#!/usr/bin/env python3
"""A 线批量采集（**只跑冻结 WBC、不进 VLA**）：在 Isaac 真物理里向量化地走，存真 (s, a, s', 全局行号)。

为什么需要这个（2026-10-29，用户指出）
--------------------------------------
mjlab 那条线的**第 1 步**是：`collect_mjlab.py`，1024 个 env 并行、场景里**只有冻结 WBC、没有 VLA**，
1000 tick → 约 **100 万条真实转移**，用它训出来的 WM 在 sim 分布上 `qd_corr=0.9925 / qd_ratio=1.006`。

我们 S3 的 WM 吃的却是：
* `packed.pt` 的 `ref`（634,016 帧）= **运动学参考**，
* `ref_action_lib.pt` 的 `actions` = 录制 npz 的 `decoder_target_action`（**不是** WBC 在物理里真下发的），
* 加 5% 的 B 线（VLA 在环、漂移分布）。

于是 WM 学的是"参考轨迹的连续性"，不是"机器人在 Isaac 里真跑出来的动力学" ——
实测 `qd_corr` 只有 0.743（k=1）、`qd_ratio` 0.948（**系统性低估速度 5%**）。
本脚本补上缺的那一步。

与 `scripts/isaac_g1_collect.py` 的关系
---------------------------------------
* 那个是**单 clip、默认 8 env** 的最小验证版，且 `frames` 存的是 **clip 内偏移**（缺 `starts[ci]`，
  喂给 `compute_collect_reward.py` 的 `FrameRef` 会错位）；
* 本脚本是**多 clip 批量**版：
    - 每个 env 独立分配 `(clip, start)` ⇒ N 个 env 是**不同片段**，不是同一段的复制；
    - `frames` 存 **全局库行号** `starts[ci] + start + step`（奖励/参考直接可用）；
    - 额外存 `clip_ids`，便于按 clip 切分训/留出；
    - 布局与旧的**完全相同**（每步追加一块 n 行，`prev_action` 可用"回退 n 行"近似）。

用法
----
```bash
# 冒烟（16 env × 20 步，验证链路，~2 分钟）
python3 scripts/isaac_a_collect_batch.py --out runs/wm_real/a_smoke.pt \
    --num-envs 16 --steps 20

# 正式：512 env × 300 步 = 15.4 万条/波；跑 7 波 ≈ 108 万条
python3 scripts/isaac_a_collect_batch.py --out runs/wm_real/a_w0.pt \
    --num-envs 512 --steps 300
```

产出的奖励**不在这里算**（Isaac venv 里 import 训练栈会炸），用系统 python 补：
```bash
python3 scripts/compute_collect_reward.py --in runs/wm_real/a_w*.pt --out runs/wm_real/a_rew.pt
```
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HIST = 10
DEFAULT_USD = ("third_party/HumanoidArena/isaaclab_twist2_g1/assets/robots/"
               "g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd")

# ★ 2026-11-03：**并行跑多个 Isaac 实例必踩的坑**。
#   HA 的 `image_server/shared_memory_utils.py` 在 **import 时**就执行
#   `SHM_NAME = resolve_shm_name()`，默认名写死成 `isaac_multi_image_shm`。
#   于是第 2、3、4 个并发实例会抛
#       FileExistsError: [Errno 17] File exists: '/isaac_multi_image_shm'
#   而且这个异常发生在 ImageServer（非 daemon 线程）里 ⇒ 进程**退出码仍然是 0**、
#   `.pt` 产物没写出来。多卡驱动只看 rc，就会把这种分片当成"成功"，永久漏采
#   （实测 VisionNavi off192/off224 两个满批分片就是这么丢的，A' 只采到 583/600 条 clip）。
#   官方脚本 `script/eval_scripts/sonic/sim_eval_vla.py:185` 的做法是每个进程一个唯一名，
#   这里照抄（必须在 import 到那个模块**之前**设置）。
os.environ.setdefault(
    "ISAAC_MULTI_IMAGE_SHM_NAME",
    f"isaac_multi_image_shm_{os.getpid()}_{os.urandom(4).hex()}",
)

# Isaac 侧依赖（和 isaac_g1_collect.py 同一条最小依赖链：刻意不 import 训练栈）
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="A 线批量采集（WBC-only，多 clip 向量化）")
parser.add_argument("--wbc-ckpt", type=Path,
                    default=ROOT / "runs/S1_handover/WBC_A24000_wmvel_v6.pt")
parser.add_argument("--packed", type=Path,
                    default=ROOT / "runs/isaac_motion/ha_cmd/packed.pt")
parser.add_argument("--tokens-dir", type=Path, default=ROOT / "data/npz/ha_tokens")
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--exec-steps", type=int, default=20)
parser.add_argument("--start-mode", choices=("uniform", "first", "windows"), default="uniform",
                    help="uniform=每个 env 在自己的 clip 里均匀撒起点；first=都从 clip 起点开始；"
                         "windows=确定性全覆盖（配 --plan-offset 分片）")
parser.add_argument("--plan-repeat-k", type=int, default=0,
                    help="**配对探针用**：每 K 个 env 共用同一个窗口/同一起点（0=关）。"
                         "配合 --segment 20 ⇒ 每段重置后这 K 个 env 状态**完全相同**、"
                         "只有动作(噪声)不同 ⇒ 直接得到 (s, a+δ_i, s'_i) 真配对。"
                         "num_envs=64、K=4 ⇒ 一次 launch 覆盖 16 个不同状态。")
# ★★ 2026-11-04：**窗口跨度（window-stride）与运行步数（steps）解耦**。
#   动机：单个 Isaac 进程的启动要 ~60 s，而 270 步仿真只要 ~14 s ⇒ 启动占 81%，
#   一轮 2087 个窗口要 40 分钟。最直接的"把 steps 从 270 提到 810"会把
#   `L < steps` 的 clip 全部丢掉：实测 810 时只剩 221/599 条 clip、覆盖 50.8% 帧。
#   所以真正该做的是：**窗口仍按 260 帧枚举（每条 clip 都能覆盖），
#   但一个进程连续跑 780 步 ⇒ 每个 env 自动跨 3 个相邻窗口**。
#   `frames = lib_starts + start + step` 本身就是绝对帧号，跨窗口天然连续；
#   `start + step < clips` 的掩码负责 clip 末尾，20 步重置照旧。零覆盖损失。
parser.add_argument("--window-stride", type=int, default=0,
                    help="窗口枚举跨度（默认 0 = 用 --steps）。"
                         "设成 260 + --steps 780 就是「一个进程连跑 3 个窗口」。"
                         "必须是 20 的整数倍，否则段重置相位会漂。")
parser.add_argument("--plan-offset", type=int, default=0,
                    help="windows 模式：本次取全局窗口列表的起点（多卡/多波分片用）")
# ★ 2026-11-03：windows 模式下 `[offset, offset+num_envs)` 必须是满批，否则直接 SystemExit。
#   驱动脚本为了简单会在最后不足一批时 break，导致**每个任务的最后几条 clip 永久漏采**
#   （实测漏了 17 条 / 600）。加这个开关允许尾批不足：多出来的 env 不建、不写，
#   只用前 len(chunk) 个 env。
parser.add_argument("--allow-partial", action="store_true",
                    help="windows 模式允许尾批不足 num_envs（用实际窗口数建 env）")
parser.add_argument("--clip-start", type=int, default=0,
                    help="跳过可用列表前 N 条 clip（多波次分片用）")
parser.add_argument("--segment", type=int, default=0,
                    help=">0 = 分段采集：每 segment 步把状态重置回参考帧（防止漂移累积；"
                         "joint 训练想象 H=20，推荐 20）")
parser.add_argument("--clip-end", type=int, default=-1,
                    help="只用到这个 clip 索引（不含）；-1=到末尾。配 --clip-start 做任务分组")
parser.add_argument("--clip-index", type=int, default=-1,
                    help="只跑库里这个 index 的 clip（-1=不限）；诊断/对拍录制数据时必用")
parser.add_argument("--len-min", type=int, default=0,
                    help="只收这个长度以上的 clip（全长模式按长度分桶用，0=不限）")
parser.add_argument("--len-max", type=int, default=0,
                    help="只收这个长度以下的 clip（0=不限）")
parser.add_argument("--token-source", choices=("npy", "online"), default="npy",
                    help="npy=冻结录制 token（开环，实测 100 步就漂 44 cm）；"
                         "online=每 hold 窗口用 SonicTokenBuilder(参考行, 当前 root 姿态) 实时算"
                         "（= 训练/部署同口径，G1 online 臂 2080 步只有 0.18 m）")
parser.add_argument("--ha-task", type=str, default="",
                    help="非空 = 在 **HA 任务场景**里跑（gym 任务 id，例如 "
                         "Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody）。"
                         "空 = 自建裸地面场景（旧行为，口径不对，只做诊断）")
parser.add_argument("--ha-yaml", type=str, default="tasks/common_test_config/base_test/doubledesk_sonic_test.yaml",
                    help="HA 任务 env 的 yaml 覆盖（与闭环同一份）")
parser.add_argument("--executor", choices=("wbc", "sonic"), default="wbc",
                    help="wbc=我们的 SuperMimic WBC（默认）；sonic=SONIC release decoder"
                         "（994→29，已转成可微 torch，逐位复刻 ONNX）")
parser.add_argument("--replay-target-npz", type=str, default="",
                    help="诊断用：直接回放录制里的 decoder_target_action 当关节目标（不经过 decoder），"
                         "用来判定'仿真/初始状态/PD'这一侧对不对")
parser.add_argument("--replay-decoder-obs", type=str, default="",
                    help="诊断用：喂**录制**的 decoder_obs（npz 路径）而不是自己拼 obs，"
                         "用来把'obs 拼装错'和'仿真/尺度错'分开")
parser.add_argument("--sonic-decoder", type=Path, default=None,
                    help="sonic executor 用的 ONNX（默认 GR00T release model_decoder.onnx）")
parser.add_argument("--token-lib", type=Path,
                    default=ROOT / "runs/isaac_motion/ha_cmd/packed.pt",
                    help="online 模式的命令流库（token 的 q/qd/root_quat 取自这里）")
parser.add_argument("--max-clips", type=int, default=0, help="只用前 N 个 clip（0=全部）")
parser.add_argument("--seed", type=int, default=20261029)
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--log-every", type=int, default=25)
parser.add_argument("--out", type=Path, default=ROOT / "runs/wm_real/a_line.pt")
parser.add_argument("--dry-run", action="store_true", help="只校验 clip/token 映射，不启动 Isaac")
# ---- A' 扰动（2026-11-03 从 isaac_collect_perturbed.py 移植；默认全 0 = 原 A 线行为不变）----
parser.add_argument("--action-noise-std", type=float, default=0.0,
                    help="A' L1 动作噪声（0=关）。gauss 时为 std，uniform 时为半宽")
parser.add_argument("--noise-mode", choices=("gauss", "uniform"), default="gauss")
parser.add_argument("--perturb-hold", type=int, default=20,
                    help="L1 噪声保持步数（跟闭环 token hold 同尺度，默认 20）")
parser.add_argument("--start-perturb-xy", type=float, default=0.0, help="A' L2 起点 root xy 扰动 σ(m)")
parser.add_argument("--start-perturb-yaw", type=float, default=0.0, help="A' L2 起点 yaw 扰动 σ(rad)")
parser.add_argument("--start-perturb-dq", type=float, default=0.0, help="A' L2 起点关节角扰动 σ(rad)")
# ★ 2026-11-03：**段起点扰动**（每次 --segment 重置之后再加，不是只在 episode 开头加一次）。
#   为什么要（用户追问"每 20 步重置是不是就和联合微调对齐了"）：
#     * A' 的 20 步段每段都被**瞬移回参考帧**（实测段首 root 位置误差 p50 = 0.00 cm）；
#     * 部署时 receding horizon 每 20 步重规划，但机器人**不回参考**——第 2 个窗口的起点
#       已经是漂过的状态。实测一个 20 步段内 root 偏差从 0 长到 p50 8.2 cm / p90 20.2 cm。
#   ⇒ 训练只覆盖"从参考出发的 20 步"，部署走的是"从偏移出发的 20 步"，分布对不上。
#   这个开关让每段起点带上和部署同量级的偏移，段长仍是 20 步（照样采得完、能 100% 覆盖，
#   不会像全程不重置那样漂到 1–3 m 采不动）。
# ★★ 2026-11-04 **缰绳式（leash）重置**：不按固定步数重置，而是**当某个 env 的
#   root 偏移 |ξ| 超过 δ 才把它拉回参考**。为什么（用户口径 + 实测）：
#     * 每 20 步硬重置 ⇒ 段长 20 帧，而一个合法微调窗口要"10 帧历史 + 20 步 horizon
#       = 30 帧在同一段内" ⇒ **结构上不可能**，所有窗口的历史都跨瞬移；
#     * 起点偏移恒为 0 ⇒ 训练学不到"漂了以后继续正常工作"，WM 也没见过 ξ>1.5cm 的状态；
#     * 实测无重置 6 秒漂到 10–24 cm 且**不摔** ⇒ 用 δ=0.10 把漂移夹住是安全的。
#   段长自然变成 100–150 步（2–3 秒），历史连续、起点覆盖 ξ∈[0,δ]。
parser.add_argument("--reset-max-drift", type=float, default=0.0,
                    help="leash：某 env 的 root 偏移超过该值(m)才重置它（0=关，用 --segment）")
parser.add_argument("--seg-start-perturb-xy", type=float, default=0.0,
                    help="每段重置后 root xy 扰动 σ(m)；建议 0.08（= 实测段末 p50）")
parser.add_argument("--seg-start-perturb-yaw", type=float, default=0.0,
                    help="每段重置后 root yaw 扰动 σ(rad)")
parser.add_argument("--seg-start-perturb-dq", type=float, default=0.0,
                    help="每段重置后关节角扰动 σ(rad)")
parser.add_argument("--seg-start-perturb-v", type=float, default=0.0,
                    help="每段重置后 root 线速度扰动 σ(m/s)")
parser.add_argument("--action-clip", type=float, default=0.0,
                    help="动作截断上限（0=不截断，与训练口径一致；这个 WBC 训练时 action_limit=None）")
AppLauncher.add_app_launcher_args(parser)
args_cli, _unknown = parser.parse_known_args()

DEV = torch.device("cuda:0")


DEFAULT_SONIC_DECODER = Path(
    "/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/GR00T-WholeBodyControl/"
    "gear_sonic_deploy/policy/release/model_decoder.onnx")


def _gravity_body(q_wxyz: torch.Tensor) -> torch.Tensor:
    """把世界坐标 [0,0,-1] 转到机身系 = **旋转量用 q 的共轭**（SONIC deploy 口径）。"""
    w, x, y, z = q_wxyz.unbind(-1)
    u = torch.stack([x, y, z], dim=-1)
    v = torch.zeros_like(u)
    v[..., 2] = -1.0
    t = 2.0 * torch.cross(u, v, dim=-1)          # 2(u×v)
    return v - w.unsqueeze(-1) * t + torch.cross(u, t, dim=-1)


class _PackedLibView:
    """把 ``packed.pt`` 的 ref 表包成 ``SonicTokenBuilder`` 需要的最小运动库接口。

    ref 布局 ``[p(3), quat_wxyz(4), q(nj), v(3), w(3), qd(nj)]``（nj=29）。
    builder 只用 ``.q`` / ``.qd`` / ``.root_quat(frame)`` / ``.last_frame_of(frame)``。
    """

    def __init__(self, ref: torch.Tensor, starts, lengths, device):
        self.ref = ref
        nj = (ref.shape[1] - 13) // 2
        # ★ 库 ref 布局是 [p(3), quat(4), v(3), w(3), q(29), qd(29)] ⇒ q 在 13:42
        self.q = ref[:, 13:13 + nj].to(device).contiguous()
        self.qd = ref[:, 42:42 + nj].to(device).contiguous()
        self._quat = ref[:, 3:7].to(device).contiguous()
        last = torch.empty(ref.shape[0], dtype=torch.long)
        for s, L in zip(starts, lengths):
            s, L = int(s), int(L)
            last[s:s + L] = s + L - 1
        self._last = last.to(device)

    def root_quat(self, frames: torch.Tensor) -> torch.Tensor:
        return self._quat[frames.to(self._quat.device)]

    def last_frame_of(self, frames: torch.Tensor) -> torch.Tensor:
        return self._last[frames.to(self._last.device)]


def _load_tokens(path: Path) -> np.ndarray:
    """从 npz/npy 读 64 维 token 流。

    数据集的 npz 里键名是 ``g1_motion_token_64``（不是裸数组）—— 2026-10-29 实测，
    直接 ``np.load(...)`` 当数组用会拿到 0 个名字、形状全错。
    """
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(str(path)).astype(np.float32)
    z = np.load(str(path))
    for k in ("g1_motion_token_64", "tokens", "token", "arr_0"):
        if k in z:
            return np.asarray(z[k]).astype(np.float32)
    raise SystemExit(f"[a-line] {path} 里找不到 token 键（试过 g1_motion_token_64/tokens/token/arr_0）")


def build_plan(lib_names, lib_starts, lib_lengths, steps: int, num_envs: int,
               tokens_dir: Path, max_clips: int, start_mode: str, seed: int,
               window_stride: int = 0):
    """给每个 env 分配 (clip_idx, start, token_npy)。返回 plan 列表与统计。

    ``steps`` = 这个进程要跑多少步；``window_stride`` = 窗口枚举跨度（0 ⇒ 等于 steps）。
    两者分开是为了「一个进程连跑多个窗口」：steps=780, window_stride=260 ⇒
    每个 env 从 260 边界的起点出发连跑 780 步，天然覆盖 3 个相邻窗口，
    而 clip 的长度门槛只按 260 判（所以 271 帧的 clip 也不会被丢）。
    """
    _stride = int(window_stride) if int(window_stride) > 0 else int(steps)
    if start_mode == "windows" and (int(steps) % 20) != 0:
        print(f"[a-line][warn] --steps={steps} 不是 20 的整数倍：段重置相位会在窗口之间漂移",
              flush=True)
    tdir = Path(tokens_dir)
    # 2026-11-03：A' 线的 token 是 `runs/exp_tokens/<clip>.npy`（699 个 npy，不是 npz）。
    # 这里两种后缀都认，老的 npz 目录行为不变。
    _files = sorted(list(tdir.glob("*.npz")) + list(tdir.glob("*.npy")))
    stems = [q.stem for q in _files]
    if not stems:
        raise SystemExit(f"[a-line] {tdir} 里没有 token npz/npy")

    def token_path(ci: int):
        # 库名（ep<timestamp>）与 token stem（ep000000）不一致：按**同序号**回退（旧的坑，已验证长度一致）
        name = lib_names[ci]
        for suf in (".npz", ".npy"):
            cand = tdir / f"{name}{suf}"
            if cand.exists():
                return cand
        if ci < len(stems):
            for suf in (".npz", ".npy"):
                cand = tdir / f"{stems[ci]}{suf}"
                if cand.exists():
                    return cand
        return None

    usable = []
    _lmin = int(getattr(args_cli, "len_min", 0) or 0)
    _lmax = int(getattr(args_cli, "len_max", 0) or 0)
    for ci in range(len(lib_names)):
        if max_clips and ci >= max_clips:
            break
        L = int(lib_lengths[ci])
        # first = 全长模式（起点固定 0，跑 steps 步，越界行由掩码丢掉）⇒ 允许 L < steps；
        # 其他模式起点必须留出 steps 的余量。
        # ★ 2026-11-03：原来是 `L < steps + 2`，把 L == steps+1 的 clip 也踢掉了
        #   （本库只有 clip 385，L=271、steps=270 ⇒ 覆盖永远差 1 条：599/600）。
        #   实际上 `L >= steps` 就够：窗口起点 0 与 L-steps 都合法。
        # ★ 2026-11-04：长度门槛不再等于 steps。让一个进程连跑 780 步时，
        #   271 帧的 clip 也应该能采（跑 271 步后由掩码丢掉越界行），
        #   所以门槛改成 `--len-min`（默认 20 = 至少一个 20 步段）。
        if start_mode != "first" and L < max(20, _lmin):
            continue
        if ci < int(getattr(args_cli, "clip_start", 0) or 0):
            continue
        _ce = int(getattr(args_cli, "clip_end", -1) or -1)
        if _ce >= 0 and ci >= _ce:
            continue
        _ci = int(getattr(args_cli, "clip_index", -1) or -1)
        if _ci >= 0 and ci != _ci:
            continue
        if _lmin and L < _lmin:
            continue
        if _lmax and L > _lmax:
            continue
        tp = token_path(ci)
        if tp is None or not tp.exists():
            continue
        usable.append((ci, L, tp))
    if not usable:
        raise SystemExit("[a-line] 没有可用 clip（长度/token 都不满足）")

    # ★ 2026-11-03：windows 模式 = **确定性全覆盖**。按 (clip, 窗口起点) 枚举所有窗口，
    #   一次 launch 取 [plan_offset, plan_offset+num_envs) 这一段。这样多卡多波拼起来
    #   刚好不重不漏地盖满 634,016 帧（G2 的 frames.max()≈634k 就是这么来的）。
    #   random 模式保持原行为（每个 env 随机 clip + 随机起点）。
    if start_mode == "windows":
        cells = []
        for ci, L, tp in usable:
            # 2026-11-03：补"尾窗口"，保证每条 clip 的**全长**都被覆盖
            #   （原来 range(0, L-steps-1, steps) 会丢掉每条 clip 的末尾 < steps 帧）
            _last = -1
            for st in range(0, max(1, L - _stride), _stride):
                cells.append({"env": -1, "clip_idx": ci, "start": int(st),
                              "clips": L, "token": tp})
                _last = int(st)
            if L > _stride and _last + _stride < L:
                cells.append({"env": -1, "clip_idx": ci, "start": int(L - _stride),
                              "clips": L, "token": tp})
        off = int(args_cli.plan_offset)
        _K = int(getattr(args_cli, "plan_repeat_k", 0) or 0)
        if _K > 0:
            # 配对探针：每 K 个 env 共用一个窗口 ⇒ 段重置后这 K 个 env 状态完全相同
            _nc = num_envs // _K
            if off + _nc > len(cells):
                raise SystemExit(f"[a-line] plan-repeat-k: offset={off}+{_nc} 越界（共 {len(cells)} 窗口）")
            chunk = [dict(cells[off + i // _K]) for i in range(num_envs)]
        else:
            chunk = cells[off:off + num_envs]
        if len(chunk) < num_envs:
            if not bool(getattr(args_cli, "allow_partial", False)):
                raise SystemExit(f"[a-line] windows 模式窗口不够：offset={off} 需要 {num_envs}，"
                                 f"总共 {len(cells)} 个窗口")
            if not chunk:
                raise SystemExit(f"[a-line] windows 模式 offset={off} 已越过末尾"
                                 f"（总共 {len(cells)} 个窗口）")
            print(f"[a-line] windows 模式尾批：offset={off} 只剩 {len(chunk)} 个窗口"
                  f"（< num_envs={num_envs}），按 --allow-partial 建 {len(chunk)} 个 env",
                  flush=True)
        for e, c in enumerate(chunk):
            c["env"] = e
        n_clip_used = len({p["clip_idx"] for p in chunk})
        print(f"[a-line] windows 模式：窗口总 {len(cells)}，本次取 [{off},{off + len(chunk)})",
              flush=True)
        return chunk, n_clip_used, len(usable)

    rng = np.random.RandomState(seed)
    order = rng.permutation(len(usable))
    plan = []
    for e in range(num_envs):
        ci, L, tp = usable[int(order[e % len(usable)])]
        hi = max(1, L - steps - 1)
        start = 0 if start_mode == "first" else int(rng.randint(0, hi))
        plan.append({"env": e, "clip_idx": ci, "start": start, "clips": L, "token": tp})
    n_clip_used = len({p["clip_idx"] for p in plan})
    return plan, n_clip_used, len(usable)


def main() -> int:
    ck = torch.load(str(args_cli.wbc_ckpt), map_location="cpu", weights_only=False)
    rs = ck["robot_spec"]
    joint_names = list(rs["joint_names"])
    default_q = torch.tensor(rs["default_q"], dtype=torch.float32, device=DEV)
    scale = torch.tensor(rs["action_scale"], dtype=torch.float32, device=DEV)
    nj = len(joint_names)
    # ★ USD 优先取 ckpt 里 `robot_spec.usd`（就是训练用的那个），比环境变量可靠：
    #   我们试过 `SUPERMIMIC_ROBOT_USD` 未导出时脚本直接 KeyError。
    usd_path = (os.environ.get("SM_USD_OVERRIDE") or rs.get("usd")
                or os.environ.get("SUPERMIMIC_ROBOT_USD") or str(ROOT / DEFAULT_USD))
    print(f"[a-line] USD = {usd_path}", flush=True)

    packed = torch.load(str(args_cli.packed), map_location="cpu", weights_only=False)
    ref = torch.as_tensor(packed["ref"]).float()
    lib_names = list(packed["names"])
    lib_starts = [int(x) for x in packed["starts"]]
    lib_lengths = [int(x) for x in packed["lengths"]]

    n = int(args_cli.num_envs)
    plan, n_clip_used, n_usable = build_plan(lib_names, lib_starts, lib_lengths,
                                             int(args_cli.steps), n, args_cli.tokens_dir,
                                             int(args_cli.max_clips), args_cli.start_mode,
                                             int(args_cli.seed),
                                             int(getattr(args_cli, "window_stride", 0) or 0))
    # ★ 2026-11-03：尾批可能不足 num_envs（--allow-partial）⇒ env 数必须以 plan 为准，
    #   否则 plan 里没有的 env 会用**上一个 plan 项的参考**跑，写出脏数据。
    if len(plan) != n:
        print(f"[a-line] num_envs {n} -> {len(plan)}（尾批 / --allow-partial）", flush=True)
        n = len(plan)
    print(f"[a-line] 可用 clip {n_usable} 个；本次 {n} env 覆盖 {n_clip_used} 个不同 clip；"
          f"steps={args_cli.steps} ⇒ 预计 {n * int(args_cli.steps)} 条转移", flush=True)

    # 每个 env 自己的 token 流（一次性读进内存；699 clip × ~3000 帧 × 64 太大就按需读）
    toks = {}
    if args_cli.token_source == "npy":
        for p in plan:
            ci = p["clip_idx"]
            if ci not in toks:
                toks[ci] = _load_tokens(p["token"])
        for ci, arr in toks.items():
            if arr.ndim != 2 or arr.shape[1] != 64:
                raise SystemExit(f"[a-line] clip {ci} 的 token 形状异常 {arr.shape}")

    # ---- online token（2026-11-03）：训练/部署同口径 -----------------------------
    # 用 SonicTokenBuilder(frame, robot_quat)：q/qd/root_quat 取**命令流库**（ha_cmd），
    # anchor 用**当前机器人**的 root 姿态。冻结 npy 是开环的：机器人一偏，token 还在
    # 按原计划走 → 实测 100 步漂 44 cm、800 步 2.4 m，奖励恒 0。online 臂同 WBC 只有 0.18 m。
    token_fn = None
    if args_cli.token_source == "online":
        from trackerLab.algorithms.supermimic.sonic_token import SonicTokenBuilder
        _tp = torch.load(str(args_cli.token_lib), map_location="cpu", weights_only=False)
        _tstarts = [int(x) for x in _tp["starts"]]
        _tnames = list(_tp["names"])
        if _tnames != lib_names:
            raise SystemExit("[a-line] token 库与参考库 clip 顺序不一致，拒绝 online 模式")
        _view = _PackedLibView(torch.as_tensor(_tp["ref"]).float(), _tstarts,
                               [int(x) for x in _tp["lengths"]], DEV)
        token_fn = SonicTokenBuilder(_view, DEV)
        print(f"[a-line] token=online  cmd_lib={args_cli.token_lib} "
              f"encoder={type(token_fn.encoder).__name__} window={token_fn.window}", flush=True)

    if args_cli.dry_run:
        print("[a-line] dry-run OK：映射全部解析成功，未启动 Isaac")
        for p in plan[:5]:
            L = len(toks[p["clip_idx"]]) if toks else -1
            print(f"   env{p['env']:>3} clip={p['clip_idx']:>4} start={p['start']:>5} "
                  f"len={p['clips']:>5} token_frames={L}"
                  f"{'  (online)' if token_fn is not None else ''}")
        return 0

    # ------------------------------------------------------------------ Isaac
    # ★ 顺序铁律：**必须先 `AppLauncher(...).app` 启动 Isaac Sim**，再 import
    #   `isaaclab.sim/actuators/assets/scene` —— 那些模块要 `pxr`（USD）和 `warp`，
    #   而这两个是启动 app 时才挂到 sys.path 上的。反过来写就是
    #   `ModuleNotFoundError: No module named 'pxr'` / `'warp'`（2026-10-29 踩过）。
    app = AppLauncher(args_cli).app  # noqa: F841

    import isaaclab.sim as sim_utils  # noqa: PLC0415
    from isaaclab.actuators import ImplicitActuatorCfg  # noqa: PLC0415
    from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg  # noqa: PLC0415
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: PLC0415
    from isaaclab.sim import SimulationContext  # noqa: PLC0415
    from isaaclab.utils import configclass  # noqa: PLC0415

    from trackerLab.algorithms.supermimic.networks import Policy  # noqa: PLC0415
    pol = Policy(obs_dim=93, goal_dim=0, hist=HIST, token_dim=64, num_joints=nj, action_limit=None)
    miss, unexp = pol.load_state_dict(ck["pi"], strict=False)
    if miss or unexp:
        raise SystemExit(f"[a-line] policy 权重不匹配 missing={miss[:4]} unexpected={unexp[:4]}")
    pol = pol.to(DEV).eval()

    # ---- 执行器：SONIC release decoder（2026-11-03）---------------------------
    # 994 = 64(token) + 10×[ang_vel(3) | q-default(29) | qd(29) | last_a(29) | grav(3)]
    # 顺序取自 action_provider_sonic.py::_build_live_decoder_obs_with_latent（权威）。
    dec_mod = None
    _DEC_Q_OFF = None
    if str(getattr(args_cli, "executor", "wbc")) == "sonic":
        _of = ROOT / "runs/decoder_q_offset.pt"
        if _of.exists():
            _DEC_Q_OFF = torch.load(_of, map_location="cpu", weights_only=False)["offset"].to(DEV)
            print(f"[a-line] decoder q 块偏移已加载（非零关节 {int((_DEC_Q_OFF.abs()>1e-3).sum())}/29）", flush=True)
        # ★ 2026-11-03（最小二乘反解，残差 0.0000）：动作→目标 = SONIC_DEFAULT_POS + a*SONIC_SCALE，
        #   且与 q 块**同一关节序**。所以目标必须用这两个常量，不能用 ckpt 的 default_q/action_scale
        #   （两者相差最大 0.201 rad，正是闭环发散的根源）。
        import sys as _s4
        if str(ROOT / "third_party/HumanoidArena/isaaclab_twist2_g1") not in _s4.path:
            _s4.path.insert(0, str(ROOT / "third_party/HumanoidArena/isaaclab_twist2_g1"))
        from action_provider.action_provider_sonic import (
            SONIC_DEFAULT_POS as _SDEFF, G1_ACTION_SCALE_ISAACLAB as _SSCF)
        _SDEF = torch.as_tensor(_SDEFF, dtype=torch.float32, device=DEV)
        _SSC = torch.as_tensor(_SSCF, dtype=torch.float32, device=DEV)
        print("[a-line] decoder 目标映射用 SONIC_DEFAULT_POS + a*G1_ACTION_SCALE_ISAACLAB", flush=True)
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "scripts"))
        from sonic_decoder_torch import load_decoder
        _onnx = args_cli.sonic_decoder or DEFAULT_SONIC_DECODER
        dec_mod = load_decoder(_onnx).to(DEV).eval()
        for _p in dec_mod.parameters():
            _p.requires_grad_(False)
        print(f"[a-line] executor=sonic decoder={_onnx}", flush=True)

    ha_env = None
    _decim, _pdt = 4, 1.0 / 200.0
    if str(getattr(args_cli, "ha_task", "") or ""):
        # ---- HA 任务场景（口径与部署一致：桌子/物体/接触材质全来自任务 cfg）----
        import gymnasium as gym  # noqa: PLC0415
        import tasks  # noqa: F401,PLC0415  （注册 gym 环境）
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: PLC0415
        from tasks.common_env_config import apply_env_config_yaml  # noqa: PLC0415
        _cfg = parse_env_cfg(args_cli.ha_task, device=str(DEV), num_envs=n)
        _cfg.env_name = args_cli.ha_task
        if args_cli.ha_yaml:
            apply_env_config_yaml(_cfg, args_cli.ha_yaml, task_name=args_cli.ha_task,
                                  route_name="supermimic_wholebody")
        ha_env = gym.make(args_cli.ha_task, cfg=_cfg).unwrapped
        # ★ 2026-11-03：照搬官方 sim_eval_vla.py 的场景初始化/重置流程，否则物体摆放、
        #   task reset、视觉/随机化都和录制不一致（实测跟踪目标的能力差 ~3×）。
        try:
            _init_fn = getattr(_cfg, "initialize_task_scene", None)
            if callable(_init_fn):
                _init_fn(ha_env, args_cli)
                print("[a-line] HA initialize_task_scene 已执行", flush=True)
        except Exception as _ie:  # noqa: BLE001
            print(f"[a-line] WARN initialize_task_scene: {_ie}", flush=True)
        # ★ 2026-11-03：**必须先 reset 一次**再写我们的初始状态。不 reset 的话，
        #   env 的首次 reset 会把机器人摆回默认位姿，把我们写的参考初值冲掉
        #   （实测 step 0 就和录制差 5.4 m / 1.3 rad，看起来像"控制器不行"，其实是初值没生效）。
        # ★ 2026-11-03：**不要调用 env.reset()** —— 这个 HA 任务场景在 num_envs>1 时
        #   reset 会触发 `CUDA device-side assert`（4/16/32 env 都复现）。
        #   不 reset、直接写参考初值反而是对的（之前 64 env 能跑就是这条路径）。
        if str(os.environ.get("HA_ENV_RESET", "0")) == "1":
            try:
                ha_env.reset()
                print("[a-line] HA env 已 reset（HA_ENV_RESET=1）", flush=True)
            except Exception as _re:  # noqa: BLE001
                print(f"[a-line] WARN env.reset 失败: {_re}", flush=True)
        sim = ha_env.sim
        scene = ha_env.scene
        robot: Articulation = scene["robot"]
        _decim = int(getattr(ha_env.cfg, "decimation", 4) or 4)
        _pdt = float(ha_env.physics_dt)
        origins = torch.as_tensor(scene.env_origins, device=DEV, dtype=torch.float32)
        print(f"[a-line] HA 场景 task={args_cli.ha_task} num_envs={n} decimation={_decim}", flush=True)
        _ha = True
    else:
        _ha = False

    if not _ha:
        origins = torch.zeros(n, 3, device=DEV)
        @configclass
        class SceneCfg(InteractiveSceneCfg):
            ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
            robot: ArticulationCfg = None

        # ★ 2026-11-03：**必须复制 HA 任务场景的接触材质**，否则裸场景默认摩擦（低、combine=average）
        #   会让脚打滑：同样的参考/WBC，实测只走 0.6 m（参考 4.1 m）并且 100 步内倒。
        #   HA 的口径（move_pickplace_*_hw_env_cfg.py:134-138）：
        #       physics_material.static_friction=1.0, dynamic_friction=1.0, combine=max,
        #       friction_correlation_distance=0.00625
        _sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 200.0, device=str(DEV))
        try:
            _sim_cfg.physics_material = sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
                friction_combine_mode="max", restitution_combine_mode="max")
            _sim_cfg.physx.friction_correlation_distance = 0.00625
            print("[a-line] 接触材质已对齐 HA：static/dynamic=1.0, combine=max", flush=True)
        except Exception as _e:  # noqa: BLE001
            print(f"[a-line] ⚠️ 接触材质对齐失败（沿用默认，脚会打滑）: {_e}", flush=True)
        sim = SimulationContext(_sim_cfg)
        kp = torch.as_tensor(rs["kp"], dtype=torch.float32)
        kd = torch.as_tensor(rs["kd"], dtype=torch.float32)
        ef = torch.as_tensor(rs.get("effort_limit", [0.0] * nj), dtype=torch.float32)
        acts = {
            f"j{i}": ImplicitActuatorCfg(joint_names_expr=[joint_names[i]],
                                         stiffness=float(kp[i]), damping=float(kd[i]),
                                         effort_limit=float(ef[i]) if float(ef[i]) > 0 else None)
            for i in range(nj)
        }
        robot_cfg = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.80)),
            actuators=acts,
        )
        cfg = SceneCfg(num_envs=n, env_spacing=2.5)
        cfg.robot = robot_cfg
        scene = InteractiveScene(cfg)
        robot: Articulation = scene["robot"]
        sim.reset()
        scene.update(sim.get_physics_dt())

    def _scene_update():
        if _ha:
            scene.update(dt=_pdt)
        else:
            scene.update(sim.get_physics_dt())

    def _physics_step(target_full):
        if _ha:
            for _ in range(_decim):
                robot.set_joint_position_target(target_full)
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(dt=_pdt)
        else:
            robot.set_joint_position_target(target_full)
            robot.write_data_to_sim()
            for _ in range(4):
                sim.step()
            scene.update(sim.get_physics_dt())

    # ★ 2026-11-03：在 HA 场景里，**动作的 default/scale 要用 HA 场景自己的**，
    #   不是 WBC 训练 ckpt 里的。两者实测差到 0.35 rad（比如腕关节），用错会让
    #   "关节目标→实际姿态"整体偏一个常数量，表现为机器人下沉/跟不上。
    if _ha and str(os.environ.get("SM_USE_HA_SPEC", "0")) == "1":
        # 注意：实测用 HA spec 的 default_q 反而更差（z 0.44 / up 0.47），默认关闭。
        try:
            import sys as _s3
            _ha_root = ROOT / "third_party/HumanoidArena/isaaclab_twist2_g1"
            if str(_ha_root) not in _s3.path:
                _s3.path.insert(0, str(_ha_root))
            from action_provider.action_provider_sonic import _ha_align_spec
            _spec = _ha_align_spec()
            default_q = torch.as_tensor(_spec["default_q"], dtype=torch.float32, device=DEV)
            scale = torch.as_tensor(_spec["action_scale"], dtype=torch.float32, device=DEV)
            print(f"[a-line] 动作 default/scale 改用 HA 场景 spec（{len(default_q)} 关节）", flush=True)
        except Exception as _e:  # noqa: BLE001
            print(f"[a-line] ⚠️ HA spec 读取失败，沿用 ckpt 的 default/scale: {_e}", flush=True)

    # SONIC/canonical 序里的手臂关节索引（SM_ARM_KP_SCALE 用）
    _ARM_IDX = [11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
    # ★ 2026-11-03：官方 provider 会**显式把 HA_ALIGN 的 kp/kd 写进仿真**
    #   （action_provider_sonic.py:1951，position 模式）。我们之前依赖 env 自带增益，
    #   可能与之不一致 ⇒ 同样目标、同样初始状态，动力学却不同。
    if _ha and str(os.environ.get("SM_WRITE_GAINS", "1")) == "1":
        try:
            import sys as _s5
            _ha_root = ROOT / "third_party/HumanoidArena/isaaclab_twist2_g1"
            if str(_ha_root) not in _s5.path:
                _s5.path.insert(0, str(_ha_root))
            from action_provider.action_provider_sonic import (
                _ha_align_spec as _spec_fn, SONIC_ISAACLAB_JOINT_ORDER as _SO)
            _sp = _spec_fn()
            _names = list(robot.joint_names)
            _jidx = [_names.index(nm) for nm in _SO]
            _kp = torch.as_tensor(_sp["kp"], dtype=torch.float32, device=DEV).unsqueeze(0)
            _kd = torch.as_tensor(_sp["kd"], dtype=torch.float32, device=DEV).unsqueeze(0)
            _arm_scale = float(os.environ.get("SM_ARM_KP_SCALE", "1") or 1)
            if _arm_scale != 1.0:
                _kp = _kp.clone()
                _kp[0, _ARM_IDX] *= _arm_scale
                print(f"[a-line] 手臂 kp ×{_arm_scale}（{len(_ARM_IDX)} 关节）", flush=True)
            globals()["_GAIN_KP"], globals()["_GAIN_KD"], globals()["_GAIN_IDS"] = _kp, _kd, _jidx
            robot.write_joint_stiffness_to_sim(_kp, joint_ids=_jidx)
            robot.write_joint_damping_to_sim(_kd, joint_ids=_jidx)
            if str(os.environ.get("SM_WRITE_EFFORT", "0")) == "1":
                _ef = torch.as_tensor(_sp["effort"], dtype=torch.float32, device=DEV).unsqueeze(0)
                try:
                    robot.write_joint_effort_limit_to_sim(_ef, joint_ids=_jidx)
                    print(f"[a-line] 已写入 HA effort 限幅（肘={float(_ef[0,21]):.1f} Nm）", flush=True)
                except Exception as _ee:  # noqa: BLE001
                    print(f"[a-line] WARN 写 effort 限幅失败: {_ee}", flush=True)
            print(f"[a-line] 已按 HA spec 写入 kp/kd（{len(_jidx)} 关节，kp mean {float(_kp.mean()):.1f}）", flush=True)
            # ★ 读回校验：确认仿真里**真的**是 HA 的增益（不然就是 USD 默认的硬增益在跑）
            try:
                _kpr = robot.data.joint_stiffness[0, _jidx].detach().float()
                _kdr = robot.data.joint_damping[0, _jidx].detach().float()
                print(f"[a-line] kp 读回 mean={float(_kpr.mean()):.1f} min={float(_kpr.min()):.1f} max={float(_kpr.max()):.1f} | "
                      f"kd 读回 mean={float(_kdr.mean()):.2f}", flush=True)
            except Exception as _rbe:  # noqa: BLE001
                print(f"[a-line] kp 读回失败: {_rbe}", flush=True)
            try:
                _dks = robot.data.default_joint_stiffness[0, _jidx].detach().float()
                print(f"[a-line] env/USD default kp mean={float(_dks.mean()):.1f} "
                      f"min={float(_dks.min()):.1f} max={float(_dks.max()):.1f}", flush=True)
            except Exception:  # noqa: BLE001
                pass
        except Exception as _ge:  # noqa: BLE001
            print(f"[a-line] WARN 写增益失败: {_ge}", flush=True)

    try:
        _rc = getattr(getattr(_cfg, "scene", None), "robot", None)
        print(f"[dbg-env] env robot usd = {getattr(getattr(_rc,'spawn',None),'usd_path','?')}", flush=True)
        _acts = getattr(_rc, "actuators", None)
        if _acts:
            for _an, _ac in _acts.items():
                _kp = getattr(_ac, "stiffness", None)
                _kd = getattr(_ac, "damping", None)
                _ef = getattr(_ac, "effort_limit", None)
                _ar = getattr(_ac, "armature", None)
                _names = getattr(_ac, "joint_names_expr", None)
                print(f"[dbg-env] actuator {_an}: class={type(_ac).__name__} "
                      f"kp={_kp} kd={_kd} effort={_ef} armature={_ar} names={_names}", flush=True)
    except Exception as _de:
        print(f"[dbg-env] WARN {_de}", flush=True)
    isaac_names = list(robot.joint_names)
    missing = [j for j in joint_names if j not in isaac_names]
    if missing:
        print(f"[a-line] USD 缺关节 {missing[:5]}", flush=True)
        return 3
    idx = [isaac_names.index(j) for j in joint_names]
    idx_t = torch.tensor(idx, dtype=torch.long, device=DEV)
    print(f"[a-line] scene ready num_envs={n} 关节映射 OK", flush=True)

    # ★ 2026-11-03：SONIC decoder 用的是 **SONIC IsaacLab 关节顺序**（右腿先、左腿后），
    #   而 WBC ckpt 的 robot_spec 是另一种顺序（左/右交替）。不换顺序 = 喂错 obs、动作也写错关节，
    #   实测 |a| 飙到 3–5、40 步内必倒（就是这一条造成的）。
    idx_s, default_s, scale_s = None, None, None
    if dec_mod is not None:
        import sys as _sys2
        if str(ROOT / "third_party/HumanoidArena/isaaclab_twist2_g1") not in _sys2.path:
            _sys2.path.insert(0, str(ROOT / "third_party/HumanoidArena/isaaclab_twist2_g1"))
        from action_provider.action_provider_sonic import SONIC_ISAACLAB_JOINT_ORDER
        _perm = [joint_names.index(nm) for nm in SONIC_ISAACLAB_JOINT_ORDER]
        idx_s = [isaac_names.index(nm) for nm in SONIC_ISAACLAB_JOINT_ORDER]
        default_s = default_q[torch.tensor(_perm, device=default_q.device)]
        scale_s = scale[torch.tensor(_perm, device=scale.device)]
        print(f"[a-line] SONIC 顺序映射建立：{len(idx_s)} 关节，默认角/scale 已按名重排", flush=True)

    if idx_s is None:
        # wbc 执行器也要建立 SONIC 序映射（A' 初值/对照需要）
        try:
            import sys as _sys3
            if str(ROOT / "third_party/HumanoidArena/isaaclab_twist2_g1") not in _sys3.path:
                _sys3.path.insert(0, str(ROOT / "third_party/HumanoidArena/isaaclab_twist2_g1"))
            from action_provider.action_provider_sonic import (
                SONIC_ISAACLAB_JOINT_ORDER as _SO3)
            idx_s = [isaac_names.index(nm) for nm in _SO3]
            _perm3 = [joint_names.index(nm) for nm in _SO3]
            default_s = default_q[torch.tensor(_perm3, device=default_q.device)]
            scale_s = scale[torch.tensor(_perm3, device=scale.device)]
            print(f"[a-line] SONIC 序映射（wbc 路径也建）: {len(idx_s)} 关节", flush=True)
        except Exception as _e3:  # noqa: BLE001
            print(f"[a-line] WARN SONIC 序映射建立失败: {_e3}", flush=True)

    # ★★ 2026-11-04 **契约自检**（把踩过的坑锁死；任何一条不成立就直接退出，
    #   不写任何数据）。对应 scripts/verify_contracts.py 的同名检查。
    def _contract_fail(msg: str):
        raise SystemExit(f"[a-line][CONTRACT] ❌ {msg}")

    if dec_mod is not None:
        if _DEC_Q_OFF is None:
            _contract_fail("sonic 执行器必须有 runs/decoder_q_offset.pt（q 块偏移 = −SONIC_DEFAULT_POS）")
        _d_off = float((_DEC_Q_OFF + _SDEF).abs().max())
        if _d_off > 1e-5:
            _contract_fail(f"decoder_q_offset 与 −SONIC_DEFAULT_POS 不一致 max|Δ|={_d_off:.2e}")
        if idx_s is None:
            _contract_fail("sonic 执行器需要 idx_s（SONIC 关节序）")
        if list(idx_s) == list(idx_t):
            _contract_fail("idx_s == idx_t：SONIC 序与 standard 序相同？先查 joint_names（应该是两个不同的 29 序）")
        print(f"[a-line][CONTRACT] ✅ SONIC 契约：q/qd 走 idx_s、目标走 idx_s、"
              f"offset≡−SONIC_DEFAULT_POS(max|Δ|={_d_off:.1e})", flush=True)

    # ★ 2026-11-03（晚间，关节序修正）：录制 npz 的 `robot_qpos_before_decimation` /
    #   `final_body_action_29dof` 是 **SONIC/canonical 序**（= SONIC_ISAACLAB_JOINT_ORDER）。
    #   四项独立验证：
    #     (1) decoder_obs 的 q 块末帧 − robot_qpos 在时间上恒为常数（std 5e-6）；
    #     (2) 该常数逐关节 = −SONIC_DEFAULT_POS，非零位恰是 interleave 的 0/1、9/10、11/12、
    #         13/14、15/16（左右反号）、21/22 —— 只有 canonical 序能对出来；
    #     (3) final_body_action_29dof == SONIC_DEFAULT_POS + decoder_raw_action*SCALE（max|Δ|=0）；
    #     (4) 帧 0 按 canonical 读 = HA 默认站姿；按 standard 读会出现"左膝 0、右膝 0.4、
    #         左肘 0、左腕偏航 1.2"这种鬼姿势。
    #   而 ckpt 的 robot_spec.joint_names 是 **standard 序**，所以回放录制数据必须用 idx_s
    #   写槽位；用 idx_t 等于把 29 个值落到错的关节上。
    #   （WBC 闭环那条路走的一直是 idx_s ⇒ 闭环站得住、回放站不住，这个不对称本身就是证据。）
    _rtgt_npz_path = str(getattr(args_cli, "replay_target_npz", "") or "")
    _rtgt_ids = idx_t
    _rtgt_npz = None
    _rtgt_key = None
    _rtgt_mode = "ckpt"
    if _rtgt_npz_path:
        import numpy as _npzr
        _rtgt_npz = _npzr.load(_rtgt_npz_path, allow_pickle=True)
        _key_override = str(os.environ.get("SM_REPLAY_KEY", "") or "")
        _keys = ([_key_override] if _key_override
                 else ["final_body_action_29dof", "decoder_target_action", "vla_action_joint_pos_29"])
        for _k in _keys:
            if _k in _rtgt_npz:
                _rtgt_key = _k
                break
        if _rtgt_key is None:
            raise SystemExit(f"[a-line] 回放 npz 缺少目标键：{_rtgt_npz_path}")
        _rtgt_mode = str(os.environ.get("SM_REPLAY_IDS", "sonic")).lower()
        if _rtgt_mode == "sonic":
            if idx_s is None:
                raise SystemExit("[a-line] 回放要 SONIC 关节序，请用 --executor sonic（idx_s 未建立）")
            _rtgt_ids = torch.tensor(idx_s, dtype=torch.long, device=DEV)
        print(f"[a-line] 回放：键={_rtgt_key} shape={tuple(_rtgt_npz[_rtgt_key].shape)} "
              f"关节序={_rtgt_mode} ({'idx_s/SONIC' if _rtgt_mode=='sonic' else 'idx_t/ckpt'}) "
              f"帧数={int(_rtgt_npz['robot_qpos_before_decimation'].shape[0])}", flush=True)
    # ★ 手部：官方会把手部目标一起写进 43 维 action（`hand_action_left/right`）
    _hand_l = _hand_r = None
    _lh_idx = _rh_idx = None
    if _rtgt_npz is not None:
        if "hand_action_left" in _rtgt_npz:
            _hand_l = torch.as_tensor(_rtgt_npz["hand_action_left"], dtype=torch.float32, device=DEV)
        if "hand_action_right" in _rtgt_npz:
            _hand_r = torch.as_tensor(_rtgt_npz["hand_action_right"], dtype=torch.float32, device=DEV)
        _LN = ["left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
               "left_hand_middle_0_joint", "left_hand_middle_1_joint",
               "left_hand_index_0_joint", "left_hand_index_1_joint"]
        _RN = [n.replace("left_", "right_", 1) for n in _LN]
        _m43 = {n: i for i, n in enumerate(isaac_names)}
        _lh_idx = torch.tensor([_m43[n] for n in _LN if n in _m43], dtype=torch.long, device=DEV)
        _rh_idx = torch.tensor([_m43[n] for n in _RN if n in _m43], dtype=torch.long, device=DEV)
        print(f"[a-line] 手部目标：L{tuple(_hand_l.shape) if _hand_l is not None else None} "
              f"R{tuple(_hand_r.shape) if _hand_r is not None else None} "
              f"idx L{len(_lh_idx)} R{len(_rh_idx)}", flush=True)
    # ★ 2026-11-03（判决性实验）：直接实例化官方 SonicActionProvider，用它自己的 get_action()
    #   推进仿真。若结果仍是 mae=0.0303，则差异在 env 构造；若变成 0.0014，则差异在 provider 内部。
    _official_provider = None
    if (str(os.environ.get("SM_USE_OFFICIAL_PROVIDER", "0")) == "1"
            and _rtgt_npz is not None and ha_env is not None):
        try:
            import argparse as _ap
            from action_provider.action_provider_sonic import SonicActionProvider
            _ns = _ap.Namespace()
            for _k, _v in vars(args_cli).items():
                setattr(_ns, _k, _v)
            _ns.task = str(getattr(args_cli, "ha_task", ""))
            _ns.robot_type = "g129"
            _ns.input_source = ""
            _ns.gmt_backend = "sonic"
            _ns.replay_file = str(args_cli.replay_target_npz)
            _ns.replay_mode = "direct_replay"
            _ns.replay_loop = False
            _ns.replay_data = False
            _ns.exit_when_replay_complete = True
            _ns.enable_dex3_dds = False
            _ns.enable_dex1_dds = False
            _ns.enable_inspire_dds = False
            _enc = str(getattr(args_cli, "sonic_encoder", "") or "")
            if not _enc:
                _enc = str(Path(str(args_cli.sonic_decoder or DEFAULT_SONIC_DECODER))
                           .with_name("model_encoder.onnx"))
            _ns.sonic_encoder_path = _enc
            _ns.sonic_decoder_path = str(args_cli.sonic_decoder or DEFAULT_SONIC_DECODER)
            _ns.sonic_effort_control = False
            _ns.sonic_output_delay_steps = 0
            _ns.enable_rtf_monitor = False
            _ns.sonic_debug = False
            _ns.sonic_log_every = 50
            _official_provider = SonicActionProvider(env=ha_env, args_cli=_ns)
            print("[a-line] 官方 SonicActionProvider 已实例化（由它推进仿真）", flush=True)
        except Exception as _pe:  # noqa: BLE001
            print(f"[a-line] WARN 官方 provider 实例化失败: {_pe}", flush=True)
            _official_provider = None
    if str(os.environ.get("SM_DUMP_JOINTS", "0")) == "1":
        for _i, _nm in enumerate(isaac_names):
            print(f"[a-line] isaac_joint[{_i:2d}] = {_nm}", flush=True)

    # 每个 env 的初始根位姿 / 关节角 = 它自己参考帧的值（不平移就会一出生就在追赶）
    rows = torch.stack([
        ref[lib_starts[p["clip_idx"]] + p["start"]] for p in plan
    ]).to(DEV)                                                    # (n,71)
    # ★ 2026-11-03（晚间）：回放诊断时，初值必须取 **被回放的那个 npz 的第 0 帧**，
    #   而不是库里随机挑到的 clip（之前就是拿 open_door 的初值去回放 double_desk 的目标）。
    _rtgt_init_from_npz = False
    if _rtgt_npz is not None and all(
            k in _rtgt_npz for k in ("robot_root_position", "robot_root_orientation",
                                     "robot_qpos_before_decimation", "robot_qvel_before_decimation")):
        _p0 = torch.as_tensor(_rtgt_npz["robot_root_position"][0], dtype=torch.float32, device=DEV)
        _q0w = torch.as_tensor(_rtgt_npz["robot_root_orientation"][0], dtype=torch.float32, device=DEV)
        _j0 = torch.as_tensor(_rtgt_npz["robot_qpos_before_decimation"][0], dtype=torch.float32, device=DEV)
        _jd0 = torch.as_tensor(_rtgt_npz["robot_qvel_before_decimation"][0], dtype=torch.float32, device=DEV)
        rows = rows.clone()
        rows[:, :3] = _p0.unsqueeze(0)
        rows[:, 3:7] = _q0w.unsqueeze(0)
        rows[:, 7:10] = 0.0 if "robot_root_lin_vel_world" not in _rtgt_npz else torch.as_tensor(
            _rtgt_npz["robot_root_lin_vel_world"][0], dtype=torch.float32, device=DEV).unsqueeze(0)
        rows[:, 10:13] = 0.0 if "robot_root_ang_vel_world" not in _rtgt_npz else torch.as_tensor(
            _rtgt_npz["robot_root_ang_vel_world"][0], dtype=torch.float32, device=DEV).unsqueeze(0)
        rows[:, 13:42] = _j0.unsqueeze(0)
        rows[:, 42:71] = _jd0.unsqueeze(0)
        _rtgt_init_from_npz = True
        print(f"[a-line] 回放初值取自 npz 第 0 帧（root z={float(_p0[2]):.3f}，"
              f"q 已按 {_rtgt_mode} 序对齐）", flush=True)
    pose = torch.zeros(n, 7, device=DEV)
    pose[:, :3] = rows[:, :3] + origins          # HA 多 env 必须加各自 env 原点
    pose[:, 3:] = rows[:, 3:7]
    # ★ 2026-11-03：官方 reset 流程（sim.reset → env.reset）在 **num_envs>1 时必触
    #   `CUDA device-side assert`**，所以默认关闭；单 env 对拍时用 SM_OFFICIAL_RESET=1 打开。
    if _ha and str(os.environ.get("SM_OFFICIAL_RESET", "0")) == "1":
        # 官方 _reset_environment_for_episode 的做法：sim.reset() -> env.reset() -> 触发 reset_all_self
        try:
            # ★ 2026-11-03：官方按 **episode_seed** 重置物体（object_reset_seed_source=env_seed）。
            #   录制的 npz 里有 `episode_object_seed`，用它才能复现"那一集"的物体摆放。
            _eps = int(os.environ.get("SM_EPISODE_SEED", "0") or 0)
            if _eps:
                _ns = _eps & 0x7FFFFFFF
                _cfg.seed = _ns
                _cfg.object_reset_seed_source = "env_seed"
                setattr(_cfg, "_episode_runtime_seed", _ns)
                setattr(_cfg, "_episode_object_seed_counter", 0)
                setattr(_cfg, "_current_episode_object_seed", None)
                setattr(_cfg, "_current_episode_object_seed_source", "")
                print(f"[a-line] episode_object_seed={_eps} -> cfg.seed={_ns}", flush=True)
            ha_env.sim.reset()
            ha_env.reset()
            _em = getattr(_cfg, "event_manager", None)
            if _em is not None:
                _em.trigger("reset_all_self", ha_env)
            print("[a-line] HA 场景已按官方流程 reset（sim.reset → env.reset → reset_all_self）", flush=True)
        except Exception as _re2:  # noqa: BLE001
            print(f"[a-line] WARN HA reset 流程失败: {_re2}", flush=True)

    # ★ 2026-11-03：HA env 必须先 reset 再写初值（否则会被 env 的 spawn 覆盖成默认位姿，
    #   实测差 5.4 m）。写**只做一次**——多写几次 + scene.update 会让 PhysX 报
    #   `CUDA device-side assert`（4/16/32 env 都复现过）。
    _skip_init = str(os.environ.get("SM_SKIP_INIT_WRITE", "0")) == "1"
    if _skip_init:
        print("[a-line] SM_SKIP_INIT_WRITE=1：不写初值，直接用 env reset 后的状态（与官方一致）", flush=True)
    if not _skip_init:
        robot.write_root_pose_to_sim(pose)
    try:
        robot.write_root_velocity_to_sim(
            torch.cat((rows[:, 7:10], rows[:, 10:13]), dim=-1))
    except Exception:  # noqa: BLE001
        pass
    _back = robot.data.root_pos_w
    _err = float((_back[:, :2] - (rows[:, :2] + origins[:, :2])).norm(dim=-1).max())
    print(f"[a-line] 初始位姿读回偏差 {_err:.4f} m（应 <0.05）", flush=True)
    # 状态布局：[p(3), quat_wxyz(4), q(29), v(3), w(3), qd(29)] ⇒ q=7:36, v=36:39, w=39:42, qd=42:71
    # ★ 2026-11-03：A' 采集（非回放）也支持切换初值关节序：库 ref 的 q 是 canonical 值，
    #   若按 ckpt 的 standard 标签写就会错位。SM_INIT_IDS=sonic 用 idx_s。
    _init_ids = _rtgt_ids
    # ★ 2026-11-03（默认值修正）：库 ref 的 q 是 canonical 值 + standard 标签，
    #   初值必须按 idx_s 写，否则从第 0 步姿态就是错的 —— 这是"A' 一采就倒"的根因。
    #   默认改为 sonic；要复现旧行为显式设 SM_INIT_IDS=ckpt。
    if (_rtgt_npz is None and str(os.environ.get("SM_INIT_IDS", "sonic")) == "sonic"
            and idx_s is not None):
        _init_ids = torch.tensor(idx_s, dtype=torch.long, device=DEV)
        print("[a-line] A' 初值关节序 = idx_s (SONIC/canonical)", flush=True)
    q0 = torch.zeros(n, robot.num_joints, device=DEV)
    q0[:, _init_ids] = rows[:, 13:13 + nj]     # 回放时用 SONIC 序（见上面四条验证）
    qd0 = torch.zeros_like(q0)
    qd0[:, _init_ids] = rows[:, 42:42 + nj]   # qd 也在 42:71（布局尾部）
    if not _skip_init:
        robot.write_joint_state_to_sim(q0, qd0)
    # ★ 2026-10-29（第三个真 bug）：根速度也必须按参考帧给。
    #   随机起点的参考根速度中位数 0.556 m/s（p90 1.52），而我们让机器人**零速度**出生，
    #   WBC 变不出这个速度 ⇒ 必然趔趄然后在 ~30 步内倒（实测 z 0.77 → 0.09）。
    #   闭环里机器人是连续演化的，不存在这个"从静止突然要它按 0.6 m/s 走"的跳变。
    if not _skip_init:
        robot.write_root_velocity_to_sim(torch.cat((rows[:, 7:10], rows[:, 10:13]), dim=-1))
    # ★ 2026-11-03：写完立刻读回关节，区分"初值没写进去"和"第一步动作把它带跑"
    _bq = robot.data.joint_pos[:, _init_ids]
    _qerr = float((_bq - rows[:, 13:13 + nj]).abs().max())
    print(f"[a-line] 初始关节读回偏差 {_qerr:.4f} rad（应 ≈0）", flush=True)
    _scene_update()
    print(f"[a-line] 初速度按参考帧设置：|v| mean={float(rows[:, 7:10].norm(dim=-1).mean()):.3f} m/s "
          f"|qd| mean={float(rows[:, 42:42 + nj].abs().mean()):.3f} rad/s", flush=True)

    # ---- A' L2 起点扰动（默认 0 = 不动）--------------------------------------
    _l2 = (float(args_cli.start_perturb_xy), float(args_cli.start_perturb_yaw),
           float(args_cli.start_perturb_dq))
    if any(v > 0 for v in _l2):
        _xy, _yaw, _dq = _l2
        _rp = robot.data.root_pos_w.clone()
        _rp[:, :2] += torch.randn(n, 2, device=DEV) * _xy
        _rq = robot.data.root_quat_w.clone()
        if _yaw > 0:
            _ang = torch.randn(n, device=DEV) * _yaw
            _c, _s = torch.cos(_ang / 2), torch.sin(_ang / 2)
            _w, _x, _y, _z = _rq.unbind(-1)
            _rq = torch.stack([_c * _w - _s * _z, _c * _x - _s * _y,
                               _c * _y + _s * _x, _c * _z + _s * _w], dim=-1)
        robot.write_root_pose_to_sim(torch.cat([_rp, _rq], dim=-1))
        if _dq > 0:
            _q2 = robot.data.joint_pos.clone()
            _q2[:, idx_t] += torch.randn(n, len(idx_t), device=DEV) * _dq
            robot.write_joint_state_to_sim(_q2, torch.zeros_like(_q2))
        scene.update(sim.get_physics_dt())
        print(f"[a-line] A' L2 起点扰动：xy={_xy} yaw={_yaw} dq={_dq}", flush=True)
    else:
        print("[a-line] A' L2 关闭（名义 A 线）", flush=True)

    def tokens_at(offset: int) -> torch.Tensor:
        """每个 env 在它自己 clip 的 (start+offset) 处取 64 维 token。"""
        if token_fn is not None:
            # online：参考行取**命令流库**的全局行，anchor 用当前机器人 root 姿态。
            # ★ 2026-11-03：必须按**该 env 自己 clip 的末帧**截断。全长分桶时 steps 取的是
            #   桶内最长 clip，短 clip 跑到后面会越过自己末尾 —— 不截断就会读到下一条 clip
            #   的行；若是库里最后一条 clip 还会越界 ⇒ CUDA `device-side assert`（实测 133 env
            #   那批就是这么挂的，进程还假装在跑）。越界的行后面本来也会被掩码丢掉。
            frames = torch.tensor(
                [min(_tstarts[p["clip_idx"]] + int(p["start"]) + offset,
                     _tstarts[p["clip_idx"]] + int(p["clips"]) - 1) for p in plan],
                device=DEV, dtype=torch.long)
            return token_fn(frames, robot.data.root_quat_w.detach().to(DEV))
        cols = []
        for p in plan:
            arr = toks[p["clip_idx"]]
            fi = min(int(p["start"]) + offset, int(arr.shape[0]) - 1)
            if os.environ.get("SM_DEBUG_TOK"):
                print(f"[tokdbg] offset={offset} start={p['start']} fi={fi} "
                      f"val[0:3]={[round(float(x),3) for x in arr[fi][:3]]}", flush=True)
            cols.append(torch.as_tensor(arr[fi], device=DEV, dtype=torch.float32))
        return torch.stack(cols)                                  # (n,64)

    _tok_hold = {"tok": None, "at": -1}
    _noise = {"delta": None}
    _hists = None
    _robs = None
    _rtgt = None
    _rtgt_dbg = {"off": None}
    if str(getattr(args_cli, "replay_target_npz", "") or ""):
        _rtgt = torch.from_numpy(_npzr.asarray(_rtgt_npz[_rtgt_key], dtype=_npzr.float32))
        print(f"[a-line] 诊断模式：直接回放 {_rtgt_key} {tuple(_rtgt.shape)}", flush=True)
    if dec_mod is not None and str(getattr(args_cli, "replay_decoder_obs", "") or ""):
        import numpy as _np
        _z = _np.load(args_cli.replay_decoder_obs, allow_pickle=True)
        _robs = torch.from_numpy(_np.asarray(_z["decoder_obs"], dtype=_np.float32))
        print(f"[a-line] 诊断模式：喂录制 decoder_obs {tuple(_robs.shape)}", flush=True)
    if dec_mod is not None:
        _hists = {
            "av": torch.zeros(n, HIST, 3, device=DEV),     # root ang vel (body)
            "qp": torch.zeros(n, HIST, nj, device=DEV),    # joint_pos - default
            "qv": torch.zeros(n, HIST, nj, device=DEV),    # joint_vel
            "la": torch.zeros(n, HIST, nj, device=DEV),    # last action
            "gr": torch.zeros(n, HIST, 3, device=DEV),     # projected gravity
        }

    def step_block(offset: int, noisy: bool = False):
        if _official_provider is not None:
            try:
                _official_provider.get_action(ha_env)
            except Exception as _oae:  # noqa: BLE001  ReplayComplete 表示到头
                if type(_oae).__name__ != "ReplayComplete":
                    print(f"[a-line] official provider step 异常: {_oae}", flush=True)
            _qc = min(int(offset) + 1, int(_rtgt_npz["robot_qpos_before_decimation"].shape[0]) - 1)
            if "robot_qpos_before_decimation" in _rtgt_npz:
                _q_now = robot.data.joint_pos[0, [int(i) for i in _rtgt_ids]].detach().cpu().numpy()
                _q_rec = _rtgt_npz["robot_qpos_before_decimation"][_qc].astype("float32")
                _je = abs(_q_now - _q_rec)
                try:
                    _tgt_written = float(robot.data.joint_pos_target[0, 21])
                except Exception:
                    _tgt_written = float("nan")
                print(f"[replay-off] step={int(offset):4d} 关节 mae={_je.mean():.4f} "
                      f"max={_je.max():.3f} | z={float(robot.data.root_pos_w[0, 2]):.3f} "
                      f"写入目标[21]={_tgt_written:.4f} 实测[21]={float(_q_now[21]):.4f}", flush=True)
            return torch.zeros(1, 29, device=DEV)
        # ★ 2026-10-29 修：WBC 的 token 是 **hold 窗口**（`--exec-steps`，默认 20 步一个计划）。
        #   原来我每步都换 token ⇒ WBC 每步收到不同的未来计划，是它训练时从没见过的输入，
        #   实测 28 步（1.5 个 hold 窗口）就崩：关节角冲到 −9.35 rad、速度 10.8 rad/s，
        #   清洗后只有 10% 的行是干净的（median 有效长度 28/300）。
        #   现在严格按 hold 窗口更新 token，与闭环里的 `HA_VLA_EXEC_STEPS=20` 一致。
        if _tok_hold["tok"] is None or (offset % int(args_cli.exec_steps)) == 0:
            _tok_hold["tok"] = tokens_at(offset)
            _tok_hold["at"] = offset
        tok = _tok_hold["tok"]
        if dec_mod is not None:
            # ★★ 2026-11-03 深夜修正（**这一条是 SONIC decoder 闭环发散的根因**）
            #   之前写的是「decoder 的 q 块用 STANDARD 序（idx_t），相关 1.000」—— 错的。
            #   离线把录制 npz 的 `decoder_obs` 逐块拆开，铁证如下（数值全部 max|Δ|）：
            #     obs[0:64]        == encoder_latent                     0.000000
            #     obs[64:94]       == decoder_ang_vel_hist              0.000000
            #     obs[674:964]     == decoder_last_action_hist[t-1]     0.000000（obs 早于动作更新）
            #     obs[964:994]     == decoder_gravity_dir_hist          0.000000
            #     obs[94+9*29:384] == robot_qpos_before_decimation + c  其中 c ≡ −SONIC_DEFAULT_POS
            #                         （实测 `runs/decoder_q_offset.pt` 与 −SONIC_DEFAULT_POS
            #                           max|Δ| = 6e-8，**不是**「原始 q」）
            #   而 `SONIC_DEFAULT_POS` 是 **SONIC 序**下的默认角 ⇒ decoder 的 q 块必然是
            #   `q[SONIC 序] − SONIC_DEFAULT_POS`，和官方
            #   `action_provider_sonic.py:2646-2651` 的 `q[sonic_idx] − _sonic_default_np`
            #   一字不差。用 STANDARD 序读 = 把 29 个关节的输入喂错位（standard→SONIC
            #   的置换是 [0,6,12,1,7,13,...]，**不是恒等**），decoder 直接输出垃圾。
            q = robot.data.joint_pos[:, idx_s]
            dq = robot.data.joint_vel[:, idx_s]
        else:
            q = robot.data.joint_pos[:, idx_t]
            dq = robot.data.joint_vel[:, idx_t]
        if dec_mod is not None:
            # SONIC decoder 口径：历史逐帧滚动，新帧在最后
            for k, val in (("av", robot.data.root_ang_vel_b),
                           # ★ 2026-11-03 真 bug：q 是 **SONIC 序**，而 default_q 是 **ckpt 序**。
                           #   两者相减等于把 29 个关节的偏移量错位，decoder 直接看到垃圾观测量
                           #   （实测 up 0.97→0.58）。必须用按名重排过的 default_s。
                           # 录制逐帧反解：decoder 的 q 块 = robot_qpos + 全局常数偏移
                           ("qp", q + _DEC_Q_OFF if _DEC_Q_OFF is not None else q), ("qv", dq),
                           ("la", _last_a), ("gr", _gravity_body(robot.data.root_quat_w))):
                val = val.detach().reshape(n, -1)   # 兼容 (n,d) / (n,1,d)
                _hists[k] = torch.cat((_hists[k][:, 1:], val.unsqueeze(1)), dim=1)
            dec_obs = torch.cat([tok, _hists["av"].flatten(1), _hists["qp"].flatten(1),
                                 _hists["qv"].flatten(1), _hists["la"].flatten(1),
                                 _hists["gr"].flatten(1)], dim=-1)
            if _rtgt is not None:                     # 诊断：直接回放录制的关节目标
                _hold = str(os.environ.get("SM_HOLD_Q", "0"))
                _i = 0 if _hold == "1" else min(int(offset), _rtgt.shape[0] - 1)
                if _hold == "2":                      # 体检：保持 env 自己的默认站姿
                    _tgt_now = robot.data.default_joint_pos[
                        0, [int(k) for k in _rtgt_ids]].unsqueeze(0).to(DEV)
                else:
                    _tgt_now = _rtgt[_i:_i + 1].to(DEV)
                # ★ 2026-11-03（晚间修正）：录制目标是 **SONIC/canonical 序**，必须用 `_rtgt_ids`
                #   （= idx_s，按关节名映射）；之前用 idx_t 是把 29 个值按错关节落位。
                _step_mode = str(os.environ.get("SM_STEP_MODE", "manual"))
                if _step_mode == "official" and _ha:
                    # ★ 2026-11-03：逐字节照抄官方 provider 的推进循环
                    #   （action_provider_sonic.py:5741-5745）：
                    #   每个物理步都 set_joint_position_target(43 维) -> write_data_to_sim
                    #   -> sim.step(render=False) -> scene.update(dt=physics_dt)
                    _full43 = robot.data.default_joint_pos[0].clone()
                    _full43[[int(k) for k in _rtgt_ids]] = _tgt_now[0]
                    if _hand_l is not None and _lh_idx is not None and len(_lh_idx):
                        _full43[_lh_idx] = _hand_l[_i][:_lh_idx.numel()]
                    if _hand_r is not None and _rh_idx is not None and len(_rh_idx):
                        _full43[_rh_idx] = _hand_r[_i][:_rh_idx.numel()]
                    _t_before = float(sim.current_time)
                    if str(os.environ.get("SM_WRITE_GAINS_EVERY", "0")) == "1":
                        _kp = globals().get("_GAIN_KP"); _kd = globals().get("_GAIN_KD")
                        _ids = globals().get("_GAIN_IDS")
                        if _kp is not None and _ids is not None:
                            robot.write_joint_stiffness_to_sim(_kp, joint_ids=_ids)
                            robot.write_joint_damping_to_sim(_kd, joint_ids=_ids)
                    for _ in range(_decim):
                        robot.set_joint_position_target(_full43)
                        scene.write_data_to_sim()
                        sim.step(render=False)
                        scene.update(dt=_pdt)
                    if _i < 3:
                        try:
                            _rd = robot.data
                            _extra = ""
                            for _fn in ("applied_torque", "computed_torque", "joint_armature",
                                        "joint_friction", "joint_vel"):
                                if hasattr(_rd, _fn):
                                    _extra += f" {_fn}[21]={float(getattr(_rd, _fn)[0, 21]):.3f}"
                            print(f"[dbg-step] i={_i} sim_dt={float(sim.current_time) - _t_before:.4f} "
                                  f"cmd_elbow={float(_full43[21]):.3f} "
                                  f"ach_elbow={float(_rd.joint_pos[0, 21]):.3f} "
                                  f"kp20={float(_rd.joint_stiffness[0, 21]):.1f} "
                                  f"kd20={float(_rd.joint_damping[0, 21]):.2f}{_extra}", flush=True)
                        except Exception as _dbe:  # noqa: BLE001
                            print(f"[dbg-step] WARN {_dbe}", flush=True)
                elif _step_mode == "env" and ha_env is not None:
                    # ★ 2026-11-03：与官方完全同一条通路 —— 组装 43 维 action 交给 env.step()，
                    #   由 action_manager（JointPositionActionCfg, scale=1, use_default_offset）
                    #   处理，而不是我们手写 set_joint_position_target。
                    _full = robot.data.default_joint_pos[0].clone()
                    _full[[int(k) for k in _rtgt_ids]] = _tgt_now[0]
                    if _hand_l is not None and _lh_idx is not None and len(_lh_idx):
                        _full[_lh_idx] = _hand_l[_i][:_lh_idx.numel()]
                    if _hand_r is not None and _rh_idx is not None and len(_rh_idx):
                        _full[_rh_idx] = _hand_r[_i][:_rh_idx.numel()]
                    ha_env.step(_full.unsqueeze(0))
                else:
                    robot.set_joint_position_target(_tgt_now, joint_ids=[int(i) for i in _rtgt_ids])
                    if _ha:
                        for _ in range(_decim):
                            scene.write_data_to_sim(); sim.step(render=False); scene.update(dt=_pdt)
                    else:
                        robot.write_data_to_sim()
                        for _ in range(4):
                            sim.step()
                        scene.update(sim.get_physics_dt())
                # 逐步对拍：关节（SONIC 序）/ root，与录制 npz 比
                if _rtgt_npz is not None and "robot_qpos_before_decimation" in _rtgt_npz:
                    _qc = min(_i + 1, int(_rtgt_npz["robot_qpos_before_decimation"].shape[0]) - 1)
                    _q_now = robot.data.joint_pos[0, [int(i) for i in _rtgt_ids]].detach().cpu().numpy()
                    _q_rec = _rtgt_npz["robot_qpos_before_decimation"][_qc].astype("float32")
                    _je = abs(_q_now - _q_rec)
                    _msg = (f"[replay] step={_i:4d} 关节 mae={_je.mean():.4f} max={_je.max():.3f} rad")
                    if "robot_root_position" in _rtgt_npz:
                        _rp = robot.data.root_pos_w[0].detach().cpu().numpy()
                        _rr = _rtgt_npz["robot_root_position"][_qc].astype("float32")
                        if _rtgt_dbg["off"] is None:
                            _rtgt_dbg["off"] = _rp - _rr      # env origin 常数偏移
                        _d = _rp - _rr - _rtgt_dbg["off"]
                        _msg += f" | root drift={float((_d ** 2).sum() ** 0.5):.4f} m"
                        _msg += f" z_err={float(_d[2]):+.3f}"
                    _msg += (f" | z={float(robot.data.root_pos_w[0, 2]):.3f}"
                             f" up={float(1 - 2 * (robot.data.root_quat_w[0, 1] ** 2 + robot.data.root_quat_w[0, 2] ** 2)):.3f}")
                    print(_msg, flush=True)
                return _tgt_now.reshape(n, -1)
            _dump_obs = os.environ.get("SM_DUMP_DEC_OBS", "")
            if _dump_obs:
                try:
                    _bufs = globals().setdefault("_DEC_OBS_BUF", [])
                    _bufs.append(dec_obs.detach().cpu())
                    if len(_bufs) >= 60:
                        torch.save({"obs": torch.cat(_bufs, 0)}, _dump_obs)
                        globals()["_DEC_OBS_BUF"] = []
                        print(f"[a-line] dec_obs dump -> {_dump_obs}", flush=True)
                except Exception as _de:  # noqa: BLE001
                    pass
            if _robs is not None:                     # 诊断：用录制的 decoder_obs
                _i = min(int(offset), _robs.shape[0] - 1)
                dec_obs = _robs[_i:_i + 1].to(DEV)
            with torch.no_grad():
                a = dec_mod(dec_obs).reshape(n, -1)   # (n,29)
        else:
            if str(os.environ.get("SM_WBC_IDS", "ckpt")) == "sonic" and idx_s is not None:
                q = robot.data.joint_pos[:, idx_s]
                dq = robot.data.joint_vel[:, idx_s]
                obs = torch.cat((q - default_s, dq, robot.data.root_ang_vel_b,
                                 robot.data.projected_gravity_b, _last_a), dim=-1)
            else:
                obs = torch.cat((q - default_q, dq, robot.data.root_ang_vel_b,
                                 robot.data.projected_gravity_b, _last_a), dim=-1)
            _hist[:] = torch.cat((_hist[:, 1:], obs.unsqueeze(1)), dim=1)
            with torch.no_grad():
                a = pol(_hist, tok)
        # ---- A' L1 动作噪声（2026-11-03 移植；默认 0 = 关）-------------------
        # 噪声打在 WBC 的动作输出上、闭环内保持 perturb_hold 步：和闭环里"WBC 会看到
        # 自己造成的偏差并修正"一致。离线给状态加噪声是伪造 s'，不能这么做。
        if noisy and float(args_cli.action_noise_std) > 0:
            _h = max(1, int(args_cli.perturb_hold))
            if _noise["delta"] is None or (offset % _h) == 0:
                if args_cli.noise_mode == "gauss":
                    _noise["delta"] = torch.randn_like(a) * float(args_cli.action_noise_std)
                else:
                    _noise["delta"] = (torch.rand_like(a) * 2.0 - 1.0) * float(args_cli.action_noise_std)
            a = a + _noise["delta"]
            # ★ 2026-11-03：这里原来抄了 `clamp(-1,1)`，是**致命 bug**——这个 WBC 的
            #   action_limit=None（训练就是无界的），实测 |a| 最大 18.6、34% 的分量超 1.0。
            #   一 clamp 到 ±1 等于把动作砍掉一大半，机器人 40 步内必倒（σ 从 0.05 降到
            #   1e-6 结果一模一样，可见跟噪声大小无关）。只有显式给 --action-clip 才截断。
            if float(getattr(args_cli, "action_clip", 0.0)) > 0:
                _c = float(args_cli.action_clip)
                a = a.clamp(-_c, _c)
        if dec_mod is not None:
            # 目标 = SONIC_DEFAULT_POS + a*G1_ACTION_SCALE_ISAACLAB，关节序 = **SONIC 序**
            # ★ 2026-11-03 深夜修正：之前写 `idx_t`（STANDARD 序）是错的。录制 npz 实测
            #   `decoder_target_action == raw*G1_ACTION_SCALE_ISAACLAB + SONIC_DEFAULT_POS`
            #   max|Δ| = 0.000000，而这两个常量都是 SONIC 序 ⇒ 目标必须按 `idx_s` 落到 isaac 关节。
            _tgt = _SDEF.unsqueeze(0) + a * _SSC.unsqueeze(0)
            _ids = [int(i) for i in idx_s]
        elif str(os.environ.get("SM_WBC_IDS", "ckpt")) == "sonic" and idx_s is not None:
            _tgt = default_s.unsqueeze(0) + a * scale_s.unsqueeze(0)
            _ids = [int(i) for i in idx_s]
        else:
            _tgt = default_q.unsqueeze(0) + a * scale.unsqueeze(0)
            _ids = [int(i) for i in idx]
        # ★ 2026-11-03：目标必须 clip 到关节限位（官方 provider 也这么做，
        #   action_provider_supermimic.py:1026）。漏了它，超限目标会让 PD 顶限位打，
        #   表现为机器人慢慢下沉（up 0.96→0.72、z 0.78→0.63），而官方闭环不会。
        try:
            _lim = robot.data.joint_pos_limits          # (n, joints, 2)，按 isaac 关节索引
            _tgt = torch.max(torch.min(_tgt, _lim[:, _ids, 1]), _lim[:, _ids, 0])
        except Exception as _le:  # noqa: BLE001
            pass
        if _ha:
            # HA env：先写完整 43 关节目标，再用它的 decimation/dt 推进物理
            _full = robot.data.joint_pos_target.clone() if hasattr(robot.data, "joint_pos_target") else None
            robot.set_joint_position_target(_tgt, joint_ids=_ids)
            for _ in range(_decim):
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(dt=_pdt)
        else:
            robot.set_joint_position_target(_tgt, joint_ids=_ids)
        # ★ 2026-10-29（第四个真 bug）：`set_joint_position_target` 只写**缓冲**，
        #   必须 `write_data_to_sim()` 才会进 PhysX。漏了它，WBC 的动作永远到不了物理，
        #   机器人只是被 USD 默认驱动僵在原地（旧采集器 `onpolicy_gt_*` 就是这样：
        #   300 步 z 恒等于 0.79、up 恒等于 1.0，其实是"没动"而不是"站得稳"）。
        if not _ha:
            robot.write_data_to_sim()
            for _ in range(4):
                sim.step()
            scene.update(sim.get_physics_dt())
        return a

    # ★ 2026-10-29 修（第二个真 bug）：观测历史**不能**全零初始化。
    #   零 = 告诉 WBC"机器人完美静止在 default_q、无角速度、重力朝正下"，
    #   而实际状态是参考帧的姿态（可能离 default_q 很远）⇒ WBC 第一拍就输出饱和动作
    #   （实测 |a|≈1.35），暖机 20 步内就歪到 31°、然后必倒（z 0.77→0.09）。
    #   真实闭环里历史是当前状态填满的，这里照做：用当前 obs 复制 HIST 帧。
    _last_a = torch.zeros(n, nj, device=DEV)
    _q0 = robot.data.joint_pos[:, idx_t]
    _dq0 = robot.data.joint_vel[:, idx_t]
    _obs0 = torch.cat((_q0 - default_q, _dq0, robot.data.root_ang_vel_b,
                       robot.data.projected_gravity_b, _last_a), dim=-1)
    _hist = _obs0.unsqueeze(1).repeat(1, HIST, 1).contiguous()
    print(f"[a-line] 观测历史用当前状态填满（|obs|={float(_obs0.abs().mean()):.3f}）", flush=True)
    _warm = int(args_cli.warmup)
    for w in range(_warm):
        _last_a = step_block(w).detach()
    # ★ 2026-11-03 真 bug：warmup 已经让机器人往前走了 _warm 步、token 也吃到 _warm，
    #   但录制循环又从头用 frame=start+0、token=start+0 ⇒ **录制第 0 行比参考落后 20 帧**，
    #   实测 step0 关节误差因此有 0.5~0.75 rad。这里把每个 env 的起点整体前移 _warm，
    #   让"状态 ↔ 参考帧 ↔ token"始终同步（代价：每条 clip 少最后 20 帧，掩码会处理）。
    for _p0 in plan:
        _p0["start"] = int(_p0["start"]) + _warm
    print(f"[a-line] warm-up {_warm} 步完成，录制起点已同步前移", flush=True)

    _seg = int(getattr(args_cli, "segment", 0) or 0)

    # ================= 段重置 / 参考历史（2026-11-04 修 bug ②③）=================
    def _quat2mat_wxyz(qq):
        """四元数 (w,x,y,z) -> 旋转矩阵（本体->世界）。"""
        w, x, y, z = qq.unbind(-1)
        ww, xx, yy, zz = w * w, x * x, y * y, z * z
        return torch.stack([
            torch.stack([ww + xx - yy - zz, 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            torch.stack([2 * (x * y + w * z), ww - xx + yy - zz, 2 * (y * z - w * x)], -1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), ww - xx - yy + zz], -1),
        ], dim=-2)

    def _ref_hist(frames_, mask=None):
        """★ bug ③：用**参考帧**重建 decoder 的 10 帧历史（帧 f0-9 … f0）。

        原来段重置只写机器人状态、**完全没有动 `_hists`** ⇒ 每段开头 decoder 看到的
        历史是"上一段的漂移状态 + 突然跳回参考"（0.07 rad 阶跃），而录制数据里的历史
        是连续的。逐块口径与 torch 版 decoder 一致：
          av = R_ref^T · w_ref（体坐标角速度）   gr = R_ref^T · [0,0,-1]
          qp = q_ref − SONIC_DEFAULT_POS         qv = qd_ref
          la = (q_ref − SONIC_DEFAULT_POS) / G1_ACTION_SCALE
        """
        if dec_mod is None or _hists is None:
            return
        n_ = len(plan)
        cs = torch.tensor([lib_starts[p["clip_idx"]] for p in plan], device=DEV, dtype=torch.long)
        k = torch.arange(HIST, device=DEV)
        hr = torch.clamp(frames_.to(DEV).view(n_, 1) + (k - (HIST - 1)), min=cs.view(n_, 1))
        R = ref[hr.cpu()].to(DEV)                       # (n,HIST,71)
        q_ = R[..., 13:13 + nj]
        qd_ = R[..., 42:42 + nj]
        w_ = R[..., 10:13]
        Rm = _quat2mat_wxyz(R[..., 3:7])                # (n,HIST,3,3)
        RmT = Rm.transpose(-1, -2)
        av_ = torch.einsum("nhij,nhj->nhi", RmT, w_)
        up_ = torch.tensor([0.0, 0.0, -1.0], device=DEV).expand(n_, HIST, 3)
        gr_ = torch.einsum("nhij,nhj->nhi", RmT, up_)
        qp_ = q_ - _SDEF.view(1, 1, -1)
        # ★ `last_action` 块要**比状态块早一帧**：录制里 obs[t] 的 la 末元素是 raw[t-1]
        #   （obs 在动作更新之前拼接）。所以这里用帧 f0-10 … f0-1 推 la。
        hr_la = torch.clamp(hr - 1, min=cs.view(n_, 1))
        q_la = ref[hr_la.cpu()].to(DEV)[..., 13:13 + nj]
        la_ = (q_la - _SDEF.view(1, 1, -1)) / _SSC.view(1, 1, -1)
        _m = torch.ones(n_, 1, 1, dtype=torch.bool, device=DEV) if mask is None \
            else mask.to(DEV).view(n_, 1, 1)
        _hists["av"].copy_(torch.where(_m, av_, _hists["av"]))
        _hists["qp"].copy_(torch.where(_m, qp_, _hists["qp"]))
        _hists["qv"].copy_(torch.where(_m, qd_, _hists["qv"]))
        _hists["la"].copy_(torch.where(_m, la_.clamp(-50.0, 50.0), _hists["la"]))
        _hists["gr"].copy_(torch.where(_m, gr_, _hists["gr"]))

    def _reset_envs(frames_, tag="", mask=None):
        """把（mask 选中的）env 瞬移回参考帧，并重建 decoder 历史。

        ``mask`` 为 None ⇒ 全部重置（固定 --segment 口径）；否则只重置 mask=True 的 env
        （缰绳口径，未选中的 env 保持当前状态不动）。
        """
        n_ = len(plan)
        _fr = torch.clamp(frames_.to(DEV).to(torch.long), max=ref.shape[0] - 1).cpu()
        _rr = ref[_fr].to(DEV)
        _mk = torch.ones(n_, dtype=torch.bool, device=DEV) if mask is None else mask.to(DEV)
        if not bool(_mk.any()):
            return 0
        # 当前状态（未重置的 env 保留它）
        _cur_pose = torch.cat((robot.data.root_pos_w, robot.data.root_quat_w), dim=-1)
        _cur_q = robot.data.joint_pos.clone()
        _cur_qd = robot.data.joint_vel.clone()
        _cur_v = torch.cat((robot.data.root_lin_vel_w, robot.data.root_ang_vel_w), dim=-1)
        _pose = torch.zeros(n_, 7, device=DEV)
        _pose[:, :3] = _rr[:, :3] + origins
        _pose[:, 3:] = _rr[:, 3:7]
        _pose = torch.where(_mk[:, None], _pose, _cur_pose)
        robot.write_root_pose_to_sim(_pose)
        _q = _cur_q.clone()
        _q[:, _init_ids] = torch.where(_mk[:, None], _rr[:, 13:13 + nj],
                                       _cur_q[:, _init_ids])
        _qd = _cur_qd.clone()
        _qd[:, _init_ids] = torch.where(_mk[:, None], _rr[:, 42:42 + nj],
                                        _cur_qd[:, _init_ids])
        robot.write_joint_state_to_sim(_q, _qd)
        _vel = torch.where(_mk[:, None], torch.cat((_rr[:, 7:10], _rr[:, 10:13]), dim=-1), _cur_v)
        robot.write_root_velocity_to_sim(_vel)
        # 段起点扰动（默认全 0）
        _sp = (float(args_cli.seg_start_perturb_xy), float(args_cli.seg_start_perturb_yaw),
               float(args_cli.seg_start_perturb_dq), float(args_cli.seg_start_perturb_v))
        if any(v > 0 for v in _sp):
            _sxy, _syaw, _sdq, _sv = _sp
            if _sxy > 0:
                _rp = robot.data.root_pos_w.clone()
                _rp[:, :2] += torch.randn(n_, 2, device=DEV) * _sxy
                _pose[:, :3] = _rp
            if _syaw > 0:
                _ang = torch.randn(n_, device=DEV) * _syaw
                _c, _s2 = torch.cos(_ang / 2), torch.sin(_ang / 2)
                _w, _x, _y, _z = _pose[:, 3:].unbind(-1)
                _pose[:, 3:] = torch.stack([_c * _w - _s2 * _z, _c * _x - _s2 * _y,
                                            _c * _y + _s2 * _x, _c * _z + _s2 * _w], dim=-1)
            if _sxy > 0 or _syaw > 0:
                robot.write_root_pose_to_sim(_pose)
            if _sdq > 0:
                _q2 = robot.data.joint_pos.clone()
                _q2[:, idx_t] += torch.randn(n_, len(idx_t), device=DEV) * _sdq
                robot.write_joint_state_to_sim(_q2, torch.zeros_like(_q2))
            if _sv > 0:
                _vv = torch.cat((_rr[:, 7:10], _rr[:, 10:13]), dim=-1)
                _vv[:, :3] += torch.randn(n_, 3, device=DEV) * _sv
                robot.write_root_velocity_to_sim(_vv)
        _ref_hist(_fr, mask=_mk)             # ★ 重建 10 帧历史（只重建被重置的 env）
        _scene_update()
        # ---- 非循环复位校验（★ 只校验**被重置的那些 env**；未重置的本来就该是漂移状态）----
        _tgt_q = _rr[:, 13:13 + nj]
        _alt = idx_t if _init_ids is not idx_t else idx_s
        _jp = robot.data.joint_pos
        if mask is not None:
            _jp = _jp[_mk]
            _tgt_q_chk = _tgt_q[_mk]
        else:
            _tgt_q_chk = _tgt_q
        _e_ok = float((_jp[:, _init_ids] - _tgt_q_chk).abs().max())
        _e_alt = float((_jp[:, _alt] - _tgt_q_chk).abs().max())
        if os.environ.get("SM_DBG_RESET"):
            print(f"[dbg-reset]{tag} 正序 max|Δ|={_e_ok:.4f} 错序 max|Δ|={_e_alt:.4f}", flush=True)
        if _e_ok > 0.05:
            raise SystemExit(f"[a-line][CONTRACT] ❌ 段重置没写进去：正序 max|Δ|={_e_ok:.4f} rad")
        if _e_alt <= max(3.0 * _e_ok, 0.05):
            raise SystemExit(f"[a-line][CONTRACT] ❌ 复位索引可能写成了错的关节序："
                             f"正序={_e_ok:.4f} 错序={_e_alt:.4f}")

    # ★★ bug ②（2026-11-04）：warm-up 让机器人从参考漂了 2–4 cm，所以"录制的第 1 行"
    #   并不是精确在参考上。这里补一次重置，让**每个窗口的第一行也是 ξ = 0**。
    _init_frames = torch.tensor([lib_starts[p["clip_idx"]] + int(p["start"]) for p in plan],
                                device=DEV, dtype=torch.long)
    _reset_envs(_init_frames, tag=" init")
    _last_a = torch.zeros(n, nj, device=DEV)      # 重置后历史里的 last_a 由参考推出
    print("[a-line] warm-up 后已重置到参考（窗口第一行 ξ=0）", flush=True)

    _leash = float(getattr(args_cli, "reset_max_drift", 0.0) or 0.0)
    if _leash > 0.0:
        print(f"[a-line] 缰绳式重置 ON：|ξ| > {_leash:.2f} m 才拉回参考", flush=True)
    _nres = [0]
    _seglen = []
    _last_reset_env = torch.zeros(n, dtype=torch.long, device=DEV)   # 每个 env 上次重置的步号
    S, A, S2, F, C, K, _E = [], [], [], [], [], [], []
    t0 = time.time()
    for step in range(int(args_cli.steps)):
        # ★★ 2026-11-03 真 bug（致命）：`step_block()` 内部就把仿真推进了一步，
        #   所以在它**之后**读到的已经是 s'。原代码把同一个 s' 同时写进 `states`
        #   和 `next_states` ⇒ 数据集里 **没有转移**（实测 max|states−next_states| = 0.0），
        #   且每行状态比它的 `frames` 标签**超前一步**（段起点绝对漂移因此有 7.6 cm）。
        #   正确做法：步前抓 s，步后抓 s'。
        _p0 = robot.data.root_pos_w - origins
        _s_before = torch.cat((_p0, robot.data.root_quat_w,
                               robot.data.joint_pos[:, idx_t],
                               robot.data.root_lin_vel_w, robot.data.root_ang_vel_w,
                               robot.data.joint_vel[:, idx_t]), dim=-1)
        a = step_block(step, noisy=True)
        q = robot.data.joint_pos[:, idx_t]
        dq = robot.data.joint_vel[:, idx_t]
        # HA 多 env：root 位置要减去各自 env 原点，否则和参考库对不上（env1 会偏 2.5 m）
        _p = robot.data.root_pos_w - origins
        state = torch.cat((_p, robot.data.root_quat_w, q,
                           robot.data.root_lin_vel_w, robot.data.root_ang_vel_w, dq), dim=-1)
        S.append(_s_before.cpu()); A.append(a.detach().reshape(n, -1).cpu())
        S2.append(state.cpu())
        F.append(torch.tensor([lib_starts[p["clip_idx"]] + p["start"] + step for p in plan],
                              dtype=torch.long))
        C.append(torch.tensor([p["clip_idx"] for p in plan], dtype=torch.long))
        _E.append(torch.arange(n, dtype=torch.long))   # ★ 2026-11-05: env 序号（配对探针拼轨迹用）
        # ★ 2026-11-03：越过 clip 末尾的行必须丢掉——tokens_at 会在末帧 clamp，
        #   而 `frames` 会指到下一条 clip 的库行 ⇒ 奖励查到别的参考。原来 A 线是
        #   每条 clip 单独一个进程（steps=该 clip 长度）所以没暴露。
        K.append(torch.tensor([p["start"] + step < p["clips"] for p in plan], dtype=torch.bool))
        _last_a = a.detach()
        # ★ 2026-11-03：分段采集 —— 每 _seg 步把机器人状态重置回**参考帧**，
        #   把漂移限制在段内（20 步 ≈ 2 cm），否则全长会漂到 10–30 cm、参考失去意义。
        #   2026-11-04：重置逻辑收进 `_reset_envs()`（写状态 + **重建 decoder 历史** + 校验）。
        if _seg > 0 and (step + 1) % _seg == 0 and (step + 1) < int(args_cli.steps):
            _fr = torch.tensor([lib_starts[p["clip_idx"]] + int(p["start"]) + step + 1
                                for p in plan], device=DEV, dtype=torch.long)
            _reset_envs(_fr, tag=f" step={step+1}")
        # ★ 2026-11-04 缰绳式重置：只重置 |ξ| > δ 的 env（段长自然 100–150 步）
        elif _leash > 0.0 and step + 1 < int(args_cli.steps):
            _fr = torch.tensor([lib_starts[p["clip_idx"]] + int(p["start"]) + step + 1
                                for p in plan], device=DEV, dtype=torch.long)
            _fr_c = torch.clamp(_fr, max=ref.shape[0] - 1).cpu()
            _xi = (robot.data.root_pos_w - origins
                   - ref[_fr_c].to(DEV)[:, 0:3]).norm(dim=-1)
            _need = _xi > _leash
            if bool(_need.any()):
                _nres[0] += int(_need.sum())
                # ★ 按 env 记录段长（之前记的是"任意 env 重置的间隔"，64 env 下会退化成 1 步）
                _seglen.extend((step - _last_reset_env[_need]).tolist())
                _last_reset_env[_need] = step
                _reset_envs(_fr, tag=f" leash step={step+1}", mask=_need)
        if step % int(args_cli.log_every) == 0:
            up = 1 - 2 * (robot.data.root_quat_w[:, 1] ** 2 + robot.data.root_quat_w[:, 2] ** 2)
            print(f"[a-line] {step}/{args_cli.steps} "
                  f"z={float(robot.data.root_pos_w[:, 2].mean()):.3f} "
                  f"|a|={float(a.abs().mean()):.3f} up={float(up.mean()):.3f} "
                  f"{time.time() - t0:.0f}s", flush=True)

    keep = torch.cat(K)
    n_drop = int((~keep).sum())
    # ★★ 2026-11-04 落盘前硬校验（锁死"数据集里没有转移"和"摔倒没人发现"两个坑）
    _Ss, _S2s = torch.cat(S), torch.cat(S2)
    _gap = float((_Ss - _S2s).abs().max())
    _zmin = float(_Ss[:, 2].min())
    _fin = bool(torch.isfinite(_Ss).all() and torch.isfinite(_S2s).all())
    print(f"[a-line][CONTRACT] 转移 max|s−s'|={_gap:.2f} | z_min={_zmin:.3f} | finite={_fin}",
          flush=True)
    if _leash > 0.0:
        _sm = float(torch.tensor(_seglen, dtype=torch.float32).median()) if _seglen else 0.0
        print(f"[a-line] leash 统计：重置 {_nres[0]} 次 / 段长中位 {_sm:.0f} 步（δ={_leash:.2f} m）",
              flush=True)
    if not _fin:
        raise SystemExit("[a-line][CONTRACT] ❌ 状态里有 NaN/Inf，拒绝落盘")
    if _gap < 1e-2:
        raise SystemExit(f"[a-line][CONTRACT] ❌ states==next_states（max|s-s'|={_gap:.2e}）"
                         " —— 又是'步后抓 s'的老 bug，数据里没有转移")
    if _zmin < 0.20:
        raise SystemExit(f"[a-line][CONTRACT] ❌ 采集过程中摔倒（z_min={_zmin:.3f}），拒绝落盘")
    out = {
        "states": torch.cat(S)[keep], "actions": torch.cat(A)[keep],
        "next_states": torch.cat(S2)[keep],
        "frames": torch.cat(F)[keep], "clip_ids": torch.cat(C)[keep],
        "env_ids": torch.cat(_E)[keep],
        "num_envs": n, "steps": int(args_cli.steps), "n_dropped_clip_end": n_drop,
        "meta": {"layout": "[p(3),quat_wxyz(4),q(29),v(3),w(3),qd(29)]",
                 "joint_names": joint_names, "exec_steps": int(args_cli.exec_steps),
                 "wbc_ckpt": str(args_cli.wbc_ckpt), "source": "a_line_wbc_only",
                 "start_mode": args_cli.start_mode, "seed": int(args_cli.seed),
                 "n_clips_used": n_clip_used, "n_clips_usable": n_usable,
                 "frames_unit": "global_library_row",
                 # A' 扰动口径（全 0 = 名义 A 线）
                 "action_noise_std": float(args_cli.action_noise_std),
                 "noise_mode": args_cli.noise_mode,
                 "perturb_hold": int(args_cli.perturb_hold),
                 "start_perturb_xy": float(args_cli.start_perturb_xy),
                 "start_perturb_yaw": float(args_cli.start_perturb_yaw),
                 "start_perturb_dq": float(args_cli.start_perturb_dq),
                 # 2026-11-03：段起点扰动（每 segment 重置后加），对齐部署的"从偏移出发"
                 "seg_start_perturb_xy": float(args_cli.seg_start_perturb_xy),
                 "seg_start_perturb_yaw": float(args_cli.seg_start_perturb_yaw),
                 "seg_start_perturb_dq": float(args_cli.seg_start_perturb_dq),
                 "seg_start_perturb_v": float(args_cli.seg_start_perturb_v),
                 # 2026-11-04 缰绳式重置（0 = 关，用 --segment）
                 "reset_max_drift": float(getattr(args_cli, "reset_max_drift", 0.0) or 0.0),
                 "n_leash_resets": int(_nres[0]),
                 "seg_len_median": (float(torch.tensor(_seglen, dtype=torch.float32).median())
                                    if _seglen else 0.0),
                 "n_dropped_clip_end": n_drop,
                 "note": "奖励未计算，见 scripts/compute_collect_reward.py"},
    }
    args_cli.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, str(args_cli.out))
    # ★ 2026-11-03：Isaac Sim 的 teardown 会挂死（实测写完盘后 4+ 分钟不退，
    #   把 4 卡并行的驱动脚本堵住）。批量采集保存完就硬退出。
    _hard_exit = str(os.environ.get("SM_HARD_EXIT", "1")) == "1"
    print(f"[a-line] saved {args_cli.out} n={out['states'].shape[0]} "
          f"clips={n_clip_used}/{n_usable} {time.time() - t0:.0f}s", flush=True)
    if _hard_exit:
        # 兜底：保存完直接退出，绕过 Isaac 的挂死式 teardown
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
