#!/usr/bin/env python3
"""WM 保鲜（S3）：**多步展开损失 + B 留出多步早停**。

为什么改成多步（2026-10-28 实测）：
    原来的刷新只做**一步**损失（``wm.step(s,a)`` vs 下一帧），一步 ``qd_corr`` 变好（0.827）
    但 **20 步关节误差反而 +39%**（0.154 → 0.214 rad）。训练用的是 H=20，所以"一步准"
    没有意义 —— 这一版把损失改成 2–4 步**自回归展开**，并且**早停看 20 步留出误差**。

口径（不变）：
    * 动作 = 录制侧 ``ref_action_lib.actions``（= 真正下发的动作，canonical；WM 的原生语义）
    * 参考库 = ``runs/isaac_motion/ha_cmd/packed.pt``（命令流）
    * 结构 = 显式物理积分（mjlab parity），写进 ckpt

验收：
    ``scripts/eval_wm_multistep.py`` 在**留出 B 片段**上的 20 步关节误差 ≤ 1.05× 上一版。

用法：
    PYTHONPATH=. python3 scripts/finetune_wm_velocity_policy.py \\
        --ckpt runs/S1_handover/WBC_A24000_wmvel_cmd.pt \\
        --out  runs/S1_handover/WBC_A24000_wmvel_v3.pt \\
        --packed runs/isaac_motion/ha_cmd/packed.pt --token-space label \\
        --rollout-steps 3 --steps 600 --lr 5e-5 --vel-lambda 5 \\
        --extra-transitions runs/wm_real/real_transitions_v2b.pt --extra-frac 0.5 \\
        --val-dump 'runs/B_*/scored/*_r.pt' --val-horizon 20 --val-every 100

结构（2026-11）：``--arch mlp``（默认，133->2048x3->70，逐位不变）/
``--arch gru``（单层 GRU(133->1024) + 头 Linear(1024,2048)/(2048,2048)/(2048,70)，
隐状态跨 K 步传递、BPTT 穿过全部 K 步；参数 +13.5%，与 MLP 基线差 ≤ 20%）。
GRU 冷启动：MLP ckpt 只复用最终层 ``net.9`` 与 in_norm/out_norm 统计量，其余从零初始化（打印明细）。
例如：``--arch gru --ckpt runs/S1_handover/WBC_A24000_wmvel_cmd.pt --out .../wmvel_v3_gru.pt``。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "trackerlab-supermimic/source/trackerLab"):
    sys.path.insert(0, str(p))

# LOCKED 2026-10-26 (S1.1)：与 evaluate.py / eval_wm_velocity.py 同一把口径锁。
if os.environ.get("SM_ALLOW_SCALE_MISMATCH") != "1":
    os.environ.setdefault("SUPERMIMIC_HA_ALIGN", "1")
    os.environ.setdefault(
        "SUPERMIMIC_ROBOT_USD",
        str(ROOT / "trackerlab-supermimic/data/assetslib/unitree/unitree_g1/usd/g1_29dof_rev_1_0.usd"))
print(f"[wmvel][LOCKED] SUPERMIMIC_HA_ALIGN={os.environ.get('SUPERMIMIC_HA_ALIGN')}", flush=True)

from diff_vla_wbc.isaac_joint import IsaacJointBundle, load_isaac_bundle, flat_state_tuple  # noqa: E402
from diff_vla_wbc.data import MotionBatchProvider, RecordedTokenSource  # noqa: E402
from diff_vla_wbc.rollout import _flat_history, compute_joint_loss, imagine_with_tokens, LossWeights  # noqa: E402
from trackerLab.algorithms.supermimic.networks import RunningNorm, WorldModel, mlp  # noqa: E402
from trackerLab.kinematics.supermimic.rotations import so3_exp, so3_log, yaw_mat  # noqa: E402
from trackerLab.tracker_env.mdp.supermimic import NUM_J  # noqa: E402


# --------------------------------------------------------------------------- #
# WM 结构开关（2026-11）：``--arch {mlp,gru}`` —— 定义在 diff_vla_wbc/wm_arch.py
# --------------------------------------------------------------------------- #
# 结构代码必须和**评估**共用同一份（``isaac_joint.load_isaac_bundle`` 也 import 它），
# 否则训练出 GRU 的 ckpt、裁判按 MLP 造模型 ⇒ 权重键 mismatch；更糟的情况是两边悄悄不一致。
from diff_vla_wbc.wm_arch import (  # noqa: E402
    WM_GRU_HEAD,
    WM_GRU_HIDDEN,
    WM_MLP_BASELINE_PARAMS,
    WM_PARAM_TOL,
    GRUWorldModel,
    build_wm_arch,
    detect_wm_arch,
    is_gru_state_dict,
    load_wm_weights,
    report_wm_arch,
    wm_param_count,
)


def _mag_pen(pred, tgt, eps: float = 1e-6):
    """**幅度比**惩罚：``(log mean|pred| − log mean|tgt|)²``。

    为什么需要它：现有损失只惩罚**逐元素 L2**，而 L2 训练出的确定性模型会回归**条件均值**，
    速度通道的幅度因此系统性偏小（regression-to-mean），视界越长越明显——实测 in-dist
    `root_w_ratio` k=1 0.80 / k=5 **0.61**（想象转速只有真实的 61%）。
    相关系数可以靠加权救（已从 0.14 提到 0.37），但幅度不会自己回来，所以单独加这一项。
    """
    a = pred.abs().mean()
    b = tgt.abs().mean()
    return (torch.log(a + eps) - torch.log(b + eps)).pow(2)


def _pos_vel(out, tgt, vel_lambda: float, w_lambda: float = 1.0,
             vel_scale_lambda: float = 0.0):
    """位置/朝向/关节 + 速度（归一化）误差；与旧版一步损失同口径。

    ★ 2026-10-31（A2 的前置条件）：**root 角速度 ``w`` 单独加权**。
      原来 ``vel`` 是 ``v / w / qd`` **三项等权**，而 ``w`` 只有 3 维、``qd`` 有 29 维，
      模型把力气都花在关节上 ⇒ 刷新后 `root_w_corr` 从 0.500 掉到 0.261、
      `root_w_ratio` 1.11→1.82（幅度高估 82%）。
      而奖励里的 yaw_rate 项正是吃这个量。所以给 ``w`` 加 ``w_lambda``（默认 3.0）。
      **不是"只优化 yaw"**：pos 三项、v、qd 全部保留，只是把被 3 维小项挤掉的权重补回来。
    """
    pos = ((out[0] - tgt[0]).square().mean()
           + so3_log(out[1].transpose(-1, -2) @ tgt[1]).square().mean()
           + (out[4] - tgt[4]).square().mean())
    sv = tgt[2].std(0).mean() + 1e-6
    sw = tgt[3].std(0).mean() + 1e-6
    sq = tgt[5].std(0).mean() + 1e-6
    vel = ((out[2] - tgt[2]).square().mean() / sv ** 2
           + float(w_lambda) * (out[3] - tgt[3]).square().mean() / sw ** 2
           + (out[5] - tgt[5]).square().mean() / sq ** 2)
    # 三个速度通道的**幅度比**都要守住（不只 w，避免"只优化 yaw"）
    scale = (_mag_pen(out[2], tgt[2]) + _mag_pen(out[3], tgt[3])
             + _mag_pen(out[5], tgt[5])) / 3.0
    return pos, vel, pos + vel_lambda * vel + float(vel_scale_lambda) * scale


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=ROOT / "runs/S1_handover/WBC_A24000_wmvel_cmd.pt")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "runs/S1_handover/WBC_A24000_wmvel_v3.pt")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--vel-lambda", type=float, default=5.0)
    ap.add_argument("--vel-scale-lambda", type=float, default=1.0,
                    help="三个速度通道的**幅度比**惩罚权重（(log mean|pred| − log mean|tgt|)²）。"
                         "L2 会让幅度回归均值 ⇒ 视界越长越小（实测 k=5 root_w_ratio 0.61）。0=关")
    ap.add_argument("--w-lambda", type=float, default=3.0,
                    help="A2 前置：root 角速度 w 的额外权重（原来 v/w/qd 三项等权，"
                         "w 只有 3 维被 29 维的 qd 挤掉）。1.0=旧行为")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--packed", type=Path, default=ROOT / "runs/isaac_motion/ha_cmd/packed.pt",
                    help="参考库。S3 必须用命令流 ha_cmd")
    ap.add_argument("--npz-dir", type=Path, default=ROOT / "data/npz/ha_tokens")
    ap.add_argument("--port-dir", type=Path, default=None)
    ap.add_argument("--token-space", choices=("auto", "label", "port"), default="auto")
    ap.add_argument("--ref-action-lib", type=Path, default=ROOT / "runs/wm_real/ref_action_lib.pt",
                    help="录制侧动作（= 真正下发的动作，WM 原生语义）")
    ap.add_argument("--action-source", choices=("lib", "policy"), default="lib",
                    help="lib=用录制动作（推荐，与 WM 训练数据一致）；policy=用当前 WBC 的输出")
    ap.add_argument("--rollout-steps", type=int, default=2,
                    help="多步展开步数 K（1=旧的一步损失；建议 2–4）")
    ap.add_argument("--arch", choices=("mlp", "gru"), default="mlp",
                    help="WM 结构。mlp=现有 133->2048x3->70（默认，与旧行为逐位一致）；"
                         "gru=单层 GRU(133->1024) + 同形状头(1024->2048->2048->70)，"
                         "跨 K 步保留隐状态并 BPTT 穿过全部 K 步")
    ap.add_argument("--clip-norm", type=float, default=1.0,
                    help="梯度裁剪范数（BPTT 安全阀；默认 1.0 = 旧行为）")
    ap.add_argument("--extra-transitions", type=Path, default=None,
                    help="real_transitions_v2.pt：把 B 线 on-policy 的 (s,a,s') 混进 WM 刷新")
    ap.add_argument("--extra-frac", type=float, default=0.5,
                    help="每个 batch 里 B 线 on-policy 样本的占比（默认 0.5）")
    ap.add_argument("--mix-lib", type=float, default=-1.0,
                    help="A8：一整批里**参考动作库**(prov.sample 的 packed.pt)占比。<0=沿用旧行为 "
                         "(`--traj-frac` 追加法)；0=完全不用参考库（只用 A/B 线）")
    ap.add_argument("--mix-b", type=float, default=0.5,
                    help="A8：A/B 线那一块里 **B 线**占比（0.5 = A:B 各半）")
    ap.add_argument("--pert-dump", default="",
                    help="Bw_pert（B 分布 + token 扰动）的 glob。**只进 WM 训练**，不进 B 池")
    ap.add_argument("--mix-pert", type=float, default=0.0,
                    help="整批里扰动数据的占比（0=关）。P 与 A/B 并列，用于补\"token 换一点会怎样\"")
    ap.add_argument("--pert-seg-len", type=int, default=20,
                    help="扰动数据的段长：多步窗口不得跨段（跨段目标无意义）")
    ap.add_argument("--a-order", default="ha_v31", choices=("ha_v31", "declared"),
                    help="A 线文件里关节的存储顺序：ha_v31=沿用旧行为（对 A 也做 ha_v31→standard）；"
                         "declared=A 线 meta 自称 STANDARD（实测 corr 0.62），**不再换序**")
    ap.add_argument("--traj-dump", default="",
                    help="glob：B 线**长轨迹** dump（states/q_target_applied）。多步损失真正需要的数据域"
                         "——录制库是 on-reference，B 才是 off-reference")
    # ★ 2026-10-29：补 mjlab 那条线的「第 1 步」——A 线 = 冻结 WBC 在 Isaac 真物理里
    #   向量化走出来的真 (s,a,s')。我们原来 95% 训练数据是运动学参考（packed 的 ref + 录制动作），
    #   导致 WM 的 qd_corr 只有 0.743、qd_ratio 0.948（系统性低估速度 5%）。见 docs 附录 E。
    ap.add_argument("--a-line", type=str, default="",
                    help="glob：A 线批量采集（WBC-only、Isaac 真物理）产物。"
                         "里面的 actions **已是归一化动作**、states 是 HA_V31 序，"
                         "按 num_envs 解交错后按 env 切成轨迹，直接参与多步损失。")
    ap.add_argument("--val-max-traj", type=int, default=24,
                    help="验证时最多用多少条轨迹（A 线动辄上千条，全跑一遍要十几分钟）")
    ap.add_argument("--a-line-envs", type=int, default=256,
                    help="每个 A 线文件最多取多少个 env（控制内存）")
    ap.add_argument("--traj-frac", type=float, default=0.5,
                    help="batch 里 B 轨迹窗口的占比")
    ap.add_argument("--traj-val-frac", type=float, default=0.25,
                    help="每条 B 轨迹**尾部**留出的比例（按时间切，防泄漏）")
    ap.add_argument("--val-dump", default="", help="glob：留出 B dump（多步早停用）")
    ap.add_argument("--val-horizon", type=int, default=20)
    ap.add_argument("--val-stride", type=int, default=20)
    ap.add_argument("--val-max-starts", type=int, default=64, help="0=全部重规划时刻")
    ap.add_argument("--val-every", type=int, default=100, help="0=不早停，只报首尾")
    ap.add_argument("--no-explicit-integration", action="store_true",
                    help="保持旧结构（整步增量）；默认启用 mjlab 的半隐式显式积分")
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device(a.device)
    arch = str(a.arch).lower()
    # ★ 2026-11-01 接管修正：`load_isaac_bundle` 现在**自己按 ckpt 判断 arch**
    #   （diff_vla_wbc/wm_arch.detect_wm_arch），所以续训 GRU ckpt 也走它 —— 不再用
    #   `strict=False` 糊过去（那样 pi 缺键也会被静默吞掉）。--arch mlp 路径逐位不变。
    bundle = load_isaac_bundle(a.ckpt, device=dev,
                               npz=sorted(Path(a.npz_dir).glob("*.npz")),
                               packed_motion=Path(a.packed),
                               packed_dir=None, obs_mode=None, hist=10, token_dim=64)
    if arch == "gru":
        bundle.wm = build_wm_arch("gru", num_joints=bundle.num_joints)
        load_wm_weights(bundle.wm, "gru", a.ckpt)
        bundle.wm.to(dev).eval()
        from types import MethodType  # noqa: PLC0415
        from diff_vla_wbc.isaac_joint import trackerlab_window_loss  # noqa: PLC0415

        bundle.wm.window_loss = MethodType(trackerlab_window_loss, bundle.wm)
    report_wm_arch(arch, bundle.wm)
    bundle.wm.explicit_integration = not a.no_explicit_integration
    bundle.wm.dt = 0.02
    wm_integration = ("explicit_dt%g" % bundle.wm.dt) if bundle.wm.explicit_integration else "residual"
    bundle.wm_integration = wm_integration
    print(f"[structure] ws_integration={wm_integration}", flush=True)
    lib = bundle.library
    rec = RecordedTokenSource(Path(a.npz_dir), lib, token_horizon=40, token_dim=64, device=dev,
                              port_dir=a.port_dir, token_space=a.token_space)
    prov = MotionBatchProvider(bundle, lib, batch_size=a.batch, horizon=40, token_source=None,
                               token_dim=64, token_horizon=40, seed=12345, hand_dir=None)
    wm = bundle.wm
    dflt = bundle.default_joint_pos.float().to(dev)
    K = max(1, int(a.rollout_steps))

    # ---- 录制动作库（WM 原生动作语义）----
    lib_act = None
    if K > 1 and a.action_source != "lib":
        raise SystemExit("[loss] K>1 需要 --action-source lib（policy 源拿不到后续步的动作）")
    if a.action_source == "lib" or K > 1:
        _la = torch.load(str(a.ref_action_lib), map_location="cpu", weights_only=False)
        lib_act = _la["actions"].float().to(dev)
        assert lib_act.shape[0] == lib.num_frames, \
            f"ref_action_lib 行数 {lib_act.shape[0]} != 库帧数 {lib.num_frames}"
        print(f"[action] lib={a.ref_action_lib} n={lib_act.shape[0]} "
              f"mean={float(lib_act.mean()):+.4f} std={float(lib_act.std()):.3f}", flush=True)

    # ---- B 线 on-policy 混合 ----
    ex = None
    if a.extra_transitions is not None:
        _d = torch.load(str(a.extra_transitions), map_location="cpu", weights_only=False)
        _e = _d.get("extra") or _d
        if not all(k in _e for k in ("states", "actions", "next_states")):
            raise SystemExit(f"[extra] {a.extra_transitions} 缺 states/actions/next_states")
        ex = {k: _e[k].float() for k in ("states", "actions", "next_states")}
        if ex["states"].shape[0] < 8:
            raise SystemExit(f"[extra] on-policy 样本只有 {ex['states'].shape[0]} 条，太少")
        print(f"[extra] on-policy 转移 n={ex['states'].shape[0]} frac={a.extra_frac}", flush=True)
    g_ex = torch.Generator(device="cpu")
    g_ex.manual_seed(20261028)

    # ---- 留出 B（多步早停）----
    val = []
    if a.val_dump:
        import joint_contract as JC  # noqa: PLC0415
        from diff_vla_wbc.b_start import state_to_standard  # noqa: PLC0415

        perm = torch.as_tensor(JC.permutation("ha_v31", "standard"), dtype=torch.long)
        for f in sorted(glob.glob(a.val_dump)):
            d = torch.load(f, map_location="cpu", weights_only=False)
            if "q_target_applied" not in d:
                print(f"[val] SKIP {Path(f).name}（没有 q_target_applied）", flush=True)
                continue
            val.append({"name": Path(f).name,
                        "states": state_to_standard(d["states"].float(), perm),
                        "q_tgt": d["q_target_applied"].float().reshape(-1, 29)})
        print(f"[val] 留出 B：{len(val)} 个 dump，horizon={a.val_horizon}", flush=True)
        if ex is not None and val:
            print("[val] WARN: --val-dump 与 --extra-transitions 可能重叠；"
                  "早停会被泄漏（终裁请用另一条 B episode）", flush=True)

    # ---- B 长轨迹（多步损失的数据域）----
    traj = []
    # ★ 2026-10-29 修（第五个真 bug）：perm / state_to_standard 原来只在 `if a.traj_dump:` 里定义，
    #   而 A 线的加载**也被嵌在**那个 if 里 ⇒ A 段（`--traj-dump ""`）根本没加载 A 线，
    #   等于只训了运动学库（两次跑出逐位相同的 loss 就是证据）。现在把两者都提到顶层。
    import joint_contract as JC  # noqa: PLC0415
    from diff_vla_wbc.b_start import state_to_standard  # noqa: PLC0415

    perm = torch.as_tensor(JC.permutation("ha_v31", "standard"), dtype=torch.long)

    if a.traj_dump:
        n_b = 0
        for f in sorted(glob.glob(a.traj_dump)):
            d = torch.load(f, map_location="cpu", weights_only=False)
            if "q_target_applied" not in d:
                continue
            S = state_to_standard(d["states"].float(), perm)
            A = (d["q_target_applied"].float().reshape(-1, 29) - dflt.cpu())
            n = int(S.shape[0])
            n_tr = max(K + 20, int(n * (1.0 - float(a.traj_val_frac))))
            traj.append({"name": Path(f).name, "S": S.to(dev), "A": A.to(dev),
                         "train": (10, n_tr), "val": (n_tr, n)})
            n_b += 1
        if n_b:
            print(f"[traj] B 轨迹 {n_b} 条（离线闭环 dump）", flush=True)

    # ---- A 线（WBC-only 真物理）：actions 已归一化，按 num_envs 解交错 ----
    if a.a_line:
        n_a = 0
        for f in sorted(glob.glob(a.a_line)):
            d = torch.load(f, map_location="cpu", weights_only=False)
            if "states" not in d or "actions" not in d:
                print(f"[a-line] SKIP {Path(f).name}：缺 states/actions", flush=True)
                continue
            n_env = int(d.get("num_envs", 1))
            # ★★ 2026-10-31 定案（**这是一个真 bug，不是口味问题**）：
            #   A 线文件 meta.joint_names **就是** spec/standard 序（逐项相同），实测：
            #     corr(mean q, default_q) = +0.698（原样） / +0.034（过 ha_v31→standard 换序）
            #     corr(mean action, mean q−default_q) = +0.832（原样） / +0.055（换序后）
            #   ⇒ 换序会把 **states 打乱而 actions 不动**，破坏"动作 j → 关节 j"的因果对应；
            #     而 B 线 dump 恰好相反（states 存 ha_v31、动作是 standard，已实测）。
            #   所以 `--a-order declared` = 不换序（正确）；`ha_v31` 只为复现历史数字而保留。
            _order = str(getattr(a, "a_order", "ha_v31"))
            _Sraw = d["states"].float()
            Sf = state_to_standard(_Sraw, perm) if _order == "ha_v31" else _Sraw
            # ---- 防线（防"顺手换序"再次静默发生）：用第一个文件量两种顺序谁与 default_q 同号 ----
            if not getattr(main, "_order_checked", False):
                main._order_checked = True
                _q = _Sraw[:, 7:36] if _Sraw.ndim == 2 else _Sraw.reshape(-1, _Sraw.shape[-1])[:, 7:36]
                _dq = dflt.detach().cpu()

                def _c(x):
                    x = x - x.mean()
                    return float((x * (_dq - _dq.mean())).sum()
                                 / (x.norm() * (_dq - _dq.mean()).norm() + 1e-12))

                _chosen = _c(_q.mean(0)) if _order == "declared" else _c(_q[:, perm].mean(0))
                _alt = _c(_q.mean(0)) if _order == "ha_v31" else _c(_q[:, perm].mean(0))
                print(f"[a-order] 选用 {_order}: corr(mean q, default_q)={_chosen:+.3f}"
                      f"（另一种顺序 {_alt:+.3f}）", flush=True)
                if _alt > 0.30 and _chosen < _alt - 0.15:
                    raise SystemExit(
                        f"[a-order] 选错顺序了：{_order} 的相关性 {_chosen:+.3f} 明显低于另一种 "
                        f"{_alt:+.3f} ⇒ 会把 A 线的 states 打乱而 actions 不动（因果错位）。"
                        f"请改用另一种 --a-order。")
            T = int(Sf.shape[0]) // max(1, n_env)
            if T < K + 25:
                print(f"[a-line] SKIP {Path(f).name}：T={T} 太短", flush=True)
                continue
            Sf = Sf[: T * n_env].reshape(T, n_env, -1).permute(1, 0, 2)     # (n_env,T,71)
            Af = d["actions"].float()[: T * n_env].reshape(T, n_env, -1).permute(1, 0, 2)
            vlen = d.get("valid_len")          # 清洗脚本给的"每条 env 的干净前缀"
            take = min(n_env, int(a.a_line_envs) if a.a_line_envs > 0 else n_env)
            n_skip = 0
            for e in range(take):
                if vlen is not None:
                    _v = int(vlen[e])
                    if _v < K + 25:            # 太短，切不出 "训练窗 + 验窗"
                        n_skip += 1
                        continue
                    S_e, A_e = Sf[e, :_v], Af[e, :_v]
                else:
                    S_e, A_e = Sf[e], Af[e]
                _n = int(S_e.shape[0])
                # ★ 2026-10-29 修（第六个真 bug）：`max(K+20, …)` 在短轨迹上会**超过** _n，
                #   于是 train=(10,_tr) 的 hi>_n ⇒ `t["A"][idx+k]` 越界，
                #   CUDA 直接 `index < sizes[i]` 断言崩掉（A 段 200 步就是这么死的）。
                _tr = min(_n - 1, max(K + 20, int(_n * (1.0 - float(a.traj_val_frac)))))
                if _tr <= 10 + K:
                    n_skip += 1
                    continue
                traj.append({"name": f"{Path(f).stem}#e{e}", "S": S_e.to(dev),
                             "A": A_e.to(dev), "train": (10, _tr), "val": (_tr, _n)})
                n_a += 1
            if n_skip:
                print(f"[a-line] {Path(f).name}: 跳过 {n_skip} 条（清洗后太短）", flush=True)
        if n_a:
            print(f"[a-line] 加载 {n_a} 条 WBC-only 轨迹（归一化动作，真物理）", flush=True)

    # ---- P：Bw_pert（B 起点 + token 扰动；反事实样本）----
    traj_p = []
    if getattr(a, "pert_dump", "") and float(getattr(a, "mix_pert", 0.0)) > 0.0:
        n_p = 0
        _seg_default = int(getattr(a, "pert_seg_len", 20) or 20)
        for f in sorted(glob.glob(a.pert_dump)):
            d = torch.load(f, map_location="cpu", weights_only=False)
            if "q_target_applied" not in d:
                print(f"[pert] SKIP {Path(f).name}（没有 q_target_applied）", flush=True)
                continue
            S = state_to_standard(d["states"].float(), perm)      # 我们产出=ha_v31，正确
            A = (d["q_target_applied"].float().reshape(-1, 29) - dflt.cpu())
            seg = int((d.get("meta") or {}).get("seg_len") or _seg_default)
            n = int(S.shape[0])
            # 多步窗口不能跨段：预先算好所有合法起点
            _ok = [i for i in range(0, n - K) if (i % max(1, seg)) + K < seg]
            if not _ok:
                continue
            traj_p.append({"name": "pert:" + Path(f).stem, "S": S.to(dev), "A": A.to(dev),
                           "train": (0, n), "val": (n, n), "seg": max(1, seg),
                           "ok": torch.tensor(_ok, dtype=torch.long, device=dev)})
            n_p += 1
        if n_p:
            print(f"[pert] 载入 {n_p} 个扰动文件（seg={_seg_default}，窗口不跨段）", flush=True)
        else:
            print("[pert] WARN: --mix-pert>0 但没载入任何扰动文件", flush=True)

    traj_a, traj_b = [], []
    if traj:
        # A 线的名字带 '#'（见上面的命名规则），B 线不带 —— 这是仓库里既有的判据
        traj_a = [t for t in traj if "#" in t["name"]]
        traj_b = [t for t in traj if "#" not in t["name"]]
        print(f"[traj] 合计 {len(traj)} 条轨迹（B={len(traj_b)} A={len(traj_a)}）", flush=True)

    @torch.no_grad()
    def val_multistep() -> float:
        """留出 B 上滚 val_horizon 步的关节 RMSE（越大越差）。"""
        if (not val and not traj) or a.val_horizon <= 0:
            return float("nan")
        H = int(a.val_horizon)
        errs = []
        if traj:
            # ★ 2026-10-29：轨迹多时验证极慢（885 条 × 20 步 ≈ 1.7 万次前向，>10 min）。
            #   只取前 val_max_traj 条，足够做早停/诊断。
            for t in traj[: max(1, int(getattr(a, "val_max_traj", 24) or 24))]:
                lo, hi = t["val"]
                # ★ 2026-10-29 修（第七个真 bug）：验证段比 H 还短时，`idx + H` 直接越界
                #   （A 线里 30 步的短轨迹就是这样把 CUDA 断言打崩的）。跳过即可。
                if int(hi) - int(lo) <= int(H) + 1 or int(hi) > int(t["S"].shape[0]):
                    continue
                starts = list(range(max(lo, 10), max(lo + 1, hi - H), max(1, int(a.val_stride))))
                if a.val_max_starts:
                    starts = starts[: int(a.val_max_starts)]
                if not starts:
                    continue
                idx = torch.tensor(starts, device=dev)
                s = t["S"][idx]
                # --arch gru：验证也要**带记忆**滚 H 步（隐状态从零开始，和训练窗口一致）
                if arch == "gru":
                    bundle.wm.begin_unroll(int(idx.shape[0]), device=dev)
                try:
                    for k in range(H):
                        s = bundle.step(s, t["A"][idx + k],
                                        torch.zeros(s.shape[0], 33, device=dev), step_idx=k + 1)
                finally:
                    if arch == "gru":
                        bundle.wm.end_unroll()
                errs.append(float((s[:, 7:36] - t["S"][idx + H][:, 7:36]).pow(2).mean().sqrt()))
            return sum(errs) / max(1, len(errs))
        for d in val:
            S = d["states"].to(dev)
            Qt = d["q_tgt"].to(dev)
            T = int(S.shape[0])
            starts = list(range(10, T - H - 1, max(1, int(a.val_stride))))
            if a.val_max_starts:
                starts = starts[: int(a.val_max_starts)]
            if not starts:
                continue
            idx = torch.tensor(starts, device=dev)
            s = S[idx]
            if arch == "gru":
                bundle.wm.begin_unroll(int(idx.shape[0]), device=dev)
            try:
                for k in range(H):
                    a_k = (Qt[idx + k] - dflt)
                    s = bundle.step(s, a_k, torch.zeros(s.shape[0], 33, device=dev),
                                    step_idx=k + 1)
            finally:
                if arch == "gru":
                    bundle.wm.end_unroll()
            errs.append(float((s[:, 7:36] - S[idx + H][:, 7:36]).pow(2).mean().sqrt()))
        return sum(errs) / max(1, len(errs))

    def _line_rows(n_a: int, n_b: int, n_p: int = 0):
        """从 A/B/P 轨迹里各抽 n_a / n_b / n_p 个起始帧 → (S, acts, targets)。

        P（Bw_pert）的窗口**不得跨段**：段内只有 seg_len 步，跨段的多步目标无意义。
        """
        Ss, As, Gs = [], [], []
        for lst, cnt in ((traj_a, int(n_a)), (traj_b, int(n_b)), (traj_p, int(n_p))):
            if cnt <= 0 or not lst:
                continue
            got = 0
            _is_pert = bool(lst is traj_p)
            while got < cnt:
                ti = torch.randint(0, len(lst), (1,), generator=g_ex).item()
                t = lst[ti]
                lo, hi = t["train"]
                hi = max(lo + K + 1, hi)
                take = min(cnt - got, max(1, hi - lo - K))
                if _is_pert:
                    _ok = t["ok"]
                    _pick = torch.randint(0, int(_ok.numel()), (min(take, int(_ok.numel())),),
                                          generator=g_ex)
                    tt = _ok[_pick]
                else:
                    tt = torch.randint(lo, hi - K, (take,), generator=g_ex).to(dev)
                if tt.numel() == 0:
                    break
                Ss.append(flat_state_tuple(t["S"][tt]))
                As.append(torch.stack([t["A"][tt + k] for k in range(K)], 1))
                Gs.append([flat_state_tuple(t["S"][tt + k + 1]) for k in range(K)])
                got += int(tt.shape[0])
                if len(lst) == 1 and got == 0:
                    break
        if not Ss:
            raise SystemExit("[mix] A/B 线一条都用不上（检查 --traj-dump / --a-line 与 traj-frac）")
        S_ = tuple(torch.cat([x[k] for x in Ss], 0) for k in range(6))
        A_ = torch.cat(As, 0)
        # Gs 是 (chunk, k) 两层：Gs[c][k] 是 6-tuple；这里按 horizon k 把各 chunk 拼起来
        G_ = [tuple(torch.cat([Gs[c][k][j] for c in range(len(Gs))], 0)
                    for j in range(6)) for k in range(K)]
        return S_, A_, G_

    def one_batch():
        b = prov.sample()
        n = int(b.frames.shape[0])
        frames = b.frames.to(dev)
        S = flat_state_tuple(b.start_state)
        # 多步目标：teacher_states[:, k]（k=1..K）
        tgts = [flat_state_tuple(b.teacher_states[:, k]) for k in range(K)]
        if lib_act is not None:
            _end = lib.clip_end[lib.clip_id[b.frames]].to(dev)
            idx = torch.stack([frames + k for k in range(K)], 1)
            idx = torch.minimum(idx, _end[:, None])
            acts = torch.stack([lib_act[idx[:, k].long()].to(dev) for k in range(K)], 1)
        else:
            tok = rec({"frames": b.frames})
            obs = _flat_history(bundle, b.history)
            acts = None
            with torch.no_grad():
                a0 = bundle.act(obs, tok[:, 0], ref=b.references[0])
            acts = a0[:, None, :]
        mix = float(getattr(a, "mix_lib", -1.0))
        if mix >= 0.0:
            # ---- A8：显式配比（整批 = lib + A + B，三者占比由 --mix-lib/--mix-b 决定）----
            n_tot = int(a.batch)
            n_lib = int(round(n_tot * mix))
            n_line = n_tot - n_lib
            _ha, _hb = bool(traj_a), bool(traj_b)
            _hp = bool(traj_p)
            _mp = float(getattr(a, "mix_pert", 0.0) or 0.0)
            n_p = int(round(n_line * _mp)) if (_hp and _mp > 0.0) else 0
            _rest = n_line - n_p
            if _ha and _hb:
                n_b = int(round(_rest * float(a.mix_b)))
                n_a = _rest - n_b
            elif _ha:
                n_a, n_b = _rest, 0
            elif _hb:
                n_a, n_b = 0, _rest
            else:
                raise SystemExit("[mix] A/B 线一条都没有：--traj-dump/--a-line 都没给")
            if _mp > 0.0 and not _hp:
                raise SystemExit(f"[mix] --mix-pert={_mp:g} > 0 但没有可用扰动数据（--pert-dump）")
            if n_lib < n:
                S = tuple(x[:n_lib] for x in S)
                acts = acts[:n_lib]
                tgts = [tuple(x[:n_lib] for x in tg) for tg in tgts]
            if n_line > 0:
                S_l, A_l, G_l = _line_rows(n_a, n_b, n_p)
                if n_lib > 0:
                    S = tuple(torch.cat([x, y], 0) for x, y in zip(S, S_l))
                    acts = torch.cat([acts, A_l], 0)
                    tgts = [tuple(torch.cat([x, y], 0) for x, y in zip(tgts[k], G_l[k]))
                            for k in range(K)]
                else:
                    S, acts, tgts = S_l, A_l, G_l
            if not getattr(one_batch, "_reported", False):
                one_batch._reported = True
                tot = max(1, n_lib + n_a + n_b + n_p)
                print(f"[mix] 整批配比: 参考库 {n_lib} ({100*n_lib/tot:.1f}%) / "
                      f"A 线 {n_a} ({100*n_a/tot:.1f}%) / B 线 {n_b} ({100*n_b/tot:.1f}%) / "
                      f"P 扰动 {n_p} ({100*n_p/tot:.1f}%) "
                      f"⇒ 总 {tot}（--mix-lib={mix:g} --mix-b={float(a.mix_b):g} "
                      f"--mix-pert={_mp:g}）", flush=True)
                got_lib = n_lib / tot
                if abs(got_lib - mix) > 0.02:
                    raise SystemExit(f"[mix] 参考库占比 {got_lib:.3f} != 请求 {mix:.3f}（配比没生效）")
                got_p = n_p / tot
                if _mp > 0.0 and abs(got_p - _mp) > 0.03:
                    raise SystemExit(f"[mix] 扰动占比 {got_p:.3f} != 请求 {_mp:.3f}（配比没生效）")
        elif traj and a.traj_frac > 0:
            kk = max(1, int(round(n * float(a.traj_frac))))
            for _ in range(1):
                ti = torch.randint(0, len(traj), (1,), generator=g_ex).item()
                t = traj[ti]
                lo, hi = t["train"]
                hi = max(lo + K + 1, hi)
                tt = torch.randint(lo, hi - K, (kk,), generator=g_ex).to(dev)
                S_t = flat_state_tuple(t["S"][tt])
                A_t = torch.stack([t["A"][tt + k] for k in range(K)], 1)
                G_t = [flat_state_tuple(t["S"][tt + k + 1]) for k in range(K)]
            S = tuple(torch.cat([x, y], 0) for x, y in zip(S, S_t))
            acts = torch.cat([acts, A_t], 0)
            tgts = [tuple(torch.cat([x, y], 0) for x, y in zip(tgts[k], G_t[k])) for k in range(K)]
        if ex is not None and a.extra_frac > 0:
            if K > 1:
                # extra 里只有 **一步** (s,a,s')；多步目标不存在，硬拼会教 WM"零动力学"。
                # 所以 K>1 时 on-policy 只用于 1-step 诊断，不参与多步损失（数据只占 0.3%）。
                if not getattr(one_batch, "_warned", False):
                    print("[extra] K>1：跳过 on-policy 混合（它只有一步目标）", flush=True)
                    one_batch._warned = True
            else:
                kk = max(1, int(round(n * float(a.extra_frac))))
                j = torch.randint(0, int(ex["states"].shape[0]), (kk,), generator=g_ex)
                S = tuple(torch.cat([x, y], 0) for x, y in zip(S, flat_state_tuple(ex["states"][j].to(dev))))
                acts = torch.cat([acts, ex["actions"][j].to(dev)[:, None, :]], 0)
                tgts[0] = tuple(torch.cat([x, y], 0) for x, y in zip(
                    tgts[0], flat_state_tuple(ex["next_states"][j].to(dev))))
        return S, acts, tgts

    def eval_wm(n=8):
        S, acts, tgts = one_batch()
        xi = torch.zeros(int(S[0].shape[0]), 33, device=dev)
        with torch.no_grad():
            out = wm.step(S, acts[:, 0], xi)
        S1 = tgts[0]
        corr = float(((out[5] - out[5].mean()) * (S1[5] - S1[5].mean())).mean()
                     / (out[5].std() * S1[5].std() + 1e-9))
        return {"qd_corr": corr,
                "qd_ratio": float(out[5].abs().mean() / (S1[5].abs().mean() + 1e-9)),
                "joint_mse": float((out[4] - S1[4]).square().mean()),
                "pos_mse": float((out[0] - S1[0]).square().mean())}

    def policy_eval():   # 真正关心的指标：策略 rollout
        b = prov.sample()
        tok = rec({"frames": b.frames})
        W = LossWeights(); W.state_vel = 1.0
        # --arch gru：想象 rollout 里也保留隐状态（imagine_with_tokens 内部逐步调 bundle.step）
        _gru = arch == "gru"
        if _gru:
            bundle.wm.begin_unroll(int(tok.shape[0]), device=dev)
        try:
            with torch.no_grad():
                ro = imagine_with_tokens(bundle, tokens=tok, state=b.start_state, obs_hist=b.obs_hist,
                                         history=b.history, xi=b.xi, references=b.references,
                                         prev_action=b.prev_action, horizon=8, obs_noise=0.0)
                _, t = compute_joint_loss(ro, teacher_states=b.teacher_states, gamma=0.99, weights=W)
        finally:
            if _gru:
                bundle.wm.end_unroll()
        return float(t["state"]), float(ro.rewards.mean())

    print(f"[loss] K={K} action={a.action_source}")
    print("[before wm] " + json.dumps({k: round(v, 4) for k, v in eval_wm().items()}))
    v0 = val_multistep()
    if v0 == v0:
        print(f"[before val] B {a.val_horizon} 步关节 RMSE = {v0:.4f} rad")
    se, rw = policy_eval()
    print(f"[before policy] state_term={se:.4f} reward={rw:+.5f}", flush=True)
    params = list(wm.parameters())
    opt = torch.optim.Adam(params, lr=a.lr)
    # --arch gru：BPTT 前 200 步做线性 lr warmup（手写 lr 斜坡，不用 scheduler，
    # 避免 PyTorch 版本间 "scheduler 第一格被跳过" 的差异）；MLP 路径不动 lr，逐位不变。
    lr_warm = int(min(200, max(0, a.steps))) if arch == "gru" else 0
    if lr_warm > 1:
        print(f"[warmup] arch=gru lr 线性 warmup {lr_warm} 步（第 1 步 {a.lr / lr_warm:g}）"
              f" → {a.lr:g}", flush=True)
    best = {"val": v0 if v0 == v0 else float("inf"),
            "state": {k: v.detach().cpu().clone() for k, v in wm.state_dict().items()},
            "it": 0}
    for it in range(1, a.steps + 1):
        # ★ 2026-11-01 接管修（**真 bug**，冒烟第一跑就炸）：
        #   `RuntimeError: cudnn RNN backward can only be called in training mode`
        #   —— bundle 加载后 WM 被 `.eval()` 了（旧代码对 MLP 无所谓，LN 无 running stats），
        #   而 cuDNN 的 RNN 反向**只允许**在 training 模式调用。放在循环里是自愈的：
        #   即使某个 eval 函数把模型留在 eval 模式，下一轮也会被纠正，不会静默跑偏。
        if arch == "gru":
            wm.train()
        S, acts, tgts = one_batch()
        xi = torch.zeros(int(S[0].shape[0]), 33, device=dev)
        s = S
        loss = 0.0
        # ★ 多步展开：GRU 的隐状态 h 跨 K 步传递 ⇒ loss.backward() 会穿过**全部 K 步**
        #   （这是加 GRU 的意义；MLP 路径仍是逐帧独立，逐位不变）。
        h = wm.init_hidden(int(S[0].shape[0]), device=dev) if arch == "gru" else None
        for k in range(K):
            if arch == "gru":
                out, h = wm.step_h(s, acts[:, k], xi, h)
            else:
                out = wm.step(s, acts[:, k], xi)
            _p, _v, _l = _pos_vel(out, tgts[k], a.vel_lambda, a.w_lambda,
                                  a.vel_scale_lambda)
            loss = loss + _l / K
            s = out                       # 自回归：下一步用**自己的预测**（这才是多步）
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # BPTT 安全阀：K 步展开梯度容易爆，统一裁剪（默认 1.0 = 旧行为）
        torch.nn.utils.clip_grad_norm_(params, float(a.clip_norm))
        if lr_warm > 1 and it <= lr_warm:
            for _g in opt.param_groups:
                _g["lr"] = float(a.lr) * float(it) / float(lr_warm)
        opt.step()
        if a.val_every and it % a.val_every == 0:
            vv = val_multistep()
            se, rw = policy_eval()
            print(f"  [{it:>4}] loss={float(loss.detach()):.4f} "
                  f"wm={json.dumps({k: round(v,4) for k,v in eval_wm(4).items()})} "
                  f"val20={vv:.4f} policy state={se:.3f} reward={rw:+.4f}", flush=True)
            if vv == vv and vv < best["val"]:
                best = {"val": vv, "it": it,
                        "state": {k: v.detach().cpu().clone() for k, v in wm.state_dict().items()}}
                print(f"  [{it:>4}] ↑ new best val20={vv:.4f}", flush=True)
    v1 = val_multistep()
    if v1 == v1:
        print(f"[after val] B {a.val_horizon} 步关节 RMSE = {v1:.4f} rad "
              f"(before {v0:.4f})")
    if best["it"] > 0:
        wm.load_state_dict(best["state"])
        print(f"[early-stop] 用第 {best['it']} 步的 WM（val20={best['val']:.4f}）", flush=True)
    se, rw = policy_eval()
    print(f"[after policy] state_term={se:.4f} reward={rw:+.5f}", flush=True)
    ck = dict(torch.load(a.ckpt, map_location="cpu", weights_only=False))
    ck["wm"] = {k: v.detach().cpu() for k, v in wm.state_dict().items()}
    ck["wm_integration"] = wm_integration
    if arch != "mlp":
        # 顶层键保持源 ckpt 原样（pi/wm/iteration/wall_time/sim_steps/args/reward_cfg/saved_at），
        # 只补一个结构标记：gru 的权重在 wm 里以 gru.*/head.* 命名，--arch gru 可 strict 重载。
        ck["wm_arch"] = arch
    torch.save(ck, a.out)
    print(f"[saved] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
