#!/usr/bin/env python3
"""会话 03：**纯 BPTT 可学性裁决**（Base vs Constant vs State-MLP vs Oracle）。

这份脚本把"Δtoken 能不能靠 WM 回报反传学出**比共享常向量更好**的状态相关修正"
拆成可复现、可判读的最小实验。设计约束（与 `docs/REVIEW_TWO_ROUTES_BPTT_VS_DISTILL_20261107.md`
和会话 03 任务书一致）：

1. **部署口径的共享 Δ**：``token'_t = token_t + Δ``，``Δ = f(token_0, state64)``，
   广播到**整段** token chunk（与 `diff_vla_wbc/v3_step.py::_delta_token_delta` 逐位同语义）。
   绝不用"重复 token0"替代 chunk。
2. **冻结一切**：VLA（checkpoint-8000，只用来**一次性**采 token chunk 并缓存）、
   WM（`wm_gain2.pt`）、SONIC release decoder、奖励函数。只训 Δ。
3. **最小目标**：``L = -mean_i J_i (+ λ·mean‖Δ_i‖²)``，``J_i`` = 折扣 WM 想象回报。
   第一轮关闭 critic / imagined bootstrap / window_norm / grad-target 辅损 / 反常量正则 /
   LoRA / 随机扰动。
4. **公平约束**：Constant / MLP / Oracle 共用同一个 ``apply_norm_cap`` 软上限、
   同一个 token 起点 chunk、同一个 WM、同一个 reward、同一个 H、同一个优化步预算；
   每个 arm 单独做**对称范围**的 lr 搜索，并在同一张表里同时报告未调参（共享 lr）的结果。
5. **不泄漏**：数据按 **clip（episode）** 分组切 train/val/test，同一 clip 的相邻帧不跨 split。

子命令::

    python3 scripts/diag_bptt_learnability.py build   --out runs/harness_03_bptt
    python3 scripts/diag_bptt_learnability.py exp0    --out runs/harness_03_bptt
    python3 scripts/diag_bptt_learnability.py arms    --out runs/harness_03_bptt
    python3 scripts/diag_bptt_learnability.py gradstruct --out runs/harness_03_bptt
    python3 scripts/diag_bptt_learnability.py horizon --out runs/harness_03_bptt
    python3 scripts/diag_bptt_learnability.py report  --out runs/harness_03_bptt

`build` 会落盘 `windows.pt`（含哈希），后面所有子命令都只读它 ⇒ 各 arm 逐位同数据。
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
# ★ 数据/权重都在**原始仓库**（worktree 里被 gitignore 的 runs/、data/ 不会出现）。
#   用环境变量覆盖，默认回落到 workspaces 里的主仓库绝对路径。
MAIN_REPO = Path(os.environ.get("COMIMIC_MAIN", str(ROOT)))
for _p in (str(MAIN_REPO), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("SUPERMIMIC_HA_ALIGN", "1")
os.environ.setdefault(
    "SUPERMIMIC_ROBOT_USD",
    str(MAIN_REPO / "trackerlab-supermimic/data/assetslib/unitree/unitree_g1/usd/g1_29dof_rev_1_0.usd"),
)

# --------------------------------------------------------------------------- 常量口径
TOKEN_DIM = 64
STATE_DIM = 71
GAMMA = 0.99
#: 部署用的 token 修正预算（`GRAD_TARGET_ETA` / §3.2 的 η=0.08 = 0.15/1.8）
ETA = 0.08

DEFAULTS = dict(
    wbc_ckpt="runs/S1_handover/WBC_A24000_wmvel_cmd.pt",
    wm_ckpt="runs/a_prime_sonic_exc/wm_gain2.pt",
    vlm_ckpt="runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000",
    vlm_path="groot_repro/nvidia/Cosmos-Reason2-2B-weights",
    motion="data/npz/ha_tokens",
    motion_packed="runs/isaac_motion/ha_cmd/packed.pt",
    port_tokens="data/port_tokens/ha_tokens",
    real_transitions="runs/a_prime_sonic_exc/all_transitions.pt",
    ref_drift_lib="runs/isaac_motion/ha_cmd_exec/packed.pt",
)


def reward_cfg_summary_of(bundle) -> str:
    from diff_vla_wbc.reward_cfg import reward_cfg_summary

    return reward_cfg_summary(bundle)


def _abs(p: str | Path | list) -> Path:
    if isinstance(p, (list, tuple)):
        p = p[0]
    p = Path(p)
    return p if p.is_absolute() else (MAIN_REPO / p)


def log(msg: str) -> None:
    print(f"[diag03] {msg}", flush=True)


def sha256_file(path: Path, limit: int = 1 << 30) -> str:
    h = hashlib.sha256()
    n = 0
    with Path(path).open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
            n += len(blk)
            if n >= limit:
                break
    return h.hexdigest()


def tensor_hash(*ts: torch.Tensor) -> str:
    h = hashlib.sha256()
    for t in ts:
        h.update(np.ascontiguousarray(t.detach().to(torch.float32).cpu().numpy()).tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- 环境装配
@dataclasses.dataclass
class Env:
    bundle: object
    lib: object
    device: torch.device
    online: object          # LibraryTokenBuilder（hold 窗口 + 机器人 anchor）
    chunk_src: object       # RecordedTokenSource(token_horizon=40, label 空间)
    frame_src: object       # RecordedTokenSource(token_horizon=1)
    recorded_states: torch.Tensor
    covered: torch.Tensor
    bptt_args: dict


def load_env(device: str = "cuda:0", bptt_args: Path | None = None) -> Env:
    """按 `runs/bptt1/args.json` 的口径重建**冻结** bundle（逐项记录，不猜）。"""
    from diff_vla_wbc.data import RecordedTokenSource
    from diff_vla_wbc.isaac_joint import load_isaac_bundle
    from diff_vla_wbc.reward_cfg import apply_reward_cfg, reward_cfg_summary
    from diff_vla_wbc.sonic_frontend import LibraryTokenBuilder
    from diff_vla_wbc.sonic_isaac import SonicIsaacBundle

    bargs = json.loads(Path(bptt_args).read_text()) if bptt_args else {}
    dev = torch.device(device)
    files = sorted(_abs(bargs.get("motion", DEFAULTS["motion"])).glob("*.npz"))
    if not files:
        raise SystemExit("[diag03] 找不到 motion npz")
    packed = _abs(bargs.get("motion_packed", DEFAULTS["motion_packed"]))
    wbc = _abs(bargs.get("wbc_ckpt", DEFAULTS["wbc_ckpt"]))
    wm_ckpt = _abs(bargs.get("wm_ckpt", DEFAULTS["wm_ckpt"]))
    t0 = time.time()
    base = load_isaac_bundle(wbc, device=dev, npz=files, packed_motion=packed,
                             obs_mode=None, hist=None, token_dim=TOKEN_DIM)
    bundle = SonicIsaacBundle(base, onnx_path=None,
                              token_quant=str(bargs.get("token_quant", "none")), device=dev)
    _wd = torch.load(str(wm_ckpt), map_location="cpu", weights_only=False)
    bundle.wm.load_state_dict(_wd.get("wm", _wd), strict=True)
    apply_reward_cfg(bundle, bargs)
    log(f"bundle 就绪 {time.time()-t0:.1f}s | wbc={wbc.name} wm={wm_ckpt.name} "
        f"executor={type(bundle.executor).__name__} | {reward_cfg_summary(bundle)}")

    lib = bundle.library
    online = LibraryTokenBuilder(bundle.motion, device=dev,
                                 window=str(bargs.get("wbc_token_window", "hold")),
                                 library_path=str(packed))
    _tdir = _abs(bargs.get("motion", DEFAULTS["motion"]))
    _port_dir = bargs.get("port_tokens", DEFAULTS["port_tokens"])
    space = str(bargs.get("wbc_token_space", "label"))
    chunk_src = RecordedTokenSource(_tdir, lib, token_horizon=40, token_dim=TOKEN_DIM,
                                    device=dev, port_dir=_abs(_port_dir), token_space=space)
    frame_src = RecordedTokenSource(_tdir, lib, token_horizon=1, token_dim=TOKEN_DIM,
                                    device=dev, port_dir=_abs(_port_dir), token_space=space)

    # 真转移起点：帧 → 行 映射（与 train_joint_v2.py 同口径），起点只取**真采到过**的帧
    real = torch.load(str(_abs(bargs.get("real_transitions", DEFAULTS["real_transitions"]))),
                      map_location="cpu", weights_only=False)
    n_frames = int(lib.num_frames)
    fr = real["frames"].long()
    _map = torch.full((n_frames,), -1, dtype=torch.long)
    _map[fr] = torch.arange(fr.numel())
    covered = _map >= 0
    idx = torch.arange(n_frames)
    last = torch.where(covered, idx, torch.full_like(idx, -1))
    last = torch.cummax(last, 0).values
    first = torch.where(covered, idx, torch.full_like(idx, n_frames))
    first = torch.flip(torch.cummin(torch.flip(first, [0]), 0).values, [0])
    nearest = torch.where(last >= 0, last, first)
    _map = torch.where(covered, _map, _map[nearest])
    recorded_states = real["states"][_map].to(dev)
    log(f"真转移 {tuple(real['states'].shape)} 帧覆盖 {float(covered.float().mean())*100:.1f}%")
    return Env(bundle=bundle, lib=lib, device=dev, online=online, chunk_src=chunk_src,
               frame_src=frame_src, recorded_states=recorded_states, covered=covered,
               bptt_args=bargs)


# --------------------------------------------------------------------------- 起点抽样
def start_weights(env: Env, delta: float) -> torch.Tensor:
    """`START_MAX_DRIFT`：只在**真采到过**且起点漂移 ≤ δ 的帧上抽样（与 bptt1 同口径）。"""
    real = torch.load(str(_abs(env.bptt_args.get("real_transitions",
                                                 DEFAULTS["real_transitions"]))),
                      map_location="cpu", weights_only=False)
    pk = torch.load(str(_abs(env.bptt_args.get("ref_drift_lib", DEFAULTS["ref_drift_lib"]))),
                    map_location="cpu", weights_only=False)
    ref_all = torch.as_tensor(pk["ref"]).float()
    fr = real["frames"].long()
    st = real["states"].float()
    off = (st[:, 0:3] - ref_all[fr][:, 0:3]).norm(dim=-1)
    w = torch.zeros(int(env.lib.num_frames), dtype=torch.float32)
    ok = off <= float(delta)
    try:
        ok = ok & env.covered[fr].cpu()
    except Exception:  # noqa: BLE001
        pass
    w[fr[ok]] = 1.0
    zero = (w == 0).nonzero(as_tuple=False).flatten()
    if zero.numel():
        w[zero] = w[(zero - 1).clamp(min=0)]
    return w


def clip_groups(lib) -> tuple[np.ndarray, list[str]]:
    """每帧所属 clip 序号 + clip 名（用于按 episode 分组切分）。"""
    cid = np.asarray(lib.clip_id.cpu() if torch.is_tensor(lib.clip_id) else lib.clip_id)
    names = [getattr(c, "name", f"clip{i}") for i, c in enumerate(lib.clips)]
    return cid, names


def build_windows(args) -> Path:
    """采样固定窗口、跑 burn-in、缓存 VLA token chunk，落盘 `windows.pt`。"""
    from diff_vla_wbc.burnin import sample_burnin

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = load_env(args.device, Path(args.bptt_args) if args.bptt_args else None)
    lib, dev = env.lib, env.device
    H = int(args.horizon)
    w = start_weights(env, float(args.start_max_drift))
    cid, names = clip_groups(lib)
    allowed = (w > 0).nonzero(as_tuple=False).flatten()
    frames_ok = torch.zeros(int(lib.num_frames), dtype=torch.bool)
    frames_ok[allowed] = True
    # 起点必须在 [HISTORY, num_frames-H-2]，且往后 H+1 帧都在同一 clip 内
    min_f, max_f = 10, int(lib.num_frames) - H - 3
    frames_ok[:min_f] = False
    frames_ok[max_f:] = False
    log(f"可作起点的帧 {int(frames_ok.sum())} / {int(lib.num_frames)}")

    # ---- 按 clip 切分（episode 级，不按帧）----
    frame_clip = cid
    usable_clips = sorted({int(frame_clip[f]) for f in allowed.tolist() if frames_ok[f]})
    rng_split = np.random.default_rng(int(args.split_seed))
    perm = rng_split.permutation(len(usable_clips))
    n_val = max(1, int(round(len(usable_clips) * args.val_frac)))
    n_test = max(1, int(round(len(usable_clips) * args.test_frac)))
    val_clips = [usable_clips[i] for i in perm[:n_val]]
    test_clips = [usable_clips[i] for i in perm[n_val:n_val + n_test]]
    train_clips = [usable_clips[i] for i in perm[n_val + n_test:]]
    log(f"clip 切分 train/val/test = {len(train_clips)}/{len(val_clips)}/{len(test_clips)}")

    g = torch.Generator(device="cpu")
    g.manual_seed(int(args.seed))
    picked: dict[str, torch.Tensor] = {}
    for split, clips in (("train", train_clips), ("val", val_clips), ("test", test_clips)):
        n_target = int(args.n_train if split == "train" else (
            args.n_val if split == "val" else args.n_test))
        pool = torch.tensor(
            [f for f in allowed.tolist() if frames_ok[f] and int(frame_clip[f]) in set(clips)],
            dtype=torch.long,
        )
        if pool.numel() == 0:
            log(f"[warn] split={split} 没有可用帧")
            picked[split] = torch.zeros(0, dtype=torch.long)
            continue
        # clip 分层：每个 clip 尽量均匀取
        sel: list[int] = []
        by_clip: dict[int, list[int]] = {}
        for f in pool.tolist():
            by_clip.setdefault(int(frame_clip[f]), []).append(f)
        keys = sorted(by_clip)
        rng = np.random.default_rng(int(args.seed) + {"train": 1, "val": 2, "test": 3}[split])
        for c in keys:
            rng.shuffle(by_clip[c])
        ptr = {c: 0 for c in keys}
        while len(sel) < min(n_target, int(pool.numel())):
            for c in keys:
                if len(sel) >= min(n_target, int(pool.numel())):
                    break
                if ptr[c] < len(by_clip[c]):
                    sel.append(by_clip[c][ptr[c]])
                    ptr[c] += 1
        picked[split] = torch.tensor(sorted(sel), dtype=torch.long)
        log(f"split={split}: {len(sel)} 窗口 / {len(keys)} clip")

    frames_all = torch.cat([picked[s] for s in ("train", "val", "test")], 0)
    split_id = torch.cat([
        torch.full((picked[s].numel(),), i, dtype=torch.long)
        for i, s in enumerate(("train", "val", "test"))
    ])
    N = int(frames_all.numel())
    if N == 0:
        raise SystemExit("[diag03] 没有窗口")

    # ---- burn-in（与 bptt1 完全同一条路径：recorded_states + perturb + hold online token）----
    t0 = time.time()
    st, hist, xi, prev_a, ref_f, ref_state = [], [], [], [], [], []
    for s in range(0, N, int(args.batch)):
        fr = frames_all[s:s + int(args.batch)].to(dev)
        bi = sample_burnin(
            env.bundle, lib, fr,
            port_tokens=lambda f: env.frame_src({"frames": f})[:, 0],
            token_fn=env.online.__call__,
            recorded_states=env.recorded_states,
            horizon=H, k=int(args.burnin_steps),
            generator=torch.Generator(device="cpu").manual_seed(int(args.seed) + s),
            perturb=True, xy=0.05, yaw=0.10, dq=0.05, dqd=0.30,
        )
        st.append(bi.state.detach().float().cpu())
        hist.append(bi.history.detach().float().permute(1, 0, 2).cpu())  # -> (B,10,W)
        xi.append(bi.xi.detach().float().cpu())
        prev_a.append(bi.prev_action.detach().float().cpu())
        ref_f.append(torch.stack([r.f.detach().cpu() for r in bi.references], 1))
        log(f"  burn-in {s + fr.numel()}/{N} ({time.time()-t0:.1f}s)")
    state = torch.cat(st, 0)
    history = torch.cat(hist, 0)
    xi_t = torch.cat(xi, 0)
    prev_action = torch.cat(prev_a, 0)
    refs_f = torch.cat(ref_f, 0)

    # ---- 起点帧的参考状态 + 两个 token chunk ----
    with torch.no_grad():
        ref0 = env.bundle.ref_state_at(frames_all.to(dev)).detach().float().cpu()
        last = lib.clip_end[lib.clip_id[frames_all.to(dev)]].to(dev)
        idx40 = torch.minimum(frames_all.to(dev)[:, None]
                             + torch.arange(40, device=dev)[None, :], last[:, None])
        tok_label = env.online.chunk(idx40, robot_quat=state.to(dev)[:, 3:7]).detach().float().cpu()

    meta = dict(
        n=N, horizon=H, burnin_steps=int(args.burnin_steps), seed=int(args.seed),
        split_seed=int(args.split_seed), start_max_drift=float(args.start_max_drift),
        n_train=int(picked["train"].numel()), n_val=int(picked["val"].numel()),
        n_test=int(picked["test"].numel()),
        train_clips=[names[c] for c in train_clips],
        val_clips=[names[c] for c in val_clips],
        test_clips=[names[c] for c in test_clips],
        bptt_args=str(args.bptt_args) if args.bptt_args else "",
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    payload = dict(
        frames=frames_all, split=split_id, clip_id=torch.tensor([int(cid[f]) for f in frames_all.tolist()]),
        state=state, history=history, xi=xi_t, prev_action=prev_action,
        refs_f=refs_f, ref0_state=ref0, tokens_label=tok_label, meta=meta,
    )
    torch.save(payload, out_dir / "windows.pt")
    (out_dir / "windows_meta.json").write_text(json.dumps(
        dict(meta, windows_sha256=sha256_file(out_dir / "windows.pt"),
             wbc_ckpt=str(_abs(env.bptt_args.get("wbc_ckpt", DEFAULTS["wbc_ckpt"]))),
             wbc_sha256_1m=sha256_file(_abs(env.bptt_args.get("wbc_ckpt",
                                                              DEFAULTS["wbc_ckpt"])), 1 << 20),
             wm_ckpt=str(_abs(env.bptt_args.get("wm_ckpt", DEFAULTS["wm_ckpt"]))),
             wm_sha256=sha256_file(_abs(env.bptt_args.get("wm_ckpt", DEFAULTS["wm_ckpt"]))),
             vlm_ckpt=str(_abs(env.bptt_args.get("vlm_ckpt", DEFAULTS["vlm_ckpt"]))),
             reward_cfg=reward_cfg_summary_of(env.bundle)), indent=1))
    log(f"windows.pt 写入 {out_dir/'windows.pt'} ({N} 窗口, {time.time()-t0:.1f}s)")

    # ---- VLA token chunk（部署真正用的那一段；一次采样 + 第二次采样做方差诊断）----
    if not args.skip_vla:
        _cache_vla_tokens(args, env, frames_all, out_dir)
    return out_dir / "windows.pt"


@torch.no_grad()
def _cache_vla_tokens(args, env: Env, frames: torch.Tensor, out_dir: Path) -> None:
    """跑官方 VLA（checkpoint-8000）在**同一批帧**上采 token chunk 并落盘。

    为什么必须缓存：VLA 冻结**不代表输出固定**（flow matching 有随机初噪）。
    训练里每一步都会重采，实验要复现就必须把**实际采到的那一段**存下来。
    """
    from diff_vla_wbc.action_codec import ActionCodec
    from diff_vla_wbc.fm_bc import sample_once_with_grad
    from diff_vla_wbc.g1_tasks import TaskObservationBuilder, processed_batch
    try:
        from diff_vla_wbc.g1_tasks import load_gr00t_processor
    except ImportError:  # 老版本命名
        from diff_vla_wbc.g1_tasks import load_processor as load_gr00t_processor  # type: ignore
    from diff_vla_wbc.gr00t_vlm import Gr00tVLM

    dev = env.device
    files = sorted(_abs(args.motion).glob("*.npz"))
    ckpt = _abs(args.vlm_ckpt)
    t0 = time.time()
    processor = load_gr00t_processor(ckpt)
    codec = ActionCodec(processor, device=dev)
    vlm = Gr00tVLM.from_pretrained(_abs(args.vlm_path), checkpoint=ckpt, device=dev,
                                   action_horizon=40, max_state_dim=132, freeze_vlm=True,
                                   attn_implementation="sdpa")
    vlm.action_head.eval()
    builder = TaskObservationBuilder(files) if not hasattr(args, "bucket_mode") else None
    from diff_vla_wbc.g1_tasks import TaskFrameIndex
    builder = TaskObservationBuilder(TaskFrameIndex(files))
    log(f"VLA 就绪 {time.time()-t0:.1f}s")
    outs, outs2 = [], []
    for s in range(0, int(frames.numel()), int(args.vla_batch)):
        fr = frames[s:s + int(args.vla_batch)]
        batch = processed_batch(processor, builder, fr)
        bb_in, act_in = vlm.prepare_input(dict(batch))
        with torch.no_grad():
            bb_out = vlm.backbone(bb_in)
        head = vlm.action_head
        for rep, sink in ((0, outs), (1, outs2)):
            gen = torch.Generator(device=dev)
            gen.manual_seed(int(args.seed) * 1000 + rep)
            noise = torch.randn(int(act_in["state"].shape[0]), int(head.action_horizon),
                                int(head.action_dim), device=dev, dtype=head.dtype, generator=gen)
            with torch.enable_grad():
                x_norm = sample_once_with_grad(head, bb_out, act_in, initial_noise=noise,
                                               num_steps=int(args.fm_steps))
            sink.append(codec.decode(x_norm.detach())[..., :TOKEN_DIM].float().cpu())
        log(f"  [vla] {s + fr.numel()}/{int(frames.numel())} ({time.time()-t0:.1f}s)")
    tok = torch.cat(outs, 0)
    tok2 = torch.cat(outs2, 0)
    torch.save({"tokens_vla": tok, "tokens_vla_rep2": tok2,
                "vlm_ckpt": str(ckpt), "fm_steps": int(args.fm_steps)},
               out_dir / "vla_tokens.pt")
    log(f"vla_tokens.pt 写入：{tuple(tok.shape)}；两次采样 |Δ|={float((tok-tok2).abs().mean()):.5f}")


# --------------------------------------------------------------------------- 目标函数
@dataclasses.dataclass
class WindowSet:
    """一组固定窗口（张量都在 CPU，用时搬到 device）。"""

    frames: torch.Tensor
    state: torch.Tensor
    history: torch.Tensor
    xi: torch.Tensor
    prev_action: torch.Tensor
    refs_f: torch.Tensor
    ref0_state: torch.Tensor
    tokens: torch.Tensor          # (N,40,64) 基础 token chunk（未加 Δ）
    tokens_name: str
    state64: torch.Tensor         # (N,64) 部署 state64
    token0: torch.Tensor          # (N,64) = tokens[:,0]
    clip_id: torch.Tensor = None  # (N,) 所属 clip（分组 CV / 聚类用）
    tag: str = ""

    def __len__(self) -> int:
        return int(self.frames.numel())

    def index(self, idx: torch.Tensor | np.ndarray | slice) -> "WindowSet":
        if isinstance(idx, slice):
            sel = torch.arange(self.frames.numel())[idx]
        else:
            sel = torch.as_tensor(idx, dtype=torch.long)
        return dataclasses.replace(self, **{
            k: getattr(self, k)[sel] for k in
            ("frames", "state", "history", "xi", "prev_action", "refs_f",
             "ref0_state", "tokens", "state64", "token0", "clip_id")
        })

    def to(self, dev: torch.device, H: int, *, tokens: torch.Tensor | None = None,
           state64: torch.Tensor | None = None, token0: torch.Tensor | None = None) -> dict:
        return dict(
            state=self.state.to(dev, non_blocking=True),
            history=self.history.to(dev, non_blocking=True),
            xi=self.xi.to(dev, non_blocking=True),
            prev_action=self.prev_action.to(dev, non_blocking=True),
            refs_f=self.refs_f[:, :H].to(dev, non_blocking=True),
            tokens=(self.tokens if tokens is None else tokens).to(dev, non_blocking=True),
            state64=(self.state64 if state64 is None else state64).to(dev, non_blocking=True),
            token0=(self.token0 if token0 is None else token0).to(dev, non_blocking=True),
        )


def state64_from(bundle, state: torch.Tensor, frames: torch.Tensor, dev,
                 use_full: bool = True) -> torch.Tensor:
    """部署 ``state64 = [rot6d | dof_pos29 | dof_vel29]``（与 v3 `_delta_token_features` 同序）。

    ``rot6d`` 用 `rotlocal6`：``conj(heading(q0))·q_now``，``q0`` = **clip 首帧参考朝向**
    （= `data._start_quat0` 的口径；bptt1 的 `DELTA_TOKEN_FULLSTATE=1` 就是这个）。``q0`` 拿不到时
    退回 `abs6`，并把退化事件显式记进 meta。
    """
    from diff_vla_wbc.isaac_joint import flat_state_tuple
    from diff_vla_wbc.state_extra import extra6_for

    s = state.to(dev).float()
    _p, _R, _v, _w, _q, _qd = flat_state_tuple(s)
    try:
        first = bundle.clip_start_of(frames.to(dev))
        q0 = bundle.motion.ref[first.to(bundle.motion.ref.device)][:, 3:7].float().to(dev)
        rot6 = extra6_for("rotlocal6", quat=s[:, 3:7], q_init=q0)
    except Exception as exc:  # noqa: BLE001
        log(f"[warn] rotlocal6 不可用（{exc}）⇒ 退回 abs6")
        rot6 = extra6_for("abs6", quat=s[:, 3:7])
    if not use_full:
        return torch.cat([rot6.float(), _qd.float()], dim=-1)
    return torch.cat([rot6.float(), _q.float(), _qd.float()], dim=-1)


@contextlib.contextmanager
def fixed_rng(seed: int | None):
    """目标函数必须是确定性的（否者 arm 之间的差异会混进采样噪声）。"""
    if seed is None:
        yield
        return
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def window_returns(bundle, batch: dict, H: int, gamma: float = GAMMA,
                   want_actions: bool = False) -> tuple[torch.Tensor, dict]:
    """`J_i`（可微 WM 想象回报）+ 诊断量。**唯一**的目标函数实现，所有 arm 共用。"""
    from diff_vla_wbc.isaac_joint import FrameRef
    from diff_vla_wbc.rollout import imagine_with_tokens

    refs = [FrameRef(batch["refs_f"][:, t]) for t in range(H)]
    ro = imagine_with_tokens(
        bundle, tokens=batch["tokens"], state=batch["state"],
        obs_hist=None, history=batch["history"], xi=batch["xi"], references=refs,
        prev_action=batch["prev_action"], horizon=H, obs_noise=0.0,
    )
    disc = gamma ** torch.arange(H, device=ro.rewards.device, dtype=ro.rewards.dtype)
    J = (disc[:, None] * ro.survival[:H] * ro.rewards[:H]).sum(0)
    extra = dict(rewards=ro.rewards.detach(), survival=ro.survival[-1].detach(),
                 actions=ro.actions if want_actions else None, rollout=ro)
    return J, extra


def apply_cap(delta: torch.Tensor, cap: float) -> torch.Tensor:
    """与部署/训练同一个软上限函数（`delta_token.apply_norm_cap`）。"""
    from diff_vla_wbc.delta_token import apply_norm_cap

    return apply_norm_cap(delta, cap)


# --------------------------------------------------------------------------- arms
class DeltaParam(torch.nn.Module):
    """四种参数化共用的一层壳：``Δ = cap(f(x))``，``x = (token0, state64)``。

    ``kind``：
      * ``base``     —— 恒 0（无参数，只测原始 WM 回报）
      * ``constant`` —— 一个共享 64 维向量（**核心基线**）
      * ``mlp``      —— `DeltaTokenNet(token0, state64)`（与生产同一个结构/初始化）
      * ``oracle``   —— 每窗口一个自由 64 维向量（诊断上界，不可部署）
    """

    def __init__(self, kind: str, n: int, *, hidden: int = 1024, depth: int = 2,
                 init_std: float = 4e-4, cap: float = 0.0, seed: int = 0,
                 state_dim: int = 64, token_dim: int = TOKEN_DIM):
        super().__init__()
        self.kind = kind
        self.cap = float(cap)
        self.n = int(n)
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        if kind == "constant":
            v = torch.zeros(1, token_dim)
            nn_ = torch.nn.Parameter(v)
            with torch.no_grad():
                nn_.normal_(0.0, init_std, generator=g)
            self.const = nn_
        elif kind == "mlp":
            from diff_vla_wbc.delta_token import DeltaTokenNet

            net = DeltaTokenNet(token_dim=token_dim, feat_dim=state_dim,
                                hidden=hidden, depth=depth, init_std=init_std)
            with torch.no_grad():
                # 与生产一致的初始化，但**显式播种**（可复现）
                for lm in net.net:
                    if isinstance(lm, torch.nn.Linear):
                        w = torch.empty_like(lm.weight).normal_(0.0, 1.0, generator=g)
                        fan_in = lm.weight.shape[1]
                        lm.weight.copy_(w * (math.sqrt(2.0 / fan_in)))
                        if lm.bias is not None:
                            lm.bias.zero_()
                last = [m for m in net.net if isinstance(m, torch.nn.Linear)][-1]
                last.weight.copy_(torch.empty_like(last.weight).normal_(0.0, init_std, generator=g))
                log(f"[mlp] params={sum(p.numel() for p in net.parameters()):,} "
                    f"init_std={init_std:g} hidden={hidden}x{depth}")
            self.net = net
        elif kind == "oracle":
            v = torch.zeros(self.n, token_dim)
            with torch.no_grad():
                v.normal_(0.0, init_std, generator=g)
            self.per_window = torch.nn.Parameter(v)
        elif kind == "base":
            pass
        else:
            raise ValueError(kind)

    def raw(self, token0: torch.Tensor, state64: torch.Tensor) -> torch.Tensor:
        if self.kind == "base":
            return torch.zeros(token0.shape[0], token0.shape[-1],
                               device=token0.device, dtype=token0.dtype)
        if self.kind == "constant":
            return self.const.expand(token0.shape[0], -1)
        if self.kind == "mlp":
            return self.net(token0, state64)
        if self.kind == "oracle":
            if token0.shape[0] != self.per_window.shape[0]:
                raise ValueError("oracle arm 必须整批（每窗口一个参数）")
            return self.per_window
        raise ValueError(self.kind)

    def forward(self, token0: torch.Tensor, state64: torch.Tensor) -> torch.Tensor:
        return apply_cap(self.raw(token0, state64), self.cap)

    def trainable(self) -> list[torch.nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]


__all__ = [
    "Env", "load_env", "build_windows", "WindowSet", "state64_from", "window_returns",
    "DeltaParam", "apply_cap", "fixed_rng", "start_weights", "sha256_file", "tensor_hash",
    "DEFAULTS", "ETA", "GAMMA", "log",
]


# =========================================================================== 窗口装配
def load_windows(env: Env, args, H: int, tokens_kind: str | None = None) -> WindowSet:
    """把 `windows.pt`（+ `vla_tokens.pt`）装成 `WindowSet`（含 state64 / token0）。"""
    wp = Path(args.windows) if args.windows else (Path(args.out) / "windows.pt")
    d = torch.load(str(wp), map_location="cpu", weights_only=False)
    kind = tokens_kind or args.tokens
    if kind == "vla":
        vp = (Path(args.vla_tokens) if getattr(args, "vla_tokens", None)
              else (Path(args.out) / "vla_tokens.pt"))
        if not vp.exists():
            raise SystemExit(f"[diag03] 缺 {vp}：先跑 build（不加 --skip-vla），或用 --tokens label")
        vtok = torch.load(str(vp), map_location="cpu", weights_only=False)
        tokens = vtok["tokens_vla"].float()
    else:
        tokens = d["tokens_label"].float()
    s64 = state64_from(env.bundle, d["state"], d["frames"], "cpu",
                       use_full=bool(env.bptt_args.get("delta_token_fullstate", True)))
    ws = WindowSet(
        frames=d["frames"], state=d["state"], history=d["history"], xi=d["xi"],
        prev_action=d["prev_action"], refs_f=d["refs_f"], ref0_state=d["ref0_state"],
        tokens=tokens, tokens_name=kind, state64=s64, token0=tokens[:, 0].clone(),
        clip_id=d.get("clip_id"),
    )
    ws.meta = d.get("meta", {})  # type: ignore[attr-defined]
    ws.tokens_label = d["tokens_label"]  # type: ignore[attr-defined]
    return ws


def split_sets(ws: WindowSet) -> dict[str, WindowSet]:
    meta = getattr(ws, "meta", {})
    n_tr = int(meta.get("n_train", 0))
    n_va = int(meta.get("n_val", 0))
    out = {
        "train": ws.index(slice(0, n_tr)),
        "val": ws.index(slice(n_tr, n_tr + n_va)),
        "test": ws.index(slice(n_tr + n_va, None)),
    }
    for k, v in out.items():
        v.tag = k
    return out


# =========================================================================== 一个 arm
def _eval_arm(env: Env, ws: WindowSet, arm: DeltaParam | None, H: int, dev,
              seed: int = 4321) -> dict:
    """`no_grad` 下算 J / Δ / 动作变化的 RMS（所有 arm 同一口径、同一起始 RNG）。"""
    from diff_vla_wbc.isaac_joint import FrameRef
    from diff_vla_wbc.rollout import imagine_with_tokens

    with torch.no_grad(), fixed_rng(seed):
        b = ws.to(dev, H)
        if arm is None:
            delta = torch.zeros(len(ws), TOKEN_DIM, device=dev)
        else:
            delta = arm(b["token0"], b["state64"])
        batch = dict(b)
        batch["tokens"] = b["tokens"] + delta[:, None, :]
        J, extra = window_returns(env.bundle, batch, H)
        # 动作变化 RMS：与 Δ=0 的同一起点动作比（判 deployability 的唯一硬指标）
        ro0 = imagine_with_tokens(env.bundle, tokens=b["tokens"], state=b["state"],
                                  obs_hist=None, history=b["history"], xi=b["xi"],
                                  references=[FrameRef(b["refs_f"][:, t]) for t in range(H)],
                                  prev_action=b["prev_action"], horizon=H, obs_noise=0.0)
        da = (extra["rollout"].actions - ro0.actions).float()
        act_rms = float((da.pow(2).mean(-1)).mean())
        return dict(
            J=J.detach().float().cpu(),
            delta=delta.detach().float().cpu(),
            delta_norm=float(delta.norm(dim=-1).mean()),
            act_rms=act_rms,
            act_rms_h=float((da.pow(2).mean(-1)).mean(0).mean()) if da.numel() else 0.0,
            survival=float(extra["survival"].mean()),
            reward_mean=float(extra["rewards"].mean()),
            cap_active=float((delta.norm(dim=-1) > 0.95 * arm.cap).float().mean())
            if arm is not None and arm.cap > 0 else 0.0,
        )


def run_arm(env: Env, ws_fit: WindowSet, ws_eval: dict[str, WindowSet], *,
            kind: str, H: int, cap: float, lam: float, lr: float, steps: int,
            seed: int, dev, log_every: int = 10, eval_every: int = 50,
            hidden: int = 1024, depth: int = 2, init_std: float = 4e-4,
            freeze_bias: bool = False, only_bias: bool = False,
            zero_state: bool = False,
            zero_token: bool = False, shuffle_state: bool = False,
            shuffle_token: bool = False, extra_metrics: bool = True) -> dict:
    """训一个 arm 并返回收敛曲线 + 各 split 的逐窗口回报（`per_window`）。"""
    n = len(ws_fit)
    arm = DeltaParam(kind, n, hidden=hidden, depth=depth, init_std=init_std, cap=cap,
                     seed=seed).to(dev)
    if zero_state or zero_token or shuffle_state or shuffle_token:
        _patch_inputs(arm, ws_fit, ws_eval, zero_state=zero_state, zero_token=zero_token,
                      shuffle_state=shuffle_state, shuffle_token=shuffle_token, seed=seed)
    params = arm.trainable()
    if not params:                       # base：无参数 ⇒ 只评测（训练循环空转）
        per_window = {}
        for split, wset in [("fit", ws_fit)] + list(ws_eval.items()):
            if wset is None or len(wset) == 0:
                continue
            per_window[split] = _eval_arm(env, wset, None, H, dev)
        return dict(kind=kind, lr=0.0, steps=0, seed=int(seed), cap=float(cap), lam=float(lam),
                    n_fit=n, hist=[], per_window=per_window, state_sens=0.0, token_sens=0.0,
                    n_params=0, secs=0.0)
    if freeze_bias:
        for m in arm.modules():
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                m.bias.requires_grad_(False)
        params = arm.trainable()
    if only_bias:
        # 只留输出层 bias ⇒ "常量捷径"的**严格**版本（连常数向量都不许学方向）
        for n_, p_ in arm.named_parameters():
            p_.requires_grad_(n_.endswith("bias") and n_.startswith("net.") and _is_last(arm, n_))
        params = arm.trainable()
    opt = torch.optim.Adam(params, lr=float(lr))
    b = ws_fit.to(dev, H)
    hist: list[dict] = []
    t0 = time.time()
    base_eval = None
    for step in range(1, int(steps) + 1):
        delta = arm(b["token0"], b["state64"])
        batch = dict(b)
        batch["tokens"] = b["tokens"] + delta[:, None, :]
        J, extra = window_returns(env.bundle, batch, H)
        loss = -J.mean() + float(lam) * delta.pow(2).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(
            arm.trainable(), float(getattr(args_global, "grad_clip", 1e9) or 1e9)))
        opt.step()
        if step == 1 or step % log_every == 0 or step == steps:
            with torch.no_grad():
                hist.append(dict(step=step, J=float(J.mean()), J_std=float(J.std()),
                                 loss=float(loss), delta_norm=float(delta.norm(dim=-1).mean()),
                                 grad_norm=gnorm, act_rms=0.0,
                                 survival=float(extra["survival"].mean()),
                                 secs=time.time() - t0))
    # ---- 评测 ----
    per_window: dict[str, dict] = {}
    for split, wset in [("fit", ws_fit)] + list(ws_eval.items()):
        if wset is None or len(wset) == 0:
            continue
        r = _eval_arm(env, wset, arm, H, dev)
        per_window[split] = r
    # Δ 与输入的关系（诊断：是否真的按状态变）
    with torch.no_grad():
        bf = ws_fit.to(dev, H)
        d_fit = arm(bf["token0"], bf["state64"])
        _perm = torch.randperm(n, device=dev)
        d_perm = arm(bf["token0"], bf["state64"][_perm])
        state_sens = float((d_fit - d_perm).norm(dim=-1).mean()
                           / d_fit.norm(dim=-1).mean().clamp_min(1e-9))
        d_tok_perm = arm(bf["token0"][_perm], bf["state64"])
        token_sens = float((d_fit - d_tok_perm).norm(dim=-1).mean()
                           / d_fit.norm(dim=-1).mean().clamp_min(1e-9))
    out = dict(kind=kind, lr=float(lr), steps=int(steps), seed=int(seed), cap=float(cap),
               lam=float(lam), n_fit=n, hist=hist, per_window=per_window,
               state_sens=state_sens, token_sens=token_sens,
               delta_mean=per_window.get("fit", {}).get("delta", torch.zeros(1))[:n].mean(0),
               n_params=sum(p.numel() for p in arm.parameters()),
               secs=time.time() - t0)
    if extra_metrics:
        out["arm_state"] = {k: v.detach().cpu() for k, v in arm.state_dict().items()}
    return out


args_global = argparse.Namespace(grad_clip=0.0)


def _is_last(arm: "DeltaParam", name: str) -> bool:
    """``name`` 是否属于 DeltaTokenNet 的**最后一层**线性层。"""
    keys = [k for k, _ in arm.named_parameters()]
    lin_idx = [i for i, k in enumerate(keys) if k.endswith("weight") and k.startswith("net.")]
    return bool(lin_idx) and keys[lin_idx[-1]].rsplit(".", 1)[0] == name.rsplit(".", 1)[0]


def _patch_inputs(arm: DeltaParam, ws_fit: WindowSet, ws_eval: dict, *,
                  zero_state: bool, zero_token: bool,
                  shuffle_state: bool, shuffle_token: bool, seed: int) -> None:
    """消融：把 inputs 换成常量（zero）/ 跨窗口错配（shuffle）。"""
    arm._zero_state = bool(zero_state)      # type: ignore[attr-defined]
    arm._zero_token = bool(zero_token)      # type: ignore[attr-defined]
    g = torch.Generator(device="cpu").manual_seed(int(seed) + 977)
    for ws in [ws_fit] + [w for w in ws_eval.values() if w is not None]:
        if shuffle_state:
            perm = torch.randperm(len(ws), generator=g)
            ws.state64 = ws.state64[perm]
        if shuffle_token:
            perm = torch.randperm(len(ws), generator=g)
            ws.token0 = ws.token0[perm]
    if zero_state or zero_token:
        _orig = arm.forward

        def _fwd(token0, state64, _orig=_orig, _a=arm):
            if getattr(_a, "_zero_state", False):
                state64 = torch.zeros_like(state64)
            if getattr(_a, "_zero_token", False):
                token0 = torch.zeros_like(token0)
            return _orig(token0, state64)

        arm.forward = _fwd  # type: ignore[assignment]


# =========================================================================== 实验 0
def cmd_exp0(env: Env, args) -> dict:
    """梯度路径审计：参数进没进图、梯度有没有回到 Δ、解析梯度对不对（有限差分）。"""
    from diff_vla_wbc.isaac_joint import FrameRef

    dev, H = env.device, int(args.horizon)
    ws = load_windows(env, args, H)
    sp = split_sets(ws)
    ws8 = sp["train"].index(slice(0, int(args.n)))
    rows: dict = {"n_windows": len(ws8), "H": H, "cap": float(args.cap), "tokens": ws.tokens_name}

    # ---------- (1) 参数/梯度路径 ----------
    path_rows = []
    for kind in ("constant", "mlp"):
        arm = DeltaParam(kind, len(ws8), hidden=int(args.hidden), depth=int(args.depth),
                         init_std=float(args.init_std), cap=float(args.cap), seed=0).to(dev)
        opt = torch.optim.Adam(arm.trainable(), lr=1e-3)
        n_req = sum(1 for p in arm.parameters() if p.requires_grad)
        in_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
        b = ws8.to(dev, H)
        w_before = {k: v.detach().clone() for k, v in arm.state_dict().items()}
        delta = arm(b["token0"], b["state64"])
        batch = dict(b)
        batch["tokens"] = b["tokens"] + delta[:, None, :]
        J, extra = window_returns(env.bundle, batch, H)
        loss = -J.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grads = {k: (None if p.grad is None else p.grad.detach().clone())
                 for k, p in arm.named_parameters()}
        opt.step()
        w_after = {k: v.detach().clone() for k, v in arm.state_dict().items()}
        layer = []
        for k, p in arm.named_parameters():
            g = grads.get(k)
            layer.append(dict(
                name=k, shape=list(p.shape), requires_grad=bool(p.requires_grad),
                grad_none=g is None, grad_norm=(float(g.norm()) if g is not None else 0.0),
                grad_absmax=(float(g.abs().max()) if g is not None else 0.0),
                grad_finite=(bool(torch.isfinite(g).all()) if g is not None else None),
                param_delta=(float((w_after[k] - w_before[k]).abs().max())
                             if k in w_after else float("nan")),
            ))
        path_rows.append(dict(
            kind=kind, n_params=sum(p.numel() for p in arm.parameters()),
            n_requires_grad=n_req, params_in_optimizer=in_opt,
            identity_at_init=float(delta.abs().max()),
            J_mean=float(J.mean()), J_std=float(J.std()),
            loss=float(loss), grad_sum_norm=float(sum(
                float(g.norm()) for g in grads.values() if g is not None)),
            layers=layer,
        ))
        log(f"[exp0][{kind}] params={path_rows[-1]['n_params']:,} "
            f"req_grad={n_req}/{len(list(arm.parameters()))} opt_params={in_opt:,} "
            f"|Δ|_max@init={float(delta.abs().max()):.3e} J={float(J.mean()):.5f} "
            f"sum|g|={path_rows[-1]['grad_sum_norm']:.3e} "
            f"|W|变化_max={max(r['param_delta'] for r in layer):.3e}")
    rows["grad_path"] = path_rows

    # ---------- (2) 梯度是否真的穿过 decoder 回到 token ----------
    b = ws8.to(dev, H)
    t0 = b["token0"].detach().clone().requires_grad_(True)
    tok = b["tokens"].clone()
    tok[:, 0, :] = t0                      # 只让 token0 带图（= 部署里 Δ 的挂点）
    J, _ = window_returns(env.bundle, dict(b, tokens=tok), H)
    g_tok = torch.autograd.grad(J.sum(), t0)[0]
    rows["token_grad"] = dict(
        norm_mean=float(g_tok.norm(dim=-1).mean()), finite=bool(torch.isfinite(g_tok).all()),
        n_zero=int((g_tok.norm(dim=-1) < 1e-12).sum()),
    )
    log(f"[exp0] dJ/dtoken0 = {rows['token_grad']['norm_mean']:.4e} "
        f"(finite={rows['token_grad']['finite']}, zeros={rows['token_grad']['n_zero']})")

    # ---------- (3) 有限差分：共享 Δ 的方向导数 ----------
    #   G(v) = mean_i J_i(token_i + v)。解析 ∂G/∂v 与 (G(v+h·e)−G(v−h·e))/(2h) 对比。
    v0 = torch.zeros(TOKEN_DIM, device=dev)
    if args.fd_at == "grad":
        Jg = window_returns(env.bundle, dict(b, tokens=b["tokens"] + v0), H)[0].mean()
        # 用 dG/dv 自身当方向（最有信息的那一个）
        vv = torch.zeros(1, TOKEN_DIM, device=dev, requires_grad=True)
        Jv, _ = window_returns(env.bundle, dict(b, tokens=b["tokens"] + vv[:, None, :]), H)
        gr = torch.autograd.grad(Jv.mean(), vv)[0][0].detach()
        e = gr / gr.norm().clamp_min(1e-12)
    else:
        g = torch.Generator(device=dev).manual_seed(int(args.seed))
        e = torch.randn(TOKEN_DIM, device=dev, generator=g)
        e = e / e.norm()

    vv = (e * 0.0).clone().requires_grad_(True)
    Jv, _ = window_returns(env.bundle, dict(b, tokens=b["tokens"] + vv[None, None, :]), H)
    analytic = float(torch.autograd.grad(Jv.mean(), vv)[0] @ e)
    fd = []
    with torch.no_grad():
        for h in args.fd_h:
            Jp, _ = window_returns(env.bundle, dict(b, tokens=b["tokens"] + (e * h)[None, None, :]), H)
            Jm, _ = window_returns(env.bundle, dict(b, tokens=b["tokens"] - (e * h)[None, None, :]), H)
            num = float((Jp.mean() - Jm.mean()) / (2 * float(h)))
            err = abs(num - analytic) / max(abs(analytic), 1e-12)
            fd.append(dict(h=float(h), fd=num, analytic=analytic, rel_err=float(err),
                           J_plus=float(Jp.mean()), J_minus=float(Jm.mean())))
            log(f"[exp0][FD] h={float(h):.1e} analytic={analytic:+.6e} fd={num:+.6e} "
                f"rel_err={err:.3e}")
    rows["finite_difference"] = fd
    rows["fd_direction"] = "grad" if args.fd_at == "grad" else "random"
    rows["fd_direction_norm"] = float(e.norm())

    # ---------- (4) 数值稳定性 / 尺度标定 ----------
    with torch.no_grad():
        r = _eval_arm(env, ws8, None, H, dev)
        scale_rows = []
        for mag in (0.01, 0.02, 0.04, 0.08, 0.16, 0.32):
            dv = (e * mag)[None, :].expand(len(ws8), -1)
            ro = None
            arm = DeltaParam("oracle", len(ws8), cap=0.0, init_std=0.0, seed=0).to(dev)
            with torch.no_grad():
                arm.per_window.copy_(dv)
            rr = _eval_arm(env, ws8, arm, H, dev)
            scale_rows.append(dict(mag=mag, dJ=float(rr["J"].mean() - r["J"].mean()),
                                   act_rms=rr["act_rms"], delta_norm=rr["delta_norm"]))
            log(f"[exp0][scale] ‖Δ‖={mag:.3f} → ΔJ={scale_rows[-1]['dJ']:+.5f} "
                f"动作RMS={rr['act_rms']:.4f}")
    rows["scale_calibration"] = scale_rows
    rows["J_base"] = float(r["J"].mean())
    rows["act_rms_zero"] = 0.0
    return rows


# =========================================================================== 实验 1/2：arms
def base_returns(env: Env, wsets: dict[str, WindowSet], H: int, dev,
                 cache: Path | None = None) -> dict[str, torch.Tensor]:
    if cache is not None and cache.exists():
        d = torch.load(str(cache), map_location="cpu", weights_only=False)
        return {k: v for k, v in d.items()}
    out = {}
    for k, ws in wsets.items():
        if ws is None or len(ws) == 0:
            continue
        r = _eval_arm(env, ws, None, H, dev)
        out[k] = r["J"]
        log(f"[base] split={k} n={len(ws)} J={float(r['J'].mean()):.6f} "
            f"(±{float(r['J'].std()):.4f})")
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(f".{os.getpid()}.tmp")
        torch.save(out, str(tmp))
        os.replace(tmp, cache)
    return out


def cmd_arms(env: Env, args) -> list[dict]:
    dev, H = env.device, int(args.horizon)
    ws = load_windows(env, args, H)
    sp = split_sets(ws)
    for k, v in sp.items():
        log(f"[windows] split={k} n={len(v)}")
    rows = []
    ablate = list(getattr(args, "ablate", None) or [])
    for size in args.sizes:
        fit = sp["train"].index(slice(0, int(size))) if size > 0 else sp["train"]
        eval_sets = {"fit": fit, "val": sp["val"], "test": sp["test"]}
        # ★ 基线必须按**同一个窗口子集**算（size<全量时用全量均值会系统性偏移）
        base_cache = Path(args.out) / f"base_eval_{ws.tokens_name}_n{len(fit)}_{args.tag or 'main'}.pt"
        base = base_returns(env, eval_sets, H, dev, cache=base_cache)
        for arm_kind in args.arms:
            if arm_kind == "oracle":
                # ★ Oracle 是**逐窗口**自由参数 ⇒ 跨窗口不迁移。每个 split 各自拟合一次，
                #   只把它当"该 split 上还有多少可优化空间"的诊断上界（绝不当可部署模型）。
                for scope, wset in (("fit", fit), ("val", sp["val"]), ("test", sp["test"])):
                    if wset is None or len(wset) == 0:
                        continue
                    for lr in args.lrs:
                        t0 = time.time()
                        r = run_arm(env, wset, {}, kind="oracle", H=H, cap=float(args.cap),
                                    lam=float(args.lam), lr=float(lr), steps=int(args.steps),
                                    seed=int(args.seed), dev=dev, log_every=int(args.log_every),
                                    hidden=int(args.hidden), depth=int(args.depth),
                                    init_std=float(args.init_std))
                        Jk = r["per_window"]["fit"]["J"]
                        Jb = base.get(scope if scope != "fit" else "fit")
                        row = dict(arm="oracle", size=int(size), seed=int(args.seed), lr=float(lr),
                                   steps=int(args.steps), cap=float(args.cap), lam=float(args.lam),
                                   tokens=ws.tokens_name, tag=args.tag, oracle_scope=scope,
                                   n_params=r["n_params"], secs=round(time.time() - t0, 1),
                                   state_sens=0.0, token_sens=0.0,
                                   delta_norm=r["per_window"]["fit"]["delta_norm"],
                                   act_rms=r["per_window"]["fit"]["act_rms"],
                                   cap_active=r["per_window"]["fit"]["cap_active"],
                                   survival=r["per_window"]["fit"]["survival"],
                                   J_fit=float(Jk.mean()))
                        key = "fit" if scope == "fit" else scope
                        row[f"J_{key}"] = float(Jk.mean())
                        row[f"dJ_{key}"] = float(Jk.mean() - Jb.mean()) if Jb is not None else float("nan")
                        row[f"pos_{key}"] = float((Jk > Jb).float().mean()) if Jb is not None else float("nan")
                        row[f"perwin_{key}"] = Jk.tolist()
                        row[f"baseperwin_{key}"] = Jb.tolist() if Jb is not None else []
                        row["hist"] = r["hist"]
                        rows.append(row)
                        log(f"[arm] oracle  n={len(wset):4d} scope={scope:4s} lr={lr:g} "
                            f"dJ={row[f'dJ_{key}']:+.5f} |Δ|={row['delta_norm']:.4f} "
                            f"actRMS={row['act_rms']:.4f} ({row['secs']}s)")
                        _append_jsonl(Path(args.out) / f"arms_runs_{args.tag or 'main'}.jsonl", row)
                continue
            seeds = [int(args.seed)] if arm_kind == "base" else args.seeds
            for sd in seeds:
                for lr in (args.lrs if arm_kind in ("constant", "mlp") else [0.0]):
                    t0 = time.time()
                    r = run_arm(env, fit, eval_sets, kind=arm_kind, H=H, cap=float(args.cap),
                                lam=float(args.lam), lr=float(lr), steps=int(args.steps),
                                seed=int(sd), dev=dev, log_every=int(args.log_every),
                                hidden=int(args.hidden), depth=int(args.depth),
                                init_std=float(args.init_std),
                                freeze_bias=bool(args.freeze_bias),
                                only_bias=bool(getattr(args, "only_bias", False)),
                                zero_state=("zero_state" in ablate),
                                zero_token=("zero_token" in ablate),
                                shuffle_state=("shuffle_state" in ablate),
                                shuffle_token=("shuffle_token" in ablate))
                    row = dict(
                        arm=arm_kind, size=int(size), seed=int(sd), lr=float(lr),
                        steps=int(args.steps), cap=float(args.cap), lam=float(args.lam),
                        tokens=ws.tokens_name, tag=args.tag,
                        ablate="+".join(ablate) or "none",
                        n_params=r["n_params"], secs=round(time.time() - t0, 1),
                        state_sens=r["state_sens"], token_sens=r["token_sens"],
                        delta_norm=r["per_window"]["fit"]["delta_norm"],
                        act_rms=r["per_window"]["fit"]["act_rms"],
                        cap_active=r["per_window"]["fit"]["cap_active"],
                        survival=r["per_window"]["fit"]["survival"],
                        J_fit=r["per_window"]["fit"]["J"],
                        hist=r["hist"],
                    )
                    for k in ("fit", "val", "test"):
                        if k not in r["per_window"]:
                            continue
                        Jk = r["per_window"][k]["J"]
                        Jb = base.get(k)
                        row[f"J_{k}"] = float(Jk.mean())
                        row[f"dJ_{k}"] = float(Jk.mean() - Jb.mean()) if Jb is not None else float("nan")
                        row[f"pos_{k}"] = float((Jk > Jb).float().mean()) if Jb is not None else float("nan")
                        row[f"neg_{k}"] = float((Jk < Jb - 0.05 * Jb.abs().mean()).float().mean()) \
                            if Jb is not None else float("nan")
                        row[f"perwin_{k}"] = Jk.tolist()
                        row[f"baseperwin_{k}"] = (Jb.tolist() if Jb is not None else [])
                    if arm_kind == "constant" and "fit" in r["per_window"]:
                        row["const_vec"] = r["per_window"]["fit"]["delta"].mean(0).tolist()
                    rows.append(row)
                    log(f"[arm] {arm_kind:8s} n={size:4d} seed={sd} lr={lr:g} "
                        f"J_fit={row['J_fit']:.5f} dJ_fit={row.get('dJ_fit', float('nan')):+.5f} "
                        f"dJ_val={row.get('dJ_val', float('nan')):+.5f} "
                        f"dJ_test={row.get('dJ_test', float('nan')):+.5f} "
                        f"|Δ|={row['delta_norm']:.4f} actRMS={row['act_rms']:.4f} "
                        f"state_sens={row['state_sens']:.3f} ({row['secs']}s)")
                    _append_jsonl(Path(args.out) / f"arms_runs_{args.tag or 'main'}.jsonl", row)
    return rows


def _jsonable(o):
    """递归把 tensor / numpy 变成可序列化的标量或列表（多元素 tensor 不会被 float() 混过去）。"""
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(_jsonable(row)) + "\n")


# =========================================================================== 实验 4：梯度结构
def per_window_grad(env: Env, ws: WindowSet, H: int, dev, *, cap: float = 0.0) -> torch.Tensor:
    """``g_i = dJ_i/dΔ_i``（Δ 广播到整段 chunk，逐窗口独立；这是"正确的共享修正梯度"）。

    注意：这不是旧口径的"单 token Teacher"（那只对 token[0] 求导且不广播）。
    """
    b = ws.to(dev, H)
    delta = torch.zeros(len(ws), TOKEN_DIM, device=dev, requires_grad=True)
    batch = dict(b)
    batch["tokens"] = b["tokens"] + delta[:, None, :]
    J, _ = window_returns(env.bundle, batch, H)
    return torch.autograd.grad(J.sum(), delta)[0].detach()


def _grouped_cv_predict(X: np.ndarray, Y: np.ndarray, groups: np.ndarray,
                        ridge: float = 1e-2, folds: int = 5,
                        mlp: bool = False, seed: int = 0) -> dict:
    """按 **clip 分组** 的 K 折：用 X 预测 Y，返回 held-out 的 R² / 余弦 / 逐元素相关。"""
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    if X.ndim == 1:
        X = X[:, None]
    ug = np.unique(groups)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ug))
    fold_of = {g: order[i] % folds for i, g in enumerate(ug)}
    fid = np.array([fold_of[g] for g in groups])
    pred = np.zeros_like(Y)
    for f in range(folds):
        te = fid == f
        tr = ~te
        if tr.sum() < 2 or te.sum() == 0:
            continue
        mu = X[tr].mean(0)
        sd = X[tr].std(0) + 1e-8
        Xtr = np.concatenate([(X[tr] - mu) / sd, np.ones((tr.sum(), 1))], 1)
        Xte = np.concatenate([(X[te] - mu) / sd, np.ones((te.sum(), 1))], 1)
        if mlp:
            pred[te] = _fit_torch_mlp(Xtr, Y[tr], Xte, seed=seed + f)
        else:
            W = np.linalg.solve(Xtr.T @ Xtr + ridge * np.eye(Xtr.shape[1]), Xtr.T @ Y[tr])
            pred[te] = Xte @ W
    ss_res = float(((Y - pred) ** 2).sum())
    ss_tot = float(((Y - Y.mean(0)) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    # 逐样本余弦（预测 vs 真值）
    den = (np.linalg.norm(pred, axis=-1) * np.linalg.norm(Y, axis=-1))
    cos = np.where(den > 1e-12, (pred * Y).sum(-1) / np.maximum(den, 1e-12), 0.0)
    # 逐维相关（把 (N,D) 摊平成两个长向量）
    a = pred.ravel(); b = Y.ravel()
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-12 and b.std() > 1e-12 else 0.0
    return dict(r2=float(r2), cos_mean=float(cos.mean()), cos_median=float(np.median(cos)),
                corr=corr, n=int(len(Y)), folds=int(folds))


def _fit_torch_mlp(Xtr, Ytr, Xte, *, seed: int = 0, hidden: int = 256, steps: int = 600,
                   lr: float = 3e-3) -> np.ndarray:
    torch.manual_seed(int(seed))
    X = torch.tensor(Xtr, dtype=torch.float32)
    Y = torch.tensor(Ytr, dtype=torch.float32)
    Xe = torch.tensor(Xte, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.SiLU(),
                              torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
                              torch.nn.Linear(hidden, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(net(X), Y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return net(Xe).numpy()


def cmd_gradstruct(env: Env, args) -> list[dict]:
    """逐窗口共享修正梯度 ``g_i`` 的结构：均值/残差、方向冲突、可预测性、真实收益。"""
    dev, H = env.device, int(args.horizon)
    ws = load_windows(env, args, H)
    sp = split_sets(ws)
    rows = []
    g_by_split: dict[str, torch.Tensor] = {}
    for split, wset in sp.items():
        n = min(len(wset), int(args.n_grad))
        w = wset.index(slice(0, n))
        g = per_window_grad(env, w, H, dev)
        g_by_split[split] = g.cpu()
        mu = g.mean(0)
        res = g - mu
        cos_pairs = _pairwise_cos(g)
        rows.append(dict(
            split=split, n=int(n), H=H, tokens=ws.tokens_name,
            g_norm_mean=float(g.norm(dim=-1).mean()),
            mean_norm=float(mu.norm()),
            res_rms=float(res.pow(2).mean().sqrt()),
            res_over_mean_L2=float(res.norm(dim=-1).mean() / mu.norm().clamp_min(1e-12)),
            res_over_mean_rms=float(res.pow(2).mean().sqrt() / mu.pow(2).mean().sqrt().clamp_min(1e-12)),
            cos_pairs_mean=cos_pairs["mean"], cos_pairs_median=cos_pairs["median"],
            cos_pairs_p10=cos_pairs["p10"], cos_pairs_p90=cos_pairs["p90"],
            frac_cos_neg=cos_pairs["frac_neg"],
        ))
        log(f"[grad] split={split} n={n} ‖g‖={rows[-1]['g_norm_mean']:.4e} "
            f"‖mean g‖={rows[-1]['mean_norm']:.4e} rms(res)={rows[-1]['res_rms']:.4e} "
            f"res/mean(L2)={rows[-1]['res_over_mean_L2']:.3f} "
            f"cos(pairs)={cos_pairs['mean']:+.3f} neg={cos_pairs['frac_neg']:.2f}")
    # ---- 可预测性：特征 → g_i（按 clip 分组 CV，held-out）----
    wtr = sp["train"].index(slice(0, min(len(sp["train"]), int(args.n_grad))))
    Xtok = wtr.token0.numpy()
    Xst = wtr.state64.numpy()
    Xboth = np.concatenate([Xtok, Xst], 1)
    groups = wtr.clip_id.numpy()
    Y = g_by_split["train"].numpy()
    pred_rows = []
    for name, X, y in [
        ("token_only", Xtok, Y), ("state_only", Xst, Y),
        ("token+state", Xboth, Y), ("token+state→res", Xboth, Y - Y.mean(0)),
    ]:
        for kind in ("ridge", "mlp"):
            m = _grouped_cv_predict(X, y, groups, mlp=(kind == "mlp"), seed=int(args.seed))
            pred_rows.append(dict(features=name, model=kind, target=("res" if "res" in name else "g"),
                                  **m))
            log(f"[grad][cv] {name:16s} {kind:5s} R²={m['r2']:+.3f} cos={m['cos_mean']:+.3f} "
                f"corr={m['corr']:+.3f}")
    # ---- 常向量 vs 单窗口方向的实际收益（一阶方向，不是训练）----
    b = wtr.to(dev, H)
    with torch.no_grad():
        J0, _ = window_returns(env.bundle, dict(b, tokens=b["tokens"]), H)
        mu_g = g_by_split["train"].mean(0).to(dev)
        gain_rows = []
        for scale in (0.2, 0.5, 1.0, 2.0):
            step_mu = scale * mu_g / mu_g.norm().clamp_min(1e-12)
            g = g_by_split["train"].to(dev)
            step_i = scale * g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            J_mu, _ = window_returns(env.bundle, dict(b, tokens=b["tokens"] + step_mu[None, None, :]), H)
            J_i, _ = window_returns(env.bundle, dict(b, tokens=b["tokens"] + step_i[:, None, :]), H)
            base_m = J0.mean()
            gain_rows.append(dict(scale=scale,
                                  dJ_constant=float(J_mu.mean() - base_m),
                                  dJ_perwindow=float(J_i.mean() - base_m),
                                  dJ_perwindow_minus_const=float(J_i.mean() - J_mu.mean()),
                                  cos_to_mean=float((step_i @ step_mu).sum(-1).div(
                                      step_i.norm(dim=-1).clamp_min(1e-12)).mean())))
            log(f"[grad][gain] step={scale:g}·unit  ΔJ_const={gain_rows[-1]['dJ_constant']:+.5f} "
                f"ΔJ_perwin={gain_rows[-1]['dJ_perwindow']:+.5f}")
    torch.save(dict(g={k: v for k, v in g_by_split.items()},
                    frames=wtr.frames, token0=wtr.token0, state64=wtr.state64,
                    clip_id=wtr.clip_id),
               str(Path(args.out) / f"gradstruct_{ws.tokens_name}_{args.tag or 'main'}.pt"))
    out = dict(rows=rows, cv=pred_rows, gain=gain_rows)
    (Path(args.out) / f"gradstruct_{ws.tokens_name}_{args.tag or 'main'}.json").write_text(
        json.dumps(out, indent=1, default=float))
    return rows


def _pairwise_cos(g: torch.Tensor, max_pairs: int = 20000) -> dict:
    n = g.shape[0]
    if n < 2:
        return dict(mean=0.0, median=0.0, p10=0.0, p90=0.0, frac_neg=0.0)
    gn = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    idx = torch.triu_indices(n, n, offset=1)
    if idx.shape[1] > max_pairs:
        sel = torch.randperm(idx.shape[1])[:max_pairs]
        idx = idx[:, sel]
    c = (gn[idx[0]] * gn[idx[1]]).sum(-1)
    return dict(mean=float(c.mean()), median=float(c.median()),
                p10=float(c.quantile(0.1)), p90=float(c.quantile(0.9)),
                frac_neg=float((c < 0).float().mean()))


# =========================================================================== 实验 5：视界
def cmd_horizon(env: Env, args) -> list[dict]:
    dev = env.device
    ws = load_windows(env, args, int(args.horizon))
    sp = split_sets(ws)
    rows = []
    for H in args.hs:
        fit = sp["train"].index(slice(0, int(args.n_fit)))
        base = base_returns(env, {"fit": fit, "val": sp["val"]}, H, dev,
                            cache=Path(args.out) / f"base_eval_{ws.tokens_name}_H{H}.pt")
        for kind in ("base", "constant", "mlp"):
            for sd in ([0] if kind == "base" else args.seeds):
                for lr in (args.lrs if kind != "base" else [0.0]):
                    r = run_arm(env, fit, {"val": sp["val"]}, kind=kind, H=H,
                                cap=float(args.cap), lam=float(args.lam), lr=float(lr),
                                steps=int(args.steps), seed=int(sd), dev=dev,
                                log_every=int(args.log_every), hidden=int(args.hidden),
                                depth=int(args.depth), init_std=float(args.init_std),
                                only_bias=bool(getattr(args, "only_bias", False)))
                    row = dict(H=int(H), arm=kind, seed=int(sd), lr=float(lr),
                               tokens=ws.tokens_name, tag=args.tag,
                               J_fit=r["per_window"]["fit"]["J"],
                               J_val=r["per_window"]["val"]["J"],
                               dJ_fit=float(r["per_window"]["fit"]["J"].mean() - base["fit"].mean()),
                               dJ_val=float(r["per_window"]["val"]["J"].mean() - base["val"].mean()),
                               delta_norm=r["per_window"]["fit"]["delta_norm"],
                               act_rms=r["per_window"]["fit"]["act_rms"],
                               state_sens=r["state_sens"], hist=r["hist"])
                    rows.append(row)
                    log(f"[H={H}] {kind:8s} lr={lr:g} dJ_fit={row['dJ_fit']:+.5f} "
                        f"dJ_val={row['dJ_val']:+.5f} |Δ|={row['delta_norm']:.4f}")
                    _append_jsonl(Path(args.out) / f"horizon_runs_{args.tag or 'main'}.jsonl", row)
    return rows


# =========================================================================== 汇总
def _dedup(rows: list[dict], keys: tuple) -> list[dict]:
    """同一 key 保留**最后一条**（并行 worker 重跑同一配置时不会把表算重）。"""
    seen: dict = {}
    for r in rows:
        seen[tuple(r.get(k) for k in keys)] = r
    return list(seen.values())


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def cmd_report(env: Env | None, args) -> dict:
    """把逐次运行聚合成任务书要求的 5 个产品文件。"""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs, hz = [], []
    for f in sorted(out.glob("arms_runs*.jsonl")):
        runs += _read_jsonl(f)
    for f in sorted(out.glob("horizon_runs*.jsonl")):
        hz += _read_jsonl(f)
    if args.tag:
        runs = [r for r in runs if r.get("tag") == args.tag]
        hz = [r for r in hz if r.get("tag") == args.tag]
    runs = _dedup(runs, ("arm", "size", "seed", "lr", "ablate", "oracle_scope", "tokens"))
    hz = _dedup(hz, ("arm", "H", "seed", "lr", "tokens"))
    metrics: dict = {"arms": [], "horizon": []}

    import csv

    # ---- per-window returns ----
    with (out / "per_window_returns.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["arm", "size", "seed", "lr", "tokens", "split", "window", "J_arm", "J_base",
                     "dJ", "delta_norm_mean", "act_rms_split"])
        for r in runs:
            for split in ("fit", "val", "test"):
                js = r.get(f"perwin_{split}")
                bs = r.get(f"baseperwin_{split}")
                if not js or not bs:
                    continue
                for i, (a, b) in enumerate(zip(js, bs)):
                    wr.writerow([r["arm"], r["size"], r["seed"], r["lr"], r.get("tokens", ""),
                                 split, i, a, b, a - b, r.get("delta_norm", ""),
                                 r.get(f"act_rms_{split}", "")])
    # ---- learning curves ----
    with (out / "learning_curves.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["arm", "size", "seed", "lr", "step", "J", "J_std", "loss", "delta_norm",
                     "grad_norm", "survival", "secs"])
        for r in runs + hz:
            for h in r.get("hist", []):
                wr.writerow([r["arm"], r.get("size", r.get("H")), r.get("seed", ""),
                             r.get("lr", ""), h.get("step"), h.get("J"), h.get("J_std"),
                             h.get("loss"), h.get("delta_norm"), h.get("grad_norm"),
                             h.get("survival"), h.get("secs")])
    # ---- ablation.csv ----
    with (out / "ablation.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["kind", "arm/split", "size_or_H", "seed", "lr", "J_fit", "dJ_fit",
                     "dJ_val", "dJ_test", "pos_val", "pos_test", "delta_norm", "act_rms",
                     "state_sens", "n_params", "tokens", "tag"])
        for r in runs:
            wr.writerow(["arms", r["arm"], r["size"], r["seed"], r["lr"],
                         _f(r.get("J_fit")), _f(r.get("dJ_fit")), _f(r.get("dJ_val")),
                         _f(r.get("dJ_test")), _f(r.get("pos_fit")), _f(r.get("pos_test")),
                         _f(r.get("delta_norm")), _f(r.get("act_rms")),
                         _f(r.get("state_sens")), r.get("n_params"), r.get("tokens"), r.get("tag")])
        for r in hz:
            wr.writerow(["horizon", r["arm"], r.get("H"), r.get("seed"), r.get("lr"),
                         "", _f(r.get("dJ_fit")), _f(r.get("dJ_val")), "", "", "",
                         _f(r.get("delta_norm")), _f(r.get("act_rms")),
                         _f(r.get("state_sens")), "", r.get("tokens"), r.get("tag")])
    # ---- grad_stats.csv ----
    gs = []
    for p in sorted(out.glob("gradstruct_*.json")):
        d = json.loads(p.read_text())
        for row in d.get("rows", []):
            gs.append(dict(kind="structure", **row))
        for row in d.get("cv", []):
            gs.append(dict(kind="cv", **row))
        for row in d.get("gain", []):
            gs.append(dict(kind="gain", **row))
    with (out / "grad_stats.csv").open("w", newline="") as f:
        keys = sorted({k for r in gs for k in r})
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in gs:
            wr.writerow(r)

    # ---- metrics.json：任务书那张表的原料 ----
    for r in runs:
        metrics["arms"].append({k: v for k, v in r.items()
                                if not k.startswith(("perwin", "baseperwin", "const_vec", "hist"))})
    for r in hz:
        metrics["horizon"].append({k: v for k, v in r.items()
                                   if not k.startswith(("J_fit", "J_val", "hist"))})
    for name in ("exp0.json", "gradstruct_vla_main.json", "gradstruct_label_main.json",
                 "windows_meta.json"):
        p = out / name
        if p.exists():
            metrics[name.replace(".json", "")] = json.loads(p.read_text())
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    log(f"报告原料写入 {out}：metrics.json / ablation.csv / per_window_returns.csv / "
        f"grad_stats.csv / learning_curves.csv")
    return metrics


def _f(x):
    return "" if x is None else x


# =========================================================================== CLI
def _common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--out", type=Path, default=MAIN_REPO / "runs/harness_03_bptt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bptt-args", type=Path, default=MAIN_REPO / "runs/bptt1/args.json",
                    help="口径来源（bptt1 args.json：wbc/wm/vlm/奖励/token 空间）")
    ap.add_argument("--windows", type=Path, default=None)
    ap.add_argument("--tokens", default="vla", choices=["vla", "label"])
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--cap", type=float, default=ETA)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")


def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("diag_bptt_learnability")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="采窗口 + burn-in + 缓存 VLA token chunk")
    _common(b)
    b.add_argument("--split-seed", type=int, default=20261107)
    b.add_argument("--val-frac", type=float, default=0.15)
    b.add_argument("--test-frac", type=float, default=0.15)
    b.add_argument("--n-train", type=int, default=192)
    b.add_argument("--n-val", type=int, default=96)
    b.add_argument("--n-test", type=int, default=96)
    b.add_argument("--batch", type=int, default=32)
    b.add_argument("--burnin-steps", type=int, default=10)
    b.add_argument("--start-max-drift", type=float, default=0.10)
    b.add_argument("--motion", type=Path, default=_abs(DEFAULTS["motion"]))
    b.add_argument("--vlm-ckpt", type=Path, default=_abs(DEFAULTS["vlm_ckpt"]))
    b.add_argument("--vlm-path", type=Path, default=_abs(DEFAULTS["vlm_path"]))
    b.add_argument("--vla-batch", type=int, default=16)
    b.add_argument("--fm-steps", type=int, default=4)
    b.add_argument("--skip-vla", action="store_true")

    e0 = sub.add_parser("exp0", help="梯度路径审计 + 有限差分")
    _common(e0)
    e0.add_argument("--n", type=int, default=8)
    e0.add_argument("--hidden", type=int, default=1024)
    e0.add_argument("--depth", type=int, default=2)
    e0.add_argument("--init-std", type=float, default=4e-4)
    e0.add_argument("--fd-at", default="grad", choices=["grad", "random"])
    e0.add_argument("--fd-h", type=float, nargs="+",
                    default=[1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4])

    ar = sub.add_parser("arms", help="Base / Constant / State-MLP / Oracle")
    _common(ar)
    ar.add_argument("--sizes", type=int, nargs="+", default=[8, 32, 128])
    ar.add_argument("--arms", nargs="+",
                    default=["base", "constant", "mlp", "oracle"])
    ar.add_argument("--lrs", type=float, nargs="+", default=[3e-3])
    ar.add_argument("--seeds", type=int, nargs="+", default=[0])
    ar.add_argument("--steps", type=int, default=200)
    ar.add_argument("--log-every", type=int, default=10)
    ar.add_argument("--hidden", type=int, default=1024)
    ar.add_argument("--depth", type=int, default=2)
    ar.add_argument("--init-std", type=float, default=4e-4)
    ar.add_argument("--freeze-bias", action="store_true")
    ar.add_argument("--only-bias", action="store_true",
                    help="只训输出层 bias（常量捷径的严格版本）")
    ar.add_argument("--ablate", nargs="*", default=[],
                    choices=["shuffle_state", "shuffle_token", "zero_state", "zero_token"])
    ar.add_argument("--vla-tokens", type=Path, default=None)

    gs = sub.add_parser("gradstruct", help="实验 4：逐窗口梯度结构 + 可预测性")
    _common(gs)
    gs.add_argument("--n-grad", type=int, default=256)
    gs.add_argument("--vla-tokens", type=Path, default=None)

    hz = sub.add_parser("horizon", help="实验 5：H=2/5/20")
    _common(hz)
    hz.add_argument("--hs", type=int, nargs="+", default=[2, 5, 20])
    hz.add_argument("--n-fit", type=int, default=32)
    hz.add_argument("--lrs", type=float, nargs="+", default=[3e-3])
    hz.add_argument("--seeds", type=int, nargs="+", default=[0])
    hz.add_argument("--steps", type=int, default=200)
    hz.add_argument("--log-every", type=int, default=10)
    hz.add_argument("--hidden", type=int, default=1024)
    hz.add_argument("--depth", type=int, default=2)
    hz.add_argument("--init-std", type=float, default=4e-4)
    hz.add_argument("--vla-tokens", type=Path, default=None)

    rp = sub.add_parser("report", help="聚合 metrics/csv")
    _common(rp)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_cli()
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    args_global.grad_clip = float(getattr(a, "grad_clip", 0.0) or 0.0)
    if a.cmd == "build":
        build_windows(a)
        return 0
    if a.cmd == "report":
        cmd_report(None, a)
        return 0
    env = load_env(a.device, Path(a.bptt_args) if a.bptt_args else None)
    if a.cmd == "exp0":
        rows = cmd_exp0(env, a)
        (a.out / "exp0.json").write_text(json.dumps(rows, indent=1, default=float))
        log(f"exp0 -> {a.out/'exp0.json'}")
    elif a.cmd == "arms":
        cmd_arms(env, a)
    elif a.cmd == "gradstruct":
        cmd_gradstruct(env, a)
    elif a.cmd == "horizon":
        cmd_horizon(env, a)
    else:
        raise SystemExit(f"[diag03] 未实现 {a.cmd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
