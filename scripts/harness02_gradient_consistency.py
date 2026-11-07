#!/usr/bin/env python3
"""harness02：在**真 WBC + 真 WM + 真奖励**上验证共享 Δ 的 Teacher 梯度。

验的三件事（同一个 ``J`` 定义下）
--------------------------------
1. **链式法则一致性**：``g_A = dJ/dε``（ε 是 (B,1,64) 共享变量）
   vs ``g_B = Σ_t dJ/dtoken_t``（对 (B,T,64) 叶子求导再沿 T 求和）。
   两者在数学上是同一个量；不一致 = 图里有 detach / 重复索引 / 广播写错。
2. **数值量级**：cos、max|Δ|、相对 L2、逐元素同号率。
3. **有限差分方向导数**：``(J(ε+hv) − J(ε−hv))/2h`` vs ``g·v``，扫 h 找稳定区。
   真 WM 里有 huber / exp / clamp（非光滑），所以必须**扫 h** 而不是挑一个数。

数据来源
--------
* ``--source recorded``：从 ``runs/a_prime_sonic_exc/all_transitions.pt``（真实 Isaac 转移）
  取起点，参考帧来自 motion library，token chunk 来自数据集标签（录制 token）。
  只用于**数值一致性**——不用于"可学性"结论。
* ``--source dump``：读 ``J1_TEACHER_DUMP`` 落盘的**生产窗口**（VLA 输出的 token chunk +
  同一个 batch 的 state/ref/xi/prev/history）。这是 Teacher 真正工作的那个分布。

用法::

    COMIMIC=<repo> bash -c 'source scripts/deps_env.sh; python3 scripts/harness02_gradient_consistency.py \
        --source recorded --out runs/harness_02_teacher --device cuda:1'
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("SUPERMIMIC_HA_ALIGN", "1")
os.environ.setdefault(
    "SUPERMIMIC_ROBOT_USD",
    str(ROOT / "trackerlab-supermimic/data/assetslib/unitree/unitree_g1/usd/g1_29dof_rev_1_0.usd"),
)

import torch  # noqa: E402

from diff_vla_wbc import teacher_probe as tp  # noqa: E402
from diff_vla_wbc.isaac_joint import FrameRef, default_xi, load_isaac_bundle  # noqa: E402

TRANSITIONS = ROOT / "runs/a_prime_sonic_exc/all_transitions.pt"
WBC_CKPT = ROOT / "runs/S1_handover/WBC_A24000_wmvel_cmd.pt"
WM_CKPT = ROOT / "runs/a_prime_sonic_exc/wm_gain2.pt"
PACKED = ROOT / "runs/isaac_motion/ha_cmd/packed.pt"
NPZ_DIR = ROOT / "data/npz/ha_tokens"
ARGS_JSON = ROOT / "runs/gt7fix/args.json"


def build_bundle(device: str, no_termination: bool = True):
    """与 ``scripts/train_joint_v2.py`` 的 ``--executor sonic`` 分支**同一套装配**。"""
    import argparse as _ap

    from diff_vla_wbc.reward_cfg import apply_reward_cfg, reward_cfg_summary
    from diff_vla_wbc.sonic_isaac import SonicIsaacBundle

    base = load_isaac_bundle(
        WBC_CKPT, device=device, packed_motion=str(PACKED), obs_mode=None, hist=None, token_dim=64,
    )
    bundle = SonicIsaacBundle(base, onnx_path=None, token_quant="none", device=device)
    ns = _ap.Namespace(**json.loads(ARGS_JSON.read_text())) if ARGS_JSON.exists() else _ap.Namespace()
    apply_reward_cfg(bundle, ns)
    if no_termination:
        bundle.no_termination = 1.0
    _wd = torch.load(str(WM_CKPT), map_location="cpu", weights_only=False)
    base.wm.load_state_dict(_wd.get("wm", _wd), strict=True)
    print(f"[bundle] {reward_cfg_summary(bundle)}", flush=True)
    return bundle


def _clip_end(lib, frames: torch.Tensor) -> torch.Tensor:
    return lib.clip_end[lib.clip_id[frames.long()]]


def windows_from_transitions(bundle, n: int, H: int, seed: int, device: str):
    """真实转移起点 + 数据集 token chunk（录制 token）。"""
    from diff_vla_wbc.data import RecordedTokenSource

    d = torch.load(str(TRANSITIONS), map_location="cpu", weights_only=False)
    states, actions, frames, clips = d["states"], d["actions"], d["frames"], d["clip_ids"]
    lib = bundle.library
    import numpy as np

    rng = np.random.default_rng(seed)
    end = _clip_end(lib, frames).cpu()
    clips = clips.cpu()
    ok = (frames.cpu() + H + 2 <= end) & (torch.arange(frames.numel()) >= 9)
    ok &= torch.cat([torch.zeros(9, dtype=torch.bool), clips[9:] == clips[:-9]])
    idx_all = ok.nonzero(as_tuple=False).flatten().numpy()
    idx = torch.from_numpy(np.sort(rng.choice(idx_all, size=n, replace=False)))
    dev = torch.device(device)
    start = states[idx].to(dev)
    xi = default_xi(int(n), dev)
    prev = actions[(idx - 1).clamp_min(0)].to(dev)
    obs = []
    for j in range(9, -1, -1):
        obs.append(bundle.observation(states[idx - j].to(dev), actions[idx - j].to(dev),
                                      xi, FrameRef(frames[idx - j].to(dev))))
    history = torch.stack(obs, 0)  # (10,B,W)
    f0 = frames[idx]
    _end0 = _clip_end(lib, frames[idx]).cpu()
    refs = [FrameRef(torch.minimum(f0 + t, _end0).to(dev)) for t in range(H)]
    src = RecordedTokenSource(NPZ_DIR, lib, token_horizon=40, token_dim=64, device=dev,
                              port_dir=None, token_space="label")
    chunk = src({"frames": f0}).to(dev)
    return {
        "state": start, "references": refs, "xi": xi, "prev_action": prev,
        "history": history, "obs_hist": None, "tokens": chunk,
        "frames": f0, "clips": clips[idx],
    }


def windows_from_dump(path: Path, device: str):
    """生产窗口（VLA token chunk + 同 batch 的 state/ref/xi/prev/history）。"""
    import numpy as np

    z = np.load(path)
    dev = torch.device(device)
    state = torch.as_tensor(z["state"], device=dev)
    n = int(state.shape[0])
    frames = torch.as_tensor(z["frames"], device=dev) if "frames" in z else None
    H = int(z["H"]) if "H" in z else (frames.shape[0] if frames is not None else 20)
    if frames is not None:
        refs = [FrameRef(frames[t].long()) for t in range(frames.shape[0])]
    else:
        refs = []
    hist = torch.as_tensor(z["history"], device=dev) if "history" in z else None
    oh = torch.as_tensor(z["obs_hist"], device=dev) if "obs_hist" in z else None
    if hist is not None and int(hist.shape[1]) != n and int(hist.shape[0]) != n:
        hist = None
    return {
        "state": state,
        "references": refs,
        "xi": torch.as_tensor(z["xi"], device=dev) if "xi" in z and z["xi"].size else default_xi(n, dev),
        "prev_action": (torch.as_tensor(z["prev_action"], device=dev)
                        if "prev_action" in z and z["prev_action"].size else None),
        "history": hist, "obs_hist": oh,
        "tokens": torch.as_tensor(z["token_chunk"], device=dev),
        "frames": frames, "clips": None, "H_file": H,
    }


def _sign_agree(a: torch.Tensor, b: torch.Tensor) -> float:
    sa, sb = torch.sign(a), torch.sign(b)
    nz = (sa != 0) & (sb != 0)
    if int(nz.sum()) == 0:
        return float("nan")
    return float((sa[nz] == sb[nz]).float().mean())


def run(win, H: int, gamma: float, eps_scale: float, fd_hs, device: str, seed: int = 0):
    B = int(win["state"].shape[0])
    tok = win["tokens"]
    if eps_scale > 0:
        g = torch.Generator(device="cpu").manual_seed(1000 + seed)
        eps0 = torch.randn(B, 1, tok.shape[-1], generator=g)
        eps0 = eps0 / eps0.norm(dim=-1, keepdim=True) * eps_scale
        eps0 = eps0.to(device)
    else:
        eps0 = None
    common = dict(state=win["state"], references=win["references"], xi=win["xi"],
                  prev_action=win["prev_action"], history=win["history"],
                  obs_hist=win["obs_hist"], horizon=H, gamma=gamma, reward_steps=0)
    ga = tp.shared_delta_reward_grad(tp_bundle[0], tok, eps0=eps0, **common)
    gb = tp.tokenwise_reward_grad(tp_bundle[0], tok, eps0=eps0, **common)
    diff = (ga.g - gb.g)
    cos = tp.cosine(ga.g, gb.g)
    row = {
        "H": H, "seed": seed, "n": B, "eps_scale": eps_scale,
        "cos_AB": float(cos.mean()),
        "cos_AB_min": float(cos.min()),
        "max_abs_err": float(diff.abs().max()),
        "rel_l2": float(diff.norm() / gb.g.norm().clamp_min(1e-12)),
        "sign_agree": _sign_agree(ga.g, gb.g),
        "gA_norm_mean": float(ga.g.norm(dim=-1).mean()),
        "gB_norm_mean": float(gb.g.norm(dim=-1).mean()),
        "J_mean_disc": float(ga.J.mean()),
        "J_mean_sum": float(ga.J_undisc.mean()),
    }
    # ---- 有限差分：方向 = 归一化梯度（SNR 最高）----
    v = ga.g / ga.g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    v = v[:, None, :]
    gv = (ga.g * v[:, 0]).sum(-1)
    fd_rows = []
    for h in fd_hs:
        fd = tp.finite_difference_directional(tp_bundle[0], tok, direction=v, h=h,
                                              eps0=eps0, **common)
        rel = float((fd - gv).norm() / gv.norm().clamp_min(1e-12))
        c = float(tp.cosine(fd[:, None], gv[:, None]).mean())
        fd_rows.append({"H": H, "seed": seed, "n": B, "eps_scale": eps_scale,
                        "fd_h": h, "fd_rel_err": rel, "fd_cos": c,
                        "fd_mean": float(fd.mean()), "gv_mean": float(gv.mean())})
        print(f"    H={H:<3d} seed={seed} eps={eps_scale:g} h={h:<8g} "
              f"fd_rel_err={rel:.3e} fd_cos={c:.6f}", flush=True)
    return row, fd_rows


tp_bundle: list = [None]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["recorded", "dump"], default="recorded")
    ap.add_argument("--dump", default="", help="teacher_*.npz 文件或目录（--source dump）")
    ap.add_argument("--out", default=str(ROOT / "runs/harness_02_teacher"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--horizons", default="1,2,5,20")
    ap.add_argument("--eps-scales", default="0,0.01,0.08")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--fd-h", default="0.1,0.03,0.01,0.003,0.001")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bundle = build_bundle(args.device)
    tp_bundle[0] = bundle
    torch.manual_seed(0)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    eps_scales = [float(s) for s in args.eps_scales.split(",") if s.strip()]
    fd_hs = [float(s) for s in args.fd_h.split(",") if s.strip()]

    rows, fd_rows = [], []
    if args.source == "recorded":
        for seed in seeds:
            Hmax = max(horizons)
            win = windows_from_transitions(bundle, args.n, Hmax, seed, args.device)
            print(f"[windows] seed={seed} n={args.n} Hmax={Hmax} "
                  f"token_rms={float(win['tokens'].pow(2).mean().sqrt()):.4f}", flush=True)
            for H in horizons:
                for esc in eps_scales:
                    r, f = run(win, H, args.gamma, esc, fd_hs, args.device, seed)
                    rows.append({**r, "source": "recorded"})
                    fd_rows.extend([{**x, "source": "recorded"} for x in f])
                    print(f"[gA vs gB] source=recorded seed={seed} H={H} eps={esc:g} "
                          f"cos={r['cos_AB']:.9f} max_abs={r['max_abs_err']:.3e} "
                          f"rel_l2={r['rel_l2']:.3e} sign={r['sign_agree']:.4f}", flush=True)
    else:
        files = sorted(Path(args.dump).glob("teacher_*.npz")) if Path(args.dump).is_dir() \
            else [Path(args.dump)]
        if not files:
            raise SystemExit(f"[fail] {args.dump} 下没有 teacher_*.npz")
        for fi, path in enumerate(files):
            win = windows_from_dump(path, args.device)
            if not win["references"]:
                print(f"[skip] {path.name} 没有 frames（references 缺失）", flush=True)
                continue
            print(f"[windows] dump={path.name} n={int(win['state'].shape[0])}", flush=True)
            for H in horizons:
                for esc in ([0.0] if fi else eps_scales):
                    r, f = run(win, H, args.gamma, esc, fd_hs, args.device, fi)
                    rows.append({**r, "source": f"dump:{path.name}"})
                    fd_rows.extend([{**x, "source": f"dump:{path.name}"} for x in f])
                    print(f"[gA vs gB] dump={path.name} H={H} eps={esc:g} "
                          f"cos={r['cos_AB']:.9f} max_abs={r['max_abs_err']:.3e} "
                          f"rel_l2={r['rel_l2']:.3e} sign={r['sign_agree']:.4f}", flush=True)

    import csv

    csv_path = Path(args.csv) if args.csv else (out / "gradient_consistency.csv")
    mode = "a" if csv_path.exists() else "w"
    with open(csv_path, mode, newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if mode == "w":
            w.writeheader()
        w.writerows(rows)
    fd_path = out / "gradient_consistency_fd.csv"
    mode = "a" if fd_path.exists() else "w"
    with open(fd_path, mode, newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fd_rows[0].keys()))
        if mode == "w":
            w.writeheader()
        w.writerows(fd_rows)
    print(f"[done] {len(rows)} consistency rows / {len(fd_rows)} FD rows -> {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
