#!/usr/bin/env python3
"""α 剂量扫描：**同一个方向**下，修正幅度对 WM 回报/动作偏差的影响（2026-11-07）。

口径：Δ = α · 0.08 · unit(g)，g = ∂J₂₀/∂token（窗口级），WM 用**标签口径**的
`runs/a_prime_sonic_exc/wm_gain2.pt`，K=H=20，discounted γ=0.99。
"""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=str(ROOT / "runs/harness_02_teacher/dump_h20/teacher_*.npz"))
    ap.add_argument("--ckpt", default=str(ROOT / "runs/S1_handover/WBC_A24000_wmvel_cmd.pt"))
    ap.add_argument("--wm-ckpt", default=str(ROOT / "runs/a_prime_sonic_exc/wm_gain2.pt"))
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.125, 0.25, 0.5, 1.0])
    ap.add_argument("--cap", type=float, default=0.08)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    import numpy as np, torch
    from diff_vla_wbc import teacher_probe as tp
    from diff_vla_wbc.isaac_joint import FrameRef, load_isaac_bundle
    from diff_vla_wbc.rollout import discounted_return
    dev = torch.device(a.device); K = 20; T = 40
    D = {}
    for f in sorted(glob.glob(a.dump)):
        d = np.load(f)
        for k in ("token_chunk", "state", "xi", "prev_action", "history", "obs_hist", "frames"):
            D.setdefault(k, []).append(d[k])
    ax = lambda k: 1 if k in ("history", "frames") else 0
    D = {k: np.concatenate(v, axis=ax(k)) for k, v in D.items()}
    N = int(a.n)
    b = load_isaac_bundle(a.ckpt, device=dev,
                          npz=sorted((ROOT / "data/npz/ha_tokens").glob("*.npz")),
                          packed_motion=ROOT / "runs/isaac_motion/ha_cmd/packed.pt",
                          obs_mode=None, hist=None, token_dim=64)
    if a.wm_ckpt:
        wd = torch.load(a.wm_ckpt, map_location="cpu", weights_only=False)
        b.wm.load_state_dict(wd.get("wm", wd), strict=True)
        print(f"[wm] {a.wm_ckpt}（= 生成标签用的那个）")
    tok = torch.as_tensor(D["token_chunk"][:N], dtype=torch.float32, device=dev)
    ctx = dict(state=torch.as_tensor(D["state"][:N], dtype=torch.float32, device=dev),
               references=[FrameRef(torch.as_tensor(D["frames"][t, :N], dtype=torch.long, device=dev))
                           for t in range(K)],
               xi=torch.as_tensor(D["xi"][:N], dtype=torch.float32, device=dev),
               prev_action=torch.as_tensor(D["prev_action"][:N], dtype=torch.float32, device=dev),
               history=torch.as_tensor(D["history"][:, :N], dtype=torch.float32, device=dev),
               obs_hist=None)
    mask = torch.cat([torch.ones(K, device=dev), torch.zeros(T - K, device=dev)])[None, :, None]
    with torch.no_grad():
        r0 = tp._rollout(b, tok, horizon=K, **ctx); J0 = discounted_return(r0, 0.99)
        a0 = r0.actions.detach()
    g = tp.tokenwise_reward_grad(b, tok, horizon=K, gamma=0.99, reward_steps=0, **ctx).g
    gu = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    print(f"[base] J0={float(J0.mean()):.4f}  a0_rms={float(a0.pow(2).mean().sqrt()):.3f}")
    print(f"{'alpha':>7}{'J_WM':>9}{'dJ':>9}{'dJ>0':>7}{'act_rms':>9}{'peak':>8}{'‖Δtok‖':>9}{'dJ/act':>9}")
    rows = []
    for al in a.alphas:
        d = (float(al) * float(a.cap) * gu[:, None, :].expand(-1, T, -1)).clone() * mask
        with torch.no_grad():
            r = tp._rollout(b, tok + d, horizon=K, **ctx)
            J = discounted_return(r, 0.99); an = r.actions.detach()
            dev_a = an - a0
            ar = float(dev_a.pow(2).mean().sqrt())
            pk = float(dev_a.abs().amax(dim=(0, 2)).mean())
            dJ = float((J - J0).mean())
        rows.append(dict(alpha=float(al), dJ=dJ, J=float(J.mean()), act=ar, peak=pk,
                         tok=float(d.norm(dim=-1).mean()), pos=float((J > J0).float().mean())))
        print(f"{al:>7.3f}{float(J.mean()):>9.4f}{dJ:>+9.4f}{rows[-1]['pos']:>7.2f}{ar:>9.3f}"
              f"{pk:>8.3f}{rows[-1]['tok']:>9.4f}{dJ / max(ar, 1e-9):>9.3f}")
    import json
    out = ROOT / "runs/alpha_scan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1)); print(f"[write] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
