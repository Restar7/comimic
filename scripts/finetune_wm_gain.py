#!/usr/bin/env python3
"""**两阶段 + 增益约束**的 WM 训练（2026-11-05）

阶段 1：单步（A' 单步样本）+ `L_gain`（配对探针）——**修动作增益**
阶段 2：k=20 多步（A' 窗口）+ 增量项 + **`L_gain`（必须保留）**——保住 20 步保真

为什么阶段 2 也要 `L_gain`：多步阶段用 A' 窗口（录制动作、无配对），A' 的损失对增益
几乎没有约束 ⇒ 去掉 `L_gain` 会把阶段 1 修好的增益**拉回旧解**。

用法::

    python3 scripts/finetune_wm_gain.py \
        --init-wm runs/a_prime_sonic_exc/wm_scratch_ddp.pt \
        --windows runs/a_prime_sonic_exc/roll_windows_s1.pt \
        --pairs   runs/paired_probe/gain_pairs.pt \
        --out     runs/a_prime_sonic_exc/wm_gain.pt \
        --stage1 800 --stage2 800 --gain-weight 2.0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wbc-ckpt", type=Path, default=ROOT / "runs/S1_handover/WBC_A24000_wmvel_cmd.pt")
    ap.add_argument("--init-wm", type=Path,
                    default=ROOT / "runs/a_prime_sonic_exc/wm_scratch_ddp.pt")
    ap.add_argument("--windows", type=Path,
                    default=ROOT / "runs/a_prime_sonic_exc/roll_windows_s1.pt")
    ap.add_argument("--pairs", type=Path, default=ROOT / "runs/paired_probe2/gain_pairs.pt")
    ap.add_argument("--windows2", type=Path, default=None,
                    help="**off-nominal 窗口**（探针 20 步，build_probe_windows.py 产物）；"
                         "给了就在阶段 2 里按 --mix2 比例混采")
    ap.add_argument("--mix2", type=float, default=0.2, help="阶段 2 里探针窗口占比")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/a_prime_sonic_exc/wm_gain.pt")
    ap.add_argument("--stage1", type=int, default=800, help="单步步数")
    ap.add_argument("--stage2", type=int, default=800, help="k 步自回归步数")
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--gain-batch", type=int, default=64, help="每次抽多少个配对组")
    ap.add_argument("--gain-weight", type=float, default=2.0)
    ap.add_argument("--lr1", type=float, default=2e-5)
    ap.add_argument("--lr2", type=float, default=1.5e-5)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--inc-weight", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    from diff_vla_wbc.isaac_joint import load_isaac_bundle  # noqa: PLC0415
    from diff_vla_wbc.wm_refresh import gain_loss, increment_loss, ssm_wm_loss  # noqa: PLC0415

    dev = torch.device(a.device)
    npz = sorted((ROOT / "data/npz/ha_tokens").glob("*.npz"))
    b = load_isaac_bundle(a.wbc_ckpt, device=dev, npz=npz,
                          packed_motion=ROOT / "runs/isaac_motion/ha_cmd_exec/packed.pt",
                          obs_mode=None, hist=None, token_dim=64)
    ck = torch.load(a.init_wm, map_location="cpu", weights_only=False)
    b.wm.load_state_dict(ck.get("wm", ck), strict=True)
    wm = b.wm
    wm.requires_grad_(True)
    print(f"[gain-train] 初值 {a.init_wm.name}", flush=True)

    w = torch.load(a.windows, map_location="cpu", weights_only=False)
    S, A = w["states"].float(), w["actions"].float()
    K = int(a.k)
    print(f"[gain-train] 窗口 {tuple(S.shape)}", flush=True)

    # off-nominal 窗口（可选）
    S2w = A2w = None
    if a.windows2 is not None and Path(a.windows2).exists():
        w2 = torch.load(a.windows2, map_location="cpu", weights_only=False)
        S2w, A2w = w2["states"].float(), w2["actions"].float()
        print(f"[gain-train] off-nominal 窗口 {tuple(S2w.shape)}（占比 {a.mix2:.0%}）", flush=True)

    p = torch.load(a.pairs, map_location="cpu", weights_only=False)
    Ps, Pa, Ps1 = p["s"].float(), p["a"].float(), p["s1"].float()
    Mp = int(Ps.shape[0])
    print(f"[gain-train] 配对 {tuple(Pa.shape)}（{Mp} 组 × {Pa.shape[1]} 个动作）", flush=True)

    params = list(wm.parameters())
    g = torch.Generator().manual_seed(int(a.seed))
    xi_of = lambda n: b.default_xi(n, dev)  # noqa: E731

    def one_gain(n):
        j = torch.randint(Mp, (n,), generator=g)
        return gain_loss(wm, b.step, Ps[j].to(dev), Pa[j].to(dev), Ps1[j].to(dev), xi_of)

    # ---------------- 阶段 1：单步 + 增益 ----------------
    opt = torch.optim.Adam(params, lr=float(a.lr1))
    N = int(S.shape[0])
    t0 = time.time()
    for it in range(int(a.stage1)):
        idx = torch.randint(N, (int(a.batch),), generator=g)
        s0 = S[idx, 0].to(dev)
        a0 = A[idx, 0].to(dev)
        s1 = S[idx, 1].to(dev)
        pred = b.step(s0, a0, xi_of(s0.shape[0]))
        l_wm = ssm_wm_loss(wm, pred, s1)
        l_g = one_gain(int(a.gain_batch)) if float(a.gain_weight) > 0 else pred.new_zeros(())
        loss = l_wm + float(a.gain_weight) * l_g
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if it % int(a.log_every) == 0 or it == int(a.stage1) - 1:
            print(f"  [s1 {it:4d}/{a.stage1}] wm={float(l_wm):.4f} gain={float(l_g):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # ---------------- 阶段 2：k 步自回归 + 增益 ----------------
    opt = torch.optim.Adam(params, lr=float(a.lr2))
    t1 = time.time()
    for it in range(int(a.stage2)):
        if S2w is not None and float(a.mix2) > 0:
            nb = int(a.batch)
            n2 = max(1, int(round(nb * float(a.mix2))))
            n1 = nb - n2
            i1 = torch.randint(N, (n1,), generator=g)
            i2 = torch.randint(int(S2w.shape[0]), (n2,), generator=g)
            xb = torch.cat([S[i1, 0], S2w[i2, 0]]).to(dev)
            ab = torch.cat([A[i1, :K], A2w[i2, :K]]).to(dev)
            tb = torch.cat([S[i1, 1:], S2w[i2, 1:]]).to(dev)
        else:
            idx = torch.randint(N, (int(a.batch),), generator=g)
            xb, ab, tb = S[idx, 0].to(dev), A[idx, :K].to(dev), S[idx, 1:].to(dev)
        x, acts, tgts = xb, ab, tb
        loss_m = torch.zeros((), device=dev)
        for t in range(K):
            x_prev = x
            x = b.step(x, acts[:, t], xi_of(x.shape[0]))
            loss_m = loss_m + ssm_wm_loss(wm, x, tgts[:, t])
            if float(a.inc_weight) > 0:
                loss_m = loss_m + float(a.inc_weight) * increment_loss(x, x_prev, tgts[:, t])
        loss_m = loss_m / K
        l_g = one_gain(int(a.gain_batch)) if float(a.gain_weight) > 0 else loss_m.new_zeros(())
        loss = loss_m + float(a.gain_weight) * l_g
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if it % int(a.log_every) == 0 or it == int(a.stage2) - 1:
            print(f"  [s2 {it:4d}/{a.stage2}] wm={float(loss_m):.4f} gain={float(l_g):.4f} "
                  f"({time.time()-t1:.0f}s)", flush=True)

    wm.eval()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"wm": wm.state_dict(), "gain_pairs": str(a.pairs),
                "stage1": int(a.stage1), "stage2": int(a.stage2),
                "gain_weight": float(a.gain_weight)}, a.out)
    print(f"[gain-train] -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
