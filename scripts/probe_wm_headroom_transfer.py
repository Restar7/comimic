#!/usr/bin/env python3
"""**headroom 有效性检验**：大 headroom 是"模型自欺"还是"录制动作本来就不最优"？

为什么必须做这一步（2026-11-05 自我纠错）
----------------------------------------
`probe_wm_headroom.py` 发现"自由优化 20 步动作能把跟踪误差砍 92%"，我据此判定"WM 的动作通道
是伪通道"。但这个推理有一个**漏洞**：录制动作是 SONIC decoder 跟着参考走出来的，它带**滞后/残差**，
本来就不是"最小化 20 步跟踪误差"的最优控制 ⇒ **即使动力学模型完全正确，也会显示很大 headroom**。
所以 headroom 本身不能证明模型有错。

正确的检验是**模型交叉验证**（model disagreement）：
  * 把数据按任务分成**互不相交**的两半 A / B；
  * 在 A 上训一个 WM_A，在 B 上训一个 WM_B（同初值、同步数）；
  * 在一批窗口上，用 WM_A 自由优化动作得到 a*；
  * 关键问题：**a* 在 WM_B 眼里也变好吗？**
      - 转移（B 上同样变好）⇒ 这个方向是**真实动力学** ⇒ headroom 是真的，
        那么 headroom 门必须**重新标定**（用 A→B 的转移率当判据）；
      - 不转移（B 上没变好甚至更差）⇒ 是**模型专属的伪方向** ⇒ 原结论成立。

用法::

    python3 scripts/probe_wm_headroom_transfer.py --device cuda:3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

TASK_A = ("DoubleDesk", "PPBox", "Boxing")
TASK_B = ("OpenDoor", "SitSofa", "VisionNavi")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=ROOT / "runs/a_prime_sonic_exc")
    ap.add_argument("--init", type=Path, default=None,
                    help="两半共同的初值（默认 <dir>/wm_stage2.pt）")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--pairs", type=Path, default=ROOT / "runs/paired_probe/gain_pairs.pt",
                    help="配对探针数据；给了就在两个 WM 的训练里**同样**加 L_gain（否则测的是没修过的模型）")
    ap.add_argument("--gain-steps", type=int, default=200)
    ap.add_argument("--pairs-a", type=Path, default=None)
    ap.add_argument("--pairs-b", type=Path, default=None)
    ap.add_argument("--windows2-a", type=Path, default=None,
                    help="A 半任务的探针 20 步窗口（off-nominal）")
    ap.add_argument("--windows2-b", type=Path, default=None,
                    help="B 半任务的探针 20 步窗口")
    ap.add_argument("--mix2", type=float, default=0.2, help="探针窗口在多步训练里的占比")
    ap.add_argument("--gain-weight", type=float, default=2.0)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    from diff_vla_wbc.isaac_joint import load_isaac_bundle  # noqa: PLC0415
    from diff_vla_wbc.wm_refresh import train_multistep  # noqa: PLC0415

    dev = torch.device(a.device)
    npz = sorted((ROOT / "data/npz/ha_tokens").glob("*.npz"))
    init = a.init or (a.dir / "wm_stage2.pt")
    files = sorted(a.dir.glob("*_off*.pt"))
    fa = [f for f in files if any(t in f.name for t in TASK_A)]
    fb = [f for f in files if any(t in f.name for t in TASK_B)]
    print(f"[xfer] 数据 A {len(fa)} shard / B {len(fb)} shard；初值 {init}", flush=True)

    def windows(fl):
        S, A = [], []
        for f in fl:
            z = torch.load(f, map_location="cpu", weights_only=False)
            st, ac, cid, fr = z["states"], z["actions"], z["clip_ids"], z["frames"]
            # 只取"同 clip 且帧连续"的 k+1 窗口（与 build_wm_roll_windows 同口径）
            for c in torch.unique(cid).tolist():
                m = (cid == c).nonzero().flatten()
                m = m[torch.argsort(fr[m])]
                i = 0
                while i + int(a.k) < len(m):
                    if int(fr[m[i + int(a.k)]] - fr[m[i]]) == int(a.k):
                        S.append(st[m[i:i + int(a.k) + 1]])
                        A.append(ac[m[i:i + int(a.k)]])
                        i += 4
                    else:
                        i += 1
        return torch.stack(S), torch.stack(A)

    SA, AA = windows(fa)
    SB, AB = windows(fb)
    print(f"[xfer] 窗口 A {tuple(SA.shape)} / B {tuple(SB.shape)}", flush=True)

    def make_wm():
        b = load_isaac_bundle(ROOT / "runs/S1_handover/WBC_A24000_wmvel_cmd.pt", device=dev, npz=npz,
                              packed_motion=ROOT / "runs/isaac_motion/ha_cmd_exec/packed.pt",
                              obs_mode=None, hist=None, token_dim=64)
        ck = torch.load(init, map_location="cpu", weights_only=False)
        b.wm.load_state_dict(ck.get("wm", ck), strict=True)
        return b

    wms = {}
    _pair_of = {"A": a.pairs_a, "B": a.pairs_b}
    _win_of = {"A": a.windows2_a, "B": a.windows2_b}
    for tag, (S, A) in (("A", (SA, AA)), ("B", (SB, AB))):
        # ★ 2026-11-05：把**该侧的**探针 20 步窗口按 mix2 混进来（完整配方）
        _w2 = _win_of[tag]
        if _w2 is not None and Path(_w2).exists():
            _z = torch.load(_w2, map_location="cpu", weights_only=False)
            _S2, _A2 = _z["states"].float(), _z["actions"].float()
            _n2 = max(1, int(round(S.shape[0] * float(a.mix2) / max(1e-9, 1 - float(a.mix2)))))
            _r = torch.randperm(_S2.shape[0])[:_n2]
            S = torch.cat([S, _S2[_r]])
            A = torch.cat([A, _A2[_r]])
            print(f"[xfer] WM_{tag} 训练窗口 = A/A' {39173 if tag == 'A' else 32370} + 探针 {int(_r.numel())}"
                  f"（占比 {float(a.mix2):.0%}）", flush=True)
        b = make_wm()
        t0 = __import__("time").time()
        out = train_multistep(b.wm, S.float().to(dev), A.float().to(dev),
                              steps=int(a.steps), batch=int(a.batch), lr=float(a.lr),
                              k=int(a.k), mag_weight=10.0, inc_weight=2.0, device=dev,
                              step_fn=b.step, xi_fn=(lambda n: b.default_xi(n, dev)),
                              seed=0, log=lambda m: None)
        print(f"[xfer] WM_{tag} 训完 loss {out['loss_first']:.3f}→{out['loss_last']:.3f} "
              f"({__import__('time').time()-t0:.0f}s)", flush=True)
        _pp_path = _pair_of[tag] or a.pairs
        if _pp_path is not None and Path(_pp_path).exists() and int(a.gain_steps) > 0:
            from diff_vla_wbc.wm_refresh import gain_loss  # noqa: PLC0415
            pp = torch.load(_pp_path, map_location="cpu", weights_only=False)
            Ps, Pa, Ps1 = pp["s"].float().to(dev), pp["a"].float().to(dev), pp["s1"].float().to(dev)
            wm_ = b.wm
            wm_.requires_grad_(True)
            o = torch.optim.Adam(wm_.parameters(), lr=2e-5)
            gg = torch.Generator().manual_seed(0)
            for _ in range(int(a.gain_steps)):
                j = torch.randint(Ps.shape[0], (32,), generator=gg)
                lg = gain_loss(wm_, b.step, Ps[j], Pa[j], Ps1[j],
                               lambda n: b.default_xi(n, dev))
                o.zero_grad(set_to_none=True)
                lg.backward()
                torch.nn.utils.clip_grad_norm_(wm_.parameters(), 1.0)
                o.step()
            print(f"[xfer] WM_{tag} 加了 L_gain {int(a.gain_steps)} 步，末值 {float(lg):.5f}", flush=True)
        b.wm.requires_grad_(False)
        b.wm.eval()
        wms[tag] = b

    # ---- 在 A 上优化动作，看能不能在 B 上同样变好 -----------------------------
    g = torch.Generator().manual_seed(0)
    sel = torch.randperm(int(SA.shape[0]), generator=g)[: a.n]
    s0 = SA[sel, 0].to(dev)
    acts0 = AA[sel].to(dev)
    xis = {t: wms[t].default_xi(s0.shape[0], dev) for t in wms}

    def err_on(tag: str, seq: torch.Tensor) -> float:
        b = wms[tag]
        s = s0
        with torch.no_grad():
            for t in range(int(a.k)):
                o = b.step(s, seq[:, t], xis[tag])
                s = o[0] if isinstance(o, (tuple, list)) else o
        # 与训练目标同口径：终点位置误差（cm）
        return float((s[:, 0:3] - SA[sel, int(a.k)].to(dev)[:, 0:3]).norm(dim=-1).median()) * 100

    acts = acts0.clone().requires_grad_(True)
    opt = torch.optim.Adam([acts], lr=0.02)
    bb = wms["A"]
    for it in range(int(a.iters)):
        s = s0
        loss = torch.zeros((), device=dev)
        for t in range(int(a.k)):
            o = bb.step(s, acts[:, t], xis["A"])
            s = o[0] if isinstance(o, (tuple, list)) else o
            loss = loss + (s[:, 0:3] - SA[sel, t + 1].to(dev)[:, 0:3]).norm(dim=-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    a_star = acts.detach()

    r = {}
    for tag in ("A", "B"):
        r[tag] = (err_on(tag, acts0), err_on(tag, a_star))
    print()
    print(f"[xfer] 基线（录制动作）: WM_A {r['A'][0]:.3f} cm   WM_B {r['B'][0]:.3f} cm")
    print(f"[xfer] 用 WM_A 优化出的动作: 在 WM_A 上 {r['A'][1]:.3f} cm   在 **WM_B** 上 {r['B'][1]:.3f} cm")
    gA = 1 - r["A"][1] / max(r["A"][0], 1e-9)
    gB = 1 - r["B"][1] / max(r["B"][0], 1e-9)
    print(f"[xfer] 收益：WM_A 上 {gA*100:+.1f}%   WM_B 上 {gB*100:+.1f}%   "
          f"转移率 {gB/max(gA,1e-9):.2f}")
    print("[xfer] 判读：转移率 ≥0.5 ⇒ 方向是真实动力学（headroom 门需重新标定）；"
          "≤0.2 ⇒ 模型专属伪方向（原结论成立）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
