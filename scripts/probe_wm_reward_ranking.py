#!/usr/bin/env python3
"""判据：**WM 的"动作差分回报"与真实物理一致吗？**（即：梯度有没有信号）

为什么是这个判据（2026-11-01，用户口径修正）：
    梯度反传不需要"排序两个 token"，它需要的是 `∂J/∂a` 是**有效上升方向**。
    等价可测形式：同一个起点、两串**只差一点点**的动作（真实动作 vs 加噪动作），
    WM 预测的"哪串回报更高"和真物理的"哪串回报更高"是否一致。
    一致率高 ⇒ 梯度有信号（能反传）；≈50% ⇒ 梯度是噪声（怎么传都没用）。

为什么不用新采数据：
    `runs/Bw_pert_a/` 就是现成的实验材料——B 线起点 + 真物理 + 动作空间扰动
    （σ_a = 0.02/0.05/0.10 rad），每条 rollout 都记了 `states`（ha_v31 序）、
    `q_target_applied`（standard 序）、`frames`、`src_dump`。
    基线（不扰动）就是 `src_dump` 里同一段的真实轨迹 ⇒ 天然配对。

输出：
    * σ 分档的"符号一致率"（WM 认为更好的那串，真物理里是不是真的更好）
    * ΔJ 的相关系数（WM 的 ΔJ vs 真实 ΔJ）
    * ΔJ 的量级（信号大小）vs WM 的误差量级 ⇒ 信噪比的直接观测

用法：
    PYTHONPATH=. python3 scripts/probe_wm_reward_ranking.py \
        --ckpt runs/S1_handover/wm_a24_k20.pt \
        --pert 'runs/Bw_pert_a/bw_w0.pt' --segments 512 --horizon 8 --device cuda:0
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="WM ckpt（要判的那个裁判）")
    ap.add_argument("--pert", default="runs/Bw_pert_a/bw_w0.pt")
    ap.add_argument("--segments", type=int, default=512, help="抽多少个扰动段")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    import joint_contract as JC  # noqa: PLC0415
    from diff_vla_wbc.b_start import state_to_standard  # noqa: PLC0415
    from diff_vla_wbc.isaac_joint import FrameRef, default_xi, load_isaac_bundle  # noqa: PLC0415

    dev = torch.device(a.device)
    perm = torch.as_tensor(JC.permutation("ha_v31", "standard"), dtype=torch.long)
    bundle = load_isaac_bundle(a.ckpt, device=dev,
                               npz=sorted((ROOT / "data/npz/ha_tokens").glob("*.npz")),
                               packed_motion=ROOT / "runs/isaac_motion/ha_cmd/packed.pt",
                               obs_mode=None, hist=None, token_dim=64)
    H = int(a.horizon)

    d = torch.load(a.pert, map_location="cpu", weights_only=False)
    seg_len = int((d.get("meta") or {}).get("seg_len") or 20)
    n_envs = int((d.get("meta") or {}).get("num_envs") or 4096)
    sig = d["sigma"].float()
    print(f"[rank] pert={Path(a.pert).name} rows={d['states'].shape[0]} seg_len={seg_len} "
          f"n_envs={n_envs} sigmas={sorted(set(sig.tolist()))}", flush=True)

    # ---- 抽样：跨 σ 均匀抽 ----
    n_seg = int(d["states"].shape[0]) // seg_len
    g = torch.Generator().manual_seed(0)
    per_sig = a.segments // max(1, len(set(sig.tolist())))
    chosen = []
    for s_val in sorted(set(sig.tolist())):
        pool = torch.nonzero(sig == s_val).flatten()
        if pool.numel() == 0:
            continue
        take = min(per_sig, pool.numel())
        chosen.append(pool[torch.randperm(pool.numel(), generator=g)[:take]])
    sel = torch.cat(chosen)
    print(f"[rank] 抽样 {sel.numel()} 段（每个 σ 各 {per_sig}）", flush=True)

    # ---- 逐段取：起点状态 / 扰动动作 / 真实轨迹 / 基线轨迹 ----
    src_cache: dict[str, dict] = {}
    rows = []
    for s_idx in sel.tolist():
        base = s_idx * seg_len
        st = d["states"][base:base + seg_len]                     # ha_v31
        ac = d["q_target_applied"][base:base + seg_len].float()    # standard
        fr = d["frames"][base:base + seg_len].long()
        # 基线：src_dump 里 frames[0] 之后的那段
        #   ★ `src_dump` 存的是 **basename 列表**（61440 项），不是路径 ⇒ 先建索引
        if not src_cache.get("__idx__"):
            idx = {}
            for f in glob.glob(str(ROOT / "runs/B_*/*/dump/*.pt")):
                idx[Path(f).name] = f
            src_cache["__idx__"] = idx
        base_name = str(d["src_dump"][s_idx])
        if base_name not in src_cache:
            real = src_cache["__idx__"].get(base_name)
            if real is None:
                continue
            src_cache[base_name] = torch.load(real, map_location="cpu", weights_only=False)
        sd = src_cache[base_name]
        _bs = sd["states"].float()
        # ★ 用**状态内容**对齐起点（frames 语义两套不一致：pert 存全局库帧号，dump 里是另一套）
        _d = (_bs[:, :71] - st[0][None, :]).abs().mean(-1)
        pos = int(torch.argmin(_d))
        if float(_d[pos]) > 0.02 or pos + seg_len > _bs.shape[0]:
            continue
        b_st = sd["states"][pos:pos + seg_len].float()
        _bA = sd.get("actions")
        if _bA is None:
            _bA = sd["q_target_applied"]
        b_ac = _bA[pos:pos + seg_len].float().reshape(-1, 29)
        rows.append({"sig": float(sig[s_idx]), "st": st, "ac": ac, "fr": fr,
                     "b_st": b_st, "b_ac": b_ac})
    if not rows:
        raise SystemExit("[rank] 没配成任何一段（检查 src_dump 路径与 frames 对齐）")
    print(f"[rank] 配成 {len(rows)} 段（起点对齐检查见下）", flush=True)

    # ---- 起点一致性自检（应≈0；它证明"同一段"确实起点相同）----
    dstart = max(float((state_to_standard(r["st"][0], perm) - state_to_standard(r["b_st"][0], perm)).abs().max())
                 for r in rows[:64])
    print(f"[rank] 起点最大绝对差 = {dstart:.4f}（ha→standard 后）"
          f"  {'OK' if dstart < 0.2 else '⚠ 起点没对齐，结果不可信'}", flush=True)

    def j_true(states_ha, acts, frame0):
        """真物理回报：用 bundle.reward 在**记录的真实状态**上逐帧算（与想象同口径）。"""
        S = state_to_standard(states_ha, perm)
        tot = torch.zeros(len(states_ha), device=dev)
        a_prev = acts[:, 0].to(dev)
        for t in range(H):
            s_next = S[:, t + 1].to(dev)
            act = acts[:, t].to(dev)
            ref = FrameRef(torch.full((len(states_ha),), int(frame0[0]) + t,
                                      dtype=torch.long, device=dev))
            r = bundle.reward(s_next, act, a_prev, ref)
            tot = tot + r
            a_prev = act
        return tot

    def j_wm(states_ha, acts, frame0):
        """WM 想象回报：从同一真实起点出发，用同一串动作滚 H 步。"""
        S0 = state_to_standard(states_ha[:, 0], perm).to(dev)
        xi = default_xi(len(states_ha), dev)
        s = S0
        tot = torch.zeros(len(states_ha), device=dev)
        a_prev = acts[:, 0].to(dev)
        for t in range(H):
            act = acts[:, t].to(dev)
            s = bundle.step(s, act, xi, step_idx=t + 1)
            ref = FrameRef(torch.full((len(states_ha),), int(frame0[0]) + t,
                                      dtype=torch.long, device=dev))
            r = bundle.reward(s, act, a_prev, ref)
            tot = tot + r
            a_prev = act
        return tot

    res = []
    for i in range(0, len(rows), a.batch):
        chunk = rows[i:i + a.batch]
        n = len(chunk)
        p_st = torch.stack([r["st"] for r in chunk])
        p_ac = torch.stack([r["ac"] for r in chunk])
        b_st = torch.stack([r["b_st"] for r in chunk])
        b_ac = torch.stack([r["b_ac"] for r in chunk])
        f0 = torch.tensor([int(r["fr"][0]) for r in chunk], dtype=torch.long)
        sig_v = torch.tensor([r["sig"] for r in chunk])
        with torch.no_grad():
            tp = j_true(p_st, p_ac, f0)
            tb = j_true(b_st, b_ac, f0)
            wp = j_wm(p_st, p_ac, f0)
            wb = j_wm(b_st, b_ac, f0)
        res.append(torch.stack([sig_v.cpu(), (tp - tb).cpu(), (wp - wb).cpu()], -1))
    R = torch.cat(res)                                             # (N,3)
    sg, dt, dw = R[:, 0], R[:, 1], R[:, 2]

    def rep(mask, tag):
        n = int(mask.sum())
        if n < 8:
            return
        t, w = dt[mask], dw[mask]
        agree = float((torch.sign(t) == torch.sign(w)).float().mean())
        # 去掉"真实差异≈0"的段（没有判别空间，噪声只会稀释）
        nz = t.abs() > t.abs().median() * 0.25
        agree_nz = float((torch.sign(t[nz]) == torch.sign(w[nz])).float().mean()) if int(nz.sum()) > 8 else float("nan")
        corr = float(((t - t.mean()) * (w - w.mean())).mean() / (t.std() * w.std() + 1e-12))
        print(f"\n[σ={float(sg[mask][0]):.2f}] n={n}")
        print(f"  符号一致率       = {agree:.3f}   （去掉 |ΔJ_true| 过小的段后 {agree_nz:.3f}）")
        print(f"  ΔJ 相关          = {corr:+.3f}")
        print(f"  |ΔJ_true| 中位   = {float(t.abs().median()):.5f}   均 {float(t.mean()):+.5f}")
        print(f"  |ΔJ_WM|   中位   = {float(w.abs().median()):.5f}   均 {float(w.mean()):+.5f}")
        print(f"  WM/真实 幅度比   = {float(w.abs().median() / (t.abs().median() + 1e-12)):.2f}×")

    print("\n" + "=" * 78)
    print(f"判据：同一真实起点、两串只差扰动的动作，WM 与真物理谁判得更高（H={H}）")
    print("=" * 78)
    for s_val in sorted(set(sg.tolist())):
        rep(sg == s_val, f"σ={s_val}")
    rep(torch.ones_like(sg, dtype=torch.bool), "全部")
    print("\n判读：符号一致率 ≥0.65 且 ΔJ 相关 >0.2 ⇒ **梯度有信号**；"
          "≈0.5 且相关≈0 ⇒ 梯度是噪声（只能拿 WM 做排序，不能做梯度）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
