#!/usr/bin/env python3
"""用**真转移**离线拟合 token critic，并在**留出 episode** 上报 ``v_ev``。

为什么单独一个脚本（而不是在训练里边训边看）：
    训练里的 ``critic/v_ev`` 是在**当批 minibatch** 上算的（``token_critic.critic_fit_supervised``），
    既不是留出集、batch 也只有几十条 —— 那个数**不能**当"critic 行不行"的判据。
    这里按 episode 切留出集，报 ``v_ev = 1 − MSE/Var(target)``，才是 O12 的门。

数据：
    ``--transitions runs/wm_real/real_transitions_v2.pt``（含 states/ref_rows/ret/clip_ids；
    有 ``extra`` 时把 B 线漂移帧一起用 —— **off-reference 状态才是 critic 真正要用的地方**）。

去趋势（``--center``）：
    * ``none``  ：原始 MC return（会被"剩余步数"主导）
    * ``mean``  ：每 episode 减均值（训练侧 ``critic_center`` 的默认口径）
    * ``linear``：每 episode 减时间的线性拟合（**推荐**：去掉"剩余步数"，留下"状态相关的好坏"）
    注意 bootstrap 用 ``V(s_H) − V(ref_H)`` 的相对形式，同一帧，所以去趋势不影响它的语义。

用法：
    PYTHONPATH=. python3 scripts/fit_critic_real.py \
        --transitions runs/wm_real/real_transitions_v2b.pt \
        --out runs/wm_real/critic_real_v1.pt --center linear --epochs 12
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _detrend(ret: torch.Tensor, clip: torch.Tensor, mode: str) -> torch.Tensor:
    out = ret.clone().float()
    for c in clip.unique():
        idx = (clip == c).nonzero().flatten()
        if idx.numel() < 2:
            continue
        seg = out[idx]
        if mode == "mean":
            out[idx] = seg - seg.mean()
        elif mode == "linear":
            t = torch.arange(idx.numel(), dtype=torch.float32)
            tm = t - t.mean()
            slope = (tm * (seg - seg.mean())).sum() / (tm * tm).sum().clamp_min(1e-9)
            out[idx] = seg - (seg.mean() + slope * tm)
    return out


def _v_ev(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    mse = float(((pred - tgt) ** 2).mean())
    var = float(((tgt - tgt.mean()) ** 2).mean())
    return 1.0 - mse / max(var, 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transitions", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/wm_real/critic_real_v1.pt")
    ap.add_argument("--center", choices=("none", "mean", "linear"), default="linear")
    ap.add_argument("--val-every", type=int, default=5, help="每 N 个 episode 留 1 个做验证")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--use-diff", action="store_true", default=False,
                    help="把 s−ref 显式喂进 critic（价值主要由偏差决定，强烈建议开）")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=20261028)
    # ---- critic 目标 (2026-10-30) ----
    # mc  : 整集 return-to-go（无偏、对"漂移状态"对比度最强）  ← 推荐
    # gae : 真轨迹上的 TD(λ) 回归目标 target=V+A（低方差，但依赖 V 自身，自指）
    ap.add_argument("--critic-target", choices=("mc", "gae"), default="mc")
    ap.add_argument("--lam", type=float, default=0.95, help="仅 --critic-target gae 用")
    ap.add_argument("--pass1-epochs", type=int, default=4,
                    help="gae 模式：先用 MC 目标训几轮拿到一个可用 V，再换成 GAE 目标")
    ap.add_argument("--targets-out", type=Path, default=None,
                    help="把最终目标存成小文件 {ret,kind,...}，训练时用 --critic-ret 覆盖")
    a = ap.parse_args()

    from diff_vla_wbc.isaac_joint import canonical_to_flat, default_xi  # noqa: PLC0415
    from diff_vla_wbc.token_critic import TokenCritic  # noqa: PLC0415

    dev = torch.device(a.device)
    torch.manual_seed(int(a.seed))
    d = torch.load(str(a.transitions), map_location="cpu", weights_only=False)

    def _take(blk, tag: str):
        need = ("states", "ref_rows", "ret", "clip_ids")
        miss = [k for k in need if k not in blk]
        if miss:
            raise SystemExit(f"[critic] {tag} 缺 {miss}")
        ret = blk["ret"].float()
        clip = blk["clip_ids"].long()
        # extra 的 clip id 从 n_ref 起，为了和录制侧分开去趋势，整体平移成连续 id
        clip = clip - clip.min()
        return blk["states"].float(), blk["ref_rows"].float(), ret, clip, blk.get("frames")

    parts = []
    S, R, RET, CLIP, _ = _take(d, "recorded")
    parts.append(("recorded", S, R, RET, CLIP))
    if "extra" in d:
        Se, Re, ret_e, clip_e, _ = _take(d["extra"], "extra")
        parts.append(("extra", Se, Re, ret_e, clip_e))
    print("[critic] 数据：" + " / ".join(f"{n}={x.shape[0]}" for n, x, *_ in parts), flush=True)

    S = torch.cat([p[1] for p in parts], 0)
    R = torch.cat([p[2] for p in parts], 0)
    RET = torch.cat([p[3] for p in parts], 0)
    CLIP = torch.cat([p[4] for p in parts], 0)
    PART = torch.cat([torch.full((p[1].shape[0],), i, dtype=torch.long)
                      for i, p in enumerate(parts)], 0)
    PART_NAMES = [p[0] for p in parts]

    RET = _detrend(RET, CLIP, a.center)
    print(f"[critic] center={a.center} ret mean={float(RET.mean()):+.4f} std={float(RET.std()):.4f}",
          flush=True)

    # episode 留出（不随机打乱 episode，避免同一段轨迹泄漏）
    n_clip = int(CLIP.max()) + 1
    val_clip = torch.zeros(n_clip, dtype=torch.bool)
    val_clip[torch.arange(0, n_clip, max(1, int(a.val_every)))] = True
    is_val = val_clip[CLIP]
    # B（off-reference）单独按 episode 留出：它才是 critic 工作的地方。
    # 只有 1 集时无法留出 ⇒ 打印 WARN，并**不把它算进门**（避免"用训练集自评"）。
    n_b_ep = int(parts[-1][4].unique().numel()) if len(parts) > 1 else 0
    if len(parts) > 1:
        b_idx = PART == (len(parts) - 1)
        if n_b_ep >= 2:
            b_eps = parts[-1][4].unique()
            hold = set(int(x) for x in b_eps[::2].tolist())
            is_val[b_idx] = torch.tensor([int(c) in hold for c in parts[-1][4].tolist()])
        else:
            is_val[b_idx] = False
            print(f"[critic] WARN: B 只有 {n_b_ep} 集，无法留出；"
                  "下面的 v_ev 只代表录制侧（= off-reference 之外）", flush=True)

    def prep(s, r, ret, idx):
        s = s[idx].to(dev)
        r = canonical_to_flat(r[idx]).to(dev)
        return s, r, ret[idx].to(dev)

    tr = (~is_val).nonzero().flatten()
    va = is_val.nonzero().flatten()
    if va.numel() == 0:
        raise SystemExit("[critic] 没有验证 episode，调小 --val-every")
    if tr.numel() == 0:
        raise SystemExit("[critic] 没有训练 episode")
    S_tr, R_tr, Y_tr = prep(S, R, RET, tr)
    S_va, R_va, Y_va = prep(S, R, RET, va)
    xi_tr = default_xi(S_tr.shape[0], dev)
    xi_va = default_xi(S_va.shape[0], dev)
    print(f"[critic] train n={S_tr.shape[0]} ({n_clip - int(val_clip.sum())} clips) / "
          f"val n={S_va.shape[0]} ({int(val_clip.sum())} clips)", flush=True)

    critic = TokenCritic(hidden=int(a.hidden), depth=int(a.depth),
                         use_diff=bool(a.use_diff)).to(dev)
    opt = torch.optim.AdamW(critic.parameters(), lr=float(a.lr), weight_decay=float(a.weight_decay))

    with torch.no_grad():
        v0 = critic(S_va, R_va, xi_va)
        ev0 = _v_ev(v0, Y_va)
    print(f"[critic] 初始 val v_ev={ev0:+.4f}", flush=True)

    n = int(S_tr.shape[0])
    bs = min(int(a.batch), n)
    best = {"v_ev": -1e9, "epoch": -1, "state": None}
    _ep0 = [0]                      # 连续编号（gae 两阶段共用一个计数）
    for ep in range(1, int(a.pass1_epochs if a.critic_target == "gae" else a.epochs) + 1):
        perm = torch.randperm(n, device=dev)
        tot, cnt = 0.0, 0
        for lo in range(0, n, bs):
            idx = perm[lo:lo + bs]
            v = critic(S_tr[idx], R_tr[idx], xi_tr[idx])
            loss = ((v - Y_tr[idx]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * idx.numel()
            cnt += idx.numel()
        with torch.no_grad():
            v = critic(S_va, R_va, xi_va)
            ev = _v_ev(v, Y_va)
        _ep0[0] += 1
        print(f"  [ep {_ep0[0]:>2}] train_mse={tot / max(cnt, 1):.5f} val_v_ev={ev:+.4f}", flush=True)
        if ev > best["v_ev"]:
            best = {"v_ev": ev, "epoch": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in critic.state_dict().items()}}
    # ---------------- 阶段 2（仅 gae）：用阶段 1 的 V 算 TD(λ) 目标再训 ----------------
    if a.critic_target == "gae":
        critic.load_state_dict(best["state"])
        from diff_vla_wbc.token_critic import gae_targets_padded  # noqa: PLC0415

        with torch.no_grad():
            _V_all = torch.cat([
                critic(S[i:i + 200000].to(dev),
                       canonical_to_flat(R[i:i + 200000]).to(dev),
                       default_xi(min(200000, int(S.shape[0]) - i), dev)).cpu()
                for i in range(0, int(S.shape[0]), 200000)], 0).float()
        # 用**原始** reward 与 clip 分组算 A，再 detrend
        _rew = torch.cat([d["rewards"].float(),
                          (d["extra"]["rewards"].float() if "extra" in d else torch.zeros(0))], 0) \
            if "rewards" in d else None
        if _rew is None:
            print("[critic][warn] transitions 没有 rewards ⇒ 无法算 GAE，退回 MC", flush=True)
        else:
            _A = gae_targets_padded(_rew, _V_all, CLIP, gamma=float((d.get("meta") or {}).get("gamma") or 0.99),
                                    lam=float(a.lam))
            RET = _detrend(_A + _V_all, CLIP, a.center)
            print(f"[critic] stage2 gae lam={a.lam} ret mean={float(RET.mean()):+.4f} "
                  f"std={float(RET.std()):+.4f}", flush=True)
            Y_tr = RET[tr].to(dev)
            Y_va = RET[va].to(dev)
            best = {"v_ev": -1e9, "epoch": -1, "state": None}
            for ep in range(1, int(a.epochs) + 1):
                perm = torch.randperm(n, device=dev)
                tot, cnt = 0.0, 0
                for lo in range(0, n, bs):
                    idx = perm[lo:lo + bs]
                    v = critic(S_tr[idx], R_tr[idx], xi_tr[idx])
                    loss = ((v - Y_tr[idx]) ** 2).mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                    opt.step()
                    tot += float(loss.detach()) * idx.numel()
                    cnt += idx.numel()
                with torch.no_grad():
                    ev = _v_ev(critic(S_va, R_va, xi_va), Y_va)
                _ep0[0] += 1
                print(f"  [ep {_ep0[0]:>2}g] train_mse={tot / max(cnt, 1):.5f} val_v_ev={ev:+.4f}",
                      flush=True)
                if ev > best["v_ev"]:
                    best = {"v_ev": ev, "epoch": ep,
                            "state": {k: vv.detach().cpu().clone()
                                      for k, vv in critic.state_dict().items()}}
            if a.targets_out is not None:
                a.targets_out.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"ret": RET.cpu(), "kind": "gae", "lam": float(a.lam),
                            "gamma": float((d.get("meta") or {}).get("gamma") or 0.99),
                            "source": str(a.transitions)}, str(a.targets_out))
                print(f"[critic] targets(gae) → {a.targets_out}", flush=True)
        _ = canonical_to_flat  # 保持引用

    critic.load_state_dict(best["state"])
    with torch.no_grad():
        ev_va = _v_ev(critic(S_va, R_va, xi_va), Y_va)
    print(f"[critic] best epoch={best['epoch']} val v_ev={ev_va:+.4f} "
          f"(gate: > 0.30)", flush=True)
    # 分 part 报：录制留出 / B 留出（门只看 B 留出）
    b_ev = None
    with torch.no_grad():
        for pi, name in enumerate(PART_NAMES):
            m = (PART == pi) & is_val
            if not bool(m.any()):
                continue
            sm = S[m].to(dev); rm = canonical_to_flat(R[m]).to(dev); ym = RET[m].to(dev)
            xi_m = default_xi(sm.shape[0], dev)
            ev_p = _v_ev(critic(sm, rm, xi_m), ym)
            print(f"[critic]   {name:9s} 留出 n={int(m.sum()):>7d}  v_ev={ev_p:+.4f}", flush=True)
            if pi == len(PART_NAMES) - 1:
                b_ev = ev_p
    if b_ev is not None:
        print(f"[critic] 门（B 留出 v_ev > 0.30）：{'PASS' if b_ev > 0.30 else 'FAIL'}", flush=True)
    else:
        print("[critic] 门（B 留出 v_ev > 0.30）：无法判定（B 集数 < 2）", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "critic": best["state"],
        "meta": {
            "source": str(a.transitions), "center": a.center,
            "state_dim": 71, "ref_dim": 71, "xi_dim": 33,
            "hidden": int(a.hidden), "depth": int(a.depth), "use_diff": bool(a.use_diff),
            "v_ev_val": round(float(ev_va), 4), "v_ev_init": round(float(ev0), 4),
            "v_ev_b_holdout": (None if b_ev is None else round(float(b_ev), 4)),
            "n_train": int(S_tr.shape[0]), "n_val": int(S_va.shape[0]),
            "n_clips": int(n_clip), "gamma": (d.get("meta") or {}).get("gamma"),
            "epochs": int(a.epochs), "batch": int(a.batch), "lr": float(a.lr),
        },
    }, str(a.out))
    print(f"[critic] saved {a.out}", flush=True)
    return 0 if ev_va > 0.30 else 3


if __name__ == "__main__":
    raise SystemExit(main())
