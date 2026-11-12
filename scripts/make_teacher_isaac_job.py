#!/usr/bin/env python3
"""把「WM 内优化教师」的 Δ* 变成 Isaac 可执行的 job（±两臂）。

口径（抄 `probe_wm_gradient_truth.py plan`）：
  * `states_ha` (N,71)：dump 的 canonical(standard) 状态 → **ha_v31** 序（q/qd 重排）
  * `acts_*` (N,H,29)：**standard** 序动作；`q_target = default_q + a*scale`
  * `acts_plus` = 用 Δ*（iters=20）修正后的动作序列；`acts_minus` = 不修正（基线）
用法：python3 scripts/make_teacher_isaac_job.py --n 32 --out runs/teacher_isaac
"""
import argparse, glob, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=str(ROOT / "runs/harness_02_teacher/dump_h20/teacher_*.npz"))
    ap.add_argument("--ckpt", default=str(ROOT / "runs/S1_handover/WBC_A24000_wmvel_cmd.pt"))
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--token-cap", type=float, default=0.08)
    ap.add_argument("--eta", type=float, default=0.15)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/teacher_isaac")
    ap.add_argument("--scale", type=float, default=1.0, help="把 Δ* 缩放后再执行（查剂量-反应）")
    # ★ 必须显式指定"生成标签用的那个 WM"：bundle 自带的 wm 与标签口径的 wm 不是同一个
    ap.add_argument("--wm-ckpt", default=str(ROOT / "runs/a_prime_sonic_exc/wm_gain2.pt"))
    ap.add_argument("--teacher-mode", choices=("optimized", "grad"), default="optimized",
                    help="grad = 单次梯度教师（可配合 --alpha 做剂量扫描）")
    ap.add_argument("--alpha", type=float, default=1.0, help="只在 --teacher-mode grad 下生效")
    ap.add_argument("--reward-steps", type=int, default=0, help=">0 时用截断回报 J_h 的梯度（查短视界教师）")
    a = ap.parse_args()
    import numpy as np, torch, joint_contract as JC
    from diff_vla_wbc import teacher_probe as tp
    from diff_vla_wbc.isaac_joint import FrameRef, load_isaac_bundle
    from diff_vla_wbc.rollout import discounted_return
    dev = torch.device("cuda:0"); K = 20; T = 40
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
        _wd = torch.load(a.wm_ckpt, map_location="cpu", weights_only=False)
        b.wm.load_state_dict(_wd.get("wm", _wd), strict=True)
        print(f"[wm] 覆盖为 {a.wm_ckpt}（= 生成标签用的那个）")
    tok = torch.as_tensor(D["token_chunk"][:N], dtype=torch.float32, device=dev)
    st = torch.as_tensor(D["state"][:N], dtype=torch.float32, device=dev)
    xi = torch.as_tensor(D["xi"][:N], dtype=torch.float32, device=dev)
    prev = torch.as_tensor(D["prev_action"][:N], dtype=torch.float32, device=dev)
    hist = torch.as_tensor(D["history"][:, :N], dtype=torch.float32, device=dev)
    fr = torch.as_tensor(D["frames"][:K, :N], dtype=torch.long, device=dev)
    ctx = dict(state=st, references=[FrameRef(fr[t]) for t in range(K)], xi=xi,
               prev_action=prev, history=hist, obs_hist=None)
    mask = torch.cat([torch.ones(K, device=dev), torch.zeros(T - K, device=dev)])[None, :, None]

    def para(u): return float(a.token_cap) * torch.tanh(u) * mask
    def normcap(d):
        n = d.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return torch.where(n > float(a.token_cap), d * (float(a.token_cap) / n), d) * mask

    with torch.no_grad():
        r0 = tp._rollout(b, tok, horizon=K, **ctx); J0 = discounted_return(r0, 0.99)
        a0 = r0.actions.detach()
    g = tp.tokenwise_reward_grad(b, tok, horizon=K, gamma=0.99,
                                 reward_steps=int(a.reward_steps), **ctx).g
    gu = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    if a.teacher_mode == "grad":
        _dd = float(a.alpha) * float(a.token_cap) * gu[:, None, :].expand(-1, T, -1)
        d_old = _dd * mask
    else:
        d_old = normcap(float(a.eta) * gu[:, None, :].expand(-1, T, -1))
    if a.teacher_mode == "grad":
        with torch.no_grad():
            r_new = tp._rollout(b, tok + d_old, horizon=K, **ctx)
            J_new = discounted_return(r_new, 0.99)
        U = None
    U = torch.zeros(N, T, 64, device=dev, requires_grad=True) if U is None else U
    opt = torch.optim.Adam([U], lr=float(a.lr))
    for it in (range(int(a.iters)) if a.teacher_mode != "grad" else []):
        d = para(U)
        r = tp._rollout(b, tok + d, horizon=K, **ctx); J = discounted_return(r, 0.99)
        loss = -J.mean() + 1.0 * d.pow(2).mean()
        if not torch.isfinite(loss): break
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_([U], 1.0)
        if not torch.isfinite(gn): break
        opt.step()
        with torch.no_grad(): U.clamp_(-0.08, 0.08)
    with torch.no_grad():
        d_new = d_old.detach() if a.teacher_mode == "grad" else normcap(para(U)).detach()
        d_sc = d_new * float(a.scale)
        r_new = tp._rollout(b, tok + d_sc, horizon=K, **ctx); J_new = discounted_return(r_new, 0.99)
        r_old = tp._rollout(b, tok + d_old, horizon=K, **ctx); J_old = discounted_return(r_old, 0.99)
    # canonical(standard) → ha_v31（q/qd 两段重排）
    p_ha = torch.as_tensor(JC.permutation("standard", "ha_v31"), device=dev)
    def to_ha(s):
        out = s.clone()
        out[:, 7:36] = s[:, 7:36][:, p_ha]
        out[:, 42:71] = s[:, 42:71][:, p_ha]
        return out
    a.out.mkdir(parents=True, exist_ok=True)
    common = dict(states_ha=to_ha(st).cpu(), prev=prev.cpu(), frames=D["frames"][:K, :N],
                  horizon=K, ckpt=str(a.ckpt), joint_names=list(b.kin.joint_names))
    _tag_new = ("new" if float(a.scale) == 1.0 else f"new_s{a.scale:g}")
    if a.teacher_mode == "grad":
        _tag_new = f"grad_a{a.alpha:g}" + (f"_h{a.reward_steps}" if a.reward_steps else "")
    for tag, acts, Jw in ((_tag_new, r_new.actions.detach(), J_new), ("old", r_old.actions.detach(), J_old)):
        job = dict(common)
        job.update(a0=a0.cpu(), acts_plus=acts.permute(1, 0, 2).cpu(),
                   acts_minus=a0.permute(1, 0, 2).cpu(),
                   J_WM_plus=Jw.cpu(), J_WM_minus=J0.cpu(), teacher=tag)
        p = a.out / f"job_{tag}.pt"; torch.save(job, p)
        print(f"[job] {p}  N={N}  act_rms(new={float((acts-a0).pow(2).mean().sqrt()):.3f})  "
              f"J_WM +{float(Jw.mean()):.4f} vs 0 {float(J0.mean()):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
