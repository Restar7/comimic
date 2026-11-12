#!/usr/bin/env python3
"""Gate B（可做的那一半）：**学生**的幅度扫描 + 一阶方向量 D_i。

为什么必须单独做：A.5 扫的是**教师方向** `unit(∂J/∂token)`，不是学生输出。
学生 cos 只有 0.41，方向更差 ⇒ 小幅度**可能**更安全，这一支之前从未测过。

输出：每个 α 的 `‖Δ‖`、动作偏差 RMS、`D_i = Σ_t <∇J, Δ_t>` 的正负比例、WM 的 ΔJ，
并把 `acts_plus/acts_minus` 存成 Isaac job（真实物理配对）。
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=str(ROOT / "runs/harness_02_teacher/dump_h20/teacher_*.npz"))
    ap.add_argument("--ckpt", default=str(ROOT / "runs/S1_handover/WBC_A24000_wmvel_cmd.pt"))
    ap.add_argument("--wm-ckpt", default=str(ROOT / "runs/a_prime_sonic_exc/wm_gain2.pt"))
    ap.add_argument("--student", default=str(ROOT / "runs/v5_sup_init/init_v5_supervised.pt"))
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.25, 0.5, 1.0])
    ap.add_argument("--out", default="runs/student_isaac")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    import numpy as np, torch, joint_contract as JC
    from diff_vla_wbc import teacher_probe as tp
    from diff_vla_wbc import delta_token_io_v5 as d05
    from diff_vla_wbc.isaac_joint import FrameRef, load_isaac_bundle
    from diff_vla_wbc.rollout import discounted_return
    from pretrain_v5_supervised import hist10_v5_from_930
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
    wd = torch.load(a.wm_ckpt, map_location="cpu", weights_only=False)
    b.wm.load_state_dict(wd.get("wm", wd), strict=True)
    print(f"[wm] {a.wm_ckpt}")
    tok = torch.as_tensor(D["token_chunk"][:N], dtype=torch.float32, device=dev)
    ctx = dict(state=torch.as_tensor(D["state"][:N], dtype=torch.float32, device=dev),
               references=[FrameRef(torch.as_tensor(D["frames"][t, :N], dtype=torch.long, device=dev))
                           for t in range(K)],
               xi=torch.as_tensor(D["xi"][:N], dtype=torch.float32, device=dev),
               prev_action=torch.as_tensor(D["prev_action"][:N], dtype=torch.float32, device=dev),
               history=torch.as_tensor(D["history"][:, :N], dtype=torch.float32, device=dev),
               obs_hist=None)
    # ---- 学生 ----
    _ck = torch.load(a.student, map_location="cpu", weights_only=False)
    _sd = _ck.get("delta_token_v5_state") or _ck.get("delta_token_state") or _ck
    _meta = {**(_ck.get("delta_token_v5_spec") or {}),
             **(_ck.get("delta_token_v5_meta") or _ck.get("delta_token_meta") or {})}
    net, spec = d05.load_temporal_actor_v5(_sd, _meta, device=dev)
    net.eval()
    st_t = torch.as_tensor(D["state"][:N], dtype=torch.float32)
    fr_t = torch.as_tensor(D["frames"][0, :N], dtype=torch.long)
    import diag_bptt_learnability as d03
    s64 = d03.state64_from(b, st_t, fr_t, dev).detach().float()
    hist = torch.as_tensor(hist10_v5_from_930(D["obs_hist"][:N]), dtype=torch.float32, device=dev)
    mask2d = torch.cat([torch.ones(K, device=dev), torch.zeros(T - K, device=dev)])[None, :]
    mask = mask2d[..., None]                      # 广播用 (1,T,1)
    with torch.no_grad():
        d_st = net(tok, s64, hist, mask2d.expand(N, T)).detach()          # (N,T,64)
        r0 = tp._rollout(b, tok, horizon=K, **ctx); J0 = discounted_return(r0, 0.99)
        a0 = r0.actions.detach()
    g = tp.tokenwise_reward_grad(b, tok, horizon=K, gamma=0.99, reward_steps=0, **ctx).g
    gu = (g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12))[:, None, :].expand(-1, T, -1)
    print(f"[student] ‖Δ‖ p50={float(d_st.norm(dim=-1).median()):.4f} "
          f"cos(Δ_student, unit g) p50="
          f"{float((d_st*gu).sum(-1).div(d_st.norm(dim=-1).clamp_min(1e-12)).median()):+.3f} "
          f"J0={float(J0.mean()):.4f}")
    a.out = Path(a.out); a.out.mkdir(parents=True, exist_ok=True)
    rows, jobs = [], {}
    for al in a.alphas:
        d = (float(al) * d_st) * mask
        with torch.no_grad():
            r = tp._rollout(b, tok + d, horizon=K, **ctx)
            J = discounted_return(r, 0.99); an = r.actions.detach()
            Ddot = (float(al) * d_st * g[:, None, :]).sum(-1)     # ⟨∇J, Δ⟩ 逐帧 → 窗口
            Dwin = Ddot[:, :K].sum(-1)
            dev_a = an - a0
            rows.append(dict(alpha=float(al), tok_norm=float(d.norm(dim=-1).mean()),
                             act_rms=float(dev_a.pow(2).mean().sqrt()),
                             dJ_WM=float((J - J0).mean()),
                             pos=float((J > J0).float().mean()),
                             D_pos=float((Dwin > 0).float().mean()),
                             D_mean=float(Dwin.mean())))
            jobs[float(al)] = (d.detach(), r.actions.detach())
        print(f"[α={al:g}] ‖Δ‖={rows[-1]['tok_norm']:.4f} act_rms={rows[-1]['act_rms']:.3f} "
              f"dJ_WM={rows[-1]['dJ_WM']:+.4f} 变好={rows[-1]['pos']:.2f} "
              f"D>0={rows[-1]['D_pos']:.2f} D_mean={rows[-1]['D_mean']:+.4f}", flush=True)
    p_ha = torch.as_tensor(JC.permutation("standard", "ha_v31"), device=dev)
    def to_ha(s):
        o = s.clone(); o[:, 7:36] = s[:, 7:36][:, p_ha]; o[:, 42:71] = s[:, 42:71][:, p_ha]
        return o
    common = dict(states_ha=to_ha(torch.as_tensor(D["state"][:N], dtype=torch.float32, device=dev)).cpu(),
                  prev=torch.as_tensor(D["prev_action"][:N], dtype=torch.float32).cpu(),
                  frames=torch.as_tensor(D["frames"][:K, :N]).long().T, horizon=K,
                  ckpt=str(a.ckpt), joint_names=list(b.kin.joint_names), dumps=["dump_h20"])
    for al, (d, acts) in jobs.items():
        j = dict(common)
        j.update(acts_plus=acts.permute(1, 0, 2).cpu(), acts_minus=a0.permute(1, 0, 2).cpu(),
                 a0=a0.cpu(), J_WM_plus=(J0).cpu(), J_WM_minus=J0.cpu(), teacher=f"student_a{al:g}")
        p = a.out / f"job_student_a{al:g}.pt"; torch.save(j, p)
    (a.out / "rows.json").write_text(json.dumps(rows, indent=1))
    print(f"[jobs] {a.out}/job_student_a*.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
