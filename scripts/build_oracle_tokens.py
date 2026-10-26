#!/usr/bin/env python3
"""造 **oracle token**：在可微 WM 里直接优化 token，使"执行回报"最大化（不需要仿真器）。

用途：闭环回放实验的第三臂。三臂对比（见 `docs/HANDOVER_ORACLE_TOKEN_CLOSEDLOOP_4GPU_20261026.md`）：

    GT     : 录制 token（encoder_latent / hold builder）
    VLA    : 官方 checkpoint-8000 的输出（decode 后）
    ORACLE : 本脚本产出的"奖励最优 token"

判读：如果 ORACLE 在真闭环里也比 GT/VLA 好 → 想象目标可迁移 → 联合微调值得投；
      如果 ORACLE ≈ GT 或更差 → 想象里的 30% 增益搬不到真机，别烧训练卡。

做法（**逐 chunk**，与闭环的重取间隔一致）：
    for chunk_start in 0..T step CHUNK:
        起点 = 录制执行状态[chunk_start] + 扰动
        burn-in K=10（GT hold token，仿真 anchor）          # 和训练口径一致
        T_chunk = 可学习 (CHUNK,64)，初值 = GT hold token
        对 −回报 做 Adam（60 步），clamp 到 codec 原始范围，锚定 GT chunk 的软惩罚 λ=0.1
        写回 out[chunk_start:chunk_start+CHUNK]

输出：runs/oracle_tokens/<clip>.npy  (T,64)，以及同名 `.json` 审计。

    python3 scripts/build_oracle_tokens.py --clips HOI_double_desk__tw__ep000000 --device cuda:0
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("SUPERMIMIC_HA_ALIGN", "1")
os.environ.setdefault(
    "SUPERMIMIC_ROBOT_USD",
    str(ROOT / "trackerlab-supermimic/data/assetslib/unitree/unitree_g1/usd/g1_29dof_rev_1_0.usd"),
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True, help="clip 名（npz stem）或 npz 路径")
    ap.add_argument("--npz-dir", type=Path, default=ROOT / "data/npz/ha_tokens")
    ap.add_argument("--wbc-ckpt", type=Path, default=ROOT / "runs/S1_handover/WBC_A20000_wmvel_cmd.pt")
    ap.add_argument("--motion-packed", type=Path, default=ROOT / "runs/isaac_motion/ha_cmd/packed.pt")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs/oracle_tokens")
    ap.add_argument("--chunk", type=int, default=40)
    ap.add_argument("--opt-steps", type=int, default=60)
    ap.add_argument("--opt-lr", type=float, default=0.05)
    ap.add_argument("--anchor-lambda", type=float, default=0.1)
    ap.add_argument("--perturb", type=float, default=3.0, help="burn-in 扰动倍数（1x = 5cm/0.1rad）")
    ap.add_argument("--burnin", type=int, default=10)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from diff_vla_wbc.action_codec import ActionCodec
    from diff_vla_wbc.burnin import sample_burnin
    from diff_vla_wbc.data import RecordedTokenSource, resolve_clip_files
    from diff_vla_wbc.g1_tasks import load_gr00t_processor
    from diff_vla_wbc.isaac_joint import canonical_to_flat, load_isaac_bundle
    from diff_vla_wbc.rollout import discounted_return, imagine_with_tokens
    from diff_vla_wbc.sonic_frontend import LibraryTokenBuilder

    dev = torch.device(a.device)
    torch.manual_seed(int(a.seed))
    files = sorted(a.npz_dir.glob("*.npz"))
    bundle = load_isaac_bundle(a.wbc_ckpt, device=dev, npz=files,
                               packed_motion=a.motion_packed, token_dim=64)
    lib = bundle.library
    online = LibraryTokenBuilder(bundle.motion, device=dev, window="hold", library_path=a.motion_packed)
    name_to_idx = {n: i for i, n in enumerate(lib.clips and [c.name for c in lib.clips] or [])}
    names = [c.name for c in lib.clips]
    clipped = resolve_clip_files(a.npz_dir, names)      # 名字→npz（同名失败时按序号）
    idx_of = {p.stem: i for i, p in enumerate(clipped)}
    frame_src = RecordedTokenSource(a.npz_dir, lib, token_horizon=1, token_dim=64,
                                   device=dev, token_space="label")
    processor = load_gr00t_processor(ROOT / "runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000")
    codec = ActionCodec(processor, device=dev)
    lo = codec.raw_min[:64]; hi = codec.raw_max[:64]
    lo_t = torch.tensor(lo, device=dev, dtype=torch.float32)
    hi_t = torch.tensor(hi, device=dev, dtype=torch.float32)

    def recorded_state_of(ci: int, clip_path: Path) -> torch.Tensor:
        """整段真执行状态 (T,71)：root 用库（机器人轨迹代理），q/qd 用 npz（真执行）。"""
        st = lib.clip_start[ci].item(); ln = lib.clip_len[ci].item()
        rows = lib.ref[st:st + ln].float()
        with np.load(clip_path) as z:
            q = np.asarray(z["joint_pos"], np.float32)[:ln]
            qd = np.asarray(z["joint_vel"], np.float32)[:ln]
        T = min(len(q), ln)
        rows = rows[:T].clone()
        rows[:, 13:13 + 29] = torch.from_numpy(q[:T]).to(rows.device)
        rows[:, 13 + 29:13 + 58] = torch.from_numpy(qd[:T]).to(rows.device)
        return canonical_to_flat(rows), st

    a.out_dir.mkdir(parents=True, exist_ok=True)
    for clip in a.clips:
        p = Path(clip)
        if not p.exists():
            p = a.npz_dir / f"{Path(clip).stem}.npz"
        if not p.exists():
            print(f"[oracle][skip] {clip}: npz 找不到"); continue
        ci = idx_of.get(p.stem)
        if ci is None:
            print(f"[oracle][skip] {p.stem}: 不在库里"); continue
        rec, st = recorded_state_of(ci, p)
        T = int(rec.shape[0])
        st_frame = int(lib.clip_start[ci].item())
        out = torch.zeros(T, 64)
        audit = []
        t0 = time.time()
        for c0 in range(0, T, a.chunk):
            c1 = min(c0 + a.chunk, T)
            n = c1 - c0
            frames = torch.full((1,), st_frame + c0, device=dev, dtype=torch.long)
            bi = sample_burnin(bundle, lib, frames, port_tokens=lambda f: frame_src({"frames": f})[:, 0],
                               token_fn=online.__call__, horizon=n, k=a.burnin, perturb=True,
                               xy=0.05 * a.perturb, yaw=0.10 * a.perturb,
                               dq=0.05 * a.perturb, dqd=0.30 * a.perturb)
            idx = frames[:, None] + torch.arange(n, device=dev)[None, :]
            gt = online.chunk(idx, robot_quat=bi.state[:, 3:7])[0]        # (n,64) 参考 token
            Tk = torch.nn.Parameter(gt.clone())
            opt = torch.optim.Adam([Tk], lr=a.opt_lr)
            best = None
            for it in range(a.opt_steps):
                opt.zero_grad(set_to_none=True)
                ro = imagine_with_tokens(bundle, tokens=Tk[None], state=bi.state, obs_hist=None,
                                         history=bi.history, xi=bi.xi, references=bi.references,
                                         prev_action=bi.prev_action, horizon=n, obs_noise=0.0)
                ret = discounted_return(ro, a.gamma)
                loss = -ret.mean() + a.anchor_lambda * ((Tk - gt) ** 2).mean()
                loss.backward(); opt.step()
                with torch.no_grad():
                    Tk.clamp_(min=lo_t, max=hi_t)
                    if best is None or float(ret.mean()) > best[0]:
                        best = (float(ret.mean()), Tk.detach().clone())
            with torch.no_grad():
                ro0 = imagine_with_tokens(bundle, tokens=gt[None], state=bi.state, obs_hist=None,
                                          history=bi.history, xi=bi.xi, references=bi.references,
                                          prev_action=bi.prev_action, horizon=n, obs_noise=0.0)
                r0 = float(discounted_return(ro0, a.gamma).mean())
                out[c0:c1] = best[1].cpu()
            audit.append({"chunk": c0, "n": n, "ret_gt": r0, "ret_oracle": best[0],
                          "drift": float((best[1] - gt).abs().mean())})
            if (c0 // a.chunk) % 10 == 0:
                print(f"  [{p.stem}] chunk {c0}/{T} ret {r0:.3f}→{best[0]:.3f} "
                      f"drift {audit[-1]['drift']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        np.save(a.out_dir / f"{p.stem}.npy", out.numpy().astype(np.float32))
        r0 = float(np.mean([x["ret_gt"] for x in audit])); ro = float(np.mean([x["ret_oracle"] for x in audit]))
        (a.out_dir / f"{p.stem}.json").write_text(json.dumps(
            {"clip": p.stem, "frames": T, "chunk": a.chunk, "opt_steps": a.opt_steps,
             "perturb_scale": a.perturb, "ret_gt_mean": r0, "ret_oracle_mean": ro,
             "gain_rel": (ro - r0) / max(abs(r0), 1e-9), "chunks": audit}, indent=1))
        print(f"[oracle] {p.stem}: {T} 帧 -> {a.out_dir / (p.stem + '.npy')} "
              f"| 想象回报 {r0:.3f}→{ro:.3f} ({(ro-r0)/max(abs(r0),1e-9):+.1%})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
