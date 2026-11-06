#!/usr/bin/env python3
"""**唯一该用的判据**：WM 在"策略自己走出来的动作"上还准不准（离群保真度）。

为什么必须单独测（2026-11-04 用户口径）
--------------------------------------
`probe_wm_multistep` 只在**录制动作**上验保真（ratio 0.97）——那是"看着答案考试"。
但联合微调要反传的是**策略自己挑的动作**；round-3 的 `act/rmse` 到了 1.46，
说明它离录制动作很远。所以真正要问的是：

    拿闭环 dump 里**真实下发过的 applied 动作** + 当时的真实状态，喂给 WM，
    它预测 20 步后的状态 vs 真实到达的状态，差多少？

如果这里的误差远大于数据窗口上的 1.11 cm ⇒ WM 在策略的动作分布上是失真的，
"反传想象漂移"就不能指望；如果差不多 ⇒ WM 没问题，问题在别处（那就得看闭环里
到底是"姿态对上了但没走"还是别的）。

用法::

    python3 scripts/probe_wm_ood_dump.py \
        --dump runs/closedloop_r3/ftfinal/dump --ckpt runs/a_prime_sonic_leash/wm_ssm_s0_gain.pt
    # 可同时给 --dump-base 做对照
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

#: 闭环 dump 的字段 -> 扁平 71 维 state 的布局（与 A' 数据一致）
#: ``[p(0:3) quat(3:7) q(7:36) v(36:39) w(39:42) qd(42:71)]``
def states_from_dump(rows: list[dict]) -> torch.Tensor:
    """dump -> 扁平 71 维 state。

    ★ 2026-11-04 口径修复（差点又报错数）：`roll_windows.pt` 上实测
    ``corr(dp/dt, slot36:39) = +0.953`` ⇒ **训练数据里的线/角速度是世界系**；
    而闭环 dump 给的是 `root_lin_vel_b / root_ang_vel_b`（**机体系**）。
    直接塞进去等于喂错输入，会把 WM 的误差虚高好几倍。这里按四元数转到世界系。
    """
    p = np.array([r["root_pos"] for r in rows], np.float32)
    quat = np.array([r["root_quat_wxyz"] for r in rows], np.float32)
    q = np.array([r["q"] for r in rows], np.float32)
    vb = np.array([r["root_lin_vel_b"] for r in rows], np.float32)
    wb = np.array([r["root_ang_vel_b"] for r in rows], np.float32)
    qd = np.array([r["qd"] for r in rows], np.float32)
    w0, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    R = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w0 * z), 2 * (x * z + w0 * y)], -1),
        np.stack([2 * (x * y + w0 * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w0 * x)], -1),
        np.stack([2 * (x * z - w0 * y), 2 * (y * z + w0 * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2).astype(np.float32)
    v = np.einsum("nij,nj->ni", R, vb)
    wv = np.einsum("nij,nj->ni", R, wb)
    return torch.from_numpy(np.concatenate([p, quat, q, v, wv, qd], 1))


def actions_from_dump(rows: list[dict]) -> torch.Tensor:
    return torch.from_numpy(np.array([r["applied"] for r in rows], np.float32))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wbc-ckpt", type=Path, default=ROOT / "runs/S1_handover/WBC_A24000_wmvel_cmd.pt")
    ap.add_argument("--ckpt", type=Path,
                    default=ROOT / "runs/a_prime_sonic_leash/wm_ssm_s0_gain.pt")
    ap.add_argument("--dump", type=Path, required=True, help="被查的闭环 dump（策略自己的动作）")
    ap.add_argument("--dump-base", type=Path, default=None, help="对照：base 头的 dump")
    ap.add_argument("--data-windows", type=Path,
                    default=ROOT / "runs/a_prime_sonic_leash/roll_windows.pt",
                    help="录制动作窗口（给出时打印「数据上」的误差作参照）")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    from diff_vla_wbc.isaac_joint import load_isaac_bundle  # noqa: PLC0415

    dev = torch.device(a.device)
    npz = sorted((ROOT / "data/npz/ha_tokens").glob("*.npz"))
    bundle = load_isaac_bundle(a.wbc_ckpt, device=dev, npz=npz,
                               packed_motion=ROOT / "runs/isaac_motion/ha_cmd_exec/packed.pt",
                               obs_mode=None, hist=None, token_dim=64)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    bundle.wm.load_state_dict(ck.get("wm", ck), strict=True)
    bundle.wm.eval()
    print(f"[ood] WM = {a.ckpt}", flush=True)

    K = int(a.k)

    def load_dump(p: Path):
        rows = [json.loads(l) for l in (p / "steps.jsonl").open()]
        return states_from_dump(rows), actions_from_dump(rows), p.name

    def windows(S: torch.Tensor, A: torch.Tensor):
        starts = list(range(0, max(0, len(S) - K), int(a.stride)))[: a.n]
        st = torch.tensor(starts)
        return S[st], A[st[:, None] + torch.arange(K)], S[st + K]

    def wm_err(s0, acts, realK):
        xi = bundle.default_xi(s0.shape[0], dev)
        s = s0.clone()
        with torch.no_grad():
            for t in range(K):
                out = bundle.step(s, acts[:, t], xi)
                s = out[0] if isinstance(out, (tuple, list)) else out
        e_p = (s[:, 0:3] - realK[:, 0:3]).norm(dim=-1)
        e_q = (s[:, 7:36] - realK[:, 7:36]).abs().mean(-1)
        return e_p, e_q

    def report(tag, s0, acts, realK):
        s0, acts, realK = s0.to(dev), acts.to(dev), realK.to(dev)
        e_p, e_q = wm_err(s0, acts, realK)
        print(f"  {tag:<22} n={s0.shape[0]:4d}  20 步终点误差 p50={float(e_p.median())*100:.2f}cm "
              f"p90={float(e_p.quantile(0.9))*100:.2f}cm | 关节 p50={float(e_q.median()):.4f}rad",
              flush=True)
        return float(e_p.median()) * 100

    print(f"[ood] k={K} stride={a.stride}", flush=True)
    ref = None
    if a.data_windows.exists():
        w = torch.load(a.data_windows, map_location="cpu", weights_only=False)
        Sd, Ad = w["states"].float(), w["actions"].float()
        g = torch.Generator().manual_seed(0)
        sel = torch.randperm(int(Sd.shape[0]), generator=g)[: a.n]
        ref = report("录制动作（数据上）", Sd[sel, 0], Ad[sel], Sd[sel, K])
    _s, _a, _ = load_dump(a.dump)
    e_close = report(f"闭环 {a.dump.name}（策略）", *windows(_s, _a))
    if a.dump_base is not None:
        _s2, _a2, _ = load_dump(a.dump_base)
        report(f"闭环 {a.dump_base.name}（base）", *windows(_s2, _a2))
    if ref:
        print(f"\n[ood] 离群放大系数 = {e_close / max(ref, 1e-9):.2f}×（策略动作 / 录制动作 的 20 步预测误差）")
        print("[ood] 判读：≤2× ⇒ WM 在策略分布上仍可用；≥4× ⇒ 反传想象漂移不可信，必须扩数据/加约束")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
