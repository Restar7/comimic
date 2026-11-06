#!/usr/bin/env python3
"""**方向落地探针（尺子 A）**：FT 的动作位移 Δa 与 WM 奖励下降方向 ĝ 是否同向？

为什么需要它（2026-11-05 深夜）
--------------------------------
`scripts/probe_ft_landing.py` 已经证明两件事：FT 真的把动作挪了（v10 = 2.65×目标位移），
而且三个独立训练臂的 Δa 互相 **cos 0.98–0.99**（方向被一致地实现，不是噪声）。
但那只说明"它们朝同一个方向挪"，**没说明那个方向是不是 WM 觉得该去的方向**。

这里补上最后一块：在**同一条 dump 状态**上，用训练用的同一个 bundle（WBC+WM+奖励口径）
算 `ĝ = argmax (−J)` 对动作的梯度，再与 `Δa` 求余弦。

关键工程细节（都做成自检，错了会自己报出来）
--------------------------------------------
1. **状态装配**：dump 里是 `root_pos/root_quat_wxyz/q/root_lin_vel_{b,w}/qd`，
   而 WM 吃的是 71 维 `[p, quat, q29, v3, w3, qd29]`，其中 q 的序和 v/w 的坐标系
   都没有书面保证。做法：对 4 种组合（q 原序 / 两种置换 × v/w 体/世界）各做**单步预测**，
   与 dump 的下一步比，**自动选误差最小的那种**，并把误差打出来。
   （误差大 = 装配错了，此时余弦不可信 ⇒ 脚本会拒绝下结论。）
2. **动作口径**：dump 的 `applied` 与 decoder 的输出是同一个部署口径，直接喂 WM；
   奖励里的 `action_rate` 按 `dir_accept_build_job.py` 已验证过的换算写。

用法::

    python3 scripts/probe_ft_direction.py --dump runs/closedloop_r3/base_s0 \\
        --delta runs/_landing_delta.npz --arms v10 v11 v13
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402


def quat_to_mat(q):
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-9)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.empty(q.shape[:-1] + (3, 3), q.dtype)
    m[..., 0, 0] = 1 - 2 * (y * y + z * z); m[..., 0, 1] = 2 * (x * y - z * w); m[..., 0, 2] = 2 * (x * z + y * w)
    m[..., 1, 0] = 2 * (x * y + z * w); m[..., 1, 1] = 1 - 2 * (x * x + z * z); m[..., 1, 2] = 2 * (y * z - x * w)
    m[..., 2, 0] = 2 * (x * z - y * w); m[..., 2, 1] = 2 * (y * z + x * w); m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="runs/closedloop_r3/base_s0")
    ap.add_argument("--delta", default="runs/_landing_delta.npz")
    ap.add_argument("--arms", nargs="+", default=None, help="默认取 npz 里除 base 之外的全部")
    ap.add_argument("--wbc", default="runs/S1_handover/WBC_A24000_wmvel_cmd.pt")
    ap.add_argument("--wm", default="runs/a_prime_sonic_exc/wm_gain2.pt")
    ap.add_argument("--cfg", default="runs/joint_v13/args.json", help="奖励口径（reward_cfg）")
    ap.add_argument("--npz-glob", default="data/npz/ha_tokens/*.npz")
    ap.add_argument("--packed", default="runs/isaac_motion/ha_cmd_exec/packed.pt")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    import glob  # noqa: PLC0415

    import torch  # noqa: PLC0415

    import joint_contract as JC  # noqa: PLC0415
    from diff_vla_wbc.isaac_joint import FrameRef, load_isaac_bundle  # noqa: PLC0415
    from diff_vla_wbc.reward_cfg import apply_reward_cfg  # noqa: PLC0415
    from diff_vla_wbc.sonic_mjlab import action_scale_sonic, ha_to_std, std_to_ha  # noqa: PLC0415

    dev = torch.device(a.device)
    steps = [json.loads(l) for l in (ROOT / a.dump / "dump/steps.jsonl").open() if l.strip()]
    Z = np.load(ROOT / a.delta)
    req_idx = Z["req_idx"]
    arms = a.arms or [k[2:] for k in Z.files if k.startswith("a_") and k != "a_base"]

    # ---- 1) bundle（与训练同一个 WBC/WM/奖励口径）----
    b = load_isaac_bundle(str(ROOT / a.wbc), device=dev, npz=sorted(glob.glob(str(ROOT / a.npz_glob))),
                          packed_motion=str(ROOT / a.packed), obs_mode=None, hist=None, token_dim=64)
    apply_reward_cfg(b, json.load(open(ROOT / a.cfg)))
    ck = torch.load(str(ROOT / a.wm), map_location="cpu", weights_only=False)
    b.wm.load_state_dict(ck["wm"], strict=True)
    b.wm.requires_grad_(False)
    b.wm.eval()
    spec = torch.load(str(ROOT / a.wbc), map_location="cpu", weights_only=False)["robot_spec"]
    ha_scale = torch.tensor(np.asarray(spec["action_scale"], dtype="float32"), device=dev)
    ss = action_scale_sonic().numpy().astype("float32")
    ss_t = torch.tensor(ss, device=dev)
    ha = torch.tensor(ha_to_std(), dtype=torch.long, device=dev)
    sh = torch.tensor(std_to_ha(), dtype=torch.long, device=dev)
    print(f"[dir] bundle OK；奖励口径={a.cfg}；WM={Path(a.wm).name}", flush=True)

    def assemble(row, q_mode: str, vel_mode: str) -> torch.Tensor:
        p = np.asarray(row["root_pos"], np.float32)
        quat = np.asarray(row["root_quat_wxyz"], np.float32)
        q = np.asarray(row["q"], np.float32)
        qd = np.asarray(row["qd"], np.float32)
        v = np.asarray(row["root_lin_vel_b"], np.float32)
        w = np.asarray(row["root_ang_vel_b"], np.float32)
        if q_mode == "ha_to_std":
            q, qd = q[ha_to_std()], qd[ha_to_std()]
        elif q_mode == "std_to_ha":
            q, qd = q[std_to_ha()], qd[std_to_ha()]
        if vel_mode == "world":
            R = quat_to_mat(quat)
            v, w = R @ v, R @ w
        return torch.as_tensor(np.concatenate([p, quat, q, v, w, qd]), dtype=torch.float32,
                               device=dev).reshape(1, -1)

    # ---- 2) 状态装配自检：单步预测 vs dump 下一步，四种组合选最优 ----
    best = None
    t0 = int(req_idx[0]) * 20
    for qm in ("asis", "ha_to_std", "std_to_ha"):
        for vm in ("body", "world"):
            s0 = assemble(steps[t0], qm, vm)
            av = torch.as_tensor(np.asarray(steps[t0]["applied"], np.float32), device=dev).reshape(1, -1)
            xi = b.default_xi(1, dev)
            with torch.no_grad():
                s1 = b.step(s0, av, xi, step_idx=1)
            nxt = steps[t0 + 1]
            ep = float(torch.linalg.norm(s1[0, :3] - torch.as_tensor(np.asarray(nxt["root_pos"], np.float32), device=dev)))
            eq = float(torch.linalg.norm(s1[0, 7:36] - torch.as_tensor(np.asarray(nxt["q"], np.float32), device=dev)))
            score = ep + 0.05 * eq
            print(f"  [自检] q={qm:<10} v/w={vm:<5} 单步位置误差 {ep*100:6.2f} cm 关节误差 {eq:6.3f} rad"
                  f"  综合 {score:.4f}", flush=True)
            if best is None or score < best[0]:
                best = (score, qm, vm, ep, eq)
    _, q_mode, vel_mode, ep, eq = best
    print(f"[dir] 采用 q={q_mode} v/w={vel_mode}（单步位置 {ep*100:.2f} cm / 关节 {eq:.3f} rad）", flush=True)
    if ep > 0.05 or eq > 0.5:
        print("[dir] ⚠ 单步自检误差偏大 ⇒ 装配可能不对，下面的余弦**只能当参考**", flush=True)

    # ---- 3) 逐请求算 ĝ，并与各臂 Δa 求余弦 ----
    rows = []
    for ci in req_idx:
        ci = int(ci)
        t = ci * 20
        s0 = assemble(steps[t], q_mode, vel_mode)
        A = torch.as_tensor(np.stack([np.asarray(steps[t + k]["applied"], np.float32)
                                      for k in range(a.horizon)]), device=dev).unsqueeze(0)  # (1,H,29)
        fr = torch.as_tensor([[int(steps[t + k]["frame"]) for k in range(a.horizon)]], dtype=torch.long,
                             device=dev)
        xi = b.default_xi(1, dev)
        A = A.clone().requires_grad_(True)
        s = s0
        tot = torch.zeros((), device=dev)
        for k in range(a.horizon):
            s = b.step(s, A[:, k], xi, step_idx=k + 1)
            a_ha = (A[:, k].index_select(-1, ha) * ss_t[ha]) / ha_scale
            tot = tot + b.reward(s, a_ha, a_ha, FrameRef(fr[:, k]))["total"]
        g, = torch.autograd.grad(tot, A)
        ghat = g.mean(1)[0]                                    # ∂J/∂(20 帧平均动作)
        ghat = ghat / (ghat.pow(2).mean().sqrt() + 1e-9)
        da = {}
        for nm in arms:
            key = f"a_{nm}"
            if key not in Z:
                continue
            # 同一请求、所有噪声的平均
            m = Z[key].reshape(len(req_idx), -1, Z[key].shape[-1])[list(req_idx).index(ci)].mean(0)
            da[nm] = m
        rows.append((ci, ghat.cpu().numpy(), da))
        if (list(req_idx).index(ci) + 1) % 6 == 0:
            print(f"[dir] {list(req_idx).index(ci) + 1}/{len(req_idx)} 请求完成", flush=True)

    print("\n[dir] === cos(Δa, ĝ_WM)，同一条 dump 状态 ===")
    print(f"{'arm':<8}{'cos 均值':>10}{'中位':>8}{'>0 占比':>9}{'样本':>6}")
    for nm in arms:
        cs = [float(r[2][nm] @ r[1] / (np.linalg.norm(r[2][nm]) * np.linalg.norm(r[1]) + 1e-12))
              for r in rows if nm in r[2]]
        cs = np.asarray(cs)
        print(f"{nm:<8}{cs.mean():>10.3f}{np.median(cs):>8.3f}{float((cs > 0).mean()):>9.2f}{len(cs):>6}")
    # 置换对照：把 ĝ 关节顺序反过来，余弦应显著下降（否则说明余弦对顺序不敏感 = 没用）
    print("\n[dir] 对照（把 ĝ 的关节序倒过来，余弦应明显下降）:")
    for nm in arms[:1]:
        cs = [float(r[2][nm] @ r[1][::-1] / (np.linalg.norm(r[2][nm]) * np.linalg.norm(r[1]) + 1e-12))
              for r in rows if nm in r[2]]
        print(f"  {nm}: cos(Δa, reverse(ĝ)) = {np.mean(cs):+.3f}")
    print("\n[dir] 判读：≥0.5 ⇒ 压力确实落在 WM 想去的方向（闭环看不出来只是信噪比问题）；"
          "\n       ≈0 ⇒ '方向没落实'（v10/v11/v13 同病）；<0 ⇒ 压力方向相反（被锚项/目标写错了）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
