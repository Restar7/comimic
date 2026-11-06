#!/usr/bin/env python3
"""**契约校验门**：把所有踩过的坑固化成可执行断言（2026-11-04）

一条命令跑完，任何一条 FAIL 就直接非零退出。每次改动采集/训练/评估口径前、
以及交出任何数据之前，都必须先跑这个。

```bash
python3 scripts/verify_contracts.py                        # 全部
python3 scripts/verify_contracts.py --skip-shards          # 只跑离线契约（秒级）
python3 scripts/verify_contracts.py --shards runs/a_prime_sonic
```

固化了下面这些**真实发生过**的 bug（每条都在注释里写了当时的症状）：

| 检查项 | 当初的症状 |
|---|---|
| `lib_layout`      | 把 `ref[:, 7:36]` 当关节角 ⇒ 报出 0.38 rad 的假跟踪误差（真值 0.14） |
| `decoder_offset`  | `decoder_q_offset` 被当成"原始 q"，其实 ≡ −SONIC_DEFAULT_POS |
| `decoder_obs`     | SONIC decoder 的 obs q/qd 块用了 standard 序 ⇒ 闭环 50 步内发散 |
| `action_target`   | 目标按 `idx_t` 落位 ⇒ 每步动作打到错关节 |
| `joint_order`     | `standard` 与 `SONIC` 是两个不同的 29 关节序，且**不是恒等置换** |
| `reset_order`     | 每 20 步把机器人瞬移到错位姿态 ⇒ 段内 8 cm 漂移 |
| `transitions`     | `states == next_states`（数据集里没有转移） |
| `no_freefall`     | 采集时摔倒但没人发现 |
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/HumanoidArena/isaaclab_twist2_g1"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

RES: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RES.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:22s} {detail}")
    return bool(ok)


def _npz() -> Path:
    cands = sorted(glob.glob(str(ROOT / "data/ha_raw/*/*/*.npz")))
    if not cands:
        raise SystemExit("[verify] 找不到录制 npz（data/ha_raw/*/*/*.npz）")
    return Path(cands[0])


def _consts():
    from action_provider.action_provider_sonic import (  # noqa: PLC0415
        SONIC_DEFAULT_POS, SONIC_ISAACLAB_JOINT_ORDER, G1_ACTION_SCALE_ISAACLAB)
    return (np.asarray(SONIC_DEFAULT_POS, dtype=np.float32),
            list(SONIC_ISAACLAB_JOINT_ORDER),
            np.asarray(G1_ACTION_SCALE_ISAACLAB, dtype=np.float32))


def _ckpt_joint_names() -> list[str]:
    ck = torch.load(ROOT / "runs/S1_handover/exp_WBC_exec_haonly_best.pt",
                    map_location="cpu", weights_only=False)
    return list(ck["robot_spec"]["joint_names"])


def check_lib_layout(packed: Path) -> None:
    """`packed['ref']` 是 canonical 布局，且 q 块 = 录制 `robot_qpos_before_decimation` **原序**。"""
    print("\n== lib_layout ==")
    pk = torch.load(packed, map_location="cpu", weights_only=False)
    ref = torch.as_tensor(pk["ref"]).float()
    z = np.load(_npz(), allow_pickle=True)
    qpos = np.asarray(z["robot_qpos_before_decimation"], dtype=np.float32)
    qvel = np.asarray(z["robot_qvel_before_decimation"], dtype=np.float32)
    n = min(ref.shape[0], qpos.shape[0])
    d_q = float((ref[:n, 13:42] - torch.from_numpy(qpos[:n])).abs().max())
    d_qd = float((ref[:n, 42:71] - torch.from_numpy(qvel[:n])).abs().max())
    # 容差：库里的 ref 是 **fp16 量化**后存的（文档实测 max|Δ|=0.0005），
    #   qd 量级更大 ⇒ 相对误差同量级。这里用「相对容差」而不是死数字。
    q_scale = float(torch.from_numpy(qpos[:n]).abs().max()) + 1e-6
    qd_scale = float(torch.from_numpy(qvel[:n]).abs().max()) + 1e-6
    check("ref q 在 [13:42]", d_q / q_scale < 1e-2,
          f"max|Δ|={d_q:.2e} / |q|max={q_scale:.2f} = {d_q/q_scale:.2e}（fp16 量化，应 <1e-2）")
    check("ref qd 在 [42:71]", d_qd / qd_scale < 1e-2,
          f"max|Δ|={d_qd:.2e} / |qd|max={qd_scale:.2f} = {d_qd/qd_scale:.2e}")
    # 反证：如果按扁平布局 [7:36] 取，应该对不上（这就是当初 0.38 rad 的假误差来源）
    d_bad = float((ref[:n, 7:36] - torch.from_numpy(qpos[:n])).abs().max())
    check("ref [7:36] 不是 q", d_bad > 0.1, f"max|Δ|={d_bad:.3f}（按扁平布局读会错）")


def check_decoder_offset() -> None:
    print("\n== decoder_offset ==")
    sdef, _, _ = _consts()
    f = ROOT / "runs/decoder_q_offset.pt"
    if not f.exists():
        check("decoder_q_offset.pt 存在", False, str(f))
        return
    off = torch.load(f, map_location="cpu", weights_only=False)["offset"].float().numpy()
    d = float(np.abs(off + sdef).max())
    check("offset ≡ −SONIC_DEFAULT_POS", d < 1e-5, f"max|Δ|={d:.2e}")


def check_decoder_obs() -> None:
    """录制 `decoder_obs` 的分块语义（这是 SONIC decoder 闭环的唯一权威口径）。"""
    print("\n== decoder_obs（对照录制） ==")
    sdef, _, gscale = _consts()
    z = np.load(_npz(), allow_pickle=True)
    obs = np.asarray(z["decoder_obs"], dtype=np.float32)
    lat = np.asarray(z["encoder_latent"], dtype=np.float32)
    av = np.asarray(z["decoder_ang_vel_hist"], dtype=np.float32)
    gr = np.asarray(z["decoder_gravity_dir_hist"], dtype=np.float32)
    la = np.asarray(z["decoder_last_action_hist"], dtype=np.float32)
    qpos = np.asarray(z["robot_qpos_before_decimation"], dtype=np.float32)
    check("obs 宽度 994", obs.shape[1] == 994, f"{obs.shape}")
    check("latent 在 [0:64]", float(np.abs(obs[:, 0:64] - lat).max()) == 0.0)
    check("ang_vel 在 [64:94]", float(np.abs(obs[:, 64:94] - av.reshape(len(av), -1)).max()) == 0.0)
    # ★ obs 是在**动作更新之前**拼的 ⇒ obs[t] 里的 last_action 历史 = 录制的 hist[t-1]；
    #   t=0 那一行整块是 0。这一点当初就是因为按"同期"比才误判成"对不上"。
    la_flat = la.reshape(len(la), -1)
    d_la = float(np.abs(obs[1:, 674:964] - la_flat[:-1]).max())
    d_la_same = float(np.abs(obs[:, 674:964] - la_flat).max())
    check("last_action 在 [674:964]（滞后一帧）", d_la == 0.0, f"max|Δ|={d_la:.2e}")
    check("last_action 不是同期（反证）", d_la_same > 0.5, f"max|Δ|={d_la_same:.2f}")
    check("gravity 在 [964:994]", float(np.abs(obs[:, 964:994] - gr.reshape(len(gr), -1)).max()) == 0.0)
    # q 块：每 10 帧里的**最后一帧** = qpos − SONIC_DEFAULT_POS
    last = obs[:, 94 + 9 * 29: 94 + 10 * 29]
    d = float(np.abs(last - (qpos - sdef[None, :])).max())
    check("q 块 = qpos − SONIC_DEFAULT", d < 1e-5, f"max|Δ|={d:.2e}")
    # 动作目标
    raw = np.asarray(z["decoder_raw_action"], dtype=np.float32)
    tgt = np.asarray(z["decoder_target_action"], dtype=np.float32)
    fba = np.asarray(z["final_body_action_29dof"], dtype=np.float32)
    pre = np.asarray(z["final_body_action_29dof_pre_delay"], dtype=np.float32)
    d2 = float(np.abs(raw * gscale[None, :] + sdef[None, :] - tgt).max())
    check("target = raw*scale+default", d2 < 1e-5, f"max|Δ|={d2:.2e}")
    check("final_action == target", float(np.abs(fba - tgt).max()) == 0.0)
    check("无部署侧延时后处理", float(np.abs(pre - tgt).max()) == 0.0)


def check_joint_order() -> None:
    print("\n== joint_order ==")
    _, SO, _ = _consts()
    jn = _ckpt_joint_names()
    A = [jn.index(nm) for nm in SO]          # sonic_idx -> std_idx
    B = [SO.index(nm) for nm in jn]          # std_idx -> sonic_idx
    check("两个序都覆盖 29 关节", len(set(A)) == 29 and len(set(B)) == 29)
    ident = list(range(29))
    check("SONIC 序 ≠ standard 序", A != ident, f"A[:6]={A[:6]}")
    check("A∘B == 恒等", [A[b] for b in B] == ident)


def check_shards(dirpath: Path) -> None:
    """A' 分片的硬约束：有转移、段首精确复位、无自由落体、逐行漂移可控。"""
    print(f"\n== shards: {dirpath} ==")
    files = sorted(glob.glob(str(dirpath / "*_off*.pt")))
    if not files:
        check("分片存在", False, str(dirpath)); return
    _, SO, _ = _consts()
    jn = _ckpt_joint_names()
    B = torch.tensor([SO.index(nm) for nm in jn])   # 用 ref[B] 把 SONIC 序转成 standard 序
    ck = torch.load(ROOT / "runs/S1_handover/exp_WBC_exec_haonly_best.pt",
                    map_location="cpu", weights_only=False)
    nj = len(jn)
    del ck
    pk = torch.load(ROOT / "runs/isaac_motion/ha_cmd_exec/packed.pt",
                    map_location="cpu", weights_only=False)
    ref = torch.as_tensor(pk["ref"]).float()
    starts = torch.tensor(pk["starts"]).long()
    W = 270

    gap = 0.0; zmin = 9.0; drift = []; reset_ok = []; reset_wrong = []
    for fn in files[:8]:                      # 抽样 8 个分片即够
        d = torch.load(fn, map_location="cpu", weights_only=False)
        S = d["states"].float(); S2 = d["next_states"].float()
        F = d["frames"].long(); C = d["clip_ids"].long()
        gap = max(gap, float((S - S2).abs().max()))
        zmin = min(zmin, float(S[:, 2].min()))
        R = ref[F]
        off = F - starts[C]
        seg = (off % W) % 20
        m0 = (seg == 0) & ((off % W) > 0)
        if int(m0.sum()) > 0:
            reset_ok.append(float((S[m0][:, 7:36] - R[m0][:, 13:42][:, B]).abs().mean()))
            reset_wrong.append(float((S[m0][:, 7:36] - R[m0][:, 13:42]).abs().mean()))
        dstep = (S2[:, 0:3] - S[:, 0:3]) - (ref[(F + 1).clamp(max=ref.shape[0] - 1)][:, 0:3] - R[:, 0:3])
        drift.append(dstep.norm(dim=-1))
    drift = torch.cat(drift)
    p50 = float(drift.median()); p90 = float(drift.quantile(0.9))
    check("有真实转移", gap > 1e-2, f"max|s−s'|={gap:.2f}")
    check("没有摔倒", zmin > 0.25, f"z_min={zmin:.3f}")
    check("逐行漂移 p50 < 2cm", p50 < 0.02, f"p50={p50*100:.2f}cm p90={p90*100:.2f}cm")
    # ★ 2026-11-04：**两种采集口径要分开验**
    #   · 固定段重置（--segment 20，ξ=0）：每个窗口的 segpos 0 都应精确复位 ⇒ 正序 ≈ 0
    #   · 缰绳式（--reset-max-drift δ）：重置只发生在 |ξ|>δ 的 env 上，segpos 0 是**任意
    #     中途行**（偏移可达 δ）⇒ 上面那条"段首复位≈0"根本不适用（leash 数据实测 0.065，
    #     不是 bug）。leash 口径改成验"**偏移最小的那 5% 行**的关节序是正序"：
    #     取 err_correct 最小的 5%，若正序确实对，这些行在错序口径下应明显更大。
    _leash = 0.0
    try:
        _leash = float(torch.load(files[0], map_location="cpu",
                                  weights_only=False).get("meta", {}).get("reset_max_drift", 0.0) or 0.0)
    except Exception:  # noqa: BLE001
        pass
    if _leash > 0:
        check("leash 口径（跳过固定段首检查）", True, f"reset_max_drift={_leash:.2f} m")
        ec, ew = [], []
        for fn in files[:8]:
            d = torch.load(fn, map_location="cpu", weights_only=False)
            S = d["states"].float(); F = d["frames"].long()
            R = ref[F]
            e_c = (S[:, 7:36] - R[:, 13:42][:, B]).abs().mean(-1)
            e_w = (S[:, 7:36] - R[:, 13:42]).abs().mean(-1)
            k = max(1, int(0.05 * e_c.numel()))
            top = torch.argsort(e_c)[:k]
            ec.append(float(e_c[top].median())); ew.append(float(e_w[top].median()))
        ec = float(np.mean(ec)); ew = float(np.mean(ew))
        check("偏移最小 5% 行的错序读明显更大", ew > 3 * ec + 0.02,
              f"正序 {ec:.4f} / 错序 {ew:.4f}")
    else:
        ok = float(np.mean(reset_ok)); bad = float(np.mean(reset_wrong))
        check("段首复位(正序) < 0.05 rad", ok < 0.05, f"{ok:.4f} rad")
        check("错序读明显更大", bad > 3 * ok + 0.02, f"正序 {ok:.4f} / 错序 {bad:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", type=Path,
                    default=ROOT / "runs/isaac_motion/ha_cmd_exec/packed.pt")
    ap.add_argument("--shards", type=Path, nargs="*",
                    default=[ROOT / "runs/a_prime_sonic"])
    ap.add_argument("--skip-shards", action="store_true")
    a = ap.parse_args()

    print(f"[verify] ROOT={ROOT}")
    check_lib_layout(a.packed)
    check_decoder_offset()
    check_decoder_obs()
    check_joint_order()
    if not a.skip_shards:
        for d in a.shards:
            if d.exists():
                check_shards(d)
            else:
                print(f"\n[verify] 跳过不存在的分片目录 {d}")

    nfail = sum(1 for _, ok, _ in RES if not ok)
    print(f"\n[verify] {len(RES) - nfail}/{len(RES)} PASS, {nfail} FAIL")
    if nfail:
        print("[verify] ❌ 契约不成立 —— 不要继续采集/训练/评估")
        return 1
    print("[verify] ✅ 全部契约成立")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
