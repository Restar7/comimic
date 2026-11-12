#!/usr/bin/env python3
"""**闭环漂移：root + 14 个 body（相对参考动作库）** —— 2026-11-07。

为什么要重做（以前的 root 漂移数是错的）
----------------------------------------
`runs/ab36/*/*/dump/steps.jsonl` 里的 `frame` 是 **provider 自己的控制步计数器**
（`action_provider_sonic.py::_dbg_step` 写 `self._frame_count`），**不是**运动库行号。
`scripts/eval_closedloop_drift.py` 以前直接 `ref[frame]` 取参考行 ⇒ 参考是**无关动作**
（跑到 HSI_boxing / HOI_desk 的片段上去了），所以那批 "drift 2.48 m / corr −0.4" 全部作废。

正确的参考从哪来
----------------
三个任务（OpenDoor / SitSofa / VisionNavi）都是 HumanoidArena 的**任务 env**，闭环时**不播参考**；
但运动库 `runs/isaac_motion/ha_cmd/packed.pt` 里有它们的**示教**：
`HSI_open_door__*`(100) / `HSI_sit_sofa__*`(100) / `HSI_vision_navi__*`(100)，
而且**每条示教的第 0 帧 root 位姿 == env 的 reset 位姿**（实测 OpenDoor：库
`[-1.5996, 0.2000, 0.7998]` vs dump `[-1.6002, 0.2003, 0.7970]`）。
⇒ 示教就是"参考"，t=0 天然对齐，不需要任何 NN/DTW。

关节序（实测定的，不是猜的）
--------------------------
* dump 的 `q` = **HA_V31_29**（按 KA 序读：膝 +0.427/+0.437、踝 −0.191/−0.220、
  肘 +1.055/+1.050，左右对称；按 STANDARD 读则完全错乱）。
* 库的 `ref[:,7:36]` 也是 HA_V31_29，但是 **q − default**（第 0 帧全 0）。
* FK（`RobotKinematics.forward`）要 **STANDARD_29** ⇒ 两边都先 `permutation('ha_v31','standard')`。
  库是相对量 ⇒ 直接 FK 相对 q 等价于 FK 绝对 q 再减同一根位姿（只差一个常量），
  而我们的漂移口径**本来就减掉起点**，所以不需要恢复 default。

口径
----
* 参考 = 该任务 100 条示教在 body 空间的**逐帧中位数**（`--ref-mode median`，无挑参）
  或**每集最贴合的示教**（`--ref-mode best`，是"下界"，会低估漂移）。
* root 漂移  e_t = ‖(p_t − p_0) − (r_t − r_0)‖
* body 漂移  e_t^b = ‖(b_t^b − b_0^b) − (r_t^b − r_0^b)‖，b ∈ 14 个 tracked body
* 全部**减掉起点** ⇒ 与 reset 位姿/世界系无关。
* 另外给**不依赖参考**的因果量：foot slip（接触期脚位移速度）、pelvis 高度、
  躯干/手的抖动、站桩占比、手的行程 —— 这些是"漂移从哪来"的解释项。

用法::

    python3 scripts/eval_body_drift_vs_ref.py --ref-mode both \
        --cells base=runs/ab36/base delta=runs/ab36/delta --out runs/ab36/body_drift.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

TASK_PREFIX = {
    "OpenDoor": "HSI_open_door",
    "SitSofa": "HSI_sit_sofa",
    "VisionNavi": "HSI_vision_navi",
}

# ---- steps.jsonl 的快速解析（行 26 KB，只取 4 个字段；整库 ~1.3 GB）------------
_F_FRAME = b'"frame": '
_F_Q = b'"q": ['
_F_P = b'"root_pos": ['
_F_QUAT = b'"root_quat_wxyz": ['


def _num_list(buf: bytes, start: int) -> tuple[np.ndarray, int]:
    end = buf.find(b"]", start)
    return np.fromstring(buf[start:end].decode(), sep=",", dtype=np.float32), end


def parse_steps(path: Path) -> dict[str, np.ndarray]:
    """只抽 frame / q / root_pos / root_quat_wxyz（其余字段不解析）。"""
    raw = path.read_bytes()
    frames, qs, ps, quats = [], [], [], []
    pos = 0
    n = len(raw)
    while pos < n:
        nl = raw.find(b"\n", pos)
        line = raw[pos:nl if nl >= 0 else n]
        pos = (nl + 1) if nl >= 0 else n
        if not line.strip():
            continue
        i = line.find(_F_FRAME)
        if i < 0:
            continue
        j = line.find(b",", i + len(_F_FRAME))
        frames.append(int(line[i + len(_F_FRAME):j]))
        i = line.find(_F_Q)
        v, _ = _num_list(line, i + len(_F_Q))
        qs.append(v)
        i = line.find(_F_P)
        v, _ = _num_list(line, i + len(_F_P))
        ps.append(v)
        i = line.find(_F_QUAT)
        v, _ = _num_list(line, i + len(_F_QUAT))
        quats.append(v)
    return {
        "frame": np.asarray(frames, np.int64),
        "q": np.stack(qs).astype(np.float32),
        "root_pos": np.stack(ps).astype(np.float32),
        "root_quat": np.stack(quats).astype(np.float32),
    }


def segment(s: dict[str, np.ndarray]) -> list[slice]:
    """按 frame 回退切 episode（每集一个 Isaac 进程，provider 计数器从 1 重新开始）。"""
    f = s["frame"]
    cuts = [0] + [i for i in range(1, len(f)) if f[i] <= f[i - 1]] + [len(f)]
    return [slice(cuts[k], cuts[k + 1]) for k in range(len(cuts) - 1)]


def quat_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], -1).reshape(-1, 3, 3)


class FKer:
    """把 root(p,quat)+q(HA) → (T,14,3) 的 body 位置。"""

    def __init__(self, device: str = "cuda:0", chunk: int = 2048):
        import torch  # noqa: PLC0415

        from diff_vla_wbc.isaac_joint import load_isaac_bundle  # noqa: PLC0415
        from pathlib import Path as _P

        self.torch = torch
        self.dev = device
        self.chunk = chunk
        b = load_isaac_bundle(
            str(ROOT / "runs/S1_handover/WBC_A24000_wmvel_cmd.pt"), device=device,
            npz=sorted((ROOT / "data/npz/ha_tokens").glob("*.npz"))[:1],
            packed_motion=str(ROOT / "runs/isaac_motion/ha_cmd_exec/packed.pt"),
            obs_mode=None, hist=None, token_dim=64)
        self.kin = b.mdp.kin
        self.body_names = [self.kin.body_names[i] for i in self.kin.tracked_idx]

    def __call__(self, p: np.ndarray, quat: np.ndarray, q_ha: np.ndarray) -> np.ndarray:
        import joint_contract as JC  # noqa: PLC0415

        torch = self.torch
        q_std = np.ascontiguousarray(q_ha[:, JC.permutation("ha_v31", "standard")])
        out = np.empty((len(p), len(self.body_names), 3), np.float32)
        for a in range(0, len(p), self.chunk):
            bb = slice(a, a + self.chunk)
            n = len(p[bb])
            P = torch.tensor(p[bb], dtype=torch.float32, device=self.dev)
            R = torch.tensor(quat_to_R(quat[bb]), dtype=torch.float32, device=self.dev)
            z3 = torch.zeros(n, 3, device=self.dev)
            z29 = torch.zeros(n, 29, device=self.dev)
            Q = torch.tensor(q_std[bb], dtype=torch.float32, device=self.dev)
            out[bb] = self.kin.forward(P, R, z3, z3, Q, z29)[0].cpu().numpy()
        return out


# ------------------------------------------------------------------ 参考库
def build_references(fk: FKer, packed: Path) -> dict[str, dict]:
    torch = fk.torch
    pk = torch.load(packed, map_location="cpu", weights_only=False)
    ref = np.asarray(pk["ref"], np.float64)
    names = list(pk["names"])
    starts = np.asarray(pk["starts"])
    lengths = np.asarray(pk["lengths"])
    out: dict[str, dict] = {}
    for task, prefix in TASK_PREFIX.items():
        ids = [i for i, n in enumerate(names) if n.startswith(prefix)]
        if not ids:
            continue
        demos = []
        for i in ids:
            s, l = int(starts[i]), int(lengths[i])
            demos.append((names[i], s, l, ref[s:s + l]))
        L = int(max(l for _, _, l, _ in demos))
        B = np.full((len(demos), L, len(fk.body_names), 3), np.nan, np.float32)
        for k, (_, _, l, d) in enumerate(demos):
            B[k, :l] = fk(d[:l, 0:3], d[:l, 3:7], d[:l, 7:36])
        alive = np.isfinite(B[:, :, 0, 0]).sum(0)            # (L,) 每帧还有几条示教活着
        med = np.nanmedian(B, axis=0)                         # (L,14,3) 逐帧中位数
        out[task] = {
            "n_demos": len(demos), "L": L, "median_bodies": med, "alive": alive,
            "demos": [(nm, s, l) for nm, s, l, _ in demos],
            "demo_bodies": B,
        }
        print(f"[ref] {task}: {len(demos)} 示教, 最长 {L} 帧 (中位长度 "
              f"{int(np.median([l for _, _, l, _ in demos]))}); "
              f"存活示教数 @0/500/1000/2000 = "
              f"{[int(alive[min(i, L - 1)]) for i in (0, 500, 1000, 2000)]}", flush=True)
    return out


def _stat(x: np.ndarray) -> dict:
    x = np.asarray(x, np.float64)
    return {"mean": float(np.mean(x)), "p50": float(np.median(x)),
            "p90": float(np.quantile(x, 0.9)), "max": float(np.max(x))}


def analyse_episode(bp: np.ndarray, root: np.ndarray, refs: dict, mode: str,
                    pick_steps: int = 300) -> dict:
    """一个 episode：body 位置 (T,14,3) + root → 相对参考的 root/body 漂移。

    参考帧不足（示教比 episode 短）的地方按 **可用帧** 统计，不做外推/填充。
    """
    T = len(bp)
    n = min(T, refs["L"])
    if mode == "median":
        r = refs["median_bodies"][:n]
    else:
        demos = refs["demo_bodies"][:, :n]
        k = min(pick_steps, n)
        ok = np.isfinite(demos[:, :k, 0, 0]).all(1)          # 前 k 步完整的示教
        dd = demos[ok, :k] - demos[ok, 0:1]
        aa = bp[:k] - bp[:1]
        e = np.linalg.norm(dd - aa[None], axis=-1).mean(axis=(1, 2))   # (D,)
        r = demos[np.flatnonzero(ok)[int(np.argmin(e))]]
    valid = np.isfinite(r[:, 0, 0])                           # (n,) 有参考的帧
    db_act = bp[:n] - bp[:1]
    db_ref = r - r[:1]
    e_body = np.linalg.norm(db_act - db_ref, axis=-1)         # (n,14) 米
    dp_act = root[:n] - root[:1]
    e_root = np.linalg.norm(dp_act - (r[:, 0] - r[0, 0]), axis=-1)   # (n,) 用 FK pelvis 当参考 root
    eb, er = e_body[valid], e_root[valid]
    # ---- 误差分解：沿参考切向 / 垂直参考切向（说清"走短了"还是"拐歪了"）
    dpa = np.diff(db_act, axis=0)                    # (n-1,3) 实际每步位移
    dpr = np.diff(db_ref[:, 0], axis=0)              # (n-1,3) 参考 pelvis 每步位移（切向）
    nrm = np.linalg.norm(dpr, axis=-1)
    good = nrm > 1e-4
    that = np.zeros_like(dpr)
    that[good] = dpr[good] / nrm[good][:, None]
    err = db_act - db_ref                            # (n,14,3) 误差向量（含 14 个 body）
    e1 = err[1:]
    that_b = that[:, None, :]                        # (n-1,1,3) 广播到 body
    along = np.sum(e1 * that_b, axis=-1)             # (n-1,14) + = 走过头
    cross = np.linalg.norm(e1 - along[..., None] * that_b, axis=-1)
    na = np.linalg.norm(dpa, axis=-1)                # (n-1,14)
    use = good & (na > 1e-3).any(axis=-1) & valid[1:]
    cos_dir = (np.sum(dpa * that_b, axis=-1) / np.maximum(na, 1e-9)).mean(axis=-1)
    # 净朝向误差（实际净位移方向 vs 参考净位移方向）
    nd_a, nd_r = db_act[-1, 0], db_ref[-1, 0]
    na_, nr_ = np.linalg.norm(nd_a), np.linalg.norm(nd_r)
    head_deg = float("nan")
    if na_ > 0.2 and nr_ > 0.2:
        head_deg = float(np.degrees(np.arccos(np.clip(nd_a @ nd_r / (na_ * nr_), -1, 1))))
    bins = ((0, 300), (300, 600), (600, 1200), (1200, 10 ** 9))
    by_bin = {}
    for lo, hi in bins:
        m = valid & (np.arange(n) >= lo) & (np.arange(n) < hi)
        key = f"{lo}-{hi if hi < 10**9 else '+'}"
        by_bin[key] = {"body_p50": float(np.median(e_body[m])) if m.any() else float("nan"),
                       "root_p50": float(np.median(e_root[m])) if m.any() else float("nan")}
    return {
        "n": int(valid.sum()), "T": int(T), "ref_horizon": int(n),
        "body_mean": float(eb.mean()), "body_p50": float(np.median(eb)),
        "body_p90": float(np.quantile(eb, 0.9)),
        "body_per_body_p50": np.median(e_body[valid], axis=0).tolist(),
        "root_mean": float(er.mean()), "root_p50": float(np.median(er)),
        "root_p90": float(np.quantile(er, 0.9)),
        "root_tail_p50": float(np.median(e_root[valid][-200:])),
        "bins": by_bin,
        "along_p50": float(np.median(np.abs(along[valid[1:]]))) if valid[1:].any() else float("nan"),
        "cross_p50": float(np.median(cross[valid[1:]])) if valid[1:].any() else float("nan"),
        "along_mean_signed": float(np.mean(along[valid[1:]])) if valid[1:].any() else float("nan"),
        "dir_cos_mean": float(np.mean(cos_dir[use])) if use.any() else float("nan"),
        "net_dir_err_deg": head_deg,
        "net_act": float(na_), "net_ref": float(nr_),
    }


def locomotion_metrics(bp: np.ndarray, root: np.ndarray, dt: float = 0.02) -> dict:
    """不依赖参考的因果量。"""
    v = np.linalg.norm(np.diff(root, axis=0), axis=-1) / dt          # m/s
    still = float(np.mean(v < 0.05))
    # foot slip：脚在接触（踝高 < 6 cm）时的水平位移速度
    slip = {}
    for side, idx in (("L", 3), ("R", 6)):
        f = bp[:, idx]
        contact = f[:, 2] < 0.06
        if contact.sum() > 5:
            dv = np.linalg.norm(np.diff(f[:, :2], axis=0), axis=-1) / dt
            slip[side] = float(np.mean(dv[contact[1:]]))
        else:
            slip[side] = float("nan")
    # 抖动：躯干/手的二阶差分
    jit = {}
    for nm, idx in (("torso", 7), ("lwrist", 10), ("rwrist", 13)):
        a = bp[:, idx]
        jit[nm] = float(np.mean(np.linalg.norm(a[2:] - 2 * a[1:-1] + a[:-2], axis=-1)) / dt ** 2)
    return {
        "steps": int(len(bp)), "dur_s": float(len(bp) * dt),
        "root_speed_mean": float(v.mean()), "still_frac": still,
        "pelvis_h_p50": float(np.median(bp[:, 0, 2])),
        "pelvis_h_min": float(np.min(bp[:, 0, 2])),
        "slip_L": slip["L"], "slip_R": slip["R"],
        "jitter_torso": jit["torso"], "jitter_lwrist": jit["lwrist"],
        "jitter_rwrist": jit["rwrist"],
        "lwrist_range": float(np.linalg.norm(bp[:, 10] - bp[:, 10].mean(0)) .max()),
        "rwrist_range": float(np.linalg.norm(bp[:, 13] - bp[:, 13].mean(0)).max()),
        "path_len": float(v.sum() * dt),
        "net_disp": float(np.linalg.norm(root[-1] - root[0])),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True, help="ARM=DIR（DIR 下有 TASK/）")
    ap.add_argument("--ref", type=Path, default=ROOT / "runs/isaac_motion/ha_cmd/packed.pt")
    ap.add_argument("--ref-mode", choices=["median", "best", "both"], default="both")
    ap.add_argument("--tasks", nargs="+", default=list(TASK_PREFIX))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    fk = FKer(a.device)
    refs = build_references(fk, a.ref)
    modes = ["median", "best"] if a.ref_mode == "both" else [a.ref_mode]

    rows = []
    for spec in a.cells:
        arm, _, d = spec.partition("=")
        cell = Path(d)
        cell = cell if cell.is_absolute() else ROOT / cell
        for task in a.tasks:
            p = cell / task / "dump" / "steps.jsonl"
            if not p.exists():
                print(f"[skip] {arm}/{task}: 没有 dump", flush=True)
                continue
            s = parse_steps(p)
            segs = segment(s)
            eps = [json.loads((cell / task / f"ep_seed{i}.json").read_text()) for i in range(6)]
            if len(segs) != len(eps):
                print(f"[warn] {arm}/{task}: {len(segs)} 段 vs {len(eps)} 个 ep_seed.json", flush=True)
            for i, sl in enumerate(segs):
                e = eps[i] if i < len(eps) else {"success": None, "episode_steps": -1, "failure_reason": "?"}
                if len(s["frame"][sl]) != e.get("episode_steps"):
                    print(f"[warn] {arm}/{task} seg{i}: dump {len(s['frame'][sl])} 步 vs "
                          f"json {e.get('episode_steps')} 步", flush=True)
                bp = fk(s["root_pos"][sl], s["root_quat"][sl], s["q"][sl])
                base = {"arm": arm, "task": task, "seed": i, "success": e.get("success"),
                        "failure_reason": e.get("failure_reason"),
                        "ep_steps": e.get("episode_steps")}
                lm = locomotion_metrics(bp, s["root_pos"][sl])
                base.update(lm)
                for mode in modes:
                    r = analyse_episode(bp, s["root_pos"][sl], refs[task], mode)
                    base[f"ref_{mode}"] = r
                rows.append(base)
                print(f"[ok] {arm}/{task} s{i} ok={base['success']} steps={lm['steps']} "
                      f"root_p50={base['ref_median']['root_p50']:.2f} "
                      f"body_p50={base['ref_median']['body_p50']:.3f}", flush=True)

    out = a.out or (ROOT / "runs/ab36/body_drift.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"\n[write] {out}  ({len(rows)} episodes)")

    # ---------------- 汇总表 ----------------
    bodies = [fk.body_names[i] for i in range(len(fk.body_names))]
    def agg(sel):
        if not sel:
            return None
        k = lambda f: float(np.mean([f(r) for r in sel]))
        return k
    print("\n" + "=" * 118)
    print("参考 = 该任务 100 条示教的逐帧中位数（body 空间）；全部减掉起点；单位 m")
    print("=" * 118)
    hdr = f"{'arm/task':<22}{'ok':>4}{'n':>5}{'root p50':>9}{'root p90':>9}{'body mean':>10}{'body p50':>9}{'body p90':>9}{'slip':>7}{'still':>7}{'pelv h':>8}"
    print(hdr)
    for arm in dict.fromkeys(r["arm"] for r in rows):
        for task in dict.fromkeys(r["task"] for r in rows):
            sel = [r for r in rows if r["arm"] == arm and r["task"] == task]
            if not sel:
                continue
            nok = sum(1 for r in sel if r["success"])
            m = lambda key, d: float(np.mean([r["ref_median"][key] for r in sel]))
            print(f"{arm + '/' + task:<22}{nok:>4}{len(sel):>5}{m('root_p50', 0):>9.2f}"
                  f"{m('root_p90', 0):>9.2f}{m('body_mean', 0):>10.3f}{m('body_p50', 0):>9.3f}"
                  f"{m('body_p90', 0):>9.3f}"
                  f"{np.nanmean([r['slip_L'] for r in sel]):>7.2f}"
                  f"{np.mean([r['still_frac'] for r in sel]):>7.2f}"
                  f"{np.mean([r['pelvis_h_p50'] for r in sel]):>8.3f}")

    print("\n[按 success 拆]（中位数模式）")
    print(f"{'arm/task/ok':<26}{'n':>4}{'root p50':>9}{'body p50':>9}{'body p90':>9}")
    for arm in dict.fromkeys(r["arm"] for r in rows):
        for task in dict.fromkeys(r["task"] for r in rows):
            for ok in (True, False):
                sel = [r for r in rows if r["arm"] == arm and r["task"] == task and r["success"] is ok]
                if not sel:
                    continue
                mm = lambda key: float(np.mean([r["ref_median"][key] for r in sel]))
                print(f"{arm + '/' + task + '/' + ('ok' if ok else 'fail'):<26}{len(sel):>4}"
                      f"{mm('root_p50'):>9.2f}{mm('body_p50'):>9.3f}{mm('body_p90'):>9.3f}")

    print("\n[每个 body 的漂移 p50（全 episode 平均，median 参考）]")
    print(f"{'body':<26}" + "".join(f"{a:>12}" for a in dict.fromkeys(r['arm'] for r in rows)))
    for b, nm in enumerate(bodies):
        line = f"{nm:<26}"
        for arm in dict.fromkeys(r["arm"] for r in rows):
            sel = [r for r in rows if r["arm"] == arm]
            line += f"{np.mean([r['ref_median']['body_per_body_p50'][b] for r in sel]):>12.3f}" if sel else f"{'-':>12}"
        print(line)

    print("\n[best 参考 = 下界（挑最贴合的示教）]")
    print(f"{'arm/task':<22}{'body p50':>10}{'root p50':>9}")
    for arm in dict.fromkeys(r["arm"] for r in rows):
        for task in dict.fromkeys(r["task"] for r in rows):
            sel = [r for r in rows if r["arm"] == arm and r["task"] == task]
            if not sel:
                continue
            print(f"{arm + '/' + task:<22}"
                  f"{np.mean([r['ref_best']['body_p50'] for r in sel]):>10.3f}"
                  f"{np.mean([r['ref_best']['root_p50'] for r in sel]):>9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
