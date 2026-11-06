#!/usr/bin/env python3
"""P1c G1 指标：参考回放（Isaac 闭环）vs 参考库，算 root / 关节误差 + 摔倒 + 完成度。

G1 门（docs/PLAN_REFSTART_WM_PERTURB_20261102.md §6.3）::

    root 误差（xy）   < 0.10 m      （旧 WBC × ha_cmd：0.32 m）
    关节 RMSE         < 0.2 rad
    摔倒              0
    走完整条 clip      是

输入是 ``SM_DUMP_STATES`` 落盘的状态：

* ``states (T,71)`` = ``[p(3), quat_wxyz(4), q(29), v(3), w(3), qd(29)]``，**q 是 asset/HA 顺序**
  （``_record_state_dump`` 直接取 ``robot.data.joint_pos[:, body_idx]``），报告里的关节误差
  会先按 ``joint_contract`` 换到 STANDARD_29（= 参考库顺序）再比；
* ``frames``：online/http 模式是**真实库行号**；npy 模式是步号，参考行 = ``--start-row + k``
  （控制 50 Hz = 库 50 Hz，一步一帧）。

用法::

    # 旧 WBC × ha_cmd 的基线（复现 0.32 m）
    python3 scripts/eval_g1_replay.py \\
        --dump runs/B_success/replay/ref_doubledesk_0/dump_online/DoubleDesk_refonline_ep000.pt \\
        --packed runs/isaac_motion/ha_cmd/packed.pt --start-row 0 \\
        --result runs/B_success/replay/ref_doubledesk_0/isaac_online/*/result.json --out /tmp/base.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GATE_ROOT_XY_M = 0.10
GATE_JOINT_RMSE_RAD = 0.20
NUM_JOINTS = 29


def _perm_asset_to_standard() -> list[int]:
    """asset(=HA_V31_29 交错) 顺序 -> STANDARD_29（参考库顺序）。"""
    try:
        import joint_contract as JC

        return JC.permutation(src="ha_v31", dst="standard")
    except Exception:  # noqa: BLE001
        return list(range(NUM_JOINTS))


def _clip_of_row(starts: list[int], lengths: list[int], row: int) -> tuple[int, int, int]:
    """返回 (clip_index, clip_start, clip_end)。"""
    import bisect

    ci = max(0, min(bisect.bisect_right(starts, int(row)) - 1, len(starts) - 1))
    return ci, starts[ci], starts[ci] + lengths[ci] - 1


def _load_result(paths: list[str]) -> dict | None:
    for p in paths:
        for f in sorted(glob.glob(p)):
            try:
                return json.loads(Path(f).read_text())
            except Exception:  # noqa: BLE001
                continue
    return None


def analyse(dump: Path, packed: Path, start_row: int | None = None) -> dict:
    d = torch.load(dump, map_location="cpu", weights_only=False)
    S = d["states"].float()
    T = int(S.shape[0])
    maybe_frames = d.get("frames")
    meta = d.get("meta", {}) or {}
    lib = torch.load(packed, map_location="cpu", weights_only=False)
    ref = lib["ref"].float()
    starts = [int(x) for x in lib["starts"]]
    lengths = [int(x) for x in lib["lengths"]]
    n_lib = int(ref.shape[0])

    frames_src = str(meta.get("frames_source", "step"))
    use_lib_rows = False
    if maybe_frames is not None and int(maybe_frames.numel()) == T:
        f = maybe_frames.long()
        if frames_src == "library_row" and int(f.max()) < n_lib and int(f.max()) > int(f.min()):
            use_lib_rows = True
    if use_lib_rows:
        rows = maybe_frames.long().clamp(0, n_lib - 1)
        row_mode = "library_row"
    else:
        base = int(start_row if start_row is not None else (int(maybe_frames[0]) if maybe_frames is not None and maybe_frames.numel() else 0))
        ci, cs, ce = _clip_of_row(starts, lengths, base)
        rows = torch.clamp(torch.arange(base, base + T, dtype=torch.long), max=ce)
        row_mode = "start_row+step"

    ci, clip_start, clip_end = _clip_of_row(starts, lengths, int(rows[0]))
    R = ref[rows]
    clip_len = int(lengths[ci])

    dp = S[:, :2] - R[:, :2]
    dist_xy = dp.norm(dim=-1)
    dist_3d = (S[:, :3] - R[:, :3]).norm(dim=-1)
    dz = (S[:, 2] - R[:, 2]).abs()

    q_asset = S[:, 13:13 + NUM_JOINTS]
    q_ref = R[:, 13:13 + NUM_JOINTS]
    perm = _perm_asset_to_standard()
    q_std = q_asset[:, perm]
    jerr = (q_std - q_ref).abs()
    rmse_per_step = (jerr.pow(2).mean(dim=-1)).sqrt()

    def q(x):
        x = x.numpy()
        return {"mean": float(x.mean()), "p50": float(np.percentile(x, 50)),
                "p90": float(np.percentile(x, 90)), "max": float(x.max()),
                "final": float(x[-1])}

    out = {
        "dump": str(dump),
        "packed": str(packed),
        "steps": T,
        "row_mode": row_mode,
        "clip_index": int(ci),
        "clip_name": lib["names"][ci],
        "clip_start_row": int(clip_start),
        "clip_end_row": int(clip_end),
        "clip_len": clip_len,
        "completion": round(T / max(clip_len, 1), 4),
        "root_xy_err_m": q(dist_xy),
        "root_3d_err_m": q(dist_3d),
        "root_z_abs_err_m": q(dz),
        "joint_abs_err_rad": q(jerr.mean(dim=-1)),
        "joint_rmse_rad": q(rmse_per_step),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help=".pt 路径（可 glob）")
    ap.add_argument("--packed", default=str(ROOT / "runs/isaac_motion/ha_cmd_exec/packed.pt"))
    ap.add_argument("--start-row", type=int, default=None)
    ap.add_argument("--result", nargs="*", default=[], help="result.json 路径（可 glob）")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    dumps = sorted(glob.glob(a.dump))
    if not dumps:
        raise SystemExit(f"[g1] no dump matched {a.dump}")
    res = _load_result(a.result) if a.result else None
    reports = []
    for f in dumps:
        r = analyse(Path(f), Path(a.packed), a.start_row)
        if res is not None:
            r["result"] = {k: res.get(k) for k in
                           ("success", "failure_reason", "episode_steps", "max_steps",
                            "terminal_detail", "duration_sec")}
            r["fall"] = str(res.get("failure_reason")) == "fall"
            r["walked_full_clip"] = (not r["fall"]) and int(res.get("episode_steps", 0)) >= r["clip_len"] - 2
        else:
            r["fall"] = None
            r["walked_full_clip"] = r["completion"] > 0.98
        r["gates"] = {
            "root_xy_mean_lt_0.10m": r["root_xy_err_m"]["mean"] < GATE_ROOT_XY_M,
            "joint_rmse_mean_lt_0.2rad": r["joint_rmse_rad"]["mean"] < GATE_JOINT_RMSE_RAD,
            "no_fall": (r["fall"] is False) if r["fall"] is not None else None,
            "walked_full_clip": bool(r["walked_full_clip"]),
        }
        reports.append(r)
        tag = a.tag or Path(f).parent.name
        print(f"\n[g1] {tag}: {r['clip_name']} steps={r['steps']}/{r['clip_len']} "
              f"({r['completion'] * 100:.1f}%) mode={r['row_mode']}")
        print(f"     root xy  mean={r['root_xy_err_m']['mean']:.3f} p90={r['root_xy_err_m']['p90']:.3f} "
              f"max={r['root_xy_err_m']['max']:.3f} final={r['root_xy_err_m']['final']:.3f} m")
        print(f"     joint    RMSE={r['joint_rmse_rad']['mean']:.3f} rad "
              f"(p90 {r['joint_rmse_rad']['p90']:.3f}, max {r['joint_rmse_rad']['max']:.3f})")
        print(f"     fall={r['fall']} walked_full={r['walked_full_clip']} gates={json.dumps(r['gates'])}")

    payload = {"reports": reports}
    if len(reports) == 1:
        payload["verdict"] = "PASS" if all(v is not False for v in reports[0]["gates"].values()) else "FAIL"
        print(f"\n[g1] VERDICT = {payload['verdict']}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
        print(f"[g1] report -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
