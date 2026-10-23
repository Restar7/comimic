"""Apply the deployment's success criteria to the state logs from run_episode.py.

Criteria are copied from the checks the deployment actually used:
  * check_handover_success.py            (handover)
  * check_redcup_success.py              (red cup)
  * check_pickplace_success.py           (apple)
  * carton: same "object in target box, static" rule, target = sugar_box / box
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BOX_INNER_HALF_XY = 0.12
BOX_INNER_HEIGHT = 0.15
LIFT_THRESHOLD = 0.05
STATIC_THRESHOLD = 0.10
STATIC_HOLD_FRAMES = 25


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pos(rows, name, i=None):
    i = len(rows) - 1 if i is None else i
    return np.asarray(rows[i]["tracked_bodies"][name]["position"], dtype=float)


def speeds(rows, name):
    p = np.stack([np.asarray(r["tracked_bodies"][name]["position"]) for r in rows])
    v = np.linalg.norm(np.diff(p, axis=0), axis=1) * 50.0
    return np.concatenate([[0.0], v])


def check_handover(rows):
    sugar = [pos(rows, "sugar_box", i) for i in range(len(rows))]
    tgt = [pos(rows, "tgt_box", i) for i in range(len(rows))]
    inside_since = None
    first = None
    last = None
    for i, (s, t) in enumerate(zip(sugar, tgt)):
        d = s - t
        inside = abs(d[0]) < 0.10 and abs(d[1]) < 0.10 and -0.07 < d[2] < 0.16
        moving = last is not None and np.linalg.norm(s - last) > 0.02
        last = s.copy()
        tt = rows[i]["elapsed"]
        if inside and not moving:
            if inside_since is None:
                inside_since = tt
            elif tt - inside_since >= 1.0 and first is None:
                first = tt
        else:
            inside_since = None
    return {"success": first is not None, "success_time_s": first,
            "final_delta": (sugar[-1] - tgt[-1]).round(3).tolist()}


def check_redcup(rows):
    cups = np.stack([pos(rows, "mug", i) for i in range(len(rows))])
    drawer = np.stack([pos(rows, "drawer", i) for i in range(len(rows))])
    rel = cups[-1] - drawer[-1]
    in_drawer = (
        abs(rel[0]) <= 0.13 and abs(rel[1]) <= 0.12
        and -0.08 <= rel[2] <= 0.11 and cups[-1][2] >= 0.74
    )
    opened = float((drawer[:, 0] - drawer[0, 0]).max())
    closed = float(drawer[-1, 0] - drawer[0, 0]) >= 0.075
    return {
        "success": bool(in_drawer and closed),
        "stage": "drawer_closed_success" if (in_drawer and closed)
        else "object_in_drawer" if in_drawer
        else "object_moved" if np.linalg.norm(cups[-1] - cups[0]) >= 0.02
        else "no_object_interaction",
        "in_drawer": bool(in_drawer),
        "drawer_max_open_m": round(opened, 4),
        "drawer_final_offset_m": round(float(drawer[-1, 0] - drawer[0, 0]), 4),
        "cup_lift_m": round(float((cups[:, 2] - cups[0, 2]).max()), 4),
        "cup_final": cups[-1].round(3).tolist(),
        "drawer_final": drawer[-1].round(3).tolist(),
    }


def check_pickplace(rows, obj="apple", box="box"):
    o = np.stack([pos(rows, obj, i) for i in range(len(rows))])
    b = np.stack([pos(rows, box, i) for i in range(len(rows))])
    v = speeds(rows, obj)
    zlo = b[0][2] - 0.02
    in_box = [
        abs(o[i][0] - b[i][0]) < BOX_INNER_HALF_XY
        and abs(o[i][1] - b[i][1]) < BOX_INNER_HALF_XY
        and zlo < o[i][2] < zlo + BOX_INNER_HEIGHT
        for i in range(len(rows))
    ]
    hold = 0
    success_time = None
    for i in range(len(rows)):
        if in_box[i] and v[i] < STATIC_THRESHOLD:
            hold += 1
            if hold >= STATIC_HOLD_FRAMES and success_time is None:
                success_time = rows[i]["elapsed"]
        else:
            hold = 0
    return {
        "success": success_time is not None,
        "success_time_s": success_time,
        "object_final": o[-1].round(3).tolist(),
        "box_final": b[-1].round(3).tolist(),
        "min_obj_box_dist_m": round(float(np.linalg.norm(o - b, axis=1).min()), 4),
        "max_lift_m": round(float((o[:, 2] - o[0, 2]).max()), 4),
        "ever_inside_box": bool(any(in_box)),
    }


def check_carton(rows):
    res = check_pickplace(rows, obj="sugar_box", box="box")
    res["min_carton_box_dist_m"] = res["min_obj_box_dist_m"]
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/home/wusichen/groot_repro/out/runs")
    args = ap.parse_args()
    runs = Path(args.runs)
    report = {}
    for name, fn in (("handover", check_handover), ("redcup", check_redcup),
                     ("apple", check_pickplace), ("carton", check_carton)):
        p = runs / name / "state.jsonl"
        if not p.exists():
            continue
        rows = load(p)
        meta = json.loads((runs / name / "meta.json").read_text())
        out = fn(rows)
        out["seconds"] = meta["seconds"]
        out["final_tilt_deg"] = meta["final_tilt_deg"]
        out["fell"] = meta["fell"]
        report[name] = out
        print(f"{name:9s} success={out['success']!s:5s} " +
              " ".join(f"{k}={v}" for k, v in out.items() if k != "success"))
    (runs / "success_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {runs/'success_report.json'}")


if __name__ == "__main__":
    main()
