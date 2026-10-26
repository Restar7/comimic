#!/usr/bin/env python3
"""S3 两臂（或任意臂）训练监督器：每分钟把关键读数落一行，供人扫。

不做任何判断、不发通知，只把下面这些"能一眼看出跑歪"的量并排写出来：

    step  s/step  loss  hand_fm  anchor  reward_norm  hard_rew  critic_v_ev  token_mse  imag_ret_vla/gt

用法：
    python3 scripts/s3_watch.py --arms A0A1h20_A0 A0A1h20_A1 --interval 60
    python3 scripts/s3_watch.py --once            # 打一次当前快照就退出
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

ROOT = Path("/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic")
LOG_DIR = ROOT / "logs"

STEP_RE = re.compile(r"^\[v2 (\d+)/(\d+)\] (\{.*\})\s*$")
EVAL_RE = re.compile(r"^\[eval-v2 (\d+)\] (\{.*\})\s*$")

FIELDS = [
    ("loss", "loss"),
    ("hand_fm", "loss/hand_fm"),
    ("anchor", "loss/anchor"),
    ("ret", "_ret"),          # = -(loss/reward_norm) = 想象回报 ret+boot
    ("hard", "bucket/hard_rew"),
    ("v_ev", "critic/v_ev"),
    ("r/h", "grad/r_over_hand"),
]
EVAL_FIELDS = [
    ("tok_mse", "token_mse_decoded"),
    ("hand_mse", "hand_mse_decoded"),
    ("hand170", "hand_gt170_pred"),
    ("shift", "token_shift"),
    ("ret_vla", "imag_return_vla"),
    ("ret_gt", "imag_return_gt"),
    ("surv", "imag_survival_vla"),
]


def tail_state(path: Path) -> dict:
    st: dict = {"step": None, "total": None, "last": None, "eval": None, "s_per_step": None}
    if not path.exists():
        return st
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return st
    for ln in reversed(lines[-4000:]):
        m = STEP_RE.match(ln)
        if m and st["last"] is None:
            st["step"], st["total"] = int(m.group(1)), int(m.group(2))
            try:
                last = json.loads(m.group(3))
                # loss/reward_norm = -(ret+boot)：报表里翻回正号，别再读反
                if "loss/reward_norm" in last:
                    last["_ret"] = -float(last["loss/reward_norm"])
                st["last"] = last
                st["s_per_step"] = last.get("s_per_step")
            except json.JSONDecodeError:
                pass
        m2 = EVAL_RE.match(ln)
        if m2 and st["eval"] is None:
            try:
                st["eval"] = json.loads(m2.group(2))
            except json.JSONDecodeError:
                pass
        if st["last"] is not None and st["eval"] is not None:
            break
    return st


def fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:+.4f}" if abs(v) < 100 else f"{v:+.1f}"
    return str(v)


def snapshot(arms: list[str]) -> str:
    out = [f"=== {time.strftime('%H:%M:%S')} ==="]
    for arm in arms:
        st = tail_state(LOG_DIR / f"{arm}.log")
        head = f"[{arm}] {st['step']}/{st['total']}  {fmt(st['s_per_step'])}s/it"
        if st["step"] is not None and st["s_per_step"]:
            left = (st["total"] - st["step"]) * st["s_per_step"] / 60
            head += f"  ETA {left:.0f}min"
        out.append(head)
        if st["last"]:
            out.append("   train " + "  ".join(f"{k}={fmt(st['last'].get(v))}" for k, v in FIELDS))
        if st["eval"]:
            out.append("   eval  " + "  ".join(f"{k}={fmt(st['eval'].get(v))}" for k, v in EVAL_FIELDS))
        if st["step"] is None:
            out.append("   (尚未开始 step：加载/门禁阶段)")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["A0A1h20_A0", "A0A1h20_A1"])
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--out", type=Path, default=LOG_DIR / "A0A1h20.watch.log")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()

    if a.once:
        print(snapshot(a.arms))
        return 0
    with a.out.open("a", encoding="utf-8") as f:
        while True:
            s = snapshot(a.arms)
            f.write(s + "\n")
            f.flush()
            time.sleep(a.interval)


if __name__ == "__main__":
    raise SystemExit(main())
