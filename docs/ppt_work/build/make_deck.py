# -*- coding: utf-8 -*-
"""组装 17 页 deck。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import charts
from pptxlib import build
import slides_a as A
import slides_b as B
import slides_c as C
from notes import NOTES

SKEL = os.path.join(os.path.dirname(HERE), "unz")
OUTDIR = os.path.dirname(HERE)
OUT = os.path.join(OUTDIR, "COMIMIC_课程设计汇报_Aprime世界模型与可微联合微调.pptx")


def main() -> int:
    os.makedirs(os.path.join(HERE, "assets"), exist_ok=True)
    dj = charts.chart_dj()
    wm = charts.chart_wm()
    slides = [A.s01_cover(), A.s02_problem(), A.s03_system(), A.s04_idea(),
              A.s05_data(), A.s06_wm_train(), A.s07_wm_verify(),
              B.s07_cost1(), B.s08_cost2(), B.s09_chain(),
              B.s10_starts(), B.s11_pipeline(),
              C.s13_result(dj), C.s14_diag(), C.s15_eval(), C.s16_concl(), C.s17_appendix()]
    assets = {"wm_refresh.png": wm, "dj_curve.png": dj, "dj_curve_target.png": charts.chart_dj_target()}
    warns = build(slides, OUT, SKEL, assets, notes=NOTES)
    print(f"[deck] {len(slides)} pages -> {OUT}")
    tight = []
    for i, sl in enumerate(slides, 1):
        for r, t in sl.fill:
            tight.append((r, i, t))
    tight.sort(reverse=True)
    print("[deck] 最紧的 12 个文本框：")
    for r, i, t in tight[:12]:
        print(f"    p{i:02d}  {r*100:5.0f}%  {t}")
    if warns:
        print(f"[deck] {len(warns)} warnings:")
        for w in warns:
            print("   ", w)
    else:
        print("[deck] 无溢出警告")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
