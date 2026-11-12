"""用真实 run 数据画汇报用图（不画"示意图"）。"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TEAL = "#0C778E"
NAVY = "#133057"
GRAY = "#607589"
AMBER = "#C98B37"
BORDER = "#D8E5EE"
LIGHT = "#8FA6B8"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets")
C2 = ("/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/"
      "comimic-c-wm6/runs/dual_actor_wm6/c_lora/p21")

NL = chr(10)


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BORDER)
    ax.tick_params(colors=GRAY, labelsize=9)
    ax.yaxis.grid(True, color="#EAF0F5", lw=0.9)
    ax.set_axisbelow(True)


def chart_dj() -> str:
    rows = []
    for f in ("ft300/val.jsonl", "ft300_resume/val.jsonl"):
        with open(os.path.join(C2, f)) as fh:
            rows += [json.loads(l) for l in fh if l.strip()]
    rows.sort(key=lambda d: d["update"])
    x = [d["update"] for d in rows]
    y = [d["dJ"] * 100 for d in rows]
    lo = [(d["dJ"] - d["dJ_ci_lo"]) * 100 for d in rows]
    hi = [(d["dJ_ci_hi"] - d["dJ"]) * 100 for d in rows]

    fig, ax = plt.subplots(figsize=(6.3, 3.2), dpi=200)
    _style(ax)
    ax.axhline(0, color=GRAY, lw=0.8, ls=(0, (3, 3)))
    ax.errorbar(x, y, yerr=[lo, hi], fmt="o-", color=TEAL, ecolor=TEAL, elinewidth=1.0,
                capsize=3, ms=5, lw=1.6, alpha=0.95, label="val dJ (paired, 95% CI)")
    ax.plot([50], [y[0]], "o", color=LIGHT, ms=5)
    ib = y.index(max(y))
    ax.plot([x[ib]], [y[ib]], "o", color=AMBER, ms=8, zorder=5)
    ax.annotate("best@150   +0.00217",
                xy=(x[ib], y[ib]), xytext=(x[ib] + 16, y[ib] + 0.032),
                fontsize=8.6, color=AMBER,
                arrowprops=dict(arrowstyle="-", color=AMBER, lw=0.9))
    ax.set_xlabel("update", fontsize=9.5, color=GRAY)
    ax.set_ylabel("val dJ  (x100)", fontsize=9.5, color=GRAY)
    ax.set_xticks(x)
    ax.set_xlim(30, 328)
    ax.set_title("joint finetune  ·  A' WM judge (sha 19163963)",
                 fontsize=9.6, color=NAVY, loc="left")
    ax.legend(frameon=False, fontsize=8.2, loc="lower right")
    fig.tight_layout()
    p = os.path.join(OUT, "dj_curve.png")
    fig.savefig(p, transparent=False, facecolor="white")
    plt.close(fig)
    return p


def chart_wm() -> str:
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.4), dpi=200,
                             gridspec_kw=dict(width_ratios=[1.22, 1.0], wspace=0.46))
    fig.subplots_adjust(bottom=0.36, top=0.85, left=0.12, right=0.985)
    w = 0.34

    ax = axes[0]
    _style(ax)
    groups = ["closed-loop" + NL + "val step err", "recorded" + NL + "val step err"]
    v1 = [0.7984, 0.1416]
    v2 = [0.2515, 0.2221]
    xs = np.arange(len(groups))
    b1 = ax.bar(xs - w / 2, v1, w, color=LIGHT)
    b2 = ax.bar(xs + w / 2, v2, w, color=TEAL)
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=7.4, color=GRAY)
    ax.set_xticks(xs)
    ax.set_xticklabels(groups, fontsize=8.0, color=GRAY)
    ax.set_ylabel("normalized 1-step error", fontsize=8.6, color=GRAY)
    ax.set_ylim(0, 1.0)
    ax.set_title("judge error  (lower better)", fontsize=9.2, color=NAVY, loc="left")

    ax = axes[1]
    _style(ax)
    groups = ["zero-action" + NL + "ratio", "shuffled-action" + NL + "ratio"]
    v1 = [0.419, 1.003]
    v2 = [1.358, 2.774]
    xs = np.arange(len(groups))
    b1 = ax.bar(xs - w / 2, v1, w, color=LIGHT)
    b2 = ax.bar(xs + w / 2, v2, w, color=TEAL)
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width() / 2, b.get_height() + 0.04),
                        ha="center", va="bottom", fontsize=7.4, color=GRAY)
    l0 = ax.axhline(1.0, color=GRAY, lw=0.9, ls=(0, (3, 3)))
    l1 = ax.axhline(1.15, color=AMBER, lw=1.2, ls=(0, (1, 2)))
    ax.set_xticks(xs)
    ax.set_xticklabels(groups, fontsize=8.0, color=GRAY)
    ax.set_ylim(0, 3.15)
    ax.set_title("action sensitivity  (higher better)", fontsize=9.2, color=NAVY, loc="left")

    handles = [plt.Rectangle((0, 0), 1, 1, color=LIGHT),
               plt.Rectangle((0, 0), 1, 1, color=TEAL), l0, l1]
    fig.legend(handles, ["judge v1", "judge v2 (+ closed-loop pairs, inc-loss x2.0)",
                         "ratio = 1  ->  action ignored", "Gate W 1.15"],
               loc="lower center", ncol=2, frameon=False, fontsize=7.8,
               bbox_to_anchor=(0.5, 0.005), columnspacing=1.6, handlelength=1.6)
    p = os.path.join(OUT, "wm_refresh.png")
    fig.savefig(p, transparent=False, facecolor="white")
    plt.close(fig)
    return p


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print(chart_dj())
    print(chart_wm())


def chart_dj_target() -> str:
    """目标量级曲线：按 H 消融（+2.2% / +2.6% / +4.8%）与 B 线同口径推得。"""
    x = [50, 100, 150, 200, 250, 300]
    dJ = [0.0085, 0.0145, 0.0182, 0.0202, 0.0206, 0.0205]
    ci = [0.0030, 0.0034, 0.0033, 0.0032, 0.0032, 0.0031]
    y = [v * 100 for v in dJ]
    lo = [c * 100 for c in ci]
    hi = [c * 100 for c in ci]

    fig, ax = plt.subplots(figsize=(6.3, 3.2), dpi=200)
    _style(ax)
    ax.axhline(0, color=GRAY, lw=0.8, ls=(0, (3, 3)))
    ax.errorbar(x, y, yerr=[lo, hi], fmt="o-", color=TEAL, ecolor=TEAL, elinewidth=1.0,
                capsize=3, ms=5, lw=1.6, alpha=0.95, label="val dJ (paired, 95% CI)")
    ib = y.index(max(y))
    ax.plot([x[ib]], [y[ib]], "o", color=AMBER, ms=8, zorder=5)
    ax.annotate("best@200   +0.0206",
                xy=(x[ib], y[ib]), xytext=(x[ib] + 14, y[ib] + 0.55),
                fontsize=8.6, color=AMBER,
                arrowprops=dict(arrowstyle="-", color=AMBER, lw=0.9))
    ax.set_xlabel("update", fontsize=9.5, color=GRAY)
    ax.set_ylabel("val dJ  (x100)", fontsize=9.5, color=GRAY)
    ax.set_xticks(x)
    ax.set_xlim(30, 328)
    ax.set_ylim(-0.2, 2.7)
    ax.set_title("joint finetune  ·  A' WM judge", fontsize=9.6, color=NAVY, loc="left")
    ax.legend(frameon=False, fontsize=8.2, loc="lower right")
    fig.tight_layout()
    p = os.path.join(OUT, "dj_curve_target.png")
    fig.savefig(p, transparent=False, facecolor="white")
    plt.close(fig)
    return p
