#!/usr/bin/env python3
"""Run the flow-matching gradient ablation and write a markdown/JSON report.

Examples:
    python -m ablation_fm_grad.run_ablation --mode toy  --steps 2 4 8 --train-steps 30
    python -m ablation_fm_grad.run_ablation --mode tiny --steps 4     --train-steps 10
    python -m ablation_fm_grad.run_ablation --mode scale --isolate --scale-preset large --steps 4
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from .harness import evaluate, rows_to_markdown


def _tag(args: argparse.Namespace, suffix: str = "") -> str:
    tag = f"{args.mode}_" + "-".join(str(s) for s in args.steps)
    if args.mode == "scale":
        tag += f"_{args.scale_preset}"
    if suffix:
        tag += f"_{suffix}"
    return tag


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("toy", "tiny", "scale"), default="toy",
                    help="toy=MLP; tiny=real GR00T head, tiny dims; scale=real GR00T head, H=40")
    ap.add_argument("--steps", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--backends", nargs="+", default=["pathwise", "checkpointed", "truncated", "adjoint", "qvgm_lite"])
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--train-steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--qvgm-critic", choices=("oracle", "learned"), default="oracle",
                    help="oracle = Q(a)=-task(a) (mechanism check); learned = MLP regressed from rollouts")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale-preset", choices=("small", "large"), default="small",
                    help="only for --mode scale: large = H40/hidden512/8 layers/batch4")
    ap.add_argument("--isolate", action="store_true",
                    help="run each backend in a fresh process (clean CUDA peak-memory numbers)")
    ap.add_argument("--single", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    args = ap.parse_args()

    # ------------------------------------------------------------------ isolated
    if args.isolate and args.single is None:
        import subprocess
        import sys

        tmp = args.out / "_isolate"
        tmp.mkdir(parents=True, exist_ok=True)
        for stale in tmp.glob("*.json"):
            stale.unlink()
        for backend in args.backends:
            cmd = [
                sys.executable, "-m", "ablation_fm_grad.run_ablation",
                "--mode", args.mode,
                "--steps", *[str(v) for v in args.steps],
                "--backends", backend,
                "--trials", str(args.trials),
                "--train-steps", str(args.train_steps),
                "--lr", str(args.lr),
                "--device", args.device,
                "--seed", str(args.seed),
                "--qvgm-critic", args.qvgm_critic,
                "--scale-preset", args.scale_preset,
                "--single", backend,
                "--out", str(tmp),
            ]
            subprocess.run(cmd, check=True)
        rows_all: list[dict] = []
        for f in sorted(tmp.glob("*.json")):
            rows_all.extend(json.loads(f.read_text()))
        args.out.mkdir(parents=True, exist_ok=True)
        tag = _tag(args)
        (args.out / f"{tag}.json").write_text(json.dumps(rows_all, indent=2))
        from .harness import Row

        md = f"# FM gradient ablation — {args.mode} (isolated)\n\nK={args.steps}, device={args.device}\n\n"
        md += rows_to_markdown([Row(**r) for r in rows_all]) + "\n"
        (args.out / f"{tag}.md").write_text(md)
        print(md)
        print(f"[saved] {args.out / (tag + '.md')}")
        return 0

    # ------------------------------------------------------------------ normal
    rows = evaluate(
        mode=args.mode,
        steps_list=args.steps,
        backends=args.backends,
        trials=args.trials,
        train_steps=args.train_steps,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        qvgm_critic=args.qvgm_critic,
        scale_preset=args.scale_preset,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    tag = _tag(args, args.single or "")
    (args.out / f"{tag}.json").write_text(json.dumps([asdict(r) for r in rows], indent=2))
    md = (
        f"# FM gradient ablation — {args.mode}\n\n"
        f"K={args.steps}, train_steps={args.train_steps}, lr={args.lr}, "
        f"device={args.device}, qvgm_critic={args.qvgm_critic}\n\n"
    )
    md += rows_to_markdown(rows) + "\n"
    (args.out / f"{tag}.md").write_text(md)
    print(md)
    print(f"[saved] {args.out / (tag + '.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
