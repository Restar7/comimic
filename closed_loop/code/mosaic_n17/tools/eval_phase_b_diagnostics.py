"""Re-run Phase B consequence/candidate-ranking diagnostics on a saved checkpoint.

``train_phase_b`` only reports diagnostics for the in-memory model at the end of training, and
the report then reflects whatever diagnostics code was imported when the process started.  This
driver loads a checkpoint plus its source manifest and produces the same report shape, including
the candidate-ranking metrics, so a finished run can be scored without retraining.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ..ac.checkpoint import load_phase_b_checkpoint
from ..ac.phase_b_runtime import PhaseBConfig, Sonic78PhaseB
from ..eval.phase_b_diagnostics import DiagnosticsConfig, evaluate_phase_b
from ..train.train_phase_b import build_split, load_source_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Defaults to <checkpoint>.diagnostics.json")
    parser.add_argument("--split", choices=("val", "train", "all"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--ranking-candidates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing report; without it the driver refuses to clobber run artifacts",
    )
    args = parser.parse_args()

    payload = load_phase_b_checkpoint(args.checkpoint)
    model = Sonic78PhaseB(PhaseBConfig(**payload["config"])).to(args.device)
    model.load_state_dict(payload["model"])
    model.eval()
    phase_a_sha256 = payload["contract"]["phase_a_sha256"]
    sources = load_source_manifest(args.sources)
    splits = ("train", "val") if args.split == "all" else (args.split,)

    model_config = PhaseBConfig(**payload["config"])
    config = DiagnosticsConfig(
        stride=args.stride,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
        seed=args.seed,
        ranking_candidates=args.ranking_candidates,
        action_horizon=model_config.control_horizon,
    )
    reports: dict[str, object] = {}
    for split in splits:
        episodes = build_split(sources, split, phase_a_sha256=phase_a_sha256)
        if not episodes:
            continue
        reports[split] = evaluate_phase_b(model, episodes, config=config, device=args.device)

    output = args.output or args.checkpoint.with_suffix(".diagnostics.json")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"diagnostics report already exists: {output} (pass --overwrite)")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema": "mosaic.n17.phase_b_diagnostics_report.v1",
                "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                "phase_a_sha256": phase_a_sha256,
                "splits": reports,
            },
            indent=1,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "verdicts": {split: report["verdict"] for split, report in reports.items()},
                "ranking": {
                    split: {
                        key: report["overall"][key]
                        for key in (
                            "ranking_candidates",
                            "ranking_top1_accuracy",
                            "ranking_mean_rank",
                            "ranking_mrr",
                            "ranking_random_top1",
                        )
                    }
                    for split, report in reports.items()
                },
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
