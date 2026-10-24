#!/usr/bin/env python3
"""Policy server for K1/K RTC inference (zero-shot, no retraining).

Two endpoints:

* ``get_action``     -- one committed plan (K=1) or the fixed/gated selection over K
                        candidates; accepts ``options = {"anchor_tick": int, "rtc": {...}}``.
                        ``PolicyClient._get_action`` already forwards ``options``.
* ``get_candidates`` -- only registered when ``--k > 1``: returns ``[K, 40, 78]``
                        decoded candidates plus ``[K, 40, 132]`` normalized rows for
                        the caller to commit after its own arbitration.

The server never commits anything itself; the deployment loop owns the committed
plan and passes its tail back through ``options["rtc"]["previous_action"]``.

Example (K=1 RTC)::

    python -m mosaic_n17.tools.run_rtc_server --ckpt runs/.../checkpoint-20000 --port 5555

Example (K=4, fixed selection)::

    python -m mosaic_n17.tools.run_rtc_server --ckpt runs/.../checkpoint-20000 --k 4

``--k-select gate`` requires a calibrated AC scorer and is rejected here on
purpose: a random or placeholder ranking would silently change commands.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--ckpt", required=True, help="offline-loadable GR00T checkpoint path")
    parser.add_argument("--embodiment-tag", default="unitree_g1_sonic")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="*")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--api-token", default=None)
    parser.add_argument("--k", type=int, default=1, help="candidates per plan (K>=1)")
    parser.add_argument(
        "--k-select",
        choices=("first", "gate"),
        default="first",
        help="'first' = fixed primary candidate (no scorer); 'gate' needs a calibrated AC scorer",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.k < 1:
        raise SystemExit("--k must be >= 1")
    if args.k > 1 and args.k_select == "gate":
        raise SystemExit(
            "--k-select gate needs a calibrated AC scorer; this server only does fixed selection"
        )
    if args.k == 1 and args.k_select != "first":
        raise SystemExit("--k-select only applies to --k > 1")


def build_policy(args: argparse.Namespace):
    from mosaic_n17.policy.rtc_policy import RtcGr00tPolicy
    import torch

    torch.manual_seed(0)
    policy = RtcGr00tPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=args.ckpt,
        device=args.device,
        strict=True,
    )
    if args.k == 1:
        return policy

    from mosaic_n17.policy.policy_wrapper import MosaicPolicyWrapper

    return MosaicPolicyWrapper(policy, K=args.k, selection="first", shadow=False)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    validate_args(args)
    policy = build_policy(args)

    from gr00t.policy.server_client import PolicyServer

    with PolicyServer(policy, host=args.host, port=args.port, api_token=args.api_token) as server:
        if args.k > 1:
            server.register_endpoint("get_candidates", policy.get_candidates)
        server.run()


if __name__ == "__main__":
    main()
