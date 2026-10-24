"""Candidate sampling, decoding, arbitration and shadow execution (M1/M4)."""

from __future__ import annotations

from .candidate_sampler import CandidateSet, sample_candidates, sample_from_features
from .decision_gate import GateConfig, GateDecision, decide
from .decode import assemble_sonic78, decode_candidates, decode_group_dict
from .policy_wrapper import MosaicPolicyWrapper, ScoreFn
from .rtc import RtcRequest, RtcWindow, encode_committed_action, prefix_equal
from .shadow import ShadowRecord, run_shadow_step


__all__ = [
    "CandidateSet",
    "MosaicPolicyWrapper",
    "RtcRequest",
    "RtcWindow",
    "ScoreFn",
    "GateConfig",
    "GateDecision",
    "ShadowRecord",
    "assemble_sonic78",
    "decide",
    "decode_candidates",
    "decode_group_dict",
    "encode_committed_action",
    "prefix_equal",
    "run_shadow_step",
    "sample_candidates",
    "sample_from_features",
]
