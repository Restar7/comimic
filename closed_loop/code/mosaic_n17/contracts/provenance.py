"""Provenance tags and the record that binds an action to its generator.

Two different actions can look identical and mean different things: a recorded
teleop token, a token re-encoded from a reference motion, and a token produced
by an offline encoder from measured joints are not interchangeable supervision.
The tag travels with the action, and post-training refuses anything that is not
in :data:`VALID_PROVENANCE`.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


#: Tags accepted as supervision / deployable commands.
VALID_PROVENANCE = frozenset(
    {
        "recorded_token",
        "encoder_reference",
        "measured_motion_encoded",
        "selected_teacher",
    }
)
#: Never valid: a zero/filled placeholder must never become a label.
PLACEHOLDER_PROVENANCE = "placeholder"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def validate_provenance(tag: object, *, require: bool = True) -> str:
    """Return ``tag`` when admissible, otherwise raise."""

    if tag is None:
        if require:
            raise ValueError("provenance is required")
        return ""
    if not isinstance(tag, str):
        raise TypeError(f"provenance must be a string, got {type(tag).__name__}")
    if tag == PLACEHOLDER_PROVENANCE:
        raise ValueError("placeholder provenance is never admissible")
    if require and tag not in VALID_PROVENANCE:
        raise ValueError(f"provenance {tag!r} is not one of {sorted(VALID_PROVENANCE)}")
    return tag


def _optional_sha256(value: object, label: str) -> str:
    if value is None or value == "":
        return ""
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase sha256 hex digest")
    return value


@dataclass(frozen=True)
class ProvenanceRecord:
    """Everything needed to reproduce and audit one teacher/action record."""

    candidate_id: str
    provenance: str
    generator_sha256: str
    processor_sha256: str
    stats_sha256: str
    ac_sha256: str = ""
    stage_sha256: str = ""
    sim_receipt_sha256: str = ""
    bank_generation: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id must be a non-empty string")
        validate_provenance(self.provenance)
        for field in (
            "generator_sha256",
            "processor_sha256",
            "stats_sha256",
        ):
            _optional_sha256(getattr(self, field), field)
        for field in ("ac_sha256", "stage_sha256", "sim_receipt_sha256"):
            _optional_sha256(getattr(self, field), field)
        if isinstance(self.bank_generation, bool) or not isinstance(self.bank_generation, int):
            raise TypeError("bank_generation must be an int")
        if self.bank_generation < 0:
            raise ValueError("bank_generation must be non-negative")

    @property
    def is_sim_verified(self) -> bool:
        """A teacher is verified only with a bound simulation receipt."""

        return self.provenance == "selected_teacher" and bool(self.sim_receipt_sha256)


__all__ = [
    "PLACEHOLDER_PROVENANCE",
    "ProvenanceRecord",
    "VALID_PROVENANCE",
    "validate_provenance",
]
