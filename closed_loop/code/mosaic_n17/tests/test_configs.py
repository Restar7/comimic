"""Configuration tests: the shipped YAMLs must parse into validated objects."""

from __future__ import annotations

from pathlib import Path

import yaml

from mosaic_n17.policy.decision_gate import GateConfig
from mosaic_n17.posttrain import load_posttrain_config


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_k_ode_config_parses_gate() -> None:
    payload = yaml.safe_load((CONFIG_DIR / "k_ode.yaml").read_text())
    assert payload["schema"] == "mosaic.n17.k_ode_config.v1"
    assert payload["sampling"]["K"] >= 1
    assert payload["sampling"]["seeds"] is None
    gate = GateConfig.from_mapping(payload["gate"])
    assert gate.motion_abs_bound == 1.25
    assert gate.active_terms() == ("stage_gain", "support", "uncertainty")
    assert payload["budget"]["replan_hz"] == 2.5


def test_k_ode_gate_rejects_unknown_keys() -> None:
    import pytest

    with pytest.raises(ValueError, match="unknown gate config keys"):
        GateConfig.from_mapping({"motion_abs_bound": 1.25, "token_velocity_limit": 1.0})


def test_posttrain_config_parses_two_specs() -> None:
    config = load_posttrain_config(CONFIG_DIR / "posttrain_teacher78.yaml")
    assert len(config.demo_roots) == 9
    assert len(config.teacher_roots) == 1
    assert config.teacher_mix_ratio == 0.2
    assert config.tune_llm is False and config.tune_top_llm_layers == 0
    payload = config.to_dict()
    specs = payload["data"]["datasets"]
    assert len(specs) == 2 and specs[0]["mix_ratio"] == 1.0


def test_posttrain_config_requires_teacher_spec() -> None:
    import pytest

    from mosaic_n17.posttrain import posttrain_config_from_dict

    with pytest.raises(ValueError, match="demo spec followed by a teacher spec"):
        posttrain_config_from_dict(
            {
                "data": {
                    "datasets": [{"dataset_paths": ["/a"], "embodiment_tag": "unitree_g1_sonic"}]
                }
            }
        )
