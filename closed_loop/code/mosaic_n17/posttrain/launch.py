"""Post-training launcher: mix demo and verified-teacher data, then fine-tune.

N1.7's FM loss has no per-sample weight, so ``L_demo + lambda * L_teacher`` is
expressed as two dataset specs whose sampling weights are
``relative_length * mix_ratio`` (``gr00t/data/dataset/factory.py:71-76``).  This
module only builds and validates that config: it never edits the trainer, the
loss, or the processed datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

import yaml


LAUNCH_SCRIPT = "gr00t/experiment/launch_train.py"
CONFIG_SCHEMA = "mosaic.n17.posttrain_config.v1"


@dataclass(frozen=True)
class PosttrainConfig:
    base_model_path: str
    demo_roots: tuple[str, ...]
    teacher_roots: tuple[str, ...]
    output_dir: str
    teacher_mix_ratio: float = 0.2
    max_steps: int = 3000
    global_batch_size: int = 32
    save_steps: int = 500
    learning_rate: float = 1e-4
    state_dropout_prob: float = 0.2
    tune_llm: bool = False
    tune_visual: bool = False
    tune_top_llm_layers: int = 0
    embodiment_tag: str = "unitree_g1_sonic"
    schema: str = CONFIG_SCHEMA
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema != CONFIG_SCHEMA:
            raise ValueError(f"unsupported post-training schema {self.schema!r}")
        if not self.base_model_path:
            raise ValueError("base_model_path is required")
        if not self.demo_roots:
            raise ValueError("at least one demo root is required")
        if not self.teacher_roots:
            raise ValueError("at least one teacher root is required")
        if not 0.0 < self.teacher_mix_ratio:
            raise ValueError("teacher_mix_ratio must be positive")
        if self.tune_llm and self.tune_top_llm_layers:
            raise ValueError("choose either full tune_llm or tune_top_llm_layers, not both")
        for name in ("max_steps", "global_batch_size", "save_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive int")
        if self.tune_top_llm_layers < 0:
            raise ValueError("tune_top_llm_layers must be non-negative")

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "data": {
                "datasets": [
                    {
                        "dataset_paths": list(self.demo_roots),
                        "embodiment_tag": self.embodiment_tag,
                        "mix_ratio": 1.0,
                    },
                    {
                        "dataset_paths": list(self.teacher_roots),
                        "embodiment_tag": self.embodiment_tag,
                        "mix_ratio": float(self.teacher_mix_ratio),
                    },
                ]
            },
            "model": {
                "tune_llm": self.tune_llm,
                "tune_visual": self.tune_visual,
                "tune_projector": True,
                "tune_diffusion_model": True,
                "tune_vlln": True,
                "tune_top_llm_layers": self.tune_top_llm_layers,
                "state_dropout_prob": self.state_dropout_prob,
            },
            "training": {
                "start_from_checkpoint": self.base_model_path,
                "output_dir": self.output_dir,
                "max_steps": self.max_steps,
                "global_batch_size": self.global_batch_size,
                "save_steps": self.save_steps,
                "learning_rate": self.learning_rate,
                "warmup_ratio": 0.05,
                "use_wandb": False,
            },
            "extra": self.extra,
        }


def posttrain_config_from_dict(payload: dict) -> PosttrainConfig:
    """Build a :class:`PosttrainConfig` from a parsed YAML mapping."""

    if not isinstance(payload, dict):
        raise TypeError("post-training config must be a mapping")
    data = payload.get("data") or {}
    specs = data.get("datasets")
    if not isinstance(specs, list) or len(specs) < 2:
        raise ValueError("data.datasets must contain a demo spec followed by a teacher spec")
    demo_roots = [path for spec in specs[:-1] for path in (spec.get("dataset_paths") or [])]
    teacher_roots = list(specs[-1].get("dataset_paths") or [])
    if not demo_roots or not teacher_roots:
        raise ValueError("demo and teacher specs must both carry dataset_paths")
    teacher_mix = float(specs[-1].get("mix_ratio", 0.2))
    model = payload.get("model") or {}
    training = payload.get("training") or {}
    return PosttrainConfig(
        base_model_path=str(training.get("start_from_checkpoint", "")),
        demo_roots=tuple(demo_roots),
        teacher_roots=tuple(teacher_roots),
        output_dir=str(training.get("output_dir", "")),
        teacher_mix_ratio=float(teacher_mix if teacher_mix is not None else 0.2),
        max_steps=int(training.get("max_steps", 3000)),
        global_batch_size=int(training.get("global_batch_size", 32)),
        save_steps=int(training.get("save_steps", 500)),
        learning_rate=float(training.get("learning_rate", 1e-4)),
        state_dropout_prob=float(model.get("state_dropout_prob", 0.2)),
        tune_llm=bool(model.get("tune_llm", False)),
        tune_visual=bool(model.get("tune_visual", False)),
        tune_top_llm_layers=int(model.get("tune_top_llm_layers", 0)),
    )


def load_posttrain_config(path: str | os.PathLike[str]) -> PosttrainConfig:
    return posttrain_config_from_dict(yaml.safe_load(Path(path).read_text()))


def write_config(path: str | os.PathLike[str], config: PosttrainConfig) -> Path:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
    return target


def build_command(config_path: str | os.PathLike[str], *, python: str = "python") -> list[str]:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return [python, LAUNCH_SCRIPT, "--load-config-path", str(path)]


def launch(
    config_path: str | os.PathLike[str],
    *,
    python: str = "python",
    dry_run: bool = True,
    log_path: str | os.PathLike[str] | None = None,
):
    """Run the post-training entry point (``dry_run`` only returns the command)."""

    import subprocess

    command = build_command(config_path, python=python)
    if dry_run:
        return command
    if log_path is None:
        raise ValueError("a log_path is required for a real launch")
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    return completed


__all__ = [
    "CONFIG_SCHEMA",
    "LAUNCH_SCRIPT",
    "PosttrainConfig",
    "build_command",
    "launch",
    "load_posttrain_config",
    "posttrain_config_from_dict",
    "write_config",
]
