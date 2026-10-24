"""L2 "VLM trained by the world model": latent consequence auxiliary head (M6).

The official N1.7 forward has no auxiliary-loss hook and its batch schema is
fixed, so this objective gets its own training entry instead of patching the
trainer:

* the N1.7 backbone is frozen and only produces ``backbone_features``;
* a small :class:`AuxSlotHead` reads those features plus the executed action
  chunk and predicts the *future typed slots* of the frozen Phase A;
* targets are stop-grad and come from the precomputed slot store.

The head is a side artefact: inference does not load it, so a regression here
cannot change control behaviour.  Promotion requires the downstream action
metrics, not the auxiliary loss alone.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..ac.slots import SLOT_DIM, TOTAL_SLOTS, masked_huber
from ..contracts.sonic78 import ACTION78_DIM, SONIC78_V1
from ..data.slots_h5 import read_slots_h5


AUX_SCHEMA = "mosaic.n17.vlm_world_model_head.v1"
ACTION_KEYS = ("motion_token", "left_hand_joints", "right_hand_joints")


class AuxSlotHead(nn.Module):
    """``(backbone_tokens, action[H,78]) -> future typed slots [14,384]``."""

    def __init__(
        self,
        *,
        token_dim: int = 2048,
        hidden_dim: int = 512,
        slot_dim: int = SLOT_DIM,
        horizon: int = SONIC78_V1.horizon,
        num_heads: int = 8,
        depth: int = 1,
    ) -> None:
        super().__init__()
        if horizon != SONIC78_V1.horizon:
            raise ValueError(f"aux head horizon must be {SONIC78_V1.horizon}")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.token_proj = nn.Linear(token_dim, hidden_dim)
        self.action_proj = nn.Linear(ACTION78_DIM, hidden_dim)
        self.step_embedding = nn.Embedding(horizon, hidden_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=depth)
        self.query = nn.Parameter(torch.zeros(TOTAL_SLOTS, hidden_dim))
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, slot_dim)

    def forward(self, backbone_features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if backbone_features.ndim != 3:
            raise ValueError(
                f"backbone features must be [B,S,D], got {tuple(backbone_features.shape)}"
            )
        if actions.ndim != 3 or actions.shape[-1] != ACTION78_DIM:
            raise ValueError(f"actions must be [B,H,{ACTION78_DIM}], got {tuple(actions.shape)}")
        if backbone_features.shape[0] != actions.shape[0]:
            raise ValueError("backbone features and actions must share a batch size")
        if actions.shape[1] != SONIC78_V1.horizon:
            raise ValueError(f"actions must carry H={SONIC78_V1.horizon}, got {actions.shape[1]}")
        memory = self.token_proj(backbone_features)
        steps = torch.arange(actions.shape[1], device=actions.device)
        action_tokens = self.action_proj(actions) + self.step_embedding(steps).unsqueeze(0)
        memory = torch.cat((memory, action_tokens), dim=1)
        queries = self.query.unsqueeze(0).expand(actions.shape[0], -1, -1)
        decoded = self.decoder(queries, memory)
        return self.output_proj(self.output_norm(queries + decoded))


@dataclass(frozen=True)
class AuxTrainingConfig:
    checkpoint_path: str
    dataset_root: str
    slots_dir: str
    output_path: str
    phase_a_sha256: str
    max_steps: int = 200
    batch_size: int = 2
    learning_rate: float = 3e-4
    supervision_step: int = SONIC78_V1.execute_steps
    device: str = "cuda"

    def __post_init__(self) -> None:
        if len(self.phase_a_sha256) != 64:
            raise ValueError("phase_a_sha256 must be a 64-char digest")
        if not 1 <= self.supervision_step <= SONIC78_V1.execute_steps:
            raise ValueError(
                f"supervision_step must be in [1,{SONIC78_V1.execute_steps}] "
                "(only the executed prefix is observed)"
            )


def action_chunk_from_step(step_data) -> np.ndarray:
    """Assemble the recorded ``[H,78]`` chunk from one ``VLAStepData``."""

    missing = [key for key in ACTION_KEYS if key not in step_data.actions]
    if missing:
        raise ValueError(f"step data is missing action groups {missing}")
    group = [np.asarray(step_data.actions[key], dtype=np.float64) for key in ACTION_KEYS]
    chunk = np.concatenate(group, axis=-1)
    SONIC78_V1.validate_block(chunk[None, ...])
    return chunk


def train(config: AuxTrainingConfig) -> dict:
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import MessageType
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
    from transformers import AutoProcessor

    dataset_root = Path(config.dataset_root)
    slots_dir = Path(config.slots_dir)
    processor_dir = Path(config.checkpoint_path)
    if (
        not (processor_dir / "processor_config.json").exists()
        and not (processor_dir / "processor").is_dir()
    ):
        raise FileNotFoundError(
            f"{processor_dir} has no processor_config.json or processor/ subdir"
        )
    processor = AutoProcessor.from_pretrained(
        processor_dir
        if (processor_dir / "processor_config.json").exists()
        else processor_dir / "processor"
    )
    model = Gr00tN1d7.from_pretrained(config.checkpoint_path).to(config.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head = AuxSlotHead().to(config.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.learning_rate)
    loader = LeRobotEpisodeLoader(
        dataset_root, MODALITY_CONFIGS[EmbodimentTag.UNITREE_G1_SONIC.value]
    )
    rng = np.random.default_rng(0)
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    episodes = [int(i) for i in range(int(info["total_episodes"]))]
    history: list[float] = []
    processor.eval()
    modality_configs = MODALITY_CONFIGS[EmbodimentTag.UNITREE_G1_SONIC.value]
    embodiment = EmbodimentTag.UNITREE_G1_SONIC
    history: list[float] = []
    for step in range(config.max_steps):
        features, action_chunks, targets = [], [], []
        attempts = 0
        while len(features) < config.batch_size:
            attempts += 1
            if attempts > config.batch_size * 100:
                raise RuntimeError("could not assemble a batch from valid slot targets")
            episode = episodes[int(rng.integers(len(episodes)))]
            frame_data = loader[episode]
            length = len(frame_data)
            slots_path = slots_dir / f"episode_{episode:06d}.slots.h5"
            slot_payload = read_slots_h5(slots_path)
            slots, valid = slot_payload["slots"], slot_payload["valid"]
            if slots.shape[0] != length:
                raise ValueError(
                    f"slots for episode {episode} cover {slots.shape[0]} steps but the "
                    f"LeRobot episode has {length}"
                )
            t = int(rng.integers(0, length - SONIC78_V1.horizon))
            target = t + config.supervision_step
            if target >= length or not valid[target]:
                continue
            step_data = extract_step_data(frame_data, t, modality_configs, embodiment, False)
            processed = processor([{"type": MessageType.EPISODE_STEP.value, "content": step_data}])
            batch = processor.collator([processed])["inputs"]
            with torch.no_grad():
                backbone_inputs, _ = model.prepare_input(dict(batch))
                backbone_output = model.backbone(backbone_inputs)
            features.append(backbone_output["backbone_features"].float())
            action_chunks.append(torch.from_numpy(action_chunk_from_step(step_data)).float())
            targets.append(torch.from_numpy(slots[target]).float())

        batch_features = torch.cat(features, dim=0).to(config.device)
        batch_actions = torch.stack(action_chunks).to(config.device)
        batch_targets = torch.stack(targets).to(config.device)
        predicted = head(batch_features, batch_actions)
        mask = torch.ones_like(batch_targets, dtype=torch.bool)
        loss = masked_huber(predicted, batch_targets.detach(), mask)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite aux loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        history.append(float(loss.detach().cpu()))

    payload = {
        "schema": AUX_SCHEMA,
        "phase_a_sha256": config.phase_a_sha256,
        "checkpoint": config.checkpoint_path,
        "dataset_root": config.dataset_root,
        "slots_dir": config.slots_dir,
        "supervision_step": config.supervision_step,
        "steps": config.max_steps,
        "loss_history": history,
        "head": head.state_dict(),
    }
    output = Path(config.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return {"checkpoint_sha256": digest, "final_loss": history[-1], "steps": config.max_steps}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--slots-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase-a-sha256", required=True)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    summary = train(
        AuxTrainingConfig(
            checkpoint_path=args.checkpoint_path,
            dataset_root=args.dataset_root,
            slots_dir=args.slots_dir,
            output_path=args.output,
            phase_a_sha256=args.phase_a_sha256,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
        )
    )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
