"""``Gr00tPolicy`` + zero-shot RTC, without touching the vendored upstream tree.

Why a subclass instead of patching ``gr00t/policy/gr00t_policy.py``: the
migration plan (``docs/projects/mosaic-n17-migration-plan-20261022.md``) keeps
``gr00t/`` free of local edits, and the official ``_get_action`` drops
``options`` at ``self.model.get_action(**collated_inputs)`` (line 417) and never
puts an ``action`` field into the model input (inference has no ground-truth
action, ``actions={}`` in ``_to_vla_step_data``).  Re-implementing that one
method here keeps upstream diff-free while ``BasePolicy.get_action`` still runs
``check_observation`` / ``check_action``.

Only ``UNITREE_G1_SONIC`` accepts an RTC request; a request for any other
embodiment raises instead of silently falling back.
"""

from __future__ import annotations

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy, MessageType, _rec_to_dtype
import numpy as np
import torch

from .rtc import RtcRequest, validate_prefix_tensor


__all__ = ["RtcGr00tPolicy"]


class RtcGr00tPolicy(Gr00tPolicy):
    """``Gr00tPolicy`` whose ``_get_action`` honours ``options["rtc"]``.

    The request contract (built by the deployment loop, never by the model)::

        options = {
            "anchor_tick": int,  # 50 Hz control tick, log/validation only
            "rtc": {
                "previous_action": [B, 40, 132],  # normalized + zero padded
                "horizon": 40,
                "overlap": O,
                "frozen": F,
                "ramp_rate": 6.0,
                "action_horizon": 40,
            },
        }

    The returned ``info["action_normalized"]`` is the sampler's own
    ``[B, H, max_action_dim]`` output; the caller must commit **that** tensor
    instead of re-encoding the decoded 78D block (a float round trip would break
    the frozen-prefix equality check).
    """

    def _get_action(  # noqa: C901 - mirrors upstream control flow on purpose
        self, observation: dict, options: dict | None = None
    ):
        request = self._rtc_request(options)
        if request is None:
            return super()._get_action(observation, options)

        unbatched = self._unbatch_observation(observation)
        processed, states = [], []
        for obs in unbatched:
            step = self._to_vla_step_data(obs)
            states.append(step.states)
            processed.append(
                self.processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
            )
        collated = _rec_to_dtype(self.collate_fn(processed), dtype=torch.bfloat16)["inputs"]

        action_dim = int(self.model.config.max_action_dim)
        prefix = validate_prefix_tensor(
            request.previous_action, request.window, action_dim=action_dim
        )
        if prefix.shape[0] != len(unbatched):
            raise ValueError(
                f"prefix batch {prefix.shape[0]} != observation batch {len(unbatched)}; "
                "the committed chunk must belong to this observation"
            )
        collated["action"] = torch.from_numpy(prefix).to(
            device=self.model.device, dtype=self.model.dtype
        )

        with torch.inference_mode():
            model_pred = self.model.get_action(
                inputs=collated, options=request.window.to_model_options()
            )
        normalized = model_pred["action_pred"].float()
        if tuple(normalized.shape) != (len(unbatched), request.window.horizon, action_dim):
            raise ValueError(
                f"model returned normalized actions {tuple(normalized.shape)}, expected "
                f"{(len(unbatched), request.window.horizon, action_dim)}"
            )
        if not bool(torch.isfinite(normalized).all()):
            raise ValueError("model produced non-finite actions")

        batched_states = {
            key: np.stack([state[key] for state in states], axis=0)
            for key in self.modality_configs["state"].modality_keys
        }
        decoded = self.processor.decode_action(
            normalized.cpu().numpy(), self.embodiment_tag, batched_states
        )
        casted = {key: np.asarray(value, dtype=np.float32) for key, value in decoded.items()}
        return casted, {"action_normalized": normalized.cpu().numpy()}

    def _rtc_request(self, options: dict | None) -> RtcRequest | None:
        """Resolve and validate ``options``; ``None`` means "plain upstream call"."""

        if options is None:
            return None
        unknown = set(options) - {"rtc", "anchor_tick"}
        if unknown:
            raise ValueError(f"unsupported policy options {sorted(unknown)}")
        anchor = options.get("anchor_tick")
        if anchor is not None and (
            isinstance(anchor, bool) or not isinstance(anchor, int) or anchor < 0
        ):
            raise ValueError(f"anchor_tick must be a non-negative int, got {anchor!r}")
        raw_request = options.get("rtc")
        if raw_request is None:
            return None
        if self.embodiment_tag is not EmbodimentTag.UNITREE_G1_SONIC:
            raise ValueError(
                f"RTC is only wired for UNITREE_G1_SONIC, got {self.embodiment_tag.name}"
            )
        request = RtcRequest.from_mapping(raw_request)
        request.window.validate()
        return request
