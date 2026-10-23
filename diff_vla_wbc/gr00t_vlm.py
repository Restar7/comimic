"""Top-level VLM (Cosmos-Reason2-2B / Qwen3-VL) for the joint framework.

``diff_vla_wbc``'s graph starts at *FM tokens*; everything above them was
missing.  This module adds that missing top:

```text
RGB + instruction (+ robot state)
        │
        ▼
Cosmos-Reason2-2B backbone      (Isaac-GR00T/gr00t/model/modules/qwen3_backbone.py)
        │  backbone_features [B, T, 2048]
        ▼
Gr00tN1d7ActionHead             (Isaac-GR00T/gr00t/model/gr00t_n1d7/gr00t_n1d7.py)
        │  action chunk [B, 40, max_action_dim]
        ▼
MotionGoalAdapter -> SuperMimic WBC -> world model -> task loss
```

Nothing here re-implements the VLM: :class:`Gr00tVLM` assembles the *vendored*
upstream classes (``Qwen3Backbone`` + ``Gr00tN1d7ActionHead``) under the config
contract of ``Gr00tN1d7``, so it is a drop-in ``gr00t_model`` argument for
:func:`diff_vla_wbc.gr00t_fm.sample_model_with_grad`.

Two reasons not to call ``Gr00tN1d7.from_pretrained`` directly:

* it requires a full GR00T checkpoint (2.5 GB action head **and** the gated
  ``nvidia/Cosmos-Reason2-2B`` backbone) -- here the VLM is loaded from a local
  directory / local HF cache with ``local_files_only=True`` and the head follows
  the same config;
* ``Gr00tN1d7.get_action`` is ``@torch.no_grad()``; the joint trainer needs the
  grad-enabled ``.prepare_input`` / ``.backbone`` / ``.action_head`` triple.

Weight resolution order (first hit wins):

1. ``model_name=`` argument passed by the caller;
2. ``$GR00T_VLM_PATH``;
3. ``groot_repro/nvidia/Cosmos-Reason2-2B-weights`` inside this checkout, when
   it holds real weights (``scripts/link_vlm_weights.py`` builds that view);
4. a ``models--nvidia--Cosmos-Reason2-2B`` snapshot in the local Hugging Face
   caches (``$HF_HOME``/``$HF_HUB_CACHE``, ``~/.cache/huggingface``, and any
   ``cache/huggingface/hub`` beside this checkout -- where the workspace keeps
   it);
5. the bare repo id ``nvidia/Cosmos-Reason2-2B`` (needs network or a populated
   default cache).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn

from .compat import WORKSPACE_ROOT

VLM_REPO_ID = "nvidia/Cosmos-Reason2-2B"
#: Local, offline-loadable view of the VLM inside this checkout.  Built by
#: ``scripts/link_vlm_weights.py`` and git-ignored: the weights stay in the HF
#: cache (or wherever they already are) and only symlinks live here.  The
#: config-only stub at ``groot_repro/nvidia/Cosmos-Reason2-2B`` is a different
#: thing -- it is the 16-layer donor config used to read GR00T checkpoints.
LOCAL_VLM_DIR = WORKSPACE_ROOT / "groot_repro" / "nvidia" / "Cosmos-Reason2-2B-weights"
_WEIGHT_SUFFIXES = (".safetensors", ".bin")
_VLM_ENV = "GR00T_VLM_PATH"
#: Extra Qwen3-VL processor outputs that newer transformers needs (M-RoPE etc.).
_OPTIONAL_VL_KEYS = (
    "mm_token_type_ids",
    "token_type_ids",
    "pixel_values_videos",
    "video_grid_thw",
    "image_sizes",
)
#: LLM layers kept by GR00T N1.7's truncation (``Gr00tN1d7Config.select_layer``);
#: the backbone pops the rest to save compute.  Cosmos-Reason2-2B ships 28.
DEFAULT_SELECT_LAYER = 16


def _has_weights(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(p.is_file() and p.suffix in _WEIGHT_SUFFIXES for p in path.iterdir())


def _hf_cache_roots() -> list[Path]:
    roots: list[Path] = []
    hub_cache = os.environ.get("HF_HUB_CACHE")
    if hub_cache:
        roots.append(Path(hub_cache))
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return roots


def _snapshot_dirs(hub_root: Path) -> list[Path]:
    snapshots = hub_root / ("models--" + VLM_REPO_ID.replace("/", "--")) / "snapshots"
    if not snapshots.is_dir():
        return []
    return sorted((p for p in snapshots.iterdir() if p.is_dir()), reverse=True)


def _workspace_snapshot_dirs() -> list[Path]:
    """Snapshots from a ``cache/huggingface/hub`` next to this checkout.

    The shared workspace keeps the Hugging Face cache beside the code rather than
    in ``$HOME``; walking the ancestors finds it without hard-coding a user path.
    """
    found: list[Path] = []
    for parent in [WORKSPACE_ROOT, *WORKSPACE_ROOT.parents]:
        found.extend(_snapshot_dirs(parent / "cache" / "huggingface" / "hub"))
    return found


def local_vlm_candidates() -> list[Path]:
    """Every local directory that may hold Cosmos-Reason2-2B weights, best first."""
    candidates: list[Path] = []
    env = os.environ.get(_VLM_ENV)
    if env:
        candidates.append(Path(env))
    candidates.append(LOCAL_VLM_DIR)
    for root in _hf_cache_roots():
        candidates.extend(_snapshot_dirs(root))
    candidates.extend(_workspace_snapshot_dirs())
    seen: set[str] = set()
    unique: list[Path] = []
    for path in candidates:
        key = os.path.realpath(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def resolve_vlm_source(model_name: str | Path | None = None) -> tuple[str, str]:
    """Resolve the VLM source and report where it came from.

    Returns ``(source, origin)`` where ``source`` is a local directory or a
    Hugging Face repo id and ``origin`` is one of ``argument``, ``env``,
    ``checkout``, ``hf-cache`` or ``repo-id``.
    """
    if model_name is not None:
        return str(model_name), "argument"
    env = os.environ.get(_VLM_ENV)
    if env:
        if not _has_weights(Path(env)):
            raise FileNotFoundError(
                f"${_VLM_ENV}={env} contains no VLM weights (*.safetensors)"
            )
        return env, "env"
    if _has_weights(LOCAL_VLM_DIR):
        return str(LOCAL_VLM_DIR), "checkout"
    for path in local_vlm_candidates():
        if _has_weights(path):
            return str(path), "hf-cache"
    return VLM_REPO_ID, "repo-id"


def vlm_available(model_name: str | Path | None = None) -> bool:
    """True when the VLM weights resolve to something local (no download needed)."""
    try:
        _source, origin = resolve_vlm_source(model_name)
    except FileNotFoundError:
        return False
    return origin != "repo-id"


def _gr00t_classes():
    """Import the vendored GR00T config/backbone/head (tolerating a partial tree)."""
    from .gr00t_tiny import ensure_gr00t_importable

    ensure_gr00t_importable()
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead  # noqa: E402
    from gr00t.model.modules.qwen3_backbone import Qwen3Backbone  # noqa: E402

    return Gr00tN1d7ActionHead, Qwen3Backbone


class Gr00tVLM(nn.Module):
    """Cosmos-Reason2-2B backbone + GR00T N1.7 flow-matching action head.

    The VLM is loaded with its real pretrained weights; the action head is the
    real architecture but stays randomly initialised unless a full GR00T
    checkpoint is supplied (GR00T's head weights are not part of this
    repository -- see ``checkpoint=``).
    """

    def __init__(
        self,
        config: Any,
        backbone: nn.Module,
        action_head: nn.Module,
        *,
        processor: Any | None = None,
        vlm_source: str | None = None,
        hand_head: nn.Module | None = None,
        goal_head: nn.Module | None = None,
    ):
        super().__init__()
        self.config = config
        self.backbone = backbone
        self.action_head = action_head
        self._processor = processor
        self.vlm_source = vlm_source
        #: Optional 14-D hand head.  Its output is used directly by
        #: ``diff_vla_wbc.hand_stream`` and never enters SuperMimic.
        self.hand_head = hand_head
        #: Optional ``[B, 10, 68]`` physical-goal head.  When present, SuperMimic
        #: is driven **directly** by the goal (no token adapter at all); the
        #: action-token path becomes optional/legacy.
        self.goal_head = goal_head

    # -- construction ------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        model_name: str | Path | None = None,
        *,
        checkpoint: str | Path | None = None,
        select_layer: int = DEFAULT_SELECT_LAYER,
        load_bf16: bool = True,
        head_dtype: torch.dtype | None = None,
        use_flash_attention: bool = False,
        local_files_only: bool = True,
        freeze_vlm: bool = True,
        action_horizon: int = 40,
        max_state_dim: int = 64,
        max_action_dim: int = 64,
        state_history_length: int = 1,
        num_inference_timesteps: int = 4,
        device: str | torch.device | None = None,
        attn_implementation: str | None = None,
        processor: Any | None = None,
        **config_overrides: Any,
    ) -> "Gr00tVLM":
        """Load the VLM backbone (local only by default) and build the action head.

        ``model_name`` accepts a local directory, a local HF cache snapshot or a
        repo id.  ``checkpoint`` optionally loads a full GR00T checkpoint
        (backbone + action head) on top, with the checkpoint's own config
        deciding the shapes.
        """
        if checkpoint is not None:
            return cls._from_gr00t_checkpoint(
                Path(checkpoint),
                model_name=model_name,
                local_files_only=local_files_only,
                device=device,
                processor=processor,
                freeze_vlm=freeze_vlm,
            )

        Gr00tN1d7ActionHead, Qwen3Backbone = _gr00t_classes()
        from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config

        source, origin = resolve_vlm_source(model_name)
        config_kwargs: dict[str, Any] = dict(
            model_name=source,
            select_layer=int(select_layer),
            backbone_embedding_dim=2048,
            tune_llm=not freeze_vlm,
            tune_visual=not freeze_vlm,
            action_horizon=int(action_horizon),
            max_state_dim=int(max_state_dim),
            max_action_dim=int(max_action_dim),
            state_history_length=int(state_history_length),
            num_inference_timesteps=int(num_inference_timesteps),
            use_flash_attention=bool(use_flash_attention),
        )
        config_kwargs.update(config_overrides)
        config = Gr00tN1d7Config(**config_kwargs)

        loading_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
            "trust_remote_code": True,
        }
        if attn_implementation is not None:
            loading_kwargs["attn_implementation"] = attn_implementation
        backbone = Qwen3Backbone(
            model_name=source,
            tune_llm=not freeze_vlm,
            tune_visual=not freeze_vlm,
            select_layer=int(select_layer),
            use_flash_attention=bool(use_flash_attention),
            load_bf16=bool(load_bf16),
            transformers_loading_kwargs=loading_kwargs,
        )
        backbone_dtype = next(backbone.parameters()).dtype
        if freeze_vlm:
            backbone.requires_grad_(False)
            backbone.eval()

        action_head = Gr00tN1d7ActionHead(config)
        # The flow-matching head must run in the backbone's dtype: torch's
        # LayerNorm/attention kernels reject mixed bf16/fp32 operands.  GR00T
        # itself trains in bf16 (`model_dtype: bfloat16`), so following the
        # backbone is also the upstream behaviour.
        target_dtype = head_dtype if head_dtype is not None else (backbone_dtype if load_bf16 else None)
        if target_dtype is not None:
            action_head.to(dtype=target_dtype)
        obj = cls(config, backbone, action_head, processor=processor, vlm_source=f"{source} ({origin})")
        if processor is None:
            obj.attach_processor(source, local_files_only=local_files_only)
        if device is not None:
            obj.to(torch.device(device))
        return obj

    @classmethod
    def _from_gr00t_checkpoint(
        cls,
        checkpoint: Path,
        *,
        model_name: str | Path | None,
        local_files_only: bool,
        device: str | torch.device | None,
        processor: Any | None,
        freeze_vlm: bool,
    ) -> "Gr00tVLM":
        """Load a GR00T checkpoint whose ``model_name`` points at a local VLM.

        GR00T checkpoints written on the deployment box hard-code an absolute
        ``model_name`` for the gated backbone, so the config is overridden with
        the locally resolved source before the weights are read.  This path is
        **not exercised in this workspace** (no GR00T checkpoint here); the
        standalone path above is.
        """
        # ``gr00t`` is not installed as a package in this workspace; the vendored
        # checkout has to be made importable first (the standalone path does this
        # inside ``_gr00t_classes``, so this branch needs it explicitly).
        from .gr00t_tiny import ensure_gr00t_importable

        ensure_gr00t_importable()
        from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config  # noqa: E402
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7  # noqa: E402

        source, origin = resolve_vlm_source(model_name)
        config = Gr00tN1d7Config.from_pretrained(str(checkpoint), local_files_only=True)
        config.model_name = source
        if freeze_vlm:
            config.tune_llm = False
            config.tune_visual = False
        model = Gr00tN1d7.from_pretrained(
            str(checkpoint),
            config=config,
            transformers_loading_kwargs={
                "local_files_only": local_files_only,
                "trust_remote_code": True,
            },
        )
        if freeze_vlm:
            model.backbone.requires_grad_(False)
            model.backbone.eval()
        obj = cls(
            model.config,
            model.backbone,
            model.action_head,
            processor=processor,
            vlm_source=f"{checkpoint} (checkpoint) + {source} ({origin})",
        )
        if processor is None:
            obj.attach_processor(source, local_files_only=local_files_only)
        if device is not None:
            obj.to(torch.device(device))
        return obj

    def attach_processor(self, source: str, *, local_files_only: bool = True) -> None:
        """Attach the Qwen3-VL processor so :meth:`build_batch` can run."""
        from transformers import AutoProcessor

        try:
            self._processor = AutoProcessor.from_pretrained(
                source,
                local_files_only=local_files_only,
                trust_remote_code=True,
            )
        except Exception:  # pragma: no cover - processor is optional for token replay
            self._processor = None

    # -- module plumbing ---------------------------------------------------
    @property
    def processor(self) -> Any:
        if self._processor is None:
            raise RuntimeError(
                "No Qwen3-VL processor attached; pass processor=... or feed "
                "preprocessed VLM inputs through prepare_input()."
            )
        return self._processor

    @property
    def device(self) -> torch.device:
        return next(iter(self.parameters())).device

    @property
    def dtype(self) -> torch.dtype:
        return next(iter(self.parameters())).dtype

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def parameter_groups(self, lr: float) -> list[dict[str, Any]]:
        params = self.trainable_parameters()
        if not params:
            raise ValueError("Gr00tVLM has no trainable parameters (freeze_vlm=True)")
        return [{"params": params, "lr": lr}]

    # -- inputs ------------------------------------------------------------
    def build_batch(
        self,
        images: Any,
        instructions: Sequence[str] | str,
        state: torch.Tensor | None = None,
        *,
        embodiment_id: torch.Tensor | int | None = None,
        prompts: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Turn raw (image, instruction) observations into a GR00T-ready batch.

        ``images`` may be a PIL image, a list of PIL images, or a ``[B,C,H,W]``
        float tensor in ``[0, 1]``.  ``state`` is ``[B, state_dim]`` (or
        ``[B, T, state_dim]``) and is zero-padded to ``config.max_state_dim``.

        The returned dict is exactly what :func:`sample_model_with_grad` expects:
        the Qwen3-VL processor outputs plus ``state`` and ``embodiment_id``.
        """
        if isinstance(instructions, str):
            instructions = [instructions]
        instructions = list(instructions)
        if not instructions:
            raise ValueError("instructions must be non-empty")

        processor = self.processor
        if prompts is None:
            prompts = [self._chat_prompt(processor, instruction) for instruction in instructions]

        image_list = self._as_pil_list(images, len(instructions))
        inputs = processor(text=prompts, images=image_list, return_tensors="pt", padding=True)
        batch: dict[str, Any] = dict(inputs)
        batch_size = int(batch["input_ids"].shape[0])

        if state is None:
            state = torch.zeros(batch_size, self.config.max_state_dim)
        else:
            state = torch.as_tensor(state).detach().clone()
        if state.dim() == 2:
            state = state.unsqueeze(1)
        if state.dim() != 3:
            raise ValueError(
                f"state must be [B, state_dim] or [B, T, state_dim], got {tuple(state.shape)}"
            )
        if state.shape[0] != batch_size:
            raise ValueError(f"state batch {state.shape[0]} != image batch {batch_size}")
        target_steps, target_dim = self.config.state_history_length, self.config.max_state_dim
        if state.shape[1] != target_steps:
            raise ValueError(f"state_history_length={target_steps}, got {state.shape[1]} step(s)")
        if state.shape[2] > target_dim:
            raise ValueError(f"state dim {state.shape[2]} > max_state_dim={target_dim}")
        if state.shape[2] < target_dim:
            pad = state.new_zeros(state.shape[0], state.shape[1], target_dim - state.shape[2])
            state = torch.cat((state, pad), dim=-1)

        batch["state"] = state
        if embodiment_id is None:
            batch["embodiment_id"] = torch.zeros(batch_size, dtype=torch.long)
        elif isinstance(embodiment_id, int):
            batch["embodiment_id"] = torch.full((batch_size,), int(embodiment_id), dtype=torch.long)
        else:
            batch["embodiment_id"] = torch.as_tensor(embodiment_id, dtype=torch.long)
        return batch

    @staticmethod
    def _chat_prompt(processor: Any, instruction: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": instruction}],
            }
        ]
        if hasattr(processor, "apply_chat_template"):
            return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return instruction

    @staticmethod
    def _as_pil_list(images: Any, batch_size: int) -> list[Any]:
        if isinstance(images, (list, tuple)):
            return list(images)
        if torch.is_tensor(images):
            if images.dim() == 3:
                images = images.unsqueeze(0)
            if images.dim() != 4:
                raise ValueError(f"image tensor must be [B,C,H,W], got {tuple(images.shape)}")
            from PIL import Image

            prepared = []
            for image in images.detach().cpu():
                if image.shape[0] not in (1, 3, 4):
                    raise ValueError(f"image tensor must be CHW, got {tuple(image.shape)}")
                array = (image[:3].clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
                prepared.append(Image.fromarray(array))
            if len(prepared) == 1 and batch_size > 1:
                prepared = prepared * batch_size
            return prepared
        # A single PIL image has no ``__len__``; ``len()`` on it raises
        # ``TypeError`` instead of taking the broadcast branch below.
        if hasattr(images, "__len__") and len(images) == batch_size:
            return list(images)
        return [images] * batch_size

    # -- the Gr00tN1d7-compatible interface used by gr00t_fm ---------------
    def prepare_input(self, inputs: dict) -> tuple[Any, Any]:
        """Same split as ``Gr00tN1d7.prepare_input``: (backbone inputs, head inputs).

        Inputs must already be processor outputs (``input_ids``,
        ``attention_mask``, ``pixel_values``, ``image_grid_thw``) plus ``state``
        and ``embodiment_id``; ``vlm_content`` is rejected because that path
        needs the training collator and its augmentation stack.
        """
        if "vlm_content" in inputs:
            raise ValueError(
                "Gr00tVLM takes preprocessed VLM inputs; use build_batch() for raw "
                "images/instructions instead of passing 'vlm_content'."
            )
        from transformers.feature_extraction_utils import BatchFeature

        device, dtype = self.device, self.dtype

        def cast(value: Any) -> Any:
            if not torch.is_tensor(value):
                return value
            if torch.is_floating_point(value):
                return value.to(device=device, dtype=dtype)
            return value.to(device)

        vlm_keys = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
        missing = [key for key in vlm_keys if key not in inputs]
        if missing:
            raise KeyError(f"VLM batch is missing {missing}; call build_batch() first")
        # transformers>=5 emits extra multimodal tensors (e.g. mm_token_type_ids
        # for M-RoPE); forward whichever of them the processor produced.
        optional = [key for key in _OPTIONAL_VL_KEYS if key in inputs]
        backbone_inputs = BatchFeature(
            data={key: cast(inputs[key]) for key in (*vlm_keys, *optional)}
        )

        for key in ("state", "embodiment_id"):
            if key not in inputs:
                raise KeyError(f"action inputs are missing '{key}'; call build_batch() first")
        action_inputs = BatchFeature(
            data={key: cast(inputs[key]) for key in ("state", "embodiment_id")}
        )
        return backbone_inputs, action_inputs

    def forward(self, inputs: dict) -> torch.Tensor:
        """Grad-enabled VLM + action-head forward returning the raw action chunk."""
        from .gr00t_fm import sample_model_with_grad

        return sample_model_with_grad(self, inputs)

    def backbone_features(self, inputs: dict) -> torch.Tensor:
        """Backbone features ``[B, T, D]`` (what both heads consume).

        Runs the VLM + the action head's feature encoder once; no flow-matching
        sampling, so it is the cheap path when only goal/hand predictions are
        needed.
        """
        backbone_inputs, action_inputs = self.prepare_input(dict(inputs))
        backbone_output = self.backbone(backbone_inputs)
        features = self.action_head._encode_features(backbone_output, action_inputs)
        return features.backbone_features

    def forward_goal_and_hands(
        self, inputs: dict, *, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """The split this repo actually uses to drive SuperMimic.

        * ``goal [B, 10, 68]`` -- physical reference for SuperMimic's policy,
          consumed **directly** (``flatten_goal`` -> ``SupermimicBundle.act``).
        * ``hands [B, H, 14]`` -- hand joint angles, used **directly** by the
          hand stream, never entering SuperMimic.

        No motion tokens and no adapter are involved.
        """
        features = self.backbone_features(inputs)
        goal = self.goal_head(features, mask=mask) if self.goal_head is not None else None
        hands = self.hand_head(features, mask=mask) if self.hand_head is not None else None
        return goal, hands

    def forward_with_hands(
        self, inputs: dict, *, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(tokens, hands)``: the two VLA outputs of the decoupled design.

        * ``tokens [B, H_t, 64]`` -- motion token chunk, the **only** thing that
          goes into the adapter / SuperMimic (WBC + world model).
        * ``hands [B, H, 14]`` -- hand joint angles, taken **directly** from the
          VLM and used directly by ``diff_vla_wbc.hand_stream``.  They are not
          part of the token and never enter SuperMimic.

        Requires ``self.hand_head``; without it ``hands`` is ``None`` and this is
        equivalent to :meth:`forward`.
        """
        from .gr00t_fm import sample_model_with_grad

        tokens, features, _state = sample_model_with_grad(self, inputs, return_features=True)
        if self.hand_head is None:
            return tokens, None
        return tokens, self.hand_head(features, mask=mask)


class Gr00tVLMTokenSource(nn.Module):
    """``fm_sampler`` for :class:`~diff_vla_wbc.train.JointTrainer`.

    Runs the real top-level VLM (and the flow-matching head) per training step so
    gradients reach the model, and returns the ``[B, 40, action_dim]`` chunk that
    the motion-goal adapter consumes.
    """

    def __init__(
        self,
        vlm: Gr00tVLM,
        *,
        num_steps: int | None = None,
        initial_noise: torch.Tensor | None = None,
        return_features: bool = False,
    ):
        super().__init__()
        self.vlm = vlm
        self.num_steps = num_steps
        self.return_features = return_features
        self.register_buffer(
            "fixed_noise",
            initial_noise if initial_noise is not None else torch.empty(0),
            persistent=True,
        )

    def forward(self, model_inputs: dict) -> torch.Tensor:
        from .gr00t_fm import sample_model_with_grad

        noise = self.fixed_noise if self.fixed_noise.numel() else None
        return sample_model_with_grad(
            self.vlm,
            model_inputs,
            initial_noise=noise,
            num_steps=self.num_steps,
            return_features=self.return_features,
        )


def _backbone_layer_count(backbone: nn.Module) -> int | None:
    """Number of LLM layers actually kept after ``select_layer`` truncation."""
    inner = getattr(backbone, "model", None)
    for holder in (inner, getattr(inner, "model", None)):
        language_model = getattr(holder, "language_model", None)
        layers = getattr(language_model, "layers", None)
        if layers is not None:
            return len(layers)
    return None


def describe_vlm(vlm: Gr00tVLM) -> dict[str, Any]:
    """Small summary used by the CLIs and the smoke script."""
    params = list(vlm.parameters())
    trainable = [p for p in params if p.requires_grad]
    return {
        "source": vlm.vlm_source,
        "total_params": sum(p.numel() for p in params),
        "trainable_params": sum(p.numel() for p in trainable),
        "backbone_layers": _backbone_layer_count(vlm.backbone),
        "action_horizon": int(vlm.action_head.action_horizon),
        "action_dim": int(vlm.action_head.action_dim),
        "state_history_length": int(vlm.config.state_history_length),
        "max_state_dim": int(vlm.config.max_state_dim),
    }


__all__ = [
    "DEFAULT_SELECT_LAYER",
    "Gr00tVLM",
    "Gr00tVLMTokenSource",
    "LOCAL_VLM_DIR",
    "VLM_REPO_ID",
    "describe_vlm",
    "local_vlm_candidates",
    "resolve_vlm_source",
    "vlm_available",
]
