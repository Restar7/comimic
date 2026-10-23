# GR00T N1.7 + Uni-SuperMimic Differentiable Bridge (research prototype)

**Status:** Tested on synthetic PyTorch models only. This package does NOT
include checkpoints, motion data, a ready-to-run MuJoCo scene, robot deployments,
or a demonstrated physical-world task success. It establishes and tests the
proposed gradient path and encodes the primary repository interfaces.

## Target repos

- GR00T fork: https://github.com/Restar7/issac_groot
- Uni-SuperMimic: https://github.com/Motphys/Uni-SuperMimic
- SuperMimic paper: https://arxiv.org/abs/2610.28480
- BONES-SEED: https://huggingface.co/datasets/bones-studio/seed

## Grad path

RGB/Language/Proprio -> `Gr00tN1d7.backbone` ->
`sample_fm_with_grad(Gr00tN1d7.action_head)` -> `[B,40,64]` ->
`MotionGoalAdapter` -> `[B,40,730]` ->
`Policy(obs_hist[B,10,99], goal[B,730])` -> `raw_action[B,29]` ->
`WorldModel.step(State,raw_action,normalize_context(xi))` ->
`proprio_obs` -> policy rollouts 4-8 steps -> task/tracking loss -> `.backward()`.

WorldModel is frozen for policy updates; input derivatives remain live.
WorldModel should be updated separately using physical simulation transitions.

**Critical**: `Gr00tN1d7ActionHead.get_action()` and
`get_action_with_features()` are both decorated `@torch.no_grad()` in
the provided repository. Do not call these during joint fine-tuning.
`sample_fm_with_grad` replicates their Euler loop without disabling gradients
(excluding optional RTC/inpainting).
`Gr00tPolicy.get_action` and SONIC ONNX are inference boundaries and must
not appear inside the training computational graph.

## Quick local verification

From the unzipped package root (only requires PyTorch for smoke test):

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Expected nonzero gradient norms for (a) FM velocity/DiT weights,
(b) goal adapter and (c) WBC policy, while the WM weights remain frozen.

## Minimal integration snippet (after training and installing both repositories)

```python
from gr00t_supermimic_bridge.core import (
    MotionGoalAdapter, sample_model_with_grad, imagined_rollout,
    compute_loss, freeze_world_model, load_supermimic_checkpoint,
)
import torch

device = "cuda"
gr00t = ...             # your loaded Gr00tN1d7 model
supermimic_ckpt = "runs/supermimic_mjwarp_001/last.pt"
wm, pi = load_supermimic_checkpoint(supermimic_ckpt, device)
freeze_world_model(wm)
adapter = MotionGoalAdapter(in_dim=64).to(device)

optimizer = torch.optim.AdamW(
    [
        {"params": gr00t.action_head.parameters(), "lr": 1e-5},
        {"params": adapter.parameters(), "lr": 1e-4},
        {"params": pi.parameters(), "lr": 1e-5},
    ],
    weight_decay=1e-4,
)

# ALL these come from a real synchronized training batch:
batch = ...             # GR00T-ready preprocessed RGB/text/state tensors
initial_state = ...     # six tensors: p,R,v,w,q,dq
obs_hist = ...          # [B,10,99] in Uni-SuperMimic natural joint order
xi = ...                # [B,33] simulation context
default_q = ...         # [29] same MJCF as SuperMimic
teacher_goals = ...     # [B,40,730] from MotionLib.goal or equivalent
teacher_q = ...         # [B,8,29] for short rollout
teacher_root_p = ...    # [B,8,3]
ref_world_R = ...       # [B,8,3,3] teacher alignment (optional)

# Use a fixed/separately sampled FM noise tensor for reproducible gradients:
fm_tokens = sample_model_with_grad(gr00t, batch)
trajectory = imagined_rollout(
    fm_tokens, adapter, pi, wm,
    initial_state, obs_hist, xi, default_q, horizon=8,
    reference_world_rotations=ref_world_R,
)
loss, terms = compute_loss(
    trajectory,
    teacher_goals=teacher_goals,
    teacher_joint_positions=teacher_q,
    teacher_root_positions=teacher_root_p,
    # task_loss_fn=lambda trajectory: YOUR_DIFFERENTIABLE_TASK_LOSS(trajectory),
    # fm_supervised_loss=your_existing_flow_matching_sft_loss,
)
optimizer.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(gr00t.action_head.parameters(), 0.5)
torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.)
torch.nn.utils.clip_grad_norm_(pi.parameters(), 1.)
optimizer.step()
```

## Dataset, geometry, and rollout interface contract

`MotionLib.goal()` returns 10 future reference frames each containing:
- root delta in heading frame: 3
- root 6D relative rotation: 6
- root local linear velocity: 3
- root local angular velocity: 3
- 29 joint positions: 29
- 29 joint velocities multiplied by 0.1: 29

`MotionLib` uses 50Hz CONTROL_DT=0.02; BONES-SEED G1 material is 120Hz.
Convert at correct FPS and joint order; do not treat 120Hz as 30Hz.
The LAFAN1 sample loader expects exactly `[root xyz (3), root quat
xyzw (4), q (29)]` (36 CSV columns) and internally converts to `wxyz`.
BONES-SEED CSV must be checked for metadata and mapped explicitly; never
silently assume its columns are compatible.

The BONES-SEED dataset is gated and custom licensed. Review licensing before
using it, particularly for commercial training. BONES-SEED does not provide
the synchronized ego camera/object dynamics used by an RGB-conditioned
manipulation VLA. Use BONES for WBC pretraining; use real or simulated
RGB+language+proprio+joint+object/action demonstrations for VLA finetuning.

**Scalability:** 288h at 50 Hz = about 51.8 million frames; the present
Uni-SuperMimic MotionLib eagerly loads and precomputes FK for all clips.
Implement sharded lazy motion sampling / streaming for BONES size.

## Suggested steps for the real experiment

1. Prepare G1 joint/name/actuator map, phase alignment, physics assets and frame formats.
   Verify WBC action is **joint position offset**, converted by:
   `q_target = default_q + xi[:,1:30] + alpha * raw_action`
   (`alpha` per joint from Uni-SuperMimic `robot.action_scale`).
   Preserve GR00T 14D hands on their own task head.
2. Pretrain SuperMimic pi+WM using BONES-SEED G1 trajectories, and verify
   tracking, falls, sim2sim and 8-step holdout WM errors.
3. Train the `64 -> 730` adapter supervised using aligned `MotionLib.goal`
   targets before enabling imagined loss. Freeze GR00T and WBC at first.
4. Joint fine-tune adapter+FM head (WBC initially frozen, later selectively
   unfrozen); use `L_flow + L_goal + L_track + L_task + L_safety`.
   Alternate WM updates with detached genuine physics transitions;
   freeze WM parameters for each policy improvement step.
5. Object tasks require a world model including object pose/velocity,
   robot-object contact and task-specific rewards. Existing WM is robot-only.
   Use short horizons, gradient clipping, rollout trust-region checks and
   MuJoCo/real closed-loop evaluation. Real MuJoCo `step` itself is not
   the differentiable part of this implementation.

## Unsolved/not included in this starter

Actual checkout/merging of the two GitHub repositories; exact version
dependency resolution; GR00T weights or model state normalization files;
BONES access and large-scale sharded converter; training of the actual
G1 WM; robot/object/contact dynamics; hardware safe controller;
simulation task success. Thus this is a **testable interface and autograd
prototype**, not a claim that end-to-end training already succeeded.

## Related research

- SuperMimic (2026): https://arxiv.org/abs/2610.28480
- VITA (2025/2026): https://arxiv.org/abs/2507.13231
- Q-VGM (2026): https://arxiv.org/abs/2606.08015
- Flow Policy Gradients (2026): https://arxiv.org/abs/2602.02481
