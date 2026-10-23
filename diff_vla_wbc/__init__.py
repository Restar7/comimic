"""Differentiable VLA-Whole-Body Co-Training Framework.

GR00T N1.7 flow-matching action head -> motion-goal adapter -> SuperMimic WBC
-> differentiable world model -> task loss, with one autograd graph.

The upstream SuperMimic reproduction is vendored under
``third_party/SuperMimic-Reproduced`` (CuboiLeo/SuperMimic-Reproduced).
"""
from .adapter import AdapterConfig, MotionGoalAdapter, default_goal_prior
from .compat import ensure_supermimic_importable, install_mjlab_shim
from .gr00t_fm import (
    Gr00tTokenSource,
    sample_fm_from_features,
    sample_fm_with_grad,
    sample_model_with_grad,
)
from .data import CachedTokenSource
from .rollout import (
    LossWeights,
    Rollout,
    compute_joint_loss,
    freeze_world_model,
    imagine_with_goals,
    unfreeze_world_model,
)
from .supermimic_backend import (
    G1Assets,
    SuperMimicBundle,
    build_g1_assets,
    default_xi,
    load_motion_library,
)
from .task_losses import body_position_loss, end_effector_loss, root_height_loss, upright_loss
from .train import JointTrainer, JointTrainerConfig, pretrain_world_model

__version__ = "0.1.0"

__all__ = [
    "AdapterConfig",
    "CachedTokenSource",
    "G1Assets",
    "Gr00tTokenSource",
    "JointTrainer",
    "JointTrainerConfig",
    "LossWeights",
    "MotionGoalAdapter",
    "Rollout",
    "SuperMimicBundle",
    "body_position_loss",
    "build_g1_assets",
    "compute_joint_loss",
    "default_goal_prior",
    "default_xi",
    "end_effector_loss",
    "ensure_supermimic_importable",
    "freeze_world_model",
    "imagine_with_goals",
    "install_mjlab_shim",
    "load_motion_library",
    "pretrain_world_model",
    "root_height_loss",
    "sample_fm_from_features",
    "sample_fm_with_grad",
    "sample_model_with_grad",
    "unfreeze_world_model",
    "upright_loss",
]
