"""M1 回归：mjlab/MuJoCo 后端的联合微调 bundle（obs **93**）能跑通整条可微链路。

覆盖三件事（都是迁移里最容易静默跑偏的点）：
1. 观测宽度固定 93、term-major 历史 930、首 29 维 == ``q - q0``；
2. ``reference_start`` / ``teacher_states`` 的 state 元组走 :func:`_ref71_body`
   （运动库是世界系角速度，状态/观测/WM 是体系）；
3. ``imagine_with_tokens`` 端到端可微：奖励 → WBC → WM → **token 有梯度**，
   且 policy/WM 参数本身不可训（冻结）。

用 2 条 clip + CPU，几秒内跑完；不需要 GPU、不需要 Isaac、不需要 mjlab 运行环境
（``ensure_supermimic_importable`` 会装 ``mjlab.utils.lab_api.math`` 的纯 torch shim）。
"""

from __future__ import annotations

import glob
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "third_party/SuperMimic-Reproduced/logs/e9_bones_seed_token_h200x2/ckpt_0024000.pt"
NPZ_GLOB = str(ROOT / "data/npz/g1_tasks/*.npz")
TOKENS = ROOT / "data/library/ft318_tokens.pt"

pytestmark = pytest.mark.skipif(
    not (CKPT.is_file() and glob.glob(NPZ_GLOB) and TOKENS.is_file()),
    reason="需要 mjlab 底模 ckpt + 318 npz + 录制 token 包",
)


@pytest.fixture(scope="module")
def bundle():
    from diff_vla_wbc.mjlab_joint import load_mjlab_bundle

    npz = sorted(glob.glob(NPZ_GLOB))[:2]          # 2 条 clip，够做形状/梯度回归
    return load_mjlab_bundle(CKPT, device="cpu", npz=npz, token_dim=64)


def test_observation_is_93(bundle):
    from diff_vla_wbc.mjlab_joint import OBS_DIM, TERM_DIMS

    assert OBS_DIM == 93 and TERM_DIMS == (29, 29, 3, 3, 29)
    assert bundle.obs_width == 93 and bundle.term_dims == TERM_DIMS

    frames = torch.tensor([120, 400])
    start = bundle.reference_start(bundle.library, frames)
    assert start["state"].shape == (2, 71)
    assert start["obs_hist"].shape == (2, 930)
    assert start["observations"].shape == (10, 2, 93)      # rollout 顺序 (H,B,W)

    obs = bundle.observation(start["state"], start["prev_action"], start["xi"])
    assert obs.shape == (2, 93)
    # 首 29 维 == q - q0（xi 偏置为 0 时；actor 的 q0 来自 MJCF default pose）
    q = start["state"][:, 7:36]
    assert torch.allclose(obs[:, :29], q - bundle.default_joint_pos, atol=1e-4)


def test_state_uses_body_frame_angvel(bundle):
    """库里的 ``[39:42]`` 是世界系；state/观测/WM 要体系（§1.3）。"""
    from diff_vla_wbc.mjlab_joint import _ref71_body
    from supermimic.kinematics.rotations import quat_apply_inverse

    lib = bundle.library
    g = torch.Generator().manual_seed(0)
    frames = torch.randint(lib.num_frames, (16,), generator=g)
    raw = lib.ref71(frames).float()
    conv = _ref71_body(lib, frames)

    # 其余维度逐位不变
    assert torch.equal(raw[:, :39], conv[:, :39])
    assert torch.equal(raw[:, 42:], conv[:, 42:])
    # [39:42] 恰好是 world → body 的旋转
    assert torch.allclose(
        conv[:, 39:42], quat_apply_inverse(raw[:, 3:7], raw[:, 39:42]), atol=1e-6
    )
    # reference_start / teacher_states 都必须用转换后的版本
    assert torch.allclose(bundle.reference_start(lib, frames)["state"], conv, atol=1e-6)
    assert torch.allclose(bundle.teacher_states(lib, frames, 2)[:, 0],
                          _ref71_body(lib, frames + 1), atol=1e-6)


def test_reward_backprops_into_token(bundle):
    from diff_vla_wbc.rollout import imagine_with_tokens

    torch.manual_seed(0)
    lib = bundle.library
    frames = torch.tensor([200, 600, 900, 1200])
    start = bundle.reference_start(lib, frames)
    refs = bundle.teacher_references(lib, frames, 4)

    tok = torch.load(TOKENS, map_location="cpu", weights_only=False)["tokens"]
    tokens = tok[frames[:, None] + torch.arange(4)].float().requires_grad_(True)   # (B,4,64)

    roll = imagine_with_tokens(
        bundle, tokens=tokens, state=start["state"], obs_hist=start["obs_hist"],
        history=None, xi=start["xi"], references=refs, prev_action=start["prev_action"],
        horizon=4,
    )
    assert roll.states.shape == (5, 4, 71)
    assert roll.actions.shape == (4, 4, 29)
    assert roll.rewards.shape == (4, 4)

    (-roll.rewards.sum()).backward()

    assert tokens.grad is not None, "梯度没有回到 token —— 可微链路断了"
    assert float(tokens.grad.abs().sum()) > 0.0
    # 冻结检查：执行器/动力学参数不可训（训练器会再调一次 _freeze_executor，这里验基线）
    assert not any(p.requires_grad for p in bundle.policy.parameters())
    assert not any(p.requires_grad for p in bundle.wm.parameters())
