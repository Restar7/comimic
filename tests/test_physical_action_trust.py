"""物理动作 trust（29D rad）手算对拍（规范 §7.1 / §9 `tests/test_physical_action_trust.py`）。

断言：
1. `q_target` 单位是 **rad**：`q = default + scale·a`；
2. 同 obs 下 `e = scale·(a_corr − a_base)`（即 `joint_error_rad(q_corr, q_base)`）；
3. `L_act` / `RMS` / `L_smooth` / `L_anchor` 与手算逐位一致；
4. `q_base` 未 detach 时直接抛（参照必须是常数）。
"""
from __future__ import annotations

import math

import pytest
import torch

from diff_vla_wbc.action_trust_v5 import (
    action_budget_stats, anchor_penalty, joint_error_rad, masked_rms, physical_action_trust,
    q_target_from_action, smoothness_penalty)

DEFAULT = torch.linspace(-0.4, 0.5, 29)
SCALE = torch.linspace(0.5, 1.5, 29)


def test_q_target_is_radians():
    a = torch.randn(4, 6, 29)
    q = q_target_from_action(DEFAULT, SCALE, a)
    assert tuple(q.shape) == (4, 6, 29)
    assert torch.allclose(q, DEFAULT + SCALE * a, atol=1e-6)
    assert torch.allclose(q[0, 0], DEFAULT + SCALE * a[0, 0])
    # 物理量级检查：scale 是 rad/动作单位，a=1 时偏移就是 scale
    ones = torch.ones(1, 29)
    assert torch.allclose(q_target_from_action(DEFAULT, SCALE, ones)[0] - DEFAULT, SCALE)


def test_error_equals_scale_times_action_difference():
    a_corr = torch.randn(5, 4, 29)
    a_base = torch.randn(5, 4, 29)
    q_corr = q_target_from_action(DEFAULT, SCALE, a_corr)
    q_base = q_target_from_action(DEFAULT, SCALE, a_base)
    e = joint_error_rad(q_corr, q_base.detach())
    assert torch.allclose(e, SCALE * (a_corr - a_base), atol=1e-6)


def test_error_requires_detached_base():
    q_corr = torch.zeros(2, 3, 29)
    q_base = torch.zeros(2, 3, 29, requires_grad=True)
    with pytest.raises(ValueError):
        joint_error_rad(q_corr, q_base)


def test_masked_rms_matches_hand_calc():
    torch.manual_seed(0)
    e = torch.randn(3, 7, 29)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0],
                         [1, 1, 1, 1, 1, 1, 1],
                         [0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32)
    rms, n = masked_rms(e, mask)
    for i in range(3):
        k = int(mask[i].sum())
        if k == 0:
            assert float(rms[i]) == 0.0
            continue
        manual = math.sqrt(float((e[i, :k] ** 2).sum()) / (29 * k))
        assert rms[i].item() == pytest.approx(manual, rel=1e-6)
        assert float(n[i]) == k


def test_trust_hinge_hand_calc():
    tau = 0.02
    # 构造逐样本 RMS 恰为 0.01 / 0.03 rad：e = rms * ones(29) ⇒ ‖e‖² = 29·rms²
    e = torch.stack([torch.full((4, 29), 0.01), torch.full((4, 29), 0.03)], 0)  # (2,4,29)
    mask = torch.ones(2, 4)
    loss, rms = physical_action_trust(e, mask, tau=tau)
    assert torch.allclose(rms, torch.tensor([0.01, 0.03]), atol=1e-6)
    manual = (max(0.01 / tau - 1.0, 0.0) ** 2 + (0.03 / tau - 1.0) ** 2) / 2
    assert loss.item() == pytest.approx(manual, rel=1e-6)
    # 全部低于 tau ⇒ loss = 0（hinge 不惩罚更小）
    small = torch.full((2, 4, 29), 0.005)
    loss0, _ = physical_action_trust(small, mask, tau=tau)
    assert float(loss0) == 0.0


def test_smoothness_hand_calc():
    tau = 0.02
    e = torch.zeros(2, 3, 29)
    e[0, 0] = 0.01
    e[0, 1] = 0.02
    mask = torch.ones(2, 3)
    loss = smoothness_penalty(e, mask, tau=tau)
    # 两段非零差：Δe = +0.01 与 −0.02（各 29 维）⇒ mean over 29 = 1e-4 / 4e-4；
    # denom = mask[:,1:]·mask[:,:-1] 的和 = 4（2 个 batch × 2 段）
    manual = ((1e-4 + 4e-4) / 4.0) / tau ** 2
    assert loss.item() == pytest.approx(manual, rel=1e-6)
    # mask 把第一段裁掉 ⇒ 0
    m = mask.clone()
    m[:, 1] = 0.0
    assert float(smoothness_penalty(e, m, tau=tau)) == 0.0


def test_anchor_hand_calc():
    cap = 0.08
    delta = torch.full((2, 40, 64), 0.01)
    mask = torch.zeros(2, 40)
    mask[:, :20] = 1.0
    loss = anchor_penalty(delta, mask, cap=cap)
    # Σ(Δ²·mask)/(K·B)/cap² = (2·20·64·1e-4)/(2·20)/cap²
    manual = (2 * 20 * 64 * 1e-4) / (2 * 20) / cap ** 2
    assert loss.item() == pytest.approx(manual, rel=1e-6)


def test_action_budget_stats():
    e = torch.zeros(2, 3, 29)
    e[0, 0, 0] = 0.03
    e[1, 2, 28] = -0.02
    mask = torch.ones(2, 3)
    st = action_budget_stats(e, mask)
    assert st["peak_mrad"] == pytest.approx(30.0, rel=1e-6)
    # RMS：每样本 sqrt(Σe²/(29*3))
    rms0 = math.sqrt(0.03 ** 2 / (29 * 3))
    rms1 = math.sqrt(0.02 ** 2 / (29 * 3))
    assert st["rms_mrad"] == pytest.approx((rms0 + rms1) / 2 * 1e3, rel=1e-5)
    assert st["steps"] == pytest.approx(3.0)
    assert st["samples"] == pytest.approx(6.0)
