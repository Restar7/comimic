"""会话 01（部署加载器）× 会话 02（schema v2 metadata）的接口合同测试。

为什么单独一个文件：两个会话各自是对的，但**接口**才是历史上的事故点——
训练侧写出的 metadata 字段名/语义一变，部署侧的推断就会静默退回 35 维。
本文件把"训练侧真正会写出的 v2 meta"当作输入，断言加载器能**无歧义**读出来，
并且在字段自相矛盾时 fail-closed（不是猜一个继续跑）。
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from diff_vla_wbc.delta_token import STATE64_DIM, TOKEN_DIM, DeltaTokenNet
from diff_vla_wbc.delta_token_io import (
    DeltaTokenContractError,
    delta_features_from_state64,
    load_delta_token_from_checkpoint,
)


def _trained_net(feat_dim: int, seed: int = 0) -> DeltaTokenNet:
    """造一个"训练过的"网（非零末层，才检验得了特征是否喂对）。"""
    torch.manual_seed(seed)
    net = DeltaTokenNet(feat_dim=feat_dim, hidden=64, depth=2, init_std=4e-4)
    with torch.no_grad():
        net.net[-1].weight.normal_(0.0, 0.05)
    return net


def _v2_meta(feat_dim: int, *, hidden: int = 64, depth: int = 2) -> dict:
    """**与 `V3Trainer.delta_token_meta()` 完全同键**的 schema v2 metadata。"""
    return {
        "schema_version": 2,
        "hidden": hidden,
        "init_std": 4e-4,
        "depth": depth,
        "cap": 0.0,
        "feat_dim": feat_dim,
        "token_dim": TOKEN_DIM,
        "input_features": ("state64[rot6d6|dof_pos29|dof_vel29]" if feat_dim == STATE64_DIM
                           else "rot6d6+dof_vel29"),
        "input_features_version": ("deploy_state64_v1" if feat_dim == STATE64_DIM
                                   else "legacy_35_v1"),
        "apply": "broadcast_shared_over_chunk",
        "norm_cap": "apply_norm_cap(tanh)",
        "teacher_objective": "shared_delta_discounted_v1",
        "teacher_reward": "discounted",
        "teacher_horizon": 20,
        "teacher_eta": 0.08,
        "teacher_n": 8,
        "grad_target_weight": 1.0,
        "trained_steps": 150,
        "git_commit": "deadbeef",
    }


@pytest.mark.parametrize("feat_dim", [35, STATE64_DIM])
def test_v2_meta_roundtrip_from_training(feat_dim):
    net = _trained_net(feat_dim)
    ck = {"delta_token_state": {k: v.clone() for k, v in net.state_dict().items()},
          "delta_token_meta": _v2_meta(feat_dim)}

    loaded, spec = load_delta_token_from_checkpoint(ck, device="cpu")
    assert int(spec.meta.get("schema_version", 0)) == 2
    assert int(spec.feat_dim) == int(feat_dim)
    assert int(spec.token_dim) == TOKEN_DIM
    # 训练侧约定关节块是 standard 序；meta 没写时加载器按训练约定兜底并记录来源
    assert spec.feat_joint_order == "standard"
    # 前向逐元素一致（同一 token / 同一特征）
    tok = torch.zeros(1, TOKEN_DIM)
    feat = torch.zeros(1, feat_dim)
    with torch.no_grad():
        a = net(tok, feat)
        b = loaded(tok, feat)
    assert torch.allclose(a, b, atol=1e-6)


def test_v2_meta_conflicting_feat_dim_fails_closed():
    net = _trained_net(STATE64_DIM)
    meta = _v2_meta(STATE64_DIM)
    meta["feat_dim"] = 35                      # 与首层宽度 (1024, 128) 矛盾
    ck = {"delta_token_state": {k: v.clone() for k, v in net.state_dict().items()},
          "delta_token_meta": meta}
    with pytest.raises(DeltaTokenContractError):
        load_delta_token_from_checkpoint(ck, device="cpu")


def test_v2_meta_state64_feature_layout_matches_training_order():
    """部署 state64（ha_v31 关节序）→ 特征（standard 序）：与训练侧同序。

    训练侧 `_delta_token_features(fullstate=1)` = ``[rot6d(6), q(29), qd(29)]``，其中 q/qd
    来自 bundle 的 standard 序；部署侧 state64 的关节块是 ha_v31 序。
    这里用"关节值 = 序号"构造，断言加载器的置换与 `joint_contract` 一致。
    """
    import joint_contract as JC

    perm = np.asarray(JC.permutation("ha_v31", "standard"), dtype=np.int64)
    q_ha = np.arange(29, dtype=np.float32) + 0.5
    qd_ha = np.arange(29, dtype=np.float32) + 100.0
    state64 = np.concatenate([np.zeros(6, np.float32), q_ha, qd_ha])

    net = _trained_net(STATE64_DIM)
    ck = {"delta_token_state": {k: v.clone() for k, v in net.state_dict().items()},
          "delta_token_meta": _v2_meta(STATE64_DIM)}
    _loaded, spec = load_delta_token_from_checkpoint(ck, device="cpu", state_joint_order="ha_v31")
    feat = delta_features_from_state64(state64, spec)

    expected = np.concatenate([np.zeros(6, np.float32), q_ha[perm], qd_ha[perm]])
    assert feat.shape == (STATE64_DIM,)
    assert np.allclose(feat, expected, atol=1e-6)


def test_legacy_ckpt_still_infers_without_v2_meta():
    net = _trained_net(STATE64_DIM)
    ck = {"delta_token_state": {k: v.clone() for k, v in net.state_dict().items()},
          "delta_token_meta": {"hidden": 64, "depth": 2, "init_std": 4e-4, "cap": 0.0}}
    loaded, spec = load_delta_token_from_checkpoint(ck, device="cpu")
    assert int(spec.feat_dim) == STATE64_DIM          # 由首层宽度反推
    assert int(spec.meta.get("schema_version", 1)) == 1  # 无 v2 声明 ⇒ 记作 legacy 推断
