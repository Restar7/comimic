"""关节顺序契约的回归测试 —— 防止"顺序又悄悄错位"。

三个真源必须和 `joint_contract` 完全一致：

1. **MuJoCo MJCF**（``reproduce_scenes/g1/g1_29dof_with_hand.xml``）的关节声明顺序
   —— 物理/策略顺序（``STANDARD_29``）
2. **trackerlab** ``get_robot("g1").joint_names``
   —— WBC 训练时的动作/观测顺序（必须等于 ``STANDARD_29``）
3. **HumanoidArena** ``CANONICAL_G1_JOINT_NAMES_29``
   —— ``vla_state`` 的 29 维顺序（必须等于 ``HA_V31_29``）

任何一处被改，这里就会红。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joint_contract as JC  # noqa: E402

MJCF = ROOT / "reproduce_scenes" / "g1" / "g1_29dof_with_hand.xml"
TRACKERLAB_SRC = ROOT / "trackerlab-supermimic" / "source" / "trackerTask"
HA_RUNTIME = (ROOT / "third_party" / "HumanoidArena" / "isaaclab_twist2_g1"
              / "action_provider" / "vla_smpl_runtime.py")
HA_PROVIDER = (ROOT / "third_party" / "HumanoidArena" / "isaaclab_twist2_g1"
               / "action_provider" / "action_provider_supermimic.py")


def _mjcf_joint_names() -> list[str]:
    text = MJCF.read_text(encoding="utf-8")
    return re.findall(r'<joint\s+name="([a-z0-9_]+)"', text)


def test_mjcf_matches_standard():
    """MJCF（MuJoCo 部署模型）的 29 个 body 关节顺序 == STANDARD_29。"""
    names = _mjcf_joint_names()
    body = [n for n in names if not n.startswith(("left_hand_", "right_hand_"))
            and n != "floating_base_joint"]
    assert len(body) == JC.NUM_BODY, f"MJCF body joints: {len(body)}"
    assert tuple(body) == JC.STANDARD_29, (
        "MJCF joint order drifted from joint_contract.STANDARD_29\n"
        f"  mjcf     : {body[:8]}\n  contract : {list(JC.STANDARD_29)[:8]}"
    )
    hands = [n for n in names if n.startswith(("left_hand_", "right_hand_"))]
    assert tuple(hands) == JC.HAND_14


def test_trackerlab_matches_standard():
    """WBC 策略的关节契约 == STANDARD_29。"""
    sys.path.insert(0, str(TRACKERLAB_SRC))
    from trackerTask.supermimic.robots import get_robot  # noqa: PLC0415

    names = tuple(get_robot("g1").joint_names)
    assert names == JC.STANDARD_29, (
        "trackerlab joint contract drifted from joint_contract.STANDARD_29\n"
        f"  trackerlab: {names[:8]}\n  contract  : {list(JC.STANDARD_29)[:8]}"
    )
    assert len(get_robot("g1").action_scale) == JC.NUM_BODY


def test_ha_v31_matches_upstream_constant():
    """HA ``vla_state`` 的顺序 == HA_V31_29（直接读上游源码，避免 import isaaclab）。"""
    text = HA_RUNTIME.read_text(encoding="utf-8")
    m = re.search(r"CANONICAL_G1_JOINT_NAMES_29\s*=\s*\[(.*?)\]", text, re.S)
    assert m, "cannot find CANONICAL_G1_JOINT_NAMES_29 in HA runtime"
    names = tuple(re.findall(r'"([a-z0-9_]+_joint)"', m.group(1)))
    assert len(names) == JC.NUM_BODY, f"HA canonical names: {len(names)}"
    assert names == JC.HA_V31_29, (
        "HumanoidArena vla_state order drifted from joint_contract.HA_V31_29\n"
        f"  HA      : {names[:8]}\n  contract: {list(JC.HA_V31_29)[:8]}"
    )


def test_two_orders_are_actually_different():
    """两套顺序必须真的不同 —— 否则这整套契约就是自欺欺人。"""
    assert JC.STANDARD_29 != JC.HA_V31_29
    perm = JC.permutation("ha_v31", "standard")
    assert perm != list(range(JC.NUM_BODY))


def test_permutation_roundtrip():
    """ha_v31 -> standard -> ha_v31 必须还原。"""
    pytest.importorskip("numpy")
    import numpy as np

    rng = np.random.default_rng(0)
    v = np.arange(JC.NUM_BODY, dtype=np.float64)
    std = JC.to_standard(v, src="ha_v31")
    back = JC.from_standard(std, dst="ha_v31")
    assert np.allclose(back, v)
    # 逐元素按名字核对
    for j, name in enumerate(JC.HA_V31_29):
        assert std[list(JC.STANDARD_29).index(name)] == v[j]
    # 多维同样适用
    batch = rng.normal(size=(4, JC.NUM_BODY))
    assert np.allclose(JC.from_standard(JC.to_standard(batch), dst="ha_v31"), batch)


def test_assert_order_rejects_permuted_names():
    """顺序/名字不对必须当场炸，而不是静默跑。"""
    JC.assert_order(JC.STANDARD_29, "standard", who="test")          # ok
    JC.assert_order(JC.HA_V31_29, "ha_v31", who="test")              # ok
    with pytest.raises(RuntimeError, match="joint ORDER"):
        JC.assert_order(JC.HA_V31_29, "standard", who="test")
    with pytest.raises(RuntimeError, match="joint NAMES"):
        JC.assert_order(JC.STANDARD_29[:-1] + ("bogus_joint",), "standard", who="test")


def test_isaac_provider_uses_the_contract():
    """Isaac provider 不许自带一份关节顺序字面量。"""
    src = HA_PROVIDER.read_text(encoding="utf-8")
    assert "joint_contract" in src, (
        "action_provider_supermimic.py must import joint_contract instead of keeping its own list"
    )
    assert "_CANONICAL_G1_JOINTS" not in src, "provider still carries a private joint list"


def test_mujoco_sim_map_uses_the_contract():
    """MuJoCo 部署侧（groot_repro/g1_sim.py）必须直接用契约，而不是自带字面量。"""
    src = (ROOT / "groot_repro" / "g1_sim.py").read_text(encoding="utf-8")
    assert "joint_contract" in src, "g1_sim.py must import joint_contract"
    assert "_JC.STANDARD_29" in src, "g1_sim.BODY_JOINT_NAMES must come from the contract"

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "groot_repro"))
    import g1_sim  # noqa: PLC0415

    # g1_sim 用 "名字 + _joint" 去 mj_name2id 查，所以这里补后缀再比对
    assert tuple(n + "_joint" for n in g1_sim.BODY_JOINT_NAMES) == JC.STANDARD_29
    assert tuple(g1_sim.HAND_JOINT_NAMES) == JC.HAND_14


def test_sonic_provider_order_is_one_of_the_known_orders():
    """上游 SONIC provider 的 29 关节顺序也必须是契约里的两套之一。

    它虽然按名字映射（``idx_map[n]``），但如果哪天上游换了一套没人认识的顺序，
    我们至少要在这里看到红，而不是等到闭环摔倒再排查。
    """
    src = (ROOT / "third_party" / "HumanoidArena" / "isaaclab_twist2_g1"
           / "action_provider" / "action_provider_sonic.py").read_text(encoding="utf-8")
    m = re.search(r"SONIC_ISAACLAB_JOINT_ORDER\s*=\s*\[(.*?)\]", src, re.S)
    assert m, "SONIC_ISAACLAB_JOINT_ORDER not found"
    names = tuple(re.findall(r'"([a-z0-9_]+_joint)"', m.group(1)))
    assert names in (JC.STANDARD_29, JC.HA_V31_29), (
        f"SONIC provider uses an unknown joint order: {names[:8]}"
    )
