"""G1 29-DoF 关节顺序的**唯一**契约（single source of truth）。

这个文件存在的唯一目的：让"关节顺序"只在这里定义一次，其他地方一律 import，
不许再各写一份字面量。2026-10-24 的闭环就是栽在两份顺序不一致上（动作打到错误关节，
机器人 1 秒就摔，而且看起来像"WBC 太弱"）。

两套顺序
--------
* ``STANDARD_29``（物理/策略顺序）
  MuJoCo MJCF 的关节声明顺序（``reproduce_scenes/g1/g1_29dof_with_hand.xml``），
  也是 trackerlab ``get_robot("g1").joint_names`` 的顺序（训练时按名字重排到这个顺序），
  也是 HumanoidArena 的 ``TWIST2_G1_JOINT_NAMES_29``。
  形如：左腿 → 右腿 → 腰 → 左臂 → 右臂。

* ``HA_V31_29``（HumanoidArena v3.1 的 ``vla_state`` 顺序）
  HumanoidArena ``action_provider/vla_smpl_runtime.py`` 里的
  ``CANONICAL_G1_JOINT_NAMES_29``，也是 Isaac 资产 ``robot.data.joint_names``
  里这 29 个 body 关节的实际顺序。
  形如：左髋pitch → 右髋pitch → 腰yaw → 左髋roll → 右髋roll → …（左右交错）

**名字上的坑**：HA 把交错顺序叫 "canonical"，而 trackerlab/MuJoCo 把另一套叫 canonical。
所以本项目里禁止使用 "canonical" 这个词，统一叫 ``STANDARD_29`` / ``HA_V31_29``。

用法
----
    from joint_contract import STANDARD_29, HA_V31_29, to_standard, from_standard

    q_std = to_standard(q_ha, src=HA_V31_29)        # 交错 -> 标准
    q_ha  = from_standard(q_std, dst=HA_V31_29)     # 标准 -> 交错

    perm = permutation(src=HA_V31_29, dst=STANDARD_29)   # perm[i] = src 里第 i 个在 dst 的位置

运行时**必须**用 :func:`assert_order` 显式校验名字列表；对不上就当场报错并打印差异，
绝不允许静默错位（这就是"锁死"）。
"""
from __future__ import annotations

from typing import Iterable, Sequence

# --------------------------------------------------------------------------- 两套顺序
#: 物理/策略顺序：MuJoCo MJCF 声明顺序 == trackerlab g1 契约 == HA TWIST2 顺序
STANDARD_29: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

#: HumanoidArena v3.1 的 vla_state / Isaac 资产顺序（HA 自己叫它 "canonical"）
HA_V31_29: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

#: 14 维手关节（左右各 7），MuJoCo MJCF 与 HA 资产都是这个顺序
HAND_14: tuple[str, ...] = tuple(
    f"{side}_hand_{finger}_joint"
    for side in ("left", "right")
    for finger in ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")
)

NUM_BODY = len(STANDARD_29)      # 29
NUM_HANDS = len(HAND_14)         # 14

_KNOWN = {"standard": STANDARD_29, "ha_v31": HA_V31_29}


# --------------------------------------------------------------------------- 基础工具
def _resolve(order: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(order, str):
        try:
            return _KNOWN[order]
        except KeyError as exc:  # noqa: TRY003
            raise KeyError(f"unknown joint order {order!r}; expected one of {sorted(_KNOWN)}") from exc
    return tuple(order)


def permutation(src: str | Sequence[str] = "ha_v31",
                dst: str | Sequence[str] = "standard") -> list[int]:
    """把 ``src`` 顺序的向量变成 ``dst`` 顺序时用的索引。

    ``dst_vec[i] = src_vec[perm[i]]``，即 ``perm[i]`` 是"输出第 i 个（dst 顺序）
    应该取自 src 的第几个"。
    """
    s, d = _resolve(src), _resolve(dst)
    if len(s) != len(d) or set(s) != set(d):
        missing = sorted(set(s) - set(d))
        extra = sorted(set(d) - set(s))
        raise ValueError(f"joint sets differ: missing={missing} extra={extra}")
    pos = {name: i for i, name in enumerate(s)}
    return [pos[name] for name in d]


def reorder(vec, src: str | Sequence[str] = "ha_v31",
            dst: str | Sequence[str] = "standard"):
    """按名字把最后一维从 ``src`` 顺序重排到 ``dst`` 顺序（numpy 数组或 list of lists）。"""
    perm = permutation(src, dst)
    if hasattr(vec, "shape") and len(getattr(vec, "shape", ())) == 1:
        return vec[perm]
    return [row[perm] for row in vec]


def to_standard(vec, src: str | Sequence[str] = "ha_v31"):
    return reorder(vec, src=src, dst=STANDARD_29)


def from_standard(vec, dst: str | Sequence[str] = "ha_v31"):
    return reorder(vec, src=STANDARD_29, dst=dst)


def assert_order(names: Iterable[str], expected: str | Sequence[str], *, who: str) -> None:
    """运行时闸门：名字列表必须**恰好**等于某个已知顺序，否则立刻失败。

    这是"锁死"的核心：任何 provider / 数据脚本拿到关节名字后都要先过这一关，
    对不上就报错，不允许静默错位。
    """
    got = tuple(names)
    exp = _resolve(expected)
    if got == exp:
        return
    if set(got) != set(exp):
        missing = [n for n in exp if n not in set(got)]
        extra = [n for n in got if n not in set(exp)]
        raise RuntimeError(
            f"[joint_contract] {who}: joint NAMES do not match {expected!r}. "
            f"missing={missing[:6]} extra={extra[:6]}"
        )
    # 同一批名字、不同顺序：把"应该是什么"直接打出来
    perm = [got.index(n) for n in exp]
    raise RuntimeError(
        f"[joint_contract] {who}: joint ORDER does not match {expected!r}.\n"
        f"  expected: {list(exp)[:8]} ...\n"
        f"  got     : {list(got)[:8]} ...\n"
        f"  perm (expected[i] sits at got index): {perm}\n"
        f"  fix: build the permutation with joint_contract.permutation(src={expected!r}, dst=<your order>)"
    )


def match_order(names: Iterable[str], *, who: str = "") -> str:
    """返回 ``names`` 到底是哪套顺序（``"ha_v31"`` / ``"standard"``），否则抛错。

    比 :func:`assert_order` 更好用：调用方拿到运行时关节名后不必先假设是哪套，
    直接问它，然后用 :func:`permutation` 转换。
    """
    got = tuple(names)
    for key in ("ha_v31", "standard"):
        if got == _KNOWN[key]:
            return key
    assert_order(got, "standard", who=who or "match_order")   # 给出同顺序/缺名字的详细报错
    raise AssertionError("unreachable")


def describe() -> str:
    return (f"STANDARD_29[0:6]={list(STANDARD_29[:6])} ...\n"
            f"HA_V31_29[0:6]={list(HA_V31_29[:6])} ...\n"
            f"perm(ha_v31 -> standard)={permutation('ha_v31', 'standard')}")
