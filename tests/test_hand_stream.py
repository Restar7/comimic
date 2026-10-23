"""Tests for the decoupled 14-DoF hand channel (``diff_vla_wbc.hand_stream``).

The hand channel mirrors SONIC/``gear_sonic``'s separate hand stream: the body
policy keeps its 29-DoF action space and the hands are driven by an independent
14-D joint-angle channel.  These tests pin down the two things that are easy to
get wrong: the **interleaved actuator layout** of the 43-DoF MJCF, and the
**mirrored joint signs** of the right hand.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from diff_vla_wbc.hand_stream import (
    HAND_DIM,
    HAND_JOINT_NAMES,
    HAND_POSES,
    G1BodyIndex,
    HandStream,
    HandStreamCfg,
    load_hand_model,
)

ROOT = Path(__file__).resolve().parents[1]
HAND_XML = ROOT / "reproduce_scenes" / "g1" / "g1_29dof_with_hand.xml"
BODY_XML = ROOT / "GR00T-WholeBodyControl" / "gear_sonic_deploy" / "g1" / "g1_29dof.xml"


def _joint_ranges(model) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    jids = [names.index(n) for n in HAND_JOINT_NAMES]
    return model.jnt_range[jids, 0], model.jnt_range[jids, 1]


class HandStreamLayoutTest(unittest.TestCase):
    def test_actuator_layout_is_interleaved(self):
        model = load_hand_model(HAND_XML)
        hand = HandStream(model, HandStreamCfg())
        self.assertTrue(hand.enabled)
        # XML-specific regression: hands sit between the left and right arm banks.
        self.assertEqual(hand.ctrl_ids, [22, 23, 24, 25, 26, 27, 28, 36, 37, 38, 39, 40, 41, 42])
        self.assertEqual(len(hand.info()["body_ctrl_ids"]), 29)

    def test_body_only_model_disables_channel(self):
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(BODY_XML))
        hand = HandStream(model, HandStreamCfg())
        self.assertFalse(hand.enabled)
        body = np.arange(29, dtype=np.float64)
        ctrl = hand.assemble_ctrl(body)
        self.assertEqual(ctrl.shape, (29,))
        np.testing.assert_allclose(ctrl, body)
        # Disabled channel never writes hand torque.
        np.testing.assert_allclose(hand.pd_torque(None), np.zeros(HAND_DIM))

    def test_assemble_ctrl_scatters_by_name(self):
        model = load_hand_model(HAND_XML)
        hand = HandStream(model, HandStreamCfg())
        body = np.arange(29, dtype=np.float64)
        fingers = np.arange(1, HAND_DIM + 1, dtype=np.float64)
        ctrl = hand.assemble_ctrl(body, fingers)
        self.assertEqual(ctrl.shape, (43,))
        np.testing.assert_allclose(ctrl[0:22], body[0:22])
        np.testing.assert_allclose(ctrl[29:36], body[22:29])
        np.testing.assert_allclose(ctrl[22:29], fingers[:7])
        np.testing.assert_allclose(ctrl[36:43], fingers[7:])
        # No actuator is written twice: zero body + zero hand -> all zeros.
        np.testing.assert_allclose(hand.assemble_ctrl(np.zeros(29), np.zeros(HAND_DIM)), 0.0)

    def test_pose_values_respect_joint_limits(self):
        model = load_hand_model(HAND_XML)
        lo, hi = _joint_ranges(model)
        for name, pose in HAND_POSES.items():
            self.assertEqual(pose.shape, (HAND_DIM,), name)
            self.assertTrue(
                np.all((pose >= lo - 1e-9) & (pose <= hi + 1e-9)),
                f"pose {name} exceeds joint limits",
            )

    def test_right_hand_is_mirrored(self):
        # Same grasp must have opposite signs on the mirrored right hand.
        grasp = HAND_POSES["grasp"]
        np.testing.assert_allclose(grasp[:7], -grasp[7:])


class G1BodyIndexTest(unittest.TestCase):
    def test_body_state_is_not_contiguous_on_43dof(self):
        model = load_hand_model(HAND_XML)
        idx = G1BodyIndex(model)
        np.testing.assert_array_equal(idx.qpos_idx, list(range(7, 29)) + list(range(36, 43)))
        np.testing.assert_array_equal(idx.ctrl_idx, list(range(0, 22)) + list(range(29, 36)))

    def test_body_ctrl_leaves_hands_untouched(self):
        model = load_hand_model(HAND_XML)
        idx = G1BodyIndex(model)
        ctrl = idx.assemble_ctrl(np.arange(29, dtype=np.float64))
        np.testing.assert_allclose(ctrl[[22, 23, 24, 25, 26, 27, 28, 36, 37, 38, 39, 40, 41, 42]], 0.0)
        np.testing.assert_allclose(ctrl[29:36], np.arange(29, dtype=np.float64)[22:29])

    def test_body_index_on_29dof_model(self):
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(BODY_XML))
        idx = G1BodyIndex(model)
        np.testing.assert_array_equal(idx.qpos_idx, np.arange(7, 36))
        np.testing.assert_array_equal(idx.ctrl_idx, np.arange(29))


class HandStreamSourceTest(unittest.TestCase):
    def setUp(self):
        self.model = load_hand_model(HAND_XML)

    def test_slew_rate_limits_step(self):
        hand = HandStream(self.model, HandStreamCfg(pose="open", slew_rate=5.0))
        hand.set_pose("fist")
        before = hand.targets()
        after = hand.step(0.02)
        delta = np.abs(after - before)
        self.assertLessEqual(float(delta.max()), 5.0 * 0.02 + 1e-9)
        self.assertGreater(float(delta.max()), 0.0)

    def test_immediate_pose_jumps(self):
        hand = HandStream(self.model, HandStreamCfg(pose="open"))
        hand.set_pose("grasp", immediate=True)
        np.testing.assert_allclose(hand.targets(), HAND_POSES["grasp"])

    def test_motion_source_indexes_and_clamps(self):
        ramp = np.linspace(-0.5, 0.5, 5 * HAND_DIM).reshape(5, HAND_DIM)
        lo, hi = _joint_ranges(self.model)
        expected = np.clip(ramp, lo, hi)  # motion source clamps to joint limits
        hand = HandStream(self.model, HandStreamCfg(source="motion", motion=ramp))
        np.testing.assert_allclose(hand.targets(0), expected[0])
        np.testing.assert_allclose(hand.targets(3), expected[3])
        np.testing.assert_allclose(hand.targets(99), expected[-1])  # frame clamped
        np.testing.assert_allclose(hand.targets(-5), expected[0])

    def test_motion_source_requires_array(self):
        with self.assertRaises(ValueError):
            HandStream(self.model, HandStreamCfg(source="motion"))

    def test_unknown_pose_raises(self):
        hand = HandStream(self.model, HandStreamCfg())
        with self.assertRaises(KeyError):
            hand.set_pose("high_five")


class HandStreamSettleTest(unittest.TestCase):
    """Closed-loop MuJoCo check: the hand PD actually reaches a grasp pose."""

    def test_grasp_settles(self):
        import mujoco

        from diff_vla_wbc.compat import G1_KD, G1_KP, g1_default_joint_pos

        model = load_hand_model(HAND_XML)
        model.opt.timestep = 0.002
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        data = mujoco.MjData(model)

        q0 = g1_default_joint_pos().numpy().astype(np.float64)
        kp = G1_KP.numpy().astype(np.float64)
        kd = G1_KD.numpy().astype(np.float64)
        data.qpos[7:36] = q0
        mujoco.mj_forward(model, data)

        hand = HandStream(model, HandStreamCfg(pose="open", kp=12.0, kd=0.5, slew_rate=40.0))
        hand.set_pose("grasp")
        goal = HAND_POSES["grasp"].copy()

        dt = 0.002
        for _ in range(1500):  # 3 s
            q = data.qpos[7:36]
            dq = data.qvel[6:35]
            body_tau = kp * (q0 - q) - kd * dq
            hand.step(dt)
            data.ctrl[:] = hand.assemble_ctrl(body_tau, data=data)
            mujoco.mj_step(model, data)
            self.assertTrue(np.all(np.isfinite(data.qpos)))

        q_hand, _ = hand.read_state(data)
        err = np.abs(q_hand - goal)
        self.assertLess(float(err.max()), 0.25, f"hand tracking error too large: {err}")
        self.assertLess(float(np.mean(err)), 0.08)

        lo, hi = _joint_ranges(model)
        self.assertTrue(np.all(q_hand >= lo - 0.05) and np.all(q_hand <= hi + 0.05))

    def test_open_pose_holds(self):
        import mujoco

        from diff_vla_wbc.compat import G1_KD, G1_KP, g1_default_joint_pos

        model = load_hand_model(HAND_XML)
        model.opt.timestep = 0.002
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        data = mujoco.MjData(model)
        q0 = g1_default_joint_pos().numpy().astype(np.float64)
        kp, kd = G1_KP.numpy(), G1_KD.numpy()
        data.qpos[7:36] = q0
        mujoco.mj_forward(model, data)

        hand = HandStream(model, HandStreamCfg(pose="open", kp=12.0, kd=0.5))
        for _ in range(500):
            q, dq = data.qpos[7:36], data.qvel[6:35]
            data.ctrl[:] = hand.assemble_ctrl(kp * (q0 - q) - kd * dq, data=data)
            mujoco.mj_step(model, data)

        q_hand, _ = hand.read_state(data)
        self.assertLess(float(np.abs(q_hand).max()), 0.05)


if __name__ == "__main__":
    unittest.main()
