import os, numpy as np, mujoco
os.environ.setdefault("MUJOCO_GL", "glfw")
import g1_sim as G

SCENE = "/home/wusichen/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)
print("nq", m.nq, "nu", m.nu, "dt", m.opt.timestep)

def reset():
    mujoco.mj_resetData(m, d)
    d.qpos[0:3] = [0.0, 0.0, 0.75]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    d.qpos[7:36] = G.DEFAULT_ANGLES
    d.qpos[36:50] = 0.0
    mujoco.mj_forward(m, d)

reset()
q_target = G.DEFAULT_ANGLES.copy()
for step in range(1500):  # 3 s at 500 Hz
    q = d.qpos[7:36]
    dq = d.qvel[6:35]
    tau = G.KP * (q_target - q) - G.KD * dq
    d.ctrl[:29] = np.clip(tau, -88, 88)
    d.ctrl[29:] = 0.0
    mujoco.mj_step(m, d)

print("base z after 3s:", round(float(d.qpos[2]), 4))
print("tilt deg:", round(float(np.degrees(np.arccos(np.clip(G.quat_to_rotmat(d.qpos[3:7])[2,2],-1,1)))),2))
print("max joint err:", round(float(np.abs(d.qpos[7:36]-q_target).max()),3))
