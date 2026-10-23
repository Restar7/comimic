import os, numpy as np, mujoco
os.environ.setdefault("MUJOCO_GL", "glfw")
import g1_sim as G
from sonic_loop import SonicController, pd_torque, CONTROL_DT

SCENE = "/home/wusichen/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
m = mujoco.MjModel.from_xml_path(SCENE); d = mujoco.MjData(m)
mujoco.mj_resetData(m, d)
d.qpos[0:3] = [0,0,0.75]; d.qpos[3:7] = [1,0,0,0]
d.qpos[7:36] = G.DEFAULT_ANGLES; d.qpos[36:50] = 0
mujoco.mj_forward(m, d)

ctrl = SonicController("sonic_models/model_decoder.onnx")
token = np.zeros(64)
sub = int(round(CONTROL_DT / m.opt.timestep))
for tick in range(150):          # 3 s
    q = d.qpos[7:36].copy(); dq = d.qvel[6:35].copy()
    quat = d.qpos[3:7].copy()
    # angular velocity in body frame: MuJoCo free joint qvel[3:6] is in body frame already
    ang = d.qvel[3:6].copy()
    q_target = ctrl.step(token, q, dq, quat, ang)
    for _ in range(sub):
        q = d.qpos[7:36]; dq = d.qvel[6:35]
        d.ctrl[:29] = np.clip(pd_torque(q_target, q, dq), -88, 88)
        d.ctrl[29:] = 0.0
        mujoco.mj_step(m, d)
    if tick % 25 == 0:
        tilt = np.degrees(np.arccos(np.clip(G.quat_to_rotmat(d.qpos[3:7])[2,2],-1,1)))
        print(f"t={tick*CONTROL_DT:.2f}s z={d.qpos[2]:.3f} tilt={tilt:.1f}deg")
print("final z", round(float(d.qpos[2]),3), "tilt", round(float(np.degrees(np.arccos(np.clip(G.quat_to_rotmat(d.qpos[3:7])[2,2],-1,1)))),1))
