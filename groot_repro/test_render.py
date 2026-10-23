import os, numpy as np, mujoco, cv2
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
renderer = mujoco.Renderer(m, 480, 640)
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
cam.fixedcamid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, 'head_camera')
w = cv2.VideoWriter("out/test_stand.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 25, (640,480))
for tick in range(250):  # 5 s at 50 Hz
    q = d.qpos[7:36].copy(); dq = d.qvel[6:35].copy()
    q_target = ctrl.step(token, q, dq, d.qpos[3:7].copy(), d.qvel[3:6].copy())
    for _ in range(sub):
        d.ctrl[:29] = np.clip(pd_torque(q_target, d.qpos[7:36], d.qvel[6:35]), -88, 88)
        d.ctrl[29:] = 0.0
        mujoco.mj_step(m, d)
    if tick % 2 == 0:
        renderer.update_scene(d, cam)
        w.write(renderer.render()[..., ::-1])
w.release()
print("tilt final:", round(float(np.degrees(np.arccos(np.clip(G.quat_to_rotmat(d.qpos[3:7])[2,2],-1,1)))),1),
      "z", round(float(d.qpos[2]),3))
