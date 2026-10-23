import os, numpy as np, mujoco, cv2, pyarrow.parquet as pq
os.environ.setdefault("MUJOCO_GL","glfw")
import g1_sim as G
from sonic_loop import SonicController, pd_torque, CONTROL_DT, HISTORY

ROOT="/home/wusichen/SONIC_DATA/2026-07-14/datasets/2026-07-14-20-30-01"
t=pq.read_table(f"{ROOT}/data/chunk-000/episode_000000.parquet")
raw_q=np.asarray(t["observation.state"].to_pylist(),float)
tok=np.asarray(t["action.motion_token"].to_pylist(),float)
grav=np.asarray(t["observation.projected_gravity"].to_pylist(),float)
BODY=list(range(0,22))+list(range(29,36)); q_mj=raw_q[:,BODY]
q_is=(q_mj-G.DEFAULT_ANGLES)[:,G.MUJOCO_TO_ISAACLAB]
dq_is=np.gradient(q_mj,axis=0)[:,G.MUJOCO_TO_ISAACLAB]*50.0

SCENE="/home/wusichen/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
m=mujoco.MjModel.from_xml_path(SCENE); d=mujoco.MjData(m)
mujoco.mj_resetData(m,d)
d.qpos[0:3]=[0,0,0.781]; d.qpos[3:7]=[1,0,0,0]
d.qpos[7:36]=q_mj[0]; d.qpos[36:50]=0
mujoco.mj_forward(m,d)

ctrl=SonicController("sonic_models/model_decoder.onnx", init_q=q_mj[0])
ctrl.hist.clear()
for i in range(HISTORY):
    ctrl.hist.append({"base_ang_vel":np.zeros(3),"q":q_is[i],"dq":dq_is[i],
                      "last_action":np.zeros(29),"gravity":grav[i]})
sub=int(round(CONTROL_DT/m.opt.timestep))
renderer=mujoco.Renderer(m,480,640)
cam=mujoco.MjvCamera(); cam.type=mujoco.mjtCamera.mjCAMERA_FIXED
cam.fixedcamid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_CAMERA,'head_camera')
w=cv2.VideoWriter("out/replay_recorded2.mp4",cv2.VideoWriter_fourcc(*"mp4v"),25,(640,480))
for tick in range(300):
    qt=ctrl.step(tok[tick], d.qpos[7:36].copy(), d.qvel[6:35].copy(), d.qpos[3:7].copy(), d.qvel[3:6].copy())
    for _ in range(sub):
        d.ctrl[:29]=np.clip(pd_torque(qt,d.qpos[7:36],d.qvel[6:35]),-88,88); d.ctrl[29:]=0
        mujoco.mj_step(m,d)
    if tick%2==0:
        renderer.update_scene(d,cam); w.write(renderer.render()[...,::-1])
    if tick%50==0:
        tilt=np.degrees(np.arccos(np.clip(G.quat_to_rotmat(d.qpos[3:7])[2,2],-1,1)))
        print(f"t={tick*0.02:.1f}s z={d.qpos[2]:.3f} tilt={tilt:.1f}")
w.release()
