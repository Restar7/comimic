"""PD track the recorded WBC joint targets directly (no decoder)."""
import os, numpy as np, mujoco, cv2, pyarrow.parquet as pq
os.environ.setdefault("MUJOCO_GL", "glfw")
import g1_sim as G
from sonic_loop import pd_torque

ROOT="/home/wusichen/SONIC_DATA/2026-07-14/datasets/2026-07-14-20-30-01"
t=pq.read_table(f"{ROOT}/data/chunk-000/episode_000000.parquet")
raw_q=np.asarray(t["observation.state"].to_pylist(),float)
raw_a=np.asarray(t["action.wbc"].to_pylist(),float)
BODY=list(range(0,22))+list(range(29,36))
q_mj=raw_q[:,BODY]; a_mj=raw_a[:,BODY]

m=mujoco.MjModel.from_xml_path("/home/wusichen/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml")
d=mujoco.MjData(m)
force = np.array([88,88,88,139,50,50]*2 + [88,50,50] + [25,25,25,25,25,5,5]*2, float)
mujoco.mj_resetData(m,d)
d.qpos[0:3]=[0,0,0.78]; d.qpos[3:7]=[1,0,0,0]
d.qpos[7:36]=q_mj[0]; d.qpos[36:50]=0
mujoco.mj_forward(m,d)
sub=int(round(0.02/m.opt.timestep))
for tick in range(300):
    qt=a_mj[tick]
    for _ in range(sub):
        d.ctrl[:29]=np.clip(pd_torque(qt,d.qpos[7:36],d.qvel[6:35]),-force,force); d.ctrl[29:]=0
        mujoco.mj_step(m,d)
    if tick%50==0:
        tilt=np.degrees(np.arccos(np.clip(G.quat_to_rotmat(d.qpos[3:7])[2,2],-1,1)))
        print(f"t={tick*0.02:.1f}s z={d.qpos[2]:.3f} tilt={tilt:.1f} track_err={np.abs(d.qpos[7:36]-qt).max():.3f}")
print("final z", round(float(d.qpos[2]),3))
