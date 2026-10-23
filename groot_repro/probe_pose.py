import numpy as np, mujoco
import g1_sim as G
SCENE = "/home/wusichen/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
m = mujoco.MjModel.from_xml_path(SCENE); d = mujoco.MjData(m)
def set_pose(z):
    mujoco.mj_resetData(m, d)
    d.qpos[0:3] = [0,0,z]; d.qpos[3:7] = [1,0,0,0]
    d.qpos[7:36] = G.DEFAULT_ANGLES; d.qpos[36:50] = 0
    mujoco.mj_forward(m, d)
    return d
for z in (0.78, 0.75, 0.72, 0.70):
    set_pose(z)
    # lowest geom z among foot-related bodies
    zs = [(d.geom_xpos[i][2], mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)) for i in range(m.ngeom)]
    zs.sort()
    print(f"base z={z}: lowest geoms", [(round(a,3), b) for a,b in zs[:3]])
print("total mass", m.body_subtreemass[1], "kg")
