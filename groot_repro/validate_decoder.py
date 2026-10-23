"""Feed recorded motion tokens through the SONIC decoder and compare with the
recorded WBC joint command (action.wbc)."""
import numpy as np, pyarrow.parquet as pq
import g1_sim as G
from sonic_loop import Decoder

ROOT = "/home/wusichen/SONIC_DATA/2026-07-14/datasets/2026-07-14-20-30-01"
t = pq.read_table(f"{ROOT}/data/chunk-000/episode_000000.parquet")
raw_q = np.asarray(t["observation.state"].to_pylist(), np.float64)
raw_a = np.asarray(t["action.wbc"].to_pylist(), np.float64)
grav  = np.asarray(t["observation.projected_gravity"].to_pylist(), np.float64)
tok   = np.asarray(t["action.motion_token"].to_pylist(), np.float64)
quat  = np.asarray(t["observation.root_orientation"].to_pylist(), np.float64)  # wxyz

BODY = list(range(0, 22)) + list(range(29, 36))   # dataset order -> MuJoCo order
q_mj, a_mj = raw_q[:, BODY], raw_a[:, BODY]

def ang_vel(quats, dt=0.02):
    out = np.zeros((len(quats), 3))
    for i in range(1, len(quats)-1):
        q0, q1 = quats[i-1], quats[i+1]
        # dq/dt ≈ (q1 - q0)/2dt in quaternion space
        dq = (q1 - q0) / (2*dt)
        w = q0[0]*dq[1:] - dq[0]*q0[1:] + np.cross(q0[1:], dq[1:])  # not exact, fine for viz
        out[i] = 2*w
    return out

dec = Decoder("sonic_models/model_decoder.onnx")
H = 10
q_norm = q_mj - G.DEFAULT_ANGLES                      # what the C++ logger stores
q_is  = q_norm[:, G.MUJOCO_TO_ISAACLAB]
dq_is = np.gradient(q_mj, axis=0)[:, G.MUJOCO_TO_ISAACLAB] * 50.0
w_is  = ang_vel(quat)

hist_q=[q_is[0]]*H; hist_dq=[dq_is[0]]*H; hist_a=[np.zeros(29)]*H
hist_g=[grav[0]]*H; hist_w=[w_is[0]]*H
pred=[]
for i in range(len(tok)):
    obs=np.concatenate([tok[i]]+[np.concatenate(h) for h in (hist_w,hist_q,hist_dq,hist_a,hist_g)])
    a=dec(obs)
    pred.append(G.DEFAULT_ANGLES + a[G.ISAACLAB_TO_MUJOCO]*G.ACTION_SCALE)
    hist_q=(hist_q+[q_is[i]])[-H:]; hist_dq=(hist_dq+[dq_is[i]])[-H:]
    hist_a=(hist_a+[a])[-H:]; hist_g=(hist_g+[grav[i]])[-H:]; hist_w=(hist_w+[w_is[i]])[-H:]
pred=np.array(pred)

for skip in (0, 50, 200):
    p,tgt=pred[skip:],a_mj[skip:]
    print(f"from {skip}: MAE={np.degrees(np.abs(p-tgt).mean()):.2f} deg  corr={np.corrcoef(p.ravel(),tgt.ravel())[0,1]:.3f}")
print("pred[0][:6]:", np.round(pred[0][:6],3))
print("targ[0][:6]:", np.round(a_mj[0][:6],3))
