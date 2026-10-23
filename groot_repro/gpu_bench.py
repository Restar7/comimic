import os, time
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE","1"); os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
import numpy as np, torch
import meta_patch; meta_patch.install()
import gpu_patch; gpu_patch.install()
from gr00t.policy.gr00t_policy import Gr00tPolicy
import pyarrow.parquet as pq, cv2
ROOT="/home/wusichen/SONIC_DATA/2026-07-14/datasets/2026-07-14-20-30-01"
t=pq.read_table(f"{ROOT}/data/chunk-000/episode_000000.parquet")
q=np.asarray(t["observation.state"].to_pylist(),float)[1000]
cap=cv2.VideoCapture(f"{ROOT}/videos/chunk-000/observation.images.ego_view/episode_000000.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES,1000); ok,bgr=cap.read(); cap.release()
rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
BODY=list(range(0,22))+list(range(29,36)); qm=q[BODY]
state={"left_leg":qm[0:6],"right_leg":qm[6:12],"waist":qm[12:15],"left_arm":qm[15:22],
       "right_arm":qm[22:29],"left_hand":q[22:29],"right_hand":q[36:43],"projected_gravity":np.array([0,0,-1.0])}
obs={"video":{"ego_view":rgb[None,None]},"state":{k:np.asarray(v,np.float32)[None,None] for k,v in state.items()},
     "language":{"annotation.human.task_description":[["Pick up the apple from the table."]]}}
p=Gr00tPolicy(embodiment_tag="unitree_g1_sonic",model_path="/home/wusichen/groot_repro/ckpt_sirius_50k",device="cuda:0",strict=True)
for i in range(2):
    t0=time.time(); a,_=p.get_action(obs); print(f"infer {i}: {time.time()-t0:.2f}s token range [{a['motion_token'].min():.2f},{a['motion_token'].max():.2f}]", flush=True)
free,total=torch.cuda.mem_get_info(); print(f"vram free {free/2**30:.2f}/{total/2**30:.2f} GiB")
