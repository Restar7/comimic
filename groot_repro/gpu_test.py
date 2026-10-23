import os, time
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE","1")
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
import torch
import meta_patch; meta_patch.install()
import gpu_patch; gpu_patch.install()
from gr00t.policy.gr00t_policy import Gr00tPolicy
t0=time.time()
p=Gr00tPolicy(embodiment_tag="unitree_g1_sonic", model_path="/home/wusichen/groot_repro/ckpt_sirius_50k", device="cuda:0", strict=True)
print("loaded in", round(time.time()-t0,1),"s")
dev={}
for n,mod in p.model.named_modules():
    d=str(next(mod.parameters()).device) if list(mod.parameters(recurse=False)) else None
    if d: dev[d]=dev.get(d,0)+1
print("modules by device:", dev)
free,total=torch.cuda.mem_get_info(); print(f"vram free {free/2**30:.2f}/{total/2**30:.2f} GiB")
