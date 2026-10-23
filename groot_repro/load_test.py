import os, time, resource
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE","1")
os.environ.setdefault("HF_HUB_OFFLINE","1")
os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
import torch
import meta_patch; meta_patch.install()
from gr00t.policy.gr00t_policy import Gr00tPolicy

t0=time.time()
policy = Gr00tPolicy(
    embodiment_tag="unitree_g1_sonic",
    model_path="/home/wusichen/groot_repro/ckpt_sirius_50k",
    device="cpu",
    strict=True,
)
print("policy ready in", round(time.time()-t0,1), "s")
print("rss GB:", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1048576,2))
print("modality keys:", {k: v.modality_keys for k,v in policy.modality_configs.items()})
print("action horizon:", policy.modality_configs["action"].delta_indices[-1]+1)
