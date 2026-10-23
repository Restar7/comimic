"""Verify every checkpoint tensor actually lands in the loaded model."""
import json, torch
import meta_patch; meta_patch.install()
import gr00t.model  # register
from transformers import AutoModel
from safetensors import safe_open

CKPT="/home/wusichen/groot_repro/ckpt_sirius_50k"
model, info = AutoModel.from_pretrained(CKPT, output_loading_info=True)
print("missing_keys:", len(info["missing_keys"]), info["missing_keys"][:5])
print("unexpected_keys:", len(info["unexpected_keys"]), info["unexpected_keys"][:5])
print("mismatched:", len(info.get("mismatched_keys", [])))

sd = model.state_dict()
idx = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
handles = {s: safe_open(f"{CKPT}/{s}", "pt") for s in set(idx.values())}
import random; random.seed(0)
keys = random.sample(sorted(idx), 12)
worst = 0.0
for k in keys:
    ref = handles[idx[k]].get_tensor(k)
    got = sd[k]
    diff = (ref.to(torch.float32) - got.to(torch.float32)).abs().max().item()
    worst = max(worst, diff)
    print(f"  {k}: max|diff|={diff:.3e} shape={tuple(got.shape)}")
print("WORST max|diff| over sample:", worst)
print("meta tensors left:", sum(1 for p in model.parameters() if p.device.type == "meta"))
