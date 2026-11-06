#!/usr/bin/env bash
# 启动官方 GR00T VLA latent64 服务（L1 闭环要用）
set -uo pipefail
C=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
cd "$C" || exit 1
export PYTHONPATH="$C/third_party/gr00t_deps:$C/Isaac-GR00T:$C"
export NO_ALBUMENTATIONS_UPDATE=1
export HF_HOME=/inspire/hdd/global_user/liumengfan-253108110079/cache/huggingface
exec python3 scripts/ha_vla_latent64_server.py \
  --ckpt runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000 \
  --vlm-path groot_repro/nvidia/Cosmos-Reason2-2B-weights \
  --port 18080
