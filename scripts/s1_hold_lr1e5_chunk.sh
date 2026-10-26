#!/usr/bin/env bash
# A' 臂：hold token，lr 1e-5，从 16000 起，1000 步；跑完对 16500/17000 各做一次官方 HA-699 抽检
set -uo pipefail
C=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
R=$C/trackerlab-supermimic
cd $R
export SONIC_ENCODER_ONNX=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RESUME=$C/runs/wbc_v2_label/wbc_v2.pt OBS_MODE=paper_obs93 FREEZE_WM=0 NO_CRITIC=1 WM_ONLY_ITERS=0 \
LR=1e-5 MODE=train NPROC=2 ENVS_PER_RANK=8192 ITERATIONS=17000 MAX_EPISODE_STEPS=0 \
LIBRARY=$R/data/bones_seed/bones_seed_plus_ha_cmd_x20.pt INTERFACE=token COMPILE=1 HA_ALIGN=1 \
CKPT_EVERY=250 PORT_BASE=29760 LOG_DIR=logs/wbc_hold_lr1e5 \
bash scripts/rls/supermimic/train_bones_seed.sh > $C/logs/s1/wbc_hold_lr1e5.launcher.log 2>&1
for IT in 16500 17000; do
  CK=logs/wbc_hold_lr1e5/ckpt_0$IT.pt
  [ -f "$CK" ] || continue
  ../third_party/trackerlab_env/bin/python scripts/rls/supermimic/evaluate.py \
    --run_dir logs/wbc_hold_lr1e5 --ckpt "$CK" \
    --motion_packed $C/runs/isaac_motion/ha_tokens/packed.pt --device cuda:0 --kit_args=--allow-root \
    > $C/logs/s1/eval_lr1e5_$IT.log 2>&1
  cp logs/wbc_hold_lr1e5/ckpt_0${IT}_eval.json $C/runs/S1_gates/lr1e5_HA699_$IT.json 2>/dev/null
done
