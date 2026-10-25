#!/usr/bin/env bash
# SuperMimic 整网微调（HA token 条件 + 速度通道修复）—— 2×RTX4090，Isaac Sim 5.0 + IsaacLab 2.2。
#
# 与底模 `bones_seed_token_ha_v2` **同一套栈**（`third_party/isaacsim_env` + `third_party/isaaclab_220`），
# 不是 5.1/2.3.2 那条（那条只用于更早的 BONES-SEED run）。
#
# 两段式（用户定的）：
#   STAGE=1  —— **锁定速度通道/保持端口 token**：条件用 `data/token_motions/ha_port`（端口 token 侧车），
#              从 `ckpt_010500_physint.pt`（残差→显式积分 + 800 步速度标定）续 300 iter，验证 sim_rew 不掉；
#   STAGE=2  —— **切 HA 标签 token**：条件换 `data/token_motions/ha`（`g1_motion_token_64` 原样），
#              从 stage1 的 last.pt 续 2500 iter，验证能不能把 sim_rew 救回 ≥0.84。
#
# 用法：
#   bash scripts/launch_sm_finetune.sh 1        # 300 iter
#   bash scripts/launch_sm_finetune.sh 2        # 2500 iter（resume stage1 的 last.pt）
set -uo pipefail

COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
REPO=$COMIMIC/trackerlab-supermimic
PY=${PY:-$COMIMIC/third_party/isaacsim_env/bin/python}     # Isaac Sim 5.0.0 + IsaacLab 2.2.0
STAGE=${1:?usage: launch_sm_finetune.sh <1|2>}
NPROC=${NPROC:-2}
ENVS_PER_RANK=${ENVS_PER_RANK:-4096}
PORT_BASE=${PORT_BASE:-29690}
CKPT_EVERY=${CKPT_EVERY:-100}

export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export SUPERMIMIC_EXPLICIT_INTEGRATION=1                   # 显式积分结构（与 physint ckpt 一致）
# 机器人资产/驱动必须与底模 `bones_seed_token_ha_v2` 完全一致（该 run 的 launcher 日志实测）：
#   robot asset: g1_29dof_rev_1_0.usd | drive: HA/SONIC PD + HA default pose (HA_ALIGN=1)
# 不设这两个变量会回落到默认 29-DoF 驱动 → policy 直接崩（term≈0.95），不是模型的问题。
export SUPERMIMIC_HA_ALIGN=1
export SUPERMIMIC_ROBOT_USD=${SUPERMIMIC_ROBOT_USD:-$REPO/data/assetslib/unitree/unitree_g1/usd/g1_29dof_rev_1_0.usd}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
cd "$REPO"

case "$STAGE" in
  1)
    MOTION=../data/token_motions/ha_port
    RESUME=logs/bones_seed_token_ha_v2/ckpt_010500_physint.pt
    ITERATIONS=$((10500 + ${ITERS:-300}))
    LOG_DIR=logs/ha_stage1_port_${TAG:-$(date +%m%d_%H%M)}
    ;;
  2)
    MOTION=../data/token_motions/ha
    RESUME=${RESUME:-$(ls -dt logs/ha_stage1_port_* 2>/dev/null | head -1)/last.pt}
    ITERATIONS=$((10500 + ${ITERS:-300} + ${ITERS2:-2500}))
    LOG_DIR=logs/ha_stage2_hatoken_${TAG:-$(date +%m%d_%H%M)}
    ;;
  *) echo "[fail] unknown STAGE=$STAGE"; exit 2;;
esac

echo "[sm-finetune] stage=$STAGE  isaac=$(basename $PY)  motion=$MOTION"
echo "[sm-finetune] resume=$RESUME  iterations=$ITERATIONS  envs=${NPROC}x${ENVS_PER_RANK}"
echo "[sm-finetune] log_dir=$LOG_DIR  ckpt_every=$CKPT_EVERY"

exec "$PY" -m torch.distributed.run --nproc_per_node "$NPROC" --master_port "$PORT_BASE" \
  scripts/rls/supermimic/train.py \
    --task g1 --interface token \
    --motion_dir "$MOTION" --no-compact_library \
    --goal_frames 10 --hist 10 --horizon 32 \
    --utd 0.4 --lr 5e-5 --no-compile \
    --num_envs "$ENVS_PER_RANK" --iterations "$ITERATIONS" \
    --shard_motion --shard_mode interleave \
    --buffer_steps 1000 --warmup_steps 64 --wm_only_iters ${WM_ONLY:-500} \
    --wm_batches 10 --wm_batch 2048 --pi_batch 4096 \
    --ckpt_every "$CKPT_EVERY" --log_every 10 \
    --amp bf16 --resume "$RESUME" \
    --log_dir "$LOG_DIR" --kit_args=--allow-root
