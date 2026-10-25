#!/usr/bin/env bash
# A3 · PPO vs BPTT, fixed wall clock, 2 GPUs (this box has cards 0,1 only).
#
#   ARM=bptt MINUTES=120 bash scripts/run_a3_ppo_vs_bptt.sh
#   ARM=ppo  MINUTES=120 bash scripts/run_a3_ppo_vs_bptt.sh
#   ARM=both MINUTES=120 bash scripts/run_a3_ppo_vs_bptt.sh     # sequential, ~4 h
#
# Both arms share everything except the gradient source:
#   * data          data/npz/ha_tokens            (pure HA, mixed/ft318 rejected upstream)
#   * WBC + world   trackerlab-supermimic/.../ckpt_010500.pt  (the OFFICIAL pinned Isaac ckpt,
#                   see runs/OFFICIAL_SUPERMIMIC.json; parity-verified by scripts/check_isaac_bundle.py)
#   * VLA           runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000
#   * hyper         batch 2, horizon 8, hand loss 0.2 / grasp boost 5, same wall clock
#
# Caveats recorded for the write-up (see docs/ABLATION_ALLOCATION.md §A3):
#   - PPO's effective horizon is 1 (one token chunk per step) -> a T=1 clipped PG, not multi-step GAE.
#   - identical wall clock is what makes the two arms comparable; `--steps` is set huge on purpose.
set -uo pipefail

COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
cd "$COMIMIC"

ARM=${ARM:-both}
MINUTES=${MINUTES:-120}
GPU=${GPU:-0,1}
NPROC=${NPROC:-2}
MOTION=${MOTION:-data/npz/ha_tokens}
SM_CKPT=${SM_CKPT:-trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500.pt}
VLM_PATH=${VLM_PATH:-groot_repro/nvidia/Cosmos-Reason2-2B-weights}
VLM_CKPT=${VLM_CKPT:-runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000}
OUT=${OUT:-runs/ablations/A3_bptt_vs_ppo}
mkdir -p "$OUT"

# tyro/transformers come from gr00t_deps; the fake flash_attn in _noflash forces transformers
# to fall back to sdpa (real FlashAttention rejects the fp32 VLM inputs used for backprop).
export PYTHONPATH="$COMIMIC:$COMIMIC/Isaac-GR00T:$COMIMIC/third_party/gr00t_deps:$COMIMIC/third_party/gr00t_deps/_noflash${PYTHONPATH:+:$PYTHONPATH}"

for p in "$MOTION" "$SM_CKPT" "$VLM_PATH" "$VLM_CKPT"; do
  [[ -e "$p" ]] || { echo "[fail] missing $p"; exit 1; }
done
echo "[a3] arm=$ARM minutes=$MINUTES gpu=$GPU motion=$MOTION sm=$SM_CKPT"
echo "[a3] out=$OUT"

run_bptt() {
  local o="$OUT/bptt"
  mkdir -p "$o"
  CUDA_VISIBLE_DEVICES="$GPU" /usr/bin/python3 -m torch.distributed.run \
    --nproc_per_node "$NPROC" --master_port 29672 scripts/train_joint.py \
    --motion "$MOTION" --supermimic-ckpt "$SM_CKPT" \
    --vlm --vlm-path "$VLM_PATH" --vlm-ckpt "$VLM_CKPT" --vlm-dataset --vlm-steps 2 \
    --batch-size 2 --horizon 8 --steps 1000000 --fm-grad-backend pathwise \
    --time-budget-minutes "$MINUTES" --hand-loss-weight 0.2 --hand-grasp-boost 5 \
    --eval-every 300 --eval-batches 2 --eval-out "$o/eval_curve.jsonl" \
    --save --save-every 300 --save-vla-every 0 --out "$o"
}

run_ppo() {
  local o="$OUT/ppo"
  mkdir -p "$o"
  CUDA_VISIBLE_DEVICES="$GPU" /usr/bin/python3 -m torch.distributed.run \
    --nproc_per_node "$NPROC" --master_port 29674 scripts/train_ppo.py \
    --motion "$MOTION" --supermimic-ckpt "$SM_CKPT" \
    --vlm --vlm-path "$VLM_PATH" --vlm-ckpt "$VLM_CKPT" --vlm-steps 2 \
    --batch-size 2 --horizon 8 --ppo-epochs 4 --sigma-init 0.25 --steps 1000000 \
    --time-budget-minutes "$MINUTES" --hand-loss-weight 0.2 \
    --eval-every 100 --eval-batches 2 --eval-out "$o/eval_curve.jsonl" --out "$o"
}

case "$ARM" in
  bptt) run_bptt ;;
  ppo)  run_ppo ;;
  both) run_bptt; echo "[a3] bptt arm done at $(date +%H:%M:%S), starting ppo"; run_ppo ;;
  *) echo "ARM must be bptt|ppo|both"; exit 1 ;;
esac
echo "[a3] arm=$ARM finished at $(date +%H:%M:%S)"
