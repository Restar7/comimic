#!/usr/bin/env bash
# 等 P1b-A（exec token 版）产出交付 ckpt 后，立刻开跑 P1b-B（双库 token 版：
# ref=执行参考库，token=命令流库）。B 才是"能接 VLA/部署"的配方。
set -uo pipefail
C=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
cd "$C" || exit 1
A_CKPT=$C/runs/S1_handover/exp_WBC_exec.pt
echo "[chain] waiting for $A_CKPT ..."
for _ in $(seq 1 480); do
  [[ -f "$A_CKPT" ]] && break
  sleep 30
done
[[ -f "$A_CKPT" ]] || { echo "[chain] FATAL: A ckpt never appeared"; exit 1; }
echo "[chain] A deliverable ready: $(ls -la "$A_CKPT")"
sleep 20
MODE=full TAG=cmd ENVS_PER_RANK=4096 \
TOKEN_LIB_TRAIN=$C/runs/isaac_motion/ha_cmd/packed.pt \
OUT_CKPT=$C/runs/S1_handover/exp_WBC_exec_cmd.pt \
  bash scripts/p1b_finetune_wbc_exec.sh
echo "[chain] B exit=$?"
