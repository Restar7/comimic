#!/usr/bin/env bash
# A 线：采 on-policy 真转移（GT / ORACLE 两臂），输出到 runs/wm_real/onpolicy_{arm}_dd_v2.pt
# 直接用 $ISAAC_VENV/bin/python（不要 isaaclab.sh：TERM=dumb 下会失败），不设 CUDA_VISIBLE_DEVICES。
set -u
COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
cd "$COMIMIC" || exit 1
source scripts/isaac_env.sh >/dev/null 2>&1
unset CUDA_VISIBLE_DEVICES
export TERM=xterm
export PYTHONPATH="$COMIMIC:$COMIMIC/trackerlab-supermimic/source/trackerTask:$COMIMIC/trackerlab-supermimic/source/trackerLab:${PYTHONPATH:-}"

CLIPS=${CLIPS:-"HOI_double_desk__tw__ep000000 HOI_double_desk__tw__ep000001 HOI_double_desk__tw__ep000002 HOI_double_desk__tw__ep000003 HOI_pp_box__yb__ep000000"}
ARMS=${ARMS:-gt}
ENVS=${ENVS:-16}
STEPS=${STEPS:-300}

for CLIP in $CLIPS; do
  for ARM in $ARMS; do
    tag=$(echo "$CLIP" | sed 's/HOI_//; s/__tw__/_/; s/__yb__/_/')
    echo "=== collect $CLIP/$ARM start $(date +%T) ==="
    "$ISAAC_VENV/bin/python" -u scripts/isaac_g1_collect.py \
      --token-npy "runs/arm_tokens/$CLIP/$ARM.npy" --clip "$CLIP" \
      --num-envs "$ENVS" --steps "$STEPS" --log-every 200 \
      --out "runs/wm_real/onpolicy_${ARM}_${tag}.pt" --headless
    echo "=== collect $CLIP/$ARM done rc=$? $(date +%T) ==="
  done
done
