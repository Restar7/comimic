#!/usr/bin/env bash
# A' 采集之后的三步流水线（2026-11-03）
#   1) 合并全部分片 + 逐行过滤（默认 5cm）
#   2) 用新转移刷新 WM（pretrain_wm，6000 步）
#   3) 校验：imagined vs real 的 20 步 headroom
#
# 用法：
#   bash scripts/run_a_prime_pipeline.sh                 # 全跑
#   WM_STEPS=200 bash scripts/run_a_prime_pipeline.sh    # 冒烟
set -uo pipefail
C=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
DIR=${DIR:-$C/runs/a_prime_seg20}
MAX_DRIFT=${MAX_DRIFT:-0.05}
WM_STEPS=${WM_STEPS:-6000}
JOINT_CKPT=${JOINT_CKPT:-$C/runs/S1_handover/WBC_A24000_wmvel_cmd.pt}
OUT_WM=${OUT_WM:-$DIR/wm_pre.pt}
GPU=${GPU:-0}

cd "$C" || exit 1
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export SUPERMIMIC_HA_ALIGN=1
# headroom 的 VLA 支路需要 gr00t 依赖（diffusers/transformers）
export PYTHONPATH="$C/third_party/gr00t_deps:$C/Isaac-GR00T:$C:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-/inspire/hdd/global_user/liumengfan-253108110079/cache/huggingface}
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export SUPERMIMIC_ROBOT_USD=${SUPERMIMIC_ROBOT_USD:-$C/third_party/HumanoidArena/isaaclab_twist2_g1/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1.0.usd}
export SUPERMIMIC_ROBOT_USD=${SUPERMIMIC_ROBOT_USD/rev_1.0/rev_1_0}

echo "=== [1/4] 合并 + 过滤（逐行，阈值 ${MAX_DRIFT} m）==="
python3 scripts/post_a_prime_seg20.py --dir "$DIR" --pattern "*_off*.pt" \
        --max-drift "$MAX_DRIFT" || exit 10
echo

echo "=== [2/4] 刷新 WM（${WM_STEPS} 步）==="
CUDA_VISIBLE_DEVICES=$GPU python3 scripts/pretrain_wm.py \
    --joint-ckpt "$JOINT_CKPT" \
    --transitions "$DIR/filtered.pt" \
    --steps "$WM_STEPS" --no-stage1 --stage3-steps 0 \
    --device cuda --out "$OUT_WM" || exit 20
echo

echo "=== [3/4] 真回报（critic 用，δ_pos=${REWARD_DELTA_POS:-0.10}）==="
REWARD_DELTA_POS=${REWARD_DELTA_POS:-0.10}
python3 scripts/build_real_returns.py \
    --transitions "$DIR/filtered.pt" --reward-delta-pos "$REWARD_DELTA_POS" \
    --out "$DIR/real_returns.pt" || exit 25
echo

echo "=== [4/4] headroom 校验（imag vs real, H=20）==="
if [[ -f scripts/probe_headroom.py ]]; then
  CUDA_VISIBLE_DEVICES=$GPU python3 scripts/probe_headroom.py \
      --wbc-ckpt "$JOINT_CKPT" --windows "${HEADROOM_WINDOWS:-32}" \
      --device cuda:$GPU || exit 30
else
  echo "[warn] 没找到 scripts/probe_headroom.py，跳过（列出候选）:"
  ls scripts/*headroom* scripts/*wm*eval* 2>/dev/null | head -5
fi
echo "=== 完成 ==="
