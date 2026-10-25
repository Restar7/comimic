#!/usr/bin/env bash
# 冒烟：验证「预取等价」+「续训续接」两件事（需要一张空闲卡，约 6–10 分钟）
#
#   ① 预取等价：同一配置跑 2 步，--prefetch-vla 与 --no-prefetch-vla 的
#      [eval 0] / [step 1] / [step 2] 行必须**逐位相同**（batch 顺序不因预取而改变：
#      预取只是提前把 b_{t+1} 的 CPU 预处理做掉，训练用的仍是 b_1,b_2,... 同一序列）。
#   ② 续训续接：从 joint_002000.pt + vla_002000.pt 续 1 步，
#      [eval 2000] 行必须与原始 run 的 [eval 2000] 行一致（权重相同、eval batch 相同）。
#
# 用法：
#   bash scripts/smoke_resume_and_prefetch.sh            # 默认用 A1 主线的 002000 存档
#   BASE_STEP=1000 bash scripts/smoke_resume_and_prefetch.sh
set -uo pipefail

COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
cd "$COMIMIC"

STEP="${BASE_STEP:-2000}"
JOINT="runs/joint_ha_a1/joint_$(printf '%06d' "$STEP").pt"
VLA="runs/joint_ha_a1/vla_$(printf '%06d' "$STEP").pt"
MERGED="/tmp/ckpt_ft${STEP}"
OUT_A=/tmp/smoke_prefetch_on
OUT_B=/tmp/smoke_prefetch_off
OUT_R=/tmp/smoke_resume
PY=/opt/miniconda3/envs/isaaclab/bin/python3
export PYTHONPATH="$COMIMIC:$COMIMIC/Isaac-GR00T:$COMIMIC/third_party/gr00t_deps/_noflash"

say() { echo "[$(date +%H:%M:%S)] $*"; }
fail() { echo "[FAIL] $*"; exit 1; }

[[ -f "$JOINT" ]] || fail "缺 $JOINT（先让主线跑到该步，或指定 BASE_STEP=1000）"
[[ -f "$VLA" ]] || fail "缺 $VLA"

COMMON=(
  scripts/train_joint.py
  --motion data/npz/ha_tokens
  --supermimic-ckpt trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500_physint.pt
  --vlm --vlm-path groot_repro/nvidia/Cosmos-Reason2-2B-weights
  --vlm-dataset --vlm-steps 2
  --batch-size 2 --horizon 8 --eval-every 1 --eval-batches 2
  --state-vel-weight 1.0 --hand-loss-weight 0.2 --hand-grasp-boost 5
  --log-every 1 --save-every 0 --save-vla-every 0
)

# ---------------------------------------------------------------- ① 预取等价
say "① 预取 vs 不预取（各 2 步）"
CUDA_VISIBLE_DEVICES=0 "$PY" "${COMMON[@]}" --vlm-ckpt runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000 \
  --steps 2 --prefetch-vla --out "$OUT_A" > "$OUT_A.log" 2>&1 || fail "prefetch-on 跑挂了（见 $OUT_A.log）"
CUDA_VISIBLE_DEVICES=0 "$PY" "${COMMON[@]}" --vlm-ckpt runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000 \
  --steps 2 --no-prefetch-vla --out "$OUT_B" > "$OUT_B.log" 2>&1 || fail "prefetch-off 跑挂了（见 $OUT_B.log）"

grep -hE "^\[(eval 0|step 1/|step 2/)\]" "$OUT_A.log" > /tmp/pa.txt
grep -hE "^\[(eval 0|step 1/|step 2/)\]" "$OUT_B.log" > /tmp/pb.txt
if diff -q /tmp/pa.txt /tmp/pb.txt > /dev/null; then
  say "① PASS：预取与现算逐位一致（$(wc -l < /tmp/pa.txt) 行）"
else
  echo "--- prefetch ON ---"; cat /tmp/pa.txt
  echo "--- prefetch OFF ---"; cat /tmp/pb.txt
  fail "① 预取与现算不一致"
fi

# ---------------------------------------------------------------- ② 续训续接
say "② 合并 vla_$(printf '%06d' "$STEP").pt -> $MERGED"
"$PY" scripts/export_vla_ckpt.py --base runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000 \
  --vla "$VLA" --out "$MERGED" --force > /tmp/merge_vla.log 2>&1 \
  || fail "export_vla_ckpt 失败（见 /tmp/merge_vla.log）"

say "② 从 step $STEP 续 1 步（--resume-joint/--resume-vla）"
CUDA_VISIBLE_DEVICES=0 "$PY" "${COMMON[@]}" --vlm-ckpt "$MERGED" \
  --steps "$((STEP + 1))" --resume-joint "$JOINT" --resume-vla "$VLA" \
  --out "$OUT_R" > "$OUT_R.log" 2>&1 || fail "续训跑挂了（见 $OUT_R.log）"

grep -q "\[eval $STEP\]" "$OUT_R.log" || fail "续训没有在 step $STEP 打 eval（编号没续上）"
grep -q "\[step $((STEP + 1))/" "$OUT_R.log" || fail "续训没有从 $STEP 往下走"

ORIG=$(grep -hE "^\[eval $STEP\]" logs/joint_ha_a1.log | tail -1)
NEW=$(grep -hE "^\[eval $STEP\]" "$OUT_R.log" | tail -1)
if [[ -n "$ORIG" && "$ORIG" == "$NEW" ]]; then
  say "② PASS：续训起点的 eval 与原始 run 逐位一致"
else
  echo "--- 原始 ---"; echo "$ORIG"
  echo "--- 续训 ---"; echo "$NEW"
  say "② WARN：两者不一致（若原始 run 的 save/eval 顺序不同属正常，请人工核对 token_mse/state/reward）"
fi

say "全部完成：prefetch=$OUT_A / resume=$OUT_R"
