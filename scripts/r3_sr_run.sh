#!/usr/bin/env bash
# v13 vs base 成功率实验（VisionNavi / SitSofa），无视频。
#
# 4 个格并行：v13 两个格在 GPU0，base 两个格在 GPU1；每格自带一个 VLA server（跨 seed 复用）
# （避免两个 Isaac 进程共享 server 时的手部反馈串台），格内逐 seed 起 Isaac 进程
# （= 官方单集路径；--episode_batch_json 第 2 集起会 sim_error，已弃用）。
#
# 用法：
#   SEEDS="0 1 2 3 4 5" MAX_STEPS=3000 WAVE=w1 bash scripts/r3_sr_run.sh
set -uo pipefail
C=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
cd "$C" || exit 1

SEEDS=${SEEDS:-"0 1 2 3 4 5"}
MAX_STEPS=${MAX_STEPS:-3000}
WAVE=${WAVE:-w1}
HEAD_V13=${HEAD_V13:-runs/joint_v13/vla_final.pt}
BASE=${C}/runs/closedloop_sr/${WAVE}
mkdir -p "$BASE"

SIT_ID=Isaac-Move-Sit-Sofa-G129-Dex3-Wholebody
VN_ID=Isaac-Move-SmallWarehouse-VisionNavigation-G129-Dex3-Wholebody

echo "[sr] wave=$WAVE seeds='$SEEDS' max_steps=$MAX_STEPS out=$BASE"

TASK=SitSofa    TASK_ID=$SIT_ID HEAD=$HEAD_V13 GPU=0 PORT=18311 CELLDIR=$BASE/sits_v13  LABEL=v13   SEEDS="$SEEDS" MAX_STEPS=$MAX_STEPS bash scripts/r3_sr_cell.sh &
P1=$!
TASK=VisionNavi TASK_ID=$VN_ID  HEAD=$HEAD_V13 GPU=0 PORT=18312 CELLDIR=$BASE/vn_v13    LABEL=v13   SEEDS="$SEEDS" MAX_STEPS=$MAX_STEPS bash scripts/r3_sr_cell.sh &
P2=$!
TASK=SitSofa    TASK_ID=$SIT_ID HEAD=none     GPU=1 PORT=18313 CELLDIR=$BASE/sits_base LABEL=base  SEEDS="$SEEDS" MAX_STEPS=$MAX_STEPS bash scripts/r3_sr_cell.sh &
P3=$!
TASK=VisionNavi TASK_ID=$VN_ID  HEAD=none     GPU=1 PORT=18314 CELLDIR=$BASE/vn_base   LABEL=base  SEEDS="$SEEDS" MAX_STEPS=$MAX_STEPS bash scripts/r3_sr_cell.sh &
P4=$!

echo "[sr] pids: sits_v13=$P1 vn_v13=$P2 sits_base=$P3 vn_base=$P4"
wait $P1; R1=$?
wait $P2; R2=$?
wait $P3; R3=$?
wait $P4; R4=$?
echo "[sr] rc: sits_v13=$R1 vn_v13=$R2 sits_base=$R3 vn_base=$R4"

python3 scripts/r3_sr_summary.py --dir "$BASE" | tee "$BASE/SUMMARY.txt"
echo "[sr] done $(date +%F' '%T)"
