#!/usr/bin/env bash
# 会话 03 的统一运行环境：worktree 代码 + 主仓库（只读）数据/权重
set -euo pipefail
WT=${WT:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-wt03}
MAIN=${COMIMIC_MAIN:-/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic}
export COMIMIC=$MAIN COMIMIC_MAIN=$MAIN
# shellcheck disable=SC1091
source "$MAIN/scripts/deps_env.sh" >/dev/null 2>&1
export PYTHONPATH="$WT:$MAIN/trackerlab-supermimic/source/trackerLab:$MAIN/trackerlab-supermimic/source/trackerTask:$PYTHONPATH"
export SUPERMIMIC_HA_ALIGN=1
export SUPERMIMIC_ROBOT_USD="$MAIN/trackerlab-supermimic/data/assetslib/unitree/unitree_g1/usd/g1_29dof_rev_1_0.usd"
export NO_ALBUMENTATIONS_UPDATE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
