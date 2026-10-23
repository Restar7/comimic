#!/usr/bin/env bash
# 2×H200 离线训练入口（token 架构）：只是给 launch_token_train_4xh200.sh 钉死 NPROC=2 / PROFILE=paper。
# 单点真相仍在 launch_token_train_4xh200.sh（token 参数、TF32、离线环境变量都在那里），本文不复制任何参数。
#
# 用法（工作目录随意，脚本自己找 REPO）：
#   bash scripts/launch_token_train_2xh200.sh                    # 正式 320k（论文口径）
#   ITER=200 RUN=bench_2xh200 bash scripts/launch_token_train_2xh200.sh    # 200 迭代基准
#   bash scripts/launch_token_train_2xh200.sh --resume logs/e9_bones_seed_token_h200x2/ckpt_latest.pt
#   ENVS=8192 BATCH=4096 WMB=4096 bash scripts/launch_token_train_2xh200.sh   # 自定义覆盖
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NPROC="${NPROC:-2}"
export PROFILE="${PROFILE:-paper}"

exec bash "$HERE/launch_token_train_4xh200.sh" "$@"
