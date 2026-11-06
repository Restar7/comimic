#!/usr/bin/env bash
# 联合微调第 2 轮：修掉 act hinge 的量级问题（H=20 的轨迹差 eps 不能按单步 0.10）
set -uo pipefail
C=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
DIR=${DIR:-$C/runs/a_prime_seg20}
cd "$C" || exit 1
export V3=1 H=20
export REWARD_DELTA_POS=${REWARD_DELTA_POS:-0.10}
export CRITIC_ON=1 CRITIC_REAL="$DIR/critic_real.pt"
export EXTRA="${EXTRA:---real-transitions $DIR/filtered.pt}"
export ACT_EPS=${ACT_EPS:-0.40}     # H=20 轨迹差的合理量级（原 0.10 → 100% 出界）
export ACT_WEIGHT=${ACT_WEIGHT:-0.30}
export DDP=1 NPROC=${NPROC:-4} DDP_GPUS=${DDP_GPUS:-0,1,2,3}
export RUN=${RUN:-joint_v3_round2} STEPS=${STEPS:-300}
# ★ 2026-11-03：第 1 轮**根本没加载刷新过的 WM**（launcher 的 WM_CKPT 默认空 ⇒
#   FT 用的还是 WBC ckpt 里那份 qd_corr=0.369 的旧 WM）。这里显式指向
#   多 seed 扫+held-out 选出来的 wm_pre_full.pt（qd_corr=0.676 / ratio 0.996）。
export WM_CKPT=${WM_CKPT:-$DIR/wm_pre_full.pt}
export EVAL_EVERY=${EVAL_EVERY:-100} SAVE_EVERY=${SAVE_EVERY:-100}

# ★★ 2026-11-04 硬门禁（把踩过的坑锁死）：
#   1) `WM_CKPT` 之前默认空 ⇒ 微调**根本没加载刷新过的 WM**，300 步白跑；
#   2) `ACT_EPS` 太小 ⇒ act hinge 100% 出界（frac_out=1.0）、梯度全是噪声；
#   3) 数据目录必须过契约门（有转移 / 不摔 / 没进废数据）。
[[ -f "${WM_CKPT:-}" ]] || { echo "[joint-r2][FAIL] WM_CKPT 不存在：${WM_CKPT:-<空>}" >&2; exit 4; }
python3 - "$WM_CKPT" "$ACT_EPS" <<'PYG'
import sys, torch
p, eps = sys.argv[1], float(sys.argv[2])
ck = torch.load(p, map_location="cpu", weights_only=False)
if "wm" not in ck:
    raise SystemExit(f"[joint-r2][FAIL] {p} 里没有 'wm' 权重")
rec = ck.get("refresh_recipe")
print(f"[joint-r2] WM={p.split('/')[-1]} recipe={rec}")
if eps < 0.20:
    raise SystemExit(f"[joint-r2][FAIL] ACT_EPS={eps} 太小：H=20 的轨迹差量级约 0.4，"
                     "过小会让 hinge 全部出界（实测 frac_out=1.0、loss/act≈8）")
PYG
bash scripts/check_all_contracts.sh "${DIR}" >/tmp/joint_r2_contracts.log 2>&1 || {
  echo "[joint-r2][FAIL] 契约门没过，见 /tmp/joint_r2_contracts.log" >&2; tail -5 /tmp/joint_r2_contracts.log >&2; exit 5; }
echo "[joint-r2] 契约门 PASS"

echo "[joint-r2] ACT_EPS=$ACT_EPS ACT_WEIGHT=$ACT_WEIGHT RUN=$RUN STEPS=$STEPS"
exec bash scripts/launch_joint_ha_v2.sh
