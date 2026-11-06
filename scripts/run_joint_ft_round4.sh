#!/usr/bin/env bash
# 联合微调 第 4 轮（2026-11-04，**收紧 token 信任域**）
#
# 为什么是"收紧 trust"而不是"加 token BC"
# ------------------------------------------
# v3 路径里 **`lambda_bc` 根本不生效**（`v3_step.py:214` 会对非零值打 warn：
# "[AUDIT][v3][warn] lambda_bc=... 在 v3 路径不生效"）。v3 里 token 64 维**没有任何标签监督**，
# 唯一的锚就是 token 空间的信任域（`TRUST_CENTER` / `TRUST_WEIGHT` / `TRUST_EPS`）。
# round-3（A 臂）用的是 weight 0.05 / eps 0.15 ⇒ 实测 token 在**同分布输入**上
# `cos(tok,GT)` 从 0.987 掉到 0.663，真机净位移 1.006 → 0.111 m。
#
# 本轮唯一改动：把那个锚收紧到 eps 0.05 / weight 0.5，动作 hinge 从 0.50 收到 0.30；
# 裁判换成刚过门的 `wm_ssm_s0_gain.pt`（多步 ratio 0.97/0.97、corr 0.993、终点 1.11 cm）。
# 其余（λ_r=5、奖励口径、起点分布、γ）与 round-3 逐位相同，保证可比。
set -uo pipefail
C=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
DIR=${DIR:-$C/runs/a_prime_sonic_leash}
cd "$C" || exit 1
[[ -f "$DIR/filtered_std.pt" ]] || { echo "[r4] 缺 $DIR/filtered_std.pt" >&2; exit 2; }

export V3=1 H=20
export WM_CKPT=${WM_CKPT:-$DIR/wm_ssm_s0_gain.pt}
export EXTRA="${EXTRA:---real-transitions $DIR/filtered_std.pt}"
export START_SOURCE=recorded START_MAX_DRIFT=0.10
export REWARD_MODE=huber_err ROOT_ORI_MODE=split_huber
export REWARD_DELTA_POS=0.05 REWARD_DELTA_ORI=0.05 REWARD_D_MAX=0.30
export REWARD_VEL_SCALE=0
export LAMBDA_R=${LAMBDA_R:-5.0} LAMBDA_BC=0.0 LAMBDA_HAND=0.05 TOKEN_BC=0
# ---- 唯一改动：信任域收紧（token 锚）+ 动作 hinge 收紧 ----
export TRUST_CENTER=base
export TRUST_WEIGHT=${TRUST_WEIGHT:-0.5}
export TRUST_EPS=${TRUST_EPS:-0.05}
export ACT_EPS=${ACT_EPS:-0.30} ACT_WEIGHT=${ACT_WEIGHT:-0.20}
export GAMMA=0.95 CRITIC_ON=0
export ADV_BASELINE=none ADV_TAU=0
export RUN=${RUN:-joint_v4_trusttight} STEPS=${STEPS:-400}
export EVAL_EVERY=${EVAL_EVERY:-50} SAVE_EVERY=${SAVE_EVERY:-200}
export DDP=1 NPROC=${NPROC:-4} DDP_GPUS=${DDP_GPUS:-0,1,2,3}

# ---- 铁律自检：锚必须真的收紧了，且裁判是过门的那份 ----
[[ -f "$WM_CKPT" ]] || { echo "[r4][FAIL] WM_CKPT 不存在: $WM_CKPT" >&2; exit 9; }
python3 - "$WM_CKPT" "$TRUST_EPS" "$TRUST_WEIGHT" "$ACT_EPS" <<'PYG'
import sys
from pathlib import Path
wm, eps, w, aeps = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
if Path(wm).stat().st_size < 1e6:
    raise SystemExit(f"[r4][FAIL] WM_CKPT 太小: {wm}")
if "wm_ssm_s0_gain" not in wm:
    print(f"[r4][warn] 裁判不是过了增益/保真门的 wm_ssm_s0_gain.pt：{wm}")
if eps > 0.06 or w < 0.4:
    raise SystemExit(f"[r4][FAIL] 信任域没收紧：eps={eps} weight={w}")
if aeps > 0.35:
    raise SystemExit(f"[r4][FAIL] 动作 hinge 没收紧：ACT_EPS={aeps}")
print(f"[r4][OK] WM={Path(wm).name} trust=eps{eps}/w{w} act_eps={aeps}")
PYG
[[ $? -ne 0 ]] && exit 9

echo "[r4] WM=$WM_CKPT"
echo "[r4] trust: center=$TRUST_CENTER eps=$TRUST_EPS weight=$TRUST_WEIGHT  (token 的唯一锚)"
echo "[r4] act: eps=$ACT_EPS weight=$ACT_WEIGHT | λ_r=$LAMBDA_R λ_hand=$LAMBDA_HAND γ=$GAMMA"
echo "[r4] steps=$STEPS save_every=$SAVE_EVERY eval_every=$EVAL_EVERY"
exec bash scripts/launch_joint_ha_v2.sh
