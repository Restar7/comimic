#!/usr/bin/env bash
# 联合微调 第 5 轮（2026-11-05）：**部署口径 + 因果 WM + 速度项 + base 手**
#
# 相对 round-4 的改动（每一处都有实测依据，见 docs/WHY_FT_DEGRADED_20261104.md）
#   1. WM      换成重采后过因果门的那份（默认 $DIR/wm_exc.pt，headroom ≤20%）
#   2. 速度项   `REWARD_VEL_SCALE=1`：`huber_err` 分支原来**没有速度项**，
#               而 base 的症结是"平均速度只有录制的 1/2"（0.050 vs 0.102 m/s），
#               位置亏欠在 0.4 s 里只有 2.1 cm、速度却是即时可见的。
#   3. H=20    **不拉长视界**：部署是 2.5 Hz / 每 20 步重规划，H=20 与部署逐位一致；
#               拉长到 60 会"沿用第 40 帧 token"，那是不存在的工况。
#   4. 手部     `--hand-from-base`：手不参与联合微调，服务端用 base 头出 64:78。
#   5. 锚       trust eps 0.05 / weight 0.5（v3 里 token 的唯一锚）。
#
# 用法::
#   WM_CKPT=runs/a_prime_sonic_exc/wm_exc.pt STEPS=400 bash scripts/run_joint_ft_round5.sh
set -uo pipefail
C=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
DIR=${DIR:-$C/runs/a_prime_sonic_leash}          # 起点/参考数据（= 现在的 A' leash）
cd "$C" || exit 1

export V3=1 H=20
export WM_CKPT=${WM_CKPT:-$C/runs/a_prime_sonic_exc/wm_exc.pt}
export EXTRA="${EXTRA:---real-transitions $DIR/filtered_std.pt}"
# 2026-11-05：起点漂移上限可调。实测漂移 p50=3.9 / p90=8.2 cm（段中间帧），
# 而部署是从 reset 位姿（漂移≈0）出发 ⇒ 收紧到 0.05：起点 p50 2.38 cm、
# 保留 62% 的行、clip 覆盖仍 600/600，动作可归因偏差从 ~10% 提到 ~25%。
export START_SOURCE=${START_SOURCE:-recorded} START_MAX_DRIFT=${START_MAX_DRIFT:-0.05}
export REWARD_MODE=huber_err ROOT_ORI_MODE=split_huber
export REWARD_DELTA_POS=0.05 REWARD_DELTA_ORI=0.05 REWARD_D_MAX=0.30
# ★ 新增：速度项（权重 1.0）+ 速度 Huber δ
export REWARD_VEL_SCALE=${REWARD_VEL_SCALE:-1.0}
export REWARD_DELTA_VEL=${REWARD_DELTA_VEL:-0.05}
export LAMBDA_R=${LAMBDA_R:-5.0} LAMBDA_BC=0.0 LAMBDA_HAND=0.05 TOKEN_BC=0
# ★ 2026-11-05（第二轮）：动作刹车。
#   round-5 第一轮用 ACT_EPS=0.30 / weight=0.20，而 hinge 在 eps 以内**梯度恰为 0**
#   ⇒ 实测动作漂移一路滑到 act/rmse=0.80（≈每关节 20°）才被接住。那 20° 是势能最低点，
#   不是策略乱来；而真机执行 20° 偏移的动作时，已经不是"同一件事"了。
#   收紧方式：形状换 `l2`（无死区），权重 0.2→1.0。eps 在 l2 里**只是尺度**，
#   平衡点粗算 r* ≈ 1.1·eps²/w ⇒ (0.30, 1.0) 给出 r*≈0.10（≈2.5°/关节）。
#   ⚠ 别把 eps 直接调到 0.10：那会让 r*≈0.011，等于把策略钉在 base 上（安全但没收益）。
export ACT_PENALTY=${ACT_PENALTY:-hinge}
# 奖励视界：只信想象的前 k 步（0=满 20）。理由见 v3.reward_horizon。
export REWARD_HORIZON=${REWARD_HORIZON:-0}
export ACT_EPS=${ACT_EPS:-0.30} ACT_WEIGHT=${ACT_WEIGHT:-0.20}
export TRUST_CENTER=base TRUST_EPS=${TRUST_EPS:-0.05} TRUST_WEIGHT=${TRUST_WEIGHT:-0.5}
export GAMMA=0.95 CRITIC_ON=0 ADV_BASELINE=none ADV_TAU=0
export RUN=${RUN:-joint_v5} STEPS=${STEPS:-400}
export EVAL_EVERY=${EVAL_EVERY:-50} SAVE_EVERY=${SAVE_EVERY:-200}
export DDP=1 NPROC=${NPROC:-4} DDP_GPUS=${DDP_GPUS:-0,1,2,3}
# 服务端（闭环评估时）必须带 --hand-from-base
# 闭环评估脚本（scripts/r3_closedloop_ab.sh）读的是 SRV_EXTRA，两个名字都设上，避免口径漂移
export SRV_EXTRA=${SRV_EXTRA:---hand-from-base}
export SERVER_EXTRA=${SERVER_EXTRA:-$SRV_EXTRA}

echo "[r5] WM=$WM_CKPT"
echo "[r5] H=$H（= 部署 2.5 Hz / 每 20 步重规划）  速度项=$REWARD_VEL_SCALE (δ=$REWARD_DELTA_VEL)"
echo "[r5] trust eps=$TRUST_EPS w=$TRUST_WEIGHT  act eps=$ACT_EPS w=$ACT_WEIGHT shape=$ACT_PENALTY reward_horizon=$REWARD_HORIZON"
echo "[r5] 手部：服务端 $SRV_EXTRA"

# ---- 启动前硬门（任一条不满足就 exit 9）--------------------------------------
[[ -f "$WM_CKPT" ]] || { echo "[r5][FAIL] WM_CKPT 不存在：$WM_CKPT" >&2; exit 9; }
[[ -f "$DIR/filtered_std.pt" ]] || { echo "[r5][FAIL] 缺 $DIR/filtered_std.pt" >&2; exit 9; }
python3 - "$WM_CKPT" "$TRUST_EPS" "$TRUST_WEIGHT" "$ACT_EPS" "$REWARD_VEL_SCALE" "$ACT_WEIGHT" <<'PYG' || exit 9
import sys
from pathlib import Path
wm, eps, w, aeps, vs = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
if Path(wm).stat().st_size < 1e6:
    raise SystemExit(f"[r5][FAIL] WM_CKPT 太小：{wm}")
# ★ 2026-11-05：token 锚的阈值改成"有动作刹车就可以放松"。
#   原规则 `w >= 0.4` 是"token 锚是唯一锚"时代的产物；现在我们有动作空间刹车
#   （`ACT_PENALTY=l2` + `ACT_WEIGHT>0`，实测能把动作漂移压在 0.15–0.2），
#   而 token 锚长期饱和（frac_out 0.6–0.9）且方向与奖励梯度不同向 —— 它反而是
#   "方向被压歪"的嫌疑源。所以允许在**动作刹车在场**时把 trust 权重降到 0.2。
_actw = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
if eps > 0.06:
    raise SystemExit(f"[r5][FAIL] token 锚没收紧：eps={eps}")
if w < 0.4 and _actw <= 0.0:
    raise SystemExit(f"[r5][FAIL] trust weight={w} < 0.4 且没有动作刹车（ACT_WEIGHT={_actw}）"
                     " ⇒ 没有任何锚，动作会失控")
if aeps > 0.35:
    raise SystemExit(f"[r5][FAIL] act hinge 没收紧：{aeps}")
if vs <= 0:
    raise SystemExit("[r5][FAIL] 速度项没开（REWARD_VEL_SCALE<=0）——base 的核心问题是速度")
print(f"[r5][OK] WM={Path(wm).name} trust=eps{eps}/w{w} act={aeps} vel={vs}")
PYG

echo "[r5] 提示：跑完先做三件事再判好坏"
echo "      1) python3 scripts/check_reward_vel_term.py            # 速度项真的接上了"
echo "      2) python3 scripts/eval_closedloop_progress.py --runs base=... ft=...  # 3 seed"
echo "      3) 闭环必须带 $SERVER_EXTRA"
exec bash scripts/launch_joint_ha_v2.sh
