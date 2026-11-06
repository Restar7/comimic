#!/usr/bin/env bash
# 新数据 → WM 重训 → **三道门** → 打印判决（2026-11-05）
#
# 门（缺一不可）
#   1) 多步保真：20 步 imag/real ratio ∈ [0.8, 1.2]（`probe_wm_multistep.py`）
#   2) **因果门**：自由优化动作的 headroom ≤ 20%（`probe_wm_headroom.py --gate`）
#   3) 通道可分辨：位置/速度通道的 误差÷单步变化（`probe_wm_channels.py`，只打印）
#
# 任一门不过 ⇒ exit 3，**不要**开联合微调（目标函数还是假的）。
#
# 用法::
#   DIR=runs/a_prime_sonic_exc bash scripts/wm_retrain_from_new_data.sh
#   FAST=1 ...        # 跳过"追加 3000 步"，省 ~19 min（门可能过不利索）
set -uo pipefail
C=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
cd "$C" || exit 1

DIR=${DIR:-$C/runs/a_prime_sonic_exc}
TAG=${TAG:-exc}
INIT=${INIT:-$C/runs/a_prime_sonic_leash/wm_ssm_s0_gain.pt}
OUT_WM=${OUT_WM:-$DIR/wm_${TAG}.pt}
FAST=${FAST:-0}
export SUPERMIMIC_HA_ALIGN=1
export SUPERMIMIC_ROBOT_USD=${SUPERMIMIC_ROBOT_USD:-$C/third_party/HumanoidArena/isaaclab_twist2_g1/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
GPU=${GPU:-0}

[[ -d "$DIR" ]] || { echo "[wm-re][FAIL] 数据目录不存在: $DIR" >&2; exit 2; }
ls "$DIR"/*_off*.pt >/dev/null 2>&1 || { echo "[wm-re][FAIL] $DIR 下没有 *_off*.pt" >&2; exit 2; }
[[ -f "$INIT" ]] || { echo "[wm-re][FAIL] 初值 WM 不存在: $INIT" >&2; exit 2; }

say() { echo; echo "=== $* ==="; }

say "0/4 建窗口（k=20）"
python3 scripts/build_wm_roll_windows.py --dir "$DIR" --k 20 --stride 4 \
  --out "$DIR/roll_windows.pt" || exit 3

say "1/4 单步 refresh（新数据，2000 步）"
python3 - "$DIR" "$INIT" "$GPU" <<'PY' || exit 3
import sys, torch
sys.path.insert(0, '.')
from pathlib import Path
d, init, gpu = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
from diff_vla_wbc.isaac_joint import load_isaac_bundle
from diff_vla_wbc.wm_refresh import refresh_wm, ssm_wm_loss_parts
dev = torch.device(f"cuda:{gpu}")
npz = sorted(Path('data/npz/ha_tokens').glob('*.npz'))
b = load_isaac_bundle(Path('runs/S1_handover/WBC_A24000_wmvel_cmd.pt'), device=dev, npz=npz,
                      packed_motion=Path('runs/isaac_motion/ha_cmd_exec/packed.pt'),
                      obs_mode=None, hist=None, token_dim=64)
ck = torch.load(init, map_location='cpu', weights_only=False)
b.wm.load_state_dict(ck.get('wm', ck), strict=True)
files = sorted(d.glob('*_off*.pt'))
S, A, NS = [], [], []
for f in files:
    z = torch.load(f, map_location='cpu', weights_only=False)
    S.append(z['states']); A.append(z['actions']); NS.append(z['next_states'])
S = torch.cat(S); A = torch.cat(A); NS = torch.cat(NS)
print(f'[wm-re] 数据 {tuple(S.shape)} 来自 {len(files)} 个 shard', flush=True)
out = refresh_wm(b.wm, {'states': S, 'actions': A, 'next_states': NS},
                 steps=2000, batch=2048, lr=1e-4, device=dev,
                 step_fn=b.step, log=lambda m: print(m, flush=True))
print('[wm-re] refresh 完成', out, flush=True)
torch.save({'wm': b.wm.state_dict()}, d / 'wm_stage2.pt')
print('[wm-re] ->', d / 'wm_stage2.pt')
PY

say "2/4 多步微调（20 步；默认再追加 3000 步）"
EXTRA=$( [ "$FAST" = "1" ] && echo 600 || echo 3000 )
python3 scripts/finetune_wm_multistep.py --init-wm "$DIR/wm_stage2.pt" \
  --windows "$DIR/roll_windows.pt" --steps "$EXTRA" --batch 1024 --lr 1.5e-5 \
  --inc-weight 2.0 --device "cuda:$GPU" --out "$OUT_WM" || exit 3

say "3/4 门禁 1：多步保真"
python3 scripts/probe_wm_multistep.py --ckpt "$OUT_WM" --windows "$DIR/roll_windows.pt" \
  --n 768 --device "cuda:$GPU" || { echo "[wm-re][FAIL] 多步保真门未过" >&2; exit 3; }

say "4/4 门禁 2：因果门（headroom ≤20%）"
if python3 scripts/probe_wm_headroom.py --ckpt "$OUT_WM" --windows "$DIR/roll_windows.pt" \
     --gate --device "cuda:$GPU"; then
  echo "[wm-re][PASS] 因果门通过"
else
  echo "[wm-re][FAIL] 因果门未通过 —— **不要**开联合微调（反传仍是假目标）" >&2
  exit 3
fi

say "附：通道可分辨性（只打印，不设门）"
python3 scripts/probe_wm_channels.py --ckpt "$OUT_WM" --windows "$DIR/roll_windows.pt" \
  --device "cuda:$GPU" || true

say "结论"
echo "✅ WM 可用于联合微调：$OUT_WM"
echo "   下一步：WM_CKPT=$OUT_WM bash scripts/run_joint_ft_round5.sh"
