#!/usr/bin/env bash
# 单任务 × N 集 / **单进程持久仿真**（走 sim_eval_vla.py 的 --episode_batch_json）。
#
# 为什么不用 ha_isaac_closedloop.sh 逐集起进程：那样每集都要付一次 Isaac 启动 + 建场景
# 的 30–60 s，10 集就是 10 次。这里一个进程把 N 集连着跑，只付一次启动。
#
# 用法（队列脚本会自动调）：
#   ROUTE=L1 TASK_KEY=DoubleDesk EPISODES=10 OUT=runs/xx/L1/DoubleDesk \
#     SONIC_DIR=.../policy/release bash scripts/ha_fast_eval.sh
#
# 产物：$OUT/episodes/*.json、$OUT/videos_{success,failure}/*.mp4、$OUT/summary.json（与
# ha_isaac_batch.sh 同 schema，compare_two_routes.py 直接吃）。
set -uo pipefail

COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
PROJECT_ROOT=$COMIMIC/third_party/HumanoidArena/isaaclab_twist2_g1
export PROJECT_ROOT
export HF_HOME=${HF_HOME:-/inspire/hdd/global_user/liumengfan-253108110079/cache/huggingface}
export NO_ALBUMENTATIONS_UPDATE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
# 关键：我们的 server 回的是 latent64_chunk(40x64)，provider 必须用 latent64 模式。
# 漏了这一行 → provider 走 semantic_v3（期望 Nx40）→ 每步抛异常、静默保持上一条动作、
# 还每步白跑一次推理（2026-10-25 15:0x 的 L1 三集 timeout 就是这么来的）。
export SONIC_VLA_ACTION_FORMAT=${SONIC_VLA_ACTION_FORMAT:-latent64}
export HA_VLA_EXEC_STEPS=${HA_VLA_EXEC_STEPS:-20}

TASK_KEY=${TASK_KEY:?TASK_KEY 必填}
OUT=${OUT:?OUT 必填}
ROUTE=${ROUTE:-eval}
EPISODES=${EPISODES:-10}
MAX_STEPS=${MAX_STEPS:-300}
PORT=${PORT:-18080}
ISAAC_GPU=${ISAAC_GPU:-0}
SM_CKPT=${SM_CKPT:-$COMIMIC/trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500_physint.pt}
SONIC_DIR=${SONIC_DIR:-$COMIMIC/../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release}
VIDEO_FPS=${VIDEO_FPS:-30}

say() { echo "[$(date '+%F %T')] [$TASK_KEY/$ROUTE] $*"; }

# ---- 锁死契约校验（场景 / 资产 / 机器人初始位姿 / 仿真参数 / 编解码器）--------
# 契约文件 configs/ha_closedloop_pin.json，说明 docs/HA_CLOSEDLOOP_PIN.md。
# 任何一项漂了都拒绝启动 —— 否则"同一个任务"两次跑出来的东西不可比。
if [[ "${HA_PIN_BYPASS:-0}" == "1" ]]; then
  say "[pin] WARN: HA_PIN_BYPASS=1，跳过契约校验（结果不得与正式结果混用）"
else
  python3 "$COMIMIC/scripts/pin_ha_closedloop.py" --check --quiet || {
    echo "[pin] HA 闭环契约校验失败，拒绝启动。修法见 docs/HA_CLOSEDLOOP_PIN.md" >&2
    exit 9
  }
fi

case "$TASK_KEY" in
  DoubleDesk|HOI_double_desk)
      TASK_ID=Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody
      TASK_YAML=tasks/common_test_config/base_test/doubledesk_sonic_test.yaml
      INSTR="Pick up the object from the double desk and place it into the basket." ;;
  PPBox|HOI_pp_box)
      TASK_ID=Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby
      TASK_YAML=tasks/common_test_config/base_test/pp_box_sonic_test.yaml
      INSTR="Pick up the box and place it at the target location." ;;
  Boxing|HSI_boxing)
      TASK_ID=Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody
      TASK_YAML=tasks/common_test_config/base_test/boxing_sonic_test.yaml
      INSTR="Throw punches at the target following the reference motion." ;;
  OpenDoor|HSI_open_door)
      TASK_ID=Isaac-Move-Open-Door-G129-Dex3-Wholebody
      TASK_YAML=tasks/common_test_config/base_test/open_door_sonic_test.yaml
      INSTR="Walk to the door, open it, and step through." ;;
  SitSofa|HSI_sit_sofa)
      TASK_ID=Isaac-Move-Sit-Sofa-G129-Dex3-Wholebody
      TASK_YAML=tasks/common_test_config/base_test/sit_sofa_sonic_test.yaml
      INSTR="Walk to the sofa and sit down." ;;
  VisionNavi|HSI_vision_navi)
      TASK_ID=Isaac-Move-SmallWarehouse-VisionNavigation-G129-Dex3-Wholebody
      TASK_YAML=tasks/common_test_config/base_test/vision_navi_sonic_test.yaml
      INSTR="Navigate to the goal location shown in the instruction." ;;
  FootballSingle|football|HSI_football)
      TASK_ID=Isaac-Move-Football-Single-G129-Dex3-Wholebody
      TASK_YAML=tasks/common_test_config/base_test/football_single_sonic_test.yaml
      INSTR="Kick the football into the goal." ;;
  *)
      echo "unknown TASK_KEY=$TASK_KEY" >&2; exit 2 ;;
esac

mkdir -p "$OUT/episodes" "$OUT/videos_success" "$OUT/videos_failure" "$OUT/recordings"

# ---- 组装持久仿真批次的 episode 清单 ---------------------------------------
python3 - "$OUT" "$EPISODES" "$ROUTE" "$MAX_STEPS" "$TASK_ID" "$VIDEO_FPS" <<'PY'
import json, sys
from pathlib import Path
out, n, label, max_steps, task_id, fps = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5], int(sys.argv[6])
out = Path(out)
eps = []
for i in range(n):
    eps.append({
        "seed": i,
        "repeat_idx": 0,
        "episode_seed": i,
        "episode_index": i,
        "result_json": str(out / "episodes" / f"ep_{i:02d}.json"),
        "success_video_dir": str(out / "videos_success"),
        "failure_video_dir": str(out / "videos_failure"),
        "video_fps": fps,
        "post_termination_record_steps": 0,
        "model_label": label,
        "eval_model_path": "",
        "recording_save_dir": str(out / "recordings"),
        "max_steps": max_steps,
    })
(out / "batch.json").write_text(json.dumps({"task": task_id, "episodes": eps}, indent=2) + "\n")
print(f"[batch-json] {out / 'batch.json'} episodes={len(eps)}")
PY

# ---- 环境 ----------------------------------------------------------------
source "$COMIMIC/scripts/isaac_env.sh" >/dev/null 2>&1
export PYTHONPATH=$COMIMIC/third_party/unitree_sdk2_python:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}
unset CUDA_VISIBLE_DEVICES
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y

cd "$PROJECT_ROOT"
say "start: task=$TASK_ID episodes=$EPISODES max_steps=$MAX_STEPS decoder=$SONIC_DIR/model_decoder.onnx"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  say "[dry-run] 不启动 Isaac；batch.json 与命令已就绪"
  exit 0
fi
T0=$(date +%s)
python -u script/eval_scripts/sonic/sim_eval_vla.py \
  --task "$TASK_ID" \
  --env_config_yaml "$TASK_YAML" \
  --seed 0 --max_steps "$MAX_STEPS" \
  --episode_batch_json "$OUT/batch.json" \
  --sm_ckpt "$SM_CKPT" --sm_obs_mode relative_ori99 --sm_token_source http \
  --sv_task_name "$INSTR" --model_label "$ROUTE" \
  --sonic_encoder_path "$SONIC_DIR/model_encoder.onnx" \
  --sonic_decoder_path "$SONIC_DIR/model_decoder.onnx" \
  --lerobot_server_url "http://127.0.0.1:$PORT" \
  --lerobot_server_timeout 60 \
  --robot_type unitree_g1_refpose_v3_1 \
  --device "cuda:$ISAAC_GPU" --headless \
  > "$OUT/sim.log" 2>&1
RC=$?
T1=$(date +%s)
say "sim exit=$RC  wall=$((T1-T0))s  (log: $OUT/sim.log)"

# ---- 汇总成 ha_isaac_batch 同 schema --------------------------------------
python3 - "$OUT" "$TASK_ID" "$ROUTE" <<'PY'
import json, sys
from pathlib import Path
out, task_id, label = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
rows = []
for f in sorted((out / "episodes").glob("*.json")):
    try:
        rows.append(json.loads(f.read_text()))
    except Exception:
        pass
ok = sum(1 for r in rows if r.get("success"))
steps = [int(r.get("episode_steps") or 0) for r in rows]
rew = [float(r.get("max_reward") or 0.0) for r in rows]
per_task = [{
    "variant": label, "task": task_id, "episodes": len(rows), "success": ok,
    "success_rate": round(ok / len(rows), 4) if rows else 0.0,
    "avg_steps": round(sum(steps) / len(steps), 1) if steps else 0.0,
    "avg_max_reward": round(sum(rew) / len(rew), 4) if rew else 0.0,
}]
summary = {"rows": rows, "per_task": per_task,
           "per_variant": {label: {"success": ok, "episodes": len(rows),
                                   "success_rate": round(ok / len(rows), 4) if rows else 0.0}}}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
modes = {}
for r in rows:
    k = "success" if r.get("success") else str(r.get("failure_reason") or "unknown")
    modes[k] = modes.get(k, 0) + 1
print(f"[{label}/{task_id}] {ok}/{len(rows)} = {(ok/len(rows) if rows else 0):.1%}  steps_avg={per_task[0]['avg_steps']}  modes={modes}")
if len(rows) < 1:
    raise SystemExit("[fast-eval] 没有任何 episode 结果，按失败处理")
PY
