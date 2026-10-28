#!/usr/bin/env bash
# Isaac Sim 5.0 + IsaacLab 2.2 闭环（HumanoidArena 官方 sim_eval_vla.py）+
# 我们的 VLA（scripts/ha_vla_latent64_server.py，走 HTTP）。
#
# 这条链路在 570.124.06 驱动上验证通过（见 docs/HANDOVER_ISAAC_CLOSEDLOOP.md §10）。
#
# 用法：
#   bash scripts/ha_isaac_closedloop.sh                       # doubledesk, seed 0, 1 集
#   TASK=Boxing bash scripts/ha_isaac_closedloop.sh           # 换任务
#   SEED=1 REPEAT=3 MAX_STEPS=300 bash scripts/ha_isaac_closedloop.sh
#   EXTRA_ARGS="--step_log_every_n 50" bash scripts/ha_isaac_closedloop.sh
#
# 关键环境变量（一般不用改）：
#   SERVER_URL   我们的 VLA 服务地址（默认 http://127.0.0.1:18080）
#   ISAAC_GPU    给 Isaac Sim 的物理 GPU 序号（默认 0；不改 CUDA_VISIBLE_DEVICES）
#   OUT_ROOT     结果目录（默认 runs/ha_isaac_eval）
#   SONIC_DIR    GEAR-SONIC low_latency 目录（默认从 HF cache 找）
set -uo pipefail

COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
export PROJECT_ROOT=$COMIMIC/third_party/HumanoidArena/isaaclab_twist2_g1

# ---- LOCKED 2026-10-25：闭环必须与 WBC 训练同口径（HA_ALIGN=1）---------------------
# 缺这两个变量 -> action_scale 退回 BeyondMimic（hip_pitch 放大 2.85x、腕缩小 10x）-> 必摔。
# 见 docs/HANDOVER_CLOSEDLOOP_ACTION_SCALE_20261025.md；A/B 才允许 SM_ALLOW_SCALE_MISMATCH=1。
export SUPERMIMIC_HA_ALIGN=1
export SUPERMIMIC_ROBOT_USD=${SUPERMIMIC_ROBOT_USD:-$COMIMIC/trackerlab-supermimic/data/assetslib/unitree/unitree_g1/usd/g1_29dof_rev_1_0.usd}
export HF_HOME=${HF_HOME:-/inspire/hdd/global_user/liumengfan-253108110079/cache/huggingface}
export NO_ALBUMENTATIONS_UPDATE=1
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

# ---- 参数 -------------------------------------------------------------------
TASK=${TASK:-DoubleDesk}
SEED=${SEED:-0}
REPEAT=${REPEAT:-0}
MAX_STEPS=${MAX_STEPS:-300}
SERVER_URL=${SERVER_URL:-http://127.0.0.1:18080}
ISAAC_GPU=${ISAAC_GPU:-0}
OUT_ROOT=${OUT_ROOT:-$COMIMIC/runs/ha_isaac_eval}
# 1 = headless（默认；出视频/跑批量）；0 = 开 Isaac GUI（配合 DISPLAY=:N，NoVNC 可视化用）
HEADLESS=${HEADLESS:-1}
if [[ "$HEADLESS" == "1" ]]; then HEADLESS_FLAG="--headless"; else HEADLESS_FLAG=""; fi
# 我们的 SuperMimic WBC（token-conditioned，trackerlab 训练产物，ckpt 里要有 "pi" 键）
# LOCKED 2026-10-26：执行器必须与 S3 联合微调**同一个** ckpt —— C6h
# （C2 策略 + 按 ha_cmd 命令流域重标的 WM；hold 窗口 + robot_spec）。
# 旧默认 bones_seed_token_2x8192（future 窗口、旧库）会让闭环与训练各说各话。
SM_CKPT=${SM_CKPT:-$COMIMIC/runs/S1_handover/WBC_A20000_wmvel_cmd.pt}
# 回放模式（SM_TOKEN_SOURCE=npy）：不需要 VLA 服务/端口/相机，SM_TOKEN_NPY 指向 (T,64) token 流。
# receding horizon：部署每 20 步重规划（= 训练 H=20 的口径来源）。
# supermimic provider 默认 0（旧行为：全消费 40 帧），闭环必须显式打开。
export HA_VLA_EXEC_STEPS=${HA_VLA_EXEC_STEPS:-20}
# http = 真闭环（VLA 出 token）；zeros = 对照实验（不调 VLA，用零 token）
SM_TOKEN_SOURCE=${SM_TOKEN_SOURCE:-http}
# 上游 SONIC provider 用这个开关决定吃 40 维 semantic 还是我们返回的 64 维 latent
export SONIC_VLA_ACTION_FORMAT=${SONIC_VLA_ACTION_FORMAT:-latent64}

# task 别名 -> (任务 id, 测试 yaml)
case "$TASK" in
  DoubleDesk|doubledesk|HOI_double_desk)
      TASK_ID=Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody
      TASK_YAML=tasks/common_test_config/base_test/doubledesk_sonic_test.yaml
      TASK_INSTRUCTION="Pick up the object from the double desk and place it into the basket." ;;
  PPBox|ppbox|HOI_pp_box)
      TASK_ID=Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby
      TASK_YAML=tasks/common_test_config/base_test/pp_box_sonic_test.yaml
      TASK_INSTRUCTION="Pick up the box and place it at the target location." ;;
  Boxing|boxing|HSI_boxing)
      TASK_ID=Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody
      TASK_YAML=tasks/common_test_config/base_test/boxing_sonic_test.yaml
      TASK_INSTRUCTION="Throw punches at the target following the reference motion." ;;
  OpenDoor|opendoor|HSI_open_door)
      TASK_ID=Isaac-Move-Open-Door-G129-Dex3-Wholebody
      TASK_YAML=tasks/common_test_config/base_test/open_door_sonic_test.yaml
      TASK_INSTRUCTION="Walk to the door, open it, and step through." ;;
  SitSofa|sitsofa|HSI_sit_sofa)
      TASK_ID=Isaac-Move-Sit-Sofa-G129-Dex3-Wholebody
      TASK_YAML=tasks/common_test_config/base_test/sit_sofa_sonic_test.yaml
      TASK_INSTRUCTION="Walk to the sofa and sit down." ;;
  VisionNavi|visionnavi|HSI_vision_navi)
      TASK_ID=Isaac-Move-SmallWarehouse-VisionNavigation-G129-Dex3-Wholebody
      TASK_YAML=tasks/common_test_config/base_test/vision_navi_sonic_test.yaml
      TASK_INSTRUCTION="Navigate to the goal location shown in the instruction." ;;
  FootballSingle|football)
      TASK_ID=Isaac-Move-Football-Single-G129-Dex3-Wholebody
      TASK_YAML=tasks/common_test_config/base_test/football_single_sonic_test.yaml ;;
  GrapCup|grapcup|HOI_grap_cup)
      # 本 checkout（shallow, HEAD 6847928）没有注册这个任务，也没有 ArtVIP Livingroom 资产。
      echo "GrapCup 任务在本仓库 checkout 中缺失（未注册 + 无资产），见 docs/HANDOVER_ISAAC_CLOSEDLOOP.md §11" >&2
      exit 4 ;;
  Isaac-*)  # 直接给官方任务 id
      TASK_ID=$TASK
      TASK_YAML=${TASK_YAML:-tasks/common_test_config/base_test/doubledesk_sonic_test.yaml}
      TASK_INSTRUCTION=${TASK_INSTRUCTION:-} ;;
  *)
      echo "unknown TASK=$TASK (DoubleDesk|GrapCup|PPBox|Boxing|OpenDoor|SitSofa|VisionNavi|Isaac-...)" >&2
      exit 2 ;;
esac

# ---- SONIC ONNX（encoder+decoder 都必须给，provider 两个一起加载）------------
# **必须用 release 变体**：HA 的录制数据（decoder_obs → decoder_target_action）就是 release
# 权重产生的 —— release decoder 复现录制动作的残差 1e-5 度；low_latency 是另一个模型，
# 残差 5.0 度（最差单关节 19.5 度）。HF cache 里只有 low_latency，release 在 clone 里。
# 证据见 docs/SONIC_DECODER_FOUND.md。
if [[ -z "${SONIC_DIR:-}" ]]; then
  RELEASE_DIR=$COMIMIC/../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release
  if [[ -f "$RELEASE_DIR/model_decoder.onnx" && -f "$RELEASE_DIR/model_encoder.onnx" ]]; then
    SONIC_DIR=$RELEASE_DIR
  else
    echo "[closedloop] WARN: release 权重不在 $RELEASE_DIR，退回 HF cache 的 low_latency（口径与录制数据不一致）" >&2
    SONIC_DIR=$(ls -d "$HF_HOME"/hub/models--nvidia--GEAR-SONIC/snapshots/*/low_latency 2>/dev/null | head -1)
  fi
fi
if [[ -z "$SONIC_DIR" || ! -f "$SONIC_DIR/model_decoder.onnx" ]]; then
  echo "cannot find model_decoder.onnx; set SONIC_DIR" >&2
  exit 3
fi

# ---- venv / 系统环境 --------------------------------------------------------
source "$COMIMIC/scripts/isaac_env.sh" >/dev/null 2>&1
export PYTHONPATH=$COMIMIC/third_party/unitree_sdk2_python:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}

# Isaac Sim 5.0 的 carb 不认 CUDA_VISIBLE_DEVICES（会判 "CUDA in bad state"）；
# 选卡只能用 --device，所以这里只清掉它，不动 device。
unset CUDA_VISIBLE_DEVICES
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y

# ---- 锁死契约校验（场景 / 资产 / 机器人初始位姿 / 仿真参数）----------------
if [[ "${HA_PIN_BYPASS:-0}" == "1" ]]; then
  echo "[pin] WARN: HA_PIN_BYPASS=1，跳过契约校验（结果不得与正式结果混用）" >&2
else
  python3 "$COMIMIC/scripts/pin_ha_closedloop.py" --check --quiet || {
    echo "[pin] HA 闭环契约校验失败，拒绝启动。修法见 docs/HA_CLOSEDLOOP_PIN.md" >&2
    exit 9
  }
fi

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR=$OUT_ROOT/${TASK}__seed${SEED}__repeat${REPEAT}__$STAMP
mkdir -p "$RUN_DIR/videos_success" "$RUN_DIR/videos_failure"
RESULT_JSON=$RUN_DIR/result.json

# 测试 yaml 默认 wait_for_textures: false（灰模，VLA 看到的是训练时没见过的无贴图画面）。
# TEXTURES=1 时生成一份 wait_for_textures: true 的同款 yaml 再跑，用于和训练分布对齐。
if [[ "${TEXTURES:-0}" == "1" && -f "$PROJECT_ROOT/$TASK_YAML" ]]; then
  sed 's/wait_for_textures: false/wait_for_textures: true/' "$PROJECT_ROOT/$TASK_YAML" > "$RUN_DIR/env_textured.yaml"
  TASK_YAML=$RUN_DIR/env_textured.yaml
fi

echo "[closedloop] task=$TASK_ID"
echo "[closedloop] yaml=$TASK_YAML"
echo "[closedloop] server=$SERVER_URL  isaac_gpu=cuda:$ISAAC_GPU"
echo "[closedloop] sm_ckpt=$SM_CKPT"
echo "[closedloop] instruction=${TASK_INSTRUCTION:-<none>}"
echo "[closedloop] out=$RUN_DIR"

cd "$PROJECT_ROOT"
# GUI over NoVNC: Xtigervnc has no hardware GLX, so the RTX viewport stays black unless
# VirtualGL interposes.  VGLRUN=1 + HEADLESS=0 -> ``vglrun -d egl python ...``.
if [[ "${VGLRUN:-0}" == "1" ]]; then
  VGL_PREFIX="vglrun -d ${VGL_DISPLAY:-egl}"
else
  VGL_PREFIX=""
fi
exec ${VGL_PREFIX} python -u script/eval_scripts/sonic/sim_eval_vla.py \
  --task "$TASK_ID" \
  --env_config_yaml "$TASK_YAML" \
  --seed "$SEED" --repeat_idx "$REPEAT" \
  --max_steps "$MAX_STEPS" \
  --sm_ckpt "$SM_CKPT" --sm_obs_mode "${SM_OBS_MODE:-auto}" --sm_token_source "$SM_TOKEN_SOURCE" \
  --sm_token_npy "${SM_TOKEN_NPY:-}" --sm_hands_npy "${SM_HANDS_NPY:-}" \
  --sv_task_name "${TASK_INSTRUCTION:-$TASK_ID}" \
  --model_label "${MODEL_LABEL:-$TASK}" \
  --sonic_encoder_path "$SONIC_DIR/model_encoder.onnx" \
  --sonic_decoder_path "$SONIC_DIR/model_decoder.onnx" \
  --lerobot_server_url "$SERVER_URL" \
  --lerobot_server_timeout 60 \
  --robot_type unitree_g1_refpose_v3_1 \
  --device "cuda:$ISAAC_GPU" \
  $HEADLESS_FLAG \
  --result_json "$RESULT_JSON" \
  --success_video_dir "$RUN_DIR/videos_success" \
  --failure_video_dir "$RUN_DIR/videos_failure" \
  --recording_save_dir "$RUN_DIR" \
  ${EXTRA_ARGS:-}
