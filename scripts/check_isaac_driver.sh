#!/usr/bin/env bash
# Isaac 闭环启动前的驱动预检。
#
# 背景（2026-11-02 实测）：Isaac Sim 5.0 自带的 Omniverse Warp / PhysX 走老式驱动入口点
# （cuDeviceGetUuid）。宿主驱动 595.71.05 上解析失败，日志表现为：
#   Warp CUDA error: Failed to get driver entry point 'cuDeviceGetUuid'
#   [Error] [omni.physx.plugin] CUDA libs are present, but no suitable CUDA GPU was found!
#   PhysX warning: GPU solver pipeline failed, switching to software
# 后果不是"报错退出"，而是**静默退回 CPU 物理**：一条 episode 从 ~100 s 变成 >20 min，
# 且动力学口径与已验证数据不一致（采出来的东西必须丢弃）。所以必须在开跑前拦住。
#
# 已验证可用的宿主驱动：550.163.01 / 570.124.06 / 570.195.03
# 已知不可用：595.71.05（本脚本阈值按 major 判断：>575 一律拒绝）
#
# 用法：
#   bash scripts/check_isaac_driver.sh            # 检查，OK 退出 0，不兼容退出 9
#   ALLOW_UNKNOWN_DRIVER=1 bash ...               # 强行通过（产物必须标注非验证驱动）
set -uo pipefail

MAX_MAJOR=${MAX_DRIVER_MAJOR:-575}

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[driver-check][FAIL] 没有 nvidia-smi" >&2
  exit 9
fi

VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
if [[ -z "$VER" ]]; then
  echo "[driver-check][FAIL] 读不到驱动版本（GPU 没挂上？）" >&2
  exit 9
fi
MAJOR=${VER%%.*}

NVRM=$(grep -oE '[0-9]+\.[0-9]+\.[0-9]+' /proc/driver/nvidia/version 2>/dev/null | head -1)
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
HOSTID=$(cat /sys/class/dmi/id/product_serial 2>/dev/null | tr -d ' ')
MODEL=$(cat /sys/class/dmi/id/product_name 2>/dev/null | tr -d ' ')

echo "[driver-check] gpu_driver=$VER  kernel_nvrm=${NVRM:-?}  n_gpu=$NGPU  host=${MODEL:-?}/${HOSTID:-?}"

if [[ "$MAJOR" =~ ^[0-9]+$ ]] && (( MAJOR <= MAX_MAJOR )); then
  echo "[driver-check] ✅ 在已验证范围（<= ${MAX_MAJOR}）"
  exit 0
fi

if [[ "${ALLOW_UNKNOWN_DRIVER:-0}" == "1" ]]; then
  echo "[driver-check] ⚠️ 驱动 $VER 未验证，但 ALLOW_UNKNOWN_DRIVER=1 ⇒ 放行" >&2
  exit 0
fi

cat >&2 <<EOF
[driver-check] ❌ 驱动 $VER 未验证（阈值 <= ${MAX_MAJOR}）。
  Isaac Sim 5.0 的 Omniverse Warp/PhysX 在这类新驱动上会**静默退回 CPU 物理**：
  不报错、不退出，只是一条 episode 从 ~100 s 变成 >20 min，且采出的数据动力学口径不对。
  已验证可用：550.163.01 / 570.124.06 / 570.195.03（见 docs/HANDOVER_2GPU_EVAL 与本轮记录）。

  处理办法（任一）：
    1. 把容器调度回驱动 <= ${MAX_MAJOR} 的节点（首选）；把下面这行给运维：
       host=${MODEL:-?} serial=${HOSTID:-?} driver=$VER 需要 <= ${MAX_MAJOR} 的驱动
    2. 确认这次不跑 Isaac（只训 joint）⇒ 用 run_stage，不要走采集/评测阶段。
    3. 明知故犯 ⇒ ALLOW_UNKNOWN_DRIVER=1，并在所有产出的名字/报告里标注。
EOF
exit 9
