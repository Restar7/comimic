#!/usr/bin/env bash
# Install THE project Isaac stack: Isaac Sim 5.0.0 + Isaac Lab release/2.2.0 (single version).
#
# VERSION DECISION (determined 2026-10-24, see docs/HANDOVER_CROSS_CONTAINER.md §3):
# the HumanoidArena dataset was recorded with Isaac Sim 5.0.0 + Isaac Lab release/2.2.0, so
# that is the version that keeps observations, assets and the closed loop on one physics
# build.  trackerlab-supermimic (WBC training) pins 5.1/2.3.2 upstream, but a static API
# comparison shows it only needs 6 isaaclab modules / 10 symbols and **all of them exist in
# Isaac Lab 2.2** — so it can be trained on 5.0/2.2 as well.  One version, both sides.
#
# Why Isaac Sim at all: the arena scenes (182 `.usd`) and — more importantly — the
# training observations (front RGB mp4 recorded *inside* Isaac Sim) only match if the
# closed loop renders the same assets with the same renderer.  A MuJoCo proxy changes
# the visual distribution, so the VLA would see out-of-distribution frames.
#
# Versions are pinned by the release docs (docs/04_environment_setup.md):
#   Python 3.11, torch 2.7.0+cu128, isaacsim[all,extscache]==5.0.0, IsaacLab release/2.2.0
#
# This box has no conda, so the environment is a plain uv venv (python 3.11).
#
# Usage:  bash scripts/install_isaacsim.sh            # full install
#         STEP=torch bash scripts/install_isaacsim.sh # only one step
set -uo pipefail

COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
# Do not introduce a second Isaac version: one physics build for training and evaluation.
VENV="${VENV:-$COMIMIC/third_party/isaacsim_env}"
ISSUE_LAB="${ISSUE_LAB:-$COMIMIC/third_party/isaaclab_220}"
ISAACSIM_VERSION="${ISAACSIM_VERSION:-5.0.0}"
ISAACLAB_REF="${ISAACLAB_REF:-release/2.2.0}"
LOG="${LOG:-$COMIMIC/logs/isaacsim_install.log}"
STEP="${STEP:-all}"
# Isaac Sim blocks on an interactive EULA prompt; in a headless shell that is an
# EOFError inside `import isaacsim`.  These three env vars make it non-interactive.
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
export PRIVACY_CONSENT=Y

mkdir -p "$COMIMIC/logs" "$COMIMIC/third_party"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== install start (step=$STEP) ==="

if [[ "$STEP" == "all" || "$STEP" == "python" ]]; then
  say "uv python install 3.11"
  uv python install 3.11 2>&1 | tee -a "$LOG"
  say "create venv $VENV"
  uv venv --python 3.11 "$VENV" 2>&1 | tee -a "$LOG"
fi

PY="$VENV/bin/python"
[[ -x "$PY" ]] || { say "FATAL: no $PY"; exit 1; }

if [[ "$STEP" == "all" || "$STEP" == "torch" ]]; then
  say "installing torch 2.7.0 / torchvision 0.22.0 / torchaudio 2.7.0 (cu128)"
  uv pip install -p "$PY" \
    torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -20 | tee -a "$LOG"
fi

if [[ "$STEP" == "all" || "$STEP" == "isaacsim" ]]; then
  say "installing isaacsim[all,extscache]==$ISAACSIM_VERSION (~15 GB, this is the long one)"
  uv pip install -p "$PY" "isaacsim[all,extscache]==$ISAACSIM_VERSION" \
    --extra-index-url https://pypi.nvidia.com 2>&1 | tail -40 | tee -a "$LOG"
fi

if [[ "$STEP" == "all" || "$STEP" == "isaaclab" ]]; then
  say "cloning IsaacLab $ISAACLAB_REF"
  if [[ ! -d "$ISSUE_LAB/.git" ]]; then
    git clone --depth 1 -b "$ISAACLAB_REF" https://github.com/isaac-sim/IsaacLab.git "$ISSUE_LAB" 2>&1 | tail -5 | tee -a "$LOG"
  fi
  say "installing Isaac Lab packages into the venv"
  ( cd "$ISSUE_LAB" && ./isaaclab.sh --install 2>&1 | tail -40 | tee -a "$LOG" )
fi

if [[ "$STEP" == "all" || "$STEP" == "check" ]]; then
  say "verifying"
  "$PY" - <<'PY' 2>&1 | tee -a "$LOG"
import importlib
for mod in ("torch", "isaacsim", "isaaclab"):
    try:
        m = importlib.import_module(mod)
        print(f"  {mod}: OK {getattr(m,'__version__','')}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {mod}: FAIL {exc}")
PY
fi

say "=== install done (step=$STEP) ==="
