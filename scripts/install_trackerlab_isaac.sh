#!/usr/bin/env bash
# Install Isaac Sim 5.1.0 + Isaac Lab 2.3.2 + trackerlab-supermimic in their own
# Python 3.11 venv, so the MuJoCo line and the concurrent HA session are untouched.
#
# Versions come from trackerlab-supermimic/docs/installation.md:
#   Python 3.11, torch 2.7.0+cu128, isaacsim[all,extscache]==5.1.0,
#   IsaacLab commit 37ddf626871758333d6ed89cf64ad702aef127d0 (release 2.3.2,
#   package version 0.54.2).
#
# Usage:
#   bash scripts/install_trackerlab_isaac.sh                 # everything
#   STEP=isaacsim bash scripts/install_trackerlab_isaac.sh   # one step
#
# The box is a root-owned container, so scripts that launch Kit need --allow-root;
# the EULA is accepted non-interactively through OMNI_KIT_ACCEPT_EULA below.
set -uo pipefail

COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
VENV="$COMIMIC/third_party/trackerlab_env"
PY="$VENV/bin/python"
ISAACLAB="$COMIMIC/third_party/IsaacLab_232"
REPO="$COMIMIC/trackerlab-supermimic"
LOG="$COMIMIC/logs/isaaclab_232_install.log"
STEP="${STEP:-all}"

export UV_PYTHON_INSTALL_DIR="$COMIMIC/third_party/uv_python"
export UV_PYTHON_PREFERENCE=only-managed
export UV_HTTP_TIMEOUT=600
export UV_CONCURRENT_DOWNLOADS=8
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y

mkdir -p "$COMIMIC/logs" "$COMIMIC/third_party"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
uvp() { uv pip install -p "$PY" --index-strategy unsafe-best-match "$@"; }

say "=== install start (step=$STEP) ==="

# ---------------------------------------------------------------- python + venv
if [[ ! -x "$PY" ]]; then
  say "creating venv $VENV (python 3.11)"
  uv venv --python 3.11 "$VENV" 2>&1 | tail -2 | tee -a "$LOG"
fi
[[ -x "$PY" ]] || { say "FATAL: no $PY"; exit 1; }

# ------------------------------------------------------------------- base tools
if [[ "$STEP" == "all" || "$STEP" == "base" ]]; then
  say "base tools: pip/setuptools/wheel/flatdict"
  uvp pip "setuptools==80.9.0" "wheel==0.45.1" 2>&1 | tail -5 | tee -a "$LOG"
  uvp "flatdict==4.0.1" --no-build-isolation 2>&1 | tail -5 | tee -a "$LOG"
fi

# ------------------------------------------------------------------------ torch
if [[ "$STEP" == "all" || "$STEP" == "torch" ]]; then
  say "torch 2.7.0 / torchvision 0.22.0 / torchaudio 2.7.0 (cu128)"
  uv pip install -p "$PY" \
    torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -8 | tee -a "$LOG"
fi

# --------------------------------------------------------------------- isaacsim
if [[ "$STEP" == "all" || "$STEP" == "isaacsim" ]]; then
  say "isaacsim[all,extscache]==5.1.0 (large: ~15-20 GB download)"
  uvp "isaacsim[all,extscache]==5.1.0" \
    --extra-index-url https://pypi.nvidia.com 2>&1 | tail -15 | tee -a "$LOG"
fi

# -------------------------------------------------------------------- isaaclab
if [[ "$STEP" == "all" || "$STEP" == "isaaclab" ]]; then
  if [[ ! -d "$ISAACLAB/.git" ]]; then
    say "cloning IsaacLab"
    git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB" 2>&1 | tail -3 | tee -a "$LOG"
  fi
  say "checkout 37ddf626871758333d6ed89cf64ad702aef127d0 (2.3.2)"
  git -C "$ISAACLAB" checkout -q 37ddf626871758333d6ed89cf64ad702aef127d0 2>&1 | tail -3 | tee -a "$LOG"
  say "pip install -e $ISAACLAB/source/isaaclab"
  uvp -e "$ISAACLAB/source/isaaclab" 2>&1 | tail -8 | tee -a "$LOG"
  say "isaaclab extras: ipython/psutil"
  uvp "ipython==8.30.0" "psutil==5.9.8" 2>&1 | tail -5 | tee -a "$LOG"
fi

# ------------------------------------------------------------------- trackerlab
if [[ "$STEP" == "all" || "$STEP" == "trackerlab" ]]; then
  say "trackerlab-supermimic requirements + editable install"
  uvp -r "$REPO/requirements.txt" 2>&1 | tail -8 | tee -a "$LOG"
  ( cd "$REPO" && uv pip install -p "$PY" --no-deps -e . 2>&1 | tail -5 | tee -a "$LOG" )
fi

# ------------------------------------------------------------------------ check
if [[ "$STEP" == "all" || "$STEP" == "check" ]]; then
  say "verifying imports"
  "$PY" - <<'PY' 2>&1 | tee -a "$LOG"
import importlib
for mod in ("torch", "isaacsim", "isaaclab", "trackerLab"):
    try:
        m = importlib.import_module(mod)
        print(f"  {mod}: OK {getattr(m, '__version__', '')} {getattr(m, '__file__', '')}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {mod}: FAIL {type(exc).__name__}: {exc}")
PY
fi

say "=== install done (step=$STEP) ==="
