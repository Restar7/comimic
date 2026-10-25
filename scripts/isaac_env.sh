#!/usr/bin/env bash
# Source this to get the project Isaac environment on this box.
#
#   source scripts/isaac_env.sh            # the one stack: Isaac Sim 5.0.0 + Isaac Lab 2.2.0
#   source scripts/isaac_env.sh check      # ...and print what it found
#
# VERSION DECISION (determined 2026-10-24):
# HumanoidArena was recorded with Isaac Sim 5.0.0 + Isaac Lab release/2.2.0, so that is the
# single version for BOTH training (trackerlab-supermimic) and evaluation.  trackerlab pins
# 5.1/2.3.2 upstream, but a static API comparison shows it uses only 6 isaaclab modules and
# 10 symbols, all present in 2.2 — so no second stack is needed.  See
# docs/HANDOVER_CROSS_CONTAINER.md §3 for the evidence and the residual risk.
#
# Non-interactive EULA: `import isaacsim` prompts on stdin and dies with EOFError in a
# headless shell.  These variables must be exported BEFORE anything imports isaacsim
# (the installer needs them too).
set +u 2>/dev/null || true

COMIMIC=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
STACK="${1:-isaac}"       # kept for backwards compatibility: only one stack exists now
MODE="${2:-}"

export ISAAC_VENV="$COMIMIC/third_party/isaacsim_env"
export ISAAC_LAB_DIR="$COMIMIC/third_party/isaaclab_220"
export ISAAC_VERSION_NOTE="isaacsim 5.0.0 + Isaac Lab release/2.2.0 (matches the HA dataset)"

# --- EULA / headless -------------------------------------------------------------
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
export PRIVACY_CONSENT=Y
export OMNI_KIT_ACCEPT_NUCLEUS_EULA=YES

# --- runtime ---------------------------------------------------------------------
export CARB_LOG_LEVEL="${CARB_LOG_LEVEL:-WARN}"
export HF_HOME=/inspire/hdd/global_user/liumengfan-253108110079/cache/huggingface
export NO_ALBUMENTATIONS_UPDATE=1
export SUPERMIMIC_ASSETLIB_ROOT="${SUPERMIMIC_ASSETLIB_ROOT:-$COMIMIC/trackerlab-supermimic/data/assetslib}"

# `joint_contract` (G1 joint-order single source of truth) lives in the repo root and is
# imported by the Isaac action providers.  Put it on PYTHONPATH here so EVERY entry point
# (our scripts, plain `python sim_main.py`, ...) sees it.
export PYTHONPATH="$COMIMIC${PYTHONPATH:+:$PYTHONPATH}"

export PATH="$ISAAC_VENV/bin:$PATH"
export VIRTUAL_ENV="$ISAAC_VENV"
if [[ -f "$ISAAC_LAB_DIR/isaaclab.sh" ]]; then
  export ISAACLAB_PATH="$ISAAC_LAB_DIR"
fi

if [[ "$MODE" == "check" || "$STACK" == "check" ]]; then
  echo "[isaac_env] stack : $ISAAC_VERSION_NOTE"
  echo "[isaac_env] venv  : $ISAAC_VENV"
  echo "[isaac_env] lab   : $ISAAC_LAB_DIR"
  python - <<'PY'
import importlib
for mod in ("torch", "isaacsim", "isaaclab"):
    try:
        m = importlib.import_module(mod)
        print(f"  {mod}: OK {getattr(m, '__version__', '')}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {mod}: FAIL {type(exc).__name__}: {str(exc)[:120]}")
PY
fi
