#!/usr/bin/env bash
# Recreate the mesh link used by the deploy G1 MJCF.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
G1_DIR="$ROOT/GR00T-WholeBodyControl/gear_sonic_deploy/g1"
MESHES="$ROOT/reproduce_scenes/g1/meshes"
if [ ! -d "$MESHES" ]; then
  echo "missing mesh directory: $MESHES" >&2
  exit 1
fi
if [ -L "$G1_DIR/meshes" ]; then
  echo "mesh symlink already exists: $G1_DIR/meshes"
elif [ -d "$G1_DIR/meshes" ] && [ -z "$(ls -A "$G1_DIR/meshes")" ]; then
  rmdir "$G1_DIR/meshes"
  ln -s ../../../reproduce_scenes/g1/meshes "$G1_DIR/meshes"
  echo "created mesh symlink"
else
  echo "mesh directory exists and is not empty; leaving it alone"
fi
