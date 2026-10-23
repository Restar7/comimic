#!/usr/bin/env bash
# Build the SONIC C++ deployment on an x86_64 workstation.
#
# Everything here is already present on this machine; nothing is downloaded.
# Reusing what existed saved ~7 GB of installs:
#
#   TensorRT 10.13.3.9  -> pip wheel  (tensorrt-cu12 + tensorrt-cu12-libs)
#   headers             -> cdn.jsdelivr.net gh/NVIDIA/TensorRT@v10.13.3/include
#   onnxruntime 1.19.2  -> /tmp/ort/onnxruntime-linux-x64-1.19.2 (headers + so)
#   msgpack-cxx 6.1.1   -> /tmp/msgpack/msgpack-cxx-6.1.1/include  (MSGPACK_NO_BOOST)
#   libzmq + Eigen3     -> conda env ros2h
#   nlohmann/json.hpp   -> external_dependencies/XRoboToolkit-.../include
#   CUDA 12.8           -> /usr/local/cuda
#   unitree_sdk2 (x86)  -> gear_sonic_deploy/thirdparty (vendored)
set -euo pipefail

REPO=/home/wusichen/GR00T-WholeBodyControl
DEPLOY="$REPO/gear_sonic_deploy"
TRT_ROOT=/home/wusichen/groot_repro/trt_root
ORT_ROOT=/tmp/ort/onnxruntime-linux-x64-1.19.2
MSGPACK=/tmp/msgpack/msgpack-cxx-6.1.1/include
CONDA=/home/wusichen/miniconda3/envs/ros2h
XRT_INC="$REPO/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/include"

cd "$DEPLOY"
rm -rf build && mkdir -p build && cd build

env TensorRT_ROOT="$TRT_ROOT" \
    onnxruntime_ROOT="$ORT_ROOT" \
    CUDAToolkit_ROOT=/usr/local/cuda \
  cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$CONDA" \
    -DMSGPACK_INCLUDE_DIR="$MSGPACK" \
    -DZMQ_INCLUDE_DIR="$CONDA/include" \
    -DZMQ_LIBRARY="$CONDA/lib/libzmq.so" \
    -DEigen3_DIR="$CONDA/share/eigen3/cmake" \
    -DCMAKE_CXX_FLAGS="-DMSGPACK_NO_BOOST -I$XRT_INC -I$MSGPACK"

# tests need nlohmann + gtest; only the controller binary matters
cmake --build . --target g1_deploy_onnx_ref -j"$(nproc)"

echo
echo "built: $DEPLOY/target/release/g1_deploy_onnx_ref"
echo "run with:"
echo "  export LD_LIBRARY_PATH=$TRT_ROOT/lib:$ORT_ROOT/lib:$CONDA/lib:\$LD_LIBRARY_PATH"
