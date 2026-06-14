#!/usr/bin/env bash
# ==========================================================================
# build.sh — Titan Audio Ecosystem build/run wrapper
# Activates the llama-turbo-61 micromamba environment which provides:
#   • CUDA 12.4  (compatible with candle 0.8 / cudarc 0.13.9)
#   • GCC 12.4   (compatible with CUDA 12.4 nvcc)
#   • sm_61      (GTX 1080 Ti compute capability)
#
# Usage:
#   ./build.sh              — cargo build --release
#   ./build.sh run          — cargo run   --release
#   ./build.sh check        — cargo check (fast, no CUDA kernel compile)
# ==========================================================================

set -euo pipefail

ENV_PREFIX=/home/anon/micromamba/envs/llama-turbo-61
TARGETS="${ENV_PREFIX}/targets/x86_64-linux"

# ---- Toolchain ----
# Prepend the conda env bin so nvcc, gcc, g++ resolve to the right versions.
export PATH="${ENV_PREFIX}/bin:${PATH}"

# ---- CUDA root for bindgen_cuda + cudarc ----
# bindgen_cuda looks for $CUDA_PATH/include/cuda.h; in the conda env the
# headers are under targets/x86_64-linux/include/, not the env root.
export CUDA_PATH="${TARGETS}"
export CUDA_ROOT="${TARGETS}"
export CUDA_TOOLKIT_ROOT_DIR="${TARGETS}"

# ---- Host compiler overrides ----
# bindgen_cuda uses NVCC_CCBIN for the -ccbin flag.
# cudarc uses CUDAHOSTCXX.
export NVCC_CCBIN="${ENV_PREFIX}/bin/g++"
export CUDAHOSTCXX="${ENV_PREFIX}/bin/g++"
export CUDACXX="${ENV_PREFIX}/bin/nvcc"
export CC="${ENV_PREFIX}/bin/gcc"
export CXX="${ENV_PREFIX}/bin/g++"

# ---- Runtime library path ----
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${TARGETS}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "=== Build environment ==="
echo "  nvcc : $(which nvcc) — $(nvcc --version | grep 'release')"
echo "  g++  : $(which g++) — $(g++ --version | head -1)"
echo "  gcc  : $(which gcc) — $(gcc --version | head -1)"
echo "  CUDA_PATH : ${CUDA_PATH}"
echo "  cuda.h    : $(ls ${CUDA_PATH}/include/cuda.h 2>/dev/null && echo FOUND || echo MISSING)"
echo "========================="

CARGO_CMD="${1:-build}"
shift 2>/dev/null || true

cargo "${CARGO_CMD}" --release "$@"
