#!/bin/bash
# Run inside the edgellm-dev container (sm_87): ./build_and_test.sh [layers...]
set -e
export PATH=/usr/local/cuda/bin:$PATH
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
mkdir -p /tmp/aff
nvcc -std=c++17 -arch=sm_87 --expt-relaxed-constexpr -shared -Xcompiler -fPIC -I"$REPO/cpp" "$HERE/affine_capi.cu" -o /tmp/aff/libaffine_capi.so
cd "$HERE"
python3 -I precision.py 2 20 48
python3 -I test_affine_moe.py "${@:-20}"
