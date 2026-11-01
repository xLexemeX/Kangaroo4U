#!/usr/bin/env bash
# Build the solver into bin/. Needs the CUDA Toolkit (nvcc), CMake 3.18+ and g++.
set -e
cd "$(dirname "$0")/engine"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build build --config Release -j
echo "built: $(cd .. && pwd)/bin/rckangaroo"
