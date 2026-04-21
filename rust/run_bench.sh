#!/usr/bin/env bash
# Runs the C++ baseline and the Rust port, both with -O3/native, and prints
# their outputs back-to-back so they can be compared.
set -euo pipefail

cd "$(dirname "$0")"

echo "=== Building C++ baseline ==="
g++ -O3 -march=native -std=c++17 -DNDEBUG \
    cpp_baseline/bench_gemm.cpp -o cpp_baseline/bench_gemm_cpp

echo "=== Building Rust port ==="
cargo build --release --bin bench_gemm --quiet

echo
echo "################ C++ baseline ################"
./cpp_baseline/bench_gemm_cpp
echo
echo "################ Rust port ###################"
./target/release/bench_gemm
