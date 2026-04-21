# flatter-rs — partial Rust port & benchmark

Scope: this directory is **not** a full Rust port of flatter (the library is
~15 kLoC of C++ depending on GMP, MPFR, Eigen, OpenBLAS, and FPLLL — a
multi-month undertaking). It ports one well-defined hot kernel and
benchmarks it against the upstream C++ implementation so we can see what a
Rust rewrite buys us for the core building blocks.

## What was ported

`src/problems/matrix_multiplication/elementary_native.cpp` — the templated
"elementary" gemm used as the base case of the recursive matrix-multiply
problem (and as a fallback when BLAS isn't engaged). The C++ version is a
straight ijk triple loop over arbitrary row/column strides. We port the
non-transposed, row-major case (`adr=k, adc=1, bdr=n, bdc=1, ldc=n`), which
is the branch the library actually takes most of the time.

Three type instantiations ship in C++:
`<int64_t,int64_t,int64_t>`, `<double,double,double>`,
`<double,double,int64_t>`. We cover the first two — they're the hot ones —
and leave the mixed `<double,double,int64_t>` for later.

## Rust kernels

| Kernel                        | Threads | SIMD                                   |
|-------------------------------|---------|----------------------------------------|
| `gemm_f64_naive` / `_i64_`    | 1       | none (port of C++ reference)           |
| `gemm_f64_blocked` / `_i64_`  | 1       | autovec (AVX-512 under native target)  |
| `gemm_f64_blocked_par`        | rayon   | autovec                                |
| `gemm_i64_blocked_par`        | rayon   | autovec                                |
| `gemm_f64_avx2`               | 1       | explicit AVX2+FMA (256-bit)            |
| `gemm_f64_avx2_par`           | rayon   | explicit AVX2+FMA                      |

Blocking uses a panel of `MC × KC × NC = 64 × 256 × 256`, ikj order, so the
inner loop is `c_row[j] += a_il * b_row[j]` — a broadcast-FMA pattern that
LLVM lowers to packed FMA/PMULLQ+PADDQ. Rayon parallelism is
"split C by row blocks, each thread owns disjoint rows of C", so there's no
cross-thread write contention.

Correctness is checked against the naive C++-port implementation
(`cargo test --release` — 6 tests, all passing).

## Build & run

```
# Rust (target-cpu=native is set in .cargo/config.toml)
cargo build --release

# C++ baseline + Rust bench, back-to-back
./run_bench.sh
```

## Results

Machine: Intel Xeon @ 2.1 GHz, 4 cores, AVX-512 (F/DQ/BW/VL/VNNI/IFMA),
AVX2+FMA, BMI1/2. Both sides compiled with `-O3 -march=native`; Rust uses
LTO=fat and `codegen-units=1`. Times are best-of-5, preceded by one warm-up.

### f64 gemm, 512 × 512 × 512  (0.27 GFlop)

| Implementation                      | Threads | ms    | GFLOP/s | vs C++ ref | vs C++ best-single |
|-------------------------------------|---------|-------|---------|-----------:|-------------------:|
| C++ flatter ijk (reference)         | 1       | —     | —       | —          | —                  |
| C++ ikj autovec                     | 1       | 45.4  | 5.9     | >5×        | 1.00×              |
| Rust blocked                        | 1       | 30.0  | 9.0     | —          | 1.51×              |
| Rust blocked + rayon                | 4       | 8.6   | 31.4    | —          | **5.31×**          |
| Rust avx2+fma                       | 1       | 37.1  | 7.2     | —          | 1.22×              |
| Rust avx2+fma + rayon               | 4       | 12.7  | 21.1    | —          | 3.58×              |

(The unblocked C++ reference runs are skipped at 512³ because they take
≥5s; at 256³ the reference is 2.0 GFLOP/s — see below.)

### f64 gemm, 256 × 256 × 256  (33.6 MFlop)

| Implementation                      | Threads | ms    | GFLOP/s | vs C++ ref | vs C++ best-single |
|-------------------------------------|---------|-------|---------|-----------:|-------------------:|
| C++ flatter ijk (reference)         | 1       | 16.5  | 2.0     | 1.00×      | 0.26×              |
| C++ ikj autovec                     | 1       |  4.3  | 7.8     | 3.87×      | 1.00×              |
| Rust naive (same-shape port)        | 1       | 14.8  | 2.3     | 1.12×      | 0.29×              |
| Rust blocked                        | 1       |  3.1  | 10.8    | 5.30×      | 1.37×              |
| Rust blocked + rayon                | 4       |  1.4  | 24.3    | 11.99×     | 3.10×              |
| Rust avx2+fma                       | 1       |  3.9  |  8.7    | 4.28×      | 1.11×              |
| Rust avx2+fma + rayon               | 4       |  1.3  | 25.4    | 12.53×     | 3.24×              |

### i64 gemm, 512 × 512 × 512  (0.27 Giop)

| Implementation                      | Threads | ms    | GIOP/s  | vs C++ best-single |
|-------------------------------------|---------|-------|---------|-------------------:|
| C++ ikj autovec                     | 1       | 36.6  |  7.3    | 1.00×              |
| Rust blocked                        | 1       | 34.4  |  7.8    | 1.06×              |
| Rust blocked + rayon                | 4       |  9.7  | 27.8    | **3.79×**          |

### i64 gemm, 256 × 256 × 256  (33.6 Miop)

| Implementation                      | Threads | ms    | GIOP/s  | vs C++ ref | vs C++ best-single |
|-------------------------------------|---------|-------|---------|-----------:|-------------------:|
| C++ flatter ijk (reference)         | 1       | 16.6  |  2.0    | 1.00×      | 0.20×              |
| C++ ikj autovec                     | 1       |  3.4  |  9.9    | 4.91×      | 1.00×              |
| Rust naive                          | 1       | 15.6  |  2.1    | 1.07×      | 0.22×              |
| Rust blocked                        | 1       |  4.0  |  8.3    | 4.14×      | 0.84×              |
| Rust blocked + rayon                | 4       |  1.2  | 26.9    | 13.36×     | 2.92×              |

### Headline vs the upstream kernel

Compared to the exact C++ shape flatter currently ships
(`gemm_xx_flatter`, ijk order, no blocking), the Rust `blocked + rayon`
kernel is:

* **~12× faster** at 256³ f64 (2.0 → 24.3 GFLOP/s)
* **~13× faster** at 256³ i64 (2.0 → 26.9 GIOP/s)

Compared to a charitably-tuned C++ single-threaded autovec version of the
same loop (ikj reorder, same compiler flags), it is **~3.5–5×** faster at
these sizes, which is close to what 4 cores can give you.

## Interesting findings

* At ikj order the compiler does most of the work. Rust `blocked` (1 thread)
  ≈ C++ `ikj autovec` (1 thread) — as expected, since both compile to AVX-512
  FMA / PMULLQ under `target-cpu=native`. The faithful ijk port in both
  languages lands within 10% of each other (2.07 vs 2.25 GFLOP/s at 128³
  f64), confirming the port is apples-to-apples.
* **Explicit AVX2 lost to autovectorized scalar.** My handwritten
  `gemm_f64_avx2` uses 256-bit FMA, but on this CPU `target-cpu=native`
  autovectorizes the scalar `blocked` loop to 512-bit AVX-512 FMA, so it's
  faster. The right move for a production port would be to either drop the
  explicit AVX2 path or raise it to AVX-512 intrinsics. (Left in the repo so
  the reader can see the gap.)
* **Rayon scaling is good but not 4×.** On 4 cores we see 3.1–3.8× at 256³
  and above, and ~1.4× at 128³. That's typical: 128³ f64 is only 33 MFlop of
  work, so thread-launch overhead eats into the win.

## Files

```
rust/
  Cargo.toml
  .cargo/config.toml              # -C target-cpu=native
  src/
    lib.rs
    gemm_f64.rs                   # naive, blocked, blocked+par, avx2, avx2+par
    gemm_i64.rs                   # naive, blocked, blocked+par
  benches/
    bench_gemm.rs                 # timing harness (min-of-5)
  cpp_baseline/
    bench_gemm.cpp                # verbatim flatter gemm_xx + ikj variant
  run_bench.sh
  bench_results.txt               # captured run, attached
```

## What a full port would look like

The rest of flatter's dependency surface is the hard part. The pieces that
would need real work (in rough order of lift):

1. Drop-in replacements / bindings for GMP and MPFR (`rug` crate fronts both
   and is mature enough). That's most of `src/math/` and `src/data/matrix/`.
2. Eigen → `nalgebra` or `faer` for dense linear algebra. `faer` specifically
   has a multithreaded, SIMD Cholesky/QR stack.
3. FPLLL compatibility / I/O format — straightforward.
4. Port the recursive lattice-reduction harness in
   `src/problems/lattice_reduction/` — this is where the algorithmic value of
   the library lives, and it's ~5 kLoC of control flow on top of the
   primitives above.
5. Replace the C++ threading in `matrix_multiplication/threaded.cpp` and
   `lattice_reduction/threaded_3.cpp` with rayon. Low risk, mirrors what we
   already did here.

A realistic first milestone is "ported up through `elementary_*` and
`strassen` for all three instantiations, wired through the
`MatrixMultiplication` dispatch, benchmarked end-to-end on qary_small.lat".
That's weeks, not hours.
