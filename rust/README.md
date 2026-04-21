# flatter-rs — Rust port of flatter

A Rust port of the [flatter](https://github.com/keeganryan/flatter) lattice
reduction library. Same CLI, same FPLLL I/O format, same reduction-quality
flags. Drop-in replacement for the `flatter` binary.

## Status

| Component                                  | Status | Notes |
|--------------------------------------------|--------|-------|
| CLI (`-h -v -q -p -alpha -rhf -delta …`)   | ✅     | byte-compatible with `apps/flatter.cpp` |
| FPLLL lattice I/O                          | ✅     | `Lattice::read_fplll` / `Display` |
| `Lattice`, `Profile` data types            | ✅     | ports of `src/data/lattice/*`, `src/profile.cpp` |
| `LatticeReductionGoal` (alpha/rhf/delta)   | ✅     | |
| Lagrange reduction (n ≤ 2)                 | ✅     | port of `lagrange.cpp`; uses `rug::Integer`/`rug::Float` |
| Classical LLL (general n)                  | ✅     | native Rust — stands in for the `FPLLL` dispatch path |
| Iterated-compression core (Heuristic2/3)   | ⏳     | left for the next session, see below |
| Elementary gemm kernels (f64, i64)         | ✅     | in `src/gemm_*.rs`, benchmarked vs C++ |
| MPFR triangular matmul                     | ✅     | `src/tri_matmul_mpfr.rs`, via `rug` |

No pure-Rust reimplementation of GMP/MPFR — we bind to the same C
libraries flatter uses, via the [`rug`](https://crates.io/crates/rug)
crate (feature `use-system-libs` so the build reuses the system
`libgmp` / `libmpfr`).

## End-to-end comparison with C++ flatter

Running `latticegen q d d/2 b b` for various `(d, b)` and piping the
output into both binaries with default flags (`rhf=1.0219`):

```
dim  bits   CPP_alpha   CPP_ms   Rust_alpha   Rust_ms   speedup
 10  100    0.0282      10       0.0129       1         10×
 20  200    0.0404      135      0.0168       1         135×
 20  500    0.0318      201      0.0372       101       2.0×
 20 1000    0.0406      292      0.0115       195       1.5×
 20 2000    0.0356      444      0.0109       661       0.7×
 30  500    0.0407      442      0.0438       489       0.9×
 50  500    0.0558      1421     0.0543       4333      0.3×
 50  100    0.0543      546      0.0289       13        42×
100   50    0.0547      4169     0.0296       172       24×
200   50    0.0582      31925    0.0184       651       49×
```

Both produce valid LLL-reduced bases (achieved α well below target
≈ 0.0625 in every case). Summary:

| Regime                       | Speedup         |
|------------------------------|-----------------|
| small entries (≤ 200 bits)    | **10 – 135×**  |
| large dimensions, few bits   | **24 – 49×**   |
| medium entries (500–1000 b)  | 1.5 – 2× or parity |
| large entries on big n       | 0.3 – 0.7× (C++ wins) |

Where C++ still wins is the combination of **mid-to-high dimension
(n ≥ 50) with large entries (≥ 500 bits)** — that's exactly the regime
flatter's recursive iterated-compression heuristic is designed for. Our
implementation falls back to classical LLL in MPFR there, which works
but has the wrong asymptotic scaling. See "What isn't ported" below.

### Where the speed comes from

* **Incremental GSO update** (`src/reduction/lll.rs:swap_update_f64`,
  Cohen Alg 2.6.3): each basis swap updates μ and ‖b*‖² in O(n)
  instead of recomputing in O(n²·dim). This is the biggest lever for
  dimensions above ~20.
* **f64 path for small-entry lattices**: entries up to ~300 bits use
  plain f64 GSO; above that we switch to MPFR automatically.
* **Right-sized MPFR precision**: `max_bits + log₂(n) + 64`, which is
  just enough to round μ correctly — not the safer but far slower
  `2·max_bits`.
* **`-C target-cpu=native` + LTO**.

For identical profiles between the two binaries, compare at d=6:

```
$ latticegen q 6 3 8 b | flatter -v -p
Output profile:
3.22972 3.44049 3.46242 3.42766 3.55288 3.46077

$ latticegen q 6 3 8 b | rust/target/release/flatter -v -p
Output profile:
3.22972 3.44049 3.46242 3.42766 3.55288 3.46077
```

Profiles agree to 5+ digits; bases agree up to sign (which is all LLL
guarantees).

## What's ported vs. what isn't

**Ported and producing matching output**:
* Full CLI: `-h -v -q -p -alpha -rhf -delta -logcond`
* FPLLL lattice I/O (read + write)
* Lagrange reduction for n ≤ 2
* Classical LLL for general n — f64 path for entries ≤ 300 bits,
  MPFR path via `rug::Float` above that, with the Cohen incremental
  swap update.
* Profile computation (`get_drop`, `get_spread`) ported from
  `src/profile.cpp`.

**Not ported — the `RecursiveGeneric` core**: flatter's actual named
contribution is the iterated-compression heuristic in
`src/problems/lattice_reduction/recursive_generic.cpp` (and its
Heuristic2/Heuristic3/Threaded3 subclasses). The trick:

 1. QR-factor the integer basis `B` into `Q · R` at MPFR precision.
 2. Shift `R`'s columns to bring them into a fixed-precision window
    (`compress_R` in recursive_generic.cpp:378).
 3. Recursively reduce that **compressed shadow** at low precision.
 4. Lift the resulting unimodular `U` back to the exact basis by a
    big-integer matmul: `B := B · U`.
 5. Repeat until the profile is reduced.

The key property is that each recursive call runs on O(n) basis
vectors with O(precision)-bit entries, regardless of how large the
*original* entries were. Classical LLL in MPFR has to keep full
precision throughout, which is why we see the `n·b` vs `n³·b` gap in
the table's bottom-right cells.

Porting the recursive core is a bounded amount of work but not a
one-liner:

* `QRFactorization` in MPFR — a Householder variant already lives in
  `src/problems/qr_factorization/householder_mpfr.cpp` (~400 LOC). `rug`
  makes the port mechanical.
* `FusedQRSizeReduction` — inner loop in
  `src/problems/fused_qr_sizered/columnwise_double.cpp` (~500 LOC).
* `RecursiveGeneric::solve` + Heuristic2/3's `setup_sublattice_reductions`
  — ~1200 LOC total across the three files.
* `SublatticeSplit` — trivial.

Between 2000 – 3000 LOC including tests. One focused session on top of
what's here.

The gemm kernels already ported (`src/gemm_f64.rs`, `src/gemm_i64.rs`,
`src/tri_matmul_mpfr.rs`) are the multiplication primitives this layer
would call.

Sketch of what's needed:
* `MatrixData<mpfr_t>` / `MatrixData<mpz_t>` wrappers (we have the int
  matrix; the MPFR one is trivial on top of `Vec<rug::Float>`).
* `QRFactorization` — Householder QR at MPFR precision. Bindings to
  LAPACK via `lapack`/`openblas-src` work for the double-precision path,
  `rug` handles MPFR.
* `FusedQRSizeReduction` — the inner loop of the recursion. One file
  (`src/problems/fused_qr_sizered/`).
* `RecursiveGeneric` + `Heuristic2`/`Heuristic3` — the algorithm proper.
* `SublatticeSplit` — trivial.
* Rayon to replace the OpenMP tasks in `threaded_3.cpp`.

The gemm kernels ported earlier in this directory
(`gemm_f64.rs`, `gemm_i64.rs`) already cover the matrix-multiplication
building blocks needed by that layer.

## Build

Needs `libgmp-dev`, `libmpfr-dev` (the `rug` crate uses
`gmp-mpfr-sys`'s `use-system-libs` feature so it links against the
system libraries — no lengthy source build).

```
cd rust/
cargo build --release
./target/release/flatter -h
```

## Test against C++ flatter

```
# from the project root, with the C++ binary already built in build/bin/flatter:
latticegen q 20 10 20 b > /tmp/q20.lat

LD_LIBRARY_PATH=build/lib build/bin/flatter -q -v < /tmp/q20.lat
rust/target/release/flatter                   -q -v < /tmp/q20.lat
```

## Repo layout (rust/)

```
rust/
  Cargo.toml, Cargo.lock, .cargo/config.toml  (-C target-cpu=native)
  src/
    lib.rs              # re-exports
    lattice.rs          # Lattice, IntMatrix, FPLLL I/O
    profile.rs          # Profile + get_drop / get_spread
    reduction/
      mod.rs            # LatticeReduction::solve dispatch
      goal.rs           # LatticeReductionGoal (alpha/rhf/delta)
      params.rs         # LatticeReductionParams
      lagrange.rs       # n ≤ 2 (port of lagrange.cpp)
      lll.rs            # classical LLL for general n
    gemm_f64.rs         # elementary f64 gemm (naive, blocked, rayon, AVX2+FMA)
    gemm_i64.rs         # elementary i64 gemm (naive, blocked, rayon)
    tri_matmul_mpfr.rs  # MPFR triangular matmul via rug (sequential + rayon)
    bin/
      flatter.rs        # CLI port of apps/flatter.cpp
  benches/
    bench_gemm.rs       # kernel microbench
  cpp_baseline/
    bench_gemm.cpp      # C++ baseline for the kernel microbench
  bench_results.txt     # captured microbench run
  end_to_end.txt        # captured CLI-vs-CLI run on qary lattices
  run_bench.sh          # builds both and prints the microbench tables
```

## Tests

```
cargo test --release     # 11 tests, all passing
```

Coverage: lattice I/O roundtrip, Lagrange vs hand-reduced 2x2, LLL vs
hand-verified 3x3 (determinant preservation + finite profile), gemm
port matches the naive reference at f64/i64, MPFR tri-matmul matches
an independent sequential reference.
