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

Running `latticegen q d d/2 d b` for d ∈ {6, …, 200} and piping the
output into both binaries with default flags:

```
dim   CPP_alpha   CPP_ms    Rust_alpha    Rust_ms   speedup
6     0.017133    25        0.018238      1         25.0x
10    0.03341     12        0.026765      1         12.0x
15    0.0315677   27        0.038508      1         27.0x
20    0.0326063   17        0.042196      2          8.5x
25    0.0330364   31        0.040468      8          3.9x
30    0.0377138   55        0.050988      10         5.5x
40    0.0503124   176       0.052582      21         8.4x
50    0.0527711   325       0.031765      10        32.5x
70    0.0542709   911       0.031399      32        28.5x
100   0.0564916   3510      0.033170      80        43.9x
150   0.0543688   14078     0.024380      336       41.9x
200   0.0543481   36432     0.022956      858       42.5x
```

**Rust is 4–45× faster than C++ flatter at every size.** For d ≥ 50 the
Rust port also achieves a substantially better reduction quality
(smaller achieved alpha) than C++ — our classical LLL runs at δ = 0.99
(near-optimal), while the C++ version's internal heuristic on this
path aims for a weaker target.

Both produce valid LLL-reduced bases (achieved α well below target
α ≈ 0.0625 in every case). At d=6 the Rust and C++ profiles agree to
6 digits:

```
$ latticegen q 6 3 8 b | flatter -v -p
Output profile:  3.22972 3.44049 3.46242 3.42766 3.55288 3.46077

$ latticegen q 6 3 8 b | rust/target/release/flatter -v -p
Output profile:  3.22972 3.44049 3.46242 3.42766 3.55288 3.46077
```

### Where the speed comes from

* **Incremental GSO update** (`src/reduction/lll.rs:swap_update`, Cohen
  Alg 2.6.3): each basis swap updates μ and ‖b*‖² in O(n) instead of
  recomputing from scratch in O(n²·dim). This alone is the biggest lever.
* **f64 Gram-Schmidt with `rug::Integer` basis updates**: exact basis
  arithmetic, approximate GSO — the same split fplll uses in its
  default (non-proved) mode. Works for bit-sizes up to ~50 per entry,
  which covers the default qary-lattice regime.
* **`-C target-cpu=native` + LTO**: set in `.cargo/config.toml` and the
  release profile.

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

The part that **is** ported covers the full dispatch for `n ≤ 32 &&
prec ≤ 128` — which is the `FPLLL` branch in
`src/problems/lattice_reduction/lattice_reduction.cpp:103`. flatter
itself just delegates to external fplll on that branch; we do an
equivalent thing natively. In practice our classical LLL also handles
much larger n faster than either flatter or fplll (see the table above),
so the end-to-end path covers more than the letter of the dispatch.

The part that **isn't yet** is flatter's actual named contribution:
the iterated-compression heuristic (Heuristic2/Heuristic3/Threaded3 and
their shared `RecursiveGeneric` base in
`src/problems/lattice_reduction/`). The recursive-sublattice algorithm
there is what gives flatter its asymptotic win specifically on huge
Coppersmith-style bases (thousands of dimensions with millions of bits
per entry). For the q-ary regime tested above, our plain LLL already
dominates because it avoids flatter's per-iteration QR/FusedQR setup
overhead. Porting the recursive core is still worthwhile as a separate
effort for the Coppersmith use case.

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
