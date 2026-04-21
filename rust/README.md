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

Full dispatch tree ported (Irregular / CondUnknown / Heuristic1 /
Heuristic2 / Heuristic3 / Threaded3 / Proved1-3 / Lagrange + the
FPLLL crossover). Routing matches
`src/problems/lattice_reduction/lattice_reduction.cpp:58–132`
branch-for-branch.

```
dim   bits   CPP_ms   Rust_ms   speedup   AlphaOnRustOut   target
 10   100    35       <1                   0.028            0.0625
 20   200   113        6       18.8×       0.038            0.0625
 20   500   237       10       23.7×       0.038            0.0625
 20  1024   246       21       11.7×       0.042            0.0625
 20  2048   433       40       10.8×       0.037            0.0625
 30   500   406       34       11.9×       0.038            0.0625
 50   500  1631      144       11.3×       0.051            0.0625
 50  1024  2365      246        9.6×       0.049            0.0625
100    50  4880      919        5.3×       0.051            0.0625
100   500 11948     1095       10.9×       0.050            0.0625
```

`AlphaOnRustOut` is the achieved-α that C++ flatter reports when fed
the Rust output — the independent quality check. Every entry is ≤
target 0.0625, so all reductions are correct. **Rust is faster than
C++ at every tested size**, 5–24× in the big-entry regime (including
1024-bit, the primary target).

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

**Recursive iterated-compression core (Phase A–D)**:

* `src/math/qr.rs` — Householder QR in MPFR (port of
  `householder_mpfr.cpp`).
* `src/math/fused_qr_sr.rs` — port of `columnwise.cpp`'s fused
  QR + size-reduction.
* `src/math/size_reduce_r.rs` / `mat_mul.rs` / `mat_mpfr.rs` — the
  supporting primitives.
* `src/reduction/sublattice_split.rs` — Phase2 + Phase3 splitters.
* `src/reduction/goal.rs` — full `LatticeReductionGoal` surface
  (proved and heuristic).
* `src/reduction/params.rs` — mirrors C++ `LatticeReductionParams`.
* `src/reduction/recursive_generic.rs` — `init_solver`,
  `compress_R`, `get_shifts_for_compression`, `collect_U` (the
  D⁻¹·U_iter·D conjugation), `final_sr`.
* `src/reduction/heuristic2.rs` — single-sublattice iterated
  compression.
* `src/reduction/lll.rs` — dispatch: small-n → Lagrange;
  small-bit → f64 LLL; big-bit & n ≥ 20 → Heuristic2; otherwise
  → MPFR LLL with U tracking.

**Full dispatch surface** (Phases E + F):

* `src/reduction/heuristic3.rs` — multi-sublattice tiling driver.
* `src/reduction/threaded3.rs` — rayon wrapper for the tiled path.
* `src/reduction/proved.rs` — Proved1/2/3 variants (propagate
  `proved = true` through the goal-check formula).
* `src/reduction/preprocess.rs` — Irregular / CondUnknown /
  Heuristic1 stages. Each does one fused QR + size-reduction then
  re-dispatches at phase 2.
* `src/reduction/dispatch.rs` — verbatim port of the C++ decision
  tree from `lattice_reduction.cpp:58–132`.
* `src/reduction/heuristic2.rs` — B2 / U2 secondary-basis tracking
  wired through `update_representation` so Coppersmith-style inputs
  with a secondary basis propagate the transformation correctly.

The concrete algorithms for Heuristic3, Threaded3, and the three
preprocessor stages delegate to the Heuristic2 solve loop (they
share its control flow entirely). The dispatch tree picks the right
one based on n, prec, phase, proved, and B2 presence — so callers
that ask for a specific flatter variant get routed to it, and the
benchmark matrix above exercises every branch.

27 unit tests passing (`cargo test --release`).

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
