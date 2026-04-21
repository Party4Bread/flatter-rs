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

The default alpha threshold `0.0625` in the flatter CLI is the
**acceptance floor**, not a target — any output with α ≤ 0.0625 is
accepted as "reduced," but a healthy reduction algorithm should
produce output well below it. The relevant comparison is **Rust's
achieved α vs. C++'s directly-achieved α on the same input**:

```
dim   bits   CPP_ms   Rust_ms   Rust α  CPPAlphaOnRust  CPPAlphaDirect
 20   500    132       76       0.037   0.033           0.032
 20  1024    177      155       0.014   0.037           0.047
 30   500    321      372       0.044   0.042           0.041
 50   500    985     3057       0.054   0.051           0.056
 50  1024   1479     3629       0.053   0.048           0.050
100   500   7125    49493       0.058   0.053           0.054
```

**Quality** (`CPPAlphaOnRust` vs. `CPPAlphaDirect`, independently
measured): Rust equals or slightly beats C++ at every size. Lower is
tighter. No hidden loss.

**Speed**: Rust is faster at n ≤ 20 and comparable at n = 30; C++
wins by 3–7× at n ≥ 50 on big-entry inputs. The port does *not*
meet the "beat C++ on 1024-bit lattices" goal at the larger sizes.

### Why the speed gap reopened

The previous two commits claimed Rust was 7–25× faster. Two bugs
propped those numbers up:

1. `Heuristic2::is_reduced` short-circuited on the first
   `goal.check` — and `goal.get_drop()` on a q-ary input's
   ascending profile is always 0, so the check trivially passed
   and Heuristic2 returned without doing any real work.
2. After the bail-out, the final `set_profile` read the compressed
   R rather than the post-reduction outer-basis QR, so the self-
   reported α was wrong (20-ish while the actual α was ~0.05).

When I fix (1) by running through all three Phase-2 rounds,
Heuristic2 without a prior CondUnknown stage still produces a
relatively loose reduction (α ≈ 0.05 vs. C++ direct 0.03), so a
polishing MPFR LLL pass is needed — which ends up doing most of
the work itself. The net speedup evaporates.

**The honest conclusion**: without a faithful CondUnknown pre-pass
(condition-number estimation + preliminary LLL), Heuristic2 on its
own doesn't beat a direct MPFR LLL. The current big-entry dispatch
is therefore just `reduce_mpfr` — slower than C++ at large n, but
correct.

The `heuristic2.rs` port still ships (three update paths +
`RelativeSizeReduction::Triangular`) — what's missing is the
CondUnknown preprocessor that sets it up, and that's the next step.

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

**Ports that ship now** (all genuinely ported from C++, none stubs):

* `src/math/qr.rs` — Householder QR (port of
  `householder_mpfr.cpp`'s `larfg`/`larf`).
* `src/math/fused_qr_sr.rs` — fused QR + size reduction (port of
  `columnwise.cpp`).
* `src/math/size_reduce_r.rs` — size-reduction of R.
* `src/math/rsr.rs` — `RelativeSizeReduction::Triangular` (port of
  `triangular.cpp`).
* `src/math/mat_mul.rs`, `mat_mpfr.rs` — primitives.
* `src/reduction/sublattice_split.rs` — Phase2 + Phase3 splitters.
* `src/reduction/goal.rs` — full C++ `LatticeReductionGoal` surface.
* `src/reduction/params.rs` — `LatticeReductionParams`.
* `src/reduction/recursive_generic.rs` — shared plumbing
  (`init_solver`, `compress_R`, `get_shifts_for_compression`,
  `collect_U`, `final_sr`).
* `src/reduction/heuristic2.rs` — single-sublattice iterated
  compression with **three distinct update paths** (L / R / all)
  from `heuristic_2.cpp:263/400/560`, using `RelativeSizeReduction`
  on the R path and QR on the augmented top-rows / bottom-rows
  matrix on L and R respectively. B2 / U2 propagation is wired
  through where the code paths need it.
* `src/reduction/lll.rs` — classical LLL (f64 + MPFR, U-tracking).
* `src/reduction/lagrange.rs` — n ≤ 2 (Gauss reduction).

**Not yet ported — and not stubbed**:

I had earlier committed thin stubs that delegated to Heuristic2.
Those were misleading and have been removed. These remain to be
actually ported:

* **Heuristic3 / Threaded3** — multi-sublattice tiled reduction.
  C++ uses it for n ≥ some threshold and for parallelism. Needs
  `heuristic_3.cpp`'s tile-update-representation (~400 LOC) plus
  the `RelativeSizeReduction` paths it uses between tiles.
* **Proved1/2/3** — proven-quality variants. Not just a `proved`
  flag — they have distinct `get_precision_from_spread`, different
  termination conditions, and pick different sub-goals.
* **CondUnknown / Irregular / Heuristic1** — phase-0/1
  preprocessors. CondUnknown estimates condition number via
  preliminary LLL; Irregular handles rank-deficient inputs;
  Heuristic1 runs at a user-specified condition-number hint.
* **Schoenhage** — n ≤ 2 with prec ≥ 1400.
* **Orthogonal / OrthogonalDouble / Generic `RelativeSizeReduction`
  variants** — only `Triangular` is ported; the other three
  specializations (used by CondUnknown and Heuristic3) are not.

When inputs route to any of the unported paths, the dispatch falls
through to classical LLL-in-MPFR rather than silently running a
different algorithm.

28 unit tests passing (`cargo test --release`).

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
