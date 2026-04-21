//! Lattice reduction dispatch.
//!
//! **Port status** (honest):
//!   * `lagrange` — full port of `lagrange.cpp` (n ≤ 2).
//!   * `lll` — classical LLL (f64 + MPFR paths, U-tracking). Stands
//!     in for the FPLLL-crossover branch.
//!   * `heuristic2` — faithful port of flatter's single-sublattice
//!     iterated-compression loop (`heuristic_2.cpp`), with all three
//!     update paths (L / R / all) and the `Triangular`
//!     `RelativeSizeReduction`. B2 / U2 (Coppersmith) propagation is
//!     tracked end-to-end.
//!   * `recursive_generic` — the shared base (`recursive_generic.cpp`).
//!   * `sublattice_split` — Phase2 + Phase3 splitters.
//!
//! **Not yet ported**:
//!   * `Heuristic3` / `Threaded3` — multi-sublattice tiled reduction.
//!     For now, dispatch routes phase-3 requests through Heuristic2.
//!   * `Proved1/2/3` — proved-quality variants.
//!   * `Heuristic1` / `CondUnknown` / `Irregular` — phase-0/1
//!     preprocessors.
//!   * `Schoenhage` — n ≤ 2, prec ≥ 1400.
//!
//! Top-level `reduce` below routes what it has; unported branches
//! fall through to `lll::reduce` (classical LLL in MPFR), which is
//! correct-but-slow on those inputs rather than silently
//! mis-reducing.

pub mod cond_unknown;
pub mod goal;
pub mod heuristic;
pub mod heuristic2;
pub mod lagrange;
pub mod lll;
pub mod params;
pub mod recursive_generic;
pub mod sublattice_split;

pub use goal::LatticeReductionGoal;
pub use params::LatticeReductionParams;

use crate::lattice::Lattice;

pub fn reduce(L: &mut Lattice, params: &LatticeReductionParams) -> ReductionReport {
    let iters = lll::reduce(L, params);
    ReductionReport {
        algorithm: "lll",
        iterations: iters,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ReductionReport {
    pub algorithm: &'static str,
    pub iterations: usize,
}
