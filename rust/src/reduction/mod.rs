//! Lattice reduction dispatch. Port of
//! `src/problems/lattice_reduction/lattice_reduction.cpp` with a trimmed
//! implementation surface — see the implementation status in each module.

pub mod dispatch;
pub mod goal;
pub mod heuristic;
pub mod heuristic2;
pub mod heuristic3;
pub mod lagrange;
pub mod lll;
pub mod params;
pub mod preprocess;
pub mod proved;
pub mod recursive_generic;
pub mod sublattice_split;
pub mod threaded3;

pub use goal::LatticeReductionGoal;
pub use params::LatticeReductionParams;

use crate::lattice::Lattice;

/// Top-level dispatch + solve. Now routes through
/// `dispatch::reduce`, the verbatim port of flatter's decision tree
/// from `lattice_reduction.cpp:58–132`.
pub fn reduce(L: &mut Lattice, params: &LatticeReductionParams) -> ReductionReport {
    let iters = dispatch::reduce(L, params);
    ReductionReport {
        algorithm: "dispatch",
        iterations: iters,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ReductionReport {
    pub algorithm: &'static str,
    pub iterations: usize,
}
