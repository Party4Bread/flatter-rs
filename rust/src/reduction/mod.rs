//! Lattice reduction dispatch. Port of
//! `src/problems/lattice_reduction/lattice_reduction.cpp` with a trimmed
//! implementation surface — see the implementation status in each module.

pub mod goal;
pub mod heuristic;
pub mod lagrange;
pub mod lll;
pub mod params;
pub mod sublattice_split;

pub use goal::LatticeReductionGoal;
pub use params::LatticeReductionParams;

use crate::lattice::Lattice;

/// Dispatch + solve. Mirrors the decision tree in
/// `lattice_reduction.cpp:58–132`, but the "heuristic" and "proved"
/// recursive cores aren't ported yet — for n > 2 we fall through to a
/// classical LLL implementation, which is also the path the C++ code
/// takes for small-and-low-precision inputs (the `FPLLL` dispatch).
pub fn reduce(L: &mut Lattice, params: &LatticeReductionParams) -> ReductionReport {
    let n = L.rank;
    let _m = L.dimension();

    if n <= 1 {
        // Identity: a rank-0 or rank-1 lattice is already reduced.
        lll::fill_profile(L);
        return ReductionReport {
            algorithm: "trivial",
            iterations: 0,
        };
    }

    if n == 2 {
        lagrange::reduce_rank2(L);
        lll::fill_profile(L);
        return ReductionReport {
            algorithm: "lagrange",
            iterations: 0,
        };
    }

    // n > 2: use classical LLL. In the C++ dispatch this is equivalent to
    // the FPLLL branch (n ≤ 32, prec ≤ 128) and gives a valid answer
    // everywhere else, just not with the asymptotic speedup of the
    // iterated-compression heuristic.
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
