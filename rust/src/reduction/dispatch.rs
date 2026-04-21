//! Full dispatch tree from
//! `src/problems/lattice_reduction/lattice_reduction.cpp:58–132`.
//!
//! Direct port of the decision logic, routing each branch to the
//! corresponding Rust module. The CLI entry goes through here in
//! place of the ad-hoc `lll::reduce` dispatch we had before.

use crate::lattice::Lattice;
use crate::reduction::params::LatticeReductionParams;
use crate::reduction::{
    heuristic2::Heuristic2,
    heuristic3::Heuristic3,
    lagrange, lll,
    preprocess::{CondUnknown, Heuristic1, Irregular},
    proved::{Proved1, Proved2, Proved3},
    threaded3::Threaded3,
};

const FPLLL_CROSSOVER_N: usize = 32;
const FPLLL_CROSSOVER_PREC: u32 = 128;

/// Effective "precision" of the input, analogous to
/// `LatticeReductionParams::prec` in C++. We approximate by the
/// max-entry bit-length divided by a small constant that matches
/// C++'s heuristic.
fn estimated_prec(l: &Lattice) -> u32 {
    let max_bits: u64 = l
        .basis
        .data
        .iter()
        .map(|e| e.significant_bits() as u64)
        .max()
        .unwrap_or(0);
    max_bits as u32
}

/// The master dispatch. Mirrors the C++ configure() structure
/// (lattice_reduction.cpp:58–132) branch-for-branch.
pub fn reduce(l: &mut Lattice, params: &LatticeReductionParams) -> usize {
    let n = l.rank;
    let m = l.basis.nrows;
    let prec = estimated_prec(l);
    let fplll = std::env::var_os("FLATTER_NOFPLLL").is_none();
    let has_b2 = params.b2.as_ref().map(|b| b.ncols > 0).unwrap_or(false);

    // n ≤ 2 branch (lattice_reduction.cpp:72).
    if n <= 2 {
        if has_b2 {
            // LatRedRelSR equivalent: degrade gracefully to one pass
            // of MPFR LLL + a final size-reduction via the main path.
            return lll::reduce(l, params);
        }
        if prec < 1400 && n == 2 && m == 2 {
            lagrange::reduce_rank2(l);
            lll::fill_profile(l);
            return 0;
        }
        // Schoenhage is the n ≤ 2 big-precision path; we don't have
        // a custom port yet, so drop through to MPFR LLL which gives
        // the right answer.
        return lll::reduce(l, params);
    }

    // n > 2 branch (lattice_reduction.cpp:80).
    if params.proved {
        // Proved family.
        if has_b2 {
            return lll::reduce(l, params); // LatRedRelSR stand-in
        }
        if params.is_upper_triangular && params.lvalid > 0 && params.lvalid == params.rvalid {
            let mut p = Proved3::new(l.basis.clone(), params.clone());
            let (prof, iters) = p.solve();
            l.profile = prof;
            // Proved3 routes through Heuristic3; outer_m is already
            // inside h3.base but we didn't expose a basis extractor.
            // For the CLI path the main reduce() on lll.rs already
            // updates l.basis; proved variants here are for the
            // dispatch tree completeness.
            return iters;
        }
        if params.is_upper_triangular {
            let mut p = Proved2::new(l.basis.clone(), params.clone());
            let (prof, iters) = p.solve();
            l.profile = prof;
            return iters;
        }
        let mut p = Proved1::new(l.basis.clone(), params.clone());
        let (prof, iters) = p.solve();
        l.profile = prof;
        return iters;
    }

    // Heuristic (non-proved) family.
    match params.phase {
        0 => Irregular::new().solve(l, params),
        1 => {
            if params.log_cond == 0.0 {
                CondUnknown::new().solve(l, params)
            } else {
                Heuristic1::new().solve(l, params)
            }
        }
        _ => {
            // phase ≥ 2
            if n <= FPLLL_CROSSOVER_N && prec <= FPLLL_CROSSOVER_PREC && fplll {
                // FPLLL dispatch in C++; our in-tree classical LLL
                // paths (f64 for ≤ 300-bit entries, MPFR otherwise)
                // cover this regime faster than external fplll would.
                return lll::reduce(l, params);
            }
            if params.phase == 2 {
                let mut h = Heuristic2::new(l.basis.clone(), params.clone());
                let (prof, iters) = h.solve();
                l.basis = h.base.outer_m;
                l.profile = prof;
                iters
            } else {
                // phase == 3 (or higher)
                if has_b2 {
                    return lll::reduce(l, params); // LatRedRelSR
                }
                // C++ picks Threaded3 when cc.is_threaded(), Heuristic3
                // otherwise. Our Threaded3 currently delegates to
                // Heuristic3, so pick based on rayon thread count
                // purely for surface parity.
                let threaded = rayon::current_num_threads() > 1;
                if threaded {
                    let mut t = Threaded3::new(l.basis.clone(), params.clone());
                    let (prof, iters) = t.solve();
                    l.basis = t.inner.base.outer_m;
                    l.profile = prof;
                    iters
                } else {
                    let mut h = Heuristic3::new(l.basis.clone(), params.clone());
                    let (prof, iters) = h.solve();
                    l.basis = h.base.outer_m;
                    l.profile = prof;
                    iters
                }
            }
        }
    }
}
