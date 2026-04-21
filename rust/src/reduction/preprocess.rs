//! Preprocessor stages: `Irregular` (phase 0), `CondUnknown` /
//! `Heuristic1` (phase 1). Ports of
//! `src/problems/lattice_reduction/{irregular,cond_unknown,heuristic_1}.cpp`,
//! simplified.
//!
//! What these stages do in C++:
//!   * **Irregular** (phase 0): handles rank-deficient or malformed
//!     inputs — permutes out zero vectors, size-reduces, then calls
//!     recursively at phase 1.
//!   * **CondUnknown** (phase 1, `log_cond == 0`): estimates the
//!     condition number by running an inexpensive preliminary
//!     reduction; then dispatches to phase 2 with the estimate.
//!   * **Heuristic1** (phase 1, `log_cond != 0`): same shape as
//!     Heuristic2 but uses `log_cond` to pick precision upfront and
//!     runs a single pass at that precision.
//!
//! All three ultimately hand off to `Heuristic2`/`Heuristic3`. Our
//! port keeps that contract: we QR + size-reduce the outer basis
//! once (enough for the non-degenerate q-ary inputs we care about),
//! then re-enter the dispatch at phase 2.

use crate::lattice::{IntMatrix, Lattice};
use crate::math::{fused_qr_sr, mat_mpfr::MatMpfr};
use crate::profile::Profile;
use crate::reduction::params::LatticeReductionParams;

/// Common entry point: size-reduce once, bump phase, re-dispatch.
pub fn run_preprocessor(
    l: &mut Lattice,
    params: &LatticeReductionParams,
) -> usize {
    // Preprocess: one fused QR + size reduction at generous precision.
    // Makes the subsequent Heuristic2 init_compressed_B start from an
    // already-size-reduced basis.
    let max_bits: u64 = l
        .basis
        .data
        .iter()
        .map(|e| e.significant_bits() as u64)
        .max()
        .unwrap_or(0);
    let m = l.basis.nrows;
    let n = l.basis.ncols;
    let prec =
        (max_bits as u32).saturating_add((n as u32).next_power_of_two().trailing_zeros() + 64).max(128);
    let mut r = MatMpfr::zeros(m, n, prec);
    let mut u = IntMatrix::zeros(n, n);
    fused_qr_sr::fused_qr_sr(&mut l.basis, &mut r, &mut u);

    // Re-dispatch with phase bumped. At phase ≥ 2 reduce() routes
    // through the main LLL → Heuristic2 path.
    let mut new_params = params.clone();
    new_params.phase = 2;
    crate::reduction::lll::reduce(l, &new_params)
}

pub struct Irregular;
pub struct CondUnknown;
pub struct Heuristic1;

impl Irregular {
    pub fn new() -> Self {
        Self
    }
    pub fn solve(&self, l: &mut Lattice, params: &LatticeReductionParams) -> usize {
        // C++ Irregular handles rank-0 / rank-1 / zero-column cases.
        // Delegate to the trivial-rank branches already in
        // reduction::mod::reduce, then preprocess if we still have
        // rank ≥ 2.
        if l.rank <= 1 {
            let _ = Profile::new(l.rank);
            return 0;
        }
        run_preprocessor(l, params)
    }
}

impl CondUnknown {
    pub fn new() -> Self {
        Self
    }
    pub fn solve(&self, l: &mut Lattice, params: &LatticeReductionParams) -> usize {
        run_preprocessor(l, params)
    }
}

impl Heuristic1 {
    pub fn new() -> Self {
        Self
    }
    pub fn solve(&self, l: &mut Lattice, params: &LatticeReductionParams) -> usize {
        run_preprocessor(l, params)
    }
}
