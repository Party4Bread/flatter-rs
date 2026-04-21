//! `LatticeReductionParams`: inputs to the reduction algorithm. Trimmed
//! port of `src/problems/lattice_reduction/params.cpp` — we keep the
//! fields the single-basis CLI path actually sets.

use crate::reduction::goal::LatticeReductionGoal;

#[derive(Clone, Debug)]
pub struct LatticeReductionParams {
    pub goal: LatticeReductionGoal,
    pub rhf: f64,
    pub log_cond: f64,
    pub proved: bool,
    /// Classical-LLL delta derived from `goal.alpha`. For the native LLL
    /// path we clamp to [0.5, 0.99] to avoid both loop divergence (>1) and
    /// the algorithm giving up too easily (too low).
    pub delta: f64,
}

impl LatticeReductionParams {
    pub fn from_goal(goal: LatticeReductionGoal) -> Self {
        let rhf = goal.rhf();
        // `goal.delta()` maps the flatter RHF target to a classical LLL δ
        // via `0.255 / sqrt(alpha)`. For alpha=0.0625 (default) this gives
        // δ ≈ 1.02, above LLL's convergence range — we clamp to 0.99. δ=0.99
        // gives near-optimal LLL quality at the cost of more swaps than a
        // looser δ like 0.75 would need, but the Cohen incremental update
        // makes each swap cheap, so end-to-end it's competitive and we
        // reliably meet any reasonable `rhf` target.
        let delta = goal.delta().clamp(0.5, 0.99);
        Self {
            goal,
            rhf,
            log_cond: 0.0,
            proved: false,
            delta,
        }
    }
}
