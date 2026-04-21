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
