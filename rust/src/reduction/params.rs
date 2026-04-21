//! `LatticeReductionParams`: inputs to the recursive reduction. Ported
//! from `include/flatter/problems/lattice_reduction/params.h` and
//! `src/problems/lattice_reduction/params.cpp`.
//!
//! We keep everything the C++ header carries (phase, proved, offset,
//! profile_offset, lvalid/rvalid, B2/U2, log_cond, aggressive_precision)
//! so sub-params we build during the recursive descent look exactly
//! like what flatter's setup routines produce.

use std::cell::RefCell;
use std::rc::Rc;

use crate::lattice::IntMatrix;
use crate::reduction::goal::LatticeReductionGoal;
use crate::reduction::sublattice_split::SplitRc;

#[derive(Clone)]
pub struct LatticeReductionParams {
    pub goal: LatticeReductionGoal,
    /// Target RHF — information-carrying; the reduction decision is
    /// driven by `goal`, but several sub-calls re-derive from `rhf`.
    pub rhf: f64,
    pub proved: bool,
    pub phase: u32,
    /// True iff the input basis is already upper-triangular (skips the
    /// initial QR in recursive_generic).
    pub is_upper_triangular: bool,

    /// Optional secondary basis (Coppersmith-style). `None` for pure
    /// q-ary inputs from the CLI.
    pub b2: Option<IntMatrix>,
    pub u2: Option<IntMatrix>,

    /// Classical-LLL δ derived from `goal.quality`. Used only by the
    /// non-recursive LLL paths.
    pub delta: f64,

    /// Condition-number bound (`-logcond` CLI flag). 0 means "unknown".
    pub log_cond: f64,
    pub aggressive_precision: bool,

    /// Profile offset carried across recursive calls to preserve the
    /// global notion of "log₂|b*_i|" even as we descend into compressed
    /// sublattices.
    pub profile_offset: Vec<f64>,

    /// Global offset of this sub-problem's column 0 in the original
    /// basis. Used for logging / profile_update callbacks.
    pub offset: usize,

    /// `[lvalid, rvalid)` — the sub-range of columns currently
    /// considered validly reduced. Used by the "proved" variants.
    pub lvalid: usize,
    pub rvalid: usize,

    /// Sublattice splitter for this (sub-)problem. Optional because the
    /// top-level CLI entry constructs it just-in-time via the dispatch.
    pub split: Option<SplitRc>,
}

impl LatticeReductionParams {
    pub fn from_goal(goal: LatticeReductionGoal) -> Self {
        let n = goal.n;
        let rhf = goal.get_rhf();
        let slope = goal.get_slope();
        let delta = if slope > 0.0 {
            (0.255 / slope.sqrt()).clamp(0.5, 0.99)
        } else {
            0.99
        };
        Self {
            goal,
            rhf,
            proved: false,
            // Match C++ params.cpp:43 — default phase is 0, so the CLI
            // entry routes through Irregular → CondUnknown → Heuristic2.
            phase: 0,
            is_upper_triangular: false,
            b2: None,
            u2: None,
            delta,
            log_cond: 0.0,
            aggressive_precision: false,
            profile_offset: vec![0.0; n],
            offset: 0,
            lvalid: 0,
            rvalid: 0,
            split: None,
        }
    }

    /// Build sub-params for a recursive call on columns `[start..end]`.
    /// Mirrors the parameter-construction logic in
    /// `Heuristic2::setup_sublattice_reductions` (heuristic_2.cpp:89).
    pub fn subparams(&self, start: usize, end: usize, child_split: SplitRc) -> Self {
        let sub_goal = self.goal.subgoal(start, end);
        let profile_offset =
            self.profile_offset[start..end].to_vec();
        Self {
            goal: sub_goal.clone(),
            rhf: sub_goal.get_rhf(),
            proved: self.proved,
            phase: self.phase,
            is_upper_triangular: false,
            b2: None,
            u2: None,
            delta: self.delta,
            log_cond: self.log_cond,
            aggressive_precision: self.aggressive_precision,
            profile_offset,
            offset: self.offset + start,
            lvalid: 0,
            rvalid: 0,
            split: Some(child_split),
        }
    }
}

/// Small convenience for making sublattice-split chains shareable.
pub fn share(s: SplitRc) -> Rc<RefCell<super::sublattice_split::Split>> {
    s
}
