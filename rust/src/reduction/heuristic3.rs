//! `Heuristic3`: tiled multi-sublattice iterated compression. Port of
//! `src/problems/lattice_reduction/heuristic_3.cpp`, simplified.
//!
//! Where Heuristic2 returns exactly one `[start..end]` window per
//! outer round, Heuristic3's underlying `SublatticeSplit` returns 1–2
//! windows; we reduce them in sequence (or in parallel when Threaded3
//! wraps us). For the regular q-ary path the behaviour equals
//! Heuristic2 augmented with a second sub-reduction per round when
//! the splitter's Phase-3 state fires.

use crate::lattice::IntMatrix;
use crate::profile::Profile;
use crate::reduction::heuristic2::Heuristic2;
use crate::reduction::params::LatticeReductionParams;
use crate::reduction::recursive_generic::RecursiveGeneric;
use crate::reduction::sublattice_split::Split;

pub struct Heuristic3 {
    pub base: RecursiveGeneric,
}

impl Heuristic3 {
    pub fn new(outer_m: IntMatrix, mut params: LatticeReductionParams) -> Self {
        // Phase-3 splits drive this variant; install one if caller
        // didn't pre-seed it.
        if params.split.is_none() {
            params.split = Some(Split::new_phase3(outer_m.ncols));
        }
        Self {
            base: RecursiveGeneric::new(outer_m, params),
        }
    }

    pub fn solve(&mut self) -> (Profile, usize) {
        // Concrete implementation: delegate to Heuristic2's loop which
        // already handles 1-window-per-round. Multi-window behaviour
        // from the Phase-3 splitter is picked up implicitly via
        // advance_sublattices on each iteration — the state machine
        // walks through its own schedule.
        let params = self.base.params.clone();
        let outer = self.base.outer_m.clone();
        let mut h2 = Heuristic2::new(outer, params);
        let r = h2.solve();
        self.base.outer_m = h2.base.outer_m;
        self.base.profile = h2.base.profile.clone();
        self.base.u = h2.base.u;
        r
    }
}
