//! `Threaded3`: rayon-parallel wrapper around Heuristic3. Port of
//! `src/problems/lattice_reduction/threaded_3.cpp`, simplified.
//!
//! C++ uses OpenMP tasks to process multiple disjoint sub-windows in
//! parallel when the Phase-3 splitter hands out more than one at a
//! time. Our `Heuristic3` currently walks them sequentially via the
//! `Heuristic2` driver; the hook for parallelism is here so the
//! dispatch tree can route at `phase == 3 && threaded` without
//! breaking. When the inner loop grows a real multi-window scheduler
//! this is where `rayon::scope` will live.

use crate::lattice::IntMatrix;
use crate::profile::Profile;
use crate::reduction::heuristic3::Heuristic3;
use crate::reduction::params::LatticeReductionParams;

pub struct Threaded3 {
    pub inner: Heuristic3,
}

impl Threaded3 {
    pub fn new(outer_m: IntMatrix, params: LatticeReductionParams) -> Self {
        Self {
            inner: Heuristic3::new(outer_m, params),
        }
    }

    pub fn solve(&mut self) -> (Profile, usize) {
        self.inner.solve()
    }
}
