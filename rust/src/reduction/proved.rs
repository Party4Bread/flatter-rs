//! `Proved1`, `Proved2`, `Proved3`: proved-quality variants of the
//! corresponding heuristics. Ports of
//! `src/problems/lattice_reduction/proved_{1,2,3}.cpp`.
//!
//! Structurally identical to the heuristic versions — the difference
//! is that `params.proved = true` propagates down to
//! `LatticeReductionGoal::check`, which uses the proved α bound
//! (`get_alpha_n` formula) instead of the heuristic one. All three
//! proved variants therefore delegate to Heuristic2/3 with the proved
//! flag set; the ported `goal.rs` already handles the branch.

use crate::lattice::IntMatrix;
use crate::profile::Profile;
use crate::reduction::heuristic2::Heuristic2;
use crate::reduction::heuristic3::Heuristic3;
use crate::reduction::params::LatticeReductionParams;

pub struct Proved1 {
    inner: Heuristic2,
}
pub struct Proved2 {
    inner: Heuristic2,
}
pub struct Proved3 {
    inner: Heuristic3,
}

macro_rules! proved_impl {
    ($name:ident, $wrapped:ty, $ctor:expr) => {
        impl $name {
            pub fn new(outer_m: IntMatrix, mut params: LatticeReductionParams) -> Self {
                params.proved = true;
                params.goal.proved = true;
                Self {
                    inner: $ctor(outer_m, params),
                }
            }
            pub fn solve(&mut self) -> (Profile, usize) {
                self.inner.solve()
            }
        }
    };
}

proved_impl!(Proved1, Heuristic2, Heuristic2::new);
proved_impl!(Proved2, Heuristic2, Heuristic2::new);
proved_impl!(Proved3, Heuristic3, Heuristic3::new);
