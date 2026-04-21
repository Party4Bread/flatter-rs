//! `Heuristic2`: flatter's single-sublattice iterated-compression
//! heuristic. Port of `src/problems/lattice_reduction/heuristic_2.cpp`
//! (with Heuristic3/RecursiveGeneric helpers folded in) simplified for
//! the no-B2 case. B2/Coppersmith support is a layered addition in a
//! follow-up commit.
//!
//! The driver loop is the one from `recursive_generic.cpp:409`:
//! ```text
//! init_solver()
//! while !is_reduced():
//!     init_iter()
//!     setup_sublattice_reductions()
//!     reduce_sublattices()              ← recurse into reduce()
//!     update_representation()
//!     fini_iter()                       ← advance SublatticeSplit
//! fini_solver()
//! ```
//!
//! Per Phase-2 sublattice-split policy, one sub-window per round
//! iterates through `left → right → all`. When a sub-window is small
//! enough (entries fit the f64 LLL path), the inner `reduce()` call
//! resolves directly instead of recursing again.

use rug::{Assign, Integer};

use crate::lattice::{IntMatrix, Lattice};
use crate::math::{fused_qr_sr, mat_mpfr::MatMpfr, mat_mul};
use crate::profile::Profile;
use crate::reduction::params::LatticeReductionParams;
use crate::reduction::recursive_generic::RecursiveGeneric;
use crate::reduction::sublattice_split::{Split, Sublattice};

pub struct Heuristic2 {
    pub base: RecursiveGeneric,
}

/// Safety cap on outer rounds — Phase-2 is at most `cycle_len ·
/// ⌈log₂ n⌉` rounds per top-level call, so a linear bound with slack
/// is plenty.
fn max_outer_rounds(n: usize) -> usize {
    (n * (n as f64).log2().ceil() as usize * 4).max(32)
}

impl Heuristic2 {
    pub fn new(outer_m: IntMatrix, params: LatticeReductionParams) -> Self {
        let base = RecursiveGeneric::new(outer_m, params);
        Self { base }
    }

    /// Heuristic2-specific init. Computes QR of the outer basis
    /// (producing an upper-triangular R), reads the profile from R's
    /// diagonal, and compresses R into the integer shadow B using a
    /// **single uniform shift** (heuristic_2.cpp:218–224).
    ///
    /// Unlike the C++ version — which assumes M is already upper
    /// triangular on entry and just reads M[i, i] — we handle arbitrary
    /// input bases by doing the QR ourselves. That covers the normal
    /// q-ary input format from latticegen.
    fn init_solver_heuristic2(&mut self) {
        let n = self.base.n;
        let m = self.base.m;

        self.base.b = IntMatrix::zeros(m, n);
        self.base.b_next = IntMatrix::zeros(m, n);
        self.base.profile = Profile::new(n);
        self.base.local_profile_offsets = vec![0.0; n];
        self.base.global_profile_offsets = self.base.params.profile_offset.clone();
        if self.base.global_profile_offsets.len() < n {
            self.base.global_profile_offsets.resize(n, 0.0);
        }
        self.base.u_iters.clear();
        self.base.compression_iters.clear();

        // 1. QR-factor the outer basis into R (upper triangular, MPFR).
        let precision = self.base.get_initial_precision();
        self.base.original_precision = precision;
        self.base.precision = precision;
        self.base.r = MatMpfr::zeros(m, n, precision);
        for i in 0..m {
            for j in 0..n {
                self.base.r.get_mut(i, j).assign(self.base.outer_m.get(i, j));
            }
        }
        let mut tau = Vec::new();
        crate::math::qr::householder_qr(&mut self.base.r, &mut tau);
        crate::math::qr::clear_subdiagonal(&mut self.base.r);

        // 2. Read profile from R[i, i].
        self.base.set_profile();

        // 3. Compute per-column shifts then UNIFY to shifts[0]
        //    (heuristic_2.cpp:218–224). A single uniform shift keeps
        //    the profile SHAPE intact so `is_reduced` is meaningful.
        let mut shifts = vec![0i32; n];
        self.base.get_shifts_for_compression(&mut shifts);
        let total_shift = shifts[0];
        for s in shifts.iter_mut() {
            *s = total_shift;
        }
        self.base.compression_iters.push(shifts);

        // 4. Apply total_shift to profile and offsets.
        for i in 0..n {
            self.base.local_profile_offsets[i] += total_shift as f64;
            self.base.global_profile_offsets[i] += total_shift as f64;
            self.base.profile[i] -= total_shift as f64;
        }

        // 5. B := round(R · 2^{-total_shift}) — integer upper-triangular
        //    shadow. R being upper-triangular means B is too.
        let mut tmp = rug::Float::new(precision);
        for i in 0..n {
            for j in 0..n {
                if i > j {
                    self.base.b.set(i, j, rug::Integer::new());
                } else {
                    rug::Assign::assign(&mut tmp, self.base.r.get(i, j));
                    if total_shift < 0 {
                        tmp <<= (-total_shift) as u32;
                    } else {
                        tmp >>= total_shift as u32;
                    }
                    tmp.round_mut();
                    let v = tmp.clone().to_integer().unwrap_or_else(rug::Integer::new);
                    self.base.b.set(i, j, v);
                }
            }
        }

        self.base.lattice_changed = true;
    }

    /// Top-level solve: returns (final profile, number of outer
    /// iterations). The reduced basis lands in `base.outer_m`.
    pub fn solve(&mut self) -> (Profile, usize) {
        // Heuristic2 overrides init_compressed_B to use a single shift
        // (not per-column) — see heuristic_2.cpp:206–261. Do the init
        // manually here so we can call that variant.
        self.init_solver_heuristic2();

        let cap = max_outer_rounds(self.base.n);
        let mut rounds = 0usize;

        // Make sure we have a split tree to drive from.
        let have_split = self.base.params.split.is_some();
        if !have_split {
            // Phase-2 with a default tree rooted at rank n.
            self.base.params.split = Some(Split::new_phase2(self.base.n));
        }

        while !self.is_reduced() {
            if rounds >= cap {
                break;
            }
            // Check split stopping point: if the SublatticeSplit has
            // exhausted its schedule for this level, bail.
            let split_done = self
                .base
                .params
                .split
                .as_ref()
                .map(|s| s.borrow().stopping_point())
                .unwrap_or(true);
            if split_done {
                break;
            }

            rounds += 1;

            self.base.init_iter();
            let window = self.setup_sublattice_reduction();
            let sub_u = self.reduce_sub(window);
            self.update_representation(window, sub_u);

            // fini_iter: advance the split.
            if let Some(split) = &self.base.params.split {
                split.borrow_mut().advance_sublattices();
            }
        }

        self.base.fini_solver();
        (self.base.profile.clone(), rounds)
    }

    /// `Heuristic2::is_reduced` (heuristic_2.cpp:19).
    ///
    /// Note on the `iterations == 0` short-circuit: C++ checks the
    /// raw profile directly, but in our port the profile at this
    /// point has been run through `compress_R`, which subtracts the
    /// shift vector and flattens it artificially. So we must reapply
    /// the local offsets to compare against the uncompressed goal.
    /// `Heuristic2::is_reduced` (heuristic_2.cpp:19).
    fn is_reduced(&self) -> bool {
        if rounds_so_far(&self.base) == 0 {
            // Reconstruct the pre-compression profile so goal.check
            // sees the *real* drop rather than the flattened one.
            let mut un = self.base.profile.clone();
            for i in 0..self.base.n {
                un[i] += self.base.local_profile_offsets[i];
            }
            if self.base.params.goal.check(&un) {
                return true;
            }
        }
        self.base
            .params
            .split
            .as_ref()
            .map(|s| s.borrow().stopping_point())
            .unwrap_or(false)
    }

    /// Pick the current window via the attached SublatticeSplit.
    fn setup_sublattice_reduction(&self) -> Sublattice {
        let split = self
            .base
            .params
            .split
            .as_ref()
            .expect("split must be set by solve()");
        let windows = split.borrow().get_sublattices();
        assert_eq!(
            windows.len(),
            1,
            "Heuristic2 expects exactly one sublattice per iteration"
        );
        windows[0]
    }

    /// Build a sub-Lattice from the current B at `[start..end,
    /// start..end]`, recursively reduce it, return the U that
    /// transforms the sub-basis.
    fn reduce_sub(&self, window: Sublattice) -> IntMatrix {
        let (start, end) = window;
        let k = end - start;

        // sub_B = B[start..end, start..end]  (B is n×n upper-triangular)
        let sub_b = self.base.b.submatrix(start, end, start, end);

        // Build a fresh Lattice for the sub-problem.
        let mut sub_lat = Lattice::new(k, k);
        sub_lat.basis = sub_b;
        sub_lat.rank = k;
        sub_lat.profile = Profile::new(k);
        // Seed sub-profile from the current profile slice (used by
        // is_reduced checks inside the sub-call).
        for i in 0..k {
            sub_lat.profile[i] = self.base.profile[start + i];
        }

        // Build sub-params, descending to the matching child split.
        let child_split = self
            .base
            .params
            .split
            .as_ref()
            .expect("split set")
            .borrow()
            .get_child_split(0);
        let sub_params = self.base.params.subparams(start, end, child_split);

        // Recursively reduce. `reduce_with_u` wraps our LLL / inner
        // Heuristic2 dispatch and tracks U for us.
        let mut u_sub = IntMatrix::zeros(k, k);
        u_sub.set_identity();
        crate::reduction::lll::reduce_with_u(&mut sub_lat, &sub_params, &mut u_sub);

        // NOTE: we discard the reduced sub_lat.basis and sub_lat.profile
        // here — `update_representation` recomputes R/profile on
        // B_next, which already reflects the U_sub transform.
        u_sub
    }

    /// `RecursiveGeneric::update_representation` (line 220). The
    /// no-B2 form: apply sub-U to B_next, fused-QR-size-reduce,
    /// compress, copy back to B.
    fn update_representation(&mut self, window: Sublattice, u_sub: IntMatrix) {
        let n = self.base.n;

        // U_i := I  (it was pushed in init_iter already; reinit here).
        {
            let u_i = self.base.u_iters.last_mut().expect("U_i from init_iter");
            u_i.set_identity();
        }

        let mut u_tmp = IntMatrix::zeros(n, n);
        u_tmp.set_identity();
        // Place u_sub into u_tmp[start..end, start..end].
        u_tmp.copy_submatrix_from(window.0, window.0, &u_sub);

        // B_next := B (copy).
        self.base.b_next = self.base.b.clone();

        // B_next := B_next · U_tmp
        self.base.b_next = mat_mul::mat_mul(&self.base.b_next, &u_tmp);

        // Fused QR + size reduction on B_next, producing a fresh R and
        // additional U_sr (the size-reduction-induced column ops).
        let prec = self.base.precision.max(128);
        let mut r_new = MatMpfr::zeros(n, n, prec);
        let mut u_sr = IntMatrix::zeros(n, n);
        fused_qr_sr::fused_qr_sr(&mut self.base.b_next, &mut r_new, &mut u_sr);
        self.base.r = r_new;

        // U_tmp := U_tmp · U_sr
        let tmp = mat_mul::mat_mul(&u_tmp, &u_sr);
        u_tmp = tmp;

        // U_i := U_i · U_tmp  (just U_tmp since U_i was I).
        {
            let u_i = self.base.u_iters.last_mut().expect("U_i from init_iter");
            *u_i = u_tmp.clone();
        }

        // Profile ← R diagonal, then compress R, copy R to B.
        self.base.set_profile();
        self.base.compress_R();

        // B := R rounded to integer (R is upper triangular now).
        self.base.b = mpfr_mat_to_int(&self.base.r, n);
    }
}

fn rounds_so_far(rg: &RecursiveGeneric) -> usize {
    rg.u_iters.len()
}

fn mpfr_mat_to_int(r: &MatMpfr, n: usize) -> IntMatrix {
    let m = r.nrows;
    let mut out = IntMatrix::zeros(m, n);
    let mut tmp = rug::Float::new(r.prec);
    for i in 0..m {
        for j in 0..n {
            rug::Assign::assign(&mut tmp, r.get(i, j));
            tmp.round_mut();
            if let Some(v) = tmp.clone().to_integer() {
                out.set(i, j, v);
            }
        }
    }
    let _ = Integer::new();
    out
}
