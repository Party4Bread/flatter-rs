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
use crate::math::{mat_mpfr::MatMpfr, mat_mul, qr};
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
            let (sub_basis, u_sub, sub_profile) = self.reduce_sub(window);
            // Dispatch to the correct C++ update_*_representation based
            // on where the window sits — `heuristic_2.cpp:193`.
            let (start, end) = window;
            let n = self.base.n;
            if start == 0 && end < n {
                self.update_l_representation(window, sub_basis, u_sub, sub_profile);
            } else if start > 0 && end == n {
                self.update_r_representation(window, sub_basis, u_sub, sub_profile);
            } else {
                self.update_all_representation(window, u_sub, sub_profile);
            }

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
    /// start..end]`, recursively reduce it, return the reduced sub-basis
    /// and the U that transforms it.
    fn reduce_sub(&self, window: Sublattice) -> (IntMatrix, IntMatrix, Profile) {
        let (start, end) = window;
        let k = end - start;

        // sub_B = B[start..end, start..end]  (B is n×n upper-triangular)
        let sub_b = self.base.b.submatrix(start, end, start, end);

        let mut sub_lat = Lattice::new(k, k);
        sub_lat.basis = sub_b;
        sub_lat.rank = k;
        sub_lat.profile = Profile::new(k);
        for i in 0..k {
            sub_lat.profile[i] = self.base.profile[start + i];
        }

        let child_split = self
            .base
            .params
            .split
            .as_ref()
            .expect("split set")
            .borrow()
            .get_child_split(0);
        let sub_params = self.base.params.subparams(start, end, child_split);

        let mut u_sub = IntMatrix::zeros(k, k);
        u_sub.set_identity();
        crate::reduction::lll::reduce_with_u(&mut sub_lat, &sub_params, &mut u_sub);

        (sub_lat.basis, u_sub, sub_lat.profile)
    }

    // -------------------------------------------------------------
    // The three update paths from heuristic_2.cpp. Each one rebuilds
    // R for a different slice of the basis — this is the key detail
    // that propagates the sub-reduction's effect across block
    // boundaries in the full lattice.
    // -------------------------------------------------------------

    /// `Heuristic2::update_L_representation` (heuristic_2.cpp:263).
    /// Sub-window is `[0, end)` with `end < n`. QRs the TOP `end`
    /// rows of the full basis (not just the `end × end` submatrix)
    /// so the reduction propagates into the right columns.
    fn update_l_representation(
        &mut self,
        window: Sublattice,
        sub_basis: IntMatrix,
        u_sub: IntMatrix,
        sub_profile: Profile,
    ) {
        let (start, end) = window;
        assert_eq!(start, 0);
        let n = self.base.n;

        // profile[0..end] := sub_prof.
        for i in 0..end {
            self.base.profile[i] = sub_profile[i];
        }

        // U_i[start:end, start:end] := U_sub.
        {
            let u_i = self.base.u_iters.last_mut().expect("U_i");
            u_i.set_identity();
            u_i.copy_submatrix_from(start, start, &u_sub);
        }

        // B_next[start:end, start:end] := L_sub.basis  (the reduced
        // sub-basis; heuristic_2.cpp:309).
        // B_next[start:end, end:n] := snapshot of B[start:end, end:n]
        // (the "B2s" matrix in C++, just the right columns of B).
        // B_next[end:n, end:n] := B[end:n, end:n] (unchanged).
        self.base.b_next = IntMatrix::zeros(n, n);
        self.base.b_next.copy_submatrix_from(start, start, &sub_basis);
        let right_snapshot = self.base.b.submatrix(start, end, end, n);
        self.base.b_next.copy_submatrix_from(start, end, &right_snapshot);
        let untouched = self.base.b.submatrix(end, n, end, n);
        self.base.b_next.copy_submatrix_from(end, end, &untouched);

        // R := B_next[0:end, 0:n] as MPFR, then Householder QR.
        let spread = self.base.profile.get_spread();
        let precision = self.base.get_precision_from_spread(spread);
        let mut r = MatMpfr::zeros(end, n, precision);
        for i in 0..end {
            for j in 0..n {
                r.get_mut(i, j).assign(self.base.b_next.get(i, j));
            }
        }
        let mut tau = Vec::new();
        qr::householder_qr(&mut r, &mut tau);
        qr::clear_subdiagonal(&mut r);

        // profile[0..end] := log₂|R[i,i]|.
        for i in 0..end {
            let r_ii = r.get(i, i);
            self.base.profile[i] = if r_ii.is_zero() {
                f64::NEG_INFINITY
            } else {
                let (d, e) = r_ii.to_f64_exp();
                d.abs().log2() + e as f64
            };
        }

        // Compute single uniform total_shift; apply to R (top end rows),
        // profile, offsets, and B (bottom n-end rows untouched by QR).
        let mut shifts = vec![0i32; n];
        self.base.get_shifts_for_compression(&mut shifts);
        let total_shift = shifts[0];
        for s in shifts.iter_mut() {
            *s = total_shift;
        }
        *self.base.compression_iters.last_mut().unwrap() = shifts;
        for i in 0..n {
            self.base.local_profile_offsets[i] += total_shift as f64;
            self.base.global_profile_offsets[i] += total_shift as f64;
            self.base.profile[i] -= total_shift as f64;
        }

        // Scale R by 2^{-total_shift}, zero the strict lower.
        for i in 0..end {
            for j in 0..n {
                if i > j {
                    r.get_mut(i, j).assign(0);
                } else {
                    let cell = r.get_mut(i, j);
                    if total_shift < 0 {
                        *cell <<= (-total_shift) as u32;
                    } else {
                        *cell >>= total_shift as u32;
                    }
                }
            }
        }
        // Scale the untouched bottom tile B[end:n, end:n] by same shift,
        // zero strict lower (already upper-triangular so no-op).
        for i in end..n {
            for j in 0..n {
                if i > j {
                    self.base.b_next.set(i, j, rug::Integer::new());
                } else if j >= end {
                    let cell = self.base.b_next.get_mut(i, j);
                    if total_shift < 0 {
                        *cell <<= (-total_shift) as u32;
                    } else {
                        *cell >>= total_shift as u32;
                    }
                }
            }
        }

        // B[0:end, 0:n] := R (rounded to integer). B[end:n, end:n] :=
        // B_next[end:n, end:n] (already scaled).
        self.base.b = IntMatrix::zeros(n, n);
        let r_int = mpfr_mat_to_int_rows(&r, end, n);
        self.base.b.copy_submatrix_from(0, 0, &r_int);
        let bot = self.base.b_next.submatrix(end, n, end, n);
        self.base.b.copy_submatrix_from(end, end, &bot);

        // Keep R in state (bottom rows empty for the n × n slot,
        // upper-triangular after QR + clear).
        let mut full_r = MatMpfr::zeros(n, n, precision);
        for i in 0..end {
            for j in 0..n {
                full_r.get_mut(i, j).assign(r.get(i, j));
            }
        }
        self.base.r = full_r;
        self.base.precision = precision;
    }

    /// `Heuristic2::update_R_representation` (heuristic_2.cpp:400).
    /// Sub-window is `[start, n)` with `start > 0`. Uses
    /// `RelativeSizeReduction` on the left tile before QR-ing the
    /// bottom-right sub-matrix.
    fn update_r_representation(
        &mut self,
        window: Sublattice,
        sub_basis: IntMatrix,
        u_sub: IntMatrix,
        sub_profile: Profile,
    ) {
        let (start, end) = window;
        let n = self.base.n;
        assert_eq!(end, n);

        for i in start..n {
            self.base.profile[i] = sub_profile[i - start];
        }

        // U_i[start:n, start:n] := U_sub.
        {
            let u_i = self.base.u_iters.last_mut().expect("U_i");
            u_i.set_identity();
            u_i.copy_submatrix_from(start, start, &u_sub);
        }

        // B_next[0:start, 0:start] := B[0:start, 0:start]
        // B_next[0:start, start:n] := B[0:start, start:n] · U_sub   (matmul)
        // B_next[start:n, start:n] := sub_basis
        self.base.b_next = IntMatrix::zeros(n, n);
        let left_top = self.base.b.submatrix(0, start, 0, start);
        self.base.b_next.copy_submatrix_from(0, 0, &left_top);
        let right_top = self.base.b.submatrix(0, start, start, n);
        let right_top_updated = mat_mul::mat_mul(&right_top, &u_sub);
        self.base.b_next.copy_submatrix_from(0, start, &right_top_updated);
        self.base.b_next.copy_submatrix_from(start, start, &sub_basis);

        // RelativeSizeReduction: reduce B_next[0:start, start:n]
        // against B_next[0:start, 0:start] (which is upper-triangular).
        let b1 = self.base.b_next.submatrix(0, start, 0, start);
        let mut b2 = self.base.b_next.submatrix(0, start, start, n);
        let mut u_sr_slice = IntMatrix::zeros(start, n - start);
        {
            let mut rsr = crate::math::rsr::RsrTriangular::new(&b1, &mut b2, &mut u_sr_slice);
            rsr.solve();
        }
        // Write the reduced b2 and u_sr_slice back.
        self.base.b_next.copy_submatrix_from(0, start, &b2);

        // U_i := U_i · U_sr (where U_sr = I with u_sr_slice at
        // [0:start, start:n]).
        let mut u_sr_full = IntMatrix::zeros(n, n);
        u_sr_full.set_identity();
        u_sr_full.copy_submatrix_from(0, start, &u_sr_slice);
        let u_i_new = {
            let u_i = self.base.u_iters.last().unwrap();
            mat_mul::mat_mul(u_i, &u_sr_full)
        };
        *self.base.u_iters.last_mut().unwrap() = u_i_new;

        // R := B_next[start:n, start:n] as MPFR, then QR.
        let k = n - start;
        let spread = self.base.profile.get_spread();
        let precision = self.base.get_precision_from_spread(spread);
        let mut r = MatMpfr::zeros(k, k, precision);
        for i in 0..k {
            for j in 0..k {
                r.get_mut(i, j).assign(self.base.b_next.get(start + i, start + j));
            }
        }
        let mut tau = Vec::new();
        qr::householder_qr(&mut r, &mut tau);
        qr::clear_subdiagonal(&mut r);

        // profile[start..n] := log₂|R[i-start, i-start]|.
        for i in start..n {
            let r_ii = r.get(i - start, i - start);
            self.base.profile[i] = if r_ii.is_zero() {
                f64::NEG_INFINITY
            } else {
                let (d, e) = r_ii.to_f64_exp();
                d.abs().log2() + e as f64
            };
        }

        // Single uniform total_shift, apply across the board.
        let mut shifts = vec![0i32; n];
        self.base.get_shifts_for_compression(&mut shifts);
        let total_shift = shifts[0];
        for s in shifts.iter_mut() {
            *s = total_shift;
        }
        *self.base.compression_iters.last_mut().unwrap() = shifts;
        for i in 0..n {
            self.base.local_profile_offsets[i] += total_shift as f64;
            self.base.global_profile_offsets[i] += total_shift as f64;
            self.base.profile[i] -= total_shift as f64;
        }

        // Scale the bottom-right MPFR R by 2^{-total_shift}; zero lower.
        for i in 0..k {
            for j in 0..k {
                if i > j {
                    r.get_mut(i, j).assign(0);
                } else {
                    let cell = r.get_mut(i, j);
                    if total_shift < 0 {
                        *cell <<= (-total_shift) as u32;
                    } else {
                        *cell >>= total_shift as u32;
                    }
                }
            }
        }
        // Scale the integer top-left untouched tile B_next[0:start, :] by same shift.
        for i in 0..start {
            for j in 0..n {
                if i > j {
                    self.base.b_next.set(i, j, rug::Integer::new());
                } else {
                    let cell = self.base.b_next.get_mut(i, j);
                    if total_shift < 0 {
                        *cell <<= (-total_shift) as u32;
                    } else {
                        *cell >>= total_shift as u32;
                    }
                }
            }
        }

        // B := assemble from B_next (top-left) and R (bottom-right).
        self.base.b = IntMatrix::zeros(n, n);
        let top = self.base.b_next.submatrix(0, start, 0, n);
        self.base.b.copy_submatrix_from(0, 0, &top);
        let r_int = mpfr_mat_to_int_rows(&r, k, k);
        self.base.b.copy_submatrix_from(start, start, &r_int);

        // Stash R into state (upper-right of the n×n R matrix, zero-padded).
        let mut full_r = MatMpfr::zeros(n, n, precision);
        for i in 0..k {
            for j in 0..k {
                full_r.get_mut(start + i, start + j).assign(r.get(i, j));
            }
        }
        self.base.r = full_r;
        self.base.precision = precision;
    }

    /// `Heuristic2::update_all_representation` (heuristic_2.cpp:560).
    /// Sub-window is the whole basis `[0, n)`. Just records U_i and
    /// the sub-profile — no re-QR, no compression.
    fn update_all_representation(
        &mut self,
        window: Sublattice,
        u_sub: IntMatrix,
        sub_profile: Profile,
    ) {
        let (start, end) = window;
        assert_eq!(start, 0);
        assert_eq!(end, self.base.n);

        {
            let u_i = self.base.u_iters.last_mut().expect("U_i");
            u_i.set_identity();
            u_i.copy_submatrix_from(start, start, &u_sub);
        }
        for i in 0..self.base.n {
            self.base.profile[i] = sub_profile[i];
        }
    }
}

fn rounds_so_far(rg: &RecursiveGeneric) -> usize {
    rg.u_iters.len()
}

#[allow(dead_code)]
fn _mpfr_mat_to_int_full(r: &MatMpfr, n: usize) -> IntMatrix {
    mpfr_mat_to_int_rows(r, r.nrows, n)
}

/// Round the top `rows` of an MPFR matrix `r` into a `rows × cols`
/// integer matrix. Used by update_L / update_R to write R back into B.
fn mpfr_mat_to_int_rows(r: &MatMpfr, rows: usize, cols: usize) -> IntMatrix {
    let mut out = IntMatrix::zeros(rows, cols);
    let mut tmp = rug::Float::new(r.prec);
    for i in 0..rows {
        for j in 0..cols {
            rug::Assign::assign(&mut tmp, r.get(i, j));
            tmp.round_mut();
            if let Some(v) = tmp.clone().to_integer() {
                out.set(i, j, v);
            }
        }
    }
    out
}
