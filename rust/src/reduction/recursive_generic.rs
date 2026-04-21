//! `RecursiveGeneric`: the shared plumbing under flatter's recursive
//! heuristics. Port of
//! `src/problems/lattice_reduction/recursive_generic.cpp`. Concrete
//! heuristics (Heuristic2, Heuristic3, Proved2, Proved3) compose this
//! rather than inheriting it, since Rust prefers composition over the
//! virtual-method dance the C++ uses.
//!
//! What lives here:
//!   * the shared state (M, R, B, B_next, U, profile, offsets,
//!     per-iteration U_iters and compression_iters)
//!   * `init_solver` / `init_compressed_B`
//!   * `compress_R` + `get_shifts_for_compression` (the staircase
//!     compression formula that keeps per-column shifts safe for
//!     large-spread profiles)
//!   * `set_profile` (reads R's diagonal into the profile)
//!   * `collect_U` (the Saruchi D⁻¹ U D unrolling that prevents entry
//!     blowup when folding all the per-iteration U_iters back into a
//!     single final U)
//!   * `final_sr` and `fini_solver`
//!
//! What lives on the concrete heuristic struct (Phase D):
//!   * `solve()` driver loop
//!   * `is_reduced`
//!   * `setup_sublattice_reductions`
//!   * `update_representation`

use rug::{Assign, Float};

use crate::lattice::IntMatrix;
use crate::math::{fused_qr_sr, mat_mpfr::MatMpfr, mat_mul};
use crate::profile::Profile;
use crate::reduction::params::LatticeReductionParams;

/// Shared state mirroring `recursive_generic.cpp` fields.
pub struct RecursiveGeneric {
    /// Dimension (rows) and rank (columns) of the outer basis.
    pub m: usize,
    pub n: usize,

    /// Outer integer basis. `M` is untouched during the loop; the
    /// accumulated `U` is applied to it only in `fini_solver`.
    pub outer_m: IntMatrix,

    /// MPFR QR factor at `precision` bits. Recomputed on each
    /// `update_representation`.
    pub r: MatMpfr,

    /// Integer "compressed shadow" basis = round(R · 2^{-shifts[j]}).
    /// Fed into the recursive `LatticeReduction::solve` inside each
    /// sub-problem.
    pub b: IntMatrix,

    /// Workspace used by `update_representation` to apply each
    /// sub-lattice's U to the global basis.
    pub b_next: IntMatrix,

    /// Accumulated unimodular (top-level). Populated by `collect_U`.
    pub u: IntMatrix,

    /// Per-iteration unimodular transforms, pushed by
    /// `init_iter` and consumed by `collect_U` in reverse.
    pub u_iters: Vec<IntMatrix>,

    /// Per-iteration compression shift arrays. Length is
    /// `u_iters.len() + 1` because the initial compression happens
    /// before any U_iter.
    pub compression_iters: Vec<Vec<i32>>,

    pub profile: Profile,

    /// Per-column profile offset relative to the sub-problem's own
    /// frame (accumulated compressions applied locally).
    pub local_profile_offsets: Vec<f64>,
    /// Per-column profile offset relative to the global outer
    /// problem (passed in via params).
    pub global_profile_offsets: Vec<f64>,

    pub precision: u32,
    pub original_precision: u32,
    pub lattice_changed: bool,

    pub params: LatticeReductionParams,
}

impl RecursiveGeneric {
    /// Construct but don't initialise yet. Call `init_solver` before
    /// driving the loop.
    pub fn new(outer_m: IntMatrix, params: LatticeReductionParams) -> Self {
        let m = outer_m.nrows;
        let n = outer_m.ncols;
        Self {
            m,
            n,
            outer_m,
            r: MatMpfr::zeros(m, n, 53),
            b: IntMatrix::zeros(m, n),
            b_next: IntMatrix::zeros(m, n),
            u: IntMatrix::zeros(n, n),
            u_iters: Vec::new(),
            compression_iters: Vec::new(),
            profile: Profile::new(n),
            local_profile_offsets: vec![0.0; n],
            global_profile_offsets: vec![0.0; n],
            precision: 53,
            original_precision: 53,
            lattice_changed: true,
            params,
        }
    }

    /// Precision to carry MPFR ops at, as a function of log₂ spread of
    /// the profile. Matches `recursive_generic.cpp:77`:
    ///   `precision = 2·spread + 10 + 30`
    pub fn get_precision_from_spread(&self, spread: f64) -> u32 {
        (2.0 * spread + 40.0).max(53.0).ceil() as u32
    }

    /// Initial precision from the outer basis's maximum entry bit-length
    /// (`recursive_generic.cpp:50`).
    pub fn get_initial_precision(&self) -> u32 {
        let mut max_bits: u64 = 0;
        for e in &self.outer_m.data {
            let b = e.significant_bits() as u64;
            if b > max_bits {
                max_bits = b;
            }
        }
        self.get_precision_from_spread(max_bits as f64)
    }

    /// Change the MPFR precision of `self.r` in-place. Existing entries
    /// are re-quantised to the new precision (rug does this on assign
    /// when target and source precisions differ).
    pub fn set_precision(&mut self, prec: u32) {
        if prec == self.precision {
            return;
        }
        let mut new_r = MatMpfr::zeros(self.m, self.n, prec);
        for i in 0..self.m {
            for j in 0..self.n {
                new_r.get_mut(i, j).assign(self.r.get(i, j));
            }
        }
        self.r = new_r;
        self.precision = prec;
    }

    /// `recursive_generic.cpp:318`: read profile[i] = log₂|R[i, i]|.
    pub fn set_profile(&mut self) {
        for i in 0..self.n {
            let r_ii = self.r.get(i, i);
            let new_val = if r_ii.is_zero() {
                f64::NEG_INFINITY
            } else {
                let (d, exp) = r_ii.to_f64_exp();
                d.abs().log2() + exp as f64
            };
            if self.profile[i] != new_val {
                self.lattice_changed = true;
            }
            self.profile[i] = new_val;
        }
    }

    /// `recursive_generic.cpp:62`: populate R with M, compute the
    /// initial profile, then compress R into B.
    pub fn init_compressed_B(&mut self) {
        let precision = self.get_initial_precision();
        self.original_precision = precision;
        self.set_precision(precision);

        // R := outer_m (integer → MPFR copy).
        for i in 0..self.m {
            for j in 0..self.n {
                self.r.get_mut(i, j).assign(self.outer_m.get(i, j));
            }
        }

        self.set_profile();

        // Initial compression_iter (size n, all zeros). compress_R
        // overwrites it.
        self.compression_iters.push(vec![0i32; self.n]);
        self.compress_R();

        // B := R (rounded).
        self.b = mpfr_mat_to_int(&self.r, self.n);
    }

    /// `recursive_generic.cpp:82`: allocate working matrices, copy the
    /// global profile_offset, populate B via QR + compression.
    pub fn init_solver(&mut self) {
        self.b = IntMatrix::zeros(self.m, self.n);
        self.r = MatMpfr::zeros(self.m, self.n, 53);
        self.b_next = IntMatrix::zeros(self.m, self.n);
        self.profile = Profile::new(self.n);
        self.local_profile_offsets = vec![0.0; self.n];
        self.global_profile_offsets = self.params.profile_offset.clone();
        if self.global_profile_offsets.len() < self.n {
            self.global_profile_offsets.resize(self.n, 0.0);
        }
        self.u_iters.clear();
        self.compression_iters.clear();

        self.init_compressed_B();

        self.lattice_changed = true;
    }

    /// `recursive_generic.cpp:333`: staircase-aware per-column shifts.
    /// Writes `shifts` (length n) and returns the new MPFR precision.
    pub fn get_shifts_for_compression(&self, shifts: &mut [i32]) -> u32 {
        let n = self.n;
        let mut max_from_left = vec![0.0f64; n];
        let mut min_from_right = vec![0.0f64; n];
        max_from_left[0] = self.profile[0];
        min_from_right[n - 1] = self.profile[n - 1];
        for i in 0..n - 1 {
            max_from_left[i + 1] = self.profile[i + 1].max(max_from_left[i]);
            min_from_right[n - i - 2] =
                self.profile[n - i - 2].min(min_from_right[n - i - 1]);
        }

        shifts[0] = 0;
        for i in 1..n {
            shifts[i] = shifts[i - 1];
            let compress = min_from_right[i] - max_from_left[i - 1];
            if compress <= 1.0 {
                continue;
            }
            shifts[i] += (compress - 1.0).floor() as i32;
        }

        let spread =
            max_from_left[n - 1] - shifts[n - 1] as f64 - min_from_right[0];
        let precision = self.get_precision_from_spread(spread);
        let new_shift =
            max_from_left[n - 1].ceil() as i32 - shifts[n - 1] - precision as i32;

        for s in shifts.iter_mut() {
            *s += new_shift;
        }

        precision
    }

    /// `recursive_generic.cpp:378`: apply the per-column shifts to R,
    /// update local/global offsets, and update the profile.
    pub fn compress_R(&mut self) {
        // Compute shifts into the most-recent compression_iter slot.
        let n = self.n;
        let mut shifts = vec![0i32; n];
        let precision = self.get_shifts_for_compression(&mut shifts);

        // Store the shifts as THIS iteration's compression entry.
        {
            let slot = self.compression_iters.last_mut().expect("compression_iters push first");
            slot.clear();
            slot.extend_from_slice(&shifts);
        }

        for j in 0..n {
            self.local_profile_offsets[j] += shifts[j] as f64;
            self.global_profile_offsets[j] += shifts[j] as f64;
            self.profile[j] -= shifts[j] as f64;

            for i in 0..self.m {
                if i > j {
                    self.r.get_mut(i, j).assign(0);
                } else {
                    // R[i, j] *= 2^{-shifts[j]}
                    let shift = -shifts[j];
                    let cell = self.r.get_mut(i, j);
                    if shift >= 0 {
                        *cell <<= shift as u32;
                    } else {
                        *cell >>= (-shift) as u32;
                    }
                }
            }
        }

        self.set_precision(precision);
    }

    /// `recursive_generic.cpp:273`: compose all `U_iter`s in reverse,
    /// conjugating each by the `D⁻¹ · U_iter · D` rescaling where
    /// `D[k] = 2^{compression_iters[t][k]}`.
    pub fn collect_U(&mut self) {
        assert_eq!(
            self.u_iters.len() + 1,
            self.compression_iters.len(),
            "expected one more compression_iter than U_iter"
        );

        self.u = IntMatrix::zeros(self.n, self.n);
        self.u.set_identity();

        // Drop the final compression_iter — it has no matching U_iter
        // (no reduction happened after the last compression).
        self.compression_iters.pop();

        while let Some(mut u_iter) = self.u_iters.pop() {
            let d = self
                .compression_iters
                .last()
                .cloned()
                .expect("compression_iters underflow");

            // D⁻¹ · U_iter · D: each entry (i, j) is scaled by
            // `2^{d[j] - d[i]}`. For `d[i] > d[j]` the paper mandates
            // U_iter[i,j] = 0; debug-assert.
            for i in 0..self.n {
                for j in 0..self.n {
                    let di = d[i];
                    let dj = d[j];
                    if di == dj {
                        // no-op
                    } else if di > dj {
                        debug_assert!(
                            u_iter.get(i, j).is_zero(),
                            "collect_U invariant: U[{},{}] must be 0 (D[i]={}, D[j]={})",
                            i,
                            j,
                            di,
                            dj
                        );
                    } else {
                        let shift = (dj - di) as u32;
                        *u_iter.get_mut(i, j) <<= shift;
                    }
                }
            }

            // U := U_iter · U
            let new_u = mat_mul::mat_mul(&u_iter, &self.u);
            self.u = new_u;

            self.compression_iters.pop();
        }
        assert!(self.compression_iters.is_empty());
    }

    /// `recursive_generic.cpp:106`: one final fused QR + size reduction
    /// on the outer basis, at a tighter precision. Composes the returned
    /// U into the global `self.u`.
    pub fn final_sr(&mut self) {
        let spread = self.profile.get_spread();
        let new_precision = self.get_precision_from_spread(spread);
        self.set_precision(new_precision);

        let mut r_final = MatMpfr::zeros(self.m, self.n, new_precision);
        let mut u_tmp = IntMatrix::zeros(self.n, self.n);
        fused_qr_sr::fused_qr_sr(&mut self.outer_m, &mut r_final, &mut u_tmp);
        self.r = r_final;

        // U := U · U_tmp
        let new_u = mat_mul::mat_mul(&self.u, &u_tmp);
        self.u = new_u;
    }

    /// `recursive_generic.cpp:147`: apply the accumulated U to the
    /// outer basis, update the profile, run a final size reduction.
    pub fn fini_solver(&mut self) {
        self.collect_U();

        // M := M · U
        let new_m = mat_mul::mat_mul(&self.outer_m, &self.u);
        self.outer_m = new_m;

        for i in 0..self.n {
            self.profile[i] += self.local_profile_offsets[i];
            self.global_profile_offsets[i] -= self.local_profile_offsets[i];
            self.local_profile_offsets[i] = 0.0;
        }

        self.final_sr();
    }

    /// `recursive_generic.cpp:170` — per-iteration setup. The concrete
    /// heuristic calls this and then `setup_sublattice_reductions`.
    pub fn init_iter(&mut self) {
        self.u_iters.push({
            let mut m = IntMatrix::zeros(self.n, self.n);
            m.set_identity();
            m
        });
        self.compression_iters.push(vec![0i32; self.n]);
    }
}

/// Round an upper-triangular MPFR matrix to an integer matrix. Rounds
/// to nearest; below-diagonal entries are assumed zero and copied as 0.
fn mpfr_mat_to_int(r: &MatMpfr, n: usize) -> IntMatrix {
    let m = r.nrows;
    let mut out = IntMatrix::zeros(m, n);
    let mut tmp = Float::new(r.prec);
    for i in 0..m {
        for j in 0..n {
            tmp.assign(r.get(i, j));
            tmp.round_mut();
            if let Some(v) = tmp.clone().to_integer() {
                out.set(i, j, v);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reduction::goal::LatticeReductionGoal;
    use crate::reduction::sublattice_split::Split;
    use rug::Integer;

    fn build_rg(n: usize, entries: &[i64]) -> RecursiveGeneric {
        let mut b = IntMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                b.set(i, j, Integer::from(entries[i * n + j]));
            }
        }
        let goal = LatticeReductionGoal::from_rhf(n, 1.0219, false);
        let mut params = LatticeReductionParams::from_goal(goal);
        params.split = Some(Split::new_phase2(n));
        RecursiveGeneric::new(b, params)
    }

    #[test]
    fn init_solver_populates_B_and_profile() {
        let mut rg = build_rg(3, &[10, 3, 7, 0, 8, 2, 0, 0, 5]);
        rg.init_solver();
        assert_eq!(rg.b.nrows, 3);
        assert_eq!(rg.b.ncols, 3);
        // Profile has finite entries for non-zero diagonal.
        for i in 0..3 {
            assert!(
                rg.profile[i].is_finite(),
                "profile[{}] = {}",
                i,
                rg.profile[i]
            );
        }
        // B should be upper-triangular after compress_R.
        for i in 0..3 {
            for j in 0..i {
                assert!(
                    rg.b.get(i, j).is_zero(),
                    "B[{},{}] not zero",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn collect_u_identity_no_rounds() {
        // A RecursiveGeneric that did 0 outer rounds: U_iters is empty,
        // compression_iters has one entry (the initial). collect_U should
        // give U = I.
        let mut rg = build_rg(3, &[5, 0, 0, 0, 5, 0, 0, 0, 5]);
        rg.init_solver();
        rg.collect_U();
        assert!(rg.u.is_identity(), "u should be identity");
    }

    #[test]
    fn collect_u_single_round_preserves_product() {
        // Two rounds where each U_iter is the identity: final U = I.
        let mut rg = build_rg(3, &[10, 3, 7, 0, 8, 2, 0, 0, 5]);
        rg.init_solver();
        // Fake one iteration where U_iter = I.
        rg.init_iter();
        rg.collect_U();
        assert!(rg.u.is_identity());
    }
}
