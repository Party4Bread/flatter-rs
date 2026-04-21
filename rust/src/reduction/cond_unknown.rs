//! `CondUnknown` — phase-1 preprocessor when `log_cond == 0` (unknown
//! condition number). Port of
//! `src/problems/lattice_reduction/cond_unknown.cpp`.
//!
//! Outer loop (`solve`, cond_unknown.cpp:342):
//!   working_prec = 53
//!   B := M
//!   U := I
//!   repeat up to 10000 times:
//!       if refine_basis() == true: done
//!       sort_by_size(B)
//!   M := B
//!   update_rank
//!
//! `refine_basis` (cond_unknown.cpp:243) does:
//!   1. extract_similar: QR-ish extraction producing a prec-similar
//!      basis B_sim with `num_valid` independent vectors front-loaded,
//!      plus a permutation U_1 and shift_amount / spread.
//!   2. SizeReduction on B_sim[:, 0..num_valid] → U_2i.
//!   3. If num_valid < cols: RelativeSizeReduction on the dependent
//!      sub-basis → U_2d (composed with U_2i into U_2d).
//!   4. Recursive LatticeReduction on B_sim_indep with phase=2,
//!      proved flag propagated, B2 = B_sim_dep, profile_offset =
//!      [-shift_amount; num_valid] → fills U_3i, U_3d.
//!   5. apply_perm(U_1); apply_U(U_2i, U_2d); apply_U(U_3i, U_3d).
//!   6. Convergence check: if num_valid + zero_vecs == cols, done.
//!      Otherwise bump working_prec (either from profile spread if
//!      we've reached max_rank, or doubled).

use rug::{Assign, Integer};

use crate::lattice::{IntMatrix, Lattice};
use crate::math::mat_mpfr::MatMpfr;
use crate::math::{mat_mul, qr, rsr, size_reduction};
use crate::profile::Profile;
use crate::reduction::params::LatticeReductionParams;
use crate::reduction::sublattice_split::Split;

pub struct CondUnknown {
    params: LatticeReductionParams,
    b: IntMatrix,
    u: IntMatrix,
    rhf: f64,
    working_prec: u32,
    max_rank: usize,
    m: usize,
    n: usize,
}

impl CondUnknown {
    pub fn new(outer_m: IntMatrix, params: LatticeReductionParams) -> Self {
        let m = outer_m.nrows;
        let n = outer_m.ncols;
        let max_rank = m.min(n);
        let rhf = params.rhf;
        let mut u = IntMatrix::zeros(n, n);
        u.set_identity();
        Self {
            params,
            b: outer_m,
            u,
            rhf,
            working_prec: 53,
            max_rank,
            m,
            n,
        }
    }

    /// `CondUnknown::solve` (cond_unknown.cpp:342).
    pub fn solve(mut self) -> (IntMatrix, IntMatrix, Profile) {
        // Cap iterations at 10000 like C++ (cond_unknown.cpp:355).
        // In practice the loop converges in a few iterations for
        // well-conditioned qary inputs.
        for _ in 0..10000 {
            if self.refine_basis() {
                break;
            }
            self.sort_by_size();
        }
        // Build final profile from diagonal of B (upper-triangular after
        // the refinement loop — or trivially zero where columns are dependent).
        let mut prof = Profile::new(self.n);
        for i in 0..self.n {
            let b_ii = self.b.get(i, i);
            prof[i] = if b_ii.is_zero() {
                f64::NEG_INFINITY
            } else {
                let (d, e) = b_ii.to_f64_exp();
                d.abs().log2() + e as f64
            };
        }
        (self.b, self.u, prof)
    }

    /// `CondUnknown::refine_basis` (cond_unknown.cpp:243).
    /// Returns true iff convergence was reached.
    fn refine_basis(&mut self) -> bool {
        let n = self.n;

        // --- 1. extract_similar ---
        let mut b_sim = IntMatrix::zeros(self.m, n);
        let mut u_1 = IntMatrix::zeros(n, n);
        let (num_valid, shift_amount, spread) =
            self.extract_similar(self.working_prec, &mut b_sim, &mut u_1);

        // --- 2. SizeReduction on the independent sub-basis ---
        // B_sim_indep = B_sim[0:num_valid, 0:num_valid]  (upper-triangular)
        let mut b_sim_indep = b_sim.submatrix(0, num_valid, 0, num_valid);
        let mut u_2 = IntMatrix::zeros(n, n);
        u_2.set_identity();
        let mut u_2i = IntMatrix::zeros(num_valid, num_valid);
        if num_valid > 0 {
            size_reduction::size_reduce(&mut b_sim_indep, &mut u_2i);
            // write back
            for i in 0..num_valid {
                for j in 0..num_valid {
                    b_sim.set(i, j, b_sim_indep.get(i, j).clone());
                    u_2.set(i, j, u_2i.get(i, j).clone());
                }
            }
        }

        // --- 3. RSR on the dependent part ---
        let b2_cols = n - num_valid;
        let mut u_2d = IntMatrix::zeros(num_valid, b2_cols);
        if num_valid > 0 && num_valid < n {
            let b1 = b_sim.submatrix(0, num_valid, 0, num_valid);
            let mut b2 = b_sim.submatrix(0, num_valid, num_valid, n);
            {
                let mut rsr = rsr::RsrTriangular::new(&b1, &mut b2, &mut u_2d);
                rsr.solve();
            }
            // Write b2 back into b_sim.
            for i in 0..num_valid {
                for j in 0..b2_cols {
                    b_sim.set(i, num_valid + j, b2.get(i, j).clone());
                }
            }
            // Compose: u_2d := u_2i · u_2d  (C++ cond_unknown.cpp:275).
            let composed = mat_mul::mat_mul(&u_2i, &u_2d);
            u_2d = composed;
            // Also update U_2 top-right block to composed.
            for i in 0..num_valid {
                for j in 0..b2_cols {
                    u_2.set(i, num_valid + j, u_2d.get(i, j).clone());
                }
            }
        }

        // --- 4. Recursive LatticeReduction call ---
        // Sub-problem: B_sim_indep (post-SR), U_3i, with B2 = B_sim_dep,
        // phase=2, split=Phase2(num_valid), profile_offset=[-shift_amount].
        let mut u_3 = IntMatrix::zeros(n, n);
        u_3.set_identity();
        let mut u_3i = IntMatrix::zeros(num_valid, num_valid);
        let mut u_3d = IntMatrix::zeros(num_valid, b2_cols);

        let mut sub_profile = Profile::new(num_valid);
        if num_valid > 0 {
            let mut sub_lat = Lattice::new(num_valid, num_valid);
            sub_lat.basis = b_sim.submatrix(0, num_valid, 0, num_valid);
            sub_lat.rank = num_valid;
            sub_lat.profile = Profile::new(num_valid);

            // Build a FRESH goal sized for the sub-problem (n = num_valid)
            // so goal.check in the recursive call uses the right n.
            let sub_goal =
                crate::reduction::goal::LatticeReductionGoal::from_rhf(
                    num_valid,
                    self.rhf,
                    self.params.proved,
                );
            let mut sub_params = self.params.clone();
            sub_params.goal = sub_goal;
            sub_params.rhf = self.rhf;
            sub_params.is_upper_triangular = true;
            sub_params.proved = self.params.proved;
            sub_params.phase = 2;
            sub_params.split = Some(Split::new_phase2(num_valid));
            sub_params.aggressive_precision = self.params.aggressive_precision;
            sub_params.log_cond = spread;
            sub_params.profile_offset = vec![-(shift_amount as f64); num_valid];
            if b2_cols > 0 {
                sub_params.b2 = Some(b_sim.submatrix(0, num_valid, num_valid, n));
                sub_params.u2 = Some(u_3d.clone());
            } else {
                sub_params.b2 = None;
                sub_params.u2 = None;
            }

            // Call reduce_with_u to track the unimodular U_3i.
            crate::reduction::lll::reduce_with_u(&mut sub_lat, &sub_params, &mut u_3i);
            sub_profile = sub_lat.profile;

            // Retrieve the U2 that LatticeReduction filled in (if B2 was set).
            if let Some(u2) = sub_params.u2 {
                u_3d = u2;
            }

            // Place U_3i, U_3d into U_3.
            for i in 0..num_valid {
                for j in 0..num_valid {
                    u_3.set(i, j, u_3i.get(i, j).clone());
                }
                for j in 0..b2_cols {
                    u_3.set(i, num_valid + j, u_3d.get(i, j).clone());
                }
            }
        }

        // Update profile entries for the reduced vectors
        // (cond_unknown.cpp:304-306).
        let mut prof = self.params.goal.n.max(n);
        let _ = prof;
        let mut l_profile = Profile::new(n);
        for i in 0..num_valid {
            l_profile[i] = sub_profile[i] - shift_amount as f64;
        }
        // Track in our params.L-equivalent profile (we carry it into outputs).

        // --- 5. apply U_1, U_2, U_3 to outer B ---
        self.apply_perm(&u_1);
        if num_valid > 0 {
            self.apply_u(&u_2, num_valid);
            self.apply_u(&u_3, num_valid);
        }

        // Compose U: self.u := self.u · (U_1 · U_2 · U_3).
        let u_12 = mat_mul::mat_mul(&u_1, &u_2);
        let u_123 = mat_mul::mat_mul(&u_12, &u_3);
        let new_u = mat_mul::mat_mul(&self.u, &u_123);
        self.u = new_u;

        // --- 6. Convergence check (cond_unknown.cpp:315-339) ---
        let mut zero_vecs = 0usize;
        for j in num_valid..n {
            let mut vec_is_zero = true;
            for i in 0..self.m {
                if !self.b.get(i, j).is_zero() {
                    vec_is_zero = false;
                    break;
                }
            }
            if vec_is_zero {
                zero_vecs += 1;
            }
        }

        if num_valid + zero_vecs == n {
            return true;
        }
        if num_valid == self.max_rank {
            // working_prec = 2 * profile.spread + 40
            let sp = l_profile.get_spread();
            self.working_prec =
                ((2.0 * sp).ceil() as u32).saturating_add(40).max(53);
            return false;
        }
        self.max_rank = self.m.min(n - zero_vecs);
        self.working_prec = self.working_prec.saturating_mul(2).max(53);
        false
    }

    /// `CondUnknown::extract_similar` (cond_unknown.cpp:92).
    /// Pure-scratch MPFR computation: run a modified Householder QR
    /// that greedily picks columns with "enough" orthogonal component
    /// and pushes the rest aside as dependent. Produces a
    /// prec-similar basis + permutation U.
    fn extract_similar(
        &self,
        prec: u32,
        b_sim_out: &mut IntMatrix,
        u_out: &mut IntMatrix,
    ) -> (usize, i32, f64) {
        let m = self.m;
        let n = self.n;

        // R (sorted) and R_unsorted, both MPFR.
        let mut r = MatMpfr::zeros(m, n, prec);
        let mut r_unsorted = MatMpfr::zeros(m, n, prec);
        u_out.set_identity();
        // Zero u_out first (since set_identity just puts 1s on diag).
        // For the column-selection logic below we need to set
        // specific entries non-trivially; start from zero.
        for i in 0..n {
            for j in 0..n {
                u_out.set(i, j, Integer::new());
            }
        }

        for i in 0..m {
            for j in 0..n {
                r_unsorted.get_mut(i, j).assign(self.b.get(i, j));
            }
        }

        let max_rank = m.min(n);
        let mut tau: Vec<rug::Float> = (0..max_rank).map(|_| rug::Float::new(prec)).collect();

        let mut tmp = rug::Float::new(prec);
        let mut vec_len = rug::Float::new(prec);
        let mut orth_len = rug::Float::new(prec);

        let mut curmax = f64::NEG_INFINITY;
        let mut curmin = f64::INFINITY;
        let mut num_valid = 0usize;
        let mut num_dependent = 0usize;

        for j in 0..n {
            vec_len.assign(0);
            orth_len.assign(0);

            for i in 0..m {
                tmp.assign(r_unsorted.get(i, j));
                tmp.square_mut();
                vec_len += &tmp;
                if i >= num_valid {
                    orth_len += &tmp;
                }
            }

            let log_vec_len = if vec_len.is_zero() {
                f64::NEG_INFINITY
            } else {
                let (d, e) = vec_len.to_f64_exp();
                (d.abs().log2() + e as f64) / 2.0
            };
            let log_orth_len = if orth_len.is_zero() {
                f64::NEG_INFINITY
            } else {
                let (d, e) = orth_len.to_f64_exp();
                (d.abs().log2() + e as f64) / 2.0
            };

            let new_spread = curmax.max(log_vec_len) - curmin.min(log_orth_len);
            let required_prec = (2.0 * new_spread + 40.0).ceil() as i64;

            if new_spread.is_finite() && required_prec <= prec as i64 {
                // Copy this column from R_unsorted[:, j] to R[:, num_valid].
                for i in 0..m {
                    r.get_mut(i, num_valid).assign(r_unsorted.get(i, j));
                }
                // Mark u_out[j, num_valid] = 1; diagonal U[j, j] zero-out.
                u_out.set(j, num_valid, Integer::from(1));

                // Householder on R[:, num_valid] starting from row num_valid.
                if num_valid + 1 < m {
                    let t = qr::larfg(&mut r, num_valid, m, prec);
                    tau[num_valid].assign(&t);
                    // Apply reflector H = I - τ·v·vᵀ to R_unsorted[:, j+1..n],
                    // where v is stored in R[num_valid..m, num_valid] with v[0]=1.
                    // Matches cond_unknown.cpp:157–162 (cross-matrix larf).
                    if j + 1 < n {
                        let tau_val = rug::Float::with_val(prec, &tau[num_valid]);
                        if !tau_val.is_zero() {
                            let mut dot = rug::Float::new(prec);
                            let mut scalar = rug::Float::new(prec);
                            let mut tmp = rug::Float::new(prec);
                            for col in (j + 1)..n {
                                // dot = 1·R_unsorted[num_valid,col] + Σ v[i]·R_unsorted[i,col]
                                dot.assign(r_unsorted.get(num_valid, col));
                                for i in (num_valid + 1)..m {
                                    tmp.assign(r.get(i, num_valid));
                                    tmp *= r_unsorted.get(i, col);
                                    dot += &tmp;
                                }
                                scalar.assign(&tau_val);
                                scalar *= &dot;
                                // R_unsorted[num_valid,col] -= scalar·1
                                {
                                    let c = r_unsorted.get_mut(num_valid, col);
                                    *c -= &scalar;
                                }
                                // R_unsorted[i,col] -= scalar·v[i] for i > num_valid
                                for i in (num_valid + 1)..m {
                                    let v =
                                        rug::Float::with_val(prec, &scalar * r.get(i, num_valid));
                                    let c = r_unsorted.get_mut(i, col);
                                    *c -= &v;
                                }
                            }
                        }
                    }
                } else if num_valid + 1 == m {
                    tau[num_valid].assign(0);
                }
                num_valid += 1;
                curmax = curmax.max(log_vec_len);
                curmin = curmin.min(log_orth_len);
            } else {
                // Dependent vector. Store at the end of R.
                let dest = n - num_dependent - 1;
                for i in 0..m {
                    r.get_mut(i, dest).assign(r_unsorted.get(i, j));
                }
                u_out.set(j, dest, Integer::from(1));
                num_dependent += 1;
            }
        }

        let shift_amount = prec as i32 - curmax.ceil() as i32;
        let spread = if curmax.is_finite() && curmin.is_finite() {
            curmax - curmin
        } else {
            0.0
        };

        // Scale R's upper triangle by 2^shift_amount, zero below-diag.
        for i in 0..m {
            for j in 0..n {
                if i <= j {
                    let cell = r.get_mut(i, j);
                    if shift_amount >= 0 {
                        *cell <<= shift_amount as u32;
                    } else {
                        *cell >>= (-shift_amount) as u32;
                    }
                } else {
                    r.get_mut(i, j).assign(0);
                }
            }
        }

        // Round R into b_sim_out.
        let mut rtmp = rug::Float::new(prec);
        for i in 0..m {
            for j in 0..n {
                rtmp.assign(r.get(i, j));
                rtmp.round_mut();
                if let Some(v) = rtmp.clone().to_integer() {
                    b_sim_out.set(i, j, v);
                } else {
                    b_sim_out.set(i, j, Integer::new());
                }
            }
        }

        (num_valid, shift_amount, spread)
    }

    /// `CondUnknown::apply_perm` (cond_unknown.cpp:70).
    fn apply_perm(&mut self, u: &IntMatrix) {
        let mut b2 = IntMatrix::zeros(self.m, self.n);
        for j in 0..u.ncols {
            let mut src = 0usize;
            let mut found = false;
            for s in 0..self.n {
                if u.get(s, j).to_i64() == Some(1) {
                    src = s;
                    found = true;
                    break;
                }
            }
            if !found {
                continue;
            }
            for i in 0..self.m {
                b2.set(i, j, self.b.get(i, src).clone());
            }
        }
        self.b = b2;
    }

    /// `CondUnknown::apply_U` (cond_unknown.cpp:55).
    /// Conceptually (C++): B_right += B_left · U2;  B_left = B_left · U1.
    /// Here `u` is `n × n` with `U_1 = u[0..k, 0..k]`, `U_2 = u[0..k, k..n]`.
    fn apply_u(&mut self, u: &IntMatrix, k: usize) {
        let m = self.m;
        let n = self.n;
        // B_left (old) snapshot — we'll need it for both updates.
        let mut b_left_old = IntMatrix::zeros(m, k);
        for i in 0..m {
            for j in 0..k {
                b_left_old.set(i, j, self.b.get(i, j).clone());
            }
        }

        // B_right += B_left · U2
        if k < n {
            let mut prod = Integer::new();
            for i in 0..m {
                for jj in k..n {
                    for l in 0..k {
                        prod.assign(b_left_old.get(i, l) * u.get(l, jj));
                        let cell = self.b.get_mut(i, jj);
                        *cell += &prod;
                    }
                }
            }
        }

        // B_left = B_left · U1
        let mut prod = Integer::new();
        for i in 0..m {
            for j in 0..k {
                let mut sum = Integer::new();
                for l in 0..k {
                    prod.assign(b_left_old.get(i, l) * u.get(l, j));
                    sum += &prod;
                }
                self.b.set(i, j, sum);
            }
        }
    }

    /// `CondUnknown::sort_by_size` (cond_unknown.cpp:197).
    fn sort_by_size(&mut self) {
        let n = self.n;
        let mut log_vec_lens = vec![f64::INFINITY; n];
        let mut s = Integer::new();
        let mut tmp = Integer::new();
        for j in 0..n {
            s.assign(0);
            for i in 0..self.m {
                tmp.assign(self.b.get(i, j) * self.b.get(i, j));
                s += &tmp;
            }
            if s.is_zero() {
                log_vec_lens[j] = f64::NEG_INFINITY;
            } else {
                let (d, e) = s.to_f64_exp();
                log_vec_lens[j] = (d.abs().log2() + e as f64) / 2.0;
            }
        }

        // Build permutation matrix u_sort.
        let mut u_sort = IntMatrix::zeros(n, n);
        let mut end_idx = n as isize - 1;
        let mut start_idx = 0usize;
        let mut remaining: Vec<usize> = (0..n).collect();
        let mut taken = vec![false; n];
        for _step in 0..n {
            // min element of log_vec_lens over remaining.
            let mut min_j = None;
            let mut min_v = f64::INFINITY;
            for &j in &remaining {
                if !taken[j] && log_vec_lens[j] < min_v {
                    min_v = log_vec_lens[j];
                    min_j = Some(j);
                }
            }
            let j = match min_j {
                Some(v) => v,
                None => break,
            };
            taken[j] = true;
            if log_vec_lens[j] == f64::NEG_INFINITY {
                if end_idx >= 0 {
                    u_sort.set(j, end_idx as usize, Integer::from(1));
                    end_idx -= 1;
                }
            } else {
                u_sort.set(j, start_idx, Integer::from(1));
                start_idx += 1;
            }
            log_vec_lens[j] = f64::INFINITY;
        }

        self.apply_perm(&u_sort);
    }
}
