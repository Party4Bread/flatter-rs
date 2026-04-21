//! Fused QR + size reduction, MPFR path. Port of
//! `src/problems/fused_qr_sizered/columnwise.cpp`'s
//! `Columnwise::solve`.
//!
//! Given an integer basis `B` (m×n), produce:
//!   * `R` (MPFR, m×n, upper triangular) — QR factor of the
//!     **size-reduced** `B`.
//!   * `U` (integer, n×n, unimodular) — transform such that
//!     `B_new = B_old · U` and every `|R[i,j]| ≤ |R[i,i]|/2` for i<j.
//!
//! The important difference vs. our `size_reduce_r::size_reduce` is
//! that this routine *re-orthogonalizes* the column whenever size
//! reduction is applied, so R stays consistent with the updated B.
//! That's essential for the recursive heuristic: size_reduce_r is
//! one-shot (R gets stale as entries drift), while `fused_qr_sr` leaves
//! (B, R) self-consistent at the end.

use rug::{Assign, Float, Integer};

use crate::lattice::IntMatrix;
use crate::math::mat_mpfr::MatMpfr;
use crate::math::qr::{larf, larfg};

/// Running size-reduction tolerance factor. Flatter uses 0.51 to avoid
/// infinite ping-ponging when |R[i,j]| is almost exactly |R[i,i]|/2
/// but slightly above due to roundoff.
const SR_TOLERANCE: f64 = 0.51;

/// Number of mu "improvement" rounds below which we stop re-orthoing.
/// Flatter compares `round_mu_max` vs. `mu_max - prec/2`; we match that.
/// If no meaningful decrease, exit the loop.
fn should_stop(round_mu_max: f64, prev_mu_max: f64, prec: u32) -> bool {
    round_mu_max.is_finite() && round_mu_max >= prev_mu_max - (prec as f64 / 2.0)
}

/// Run fused QR + size reduction.
///
/// On entry: `b` is `m × n` (integer); `r` is `m × n` at MPFR
/// precision `prec` — contents ignored; `u` is `n × n`.
///
/// On exit: `b` has been size-reduced (entries may shrink, may swap-ish
/// via the U update), `r` holds the new upper-triangular factor, and
/// `u` contains the unimodular so that `b_in · u = b_out`.
pub fn fused_qr_sr(b: &mut IntMatrix, r: &mut MatMpfr, u: &mut IntMatrix) {
    let m = b.nrows;
    let n = b.ncols;
    assert_eq!(r.nrows, m);
    assert_eq!(r.ncols, n);
    assert_eq!(u.nrows, n);
    assert_eq!(u.ncols, n);
    let prec = r.prec;

    u.set_identity();
    copy_int_to_mpfr(b, r);

    // Householder τ[0..n].
    let mut tau: Vec<Float> = (0..n).map(|_| Float::new(prec)).collect();

    let mut c_int = Integer::new();
    let mut prod = Integer::new();
    let mut entry = Float::new(prec);
    let mut c_float = Float::new(prec);

    for i in 0..n {
        // Size-reduce column `i` against columns `0..i`, re-orthogonalizing
        // as needed (Columnwise::solve, columnwise.cpp:78).
        let mut mu_max = f64::INFINITY;
        loop {
            let mut must_repeat = false;
            let mut round_mu_max = 0.0f64;

            for j in 0..i {
                let row = i - j - 1;
                // Is |R[row, i]| "small enough" vs. |R[row, row]|?
                // Match columnwise.cpp:84–105 logic.
                entry.assign(r.get(row, i));
                let not_reduced = if entry.clone().abs()
                    >= r.get(row, row).clone().abs()
                {
                    true
                } else {
                    // Check |2·R[row,i]| vs. |R[row,row]|.
                    let twice = Float::with_val(prec, &entry * 2);
                    if twice.clone().abs() < r.get(row, row).clone().abs() {
                        false
                    } else {
                        let entry_d = r.get(row, i).to_f64();
                        let diag_d = r.get(row, row).to_f64();
                        if diag_d == 0.0 {
                            false
                        } else {
                            (entry_d / diag_d).abs() > SR_TOLERANCE
                        }
                    }
                };
                if not_reduced {
                    must_repeat = true;
                    // c = round(R[row, i] / R[row, row])
                    c_float.assign(r.get(row, i));
                    c_float /= r.get(row, row);
                    c_float.round_mut();
                    let mu_abs = c_float.to_f64().abs();
                    if mu_abs > round_mu_max {
                        round_mu_max = mu_abs;
                    }
                    if let Some(val) = c_float.clone().to_integer() {
                        c_int.assign(&val);
                    } else {
                        // Non-finite guard; skip this column.
                        continue;
                    }

                    // Subtract column `row` from column `i`:
                    //   B[:, i] -= c · B[:, row]
                    //   U[:, i] -= c · U[:, row]
                    //   R[k, i] -= c · R[k, row]   for k < row
                    for k in 0..m {
                        prod.assign(b.get(k, row) * &c_int);
                        *b.get_mut(k, i) -= &prod;
                    }
                    for k in 0..n {
                        prod.assign(u.get(k, row) * &c_int);
                        *u.get_mut(k, i) -= &prod;
                    }
                    for k in 0..row {
                        let v = Float::with_val(prec, &c_float * r.get(k, row));
                        *r.get_mut(k, i) -= &v;
                    }
                    // R[row, i] itself: -= c · R[row, row].
                    let v = Float::with_val(prec, &c_float * r.get(row, row));
                    *r.get_mut(row, i) -= &v;
                }
            }

            if should_stop(round_mu_max, mu_max, prec) {
                must_repeat = false;
            }
            mu_max = round_mu_max;

            if must_repeat {
                // Re-orthogonalize column i: copy B[:, i] fresh to R[:, i],
                // then apply the previous Householder reflectors.
                for k in 0..m {
                    r.get_mut(k, i).assign(b.get(k, i));
                }
                for j in 0..i {
                    larf(r, j, i, m, &tau[j], prec);
                }
            } else {
                break;
            }
        }

        // Generate Householder reflector for R[i..m, i] and apply to
        // trailing columns.
        if i < m.saturating_sub(1) {
            let t = larfg(r, i, m, prec);
            tau[i].assign(&t);
            for i2 in (i + 1)..n {
                larf(r, i, i2, m, &t, prec);
            }
        }
    }

    // Clear below diagonal (columnwise.cpp:167).
    for i in 0..m {
        for j in 0..i.min(n) {
            r.get_mut(i, j).assign(0);
        }
    }
}

fn copy_int_to_mpfr(b: &IntMatrix, r: &mut MatMpfr) {
    for i in 0..b.nrows {
        for j in 0..b.ncols {
            r.get_mut(i, j).assign(b.get(i, j));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_int_matrix(rows: &[&[i64]]) -> IntMatrix {
        let m = rows.len();
        let n = rows[0].len();
        let mut x = IntMatrix::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                x.set(i, j, Integer::from(rows[i][j]));
            }
        }
        x
    }

    fn det3(b: &IntMatrix) -> f64 {
        let mut m = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] = b.get(i, j).to_f64();
            }
        }
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }

    #[test]
    fn fused_bounds_offdiags_and_preserves_det() {
        let prec = 128u32;
        let rows: Vec<&[i64]> = vec![&[10, 3, 7], &[0, 8, 2], &[0, 0, 5]];
        let mut b = make_int_matrix(&rows);
        let det_before = det3(&b).abs();
        let mut r = MatMpfr::zeros(3, 3, prec);
        let mut u = IntMatrix::zeros(3, 3);
        fused_qr_sr(&mut b, &mut r, &mut u);

        // R: size-reduced off-diagonals.
        for i in 0..3 {
            for j in (i + 1)..3 {
                let rij = r.get(i, j).to_f64().abs();
                let rii = r.get(i, i).to_f64().abs();
                assert!(
                    rij <= rii * SR_TOLERANCE + 1e-9,
                    "R[{},{}] = {} exceeds {}·R[{},{}]",
                    i,
                    j,
                    rij,
                    SR_TOLERANCE,
                    i,
                    i
                );
            }
        }

        // |det B| is preserved.
        let det_after = det3(&b).abs();
        assert!((det_after - det_before).abs() < 1e-6);
    }

    #[test]
    fn fused_u_is_unimodular() {
        let prec = 128u32;
        let rows: Vec<&[i64]> = vec![&[5, 11, 3], &[2, 4, 1], &[0, 7, 9]];
        let mut b_orig = make_int_matrix(&rows);
        let b_initial = b_orig.clone();
        let mut r = MatMpfr::zeros(3, 3, prec);
        let mut u = IntMatrix::zeros(3, 3);
        fused_qr_sr(&mut b_orig, &mut r, &mut u);

        // Verify B_new = B_initial · U.
        let expected = crate::math::mat_mul::mat_mul(&b_initial, &u);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(
                    b_orig.get(i, j),
                    expected.get(i, j),
                    "B mismatch at ({}, {})",
                    i,
                    j
                );
            }
        }

        // |det U| = 1.
        let det_u = det3(&u).abs();
        assert!((det_u - 1.0).abs() < 1e-6, "det U = {}", det_u);
    }
}
