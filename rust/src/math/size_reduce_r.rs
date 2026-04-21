//! Size reduction of a basis given its R factor.
//!
//! Given:
//!   * `B`: integer basis (columns = basis vectors)
//!   * `R`: upper triangular R from B = Q·R (real, MPFR)
//!
//! Produces a unimodular U (same shape as B's col-side) and applies it
//! in-place to B and R so that after the pass every off-diagonal
//! R[i, j] (i < j) satisfies |R[i, j]| ≤ |R[i, i]| / 2.
//!
//! This is the inner kernel of `FusedQRSizeReduction` in
//! `src/problems/fused_qr_sizered/columnwise_double.cpp` (the non-Fused
//! variant — we keep it simple by not refreshing R on the fly). It's
//! enough to feed the recursive-compression loop.

use rug::{ops::NegAssign, Assign, Float, Integer};

use crate::lattice::IntMatrix;
use crate::math::mat_mpfr::MatMpfr;

/// In-place size reduction.
///
/// For j = 1..n and i = j-1..=0:
///   q = round(R[i, j] / R[i, i])
///   if q != 0:
///     R[:, j] -= q · R[:, i]     (only the top i+1 entries matter)
///     B[:, j] -= q · B[:, i]
pub fn size_reduce(b: &mut IntMatrix, r: &mut MatMpfr) {
    let n = b.ncols;
    let dim = b.nrows;
    let prec = r.prec;
    assert_eq!(r.ncols, n);

    let mut q_z = Integer::new();
    let mut q_f = Float::new(prec);
    let mut tmp_i = Integer::new();
    let mut tmp_f = Float::new(prec);

    for j in 1..n {
        for ii in 0..j {
            let i = j - 1 - ii;
            // q = round(R[i, j] / R[i, i])
            if r.get(i, i).is_zero() {
                continue;
            }
            q_f.assign(r.get(i, j));
            q_f /= r.get(i, i);
            q_f.round_mut();
            // Below 0.5-rounded threshold already handled by round.
            if q_f.is_zero() {
                continue;
            }
            // Convert to Integer. `to_integer` rounds toward zero for
            // fractional values — since we already rounded, the result
            // is exact.
            let maybe = q_f.clone().to_integer();
            let q = match maybe {
                Some(q) => q,
                None => continue, // non-finite guard
            };
            q_z.assign(&q);

            // R[:, j] -= q · R[:, i]    (for rows 0..=i; rows below i are
            // zero in R because it's upper triangular, so nothing to do
            // for rows > i).
            for row in 0..=i {
                tmp_f.assign(&q_z);
                tmp_f *= r.get(row, i);
                *r.get_mut(row, j) -= &tmp_f;
            }

            // B[:, j] -= q · B[:, i]    (exact integer update).
            let nc = b.ncols;
            for row in 0..dim {
                let rowslice = &mut b.data[row * nc..(row + 1) * nc];
                let (left, right) = rowslice.split_at_mut(j);
                let b_ij = &left[i];
                let b_jj = &mut right[0];
                tmp_i.assign(b_ij);
                tmp_i *= &q_z;
                *b_jj -= &tmp_i;
            }
        }
    }

    // NegAssign is a no-op; keep the import used so rustc doesn't warn.
    let _: fn(&mut Float) = <Float as NegAssign>::neg_assign;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::qr::{clear_subdiagonal, householder_qr};

    /// Build R via QR, size-reduce (B, R), check off-diagonals of R are
    /// bounded and |det(B)| is preserved.
    #[test]
    fn size_reduce_bounds_offdiags() {
        let prec = 128u32;
        let mut b = IntMatrix::zeros(3, 3);
        let init: [[i64; 3]; 3] = [[10, 3, 7], [0, 8, 2], [0, 0, 5]];
        for i in 0..3 {
            for j in 0..3 {
                b.set(i, j, Integer::from(init[i][j]));
            }
        }
        let det_before = 10.0 * 8.0 * 5.0;

        let mut r = MatMpfr::zeros(3, 3, prec);
        r.copy_from_int(&b);
        let mut tau = Vec::new();
        householder_qr(&mut r, &mut tau);
        clear_subdiagonal(&mut r);

        size_reduce(&mut b, &mut r);

        // After size reduction: |R[i,j]| ≤ |R[i,i]|/2 for i < j.
        for i in 0..3 {
            for j in (i + 1)..3 {
                let rij = r.get(i, j).to_f64().abs();
                let rii = r.get(i, i).to_f64().abs();
                assert!(
                    rij <= rii / 2.0 + 1e-9,
                    "R[{},{}] = {} exceeds R[{},{}]/2 = {}",
                    i,
                    j,
                    rij,
                    i,
                    i,
                    rii / 2.0
                );
            }
        }

        // |det B| preserved (unimodular transform).
        let det = det3(&b).abs();
        assert!((det - det_before).abs() < 1e-6);
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
}
