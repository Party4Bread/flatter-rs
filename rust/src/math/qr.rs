//! Householder QR for MPFR matrices. Port of the core algorithm in
//! `src/problems/qr_factorization/householder_mpfr.cpp`, stripped of the
//! block-reflector, workspace-buffer, and OpenMP task machinery so the
//! useful math is visible.
//!
//! Input: m×n MPFR matrix A (with m >= n).
//! Output:
//!   * A is overwritten. The upper triangle of A is R (the triangular
//!     factor). The strict lower triangle stores the essential parts of
//!     the Householder vectors (v[1..]) — these are only needed if the
//!     caller wants to apply Q later, which the flatter pipeline does
//!     not in the classical-LLL-over-R path we're building for.
//!   * `tau[i]` is the scalar for the i-th reflector, I - τ·vvᵀ.

use rug::{Assign, Float};

use crate::math::mat_mpfr::MatMpfr;

/// Computes QR = A in place, with R stored in the upper triangle of A
/// and Householder vectors implicitly in the strict lower triangle
/// (with an implicit leading 1 not stored).
pub fn householder_qr(a: &mut MatMpfr, tau: &mut Vec<Float>) {
    let (m, n) = (a.nrows, a.ncols);
    let prec = a.prec;
    tau.clear();
    tau.resize(n.min(m), Float::new(prec));

    for i in 0..n.min(m) {
        // Generate Householder reflector on the sub-column A[i..m, i].
        let t = larfg(a, i, m, prec);
        tau[i].assign(&t);

        // Apply reflector to the trailing submatrix A[i..m, i+1..n].
        if i + 1 < n {
            for j in (i + 1)..n {
                larf(a, i, j, m, &t, prec);
            }
        }
    }
}

/// LARFG: given an (m-i)-vector x = A[i..m, i], compute alpha and τ and
/// an implicit v (v[0] = 1, v[1..] stored in A[i+1..m, i]) such that
/// `(I - τ·vvᵀ)·x = (alpha, 0, ..., 0)ᵀ`.
///
/// Convention (LAPACK): alpha overwrites `A[i, i]`, v[1..] stays in
/// `A[i+1..m, i]` scaled so that v[0] is implicitly 1.
pub(crate) fn larfg(a: &mut MatMpfr, i: usize, m: usize, prec: u32) -> Float {
    let n = a.ncols;

    // xnorm2 = sum_{k=i+1}^{m-1} A[k,i]²
    let mut xnorm2 = Float::new(prec);
    let mut tmp = Float::new(prec);
    for k in (i + 1)..m {
        tmp.assign(a.get(k, i));
        tmp.square_mut();
        xnorm2 += &tmp;
    }

    let alpha_old = Float::with_val(prec, a.get(i, i));

    // If the sub-column below the diagonal is zero, no reflection needed;
    // τ = 0 and v = 0.
    if xnorm2.is_zero() {
        return Float::new(prec);
    }

    // beta = -sign(alpha) * sqrt(alpha^2 + xnorm2)
    let mut alpha_sq = alpha_old.clone();
    alpha_sq.square_mut();
    let mut beta_sq = Float::with_val(prec, &alpha_sq + &xnorm2);
    beta_sq.sqrt_mut();
    let mut beta = beta_sq;
    if alpha_old.is_sign_positive() {
        beta = -beta;
    }

    // τ = (beta - alpha) / beta
    let mut tau = Float::with_val(prec, &beta - &alpha_old);
    tau /= &beta;

    // v[0] = 1 (implicit). For k > 0: v[k] = x[k] / (alpha - beta).
    //   scale := 1 / (alpha - beta)
    let scale = Float::with_val(prec, &alpha_old - &beta);
    // scale is "alpha - beta"; we divide v entries by this.
    // Guard against zero (shouldn't happen given the norm check above).
    if scale.is_zero() {
        return Float::new(prec);
    }
    // Overwrite the sub-column with scaled values.
    for k in (i + 1)..m {
        // a[k,i] = a[k,i] / scale
        let v = Float::with_val(prec, a.get(k, i) / &scale);
        a.get_mut(k, i).assign(&v);
    }
    // A[i, i] := beta  (the new diagonal entry of R)
    a.get_mut(i, i).assign(&beta);

    let _ = n;
    tau
}

/// LARF: apply the Householder reflector H = I - τ·vvᵀ to column j of A,
/// where v is stored implicitly in A[i..m, i] with v[0] = 1.
pub(crate) fn larf(a: &mut MatMpfr, i: usize, j: usize, m: usize, tau: &Float, prec: u32) {
    if tau.is_zero() {
        return;
    }
    // dot = vᵀ · A[i..m, j]
    //     = 1·A[i,j] + sum_{k=i+1..m} A[k,i] · A[k,j]
    let mut dot = Float::with_val(prec, a.get(i, j));
    let mut tmp = Float::new(prec);
    for k in (i + 1)..m {
        tmp.assign(a.get(k, i));
        tmp *= a.get(k, j);
        dot += &tmp;
    }
    // scalar = τ · dot
    let mut scalar = Float::with_val(prec, tau);
    scalar *= &dot;

    // A[i, j] -= scalar · 1
    {
        let aij = a.get_mut(i, j);
        *aij -= &scalar;
    }
    for k in (i + 1)..m {
        // A[k, j] -= scalar · A[k, i]
        let v = Float::with_val(prec, &scalar * a.get(k, i));
        *a.get_mut(k, j) -= &v;
    }
}

/// After `householder_qr`, zero the strict lower triangle to leave a
/// clean R. Useful for size-reduction passes that only read the upper
/// triangle of R but don't want stray Householder bytes.
pub fn clear_subdiagonal(a: &mut MatMpfr) {
    let m = a.nrows;
    let n = a.ncols;
    for i in 0..m {
        for j in 0..i.min(n) {
            a.get_mut(i, j).assign(0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mat_from_rows(rows: &[&[f64]], prec: u32) -> MatMpfr {
        let m = rows.len();
        let n = rows[0].len();
        let mut a = MatMpfr::zeros(m, n, prec);
        for i in 0..m {
            for j in 0..n {
                a.get_mut(i, j).assign(rows[i][j]);
            }
        }
        a
    }

    /// Upper triangle of A should satisfy ||R[i,i]|| = ||b*_i|| where
    /// b*_i is the i-th Gram-Schmidt vector of the original column i.
    /// We test that the diagonal of R matches the GS norms for a simple
    /// test matrix.
    #[test]
    fn qr_diagonal_is_gs_norms() {
        let prec = 128u32;
        let rows: Vec<&[f64]> = vec![&[2.0, 1.0], &[1.0, 3.0], &[0.0, 0.0]];
        let mut a = mat_from_rows(&rows, prec);
        let mut tau = Vec::new();
        householder_qr(&mut a, &mut tau);
        clear_subdiagonal(&mut a);

        // For input with cols (2,1,0) and (1,3,0):
        //   ||b_0|| = sqrt(5)
        //   b*_1 = b_1 - <b_1, b_0>/<b_0, b_0> · b_0
        //        = (1,3,0) - 5/5 · (2,1,0) = (-1, 2, 0)
        //   ||b*_1|| = sqrt(5)
        // Diagonal of R is (±sqrt(5), ±sqrt(5)) up to sign.
        let r00 = a.get(0, 0).to_f64().abs();
        let r11 = a.get(1, 1).to_f64().abs();
        assert!((r00 - 5f64.sqrt()).abs() < 1e-10, "r00 = {}", r00);
        assert!((r11 - 5f64.sqrt()).abs() < 1e-10, "r11 = {}", r11);
        // Strict lower must be zero after clear_subdiagonal.
        assert!(a.get(1, 0).is_zero());
        assert!(a.get(2, 0).is_zero());
        assert!(a.get(2, 1).is_zero());
    }
}
