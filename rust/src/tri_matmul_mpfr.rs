//! Port of `src/problems/matrix_multiplication/tri_matmul.cpp` using the
//! `rug` crate (Rust bindings for GMP/MPFR). This shows that the MPFR-using
//! parts of flatter don't need a full rewrite — we bind to the same C
//! library flatter uses, with a safe Rust surface.
//!
//! We port the `side=R, uplo=L, transa=N, diag=U` branch — the most common
//! one on the hot path (used by the QR right-sweep during reduction). Layout:
//! row-major, no strides, which matches the bench harness. The C++ version
//! supports arbitrary `lda`/`ldb`; that's straightforward to add but not
//! needed to measure the kernel.
//!
//! Two variants ship: a direct port and a rayon-parallel version that
//! processes independent `(j, k)` column pairs in parallel. Rug's `Float`
//! is `Send + Sync` so rayon works without juggling raw mpfr_t pointers.

use rayon::prelude::*;
use rug::{
    float::Round,
    ops::{AddAssignRound, MulAssignRound},
    Assign, Float,
};

/// Computes `B := alpha * B * A` where `A` is `n x n` unit-lower-triangular
/// (the diagonal is implicit 1, strictly-upper entries are ignored).
/// `B` is `m x n` row-major. Corresponds to the
/// `side='R', uplo='L', transa='N', diag='U'` branch of `tri_matmul.cpp`.
pub fn tri_mm_rlnu(a: &[Float], b: &mut [Float], alpha: &Float, m: usize, n: usize) {
    assert_eq!(a.len(), n * n);
    assert_eq!(b.len(), m * n);
    let prec = alpha.prec();
    let mut tmp = Float::new(prec);
    let mut prod = Float::new(prec);

    // C++:
    //   for j = 0..n:
    //     tmp = alpha * 1                       (diag='U' means A[j,j]=1)
    //     for i = 0..m: B[i,j] *= tmp
    //     for k = j+1..n:
    //       if A[k,j] != 0:
    //         tmp = alpha * A[k,j]
    //         for i = 0..m:
    //           prod = tmp * B[i,k]
    //           B[i,j] += prod
    for j in 0..n {
        tmp.assign(alpha);
        for i in 0..m {
            b[i * n + j].mul_assign_round(&tmp, Round::Nearest);
        }
        for k in (j + 1)..n {
            let a_kj = &a[k * n + j];
            if !a_kj.is_zero() {
                tmp.assign(alpha);
                tmp.mul_assign_round(a_kj, Round::Nearest);
                for i in 0..m {
                    prod.assign(&tmp);
                    prod.mul_assign_round(&b[i * n + k], Round::Nearest);
                    b[i * n + j].add_assign_round(&prod, Round::Nearest);
                }
            }
        }
    }
}

/// Rayon-parallel version. We flip to an i-outer / j-inner sweep so the
/// outer loop is embarrassingly parallel — each row of B is independent.
/// The required reads from A are read-only, so no locking is needed.
///
/// Semantically identical to `tri_mm_rlnu` up to MPFR's deterministic
/// round-to-nearest; floating-point reassociation is avoided by keeping
/// the same intra-row order as the C++ reference.
pub fn tri_mm_rlnu_par(a: &[Float], b: &mut [Float], alpha: &Float, m: usize, n: usize) {
    assert_eq!(a.len(), n * n);
    assert_eq!(b.len(), m * n);
    let prec = alpha.prec();

    b.par_chunks_mut(n).for_each(|b_row| {
        let mut tmp = Float::new(prec);
        let mut prod = Float::new(prec);
        // Replicate the per-j sweep from the sequential version but only
        // over this single row of B.
        for j in 0..n {
            tmp.assign(alpha);
            b_row[j].mul_assign_round(&tmp, Round::Nearest);
            for k in (j + 1)..n {
                let a_kj = &a[k * n + j];
                if !a_kj.is_zero() {
                    tmp.assign(alpha);
                    tmp.mul_assign_round(a_kj, Round::Nearest);
                    prod.assign(&tmp);
                    prod.mul_assign_round(&b_row[k], Round::Nearest);
                    b_row[j].add_assign_round(&prod, Round::Nearest);
                }
            }
        }
    });
}

/// Helper: allocate an `m x n` matrix of Floats at precision `prec`,
/// initialised to 0. Useful for the bench harness.
pub fn zeros(m: usize, n: usize, prec: u32) -> Vec<Float> {
    (0..m * n).map(|_| Float::new(prec)).collect()
}

/// Helper: fill an `m x n` matrix with pseudo-random (deterministic) values
/// for benchmarking. The actual values don't matter — we just want the
/// same workload across runs.
pub fn fill_deterministic(data: &mut [Float], seed: u64) {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
    for x in data.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // map top 53 bits to a double in [-1, 1), then into MPFR.
        let bits = (s >> 11) as f64 / (1u64 << 53) as f64;
        *x = Float::with_val(x.prec(), bits * 2.0 - 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independent reference — direct transcription of the C++ `tri_matmul.cpp`
    // branch used for correctness cross-check. Kept separate from the
    // production variant so any mistakes in the port stand out.
    fn reference(a: &[Float], b: &mut [Float], alpha: &Float, m: usize, n: usize) {
        let prec = alpha.prec();
        let mut tmp = Float::new(prec);
        let mut prod = Float::new(prec);
        for j in 0..n {
            tmp.assign(alpha);
            for i in 0..m {
                let t = Float::with_val(prec, &b[i * n + j] * &tmp);
                b[i * n + j] = t;
            }
            for k in (j + 1)..n {
                let a_kj = &a[k * n + j];
                if !a_kj.is_zero() {
                    tmp.assign(alpha);
                    tmp *= a_kj;
                    for i in 0..m {
                        prod.assign(&tmp);
                        prod *= &b[i * n + k];
                        b[i * n + j] += &prod;
                    }
                }
            }
        }
    }

    fn make_a(n: usize, prec: u32) -> Vec<Float> {
        // Unit-lower: A[k,k]=1 (implicit), A[k,j]=sin(kj)/n for k>j.
        let mut a = zeros(n, n, prec);
        for k in 0..n {
            for j in 0..k {
                a[k * n + j] =
                    Float::with_val(prec, ((k * 7 + j * 13) as f64 * 0.037).sin() / n as f64);
            }
        }
        a
    }

    fn make_b(m: usize, n: usize, prec: u32) -> Vec<Float> {
        let mut b = zeros(m, n, prec);
        fill_deterministic(&mut b, 0xCAFE);
        b
    }

    fn eq(a: &[Float], b: &[Float]) -> bool {
        a.iter().zip(b.iter()).all(|(x, y)| x == y)
    }

    #[test]
    fn port_matches_reference() {
        let prec = 128;
        let (m, n) = (9, 11);
        let alpha = Float::with_val(prec, 0.75);
        let a = make_a(n, prec);
        let mut b1 = make_b(m, n, prec);
        let mut b2 = b1.clone();
        reference(&a, &mut b1, &alpha, m, n);
        tri_mm_rlnu(&a, &mut b2, &alpha, m, n);
        assert!(eq(&b1, &b2));
    }

    #[test]
    fn par_matches_reference() {
        let prec = 128;
        let (m, n) = (17, 13);
        let alpha = Float::with_val(prec, -0.375);
        let a = make_a(n, prec);
        let mut b1 = make_b(m, n, prec);
        let mut b2 = b1.clone();
        reference(&a, &mut b1, &alpha, m, n);
        tri_mm_rlnu_par(&a, &mut b2, &alpha, m, n);
        assert!(eq(&b1, &b2));
    }
}
