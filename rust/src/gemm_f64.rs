//! f64 gemm kernels. C = A * B where A is m x k (row-major), B is k x n
//! (row-major), C is m x n (row-major). `accumulate` chooses C += A*B vs
//! C = A*B.
//!
//! Four implementations, roughly in order of increasing sophistication:
//!   * `gemm_f64_naive`         — direct port of the C++ triple loop.
//!   * `gemm_f64_blocked`       — cache-blocked ikj variant, autovec-friendly.
//!   * `gemm_f64_blocked_par`   — rayon-parallel over row blocks.
//!   * `gemm_f64_avx2`          — explicit AVX2+FMA inner kernel.
//!   * `gemm_f64_avx2_par`      — AVX2+FMA kernel + rayon row parallelism.

use rayon::prelude::*;

const MC: usize = 64;
const KC: usize = 256;
const NC: usize = 256;

/// Direct port of the C++ reference (`elementary_native.cpp` `gemm_xx` with
/// adr=k, adc=1, bdr=n, bdc=1). Kept for correctness cross-checks.
pub fn gemm_f64_naive(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    m: usize,
    n: usize,
    k: usize,
    accumulate: bool,
) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            if accumulate {
                c[i * n + j] += sum;
            } else {
                c[i * n + j] = sum;
            }
        }
    }
}

/// Cache-blocked ikj. The inner-most loop runs over j with a fixed a_il
/// scalar and a contiguous row of B, which lets the compiler autovectorize
/// trivially.
pub fn gemm_f64_blocked(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    m: usize,
    n: usize,
    k: usize,
    accumulate: bool,
) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    if !accumulate {
        c.fill(0.0);
    }

    let mut ii = 0;
    while ii < m {
        let ib = (m - ii).min(MC);
        let mut ll = 0;
        while ll < k {
            let lb = (k - ll).min(KC);
            let mut jj = 0;
            while jj < n {
                let jb = (n - jj).min(NC);
                kernel_ikj(a, b, c, m, n, k, ii, ib, ll, lb, jj, jb);
                jj += NC;
            }
            ll += KC;
        }
        ii += MC;
    }
}

#[inline(always)]
fn kernel_ikj(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    _m: usize,
    n: usize,
    k: usize,
    i0: usize,
    ib: usize,
    l0: usize,
    lb: usize,
    j0: usize,
    jb: usize,
) {
    for i in 0..ib {
        let c_row = &mut c[(i0 + i) * n + j0..(i0 + i) * n + j0 + jb];
        let a_row = &a[(i0 + i) * k + l0..(i0 + i) * k + l0 + lb];
        for l in 0..lb {
            let a_il = a_row[l];
            let b_row = &b[(l0 + l) * n + j0..(l0 + l) * n + j0 + jb];
            for j in 0..jb {
                c_row[j] += a_il * b_row[j];
            }
        }
    }
}

/// Rayon-parallel variant of `gemm_f64_blocked`. We partition C along rows
/// and let each thread own a disjoint slice of the output, which keeps the
/// parallel section free of synchronization.
pub fn gemm_f64_blocked_par(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    m: usize,
    n: usize,
    k: usize,
    accumulate: bool,
) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    if !accumulate {
        c.fill(0.0);
    }

    c.par_chunks_mut(MC * n)
        .enumerate()
        .for_each(|(block_idx, c_block)| {
            let i0 = block_idx * MC;
            let ib = (m - i0).min(MC);
            let a_block = &a[i0 * k..(i0 + ib) * k];
            // Iterate over k-panels of B and j-panels to stay cache-friendly.
            let mut ll = 0;
            while ll < k {
                let lb = (k - ll).min(KC);
                let mut jj = 0;
                while jj < n {
                    let jb = (n - jj).min(NC);
                    for i in 0..ib {
                        let c_row = &mut c_block[i * n + jj..i * n + jj + jb];
                        let a_row = &a_block[i * k + ll..i * k + ll + lb];
                        for l in 0..lb {
                            let a_il = a_row[l];
                            let b_row = &b[(ll + l) * n + jj..(ll + l) * n + jj + jb];
                            for j in 0..jb {
                                c_row[j] += a_il * b_row[j];
                            }
                        }
                    }
                    jj += NC;
                }
                ll += KC;
            }
        });
}

// ---------------------------------------------------------------------------
// Explicit AVX2 + FMA kernels. Guarded by `is_x86_feature_detected` so the
// code falls back to the scalar blocked version on machines without AVX2.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
pub fn gemm_f64_avx2(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    m: usize,
    n: usize,
    k: usize,
    accumulate: bool,
) {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return gemm_f64_blocked(a, b, c, m, n, k, accumulate);
    }
    if !accumulate {
        c.fill(0.0);
    }
    unsafe { gemm_f64_avx2_impl(a, b, c, m, n, k) }
}

#[cfg(target_arch = "x86_64")]
pub fn gemm_f64_avx2_par(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    m: usize,
    n: usize,
    k: usize,
    accumulate: bool,
) {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return gemm_f64_blocked_par(a, b, c, m, n, k, accumulate);
    }
    if !accumulate {
        c.fill(0.0);
    }
    c.par_chunks_mut(MC * n)
        .enumerate()
        .for_each(|(block_idx, c_block)| {
            let i0 = block_idx * MC;
            let ib = (m - i0).min(MC);
            let a_block = &a[i0 * k..(i0 + ib) * k];
            unsafe { avx2_block(a_block, b, c_block, ib, n, k) };
        });
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_f64_avx2_impl(a: &[f64], b: &[f64], c: &mut [f64], m: usize, n: usize, k: usize) {
    let mut ii = 0;
    while ii < m {
        let ib = (m - ii).min(MC);
        let a_block = &a[ii * k..(ii + ib) * k];
        let c_block = &mut c[ii * n..(ii + ib) * n];
        avx2_block(a_block, b, c_block, ib, n, k);
        ii += MC;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_block(a: &[f64], b: &[f64], c: &mut [f64], ib: usize, n: usize, k: usize) {
    use std::arch::x86_64::*;

    // ikj order. For each (i, l), broadcast a[i*k+l] then FMA across a B row
    // in chunks of 4 f64s.
    let n_simd = n - (n % 4);
    let mut ll = 0;
    while ll < k {
        let lb = (k - ll).min(KC);
        let mut jj = 0;
        while jj < n {
            let jb = (n - jj).min(NC);
            let j_end_simd = jj + (jb - (jb % 4));
            for i in 0..ib {
                let a_row = a.as_ptr().add(i * k + ll);
                let c_row = c.as_mut_ptr().add(i * n);
                for l in 0..lb {
                    let a_il = _mm256_set1_pd(*a_row.add(l));
                    let b_row = b.as_ptr().add((ll + l) * n);
                    let mut j = jj;
                    while j < j_end_simd {
                        let c_vec = _mm256_loadu_pd(c_row.add(j));
                        let b_vec = _mm256_loadu_pd(b_row.add(j));
                        let r = _mm256_fmadd_pd(a_il, b_vec, c_vec);
                        _mm256_storeu_pd(c_row.add(j), r);
                        j += 4;
                    }
                    // Scalar tail inside this (l, i) pair.
                    while j < jj + jb {
                        *c_row.add(j) += *a_row.add(l) * *b_row.add(j);
                        j += 1;
                    }
                }
            }
            jj += NC;
        }
        ll += KC;
    }
    // Silence unused warnings when n is exactly divisible.
    let _ = n_simd;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: &[f64], b: &[f64], eps: f64) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            let err = (x - y).abs();
            let scale = x.abs().max(y.abs()).max(1.0);
            assert!(err / scale < eps, "mismatch: {} vs {}", x, y);
        }
    }

    fn reference(a: &[f64], b: &[f64], m: usize, n: usize, k: usize) -> Vec<f64> {
        let mut c = vec![0.0; m * n];
        gemm_f64_naive(a, b, &mut c, m, n, k, false);
        c
    }

    #[test]
    fn blocked_matches_naive() {
        let (m, n, k) = (37, 41, 29);
        let a: Vec<f64> = (0..m * k).map(|i| (i as f64 * 0.37).sin()).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i as f64 * 0.91).cos()).collect();
        let r = reference(&a, &b, m, n, k);
        let mut c = vec![0.0; m * n];
        gemm_f64_blocked(&a, &b, &mut c, m, n, k, false);
        assert_close(&r, &c, 1e-12);
    }

    #[test]
    fn blocked_par_matches_naive() {
        let (m, n, k) = (128, 96, 80);
        let a: Vec<f64> = (0..m * k).map(|i| (i as f64 * 0.3).sin()).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i as f64 * 0.7).cos()).collect();
        let r = reference(&a, &b, m, n, k);
        let mut c = vec![0.0; m * n];
        gemm_f64_blocked_par(&a, &b, &mut c, m, n, k, false);
        assert_close(&r, &c, 1e-12);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_naive() {
        let (m, n, k) = (64, 48, 40);
        let a: Vec<f64> = (0..m * k).map(|i| (i as f64 * 0.5).sin()).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i as f64 * 0.2).cos()).collect();
        let r = reference(&a, &b, m, n, k);
        let mut c = vec![0.0; m * n];
        gemm_f64_avx2(&a, &b, &mut c, m, n, k, false);
        assert_close(&r, &c, 1e-12);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_par_matches_naive() {
        let (m, n, k) = (130, 77, 53);
        let a: Vec<f64> = (0..m * k).map(|i| (i as f64 * 0.15).sin()).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i as f64 * 0.43).cos()).collect();
        let r = reference(&a, &b, m, n, k);
        let mut c = vec![0.0; m * n];
        gemm_f64_avx2_par(&a, &b, &mut c, m, n, k, false);
        assert_close(&r, &c, 1e-12);
    }
}
