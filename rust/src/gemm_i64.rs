//! i64 gemm kernels. Mirrors gemm_f64 but with wrapping integer arithmetic —
//! the C++ version also just lets i64 overflow wrap, because the library
//! only calls this once the intermediate lattice coefficients are known to
//! fit (a reduction-and-scale invariant maintained by the surrounding code).
//!
//! AVX-512DQ provides a single-instruction 64-bit packed mullo
//! (`_mm512_mullo_epi64`). We use it when available, which is the case on
//! the Xeon this bench runs on; otherwise we fall back to scalar, since
//! emulating 64x64->64 mul on AVX2 via PMULUDQ shuffles ends up slower than
//! the autovectorized scalar loop in practice.

use rayon::prelude::*;

const MC: usize = 64;
const KC: usize = 256;
const NC: usize = 256;

/// Direct port of the C++ reference (`elementary_native.cpp` `gemm_xx`).
pub fn gemm_i64_naive(
    a: &[i64],
    b: &[i64],
    c: &mut [i64],
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
            let mut sum: i64 = 0;
            for l in 0..k {
                sum = sum.wrapping_add(a[i * k + l].wrapping_mul(b[l * n + j]));
            }
            if accumulate {
                c[i * n + j] = c[i * n + j].wrapping_add(sum);
            } else {
                c[i * n + j] = sum;
            }
        }
    }
}

/// Cache-blocked ikj. The inner loop is a contiguous broadcast-FMA pattern
/// which LLVM reliably autovectorizes to AVX-512 VPMULLQ+VPADDQ on the bench
/// machine (target-cpu=native enables avx512dq).
pub fn gemm_i64_blocked(
    a: &[i64],
    b: &[i64],
    c: &mut [i64],
    m: usize,
    n: usize,
    k: usize,
    accumulate: bool,
) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    if !accumulate {
        c.fill(0);
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
                for i in 0..ib {
                    let c_row = &mut c[(ii + i) * n + jj..(ii + i) * n + jj + jb];
                    let a_row = &a[(ii + i) * k + ll..(ii + i) * k + ll + lb];
                    for l in 0..lb {
                        let a_il = a_row[l];
                        let b_row = &b[(ll + l) * n + jj..(ll + l) * n + jj + jb];
                        for j in 0..jb {
                            c_row[j] = c_row[j].wrapping_add(a_il.wrapping_mul(b_row[j]));
                        }
                    }
                }
                jj += NC;
            }
            ll += KC;
        }
        ii += MC;
    }
}

/// Rayon-parallel variant. Split C by row blocks so there's no cross-thread
/// write contention.
pub fn gemm_i64_blocked_par(
    a: &[i64],
    b: &[i64],
    c: &mut [i64],
    m: usize,
    n: usize,
    k: usize,
    accumulate: bool,
) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    if !accumulate {
        c.fill(0);
    }

    c.par_chunks_mut(MC * n)
        .enumerate()
        .for_each(|(block_idx, c_block)| {
            let i0 = block_idx * MC;
            let ib = (m - i0).min(MC);
            let a_block = &a[i0 * k..(i0 + ib) * k];
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
                                c_row[j] = c_row[j].wrapping_add(a_il.wrapping_mul(b_row[j]));
                            }
                        }
                    }
                    jj += NC;
                }
                ll += KC;
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(a: &[i64], b: &[i64], m: usize, n: usize, k: usize) -> Vec<i64> {
        let mut c = vec![0i64; m * n];
        gemm_i64_naive(a, b, &mut c, m, n, k, false);
        c
    }

    #[test]
    fn blocked_matches_naive() {
        let (m, n, k) = (40, 50, 30);
        let a: Vec<i64> = (0..m * k).map(|i| (i as i64) * 3 - 17).collect();
        let b: Vec<i64> = (0..k * n).map(|i| (i as i64) * 5 + 11).collect();
        let r = reference(&a, &b, m, n, k);
        let mut c = vec![0i64; m * n];
        gemm_i64_blocked(&a, &b, &mut c, m, n, k, false);
        assert_eq!(r, c);
    }

    #[test]
    fn blocked_par_matches_naive() {
        let (m, n, k) = (130, 97, 71);
        let a: Vec<i64> = (0..m * k).map(|i| (i as i64) - 5).collect();
        let b: Vec<i64> = (0..k * n).map(|i| -(i as i64) + 9).collect();
        let r = reference(&a, &b, m, n, k);
        let mut c = vec![0i64; m * n];
        gemm_i64_blocked_par(&a, &b, &mut c, m, n, k, false);
        assert_eq!(r, c);
    }
}
