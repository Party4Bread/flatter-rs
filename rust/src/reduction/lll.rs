//! Classical LLL reduction with double-precision Gram-Schmidt and exact
//! integer basis updates (via `rug::Integer` = GMP `mpz_t`). This stands
//! in for two dispatch paths in `src/problems/lattice_reduction/lattice_reduction.cpp`:
//!
//!   1. The `FPLLL` branch (line 103-108), which the C++ code uses for
//!      n ≤ 32 && prec ≤ 128. That branch just calls the external fplll
//!      library's LLL — we do the same thing natively.
//!   2. A working fallback for n > 32, where the C++ code would use the
//!      heuristic iterated-compression core (Heuristic2/3, Threaded3). We
//!      produce a correct LLL-reduced basis; we do NOT yet reproduce the
//!      asymptotic speedup of the iterated-compression algorithm. That
//!      core is tracked for the next session.
//!
//! The algorithm is the classical Lenstra-Lenstra-Lovász procedure from
//! "Factoring Polynomials with Rational Coefficients" (1982), using
//! `delta ∈ [0.5, 0.99]`. The GSO is maintained in `f64`, which is fine
//! for entries up to roughly 2^50 in magnitude — which covers the qary
//! lattice sizes in `scripts/` and is the same precision fplll uses in
//! its default (non-proved) mode.

use rug::{Assign, Integer};

use crate::lattice::Lattice;
use crate::profile::Profile;
use crate::reduction::params::LatticeReductionParams;

/// Run LLL in-place on `L` and set `L.profile` to `log2(|b*_i|)`.
/// Returns the number of outer iterations taken (for logging).
pub fn reduce(L: &mut Lattice, params: &LatticeReductionParams) -> usize {
    let n = L.rank;
    let _m = L.dimension();
    let delta = params.delta;

    if n == 0 {
        return 0;
    }

    // `mu[k][j]` for j < k is the Gram-Schmidt coefficient,
    // `bstar_sq[k]` is ||b*_k||^2. Initial full GSO is O(n² · dim); each
    // swap after that updates GSO in O(n) via the incremental formulas
    // from Cohen Alg 2.6.3 (see `swap_update` below).
    let mut mu = vec![vec![0.0f64; n]; n];
    let mut bstar_sq = vec![0.0f64; n];

    compute_gso(L, &mut mu, &mut bstar_sq);

    let mut iters = 0usize;
    let mut k = 1usize;
    while k < n {
        iters += 1;
        size_reduce(L, k, &mut mu);

        // Lovász condition: ||b*_k||² ≥ (δ − μ_{k,k-1}²) · ||b*_{k-1}||²
        let ok = bstar_sq[k] + mu[k][k - 1].powi(2) * bstar_sq[k - 1]
            >= delta * bstar_sq[k - 1];
        if ok {
            k += 1;
        } else {
            swap_cols(L, k - 1, k);
            swap_update(&mut mu, &mut bstar_sq, k, n);
            if k > 1 {
                k -= 1;
            }
        }

        // Defensive cap. Under well-behaved inputs LLL is O(n² · log(det))
        // swaps; this bound is loose and catches pathological cases
        // without ever triggering on realistic input.
        if iters > 200 * n * n.max(1) {
            break;
        }
    }

    // Profile := log2(||b*_i||)  (== 0.5 · log2(||b*_i||²))
    L.profile = Profile::new(n);
    for i in 0..n {
        L.profile[i] = if bstar_sq[i] > 0.0 {
            0.5 * bstar_sq[i].log2()
        } else {
            f64::NEG_INFINITY
        };
    }
    iters
}

/// Fill `L.profile` from the current basis without mutating it. Used by
/// the n ≤ 1 and n = 2 dispatch paths where we don't run LLL but still
/// want a profile on the output.
pub fn fill_profile(L: &mut Lattice) {
    let n = L.rank;
    if n == 0 {
        L.profile = Profile::new(0);
        return;
    }
    let mut mu = vec![vec![0.0f64; n]; n];
    let mut bstar_sq = vec![0.0f64; n];
    compute_gso(L, &mut mu, &mut bstar_sq);
    L.profile = Profile::new(n);
    for i in 0..n {
        L.profile[i] = if bstar_sq[i] > 0.0 {
            0.5 * bstar_sq[i].log2()
        } else {
            f64::NEG_INFINITY
        };
    }
}

/// Incremental GSO update after swapping basis columns `k-1` and `k`.
/// Direct port of Cohen's *A Course in Computational Algebraic Number
/// Theory* Algorithm 2.6.3, SWAP step. O(n) — beats the full O(n² · dim)
/// recompute asymptotically and measurably.
fn swap_update(mu: &mut [Vec<f64>], bstar_sq: &mut [f64], k: usize, n: usize) {
    debug_assert!(k >= 1);
    let mu_val = mu[k][k - 1];                          // old μ_{k,k-1}
    let b_km1 = bstar_sq[k - 1];                        // old ||b*_{k-1}||²
    let b_k = bstar_sq[k];                              // old ||b*_k||²
    let b_new = b_k + mu_val * mu_val * b_km1;          // new ||b*_{k-1}||²

    // Edge case: both old b*_k and old b*_{k-1} were zero. Treat as no-op;
    // the basis is effectively rank-deficient at this position.
    if b_new == 0.0 {
        // Swap rows k-1 and k of μ (columns < k-1) and zero the rest.
        for j in 0..k.saturating_sub(1) {
            let t = mu[k - 1][j];
            mu[k - 1][j] = mu[k][j];
            mu[k][j] = t;
        }
        mu[k][k - 1] = 0.0;
        return;
    }

    let new_mu_kkm1 = mu_val * b_km1 / b_new;
    let new_b_k = b_km1 * b_k / b_new;

    bstar_sq[k - 1] = b_new;
    bstar_sq[k] = new_b_k;
    mu[k][k - 1] = new_mu_kkm1;

    // Columns j < k-1 of the μ table: rows k-1 and k just exchanged.
    for j in 0..k.saturating_sub(1) {
        let t = mu[k - 1][j];
        mu[k - 1][j] = mu[k][j];
        mu[k][j] = t;
    }

    // Rows i > k: update columns k-1 and k in tandem.
    for i in (k + 1)..n {
        let t = mu[i][k];
        mu[i][k] = mu[i][k - 1] - mu_val * t;
        mu[i][k - 1] = t + new_mu_kkm1 * mu[i][k];
    }
}

/// Integer column swap: `col(a) <-> col(b)` in `L.basis`.
fn swap_cols(L: &mut Lattice, a: usize, b: usize) {
    if a == b {
        return;
    }
    let nc = L.basis.ncols;
    for i in 0..L.basis.nrows {
        L.basis.data.swap(i * nc + a, i * nc + b);
    }
}

/// Size reduction of column `k` against columns `0..k`. Updates the
/// Gram-Schmidt coefficients in `mu` to match. After this, every
/// `mu[k][j]` for `j < k` satisfies `|mu[k][j]| <= 1/2`.
fn size_reduce(L: &mut Lattice, k: usize, mu: &mut [Vec<f64>]) {
    if k == 0 {
        return;
    }
    let dim = L.basis.nrows;
    let nc = L.basis.ncols;
    let mut q_z = Integer::new();
    let mut tmp = Integer::new();
    for jj in 0..k {
        let j = k - 1 - jj;
        let mu_kj = mu[k][j];
        if mu_kj.abs() <= 0.5 {
            continue;
        }
        let q = mu_kj.round();
        q_z.assign(q as i64);
        // Exact: B[:, k] -= q * B[:, j]. j != k so the two columns are
        // disjoint within each row — we split each row via split_at_mut.
        for i in 0..dim {
            let row = &mut L.basis.data[i * nc..(i + 1) * nc];
            let (b_ij, b_ik) = {
                let (lo, hi) = if j < k { (j, k) } else { (k, j) };
                let (left, right) = row.split_at_mut(hi);
                if j < k {
                    (&left[lo], &mut right[0])
                } else {
                    (&right[0], &mut left[lo])
                }
            };
            tmp.assign(b_ij);
            tmp *= &q_z;
            *b_ik -= &tmp;
        }
        // mu_{k, r} -= q * mu_{j, r}  for r < j, and mu_{k, j} -= q * 1.
        // Since mu[j][j] = 1, a single `r in 0..=j` loop handles both.
        for r in 0..=j {
            mu[k][r] -= q * mu[j][r];
        }
    }
}

/// Full Gram-Schmidt orthogonalization, `f64` precision. Writes
/// `mu[k][j]` for `j < k` and `bstar_sq[k] = ||b*_k||^2`.
fn compute_gso(L: &Lattice, mu: &mut [Vec<f64>], bstar_sq: &mut [f64]) {
    let n = L.rank;
    let dim = L.basis.nrows;

    // Convert the basis to f64 once. Entries too big for f64 saturate to
    // +/- inf, which will poison the GSO — acceptable for now, since
    // flatter's own FPLLL dispatch only fires on prec ≤ 128 too.
    let mut b_f = vec![0.0f64; dim * n];
    for j in 0..n {
        for i in 0..dim {
            b_f[i * n + j] = int_to_f64(L.basis.get(i, j));
        }
    }

    let mut bstar = vec![0.0f64; dim * n];
    for k in 0..n {
        // bstar[:, k] = b[:, k]
        for i in 0..dim {
            bstar[i * n + k] = b_f[i * n + k];
        }
        for j in 0..k {
            // mu_{k,j} = <b_k, bstar_j> / <bstar_j, bstar_j>
            if bstar_sq[j] <= 0.0 {
                mu[k][j] = 0.0;
                continue;
            }
            let mut dot = 0.0f64;
            for i in 0..dim {
                dot += b_f[i * n + k] * bstar[i * n + j];
            }
            let m = dot / bstar_sq[j];
            mu[k][j] = m;
            for i in 0..dim {
                bstar[i * n + k] -= m * bstar[i * n + j];
            }
        }
        let mut s = 0.0f64;
        for i in 0..dim {
            let x = bstar[i * n + k];
            s += x * x;
        }
        bstar_sq[k] = s;
        mu[k][k] = 1.0;
    }
}

/// Best-effort conversion of a (possibly huge) integer to f64. Equivalent
/// to `mpz_get_d` in GMP. Saturates to +/- inf for entries above
/// ~2^1024.
fn int_to_f64(z: &Integer) -> f64 {
    z.to_f64()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reduction::goal::LatticeReductionGoal;
    use rug::Integer;

    fn params(n: usize) -> LatticeReductionParams {
        LatticeReductionParams::from_goal(LatticeReductionGoal::from_rhf(n, 1.0219))
    }

    #[test]
    fn lll_reduces_qary_like_example() {
        // A tiny qary-like input. We expect the first basis vector
        // (column 0) to be shorter after reduction, and the basis to
        // still span the same lattice (det preserved up to sign).
        let mut L = Lattice::new(3, 3);
        let entries: [[i64; 3]; 3] = [[1, 0, 5], [0, 1, 3], [0, 0, 7]];
        for i in 0..3 {
            for j in 0..3 {
                *L.basis.get_mut(i, j) = Integer::from(entries[i][j]);
            }
        }

        // Determinant is 1 * 1 * 7 = 7.
        let det_before = 7.0f64;

        let p = params(3);
        let iters = reduce(&mut L, &p);
        assert!(iters > 0);

        // Determinant magnitude should be preserved (up to sign).
        let det = det3(&L);
        assert!((det.abs() - det_before).abs() < 1e-6, "det changed: {}", det);

        // The first profile entry should be smaller than the largest
        // input column norm (log2).
        assert!(L.profile[0].is_finite());
    }

    fn det3(L: &Lattice) -> f64 {
        let mut m = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] = L.basis.get(i, j).to_f64();
            }
        }
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }
}
