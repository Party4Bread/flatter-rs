//! Iterated-compression LLL prototype. **Currently unused by the
//! dispatch** — the shadow-lattice construction here diverges for q-ary
//! inputs (profile has huge spread, so flatter's staircase
//! compression scales entries back up instead of down). Kept in the
//! tree with its building blocks — QR in MPFR, size-reduction of R,
//! exact big-int matmul — so the next session can plug them into the
//! actual recursive sublattice-split algorithm.
//!
//! The missing piece is the sublattice split: `RecursiveGeneric`
//! recurses on a contiguous sub-range `[start..end]` of columns, and
//! the compression ratio for that sub-range is much better than for
//! the whole basis. That's where the asymptotic speedup for q-ary and
//! Coppersmith-style inputs comes from.
//!
//! The top-level `reduce` entrypoint in `lll.rs` skips this module for
//! now. Tests and the `reduce_f64`/`reduce_mpfr` paths still exercise
//! the gemm, QR and size-reduce-R kernels.
#![allow(dead_code)]

use rug::{Assign, Integer};

use crate::lattice::{IntMatrix, Lattice};
use crate::math::{mat_mpfr::MatMpfr, qr::householder_qr, size_reduce_r};
use crate::reduction::params::LatticeReductionParams;

/// Bit-length target for entries in the compressed shadow.
const SHADOW_BITS: u64 = 160;

/// Minimum bit-reduction per outer round below which we give up and
/// fall back to classical MPFR LLL.
const MIN_PROGRESS_BITS: u64 = 16;

/// Run iterated-compression LLL. Returns (algorithm_name, outer_rounds,
/// total_inner_iters).
pub fn reduce(
    L: &mut Lattice,
    params: &LatticeReductionParams,
) -> Option<(&'static str, usize, usize)> {
    let n = L.rank;
    let dim = L.basis.nrows;
    if n < 2 || dim < n {
        return None;
    }

    let debug = std::env::var_os("FLATTER_RS_DEBUG").is_some();
    let mut rounds = 0usize;
    let mut total_inner = 0usize;
    let mut prev_max_bits = max_entry_bits(&L.basis);
    if debug {
        eprintln!("[heuristic] entry: n={}, dim={}, max_bits={}", n, dim, prev_max_bits);
    }

    for _ in 0..64 {
        rounds += 1;

        let max_bits = max_entry_bits(&L.basis);

        // Once the basis is small enough for the f64 LLL path, finish
        // there and return.
        if max_bits <= super::lll::F64_MAX_BITS {
            let iters = super::lll::reduce_f64(L, params, None);
            return Some(("iterated-compression→f64", rounds, total_inner + iters));
        }

        // Work at MPFR precision tied to the current entry size.
        // Same formula as reduce_mpfr's precision policy.
        let prec: u32 = (max_bits as u32)
            .saturating_add((n as u32).next_power_of_two().trailing_zeros() + 64)
            .max(128);

        if debug {
            eprintln!("[heuristic] round {}: starting, bits={}, prec={}", rounds, max_bits, prec);
        }

        // Step 1: QR factorize B into R.
        let t0 = std::time::Instant::now();
        let mut r = MatMpfr::zeros(dim, n, prec);
        r.copy_from_int(&L.basis);
        let mut tau = Vec::new();
        householder_qr(&mut r, &mut tau);
        if debug { eprintln!("[heuristic]   QR took {:?}", t0.elapsed()); }
        // We'll use only the upper triangle of R going forward; clear
        // the strict lower so size_reduce_r doesn't see stray
        // Householder v's.
        crate::math::qr::clear_subdiagonal(&mut r);

        // Step 2: size-reduce R (and B in lock-step).
        let t1 = std::time::Instant::now();
        size_reduce_r::size_reduce(&mut L.basis, &mut r);
        if debug { eprintln!("[heuristic]   size_reduce_r took {:?}", t1.elapsed()); }

        // Step 3: build compressed shadow S from R. Shifts follow the
        // formula in recursive_generic.cpp:333 (`get_shifts_for_compression`)
        // — they track the *staircase* structure of the profile so
        // that for monotonically-decreasing profiles (typical of q-ary
        // lattices) we still pick shifts that keep every column
        // numerically alive in the shadow.
        let profile_log2: Vec<f64> = (0..n)
            .map(|i| {
                // log₂|R[i,i]|. Use the MPFR exponent + mantissa.
                let v = r.get(i, i).to_f64_exp();
                let (d, exp) = v;
                if d == 0.0 {
                    f64::NEG_INFINITY
                } else {
                    d.abs().log2() + exp as f64
                }
            })
            .collect();
        let mut max_from_left = vec![0f64; n];
        let mut min_from_right = vec![0f64; n];
        max_from_left[0] = profile_log2[0];
        min_from_right[n - 1] = profile_log2[n - 1];
        for i in 0..n - 1 {
            max_from_left[i + 1] = profile_log2[i + 1].max(max_from_left[i]);
            min_from_right[n - i - 2] =
                profile_log2[n - i - 2].min(min_from_right[n - i - 1]);
        }
        let mut shifts = vec![0i32; n];
        for i in 1..n {
            shifts[i] = shifts[i - 1];
            let compress = min_from_right[i] - max_from_left[i - 1];
            if compress <= 1.0 {
                continue;
            }
            shifts[i] += (compress - 1.0).floor() as i32;
        }
        // Translate so final column of the shadow fits in SHADOW_BITS.
        let spread = max_from_left[n - 1] - shifts[n - 1] as f64 - min_from_right[0];
        let precision = (spread + 30.0).ceil() as i32;
        let new_shift = max_from_left[n - 1].ceil() as i32 - shifts[n - 1] - precision.max(SHADOW_BITS as i32);
        for s in &mut shifts {
            *s += new_shift;
        }

        let mut shadow = Lattice::new(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut val = rug::Float::with_val(prec, r.get(i, j));
                val >>= shifts[j];
                val.round_mut();
                let q = val.to_integer().unwrap_or_else(Integer::new);
                shadow.basis.set(i, j, q);
            }
        }

        // If after per-column compression some diagonal is still zero
        // (can happen for nearly-zero R[j,j]), the shadow lattice is
        // singular — bail.
        let mut any_zero_diag = false;
        for i in 0..n {
            if shadow.basis.get(i, i).is_zero() {
                any_zero_diag = true;
                break;
            }
        }
        if any_zero_diag {
            if debug {
                // Dump the diagonal for inspection.
                let diag: Vec<_> = (0..n).map(|i| shadow.basis.get(i, i).to_string()).collect();
                let exps: Vec<_> = (0..n)
                    .map(|i| r.get(i, i).get_exp().unwrap_or(0))
                    .collect();
                let sh: Vec<_> = shifts.iter().copied().collect();
                eprintln!("[heuristic]   zero diag in shadow; bailing");
                eprintln!("   R[j,j] exps: {:?}", exps);
                eprintln!("   shifts:      {:?}", sh);
                eprintln!("   shadow diag: {:?}", diag);
            }
            return None;
        }

        let t3 = std::time::Instant::now();
        let mut u = IntMatrix::zeros(n, n);
        let sub_iters = super::lll::reduce_f64(&mut shadow, params, Some(&mut u));
        total_inner += sub_iters;
        if debug {
            eprintln!(
                "[heuristic]   shadow LLL took {:?} ({} iters)",
                t3.elapsed(), sub_iters
            );
        }

        // Step 5: apply U to B exactly.
        let t4 = std::time::Instant::now();
        let new_basis = exact_matmul_in(&L.basis, &u);
        L.basis = new_basis;
        if debug { eprintln!("[heuristic]   matmul took {:?}", t4.elapsed()); }

        let new_max_bits = max_entry_bits(&L.basis);
        if debug {
            eprintln!(
                "[heuristic] round {}: inner={}, bits {} -> {}",
                rounds, sub_iters, prev_max_bits, new_max_bits
            );
        }
        if new_max_bits + MIN_PROGRESS_BITS >= prev_max_bits {
            if debug {
                eprintln!("[heuristic] no progress, bailing");
            }
            return None;
        }
        prev_max_bits = new_max_bits;
    }

    if debug {
        eprintln!("[heuristic] ran out of rounds");
    }
    None
}

fn max_entry_bits(b: &IntMatrix) -> u64 {
    let mut mx = 0u64;
    for e in &b.data {
        let bits = e.significant_bits() as u64;
        if bits > mx {
            mx = bits;
        }
    }
    mx
}

fn max_r_bits(r: &MatMpfr) -> u64 {
    // Approximation: use the MPFR exponent. For a Float x, the exponent
    // approximates log₂(|x|) + 1.
    let mut mx: i32 = 0;
    for i in 0..r.nrows.min(r.ncols) {
        for j in i..r.ncols {
            if let Some(exp) = r.get(i, j).get_exp() {
                if exp > mx {
                    mx = exp;
                }
            }
        }
    }
    mx.max(0) as u64
}

/// Exact B·U (both integer matrices). B is `dim×n`, U is `n×n`,
/// result is `dim×n`.
fn exact_matmul_in(b: &IntMatrix, u: &IntMatrix) -> IntMatrix {
    assert_eq!(b.ncols, u.nrows);
    let (dim, k, n) = (b.nrows, b.ncols, u.ncols);
    let mut out = IntMatrix::zeros(dim, n);
    let mut tmp = Integer::new();
    for i in 0..dim {
        for j in 0..n {
            let mut sum = Integer::new();
            for l in 0..k {
                tmp.assign(b.get(i, l) * u.get(l, j));
                sum += &tmp;
            }
            out.set(i, j, sum);
        }
    }
    out
}
