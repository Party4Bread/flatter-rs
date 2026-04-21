//! `SizeReduction` — size-reduce an integer upper-triangular matrix
//! `R`. Port of `src/problems/size_reduction/elementary_ZZ.cpp`
//! (`ElementaryZZ::solve`).
//!
//! Given an upper-triangular integer `R` (n×n), produce:
//!   * updated `R` with each column `i` size-reduced against columns
//!     `j < i`, so that `|R[j, i]| ≤ |R[j, j]|/2` for `j < i`.
//!   * `U` (n×n) such that `R_new = R_old · U`.
//!
//! This is the *non-relative* size reduction — both input and output
//! are the same matrix. Used by `CondUnknown::refine_basis` to clean
//! up the independent sub-basis before the recursive LatticeReduction
//! call.

use rug::{ops::DivRounding, Assign, Integer};

use crate::lattice::IntMatrix;

/// In-place size reduction of an upper-triangular integer matrix
/// `r`. `u` is set to identity then updated to reflect the same
/// column operations. Matches `ElementaryZZ::solve`
/// (elementary_ZZ.cpp:47).
pub fn size_reduce(r: &mut IntMatrix, u: &mut IntMatrix) {
    let n = r.ncols;
    assert_eq!(r.nrows, n, "R must be square");
    assert_eq!(u.nrows, n);
    assert_eq!(u.ncols, n);

    u.set_identity();

    let mut mu = Integer::new();
    let mut num = Integer::new();
    let mut den = Integer::new();
    let mut prod = Integer::new();

    for i in 0..n {
        for ind in 0..i {
            let j = i - ind - 1;

            // mu = round(R[j,i] / R[j,j])
            //    = floor((2·R[j,i] + R[j,j]) / (2·R[j,j]))
            let a = r.get(j, i);
            let b = r.get(j, j);
            if b.is_zero() {
                continue;
            }
            num.assign(a);
            num <<= 1u32;
            num += b;
            den.assign(b);
            den <<= 1u32;
            mu.assign(&num);
            mu = std::mem::take(&mut mu).div_floor(den.clone());

            if mu.is_zero() {
                continue;
            }

            // R[:, i] -= mu · R[:, j];  U[:, i] -= mu · U[:, j]
            for k in 0..n {
                prod.assign(r.get(k, j) * &mu);
                {
                    let rki = r.get_mut(k, i);
                    *rki -= &prod;
                }
                prod.assign(u.get(k, j) * &mu);
                {
                    let uki = u.get_mut(k, i);
                    *uki -= &prod;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_reduce_upper_triangular() {
        // R = [[10, 23, 4],
        //      [ 0,  7, 9],
        //      [ 0,  0, 5]]
        let mut r = IntMatrix::zeros(3, 3);
        r.set(0, 0, Integer::from(10));
        r.set(0, 1, Integer::from(23));
        r.set(0, 2, Integer::from(4));
        r.set(1, 1, Integer::from(7));
        r.set(1, 2, Integer::from(9));
        r.set(2, 2, Integer::from(5));
        let mut u = IntMatrix::zeros(3, 3);
        size_reduce(&mut r, &mut u);

        // After: |R[j, i]| ≤ |R[j, j]|/2 for j < i.
        for i in 0..3 {
            for j in 0..i {
                let v = r.get(j, i).to_i64().unwrap().abs();
                let d = r.get(j, j).to_i64().unwrap().abs();
                assert!(v * 2 <= d, "|R[{},{}]|={} > |R[{},{}]|/2", j, i, v, j, j);
            }
        }
        // |det R| unchanged (= 10 · 7 · 5 = 350).
        let det = r.get(0, 0).to_i64().unwrap()
            * r.get(1, 1).to_i64().unwrap()
            * r.get(2, 2).to_i64().unwrap();
        assert_eq!(det.abs(), 350);
    }
}
