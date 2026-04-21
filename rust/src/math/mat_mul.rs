//! Integer matrix multiplication helpers. Promoted from the parked
//! `reduction::heuristic::exact_matmul_in` to a shared module since the
//! recursive heuristic uses this everywhere to fold a sub-lattice's
//! unimodular transform back into the outer basis.

use rug::{Assign, Integer};

use crate::lattice::IntMatrix;

/// Exact `C = A · B`. Dimensions: A is `m×k`, B is `k×n`, C is `m×n`.
/// Entries are `rug::Integer` (GMP `mpz_t`), so no overflow concerns.
pub fn mat_mul(a: &IntMatrix, b: &IntMatrix) -> IntMatrix {
    assert_eq!(a.ncols, b.nrows, "inner dimension mismatch");
    let (m, k, n) = (a.nrows, a.ncols, b.ncols);
    let mut c = IntMatrix::zeros(m, n);
    let mut tmp = Integer::new();
    for i in 0..m {
        for j in 0..n {
            let mut sum = Integer::new();
            for l in 0..k {
                tmp.assign(a.get(i, l) * b.get(l, j));
                sum += &tmp;
            }
            c.set(i, j, sum);
        }
    }
    c
}

/// In-place: `A := A · B` where both are square `n × n`. Allocates
/// scratch once instead of a fresh result matrix each time.
pub fn mat_mul_inplace_square(a: &mut IntMatrix, b: &IntMatrix) {
    assert_eq!(a.nrows, a.ncols);
    assert_eq!(a.ncols, b.nrows);
    assert_eq!(b.nrows, b.ncols);
    let n = a.nrows;
    let mut row_buf: Vec<Integer> = (0..n).map(|_| Integer::new()).collect();
    let mut tmp = Integer::new();
    for i in 0..n {
        for j in 0..n {
            let mut sum = Integer::new();
            for l in 0..n {
                tmp.assign(a.get(i, l) * b.get(l, j));
                sum += &tmp;
            }
            row_buf[j] = sum;
        }
        for j in 0..n {
            *a.get_mut(i, j) = std::mem::take(&mut row_buf[j]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make(n: usize, vals: &[i64]) -> IntMatrix {
        let mut m = IntMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                m.set(i, j, Integer::from(vals[i * n + j]));
            }
        }
        m
    }

    #[test]
    fn identity_matmul() {
        let n = 3;
        let mut i3 = IntMatrix::zeros(n, n);
        for i in 0..n {
            i3.set(i, i, Integer::from(1));
        }
        let b = make(n, &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let c = mat_mul(&i3, &b);
        for i in 0..n {
            for j in 0..n {
                assert_eq!(c.get(i, j), b.get(i, j));
            }
        }
    }

    #[test]
    fn inplace_matches_fresh() {
        let a = make(3, &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let b = make(3, &[9, 8, 7, 6, 5, 4, 3, 2, 1]);
        let c = mat_mul(&a, &b);
        let mut a2 = a.clone();
        mat_mul_inplace_square(&mut a2, &b);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(c.get(i, j), a2.get(i, j), "row {} col {}", i, j);
            }
        }
    }

    #[test]
    fn matmul_big_entries() {
        // 200-bit-ish entries.
        let base: Integer = Integer::from(1u64) << 200;
        let mut a = IntMatrix::zeros(2, 2);
        a.set(0, 0, base.clone());
        a.set(0, 1, Integer::from(3));
        a.set(1, 0, Integer::from(5));
        a.set(1, 1, base.clone());
        let c = mat_mul(&a, &a);
        // c[0,0] = base^2 + 3*5 = base^2 + 15
        let expected = Integer::from(&base * &base) + 15;
        assert_eq!(c.get(0, 0), &expected);
    }
}
