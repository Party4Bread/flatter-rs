//! `RelativeSizeReduction` — size-reduce matrix B2 against B1.
//! Port of `src/problems/relative_size_reduction/relative_size_reduction.cpp`
//! and its `Triangular` implementation
//! (`src/problems/relative_size_reduction/triangular.cpp`).
//!
//! Given an upper-triangular B1 (m×m) and a general B2 (m×n_B2), produce:
//!   * updated B2 with each column size-reduced against B1
//!     (off-diagonal magnitude bounded by |B1[i,i]|/2)
//!   * accumulated unimodular U (m×n_B2) with U[i,j] = -c_int[i,j]
//!     (the integer multipliers used)
//!   * if `new_shift != 0`, scale B2 rows by `2^{-new_shift}` on the
//!     way out — matches the C++ triangular.cpp:80 post-scaling.
//!
//! This port only implements the `Triangular` specialization — the
//! `Generic`/`Orthogonal`/`OrthogonalDouble` paths from C++ aren't
//! ported. `Triangular` is what Heuristic2's update_R path uses.

use rug::{ops::DivRounding, Assign, Integer};

use crate::lattice::IntMatrix;

pub struct RsrTriangular<'a> {
    pub b1: &'a IntMatrix,
    pub b2: &'a mut IntMatrix,
    pub u: &'a mut IntMatrix,
    pub new_shift: i32,
}

impl<'a> RsrTriangular<'a> {
    /// Build a Triangular RSR. Assumes `b1.is_upper_triangular`.
    pub fn new(b1: &'a IntMatrix, b2: &'a mut IntMatrix, u: &'a mut IntMatrix) -> Self {
        assert_eq!(b1.nrows, b1.ncols, "B1 must be square");
        assert_eq!(b2.nrows, b1.nrows, "B1 and B2 row count");
        assert_eq!(u.nrows, b1.nrows, "U and B1 row count");
        assert_eq!(u.ncols, b2.ncols, "U and B2 col count");
        Self {
            b1,
            b2,
            u,
            new_shift: 0,
        }
    }

    /// `Triangular::solve` from `triangular.cpp:44`.
    pub fn solve(&mut self) {
        let rows = self.b2.nrows;
        let cols = self.b2.ncols;

        let mut c_int = Integer::new();
        let mut num = Integer::new();
        let mut den = Integer::new();
        let mut prod = Integer::new();

        for j in 0..cols {
            for i in 0..rows {
                let row = rows - i - 1;

                let diag = self.b1.get(row, row);
                if diag.is_zero() {
                    // degenerate column — nothing to reduce against
                    continue;
                }
                let entry = self.b2.get(row, j).clone();

                // c_int = round(entry / diag)
                //       = floor((2·entry + diag) / 2·diag)   for positive diag
                // Match the C++ formula using fdiv_q.
                num.assign(&entry);
                num <<= 1u32;
                num += diag;
                den.assign(diag);
                den <<= 1u32;
                c_int.assign(&num);
                c_int.div_floor_assign(&den);

                if !c_int.is_zero() {
                    // U[row, j] := -c_int (we're RSR so the sign convention
                    // matches triangular.cpp:73).
                    let mut neg = c_int.clone();
                    neg.neg_assign();
                    self.u.set(row, j, neg);

                    for k in 0..=row {
                        prod.assign(self.b1.get(k, row) * &c_int);
                        let b2_cell = self.b2.get_mut(k, j);
                        *b2_cell -= &prod;
                    }
                }

                // Post-scale the (row, j) entry by 2^{-new_shift}, matching
                // triangular.cpp:80–84.
                if self.new_shift != 0 {
                    let cell = self.b2.get_mut(row, j);
                    if self.new_shift < 0 {
                        *cell <<= (-self.new_shift) as u32;
                    } else {
                        *cell >>= self.new_shift as u32;
                    }
                }
            }
        }
    }
}

/// Extension trait — `rug::Integer` has `div_floor` but we need
/// in-place. Simple wrapper.
trait IntDivFloorAssign {
    fn div_floor_assign(&mut self, divisor: &Integer);
}
impl IntDivFloorAssign for Integer {
    fn div_floor_assign(&mut self, divisor: &Integer) {
        let num = std::mem::replace(self, Integer::new());
        *self = num.div_floor(divisor.clone());
    }
}

/// Extension trait for in-place negation.
trait IntNegAssign {
    fn neg_assign(&mut self);
}
impl IntNegAssign for Integer {
    fn neg_assign(&mut self) {
        let x = std::mem::replace(self, Integer::new());
        *self = -x;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangular_size_reduces() {
        // B1 = diag(10, 7), B2 column = [13, 5]
        //   row=1: q = round(5/7) = 1; B2[1] -= 7, B2[0] -= 0 (since B1[0,1]=0).
        //     wait — the loop is row = rows-i-1, starting with the BOTTOM row.
        //     i=0: row=1. diag=7. entry=5. q = round(5/7) = 1.
        //       Update: B2[row, j] -= q * B1[k, row] for k <= row.
        //       row=1, so k=0,1: B2[0,0] -= 1 * B1[0,1] = 1 * 0 = 0 ; B2[1,0] -= 1 * 7 = -2
        //     After: B2 = [13, -2]
        //     i=1: row=0. diag=10. entry=13. q = round(13/10) = 1.
        //       B2[0,0] -= 1 * B1[0,0] = 10. B2 = [3, -2]
        let mut b1 = IntMatrix::zeros(2, 2);
        b1.set(0, 0, Integer::from(10));
        b1.set(1, 1, Integer::from(7));
        let mut b2 = IntMatrix::zeros(2, 1);
        b2.set(0, 0, Integer::from(13));
        b2.set(1, 0, Integer::from(5));
        let mut u = IntMatrix::zeros(2, 1);

        let mut rsr = RsrTriangular::new(&b1, &mut b2, &mut u);
        rsr.solve();

        assert_eq!(b2.get(0, 0).to_i64().unwrap(), 3);
        assert_eq!(b2.get(1, 0).to_i64().unwrap(), -2);

        // Every B2 entry satisfies |b2[i,j]| <= |b1[i,i]|/2 (post-reduce).
        for j in 0..b2.ncols {
            for i in 0..b2.nrows {
                let v = b2.get(i, j).to_i64().unwrap().abs();
                let d = b1.get(i, i).to_i64().unwrap().abs();
                assert!(v * 2 <= d, "|b2[{},{}]|={} > |b1[{},{}]|/2={}", i, j, v, i, i, d / 2);
            }
        }
    }
}
