//! Dense row-major MPFR matrix. Thin wrapper around `Vec<rug::Float>`
//! with the handful of helpers used by the QR and size-reduction passes.

use rug::{Assign, Float};

#[derive(Clone)]
pub struct MatMpfr {
    pub nrows: usize,
    pub ncols: usize,
    pub prec: u32,
    pub data: Vec<Float>,
}

impl MatMpfr {
    pub fn zeros(nrows: usize, ncols: usize, prec: u32) -> Self {
        Self {
            nrows,
            ncols,
            prec,
            data: (0..nrows * ncols).map(|_| Float::new(prec)).collect(),
        }
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> &Float {
        &self.data[i * self.ncols + j]
    }
    #[inline]
    pub fn get_mut(&mut self, i: usize, j: usize) -> &mut Float {
        &mut self.data[i * self.ncols + j]
    }

    pub fn copy_from_int(&mut self, b: &crate::lattice::IntMatrix) {
        assert_eq!(self.nrows, b.nrows);
        assert_eq!(self.ncols, b.ncols);
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                self.get_mut(i, j).assign(b.get(i, j));
            }
        }
    }
}
