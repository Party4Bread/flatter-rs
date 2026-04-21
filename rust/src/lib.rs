#![allow(non_snake_case)]
//! Rust port of flatter's elementary matrix-multiplication kernels.
//!
//! The C++ source lives at `src/problems/matrix_multiplication/elementary_native.cpp`
//! and implements the inner `gemm_xx` triple loop used as a fallback when
//! BLAS is not engaged. The loop is:
//!
//! ```text
//! for i in 0..m:
//!     for j in 0..n:
//!         sum = 0
//!         for l in 0..k:
//!             sum += A[i*adr + l*adc] * B[l*bdr + j*bdc]
//!         C[i*ldc + j] = sum   (or += sum)
//! ```
//!
//! We port the common case `adr=k, adc=1, bdr=n, bdc=1, ldc=n` (row-major,
//! non-transposed A and B) which matches the path taken by the library's
//! `!dA.is_transposed() && !dB.is_transposed()` branch.

pub mod gemm_f64;
pub mod gemm_i64;
pub mod lattice;
pub mod math;
pub mod profile;
pub mod reduction;
pub mod tri_matmul_mpfr;

pub use gemm_f64::*;
pub use gemm_i64::*;
pub use lattice::*;
pub use profile::*;
pub use reduction::*;
pub use tri_matmul_mpfr::*;
