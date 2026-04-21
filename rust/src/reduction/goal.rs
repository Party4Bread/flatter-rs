//! `LatticeReductionGoal`: quality parameter. Port of
//! `include/flatter/problems/lattice_reduction/goal.h`.
//!
//! The CLI surfaces three equivalent ways to specify quality:
//!   * `alpha`  — directly (slope parameter, default 2·log2(1.0219))
//!   * `rhf`    — root Hermite factor; alpha = 2·log2(rhf)
//!   * `delta`  — LLL delta; alpha ≈ (0.255 / delta)^2
//!
//! From the C++ `apps/flatter.cpp` parsing code.

#[derive(Clone, Debug)]
pub struct LatticeReductionGoal {
    pub n: usize,
    pub alpha: f64,
}

impl LatticeReductionGoal {
    pub fn from_slope(n: usize, alpha: f64) -> Self {
        Self { n, alpha }
    }

    pub fn from_rhf(n: usize, rhf: f64) -> Self {
        Self::from_slope(n, 2.0 * rhf.log2())
    }

    /// LLL delta → alpha approximation, matching apps/flatter.cpp:89.
    /// `delta = 0.255 / sqrt(alpha)` ⇒ `alpha = (0.255 / delta)^2`.
    pub fn from_delta(n: usize, delta: f64) -> Self {
        Self::from_slope(n, (0.255 / delta).powi(2))
    }

    pub fn rhf(&self) -> f64 {
        (2f64).powf(self.alpha / 2.0)
    }

    /// LLL delta, inverse of `from_delta`. Only used for reporting.
    pub fn delta(&self) -> f64 {
        0.255 / self.alpha.sqrt()
    }
}
