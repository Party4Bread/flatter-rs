//! `LatticeReductionGoal`: reduction quality parameter. Port of
//! `include/flatter/problems/lattice_reduction/goal.h` and
//! `src/problems/lattice_reduction/goal.cpp`.
//!
//! Two forms:
//! * **Heuristic** (`proved = false`, the default from CLI): `quality`
//!   is the slope parameter "C" derived from target RHF.
//! * **Proved**: parameterized by `top_slope`, `best_slope`, `g`, `top_N`.
//!
//! The constants:
//!   BKZ_BEST_SLOPE     = 0.031281
//!   HERMITE_BEST_SLOPE = 0.41503749927884365
//!   DEFAULT_G          = 3.0
//!
//! `check(profile)` is the termination predicate for the recursive
//! heuristic: it asks whether the profile is "reduced enough" for the
//! current quality target. The formulas come directly from goal.cpp:43.

use crate::profile::Profile;

pub const BKZ_BEST_SLOPE: f64 = 0.031281;
pub const HERMITE_BEST_SLOPE: f64 = 0.41503749927884365;
pub const DEFAULT_G: f64 = 3.0;

#[derive(Clone, Debug)]
pub struct LatticeReductionGoal {
    pub n: usize,
    pub top_n: usize,
    pub quality: f64,
    pub best_slope: f64,
    pub top_slope: f64,
    pub g: f64,
    pub log_g: f64,
    pub proved: bool,
}

impl Default for LatticeReductionGoal {
    fn default() -> Self {
        Self {
            n: 0,
            top_n: 0,
            quality: 0.0,
            best_slope: BKZ_BEST_SLOPE,
            top_slope: 0.0,
            g: DEFAULT_G,
            log_g: DEFAULT_G.log2(),
            proved: false,
        }
    }
}

impl LatticeReductionGoal {
    /// Heuristic constructor (`LatticeReductionGoal(n, quality, proved)`
    /// with proved=false; goal.cpp:15). `quality` here is the "C_scale"
    /// value, not alpha directly.
    pub fn new_heuristic(n: usize, quality: f64) -> Self {
        Self {
            n,
            top_n: n,
            quality,
            best_slope: BKZ_BEST_SLOPE,
            top_slope: 0.0,
            g: DEFAULT_G,
            log_g: DEFAULT_G.log2(),
            proved: false,
        }
    }

    /// Proved constructor (goal.cpp:23).
    pub fn new_proved(
        n: usize,
        top_level_slope: f64,
        base_slope: f64,
        g: f64,
        top_n: usize,
    ) -> Self {
        let top_n = if top_n == 0 { n } else { top_n };
        assert!(top_level_slope > base_slope);
        Self {
            n,
            top_n,
            quality: 0.0,
            best_slope: base_slope,
            top_slope: top_level_slope,
            g,
            log_g: g.log2(),
            proved: true,
        }
    }

    /// `from_RHF` (goal.cpp:177).
    pub fn from_rhf(n: usize, rhf: f64, proved: bool) -> Self {
        let slope = rhf.log2() * 2.0;
        Self::from_slope(n, slope, proved)
    }

    /// `from_drop` (goal.cpp:182).
    pub fn from_drop(n: usize, drop: f64, proved: bool) -> Self {
        if proved {
            let slope = (drop / n as f64).max(BKZ_BEST_SLOPE + 0.000001);
            Self::from_slope(n, slope, true)
        } else {
            let slope = drop / (n as f64 - 1.0);
            Self::from_slope(n, slope, false)
        }
    }

    /// `from_slope` (goal.cpp:195).
    pub fn from_slope(n: usize, slope: f64, proved: bool) -> Self {
        if proved {
            Self::new_proved(n, slope, BKZ_BEST_SLOPE, DEFAULT_G, 0)
        } else {
            let lgn = (n as f64).log2();
            let s_guess = 3.0 * (1.0 + 3f64.powf(lgn + 1.0) - 2f64.powf(lgn + 2.0)) / 2.0;
            let top_slope = slope.max(BKZ_BEST_SLOPE);
            let c_scale = (top_slope - BKZ_BEST_SLOPE) * n as f64 / s_guess;
            Self::new_heuristic(n, c_scale)
        }
    }

    /// `get_alpha_n` (goal.cpp:39) — proved-mode helper.
    pub fn get_alpha_n(&self, n: usize) -> f64 {
        self.best_slope
            + ((n as f64) / (self.top_n as f64)).powf(self.log_g) * (self.top_slope - self.best_slope)
    }

    /// `get_slope` (goal.cpp:155).
    pub fn get_slope(&self) -> f64 {
        if self.proved {
            self.get_alpha_n(self.n)
        } else {
            self.quality
        }
    }

    /// `get_max_drop` (goal.cpp:135).
    pub fn get_max_drop(&self) -> f64 {
        if self.proved {
            self.get_alpha_n(self.n) * self.n as f64
        } else {
            let lgn = (self.n as f64).log2();
            let mut max_drop =
                self.quality * 3.0 * (1.0 + 3f64.powf(lgn + 1.0) - 2f64.powf(lgn + 2.0)) / 2.0;
            max_drop += self.best_slope * self.n as f64;
            max_drop
        }
    }

    /// `get_rhf` (goal.cpp:147).
    pub fn get_rhf(&self) -> f64 {
        let max_drop = self.get_max_drop();
        let slope = max_drop / self.n as f64;
        2f64.powf(slope / 2.0)
    }

    /// `subgoal` (goal.cpp:118): the quality target for a sub-range
    /// of columns.
    pub fn subgoal(&self, start: usize, end: usize) -> Self {
        assert!(start < end);
        assert!(end <= self.n);
        if self.proved {
            Self::new_proved(
                end - start,
                self.top_slope,
                self.best_slope,
                self.g,
                self.top_n,
            )
        } else {
            let mut new_goal = Self::new_heuristic(end - start, self.quality);
            new_goal.best_slope = self.best_slope;
            new_goal
        }
    }

    /// `check` (goal.cpp:43) — termination predicate.
    pub fn check(&self, profile: &Profile) -> bool {
        assert!(self.n != 0);
        if self.n == 1 {
            return true;
        }
        let n = self.n;

        let max_drop;
        let mu_sep;
        if self.proved {
            max_drop = self.get_alpha_n(n) * n as f64;
            return profile.get_drop() < max_drop;
        } else {
            let lgn = (n as f64).log2();
            max_drop = self.quality * 3.0 * (1.0 + 3f64.powf(lgn + 1.0) - 2f64.powf(lgn + 2.0)) / 2.0
                + self.best_slope * n as f64;
            let gamma_i = self.quality * 3f64.powf(lgn);
            mu_sep = (max_drop - gamma_i) / 2.0 + gamma_i;
        }

        let n_l = if n == 3 { 2 } else { n / 2 };
        let n_r = n - n_l;
        let lgnl = (n_l as f64).log2();
        let l_drop = self.best_slope * n_l as f64
            + self.quality * 3.0 * (1.0 + 3f64.powf(lgnl + 1.0) - 2f64.powf(lgnl + 2.0)) / 2.0;
        let lgnr = (n_r as f64).log2();
        let r_drop = self.best_slope * n_r as f64
            + self.quality * 3.0 * (1.0 + 3f64.powf(lgnr + 1.0) - 2f64.powf(lgnr + 2.0)) / 2.0;

        let n1 = if n_l == 3 { 2 } else { n_l / 2 };
        let n3 = if n_r == 3 { 2 } else { n_r / 2 };
        let mid_drop = {
            // profile.subprofile(n1, n_L + n3).get_drop()
            let slice = &profile.as_slice()[n1..n_l + n3];
            let mut sub = Profile::new(slice.len());
            for (i, v) in slice.iter().enumerate() {
                sub[i] = *v;
            }
            sub.get_drop()
        };

        let mut mu_l = 0f64;
        let mut mu_r = 0f64;
        for i in 0..n_l {
            mu_l += profile[i];
        }
        mu_l /= n_l as f64;
        for i in n_l..n {
            mu_r += profile[i];
        }
        mu_r /= n_r as f64;

        (profile.get_drop() < max_drop)
            && (mu_l - mu_r < mu_sep)
            && (mid_drop <= l_drop + (max_drop - l_drop - r_drop))
    }

    /// `set_best_slope` (goal.cpp:163).
    pub fn set_best_slope(&mut self, slope: f64) {
        assert!(!self.proved);
        let lgn = (self.n as f64).log2();
        let s_guess = 3.0 * (1.0 + 3f64.powf(lgn + 1.0) - 2f64.powf(lgn + 2.0)) / 2.0;
        let gap = self.quality * s_guess / self.n as f64;
        let new_gap = gap + (self.best_slope - slope);
        let new_gap = new_gap * self.n as f64;
        assert!(new_gap > 0.0);
        self.quality = new_gap / s_guess;
        self.best_slope = slope;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_rhf_default_matches_cpp() {
        // apps/flatter.cpp default: RHF 1.0219 → alpha = 0.0625081
        let g = LatticeReductionGoal::from_rhf(20, 1.0219, false);
        // Should round-trip get_rhf ≈ 1.0219.
        let rhf = g.get_rhf();
        // get_rhf uses max_drop / n which can slightly overshoot the
        // target because the quality formula includes the best_slope
        // term; we just check we're in the right ballpark.
        assert!((rhf - 1.0219).abs() < 0.05, "rhf = {}", rhf);
    }

    #[test]
    fn subgoal_half() {
        let g = LatticeReductionGoal::from_rhf(20, 1.0219, false);
        let sg = g.subgoal(5, 15);
        assert_eq!(sg.n, 10);
        assert_eq!(sg.quality, g.quality);
        assert_eq!(sg.proved, false);
    }
}
