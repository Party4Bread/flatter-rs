//! Profile: the vector of log2 Gram-Schmidt norms. Direct port of
//! `src/profile.cpp` (`get_drop`, `get_spread`) and the accompanying header.

#[derive(Clone, Debug, Default)]
pub struct Profile {
    data: Vec<f64>,
}

impl Profile {
    pub fn new(n: usize) -> Self {
        Self { data: vec![f64::NAN; n] }
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }
    pub fn resize(&mut self, n: usize) {
        self.data.resize(n, f64::NAN);
    }

    /// Sum of entries — log2(|det(B)|) for a full-rank basis.
    pub fn logdet(&self) -> f64 {
        self.data.iter().sum()
    }

    /// Port of `Profile::get_drop` from `src/profile.cpp`.
    pub fn get_drop(&self) -> f64 {
        let n = self.data.len();
        if n == 0 {
            return 0.0;
        }
        let mut max_from_left = vec![0.0f64; n];
        let mut min_from_right = vec![0.0f64; n];
        max_from_left[0] = self.data[0];
        min_from_right[n - 1] = self.data[n - 1];
        for i in 0..n - 1 {
            max_from_left[i + 1] = self.data[i + 1].max(max_from_left[i]);
            min_from_right[n - i - 2] = self.data[n - i - 2].min(min_from_right[n - i - 1]);
        }
        let mut spread = max_from_left[n - 1] - min_from_right[0];
        for i in 0..n - 1 {
            if min_from_right[i + 1] > max_from_left[i] {
                spread -= min_from_right[i + 1] - max_from_left[i];
            }
        }
        spread
    }

    /// Port of `Profile::get_spread` from `src/profile.cpp`.
    pub fn get_spread(&self) -> f64 {
        let n = self.data.len();
        if n == 0 {
            return 0.0;
        }
        let (mut mx, mut mn) = (self.data[0], self.data[0]);
        for &v in &self.data {
            if v > mx {
                mx = v;
            }
            if v < mn {
                mn = v;
            }
        }
        mx - mn
    }
}

impl std::ops::Index<usize> for Profile {
    type Output = f64;
    fn index(&self, i: usize) -> &f64 {
        &self.data[i]
    }
}
impl std::ops::IndexMut<usize> for Profile {
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        &mut self.data[i]
    }
}
