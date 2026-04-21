//! `Lattice`: an integer lattice basis and its Gram-Schmidt profile.
//!
//! Port of `src/data/lattice/lattice.cpp` and `include/flatter/data/lattice/lattice.h`.
//!
//! ## Layout
//!
//! We store the basis as a `dimension × rank` matrix of arbitrary-precision
//! integers in **row-major** order: `B[i, j]` is coordinate `i` of basis
//! vector `j`. The FPLLL text format treats each top-level row as a basis
//! vector, so on read we transpose: inner array `j` becomes column `j` of
//! our matrix. This is exactly what the C++ version does (see
//! `lattice.cpp:93`).

use rug::{Complete, Integer};
use std::io::{BufRead, Write};

use crate::profile::Profile;

/// Dense integer matrix, row-major, `nrows × ncols`.
#[derive(Clone, Debug, Default)]
pub struct IntMatrix {
    pub nrows: usize,
    pub ncols: usize,
    pub data: Vec<Integer>,
}

impl IntMatrix {
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            data: (0..nrows * ncols).map(|_| Integer::new()).collect(),
        }
    }
    pub fn get(&self, i: usize, j: usize) -> &Integer {
        &self.data[i * self.ncols + j]
    }
    pub fn get_mut(&mut self, i: usize, j: usize) -> &mut Integer {
        &mut self.data[i * self.ncols + j]
    }
    pub fn set(&mut self, i: usize, j: usize, v: Integer) {
        self.data[i * self.ncols + j] = v;
    }

    /// True iff every entry below the main diagonal is zero.
    pub fn is_upper_triangular(&self) -> bool {
        for i in 0..self.nrows {
            for j in 0..i.min(self.ncols) {
                if !self.get(i, j).is_zero() {
                    return false;
                }
            }
        }
        true
    }

    /// True iff `self` is the identity (square, 1s on diag, 0s off).
    pub fn is_identity(&self) -> bool {
        if self.nrows != self.ncols {
            return false;
        }
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                let v = self.get(i, j);
                if i == j {
                    if v.to_i64() != Some(1) {
                        return false;
                    }
                } else if !v.is_zero() {
                    return false;
                }
            }
        }
        true
    }

    /// Overwrite with the identity matrix. Panics unless square.
    pub fn set_identity(&mut self) {
        assert_eq!(self.nrows, self.ncols);
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                self.set(i, j, Integer::from(if i == j { 1 } else { 0 }));
            }
        }
    }

    /// Copy the submatrix `[i0..i1, j0..j1]` out as a new owned matrix.
    /// Mirrors `Matrix::submatrix` in the C++ layer.
    pub fn submatrix(&self, i0: usize, i1: usize, j0: usize, j1: usize) -> IntMatrix {
        assert!(i0 <= i1 && i1 <= self.nrows);
        assert!(j0 <= j1 && j1 <= self.ncols);
        let mut out = IntMatrix::zeros(i1 - i0, j1 - j0);
        for i in 0..(i1 - i0) {
            for j in 0..(j1 - j0) {
                out.set(i, j, self.get(i0 + i, j0 + j).clone());
            }
        }
        out
    }

    /// Write `src` into `self` at offset `(i0, j0)`.
    pub fn copy_submatrix_from(&mut self, i0: usize, j0: usize, src: &IntMatrix) {
        assert!(i0 + src.nrows <= self.nrows);
        assert!(j0 + src.ncols <= self.ncols);
        for i in 0..src.nrows {
            for j in 0..src.ncols {
                self.set(i0 + i, j0 + j, src.get(i, j).clone());
            }
        }
    }
}

/// Port of `flatter::Lattice`.
#[derive(Clone, Debug, Default)]
pub struct Lattice {
    pub basis: IntMatrix,
    pub rank: usize,
    pub profile: Profile,
}

impl Lattice {
    pub fn new(nvecs: usize, dimension: usize) -> Self {
        let basis = IntMatrix::zeros(dimension, nvecs);
        let rank = dimension.min(nvecs);
        Self {
            basis,
            rank,
            profile: Profile::new(rank),
        }
    }

    pub fn resize(&mut self, nvecs: usize, dimension: usize) {
        *self = Self::new(nvecs, dimension);
    }

    pub fn dimension(&self) -> usize {
        self.basis.nrows
    }
    pub fn nvecs(&self) -> usize {
        self.basis.ncols
    }

    /// Port of `Lattice::update_rank`: trim trailing all-zero columns
    /// from the effective rank.
    pub fn update_rank(&mut self) {
        let B = &self.basis;
        let mut r = self.rank;
        for i in 0..B.ncols {
            let col = B.ncols - 1 - i;
            let mut any_nonzero = false;
            for j in 0..B.nrows {
                if !B.get(j, col).is_zero() {
                    any_nonzero = true;
                    break;
                }
            }
            if !any_nonzero {
                r = col;
            }
        }
        r = r.min(B.nrows);
        self.rank = r;
        let mut p = Profile::new(r);
        for i in 0..r.min(self.profile.len()) {
            p[i] = self.profile[i];
        }
        self.profile = p;
    }

    // -- FPLLL format I/O ---------------------------------------------------
    //
    // Parse rules (from `lattice.cpp:93`):
    //   * The file starts with '['.
    //   * Each basis vector is an inner '[' ... ']' block, with whitespace-
    //     separated integer literals. We consume chars until a ']' then split
    //     the interior on whitespace.
    //   * The top-level ']' closes the file. Intervening whitespace and EOL
    //     are ignored.
    //   * Each integer is parsed with `mpz_set_str(..., 0)` — base detected
    //     from prefix (0x, 0, decimal). rug::Integer::parse_radix(s, 0) is
    //     the equivalent.
    //   * All inner rows must have the same length (the "dimension").
    //   * Nvec rows × dim entries → internally stored as a `dim × nvec`
    //     matrix: `B[i, j]` = row j's entry i (i.e. we transpose on load).

    pub fn read_fplll<R: BufRead>(r: &mut R) -> std::io::Result<Self> {
        use std::io::{Error, ErrorKind};
        let mut all = String::new();
        r.read_to_string(&mut all)?;
        let bytes = all.as_bytes();

        let mut idx = 0;
        let skip_ws = |s: &[u8], i: &mut usize| {
            while *i < s.len() && matches!(s[*i], b' ' | b'\t' | b'\r' | b'\n') {
                *i += 1;
            }
        };

        skip_ws(bytes, &mut idx);
        if idx >= bytes.len() || bytes[idx] != b'[' {
            return Err(Error::new(ErrorKind::InvalidData, "expected '['"));
        }
        idx += 1;

        let mut rows: Vec<Vec<Integer>> = Vec::new();
        loop {
            skip_ws(bytes, &mut idx);
            if idx >= bytes.len() {
                return Err(Error::new(ErrorKind::UnexpectedEof, "unterminated lattice"));
            }
            if bytes[idx] == b']' {
                break;
            }
            if bytes[idx] != b'[' {
                return Err(Error::new(ErrorKind::InvalidData, "expected '[' for row"));
            }
            idx += 1;
            // find closing ']'
            let start = idx;
            while idx < bytes.len() && bytes[idx] != b']' {
                idx += 1;
            }
            if idx >= bytes.len() {
                return Err(Error::new(ErrorKind::UnexpectedEof, "unterminated row"));
            }
            let inner = &all[start..idx];
            idx += 1; // skip ']'

            let mut row = Vec::new();
            for tok in inner.split_ascii_whitespace() {
                // Replicate mpz_set_str(..., base=0): detect prefix.
                //   "0x" or "-0x" → hex, "0" prefix → octal (only for len > 1),
                //   otherwise decimal. FPLLL output is decimal in practice;
                //   we support the other two for parity with the C++ parser.
                let (sign, rest) = match tok.strip_prefix('-') {
                    Some(r) => ("-", r),
                    None => ("", tok),
                };
                let (radix, rest) = if let Some(r) = rest
                    .strip_prefix("0x")
                    .or_else(|| rest.strip_prefix("0X"))
                {
                    (16, r)
                } else if rest.len() > 1 && rest.starts_with('0') && rest[1..].bytes().all(|b| (b'0'..=b'7').contains(&b)) {
                    (8, &rest[1..])
                } else {
                    (10, rest)
                };
                let signed = if sign.is_empty() {
                    rest.to_string()
                } else {
                    format!("-{}", rest)
                };
                match Integer::parse_radix(&signed, radix) {
                    Ok(p) => row.push(Integer::from(p.complete())),
                    Err(_) => {
                        return Err(Error::new(
                            ErrorKind::InvalidData,
                            format!("bad integer '{}'", tok),
                        ))
                    }
                }
            }
            rows.push(row);
        }

        let nvecs = rows.len();
        if nvecs == 0 {
            return Ok(Lattice::new(0, 0));
        }
        let dim = rows[0].len();
        for row in &rows {
            if row.len() != dim {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "inconsistent row widths",
                ));
            }
        }
        let (nvecs, dim) = if dim == 0 { (0, 0) } else { (nvecs, dim) };
        let mut L = Lattice::new(nvecs, dim);
        // `rows[j][i]` = input row j (a basis vector), coord i. Store as
        // column j of an (dim × nvecs) matrix.
        for j in 0..nvecs {
            for i in 0..dim {
                L.basis.set(i, j, std::mem::take(&mut rows[j][i]));
            }
        }
        Ok(L)
    }

    pub fn write_fplll<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        write!(w, "{}", self)
    }
}

/// Matches the C++ `operator<<` byte-for-byte ("[" then rank rows, then "]"
/// each on its own line, columns of the internal matrix printed as rows).
impl std::fmt::Display for Lattice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let rank = self.rank;
        let dim = self.dimension();
        write!(f, "[")?;
        for i in 0..rank {
            write!(f, "[")?;
            for j in 0..dim {
                write!(f, "{}", self.basis.get(j, i))?;
                if j + 1 < dim {
                    write!(f, " ")?;
                }
            }
            writeln!(f, "]")?;
        }
        writeln!(f, "]")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn roundtrip_small() {
        let src = "[[1 0 0 75 47 67]\n\
                   [0 1 0 72 97 59]\n\
                   [0 0 1 99 9 11]\n\
                   [0 0 0 116 0 0]\n\
                   [0 0 0 0 116 0]\n\
                   [0 0 0 0 0 116]]\n";
        let mut r = Cursor::new(src);
        let lat = Lattice::read_fplll(&mut r).unwrap();
        assert_eq!(lat.dimension(), 6);
        assert_eq!(lat.nvecs(), 6);
        assert_eq!(lat.rank, 6);
        // B[0, 3] corresponds to input row 3, column 0, which is 0.
        assert!(lat.basis.get(0, 3).is_zero());
        // B[3, 3] corresponds to input row 3, col 3: 116
        assert_eq!(lat.basis.get(3, 3).to_i64().unwrap(), 116);

        let out = lat.to_string();
        // Re-parse and compare numerically.
        let mut r2 = Cursor::new(out);
        let lat2 = Lattice::read_fplll(&mut r2).unwrap();
        assert_eq!(lat2.dimension(), lat.dimension());
        for i in 0..lat.basis.nrows {
            for j in 0..lat.basis.ncols {
                assert_eq!(lat.basis.get(i, j), lat2.basis.get(i, j));
            }
        }
    }
}
