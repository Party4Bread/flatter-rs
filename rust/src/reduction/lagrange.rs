//! Lagrange reduction (a.k.a. Gauss reduction) for rank-2 lattices. Port
//! of `src/problems/lattice_reduction/lagrange.cpp`. Uses `rug::Integer`
//! (which is a thin wrapper around GMP's `mpz_t` — the same representation
//! flatter uses) so arithmetic behaviour matches the C++ version.

use rug::{Assign, Integer};

use crate::lattice::Lattice;

/// In-place Lagrange reduction of a rank-2 lattice. Matches the flow in
/// `lagrange.cpp:104` for the non-degenerate case (the rank-0/1 edge
/// cases fall out of `reduce.rs`'s dispatch, which doesn't call us for
/// rank < 2).
pub fn reduce_rank2(L: &mut Lattice) {
    assert_eq!(L.rank, 2);
    let dim = L.dimension();
    assert!(dim >= 2);

    // `a = basis col 0`, `b = basis col 1` — each a vector of `dim`
    // integers. We do length comparisons with `rug::Float` (arbitrary
    // precision, matching flatter's MPFR) because the basis entries can
    // be huge.
    let prec: u32 = 128;
    let mut a_len = rug::Float::new(prec);
    let mut b_len = rug::Float::new(prec);
    norm2(&mut a_len, L, 0);
    norm2(&mut b_len, L, 1);

    // Degenerate: one of the vectors is zero. Reduction is trivial.
    if a_len.is_zero() && b_len.is_zero() {
        return;
    }
    if a_len.is_zero() {
        swap_cols(L, 0, 1);
        return;
    }
    if b_len.is_zero() {
        // Already `a` in column 0; nothing to do.
        return;
    }

    // Ensure column 0 is the longer vector — then each swap-and-reduce
    // step makes the second column strictly smaller.
    if a_len < b_len {
        swap_cols(L, 0, 1);
        std::mem::swap(&mut a_len, &mut b_len);
    }

    let mut adotb = rug::Float::new(prec);
    let mut tmp = rug::Float::new(prec);

    loop {
        dot(&mut adotb, L, 0, 1);
        // q = round(adotb / b_len)   (b_len here is ||b||^2, so we need
        // <a, b> / <b, b>). We compute as a Float then round to Integer.
        tmp.assign(&adotb);
        tmp /= &b_len;
        let q_f = tmp.clone().round();
        let q_z = q_f.to_integer().unwrap_or_else(Integer::new);

        // Swap columns 0 and 1, then subtract q * new_a from new_b so
        // new_b = old_a - q * old_b (following the C++ control flow).
        swap_cols(L, 0, 1);
        std::mem::swap(&mut a_len, &mut b_len);
        for i in 0..dim {
            let a_i = L.basis.get(i, 0).clone();
            let mut tmp_i = Integer::from(&a_i * &q_z);
            *L.basis.get_mut(i, 1) -= &tmp_i;
            tmp_i.assign(0);
            let _ = tmp_i;
        }
        // Update ||new_b||^2.
        norm2(&mut b_len, L, 1);

        if a_len <= b_len {
            break;
        }
    }
}

fn swap_cols(L: &mut Lattice, j0: usize, j1: usize) {
    let ncols = L.basis.ncols;
    for i in 0..L.basis.nrows {
        let a = i * ncols + j0;
        let b = i * ncols + j1;
        L.basis.data.swap(a, b);
    }
}

/// `r := ||col_j||^2` as an MPFR Float.
fn norm2(r: &mut rug::Float, L: &Lattice, j: usize) {
    let prec = r.prec();
    r.assign(0);
    let mut t = rug::Float::new(prec);
    for i in 0..L.basis.nrows {
        let v = L.basis.get(i, j);
        t.assign(v);
        t.square_mut();
        *r += &t;
    }
}

/// `r := <col_j0, col_j1>` as an MPFR Float.
fn dot(r: &mut rug::Float, L: &Lattice, j0: usize, j1: usize) {
    let prec = r.prec();
    r.assign(0);
    let mut t = rug::Float::new(prec);
    let mut u = rug::Float::new(prec);
    for i in 0..L.basis.nrows {
        t.assign(L.basis.get(i, j0));
        u.assign(L.basis.get(i, j1));
        t *= &u;
        *r += &t;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauss_2x2() {
        // A simple input that gets reduced.
        let mut L = Lattice::new(2, 2);
        *L.basis.get_mut(0, 0) = Integer::from(100);
        *L.basis.get_mut(1, 0) = Integer::from(0);
        *L.basis.get_mut(0, 1) = Integer::from(51);
        *L.basis.get_mut(1, 1) = Integer::from(1);

        reduce_rank2(&mut L);

        // After Lagrange reduction the shorter vector should be quite
        // small — the classic example reduces to something like (1, 2).
        let a0 = L.basis.get(0, 0).to_i64().unwrap();
        let a1 = L.basis.get(1, 0).to_i64().unwrap();
        let b0 = L.basis.get(0, 1).to_i64().unwrap();
        let b1 = L.basis.get(1, 1).to_i64().unwrap();
        let la = a0 * a0 + a1 * a1;
        let lb = b0 * b0 + b1 * b1;
        assert!(la <= lb, "col 0 must be the shorter vector");
        // |<a,b>| ≤ |a|^2 / 2   (size reduction criterion, ≤ not <)
        assert!((a0 * b0 + a1 * b1).abs() * 2 <= la);
    }
}
