//! `SublatticeSplit`: the state machine that picks which window of basis
//! columns to reduce at each step of the recursive heuristic. Port of
//! `src/problems/lattice_reduction/sublattice_split.h` +
//! `sublattice_split_2.cpp` + `sublattice_split_3.cpp`.
//!
//! The split is **purely structural** (depth-`log₂ n` binary tree, no
//! profile awareness). At every level, Phase-2 runs `left → right → all`
//! and Phase-3 alternates between a boundary-straddling "mid" window and
//! separate `left`/`right` windows — C++ uses `SubSplitPhase3` for the
//! `all` child of every Phase-2 node, which is what this Rust port does
//! too. The Phase-3 `mid_child` reference is shared with the left/right
//! children: we model that with `Rc<RefCell<…>>` rather than raw
//! pointers.

use std::cell::RefCell;
use std::rc::Rc;

pub type SplitRc = Rc<RefCell<Split>>;

/// One sublattice window, half-open `[start, end)`.
pub type Sublattice = (usize, usize);

#[derive(Debug)]
pub enum SplitKind {
    Phase2 {
        k: usize,
        left: Option<SplitRc>,
        right: Option<SplitRc>,
        all: SplitRc,
    },
    Phase3 {
        k: usize,
        left: Option<SplitRc>,
        right: Option<SplitRc>,
        mid: Option<SplitRc>,
    },
}

#[derive(Debug)]
pub struct Split {
    pub n: usize,
    pub iter: usize,
    pub kind: SplitKind,
}

impl Split {
    /// Build a fresh Phase-2 split for a lattice of rank `n`.
    /// Matches `SubSplitPhase2::SubSplitPhase2` (sublattice_split_2.cpp:8).
    pub fn new_phase2(n: usize) -> SplitRc {
        if n > 1 {
            let k = next_smaller(n);
            let left = Split::new_phase2(k);
            let right = Split::new_phase2(n - k);
            // `all` child is a Phase-3 whose children are the `all` of
            // left and right. Those `all`s are themselves Phase-3s.
            let l_all = match &left.borrow().kind {
                SplitKind::Phase2 { all, .. } => Some(Rc::clone(all)),
                _ => unreachable!(),
            };
            let r_all = match &right.borrow().kind {
                SplitKind::Phase2 { all, .. } => Some(Rc::clone(all)),
                _ => unreachable!(),
            };
            let all = Split::new_phase3_from_children(l_all, r_all);
            Rc::new(RefCell::new(Split {
                n,
                iter: 0,
                kind: SplitKind::Phase2 {
                    k,
                    left: Some(left),
                    right: Some(right),
                    all,
                },
            }))
        } else {
            // Base case: n == 1. No sub-splitting; `all` is a trivial
            // Phase-3 of size 1.
            let all = Split::new_phase3(1);
            Rc::new(RefCell::new(Split {
                n,
                iter: 0,
                kind: SplitKind::Phase2 {
                    k: n,
                    left: None,
                    right: None,
                    all,
                },
            }))
        }
    }

    /// Build a fresh Phase-3 split for size `n` with default pivot `k = n/2`.
    /// Matches `SubSplitPhase3::SubSplitPhase3(n)` (sublattice_split_3.cpp:8).
    pub fn new_phase3(n: usize) -> SplitRc {
        Split::new_phase3_with_k(n, n / 2)
    }

    fn new_phase3_with_k(n: usize, k: usize) -> SplitRc {
        if n == 3 {
            let k = 2;
            let left = Split::new_phase3_with_k(2, 1);
            let right = Split::new_phase3_with_k(1, 0);
            // mid_child uses left.right_child and right (sublattice_split_3.cpp:25).
            let l_r = match &left.borrow().kind {
                SplitKind::Phase3 { right, .. } => right.clone(),
                _ => unreachable!(),
            };
            let mid = Split::new_phase3_from_children(l_r, Some(Rc::clone(&right)));
            Rc::new(RefCell::new(Split {
                n,
                iter: 0,
                kind: SplitKind::Phase3 {
                    k,
                    left: Some(left),
                    right: Some(right),
                    mid: Some(mid),
                },
            }))
        } else if n >= 2 {
            let left = Split::new_phase3_with_k(k, k / 2);
            let right = Split::new_phase3_with_k(n - k, (n - k) / 2);
            // mid uses left.right and right.left.
            let l_r = match &left.borrow().kind {
                SplitKind::Phase3 { right, .. } => right.clone(),
                _ => unreachable!(),
            };
            let r_l = match &right.borrow().kind {
                SplitKind::Phase3 { left, .. } => left.clone(),
                _ => unreachable!(),
            };
            let mid = Split::new_phase3_from_children(l_r, r_l);
            Rc::new(RefCell::new(Split {
                n,
                iter: 0,
                kind: SplitKind::Phase3 {
                    k,
                    left: Some(left),
                    right: Some(right),
                    mid: Some(mid),
                },
            }))
        } else {
            // n = 0 or 1. Leaf.
            Rc::new(RefCell::new(Split {
                n,
                iter: 0,
                kind: SplitKind::Phase3 {
                    k: n / 2,
                    left: None,
                    right: None,
                    mid: None,
                },
            }))
        }
    }

    /// Phase-3 constructor from existing children (sublattice_split_3.cpp:34).
    fn new_phase3_from_children(l: Option<SplitRc>, r: Option<SplitRc>) -> SplitRc {
        let l_n = l.as_ref().map_or(1, |x| x.borrow().n);
        let r_n = r.as_ref().map_or(1, |x| x.borrow().n);
        let k = l_n;
        let n = k + r_n;

        let mid = if n == 3 {
            if l_n == 2 {
                let l_r = l
                    .as_ref()
                    .and_then(|x| match &x.borrow().kind {
                        SplitKind::Phase3 { right, .. } => right.clone(),
                        _ => None,
                    });
                Some(Split::new_phase3_from_children(l_r, r.as_ref().map(Rc::clone)))
            } else {
                let r_l = r
                    .as_ref()
                    .and_then(|x| match &x.borrow().kind {
                        SplitKind::Phase3 { left, .. } => left.clone(),
                        _ => None,
                    });
                Some(Split::new_phase3_from_children(l.as_ref().map(Rc::clone), r_l))
            }
        } else {
            let l_r = l
                .as_ref()
                .and_then(|x| match &x.borrow().kind {
                    SplitKind::Phase3 { right, .. } => right.clone(),
                    _ => None,
                });
            let r_l = r
                .as_ref()
                .and_then(|x| match &x.borrow().kind {
                    SplitKind::Phase3 { left, .. } => left.clone(),
                    _ => None,
                });
            if l_r.is_some() && r_l.is_some() {
                Some(Split::new_phase3_from_children(l_r, r_l))
            } else {
                None
            }
        };

        Rc::new(RefCell::new(Split {
            n,
            iter: 0,
            kind: SplitKind::Phase3 {
                k,
                left: l,
                right: r,
                mid,
            },
        }))
    }

    /// Current windows to reduce. Phase-2 always returns a single
    /// window; Phase-3 returns 1 or 2.
    /// (`SubSplitPhase2::get_sublattices` at sublattice_split_2.cpp:37;
    /// `SubSplitPhase3::get_sublattices` at sublattice_split_3.cpp:72.)
    pub fn get_sublattices(&self) -> Vec<Sublattice> {
        match &self.kind {
            SplitKind::Phase2 { k, .. } => {
                let iter = self.iter;
                let n = self.n;
                if n == 3 {
                    if iter == 0 {
                        vec![(0, *k)]
                    } else {
                        vec![(0, n)]
                    }
                } else if iter == 0 {
                    vec![(0, *k)]
                } else if iter == 1 {
                    vec![(*k, n)]
                } else {
                    vec![(0, n)]
                }
            }
            SplitKind::Phase3 { k, .. } => {
                let iter = self.iter;
                let n = self.n;
                if n != 3 {
                    if iter % 2 == 0 {
                        // "mid" window straddling k.
                        // left_child->k is the pivot of the left child (= left.k).
                        let left_k = match &self.kind {
                            SplitKind::Phase3 { left: Some(l), .. } => match &l.borrow().kind {
                                SplitKind::Phase3 { k, .. } => *k,
                                _ => unreachable!(),
                            },
                            _ => 0,
                        };
                        let right_k = match &self.kind {
                            SplitKind::Phase3 { right: Some(r), .. } => match &r.borrow().kind {
                                SplitKind::Phase3 { k, .. } => *k,
                                _ => unreachable!(),
                            },
                            _ => 0,
                        };
                        vec![(left_k, *k + right_k)]
                    } else {
                        vec![(0, *k), (*k, n)]
                    }
                } else {
                    // n = 3 special case.
                    if iter % 2 == 0 {
                        vec![(0, 2)]
                    } else {
                        vec![(1, 3)]
                    }
                }
            }
        }
    }

    pub fn advance_sublattices(&mut self) {
        match &self.kind {
            SplitKind::Phase2 { .. } => {}
            SplitKind::Phase3 { left, right, .. } => {
                // Phase-3 resets its children every tick — they're
                // visited via `get_child_split` and their local state
                // would otherwise stale.
                if let Some(l) = left {
                    l.borrow_mut().reset();
                }
                if let Some(r) = right {
                    r.borrow_mut().reset();
                }
            }
        }
        self.iter += 1;
    }

    pub fn reset(&mut self) {
        self.iter = 0;
        match &mut self.kind {
            SplitKind::Phase2 { left, right, all, .. } => {
                if let Some(l) = left {
                    l.borrow_mut().reset();
                }
                if let Some(r) = right {
                    r.borrow_mut().reset();
                }
                all.borrow_mut().reset();
            }
            SplitKind::Phase3 { left, right, mid, .. } => {
                if let Some(l) = left {
                    l.borrow_mut().reset();
                }
                if let Some(r) = right {
                    r.borrow_mut().reset();
                }
                if let Some(m) = mid {
                    m.borrow_mut().reset();
                }
            }
        }
    }

    /// Which child handles `get_sublattices()[i]`. Matches
    /// `SubSplitPhase2::get_child_split` (sublattice_split_2.cpp:76)
    /// and `SubSplitPhase3::get_child_split` (sublattice_split_3.cpp:128).
    pub fn get_child_split(&self, i: usize) -> SplitRc {
        match &self.kind {
            SplitKind::Phase2 { left, right, all, .. } => {
                assert_eq!(i, 0);
                let n = self.n;
                let iter = self.iter;
                if n == 3 {
                    if iter == 0 {
                        Rc::clone(left.as_ref().unwrap())
                    } else {
                        Rc::clone(all)
                    }
                } else if iter == 0 {
                    Rc::clone(left.as_ref().unwrap())
                } else if iter == 1 {
                    Rc::clone(right.as_ref().unwrap())
                } else {
                    Rc::clone(all)
                }
            }
            SplitKind::Phase3 { k, left, right, mid } => {
                let iter = self.iter;
                if self.n == 3 {
                    assert_eq!(i, 0);
                    if *k == 2 {
                        if iter % 2 == 0 {
                            Rc::clone(left.as_ref().unwrap())
                        } else {
                            Rc::clone(mid.as_ref().unwrap())
                        }
                    } else if iter % 2 == 0 {
                        Rc::clone(mid.as_ref().unwrap())
                    } else {
                        Rc::clone(right.as_ref().unwrap())
                    }
                } else {
                    assert!(i < 2);
                    if iter % 2 == 0 {
                        Rc::clone(mid.as_ref().unwrap())
                    } else if i == 0 {
                        Rc::clone(left.as_ref().unwrap())
                    } else {
                        Rc::clone(right.as_ref().unwrap())
                    }
                }
            }
        }
    }

    pub fn stopping_point(&self) -> bool {
        match &self.kind {
            SplitKind::Phase2 { .. } => {
                let iter = self.iter;
                if self.n == 3 {
                    iter > 1
                } else {
                    iter > 2
                }
            }
            SplitKind::Phase3 { .. } => self.iter % 2 == 0,
        }
    }
}

/// `SubSplitPhase2::next_smaller` (sublattice_split_2.cpp:111).
fn next_smaller(n: usize) -> usize {
    assert!(n >= 2);
    if n == 3 {
        2
    } else {
        n / 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Walk the Phase-2 state machine and record the `(start, end)`
    /// pairs produced at the top level. These must match the C++
    /// sequence: left → right → all.
    #[test]
    fn phase2_top_level_sequence() {
        for n in [4usize, 8, 16, 32, 50] {
            let split = Split::new_phase2(n);
            let k = n / 2;
            let mut got = Vec::new();
            for _ in 0..3 {
                let s = split.borrow().get_sublattices();
                got.push(s);
                split.borrow_mut().advance_sublattices();
            }
            assert_eq!(
                got,
                vec![vec![(0, k)], vec![(k, n)], vec![(0, n)]],
                "phase2 n={} sequence wrong",
                n
            );
            assert!(split.borrow().stopping_point());
        }
    }

    #[test]
    fn phase2_n3_two_iters() {
        let split = Split::new_phase2(3);
        assert_eq!(split.borrow().get_sublattices(), vec![(0, 2)]);
        split.borrow_mut().advance_sublattices();
        assert_eq!(split.borrow().get_sublattices(), vec![(0, 3)]);
        split.borrow_mut().advance_sublattices();
        assert!(split.borrow().stopping_point());
    }

    #[test]
    fn phase3_alternates() {
        let split = Split::new_phase3(8);
        // n=8, k=4; left.k = 2, right.k = 2. iter=0: mid window is [2, 6).
        let s0 = split.borrow().get_sublattices();
        assert_eq!(s0, vec![(2, 6)]);
        split.borrow_mut().advance_sublattices();
        let s1 = split.borrow().get_sublattices();
        assert_eq!(s1, vec![(0, 4), (4, 8)]);
    }

    /// Descending through `get_child_split` should preserve invariants
    /// on window size.
    #[test]
    fn child_split_size_matches_sublattice() {
        let split = Split::new_phase2(16);
        for _ in 0..3 {
            let sl = split.borrow().get_sublattices();
            for (i, (a, b)) in sl.iter().enumerate() {
                let child = split.borrow().get_child_split(i);
                assert_eq!(
                    child.borrow().n,
                    b - a,
                    "child.n must equal window size"
                );
            }
            split.borrow_mut().advance_sublattices();
        }
    }
}
