//! Bench harness for the ported gemm kernels.
//!
//! Runs each kernel on the same input sizes and prints a table of wall time
//! and GFLOP/s (or GIOP/s for integers). Sizes are chosen to match the
//! regime where flatter's elementary kernel actually lives: smallish (the
//! recursive base cases) up through medium (pre-Strassen crossover).

use flatter_rs::*;
use std::hint::black_box;
use std::time::{Duration, Instant};

fn time_once<F: FnMut()>(mut f: F) -> Duration {
    let t0 = Instant::now();
    f();
    t0.elapsed()
}

fn run<F: FnMut()>(name: &str, flops: f64, mut f: F) {
    // Warm up once, then take min over a few iterations — stable on short runs.
    f();
    let mut best = Duration::from_secs(u64::MAX / 2);
    let iters = 5;
    for _ in 0..iters {
        let d = time_once(&mut f);
        if d < best {
            best = d;
        }
    }
    let secs = best.as_secs_f64();
    let gflops = flops / secs / 1e9;
    println!("{:<32} {:>10.3} ms   {:>10.3} G/s", name, secs * 1e3, gflops);
}

fn bench_f64(m: usize, n: usize, k: usize) {
    println!("\n== f64 gemm  m={} n={} k={}  ({:.1} Mflop) ==",
             m, n, k, (2.0 * m as f64 * n as f64 * k as f64) / 1e6);

    let a: Vec<f64> = (0..m * k).map(|i| ((i as f64) * 0.017).sin()).collect();
    let b: Vec<f64> = (0..k * n).map(|i| ((i as f64) * 0.031).cos()).collect();
    let flops = 2.0 * m as f64 * n as f64 * k as f64;

    // Naive is O(mnk) with poor locality — cap it to small sizes, else skip.
    if m * n * k <= 512 * 512 * 64 {
        let mut c = vec![0.0f64; m * n];
        run("naive (C++-port triple loop)", flops, || {
            gemm_f64_naive(black_box(&a), black_box(&b), &mut c, m, n, k, false);
        });
    } else {
        println!("{:<32} {:>10}      {:>10}", "naive (C++-port triple loop)", "skipped", "too slow");
    }

    let mut c = vec![0.0f64; m * n];
    run("blocked  (1 thread)", flops, || {
        gemm_f64_blocked(black_box(&a), black_box(&b), &mut c, m, n, k, false);
    });

    let mut c = vec![0.0f64; m * n];
    run("blocked  + rayon",    flops, || {
        gemm_f64_blocked_par(black_box(&a), black_box(&b), &mut c, m, n, k, false);
    });

    #[cfg(target_arch = "x86_64")]
    {
        let mut c = vec![0.0f64; m * n];
        run("avx2+fma (1 thread)", flops, || {
            gemm_f64_avx2(black_box(&a), black_box(&b), &mut c, m, n, k, false);
        });

        let mut c = vec![0.0f64; m * n];
        run("avx2+fma + rayon",    flops, || {
            gemm_f64_avx2_par(black_box(&a), black_box(&b), &mut c, m, n, k, false);
        });
    }
}

fn bench_i64(m: usize, n: usize, k: usize) {
    println!("\n== i64 gemm  m={} n={} k={}  ({:.1} Miop) ==",
             m, n, k, (2.0 * m as f64 * n as f64 * k as f64) / 1e6);

    let a: Vec<i64> = (0..m * k).map(|i| (i as i64).wrapping_mul(31) ^ 0x5a5a).collect();
    let b: Vec<i64> = (0..k * n).map(|i| (i as i64).wrapping_mul(17) ^ 0xa5a5).collect();
    let ops = 2.0 * m as f64 * n as f64 * k as f64;

    if m * n * k <= 512 * 512 * 64 {
        let mut c = vec![0i64; m * n];
        run("naive (C++-port triple loop)", ops, || {
            gemm_i64_naive(black_box(&a), black_box(&b), &mut c, m, n, k, false);
        });
    } else {
        println!("{:<32} {:>10}      {:>10}", "naive (C++-port triple loop)", "skipped", "too slow");
    }

    let mut c = vec![0i64; m * n];
    run("blocked  (1 thread)", ops, || {
        gemm_i64_blocked(black_box(&a), black_box(&b), &mut c, m, n, k, false);
    });

    let mut c = vec![0i64; m * n];
    run("blocked  + rayon",    ops, || {
        gemm_i64_blocked_par(black_box(&a), black_box(&b), &mut c, m, n, k, false);
    });
}

fn main() {
    let threads = rayon::current_num_threads();
    println!("flatter-rs gemm bench   (rayon threads = {})", threads);
    println!("{:<32} {:>10}        {:>10}", "kernel", "best ms", "GFLOP/GIOP s");
    println!("{}", "-".repeat(66));

    for &(m, n, k) in &[
        (128usize, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
    ] {
        bench_f64(m, n, k);
    }
    for &(m, n, k) in &[
        (128usize, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
    ] {
        bench_i64(m, n, k);
    }
}
