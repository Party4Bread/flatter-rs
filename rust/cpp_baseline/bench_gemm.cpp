// C++ baseline for the gemm port. The `gemm_xx` function is copied
// verbatim from `src/problems/matrix_multiplication/elementary_native.cpp`
// (non-transposed case, adr=k, adc=1, bdr=n, bdc=1, ldc=n), then driven by
// a simple timing harness that matches the Rust bench.
//
// Build:  g++ -O3 -march=native -std=c++17 bench_gemm.cpp -o bench_gemm_cpp
//
// Also provides a naive auto-vectorization-friendly ikj variant to give the
// C++ compiler every chance it has; we report both against Rust.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

// ----- Exact port of flatter's elementary_native.cpp gemm_xx (ijk order). --
template <class T, class U, class V>
static void gemm_xx_flatter(
    T* C, const U* A, const V* B,
    unsigned m, unsigned n, unsigned k,
    bool accumulate_C)
{
    unsigned adr = k, adc = 1, bdr = n, bdc = 1, ldc = n;
    T prod, sum;
    for (unsigned i = 0; i < m; ++i) {
        for (unsigned j = 0; j < n; ++j) {
            sum = 0;
            for (unsigned l = 0; l < k; ++l) {
                prod = A[i*adr + l*adc] * B[l*bdr + j*bdc];
                sum += prod;
            }
            if (accumulate_C) C[i*ldc + j] += sum;
            else              C[i*ldc + j]  = sum;
        }
    }
}

// ----- ikj reordering: often autovectorizes better than the flatter form.
template <class T, class U, class V>
static void gemm_ikj(
    T* C, const U* A, const V* B,
    unsigned m, unsigned n, unsigned k,
    bool accumulate_C)
{
    if (!accumulate_C) {
        for (unsigned i = 0; i < m * n; ++i) C[i] = 0;
    }
    for (unsigned i = 0; i < m; ++i) {
        for (unsigned l = 0; l < k; ++l) {
            U a_il = A[i*k + l];
            const V* b_row = B + l*n;
            T* c_row = C + i*n;
            for (unsigned j = 0; j < n; ++j) {
                c_row[j] += a_il * b_row[j];
            }
        }
    }
}

// -- timing helper -----------------------------------------------------------
template <class F>
static double best_of(F f, int iters) {
    f(); // warm
    double best = 1e300;
    for (int i = 0; i < iters; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        f();
        auto t1 = std::chrono::steady_clock::now();
        double s = std::chrono::duration<double>(t1 - t0).count();
        if (s < best) best = s;
    }
    return best;
}

static void row(const char* name, double ops, double secs) {
    double g = ops / secs / 1e9;
    std::printf("%-32s %10.3f ms   %10.3f G/s\n", name, secs * 1e3, g);
}

template <class T>
static std::vector<T> make_vec(size_t n, double sign) {
    std::vector<T> v(n);
    for (size_t i = 0; i < n; ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            v[i] = std::sin(double(i) * sign);
        } else {
            v[i] = T(i) * T(sign > 0 ? 31 : 17) ^ T(sign > 0 ? 0x5a5a : 0xa5a5);
        }
    }
    return v;
}

static void bench_f64(unsigned m, unsigned n, unsigned k) {
    std::printf("\n== f64 gemm  m=%u n=%u k=%u  (%.1f Mflop) ==\n",
                m, n, k, (2.0 * m * n * k) / 1e6);
    auto A = make_vec<double>(size_t(m)*k,  0.017);
    auto B = make_vec<double>(size_t(k)*n, -0.031);
    std::vector<double> C(size_t(m)*n, 0.0);
    double ops = 2.0 * m * n * k;

    if (size_t(m)*n*k <= 512ull*512*64) {
        double s = best_of([&]{
            gemm_xx_flatter<double,double,double>(C.data(), A.data(), B.data(), m, n, k, false);
        }, 5);
        row("C++ flatter ijk (1 thread)", ops, s);
    } else {
        std::printf("%-32s %10s      %10s\n",
                    "C++ flatter ijk (1 thread)", "skipped", "too slow");
    }
    {
        double s = best_of([&]{
            gemm_ikj<double,double,double>(C.data(), A.data(), B.data(), m, n, k, false);
        }, 5);
        row("C++ ikj autovec (1 thread)", ops, s);
    }
}

static void bench_i64(unsigned m, unsigned n, unsigned k) {
    std::printf("\n== i64 gemm  m=%u n=%u k=%u  (%.1f Miop) ==\n",
                m, n, k, (2.0 * m * n * k) / 1e6);
    auto A = make_vec<int64_t>(size_t(m)*k,  1.0);
    auto B = make_vec<int64_t>(size_t(k)*n, -1.0);
    std::vector<int64_t> C(size_t(m)*n, 0);
    double ops = 2.0 * m * n * k;

    if (size_t(m)*n*k <= 512ull*512*64) {
        double s = best_of([&]{
            gemm_xx_flatter<int64_t,int64_t,int64_t>(C.data(), A.data(), B.data(), m, n, k, false);
        }, 5);
        row("C++ flatter ijk (1 thread)", ops, s);
    } else {
        std::printf("%-32s %10s      %10s\n",
                    "C++ flatter ijk (1 thread)", "skipped", "too slow");
    }
    {
        double s = best_of([&]{
            gemm_ikj<int64_t,int64_t,int64_t>(C.data(), A.data(), B.data(), m, n, k, false);
        }, 5);
        row("C++ ikj autovec (1 thread)", ops, s);
    }
}

int main() {
    std::printf("flatter C++ baseline gemm bench\n");
    std::printf("%-32s %10s        %10s\n", "kernel", "best ms", "GFLOP/GIOP s");
    for (auto [m,n,k] : {std::tuple<unsigned,unsigned,unsigned>{128,128,128},
                         {256,256,256}, {512,512,512}}) bench_f64(m,n,k);
    for (auto [m,n,k] : {std::tuple<unsigned,unsigned,unsigned>{128,128,128},
                         {256,256,256}, {512,512,512}}) bench_i64(m,n,k);
    return 0;
}
