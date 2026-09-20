#include "murdok/kernels.h"

#include <immintrin.h>
#include <chrono>
#include <vector>
#include <cmath>

namespace murdok {
namespace kernels {

float vec_dot_f32_scalar(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

float vec_dot_f32_avx2(const float* a, const float* b, size_t n) {
    // 4 accumulators unrolled (32 floats = 128 bytes per loop iteration)
    // Saturates Skylake / Kaby Lake FMA pipeline with 2 parallel ports
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 31 < n; i += 32) {
        __m256 a0 = _mm256_loadu_ps(a + i + 0);
        __m256 b0 = _mm256_loadu_ps(b + i + 0);
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);

        __m256 a1 = _mm256_loadu_ps(a + i + 8);
        __m256 b1 = _mm256_loadu_ps(b + i + 8);
        acc1 = _mm256_fmadd_ps(a1, b1, acc1);

        __m256 a2 = _mm256_loadu_ps(a + i + 16);
        __m256 b2 = _mm256_loadu_ps(b + i + 16);
        acc2 = _mm256_fmadd_ps(a2, b2, acc2);

        __m256 a3 = _mm256_loadu_ps(a + i + 24);
        __m256 b3 = _mm256_loadu_ps(b + i + 24);
        acc3 = _mm256_fmadd_ps(a3, b3, acc3);
    }

    // Combine accumulators
    __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));

    // Handle remaining 8-wide blocks
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }

    // Horizontal sum of __m256
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float total = _mm_cvtss_f32(sum128);

    // Remaining tail elements
    for (; i < n; ++i) {
        total += a[i] * b[i];
    }

    return total;
}

void vec_axpy_f32_avx2(float* dst, const float* src, float alpha, size_t n) {
    __m256 valpha = _mm256_set1_ps(alpha);
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 vd = _mm256_loadu_ps(dst + i);
        __m256 vs = _mm256_loadu_ps(src + i);
        vd = _mm256_fmadd_ps(vs, valpha, vd);
        _mm256_storeu_ps(dst + i, vd);
    }
    for (; i < n; ++i) {
        dst[i] += alpha * src[i];
    }
}

KernelBenchmarkResult benchmark_simd_kernels(size_t vector_size, int iterations) {
    KernelBenchmarkResult res;

    std::vector<float> a(vector_size, 1.0f);
    std::vector<float> b(vector_size, 2.0f);

    // 1. Scalar benchmark
    auto t0 = std::chrono::high_resolution_clock::now();
    float s_sum = 0.0f;
    for (int it = 0; it < iterations; ++it) {
        s_sum += vec_dot_f32_scalar(a.data(), b.data(), vector_size);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double s_sec = std::chrono::duration<double>(t1 - t0).count();
    double total_flops = 2.0 * static_cast<double>(vector_size) * iterations;
    res.scalar_gflops = (total_flops / s_sec) / 1e9;

    // 2. AVX2 benchmark
    auto t2 = std::chrono::high_resolution_clock::now();
    float v_sum = 0.0f;
    for (int it = 0; it < iterations; ++it) {
        v_sum += vec_dot_f32_avx2(a.data(), b.data(), vector_size);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double v_sec = std::chrono::duration<double>(t3 - t2).count();
    res.avx2_gflops = (total_flops / v_sec) / 1e9;

    res.speedup = (v_sec > 0.0) ? (s_sec / v_sec) : 1.0;
    res.avx2_verified = (std::abs(s_sum - v_sum) < 1e-2);

    return res;
}

} // namespace kernels
} // namespace murdok
