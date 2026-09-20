#pragma once

#include <cstdint>
#include <cstddef>

namespace murdok {
namespace kernels {

// Cache line alignment constant
constexpr size_t CACHE_LINE_BYTES = 64;

// Check if pointer is aligned to CPU cache line
inline bool is_cache_aligned(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) % CACHE_LINE_BYTES) == 0;
}

// Custom AVX2 + FMA Vector Dot Product: y = dot(a, b) for float arrays
float vec_dot_f32_avx2(const float* a, const float* b, size_t n);

// Custom Scalar Reference Dot Product (for benchmark comparison)
float vec_dot_f32_scalar(const float* a, const float* b, size_t n);

// Vectorized Scale and Accumulate: dst[i] += alpha * src[i]
void vec_axpy_f32_avx2(float* dst, const float* src, float alpha, size_t n);

// Micro-benchmark measuring throughput (GB/s and GFLOPs) of AVX2 vs Scalar
struct KernelBenchmarkResult {
    double scalar_gflops = 0.0;
    double avx2_gflops = 0.0;
    double speedup = 0.0;
    bool avx2_verified = false;
};

KernelBenchmarkResult benchmark_simd_kernels(size_t vector_size = 65536, int iterations = 1000);

} // namespace kernels
} // namespace murdok
