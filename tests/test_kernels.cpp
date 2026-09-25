#include "murdok/kernels.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>

void test_kernel_tail_handling() {
    std::cout << "[TEST] Running SIMD Kernel Numerical & Tail Tests...\n";

    std::vector<size_t> test_sizes = {0, 1, 2, 7, 8, 9, 15, 16, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1024, 65536};

    for (size_t n : test_sizes) {
        std::vector<float> a(n);
        std::vector<float> b(n);
        for (size_t i = 0; i < n; ++i) {
            a[i] = static_cast<float>(i % 17) - 8.5f;
            b[i] = static_cast<float>((i * 3) % 19) * 0.25f;
        }

        float scalar_res = murdok::kernels::vec_dot_f32_scalar(a.data(), b.data(), n);
        float avx2_res = murdok::kernels::vec_dot_f32_avx2(a.data(), b.data(), n);
        float dispatch_res = murdok::kernels::dot_product(a.data(), b.data(), n);

        float diff1 = std::abs(scalar_res - avx2_res);
        float diff2 = std::abs(scalar_res - dispatch_res);

        float tol = std::max(1e-4f, std::abs(scalar_res) * 1e-4f);
        if (diff1 > tol || diff2 > tol) {
            std::cerr << "Mismatch at N=" << n << ": scalar=" << scalar_res
                      << ", avx2=" << avx2_res << ", dispatch=" << dispatch_res << "\n";
            assert(false);
        }
    }

    std::cout << "  Verified tail handling across " << test_sizes.size() << " vector sizes.\n";

    // Test axpy
    std::vector<float> dst(100, 1.0f);
    std::vector<float> src(100, 2.0f);
    murdok::kernels::vec_axpy_f32(dst.data(), src.data(), 0.5f, 100);
    for (size_t i = 0; i < 100; ++i) {
        assert(std::abs(dst[i] - 2.0f) < 1e-5f);
    }
    std::cout << "  Verified vec_axpy_f32 vector scaling.\n";

    std::cout << "[PASS] SIMD Kernel Tests Passed.\n\n";
}
