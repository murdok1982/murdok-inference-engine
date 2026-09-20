#include "murdok/hardware.h"
#include <iostream>

int main(int argc, char* argv[]) {
    std::cout << "==========================================================\n";
    std::cout << "         MuRDoK Inference Engine (MIE) v0.1.0             \n";
    std::cout << "    Move less. Compute smarter. Infer faster. Benchmark.  \n";
    std::cout << "==========================================================\n";

    auto hw = murdok::HardwareDetector::detect();
    std::cout << "Target Hardware: " << hw.cpu_brand << "\n";
    std::cout << "Profile:         " << murdok::HardwareDetector::profile_to_string(hw.recommended_profile) << "\n";
    std::cout << "Status:          Milestone M0 (Baseline & Hardware Engine Active)\n\n";

    if (argc < 2) {
        std::cout << "Usage:\n";
        std::cout << "  murdok-hardware         Inspect system CPU, SIMD, cache, memory\n";
        std::cout << "  murdok-bench --model <model.gguf> [options] Run benchmark\n";
        return 0;
    }

    std::cout << "Running with argument: " << argv[1] << "\n";
    return 0;
}
