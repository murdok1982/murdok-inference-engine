#include "murdok/hardware.h"
#include <iostream>
#include <iomanip>

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;

    auto hw = murdok::HardwareDetector::detect();

    std::cout << "========================================\n";
    std::cout << "        MuRDoK Hardware Analyzer        \n";
    std::cout << "========================================\n\n";

    std::cout << "CPU:\n";
    std::cout << "  Model:    " << hw.cpu_brand << "\n";
    std::cout << "  Vendor:   " << hw.cpu_vendor << "\n";
    std::cout << "  Physical: " << hw.physical_cores << " cores\n";
    std::cout << "  Logical:  " << hw.logical_threads << " threads\n\n";

    std::cout << "SIMD Extensions:\n";
    std::cout << "  SSE4.2:   " << (hw.simd.sse42 ? "[YES]" : "[NO]") << "\n";
    std::cout << "  FMA:      " << (hw.simd.fma   ? "[YES]" : "[NO]") << "\n";
    std::cout << "  AVX:      " << (hw.simd.avx   ? "[YES]" : "[NO]") << "\n";
    std::cout << "  AVX2:     " << (hw.simd.avx2  ? "[YES]" : "[NO]") << "\n";
    std::cout << "  AVX-512F: " << (hw.simd.avx512f ? "[YES]" : "[NO]") << "\n";
    std::cout << "  AVX-512BW:" << (hw.simd.avx512bw ? "[YES]" : "[NO]") << "\n";
    std::cout << "  VNNI:     " << (hw.simd.avx512_vnni ? "[YES]" : "[NO]") << "\n";
    std::cout << "  AMX:      " << ((hw.simd.amx_tile || hw.simd.amx_int8) ? "[YES]" : "[NO]") << "\n\n";

    std::cout << "Cache Hierarchy:\n";
    std::cout << "  L1 Data:        " << hw.cache.l1d_kb << " KB\n";
    std::cout << "  L1 Instruction: " << hw.cache.l1i_kb << " KB\n";
    std::cout << "  L2 Cache:       " << hw.cache.l2_kb << " KB\n";
    std::cout << "  L3 Cache:       " << hw.cache.l3_kb << " KB\n";
    std::cout << "  Cache Line:     " << hw.cache.cache_line_size << " bytes\n\n";

    double total_ram_gb = static_cast<double>(hw.memory.total_ram_bytes) / (1024.0 * 1024.0 * 1024.0);
    double avail_ram_gb = static_cast<double>(hw.memory.available_ram_bytes) / (1024.0 * 1024.0 * 1024.0);

    std::cout << "System Memory:\n";
    std::cout << "  Total RAM:     " << std::fixed << std::setprecision(2) << total_ram_gb << " GB\n";
    std::cout << "  Available RAM: " << std::fixed << std::setprecision(2) << avail_ram_gb << " GB\n\n";

    std::cout << "----------------------------------------\n";
    std::cout << "Recommended Profile:\n";
    std::cout << "  " << murdok::HardwareDetector::profile_to_string(hw.recommended_profile) << "\n";
    std::cout << "========================================\n";

    return 0;
}
