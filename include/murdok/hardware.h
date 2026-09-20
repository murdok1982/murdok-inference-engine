#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace murdok {

enum class ExecutionProfile {
    Auto,
    CpuBalanced,
    CpuPerformance,
    CpuLowMemory,
    GpuOffload,
    Hybrid
};

struct CacheInfo {
    uint32_t l1d_kb = 0;
    uint32_t l1i_kb = 0;
    uint32_t l2_kb = 0;
    uint32_t l3_kb = 0;
    uint32_t cache_line_size = 64; // Default standard cache line
};

struct SimdCapabilities {
    bool sse42 = false;
    bool avx = false;
    bool avx2 = false;
    bool fma = false;
    bool avx512f = false;
    bool avx512dq = false;
    bool avx512cd = false;
    bool avx512bw = false;
    bool avx512vl = false;
    bool avx512_vnni = false;
    bool amx_tile = false;
    bool amx_int8 = false;
    bool amx_bf16 = false;
    bool neon = false;
};

struct MemoryInfo {
    uint64_t total_ram_bytes = 0;
    uint64_t available_ram_bytes = 0;
};

struct HardwareInfo {
    std::string cpu_brand;
    std::string cpu_vendor;
    uint32_t physical_cores = 0;
    uint32_t logical_threads = 0;
    SimdCapabilities simd;
    CacheInfo cache;
    MemoryInfo memory;
    ExecutionProfile recommended_profile = ExecutionProfile::CpuBalanced;
};

class HardwareDetector {
public:
    static HardwareInfo detect();
    static const char* profile_to_string(ExecutionProfile profile);
};

} // namespace murdok
