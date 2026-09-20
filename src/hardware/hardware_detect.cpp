#include "murdok/hardware.h"

#include <iostream>
#include <cstring>
#include <thread>
#include <algorithm>

#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace murdok {

static void run_cpuid(int leaf, int subleaf, int regs[4]) {
#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
    __cpuidex(regs, leaf, subleaf);
#elif defined(__GNUC__) || defined(__clang__)
    __cpuid_count(leaf, subleaf, regs[0], regs[1], regs[2], regs[3]);
#else
    regs[0] = regs[1] = regs[2] = regs[3] = 0;
#endif
}

const char* HardwareDetector::profile_to_string(ExecutionProfile profile) {
    switch (profile) {
        case ExecutionProfile::CpuBalanced: return "MURDOK_BALANCED";
        case ExecutionProfile::CpuPerformance: return "CPU_PERFORMANCE";
        case ExecutionProfile::CpuLowMemory: return "CPU_LOW_MEMORY";
        case ExecutionProfile::GpuOffload: return "GPU_OFFLOAD";
        case ExecutionProfile::Hybrid: return "HYBRID";
        default: return "AUTO";
    }
}

HardwareInfo HardwareDetector::detect() {
    HardwareInfo info;

    // Logical thread count
    info.logical_threads = std::thread::hardware_concurrency();
    if (info.logical_threads == 0) info.logical_threads = 1;

    int regs[4] = {0};

    // CPUID Leaf 0: Vendor String and Max Leaf
    run_cpuid(0, 0, regs);
    int max_leaf = regs[0];
    char vendor[13] = {0};
    *reinterpret_cast<int*>(vendor + 0) = regs[1];
    *reinterpret_cast<int*>(vendor + 4) = regs[3];
    *reinterpret_cast<int*>(vendor + 8) = regs[2];
    info.cpu_vendor = vendor;

    // CPUID Leaf 1: Basic features (SSE4.2, AVX, FMA)
    if (max_leaf >= 1) {
        run_cpuid(1, 0, regs);
        int ecx = regs[2];
        int edx = regs[3];

        info.simd.sse42 = (ecx & (1 << 20)) != 0;
        info.simd.fma   = (ecx & (1 << 12)) != 0;
        info.simd.avx   = (ecx & (1 << 28)) != 0;
    }

    // CPUID Leaf 7 Subleaf 0: Extended features (AVX2, AVX-512, AMX)
    if (max_leaf >= 7) {
        run_cpuid(7, 0, regs);
        int ebx = regs[1];
        int ecx = regs[2];
        int edx = regs[3];

        info.simd.avx2       = (ebx & (1 << 5)) != 0;
        info.simd.avx512f    = (ebx & (1 << 16)) != 0;
        info.simd.avx512dq   = (ebx & (1 << 17)) != 0;
        info.simd.avx512cd   = (ebx & (1 << 28)) != 0;
        info.simd.avx512bw   = (ebx & (1 << 30)) != 0;
        info.simd.avx512vl   = (ebx & (1 << 31)) != 0;

        info.simd.avx512_vnni = (ecx & (1 << 11)) != 0;

        info.simd.amx_bf16 = (edx & (1 << 22)) != 0;
        info.simd.amx_tile = (edx & (1 << 24)) != 0;
        info.simd.amx_int8 = (edx & (1 << 25)) != 0;
    }

    // CPU Brand String: Extended Leaves 0x80000002 to 0x80000004
    run_cpuid(0x80000000, 0, regs);
    unsigned int max_ext_leaf = static_cast<unsigned int>(regs[0]);
    if (max_ext_leaf >= 0x80000004) {
        char brand[49] = {0};
        for (int i = 0; i < 3; ++i) {
            run_cpuid(0x80000002 + i, 0, regs);
            std::memcpy(brand + (i * 16), regs, 16);
        }
        // Trim leading spaces
        char* start = brand;
        while (*start == ' ') start++;
        info.cpu_brand = start;
    } else {
        info.cpu_brand = "Unknown x86/x64 Processor";
    }

#if defined(_WIN32)
    // Physical cores & cache detection on Windows
    DWORD buffer_size = 0;
    GetLogicalProcessorInformationEx(RelationAll, nullptr, &buffer_size);
    if (buffer_size > 0) {
        std::vector<uint8_t> buffer(buffer_size);
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info_ptr =
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());

        if (GetLogicalProcessorInformationEx(RelationAll, info_ptr, &buffer_size)) {
            uint32_t physical_core_count = 0;
            DWORD offset = 0;
            while (offset < buffer_size) {
                PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX curr =
                    reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data() + offset);

                if (curr->Relationship == RelationProcessorCore) {
                    physical_core_count++;
                } else if (curr->Relationship == RelationCache) {
                    CACHE_RELATIONSHIP& cache = curr->Cache;
                    uint32_t size_kb = cache.CacheSize / 1024;
                    if (cache.Level == 1) {
                        if (cache.Type == CacheData) info.cache.l1d_kb = size_kb;
                        else if (cache.Type == CacheInstruction) info.cache.l1i_kb = size_kb;
                    } else if (cache.Level == 2) {
                        info.cache.l2_kb = size_kb;
                    } else if (cache.Level == 3) {
                        info.cache.l3_kb = size_kb;
                    }
                    if (cache.LineSize > 0) {
                        info.cache.cache_line_size = cache.LineSize;
                    }
                }
                offset += curr->Size;
            }
            info.physical_cores = physical_core_count;
        }
    }

    // System Memory
    MEMORYSTATUSEX mem_status;
    mem_status.dwLength = sizeof(mem_status);
    if (GlobalMemoryStatusEx(&mem_status)) {
        info.memory.total_ram_bytes = mem_status.ullTotalPhys;
        info.memory.available_ram_bytes = mem_status.ullAvailPhys;
    }
#else
    info.physical_cores = info.logical_threads / 2;
    if (info.physical_cores == 0) info.physical_cores = 1;
#endif

    if (info.physical_cores == 0) {
        info.physical_cores = (info.logical_threads > 1) ? (info.logical_threads / 2) : 1;
    }

    // Determine Recommended Execution Profile
    double total_ram_gb = static_cast<double>(info.memory.total_ram_bytes) / (1024.0 * 1024.0 * 1024.0);
    if (total_ram_gb < 8.0) {
        info.recommended_profile = ExecutionProfile::CpuLowMemory;
    } else if (info.simd.avx2 && info.physical_cores >= 4) {
        info.recommended_profile = ExecutionProfile::CpuBalanced;
    } else {
        info.recommended_profile = ExecutionProfile::CpuBalanced;
    }

    return info;
}

} // namespace murdok
