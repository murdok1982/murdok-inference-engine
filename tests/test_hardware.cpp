#include "murdok/hardware.h"
#include <cassert>
#include <iostream>

void test_hardware_detection() {
    std::cout << "[TEST] Running Hardware Detection Test...\n";
    auto hw = murdok::HardwareDetector::detect();

    assert(!hw.cpu_brand.empty());
    assert(!hw.cpu_vendor.empty());
    assert(hw.physical_cores > 0);
    assert(hw.logical_threads >= hw.physical_cores);
    assert(hw.cache.cache_line_size > 0);
    assert(hw.memory.total_ram_bytes > 0);

    std::string prof = murdok::HardwareDetector::profile_to_string(hw.recommended_profile);
    assert(!prof.empty());

    std::cout << "  CPU: " << hw.cpu_brand << "\n";
    std::cout << "  Physical Cores: " << hw.physical_cores << ", Logical: " << hw.logical_threads << "\n";
    std::cout << "  Profile: " << prof << "\n";
    std::cout << "[PASS] Hardware Detection Test Passed.\n\n";
}
