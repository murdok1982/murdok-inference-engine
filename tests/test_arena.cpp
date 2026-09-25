#include "murdok/static_graph.h"
#include <cassert>
#include <iostream>
#include <cstdint>

void test_static_arena() {
    std::cout << "[TEST] Running Static Memory Arena Tests...\n";

    // 1. Allocation & 64-byte alignment
    murdok::graph::StaticMemoryArena arena(4096, 64);
    assert(arena.is_valid());
    assert(arena.capacity() == 4096);
    assert(arena.alignment() == 64);

    void* ptr0 = arena.get_pointer(0);
    assert(ptr0 != nullptr);
    assert((reinterpret_cast<uintptr_t>(ptr0) % 64) == 0);

    void* ptr64 = arena.get_pointer(64);
    assert(ptr64 != nullptr);
    assert((reinterpret_cast<uintptr_t>(ptr64) % 64) == 0);

    // 2. Out of bounds check
    void* oob = arena.get_pointer(4096);
    assert(oob == nullptr);

    // 3. Write & Reset
    int* data = static_cast<int*>(ptr0);
    data[0] = 42;
    assert(data[0] == 42);

    arena.reset();
    assert(data[0] == 0);

    // 4. Move semantics
    murdok::graph::StaticMemoryArena moved = std::move(arena);
    assert(moved.is_valid());
    assert(!arena.is_valid());
    assert(moved.get_pointer(0) != nullptr);

    // 5. Graph Planner verification
    auto planner = murdok::graph::StaticGraphPlanner::create_for_transformer(24, 896, 512, 14, 151936);
    const auto& metrics = planner.get_metrics();
    assert(metrics.total_nodes > 0);
    assert(metrics.optimized_arena_size < metrics.unoptimized_peak_memory);
    assert(metrics.memory_reduction_pct > 90.0);

    std::cout << "  Arena Capacity: " << moved.capacity() << " bytes (64-byte aligned)\n";
    std::cout << "  Graph Planner nodes: " << metrics.total_nodes << ", memory reduction: "
              << metrics.memory_reduction_pct << " %\n";
    std::cout << "[PASS] Static Memory Arena Tests Passed.\n\n";
}
