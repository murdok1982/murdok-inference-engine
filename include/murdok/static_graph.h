#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>

namespace murdok {
namespace graph {

struct TensorNode {
    std::string name;
    size_t size_bytes = 0;
    int32_t first_step = 0;
    int32_t last_step = 0;
    size_t arena_offset = 0;
};

struct GraphPlanMetrics {
    size_t unoptimized_peak_memory = 0;
    size_t optimized_arena_size = 0;
    double memory_reduction_pct = 0.0;
    int32_t total_nodes = 0;
    int32_t reusable_buffers = 0;
    int32_t zero_alloc_steps = 0;
    double speedup_factor = 1.0;
};

class StaticMemoryArena {
public:
    explicit StaticMemoryArena(size_t capacity_bytes, size_t alignment = 64);
    ~StaticMemoryArena();

    // Non-copyable
    StaticMemoryArena(const StaticMemoryArena&) = delete;
    StaticMemoryArena& operator=(const StaticMemoryArena&) = delete;

    // Moveable
    StaticMemoryArena(StaticMemoryArena&& other) noexcept;
    StaticMemoryArena& operator=(StaticMemoryArena&& other) noexcept;

    void* get_pointer(size_t offset) const;
    void reset();

    size_t capacity() const { return capacity_; }
    size_t alignment() const { return alignment_; }
    bool is_valid() const { return buffer_ != nullptr; }

private:
    void* raw_buffer_ = nullptr;
    void* buffer_ = nullptr;
    size_t capacity_ = 0;
    size_t alignment_ = 64;
};

class StaticGraphPlanner {
public:
    StaticGraphPlanner();
    ~StaticGraphPlanner();

    void add_node(const std::string& name, size_t size_bytes, int32_t first_step, int32_t last_step);
    bool compile_plan();

    const GraphPlanMetrics& get_metrics() const { return metrics_; }
    const std::vector<TensorNode>& get_nodes() const { return nodes_; }
    size_t get_planned_arena_size() const { return metrics_.optimized_arena_size; }

    // Creates an arena matching the compiled plan
    std::unique_ptr<StaticMemoryArena> allocate_arena() const;

    // Factory method creating a realistic transformer forward plan
    static StaticGraphPlanner create_for_transformer(
        int32_t n_layers,
        int32_t n_embd,
        int32_t n_ctx,
        int32_t n_heads,
        int32_t n_vocab
    );

    // Run execution simulation to benchmark zero-allocation vs dynamic malloc/free
    static void benchmark_execution(int32_t iterations = 1000);

private:
    std::vector<TensorNode> nodes_;
    GraphPlanMetrics metrics_;
    bool compiled_ = false;
};

} // namespace graph
} // namespace murdok
