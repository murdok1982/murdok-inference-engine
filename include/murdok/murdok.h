#pragma once

#include "murdok/hardware.h"
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <cstdint>

namespace murdok {

struct EngineConfig {
    std::string model_path;
    int32_t n_ctx = 2048;
    int32_t n_batch = 512;
    int32_t n_threads_prompt = 0;   // 0 = auto-detect optimal (all logical cores)
    int32_t n_threads_gen = 0;      // 0 = auto-detect optimal (physical cores to prevent L1/L2 thrashing)
    float temperature = 0.7f;
    float top_p = 0.9f;
    int32_t max_tokens = 512;
    bool verbose = false;
};

struct InferenceMetrics {
    int32_t prompt_tokens = 0;
    int32_t generated_tokens = 0;
    double prompt_duration_ms = 0.0;
    double gen_duration_ms = 0.0;
    double prompt_tok_per_sec = 0.0;
    double gen_tok_per_sec = 0.0;
    double ttft_ms = 0.0;
    double tpot_ms = 0.0;
    double peak_ram_mb = 0.0;
};

class EngineImpl;

class Engine {
public:
    Engine();
    ~Engine();

    // Initialize and load model
    bool load(const EngineConfig& config);
    bool is_loaded() const;

    // Text generation with streaming callback.
    // If stream_cb returns false, generation aborts early.
    bool generate(
        const std::string& prompt,
        std::function<bool(const std::string& token)> stream_cb,
        InferenceMetrics* metrics = nullptr
    );

    // One-shot completion
    std::string complete(const std::string& prompt, InferenceMetrics* metrics = nullptr);

    // Context management
    void reset_context();

    // Metadata
    const EngineConfig& get_config() const;
    const HardwareInfo& get_hardware() const;
    InferenceMetrics get_last_metrics() const;
    std::string get_model_name() const;
    uint64_t get_model_size_bytes() const;

private:
    std::unique_ptr<EngineImpl> pimpl_;
};

} // namespace murdok
