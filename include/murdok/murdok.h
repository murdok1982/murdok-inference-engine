#pragma once

#include "murdok/hardware.h"
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <cstdint>

namespace murdok {

enum class BackendType {
    Auto = 0,
    LlamaCpp = 1,  // Pristine reference backend (scientific baseline)
    Murdok = 2     // MuRDoK hardware-adaptive optimized backend
};

enum class ErrorCode {
    Success = 0,
    InvalidConfiguration = 1,
    ModelNotFound = 2,
    ModelLoadFailed = 3,
    UnsupportedHardware = 4,
    OutOfMemory = 5,
    InvalidFormat = 6,
    InferenceFailed = 7,
    BackendUnavailable = 8
};

const char* error_code_to_string(ErrorCode code);

struct EngineConfig {
    std::string model_path;
    BackendType backend_type = BackendType::Auto;
    int32_t n_ctx = 2048;
    int32_t n_batch = 512;
    int32_t n_threads_prompt = 0;   // 0 = auto-detect optimal (all logical cores)
    int32_t n_threads_gen = 0;      // 0 = auto-detect optimal (physical cores to prevent L1/L2 thrashing)
    float temperature = 0.7f;
    float top_p = 0.9f;
    int32_t max_tokens = 512;
    std::string kv_type = "f16";    // "f16", "q8_0", "q4_0"
    bool verbose = false;
};

struct InferenceMetrics {
    std::string backend_name;
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

using StreamCallback = std::function<bool(const std::string& token)>;

class EngineImpl;

class Engine {
public:
    Engine();
    ~Engine();

    // Non-copyable
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // Movable
    Engine(Engine&&) noexcept;
    Engine& operator=(Engine&&) noexcept;

    // Initialize and load model
    ErrorCode load(const EngineConfig& config);
    bool is_loaded() const;

    // Text generation with streaming callback.
    // If stream_cb returns false, generation aborts early.
    ErrorCode generate(
        const std::string& prompt,
        StreamCallback stream_cb,
        InferenceMetrics* metrics = nullptr
    );

    // One-shot completion
    std::string complete(const std::string& prompt, InferenceMetrics* metrics = nullptr);

    // Context management
    void reset_context();

    // Diagnostics & Metadata
    const EngineConfig& get_config() const;
    const HardwareInfo& get_hardware() const;
    InferenceMetrics get_last_metrics() const;
    ErrorCode get_last_error() const;
    std::string get_model_name() const;
    uint64_t get_model_size_bytes() const;
    std::string get_backend_name() const;

private:
    std::unique_ptr<EngineImpl> pimpl_;
};

} // namespace murdok
