#pragma once

#include "murdok/murdok.h"

namespace murdok {

class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;

    virtual ErrorCode load(const EngineConfig& config) = 0;
    virtual bool is_loaded() const = 0;

    virtual ErrorCode generate(
        const std::string& prompt,
        StreamCallback stream_cb,
        InferenceMetrics* metrics
    ) = 0;

    virtual void reset_context() = 0;

    virtual const HardwareInfo& get_hardware() const = 0;
    virtual InferenceMetrics get_last_metrics() const = 0;
    virtual std::string get_model_name() const = 0;
    virtual uint64_t get_model_size_bytes() const = 0;
    virtual std::string get_backend_name() const = 0;
};

// Factory functions
std::unique_ptr<InferenceBackend> create_llama_backend();
std::unique_ptr<InferenceBackend> create_murdok_backend();

} // namespace murdok
