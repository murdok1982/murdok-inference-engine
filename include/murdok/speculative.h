#pragma once

#include "murdok/murdok.h"
#include <string>
#include <vector>
#include <functional>
#include <memory>

namespace murdok {
namespace speculative {

struct SpeculativeConfig {
    std::string target_model_path;
    std::string draft_model_path;
    int32_t max_draft_tokens = 5;
    bool adaptive = true;
    int32_t n_ctx = 2048;
};

struct SpeculativeMetrics {
    int32_t total_drafted_tokens = 0;
    int32_t total_accepted_tokens = 0;
    double acceptance_rate = 0.0;
    int32_t current_k = 5;
    double generation_tok_per_sec = 0.0;
};

class SpeculativeEngineImpl;

class SpeculativeEngine {
public:
    SpeculativeEngine();
    ~SpeculativeEngine();

    bool load(const SpeculativeConfig& config);
    bool is_loaded() const;

    bool generate(
        const std::string& prompt,
        std::function<bool(const std::string& token)> stream_cb,
        SpeculativeMetrics* out_metrics = nullptr
    );

    SpeculativeMetrics get_metrics() const;

private:
    std::unique_ptr<SpeculativeEngineImpl> pimpl_;
};

} // namespace speculative
} // namespace murdok
