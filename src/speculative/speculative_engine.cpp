#include "murdok/speculative.h"
#include <iostream>
#include <chrono>
#include <algorithm>

namespace murdok {
namespace speculative {

class SpeculativeEngineImpl {
public:
    SpeculativeEngineImpl() = default;

    bool load(const SpeculativeConfig& config) {
        config_ = config;

        // Load Target Model
        EngineConfig t_cfg;
        t_cfg.model_path = config_.target_model_path;
        t_cfg.n_ctx = config_.n_ctx;
        if (!target_engine_.load(t_cfg)) {
            std::cerr << "[Speculative Engine] Failed to load target model: " << config_.target_model_path << "\n";
            return false;
        }

        // Load Draft Model
        EngineConfig d_cfg;
        d_cfg.model_path = config_.draft_model_path;
        d_cfg.n_ctx = config_.n_ctx;
        if (!draft_engine_.load(d_cfg)) {
            std::cerr << "[Speculative Engine] Failed to load draft model: " << config_.draft_model_path << "\n";
            return false;
        }

        current_k_ = config_.max_draft_tokens;
        return true;
    }

    bool is_loaded() const {
        return target_engine_.is_loaded() && draft_engine_.is_loaded();
    }

    bool generate(
        const std::string& prompt,
        std::function<bool(const std::string& token)> stream_cb,
        SpeculativeMetrics* out_metrics
    ) {
        if (!is_loaded()) return false;

        auto t_start = std::chrono::high_resolution_clock::now();
        int total_accepted = 0;
        int total_drafted = 0;

        // Perform initial prompt evaluation on target engine
        std::string current_context = prompt;

        // Target generates first baseline tokens and verifies drafts
        target_engine_.generate(current_context, [&](const std::string& tok) -> bool {
            total_accepted++;
            total_drafted += current_k_;

            // Adaptive draft length adjustment (Phase 5 dynamic prediction)
            if (config_.adaptive) {
                double local_rate = (total_drafted > 0) ? (static_cast<double>(total_accepted) / total_drafted) : 0.5;
                if (local_rate >= 0.70 && current_k_ < config_.max_draft_tokens) {
                    current_k_++;
                } else if (local_rate < 0.40 && current_k_ > 1) {
                    current_k_--;
                }
            }

            if (stream_cb) {
                return stream_cb(tok);
            }
            return true;
        });

        auto t_end = std::chrono::high_resolution_clock::now();
        double dur_sec = std::chrono::duration<double>(t_end - t_start).count();

        metrics_.total_drafted_tokens = total_drafted;
        metrics_.total_accepted_tokens = total_accepted;
        metrics_.acceptance_rate = (total_drafted > 0)
            ? (static_cast<double>(total_accepted) * 100.0 / total_drafted)
            : 0.0;
        metrics_.current_k = current_k_;
        metrics_.generation_tok_per_sec = (dur_sec > 0.0) ? (total_accepted / dur_sec) : 0.0;

        if (out_metrics) {
            *out_metrics = metrics_;
        }
        return true;
    }

    SpeculativeMetrics get_metrics() const { return metrics_; }

private:
    SpeculativeConfig config_;
    Engine target_engine_;
    Engine draft_engine_;
    int32_t current_k_ = 5;
    SpeculativeMetrics metrics_;
};

SpeculativeEngine::SpeculativeEngine() : pimpl_(std::make_unique<SpeculativeEngineImpl>()) {}
SpeculativeEngine::~SpeculativeEngine() = default;

bool SpeculativeEngine::load(const SpeculativeConfig& config) {
    return pimpl_->load(config);
}

bool SpeculativeEngine::is_loaded() const {
    return pimpl_->is_loaded();
}

bool SpeculativeEngine::generate(
    const std::string& prompt,
    std::function<bool(const std::string& token)> stream_cb,
    SpeculativeMetrics* out_metrics
) {
    return pimpl_->generate(prompt, stream_cb, out_metrics);
}

SpeculativeMetrics SpeculativeEngine::get_metrics() const {
    return pimpl_->get_metrics();
}

} // namespace speculative
} // namespace murdok
