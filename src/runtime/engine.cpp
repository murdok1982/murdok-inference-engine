#include "murdok/murdok.h"
#include "murdok/profile_manager.h"
#include "llama.h"
#include "ggml.h"

#include <iostream>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <cstring>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#else
#include <sys/resource.h>
#endif

namespace fs = std::filesystem;

namespace murdok {

static size_t get_current_peak_rss() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS info;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info))) {
        return info.PeakWorkingSetSize;
    }
    return 0;
#else
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return usage.ru_maxrss * 1024;
    }
    return 0;
#endif
}

class EngineImpl {
public:
    EngineImpl() : hw_(HardwareDetector::detect()) {
        llama_backend_init();
    }

    ~EngineImpl() {
        cleanup();
        llama_backend_free();
    }

    void cleanup() {
        if (ctx_) {
            llama_free(ctx_);
            ctx_ = nullptr;
        }
        if (model_) {
            llama_model_free(model_);
            model_ = nullptr;
        }
    }

    bool load(const EngineConfig& config) {
        cleanup();
        config_ = config;

        // Load persisted calibration profile if available
        MurdokProfile profile;
        if (ProfileManager::load_profile(profile)) {
            if (config_.n_threads_gen <= 0) config_.n_threads_gen = profile.optimal_gen_threads;
            if (config_.n_threads_prompt <= 0) config_.n_threads_prompt = profile.optimal_prompt_threads;
            if (config_.n_batch <= 0) config_.n_batch = profile.optimal_batch_size;
            if (config_.kv_type.empty() || config_.kv_type == "f16") config_.kv_type = profile.kv_cache_type;
        }

        // Auto-configure optimal thread counts based on empirical cache topology findings
        if (config_.n_threads_gen <= 0) {
            config_.n_threads_gen = hw_.physical_cores > 0 ? hw_.physical_cores : 4;
        }
        if (config_.n_threads_prompt <= 0) {
            config_.n_threads_prompt = hw_.logical_threads > 0 ? hw_.logical_threads : config_.n_threads_gen;
        }

        if (!fs::exists(config_.model_path)) {
            std::cerr << "[MuRDoK Engine] Error: Model not found at: " << config_.model_path << "\n";
            return false;
        }

        model_size_bytes_ = fs::file_size(config_.model_path);
        model_name_ = fs::path(config_.model_path).stem().string();

        llama_model_params mparams = llama_model_default_params();
        model_ = llama_model_load_from_file(config_.model_path.c_str(), mparams);
        if (!model_) {
            std::cerr << "[MuRDoK Engine] Error: Failed to load model weights\n";
            return false;
        }

        vocab_ = llama_model_get_vocab(model_);

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = config_.n_ctx;
        cparams.n_batch = config_.n_batch;
        cparams.n_threads = config_.n_threads_gen;
        cparams.n_threads_batch = config_.n_threads_prompt;

        // KV Cache Quantization configuration (Phase 4)
        if (config_.kv_type == "q8_0") {
            cparams.type_k = GGML_TYPE_Q8_0;
            cparams.type_v = GGML_TYPE_Q8_0;
        } else if (config_.kv_type == "q4_0") {
            cparams.type_k = GGML_TYPE_Q4_0;
            cparams.type_v = GGML_TYPE_Q4_0;
        } else {
            cparams.type_k = GGML_TYPE_F16;
            cparams.type_v = GGML_TYPE_F16;
        }

        ctx_ = llama_init_from_model(model_, cparams);
        if (!ctx_) {
            std::cerr << "[MuRDoK Engine] Error: Failed to initialize inference context\n";
            llama_model_free(model_);
            model_ = nullptr;
            return false;
        }

        return true;
    }

    bool is_loaded() const {
        return model_ != nullptr && ctx_ != nullptr;
    }

    void reset_context() {
        if (ctx_) {
            llama_memory_clear(llama_get_memory(ctx_), true);
        }
    }

    bool generate(
        const std::string& prompt,
        std::function<bool(const std::string& token)> stream_cb,
        InferenceMetrics* out_metrics
    ) {
        if (!is_loaded()) {
            std::cerr << "[MuRDoK Engine] Error: Engine not loaded\n";
            return false;
        }

        // Tokenize
        std::vector<llama_token> tokens(config_.n_ctx);
        int32_t n_tokens = llama_tokenize(
            vocab_, prompt.c_str(), static_cast<int32_t>(prompt.length()),
            tokens.data(), static_cast<int32_t>(tokens.size()), true, true
        );
        if (n_tokens < 0) {
            n_tokens = -n_tokens;
            tokens.resize(n_tokens);
            llama_tokenize(
                vocab_, prompt.c_str(), static_cast<int32_t>(prompt.length()),
                tokens.data(), static_cast<int32_t>(tokens.size()), true, true
            );
        } else {
            tokens.resize(n_tokens);
        }

        // Initialize Sampler
        struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
        struct llama_sampler* sampler = llama_sampler_chain_init(sparams);
        if (config_.temperature <= 0.0f) {
            llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
        } else {
            llama_sampler_chain_add(sampler, llama_sampler_init_top_p(config_.top_p, 1));
            llama_sampler_chain_add(sampler, llama_sampler_init_temp(config_.temperature));
            llama_sampler_chain_add(sampler, llama_sampler_init_dist(1234));
        }

        InferenceMetrics metrics;
        metrics.prompt_tokens = n_tokens;

        // Reset context prior to evaluation
        reset_context();

        // 1. Ingest Prompt (Compute-bound batch processing)
        auto t_prompt_start = std::chrono::high_resolution_clock::now();
        for (int32_t i = 0; i < n_tokens; i += config_.n_batch) {
            int32_t n_eval = std::min(static_cast<int32_t>(n_tokens - i), config_.n_batch);
            llama_batch pbatch = llama_batch_get_one(tokens.data() + i, n_eval);
            if (llama_decode(ctx_, pbatch) != 0) {
                std::cerr << "[MuRDoK Engine] Decode failed during prompt processing\n";
                llama_sampler_free(sampler);
                return false;
            }
        }
        auto t_prompt_end = std::chrono::high_resolution_clock::now();
        metrics.prompt_duration_ms = std::chrono::duration<double, std::milli>(t_prompt_end - t_prompt_start).count();
        metrics.prompt_tok_per_sec = (metrics.prompt_duration_ms > 0.0) ? (n_tokens * 1000.0 / metrics.prompt_duration_ms) : 0.0;

        // 2. First Token Generation
        auto t_first_tok_start = std::chrono::high_resolution_clock::now();
        llama_token curr_token = llama_sampler_sample(sampler, ctx_, -1);
        auto t_first_tok_end = std::chrono::high_resolution_clock::now();

        metrics.ttft_ms = metrics.prompt_duration_ms +
            std::chrono::duration<double, std::milli>(t_first_tok_end - t_first_tok_start).count();

        // 3. Autoregressive Generation Loop
        int32_t gen_count = 0;
        auto t_gen_start = std::chrono::high_resolution_clock::now();

        char piece_buf[256];

        while (gen_count < config_.max_tokens) {
            if (llama_vocab_is_eog(vocab_, curr_token)) {
                break;
            }

            int32_t piece_len = llama_token_to_piece(vocab_, curr_token, piece_buf, sizeof(piece_buf), 0, false);
            if (piece_len > 0) {
                std::string piece_str(piece_buf, piece_len);
                if (stream_cb) {
                    if (!stream_cb(piece_str)) {
                        break; // User aborted
                    }
                }
            }

            llama_batch gbatch = llama_batch_get_one(&curr_token, 1);
            if (llama_decode(ctx_, gbatch) != 0) {
                break;
            }

            curr_token = llama_sampler_sample(sampler, ctx_, -1);
            gen_count++;
        }

        auto t_gen_end = std::chrono::high_resolution_clock::now();
        metrics.gen_duration_ms = std::chrono::duration<double, std::milli>(t_gen_end - t_gen_start).count();
        metrics.generated_tokens = gen_count;
        metrics.gen_tok_per_sec = (metrics.gen_duration_ms > 0.0 && gen_count > 0)
            ? (gen_count * 1000.0 / metrics.gen_duration_ms)
            : 0.0;
        metrics.tpot_ms = (metrics.gen_tok_per_sec > 0.0) ? (1000.0 / metrics.gen_tok_per_sec) : 0.0;
        metrics.peak_ram_mb = static_cast<double>(get_current_peak_rss()) / (1024.0 * 1024.0);

        last_metrics_ = metrics;
        if (out_metrics) {
            *out_metrics = metrics;
        }

        llama_sampler_free(sampler);
        return true;
    }

    std::string complete(const std::string& prompt, InferenceMetrics* out_metrics) {
        std::string result;
        generate(prompt, [&](const std::string& piece) -> bool {
            result += piece;
            return true;
        }, out_metrics);
        return result;
    }

    const EngineConfig& get_config() const { return config_; }
    const HardwareInfo& get_hardware() const { return hw_; }
    InferenceMetrics get_last_metrics() const { return last_metrics_; }
    std::string get_model_name() const { return model_name_; }
    uint64_t get_model_size_bytes() const { return model_size_bytes_; }

private:
    HardwareInfo hw_;
    EngineConfig config_;
    struct llama_model* model_ = nullptr;
    const struct llama_vocab* vocab_ = nullptr;
    struct llama_context* ctx_ = nullptr;
    std::string model_name_;
    uint64_t model_size_bytes_ = 0;
    InferenceMetrics last_metrics_;
};

Engine::Engine() : pimpl_(std::make_unique<EngineImpl>()) {}
Engine::~Engine() = default;

bool Engine::load(const EngineConfig& config) {
    return pimpl_->load(config);
}

bool Engine::is_loaded() const {
    return pimpl_->is_loaded();
}

bool Engine::generate(
    const std::string& prompt,
    std::function<bool(const std::string& token)> stream_cb,
    InferenceMetrics* metrics
) {
    return pimpl_->generate(prompt, stream_cb, metrics);
}

std::string Engine::complete(const std::string& prompt, InferenceMetrics* metrics) {
    return pimpl_->complete(prompt, metrics);
}

void Engine::reset_context() {
    pimpl_->reset_context();
}

const EngineConfig& Engine::get_config() const {
    return pimpl_->get_config();
}

const HardwareInfo& Engine::get_hardware() const {
    return pimpl_->get_hardware();
}

InferenceMetrics Engine::get_last_metrics() const {
    return pimpl_->get_last_metrics();
}

std::string Engine::get_model_name() const {
    return pimpl_->get_model_name();
}

uint64_t Engine::get_model_size_bytes() const {
    return pimpl_->get_model_size_bytes();
}

} // namespace murdok
