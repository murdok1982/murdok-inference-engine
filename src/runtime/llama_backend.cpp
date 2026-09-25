#include "murdok/backend.h"
#include "llama.h"
#include "ggml.h"

#include <iostream>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <cstring>
#include <vector>

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

static size_t get_current_rss() {
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

class LlamaCppBackend : public InferenceBackend {
public:
    LlamaCppBackend() : hw_(HardwareDetector::detect()) {
        llama_backend_init();
    }

    ~LlamaCppBackend() override {
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

    ErrorCode load(const EngineConfig& config) override {
        cleanup();
        config_ = config;

        if (!fs::exists(config_.model_path)) {
            return ErrorCode::ModelNotFound;
        }

        model_size_bytes_ = fs::file_size(config_.model_path);
        model_name_ = fs::path(config_.model_path).stem().string();

        llama_model_params mparams = llama_model_default_params();
        model_ = llama_model_load_from_file(config_.model_path.c_str(), mparams);
        if (!model_) {
            return ErrorCode::ModelLoadFailed;
        }

        vocab_ = llama_model_get_vocab(model_);

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = config_.n_ctx;
        cparams.n_batch = config_.n_batch;

        // Baseline: use configured threads or all available logical threads
        int threads = (config_.n_threads_gen > 0) ? config_.n_threads_gen
                    : (hw_.logical_threads > 0 ? hw_.logical_threads : 4);
        cparams.n_threads = threads;
        cparams.n_threads_batch = (config_.n_threads_prompt > 0) ? config_.n_threads_prompt : threads;

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
            llama_model_free(model_);
            model_ = nullptr;
            return ErrorCode::ModelLoadFailed;
        }

        return ErrorCode::Success;
    }

    bool is_loaded() const override {
        return model_ != nullptr && ctx_ != nullptr;
    }

    void reset_context() override {
        if (ctx_) {
            llama_memory_clear(llama_get_memory(ctx_), true);
        }
    }

    ErrorCode generate(
        const std::string& prompt,
        StreamCallback stream_cb,
        InferenceMetrics* out_metrics
    ) override {
        if (!is_loaded()) {
            return ErrorCode::ModelLoadFailed;
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
        metrics.backend_name = "llama.cpp (Reference Baseline)";
        metrics.prompt_tokens = n_tokens;

        reset_context();

        // 1. Ingest Prompt
        auto t_prompt_start = std::chrono::high_resolution_clock::now();
        for (int32_t i = 0; i < n_tokens; i += config_.n_batch) {
            int32_t n_eval = std::min(static_cast<int32_t>(n_tokens - i), config_.n_batch);
            llama_batch pbatch = llama_batch_get_one(tokens.data() + i, n_eval);
            if (llama_decode(ctx_, pbatch) != 0) {
                llama_sampler_free(sampler);
                return ErrorCode::InferenceFailed;
            }
        }
        auto t_prompt_end = std::chrono::high_resolution_clock::now();
        metrics.prompt_duration_ms = std::chrono::duration<double, std::milli>(t_prompt_end - t_prompt_start).count();
        metrics.prompt_tok_per_sec = (metrics.prompt_duration_ms > 0.0) ? (n_tokens * 1000.0 / metrics.prompt_duration_ms) : 0.0;

        // 2. Sample First Token
        auto t_first_tok_start = std::chrono::high_resolution_clock::now();
        llama_token curr_token = llama_sampler_sample(sampler, ctx_, -1);
        auto t_first_tok_end = std::chrono::high_resolution_clock::now();
        metrics.ttft_ms = metrics.prompt_duration_ms +
            std::chrono::duration<double, std::milli>(t_first_tok_end - t_first_tok_start).count();

        // 3. Autoregressive Loop
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
                if (stream_cb && !stream_cb(piece_str)) {
                    break;
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
            ? (gen_count * 1000.0 / metrics.gen_duration_ms) : 0.0;
        metrics.tpot_ms = (metrics.gen_tok_per_sec > 0.0) ? (1000.0 / metrics.gen_tok_per_sec) : 0.0;
        metrics.peak_ram_mb = static_cast<double>(get_current_rss()) / (1024.0 * 1024.0);

        last_metrics_ = metrics;
        if (out_metrics) {
            *out_metrics = metrics;
        }

        llama_sampler_free(sampler);
        return ErrorCode::Success;
    }

    const HardwareInfo& get_hardware() const override { return hw_; }
    InferenceMetrics get_last_metrics() const override { return last_metrics_; }
    std::string get_model_name() const override { return model_name_; }
    uint64_t get_model_size_bytes() const override { return model_size_bytes_; }
    std::string get_backend_name() const override { return "llama.cpp (Reference Baseline)"; }

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

std::unique_ptr<InferenceBackend> create_llama_backend() {
    return std::make_unique<LlamaCppBackend>();
}

} // namespace murdok
