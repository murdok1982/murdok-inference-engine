#include "murdok/murdok.h"
#include <cassert>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

void test_runtime_engine() {
    std::cout << "[TEST] Running Engine & Backend Abstraction Tests...\n";

    murdok::Engine engine;
    assert(!engine.is_loaded());

    // 1. Missing model error handling
    murdok::EngineConfig bad_cfg;
    bad_cfg.model_path = "non_existent_path.gguf";
    auto err = engine.load(bad_cfg);
    assert(err == murdok::ErrorCode::ModelNotFound);
    assert(!engine.is_loaded());

    // 2. Real model test (if exists)
    std::string test_model = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    if (fs::exists(test_model)) {
        // Test Reference LlamaCpp Backend
        murdok::EngineConfig llama_cfg;
        llama_cfg.model_path = test_model;
        llama_cfg.backend_type = murdok::BackendType::LlamaCpp;
        llama_cfg.max_tokens = 8;
        llama_cfg.temperature = 0.0f;

        murdok::Engine llama_engine;
        auto l_err = llama_engine.load(llama_cfg);
        assert(l_err == murdok::ErrorCode::Success);
        assert(llama_engine.is_loaded());
        assert(llama_engine.get_backend_name() == "llama.cpp (Reference Baseline)");

        murdok::InferenceMetrics l_metrics;
        std::string l_out = llama_engine.complete("Hello", &l_metrics);
        assert(!l_out.empty());
        assert(l_metrics.prompt_tokens > 0);
        assert(l_metrics.generated_tokens > 0);
        std::cout << "  [LlamaCppBackend] generated " << l_metrics.generated_tokens
                  << " tokens (" << l_metrics.gen_tok_per_sec << " tok/s)\n";

        // Test Murdok Backend
        murdok::EngineConfig murdok_cfg;
        murdok_cfg.model_path = test_model;
        murdok_cfg.backend_type = murdok::BackendType::Murdok;
        murdok_cfg.max_tokens = 8;
        murdok_cfg.temperature = 0.0f;

        murdok::Engine murdok_engine;
        auto m_err = murdok_engine.load(murdok_cfg);
        assert(m_err == murdok::ErrorCode::Success);
        assert(murdok_engine.is_loaded());
        assert(murdok_engine.get_backend_name() == "MuRDoK Optimized Backend");

        murdok::InferenceMetrics m_metrics;
        std::string m_out = murdok_engine.complete("Hello", &m_metrics);
        assert(!m_out.empty());
        assert(m_metrics.prompt_tokens > 0);
        assert(m_metrics.generated_tokens > 0);
        std::cout << "  [MurdokBackend]   generated " << m_metrics.generated_tokens
                  << " tokens (" << m_metrics.gen_tok_per_sec << " tok/s)\n";

        // Test Context Reset
        murdok_engine.reset_context();
    } else {
        std::cout << "  (Skipping real model inference test: " << test_model << " not found)\n";
    }

    std::cout << "[PASS] Engine & Backend Abstraction Tests Passed.\n\n";
}
