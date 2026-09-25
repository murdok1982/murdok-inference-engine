#include "murdok/murdok.h"
#include "murdok/backend.h"

#include <iostream>
#include <memory>

namespace murdok {

const char* error_code_to_string(ErrorCode code) {
    switch (code) {
        case ErrorCode::Success: return "Success";
        case ErrorCode::InvalidConfiguration: return "Invalid Configuration";
        case ErrorCode::ModelNotFound: return "Model File Not Found";
        case ErrorCode::ModelLoadFailed: return "Failed to Load Model Weights";
        case ErrorCode::UnsupportedHardware: return "Hardware Lacks Required Features";
        case ErrorCode::OutOfMemory: return "Insufficient System RAM";
        case ErrorCode::InvalidFormat: return "Invalid or Corrupted Model Format";
        case ErrorCode::InferenceFailed: return "Inference Execution Failed";
        case ErrorCode::BackendUnavailable: return "Requested Backend Unavailable";
        default: return "Unknown Error";
    }
}

class EngineImpl {
public:
    EngineImpl() : hw_(HardwareDetector::detect()) {}

    ErrorCode load(const EngineConfig& config) {
        config_ = config;

        // Select backend
        if (config_.backend_type == BackendType::LlamaCpp) {
            backend_ = create_llama_backend();
        } else {
            // Default Auto / Murdok
            backend_ = create_murdok_backend();
        }

        if (!backend_) {
            last_error_ = ErrorCode::BackendUnavailable;
            return last_error_;
        }

        last_error_ = backend_->load(config_);
        return last_error_;
    }

    bool is_loaded() const {
        return backend_ && backend_->is_loaded();
    }

    ErrorCode generate(
        const std::string& prompt,
        StreamCallback stream_cb,
        InferenceMetrics* metrics
    ) {
        if (!backend_ || !backend_->is_loaded()) {
            last_error_ = ErrorCode::ModelLoadFailed;
            return last_error_;
        }

        last_error_ = backend_->generate(prompt, stream_cb, metrics);
        return last_error_;
    }

    std::string complete(const std::string& prompt, InferenceMetrics* metrics) {
        std::string result;
        generate(prompt, [&](const std::string& piece) -> bool {
            result += piece;
            return true;
        }, metrics);
        return result;
    }

    void reset_context() {
        if (backend_) {
            backend_->reset_context();
        }
    }

    const EngineConfig& get_config() const { return config_; }
    const HardwareInfo& get_hardware() const { return hw_; }
    InferenceMetrics get_last_metrics() const {
        if (backend_) return backend_->get_last_metrics();
        return {};
    }
    ErrorCode get_last_error() const { return last_error_; }
    std::string get_model_name() const {
        if (backend_) return backend_->get_model_name();
        return "";
    }
    uint64_t get_model_size_bytes() const {
        if (backend_) return backend_->get_model_size_bytes();
        return 0;
    }
    std::string get_backend_name() const {
        if (backend_) return backend_->get_backend_name();
        return "None";
    }

private:
    HardwareInfo hw_;
    EngineConfig config_;
    ErrorCode last_error_ = ErrorCode::Success;
    std::unique_ptr<InferenceBackend> backend_;
};

Engine::Engine() : pimpl_(std::make_unique<EngineImpl>()) {}
Engine::~Engine() = default;

Engine::Engine(Engine&&) noexcept = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

ErrorCode Engine::load(const EngineConfig& config) {
    return pimpl_->load(config);
}

bool Engine::is_loaded() const {
    return pimpl_->is_loaded();
}

ErrorCode Engine::generate(
    const std::string& prompt,
    StreamCallback stream_cb,
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

ErrorCode Engine::get_last_error() const {
    return pimpl_->get_last_error();
}

std::string Engine::get_model_name() const {
    return pimpl_->get_model_name();
}

uint64_t Engine::get_model_size_bytes() const {
    return pimpl_->get_model_size_bytes();
}

std::string Engine::get_backend_name() const {
    return pimpl_->get_backend_name();
}

} // namespace murdok
