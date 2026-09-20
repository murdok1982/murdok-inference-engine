#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
#undef CPPHTTPLIB_OPENSSL_SUPPORT
#endif
#ifdef CPPHTTPLIB_ZLIB_SUPPORT
#undef CPPHTTPLIB_ZLIB_SUPPORT
#endif
#include "httplib.h"
#include "nlohmann/json.hpp"

#include "murdok/murdok.h"
#include "src/server/web_ui.h"

#include <iostream>
#include <string>
#include <filesystem>
#include <mutex>

namespace fs = std::filesystem;
using json = nlohmann::json;

struct ServerConfig {
    std::string model_path = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    std::string host = "127.0.0.1";
    int port = 8080;
    int n_ctx = 2048;
    int n_threads = 0;
};

static void print_usage(const char* exe) {
    std::cout << "Usage: " << exe << " [options]\n\n"
              << "Options:\n"
              << "  --model <path>  Path to GGUF model (default: models/qwen2.5-0.5b-instruct-q4_k_m.gguf)\n"
              << "  --host <ip>     Host address to bind (default: 127.0.0.1)\n"
              << "  --port <port>   Port number to listen on (default: 8080)\n"
              << "  --ctx <N>       Context size (default: 2048)\n"
              << "  --threads <N>   Number of threads (default: auto)\n"
              << "  --help          Show this message\n";
}

int main(int argc, char* argv[]) {
    ServerConfig s_cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            s_cfg.model_path = argv[++i];
        } else if (arg == "--host" && i + 1 < argc) {
            s_cfg.host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            s_cfg.port = std::stoi(argv[++i]);
        } else if (arg == "--ctx" && i + 1 < argc) {
            s_cfg.n_ctx = std::stoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            s_cfg.n_threads = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    // Auto-discover model if default not present
    if (!fs::exists(s_cfg.model_path)) {
        for (const auto& entry : fs::directory_iterator("models")) {
            if (entry.path().extension() == ".gguf") {
                s_cfg.model_path = entry.path().string();
                break;
            }
        }
    }

    std::cout << "==========================================================\n";
    std::cout << "             MuRDoK Inference Engine — Server             \n";
    std::cout << "==========================================================\n";
    std::cout << "Model:    " << s_cfg.model_path << "\n";
    std::cout << "Binding:  http://" << s_cfg.host << ":" << s_cfg.port << "\n";

    murdok::Engine engine;
    murdok::EngineConfig e_cfg;
    e_cfg.model_path = s_cfg.model_path;
    e_cfg.n_ctx = s_cfg.n_ctx;
    e_cfg.n_threads_gen = s_cfg.n_threads;

    if (!engine.load(e_cfg)) {
        std::cerr << "Failed to initialize and load model into MuRDoK Engine.\n";
        return 1;
    }

    auto hw = engine.get_hardware();
    std::cout << "Hardware: " << hw.cpu_brand << " (" << hw.physical_cores << " cores / "
              << hw.logical_threads << " threads)\n";
    std::cout << "Profile:  " << murdok::HardwareDetector::profile_to_string(hw.recommended_profile) << "\n";
    std::cout << "Status:   Engine loaded and ready.\n";
    std::cout << "OpenAI API: http://" << s_cfg.host << ":" << s_cfg.port << "/v1/chat/completions\n";
    std::cout << "Web UI:     http://" << s_cfg.host << ":" << s_cfg.port << "/\n";
    std::cout << "==========================================================\n";

    httplib::Server svr;
    std::mutex engine_mutex;

    // Root UI
    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(murdok::get_embedded_web_ui(), "text/html; charset=utf-8");
    });

    // Health check
    svr.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        json j;
        j["status"] = "ok";
        j["model"] = engine.get_model_name();
        res.set_content(j.dump(), "application/json");
    });

    // OpenAI Models list
    svr.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        json j;
        j["object"] = "list";
        j["data"] = json::array({
            {
                {"id", engine.get_model_name()},
                {"object", "model"},
                {"created", 1700000000},
                {"owned_by", "murdok"}
            }
        });
        res.set_content(j.dump(), "application/json");
    });

    // OpenAI Chat Completions endpoint
    svr.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception& ex) {
            res.status = 400;
            res.set_content(json({{"error", "Invalid JSON: " + std::string(ex.what())}}).dump(), "application/json");
            return;
        }

        // Build prompt from messages
        std::string prompt;
        if (body.contains("messages") && body["messages"].is_array()) {
            for (const auto& msg : body["messages"]) {
                std::string role = msg.value("role", "user");
                std::string content = msg.value("content", "");
                if (role == "system") {
                    prompt += "<|im_start|>system\n" + content + "<|im_end|>\n";
                } else if (role == "user") {
                    prompt += "<|im_start|>user\n" + content + "<|im_end|>\n";
                } else if (role == "assistant") {
                    prompt += "<|im_start|>assistant\n" + content + "<|im_end|>\n";
                }
            }
            prompt += "<|im_start|>assistant\n";
        } else if (body.contains("prompt") && body["prompt"].is_string()) {
            prompt = body["prompt"].get<std::string>();
        } else {
            prompt = "Hello!";
        }

        bool stream = body.value("stream", false);

        std::lock_guard<std::mutex> lock(engine_mutex);

        if (stream) {
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");

            murdok::InferenceMetrics metrics;
            res.set_chunked_content_provider("text/event-stream",
                [&engine, prompt, metrics](size_t, httplib::DataSink& sink) mutable -> bool {
                    engine.generate(prompt, [&](const std::string& token) -> bool {
                        json chunk;
                        chunk["id"] = "chatcmpl-murdok";
                        chunk["object"] = "chat.completion.chunk";
                        chunk["choices"] = json::array({
                            {
                                {"index", 0},
                                {"delta", {{"content", token}}},
                                {"finish_reason", nullptr}
                            }
                        });
                        std::string payload = "data: " + chunk.dump() + "\n\n";
                        return sink.write(payload.data(), payload.size());
                    }, &metrics);

                    // Send finish chunk with metrics
                    json finish_chunk;
                    finish_chunk["id"] = "chatcmpl-murdok";
                    finish_chunk["object"] = "chat.completion.chunk";
                    finish_chunk["choices"] = json::array({
                        {
                            {"index", 0},
                            {"delta", json::object()},
                            {"finish_reason", "stop"}
                        }
                    });
                    finish_chunk["metrics"] = {
                        {"prompt_tokens", metrics.prompt_tokens},
                        {"prompt_tok_per_sec", metrics.prompt_tok_per_sec},
                        {"generated_tokens", metrics.generated_tokens},
                        {"gen_tok_per_sec", metrics.gen_tok_per_sec},
                        {"ttft_ms", metrics.ttft_ms},
                        {"tpot_ms", metrics.tpot_ms},
                        {"peak_ram_mb", metrics.peak_ram_mb}
                    };
                    std::string payload = "data: " + finish_chunk.dump() + "\n\ndata: [DONE]\n\n";
                    sink.write(payload.data(), payload.size());
                    sink.done();
                    return true;
                }
            );
        } else {
            murdok::InferenceMetrics metrics;
            std::string text = engine.complete(prompt, &metrics);

            json reply;
            reply["id"] = "chatcmpl-murdok";
            reply["object"] = "chat.completion";
            reply["choices"] = json::array({
                {
                    {"index", 0},
                    {"message", {{"role", "assistant"}, {"content", text}}},
                    {"finish_reason", "stop"}
                }
            });
            reply["usage"] = {
                {"prompt_tokens", metrics.prompt_tokens},
                {"completion_tokens", metrics.generated_tokens},
                {"total_tokens", metrics.prompt_tokens + metrics.generated_tokens}
            };
            reply["metrics"] = {
                {"prompt_tok_per_sec", metrics.prompt_tok_per_sec},
                {"gen_tok_per_sec", metrics.gen_tok_per_sec},
                {"ttft_ms", metrics.ttft_ms},
                {"tpot_ms", metrics.tpot_ms},
                {"peak_ram_mb", metrics.peak_ram_mb}
            };

            res.set_content(reply.dump(), "application/json");
        }
    });

    svr.listen(s_cfg.host, s_cfg.port);
    return 0;
}
