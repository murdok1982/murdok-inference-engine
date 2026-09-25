#include "murdok/hardware.h"
#include "bench.h"
#include "llama.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <ctime>

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

static size_t get_peak_rss_bytes() {
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

struct BenchConfig {
    std::string model_path;
    std::string prompt = "Explain the concept of cache locality in CPU architecture.";
    std::string prompt_file;
    int32_t n_predict = 128;
    int32_t n_threads = 0; // 0 = auto
    int32_t n_batch = 512;
    int32_t n_ctx = 2048;
    int32_t n_warmup = 1;
    std::string out_dir = "benchmarks/results";
    bool compare_mode = false;
};

struct BenchResult {
    std::string timestamp;
    std::string engine_name = "llama.cpp (Baseline Reference)";
    murdok::HardwareInfo hw;
    std::string model_path;
    uint64_t model_size_bytes = 0;
    int32_t prompt_tokens = 0;
    int32_t generated_tokens = 0;
    int32_t threads_used = 0;
    int32_t batch_size = 0;
    int32_t ctx_size = 0;

    double ttft_ms = 0.0;
    double prompt_tok_per_sec = 0.0;
    double gen_tok_per_sec = 0.0;
    double tpot_ms = 0.0;
    double total_time_ms = 0.0;
    double peak_memory_mb = 0.0;
};

static void print_usage(const char* exe) {
    std::cout << "Usage: " << exe << " [options]\n\n"
              << "Options:\n"
              << "  --model <path>        Path to GGUF model (required)\n"
              << "  --prompt <text>       Prompt text (default: explanation of cache locality)\n"
              << "  --prompt-file <path>  Load prompt from file\n"
              << "  --tokens <N>          Tokens to generate (default: 128)\n"
              << "  --threads <N>         Number of CPU threads (default: auto)\n"
              << "  --batch <N>           Batch size (default: 512)\n"
              << "  --ctx <N>             Context size (default: 2048)\n"
              << "  --warmup <N>          Warmup runs (default: 1)\n"
              << "  --out-dir <dir>       Directory for results (default: benchmarks/results)\n"
              << "  --compare             Print comparative benchmark table\n"
              << "  --help                Show this help message\n";
}

static bool parse_args(int argc, char* argv[], BenchConfig& config) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            config.model_path = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            config.prompt = argv[++i];
        } else if (arg == "--prompt-file" && i + 1 < argc) {
            config.prompt_file = argv[++i];
        } else if (arg == "--tokens" && i + 1 < argc) {
            config.n_predict = std::stoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            std::string t_str = argv[++i];
            if (t_str == "auto") config.n_threads = 0;
            else config.n_threads = std::stoi(t_str);
        } else if (arg == "--batch" && i + 1 < argc) {
            config.n_batch = std::stoi(argv[++i]);
        } else if (arg == "--ctx" && i + 1 < argc) {
            config.n_ctx = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            config.n_warmup = std::stoi(argv[++i]);
        } else if (arg == "--out-dir" && i + 1 < argc) {
            config.out_dir = argv[++i];
        } else if (arg == "--compare") {
            config.compare_mode = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return false;
        }
    }

    if (config.model_path.empty()) {
        std::cerr << "Error: --model <path> is required.\n\n";
        print_usage(argv[0]);
        return false;
    }

    if (!config.prompt_file.empty()) {
        std::ifstream file(config.prompt_file);
        if (file.is_open()) {
            std::stringstream ss;
            ss << file.rdbuf();
            config.prompt = ss.str();
        } else {
            std::cerr << "Warning: Could not open prompt file: " << config.prompt_file << ", using default prompt.\n";
        }
    }

    return true;
}

static std::string get_current_iso_time() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    struct tm buf;
#if defined(_WIN32)
    gmtime_s(&buf, &now_c);
#else
    gmtime_r(&now_c, &buf);
#endif
    char str[64];
    std::strftime(str, sizeof(str), "%Y-%m-%dT%H:%M:%SZ", &buf);
    return std::string(str);
}

namespace murdok {
namespace bench {

int run_murdok_bench(int argc, char* argv[]) {
    BenchConfig config;
    if (!parse_args(argc, argv, config)) {
        return 1;
    }

    auto hw = murdok::HardwareDetector::detect();
    if (config.n_threads <= 0) {
        config.n_threads = hw.physical_cores > 0 ? hw.physical_cores : 4;
    }

    std::cout << "==========================================================\n";
    std::cout << "          MuRDoK Inference Engine — Benchmark             \n";
    std::cout << "==========================================================\n";
    std::cout << "Model:    " << config.model_path << "\n";
    std::cout << "Threads:  " << config.n_threads << " (Physical: " << hw.physical_cores << ", Logical: " << hw.logical_threads << ")\n";
    std::cout << "Tokens:   " << config.n_predict << "\n";
    std::cout << "Batch:    " << config.n_batch << "\n";
    std::cout << "Context:  " << config.n_ctx << "\n";
    std::cout << "----------------------------------------------------------\n";

    if (!fs::exists(config.model_path)) {
        std::cerr << "Error: Model file does not exist at: " << config.model_path << "\n";
        return 1;
    }
    uint64_t model_size = fs::file_size(config.model_path);

    // Initialize llama backend
    llama_backend_init();

    // Model parameters
    llama_model_params mparams = llama_model_default_params();
    struct llama_model* model = llama_model_load_from_file(config.model_path.c_str(), mparams);
    if (!model) {
        std::cerr << "Failed to load model from " << config.model_path << "\n";
        llama_backend_free();
        return 1;
    }

    const struct llama_vocab* vocab = llama_model_get_vocab(model);

    // Context parameters
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = config.n_ctx;
    cparams.n_batch = config.n_batch;
    cparams.n_threads = config.n_threads;
    cparams.n_threads_batch = config.n_threads;

    struct llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::cerr << "Failed to initialize context\n";
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // Tokenize prompt
    std::vector<llama_token> tokens(config.n_ctx);
    int32_t n_prompt_tokens = llama_tokenize(vocab, config.prompt.c_str(), static_cast<int32_t>(config.prompt.length()),
                                             tokens.data(), static_cast<int32_t>(tokens.size()), true, true);
    if (n_prompt_tokens < 0) {
        n_prompt_tokens = -n_prompt_tokens;
        tokens.resize(n_prompt_tokens);
        llama_tokenize(vocab, config.prompt.c_str(), static_cast<int32_t>(config.prompt.length()),
                       tokens.data(), static_cast<int32_t>(tokens.size()), true, true);
    } else {
        tokens.resize(n_prompt_tokens);
    }

    std::cout << "Prompt tokens: " << n_prompt_tokens << "\n";
    std::cout << "Beginning benchmark execution...\n\n";

    // Setup sampler (greedy)
    struct llama_sampler* sampler = llama_sampler_init_greedy();

    // Warmup run
    for (int w = 0; w < config.n_warmup; ++w) {
        llama_token warmup_token = tokens.empty() ? 1 : tokens[0];
        llama_batch wbatch = llama_batch_get_one(&warmup_token, 1);
        llama_decode(ctx, wbatch);
    }
    // Clear KV cache after warmup
    llama_memory_clear(llama_get_memory(ctx), true);

    // Benchmark Run
    BenchResult res;
    res.timestamp = get_current_iso_time();
    res.hw = hw;
    res.model_path = config.model_path;
    res.model_size_bytes = model_size;
    res.prompt_tokens = n_prompt_tokens;
    res.threads_used = config.n_threads;
    res.batch_size = config.n_batch;
    res.ctx_size = config.n_ctx;

    auto t_start_all = std::chrono::high_resolution_clock::now();

    // 1. Prompt Processing
    auto t_prompt_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_prompt_tokens; i += config.n_batch) {
        int n_eval = std::min(static_cast<int>(n_prompt_tokens - i), config.n_batch);
        llama_batch pbatch = llama_batch_get_one(tokens.data() + i, n_eval);
        if (llama_decode(ctx, pbatch) != 0) {
            std::cerr << "llama_decode failed during prompt processing\n";
            break;
        }
    }
    auto t_prompt_end = std::chrono::high_resolution_clock::now();
    double prompt_duration_sec = std::chrono::duration<double>(t_prompt_end - t_prompt_start).count();
    res.prompt_tok_per_sec = (prompt_duration_sec > 0.0) ? (n_prompt_tokens / prompt_duration_sec) : 0.0;

    // 2. First Token Generation
    auto t_first_tok_start = std::chrono::high_resolution_clock::now();
    llama_token curr_token = llama_sampler_sample(sampler, ctx, -1);
    auto t_first_tok_end = std::chrono::high_resolution_clock::now();
    double ttft_sec = prompt_duration_sec + std::chrono::duration<double>(t_first_tok_end - t_first_tok_start).count();
    res.ttft_ms = ttft_sec * 1000.0;

    // 3. Autoregressive Generation
    int generated_count = 0;
    auto t_gen_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < config.n_predict; ++i) {
        if (llama_vocab_is_eog(vocab, curr_token)) {
            break;
        }
        llama_batch gbatch = llama_batch_get_one(&curr_token, 1);
        if (llama_decode(ctx, gbatch) != 0) {
            break;
        }
        curr_token = llama_sampler_sample(sampler, ctx, -1);
        generated_count++;
    }

    auto t_gen_end = std::chrono::high_resolution_clock::now();
    auto t_end_all = std::chrono::high_resolution_clock::now();

    double gen_duration_sec = std::chrono::duration<double>(t_gen_end - t_gen_start).count();
    res.generated_tokens = generated_count;
    res.gen_tok_per_sec = (gen_duration_sec > 0.0 && generated_count > 0) ? (generated_count / gen_duration_sec) : 0.0;
    res.tpot_ms = (res.gen_tok_per_sec > 0.0) ? (1000.0 / res.gen_tok_per_sec) : 0.0;
    res.total_time_ms = std::chrono::duration<double, std::milli>(t_end_all - t_start_all).count();
    res.peak_memory_mb = static_cast<double>(get_peak_rss_bytes()) / (1024.0 * 1024.0);

    // Print Results to Console
    std::cout << "==========================================================\n";
    std::cout << "                  BENCHMARK RESULTS                       \n";
    std::cout << "==========================================================\n";
    std::cout << std::left << std::setw(28) << "Engine:" << res.engine_name << "\n";
    std::cout << std::left << std::setw(28) << "Prompt Processing:" << std::fixed << std::setprecision(2)
              << res.prompt_tok_per_sec << " tokens/sec (" << n_prompt_tokens << " tokens in "
              << prompt_duration_sec << " s)\n";
    std::cout << std::left << std::setw(28) << "Generation Throughput:" << std::fixed << std::setprecision(2)
              << res.gen_tok_per_sec << " tokens/sec (" << generated_count << " tokens in "
              << gen_duration_sec << " s)\n";
    std::cout << std::left << std::setw(28) << "Time To First Token (TTFT):" << std::fixed << std::setprecision(2)
              << res.ttft_ms << " ms\n";
    std::cout << std::left << std::setw(28) << "Time Per Output Token (TPOT):" << std::fixed << std::setprecision(2)
              << res.tpot_ms << " ms/token\n";
    std::cout << std::left << std::setw(28) << "Peak RAM Working Set:" << std::fixed << std::setprecision(2)
              << res.peak_memory_mb << " MB\n";
    std::cout << "==========================================================\n\n";

    // Write JSON and CSV files
    try {
        fs::create_directories(config.out_dir);

        // JSON Output
        std::string json_file = config.out_dir + "/baseline_run.json";
        std::ofstream jf(json_file);
        if (jf.is_open()) {
            jf << "{\n"
               << "  \"timestamp\": \"" << res.timestamp << "\",\n"
               << "  \"engine\": \"" << res.engine_name << "\",\n"
               << "  \"hardware\": {\n"
               << "    \"cpu\": \"" << hw.cpu_brand << "\",\n"
               << "    \"physical_cores\": " << hw.physical_cores << ",\n"
               << "    \"logical_threads\": " << hw.logical_threads << ",\n"
               << "    \"avx2\": " << (hw.simd.avx2 ? "true" : "false") << ",\n"
               << "    \"fma\": " << (hw.simd.fma ? "true" : "false") << ",\n"
               << "    \"total_ram_gb\": " << (static_cast<double>(hw.memory.total_ram_bytes) / (1024.0*1024.0*1024.0)) << "\n"
               << "  },\n"
               << "  \"model\": {\n"
               << "    \"path\": \"" << res.model_path << "\",\n"
               << "    \"size_mb\": " << (static_cast<double>(res.model_size_bytes) / (1024.0*1024.0)) << "\n"
               << "  },\n"
               << "  \"config\": {\n"
               << "    \"threads\": " << res.threads_used << ",\n"
               << "    \"batch_size\": " << res.batch_size << ",\n"
               << "    \"ctx_size\": " << res.ctx_size << "\n"
               << "  },\n"
               << "  \"metrics\": {\n"
               << "    \"prompt_tokens\": " << res.prompt_tokens << ",\n"
               << "    \"prompt_tok_per_sec\": " << res.prompt_tok_per_sec << ",\n"
               << "    \"generated_tokens\": " << res.generated_tokens << ",\n"
               << "    \"gen_tok_per_sec\": " << res.gen_tok_per_sec << ",\n"
               << "    \"ttft_ms\": " << res.ttft_ms << ",\n"
               << "    \"tpot_ms\": " << res.tpot_ms << ",\n"
               << "    \"total_time_ms\": " << res.total_time_ms << ",\n"
               << "    \"peak_memory_mb\": " << res.peak_memory_mb << "\n"
               << "  }\n"
               << "}\n";
            jf.close();
            std::cout << "Saved JSON benchmark: " << json_file << "\n";
        }

        // CSV Output
        std::string csv_file = config.out_dir + "/benchmarks.csv";
        bool write_header = !fs::exists(csv_file);
        std::ofstream cf(csv_file, std::ios::app);
        if (cf.is_open()) {
            if (write_header) {
                cf << "timestamp,engine,model,threads,prompt_tokens,prompt_tok_s,gen_tokens,gen_tok_s,ttft_ms,tpot_ms,peak_mem_mb\n";
            }
            cf << res.timestamp << ","
               << res.engine_name << ","
               << fs::path(res.model_path).filename().string() << ","
               << res.threads_used << ","
               << res.prompt_tokens << ","
               << res.prompt_tok_per_sec << ","
               << res.generated_tokens << ","
               << res.gen_tok_per_sec << ","
               << res.ttft_ms << ","
               << res.tpot_ms << ","
               << res.peak_memory_mb << "\n";
            cf.close();
            std::cout << "Saved CSV row: " << csv_file << "\n";
        }
    } catch (const std::exception& ex) {
        std::cerr << "Failed to write results files: " << ex.what() << "\n";
    }

    // Cleanup
    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}

} // namespace bench
} // namespace murdok

#ifndef MURDOK_NO_BENCH_MAIN
int main(int argc, char* argv[]) {
    return murdok::bench::run_murdok_bench(argc, argv);
}
#endif
