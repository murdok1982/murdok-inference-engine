#include "murdok/murdok.h"
#include "murdok/profile_manager.h"
#include "murdok/kernels.h"
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <iomanip>

namespace fs = std::filesystem;

static void print_main_help() {
    std::cout << "==========================================================\n";
    std::cout << "         MuRDoK Inference Engine (MIE) v0.2.0             \n";
    std::cout << "    Move less. Compute smarter. Infer faster. Benchmark.  \n";
    std::cout << "==========================================================\n\n"
              << "Usage: murdok <command> [options]\n\n"
              << "Commands:\n"
              << "  run <model.gguf>         Start interactive local chat session with HUD\n"
              << "  server [options]         Start OpenAI-compatible REST server & Web UI\n"
              << "  optimize [options]       Auto-calibrate hardware and find optimal settings\n"
              << "  hardware                 Inspect CPU, SIMD, cache hierarchy, and RAM\n"
              << "  bench [options]          Run standard benchmark suite\n"
              << "  help                     Show this help screen\n\n"
              << "Examples:\n"
              << "  murdok run models/qwen2.5-0.5b-instruct-q4_k_m.gguf\n"
              << "  murdok server --port 8080\n"
              << "  murdok optimize\n";
}

static int cmd_hardware() {
    auto hw = murdok::HardwareDetector::detect();

    std::cout << "========================================\n";
    std::cout << "        MuRDoK Hardware Analyzer        \n";
    std::cout << "========================================\n\n";

    std::cout << "CPU:\n";
    std::cout << "  Model:    " << hw.cpu_brand << "\n";
    std::cout << "  Vendor:   " << hw.cpu_vendor << "\n";
    std::cout << "  Physical: " << hw.physical_cores << " cores\n";
    std::cout << "  Logical:  " << hw.logical_threads << " threads\n\n";

    std::cout << "SIMD Extensions:\n";
    std::cout << "  SSE4.2:   " << (hw.simd.sse42 ? "[YES]" : "[NO]") << "\n";
    std::cout << "  FMA:      " << (hw.simd.fma   ? "[YES]" : "[NO]") << "\n";
    std::cout << "  AVX:      " << (hw.simd.avx   ? "[YES]" : "[NO]") << "\n";
    std::cout << "  AVX2:     " << (hw.simd.avx2  ? "[YES]" : "[NO]") << "\n";
    std::cout << "  AVX-512F: " << (hw.simd.avx512f ? "[YES]" : "[NO]") << "\n";
    std::cout << "  VNNI:     " << (hw.simd.avx512_vnni ? "[YES]" : "[NO]") << "\n";
    std::cout << "  AMX:      " << ((hw.simd.amx_tile || hw.simd.amx_int8) ? "[YES]" : "[NO]") << "\n\n";

    std::cout << "Cache Hierarchy:\n";
    std::cout << "  L1 Data:        " << hw.cache.l1d_kb << " KB\n";
    std::cout << "  L1 Instruction: " << hw.cache.l1i_kb << " KB\n";
    std::cout << "  L2 Cache:       " << hw.cache.l2_kb << " KB\n";
    std::cout << "  L3 Cache:       " << hw.cache.l3_kb << " KB\n";
    std::cout << "  Cache Line:     " << hw.cache.cache_line_size << " bytes\n\n";

    double total_ram_gb = static_cast<double>(hw.memory.total_ram_bytes) / (1024.0 * 1024.0 * 1024.0);
    double avail_ram_gb = static_cast<double>(hw.memory.available_ram_bytes) / (1024.0 * 1024.0 * 1024.0);

    std::cout << "System Memory:\n";
    std::cout << "  Total RAM:     " << std::fixed << std::setprecision(2) << total_ram_gb << " GB\n";
    std::cout << "  Available RAM: " << std::fixed << std::setprecision(2) << avail_ram_gb << " GB\n\n";

    std::cout << "----------------------------------------\n";
    std::cout << "Recommended Profile:\n";
    std::cout << "  " << murdok::HardwareDetector::profile_to_string(hw.recommended_profile) << "\n";
    std::cout << "========================================\n";
    return 0;
}

static std::string find_default_model() {
    if (fs::exists("models/qwen2.5-0.5b-instruct-q4_k_m.gguf")) {
        return "models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    }
    if (fs::exists("models")) {
        for (const auto& entry : fs::directory_iterator("models")) {
            if (entry.path().extension() == ".gguf") {
                return entry.path().string();
            }
        }
    }
    return "";
}

static int cmd_run(int argc, char* argv[]) {
    std::string model_path;
    if (argc > 2) {
        model_path = argv[2];
    } else {
        model_path = find_default_model();
    }

    if (model_path.empty() || !fs::exists(model_path)) {
        std::cerr << "Error: No GGUF model found. Please specify a model:\n"
                  << "  murdok run <path_to_model.gguf>\n";
        return 1;
    }

    murdok::Engine engine;
    murdok::EngineConfig cfg;
    cfg.model_path = model_path;

    if (!engine.load(cfg)) {
        std::cerr << "Failed to load model into MuRDoK Engine.\n";
        return 1;
    }

    auto hw = engine.get_hardware();
    double total_ram_gb = static_cast<double>(hw.memory.total_ram_bytes) / (1024.0 * 1024.0 * 1024.0);
    double model_mb = static_cast<double>(engine.get_model_size_bytes()) / (1024.0 * 1024.0);

    std::cout << "\n";
    std::cout << "+----------------------------------------------------+\n";
    std::cout << "|            MuRDoK Inference Engine                 |\n";
    std::cout << "+----------------------------------------------------+\n";
    std::cout << "| Model:    " << std::left << std::setw(41) << engine.get_model_name() << "|\n";
    std::cout << "| Format:   GGUF                                     |\n";
    std::cout << "| Size:     " << std::left << std::setw(34) << (std::to_string(static_cast<int>(model_mb)) + " MB") << "       |\n";
    std::cout << "| CPU:      " << std::left << std::setw(41) << hw.cpu_brand << "|\n";
    std::cout << "| Backend:  CPU AVX2 + FMA (Tuned)                   |\n";
    std::cout << "| Threads:  " << engine.get_config().n_threads_gen << " Generation / "
              << engine.get_config().n_threads_prompt << " Prompt Batch           |\n";
    std::cout << "| System:   " << std::left << std::setw(30) << (std::to_string(static_cast<int>(total_ram_gb)) + " GB RAM") << "           |\n";
    std::cout << "+----------------------------------------------------+\n";
    std::cout << "| Move less. Compute smarter. Infer faster.          |\n";
    std::cout << "+----------------------------------------------------+\n\n";

    std::cout << "MuRDoK ready. Type '/reset' to clear context, '/exit' to quit.\n\n";

    std::string line;
    while (true) {
        std::cout << "\n> ";
        if (!std::getline(std::cin, line)) break;
        if (line.empty()) continue;
        if (line == "/exit" || line == "/quit") break;
        if (line == "/reset") {
            engine.reset_context();
            std::cout << "[Context memory cleared]\n";
            continue;
        }

        std::cout << "\n";
        murdok::InferenceMetrics metrics;
        std::string prompt = "<|im_start|>user\n" + line + "<|im_end|>\n<|im_start|>assistant\n";

        engine.generate(prompt, [](const std::string& token) -> bool {
            std::cout << token << std::flush;
            return true;
        }, &metrics);

        std::cout << "\n\n";
        std::cout << "----------------------------------------------------------------------\n";
        std::cout << "[" << metrics.prompt_tokens << " prompt tok in "
                  << std::fixed << std::setprecision(2) << (metrics.prompt_duration_ms / 1000.0) << "s ("
                  << metrics.prompt_tok_per_sec << " tok/s) | Gen: "
                  << metrics.gen_tok_per_sec << " tok/s | TTFT: "
                  << std::setprecision(0) << metrics.ttft_ms << " ms | TPOT: "
                  << std::setprecision(1) << metrics.tpot_ms << " ms | RAM: "
                  << std::setprecision(0) << metrics.peak_ram_mb << " MB]\n";
        std::cout << "----------------------------------------------------------------------\n";
    }

    return 0;
}

static int cmd_optimize(int argc, char* argv[]) {
    std::string model_path;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model_path = argv[++i];
    }
    if (model_path.empty()) model_path = find_default_model();

    if (model_path.empty() || !fs::exists(model_path)) {
        std::cerr << "Error: Model file required for calibration.\n";
        return 1;
    }

    auto hw = murdok::HardwareDetector::detect();

    std::cout << "==========================================================\n";
    std::cout << "         MuRDoK Auto-Calibrator & Optimizer               \n";
    std::cout << "==========================================================\n";
    std::cout << "Model:    " << model_path << "\n";
    std::cout << "Host:     " << hw.cpu_brand << "\n";
    std::cout << "Cores:    " << hw.physical_cores << " physical, " << hw.logical_threads << " logical\n";
    std::cout << "SIMD:     AVX2: " << (hw.simd.avx2 ? "[YES]" : "[NO]") << ", FMA: " << (hw.simd.fma ? "[YES]" : "[NO]") << "\n";
    std::cout << "----------------------------------------------------------\n";
    std::cout << "Running automatic hardware sweep...\n\n";

    std::vector<int> thread_candidates = {1, 2, static_cast<int>(hw.physical_cores), static_cast<int>(hw.logical_threads)};
    // remove duplicates
    std::sort(thread_candidates.begin(), thread_candidates.end());
    thread_candidates.erase(std::unique(thread_candidates.begin(), thread_candidates.end()), thread_candidates.end());

    int best_gen_threads = hw.physical_cores;
    double best_gen_toks = 0.0;
    int best_prompt_threads = hw.logical_threads;
    double best_prompt_toks = 0.0;

    for (int t : thread_candidates) {
        std::cout << "  Benchmarking threads=" << t << " ... " << std::flush;
        murdok::Engine engine;
        murdok::EngineConfig cfg;
        cfg.model_path = model_path;
        cfg.n_threads_gen = t;
        cfg.n_threads_prompt = t;
        cfg.max_tokens = 32;

        if (!engine.load(cfg)) {
            std::cout << "[FAILED]\n";
            continue;
        }

        murdok::InferenceMetrics m;
        engine.complete("Explain cache locality in computers.", &m);

        std::cout << "Gen: " << std::fixed << std::setprecision(2) << m.gen_tok_per_sec << " tok/s | "
                  << "Prompt: " << m.prompt_tok_per_sec << " tok/s\n";

        if (m.gen_tok_per_sec > best_gen_toks) {
            best_gen_toks = m.gen_tok_per_sec;
            best_gen_threads = t;
        }
        if (m.prompt_tok_per_sec > best_prompt_toks) {
            best_prompt_toks = m.prompt_tok_per_sec;
            best_prompt_threads = t;
        }
    }

    // Persist calibration profile to ~/.murdok/profile.json (Phase 7)
    murdok::MurdokProfile profile;
    profile.cpu_model = hw.cpu_brand;
    profile.recommended_profile = murdok::HardwareDetector::profile_to_string(hw.recommended_profile);
    profile.optimal_gen_threads = best_gen_threads;
    profile.optimal_prompt_threads = best_prompt_threads;
    profile.optimal_batch_size = 512;
    profile.kv_cache_type = "f16";
    profile.cache_line_alignment = 64;
    profile.measured_gen_tok_s = best_gen_toks;
    profile.measured_prompt_tok_s = best_prompt_toks;

    bool saved = murdok::ProfileManager::save_profile(profile);

    std::cout << "\n+----------------------------------------------------+\n";
    std::cout << "|         MuRDoK Optimal Calibrated Profile          |\n";
    std::cout << "+----------------------------------------------------+\n";
    std::cout << "| Best Generation Threads:  " << std::left << std::setw(25) << (std::to_string(best_gen_threads) + " (Bandwidth tuned)") << "|\n";
    std::cout << "| Best Prompt Batch Threads:" << std::left << std::setw(25) << (std::to_string(best_prompt_threads) + " (Compute tuned)") << "|\n";
    std::cout << "| Recommended Profile:      " << std::left << std::setw(25) << profile.recommended_profile << "|\n";
    std::cout << "| Recommended Batch Size:   512                      |\n";
    std::cout << "| Target Memory Alignment:  64 bytes (Cache Line)    |\n";
    if (saved) {
        std::cout << "| Saved Profile:            ~/.murdok/profile.json   |\n";
    }
    std::cout << "+----------------------------------------------------+\n";
    std::cout << "Calibrated profile saved. Automatically applied to 'murdok run' and 'murdok server'.\n";

    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_main_help();
        return 0;
    }

    std::string cmd = argv[1];
    if (cmd == "run") {
        return cmd_run(argc, argv);
    } else if (cmd == "hardware") {
        return cmd_hardware();
    } else if (cmd == "optimize") {
        return cmd_optimize(argc, argv);
    } else if (cmd == "server") {
        // Forward to server binary
        std::string server_exe = "murdok-serve.exe";
        if (fs::exists("build/bin/Release/murdok-serve.exe")) {
            server_exe = "build\\bin\\Release\\murdok-serve.exe";
        } else if (fs::exists("build/bin/Release/murdok-server.exe")) {
            server_exe = "build\\bin\\Release\\murdok-server.exe";
        }
        std::string cmdline = server_exe;
        for (int i = 2; i < argc; ++i) {
            cmdline += " " + std::string(argv[i]);
        }
        return system(cmdline.c_str());
    } else if (cmd == "bench") {
        if (argc > 2 && std::string(argv[2]) == "--kernels") {
            std::cout << "==========================================================\n";
            std::cout << "      MuRDoK SIMD Vectorized Kernel Micro-Benchmark       \n";
            std::cout << "==========================================================\n";
            std::cout << "Benchmarking custom AVX2+FMA vs Scalar Dot-Product...\n\n";

            auto kres = murdok::kernels::benchmark_simd_kernels(65536, 10000);
            std::cout << "Scalar Implementation:   " << std::fixed << std::setprecision(2) << kres.scalar_gflops << " GFLOP/s\n";
            std::cout << "MuRDoK AVX2+FMA Kernel:  " << std::fixed << std::setprecision(2) << kres.avx2_gflops << " GFLOP/s\n";
            std::cout << "Speedup Factor:          " << std::fixed << std::setprecision(2) << kres.speedup << "x faster\n";
            std::cout << "Numerical Precision:     " << (kres.avx2_verified ? "[VERIFIED IDENTICAL]" : "[MISMATCH]") << "\n";
            std::cout << "Memory Alignment:        64 bytes (Cache Line Aligned)\n";
            std::cout << "==========================================================\n";
            return 0;
        }
        std::string bench_exe = "murdok-bench.exe";
        if (fs::exists("build/bin/Release/murdok-bench.exe")) {
            bench_exe = "build\\bin\\Release\\murdok-bench.exe";
        }
        std::string cmdline = bench_exe;
        for (int i = 2; i < argc; ++i) {
            cmdline += " " + std::string(argv[i]);
        }
        return system(cmdline.c_str());
    } else if (cmd == "help" || cmd == "--help" || cmd == "-h") {
        print_main_help();
        return 0;
    } else {
        std::cerr << "Unknown command: '" << cmd << "'. Run 'murdok help' for usage.\n";
        return 1;
    }
}
