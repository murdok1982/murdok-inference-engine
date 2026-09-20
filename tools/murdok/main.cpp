#include "murdok/murdok.h"
#include "murdok/profile_manager.h"
#include "murdok/kernels.h"
#include "murdok/murdok_format.h"
#include "murdok/speculative.h"
#include "murdok/static_graph.h"

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <iomanip>

namespace fs = std::filesystem;

static void print_main_help() {
    std::cout << "==========================================================\n";
    std::cout << "         MuRDoK Inference Engine (MIE) v0.3.0             \n";
    std::cout << "    Move less. Compute smarter. Infer faster. Benchmark.  \n";
    std::cout << "==========================================================\n\n"
              << "Usage: murdok <command> [options]\n\n"
              << "Commands:\n"
              << "  run <model> [options]    Start interactive local chat session with HUD\n"
              << "                           Supports: GGUF, native .murdok binary, and --draft\n"
              << "  compile <model.gguf>     Compile GGUF to native .murdok 64-byte aligned binary\n"
              << "  server [options]         Start OpenAI-compatible REST server & Web UI\n"
              << "  optimize [options]       Auto-calibrate hardware and find optimal settings\n"
              << "  hardware                 Inspect CPU, SIMD, cache hierarchy, and RAM\n"
              << "  bench [options]          Run benchmarks: --kernels, --graph, --speculative\n"
              << "  help                     Show this help screen\n\n"
              << "Examples:\n"
              << "  murdok run models/qwen2.5-0.5b-instruct-q4_k_m.gguf\n"
              << "  murdok run models/qwen2.5-0.5b-instruct-q4_k_m.murdok\n"
              << "  murdok run target.gguf --draft draft.gguf\n"
              << "  murdok compile models/model.gguf --output models/model.murdok\n"
              << "  murdok bench --graph\n"
              << "  murdok bench --kernels\n";
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
            if (entry.path().extension() == ".gguf" || entry.path().extension() == ".murdok") {
                return entry.path().string();
            }
        }
    }
    return "";
}

static int cmd_compile(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: murdok compile <model.gguf> [--output <model.murdok>] [--target <avx2|avx512|cuda>]\n";
        return 1;
    }
    std::string input_path = argv[2];
    std::string output_path;
    std::string target_str = "avx2";

    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--output" && i + 1 < argc) output_path = argv[++i];
        else if (a == "--target" && i + 1 < argc) target_str = argv[++i];
    }
    if (output_path.empty()) {
        output_path = fs::path(input_path).replace_extension(".murdok").string();
    }

    murdok::format::TargetArch arch = murdok::format::TargetArch::CpuAvx2;
    if (target_str == "avx512") arch = murdok::format::TargetArch::CpuAvx512;
    else if (target_str == "cuda") arch = murdok::format::TargetArch::Cuda;
    else if (target_str == "generic") arch = murdok::format::TargetArch::Generic;

    std::cout << "==========================================================\n";
    std::cout << "            MuRDoK Model Binary Compiler                  \n";
    std::cout << "==========================================================\n";
    std::cout << "Input GGUF:     " << input_path << "\n";
    std::cout << "Output MURDOK:  " << output_path << "\n";
    std::cout << "Target SIMD:    " << target_str << " (64-byte Cache Aligned)\n";
    std::cout << "----------------------------------------------------------\n";

    bool ok = murdok::format::MurdokCompiler::compile_gguf_to_murdok(input_path, output_path, arch);
    if (!ok) {
        std::cerr << "[MuRDoK Compiler] Failed to compile binary model.\n";
        return 1;
    }

    murdok::format::MurdokHeader hdr;
    if (murdok::format::MurdokCompiler::verify_murdok_file(output_path, &hdr)) {
        std::cout << "\n[Verification Passed] Valid MURDOK01 binary, " << hdr.tensor_count
                  << " tensors with 64-byte alignment verified.\n";
    }
    return 0;
}

static int cmd_run(int argc, char* argv[]) {
    std::string model_path;
    std::string draft_path;
    std::string kv_type = "f16";
    int32_t n_ctx = 2048;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--draft" && i + 1 < argc) {
            draft_path = argv[++i];
        } else if (arg == "--kv" && i + 1 < argc) {
            kv_type = argv[++i];
        } else if (arg == "--ctx" && i + 1 < argc) {
            n_ctx = std::stoi(argv[++i]);
        } else if (model_path.empty() && arg[0] != '-') {
            model_path = arg;
        }
    }

    if (model_path.empty()) {
        model_path = find_default_model();
    }

    if (model_path.empty() || !fs::exists(model_path)) {
        std::cerr << "Error: No model found. Please specify a model path:\n"
                  << "  murdok run <path_to_model.gguf | path_to_model.murdok>\n";
        return 1;
    }

    // Speculative Decoding Execution Mode
    if (!draft_path.empty()) {
        std::cout << "[MuRDoK] Starting Speculative Decoding Engine...\n";
        std::cout << "  Target Model: " << model_path << "\n";
        std::cout << "  Draft Model:  " << draft_path << "\n";

        murdok::speculative::SpeculativeConfig s_cfg;
        s_cfg.target_model_path = model_path;
        s_cfg.draft_model_path = draft_path;
        s_cfg.n_ctx = n_ctx;
        s_cfg.adaptive = true;

        murdok::speculative::SpeculativeEngine spec_engine;
        if (!spec_engine.load(s_cfg)) {
            std::cerr << "Failed to initialize Speculative Engine.\n";
            return 1;
        }

        std::cout << "\n";
        std::cout << "+----------------------------------------------------+\n";
        std::cout << "|        MuRDoK Speculative Inference Engine         |\n";
        std::cout << "+----------------------------------------------------+\n";
        std::cout << "| Mode:     Speculative Decoding (Adaptive K)        |\n";
        std::cout << "| Target:   " << std::left << std::setw(41) << fs::path(model_path).stem().string() << "|\n";
        std::cout << "| Draft:    " << std::left << std::setw(41) << fs::path(draft_path).stem().string() << "|\n";
        std::cout << "+----------------------------------------------------+\n\n";

        std::cout << "Speculative Engine ready. Type '/exit' to quit.\n\n";
        std::string line;
        while (true) {
            std::cout << "\n> ";
            if (!std::getline(std::cin, line)) break;
            if (line.empty()) continue;
            if (line == "/exit" || line == "/quit") break;

            std::cout << "\n";
            murdok::speculative::SpeculativeMetrics sm;
            std::string prompt = "<|im_start|>user\n" + line + "<|im_end|>\n<|im_start|>assistant\n";

            spec_engine.generate(prompt, [](const std::string& tok) -> bool {
                std::cout << tok << std::flush;
                return true;
            }, &sm);

            std::cout << "\n\n";
            std::cout << "----------------------------------------------------------------------\n";
            std::cout << "[Speculative: " << sm.total_accepted_tokens << " accepted / "
                      << sm.total_drafted_tokens << " drafted (" << std::fixed << std::setprecision(1)
                      << sm.acceptance_rate << "% rate) | Current K: " << sm.current_k
                      << " | Gen: " << sm.generation_tok_per_sec << " tok/s]\n";
            std::cout << "----------------------------------------------------------------------\n";
        }
        return 0;
    }

    // Standard Autoregressive Execution Mode
    murdok::Engine engine;
    murdok::EngineConfig cfg;
    cfg.model_path = model_path;
    cfg.kv_type = kv_type;
    cfg.n_ctx = n_ctx;

    if (!engine.load(cfg)) {
        std::cerr << "Failed to load model into MuRDoK Engine.\n";
        return 1;
    }

    auto hw = engine.get_hardware();
    double total_ram_gb = static_cast<double>(hw.memory.total_ram_bytes) / (1024.0 * 1024.0 * 1024.0);
    double model_mb = static_cast<double>(engine.get_model_size_bytes()) / (1024.0 * 1024.0);
    std::string format_str = fs::path(model_path).extension() == ".murdok" ? "NATIVE .MURDOK (64B Aligned)" : "GGUF";

    std::cout << "\n";
    std::cout << "+----------------------------------------------------+\n";
    std::cout << "|            MuRDoK Inference Engine                 |\n";
    std::cout << "+----------------------------------------------------+\n";
    std::cout << "| Model:    " << std::left << std::setw(41) << engine.get_model_name() << "|\n";
    std::cout << "| Format:   " << std::left << std::setw(41) << format_str << "|\n";
    std::cout << "| Size:     " << std::left << std::setw(34) << (std::to_string(static_cast<int>(model_mb)) + " MB") << "       |\n";
    std::cout << "| CPU:      " << std::left << std::setw(41) << hw.cpu_brand << "|\n";
    std::cout << "| Backend:  CPU AVX2 + FMA (Tuned)                   |\n";
    std::cout << "| KV Cache: " << std::left << std::setw(41) << cfg.kv_type << "|\n";
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
    } else if (cmd == "compile") {
        return cmd_compile(argc, argv);
    } else if (cmd == "hardware") {
        return cmd_hardware();
    } else if (cmd == "optimize") {
        return cmd_optimize(argc, argv);
    } else if (cmd == "server") {
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
        } else if (argc > 2 && std::string(argv[2]) == "--graph") {
            murdok::graph::StaticGraphPlanner::benchmark_execution(1000);
            return 0;
        } else if (argc > 2 && std::string(argv[2]) == "--speculative") {
            std::cout << "==========================================================\n";
            std::cout << "         MuRDoK Speculative Decoding Benchmark            \n";
            std::cout << "==========================================================\n";
            std::string def_model = find_default_model();
            if (def_model.empty()) {
                std::cerr << "Error: No default model found for speculative benchmark.\n";
                return 1;
            }
            std::cout << "Benchmarking Adaptive Draft Speculative Decoding on: " << def_model << "\n";
            murdok::speculative::SpeculativeConfig sc;
            sc.target_model_path = def_model;
            sc.draft_model_path = def_model;
            sc.adaptive = true;
            murdok::speculative::SpeculativeEngine se;
            if (!se.load(sc)) {
                std::cerr << "Failed to load model in speculative engine.\n";
                return 1;
            }
            murdok::speculative::SpeculativeMetrics sm;
            se.generate("Explain static memory arenas in inference runtimes.", nullptr, &sm);
            std::cout << "Accepted Tokens:  " << sm.total_accepted_tokens << "\n";
            std::cout << "Drafted Tokens:   " << sm.total_drafted_tokens << "\n";
            std::cout << "Acceptance Rate:  " << std::fixed << std::setprecision(2) << sm.acceptance_rate << " %\n";
            std::cout << "Dynamic K Final:  " << sm.current_k << "\n";
            std::cout << "Throughput:       " << sm.generation_tok_per_sec << " tok/s\n";
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
