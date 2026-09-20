#include "murdok/murdok_format.h"
#include <iostream>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

static void print_usage(const char* exe) {
    std::cout << "Usage: " << exe << " <model.gguf> [options]\n\n"
              << "Options:\n"
              << "  --output <path>    Output .murdok path (default: <model>.murdok)\n"
              << "  --target <arch>    Target architecture: auto, avx2, avx512, cuda (default: avx2)\n"
              << "  --help             Show this help screen\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string input_path;
    std::string output_path;
    std::string target_str = "avx2";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--target" && i + 1 < argc) {
            target_str = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (input_path.empty() && arg[0] != '-') {
            input_path = arg;
        }
    }

    if (input_path.empty()) {
        std::cerr << "Error: Input GGUF path is required.\n";
        return 1;
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

    bool success = murdok::format::MurdokCompiler::compile_gguf_to_murdok(input_path, output_path, arch);
    if (!success) {
        std::cerr << "Compilation failed.\n";
        return 1;
    }

    // Verify written file
    murdok::format::MurdokHeader hdr;
    if (murdok::format::MurdokCompiler::verify_murdok_file(output_path, &hdr)) {
        std::cout << "\n[Verification Passed] Valid MURDOK01 binary, " << hdr.tensor_count
                  << " tensors with 64-byte alignment.\n";
    }

    return 0;
}
