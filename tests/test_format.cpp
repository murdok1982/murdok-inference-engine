#include "murdok/murdok_format.h"
#include <cassert>
#include <iostream>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

void test_format_validation() {
    std::cout << "[TEST] Running Model Format Integrity Tests...\n";

    // 1. Non-existent file
    auto res_missing = murdok::format::MurdokCompiler::validate_container("non_existent_model.murdok");
    assert(!res_missing.is_valid);

    // 2. Truncated file (fewer than 64 bytes)
    std::string trunc_path = "test_trunc.murdok";
    {
        std::ofstream out(trunc_path, std::ios::binary);
        out << "SHORT";
    }
    auto res_trunc = murdok::format::MurdokCompiler::validate_container(trunc_path);
    assert(!res_trunc.is_valid);
    fs::remove(trunc_path);

    // 3. Corrupt magic
    std::string bad_magic_path = "test_bad_magic.murdok";
    {
        std::ofstream out(bad_magic_path, std::ios::binary);
        murdok::format::MurdokHeader bad_hdr{};
        const char bad_sig[8] = {'B', 'A', 'D', 'M', 'A', 'G', 'I', 'C'};
        std::memcpy(bad_hdr.magic, bad_sig, 8);
        bad_hdr.version = 1;
        bad_hdr.alignment = 64;
        out.write(reinterpret_cast<const char*>(&bad_hdr), sizeof(bad_hdr));
    }
    auto res_bad_magic = murdok::format::MurdokCompiler::validate_container(bad_magic_path);
    assert(!res_bad_magic.is_valid);
    fs::remove(bad_magic_path);

    // 4. Corrupt alignment
    std::string bad_align_path = "test_bad_align.murdok";
    {
        std::ofstream out(bad_align_path, std::ios::binary);
        murdok::format::MurdokHeader bad_hdr{};
        std::memcpy(bad_hdr.magic, murdok::format::MURDOK_MAGIC, 8);
        bad_hdr.version = 1;
        bad_hdr.alignment = 32; // Not 64!
        out.write(reinterpret_cast<const char*>(&bad_hdr), sizeof(bad_hdr));
    }
    auto res_bad_align = murdok::format::MurdokCompiler::validate_container(bad_align_path);
    assert(!res_bad_align.is_valid);
    fs::remove(bad_align_path);

    // 5. Existing compiled model check (if exists)
    if (fs::exists("models/qwen2.5-0.5b-instruct-q4_k_m.murdok")) {
        auto res_real = murdok::format::MurdokCompiler::validate_container("models/qwen2.5-0.5b-instruct-q4_k_m.murdok");
        assert(res_real.is_valid);
        assert(res_real.tensor_count > 0);
        assert(res_real.alignment == 64);
        std::cout << "  Real .murdok container: " << res_real.tensor_count << " tensors, verified 64B aligned.\n";
    }

    std::cout << "[PASS] Model Format Integrity Tests Passed.\n\n";
}
