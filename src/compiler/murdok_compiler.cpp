#include "murdok/murdok_format.h"
#include "gguf.h"
#include "nlohmann/json.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <filesystem>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace murdok {
namespace format {

static size_t pad_to_alignment(size_t size, size_t alignment = 64) {
    size_t rem = size % alignment;
    return (rem == 0) ? 0 : (alignment - rem);
}

bool MurdokCompiler::compile_gguf_to_murdok(
    const std::string& input_gguf_path,
    const std::string& output_murdok_path,
    TargetArch target_arch
) {
    if (!fs::exists(input_gguf_path)) {
        std::cerr << "[MuRDoK Compiler] Error: Input file does not exist: " << input_gguf_path << "\n";
        return false;
    }

    struct gguf_init_params params = {true, nullptr};
    struct gguf_context* gctx = gguf_init_from_file(input_gguf_path.c_str(), params);
    if (!gctx) {
        std::cerr << "[MuRDoK Compiler] Error: Failed to parse GGUF headers\n";
        return false;
    }

    int64_t n_tensors = gguf_get_n_tensors(gctx);
    int64_t n_kv = gguf_get_n_kv(gctx);
    size_t gguf_data_offset = gguf_get_data_offset(gctx);

    std::cout << "[MuRDoK Compiler] Reading GGUF: " << input_gguf_path << "\n";
    std::cout << "[MuRDoK Compiler] Found " << n_tensors << " tensors, " << n_kv << " metadata keys\n";

    // 1. Build Metadata JSON
    json meta;
    meta["source_format"] = "GGUF";
    meta["target_arch"] = static_cast<uint32_t>(target_arch);
    meta["tensor_count"] = n_tensors;
    meta["alignment"] = 64;

    for (int64_t i = 0; i < n_kv; ++i) {
        const char* key = gguf_get_key(gctx, i);
        enum gguf_type type = gguf_get_kv_type(gctx, i);
        if (type == GGUF_TYPE_STRING) {
            meta["kv"][key] = gguf_get_val_str(gctx, i);
        } else if (type == GGUF_TYPE_UINT32) {
            meta["kv"][key] = gguf_get_val_u32(gctx, i);
        } else if (type == GGUF_TYPE_INT32) {
            meta["kv"][key] = gguf_get_val_i32(gctx, i);
        } else if (type == GGUF_TYPE_FLOAT32) {
            meta["kv"][key] = gguf_get_val_f32(gctx, i);
        }
    }

    std::string meta_str = meta.dump();

    // 2. Open Files
    std::ofstream out(output_murdok_path, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "[MuRDoK Compiler] Error: Cannot create output file: " << output_murdok_path << "\n";
        gguf_free(gctx);
        return false;
    }

    std::ifstream in(input_gguf_path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "[MuRDoK Compiler] Error: Cannot open input file stream\n";
        gguf_free(gctx);
        return false;
    }

    // 3. Prepare Header
    MurdokHeader hdr;
    std::memset(&hdr, 0, sizeof(hdr));
    std::memcpy(hdr.magic, MURDOK_MAGIC, 8);
    hdr.version = MURDOK_VERSION;
    hdr.target_arch = static_cast<uint32_t>(target_arch);
    hdr.tensor_count = static_cast<uint32_t>(n_tensors);
    hdr.alignment = 64;

    hdr.metadata_offset = sizeof(MurdokHeader);
    hdr.metadata_size = meta_str.size();

    size_t meta_pad = pad_to_alignment(sizeof(MurdokHeader) + hdr.metadata_size, 64);
    hdr.tensor_dir_offset = hdr.metadata_offset + hdr.metadata_size + meta_pad;

    size_t dir_size = sizeof(MurdokTensorEntry) * n_tensors;
    size_t dir_pad = pad_to_alignment(hdr.tensor_dir_offset + dir_size, 64);
    hdr.tensor_data_offset = hdr.tensor_dir_offset + dir_size + dir_pad;

    // Reserve header space
    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));

    // Write metadata
    out.write(meta_str.data(), meta_str.size());
    std::vector<char> pad_bytes(64, 0);
    if (meta_pad > 0) out.write(pad_bytes.data(), meta_pad);

    // Build and write tensor directory placeholder
    std::vector<MurdokTensorEntry> entries(n_tensors);
    uint64_t current_data_offset = hdr.tensor_data_offset;

    for (int64_t i = 0; i < n_tensors; ++i) {
        MurdokTensorEntry& e = entries[i];
        std::memset(&e, 0, sizeof(e));

        const char* tname = gguf_get_tensor_name(gctx, i);
        std::strncpy(e.name, tname, sizeof(e.name) - 1);

        const int64_t* ne = gguf_get_tensor_ne(gctx, i);
        e.n_dims = 4;
        for (int d = 0; d < 4; ++d) e.dims[d] = ne[d];

        e.type = static_cast<uint32_t>(gguf_get_tensor_type(gctx, i));
        e.size_bytes = gguf_get_tensor_size(gctx, i);
        e.offset = current_data_offset;

        size_t t_pad = pad_to_alignment(e.size_bytes, 64);
        current_data_offset += e.size_bytes + t_pad;
    }

    out.write(reinterpret_cast<const char*>(entries.data()), dir_size);
    if (dir_pad > 0) out.write(pad_bytes.data(), dir_pad);

    // 4. Stream and align tensor data payloads
    std::cout << "[MuRDoK Compiler] Packing and aligning tensors to 64-byte boundaries...\n";
    std::vector<char> chunk(4 * 1024 * 1024); // 4MB stream buffer

    for (int64_t i = 0; i < n_tensors; ++i) {
        size_t src_offset = gguf_data_offset + gguf_get_tensor_offset(gctx, i);
        size_t t_size = entries[i].size_bytes;

        in.seekg(src_offset, std::ios::beg);

        size_t remaining = t_size;
        while (remaining > 0) {
            size_t to_read = std::min(remaining, chunk.size());
            in.read(chunk.data(), to_read);
            out.write(chunk.data(), to_read);
            remaining -= to_read;
        }

        size_t t_pad = pad_to_alignment(t_size, 64);
        if (t_pad > 0) {
            out.write(pad_bytes.data(), t_pad);
        }

        if (i % 50 == 0 || i == n_tensors - 1) {
            std::cout << "\r[MuRDoK Compiler] Packed: " << (i + 1) << " / " << n_tensors << " tensors" << std::flush;
        }
    }
    std::cout << "\n";

    // 5. Finalize Header
    hdr.total_file_size = current_data_offset;
    out.seekp(0, std::ios::beg);
    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));

    out.close();
    in.close();
    gguf_free(gctx);

    std::cout << "[MuRDoK Compiler] Successfully compiled to: " << output_murdok_path << "\n";
    std::cout << "[MuRDoK Compiler] Total file size: " << (hdr.total_file_size / (1024.0 * 1024.0)) << " MB\n";
    return true;
}

bool MurdokCompiler::verify_murdok_file(const std::string& murdok_path, MurdokHeader* out_hdr) {
    if (!fs::exists(murdok_path)) return false;

    std::ifstream file(murdok_path, std::ios::binary);
    if (!file.is_open()) return false;

    MurdokHeader hdr;
    file.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (file.gcount() != sizeof(hdr)) return false;

    if (std::memcmp(hdr.magic, MURDOK_MAGIC, 8) != 0) {
        return false;
    }
    if (hdr.version != MURDOK_VERSION) {
        return false;
    }
    if (hdr.alignment != 64) {
        return false;
    }

    if (out_hdr) *out_hdr = hdr;
    return true;
}

} // namespace format
} // namespace murdok
