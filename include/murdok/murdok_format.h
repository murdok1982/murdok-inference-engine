#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace murdok {
namespace format {

// Magic 8-byte signature: 'M','U','R','D','O','K','0','1'
constexpr uint8_t MURDOK_MAGIC[8] = {'M', 'U', 'R', 'D', 'O', 'K', '0', '1'};
constexpr uint32_t MURDOK_VERSION = 1;

enum class TargetArch : uint32_t {
    Generic = 0,
    CpuAvx2 = 1,
    CpuAvx512 = 2,
    Cuda = 3,
    Metal = 4
};

#pragma pack(push, 1)
struct MurdokHeader {
    uint8_t magic[8];             // 'MURDOK01'
    uint32_t version;             // 1
    uint32_t target_arch;         // TargetArch enum
    uint32_t tensor_count;        // Total tensors packaged
    uint32_t alignment;           // Cache line alignment (64 bytes)
    uint64_t metadata_offset;     // File offset to JSON metadata string
    uint64_t metadata_size;       // Size of JSON metadata in bytes
    uint64_t tensor_dir_offset;   // File offset to tensor directory
    uint64_t tensor_data_offset;  // File offset to aligned tensor payloads
    uint64_t total_file_size;     // Total file size
};

struct MurdokTensorEntry {
    char name[128];               // Tensor name (null terminated)
    uint32_t n_dims;              // Number of dimensions (up to 4)
    uint64_t dims[4];             // Dimension sizes
    uint32_t type;                // GGML/Murdok data type (e.g. Q4_K, Q8_0, F16)
    uint64_t offset;              // Absolute offset in file
    uint64_t size_bytes;          // Size in bytes
};
#pragma pack(pop)

struct ContainerValidationResult {
    bool is_valid = false;
    std::string error_message;
    uint32_t tensor_count = 0;
    uint32_t alignment = 0;
    uint32_t version = 0;
    uint64_t total_size = 0;
};

class MurdokCompiler {
public:
    static bool compile_gguf_to_murdok(
        const std::string& input_gguf_path,
        const std::string& output_murdok_path,
        TargetArch target_arch = TargetArch::CpuAvx2
    );

    static bool verify_murdok_file(const std::string& murdok_path, MurdokHeader* out_hdr = nullptr);
    static ContainerValidationResult validate_container(const std::string& murdok_path);
};

} // namespace format
} // namespace murdok
