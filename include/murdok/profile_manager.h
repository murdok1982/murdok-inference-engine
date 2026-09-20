#pragma once

#include "murdok/hardware.h"
#include <string>

namespace murdok {

struct MurdokProfile {
    std::string cpu_model;
    std::string recommended_profile = "MURDOK_BALANCED";
    int optimal_gen_threads = 4;
    int optimal_prompt_threads = 8;
    int optimal_batch_size = 512;
    std::string kv_cache_type = "f16";
    int cache_line_alignment = 64;
    double measured_gen_tok_s = 0.0;
    double measured_prompt_tok_s = 0.0;
};

class ProfileManager {
public:
    static std::string get_profile_path();
    static bool save_profile(const MurdokProfile& profile);
    static bool load_profile(MurdokProfile& profile);
    static bool exists();
};

} // namespace murdok
