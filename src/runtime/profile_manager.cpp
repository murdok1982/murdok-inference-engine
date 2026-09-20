#include "murdok/profile_manager.h"
#include "nlohmann/json.hpp"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <cstdlib>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace murdok {

std::string ProfileManager::get_profile_path() {
    std::string home;
#if defined(_WIN32)
    const char* userprofile = std::getenv("USERPROFILE");
    if (userprofile) {
        home = userprofile;
    } else {
        const char* homedrive = std::getenv("HOMEDRIVE");
        const char* homepath = std::getenv("HOMEPATH");
        if (homedrive && homepath) {
            home = std::string(homedrive) + std::string(homepath);
        } else {
            home = ".";
        }
    }
#else
    const char* h = std::getenv("HOME");
    home = h ? h : ".";
#endif

    fs::path dir = fs::path(home) / ".murdok";
    return (dir / "profile.json").string();
}

bool ProfileManager::exists() {
    return fs::exists(get_profile_path());
}

bool ProfileManager::save_profile(const MurdokProfile& profile) {
    try {
        std::string p_path = get_profile_path();
        fs::create_directories(fs::path(p_path).parent_path());

        json j;
        j["cpu_model"] = profile.cpu_model;
        j["recommended_profile"] = profile.recommended_profile;
        j["optimal_gen_threads"] = profile.optimal_gen_threads;
        j["optimal_prompt_threads"] = profile.optimal_prompt_threads;
        j["optimal_batch_size"] = profile.optimal_batch_size;
        j["kv_cache_type"] = profile.kv_cache_type;
        j["cache_line_alignment"] = profile.cache_line_alignment;
        j["measured_gen_tok_s"] = profile.measured_gen_tok_s;
        j["measured_prompt_tok_s"] = profile.measured_prompt_tok_s;

        std::ofstream file(p_path);
        if (!file.is_open()) return false;
        file << j.dump(4);
        return true;
    } catch (...) {
        return false;
    }
}

bool ProfileManager::load_profile(MurdokProfile& profile) {
    try {
        std::string p_path = get_profile_path();
        if (!fs::exists(p_path)) return false;

        std::ifstream file(p_path);
        if (!file.is_open()) return false;

        json j = json::parse(file);
        profile.cpu_model = j.value("cpu_model", "");
        profile.recommended_profile = j.value("recommended_profile", "MURDOK_BALANCED");
        profile.optimal_gen_threads = j.value("optimal_gen_threads", 4);
        profile.optimal_prompt_threads = j.value("optimal_prompt_threads", 8);
        profile.optimal_batch_size = j.value("optimal_batch_size", 512);
        profile.kv_cache_type = j.value("kv_cache_type", "f16");
        profile.cache_line_alignment = j.value("cache_line_alignment", 64);
        profile.measured_gen_tok_s = j.value("measured_gen_tok_s", 0.0);
        profile.measured_prompt_tok_s = j.value("measured_prompt_tok_s", 0.0);
        return true;
    } catch (...) {
        return false;
    }
}

} // namespace murdok
