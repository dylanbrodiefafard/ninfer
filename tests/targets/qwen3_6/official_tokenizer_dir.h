#pragma once

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>

// Official HF tokenizer files are a local maintainer prerequisite, not a CTest
// fixture. Tests that need them skip; synthetic tokenizer coverage still runs.
inline const std::optional<std::string>& official_tokenizer_dir() {
    static const std::optional<std::string> dir = []() -> std::optional<std::string> {
        const char* env = std::getenv("NINFER_OFFICIAL_TOKENIZER_DIR");
        const char* candidates[] = {
            env,
            "/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16",
            "/ssdpool2nvme/local_llm/ninfer-dylan2/profiles/bench/official-tokenizer",
        };
        for (const char* path : candidates) {
            if (path == nullptr || path[0] == '\0') { continue; }
            std::ifstream stream(std::string(path) + "/tokenizer.json", std::ios::binary);
            if (stream) { return std::string(path); }
        }
        return std::nullopt;
    }();
    return dir;
}

inline bool skip_without_official_tokenizer(const char* test_name) {
    if (official_tokenizer_dir()) { return false; }
    std::cout << "SKIP " << test_name
              << ": official tokenizer.json not found (set NINFER_OFFICIAL_TOKENIZER_DIR)\n";
    return true;
}
