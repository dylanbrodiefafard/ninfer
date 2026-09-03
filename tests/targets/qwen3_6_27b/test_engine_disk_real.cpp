#include "ninfer/engine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

constexpr std::size_t kRamBytes      = 1024ULL * 1024ULL * 1024ULL;
constexpr std::size_t kOneEntryBytes = 220ULL * 1024ULL * 1024ULL;
constexpr std::size_t kDiskBytes     = 4ULL * 1024ULL * 1024ULL * 1024ULL;

const char* source_name(ninfer::PrefixReuseSource source) {
    switch (source) {
    case ninfer::PrefixReuseSource::None:
        return "none";
    case ninfer::PrefixReuseSource::VramResident:
        return "vram";
    case ninfer::PrefixReuseSource::HostRam:
        return "ram";
    case ninfer::PrefixReuseSource::HostDisk:
        return "disk";
    }
    return "unknown";
}

ninfer::RequestOptions greedy(std::uint32_t outputs, bool reuse) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

std::vector<ninfer::TokenId> tokens_a() { return {248045, 846, 198, 5834, 248046, 198}; }

std::vector<ninfer::TokenId> tokens_b() { return {248045, 846, 198, 9906, 248046, 198}; }

std::vector<ninfer::TokenId> tokens_c() { return {248045, 846, 198, 1243, 248046, 198}; }

std::vector<ninfer::TokenId> tokens_d() { return {248045, 846, 198, 220, 248046, 198}; }

std::vector<ninfer::TokenId> tokens_e() { return {248045, 846, 198, 221, 248046, 198}; }

std::vector<ninfer::TokenId> tokens_f() { return {248045, 846, 198, 222, 248046, 198}; }

std::vector<ninfer::TokenId> tokens_g() { return {248045, 846, 198, 223, 248046, 198}; }

std::vector<ninfer::TokenId> tokens_h() { return {248045, 846, 198, 224, 248046, 198}; }

std::vector<ninfer::TokenId> concat(std::vector<ninfer::TokenId> prefix,
                                    const std::vector<ninfer::TokenId>& suffix) {
    prefix.insert(prefix.end(), suffix.begin(), suffix.end());
    return prefix;
}

std::vector<ninfer::TokenId> resume_prefix(const std::vector<ninfer::TokenId>& keep,
                                           const std::vector<ninfer::TokenId>& generated) {
    std::vector<ninfer::TokenId> prefix = keep;
    if (!generated.empty()) {
        prefix.insert(prefix.end(), generated.begin(), generated.end() - 1);
    }
    return prefix;
}

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

class ScopedDiskDir {
public:
    explicit ScopedDiskDir(const char* tag)
        : path_(std::filesystem::temp_directory_path() /
                ("ninfer-engine-disk-" + std::to_string(::getpid()) + "-" + tag)) {
        std::filesystem::remove_all(path_);
        std::filesystem::create_directories(path_);
    }
    ~ScopedDiskDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }
    ScopedDiskDir(const ScopedDiskDir&) = delete;
    ScopedDiskDir& operator=(const ScopedDiskDir&) = delete;
    ScopedDiskDir(ScopedDiskDir&&) = delete;
    ScopedDiskDir& operator=(ScopedDiskDir&&) = delete;
    operator const std::filesystem::path&() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

ScopedDiskDir make_disk_dir(const char* tag) { return ScopedDiskDir(tag); }

ninfer::EngineOptions disk_options(const char* artifact, const std::filesystem::path& disk,
                                   std::uint32_t max_concurrency, std::size_t ram_bytes) {
    ninfer::EngineOptions options;
    options.artifact_path          = artifact;
    options.max_context            = 4096;
    options.kv_capacity            = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.max_concurrency        = max_concurrency;
    options.prefill_chunk          = 1024;
    options.kv_ram_capacity_bytes  = ram_bytes;
    options.kv_disk_capacity_bytes = kDiskBytes;
    options.kv_disk_location       = disk;
    options.kv_cache               = ninfer::KvCacheStorage::Int8Group64;
    options.enable_vision          = false;
    options.use_cuda_graph         = true;
    options.pending_timeout_ms     = 120000;
    return options;
}

ninfer::EngineOptions dflash_disk_options(const char* artifact, const std::filesystem::path& disk,
                                          std::uint32_t max_concurrency, std::size_t ram_bytes) {
    ninfer::EngineOptions options     = disk_options(artifact, disk, max_concurrency, ram_bytes);
    options.speculative.backend       = ninfer::SpeculativeBackend::DFlash;
    options.speculative.draft_tokens  = 4;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    return options;
}

int verify_disk_tier(const ninfer::Engine& engine, std::size_t ram_bytes) {
    const ninfer::MemorySummary memory = engine.memory_summary();
    if (memory.kv_ram_capacity_bytes != ram_bytes) {
        std::cerr << "KV RAM capacity is " << memory.kv_ram_capacity_bytes << ", expected "
                  << ram_bytes << '\n';
        return 1;
    }
    if (memory.kv_disk_capacity_bytes != kDiskBytes) {
        std::cerr << "KV disk capacity is " << memory.kv_disk_capacity_bytes << ", expected "
                  << kDiskBytes << '\n';
        return 1;
    }
    return 0;
}

int expect_hit(const ninfer::GenerationResult& result, ninfer::PrefixReuseSource source,
                std::uint32_t history_tokens, const char* label) {
    if (result.prefix_reuse_source != source) {
        std::cerr << label << " reuse_source is " << source_name(result.prefix_reuse_source)
                  << ", expected " << source_name(source) << '\n';
        return 1;
    }
    if (result.prefix_reuse_path == ninfer::PrefixReusePath::FullReset) {
        std::cerr << label << " " << source_name(source) << " hit used FullReset\n";
        return 1;
    }
    if (result.reused_prompt_tokens != history_tokens) {
        std::cerr << label << " reused " << result.reused_prompt_tokens << ", expected "
                  << history_tokens << '\n';
        return 1;
    }
    return 0;
}

int expect_suffix_hit(const ninfer::GenerationResult& result, ninfer::PrefixReuseSource source,
                      std::uint32_t prefix_tokens, std::uint32_t prompt_tokens, const char* label) {
    if (prefix_tokens >= prompt_tokens) {
        std::cerr << label << " suffix fixture is not longer than the reused prefix\n";
        return 1;
    }
    if (result.prompt.prompt_tokens != prompt_tokens) {
        std::cerr << label << " prompt_tokens is " << result.prompt.prompt_tokens << ", expected "
                  << prompt_tokens << '\n';
        return 1;
    }
    return expect_hit(result, source, prefix_tokens, label);
}

bool wait_scheduler(ninfer::Engine& engine, std::uint32_t* max_prefilling, const auto& predicate,
                     const char* label) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(180);
    while (std::chrono::steady_clock::now() < deadline) {
        const ninfer::RuntimeStats stats = engine.runtime_stats();
        *max_prefilling = std::max(*max_prefilling, stats.prefilling_requests);
        if (stats.prefilling_requests > 1) {
            std::cerr << label << " observed prefilling_requests=" << stats.prefilling_requests
                      << '\n';
            return false;
        }
        if (predicate(stats)) { return true; }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    const ninfer::RuntimeStats stats = engine.runtime_stats();
    std::cerr << label << " timed out: running=" << stats.running_requests
              << " prefilling=" << stats.prefilling_requests
              << " decode_ready=" << stats.decode_ready_requests
              << " waiting=" << stats.waiting_requests << '\n';
    return false;
}

void wait_idle(ninfer::Engine& engine) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (std::chrono::steady_clock::now() < deadline) {
        const auto stats = engine.runtime_stats();
        if (stats.running_requests == 0 && stats.waiting_requests == 0 &&
            stats.prefilling_requests == 0) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

bool wait_disk_captures(ninfer::Engine& engine, std::uint64_t minimum, const char* label) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (std::chrono::steady_clock::now() < deadline) {
        const auto stats = engine.runtime_stats();
        if (stats.kv_disk_captures >= minimum) { return true; }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    const auto stats = engine.runtime_stats();
    std::cerr << label << " timed out waiting for kv_disk_captures>=" << minimum
              << " (have " << stats.kv_disk_captures << " drops=" << stats.kv_disk_drops
              << " ram_entries=" << stats.kv_ram_entry_count
              << " disk_entries=" << stats.kv_disk_entry_count << ")\n";
    return false;
}

int capture_to_ram(ninfer::Engine& engine, const std::vector<ninfer::TokenId>& keep,
                     const std::vector<ninfer::TokenId>& evictor,
                     std::vector<ninfer::TokenId>* generated, const char* label) {
    const auto ram_before = engine.runtime_stats().kv_ram_captures;
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare_tokens(keep), greedy(8, false));
    if (first.generated_token_ids.size() != 8) {
        std::cerr << label << " source did not generate eight tokens\n";
        return 1;
    }
    if (generated != nullptr) { *generated = first.generated_token_ids; }
    wait_idle(engine);
    const ninfer::GenerationResult other =
        engine.generate(engine.prepare_tokens(evictor), greedy(8, false));
    if (other.generated_token_ids.size() != 8 ||
        other.prefix_reuse_path != ninfer::PrefixReusePath::FullReset) {
        std::cerr << label << " evictor did not FullReset\n";
        return 1;
    }
    wait_idle(engine);
    if (engine.runtime_stats().kv_ram_captures <= ram_before) {
        std::cerr << label << " evictor did not increment kv_ram_captures\n";
        return 1;
    }
    return 0;
}

int land_on_disk(ninfer::Engine& engine, const std::vector<ninfer::TokenId>& keep,
                 const std::vector<ninfer::TokenId>& evictor,
                 const std::vector<ninfer::TokenId>& force,
                 std::vector<ninfer::TokenId>* generated, const char* label) {
    const auto captures_before = engine.runtime_stats().kv_disk_captures;
    if (const int rc = capture_to_ram(engine, keep, evictor, generated, label); rc != 0) {
        return rc;
    }
    if (engine.runtime_stats().kv_disk_captures >= captures_before + 1) { return 0; }
    const ninfer::GenerationResult extra =
        engine.generate(engine.prepare_tokens(force), greedy(8, false));
    if (extra.generated_token_ids.size() != 8 ||
        extra.prefix_reuse_path != ninfer::PrefixReusePath::FullReset) {
        std::cerr << label << " emergency evictor did not FullReset\n";
        return 1;
    }
    wait_idle(engine);
    if (!wait_disk_captures(engine, captures_before + 1, label)) { return 1; }
    return 0;
}

struct SeededChat {
    std::vector<ninfer::TokenId> keep;
    std::vector<ninfer::TokenId> generated;
    std::vector<ninfer::TokenId> history;
};

int seed_disk_chats(const char* artifact, const std::filesystem::path& disk,
                    const std::vector<std::vector<ninfer::TokenId>>& keeps,
                    std::vector<SeededChat>* out) {
    ninfer::Engine engine(disk_options(artifact, disk, 1, kRamBytes));
    if (const int rc = verify_disk_tier(engine, kRamBytes); rc != 0) { return rc; }
    out->clear();
    for (const auto& keep : keeps) {
        const ninfer::GenerationResult first =
            engine.generate(engine.prepare_tokens(keep), greedy(8, false));
        if (first.generated_token_ids.size() != 8) {
            return fail("seed chat did not generate eight tokens");
        }
        wait_idle(engine);
        SeededChat chat;
        chat.keep      = keep;
        chat.generated = first.generated_token_ids;
        chat.history   = resume_prefix(keep, first.generated_token_ids);
        out->push_back(std::move(chat));
    }
    return 0;
}

int fill_lanes(ninfer::Engine& engine, const std::vector<std::vector<ninfer::TokenId>>& keeps,
                std::vector<ninfer::GenerationResult>* results, const char* label) {
    results->clear();
    for (const auto& keep : keeps) {
        const ninfer::GenerationResult first =
            engine.generate(engine.prepare_tokens(keep), greedy(8, false));
        if (first.generated_token_ids.size() != 8) {
            std::cerr << label << " occupant did not generate eight tokens\n";
            return 1;
        }
        wait_idle(engine);
        results->push_back(first);
    }
    return 0;
}

int expect_greedy_match(const ninfer::GenerationResult& hit,
                         const ninfer::GenerationResult& baseline, const char* label) {
    if (hit.generated_token_ids != baseline.generated_token_ids) {
        std::cerr << label << " disk reuse changed greedy output\n";
        return 1;
    }
    return 0;
}

int exercise_construction(const char* artifact) {
    {
        const auto bad_dir = make_disk_dir("noram");
        ninfer::EngineOptions bad = disk_options(artifact, bad_dir, 1, 0);
        bool threw                = false;
        try {
            ninfer::Engine engine(bad);
        } catch (const std::invalid_argument&) { threw = true; }
        if (!threw) { return fail("disk without RAM did not fail Engine construction"); }
    }
    {
        ninfer::EngineOptions bad = disk_options(artifact, {}, 1, kRamBytes);
        bad.kv_disk_location.clear();
        bool threw = false;
        try {
            ninfer::Engine engine(bad);
        } catch (const std::invalid_argument&) { threw = true; }
        if (!threw) { return fail("disk capacity without location did not fail Engine construction"); }
    }
    return 0;
}

int exercise_restart_host_disk(const char* artifact) {
    const auto disk_dir = make_disk_dir("restart");
    std::vector<ninfer::TokenId> generated;
    {
        ninfer::Engine engine(disk_options(artifact, disk_dir, 1, kRamBytes));
        if (const int rc = verify_disk_tier(engine, kRamBytes); rc != 0) { return rc; }
        if (const int rc = capture_to_ram(engine, tokens_a(), tokens_b(), &generated, "restart land");
            rc != 0) {
            return rc;
        }
    }
    const auto history = static_cast<std::uint32_t>(resume_prefix(tokens_a(), generated).size());

    ninfer::Engine engine(disk_options(artifact, disk_dir, 1, kRamBytes));
    const auto prefix = resume_prefix(tokens_a(), generated);
    const auto restores_before = engine.runtime_stats().kv_disk_restores;
    const auto hit = engine.generate(engine.prepare_tokens(prefix), greedy(4, true));
    if (const int rc = expect_hit(hit, ninfer::PrefixReuseSource::HostDisk, history, "restart HostDisk");
        rc != 0) {
        return rc;
    }
    if (engine.runtime_stats().kv_disk_restores != restores_before + 1) {
        return fail("restart HostDisk did not increment kv_disk_restores");
    }
    if (hit.kv_disk_load_seconds <= 0.0) {
        return fail("restart HostDisk did not record disk load elapsed");
    }
    const auto baseline = engine.generate(engine.prepare_tokens(prefix), greedy(4, false));
    if (const int rc = expect_greedy_match(hit, baseline, "restart"); rc != 0) { return rc; }
    return 0;
}

int exercise_no_prefix_reuse(const char* artifact) {
    const auto disk_dir = make_disk_dir("noreuse");
    ninfer::Engine engine(disk_options(artifact, disk_dir, 1, kRamBytes));
    if (const int rc =
            capture_to_ram(engine, tokens_a(), tokens_b(), nullptr, "no-reuse land");
        rc != 0) {
        return rc;
    }
    const auto denied = engine.generate(engine.prepare_tokens(tokens_a()), greedy(4, false));
    if (denied.prefix_reuse_source != ninfer::PrefixReuseSource::None) {
        return fail("--no-prefix-reuse still reused");
    }
    return 0;
}

int exercise_inclusive_after_ram_consume(const char* artifact) {
    const auto disk_dir = make_disk_dir("inclusive");
    std::vector<ninfer::TokenId> generated;
    std::vector<ninfer::TokenId> history;
    {
        ninfer::Engine engine(disk_options(artifact, disk_dir, 1, kRamBytes));
        if (const int rc = verify_disk_tier(engine, kRamBytes); rc != 0) { return rc; }
        if (const int rc =
                capture_to_ram(engine, tokens_a(), tokens_b(), &generated, "inclusive land");
            rc != 0) {
            return rc;
        }
        wait_idle(engine);
        const auto disk_before_consume = engine.memory_summary().kv_disk_entry_count;
        history = resume_prefix(tokens_a(), generated);
        const auto ram_hit = engine.generate(engine.prepare_tokens(history), greedy(4, true));
        if (const int rc =
                expect_hit(ram_hit, ninfer::PrefixReuseSource::HostRam,
                           static_cast<std::uint32_t>(history.size()), "inclusive RAM consume");
            rc != 0) {
            return rc;
        }
        if (engine.memory_summary().kv_ram_entry_count != 1) {
            std::cerr << "inclusive RAM consume is not victim-only: entries="
                      << engine.memory_summary().kv_ram_entry_count << '\n';
            return 1;
        }
        if (disk_before_consume != 0 && engine.memory_summary().kv_disk_entry_count == 0) {
            return fail("inclusive RAM consume deleted the disk generation");
        }
        history = resume_prefix(history, ram_hit.generated_token_ids);
    }
    ninfer::Engine engine(disk_options(artifact, disk_dir, 1, kRamBytes));
    const auto restores_before = engine.runtime_stats().kv_disk_restores;
    const auto disk_hit = engine.generate(engine.prepare_tokens(history), greedy(4, true));
    if (const int rc =
            expect_hit(disk_hit, ninfer::PrefixReuseSource::HostDisk,
                       static_cast<std::uint32_t>(history.size()),
                       "inclusive HostDisk after RAM consume");
        rc != 0) {
        return rc;
    }
    if (engine.runtime_stats().kv_disk_restores != restores_before + 1) {
        return fail("inclusive HostDisk did not increment kv_disk_restores");
    }
    return 0;
}

int exercise_vram_wins_equal_disk(const char* artifact) {
    const auto disk_dir = make_disk_dir("vram-wins");
    ninfer::Engine engine(disk_options(artifact, disk_dir, 1, kOneEntryBytes));
    const auto keep = tokens_a();
    std::vector<ninfer::TokenId> generated;
    if (const int rc =
            land_on_disk(engine, keep, tokens_b(), tokens_c(), &generated, "VRAM-wins land");
        rc != 0) {
        return rc;
    }
    const ninfer::GenerationResult again =
        engine.generate(engine.prepare_tokens(keep), greedy(8, false));
    if (again.generated_token_ids != generated) {
        return fail("VRAM-wins rebuild did not match the disk source");
    }
    const auto history         = resume_prefix(keep, again.generated_token_ids);
    const auto restores_before = engine.runtime_stats().kv_disk_restores;
    const auto ram_before      = engine.runtime_stats().kv_ram_restores;
    const ninfer::GenerationResult hit =
        engine.generate(engine.prepare_tokens(history), greedy(4, true));
    if (hit.prefix_reuse_source != ninfer::PrefixReuseSource::VramResident ||
        hit.reused_prompt_tokens != history.size() ||
        hit.prefix_reuse_path == ninfer::PrefixReusePath::FullReset ||
        engine.runtime_stats().kv_disk_restores != restores_before ||
        engine.runtime_stats().kv_ram_restores != ram_before) {
        std::cerr << "equal reuse did not keep VRAM over disk: source="
                  << source_name(hit.prefix_reuse_source)
                  << " path=" << static_cast<int>(hit.prefix_reuse_path)
                  << " reused=" << hit.reused_prompt_tokens << '\n';
        return 1;
    }
    return 0;
}

int exercise_ram_wins_equal_disk(const char* artifact) {
    const auto disk_dir = make_disk_dir("ram-wins");
    ninfer::Engine engine(disk_options(artifact, disk_dir, 1, kRamBytes));
    std::vector<ninfer::TokenId> generated;
    if (const int rc =
            capture_to_ram(engine, tokens_a(), tokens_b(), &generated, "RAM-wins land");
        rc != 0) {
        return rc;
    }
    const auto history      = resume_prefix(tokens_a(), generated);
    const auto disk_before = engine.runtime_stats().kv_disk_restores;
    const ninfer::GenerationResult hit =
        engine.generate(engine.prepare_tokens(history), greedy(4, true));
    if (const int rc =
            expect_hit(hit, ninfer::PrefixReuseSource::HostRam,
                       static_cast<std::uint32_t>(history.size()), "equal RAM over disk");
        rc != 0) {
        return rc;
    }
    if (engine.runtime_stats().kv_disk_restores != disk_before) {
        return fail("equal-length RAM hit restored from disk");
    }
    return 0;
}

int exercise_longer_disk_beats_vram(const char* artifact) {
    const auto disk_dir = make_disk_dir("longer-disk");
    ninfer::Engine engine(disk_options(artifact, disk_dir, 1, kOneEntryBytes));
    const auto keep = tokens_a();
    std::vector<ninfer::TokenId> generated;
    if (const int rc =
            land_on_disk(engine, keep, tokens_b(), tokens_c(), &generated, "longer-disk land");
        rc != 0) {
        return rc;
    }
    const ninfer::GenerationResult shorter =
        engine.generate(engine.prepare_tokens(keep), greedy(2, false));
    if (shorter.generated_token_ids.size() != 2) {
        return fail("longer-disk shorter VRAM rebuild did not complete");
    }
    const auto history         = resume_prefix(keep, generated);
    const auto restores_before = engine.runtime_stats().kv_disk_restores;
    const ninfer::GenerationResult hit =
        engine.generate(engine.prepare_tokens(history), greedy(4, true));
    if (const int rc =
            expect_hit(hit, ninfer::PrefixReuseSource::HostDisk,
                       static_cast<std::uint32_t>(history.size()), "longer disk beats shorter VRAM");
        rc != 0) {
        return rc;
    }
    if (engine.runtime_stats().kv_disk_restores != restores_before + 1) {
        return fail("longer disk did not restore");
    }
    const auto baseline = engine.generate(engine.prepare_tokens(history), greedy(4, false));
    if (const int rc = expect_greedy_match(hit, baseline, "longer-disk"); rc != 0) { return rc; }
    return 0;
}

int exercise_suffix_prefill(const char* artifact) {
    const auto disk_dir = make_disk_dir("suffix");
    const auto keep     = tokens_a();
    std::vector<ninfer::TokenId> source_tokens;
    std::vector<ninfer::TokenId> vram_tokens;
    std::vector<ninfer::TokenId> history;
    std::vector<ninfer::TokenId> continued;
    {
        const auto suffix_vram_dir = make_disk_dir("suffix-vram");
        ninfer::Engine vram(disk_options(artifact, suffix_vram_dir, 1, kRamBytes));
        const ninfer::GenerationResult source =
            vram.generate(vram.prepare_tokens(keep), greedy(8, false));
        if (source.generated_token_ids.size() != 8) {
            return fail("suffix VRAM source did not complete");
        }
        source_tokens = source.generated_token_ids;
        history       = resume_prefix(keep, source.generated_token_ids);
        continued     = concat(history, {198, 198, 198, 198});
        const ninfer::GenerationResult vram_hit =
            vram.generate(vram.prepare_tokens(continued), greedy(4, true));
        if (const int rc =
                expect_suffix_hit(vram_hit, ninfer::PrefixReuseSource::VramResident,
                                  static_cast<std::uint32_t>(history.size()),
                                  static_cast<std::uint32_t>(continued.size()), "suffix VRAM");
            rc != 0) {
            return rc;
        }
        vram_tokens = vram_hit.generated_token_ids;
    }

    ninfer::Engine engine(disk_options(artifact, disk_dir, 1, kOneEntryBytes));
    std::vector<ninfer::TokenId> generated;
    if (const int rc =
            land_on_disk(engine, keep, tokens_c(), tokens_d(), &generated, "suffix land");
        rc != 0) {
        return rc;
    }
    if (generated != source_tokens) {
        return fail("suffix disk source did not match the VRAM source");
    }

    const auto restores_before = engine.runtime_stats().kv_disk_restores;
    const ninfer::GenerationResult hit =
        engine.generate(engine.prepare_tokens(continued), greedy(4, true));
    if (const int rc =
            expect_suffix_hit(hit, ninfer::PrefixReuseSource::HostDisk,
                             static_cast<std::uint32_t>(history.size()),
                             static_cast<std::uint32_t>(continued.size()), "suffix disk restore");
        rc != 0) {
        return rc;
    }
    if (engine.runtime_stats().kv_disk_restores != restores_before + 1) {
        return fail("suffix disk restore did not increment kv_disk_restores");
    }
    if (hit.generated_token_ids != vram_tokens) {
        return fail("suffix disk reuse changed greedy output");
    }
    return 0;
}

int exercise_ram_hit_during_idle_spill(const char* artifact) {
    const auto disk_dir = make_disk_dir("idle-spill");
    ninfer::Engine engine(disk_options(artifact, disk_dir, 1, kRamBytes));
    const auto keep = tokens_a();
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare_tokens(keep), greedy(8, false));
    if (first.generated_token_ids.size() != 8) {
        return fail("idle-spill source did not generate");
    }
    const ninfer::GenerationResult evictor =
        engine.generate(engine.prepare_tokens(tokens_b()), greedy(8, false));
    if (evictor.prefix_reuse_path != ninfer::PrefixReusePath::FullReset) {
        return fail("idle-spill evictor did not FullReset");
    }
    const auto history = resume_prefix(keep, first.generated_token_ids);
    const auto ram_hit = engine.generate(engine.prepare_tokens(history), greedy(4, true));
    if (const int rc =
            expect_hit(ram_hit, ninfer::PrefixReuseSource::HostRam,
                       static_cast<std::uint32_t>(history.size()), "RAM hit during/after spill");
        rc != 0) {
        return rc;
    }
    wait_idle(engine);
    return 0;
}

int exercise_c1_dirty_lane(const char* artifact) {
    const auto disk_dir            = make_disk_dir("c1-dirty");
    constexpr std::size_t kTightRam = 256ULL * 1024ULL * 1024ULL;
    ninfer::Engine engine(disk_options(artifact, disk_dir, 1, kTightRam));
    const auto a = engine.generate(engine.prepare_tokens(tokens_a()), greedy(8, false));
    if (a.generated_token_ids.size() != 8) { return fail("C=1 source request did not generate"); }
    const auto a_prefix = resume_prefix(tokens_a(), a.generated_token_ids);
    const auto a_hist   = static_cast<std::uint32_t>(a_prefix.size());
    wait_idle(engine);
    const std::vector<std::vector<ninfer::TokenId>> evictors = {
        tokens_b(),
        tokens_c(),
        tokens_d(),
        tokens_e(),
        {248045, 846, 198, 222, 248046, 198},
    };
    for (const auto& prompt : evictors) {
        if (engine.generate(engine.prepare_tokens(prompt), greedy(8, false))
                .generated_token_ids.size() != 8) {
            return fail("C=1 evictor did not generate");
        }
        wait_idle(engine);
    }
    if (!wait_disk_captures(engine, 1, "C=1 dirty land")) { return 1; }
    ninfer::GenerationResult dirty;
    try {
        dirty = engine.generate(engine.prepare_tokens(a_prefix), greedy(4, true));
    } catch (const std::exception& e) {
        std::cerr << "C=1 dirty-lane disk restore threw: " << e.what() << '\n';
        return 1;
    }
    if (const int rc = expect_hit(dirty, ninfer::PrefixReuseSource::HostDisk, a_hist,
                                    "C=1 dirty-lane HostDisk");
        rc != 0) {
        return rc;
    }
    return 0;
}

int exercise_c1_ram_full_disk_hit(const char* artifact) {
    const auto disk_dir = make_disk_dir("c1-ram-full");
    ninfer::Engine engine(disk_options(artifact, disk_dir, 1, kOneEntryBytes));
    if (const int rc = verify_disk_tier(engine, kOneEntryBytes); rc != 0) { return rc; }

    const ninfer::GenerationResult first_b =
        engine.generate(engine.prepare_tokens(tokens_b()), greedy(8, false));
    if (first_b.generated_token_ids.size() != 8) {
        return fail("C=1 RAM-full B source did not generate");
    }
    wait_idle(engine);
    const ninfer::GenerationResult occupant_c =
        engine.generate(engine.prepare_tokens(tokens_c()), greedy(8, false));
    if (occupant_c.generated_token_ids.size() != 8 ||
        occupant_c.prefix_reuse_path != ninfer::PrefixReusePath::FullReset) {
        return fail("C=1 RAM-full C occupant did not FullReset");
    }
    wait_idle(engine);
    const ninfer::MemorySummary after_b = engine.memory_summary();
    if (after_b.kv_ram_used_bytes == 0 || after_b.kv_ram_used_bytes * 2 <= kOneEntryBytes) {
        std::cerr << "C=1 RAM-full budget still fits two entries: used="
                  << after_b.kv_ram_used_bytes << " cap=" << kOneEntryBytes << '\n';
        return 1;
    }

    const ninfer::GenerationResult retained_a =
        engine.generate(engine.prepare_tokens(tokens_a()), greedy(8, false));
    if (retained_a.generated_token_ids.size() != 8) {
        return fail("C=1 RAM-full retained A did not generate");
    }
    wait_idle(engine);
    if (!wait_disk_captures(engine, 1, "C=1 RAM-full B durable")) { return 1; }

    const auto history_b = resume_prefix(tokens_b(), first_b.generated_token_ids);
    ninfer::GenerationResult hit;
    try {
        hit = engine.generate(engine.prepare_tokens(history_b), greedy(4, true));
    } catch (const std::exception& e) {
        std::cerr << "C=1 RAM-full disk hit of B threw: " << e.what() << '\n';
        return 1;
    }
    if (const int rc =
            expect_hit(hit, ninfer::PrefixReuseSource::HostDisk,
                       static_cast<std::uint32_t>(history_b.size()), "C=1 RAM-full HostDisk B");
        rc != 0) {
        return rc;
    }
    return 0;
}

int exercise_empty_lane_loses_to_disk(const char* artifact) {
    const auto disk_dir = make_disk_dir("empty-lane");
    std::vector<ninfer::TokenId> generated;
    {
        ninfer::Engine engine(disk_options(artifact, disk_dir, 1, kRamBytes));
        if (const int rc =
                capture_to_ram(engine, tokens_a(), tokens_b(), &generated, "empty-lane land");
            rc != 0) {
            return rc;
        }
    }

    ninfer::Engine engine(disk_options(artifact, disk_dir, 2, kRamBytes));
    const auto occupant = engine.generate(engine.prepare_tokens(tokens_c()), greedy(8, false));
    if (occupant.generated_token_ids.size() != 8) {
        return fail("empty-lane occupant did not generate");
    }
    const auto history = resume_prefix(tokens_a(), generated);
    const auto hit     = engine.generate(engine.prepare_tokens(history), greedy(4, true));
    if (const int rc =
            expect_hit(hit, ninfer::PrefixReuseSource::HostDisk,
                       static_cast<std::uint32_t>(history.size()), "empty-lane loses to disk");
        rc != 0) {
        return rc;
    }
    const auto occupant_again =
        engine.generate(engine.prepare_tokens(resume_prefix(tokens_c(), occupant.generated_token_ids)),
                        greedy(2, true));
    if (occupant_again.prefix_reuse_source != ninfer::PrefixReuseSource::VramResident) {
        return fail("empty-lane disk restore covered the unrelated VRAM occupant");
    }
    return 0;
}

int exercise_c2_dirty_disk_hit(const char* artifact) {
    const auto disk_dir = make_disk_dir("c2-dirty");
    std::vector<ninfer::TokenId> generated;
    {
        ninfer::Engine engine(disk_options(artifact, disk_dir, 1, kRamBytes));
        if (const int rc =
                capture_to_ram(engine, tokens_a(), tokens_b(), &generated, "C=2 dirty land");
            rc != 0) {
            return rc;
        }
    }

    ninfer::Engine engine(disk_options(artifact, disk_dir, 2, kRamBytes));
    if (engine.generate(engine.prepare_tokens(tokens_c()), greedy(8, false))
            .generated_token_ids.size() != 8) {
        return fail("C=2 dirty occupant C did not generate");
    }
    if (engine.generate(engine.prepare_tokens(tokens_d()), greedy(8, false))
            .generated_token_ids.size() != 8) {
        return fail("C=2 dirty occupant D did not generate");
    }
    const auto history = resume_prefix(tokens_a(), generated);
    const auto hit     = engine.generate(engine.prepare_tokens(history), greedy(4, true));
    if (const int rc =
            expect_hit(hit, ninfer::PrefixReuseSource::HostDisk,
                       static_cast<std::uint32_t>(history.size()), "C=2 dirty-only HostDisk");
        rc != 0) {
        return rc;
    }
    return 0;
}

int exercise_c3_overlapping_empty_lanes(const char* artifact) {
    const auto disk_dir = make_disk_dir("c3-empty");
    ninfer::Engine engine(disk_options(artifact, disk_dir, 3, kRamBytes));
    if (const int rc = verify_disk_tier(engine, kRamBytes); rc != 0) { return rc; }
    ninfer::GenerationHandle ha = engine.submit(engine.prepare_tokens(tokens_f()), greedy(16, false));
    ninfer::GenerationHandle hb = engine.submit(engine.prepare_tokens(tokens_g()), greedy(16, false));
    ninfer::GenerationHandle hc = engine.submit(engine.prepare_tokens(tokens_h()), greedy(16, false));
    std::uint32_t max_prefilling = 0;
    if (!wait_scheduler(
            engine, &max_prefilling,
            [](const ninfer::RuntimeStats& stats) { return stats.running_requests == 3; },
            "C=3 empty three-in-flight")) {
        return 1;
    }
    const auto a = ha.wait();
    const auto b = hb.wait();
    const auto c = hc.wait();
    if (a.generated_token_ids.size() != 16 || b.generated_token_ids.size() != 16 ||
        c.generated_token_ids.size() != 16) {
        return fail("C=3 empty overlapping submits did not complete");
    }
    if (a.prefix_reuse_path != ninfer::PrefixReusePath::FullReset ||
        b.prefix_reuse_path != ninfer::PrefixReusePath::FullReset ||
        c.prefix_reuse_path != ninfer::PrefixReusePath::FullReset ||
        a.reused_prompt_tokens != 0 || b.reused_prompt_tokens != 0 ||
        c.reused_prompt_tokens != 0) {
        return fail("C=3 empty overlapping wave reused a prefix");
    }
    if (engine.runtime_stats().kv_ram_captures != 0 ||
        engine.runtime_stats().kv_disk_restores != 0) {
        return fail("C=3 empty overlapping wave captured or disk-restored");
    }
    return 0;
}

int exercise_c3_empty_lane_loses_to_disk(const char* artifact) {
    const auto disk_dir = make_disk_dir("c3-empty-disk");
    std::vector<SeededChat> seeded;
    if (const int rc = seed_disk_chats(artifact, disk_dir, {tokens_a()}, &seeded); rc != 0) {
        return rc;
    }
    ninfer::Engine engine(disk_options(artifact, disk_dir, 3, kRamBytes));
    std::vector<ninfer::GenerationResult> occupants;
    if (const int rc = fill_lanes(engine, {tokens_f(), tokens_g()}, &occupants, "C=3 empty-disk");
        rc != 0) {
        return rc;
    }
    const auto hist_f = resume_prefix(tokens_f(), occupants[0].generated_token_ids);
    const auto hist_g = resume_prefix(tokens_g(), occupants[1].generated_token_ids);
    const auto restores_before = engine.runtime_stats().kv_disk_restores;
    ninfer::GenerationHandle disk_a =
        engine.submit(engine.prepare_tokens(seeded[0].history), greedy(8, true));
    ninfer::GenerationHandle cont_f = engine.submit(engine.prepare_tokens(hist_f), greedy(8, true));
    ninfer::GenerationHandle cont_g = engine.submit(engine.prepare_tokens(hist_g), greedy(8, true));
    std::uint32_t max_prefilling = 0;
    if (!wait_scheduler(
            engine, &max_prefilling,
            [](const ninfer::RuntimeStats& stats) { return stats.running_requests == 3; },
            "C=3 empty-lane disk plus two VRAM continues")) {
        return 1;
    }
    const auto hit_a = disk_a.wait();
    const auto hit_f = cont_f.wait();
    const auto hit_g = cont_g.wait();
    if (const int rc =
            expect_hit(hit_a, ninfer::PrefixReuseSource::HostDisk,
                       static_cast<std::uint32_t>(seeded[0].history.size()),
                       "C=3 empty-lane HostDisk");
        rc != 0) {
        return rc;
    }
    if (engine.runtime_stats().kv_disk_restores != restores_before + 1) {
        return fail("C=3 empty-lane disk restore did not increment kv_disk_restores");
    }
    if (hit_f.prefix_reuse_source != ninfer::PrefixReuseSource::VramResident ||
        hit_g.prefix_reuse_source != ninfer::PrefixReuseSource::VramResident) {
        std::cerr << "C=3 empty-lane disk restore covered a VRAM occupant: F="
                  << source_name(hit_f.prefix_reuse_source)
                  << " G=" << source_name(hit_g.prefix_reuse_source) << '\n';
        return 1;
    }
    return 0;
}

int exercise_c3_triple_disk_cover(const char* artifact) {
    const auto disk_dir = make_disk_dir("c3-triple-disk");
    std::vector<SeededChat> seeded;
    if (const int rc =
            seed_disk_chats(artifact, disk_dir, {tokens_a(), tokens_b(), tokens_c()}, &seeded);
        rc != 0) {
        return rc;
    }
    ninfer::Engine engine(disk_options(artifact, disk_dir, 3, kRamBytes));
    std::vector<ninfer::GenerationResult> occupants;
    if (const int rc =
            fill_lanes(engine, {tokens_f(), tokens_g(), tokens_h()}, &occupants, "C=3 triple");
        rc != 0) {
        return rc;
    }
    const auto ram_before  = engine.runtime_stats().kv_ram_captures;
    const auto disk_before = engine.runtime_stats().kv_disk_restores;
    ninfer::GenerationHandle ha =
        engine.submit(engine.prepare_tokens(seeded[0].history), greedy(8, true));
    ninfer::GenerationHandle hb =
        engine.submit(engine.prepare_tokens(seeded[1].history), greedy(8, true));
    ninfer::GenerationHandle hc =
        engine.submit(engine.prepare_tokens(seeded[2].history), greedy(8, true));
    std::uint32_t max_prefilling = 0;
    if (!wait_scheduler(
            engine, &max_prefilling,
            [](const ninfer::RuntimeStats& stats) {
                return stats.running_requests >= 1 &&
                       stats.running_requests + stats.waiting_requests == 3;
            },
            "C=3 triple overlapping HostDisk")) {
        return 1;
    }
    const auto hit_a = ha.wait();
    const auto hit_b = hb.wait();
    const auto hit_c = hc.wait();
    if (const int rc = expect_hit(hit_a, ninfer::PrefixReuseSource::HostDisk,
                                       static_cast<std::uint32_t>(seeded[0].history.size()),
                                       "C=3 triple A");
        rc != 0) {
        return rc;
    }
    if (const int rc = expect_hit(hit_b, ninfer::PrefixReuseSource::HostDisk,
                                       static_cast<std::uint32_t>(seeded[1].history.size()),
                                       "C=3 triple B");
        rc != 0) {
        return rc;
    }
    if (const int rc = expect_hit(hit_c, ninfer::PrefixReuseSource::HostDisk,
                                       static_cast<std::uint32_t>(seeded[2].history.size()),
                                       "C=3 triple C");
        rc != 0) {
        return rc;
    }
    if (engine.runtime_stats().kv_disk_restores != disk_before + 3) {
        std::cerr << "C=3 triple disk restores=" << engine.runtime_stats().kv_disk_restores
                  << ", expected " << disk_before + 3 << '\n';
        return 1;
    }
    if (engine.runtime_stats().kv_ram_captures < ram_before + 3) {
        std::cerr << "C=3 triple covered occupants captured "
                  << engine.runtime_stats().kv_ram_captures - ram_before << ", expected >= 3\n";
        return 1;
    }
    const auto hist_f = resume_prefix(tokens_f(), occupants[0].generated_token_ids);
    const auto ram_f  = engine.generate(engine.prepare_tokens(hist_f), greedy(2, true));
    if (ram_f.prefix_reuse_source != ninfer::PrefixReuseSource::HostRam &&
        ram_f.prefix_reuse_source != ninfer::PrefixReuseSource::HostDisk) {
        std::cerr << "C=3 covered occupant F was not restored: source="
                  << source_name(ram_f.prefix_reuse_source) << '\n';
        return 1;
    }
    return 0;
}

int exercise_c3_disk_plus_vram_continues(const char* artifact) {
    const auto disk_dir = make_disk_dir("c3-disk-vram");
    std::vector<SeededChat> seeded;
    if (const int rc = seed_disk_chats(artifact, disk_dir, {tokens_a()}, &seeded); rc != 0) {
        return rc;
    }
    ninfer::Engine engine(disk_options(artifact, disk_dir, 3, kRamBytes));
    std::vector<ninfer::GenerationResult> occupants;
    if (const int rc =
            fill_lanes(engine, {tokens_f(), tokens_g(), tokens_h()}, &occupants, "C=3 mix");
        rc != 0) {
        return rc;
    }
    const auto hist_g = resume_prefix(tokens_g(), occupants[1].generated_token_ids);
    const auto hist_h = resume_prefix(tokens_h(), occupants[2].generated_token_ids);
    const auto hist_f = resume_prefix(tokens_f(), occupants[0].generated_token_ids);
    const auto continue_g =
        engine.generate(engine.prepare_tokens(hist_g), greedy(2, true));
    if (continue_g.prefix_reuse_source != ninfer::PrefixReuseSource::VramResident) {
        return fail("C=3 mix setup did not keep G in VRAM");
    }
    const auto hist_g2 = resume_prefix(hist_g, continue_g.generated_token_ids);
    const auto ram_before  = engine.runtime_stats().kv_ram_captures;
    const auto disk_before = engine.runtime_stats().kv_disk_restores;
    ninfer::GenerationHandle disk_a =
        engine.submit(engine.prepare_tokens(seeded[0].history), greedy(12, true));
    ninfer::GenerationHandle cont_g = engine.submit(engine.prepare_tokens(hist_g2), greedy(12, true));
    ninfer::GenerationHandle cont_h = engine.submit(engine.prepare_tokens(hist_h), greedy(12, true));
    std::uint32_t max_prefilling = 0;
    if (!wait_scheduler(
            engine, &max_prefilling,
            [](const ninfer::RuntimeStats& stats) { return stats.running_requests == 3; },
            "C=3 disk restore plus two MRU VRAM continues")) {
        return 1;
    }
    const auto hit_a = disk_a.wait();
    const auto hit_g = cont_g.wait();
    const auto hit_h = cont_h.wait();
    if (const int rc =
            expect_hit(hit_a, ninfer::PrefixReuseSource::HostDisk,
                       static_cast<std::uint32_t>(seeded[0].history.size()),
                       "C=3 mix HostDisk A");
        rc != 0) {
        return rc;
    }
    if (hit_g.prefix_reuse_source != ninfer::PrefixReuseSource::VramResident ||
        hit_h.prefix_reuse_source != ninfer::PrefixReuseSource::VramResident) {
        std::cerr << "C=3 mix covered an MRU occupant: G=" << source_name(hit_g.prefix_reuse_source)
                  << " H=" << source_name(hit_h.prefix_reuse_source) << '\n';
        return 1;
    }
    if (engine.runtime_stats().kv_disk_restores != disk_before + 1) {
        return fail("C=3 mix did not restore exactly one disk chat");
    }
    if (engine.runtime_stats().kv_ram_captures <= ram_before) {
        return fail("C=3 mix disk restore did not capture the LRU occupant");
    }
    const auto ram_f = engine.generate(engine.prepare_tokens(hist_f), greedy(2, true));
    if (ram_f.prefix_reuse_source != ninfer::PrefixReuseSource::HostRam &&
        ram_f.prefix_reuse_source != ninfer::PrefixReuseSource::HostDisk) {
        std::cerr << "C=3 mix LRU occupant F was lost: source="
                  << source_name(ram_f.prefix_reuse_source) << '\n';
        return 1;
    }
    return 0;
}

int exercise_c3_duplicate_disk_submit(const char* artifact) {
    const auto disk_dir = make_disk_dir("c3-dup-disk");
    std::vector<SeededChat> seeded;
    if (const int rc = seed_disk_chats(artifact, disk_dir, {tokens_a()}, &seeded); rc != 0) {
        return rc;
    }
    ninfer::Engine engine(disk_options(artifact, disk_dir, 3, kRamBytes));
    std::vector<ninfer::GenerationResult> occupants;
    if (const int rc =
            fill_lanes(engine, {tokens_f(), tokens_g(), tokens_h()}, &occupants, "C=3 dup");
        rc != 0) {
        return rc;
    }
    ninfer::GenerationHandle first =
        engine.submit(engine.prepare_tokens(seeded[0].history), greedy(16, true));
    ninfer::GenerationHandle second =
        engine.submit(engine.prepare_tokens(seeded[0].history), greedy(16, true));
    std::uint32_t max_prefilling = 0;
    if (!wait_scheduler(
            engine, &max_prefilling,
            [](const ninfer::RuntimeStats& stats) {
                return stats.running_requests == 2 ||
                       (stats.running_requests == 1 && stats.waiting_requests >= 1);
            },
            "C=3 duplicate overlapping disk submits")) {
        return 1;
    }
    const auto hit_a = first.wait();
    const auto hit_b = second.wait();
    const bool first_disk  = hit_a.prefix_reuse_source == ninfer::PrefixReuseSource::HostDisk;
    const bool second_disk = hit_b.prefix_reuse_source == ninfer::PrefixReuseSource::HostDisk;
    if (!first_disk && !second_disk) {
        return fail("C=3 duplicate disk submit restored neither copy from disk");
    }
    const auto reused = static_cast<std::uint32_t>(seeded[0].history.size());
    if (hit_a.reused_prompt_tokens != reused || hit_b.reused_prompt_tokens != reused ||
        hit_a.prefix_reuse_path == ninfer::PrefixReusePath::FullReset ||
        hit_b.prefix_reuse_path == ninfer::PrefixReusePath::FullReset) {
        std::cerr << "C=3 duplicate disk submit dropped a prefix: A source="
                  << source_name(hit_a.prefix_reuse_source) << " reused=" << hit_a.reused_prompt_tokens
                  << " B source=" << source_name(hit_b.prefix_reuse_source)
                  << " reused=" << hit_b.reused_prompt_tokens << '\n';
        return 1;
    }
    return 0;
}

int exercise_c3_queued_disk_matcher(const char* artifact) {
    const auto disk_dir = make_disk_dir("c3-queued");
    std::vector<SeededChat> seeded;
    if (const int rc = seed_disk_chats(artifact, disk_dir, {tokens_a()}, &seeded); rc != 0) {
        return rc;
    }
    ninfer::Engine engine(disk_options(artifact, disk_dir, 3, kRamBytes));
    ninfer::GenerationHandle hf = engine.submit(engine.prepare_tokens(tokens_f()), greedy(48, false));
    ninfer::GenerationHandle hg = engine.submit(engine.prepare_tokens(tokens_g()), greedy(48, false));
    ninfer::GenerationHandle hh = engine.submit(engine.prepare_tokens(tokens_h()), greedy(48, false));
    ninfer::GenerationHandle ha =
        engine.submit(engine.prepare_tokens(seeded[0].history), greedy(8, true));
    std::uint32_t max_prefilling = 0;
    if (!wait_scheduler(
            engine, &max_prefilling,
            [](const ninfer::RuntimeStats& stats) {
                return stats.running_requests == 3 && stats.waiting_requests >= 1;
            },
            "C=3 queued disk matcher behind a full batch")) {
        return 1;
    }
    (void)hf.wait();
    (void)hg.wait();
    (void)hh.wait();
    const auto hit = ha.wait();
    if (const int rc =
            expect_hit(hit, ninfer::PrefixReuseSource::HostDisk,
                       static_cast<std::uint32_t>(seeded[0].history.size()),
                       "C=3 queued HostDisk");
        rc != 0) {
        return rc;
    }
    return 0;
}

int exercise_c3_cancel_disk_restore(const char* artifact) {
    const auto disk_dir = make_disk_dir("c3-cancel");
    std::vector<SeededChat> seeded;
    if (const int rc = seed_disk_chats(artifact, disk_dir, {tokens_a()}, &seeded); rc != 0) {
        return rc;
    }
    auto engine = std::make_unique<ninfer::Engine>(disk_options(artifact, disk_dir, 3, kRamBytes));
    std::vector<ninfer::GenerationResult> occupants;
    if (const int rc =
            fill_lanes(*engine, {tokens_f(), tokens_g(), tokens_h()}, &occupants, "C=3 cancel");
        rc != 0) {
        return rc;
    }
    const auto hist_g = resume_prefix(tokens_g(), occupants[1].generated_token_ids);
    const auto hist_h = resume_prefix(tokens_h(), occupants[2].generated_token_ids);
    const auto disk_before = engine->runtime_stats().kv_disk_restores;
    ninfer::GenerationHandle disk_a =
        engine->submit(engine->prepare_tokens(seeded[0].history), greedy(32, true),
                       ninfer::OutputDelivery::Streaming);
    ninfer::GenerationHandle cont_g = engine->submit(engine->prepare_tokens(hist_g), greedy(12, true));
    ninfer::GenerationHandle cont_h = engine->submit(engine->prepare_tokens(hist_h), greedy(12, true));
    struct CancelAfterPublish : ninfer::OutputSink {
        std::atomic<int> published{0};
        std::atomic<bool> stop{false};
        void publish(ninfer::OutputDelta) override {
            if (published.fetch_add(1) + 1 >= 2) { stop.store(true); }
        }
    } sink;
    ninfer::CancellationView cancel([&] { return sink.stop.load(); });
    const auto cancelled = disk_a.wait(&sink, cancel);
    const auto hit_g    = cont_g.wait();
    const auto hit_h    = cont_h.wait();
    if (cancelled.finish_reason != ninfer::FinishReason::Cancelled &&
        cancelled.generated_token_ids.size() == 32) {
        return fail("C=3 cancelled disk restore ran to the output limit");
    }
    if (hit_g.prefix_reuse_source != ninfer::PrefixReuseSource::VramResident ||
        hit_h.prefix_reuse_source != ninfer::PrefixReuseSource::VramResident) {
        std::cerr << "C=3 cancel covered a VRAM continue: G="
                  << source_name(hit_g.prefix_reuse_source)
                  << " H=" << source_name(hit_h.prefix_reuse_source) << '\n';
        return 1;
    }
    wait_idle(*engine);
    const auto again = engine->generate(engine->prepare_tokens(seeded[0].history), greedy(4, true));
    if (again.prefix_reuse_path == ninfer::PrefixReusePath::FullReset ||
        again.prefix_reuse_source == ninfer::PrefixReuseSource::None) {
        std::cerr << "C=3 disk A was lost after cancel: source="
                  << source_name(again.prefix_reuse_source) << '\n';
        return 1;
    }
    if (engine->runtime_stats().kv_disk_restores < disk_before + 1 &&
        again.prefix_reuse_source != ninfer::PrefixReuseSource::VramResident &&
        again.prefix_reuse_source != ninfer::PrefixReuseSource::HostRam) {
        return fail("C=3 cancel path did not restore A from any KV tier");
    }
    engine.reset();
    return 0;
}

int exercise_c3_suffix_disk_with_occupants(const char* artifact) {
    const auto disk_dir = make_disk_dir("c3-suffix");
    std::vector<SeededChat> seeded;
    if (const int rc = seed_disk_chats(artifact, disk_dir, {tokens_a()}, &seeded); rc != 0) {
        return rc;
    }
    ninfer::Engine engine(disk_options(artifact, disk_dir, 3, kRamBytes));
    std::vector<ninfer::GenerationResult> occupants;
    if (const int rc =
            fill_lanes(engine, {tokens_f(), tokens_g(), tokens_h()}, &occupants, "C=3 suffix");
        rc != 0) {
        return rc;
    }
    const auto continued = concat(seeded[0].history, {198, 198, 198, 198});
    const auto hist_g     = resume_prefix(tokens_g(), occupants[1].generated_token_ids);
    const auto hist_h     = resume_prefix(tokens_h(), occupants[2].generated_token_ids);
    ninfer::GenerationHandle disk_a =
        engine.submit(engine.prepare_tokens(continued), greedy(8, true));
    ninfer::GenerationHandle cont_g = engine.submit(engine.prepare_tokens(hist_g), greedy(8, true));
    ninfer::GenerationHandle cont_h = engine.submit(engine.prepare_tokens(hist_h), greedy(8, true));
    std::uint32_t max_prefilling = 0;
    if (!wait_scheduler(
            engine, &max_prefilling,
            [](const ninfer::RuntimeStats& stats) { return stats.running_requests == 3; },
            "C=3 suffix disk plus two VRAM continues")) {
        return 1;
    }
    const auto hit_a = disk_a.wait();
    const auto hit_g = cont_g.wait();
    const auto hit_h = cont_h.wait();
    if (const int rc =
            expect_suffix_hit(hit_a, ninfer::PrefixReuseSource::HostDisk,
                             static_cast<std::uint32_t>(seeded[0].history.size()),
                             static_cast<std::uint32_t>(continued.size()),
                             "C=3 suffix HostDisk");
        rc != 0) {
        return rc;
    }
    if (hit_g.prefix_reuse_source != ninfer::PrefixReuseSource::VramResident ||
        hit_h.prefix_reuse_source != ninfer::PrefixReuseSource::VramResident) {
        std::cerr << "C=3 suffix disk restore covered a VRAM occupant: G="
                  << source_name(hit_g.prefix_reuse_source)
                  << " H=" << source_name(hit_h.prefix_reuse_source) << '\n';
        return 1;
    }
    return 0;
}

int exercise_c3_two_disk_one_vram(const char* artifact) {
    const auto disk_dir = make_disk_dir("c3-two-disk");
    std::vector<SeededChat> seeded;
    if (const int rc = seed_disk_chats(artifact, disk_dir, {tokens_a(), tokens_b()}, &seeded);
        rc != 0) {
        return rc;
    }
    ninfer::Engine engine(disk_options(artifact, disk_dir, 3, kRamBytes));
    std::vector<ninfer::GenerationResult> occupants;
    if (const int rc =
            fill_lanes(engine, {tokens_f(), tokens_g(), tokens_h()}, &occupants, "C=3 two-disk");
        rc != 0) {
        return rc;
    }
    const auto hist_h = resume_prefix(tokens_h(), occupants[2].generated_token_ids);
    const auto hist_f = resume_prefix(tokens_f(), occupants[0].generated_token_ids);
    const auto hist_g = resume_prefix(tokens_g(), occupants[1].generated_token_ids);
    const auto ram_before  = engine.runtime_stats().kv_ram_captures;
    const auto disk_before = engine.runtime_stats().kv_disk_restores;
    ninfer::GenerationHandle disk_a =
        engine.submit(engine.prepare_tokens(seeded[0].history), greedy(12, true));
    ninfer::GenerationHandle disk_b =
        engine.submit(engine.prepare_tokens(seeded[1].history), greedy(12, true));
    ninfer::GenerationHandle cont_h = engine.submit(engine.prepare_tokens(hist_h), greedy(12, true));
    std::uint32_t max_prefilling = 0;
    if (!wait_scheduler(
            engine, &max_prefilling,
            [](const ninfer::RuntimeStats& stats) {
                return stats.running_requests >= 1 &&
                       stats.running_requests + stats.waiting_requests == 3;
            },
            "C=3 two HostDisk plus MRU VRAM continue")) {
        return 1;
    }
    const auto hit_a = disk_a.wait();
    const auto hit_b = disk_b.wait();
    const auto hit_h = cont_h.wait();
    if (const int rc = expect_hit(hit_a, ninfer::PrefixReuseSource::HostDisk,
                                    static_cast<std::uint32_t>(seeded[0].history.size()),
                                    "C=3 two-disk A");
        rc != 0) {
        return rc;
    }
    if (const int rc = expect_hit(hit_b, ninfer::PrefixReuseSource::HostDisk,
                                    static_cast<std::uint32_t>(seeded[1].history.size()),
                                    "C=3 two-disk B");
        rc != 0) {
        return rc;
    }
    if (hit_h.prefix_reuse_source != ninfer::PrefixReuseSource::VramResident) {
        std::cerr << "C=3 two-disk covered MRU occupant H: source="
                  << source_name(hit_h.prefix_reuse_source) << '\n';
        return 1;
    }
    if (engine.runtime_stats().kv_disk_restores != disk_before + 2) {
        std::cerr << "C=3 two-disk restores=" << engine.runtime_stats().kv_disk_restores
                  << ", expected " << disk_before + 2 << '\n';
        return 1;
    }
    if (engine.runtime_stats().kv_ram_captures < ram_before + 2) {
        std::cerr << "C=3 two-disk captured " << engine.runtime_stats().kv_ram_captures - ram_before
                  << " occupants, expected >= 2\n";
        return 1;
    }
    const auto ram_f = engine.generate(engine.prepare_tokens(hist_f), greedy(2, true));
    const auto ram_g = engine.generate(engine.prepare_tokens(hist_g), greedy(2, true));
    if ((ram_f.prefix_reuse_source != ninfer::PrefixReuseSource::HostRam &&
         ram_f.prefix_reuse_source != ninfer::PrefixReuseSource::HostDisk) ||
        (ram_g.prefix_reuse_source != ninfer::PrefixReuseSource::HostRam &&
         ram_g.prefix_reuse_source != ninfer::PrefixReuseSource::HostDisk)) {
        std::cerr << "C=3 two-disk lost a covered occupant: F=" << source_name(ram_f.prefix_reuse_source)
                  << " G=" << source_name(ram_g.prefix_reuse_source) << '\n';
        return 1;
    }
    return 0;
}

int exercise_c3_disk_after_vram_inflight(const char* artifact) {
    const auto disk_dir = make_disk_dir("c3-after-inflight");
    std::vector<SeededChat> seeded;
    if (const int rc = seed_disk_chats(artifact, disk_dir, {tokens_a()}, &seeded); rc != 0) {
        return rc;
    }
    ninfer::Engine engine(disk_options(artifact, disk_dir, 3, kRamBytes));
    std::vector<ninfer::GenerationResult> occupants;
    if (const int rc =
            fill_lanes(engine, {tokens_f(), tokens_g(), tokens_h()}, &occupants, "C=3 after-inflight");
        rc != 0) {
        return rc;
    }
    const auto hist_g = resume_prefix(tokens_g(), occupants[1].generated_token_ids);
    const auto hist_h = resume_prefix(tokens_h(), occupants[2].generated_token_ids);
    ninfer::GenerationHandle cont_g = engine.submit(engine.prepare_tokens(hist_g), greedy(48, true));
    ninfer::GenerationHandle cont_h = engine.submit(engine.prepare_tokens(hist_h), greedy(48, true));
    std::uint32_t max_prefilling = 0;
    if (!wait_scheduler(
            engine, &max_prefilling,
            [](const ninfer::RuntimeStats& stats) {
                return stats.running_requests == 2 && stats.decode_ready_requests == 2;
            },
            "C=3 two VRAM continues in decode before disk admit")) {
        return 1;
    }
    const auto disk_before = engine.runtime_stats().kv_disk_restores;
    ninfer::GenerationHandle disk_a =
        engine.submit(engine.prepare_tokens(seeded[0].history), greedy(8, true));
    if (!wait_scheduler(
            engine, &max_prefilling,
            [](const ninfer::RuntimeStats& stats) { return stats.running_requests == 3; },
            "C=3 disk restore onto leftover dirty lane")) {
        return 1;
    }
    const auto hit_a = disk_a.wait();
    const auto hit_g = cont_g.wait();
    const auto hit_h = cont_h.wait();
    if (const int rc =
            expect_hit(hit_a, ninfer::PrefixReuseSource::HostDisk,
                       static_cast<std::uint32_t>(seeded[0].history.size()),
                       "C=3 disk after in-flight VRAM");
        rc != 0) {
        return rc;
    }
    if (hit_g.prefix_reuse_source != ninfer::PrefixReuseSource::VramResident ||
        hit_h.prefix_reuse_source != ninfer::PrefixReuseSource::VramResident) {
        std::cerr << "C=3 disk after in-flight covered a decoding occupant: G="
                  << source_name(hit_g.prefix_reuse_source)
                  << " H=" << source_name(hit_h.prefix_reuse_source) << '\n';
        return 1;
    }
    if (engine.runtime_stats().kv_disk_restores != disk_before + 1) {
        return fail("C=3 disk after in-flight did not restore exactly one disk chat");
    }
    return 0;
}

bool truncate_first_main_object(const std::filesystem::path& disk) {
    // v4 stores main pages as extents in a generation pack rather than one
    // file per object.  Truncating the first main pack is an intentionally
    // stronger corruption injection: the entry must be rejected rather than
    // restored from a silently short extent.
    const auto root = disk / "packs";
    if (!std::filesystem::exists(root)) { return false; }
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
        if (!entry.is_regular_file() || entry.path().parent_path().filename() != "main") { continue; }
        if (entry.file_size() > 32) {
            std::filesystem::resize_file(entry.path(), 8);
            return true;
        }
    }
    return false;
}

int exercise_cancel_then_ram_continue(const char* artifact) {
    const auto disk_dir = make_disk_dir("cancel-ram");
    std::vector<SeededChat> seeded;
    if (const int rc = seed_disk_chats(artifact, disk_dir, {tokens_a()}, &seeded); rc != 0) {
        return rc;
    }
    ninfer::Engine engine(disk_options(artifact, disk_dir, 1, kRamBytes));
    std::vector<ninfer::GenerationResult> occupants;
    if (const int rc = fill_lanes(engine, {tokens_b()}, &occupants, "cancel-ram occupant");
        rc != 0) {
        return rc;
    }
    ninfer::GenerationHandle disk_a =
        engine.submit(engine.prepare_tokens(seeded[0].history), greedy(32, true));
    ninfer::CancellationView cancel([] { return true; });
    (void)disk_a.wait(nullptr, cancel);
    wait_idle(engine);
    const auto hist_b = resume_prefix(tokens_b(), occupants[0].generated_token_ids);
    const auto hit_b = engine.generate(engine.prepare_tokens(hist_b), greedy(4, true));
    if (hit_b.prefix_reuse_path == ninfer::PrefixReusePath::FullReset ||
        hit_b.prefix_reuse_source == ninfer::PrefixReuseSource::None) {
        std::cerr << "cancel-then-RAM occupant reused nothing: source="
                  << source_name(hit_b.prefix_reuse_source) << '\n';
        return 1;
    }
    if (hit_b.prefix_reuse_source == ninfer::PrefixReuseSource::HostDisk) {
        return fail("cancel-then-RAM occupant came from disk instead of RAM/VRAM");
    }
    const auto hit_a = engine.generate(engine.prepare_tokens(seeded[0].history), greedy(4, true));
    if (hit_a.prefix_reuse_source != ninfer::PrefixReuseSource::HostRam) {
        if (const int rc =
                expect_hit(hit_a, ninfer::PrefixReuseSource::HostDisk,
                           static_cast<std::uint32_t>(seeded[0].history.size()),
                           "cancel-then-RAM A");
            rc != 0) {
            return rc;
        }
    }
    return 0;
}

int exercise_destroy_during_disk_restore(const char* artifact) {
    const auto disk_dir = make_disk_dir("destroy-restore");
    std::vector<ninfer::TokenId> generated;
    {
        ninfer::Engine engine(disk_options(artifact, disk_dir, 1, kRamBytes));
        if (const int rc =
                capture_to_ram(engine, tokens_a(), tokens_b(), &generated, "destroy-restore land");
            rc != 0) {
            return rc;
        }
    }
    const auto history = resume_prefix(tokens_a(), generated);
    {
        ninfer::Engine engine(disk_options(artifact, disk_dir, 1, kRamBytes));
        ninfer::GenerationHandle restore =
            engine.submit(engine.prepare_tokens(history), greedy(64, true));
        std::uint32_t max_prefilling = 0;
        if (!wait_scheduler(
                engine, &max_prefilling,
                [](const ninfer::RuntimeStats& stats) { return stats.running_requests >= 1; },
                "destroy-during-restore start")) {
            return 1;
        }
        (void)restore;
    }
    ninfer::Engine engine(disk_options(artifact, disk_dir, 1, kRamBytes));
    const auto hit = engine.generate(engine.prepare_tokens(history), greedy(4, true));
    if (const int rc =
            expect_hit(hit, ninfer::PrefixReuseSource::HostDisk,
                       static_cast<std::uint32_t>(history.size()), "destroy-during-restore");
        rc != 0) {
        return rc;
    }
    const auto baseline = engine.generate(engine.prepare_tokens(history), greedy(4, false));
    if (const int rc = expect_greedy_match(hit, baseline, "destroy-during-restore"); rc != 0) {
        return rc;
    }
    return 0;
}

int exercise_capture_fail_does_not_leak_load(const char* artifact) {
    const auto disk_dir = make_disk_dir("capture-fail");
    std::vector<SeededChat> seeded;
    if (const int rc = seed_disk_chats(artifact, disk_dir, {tokens_a()}, &seeded); rc != 0) {
        return rc;
    }
    constexpr std::size_t kTinyRam = 1ULL << 20;
    ninfer::Engine engine(disk_options(artifact, disk_dir, 1, kTinyRam));
    const auto occupant =
        engine.generate(engine.prepare_tokens(tokens_b()), greedy(8, false));
    if (occupant.generated_token_ids.size() != 8) {
        return fail("capture-fail occupant did not generate");
    }
    wait_idle(engine);
    bool overloaded = false;
    try {
        (void)engine.generate(engine.prepare_tokens(seeded[0].history), greedy(4, true));
    } catch (const ninfer::RequestError& error) {
        if (error.kind() != ninfer::RequestErrorKind::Overloaded) {
            std::cerr << "capture-fail threw " << error.what() << '\n';
            return 1;
        }
        overloaded = true;
    }
    if (!overloaded) {
        return fail("tiny-RAM disk hit did not overload on capture");
    }
    const auto hist_b = resume_prefix(tokens_b(), occupant.generated_token_ids);
    const auto next = engine.generate(engine.prepare_tokens(hist_b), greedy(4, true));
    if (next.generated_token_ids.size() != 4) {
        return fail("capture-fail leaked into a later occupant continue");
    }
    if (next.prefix_reuse_source != ninfer::PrefixReuseSource::VramResident &&
        next.prefix_reuse_source != ninfer::PrefixReuseSource::HostRam) {
        std::cerr << "capture-fail occupant continue source="
                  << source_name(next.prefix_reuse_source) << '\n';
        return 1;
    }
    if (next.kv_disk_load_seconds > 0.0) {
        std::cerr << "capture-fail billed prefetch load onto the next request, load="
                  << next.kv_disk_load_seconds << "s\n";
        return 1;
    }
    return 0;
}

int exercise_corrupt_restore_does_not_kill_engine(const char* artifact) {
    const auto disk_dir = make_disk_dir("corrupt-restore");
    std::vector<SeededChat> seeded;
    if (const int rc = seed_disk_chats(artifact, disk_dir, {tokens_a()}, &seeded); rc != 0) {
        return rc;
    }
    ninfer::Engine engine(disk_options(artifact, disk_dir, 1, kRamBytes));
    std::vector<ninfer::GenerationResult> occupants;
    if (const int rc = fill_lanes(engine, {tokens_b()}, &occupants, "corrupt-restore occupant");
        rc != 0) {
        return rc;
    }
    wait_idle(engine);
    if (!truncate_first_main_object(disk_dir)) {
        return fail("corrupt-restore found no main object to truncate");
    }
    bool restore_failed = false;
    try {
        (void)engine.generate(engine.prepare_tokens(seeded[0].history), greedy(4, true));
    } catch (const ninfer::RequestError& error) {
        if (error.kind() != ninfer::RequestErrorKind::Unavailable) {
            std::cerr << "corrupt restore threw " << error.what() << '\n';
            return 1;
        }
        restore_failed = true;
    } catch (const std::exception& error) {
        std::cerr << "corrupt restore killed the Engine: " << error.what() << '\n';
        return 1;
    }
    if (!restore_failed) {
        return fail("truncated main object still restored");
    }
    const auto next = engine.generate(engine.prepare_tokens(tokens_c()), greedy(4, false));
    if (next.generated_token_ids.size() != 4) {
        return fail("Engine did not accept a later request after a failed disk restore");
    }
    return 0;
}

int exercise_dflash_cancel_then_continue(const char* artifact) {
    const auto disk_dir = make_disk_dir("dflash-cancel");
    std::vector<ninfer::TokenId> generated;
    try {
        ninfer::Engine engine(dflash_disk_options(artifact, disk_dir, 1, kRamBytes));
        if (const int rc =
                capture_to_ram(engine, tokens_a(), tokens_b(), &generated, "DFlash cancel land");
            rc != 0) {
            return rc;
        }
    } catch (const std::exception& e) {
        const std::string message = e.what();
        if (message.find("dflash/") != std::string::npos) {
            std::cerr << "disk_real: skip DFlash cancel-then-continue (artifact has no dflash "
                         "objects)\n";
            return 0;
        }
        std::cerr << "DFlash cancel-then-continue Engine failed: " << message << '\n';
        return 1;
    }

    ninfer::Engine engine(dflash_disk_options(artifact, disk_dir, 1, kRamBytes));
    std::vector<ninfer::GenerationResult> occupants;
    if (const int rc = fill_lanes(engine, {tokens_b()}, &occupants, "DFlash cancel occupant");
        rc != 0) {
        return rc;
    }
    const auto history_a = resume_prefix(tokens_a(), generated);
    ninfer::GenerationHandle disk_a =
        engine.submit(engine.prepare_tokens(history_a), greedy(32, true));
    ninfer::CancellationView cancel([] { return true; });
    (void)disk_a.wait(nullptr, cancel);
    wait_idle(engine);
    const auto hist_b = resume_prefix(tokens_b(), occupants[0].generated_token_ids);
    const auto hit_b  = engine.generate(engine.prepare_tokens(hist_b), greedy(4, true));
    if (hit_b.prefix_reuse_path == ninfer::PrefixReusePath::FullReset ||
        hit_b.prefix_reuse_source == ninfer::PrefixReuseSource::None) {
        std::cerr << "DFlash cancel occupant reused nothing: source="
                  << source_name(hit_b.prefix_reuse_source) << '\n';
        return 1;
    }
    if (hit_b.prefix_reuse_source == ninfer::PrefixReuseSource::HostDisk) {
        return fail("DFlash cancel occupant came from disk instead of RAM/VRAM");
    }
    if (hit_b.speculative.rounds == 0) {
        return fail("DFlash cancel occupant continue lost speculative ladders");
    }
    const auto hit_a = engine.generate(engine.prepare_tokens(history_a), greedy(4, true));
    if (hit_a.prefix_reuse_source != ninfer::PrefixReuseSource::HostRam) {
        if (const int rc =
                expect_hit(hit_a, ninfer::PrefixReuseSource::HostDisk,
                           static_cast<std::uint32_t>(history_a.size()),
                           "DFlash cancel A");
            rc != 0) {
            return rc;
        }
    }
    return 0;
}

int exercise_dflash_fingerprint(const char* artifact) {
    const auto disk_dir = make_disk_dir("fp-dflash");
    {
        ninfer::Engine greedy_engine(disk_options(artifact, disk_dir, 1, kRamBytes));
        if (const int rc = capture_to_ram(greedy_engine, tokens_a(), tokens_b(), nullptr,
                                           "fingerprint land");
            rc != 0) {
            return rc;
        }
    }
    bool threw = false;
    try {
        ninfer::Engine dflash(dflash_disk_options(artifact, disk_dir, 1, kRamBytes));
        (void)dflash;
    } catch (const std::exception& e) {
        const std::string message = e.what();
        if (message.find("dflash/") != std::string::npos) {
            std::cerr << "disk_real: skip DFlash fingerprint (artifact has no dflash objects)\n";
            return 0;
        }
        threw = true;
        if (message.find("fingerprint") == std::string::npos &&
            message.find("speculative") == std::string::npos) {
            std::cerr << "DFlash reopen of greedy directory failed for the wrong reason: " << message
                      << '\n';
            return 1;
        }
    }
    if (!threw) {
        return fail("DFlash Engine accepted a greedy (spec=off) disk directory");
    }
    return 0;
}

int exercise_dflash_three_tier(const char* artifact) {
    const auto disk_dir = make_disk_dir("dflash");
    std::vector<ninfer::TokenId> generated;
    try {
        ninfer::Engine engine(dflash_disk_options(artifact, disk_dir, 1, kRamBytes));
        if (const int rc = verify_disk_tier(engine, kRamBytes); rc != 0) { return rc; }
        if (const int rc =
                capture_to_ram(engine, tokens_a(), tokens_b(), &generated, "DFlash land");
            rc != 0) {
            return rc;
        }
    } catch (const std::exception& e) {
        const std::string message = e.what();
        if (message.find("dflash/") != std::string::npos) {
            std::cerr << "disk_real: skip DFlash three-tier (artifact has no dflash objects)\n";
            return 0;
        }
        std::cerr << "DFlash three-tier Engine failed: " << message << '\n';
        return 1;
    }

    ninfer::Engine engine(dflash_disk_options(artifact, disk_dir, 1, kRamBytes));
    const auto history = resume_prefix(tokens_a(), generated);
    const auto hit      = engine.generate(engine.prepare_tokens(history), greedy(4, true));
    if (const int rc =
            expect_hit(hit, ninfer::PrefixReuseSource::HostDisk,
                       static_cast<std::uint32_t>(history.size()), "DFlash restart HostDisk");
        rc != 0) {
        return rc;
    }
    if (hit.speculative.rounds == 0) {
        return fail("DFlash HostDisk continuation did not run speculative decode");
    }
    return 0;
}

int exercise_artifact(const char* artifact) {
    auto run = [](const char* label, auto fn) -> int {
        std::cerr << "disk_real: " << label << '\n';
        try {
            return fn();
        } catch (const ninfer::RequestError& e) {
            std::cerr << label << " threw RequestError: " << e.what() << '\n';
            return 1;
        } catch (const std::exception& e) {
            std::cerr << label << " threw: " << e.what() << '\n';
            return 1;
        }
    };
    if (const int rc = run("construction", [&] { return exercise_construction(artifact); });
        rc != 0) {
        return rc;
    }
    if (const int rc = run("restart HostDisk", [&] { return exercise_restart_host_disk(artifact); });
        rc != 0) {
        return rc;
    }
    if (const int rc = run("--no-prefix-reuse", [&] { return exercise_no_prefix_reuse(artifact); });
        rc != 0) {
        return rc;
    }
    if (const int rc = run("inclusive disk after RAM consume",
                           [&] { return exercise_inclusive_after_ram_consume(artifact); });
        rc != 0) {
        return rc;
    }
    if (const int rc = run("VRAM wins equal disk",
                           [&] { return exercise_vram_wins_equal_disk(artifact); });
        rc != 0) {
        return rc;
    }
    if (const int rc = run("RAM wins equal disk", [&] { return exercise_ram_wins_equal_disk(artifact); });
        rc != 0) {
        return rc;
    }
    if (const int rc = run("longer disk beats shorter VRAM",
                           [&] { return exercise_longer_disk_beats_vram(artifact); });
        rc != 0) {
        return rc;
    }
    if (const int rc = run("suffix prefill after disk restore",
                           [&] { return exercise_suffix_prefill(artifact); });
        rc != 0) {
        return rc;
    }
    if (const int rc = run("RAM hit during idle spill",
                           [&] { return exercise_ram_hit_during_idle_spill(artifact); });
        rc != 0) {
        return rc;
    }
    if (const int rc = run("C=1 dirty-lane HostDisk",
                           [&] { return exercise_c1_dirty_lane(artifact); });
        rc != 0) {
        return rc;
    }
    if (const int rc = run("C=1 RAM-full disk hit of B",
                           [&] { return exercise_c1_ram_full_disk_hit(artifact); });
        rc != 0) {
        return rc;
    }
    if (const int rc = run("C=2 empty lane loses to longer disk",
                           [&] { return exercise_empty_lane_loses_to_disk(artifact); });
        rc != 0) {
        return rc;
    }
    if (const int rc = run("C=2 dirty-only HostDisk",
                           [&] { return exercise_c2_dirty_disk_hit(artifact); });
        rc != 0) {
        return rc;
    }
    if (const int rc = run("C=3 overlapping empty lanes",
                           [&] { return exercise_c3_overlapping_empty_lanes(artifact); });
        rc != 0) {
        return rc;
    }
    if (const int rc = run("C=3 empty lane loses to disk",
                           [&] { return exercise_c3_empty_lane_loses_to_disk(artifact); });
        rc != 0) {
        return rc;
    }
    if (const int rc = run("C=3 triple overlapping HostDisk",
                           [&] { return exercise_c3_triple_disk_cover(artifact); });
        rc != 0) {
        return rc;
    }
    if (const int rc = run("C=3 disk restore plus VRAM continues",
                           [&] { return exercise_c3_disk_plus_vram_continues(artifact); });
        rc != 0) {
        return rc;
    }
    if (const int rc = run("C=3 duplicate disk submit",
                           [&] { return exercise_c3_duplicate_disk_submit(artifact); });
        rc != 0) {
        return rc;
    }
    if (const int rc = run("C=3 queued disk matcher",
                           [&] { return exercise_c3_queued_disk_matcher(artifact); });
        rc != 0) {
        return rc;
    }
    if (const int rc = run("C=3 cancel disk restore",
                           [&] { return exercise_c3_cancel_disk_restore(artifact); });
        rc != 0) {
        return rc;
    }
    if (const int rc = run("cancel then RAM continue",
                           [&] { return exercise_cancel_then_ram_continue(artifact); });
        rc != 0) {
        return rc;
    }
    if (const int rc = run("destroy during disk restore",
                           [&] { return exercise_destroy_during_disk_restore(artifact); });
        rc != 0) {
        return rc;
    }
    if (const int rc = run("capture-fail does not leak load",
                           [&] { return exercise_capture_fail_does_not_leak_load(artifact); });
        rc != 0) {
        return rc;
    }
    if (const int rc = run("corrupt restore does not kill Engine",
                           [&] { return exercise_corrupt_restore_does_not_kill_engine(artifact); });
        rc != 0) {
        return rc;
    }
    if (const int rc = run("C=3 suffix disk with occupants",
                           [&] { return exercise_c3_suffix_disk_with_occupants(artifact); });
        rc != 0) {
        return rc;
    }
    if (const int rc = run("C=3 two HostDisk plus MRU VRAM",
                           [&] { return exercise_c3_two_disk_one_vram(artifact); });
        rc != 0) {
        return rc;
    }
    if (const int rc = run("C=3 disk after in-flight VRAM continues",
                           [&] { return exercise_c3_disk_after_vram_inflight(artifact); });
        rc != 0) {
        return rc;
    }
    if (const int rc = run("DFlash cancel then VRAM/RAM continue",
                           [&] { return exercise_dflash_cancel_then_continue(artifact); });
        rc != 0) {
        return rc;
    }
    if (const int rc = run("MTP/off vs DFlash directory fingerprint",
                           [&] { return exercise_dflash_fingerprint(artifact); });
        rc != 0) {
        return rc;
    }
    if (const int rc = run("DFlash2 three-tier HostDisk",
                           [&] { return exercise_dflash_three_tier(artifact); });
        rc != 0) {
        return rc;
    }
    return 0;
}

} // namespace

int main() {
    const char* groupwise = std::getenv("NINFER_QWEN3_6_27B_WEIGHTS");
    const char* nvfp4     = std::getenv("NINFER_QWEN3_6_27B_NVFP4_WEIGHTS");
    const char* dflash    = std::getenv("NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS");
    if ((groupwise == nullptr || *groupwise == '\0') && (nvfp4 == nullptr || *nvfp4 == '\0') &&
        (dflash == nullptr || *dflash == '\0')) {
        std::cout << "skip: set NINFER_QWEN3_6_27B_WEIGHTS, "
                     "NINFER_QWEN3_6_27B_NVFP4_WEIGHTS, or "
                     "NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS\n";
        return 77;
    }
    if (groupwise != nullptr && *groupwise != '\0') {
        if (const int result = exercise_artifact(groupwise); result != 0) { return result; }
    }
    if (nvfp4 != nullptr && *nvfp4 != '\0') {
        if (const int result = exercise_artifact(nvfp4); result != 0) { return result; }
    }
    if (dflash != nullptr && *dflash != '\0' &&
        (nvfp4 == nullptr || *nvfp4 == '\0' || std::string(dflash) != nvfp4)) {
        if (const int result = exercise_artifact(dflash); result != 0) { return result; }
    }
    std::cout << "ok\n";
    return 0;
}
