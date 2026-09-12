#include "ninfer/engine.h"

#include <cstdlib>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <thread>
#include <vector>

namespace {

using Tokens = std::vector<ninfer::TokenId>;

ninfer::RequestOptions greedy(std::uint32_t outputs, bool reuse) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature = 0.0F;
    options.execution.allow_prefix_reuse = reuse;
    options.stop.include_model_defaults = false;
    return options;
}

struct Directory {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("ninfer-dflash-cancel-" + std::to_string(::getpid()));
    Directory() {
        std::filesystem::remove_all(path);
        std::filesystem::create_directories(path);
    }
    ~Directory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

void exercise(const char* artifact, int tier, unsigned publications) {
    Directory directory;
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.max_concurrency = 1;
    options.max_context = 256;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(256);
    options.prefill_chunk = 128;
    options.kv_ram_capacity_bytes = 2ULL << 30;
    options.enable_vision = false;
    options.speculative.backend = ninfer::SpeculativeBackend::DFlash;
    options.speculative.draft_tokens = 4;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    if (tier == 2) {
        options.kv_disk_capacity_bytes = 4ULL << 30;
        options.kv_disk_location = directory.path;
    }
    auto engine = std::make_unique<ninfer::Engine>(options);
    const Tokens prompt = {248045, 846, 198, 5834, 248046, 198};
    auto handle = engine->submit(engine->prepare_tokens(prompt), greedy(128, false),
                                 ninfer::OutputDelivery::Streaming);
    struct Published : ninfer::OutputSink {
        unsigned count = 0;
        void publish(ninfer::OutputDelta) override { ++count; }
    } sink;
    ninfer::CancellationView cancel([&] { return sink.count >= publications; });
    const auto stopped = handle.wait(&sink, cancel);
    if (stopped.finish_reason != ninfer::FinishReason::Cancelled ||
        stopped.generated_token_ids.empty() || stopped.speculative.rounds == 0) {
        throw std::runtime_error("DFlash fixture did not cancel after speculative publication");
    }
    // A committed state represents every published token except its final anchor.
    // This exact prefix needs the hidden image at E; a suffix would hide the bug by
    // computing fresh hidden state before sampling.
    Tokens exact = prompt;
    exact.insert(exact.end(), stopped.generated_token_ids.begin(),
                 stopped.generated_token_ids.end() - 1);
    if (tier != 0) {
        const Tokens replacement = {248045, 846, 198, 9906, 248046, 198};
        (void)engine->generate(engine->prepare_tokens(replacement), greedy(2, false));
        if (engine->runtime_stats().kv_ram_captures == 0) {
            throw std::runtime_error("DFlash cancellation image never reached RAM");
        }
    }
    if (tier == 2) {
        engine.reset(); // Shutdown drains spill; reopening forces the disk route.
        engine = std::make_unique<ninfer::Engine>(options);
        if (engine->memory_summary().kv_disk_entry_count == 0) {
            throw std::runtime_error("DFlash cancellation fixture did not survive disk reopen");
        }
    }
    const auto resumed = engine->generate(engine->prepare_tokens(exact), greedy(1, true));
    const auto fresh = engine->generate(engine->prepare_tokens(exact), greedy(1, false));
    if (resumed.generated_token_ids != fresh.generated_token_ids) {
        throw std::runtime_error("DFlash exact-prefix cancellation continuation changed next token");
    }
    std::cout << "DFlash cancellation exact prefix tier=" << tier
              << " publications=" << publications << " passed\n";
}

// The historical C=2 symptom had one queued request and no active GPU work.
// A blocked admission cannot honor cancellation, and Engine destruction may
// wait for that same worker. Report live counters and exit directly on timeout.
class C2Deadline {
public:
    explicit C2Deadline(ninfer::Engine& engine) : worker_([this, &engine] {
        std::unique_lock lock(mutex_);
        if (cv_.wait_for(lock, std::chrono::seconds(180), [&] { return done_; })) { return; }
        const auto stats = engine.runtime_stats();
        std::cerr << "DFlash C2 admission timeout phase=" << phase.load()
                  << " waiting=" << stats.waiting_requests
                  << " running=" << stats.running_requests
                  << " prefill=" << stats.prefilling_requests
                  << " decode_ready=" << stats.decode_ready_requests
                  << " computed=" << stats.computed_prefill_tokens
                  << " decoded=" << stats.committed_decode_tokens << std::endl;
        std::_Exit(1);
    }) {}
    ~C2Deadline() {
        { std::lock_guard lock(mutex_); done_ = true; }
        cv_.notify_all();
        worker_.join();
    }
    std::atomic<const char*> phase{"start"};
private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool done_ = false;
    std::thread worker_;
};

void exercise_c2_admission(const char* artifact) {
    Directory directory;
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.max_concurrency = 2;
    options.max_context = 256;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(512);
    options.prefill_chunk = 128;
    options.kv_ram_capacity_bytes = 2ULL << 30;
    options.kv_disk_capacity_bytes = 4ULL << 30;
    options.kv_disk_location = directory.path;
    options.enable_vision = false;
    options.speculative.backend = ninfer::SpeculativeBackend::DFlash;
    options.speculative.draft_tokens = 4;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    const Tokens prompt = {248045, 846, 198, 5834, 248046, 198};
    const Tokens other = {248045, 846, 198, 9906, 248046, 198};
    Tokens exact;
    auto engine = std::make_unique<ninfer::Engine>(options);
    {
        C2Deadline deadline(*engine);
        deadline.phase = "initial DFlash";
        const auto initial = engine->generate(engine->prepare_tokens(prompt), greedy(16, false));
        if (initial.speculative.rounds == 0 || initial.generated_token_ids.size() < 2) {
            throw std::runtime_error("C2 fixture did not execute DFlash");
        }
        exact = prompt;
        exact.insert(exact.end(), initial.generated_token_ids.begin(),
                     initial.generated_token_ids.end() - 1);
        // Every request starts with no live peer. Two distinct occupants force
        // both retained lanes through capture before the next host-cache retry.
        auto pressure = [&] {
            for (ninfer::TokenId token : {9906, 728, 2047}) {
                Tokens changed = other;
                changed[3] = token;
                (void)engine->generate(engine->prepare_tokens(changed), greedy(4, false));
            }
        };
        deadline.phase = "retain both lanes and spill";
        pressure();
        deadline.phase = "RAM exact-prefix admission";
        const auto cached = engine->generate(engine->prepare_tokens(exact), greedy(1, true));
        if (cached.prefix_reuse_source != ninfer::PrefixReuseSource::HostRam) {
            throw std::runtime_error("C2 fixture did not restore the exact prefix from RAM");
        }
        const auto fresh = engine->generate(engine->prepare_tokens(exact), greedy(1, false));
        if (cached.generated_token_ids != fresh.generated_token_ids) {
            throw std::runtime_error("C2 DFlash RAM exact-prefix next token differs");
        }
        deadline.phase = "repeated exact and suffix admissions";
        for (unsigned round = 0; round < 3; ++round) {
            Tokens suffix = exact;
            suffix.push_back(198);
            suffix.push_back(static_cast<ninfer::TokenId>(728 + round));
            (void)engine->generate(engine->prepare_tokens(exact), greedy(4, true));
            (void)engine->generate(engine->prepare_tokens(suffix), greedy(4, true));
            pressure();
        }
        deadline.phase = "cancel DFlash and readmit exact prefix";
        auto handle = engine->submit(engine->prepare_tokens(prompt), greedy(128, false),
                                     ninfer::OutputDelivery::Streaming);
        struct Sink : ninfer::OutputSink {
            unsigned publications = 0;
            void publish(ninfer::OutputDelta) override { ++publications; }
        } sink;
        const auto stopped = handle.wait(&sink, ninfer::CancellationView([&] {
            return sink.publications >= 2;
        }));
        if (stopped.finish_reason != ninfer::FinishReason::Cancelled ||
            stopped.speculative.rounds == 0 || stopped.generated_token_ids.empty()) {
            throw std::runtime_error("C2 fixture did not cancel a speculative request");
        }
        Tokens cancelled = prompt;
        cancelled.insert(cancelled.end(), stopped.generated_token_ids.begin(),
                         stopped.generated_token_ids.end() - 1);
        pressure();
        const auto resumed = engine->generate(engine->prepare_tokens(cancelled), greedy(1, true));
        const auto oracle = engine->generate(engine->prepare_tokens(cancelled), greedy(1, false));
        if (resumed.generated_token_ids != oracle.generated_token_ids) {
            throw std::runtime_error("C2 cancelled DFlash continuation changed next token");
        }
        // Keep the reopen witness recent enough for the bounded disk budget;
        // earlier lifecycle pressure is allowed to evict the original image.
        deadline.phase = "refresh disk reopen witness";
        (void)engine->generate(engine->prepare_tokens(exact), greedy(1, false));
        pressure();
    }
    engine.reset(); // Drain the actual idle spills before forcing disk admission.
    engine = std::make_unique<ninfer::Engine>(options);
    {
        C2Deadline deadline(*engine);
        deadline.phase = "disk exact-prefix admission after reopen";
        const auto restored = engine->generate(engine->prepare_tokens(exact), greedy(1, true));
        if (restored.prefix_reuse_source != ninfer::PrefixReuseSource::HostDisk) {
            throw std::runtime_error("C2 DFlash fixture did not restore from disk");
        }
        const auto oracle = engine->generate(engine->prepare_tokens(exact), greedy(1, false));
        if (restored.generated_token_ids != oracle.generated_token_ids) {
            throw std::runtime_error("C2 DFlash disk continuation changed next token");
        }
        deadline.phase = "disk session changed turn and suffix";
        (void)engine->generate(engine->prepare_tokens(other), greedy(8, false));
        Tokens suffix = exact;
        suffix.push_back(198);
        (void)engine->generate(engine->prepare_tokens(suffix), greedy(8, true));
    }
    std::cout << "DFlash C2 idle admission RAM/disk lifecycle passed\n";

    engine.reset();
    options.kv_ram_capacity_bytes = 256ULL << 20;
    options.kv_disk_location = directory.path / "full-ram";
    engine = std::make_unique<ninfer::Engine>(options);
    {
        C2Deadline deadline(*engine);
        for (unsigned round = 0; round < 3; ++round) {
            deadline.phase = "seed disk target with one-entry RAM";
            Tokens source = prompt;
            source[3] += round;
            const auto first = engine->generate(engine->prepare_tokens(source), greedy(16, false));
            if (first.speculative.rounds == 0 || first.generated_token_ids.size() < 2) {
                throw std::runtime_error("full-RAM C2 fixture did not execute DFlash");
            }
            Tokens history = source;
            history.insert(history.end(), first.generated_token_ids.begin(),
                           first.generated_token_ids.end() - 1);
            // C2 takes its empty lane first. Four distinct replacements then
            // occupy both retained lanes and move the target out of the single
            // RAM slot into disk. No idle wait is inserted between admissions.
            deadline.phase = "fill both retained lanes and exhaust RAM";
            for (ninfer::TokenId token : {9906, 728, 2047, 5830}) {
                Tokens occupant = other;
                occupant[3] = token;
                (void)engine->generate(engine->prepare_tokens(occupant), greedy(8, false));
            }
            const auto before = engine->runtime_stats();
            if (before.kv_ram_used_bytes == 0 ||
                before.kv_ram_used_bytes * 2 <= options.kv_ram_capacity_bytes) {
                throw std::runtime_error("C2 pressure budget does not constrain RAM to one image");
            }
            deadline.phase = "disk claim and prefetch before full-RAM retained capture";
            const auto restored = engine->generate(engine->prepare_tokens(history), greedy(1, true));
            const auto after = engine->runtime_stats();
            if (restored.prefix_reuse_source != ninfer::PrefixReuseSource::HostDisk ||
                after.kv_ram_captures <= before.kv_ram_captures ||
                after.kv_ram_evictions <= before.kv_ram_evictions) {
                throw std::runtime_error(
                    "C2 disk admission did not capture a retained victim under full RAM pressure");
            }
            deadline.phase = "full-RAM disk continuation oracle";
            const auto oracle = engine->generate(engine->prepare_tokens(history), greedy(1, false));
            if (restored.generated_token_ids != oracle.generated_token_ids) {
                throw std::runtime_error("full-RAM C2 disk continuation changed next token");
            }
        }
    }
    std::cout << "DFlash C2 disk admission with full RAM and retained lanes passed\n";
}

} // namespace

int main(int argc, char** argv) {
    const bool c2_only = argc == 3 && std::string(argv[1]) == "--case" &&
                         std::string(argv[2]) == "c2";
    if (argc != 1 && !c2_only) {
        std::cerr << "usage: dflash_cache_cancel_real [--case c2]\n";
        return 1;
    }
    const char* artifact = std::getenv("NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: set NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS\n";
        return 77;
    }
    try {
        if (!c2_only) {
            for (int tier = 0; tier < 3; ++tier) {
                for (unsigned publications : {2U, 3U}) { exercise(artifact, tier, publications); }
            }
        }
        exercise_c2_admission(artifact);
    } catch (const std::exception& error) {
        std::cerr << "DFlash cache cancellation: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
