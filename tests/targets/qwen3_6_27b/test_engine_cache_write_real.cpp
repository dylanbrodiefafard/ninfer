#include "ninfer/engine.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <unistd.h>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

ninfer::RequestOptions request_options() {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = 4;
    options.execution.sampling.temperature    = 0.0F;
    options.stop.include_model_defaults       = false;
    return options;
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

void ordinary_continuation(const char* artifact, std::uint32_t lanes) {
    ninfer::EngineOptions config;
    config.artifact_path         = artifact;
    config.max_concurrency       = lanes;
    config.max_context           = 256;
    config.kv_capacity           = ninfer::KvCapacityPolicy::explicit_capacity(256 * lanes);
    config.kv_ram_capacity_bytes = 1ULL << 30;
    ninfer::Engine engine(config);
    std::vector<ninfer::TokenId> original(67, 198);
    original.front() = 1000;
    const auto first = engine.generate(engine.prepare_tokens(original), request_options());
    require(first.generated_token_ids.size() == 4, "original turn did not complete");
    original.insert(original.end(), first.generated_token_ids.begin(),
                    first.generated_token_ids.end());

    if (lanes == 1) {
        std::vector<ninfer::TokenId> unrelated(32, 198);
        unrelated.front() = 2000;
        (void)engine.generate(engine.prepare_tokens(unrelated), request_options());
        require(engine.runtime_stats().kv_ram_entry_count == 1,
                "ordinary eviction did not save the original");
    }
    const auto next = engine.generate(engine.prepare_tokens(original), request_options());
    require(next.generated_token_ids.size() == 4 &&
                next.prefix_reuse_path == ninfer::PrefixReusePath::AppendAtFrontier,
            "ordinary continuation did not generate at its resume frontier");
    require(next.reused_prompt_tokens == 70,
            "ordinary continuation lost its exact resume frontier");
    require(next.prefix_reuse_source == (lanes == 1 ? ninfer::PrefixReuseSource::HostRam
                                                    : ninfer::PrefixReuseSource::VramResident),
            "ordinary continuation selected the wrong tier");
    if (lanes == 1) {
        require(engine.runtime_stats().kv_ram_restores == 1, "ordinary RAM hit was not consumed");
    }
}

ninfer::EngineOptions fork_config(const char* artifact, std::uint32_t lanes,
                                  ninfer::SpeculativeBackend backend) {
    ninfer::EngineOptions config;
    config.artifact_path            = artifact;
    config.max_concurrency          = lanes;
    config.max_context              = 256;
    config.kv_capacity              = ninfer::KvCapacityPolicy::explicit_capacity(256 * lanes);
    config.kv_ram_capacity_bytes    = 1ULL << 30;
    config.speculative.backend      = backend;
    config.speculative.draft_tokens = backend == ninfer::SpeculativeBackend::None ? 0 : 4;
    if (backend != ninfer::SpeculativeBackend::None) {
        config.context_checkpoint_marks = std::vector<std::uint32_t>{64};
    }
    config.prefill_chunk = 128;
    return config;
}

void disposable_fork(const char* artifact, std::uint32_t lanes, ninfer::SpeculativeBackend backend,
                     bool checkpoint_hit) {
    std::cout << "fork lanes=" << lanes << " backend=" << static_cast<int>(backend)
              << " checkpoint=" << checkpoint_hit << '\n';
    ninfer::Engine engine(fork_config(artifact, lanes, backend));
    std::vector<ninfer::TokenId> original(67, 198);
    original.front() = 1000;
    const auto first = engine.generate(engine.prepare_tokens(original), request_options());
    std::vector<ninfer::TokenId> continuation = original;
    continuation.insert(continuation.end(), first.generated_token_ids.begin(),
                        first.generated_token_ids.end());
    auto branch = checkpoint_hit ? original : continuation;
    branch.push_back(3000);
    auto disposable                  = request_options();
    disposable.execution.cache_write = false;
    const auto before                = engine.runtime_stats();
    const auto fork                  = engine.generate(engine.prepare_tokens(branch), disposable);
    require(fork.generated_token_ids.size() == 4, "disposable fork did not complete");
    require(fork.reused_prompt_tokens == (checkpoint_hit ? 67U : 70U),
            "fork lost the selected frontier");
    require(fork.prefix_reuse_source == (lanes == 1 ? ninfer::PrefixReuseSource::HostRam
                                                    : ninfer::PrefixReuseSource::VramResident),
            "fork initialized from the wrong tier");
    require(fork.captured_context_checkpoint_tokens == 0,
            "disposable fork captured a reuse checkpoint");
    const auto after = engine.runtime_stats();
    require(after.kv_ram_captures == before.kv_ram_captures + (lanes == 1 ? 1 : 0),
            "fork captured unexpected RAM state");
    if (lanes == 1) {
        require(after.kv_ram_entry_count == 1 && after.kv_ram_restores == 1,
                "fork consumed its preserved original");
    } else {
        // Both lanes are dirty. The disposable branch is newer, but must be evicted before A.
        std::vector<ninfer::TokenId> unrelated(32, 198);
        unrelated.front() = 2000;
        (void)engine.generate(engine.prepare_tokens(unrelated), request_options());
        require(engine.runtime_stats().kv_ram_captures == after.kv_ram_captures,
                "disposable branch was written to RAM during lane eviction");
    }
    continuation.push_back(4000);
    const auto resumed = engine.generate(engine.prepare_tokens(continuation), request_options());
    require(resumed.generated_token_ids.size() == 4 && resumed.reused_prompt_tokens == 70,
            "original no longer continues after the disposable turn");
    require(resumed.prefix_reuse_source == (lanes == 1 ? ninfer::PrefixReuseSource::HostRam
                                                       : ninfer::PrefixReuseSource::VramResident),
            "disposable eviction removed the original");
    if (lanes == 1) {
        require(engine.runtime_stats().kv_ram_entry_count == 0 &&
                    engine.runtime_stats().kv_ram_restores == 2,
                "ordinary continuation did not consume the original RAM entry");
    }
    auto cold                         = request_options();
    cold.execution.allow_prefix_reuse = false;
    const auto control_first          = engine.generate(engine.prepare_tokens(original), cold);
    require(control_first.generated_token_ids == first.generated_token_ids,
            "ordinary control did not reproduce the source turn");
    const auto control_resume =
        engine.generate(engine.prepare_tokens(continuation), request_options());
    require(control_resume.generated_token_ids == resumed.generated_token_ids,
            "fork changed the original continuation relative to ordinary reuse");
    (void)engine.generate(engine.prepare_tokens(original), cold);
    const auto control_branch = engine.generate(engine.prepare_tokens(branch), request_options());
    require(control_branch.generated_token_ids == fork.generated_token_ids,
            "fork output differs from ordinary execution at the same cached frontier");
}

void preservation_capacity(const char* artifact) {
    auto config                  = fork_config(artifact, 1, ninfer::SpeculativeBackend::None);
    config.kv_ram_capacity_bytes = 0;
    ninfer::Engine engine(config);
    std::vector<ninfer::TokenId> original(67, 198);
    const auto first = engine.generate(engine.prepare_tokens(original), request_options());
    original.insert(original.end(), first.generated_token_ids.begin(),
                    first.generated_token_ids.end());
    auto disposable                  = request_options();
    disposable.execution.cache_write = false;
    bool rejected                    = false;
    try {
        (void)engine.generate(engine.prepare_tokens(original), disposable);
    } catch (const ninfer::RequestError& error) {
        rejected = error.kind() == ninfer::RequestErrorKind::Overloaded;
    }
    require(rejected, "one-lane fork destroyed its source or waited without usable RAM");
    const auto next = engine.generate(engine.prepare_tokens(original), request_options());
    require(next.reused_prompt_tokens == 70 && next.generated_token_ids.size() == 4,
            "failed preservation changed the original");
    disposable.execution.capture_context_checkpoint = true;
    try {
        (void)engine.submit(engine.prepare_tokens(original), disposable);
        throw std::runtime_error("Engine accepted conflicting disposable capture");
    } catch (const ninfer::RequestError& error) {
        require(error.kind() == ninfer::RequestErrorKind::CacheWriteConflict,
                "Engine conflict lost its typed error");
    }
}

void disposable_promotion(const char* artifact) {
    ninfer::Engine engine(fork_config(artifact, 1, ninfer::SpeculativeBackend::None));
    std::vector<ninfer::TokenId> original(67, 198);
    auto disposable                  = request_options();
    disposable.execution.cache_write = false;
    const auto first                 = engine.generate(engine.prepare_tokens(original), disposable);
    original.insert(original.end(), first.generated_token_ids.begin(),
                    first.generated_token_ids.end());
    bool rejected = false;
    try {
        (void)engine.generate(engine.prepare_tokens(original), disposable);
    } catch (const ninfer::RequestError& error) {
        rejected = error.kind() == ninfer::RequestErrorKind::Overloaded;
    }
    require(rejected && engine.runtime_stats().kv_ram_captures == 0,
            "second disposable request offloaded its disposable source");
    const auto promoted = engine.generate(engine.prepare_tokens(original), request_options());
    require(promoted.reused_prompt_tokens == 70,
            "ordinary continuation could not promote disposable state");
    original.front() = 2000;
    (void)engine.generate(engine.prepare_tokens(original), disposable);
    require(engine.runtime_stats().kv_ram_captures == 1,
            "promoted state was not eligible for RAM capture");
}

void mtp_single_token_fork(const char* artifact) {
    ninfer::Engine engine(fork_config(artifact, 2, ninfer::SpeculativeBackend::Mtp));
    auto ordinary                              = request_options();
    ordinary.execution.requested_output_tokens = 1;
    (void)engine.generate(engine.prepare_tokens({198}), ordinary);
    auto disposable                  = ordinary;
    disposable.execution.cache_write = false;
    const auto fork = engine.generate(engine.prepare_tokens({198, 3000}), disposable);
    require(fork.reused_prompt_tokens == 1 &&
                fork.prefix_reuse_source == ninfer::PrefixReuseSource::VramResident,
            "MTP fork did not support an empty backend KV frontier");
    const auto continued = engine.generate(engine.prepare_tokens({198, 4000}), ordinary);
    require(continued.reused_prompt_tokens == 1, "single-token MTP fork changed its original");
}

void concurrent_original_and_fork(const char* artifact, ninfer::SpeculativeBackend backend) {
    ninfer::Engine engine(fork_config(artifact, 2, backend));
    std::vector<ninfer::TokenId> original(67, 198);
    original.front() = 1000;
    const auto first = engine.generate(engine.prepare_tokens(original), request_options());
    original.insert(original.end(), first.generated_token_ids.begin(),
                    first.generated_token_ids.end());
    auto branch = original;
    branch.push_back(3000);
    original.push_back(4000);
    std::vector<ninfer::TokenId> unrelated(32, 198);
    unrelated.front()                          = 2000;
    auto fork_prompt                           = engine.prepare_tokens(branch);
    auto original_prompt                       = engine.prepare_tokens(original);
    auto queued_prompt                         = engine.prepare_tokens(unrelated);
    auto ordinary                              = request_options();
    ordinary.execution.requested_output_tokens = 128;
    auto disposable                            = ordinary;
    disposable.execution.cache_write           = false;
    auto fork                                  = engine.submit(std::move(fork_prompt), disposable);
    auto continued                       = engine.submit(std::move(original_prompt), ordinary);
    auto queued_options                  = request_options();
    queued_options.execution.cache_write = false;
    auto queued                          = engine.submit(std::move(queued_prompt), queued_options);
    const auto fork_result               = fork.wait();
    const auto continued_result          = continued.wait();
    const auto queued_result             = queued.wait();
    require(fork_result.generated_token_ids.size() == 128 &&
                continued_result.generated_token_ids.size() == 128 &&
                fork_result.reused_prompt_tokens == 70 &&
                continued_result.reused_prompt_tokens == 70 &&
                fork_result.prefix_reuse_source == ninfer::PrefixReuseSource::VramResident &&
                continued_result.prefix_reuse_source == ninfer::PrefixReuseSource::VramResident,
            "concurrent original continuation and fork did not own independent lanes");
    require(fork_result.captured_context_checkpoint_tokens == 0 &&
                queued_result.generated_token_ids.size() == 4,
            "fork captured a checkpoint or FIFO admission stalled behind active lanes");
}

void insufficient_pages_and_ram(const char* artifact, bool ram_fits) {
    auto config                  = fork_config(artifact, 2, ninfer::SpeculativeBackend::None);
    config.kv_capacity           = ninfer::KvCapacityPolicy::explicit_capacity(256);
    config.kv_ram_capacity_bytes = ram_fits ? 1ULL << 30 : 1ULL << 20;
    config.pending_timeout_ms    = 2000;
    ninfer::Engine engine(config);
    std::vector<ninfer::TokenId> original(67, 198);
    original.front() = 1000;
    const auto first = engine.generate(engine.prepare_tokens(original), request_options());
    auto cold        = request_options();
    cold.execution.allow_prefix_reuse = false;
    (void)engine.generate(engine.prepare_tokens(original), cold);
    original.insert(original.end(), first.generated_token_ids.begin(),
                    first.generated_token_ids.end());
    auto branch = original;
    branch.insert(branch.end(), 80, 3000);
    auto disposable                  = request_options();
    disposable.execution.cache_write = false;
    if (ram_fits) {
        const auto fork = engine.generate(engine.prepare_tokens(branch), disposable);
        require(fork.prefix_reuse_source == ninfer::PrefixReuseSource::HostRam &&
                    fork.reused_prompt_tokens == 70,
                "page-pressure fork did not preserve its source in RAM");
        require(engine.runtime_stats().kv_ram_entry_count == 2,
                "page-pressure fork did not preserve the source and reclaimed ordinary victim");
    } else {
        bool rejected = false;
        try {
            (void)engine.generate(engine.prepare_tokens(branch), disposable);
        } catch (const ninfer::RequestError& error) {
            rejected = error.kind() == ninfer::RequestErrorKind::Overloaded;
        }
        require(rejected, "two-source preservation did not reject bounded impossible capacity");
    }
    const auto resumed = engine.generate(engine.prepare_tokens(original), request_options());
    require(resumed.reused_prompt_tokens == 70,
            "page-pressure admission changed the original source frontier");
}

class TemporaryDiskCache {
public:
    TemporaryDiskCache()
        : path(std::filesystem::temp_directory_path() /
               ("ninfer-cache-write-" + std::to_string(::getpid()))) {
        std::filesystem::remove_all(path);
        std::filesystem::create_directories(path);
    }

    ~TemporaryDiskCache() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }

    TemporaryDiskCache(const TemporaryDiskCache&)            = delete;
    TemporaryDiskCache& operator=(const TemporaryDiskCache&) = delete;
    std::filesystem::path path;
};

class CancelAfterOutput final : public ninfer::OutputSink {
public:
    void publish(ninfer::OutputDelta) override { emitted = true; }

    bool emitted = false;
};

void disk_forks_and_shutdown(const char* artifact) {
    TemporaryDiskCache disk;
    auto config                   = fork_config(artifact, 1, ninfer::SpeculativeBackend::DFlash);
    config.kv_disk_location       = disk.path;
    config.kv_disk_capacity_bytes = 2ULL << 30;
    std::vector<ninfer::TokenId> original(67, 198);
    original.front()                 = 1000;
    auto disposable                  = request_options();
    disposable.execution.cache_write = false;
    {
        ninfer::Engine engine(config);
        const auto first = engine.generate(engine.prepare_tokens(original), request_options());
        original.insert(original.end(), first.generated_token_ids.begin(),
                        first.generated_token_ids.end());
        auto branch = original;
        branch.push_back(3000);
        (void)engine.generate(engine.prepare_tokens(branch), disposable);
        require(engine.runtime_stats().kv_ram_captures == 1,
                "one-lane disk fixture did not preserve the original");
        branch.front() = 2000;
        (void)engine.generate(engine.prepare_tokens(branch), disposable);
        require(engine.runtime_stats().kv_ram_captures == 1,
                "disposable eviction wrote another RAM entry");
    }
    {
        ninfer::Engine engine(config);
        auto branch = original;
        branch.push_back(4000);
        const auto fork = engine.generate(engine.prepare_tokens(branch), disposable);
        require(fork.prefix_reuse_source == ninfer::PrefixReuseSource::HostDisk &&
                    fork.reused_prompt_tokens == 70 && fork.captured_context_checkpoint_tokens == 0,
                "disk fork did not read the original frontier");
        require(engine.runtime_stats().kv_disk_entry_count == 1 &&
                    engine.runtime_stats().kv_ram_captures == 0,
                "disk fork consumed or rewrote the original");
        disposable.execution.requested_output_tokens = 128;
        CancelAfterOutput sink;
        const ninfer::CancellationView cancel([&] { return sink.emitted; });
        const auto cancelled =
            engine.generate(engine.prepare_tokens(branch), disposable, &sink, cancel);
        require(cancelled.finish_reason == ninfer::FinishReason::Cancelled &&
                    cancelled.prefix_reuse_source == ninfer::PrefixReuseSource::HostDisk,
                "disk fork cancellation did not reach initialized execution");
        require(engine.runtime_stats().kv_disk_entry_count == 1 &&
                    engine.runtime_stats().kv_ram_captures == 0,
                "cancelled fork changed persistent cache membership");
    }
    {
        ninfer::Engine engine(config);
        const auto resumed = engine.generate(engine.prepare_tokens(original), request_options());
        require(resumed.prefix_reuse_source == ninfer::PrefixReuseSource::HostDisk &&
                    resumed.reused_prompt_tokens == 70 && resumed.generated_token_ids.size() == 4,
                "disk original was no longer continuable after forks and cancellation");
        require(engine.runtime_stats().kv_disk_entry_count == 1,
                "cancelled disposable state was persisted at shutdown");
    }
}

} // namespace

int main() {
    const char* dflash     = std::getenv("NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS");
    const char* mtp        = std::getenv("NINFER_QWEN3_8_27B_NVFP4_MTP_WEIGHTS");
    const bool have_dflash = dflash != nullptr && *dflash != '\0';
    const bool have_mtp    = mtp != nullptr && *mtp != '\0';
    if (!have_dflash && !have_mtp) {
        std::cout << "skip: set a supported NVFP4 DFlash or MTP artifact environment variable\n";
        return 77;
    }
    const char* artifact = have_dflash ? dflash : mtp;
    try {
        ordinary_continuation(artifact, 1);
        ordinary_continuation(artifact, 2);
        disposable_fork(artifact, 1, ninfer::SpeculativeBackend::None, false);
        disposable_fork(artifact, 2, ninfer::SpeculativeBackend::None, false);
        for (const auto backend :
             {ninfer::SpeculativeBackend::Mtp, ninfer::SpeculativeBackend::DFlash}) {
            const char* speculative_artifact =
                backend == ninfer::SpeculativeBackend::Mtp ? mtp : dflash;
            if (!(backend == ninfer::SpeculativeBackend::Mtp ? have_mtp : have_dflash)) {
                continue;
            }
            for (const auto lanes : {1U, 2U}) {
                disposable_fork(speculative_artifact, lanes, backend, false);
                disposable_fork(speculative_artifact, lanes, backend, true);
            }
        }
        if (have_mtp) { mtp_single_token_fork(mtp); }
        preservation_capacity(artifact);
        disposable_promotion(artifact);
        concurrent_original_and_fork(artifact, ninfer::SpeculativeBackend::None);
        if (have_dflash) {
            concurrent_original_and_fork(dflash, ninfer::SpeculativeBackend::DFlash);
        }
        insufficient_pages_and_ram(artifact, false);
        insufficient_pages_and_ram(artifact, true);
        if (have_dflash) { disk_forks_and_shutdown(dflash); }
        std::cout << "cache write Engine tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
