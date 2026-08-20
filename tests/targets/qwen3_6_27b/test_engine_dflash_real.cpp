#include "ninfer/engine.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

ninfer::EngineOptions base_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path  = artifact;
    options.max_context    = 256;
    options.kv_capacity    = ninfer::KvCapacityPolicy::explicit_capacity(256);
    options.prefill_chunk  = 128;
    options.kv_cache       = ninfer::KvCacheStorage::BFloat16;
    options.use_cuda_graph = false;
    options.enable_vision  = false;
    return options;
}

ninfer::EngineOptions speculative_engine_options(const char* artifact,
                                                 ninfer::SpeculativeBackend backend,
                                                 std::uint32_t draft_tokens) {
    ninfer::EngineOptions options     = base_engine_options(artifact);
    options.speculative.backend       = backend;
    options.speculative.draft_tokens  = draft_tokens;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    options.use_cuda_graph            = true;
    return options;
}

ninfer::RequestOptions greedy_options(std::uint32_t outputs) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = false;
    options.stop.include_model_defaults       = false;
    return options;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS is not set\n";
        return 77;
    }

    // NVFP4 AllowA4 verify at T>=4 uses W4A4 attention; ordinary T=1 decode stays A16.
    // MTP k=3 (T=4) and DFlash k=7 (T=8) share that verify family. DFlash k=1 (T=2)
    // matches ordinary T=1 on this fixture and is not the product command.
    const std::vector<ninfer::TokenId> prompt{
        248045, 846,    198, 109266, 3709,  96220, 117443, 97913,
        1710,   248046, 198, 248045, 74455, 198,   248068, 198,
    };
    std::vector<ninfer::TokenId> mtp_output;
    {
        ninfer::Engine mtp(
            speculative_engine_options(artifact, ninfer::SpeculativeBackend::Mtp, 3));
        mtp_output = mtp.generate(mtp.prepare_tokens(prompt), greedy_options(24)).generated_token_ids;
        if (mtp_output.size() != 24) {
            std::cerr << "MTP k=3 greedy baseline did not generate 24 tokens\n";
            return 1;
        }
    }

    ninfer::Engine engine(
        speculative_engine_options(artifact, ninfer::SpeculativeBackend::DFlash, 7));
    const ninfer::LoadSummary load = engine.load_summary();
    if (load.target != "qwen3_8_27b" || load.weights_id != "nvfp4" ||
        load.host_to_device_bytes == 0) {
        std::cerr << "DFlash2 Engine materialized an invalid artifact payload: target="
                  << load.target << " weights=" << load.weights_id << '\n';
        return 1;
    }
    engine.reset_memory_peaks();
    const ninfer::GenerationResult dflash =
        engine.generate(engine.prepare_tokens(prompt), greedy_options(24));
    if (dflash.generated_token_ids != mtp_output) {
        const auto mismatch =
            std::mismatch(dflash.generated_token_ids.begin(), dflash.generated_token_ids.end(),
                          mtp_output.begin(), mtp_output.end());
        std::cerr << "DFlash2 Graph route diverged from MTP k=3 greedy target output at "
                  << static_cast<std::size_t>(mismatch.first - dflash.generated_token_ids.begin())
                  << ": dflash="
                  << (mismatch.first == dflash.generated_token_ids.end() ? -1 : *mismatch.first)
                  << " mtp=" << (mismatch.second == mtp_output.end() ? -1 : *mismatch.second)
                  << " rounds=" << dflash.speculative.rounds
                  << " accepted=" << dflash.speculative.accepted_tokens
                  << " drafted=" << dflash.speculative.drafted_tokens << '\n';
        return 1;
    }
    if (dflash.speculative.backend != ninfer::SpeculativeBackend::DFlash ||
        dflash.speculative.rounds == 0) {
        std::cerr << "DFlash2 fixture did not execute speculative decode\n";
        return 1;
    }

    std::cout << "ok drafts=" << dflash.speculative.drafted_tokens
              << " accepted=" << dflash.speculative.accepted_tokens
              << " rounds=" << dflash.speculative.rounds << '\n';
    return 0;
}
