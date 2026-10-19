#include "ninfer/engine.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: set NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS\n";
        return 77;
    }
    try {
        ninfer::EngineOptions options;
        options.artifact_path              = artifact;
        options.max_context                = 256;
        options.max_concurrency            = 6;
        options.kv_capacity                = ninfer::KvCapacityPolicy::explicit_capacity(1536);
        options.kv_cache                   = ninfer::KvCacheStorage::Nvfp4;
        options.prefill_chunk              = 128;
        options.mixed_forward              = 0;
        options.use_cuda_graph             = true;
        options.enable_vision              = false;
        options.speculative.backend        = ninfer::SpeculativeBackend::DFlash;
        options.speculative.draft_tokens   = 7;
        options.speculative.adaptive_draft = true;
        options.speculative.proposal_head  = ninfer::ProposalHead::Optimized;
        // Leave verify width automatic: the W=12 tree arm establishes the wide replay-record
        // destination even when B=3 uses the initial k=3 chain. Startup validates every captured
        // context profile for B=1..6, k=3..7 and the reachable tree shapes.
        ninfer::Engine engine(options);
        const std::array<std::size_t, 3> lengths{124, 125, 127};
        std::vector<ninfer::GenerationHandle> handles;
        for (const auto length : lengths) {
            std::vector<ninfer::TokenId> tokens(length, 24);
            ninfer::RequestOptions request;
            request.execution.requested_output_tokens = 12;
            request.execution.sampling.temperature    = 0.0f;
            request.execution.allow_prefix_reuse      = false;
            request.stop.include_model_defaults       = false;
            handles.push_back(engine.submit(engine.prepare_tokens(tokens), request));
        }
        for (auto& handle : handles) {
            const auto result = handle.wait();
            if (result.generated_token_ids.size() != 12 || result.speculative.rounds == 0) {
                throw std::runtime_error("DFlash graph boundary request failed to complete");
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "DFlash graph profile regression failed: " << error.what() << '\n';
        return 1;
    }
}
