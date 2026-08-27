#include "ninfer/engine.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kMark            = 24576;
constexpr std::uint32_t kSecondMark      = 36864;
constexpr std::uint32_t kHundredMark     = 102400;
constexpr std::uint32_t k150Mark         = 151552;
constexpr std::uint32_t kCatchStart      = 8000;
// Resume 8000, then five 4096 chunks: 12096, 16192, 20288, 24384, 28480.
// Prompt is one token past that last chunk end so prefill commits 28480.
constexpr std::uint32_t kCatchFreeze     = 28481;
constexpr std::uint32_t kCatchFreezeF    = 28480;
constexpr std::uint32_t kDecodeStart     = 24000;
constexpr std::uint32_t kDecodeOutputs   = 800;
// One generated token leaves E == F, so a suffix of the frozen prefix is
// AppendAtFrontier. Two outputs move E past F and make RestoreContextCheckpoint
// observable on a probe that matches F but not E.
constexpr std::uint32_t kPastFreezeOutputs = 2;
// 24k INT8-G64 Main+MTP plus current/rewrite GDN and one ladder head is already
// >1 GiB. Catch-up at ~28k and two-mark at 36k need more; keep a 4 GiB FIFO so
// an eviction dump actually lands instead of returning FullReset.
constexpr std::size_t kRamBytes          = 4ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr ninfer::TokenId kPadA          = 198;
constexpr ninfer::TokenId kPadB          = 199;
constexpr ninfer::TokenId kPadC          = 201;
constexpr ninfer::TokenId kPadCatch      = 202;
constexpr ninfer::TokenId kPadCancel     = 203;
constexpr ninfer::TokenId kPadCancelB    = 204;
constexpr ninfer::TokenId kPadC2A        = 210;
constexpr ninfer::TokenId kPadC2B        = 211;
constexpr ninfer::TokenId kPadC2C        = 212;
constexpr ninfer::TokenId kPadC2D        = 213;
constexpr ninfer::TokenId kPadC2E        = 214;
constexpr ninfer::TokenId kPadDecode     = 215;
constexpr ninfer::TokenId kPadEvict      = 216;
constexpr ninfer::TokenId kPadC3A        = 220;
constexpr ninfer::TokenId kPadC3B        = 221;
constexpr ninfer::TokenId kPadC3C        = 222;
constexpr ninfer::TokenId kPadC3D        = 223;
constexpr ninfer::TokenId kPadC3E        = 224;
constexpr ninfer::TokenId kPadEq         = 225;
constexpr ninfer::TokenId kPadEq2        = 226;
constexpr ninfer::TokenId kPadTwoMark    = 227;
constexpr ninfer::TokenId kPadMtpOff     = 228;
constexpr ninfer::TokenId kPadRb1        = 230;
constexpr ninfer::TokenId kPadRb2        = 231;
constexpr ninfer::TokenId kPadRb3        = 232;
constexpr ninfer::TokenId kPadRbEvict    = 233;
constexpr ninfer::TokenId kPadRbC2       = 234;
constexpr ninfer::TokenId kPadRbC3       = 235;
constexpr ninfer::TokenId kPadRbCold     = 236;
constexpr ninfer::TokenId kPadOracle     = 237;
constexpr ninfer::TokenId kPadRamShort   = 238;
constexpr ninfer::TokenId kPadC3F        = 239;
constexpr ninfer::TokenId kPad150        = 240;
constexpr ninfer::TokenId kPadDropA      = 241;
constexpr ninfer::TokenId kPadDropB      = 242;
constexpr ninfer::TokenId kDiverge       = 200;
constexpr ninfer::TokenId kFlip          = 197;

ninfer::EngineOptions engine_options(const char* artifact, std::uint32_t max_concurrency = 1,
                                     std::uint32_t kv_tokens = 32768, bool vision = false,
                                     std::uint32_t max_context = 32768,
                                     ninfer::SpeculativeBackend spec =
                                         ninfer::SpeculativeBackend::Mtp) {
    ninfer::EngineOptions options;
    options.artifact_path             = artifact;
    options.max_context               = max_context;
    options.kv_capacity               = ninfer::KvCapacityPolicy::explicit_capacity(kv_tokens);
    options.max_concurrency           = max_concurrency;
    options.prefill_chunk             = 4096;
    options.kv_ram_capacity_bytes     = kRamBytes;
    options.kv_cache                  = ninfer::KvCacheStorage::Int8Group64;
    options.speculative.backend       = spec;
    options.speculative.draft_tokens  = spec == ninfer::SpeculativeBackend::None ? 0 : 3;
    options.speculative.proposal_head =
        spec == ninfer::SpeculativeBackend::None ? ninfer::ProposalHead::Full
                                                 : ninfer::ProposalHead::Optimized;
    options.enable_vision             = vision;
    options.use_cuda_graph            = true;
    return options;
}

ninfer::RequestOptions greedy(std::uint32_t outputs, bool reuse) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

std::vector<ninfer::TokenId> padded(ninfer::TokenId pad, std::size_t count) {
    // Distinct first token so concurrent prefixes do not hash-collide. Fill with
    // 198 so greedy decode stays on valid standalone UTF-8 pieces.
    std::vector<ninfer::TokenId> ids(count, 198);
    if (count != 0) { ids[0] = pad; }
    return ids;
}

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

const char* path_name(ninfer::PrefixReusePath path) {
    switch (path) {
    case ninfer::PrefixReusePath::FullReset:
        return "full_reset";
    case ninfer::PrefixReusePath::AppendAtFrontier:
        return "append_at_frontier";
    case ninfer::PrefixReusePath::RestoreTurnCheckpoint:
        return "restore_turn_checkpoint";
    case ninfer::PrefixReusePath::RestoreResponseCheckpoint:
        return "restore_response_checkpoint";
    case ninfer::PrefixReusePath::RestoreContextCheckpoint:
        return "restore_context_checkpoint";
    case ninfer::PrefixReusePath::RestoreTurnRollback:
        return "restore_turn_rollback";
    }
    return "unknown";
}

int expect_captured(const ninfer::GenerationResult& result, std::uint32_t tokens,
                    const char* label) {
    if (result.captured_context_checkpoint_tokens != tokens) {
        std::cerr << label << " captured " << result.captured_context_checkpoint_tokens
                  << ", expected " << tokens << '\n';
        return 1;
    }
    return 0;
}

int expect_restore(const ninfer::GenerationResult& result, ninfer::PrefixReuseSource source,
                   std::uint32_t reused, const char* label, std::uint32_t restored = 0) {
    if (result.prefix_reuse_path != ninfer::PrefixReusePath::RestoreContextCheckpoint) {
        std::cerr << label << " path is " << path_name(result.prefix_reuse_path)
                  << ", expected restore_context_checkpoint\n";
        return 1;
    }
    if (result.prefix_reuse_source != source) {
        std::cerr << label << " source is " << static_cast<int>(result.prefix_reuse_source)
                  << ", expected " << static_cast<int>(source) << '\n';
        return 1;
    }
    if (result.reused_prompt_tokens != reused) {
        std::cerr << label << " reused " << result.reused_prompt_tokens << ", expected " << reused
                  << '\n';
        return 1;
    }
    const std::uint32_t expected_restored = restored == 0 ? reused : restored;
    if (result.restored_context_checkpoint_tokens != expected_restored) {
        std::cerr << label << " restored " << result.restored_context_checkpoint_tokens
                  << ", expected " << expected_restored << '\n';
        return 1;
    }
    if (result.generated_token_ids.empty()) {
        std::cerr << label << " produced no tokens\n";
        return 1;
    }
    return 0;
}

int expect_rollback(const ninfer::GenerationResult& result, ninfer::PrefixReuseSource source,
                    std::uint32_t reused, const char* label) {
    if (result.prefix_reuse_path != ninfer::PrefixReusePath::RestoreTurnRollback) {
        std::cerr << label << " path is " << path_name(result.prefix_reuse_path)
                  << ", expected restore_turn_rollback reused=" << result.reused_prompt_tokens
                  << '\n';
        return 1;
    }
    if (result.prefix_reuse_source != source) {
        std::cerr << label << " source is " << static_cast<int>(result.prefix_reuse_source)
                  << ", expected " << static_cast<int>(source) << '\n';
        return 1;
    }
    if (result.reused_prompt_tokens != reused) {
        std::cerr << label << " reused " << result.reused_prompt_tokens << ", expected " << reused
                  << '\n';
        return 1;
    }
    if (result.generated_token_ids.empty()) {
        std::cerr << label << " produced no tokens\n";
        return 1;
    }
    return 0;
}

int expect_same_tokens(const std::vector<ninfer::TokenId>& got,
                       const std::vector<ninfer::TokenId>& want, const char* label) {
    if (got == want) { return 0; }
    std::cerr << label << " greedy tokens diverged";
    if (!got.empty() && !want.empty()) {
        std::cerr << " first=" << got.front() << " vs " << want.front();
    }
    std::cerr << " (got " << got.size() << ", want " << want.size() << ")\n";
    return 1;
}

int expect_rewrite(const ninfer::GenerationResult& result, ninfer::PrefixReusePath path,
                   ninfer::PrefixReuseSource source, std::uint32_t reused, const char* label) {
    if (result.prefix_reuse_path != path) {
        std::cerr << label << " path is " << path_name(result.prefix_reuse_path) << ", expected "
                  << path_name(path) << '\n';
        return 1;
    }
    if (result.prefix_reuse_source != source) {
        std::cerr << label << " source is " << static_cast<int>(result.prefix_reuse_source)
                  << ", expected " << static_cast<int>(source) << '\n';
        return 1;
    }
    if (result.reused_prompt_tokens != reused) {
        std::cerr << label << " reused " << result.reused_prompt_tokens << ", expected " << reused
                  << '\n';
        return 1;
    }
    if (result.restored_context_checkpoint_tokens != 0) {
        std::cerr << label << " restored staged head " << result.restored_context_checkpoint_tokens
                  << ", expected 0 on rewrite restore\n";
        return 1;
    }
    if (result.generated_token_ids.empty()) {
        std::cerr << label << " produced no tokens\n";
        return 1;
    }
    return 0;
}

int expect_append(const ninfer::GenerationResult& result, ninfer::PrefixReuseSource source,
                  std::uint32_t reused, const char* label) {
    if (result.prefix_reuse_path != ninfer::PrefixReusePath::AppendAtFrontier) {
        std::cerr << label << " path is " << path_name(result.prefix_reuse_path)
                  << ", expected append_at_frontier\n";
        return 1;
    }
    if (result.prefix_reuse_source != source) {
        std::cerr << label << " source is " << static_cast<int>(result.prefix_reuse_source)
                  << ", expected " << static_cast<int>(source) << '\n';
        return 1;
    }
    if (result.reused_prompt_tokens != reused) {
        std::cerr << label << " reused " << result.reused_prompt_tokens << ", expected " << reused
                  << '\n';
        return 1;
    }
    if (result.restored_context_checkpoint_tokens != 0) {
        std::cerr << label << " restored " << result.restored_context_checkpoint_tokens
                  << ", expected 0 on append\n";
        return 1;
    }
    return 0;
}

int wait_committed_decode(ninfer::Engine& engine, std::uint64_t minimum, const char* label) {
    for (int spin = 0; spin < 50'000'000; ++spin) {
        if (engine.runtime_stats().committed_decode_tokens >= minimum) { return 0; }
    }
    std::cerr << label << " decode did not start (committed="
              << engine.runtime_stats().committed_decode_tokens << ")\n";
    return 1;
}

int verify_loaded(const ninfer::Engine& engine) {
    const ninfer::LoadSummary load = engine.load_summary();
    const bool ok_target =
        (load.target == "qwen3_8_27b" || load.target == "qwen3_6_27b") &&
        (load.weights_id == "nvfp4" || load.weights_id == "groupwise-int");
    if (!ok_target || load.host_to_device_bytes == 0) {
        std::cerr << "Engine construction has an invalid load summary: target=" << load.target
                  << " weights=" << load.weights_id << '\n';
        return 1;
    }
    return 0;
}

int exercise_decode_past_mark(ninfer::Engine& engine) {
    const auto start = padded(kPadDecode, kDecodeStart);
    const ninfer::GenerationResult decoded =
        engine.generate(engine.prepare_tokens(start), greedy(kDecodeOutputs, true));
    if (decoded.generated_token_ids.size() != kDecodeOutputs) {
        return fail("decode-only run did not generate the requested tokens");
    }
    const std::uint32_t need = kMark - kDecodeStart;
    if (decoded.generated_token_ids.size() < need) {
        return fail("decode-only run did not pass 24576");
    }
    std::vector<ninfer::TokenId> prefix = start;
    prefix.insert(prefix.end(), decoded.generated_token_ids.begin(),
                  decoded.generated_token_ids.begin() + static_cast<std::ptrdiff_t>(need));
    if (prefix.size() != kMark) { return fail("decode-only 24576 prefix length is wrong"); }

    const ninfer::GenerationResult evict =
        engine.generate(engine.prepare_tokens(padded(kPadEvict, 8)), greedy(1, false));
    if (evict.generated_token_ids.size() != 1) {
        return fail("decode-only eviction did not complete");
    }

    std::vector<ninfer::TokenId> probe = prefix;
    probe.insert(probe.end(), 8, kDiverge);
    const ninfer::GenerationResult hit =
        engine.generate(engine.prepare_tokens(probe), greedy(4, true));
    if (hit.prefix_reuse_path == ninfer::PrefixReusePath::RestoreContextCheckpoint) {
        std::cerr << "decode-only past 24576 froze a ladder head: reused="
                  << hit.reused_prompt_tokens << '\n';
        return 1;
    }
    if (hit.prefix_reuse_path != ninfer::PrefixReusePath::FullReset ||
        hit.reused_prompt_tokens != 0) {
        std::cerr << "decode-only past 24576 still reused prefix state: path="
                  << path_name(hit.prefix_reuse_path)
                  << " reused=" << hit.reused_prompt_tokens << '\n';
        return 1;
    }
    return 0;
}

int exercise_single_lane(const char* artifact,
                         ninfer::SpeculativeBackend spec = ninfer::SpeculativeBackend::Mtp) {
    ninfer::Engine engine(engine_options(artifact, 1, 32768, false, 32768, spec));
    if (const int rc = verify_loaded(engine); rc != 0) { return rc; }

    const auto prompt_eq = padded(kPadEq, kMark);
    const ninfer::GenerationResult eq_capture =
        engine.generate(engine.prepare_tokens(prompt_eq), greedy(1, true));
    if (eq_capture.generated_token_ids.size() != 1) {
        return fail("E==F capture did not generate one token");
    }
    if (const int rc = expect_captured(eq_capture, kMark, "E==F capture"); rc != 0) { return rc; }
    if (eq_capture.kv_ram_save_seconds != 0.0 || eq_capture.kv_ram_load_seconds != 0.0) {
        return fail("live-lane freeze billed D2H as RAM save/load");
    }
    std::vector<ninfer::TokenId> eq_diverge = prompt_eq;
    eq_diverge.insert(eq_diverge.end(), 8, kDiverge);
    const ninfer::GenerationResult eq_vram =
        engine.generate(engine.prepare_tokens(eq_diverge), greedy(4, true));
    if (const int rc = expect_append(eq_vram, ninfer::PrefixReuseSource::VramResident, kMark,
                                     "E==F VRAM suffix");
        rc != 0) {
        return rc;
    }
    if (eq_vram.kv_ram_load_seconds != 0.0) {
        return fail("VRAM-resident append billed a RAM load");
    }

    const auto prompt_eq2 = padded(kPadEq2, kMark);
    const ninfer::GenerationResult eq2_capture =
        engine.generate(engine.prepare_tokens(prompt_eq2), greedy(1, true));
    if (eq2_capture.generated_token_ids.size() != 1) {
        return fail("E==F RAM-source capture did not generate one token");
    }
    const auto prompt_b_eq = padded(kPadB, kMark);
    const ninfer::GenerationResult eq2_evict =
        engine.generate(engine.prepare_tokens(prompt_b_eq), greedy(1, true));
    if (eq2_evict.generated_token_ids.size() != 1) {
        return fail("E==F RAM eviction did not complete");
    }
    std::vector<ninfer::TokenId> eq2_diverge = prompt_eq2;
    eq2_diverge.insert(eq2_diverge.end(), 8, kDiverge);
    const ninfer::GenerationResult eq2_ram =
        engine.generate(engine.prepare_tokens(eq2_diverge), greedy(4, true));
    if (const int rc = expect_append(eq2_ram, ninfer::PrefixReuseSource::HostRam, kMark,
                                     "E==F RAM suffix");
        rc != 0) {
        return rc;
    }
    if (eq2_ram.kv_ram_load_seconds <= 0.0) {
        return fail("RAM append reported no H2D load_ms");
    }

    const auto prompt_a = padded(kPadA, kMark);
    const ninfer::GenerationResult capture =
        engine.generate(engine.prepare_tokens(prompt_a), greedy(kPastFreezeOutputs, true));
    if (capture.generated_token_ids.size() != kPastFreezeOutputs) {
        return fail("mark-length capture did not generate the requested tokens");
    }
    if (const int rc = expect_captured(capture, kMark, "mark-length capture"); rc != 0) {
        return rc;
    }

    std::vector<ninfer::TokenId> diverge = prompt_a;
    diverge.insert(diverge.end(), 8, kDiverge);
    const ninfer::GenerationResult vram =
        engine.generate(engine.prepare_tokens(diverge), greedy(4, true));
    if (const int rc = expect_restore(vram, ninfer::PrefixReuseSource::VramResident, kMark,
                                      "VRAM suffix after freeze");
        rc != 0) {
        return rc;
    }
    if (vram.kv_ram_load_seconds != 0.0) {
        return fail("E>F VRAM restore billed a RAM load");
    }

    const auto prompt_b = padded(kPadB, kMark);
    const ninfer::GenerationResult evict =
        engine.generate(engine.prepare_tokens(prompt_b), greedy(1, true));
    if (evict.generated_token_ids.size() != 1) {
        return fail("evicting second mark-length prompt did not complete");
    }

    const ninfer::GenerationResult ram =
        engine.generate(engine.prepare_tokens(diverge), greedy(4, true));
    if (const int rc = expect_restore(ram, ninfer::PrefixReuseSource::HostRam, kMark,
                                      "RAM restore of evicted ladder head");
        rc != 0) {
        return rc;
    }
    if (ram.kv_ram_load_seconds <= 0.0) {
        return fail("E>F RAM restore reported no H2D load_ms");
    }

    const ninfer::GenerationResult keep_f =
        engine.generate(engine.prepare_tokens(prompt_a), greedy(1, true));
    if (const int rc = expect_restore(keep_f, ninfer::PrefixReuseSource::VramResident, kMark,
                                      "exact replay keeps head at F");
        rc != 0) {
        return rc;
    }

    auto flipped = prompt_a;
    flipped.back() = kFlip;
    const ninfer::GenerationResult miss =
        engine.generate(engine.prepare_tokens(flipped), greedy(1, true));
    if (miss.prefix_reuse_path != ninfer::PrefixReusePath::FullReset ||
        miss.reused_prompt_tokens != 0) {
        std::cerr << "last-token hash miss reused F: path="
                  << path_name(miss.prefix_reuse_path)
                  << " reused=" << miss.reused_prompt_tokens << '\n';
        return 1;
    }

    const auto prompt_c = padded(kPadC, kMark);
    const ninfer::GenerationResult no_capture =
        engine.generate(engine.prepare_tokens(prompt_c), greedy(1, false));
    if (no_capture.generated_token_ids.size() != 1) {
        return fail("reuse-off mark-length prompt did not complete");
    }
    if (no_capture.captured_context_checkpoint_tokens != 0) {
        return fail("reuse-off prefill froze a ladder head");
    }
    const ninfer::GenerationResult no_capture_evict =
        engine.generate(engine.prepare_tokens(padded(kPadEvict, 8)), greedy(1, false));
    if (no_capture_evict.generated_token_ids.size() != 1) {
        return fail("reuse-off eviction did not complete");
    }
    std::vector<ninfer::TokenId> no_capture_suffix = prompt_c;
    no_capture_suffix.insert(no_capture_suffix.end(), 8, kDiverge);
    const ninfer::GenerationResult no_head =
        engine.generate(engine.prepare_tokens(no_capture_suffix), greedy(4, true));
    if (no_head.prefix_reuse_path != ninfer::PrefixReusePath::AppendAtFrontier ||
        no_head.prefix_reuse_source != ninfer::PrefixReuseSource::HostRam ||
        no_head.reused_prompt_tokens != kMark) {
        std::cerr << "reuse-off then reuse-on did not RAM-append execution: path="
                  << path_name(no_head.prefix_reuse_path)
                  << " source=" << static_cast<int>(no_head.prefix_reuse_source)
                  << " reused=" << no_head.reused_prompt_tokens << '\n';
        return 1;
    }
    return exercise_decode_past_mark(engine);
}

int exercise_catch_up(const char* artifact, ninfer::SpeculativeBackend spec) {
    ninfer::Engine engine(engine_options(artifact, 1, 32768, false, 32768, spec));
    if (const int rc = verify_loaded(engine); rc != 0) { return rc; }
    const auto start = padded(kPadCatch, kCatchStart);
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare_tokens(start), greedy(1, true));
    if (first.generated_token_ids.size() != 1) {
        return fail("catch-up source did not generate one token");
    }
    std::vector<ninfer::TokenId> continued = start;
    continued.push_back(first.generated_token_ids.front());
    continued.resize(kCatchFreeze, kPadCatch);
    const ninfer::GenerationResult freeze =
        engine.generate(engine.prepare_tokens(continued), greedy(kPastFreezeOutputs, true));
    if (freeze.generated_token_ids.size() != kPastFreezeOutputs) {
        return fail("catch-up continuation did not complete");
    }
    if (freeze.captured_context_checkpoint_tokens != kCatchFreezeF) {
        std::cerr << "catch-up freeze captured " << freeze.captured_context_checkpoint_tokens
                  << ", expected advertised F " << kCatchFreezeF << '\n';
        return 1;
    }
    std::vector<ninfer::TokenId> diverge = continued;
    diverge.insert(diverge.end(), 8, kDiverge);
    const ninfer::GenerationResult restored =
        engine.generate(engine.prepare_tokens(diverge), greedy(4, true));
    if (const int rc = expect_restore(restored, ninfer::PrefixReuseSource::VramResident,
                                      kCatchFreezeF, "catch-up freeze at later chunk end");
        rc != 0) {
        return rc;
    }
    if (restored.reused_prompt_tokens == kMark) {
        return fail("catch-up restore used named mark 24576 instead of the later chunk end");
    }

    const std::uint32_t e1 = freeze.reused_prompt_tokens;
    if (e1 == 0 || e1 >= continued.size()) {
        std::cerr << "catch-up continuation reused " << e1 << ", expected occupy pin before F\n";
        return 1;
    }
    std::vector<ninfer::TokenId> rollback_edit(continued.begin(),
                                               continued.begin() + static_cast<std::ptrdiff_t>(e1));
    rollback_edit.insert(rollback_edit.end(), 8, kDiverge);
    const ninfer::GenerationResult rollback =
        engine.generate(engine.prepare_tokens(rollback_edit), greedy(kPastFreezeOutputs, true));
    if (const int rc = expect_rollback(rollback, ninfer::PrefixReuseSource::VramResident, e1,
                                       "catch-up rollback D2D after ladder freeze borrowed 2C");
        rc != 0) {
        return rc;
    }
    const ninfer::GenerationResult dropped_ladder =
        engine.generate(engine.prepare_tokens(diverge), greedy(4, true));
    if (dropped_ladder.prefix_reuse_path == ninfer::PrefixReusePath::RestoreContextCheckpoint &&
        dropped_ladder.reused_prompt_tokens == kCatchFreezeF) {
        return fail("catch-up rollback left the later ladder head hittable");
    }
    if (const int rc = expect_rollback(dropped_ladder, ninfer::PrefixReuseSource::VramResident, e1,
                                       "catch-up long prefix after rollback uses E not 28480");
        rc != 0) {
        return rc;
    }

    const ninfer::GenerationResult evict =
        engine.generate(engine.prepare_tokens(padded(kPadEvict, 8)), greedy(1, false));
    if (evict.generated_token_ids.size() != 1) {
        return fail("catch-up RAM eviction did not complete");
    }
    const ninfer::GenerationResult ram =
        engine.generate(engine.prepare_tokens(rollback_edit), greedy(kPastFreezeOutputs, true));
    if (const int rc = expect_rollback(ram, ninfer::PrefixReuseSource::HostRam, e1,
                                       "catch-up RAM unpack of rollback after freeze borrow");
        rc != 0) {
        return rc;
    }
    return 0;
}

int exercise_cancel(const char* artifact, ninfer::SpeculativeBackend spec) {
    ninfer::Engine engine(engine_options(artifact, 1, 32768, false, 32768, spec));
    if (const int rc = verify_loaded(engine); rc != 0) { return rc; }

    auto cancel_after = [&](ninfer::TokenId pad, std::uint32_t prompt_tokens,
                            std::uint64_t extra_tokens, const char* label) {
        const auto prompt = padded(pad, prompt_tokens);
        const auto before = engine.runtime_stats().computed_prefill_tokens;
        ninfer::CancellationView cancel([&] {
            return engine.runtime_stats().computed_prefill_tokens >= before + extra_tokens;
        });
        const ninfer::GenerationResult cancelled =
            engine.generate(engine.prepare_tokens(prompt), greedy(1, true), nullptr, cancel);
        if (cancelled.finish_reason != ninfer::FinishReason::Cancelled) {
            std::cerr << label << " finish_reason is "
                      << static_cast<int>(cancelled.finish_reason) << ", expected Cancelled\n";
            return 1;
        }
        if (extra_tokens < kMark && cancelled.captured_context_checkpoint_tokens != 0) {
            std::cerr << label << " captured " << cancelled.captured_context_checkpoint_tokens
                      << " before the freeze chunk\n";
            return 1;
        }
        if (cancelled.restored_context_checkpoint_tokens != 0) {
            std::cerr << label << " restored " << cancelled.restored_context_checkpoint_tokens
                      << " on a cancelled prefill\n";
            return 1;
        }
        std::vector<ninfer::TokenId> diverge = padded(pad, kMark);
        diverge.insert(diverge.end(), 8, kDiverge);
        const ninfer::GenerationResult after =
            engine.generate(engine.prepare_tokens(diverge), greedy(4, true));
        if (after.prefix_reuse_path == ninfer::PrefixReusePath::RestoreContextCheckpoint) {
            std::cerr << label
                      << " left a hittable ladder head after abort: path="
                      << path_name(after.prefix_reuse_path)
                      << " reused=" << after.reused_prompt_tokens << '\n';
            return 1;
        }
        const ninfer::GenerationResult evict =
            engine.generate(engine.prepare_tokens(padded(kPadEvict, 8)), greedy(1, false));
        if (evict.generated_token_ids.size() != 1) {
            return fail("cancel-probe eviction did not complete");
        }
        return 0;
    };
    if (const int rc =
            cancel_after(kPadCancel, kMark, 4096, "cancel after first chunk / before freeze D2D");
        rc != 0) {
        return rc;
    }
    if (const int rc = cancel_after(kPadCancelB, 28672, kMark,
                                    "cancel after freeze chunk D2D/D2H queued");
        rc != 0) {
        return rc;
    }

    const auto pin_src = padded(kPadRbCold, 256);
    const ninfer::GenerationResult pin_r1 =
        engine.generate(engine.prepare_tokens(pin_src), greedy(kPastFreezeOutputs, true));
    if (pin_r1.generated_token_ids.size() != kPastFreezeOutputs) {
        return fail("cancel-after-pin first visit did not generate");
    }
    std::vector<ninfer::TokenId> pin_follow = pin_src;
    pin_follow.insert(pin_follow.end(), pin_r1.generated_token_ids.begin(),
                      pin_r1.generated_token_ids.end());
    pin_follow.resize(8192, 198);
    const auto before_pin = engine.runtime_stats().computed_prefill_tokens;
    ninfer::CancellationView pin_cancel([&] {
        return engine.runtime_stats().computed_prefill_tokens >= before_pin + 4096;
    });
    const ninfer::GenerationResult pin_cancelled =
        engine.generate(engine.prepare_tokens(pin_follow), greedy(1, true), nullptr, pin_cancel);
    if (pin_cancelled.finish_reason != ninfer::FinishReason::Cancelled) {
        std::cerr << "cancel-after-pin finish_reason is "
                  << static_cast<int>(pin_cancelled.finish_reason) << ", expected Cancelled\n";
        return 1;
    }
    std::vector<ninfer::TokenId> pin_probe = pin_src;
    pin_probe.insert(pin_probe.end(), pin_r1.generated_token_ids.begin(),
                     pin_r1.generated_token_ids.end());
    pin_probe.insert(pin_probe.end(), 8, kDiverge);
    const ninfer::GenerationResult pin_after =
        engine.generate(engine.prepare_tokens(pin_probe), greedy(4, true));
    if ((pin_after.prefix_reuse_path != ninfer::PrefixReusePath::AppendAtFrontier &&
         pin_after.prefix_reuse_path != ninfer::PrefixReusePath::RestoreTurnRollback) ||
        pin_after.reused_prompt_tokens == 0) {
        std::cerr << "cancel after rollback pin did not keep the previous turn: path="
                  << path_name(pin_after.prefix_reuse_path)
                  << " reused=" << pin_after.reused_prompt_tokens << '\n';
        return 1;
    }
    return 0;
}

int exercise_c2(const char* artifact, ninfer::SpeculativeBackend spec) {
    ninfer::Engine engine(engine_options(artifact, 2, 65536, false, 32768, spec));
    if (const int rc = verify_loaded(engine); rc != 0) { return rc; }
    if (engine.options().max_concurrency != 2) {
        return fail("C=2 engine did not keep max_concurrency=2");
    }

    const auto prompt_a = padded(kPadC2A, kMark);
    const auto prompt_b = padded(kPadC2B, kMark);
    auto cap_a_h = engine.submit(engine.prepare_tokens(prompt_a), greedy(kPastFreezeOutputs, true));
    auto cap_b_h = engine.submit(engine.prepare_tokens(prompt_b), greedy(kPastFreezeOutputs, true));
    const ninfer::GenerationResult cap_a = cap_a_h.wait();
    const ninfer::GenerationResult cap_b = cap_b_h.wait();
    if (cap_a.generated_token_ids.size() != kPastFreezeOutputs ||
        cap_b.generated_token_ids.size() != kPastFreezeOutputs) {
        return fail("C=2 concurrent captures did not complete");
    }
    if (const int rc = expect_captured(cap_a, kMark, "C=2 concurrent capture A"); rc != 0) {
        return rc;
    }
    if (const int rc = expect_captured(cap_b, kMark, "C=2 concurrent capture B"); rc != 0) {
        return rc;
    }

    std::vector<ninfer::TokenId> diverge_a = prompt_a;
    std::vector<ninfer::TokenId> diverge_b = prompt_b;
    diverge_a.insert(diverge_a.end(), 8, kDiverge);
    diverge_b.insert(diverge_b.end(), 8, kDiverge);

    auto both_restore_a = engine.submit(engine.prepare_tokens(diverge_a), greedy(4, true));
    auto both_restore_b = engine.submit(engine.prepare_tokens(diverge_b), greedy(4, true));
    const ninfer::GenerationResult rest_a = both_restore_a.wait();
    const ninfer::GenerationResult rest_b = both_restore_b.wait();
    if (const int rc = expect_restore(rest_a, ninfer::PrefixReuseSource::VramResident, kMark,
                                      "C=2 both-restore A");
        rc != 0) {
        return rc;
    }
    if (const int rc = expect_restore(rest_b, ninfer::PrefixReuseSource::VramResident, kMark,
                                      "C=2 both-restore B");
        rc != 0) {
        return rc;
    }

    const auto prompt_c = padded(kPadC2C, kMark);
    auto restore_a      = engine.submit(engine.prepare_tokens(diverge_a), greedy(4, true));
    auto cold_c         = engine.submit(engine.prepare_tokens(prompt_c), greedy(kPastFreezeOutputs, true));
    const ninfer::GenerationResult mixed_restore = restore_a.wait();
    const ninfer::GenerationResult mixed_cold    = cold_c.wait();
    if (const int rc = expect_restore(mixed_restore, ninfer::PrefixReuseSource::VramResident, kMark,
                                      "C=2 restore+full A");
        rc != 0) {
        return rc;
    }
    if (mixed_cold.generated_token_ids.size() != kPastFreezeOutputs ||
        mixed_cold.prefix_reuse_path != ninfer::PrefixReusePath::FullReset) {
        std::cerr << "C=2 companion full chat was not a cold prefill: path="
                  << path_name(mixed_cold.prefix_reuse_path)
                  << " outputs=" << mixed_cold.generated_token_ids.size() << '\n';
        return 1;
    }
    if (const int rc = expect_captured(mixed_cold, kMark, "C=2 freeze during concurrent restore");
        rc != 0) {
        return rc;
    }
    std::vector<ninfer::TokenId> diverge_c = prompt_c;
    diverge_c.insert(diverge_c.end(), 8, kDiverge);
    const ninfer::GenerationResult rest_c =
        engine.generate(engine.prepare_tokens(diverge_c), greedy(4, true));
    if (const int rc = expect_restore(rest_c, ninfer::PrefixReuseSource::VramResident, kMark,
                                      "C=2 freeze during restore still hittable");
        rc != 0) {
        return rc;
    }

    const auto prompt_d = padded(kPadC2D, kMark);
    const auto prompt_e = padded(kPadC2E, kMark);
    auto cold_d         = engine.submit(engine.prepare_tokens(prompt_d), greedy(1, false));
    auto cold_e         = engine.submit(engine.prepare_tokens(prompt_e), greedy(1, false));
    const ninfer::GenerationResult both_d = cold_d.wait();
    const ninfer::GenerationResult both_e = cold_e.wait();
    if (both_d.generated_token_ids.size() != 1 || both_e.generated_token_ids.size() != 1 ||
        both_d.prefix_reuse_path != ninfer::PrefixReusePath::FullReset ||
        both_e.prefix_reuse_path != ninfer::PrefixReusePath::FullReset) {
        std::cerr << "C=2 both-full chats were not concurrent FullReset prefills\n";
        return 1;
    }
    return 0;
}

std::vector<std::uint8_t> gradient_ppm() {
    std::vector<std::uint8_t> ppm;
    const std::string header = "P6\n64 64\n255\n";
    ppm.insert(ppm.end(), header.begin(), header.end());
    for (int index = 0; index < 64 * 64; ++index) {
        ppm.push_back(static_cast<std::uint8_t>(index & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 3) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 7) & 0xff));
    }
    return ppm;
}

std::string pad_units(std::size_t units) {
    std::string text;
    text.reserve(units * 2);
    for (std::size_t i = 0; i < units; ++i) { text += " x"; }
    return text;
}

ninfer::MessagePart ppm_part(const std::vector<std::uint8_t>& bytes, std::string name) {
    ninfer::MessagePart image;
    image.kind              = ninfer::MessagePartKind::Media;
    image.media.kind        = ninfer::MediaKind::Image;
    image.media.bytes       = bytes;
    image.media.media_type  = "image/x-portable-pixmap";
    image.media.source_name = std::move(name);
    return image;
}

ninfer::PromptInput vision_user(const std::string& before, bool with_image,
                                const std::vector<std::uint8_t>& ppm, const std::string& after) {
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    if (!before.empty()) {
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = before, .media = {}});
    }
    if (with_image) { message.parts.push_back(ppm_part(ppm, "inline.ppm")); }
    if (!after.empty()) {
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = after, .media = {}});
    }
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;
    return input;
}

ninfer::PromptInput text_user(const std::string& text) {
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(
        ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = text, .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;
    return input;
}

int exercise_vision(const char* artifact) {
    ninfer::Engine engine(engine_options(artifact, 1, 32768, true));
    if (const int rc = verify_loaded(engine); rc != 0) { return rc; }
    const auto image_bytes = gradient_ppm();

    std::size_t hi = 1024;
    auto count_image_pad = [&](std::size_t units) {
        return engine.count_tokens(vision_user({}, true, image_bytes, pad_units(units)));
    };
    while (count_image_pad(hi) < kMark) {
        hi *= 2;
        if (hi > 65536) { return fail("vision complete-item pad search exceeded 65536 units"); }
    }
    std::size_t lo = 0;
    while (lo + 1 < hi) {
        const std::size_t mid = lo + (hi - lo) / 2;
        if (count_image_pad(mid) < kMark) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    const std::size_t mark_units    = hi;
    const std::size_t capture_units = mark_units + 2048;
    const std::string capture_text = pad_units(capture_units);
    ninfer::PromptInput complete_input =
        vision_user({}, true, image_bytes, capture_text);
    const ninfer::GenerationResult complete_capture =
        engine.generate(engine.prepare(complete_input), greedy(kPastFreezeOutputs, true));
    if (complete_capture.generated_token_ids.size() != kPastFreezeOutputs ||
        !complete_capture.prompt.has_media) {
        return fail("vision complete-item capture did not complete");
    }
    if (const int rc = expect_captured(complete_capture, kMark, "vision complete-item capture");
        rc != 0) {
        return rc;
    }
    std::string probe_text = capture_text;
    probe_text.back()      = 'z';
    ninfer::PromptInput complete_probe =
        vision_user({}, true, image_bytes, probe_text);
    const ninfer::GenerationResult complete_hit =
        engine.generate(engine.prepare(std::move(complete_probe)), greedy(4, true));
    if (const int rc = expect_restore(complete_hit, ninfer::PrefixReuseSource::VramResident, kMark,
                                      "vision item complete before 24576");
        rc != 0) {
        return rc;
    }
    const ninfer::GenerationResult vision_evict =
        engine.generate(engine.prepare_tokens(padded(kPadEvict, 8)), greedy(1, false));
    if (vision_evict.generated_token_ids.size() != 1) {
        return fail("vision RAM eviction did not complete");
    }
    ninfer::PromptInput complete_ram_probe =
        vision_user({}, true, image_bytes, capture_text);
    const ninfer::GenerationResult complete_ram =
        engine.generate(engine.prepare(std::move(complete_ram_probe)), greedy(4, true));
    if (const int rc = expect_restore(complete_ram, ninfer::PrefixReuseSource::HostRam, kMark,
                                      "vision ladder RAM restore");
        rc != 0) {
        return rc;
    }

    auto count_pad_suffix = [&](std::size_t units, bool with_image, const char* suffix) {
        return engine.count_tokens(vision_user(pad_units(units), with_image, image_bytes, suffix));
    };
    std::size_t split_hi = 1024;
    while (count_pad_suffix(split_hi, false, "What is visible?") < kMark) {
        split_hi *= 2;
        if (split_hi > 65536) { return fail("vision split pad search exceeded 65536 units"); }
    }
    std::size_t split_lo = 0;
    while (split_lo + 1 < split_hi) {
        const std::size_t mid = split_lo + (split_hi - split_lo) / 2;
        if (count_pad_suffix(mid, false, "What is visible?") < kMark) {
            split_lo = mid;
        } else {
            split_hi = mid;
        }
    }
    const std::uint32_t without_image = count_pad_suffix(split_lo, false, "What is visible?");
    const std::uint32_t with_image    = count_pad_suffix(split_lo, true, "What is visible?");
    if (!(without_image < kMark && with_image > kMark)) {
        std::cerr << "vision item did not straddle 24576: without=" << without_image
                  << " with=" << with_image << '\n';
        return 1;
    }
    ninfer::PromptInput split_input =
        vision_user(pad_units(split_lo), true, image_bytes, "What is visible?");
    const ninfer::GenerationResult split_capture =
        engine.generate(engine.prepare(split_input), greedy(kPastFreezeOutputs, true));
    if (split_capture.generated_token_ids.size() != kPastFreezeOutputs ||
        !split_capture.prompt.has_media) {
        return fail("vision split-item capture did not complete");
    }
    if (split_capture.captured_context_checkpoint_tokens == 0 ||
        split_capture.captured_context_checkpoint_tokens == kMark) {
        std::cerr << "vision split-item capture froze "
                  << split_capture.captured_context_checkpoint_tokens
                  << ", expected a later chunk end\n";
        return 1;
    }
    const std::uint32_t split_f = split_capture.captured_context_checkpoint_tokens;

    ninfer::PromptInput split_replay =
        vision_user(pad_units(split_lo), true, image_bytes, "What is visible?");
    const ninfer::GenerationResult split_complete_hit =
        engine.generate(engine.prepare(std::move(split_replay)), greedy(4, true));
    if (split_complete_hit.reused_prompt_tokens < split_f) {
        std::cerr << "vision split exact replay reused " << split_complete_hit.reused_prompt_tokens
                  << ", expected at least freeze " << split_f << " path="
                  << path_name(split_complete_hit.prefix_reuse_path) << '\n';
        return 1;
    }
    if (split_complete_hit.prefix_reuse_path ==
        ninfer::PrefixReusePath::RestoreContextCheckpoint) {
        if (split_complete_hit.reused_prompt_tokens != split_f ||
            split_complete_hit.restored_context_checkpoint_tokens != split_f) {
            std::cerr << "vision split ladder exact replay reused "
                      << split_complete_hit.reused_prompt_tokens << " restored "
                      << split_complete_hit.restored_context_checkpoint_tokens
                      << ", expected " << split_f << '\n';
            return 1;
        }
    } else if (split_complete_hit.prefix_reuse_path ==
                   ninfer::PrefixReusePath::RestoreTurnCheckpoint ||
               split_complete_hit.prefix_reuse_path ==
                   ninfer::PrefixReusePath::RestoreResponseCheckpoint) {
        // Same-F tie keeps rewrite; a later TurnClosure also beats the catch-up head.
        if (split_complete_hit.restored_context_checkpoint_tokens != 0) {
            std::cerr << "vision split rewrite exact replay restored "
                      << split_complete_hit.restored_context_checkpoint_tokens
                      << ", expected 0\n";
            return 1;
        }
    } else {
        std::cerr << "vision split exact replay path is "
                  << path_name(split_complete_hit.prefix_reuse_path)
                  << ", expected rewrite or restore_context_checkpoint at F>=" << split_f << '\n';
        return 1;
    }

    ninfer::PromptInput split_probe =
        vision_user(pad_units(split_lo), true, image_bytes, "Different suffix.");
    const ninfer::GenerationResult split_hit =
        engine.generate(engine.prepare(std::move(split_probe)), greedy(4, true));
    if (split_hit.reused_prompt_tokens == kMark &&
        split_hit.prefix_reuse_path == ninfer::PrefixReusePath::RestoreContextCheckpoint) {
        return fail("vision item that splits 24576 still froze a ladder head at the mark");
    }
    if (split_hit.prefix_reuse_path == ninfer::PrefixReusePath::RestoreContextCheckpoint &&
        split_hit.reused_prompt_tokens != split_f) {
        std::cerr << "vision split different-suffix restore reused "
                  << split_hit.reused_prompt_tokens << ", expected freeze " << split_f << '\n';
        return 1;
    }
    return 0;
}

int exercise_c3(const char* artifact, ninfer::SpeculativeBackend spec) {
    ninfer::Engine engine(engine_options(artifact, 3, 98304, false, 32768, spec));
    if (const int rc = verify_loaded(engine); rc != 0) { return rc; }

    const auto prompt_a = padded(kPadC3A, kMark);
    const auto prompt_b = padded(kPadC3B, kMark);
    const auto prompt_c = padded(kPadC3C, kMark);
    auto cap_a_h = engine.submit(engine.prepare_tokens(prompt_a), greedy(kPastFreezeOutputs, true));
    auto cap_b_h = engine.submit(engine.prepare_tokens(prompt_b), greedy(kPastFreezeOutputs, true));
    auto cap_c_h = engine.submit(engine.prepare_tokens(prompt_c), greedy(kPastFreezeOutputs, true));
    const ninfer::GenerationResult cap_a = cap_a_h.wait();
    const ninfer::GenerationResult cap_b = cap_b_h.wait();
    const ninfer::GenerationResult cap_c = cap_c_h.wait();
    if (cap_a.generated_token_ids.size() != kPastFreezeOutputs ||
        cap_b.generated_token_ids.size() != kPastFreezeOutputs ||
        cap_c.generated_token_ids.size() != kPastFreezeOutputs) {
        return fail("C=3 concurrent capture did not generate two tokens per lane");
    }
    if (const int rc = expect_captured(cap_a, kMark, "C=3 capture A"); rc != 0) { return rc; }
    if (const int rc = expect_captured(cap_b, kMark, "C=3 capture B"); rc != 0) { return rc; }
    if (const int rc = expect_captured(cap_c, kMark, "C=3 capture C"); rc != 0) { return rc; }

    std::vector<ninfer::TokenId> diverge_a = prompt_a;
    std::vector<ninfer::TokenId> diverge_b = prompt_b;
    std::vector<ninfer::TokenId> diverge_c = prompt_c;
    diverge_a.insert(diverge_a.end(), 8, kDiverge);
    diverge_b.insert(diverge_b.end(), 8, kDiverge);
    diverge_c.insert(diverge_c.end(), 8, kDiverge);

    auto rest_a_h = engine.submit(engine.prepare_tokens(diverge_a), greedy(4, true));
    auto rest_b_h = engine.submit(engine.prepare_tokens(diverge_b), greedy(4, true));
    auto rest_c_h = engine.submit(engine.prepare_tokens(diverge_c), greedy(4, true));
    const ninfer::GenerationResult rest_a = rest_a_h.wait();
    const ninfer::GenerationResult rest_b = rest_b_h.wait();
    const ninfer::GenerationResult rest_c = rest_c_h.wait();
    if (const int rc = expect_restore(rest_a, ninfer::PrefixReuseSource::VramResident, kMark,
                                      "C=3 both-restore A");
        rc != 0) {
        return rc;
    }
    if (const int rc = expect_restore(rest_b, ninfer::PrefixReuseSource::VramResident, kMark,
                                      "C=3 both-restore B");
        rc != 0) {
        return rc;
    }
    if (const int rc = expect_restore(rest_c, ninfer::PrefixReuseSource::VramResident, kMark,
                                      "C=3 both-restore C");
        rc != 0) {
        return rc;
    }

    const auto prompt_d = padded(kPadC3D, kMark);
    const auto prompt_e = padded(kPadC3E, kMark);
    auto restore_a      = engine.submit(engine.prepare_tokens(diverge_a), greedy(4, true));
    auto cold_d =
        engine.submit(engine.prepare_tokens(prompt_d), greedy(kPastFreezeOutputs, true));
    auto cold_e         = engine.submit(engine.prepare_tokens(prompt_e), greedy(1, false));
    const ninfer::GenerationResult mixed_restore = restore_a.wait();
    const ninfer::GenerationResult mixed_cold_d  = cold_d.wait();
    const ninfer::GenerationResult mixed_cold_e  = cold_e.wait();
    if (const int rc = expect_restore(mixed_restore, ninfer::PrefixReuseSource::VramResident, kMark,
                                      "C=3 restore+two-full A");
        rc != 0) {
        return rc;
    }
    if (mixed_cold_d.generated_token_ids.size() != kPastFreezeOutputs ||
        mixed_cold_d.prefix_reuse_path != ninfer::PrefixReusePath::FullReset ||
        mixed_cold_e.generated_token_ids.size() != 1 ||
        mixed_cold_e.prefix_reuse_path != ninfer::PrefixReusePath::FullReset) {
        std::cerr << "C=3 companion full chats were not cold prefills: d_path="
                  << path_name(mixed_cold_d.prefix_reuse_path) << " e_path="
                  << path_name(mixed_cold_e.prefix_reuse_path) << '\n';
        return 1;
    }
    if (const int rc = expect_captured(mixed_cold_d, kMark, "C=3 freeze during concurrent restore");
        rc != 0) {
        return rc;
    }
    if (mixed_cold_e.captured_context_checkpoint_tokens != 0) {
        return fail("C=3 reuse-off companion froze a ladder head");
    }
    std::vector<ninfer::TokenId> diverge_d = prompt_d;
    diverge_d.insert(diverge_d.end(), 8, kDiverge);
    const ninfer::GenerationResult rest_d =
        engine.generate(engine.prepare_tokens(diverge_d), greedy(4, true));
    if (const int rc = expect_restore(rest_d, ninfer::PrefixReuseSource::VramResident, kMark,
                                      "C=3 freeze during restore still hittable");
        rc != 0) {
        return rc;
    }
    return 0;
}

int exercise_restore_matches_cold(const char* artifact, ninfer::SpeculativeBackend spec) {
    ninfer::Engine engine(engine_options(artifact, 1, 32768, false, 32768, spec));
    if (const int rc = verify_loaded(engine); rc != 0) { return rc; }
    const auto prompt = padded(kPadOracle, kMark);
    const ninfer::GenerationResult capture =
        engine.generate(engine.prepare_tokens(prompt), greedy(kPastFreezeOutputs, true));
    if (capture.generated_token_ids.size() != kPastFreezeOutputs) {
        return fail("restore-vs-cold capture did not generate");
    }
    if (const int rc = expect_captured(capture, kMark, "restore-vs-cold capture"); rc != 0) {
        return rc;
    }
    std::vector<ninfer::TokenId> diverge = prompt;
    diverge.insert(diverge.end(), 8, kDiverge);
    const ninfer::GenerationResult vram =
        engine.generate(engine.prepare_tokens(diverge), greedy(4, true));
    if (const int rc = expect_restore(vram, ninfer::PrefixReuseSource::VramResident, kMark,
                                      "restore-vs-cold VRAM");
        rc != 0) {
        return rc;
    }
    const ninfer::GenerationResult vram_oracle =
        engine.generate(engine.prepare_tokens(diverge), greedy(4, false));
    if (const int rc = expect_same_tokens(vram.generated_token_ids, vram_oracle.generated_token_ids,
                                          "VRAM ladder restore vs cold prefill");
        rc != 0) {
        return rc;
    }

    const ninfer::GenerationResult recapture =
        engine.generate(engine.prepare_tokens(prompt), greedy(kPastFreezeOutputs, true));
    if (recapture.generated_token_ids.size() != kPastFreezeOutputs) {
        return fail("restore-vs-cold recapture did not generate");
    }
    const ninfer::GenerationResult evict =
        engine.generate(engine.prepare_tokens(padded(kPadEvict, 8)), greedy(1, false));
    if (evict.generated_token_ids.size() != 1) {
        return fail("restore-vs-cold eviction did not complete");
    }
    const ninfer::GenerationResult ram =
        engine.generate(engine.prepare_tokens(diverge), greedy(4, true));
    if (const int rc = expect_restore(ram, ninfer::PrefixReuseSource::HostRam, kMark,
                                      "restore-vs-cold RAM");
        rc != 0) {
        return rc;
    }
    const ninfer::GenerationResult ram_oracle =
        engine.generate(engine.prepare_tokens(diverge), greedy(4, false));
    if (const int rc = expect_same_tokens(ram.generated_token_ids, ram_oracle.generated_token_ids,
                                          "RAM ladder restore vs cold prefill");
        rc != 0) {
        return rc;
    }
    return 0;
}

int exercise_ram_shorter_ladder(const char* artifact, ninfer::SpeculativeBackend spec) {
    ninfer::Engine engine(engine_options(artifact, 1, 40960, false, 40960, spec));
    if (const int rc = verify_loaded(engine); rc != 0) { return rc; }
    const auto prompt = padded(kPadRamShort, kSecondMark);
    const ninfer::GenerationResult capture =
        engine.generate(engine.prepare_tokens(prompt), greedy(kPastFreezeOutputs, true));
    if (capture.generated_token_ids.size() != kPastFreezeOutputs) {
        return fail("RAM shorter-ladder capture did not generate");
    }
    if (const int rc = expect_captured(capture, kSecondMark, "RAM shorter-ladder last freeze");
        rc != 0) {
        return rc;
    }
    const ninfer::GenerationResult evict =
        engine.generate(engine.prepare_tokens(padded(kPadEvict, 8)), greedy(1, false));
    if (evict.generated_token_ids.size() != 1) {
        return fail("RAM shorter-ladder eviction did not complete");
    }
    std::vector<ninfer::TokenId> first = padded(kPadRamShort, kMark);
    first.insert(first.end(), 8, kDiverge);
    const ninfer::GenerationResult ram =
        engine.generate(engine.prepare_tokens(first), greedy(4, true));
    if (const int rc = expect_restore(ram, ninfer::PrefixReuseSource::HostRam, kMark,
                                      "RAM restore of first head from a longer eviction");
        rc != 0) {
        return rc;
    }
    std::vector<ninfer::TokenId> second = prompt;
    second.insert(second.end(), 8, kDiverge);
    const ninfer::GenerationResult dropped =
        engine.generate(engine.prepare_tokens(second), greedy(4, true));
    if (const int rc = expect_restore(dropped, ninfer::PrefixReuseSource::VramResident, kMark,
                                      "RAM shorter occupy dropped the later head");
        rc != 0) {
        return rc;
    }
    if (dropped.reused_prompt_tokens == kSecondMark) {
        return fail("RAM shorter occupy still restored the evicted 36864 head");
    }
    if (const int rc = expect_captured(dropped, kSecondMark,
                                       "RAM shorter occupy recaptured 36864 after drop");
        rc != 0) {
        return rc;
    }
    const ninfer::GenerationResult cold =
        engine.generate(engine.prepare_tokens(first), greedy(4, false));
    if (const int rc = expect_same_tokens(ram.generated_token_ids, cold.generated_token_ids,
                                          "RAM shorter-ladder occupy vs cold prefill");
        rc != 0) {
        return rc;
    }
    return 0;
}

int exercise_c3_cancel_during_decode(const char* artifact, ninfer::SpeculativeBackend spec) {
    ninfer::Engine engine(engine_options(artifact, 3, 98304, false, 32768, spec));
    if (const int rc = verify_loaded(engine); rc != 0) { return rc; }
    const auto prompt_a = padded(kPadC3A, kMark);
    const auto prompt_b = padded(kPadC3B, kMark);
    const ninfer::GenerationResult cap_a =
        engine.generate(engine.prepare_tokens(prompt_a), greedy(kPastFreezeOutputs, true));
    const ninfer::GenerationResult cap_b =
        engine.generate(engine.prepare_tokens(prompt_b), greedy(kPastFreezeOutputs, true));
    if (cap_a.generated_token_ids.size() != kPastFreezeOutputs ||
        cap_b.generated_token_ids.size() != kPastFreezeOutputs) {
        return fail("C=3 cancel-during-decode capture did not generate");
    }
    if (const int rc = expect_captured(cap_a, kMark, "C=3 cancel-during-decode capture A");
        rc != 0) {
        return rc;
    }
    if (const int rc = expect_captured(cap_b, kMark, "C=3 cancel-during-decode capture B");
        rc != 0) {
        return rc;
    }

    std::vector<ninfer::TokenId> restore_a = prompt_a;
    std::vector<ninfer::TokenId> restore_b = prompt_b;
    restore_a.insert(restore_a.end(), 8, kDiverge);
    restore_b.insert(restore_b.end(), 8, kDiverge);
    constexpr std::uint32_t kDecodeHold = 64;
    auto hold_a = engine.submit(engine.prepare_tokens(restore_a), greedy(kDecodeHold, true));
    auto hold_b = engine.submit(engine.prepare_tokens(restore_b), greedy(kDecodeHold, true));
    if (const int rc =
            wait_committed_decode(engine, 2, "C=3 cancel-during-decode A/B never entered decode");
        rc != 0) {
        return rc;
    }

    const auto before_prefill = engine.runtime_stats().computed_prefill_tokens;
    ninfer::CancellationView cancel([&] {
        return engine.runtime_stats().computed_prefill_tokens >= before_prefill + 4096;
    });
    const ninfer::GenerationResult cancelled = engine.generate(
        engine.prepare_tokens(padded(kPadC3F, kMark)), greedy(1, true), nullptr, cancel);
    if (cancelled.finish_reason != ninfer::FinishReason::Cancelled) {
        std::cerr << "C=3 cancel-during-decode finish_reason is "
                  << static_cast<int>(cancelled.finish_reason) << ", expected Cancelled\n";
        return 1;
    }
    if (cancelled.captured_context_checkpoint_tokens != 0) {
        return fail("C=3 cancelled prefill froze a ladder head while A/B decoded");
    }

    const ninfer::GenerationResult held_a = hold_a.wait();
    const ninfer::GenerationResult held_b = hold_b.wait();
    if (held_a.generated_token_ids.size() != kDecodeHold ||
        held_b.generated_token_ids.size() != kDecodeHold) {
        std::cerr << "C=3 cancel-during-decode holders finished with "
                  << held_a.generated_token_ids.size() << "/" << held_b.generated_token_ids.size()
                  << " tokens, expected " << kDecodeHold << "\n";
        return 1;
    }
    if (const int rc = expect_restore(held_a, ninfer::PrefixReuseSource::VramResident, kMark,
                                      "C=3 cancel-during-decode hold A");
        rc != 0) {
        return rc;
    }
    if (const int rc = expect_restore(held_b, ninfer::PrefixReuseSource::VramResident, kMark,
                                      "C=3 cancel-during-decode hold B");
        rc != 0) {
        return rc;
    }

    const ninfer::GenerationResult rest_a =
        engine.generate(engine.prepare_tokens(restore_a), greedy(4, true));
    const ninfer::GenerationResult rest_b =
        engine.generate(engine.prepare_tokens(restore_b), greedy(4, true));
    if (const int rc = expect_restore(rest_a, ninfer::PrefixReuseSource::VramResident, kMark,
                                      "C=3 A still hittable after cancel during decode");
        rc != 0) {
        return rc;
    }
    if (const int rc = expect_restore(rest_b, ninfer::PrefixReuseSource::VramResident, kMark,
                                      "C=3 B still hittable after cancel during decode");
        rc != 0) {
        return rc;
    }
    std::vector<ninfer::TokenId> cancelled_probe = padded(kPadC3F, kMark);
    cancelled_probe.insert(cancelled_probe.end(), 8, kDiverge);
    const ninfer::GenerationResult leftover =
        engine.generate(engine.prepare_tokens(cancelled_probe), greedy(4, true));
    if (leftover.prefix_reuse_path == ninfer::PrefixReusePath::RestoreContextCheckpoint) {
        return fail("C=3 cancelled prefill left a hittable ladder head");
    }
    return 0;
}

int exercise_two_marks(const char* artifact, ninfer::SpeculativeBackend spec) {
    ninfer::Engine engine(engine_options(artifact, 1, 40960, false, 40960, spec));
    if (const int rc = verify_loaded(engine); rc != 0) { return rc; }
    const auto prompt = padded(kPadTwoMark, kSecondMark);
    const ninfer::GenerationResult capture =
        engine.generate(engine.prepare_tokens(prompt), greedy(kPastFreezeOutputs, true));
    if (capture.generated_token_ids.size() != kPastFreezeOutputs) {
        return fail("two-mark capture did not generate the requested tokens");
    }
    if (const int rc = expect_captured(capture, kSecondMark, "two-mark last freeze"); rc != 0) {
        return rc;
    }
    std::vector<ninfer::TokenId> second = prompt;
    second.insert(second.end(), 8, kDiverge);
    const ninfer::GenerationResult hit_second =
        engine.generate(engine.prepare_tokens(second), greedy(4, true));
    if (const int rc = expect_restore(hit_second, ninfer::PrefixReuseSource::VramResident,
                                      kSecondMark, "two-mark restore last head");
        rc != 0) {
        return rc;
    }
    std::vector<ninfer::TokenId> first = padded(kPadTwoMark, kMark);
    first.insert(first.end(), 8, kDiverge);
    const ninfer::GenerationResult hit_first =
        engine.generate(engine.prepare_tokens(first), greedy(4, true));
    if (const int rc = expect_restore(hit_first, ninfer::PrefixReuseSource::VramResident, kMark,
                                      "two-mark restore first head");
        rc != 0) {
        return rc;
    }
    const ninfer::GenerationResult hit_dropped =
        engine.generate(engine.prepare_tokens(second), greedy(4, true));
    if (const int rc = expect_restore(hit_dropped, ninfer::PrefixReuseSource::VramResident, kMark,
                                      "two-mark last head dropped after restoring first", kMark);
        rc != 0) {
        return rc;
    }
    if (const int rc = expect_captured(hit_dropped, kSecondMark,
                                       "two-mark recapture of dropped 36864 after restore 24576");
        rc != 0) {
        return rc;
    }
    return 0;
}

int exercise_rewrite_same_f(const char* artifact, ninfer::SpeculativeBackend spec) {
    ninfer::Engine engine(engine_options(artifact, 1, 32768, false, 32768, spec));
    if (const int rc = verify_loaded(engine); rc != 0) { return rc; }
    auto text_preserve = [](const std::string& text) {
        ninfer::PromptInput input = text_user(text);
        input.options.preserve_thinking = true;
        return input;
    };
    std::size_t hi = 1024;
    auto count_pad = [&](std::size_t units) {
        return engine.count_tokens(text_preserve(pad_units(units)));
    };
    while (count_pad(hi) < kMark) {
        hi *= 2;
        if (hi > 65536) { return fail("rewrite same-F pad search exceeded 65536 units"); }
    }
    std::size_t lo = 0;
    while (lo + 1 < hi) {
        const std::size_t mid = lo + (hi - lo) / 2;
        if (count_pad(mid) < kMark) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    if (count_pad(hi) != kMark) {
        std::cerr << "rewrite same-F cannot land exactly on 24576 (got " << count_pad(hi) << ")\n";
        return 1;
    }
    const ninfer::GenerationResult capture =
        engine.generate(engine.prepare(text_preserve(pad_units(hi))), greedy(kPastFreezeOutputs, true));
    if (capture.generated_token_ids.size() != kPastFreezeOutputs) {
        return fail("rewrite same-F capture did not complete");
    }
    if (const int rc = expect_captured(capture, kMark, "rewrite same-F ladder freeze"); rc != 0) {
        return rc;
    }
    const ninfer::GenerationResult replay =
        engine.generate(engine.prepare(text_preserve(pad_units(hi))), greedy(1, true));
    // preserve_thinking plants ResponseReplay at the prompt end, so rewrite F equals the
    // 24576 ladder head and wins the same-F tie.
    if (const int rc = expect_rewrite(replay, ninfer::PrefixReusePath::RestoreResponseCheckpoint,
                                      ninfer::PrefixReuseSource::VramResident, kMark,
                                      "rewrite same-F ResponseReplay beats ladder");
        rc != 0) {
        return rc;
    }
    return 0;
}

int exercise_c2_drop_during_decode(const char* artifact, ninfer::SpeculativeBackend spec) {
    ninfer::Engine engine(engine_options(artifact, 2, 81920, false, 40960, spec));
    if (const int rc = verify_loaded(engine); rc != 0) { return rc; }
    const auto prompt_a = padded(kPadDropA, kSecondMark);
    const auto prompt_b = padded(kPadDropB, kMark);
    const ninfer::GenerationResult cap_a =
        engine.generate(engine.prepare_tokens(prompt_a), greedy(kPastFreezeOutputs, true));
    const ninfer::GenerationResult cap_b =
        engine.generate(engine.prepare_tokens(prompt_b), greedy(kPastFreezeOutputs, true));
    if (cap_a.generated_token_ids.size() != kPastFreezeOutputs ||
        cap_b.generated_token_ids.size() != kPastFreezeOutputs) {
        return fail("C=2 drop-during-decode capture did not generate");
    }
    if (const int rc = expect_captured(cap_a, kSecondMark, "C=2 drop-during-decode capture A");
        rc != 0) {
        return rc;
    }
    if (const int rc = expect_captured(cap_b, kMark, "C=2 drop-during-decode capture B"); rc != 0) {
        return rc;
    }

    std::vector<ninfer::TokenId> restore_b = prompt_b;
    restore_b.insert(restore_b.end(), 8, kDiverge);
    constexpr std::uint32_t kDecodeHold = 64;
    auto hold_b = engine.submit(engine.prepare_tokens(restore_b), greedy(kDecodeHold, true));
    if (const int rc = wait_committed_decode(engine, 1, "C=2 drop-during-decode B never entered decode");
        rc != 0) {
        return rc;
    }

    std::vector<ninfer::TokenId> first = padded(kPadDropA, kMark);
    first.insert(first.end(), 8, kDiverge);
    const ninfer::GenerationResult hit_first =
        engine.generate(engine.prepare_tokens(first), greedy(4, true));
    if (const int rc = expect_restore(hit_first, ninfer::PrefixReuseSource::VramResident, kMark,
                                      "C=2 drop-during-decode restore first head while B decodes");
        rc != 0) {
        return rc;
    }

    std::vector<ninfer::TokenId> second = prompt_a;
    second.insert(second.end(), 8, kDiverge);
    const ninfer::GenerationResult hit_dropped =
        engine.generate(engine.prepare_tokens(second), greedy(4, true));
    if (const int rc = expect_restore(hit_dropped, ninfer::PrefixReuseSource::VramResident, kMark,
                                      "C=2 drop-during-decode later head gone while B decodes");
        rc != 0) {
        return rc;
    }

    const ninfer::GenerationResult held_b = hold_b.wait();
    if (held_b.generated_token_ids.size() != kDecodeHold) {
        std::cerr << "C=2 drop-during-decode holder finished with "
                  << held_b.generated_token_ids.size() << " tokens, expected " << kDecodeHold
                  << '\n';
        return 1;
    }
    if (const int rc = expect_restore(held_b, ninfer::PrefixReuseSource::VramResident, kMark,
                                      "C=2 drop-during-decode B hold");
        rc != 0) {
        return rc;
    }
    return 0;
}

int exercise_150k(const char* artifact, ninfer::SpeculativeBackend spec) {
    constexpr std::uint32_t kCtx = 155648;
    ninfer::Engine engine(engine_options(artifact, 1, kCtx, false, kCtx, spec));
    if (const int rc = verify_loaded(engine); rc != 0) { return rc; }
    const auto prompt = padded(kPad150, k150Mark);
    const ninfer::GenerationResult capture =
        engine.generate(engine.prepare_tokens(prompt), greedy(kPastFreezeOutputs, true));
    if (capture.generated_token_ids.size() != kPastFreezeOutputs) {
        return fail("150k capture did not generate the requested tokens");
    }
    if (const int rc = expect_captured(capture, k150Mark, "150k last freeze"); rc != 0) {
        return rc;
    }
    std::vector<ninfer::TokenId> long_hit = prompt;
    long_hit.insert(long_hit.end(), 8, kDiverge);
    const ninfer::GenerationResult hit_long =
        engine.generate(engine.prepare_tokens(long_hit), greedy(4, true));
    if (const int rc = expect_restore(hit_long, ninfer::PrefixReuseSource::VramResident, k150Mark,
                                      "150k restore last head");
        rc != 0) {
        return rc;
    }
    std::vector<ninfer::TokenId> hundred = padded(kPad150, kHundredMark);
    hundred.insert(hundred.end(), 8, kDiverge);
    const ninfer::GenerationResult hit_hundred =
        engine.generate(engine.prepare_tokens(hundred), greedy(4, true));
    if (const int rc = expect_restore(hit_hundred, ninfer::PrefixReuseSource::VramResident,
                                      kHundredMark, "150k restore 102400 drops 151552");
        rc != 0) {
        return rc;
    }
    const ninfer::GenerationResult hit_dropped =
        engine.generate(engine.prepare_tokens(long_hit), greedy(4, true));
    if (const int rc = expect_restore(hit_dropped, ninfer::PrefixReuseSource::VramResident,
                                      kHundredMark, "150k later head dropped after restoring 102400");
        rc != 0) {
        return rc;
    }
    if (hit_dropped.reused_prompt_tokens == k150Mark) {
        return fail("150k restore of 102400 left 151552 hittable");
    }
    return 0;
}

int exercise_mtp_off(const char* artifact) {
    ninfer::Engine engine(engine_options(artifact, 1, 32768, false, 32768,
                                         ninfer::SpeculativeBackend::None));
    if (const int rc = verify_loaded(engine); rc != 0) { return rc; }
    const auto prompt = padded(kPadMtpOff, kMark);
    const ninfer::GenerationResult capture =
        engine.generate(engine.prepare_tokens(prompt), greedy(1, true));
    if (capture.generated_token_ids.size() != 1) {
        return fail("MTP-off mark-length prompt did not complete");
    }
    if (capture.captured_context_checkpoint_tokens != 0) {
        return fail("MTP-off prefill froze a ladder head");
    }
    std::vector<ninfer::TokenId> mtp_off_follow = prompt;
    mtp_off_follow.insert(mtp_off_follow.end(), 8, kDiverge);
    const ninfer::GenerationResult mtp_off_append =
        engine.generate(engine.prepare_tokens(mtp_off_follow), greedy(1, true));
    if (mtp_off_append.captured_context_checkpoint_tokens != 0) {
        return fail("MTP-off append pinned a turn-rollback head");
    }
    const ninfer::GenerationResult evict =
        engine.generate(engine.prepare_tokens(padded(kPadEvict, 8)), greedy(1, false));
    if (evict.generated_token_ids.size() != 1) {
        return fail("MTP-off eviction did not complete");
    }
    std::vector<ninfer::TokenId> suffix = prompt;
    suffix.insert(suffix.end(), 8, kDiverge);
    const ninfer::GenerationResult hit =
        engine.generate(engine.prepare_tokens(suffix), greedy(4, true));
    if (hit.prefix_reuse_path == ninfer::PrefixReusePath::RestoreContextCheckpoint) {
        return fail("MTP-off prefill still produced a ladder hit");
    }
    return 0;
}

int exercise_chat_rewrite_tie(const char* artifact, ninfer::SpeculativeBackend spec) {
    ninfer::Engine engine(engine_options(artifact, 1, 32768, false, 32768, spec));
    if (const int rc = verify_loaded(engine); rc != 0) { return rc; }
    std::size_t hi = 1024;
    auto count_pad = [&](std::size_t units) { return engine.count_tokens(text_user(pad_units(units))); };
    while (count_pad(hi) < kMark) {
        hi *= 2;
        if (hi > 65536) { return fail("chat rewrite-tie pad search exceeded 65536 units"); }
    }
    std::size_t lo = 0;
    while (lo + 1 < hi) {
        const std::size_t mid = lo + (hi - lo) / 2;
        if (count_pad(mid) < kMark) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    if (count_pad(hi) != kMark) {
        std::cerr << "chat rewrite-tie cannot land exactly on 24576 (got " << count_pad(hi)
                  << ")\n";
        return 1;
    }
    ninfer::PromptInput input = text_user(pad_units(hi));
    const ninfer::GenerationResult capture =
        engine.generate(engine.prepare(input), greedy(kPastFreezeOutputs, true));
    if (capture.generated_token_ids.size() != kPastFreezeOutputs) {
        return fail("chat rewrite-tie capture did not complete");
    }
    if (const int rc = expect_captured(capture, kMark, "chat rewrite-tie capture"); rc != 0) {
        return rc;
    }
    const ninfer::GenerationResult replay =
        engine.generate(engine.prepare(text_user(pad_units(hi))), greedy(1, true));
    // add_generation_prompt places TurnClosure before the think prologue, so rewrite F is
    // shorter than the 24576 ladder head and exact replay is RestoreContextCheckpoint.
    if (replay.prefix_reuse_path != ninfer::PrefixReusePath::RestoreContextCheckpoint ||
        replay.reused_prompt_tokens != kMark ||
        replay.restored_context_checkpoint_tokens != kMark) {
        std::cerr << "chat exact replay after freeze: path="
                  << path_name(replay.prefix_reuse_path)
                  << " reused=" << replay.reused_prompt_tokens
                  << " restored=" << replay.restored_context_checkpoint_tokens << '\n';
        return 1;
    }
    return 0;
}

int exercise_turn_rollback(const char* artifact,
                            ninfer::SpeculativeBackend spec = ninfer::SpeculativeBackend::Mtp) {
    constexpr std::uint32_t kRb = 256;
    const auto p1               = padded(kPadRb1, kRb);
    std::vector<ninfer::TokenId> follow;
    std::vector<ninfer::TokenId> edit;
    std::uint32_t e1 = 0;

    {
        ninfer::Engine engine(engine_options(artifact, 1, 32768, false, 32768, spec));
        if (const int rc = verify_loaded(engine); rc != 0) { return rc; }

        const ninfer::GenerationResult r1 =
            engine.generate(engine.prepare_tokens(p1), greedy(kPastFreezeOutputs, true));
        if (r1.generated_token_ids.size() != kPastFreezeOutputs) {
            return fail("turn-rollback first visit did not generate");
        }
        if (r1.captured_context_checkpoint_tokens != 0) {
            return fail("first-visit FullReset pinned a turn-rollback head");
        }

        follow = p1;
        follow.insert(follow.end(), r1.generated_token_ids.begin(), r1.generated_token_ids.end());
        follow.push_back(kPadRb2);
        follow.insert(follow.end(), 32, 198);
        const ninfer::GenerationResult r2 =
            engine.generate(engine.prepare_tokens(follow), greedy(1, true));
        if (r2.prefix_reuse_path != ninfer::PrefixReusePath::AppendAtFrontier ||
            r2.reused_prompt_tokens == 0) {
            std::cerr << "turn-rollback append: path=" << path_name(r2.prefix_reuse_path)
                      << " reused=" << r2.reused_prompt_tokens << '\n';
            return 1;
        }
        e1 = r2.reused_prompt_tokens;
        if (r2.captured_context_checkpoint_tokens != e1) {
            std::cerr << "turn-rollback pin captured " << r2.captured_context_checkpoint_tokens
                      << ", expected E1=" << e1 << '\n';
            return 1;
        }

        const ninfer::GenerationResult exact =
            engine.generate(engine.prepare_tokens(follow), greedy(1, true));
        if (exact.prefix_reuse_path != ninfer::PrefixReusePath::AppendAtFrontier) {
            std::cerr << "completed-turn exact-hit path=" << path_name(exact.prefix_reuse_path)
                      << " reused=" << exact.reused_prompt_tokens << '\n';
            return 1;
        }
        if (exact.captured_context_checkpoint_tokens != 0) {
            return fail("exact-hit of completed turn2 replaced the E1 rollback pin");
        }

        edit.assign(follow.begin(), follow.begin() + static_cast<std::ptrdiff_t>(e1));
        edit.push_back(kPadRb3);
        edit.insert(edit.end(), 32, 198);
        const ninfer::GenerationResult hit =
            engine.generate(engine.prepare_tokens(edit), greedy(kPastFreezeOutputs, true));
        if (const int rc = expect_rollback(hit, ninfer::PrefixReuseSource::VramResident, e1,
                                           "edit last suffix");
            rc != 0) {
            return rc;
        }
        if (hit.restored_context_checkpoint_tokens != e1) {
            std::cerr << "turn-rollback restored " << hit.restored_context_checkpoint_tokens
                      << ", expected E1=" << e1 << '\n';
            return 1;
        }
        const ninfer::GenerationResult vram_oracle =
            engine.generate(engine.prepare_tokens(edit), greedy(kPastFreezeOutputs, false));
        if (const int rc = expect_same_tokens(hit.generated_token_ids, vram_oracle.generated_token_ids,
                                              "VRAM rollback GDN vs cold prefill");
            rc != 0) {
            return rc;
        }

        const ninfer::GenerationResult evict =
            engine.generate(engine.prepare_tokens(padded(kPadRbEvict, 8)), greedy(1, false));
        if (evict.generated_token_ids.size() != 1) {
            return fail("turn-rollback RAM eviction did not complete");
        }
        const ninfer::GenerationResult ram =
            engine.generate(engine.prepare_tokens(edit), greedy(kPastFreezeOutputs, true));
        if (const int rc = expect_rollback(ram, ninfer::PrefixReuseSource::HostRam, e1,
                                           "RAM turn-rollback unpack");
            rc != 0) {
            return rc;
        }
        const ninfer::GenerationResult ram_oracle =
            engine.generate(engine.prepare_tokens(edit), greedy(kPastFreezeOutputs, false));
        if (const int rc = expect_same_tokens(ram.generated_token_ids, ram_oracle.generated_token_ids,
                                              "RAM rollback GDN vs cold prefill");
            rc != 0) {
            return rc;
        }
    }

    {
        ninfer::Engine denied_engine(engine_options(artifact, 1, 32768, false, 32768, spec));
        if (const int rc = verify_loaded(denied_engine); rc != 0) { return rc; }
        const ninfer::GenerationResult denied_first =
            denied_engine.generate(denied_engine.prepare_tokens(p1), greedy(kPastFreezeOutputs, true));
        if (denied_first.generated_token_ids.size() != kPastFreezeOutputs) {
            return fail("no-prefix-reuse first visit did not generate");
        }
        std::vector<ninfer::TokenId> denied_follow = p1;
        denied_follow.insert(denied_follow.end(), denied_first.generated_token_ids.begin(),
                             denied_first.generated_token_ids.end());
        denied_follow.push_back(kPadRb2);
        denied_follow.insert(denied_follow.end(), 32, 198);
        const ninfer::GenerationResult denied = denied_engine.generate(
            denied_engine.prepare_tokens(denied_follow), greedy(kPastFreezeOutputs, false));
        if (denied.captured_context_checkpoint_tokens != 0 ||
            denied.prefix_reuse_path == ninfer::PrefixReusePath::RestoreTurnRollback) {
            return fail("--no-prefix-reuse still wrote or restored turn-rollback");
        }
    }

    {
        ninfer::Engine c2(engine_options(artifact, 2, 65536, false, 32768, spec));
        if (const int rc = verify_loaded(c2); rc != 0) { return rc; }
        const ninfer::GenerationResult c2_r1 =
            c2.generate(c2.prepare_tokens(p1), greedy(kPastFreezeOutputs, true));
        if (c2_r1.generated_token_ids.size() != kPastFreezeOutputs) {
            return fail("C=2 turn-rollback first visit did not generate");
        }
        std::vector<ninfer::TokenId> c2_follow = p1;
        c2_follow.insert(c2_follow.end(), c2_r1.generated_token_ids.begin(),
                         c2_r1.generated_token_ids.end());
        c2_follow.push_back(kPadRb2);
        c2_follow.insert(c2_follow.end(), 32, 198);
        const ninfer::GenerationResult c2_r2 =
            c2.generate(c2.prepare_tokens(c2_follow), greedy(kPastFreezeOutputs, true));
        if (c2_r2.prefix_reuse_path != ninfer::PrefixReusePath::AppendAtFrontier ||
            c2_r2.captured_context_checkpoint_tokens == 0) {
            std::cerr << "C=2 turn-rollback append: path=" << path_name(c2_r2.prefix_reuse_path)
                      << " captured=" << c2_r2.captured_context_checkpoint_tokens << '\n';
            return 1;
        }
        const std::uint32_t c2_e1 = c2_r2.reused_prompt_tokens;
        const auto freeze_p       = padded(kPadRbC2, kMark);
        const ninfer::GenerationResult freeze_r =
            c2.generate(c2.prepare_tokens(freeze_p), greedy(kPastFreezeOutputs, true));
        if (const int rc =
                expect_captured(freeze_r, kMark, "C=2 freeze while rollback occupies 2C");
            rc != 0) {
            return rc;
        }
        std::vector<ninfer::TokenId> c2_edit(c2_follow.begin(),
                                             c2_follow.begin() + static_cast<std::ptrdiff_t>(c2_e1));
        c2_edit.push_back(kPadRb3);
        c2_edit.insert(c2_edit.end(), 32, 198);
        const ninfer::GenerationResult c2_hit =
            c2.generate(c2.prepare_tokens(c2_edit), greedy(kPastFreezeOutputs, true));
        if (const int rc = expect_rollback(c2_hit, ninfer::PrefixReuseSource::VramResident, c2_e1,
                                           "C=2 rollback D2D after other-lane ladder freeze");
            rc != 0) {
            return rc;
        }

        const auto other_p1 = padded(kPadRbC3, kRb);
        const ninfer::GenerationResult other_r1 =
            c2.generate(c2.prepare_tokens(other_p1), greedy(kPastFreezeOutputs, true));
        if (other_r1.generated_token_ids.size() != kPastFreezeOutputs) {
            return fail("C=2 other-lane first visit did not generate");
        }
        std::vector<ninfer::TokenId> other_follow = other_p1;
        other_follow.insert(other_follow.end(), other_r1.generated_token_ids.begin(),
                            other_r1.generated_token_ids.end());
        other_follow.push_back(kPadRb2);
        other_follow.insert(other_follow.end(), 32, 198);
        const ninfer::GenerationResult other_r2 =
            c2.generate(c2.prepare_tokens(other_follow), greedy(kPastFreezeOutputs, true));
        if (other_r2.prefix_reuse_path != ninfer::PrefixReusePath::AppendAtFrontier) {
            std::cerr << "C=2 other-lane append: path=" << path_name(other_r2.prefix_reuse_path)
                      << '\n';
            return 1;
        }

        const auto before_cold = c2.runtime_stats().computed_prefill_tokens;
        ninfer::CancellationView cold_cancel([&] {
            return c2.runtime_stats().computed_prefill_tokens >= before_cold + 4096;
        });
        const ninfer::GenerationResult cold = c2.generate(
            c2.prepare_tokens(padded(kPadRbCold, 8192)), greedy(1, false), nullptr, cold_cancel);
        if (cold.finish_reason != ninfer::FinishReason::Cancelled) {
            std::cerr << "C=2 RAM-evict canceller finish_reason is "
                      << static_cast<int>(cold.finish_reason) << ", expected Cancelled\n";
            return 1;
        }
        const ninfer::GenerationResult ram_a =
            c2.generate(c2.prepare_tokens(c2_edit), greedy(kPastFreezeOutputs, true));
        if (const int rc = expect_rollback(ram_a, ninfer::PrefixReuseSource::HostRam, c2_e1,
                                           "C=2 RAM rollback while the other lane stays live");
            rc != 0) {
            return rc;
        }
    }
    return 0;
}

int exercise_artifact(const char* artifact, ninfer::SpeculativeBackend spec) {
    if (const int rc = exercise_single_lane(artifact, spec); rc != 0) { return rc; }
    if (const int rc = exercise_catch_up(artifact, spec); rc != 0) { return rc; }
    if (const int rc = exercise_cancel(artifact, spec); rc != 0) { return rc; }
    if (const int rc = exercise_c2(artifact, spec); rc != 0) { return rc; }
    if (const int rc = exercise_c3(artifact, spec); rc != 0) { return rc; }
    if (const int rc = exercise_c3_cancel_during_decode(artifact, spec); rc != 0) { return rc; }
    if (spec != ninfer::SpeculativeBackend::DFlash) {
        if (const int rc = exercise_vision(artifact); rc != 0) { return rc; }
    }
    if (const int rc = exercise_two_marks(artifact, spec); rc != 0) { return rc; }
    if (const int rc = exercise_ram_shorter_ladder(artifact, spec); rc != 0) { return rc; }
    if (const int rc = exercise_c2_drop_during_decode(artifact, spec); rc != 0) { return rc; }
    if (const int rc = exercise_mtp_off(artifact); rc != 0) { return rc; }
    if (const int rc = exercise_chat_rewrite_tie(artifact, spec); rc != 0) { return rc; }
    if (const int rc = exercise_rewrite_same_f(artifact, spec); rc != 0) { return rc; }
    if (const int rc = exercise_turn_rollback(artifact, spec); rc != 0) { return rc; }
    if (const int rc = exercise_restore_matches_cold(artifact, spec); rc != 0) { return rc; }
    if (const int rc = exercise_150k(artifact, spec); rc != 0) { return rc; }
    return 0;
}

} // namespace

int main() {
    try {
        const char* groupwise = std::getenv("NINFER_QWEN3_6_27B_WEIGHTS");
        const char* nvfp4     = std::getenv("NINFER_QWEN3_6_27B_NVFP4_WEIGHTS");
        const char* dflash    = std::getenv("NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS");
        if ((groupwise == nullptr || *groupwise == '\0') &&
            (nvfp4 == nullptr || *nvfp4 == '\0') && (dflash == nullptr || *dflash == '\0')) {
            std::cout << "skip: set NINFER_QWEN3_6_27B_WEIGHTS, "
                         "NINFER_QWEN3_6_27B_NVFP4_WEIGHTS, or "
                         "NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS\n";
            return 77;
        }
        if (groupwise != nullptr && *groupwise != '\0') {
            if (const int result =
                    exercise_artifact(groupwise, ninfer::SpeculativeBackend::Mtp);
                result != 0) {
                return result;
            }
        }
        if (nvfp4 != nullptr && *nvfp4 != '\0') {
            if (const int result = exercise_artifact(nvfp4, ninfer::SpeculativeBackend::Mtp);
                result != 0) {
                return result;
            }
        }
        if (dflash != nullptr && *dflash != '\0') {
            if (const int result =
                    exercise_artifact(dflash, ninfer::SpeculativeBackend::DFlash);
                result != 0) {
                return result;
            }
        }
        std::cout << "ok\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
