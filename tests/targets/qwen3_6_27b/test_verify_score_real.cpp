// Qualification executable: gold histories through the production Verify phase and replay commit.
// File arguments are raw UTF-8 documents; no chat template or sampler filtering is applied.
#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "runtime/engine/kv_capacity.h"
#include "targets/qwen3_6_27b/impl/variant.h"
#define NINFER_QWEN36_VARIANT ::ninfer::targets::qwen3_6_27b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_27b_runtime
#include "targets/qwen3_6/impl/runtime/program.h"
#include "targets/qwen3_6/impl/frontend/tokenizer.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/nll_from_logits.h"
#include "ninfer/ops/embedding.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_6::detail::qwen3_6_27b_runtime {
// The only private access is materialization and the production deferred feature append.
// Token forcing, logit inspection, and pending metadata live exclusively in this executable.
struct VerificationQualification {
    static void prepare(ProgramImplCore& p, std::span<const std::uint32_t> lanes,
                        std::uint32_t width) {
        std::vector<std::uint32_t> append_lanes, starts, counts;
        for (auto lane : lanes) {
            auto& s = p.sequences[lane];
            if (s.dflash_context_frontier != s.execution_frontier) {
                append_lanes.push_back(lane);
                starts.push_back(s.dflash_context_frontier);
                counts.push_back(s.execution_frontier - s.dflash_context_frontier);
            }
        }
        if (!append_lanes.empty()) { p.enqueue_dflash_context_append(append_lanes, starts, counts); }
        for (auto lane : lanes) {
            auto& s = p.sequences[lane];
            s.dflash_context_frontier = s.execution_frontier;
            p.materialize_sequence_kv(s, s.execution_frontier + width, 0);
        }
    }
};
}

namespace {
namespace target = ninfer::targets::qwen3_6_27b::detail;
namespace family = ninfer::targets::qwen3_6;
namespace execution = family::detail::qwen3_6_27b_runtime;
using Package = ninfer::targets::qwen3_6_27b::Package;
using ninfer::TokenId;

void require(bool ok, const char* message) {
    if (!ok) { throw std::runtime_error(message); }
}

template <class T>
void upload(ninfer::Tensor tensor, const std::vector<T>& values, cudaStream_t stream) {
    require(tensor.is_contiguous(), "qualification upload must be contiguous");
    CUDA_CHECK(cudaMemcpyAsync(tensor.data, values.data(), values.size() * sizeof(T),
                               cudaMemcpyHostToDevice, stream));
}

double bf16(std::uint16_t bits) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
}

struct Sample { std::size_t document, position; TokenId token; double nll; };

std::vector<Sample> score(execution::ProgramImplCore& p, const family::Frontend& frontend,
                          const std::vector<std::vector<TokenId>>& documents,
                          std::size_t prefix, std::size_t limit, std::uint32_t width,
                          TokenId padding_token = 0, std::uint32_t commit_limit = 16) {
    std::vector<Sample> scores;
    const auto batch = static_cast<std::int32_t>(documents.size());
    std::vector<std::uint32_t> lanes(batch), counts(batch);
    std::vector<std::uint8_t> flags(batch);
    std::vector<std::size_t> ends(batch);
    for (int row = 0; row < batch; ++row) {
        lanes[row] = row;
        require(documents[row].size() > prefix + 1, "document is shorter than prefix + two tokens");
        ends[row] = std::min(documents[row].size() - 1, prefix + limit);
        ninfer::runtime::ResolvedExecutionOptions request;
        request.requested_output_tokens = static_cast<std::uint32_t>(ends[row] - prefix + 1);
        request.allow_prefix_reuse = false;
        request.sampling.temperature = 0;
        auto prompt = family::PreparedPromptAccess::take(frontend.prepare_tokens(
            std::vector<TokenId>(documents[row].begin(), documents[row].begin() + prefix)));
        auto base = p.plan_request_base(prompt, request);
        auto plan = p.plan_request_for_lane(row, prompt, base);
        auto step = p.start_prefill_lane(row, std::move(prompt), std::move(plan), {});
        while (!step.complete) { step = p.advance_prefill_lane(row); }
        p.resolve_prefill_lane(row, false);
        // Prefill consumed [0,prefix). Replace only its unexecuted sampled anchor with gold.
        p.sequences[row].ledger.back() = documents[row][prefix];
        require(p.sequences[row].execution_frontier == prefix, "prefill alignment mismatch");
    }

    auto& frame = *p.io.dflash_decode;
    const auto w = static_cast<std::int32_t>(width);
    auto ids = frame.verify_ids.slice(1, 0, batch);
    auto positions = frame.cache_positions.slice(1, 0, batch);
    auto rope = frame.target_rope_positions.slice(1, 0, batch);
    auto valid = frame.target_valid_columns.slice(0, 0, batch);
    auto rows = frame.text_kv_table_rows.slice(0, 0, batch);
    auto slots = frame.lanes.slice(0, 0, batch);
    auto hidden = frame.target_hidden.slice(2, 0, batch);
    auto logits = frame.target_logits.slice(2, 0, batch);
    auto tokens = frame.target_argmax.slice(1, 0, batch);
    require(ids.ne[0] == w, "qualification width differs from startup width");
    const std::size_t vocab = target::TextConfig::output_rows;
    const std::size_t domain = target::TextConfig::token_domain;
    std::vector<std::uint16_t> host_logits(vocab * width * batch);
    std::vector<std::int32_t> host_argmax(width * batch);
    // Qualification-only capture. Each record is four LE u32s (domain, lane, position,
    // gold ID), followed by domain BF16 logits. Default: first lane/column at each host
    // checkpoint. An explicit position list also captures token-level regression outliers.
    static_assert(std::endian::native == std::endian::little);
    std::ofstream logit_dump;
    std::ofstream record_dump;
    if (const char* path = std::getenv("NINFER_VERIFY_RECORD_DUMP")) {
        require(batch == 1, "record diagnostics require C=1");
        record_dump.open(path, std::ios::binary);
        require(bool(record_dump), "cannot open replay record dump");
    }
    if (const char* path = std::getenv("NINFER_VERIFY_LOGITS_DUMP")) {
        logit_dump.open(path, std::ios::binary);
        require(bool(logit_dump), "cannot open verifier logit dump");
    }
    std::set<std::size_t> capture_positions;
    if (const char* positions = std::getenv("NINFER_VERIFY_LOGIT_POSITIONS")) {
        std::istringstream input(positions);
        std::string item;
        while (std::getline(input, item, ',')) { capture_positions.insert(std::stoul(item)); }
    }
    std::size_t round = 0;
    while (true) {
        bool any = false;
        std::vector<std::int32_t> host_ids(width * batch), host_pos(width * batch),
            host_valid(batch), host_rows(batch), host_slots(batch), selectors(batch);
        std::uint32_t max_end = 0;
        for (int row = 0; row < batch; ++row) {
            auto& s = p.sequences[row];
            counts[row] = static_cast<std::uint32_t>(
                std::min<std::size_t>(std::min(width, commit_limit), ends[row] - s.execution_frontier));
            any |= counts[row] != 0;
            // Finished rows remain masked in the final heterogeneous batch. Stop before
            // attempting another batch with a zero-count row (documents are scored in waves).
            if (counts[row] == 0) { continue; }
            host_valid[row] = counts[row];
            host_rows[row] = s.kv->text.bound_row();
            host_slots[row] = row;
            selectors[row] = counts[row] - 1;
            max_end = std::max(max_end, s.execution_frontier + width);
            for (std::uint32_t col = 0; col < width; ++col) {
                host_pos[row * width + col] = s.execution_frontier + col;
                host_ids[row * width + col] = col < counts[row]
                    ? documents[row][s.execution_frontier + col] : padding_token;
            }
        }
        if (!any) { break; }
        require(std::all_of(counts.begin(), counts.end(), [](auto n) { return n > 0; }),
                "documents in a wave must have equal scored lengths (final partial width allowed)");
        execution::VerificationQualification::prepare(p, lanes, width);
        upload(ids, host_ids, p.device.stream);
        upload(positions, host_pos, p.device.stream);
        upload(rope, host_pos, p.device.stream);
        upload(valid, host_valid, p.device.stream);
        upload(rows, host_rows, p.device.stream);
        upload(slots, host_slots, p.device.stream);
        if (round == 0) {
            if (const char* path = std::getenv("NINFER_VERIFY_GDN_INPUT_DUMP")) {
                // Layer zero is GDN: reproduce its actual embedding + normalization/control
                // entry on these gold IDs to supply real activations to the independent Op test.
                static_assert(!target::TextConfig::is_full_attention(0));
                auto scope = p.work.scope();
                const int columns = w * batch;
                auto embedding = p.work.alloc(ninfer::DType::BF16, {target::TextConfig::hidden, columns});
                auto normalized = p.work.alloc(ninfer::DType::BF16, {target::TextConfig::hidden, columns});
                auto g = p.work.alloc(ninfer::DType::FP32, {target::TextConfig::gdn_value_heads, columns});
                auto beta = p.work.alloc(ninfer::DType::FP32, {target::TextConfig::gdn_value_heads, columns});
                ninfer::ops::embedding(ids.view({columns}), p.model.token_embedding, embedding, p.device.stream);
                const auto& first = p.model.gdn_layers[0];
                target::Variant::gdn_norm_control_projection(embedding, first.input_norm,
                    target::TextConfig::rms_epsilon, first.projection, normalized, g, beta,
                    p.work, p.device.stream, w);
                std::vector<std::uint16_t> bits(target::TextConfig::hidden * columns);
                CUDA_CHECK(cudaMemcpyAsync(bits.data(), normalized.data, bits.size() * sizeof(bits[0]),
                    cudaMemcpyDeviceToHost, p.device.stream));
                p.device.synchronize();
                std::ofstream output(path);
                require(bool(output), "cannot write real GDN activation fixture");
                output << std::setprecision(9);
                for (auto b : bits) { output << bf16(b) << '\n'; }
                require(bool(output), "failed to write real GDN activation fixture");
            }
        }
        execution::schedule::TextContext card(p.device, p.model, p.work, {},
            p.decoder->linear_attention, p.io, p.prefill_hidden, p.prefill_chunk, 0, {},
            &p.decoder->text_kv);
        card.set_gdn_state_action(execution::schedule::GdnStateAction::RecordForReplay,
                                  &*p.replay_records);
        execution::schedule::DFlashFeatureSink sink{
            .batch_features = &p.dflash->pending_features,
            .batch_lanes = &slots,
            .batch_valid_columns = &valid,
            .batch_width = w,
            .batch_size = batch,
            .layers = std::span<const int>(target::DFlashConfig::target_feature_layers),
        };
        card.target_verify_batch(ids, positions, rope, valid, rows, slots, {1, max_end},
                                  hidden, logits, tokens, sink);
        auto gold_tensor = p.work.alloc(ninfer::DType::I32, {w * batch});
        auto nll_tensor = p.work.alloc(ninfer::DType::FP32, {w * batch});
        std::vector<std::int32_t> gold_columns(width * batch);
        std::vector<float> nll_columns(width * batch);
        for (int row = 0; row < batch; ++row) {
            for (std::uint32_t col = 0; col < counts[row]; ++col) {
                gold_columns[row * width + col] =
                    documents[row][p.sequences[row].execution_frontier + col + 1];
            }
        }
        upload(gold_tensor, gold_columns, p.device.stream);
        ninfer::ops::nll_from_logits(logits.view({static_cast<int>(vocab), w * batch}),
                                     gold_tensor, nll_tensor, domain, p.device.stream);
        CUDA_CHECK(cudaMemcpyAsync(nll_columns.data(), nll_tensor.data,
            nll_columns.size() * sizeof(float), cudaMemcpyDeviceToHost, p.device.stream));
        const auto first_position = p.sequences[0].execution_frontier + 1;
        const auto requested = capture_positions.lower_bound(first_position);
        const bool check_host = round % 256 == 0 ||
            (requested != capture_positions.end() && *requested < first_position + counts[0]) ||
            std::any_of(counts.begin(), counts.end(), [width](auto n) { return n < width; });
        if (check_host) {
            CUDA_CHECK(cudaMemcpyAsync(host_logits.data(), logits.data,
                host_logits.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost, p.device.stream));
            CUDA_CHECK(cudaMemcpyAsync(host_argmax.data(), tokens.data,
                host_argmax.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost, p.device.stream));
        }
        p.device.synchronize();
        if (record_dump.is_open()) {
            for (int layer = 0; layer < p.replay_records->spec.layers; ++layer) {
                const auto record = p.replay_records->layer(layer, 1);
                const ninfer::Tensor planes[] = {record.conv, record.key, record.value, record.gate};
                for (std::uint32_t plane = 0; plane < 4; ++plane) {
                    const auto bytes = planes[plane].bytes() / width;
                    std::vector<char> values(planes[plane].bytes());
                    CUDA_CHECK(cudaMemcpy(values.data(), planes[plane].data, values.size(),
                                          cudaMemcpyDeviceToHost));
                    for (std::uint32_t col = 0; col < counts[0]; ++col) {
                        const std::uint32_t header[] = {p.sequences[0].execution_frontier + col,
                            static_cast<std::uint32_t>(layer), plane, static_cast<std::uint32_t>(bytes)};
                        record_dump.write(reinterpret_cast<const char*>(header), sizeof(header));
                        record_dump.write(values.data() + col * bytes, bytes);
                    }
                }
            }
            require(bool(record_dump), "failed writing replay record dump");
        }
        for (int row = 0; row < batch; ++row) {
            auto& s = p.sequences[row];
            for (std::uint32_t col = 0; col < counts[row]; ++col) {
                const auto position = s.execution_frontier + col + 1;
                const auto gold = documents[row][position];
                const double nll = nll_columns[row * width + col];
                if (check_host) {
                    const auto* l = host_logits.data() + (row * width + col) * vocab;
                    double maximum = -std::numeric_limits<double>::infinity();
                    std::size_t argmax = 0;
                    for (std::size_t v = 0; v < domain; ++v) {
                        if (bf16(l[v]) > maximum) { maximum = bf16(l[v]); argmax = v; }
                    }
                    require(host_argmax[row * width + col] == static_cast<std::int32_t>(argmax),
                            "verifier greedy token differs from actual-logit argmax/tie rule");
                    double sum = 0;
                    for (std::size_t v = 0; v < domain; ++v) { sum += std::exp(bf16(l[v]) - maximum); }
                    const double reference = maximum + std::log(sum) - bf16(l[gold]);
                    require(std::abs(nll - reference) < 2e-5, "GPU NLL differs from FP64 logsumexp");
                    if (logit_dump.is_open() && row == 0 &&
                        (capture_positions.empty() ? col == 0 : capture_positions.contains(position))) {
                        const std::uint32_t header[] = {static_cast<std::uint32_t>(domain),
                            static_cast<std::uint32_t>(row), position, static_cast<std::uint32_t>(gold)};
                        logit_dump.write(reinterpret_cast<const char*>(header), sizeof(header));
                        logit_dump.write(reinterpret_cast<const char*>(l), domain * sizeof(*l));
                        require(bool(logit_dump), "failed writing verifier logit dump");
                    }
                }
                require(std::isfinite(nll) && nll >= 0, "invalid verifier NLL");
                scores.push_back({static_cast<std::size_t>(row), position, gold, nll});
                p.dflash_host_egress->licensed_tokens[row * width + col] = gold;
            }
            auto& r = p.requests[row];
            r.pending = {.kind = execution::PendingKind::Speculative,
                         .base_E = s.execution_frontier, .base_S = s.ledger_frontier,
                         .produced = counts[row], .round_k = width - 1, .verify_width = width};
            r.lifecycle = execution::Lifecycle::Pending;
        }
        auto select = frame.proposal_extents.slice(0, 0, batch);
        auto selected = frame.target_continuation_hidden.slice(1, 0, batch);
        upload(select, selectors, p.device.stream);
        ninfer::ops::speculative_select_accepted_hidden(hidden, select, selected, p.device.stream);
        ninfer::ops::scatter(selected, slots, p.tail_hidden_store, p.device.stream);
        p.resolve_pending_batch(lanes, counts, flags, flags);
        for (int row = 0; row < batch; ++row) {
            const auto& s = p.sequences[row];
            require(s.ledger.back() == documents[row][s.execution_frontier],
                    "committed anchor is not next gold input");
        }
        ++round;
    }
    std::vector<std::size_t> next(batch, prefix + 1);
    for (const auto& sample : scores) {
        require(sample.position == next[sample.document]++ &&
                    sample.token == documents[sample.document][sample.position],
                "scored positions or gold tokens are misaligned");
    }
    for (int row = 0; row < batch; ++row) {
        require(next[row] == ends[row] + 1, "scored token count mismatch");
    }
    for (auto lane : lanes) { p.abort_lane(lane); }
    return scores;
}
}

// Actual production draft/Verify/sample/commit rounds. The mathematical checks use
// the candidate's represented logits, never an ordinary scorer's different history.
void check_decode(execution::ProgramImplCore& p, const family::Frontend& frontend,
                  const std::vector<std::vector<TokenId>>& documents, unsigned prefix,
                  unsigned limit, unsigned width, const std::string& mode) {
    const bool p_less = mode == "p-less";
    const bool greedy = mode == "greedy";
    require(p_less || greedy || mode == "stochastic", "decode check expects greedy, p-less or stochastic");
    require(limit >= 16 * width + 1, "decode check needs room for 16 full rounds");
    const unsigned batch = documents.size();
    std::vector<std::uint32_t> lanes(batch), counts(batch);
    std::vector<std::uint8_t> flags(batch);
    std::vector<ninfer::runtime::RoundBudget> budgets(batch);
    std::vector<std::vector<TokenId>> generated(batch);
    for (unsigned row = 0; row < batch; ++row) {
        lanes[row] = row;
        require(documents[row].size() >= prefix, "decode prompt is too short");
        ninfer::runtime::ResolvedExecutionOptions request;
        request.requested_output_tokens = limit;
        request.allow_prefix_reuse = false;
        request.sampling.temperature = greedy ? 0.0F : (p_less ? 2.0F : 0.8F);
        request.sampling.p_less = p_less;
        request.sampling.top_k = 20;
        request.sampling.top_p = 0.95F;
        request.sampling.seed = 15446143373561885318ULL + row;
        auto prompt = family::PreparedPromptAccess::take(frontend.prepare_tokens(
            std::vector<TokenId>(documents[row].begin(), documents[row].begin() + prefix)));
        auto base = p.plan_request_base(prompt, request);
        auto plan = p.plan_request_for_lane(row, prompt, base);
        auto step = p.start_prefill_lane(row, std::move(prompt), std::move(plan), {});
        while (!step.complete) { step = p.advance_prefill_lane(row); }
        p.resolve_prefill_lane(row, false);
        generated[row].push_back(p.sequences[row].ledger.back());
        budgets[row].generated_tokens_remaining = limit - 1;
    }
    const auto vocab = target::TextConfig::output_rows;
    const auto domain = target::TextConfig::token_domain;
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(vocab) * width * batch);
    std::vector<std::int32_t> drafts((width - 1) * batch);
    unsigned checked = 0;
    for (unsigned round_index = 0; round_index < 16; ++round_index) {
        std::vector<std::uint32_t> old_execution(batch), old_ledger(batch);
        for (unsigned row = 0; row < batch; ++row) {
            old_execution[row] = p.sequences[row].execution_frontier;
            old_ledger[row] = p.sequences[row].ledger_frontier;
        }
        const auto round = p.decode_batch(lanes, budgets);
        const auto& frame = *p.io.dflash_decode;
        CUDA_CHECK(cudaMemcpy(logits.data(), frame.target_logits.data, logits.size() * 2, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(drafts.data(), frame.draft_tokens.data, drafts.size() * 4, cudaMemcpyDeviceToHost));
        for (unsigned row = 0; row < batch; ++row) {
            const auto& pending = p.requests[row].pending;
            require(!pending.tree_verify && pending.verify_width == width, "decode check requires fixed chain width");
            counts[row] = round.row_counts[row];
            require(counts[row] >= 1 && counts[row] <= width, "invalid licensed count");
            for (unsigned col = 0; col < counts[row]; ++col) {
                const auto token = round.tokens[row * round.row_stride + col];
                require(token >= 0 && token < domain, "licensed token outside domain");
                const auto* values = logits.data() + (row * width + col) * vocab;
                double maximum = -std::numeric_limits<double>::infinity();
                int argmax = 0;
                for (int v = 0; v < domain; ++v) {
                    require(std::isfinite(bf16(values[v])), "nonfinite candidate logit");
                    if (bf16(values[v]) > maximum) { maximum = bf16(values[v]); argmax = v; }
                }
                if (greedy) {
                    require(token == argmax, "licensed greedy token differs from actual candidate argmax");
                } else if (p_less) {
                    double sum = 0, square_sum = 0;
                    for (int v = 0; v < domain; ++v) {
                        const double e = std::exp((bf16(values[v]) - maximum) / 2.0);
                        sum += e;
                        square_sum += e * e;
                    }
                    // Independent public formula: epsilon=1/16, M=1024, T=2.
                    const double cut = std::max(square_sum / (sum * sum) * std::exp(-1.0 / 16.0), 1.0 / 1024.0);
                    const double probability = std::exp((bf16(values[token]) - maximum) / 2.0) / sum;
                    require(probability >= cut || (1.0 / sum < cut && token == argmax),
                            "licensed p-less token outside actual candidate support");
                } else {
                    std::vector<int> ranked(domain);
                    for (int v = 0; v < domain; ++v) { ranked[v] = v; }
                    std::partial_sort(ranked.begin(), ranked.begin() + 20, ranked.end(),
                        [&](int left, int right) {
                            const double a = bf16(values[left]), b = bf16(values[right]);
                            return a != b ? a > b : left < right;
                        });
                    double weights[20], sum = 0;
                    for (int i = 0; i < 20; ++i) {
                        weights[i] = std::exp((bf16(values[ranked[i]]) - maximum) / static_cast<double>(0.8F));
                        sum += weights[i];
                    }
                    double cumulative = 0;
                    bool allowed = false;
                    for (int i = 0; i < 20; ++i) {
                        allowed |= token == ranked[i];
                        cumulative += weights[i];
                        if (cumulative >= static_cast<double>(0.95F) * sum) { break; }
                    }
                    require(allowed, "licensed stochastic token outside actual top-k/top-p support");
                }
                if (col + 1 < counts[row]) {
                    require(token == drafts[row * (width - 1) + col], "accepted token differs from draft prefix");
                }
                generated[row].push_back(token);
                ++checked;
            }
            budgets[row].generated_tokens_remaining -= counts[row];
        }
        p.resolve_pending_batch(lanes, counts, flags, flags);
        for (unsigned row = 0; row < batch; ++row) {
            const auto& sequence = p.sequences[row];
            require(sequence.execution_frontier == old_execution[row] + counts[row] &&
                    sequence.ledger_frontier == old_ledger[row] + counts[row] &&
                    sequence.ledger_frontier == sequence.execution_frontier + 1,
                    "commit frontier/anchor mismatch");
            for (unsigned col = 0; col < counts[row]; ++col) {
                require(sequence.ledger[old_ledger[row] + col] ==
                            generated[row][generated[row].size() - counts[row] + col],
                        "committed ledger differs from licensed tokens");
            }
        }
    }
    if (greedy) {
        std::cout << "Greedy row0 prefix:";
        for (std::size_t i = 0; i < std::min<std::size_t>(24, generated[0].size()); ++i) {
            std::cout << ' ' << generated[0][i];
        }
        std::cout << '\n';
    }
    for (auto lane : lanes) { p.abort_lane(lane); }
    std::cout << "PASS actual candidate " << mode << " licensing C=" << batch << " W=" << width
              << " rounds=16 tokens=" << checked << '\n';
}

int main(int argc, char** argv) {
    const char* artifact = std::getenv("NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS");
    if (!artifact || !*artifact) {
        std::cout << "SKIP: set NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS\n";
        return 77;
    }
    try {
        const unsigned batch = argc > 1 ? std::stoul(argv[1]) : 1;
        const unsigned width = argc > 2 ? std::stoul(argv[2]) : 5;
        const unsigned prefix = argc > 3 ? std::stoul(argv[3]) : 8;
        const unsigned limit = argc > 4 ? std::stoul(argv[4]) : 31;
        require(batch >= 1 && batch <= 6 && width >= 2 && width <= 6 && prefix > 0 && limit > 0,
                "usage: verify_score [C=1..6 W=2..6 prefix scored_tokens [file [--ids]]]");
        ninfer::DeviceContext device;
        ninfer::EngineOptions options;
        options.artifact_path = artifact;
        options.max_context = prefix + limit + width + 16;
        options.max_concurrency = batch;
        options.prefill_chunk = 4096;
        options.kv_capacity = ninfer::KvCapacityPolicy::automatic(1024ULL * 1024 * 1024);
        // The no-argument alignment control uses BF16 KV to separate attention arithmetic
        // partition effects from KV codecs. The NVFP4 SmallT route quantizes current K/V
        // too; do not attribute its partition differences to a BF16-current boundary.
        // Corpus qualification defaults to production NVFP4 KV.
        options.kv_cache = argc == 1 || std::getenv("NINFER_VERIFY_BF16_KV") != nullptr
            ? ninfer::KvCacheStorage::BFloat16 : ninfer::KvCacheStorage::Nvfp4;
        options.enable_vision = false;
        options.use_cuda_graph = false;
        options.speculative.backend = ninfer::SpeculativeBackend::DFlash;
        options.speculative.draft_tokens = width - 1;
        options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
        ninfer::artifact::Reader reader(artifact);
        if (const char* path = std::getenv("NINFER_VERIFY_ARTIFACT_INVENTORY")) {
            std::ofstream output(path);
            for (const auto& object : reader.objects()) {
                if (const auto* tensor = std::get_if<ninfer::artifact::TensorDescriptor>(&object)) {
                    output << tensor->name << '\t' << ninfer::artifact::format_name(tensor->format);
                    for (auto dim : tensor->shape) { output << '\t' << dim; }
                    output << '\n';
                }
            }
            require(bool(output), "failed writing artifact inventory");
        }
        ninfer::artifact::Binder binder(reader);
        options.model_id = reader.identity().model_id;
        options.weights_id = reader.identity().weights_id;
        options.artifact_file_identity = reader.file_identity();
        const auto profile = Package::resolve_weights(reader.identity(), binder);
        auto load = target::bind_artifact(binder, profile, family::startup_features(options));
        auto materialized = ninfer::artifact::materialize(reader, load.materialization, device);
        target::LoadedModelData model(std::move(load.bindings), std::move(materialized));
        auto frontend = family::make_frontend(model.frontend, false);
        family::frontend_internal::Tokenizer tokenizer({model.frontend.tokenizer_json,
            model.frontend.tokenizer_config_json, model.frontend.generation_config_json});
        std::string text = "A triangle has three sides. The sum of its interior angles in Euclidean "
            "geometry is 180 degrees. To prove this, draw a line through one vertex parallel to "
            "the opposite side. The alternate interior angles equal the other two angles, "
            "and the three angles together form a straight line. This proof uses the parallel "
            "postulate, which distinguishes Euclidean geometry from curved geometries.";
        if (argc > 5) {
            std::ifstream input(argv[5]);
            require(bool(input), "cannot open corpus document");
            text.assign(std::istreambuf_iterator<char>(input), {});
        }
        auto gold = tokenizer.encode(text, {.parse_added_tokens = false});
        if (argc > 6) {
            require(std::string(argv[6]) == "--ids", "only --ids is accepted after file");
            gold.clear();
            std::ifstream input(argv[5]);
            TokenId id;
            while (input >> id) {
                require(id >= 0 && id < target::TextConfig::token_domain && tokenizer.is_valid_token(id),
                        "corpus contains an invalid token id");
                gold.push_back(id);
            }
            require(input.eof(), "malformed token-id corpus");
        }
        if (const char* positions = std::getenv("NINFER_VERIFY_LOGIT_POSITIONS")) {
            std::istringstream input(positions);
            std::string item;
            while (std::getline(input, item, ',')) {
                const auto position = std::stoul(item);
                if (position >= gold.size()) { continue; }
                const auto begin = position > 24 ? position - 24 : 0;
                const auto end = std::min(gold.size(), position + 12);
                std::cerr << "Context position=" << position << " gold="
                          << std::quoted(tokenizer.decode(std::span(gold).subspan(position, 1)))
                          << " text=" << std::quoted(tokenizer.decode(std::span(gold).subspan(begin, end - begin)))
                          << '\n';
            }
        }
        std::vector<std::vector<TokenId>> documents(batch, gold);
        // Fixed offsets identify distinct gold histories independently of physical lane order.
        std::vector<std::size_t> document_ids(batch);
        for (unsigned row = 0; row < batch; ++row) { document_ids[row] = row; }
        if (const char* offsets = std::getenv("NINFER_VERIFY_DOCUMENT_OFFSETS")) {
            std::istringstream input(offsets);
            std::string item;
            unsigned row = 0;
            std::set<std::size_t> seen;
            while (std::getline(input, item, ',')) {
                const auto offset = std::stoul(item);
                require(row < batch && offset < gold.size() && seen.insert(offset).second,
                        "invalid or duplicate document offset");
                document_ids[row] = offset;
                documents[row++].assign(gold.begin() + offset, gold.end());
            }
            require(row == batch, "document offsets must specify every lane");
        }
        auto planner = Package::make_sequence_planner(device, options, profile);
        std::size_t free = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        const auto resolution = ninfer::runtime::resolve_kv_capacity(
            options.kv_capacity, planner.capacity_curve(), free);
        auto plan = std::move(planner).finalize(resolution.main_page_groups);
        execution::ProgramImplCore program(model.runtime, *plan.impl_, device, nullptr);
        if (const char* mode = std::getenv("NINFER_VERIFY_DECODE_CHECK")) {
            check_decode(program, frontend, documents, prefix, limit, width, mode);
            return 0;
        }
        const auto commit_limit = std::getenv("NINFER_VERIFY_COMMIT_LIMIT")
            ? std::stoul(std::getenv("NINFER_VERIFY_COMMIT_LIMIT")) : width;
        require(commit_limit > 0 && commit_limit <= width, "invalid diagnostic commit limit");
        const auto samples = score(program, frontend, documents, prefix, limit, width, 0, commit_limit);
        if (argc == 1) {
            const auto repeat = score(program, frontend, documents, prefix, limit, width, 42);
            require(samples.size() == repeat.size(), "repeat score count mismatch");
            for (std::size_t i = 0; i < samples.size(); ++i) {
                require(samples[i].nll == repeat[i].nll && samples[i].token == repeat[i].token,
                        "cold repeat verifier score mismatch");
            }
            // Check the first prediction and the prediction immediately after a replay commit
            // against two columns of the same packed forward. This is an alignment control,
            // not a claim that differently partitioned recurrent reductions are bit identical
            // over an entire continuation (state numerical qualification is separate).
            const auto serial = score(program, frontend, documents, prefix, 2, width, 42, 1);
            for (std::size_t i = 0; i < serial.size(); ++i) {
                require(samples[i].nll == serial[i].nll,
                        "first-token/replay-boundary alignment control differs");
            }
            auto changed = documents;
            changed[0][prefix + 3] = 42;
            const auto future = score(program, frontend, changed, prefix, width, width);
            for (std::size_t i = 0; i < 2; ++i) {
                require(samples[i].nll == future[i].nll,
                        "earlier prediction depends on a later valid input token");
            }
        }
        std::cout << "document\tposition\ttoken\tnll\n" << std::setprecision(17);
        double sum = 0;
        for (const auto& s : samples) {
            std::cout << document_ids[s.document] << '\t' << s.position << '\t' << s.token << '\t' << s.nll << '\n';
            sum += s.nll;
        }
        std::cerr << "Verify binary=" << argv[0] << " kv="
                  << (options.kv_cache == ninfer::KvCacheStorage::BFloat16 ? "bf16" : "nvfp4")
                  << " C=" << batch << " W=" << width << " tokens=" << samples.size()
                  << " mean_nll=" << sum / samples.size()
                  << " ppl=" << std::exp(sum / samples.size()) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
