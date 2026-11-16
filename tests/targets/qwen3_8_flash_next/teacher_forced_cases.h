// Included inside the real-artifact test's namespace after its numerical and Engine helpers.
constexpr std::size_t kTeacherPositions = 49;

struct TeacherCriterion {
    double mean_kl;
    double p95_kl;
    double maximum_kl;
    double mean_gap;
    double p95_gap;
    double maximum_gap;
    std::size_t mismatches;
    double lane_mean_kl;
    double lane_mean_gap;
    std::size_t lane_mismatches;
};

// B1 and B4 consume each B1 greedy trajectory, including the prefill distribution and 48
// decode positions. Qualification extrema across three arithmetic profiles were:
//   Regression prompts: KL mean/p95/max .044040/.117163/2.659982; selected-token logprob
//     gap mean/p95/max .100668/.385929/2.848391; top-1 mismatches 10/196, at most 6/49
//     per lane; per-lane mean KL/gap .111844/.210797.
//   Natural prompts: KL .001456/.007106/.032054; gap .017395/.081709/.226830;
//     mismatches 3/196, at most 1/49 per lane; per-lane means .003201/.036605.
// Each bound is twice its observed extremum, rounded upward. Per-lane bounds prevent the
// aggregate from hiding a damaged lane. The fixture executes only the resident Ops.
constexpr std::array<TeacherCriterion, 2> kTeacherCriteria{{
    {.mean_kl         = .09,
     .p95_kl          = .24,
     .maximum_kl      = 5.4,
     .mean_gap        = .21,
     .p95_gap         = .78,
     .maximum_gap     = 5.7,
     .mismatches      = 20,
     .lane_mean_kl    = .23,
     .lane_mean_gap   = .43,
     .lane_mismatches = 12},
    {.mean_kl         = .003,
     .p95_kl          = .015,
     .maximum_kl      = .065,
     .mean_gap        = .035,
     .p95_gap         = .17,
     .maximum_gap     = .46,
     .mismatches      = 6,
     .lane_mean_kl    = .0065,
     .lane_mean_gap   = .075,
     .lane_mismatches = 2},
}};

struct TeacherErrors {
    std::vector<double> kl;
    std::vector<double> gap;
    std::size_t mismatches = 0;
    double sum_kl          = 0.0;
    double sum_gap         = 0.0;

    void add(const DistributionDifference& difference, double selected_gap) {
        require(std::isfinite(difference.kl) && difference.kl >= -1e-10 &&
                    std::isfinite(selected_gap),
                "invalid teacher-forced distribution metrics");
        kl.push_back(std::max(0.0, difference.kl));
        gap.push_back(selected_gap);
        sum_kl += kl.back();
        sum_gap += selected_gap;
        mismatches += difference.top1_equal ? 0U : 1U;
    }

    [[nodiscard]] double mean_kl() const { return sum_kl / static_cast<double>(kl.size()); }

    [[nodiscard]] double mean_gap() const { return sum_gap / static_cast<double>(gap.size()); }
};

// Linear interpolation matches the qualification's percentile definition.
double teacher_p95(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const double position = .95 * static_cast<double>(values.size() - 1);
    const auto lower      = static_cast<std::size_t>(position);
    const double fraction = position - static_cast<double>(lower);
    return values[lower] +
           fraction * (values[std::min(lower + 1, values.size() - 1)] - values[lower]);
}

void require_teacher_errors(const TeacherErrors& errors, const TeacherCriterion& criterion,
                            const std::string& label) {
    std::cout << label << ": KL mean/p95/max " << errors.mean_kl() << '/' << teacher_p95(errors.kl)
              << '/' << *std::max_element(errors.kl.begin(), errors.kl.end()) << "; selected gap "
              << errors.mean_gap() << '/' << teacher_p95(errors.gap) << '/'
              << *std::max_element(errors.gap.begin(), errors.gap.end()) << "; top1 "
              << errors.kl.size() - errors.mismatches << '/' << errors.kl.size() << '\n';
    require(errors.mean_kl() <= criterion.mean_kl && teacher_p95(errors.kl) <= criterion.p95_kl &&
                *std::max_element(errors.kl.begin(), errors.kl.end()) <= criterion.maximum_kl,
            label + ": conditional KL exceeds its numerical envelope");
    require(errors.mean_gap() <= criterion.mean_gap &&
                teacher_p95(errors.gap) <= criterion.p95_gap &&
                *std::max_element(errors.gap.begin(), errors.gap.end()) <= criterion.maximum_gap,
            label + ": selected-token logprob gap exceeds its numerical envelope");
    require(errors.mismatches <= criterion.mismatches, label + ": conditional top-1 agreement");
}

fn::Lane teacher_prefill(fn::Program& program, const Tokens& prompt, ninfer::TokenId& first) {
    const auto reservation = program.open_lane(static_cast<std::int64_t>(prompt.size()) +
                                                   static_cast<std::int64_t>(kTeacherPositions) - 1,
                                               fn::LaneSampling{});
    if (!reservation.has_value()) { throw std::runtime_error("teacher lane reservation failed"); }
    const fn::Lane lane = reservation.value();
    std::span<const ninfer::TokenId> remaining(prompt);
    while (!remaining.empty()) {
        const auto step = program.prefill_step(lane, remaining);
        remaining       = remaining.subspan(static_cast<std::size_t>(step.consumed));
        if (step.complete) { first = step.next_token; }
    }
    return lane;
}

// Replaying the Engine's published history, rather than another free-running answer, checks
// request-to-lane publication and state routing after prefill without amplifying a near-tie.
void require_engine_batch_replay(fn::Program& program, const EngineBatch& observations) {
    const auto& criterion        = kTeacherCriteria[0];
    double total_gap             = 0.0;
    std::size_t total_mismatches = 0;
    std::size_t total_positions  = 0;
    for (std::size_t row = 0; row < observations.size(); ++row) {
        const auto& observation = observations[row];
        require(observation.tokens.size() == kShortOutputs &&
                    observation.selected_logprobs.size() == observation.tokens.size(),
                "Engine replay lost its token-aligned observations");
        ninfer::TokenId ignored_first = -1;
        const auto lane               = teacher_prefill(program, observation.prompt, ignored_first);
        double gap_sum                = 0.0;
        std::size_t mismatches        = 0;
        for (std::size_t position = 0; position < observation.tokens.size(); ++position) {
            if (position != 0) {
                const std::array lanes{lane};
                const std::array anchors{observation.tokens[position - 1]};
                std::array<ninfer::TokenId, 1> ignored{};
                program.decode(lanes, anchors, ignored);
            }
            const auto reference = logits(program, lane);
            const auto summary   = summarize_logits(reference);
            const auto token     = observation.tokens[position];
            require(token >= 0 && token < fn::kTokenDomain, "Engine replay token domain");
            const auto selected           = static_cast<std::size_t>(token);
            const double expected_logprob = bf16(reference[selected]) - summary.log_normalizer;
            const double gap = std::abs(expected_logprob - observation.selected_logprobs[position]);
            require(std::isfinite(gap), "nonfinite Engine replay logprob");
            gap_sum += gap;
            mismatches += summary.top1 == selected ? 0U : 1U;
        }
        program.close_lane(lane);
        const double mean_gap = gap_sum / static_cast<double>(observation.tokens.size());
        std::cout << "Engine teacher replay lane " << row << ": mean selected gap " << mean_gap
                  << "; top1 " << observation.tokens.size() - mismatches << '/'
                  << observation.tokens.size() << '\n';
        require(mean_gap <= criterion.lane_mean_gap && mismatches <= criterion.lane_mismatches,
                "Engine lane " + std::to_string(row) + " failed independent-history replay");
        total_gap += gap_sum;
        total_mismatches += mismatches;
        total_positions += observation.tokens.size();
    }
    require(total_gap / static_cast<double>(total_positions) <= criterion.mean_gap &&
                total_mismatches <= criterion.mismatches,
            "Engine batch failed independent-history replay");
}

void cross_batch_teacher_forcing(const Inputs& inputs, const EngineBatch& observations) {
    ninfer::DeviceContext device(0);
    const ninfer::artifact::Reader reader(inputs.artifact);
    const fn::LoadedModel model(reader, device);
    fn::ProgramOptions options;
    options.ngram_table        = inputs.ngram_table;
    options.context_tokens     = kEngineContext;
    options.kv_capacity_tokens = kEngineKvTokens;
    options.lanes              = static_cast<std::int32_t>(kLanes);
    fn::Program program(model, options, device);
    std::array<std::array<Tokens, kLanes>, 2> panels;
    for (std::uint32_t row = 0; row < kLanes; ++row) {
        panels[0][row] = prompt_tokens(kShortPrompt + 17 * row, 100 + row);
    }
    auto frontend = fn::Package::make_frontend(model);
    constexpr std::array<const char*, kLanes> text{
        "Explain how a hash table resolves collisions. Compare separate chaining with open "
        "addressing, and give a small concrete example of each.",
        "Write a C++ function that merges two sorted integer arrays. Explain the loop invariant "
        "and the time and space complexity.",
        "Describe how the water cycle connects evaporation, condensation, precipitation and "
        "groundwater. Explain what changes during a drought.",
        "Plan a three-day visit to a coastal town using public transport. Include indoor "
        "alternatives for rainy weather and explain how to keep costs low."};
    for (std::size_t row = 0; row < kLanes; ++row) {
        ninfer::PromptInput input;
        input.options.enable_thinking = false;
        input.messages = {ninfer::ChatMessage{.role  = ninfer::ChatRole::User,
                                              .parts = {ninfer::MessagePart{.text = text[row]}}}};
        const auto prepared = frontend.prepare(std::move(input));
        panels[1][row] = ninfer::targets::qwen3_6::PreparedPromptAccess::view(prepared).token_ids;
    }

    struct SoloTrace {
        std::array<ninfer::TokenId, kTeacherPositions> tokens{};
        std::vector<std::uint16_t> logits;
    };

    const auto vocab = static_cast<std::size_t>(fn::kTokenDomain);
    for (std::size_t panel_index = 0; panel_index < panels.size(); ++panel_index) {
        const auto& prompts = panels[panel_index];
        std::array<SoloTrace, kLanes> solo;
        for (std::size_t row = 0; row < kLanes; ++row) {
            auto& trace = solo[row];
            trace.logits.resize(kTeacherPositions * vocab);
            const auto lane    = teacher_prefill(program, prompts[row], trace.tokens[0]);
            const auto capture = [&](std::size_t position) {
                CUDA_CHECK(cudaMemcpy(trace.logits.data() + position * vocab,
                                      program.state().tail_logits(lane).data,
                                      vocab * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
            };
            capture(0);
            for (std::size_t position = 1; position < kTeacherPositions; ++position) {
                const std::array lanes{lane};
                const std::array anchors{trace.tokens[position - 1]};
                std::array<ninfer::TokenId, 1> next{};
                program.decode(lanes, anchors, next);
                trace.tokens[position] = next[0];
                capture(position);
            }
            program.close_lane(lane);
        }
        std::array<fn::Lane, kLanes> lanes{};
        for (std::size_t row = 0; row < kLanes; ++row) {
            ninfer::TokenId first = -1;
            lanes[row]            = teacher_prefill(program, prompts[row], first);
            require(first == solo[row].tokens[0], "teacher-forced first tokens differ");
        }
        TeacherErrors total;
        std::array<TeacherErrors, kLanes> per_lane;
        for (std::size_t position = 0; position < kTeacherPositions; ++position) {
            if (position != 0) {
                std::array<ninfer::TokenId, kLanes> anchors{}, ignored{};
                // Every lane stays busy on its own B1 history. B4 choices never feed back.
                // Greedy sampling has no penalties, so its token counters cannot affect logits.
                for (std::size_t row = 0; row < kLanes; ++row) {
                    anchors[row] = solo[row].tokens[position - 1];
                }
                program.decode(lanes, anchors, ignored);
            }
            for (std::size_t row = 0; row < kLanes; ++row) {
                const auto batch     = logits(program, lanes[row]);
                const auto reference = std::span(solo[row].logits).subspan(position * vocab, vocab);
                const auto difference = distribution_difference(batch, reference);
                require(solo[row].tokens[position] >= 0 &&
                            solo[row].tokens[position] < fn::kTokenDomain,
                        "teacher-forced selected token is outside the vocabulary");
                const auto selected = static_cast<std::size_t>(solo[row].tokens[position]);
                const double gap =
                    std::abs(static_cast<double>(bf16(reference[selected])) -
                             bf16(batch[selected]) - difference.log_normalizer_delta);
                total.add(difference, gap);
                per_lane[row].add(difference, gap);
            }
        }
        const auto& criterion   = kTeacherCriteria[panel_index];
        const std::string label = "teacher-forced B1/B4 panel " + std::to_string(panel_index);
        require_teacher_errors(total, criterion, label);
        for (std::size_t row = 0; row < kLanes; ++row) {
            const auto& errors = per_lane[row];
            require(errors.mean_kl() <= criterion.lane_mean_kl &&
                        errors.mean_gap() <= criterion.lane_mean_gap &&
                        errors.mismatches <= criterion.lane_mismatches,
                    label + ": lane " + std::to_string(row) + " exceeds its numerical envelope");
            program.close_lane(lanes[row]);
        }
    }
    require_engine_batch_replay(program, observations);
}
