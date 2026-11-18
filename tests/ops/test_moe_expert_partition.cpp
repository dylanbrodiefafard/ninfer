// moe_expert_partition and moe_expert_partial_join against exact oracles written from their
// contracts (ninfer/ops/moe_expert_partition.h).
//
// The partition is an exact selection and copy: compact_map, the count, every compact column below
// N and the untouched columns at and above N (poisoned beforehand) are compared bit for bit. The
// join is one FP32 addition followed by one BF16 round-to-nearest-even; the oracle performs that
// host FP32 addition and rounds it with the test's own RNE, so the comparison is exact. Its inputs
// include round-half ties in both directions and signed zeros.
#include "core/device.h"
#include "ninfer/ops/moe_expert_partition.h"
#include "ninfer/ops/moe_expert_record.h"
#include "ops/op_tester.h"
#include "ops/sanitizer_scope.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kHidden          = ops::kMoeHidden;
constexpr int kRoutes          = ops::kMoeHelperRoutesPerToken;
constexpr int kExperts         = ops::kMoeHelperExperts;
constexpr int kCapacity        = ops::kMoeHelperMaxTokens;
constexpr std::uint8_t kPoison = 0x5a;

// Share of local experts: every expert local (N = 0), none, half, or nine in ten.
enum class Locality : std::uint8_t { All, None, Half, Most };

const char* name(Locality locality) {
    switch (locality) {
    case Locality::All:
        return "all-local";
    case Locality::None:
        return "none-local";
    case Locality::Half:
        return "half-local";
    case Locality::Most:
        return "most-local";
    }
    return "?";
}

std::vector<bool> make_locality(Locality locality, std::mt19937& rng) {
    const double share = locality == Locality::All    ? 1.0
                         : locality == Locality::None ? 0.0
                         : locality == Locality::Half ? 0.5
                                                      : 0.9;
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::vector<bool> local(kExperts);
    for (std::size_t e = 0; e < local.size(); ++e) { local[e] = unit(rng) < share; }
    return local;
}

std::vector<std::int32_t> locality_words(const std::vector<bool>& local) {
    std::vector<std::uint32_t> words(ops::kMoeExpertMaskWords, 0);
    for (std::size_t e = 0; e < local.size(); ++e) {
        if (local[e]) { words[e / 32] |= 1U << (e % 32); }
    }
    std::vector<std::int32_t> bits(words.size());
    std::memcpy(bits.data(), words.data(), words.size() * sizeof(std::uint32_t));
    return bits;
}

struct RoundInputs {
    std::vector<std::uint16_t> x;  // BF16 bits [T][2560]
    std::vector<std::int32_t> ids; // [T][10], ten distinct ids per token
    std::vector<float> weights;    // [T][10]
};

RoundInputs make_round(int tokens, std::mt19937& rng) {
    const auto extent = static_cast<std::size_t>(tokens);
    RoundInputs in{std::vector<std::uint16_t>(extent * kHidden),
                   std::vector<std::int32_t>(extent * kRoutes),
                   std::vector<float>(extent * kRoutes)};
    std::uniform_int_distribution<int> bits(0, 0xffff);
    for (auto& value : in.x) { value = static_cast<std::uint16_t>(bits(rng)); }
    std::uniform_int_distribution<int> expert(0, kExperts - 1);
    std::uniform_real_distribution<float> weight(0.0F, 1.0F);
    for (std::size_t t = 0; t < extent; ++t) {
        for (int j = 0; j < kRoutes; ++j) {
            std::int32_t id = 0;
            bool repeated   = true;
            while (repeated) {
                id       = expert(rng);
                repeated = false;
                for (int i = 0; i < j; ++i) { repeated |= in.ids[t * kRoutes + i] == id; }
            }
            in.ids[t * kRoutes + j]     = id;
            in.weights[t * kRoutes + j] = weight(rng);
        }
    }
    return in;
}

// The partition contract evaluated on the host. Compact columns at and above N keep kPoison.
struct Partition {
    std::vector<std::int32_t> map;
    std::int32_t count = 0;
    std::vector<std::uint8_t> x, ids, weights; // raw bytes of [C][2560] BF16, [C][10] I32 and FP32
};

Partition oracle_partition(const RoundInputs& in, int tokens, const std::vector<bool>& local) {
    Partition out{std::vector<std::int32_t>(static_cast<std::size_t>(tokens), -1), 0,
                  std::vector<std::uint8_t>(std::size_t{kCapacity} * kHidden * 2, kPoison),
                  std::vector<std::uint8_t>(std::size_t{kCapacity} * kRoutes * 4, kPoison),
                  std::vector<std::uint8_t>(std::size_t{kCapacity} * kRoutes * 4, kPoison)};
    for (int t = 0; t < tokens; ++t) {
        const auto row = static_cast<std::size_t>(t) * kRoutes;
        bool helper    = false;
        for (int j = 0; j < kRoutes; ++j) {
            helper |= !local[static_cast<std::size_t>(in.ids[row + j])];
        }
        if (!helper) { continue; }
        const auto r                         = static_cast<std::size_t>(out.count++);
        out.map[static_cast<std::size_t>(t)] = static_cast<std::int32_t>(r);
        std::memcpy(out.x.data() + r * kHidden * 2,
                    in.x.data() + static_cast<std::size_t>(t) * kHidden, kHidden * 2);
        for (int j = 0; j < kRoutes; ++j) {
            const std::int32_t id  = in.ids[row + j];
            const bool remote      = !local[static_cast<std::size_t>(id)];
            const std::int32_t cid = remote ? id : ops::kMoeHelperInactiveExpert;
            const float cweight    = remote ? in.weights[row + j] : 0.0F;
            std::memcpy(out.ids.data() + (r * kRoutes + j) * 4, &cid, 4);
            std::memcpy(out.weights.data() + (r * kRoutes + j) * 4, &cweight, 4);
        }
    }
    return out;
}

// Device buffers of one round: inputs, the bitset and the partition outputs at capacity 48.
class PartitionRun {
public:
    PartitionRun(const RoundInputs& in, int tokens, const std::vector<bool>& local)
        : tokens_(tokens), x_(to_device(in.x)), ids_(to_device(in.ids)),
          weights_(to_device(in.weights)), mask_(to_device(locality_words(local))),
          map_(static_cast<std::size_t>(tokens) * 4), count_(4),
          compact_x_(std::size_t{kCapacity} * kHidden * 2),
          compact_ids_(std::size_t{kCapacity} * kRoutes * 4),
          compact_weights_(std::size_t{kCapacity} * kRoutes * 4) {}

    void upload(const RoundInputs& in, const std::vector<bool>& local) {
        x_.copy_from_host(in.x.data(), x_.bytes);
        ids_.copy_from_host(in.ids.data(), ids_.bytes);
        weights_.copy_from_host(in.weights.data(), weights_.bytes);
        const std::vector<std::int32_t> words = locality_words(local);
        mask_.copy_from_host(words.data(), mask_.bytes);
    }

    void poison() {
        map_.fill(kPoison);
        count_.fill(kPoison);
        compact_x_.fill(kPoison);
        compact_ids_.fill(kPoison);
        compact_weights_.fill(kPoison);
    }

    void invoke(cudaStream_t stream) {
        const Tensor x(x_.p, DType::BF16, {kHidden, tokens_});
        const Tensor ids(ids_.p, DType::I32, {kRoutes, tokens_});
        const Tensor weights(weights_.p, DType::FP32, {kRoutes, tokens_});
        const Tensor mask(mask_.p, DType::I32, {ops::kMoeExpertMaskWords});
        Tensor map(map_.data(), DType::I32, {tokens_});
        Tensor count(count_.data(), DType::I32, {1});
        Tensor cx(compact_x_.data(), DType::BF16, {kHidden, kCapacity});
        Tensor cids(compact_ids_.data(), DType::I32, {kRoutes, kCapacity});
        Tensor cweights(compact_weights_.data(), DType::FP32, {kRoutes, kCapacity});
        ops::moe_expert_partition(x, ids, weights, mask, map, count, cx, cids, cweights, stream);
    }

    int verify(const std::string& label, const Partition& want) const {
        int failures = 0;
        failures += verify_exact((label + " compact_map").c_str(),
                                 from_device<std::int32_t>(map_.data(), want.map.size()), want.map);
        failures += verify_exact((label + " compact_count").c_str(),
                                 from_device<std::int32_t>(count_.data(), 1),
                                 std::vector<std::int32_t>{want.count});
        failures +=
            verify_exact((label + " compact_x").c_str(),
                         from_device<std::uint8_t>(compact_x_.data(), want.x.size()), want.x);
        failures +=
            verify_exact((label + " compact_ids").c_str(),
                         from_device<std::uint8_t>(compact_ids_.data(), want.ids.size()), want.ids);
        failures += verify_exact(
            (label + " compact_weights").c_str(),
            from_device<std::uint8_t>(compact_weights_.data(), want.weights.size()), want.weights);
        for (const GuardedDeviceBuffer* buffer :
             {&map_, &count_, &compact_x_, &compact_ids_, &compact_weights_}) {
            failures += buffer->verify_guards(label);
        }
        return failures;
    }

private:
    int tokens_;
    DeviceBuffer x_, ids_, weights_, mask_;
    GuardedDeviceBuffer map_, count_, compact_x_, compact_ids_, compact_weights_;
};

int partition_case(int tokens, Locality locality, std::uint32_t seed, cudaStream_t stream) {
    std::mt19937 rng(seed);
    const std::string label =
        "moe_expert_partition T=" + std::to_string(tokens) + " " + name(locality);
    const RoundInputs in          = make_round(tokens, rng);
    const std::vector<bool> local = make_locality(locality, rng);
    PartitionRun run(in, tokens, local);
    run.poison();
    run.invoke(stream);
    cuda_synchronize(stream);
    return run.verify(label, oracle_partition(in, tokens, local));
}

// One captured partition replayed with new inputs and a new bitset uploaded before each replay.
int partition_graph_case(int tokens, std::uint32_t seed, cudaStream_t stream) {
    std::mt19937 rng(seed);
    const std::string label = "moe_expert_partition graph T=" + std::to_string(tokens);
    RoundInputs in          = make_round(tokens, rng);
    std::vector<bool> local = make_locality(Locality::Half, rng);
    PartitionRun run(in, tokens, local);
    cudaGraph_t graph = nullptr;
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
    run.invoke(stream);
    cuda_check(cudaStreamEndCapture(stream, &graph), "end capture");
    cudaGraphExec_t exec           = nullptr;
    const cudaError_t instantiated = cudaGraphInstantiate(&exec, graph, 0);
    cudaGraphDestroy(graph);
    cuda_check(instantiated, "graph instantiate");
    int failures = 0;
    for (const Locality next : {Locality::Half, Locality::Most, Locality::All, Locality::None}) {
        in    = make_round(tokens, rng);
        local = make_locality(next, rng);
        run.upload(in, local);
        run.poison();
        cuda_check(cudaGraphLaunch(exec, stream), "graph launch");
        cuda_synchronize(stream);
        failures += run.verify(label + " " + name(next), oracle_partition(in, tokens, local));
    }
    cudaGraphExecDestroy(exec);
    return failures;
}

float from_bits(std::uint32_t bits) {
    float value = 0.0F;
    std::memcpy(&value, &bits, 4);
    return value;
}

std::uint32_t to_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, 4);
    return bits;
}

// The join contract on the host: one FP32 addition, then BF16 round-to-nearest-even.
std::vector<std::uint16_t> oracle_join(const std::vector<float>& partial,
                                       const std::vector<std::int32_t>& map,
                                       const std::vector<float>& helper) {
    std::vector<std::uint16_t> y(partial.size());
    for (std::size_t t = 0; t < map.size(); ++t) {
        for (std::size_t c = 0; c < kHidden; ++c) {
            float sum = partial[t * kHidden + c];
            if (map[t] >= 0) { sum = sum + helper[static_cast<std::size_t>(map[t]) * kHidden + c]; }
            y[t * kHidden + c] = f32_to_bf16(sum);
        }
    }
    return y;
}

// Join inputs: partial values on the BF16 grid plus random low bits; for a quarter of the
// elements of participating tokens, a BF16-grid partial and a helper of exactly half its BF16 ulp
// (a tie, signs mixed); for non-participating tokens, ties in the partial itself; and signed
// zeros. Helper columns no token maps to hold NaN. compact_map assigns the participating tokens
// distinct columns in a shuffled order, which the contract allows.
int join_case(int tokens, double participating, std::uint32_t seed, cudaStream_t stream) {
    std::mt19937 rng(seed);
    const std::string label = "moe_expert_partial_join T=" + std::to_string(tokens) +
                              " participating=" + std::to_string(participating);
    const auto extent       = static_cast<std::size_t>(tokens);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::vector<std::int32_t> map(extent, -1);
    std::vector<std::int32_t> columns;
    for (std::size_t t = 0; t < extent; ++t) {
        if (unit(rng) < participating) { columns.push_back(static_cast<std::int32_t>(t)); }
    }
    std::vector<std::int32_t> order = columns;
    std::shuffle(order.begin(), order.end(), rng);
    for (std::size_t i = 0; i < order.size(); ++i) {
        map[static_cast<std::size_t>(order[i])] = static_cast<std::int32_t>(i);
    }
    // Unused columns at and above N are NaN; a read of one shows in the output.
    std::vector<float> helper(std::size_t{kCapacity} * kHidden,
                              std::numeric_limits<float>::quiet_NaN());
    std::vector<float> partial(extent * kHidden);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::uniform_int_distribution<std::uint32_t> low(0, 0xffff);
    for (std::size_t t = 0; t < extent; ++t) {
        const std::int32_t column = map[t];
        for (std::size_t c = 0; c < kHidden; ++c) {
            const float base  = bf16_to_f32(f32_to_bf16(normal(rng)));
            const double roll = unit(rng);
            float p           = from_bits(to_bits(base) | low(rng));
            float h           = normal(rng);
            if (roll < 0.25) {
                // A tie: half a BF16 ulp of base, exact in FP32 (2^-8 of base's binade).
                const int exponent = std::ilogb(base);
                const float half   = std::ldexp(1.0F, exponent - 8);
                p                  = base;
                h                  = (low(rng) & 1U) != 0 ? half : -half;
                if (column < 0) { p = from_bits(to_bits(base) | 0x8000U); }
            } else if (roll < 0.27) {
                p = (low(rng) & 1U) != 0 ? -0.0F : 0.0F;
                h = (low(rng) & 1U) != 0 ? -0.0F : 0.0F;
            }
            partial[t * kHidden + c] = p;
            if (column >= 0) { helper[static_cast<std::size_t>(column) * kHidden + c] = h; }
        }
    }
    const DeviceBuffer partial_device = to_device(partial);
    const DeviceBuffer map_device     = to_device(map);
    const DeviceBuffer helper_device  = to_device(helper);
    GuardedDeviceBuffer y(extent * kHidden * 2);
    y.fill(0xff);
    const Tensor partial_view(partial_device.p, DType::FP32, {kHidden, tokens});
    const Tensor map_view(map_device.p, DType::I32, {tokens});
    const Tensor helper_view(helper_device.p, DType::FP32, {kHidden, kCapacity});
    Tensor y_view(y.data(), DType::BF16, {kHidden, tokens});
    ops::moe_expert_partial_join(partial_view, map_view, helper_view, y_view, stream);
    cuda_synchronize(stream);
    int failures =
        verify_exact(label.c_str(), from_device<std::uint16_t>(y.data(), extent * kHidden),
                     oracle_join(partial, map, helper));
    failures += y.verify_guards(label + " y");
    return failures;
}

// Validation rejects these calls before any device access, so the views only need valid
// addresses.
int invalid_argument_cases() {
    DeviceBuffer buffer(std::size_t{kCapacity + 1} * kHidden * 4);
    DeviceBuffer partial(std::size_t{kCapacity + 1} * kHidden * 4);
    DeviceBuffer map(std::size_t{kCapacity + 1} * 4);
    int failures = 0;
    failures += expect_invalid_argument(
        [&] {
            Tensor y(buffer.p, DType::BF16, {kHidden, 4});
            ops::moe_expert_partial_join(Tensor(partial.p, DType::FP32, {kHidden, 4}),
                                         Tensor(map.p, DType::I32, {4}),
                                         Tensor(buffer.p, DType::FP32, {kHidden, 4}), y, nullptr);
        },
        "moe_expert_partial_join accepted y aliasing helper");
    failures += expect_invalid_argument(
        [&] {
            Tensor y(buffer.p, DType::BF16, {kHidden, 4});
            ops::moe_expert_partial_join(Tensor(partial.p, DType::FP32, {kHidden, 4}),
                                         Tensor(map.p, DType::I32, {4}),
                                         Tensor(partial.p, DType::FP32, {kHidden, 3}), y, nullptr);
        },
        "moe_expert_partial_join accepted a helper narrower than T");
    failures += expect_invalid_argument(
        [&] {
            Tensor y(buffer.p, DType::BF16, {kHidden, kCapacity + 1});
            ops::moe_expert_partial_join(Tensor(partial.p, DType::FP32, {kHidden, kCapacity + 1}),
                                         Tensor(map.p, DType::I32, {kCapacity + 1}),
                                         Tensor(partial.p, DType::FP32, {kHidden, kCapacity + 1}),
                                         y, nullptr);
        },
        "moe_expert_partial_join accepted T=49");
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    if (require_cuda() != 0) { return 1; }
    try {
        DeviceContext context;
        cudaStream_t stream = context.stream;
        int failures        = 0;
        if (sanitizer_scope(argc, argv)) {
            failures += partition_case(48, Locality::Half, 101U, stream);
            failures += partition_case(5, Locality::All, 103U, stream);
            failures += join_case(48, 0.6, 105U, stream);
        } else {
            failures += invalid_argument_cases();
            std::uint32_t seed = 11;
            for (const int tokens : {1, 2, 4, 12, 20, 31, 32, 33, 47, 48}) {
                for (const Locality locality :
                     {Locality::All, Locality::None, Locality::Half, Locality::Most}) {
                    failures += partition_case(tokens, locality, seed++, stream);
                }
            }
            failures += partition_graph_case(20, seed++, stream);
            failures += partition_graph_case(48, seed++, stream);
            for (const int tokens : {1, 4, 20, 48}) {
                for (const double participating : {0.0, 0.5, 1.0}) {
                    failures += join_case(tokens, participating, seed++, stream);
                }
            }
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " moe_expert_partition\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "moe_expert_partition: " << error.what() << '\n';
        return 1;
    }
}
