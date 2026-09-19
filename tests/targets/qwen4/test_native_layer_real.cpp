#include "artifact/materializer.h"
#include "artifact/typed_binding.h"
#include "ninfer/ops/qwen4_sparse_moe.h"
#include "ninfer/ops/ple.h"
#include "ops/op_tester.h"
#include "ops/ple_nvfp4_oracle.h"
#include "ops/launcher/qwen4_sparse_moe.h"
#include "targets/qwen4/native_sequence_components.h"
#include "targets/qwen4/native_component_timing.h"
#include <cuda_profiler_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::test;
constexpr int H = 2560, I = 640, E = 512, R = 10;
// Identical to the complete Qwen4 MoE represented-input implementation criterion.
constexpr ReductionCriterion output_criterion{2.5 / 255.0, 1.0 / 32768.0, 2.0 / 255.0};
// Predeclared calibrated shared-A8 profile: existing Linear A8 norm/gross allowance,
// unchanged MoE absolute floor. This is not a PPL budget or a routing relaxation.
constexpr ReductionCriterion shared_a8_criterion{.04, 1.0 / 32768.0, .06};
constexpr PointwiseCriterion route_criterion{128.0 * std::numeric_limits<float>::epsilon(),
                                            128.0 * std::numeric_limits<float>::epsilon()};

float word(std::span<const std::byte> bytes, std::size_t offset) {
    float value;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

std::vector<double> bf16_values(const artifact::Reader& reader, const std::string& name) {
    const auto bytes = reader.payload(name).data;
    std::vector<double> values(bytes.size() / 2);
    for (std::size_t i = 0; i < values.size(); ++i) {
        const auto lo = std::to_integer<unsigned>(bytes[2 * i]);
        const auto hi = std::to_integer<unsigned>(bytes[2 * i + 1]);
        values[i] = bf16_to_f32(static_cast<std::uint16_t>(lo | (hi << 8)));
    }
    return values;
}

// Independent mathematical decoder, no production codecs or converted oracle arrays.
double fp8(unsigned code) {
    const unsigned exponent = (code >> 3) & 15, mantissa = code & 7;
    if (exponent == 15 && mantissa == 7) { throw std::runtime_error("nonfinite source scale"); }
    const double magnitude = exponent == 0 ? std::ldexp(double(mantissa), -9)
        : std::ldexp(1.0 + double(mantissa) / 8.0, int(exponent) - 7);
    return (code & 128) ? -magnitude : magnitude;
}

struct Bank {
    std::span<const std::byte> bytes;
    int n, k;
    std::size_t codes() const { return std::size_t(E) * n * k / 2; }
    std::size_t scales() const { return std::size_t(E) * n * k / 16; }
    void validate() const {
        if (bytes.size() != codes() + scales() + 8 * E) {
            throw std::runtime_error("incorrect native expert payload extent");
        }
        for (int i = 0; i < 2 * E; ++i) {
            const auto scale = word(bytes, codes() + scales() + 4 * i);
            if (!(scale > 0) || !std::isfinite(scale)) {
                throw std::runtime_error("nonpositive/nonfinite native expert multiplier");
            }
        }
    }
    double at(int expert, int row, int column) const {
        const std::size_t r = std::size_t(expert) * n + row;
        const unsigned packed = std::to_integer<unsigned>(bytes[r * k / 2 + column / 2]);
        const unsigned code = (packed >> (4 * (column % 2))) & 15;
        constexpr double values[]{0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6};
        const std::size_t group = column / 16;
        const std::size_t scale_index = ((((r / 128) * (k / 64) + group / 4) * 32
            + r % 32) * 4 + (r % 128) / 32) * 4 + group % 4;
        return values[code] * fp8(std::to_integer<unsigned>(bytes[codes() + scale_index]))
            * word(bytes, codes() + scales() + 4 * expert);
    }
    std::vector<double> project(int expert, std::span<const double> input) const {
        std::vector<double> out(n);
        for (int row = 0; row < n; ++row) {
            for (int column = 0; column < k; ++column) {
                out[row] += at(expert, row, column) * input[column];
            }
        }
        return out;
    }
};

std::vector<double> project(std::span<const double> matrix, std::span<const double> input) {
    const int k = input.size(), n = matrix.size() / k;
    std::vector<double> output(n);
    for (int row = 0; row < n; ++row) {
        for (int column = 0; column < k; ++column) {
            output[row] += matrix[std::size_t(row) * k + column] * input[column];
        }
    }
    return output;
}

void swiglu(std::vector<double>& gate, std::span<const double> up) {
    for (std::size_t i = 0; i < gate.size(); ++i) {
        gate[i] = gate[i] / (1 + std::exp(-gate[i])) * up[i];
    }
}

int real_nvfp4_ple(const std::string& path) {
    artifact::Reader reader(path);
    if (reader.identity() != artifact::ArtifactIdentity{"qwen4/native-ple-qualification", "primitive-nvfp4-source-rows"}) {
        throw std::runtime_error("not a native NVFP4 PLE row fixture");
    }
    artifact::Binder binder(reader);
    const auto handle = artifact::bind_tensor(binder, "ple.rows", artifact::NumericFormat::NVFP4_PARTITION_F32M,
        {3, 4, 160}, artifact::TensorPlacement::ResidentHost);
    DeviceContext device(0);
    auto resident = artifact::materialize(reader, binder.finish(), device);
    const auto bytes = resident.mapped_tensor_bytes(handle);
    if (bytes.size() != 1092 || resident.stats().resident_tensor_bytes != 1092 ||
        resident.stats().resident_locked_bytes < 1092) {
        throw std::runtime_error("NVFP4 PLE sample not fully RAM-locked");
    }
    const auto* source = reinterpret_cast<const std::uint8_t*>(bytes.data());
    const ops::PleResidentNvfp4Table table{source, 3, 4, bytes.size()};
    int failures = 0;
    for (int width : {1, 3, 17}) {
        std::vector<std::int32_t> ids(width * 16);
        for (int i = 0; i < width * 16; ++i) { ids[i] = (i * 7 + i / 16) % 12; }
        const int count = width * 2560, packed_bytes = width * 16 * 94;
        PinnedHostBuffer pinned(packed_bytes);
        GuardedDeviceBuffer packed(packed_bytes), decoded(count * 2);
        Tensor encoded(packed.data(), DType::U8, {94, 16, width});
        Tensor output(decoded.data(), DType::BF16, {160, 16, width});
        ops::ple_nvfp4_stage_rows_batch(table, ids, width, pinned.data(), pinned.size(), encoded, nullptr);
        ops::ple_nvfp4_decode_rows(encoded, output, nullptr);
        cuda_synchronize();
        std::vector<std::uint8_t> expected_bytes(packed_bytes);
        std::vector<std::uint16_t> expected(count);
        for (int i = 0; i < width * 16; ++i) {
            auto* record = expected_bytes.data() + i * 94;
            for (int b = 0; b < 90; ++b) { record[b] = source[ids[i] * 90 + b]; }
            for (int b = 0; b < 4; ++b) { record[90 + b] = source[1080 + (ids[i] / 4) * 4 + b]; }
            for (int f = 0; f < 160; ++f) { expected[i * 160 + f] = ple_nvfp4_oracle(record, f); }
        }
        failures += verify_exact("native NVFP4 PLE exact gathered payload",
            from_device<std::uint8_t>(packed.data(), packed_bytes), expected_bytes);
        failures += verify_exact("native NVFP4 PLE independent exact decode",
            from_device<std::uint16_t>(decoded.data(), count), expected);
        failures += packed.verify_guards("native NVFP4 PLE records");
        failures += decoded.verify_guards("native NVFP4 PLE output");
    }
    return failures;
}

int real_ple(const std::string& path) {
    artifact::Reader reader(path);
    if (reader.identity() != artifact::ArtifactIdentity{"qwen4/native-ple-qualification", "nvidia-fp8-source-rows"}) {
        throw std::runtime_error("not a native selected-row PLE fixture");
    }
    artifact::Binder binder(reader);
    const auto rows = artifact::bind_tensor(binder, "ple.rows", artifact::NumericFormat::FP8_E4M3FN_TENSOR_BF16S,
        {16, 160}, artifact::TensorPlacement::ResidentHost);
    DeviceContext device(0);
    auto resident = artifact::materialize(reader, binder.finish(), device);
    const auto bytes = resident.mapped_tensor_bytes(rows);
    const std::uint16_t scale = std::to_integer<unsigned>(bytes[2560]) |
        (std::to_integer<unsigned>(bytes[2561]) << 8);
    if (scale != 0x3951) { throw std::runtime_error("unexpected pinned PLE source scale"); }
    if (resident.stats().resident_tensor_bytes != 2562 || resident.stats().resident_locked_bytes < 2562) {
        throw std::runtime_error("selected PLE sample was not eagerly locked");
    }
    const ops::PleResidentFp8Table table{reinterpret_cast<const std::uint8_t*>(bytes.data()), 16, 2560};
    int failures = 0;
    for (int width : {1, 3}) {
        const int count = width * 2560;
        std::vector<std::int32_t> ids(width * 16);
        for (int i = 0; i < width * 16; ++i) { ids[i] = (i * 7 + i / 16) % 16; }
        PinnedHostBuffer pinned(count);
        GuardedDeviceBuffer packed(count), decoded(count * 2);
        Tensor encoded(packed.data(), DType::U8, {160, 16, width});
        Tensor output(decoded.data(), DType::BF16, {160, 16, width});
        ops::ple_fp8_stage_rows_batch(table, ids, width, pinned.data(), pinned.size(), encoded, nullptr);
        ops::ple_fp8_decode_rows(encoded, scale, output, nullptr);
        cuda_synchronize();
        std::vector<std::uint8_t> expected_bytes(count);
        std::vector<std::uint16_t> expected(count);
        for (int row = 0; row < width * 16; ++row) {
            for (int d = 0; d < 160; ++d) {
                const unsigned code = std::to_integer<unsigned>(bytes[ids[row] * 160 + d]);
                expected_bytes[row * 160 + d] = code;
                expected[row * 160 + d] = f32_to_bf16(static_cast<float>(fp8(code) * bf16_to_f32(scale)));
            }
        }
        failures += verify_exact("native selected PLE row bytes", from_device<std::uint8_t>(packed.data(), count), expected_bytes);
        failures += verify_exact("native selected PLE BF16 decode", from_device<std::uint16_t>(decoded.data(), count), expected);
        failures += packed.verify_guards("native PLE packed");
        failures += decoded.verify_guards("native PLE decoded");
    }
    std::cout << "Native PLE selected rows only; not complete table residency admission\n";
    return failures;
}

int run(const std::string& path, int layer,
        const qwen4_sequence::Result* sequence_input = nullptr,
        qwen4_sequence::Result* sequence_output = nullptr, bool partitioned = false,
        bool allow_a4 = false, bool calibrated_shared = false,
        ops::Qwen4SharedExpertPolicy shared_policy = {}, const std::string& source_root = {},
        int weight_mask = 7, const std::string& shared_up_nvfp4 = {}) {
    artifact::Reader reader(path);
    if (reader.identity() != artifact::ArtifactIdentity{"qwen4/native-layer-qualification", "nvidia-nvfp4-source"}) {
        throw std::runtime_error("not a native Qwen4 layer fixture");
    }
    const std::string prefix = "model.language_model.layers." + std::to_string(layer) + ".mlp.";
    artifact::Binder binder(reader);
    std::map<std::string, artifact::ObjectHandle> handles;
    for (const auto& object : reader.objects()) {
        const auto* d = std::get_if<artifact::TensorDescriptor>(&object);
        if (!d) { throw std::runtime_error("unexpected native layer resource"); }
        if (!d->name.starts_with("model.language_model.layers." + std::to_string(layer) + ".")) {
            throw std::runtime_error("wrong native source layer");
        }
        if (d->name.starts_with(prefix + "experts.")) {
            const bool is_down = d->name == prefix + "experts.down_proj.weight";
            if ((!is_down && d->name != prefix + "experts.gate_proj.weight" &&
                 d->name != prefix + "experts.up_proj.weight") ||
                d->format != artifact::NumericFormat::NVFP4_EXPERT_F32M ||
                d->shape != std::vector<std::uint64_t>{E,
                    static_cast<std::uint64_t>(is_down ? H : I),
                    static_cast<std::uint64_t>(is_down ? I : H)}) {
                throw std::runtime_error("incorrect native expert role");
            }
        } else if (d->format != artifact::NumericFormat::BF16) {
            throw std::runtime_error("non-expert source must remain BF16");
        }
        if (d->name.starts_with(prefix + "shared_expert.")) {
            const bool is_down = d->name == prefix + "shared_expert.down_proj.weight";
            if ((!is_down && d->name != prefix + "shared_expert.gate_proj.weight" &&
                 d->name != prefix + "shared_expert.up_proj.weight") ||
                d->shape != std::vector<std::uint64_t>{static_cast<std::uint64_t>(is_down ? H : I),
                    static_cast<std::uint64_t>(is_down ? I : H)}) {
                throw std::runtime_error("incorrect native shared expert role");
            }
        }
        const auto handle = binder.require_tensor(d->name, d->format, d->layout, d->shape);
        handles.emplace(d->name, handle);
        const bool matrix = d->name.starts_with(prefix + "experts.") ||
                            d->name.starts_with(prefix + "shared_expert.") ||
                            d->name == prefix + "gate.weight" ||
                            d->name == prefix + "shared_expert_gate.weight";
        if (matrix) { binder.materialize_on_device(handle); }
        else { binder.validate_only(handle); }
    }
    auto payload = [&](const char* projection) {
        return reader.payload(prefix + "experts." + projection + ".weight").data;
    };
    const Bank gate{payload("gate_proj"), I, H}, up{payload("up_proj"), I, H}, down{payload("down_proj"), H, I};
    gate.validate(); up.validate(); down.validate();
    const auto router = bf16_values(reader, prefix + "gate.weight");
    const auto selector = bf16_values(reader, prefix + "shared_expert_gate.weight");
    const auto original_sg = bf16_values(reader, prefix + "shared_expert.gate_proj.weight");
    const auto original_su = bf16_values(reader, prefix + "shared_expert.up_proj.weight");
    const auto original_sd = bf16_values(reader, prefix + "shared_expert.down_proj.weight");
    auto sg=original_sg, su=original_su, sd=original_sd;
    if (router.size() != E * H || selector.size() != H || sg.size() != I * H ||
        su.size() != I * H || sd.size() != H * I) { throw std::runtime_error("native MoE shape mismatch"); }
    DeviceContext device(0);
    auto materialized = artifact::materialize(reader, binder.finish(), device);
    const Weight router_weight = artifact::materialized_weight(materialized,
        handles.at(prefix + "gate.weight"), artifact::NumericFormat::BF16, E, H);
    const Tensor selector_tensor = artifact::materialized_tensor(materialized,
        handles.at(prefix + "shared_expert_gate.weight"), artifact::NumericFormat::BF16, {H});
    auto bank_weight = [&](const char* p, int n, int k) {
        return artifact::materialized_nvfp4_expert_weight(materialized,
            handles.at(prefix + "experts." + p + ".weight"), E, n, k);
    };
    auto shared_weight = [&](const char* p, int n, int k) {
        return artifact::materialized_weight(materialized,
            handles.at(prefix + "shared_expert." + p + ".weight"), artifact::NumericFormat::BF16, n, k);
    };
    ops::Qwen4ResidentSparseMoeWeights weights{router_weight,
        bank_weight("gate_proj", I, H), bank_weight("up_proj", I, H), bank_weight("down_proj", H, I),
        selector_tensor, shared_weight("gate_proj", I, H),
        shared_weight("up_proj", I, H), shared_weight("down_proj", H, I)};
    std::unique_ptr<artifact::Reader> calibrated_reader;
    artifact::MaterializedArtifact calibrated_weights;
    if (calibrated_shared && weight_mask) {
        calibrated_reader=std::make_unique<artifact::Reader>(
            ((source_root.empty()?std::filesystem::path(path).parent_path():std::filesystem::path(source_root)) /
                ((layer==1 || layer==2) ? "qwen4-fp8-projection-additional.ninfer" :
                    "qwen4-fp8-projections.ninfer")).string());
        if (calibrated_reader->identity()!=artifact::ArtifactIdentity{
            "qwen4/native-fp8-projection-qualification","senfu-fp8-source"}) {
            throw std::runtime_error("wrong calibrated shared source");
        }
        artifact::Binder source_binder(*calibrated_reader);
        std::map<std::string,artifact::ObjectHandle> source_handles;
        for(const auto& object:calibrated_reader->objects()) {
            const auto* d=std::get_if<artifact::TensorDescriptor>(&object);
            if(!d) { throw std::runtime_error("non-tensor calibrated source"); }
            const auto handle=source_binder.require_tensor(d->name,d->format,d->layout,d->shape);
            const int role_bit=d->name==prefix+"shared_expert.gate_proj.weight"?1:
                d->name==prefix+"shared_expert.up_proj.weight"?2:
                d->name==prefix+"shared_expert.down_proj.weight"?4:0;
            if(role_bit & weight_mask) {
                if(d->format!=artifact::NumericFormat::FP8_E4M3FN_TENSOR_F32M ||
                   d->layout!=artifact::StorageLayout::TensorCalibratedV1) {
                    throw std::runtime_error("wrong calibrated shared format");
                }
                source_handles.emplace(d->name,handle);
                source_binder.materialize_on_device(handle);
            } else { source_binder.validate_only(handle); }
        }
        if(source_handles.size()!=std::size_t(bool(weight_mask&1)+bool(weight_mask&2)+bool(weight_mask&4))) {
            throw std::runtime_error("missing calibrated shared roles");
        }
        calibrated_weights=artifact::materialize(*calibrated_reader,source_binder.finish(),device);
        const auto assign=[&](const char* role,int n,int k,Weight& weight,std::vector<double>& logical) {
            const auto name=prefix+"shared_expert."+role+".weight";
            const auto* d=std::get_if<artifact::TensorDescriptor>(calibrated_reader->find(name));
            if(!d || d->shape!=std::vector<std::uint64_t>{static_cast<unsigned>(n),static_cast<unsigned>(k)}) {
                throw std::runtime_error("wrong calibrated shared shape");
            }
            weight=artifact::materialized_weight(calibrated_weights,source_handles.at(name),d->format,n,k);
            const auto bytes=calibrated_reader->payload(*d).data;
            const double scale=word(bytes,std::size_t(n)*k);
            logical.resize(std::size_t(n)*k);
            for(std::size_t i=0;i<logical.size();++i) { logical[i]=fp8(std::to_integer<unsigned>(bytes[i]))*scale; }
        };
        if(weight_mask&1) assign("gate_proj",I,H,weights.shared_gate_proj,sg);
        if(weight_mask&2) assign("up_proj",I,H,weights.shared_up,su);
        if(weight_mask&4) assign("down_proj",H,I,weights.shared_down,sd);
    }
    std::unique_ptr<artifact::Reader> nvfp4_reader;
    artifact::MaterializedArtifact nvfp4_weight;
    if(!shared_up_nvfp4.empty()) {
        nvfp4_reader=std::make_unique<artifact::Reader>(shared_up_nvfp4);
        const auto identity=nvfp4_reader->identity();
        if(identity.model_id!="qwen4/native-projection-candidate" ||
           (identity.weights_id!="nvfp4_diagonal_calibrated" && identity.weights_id!="nvfp4_maxabs")) {
            throw std::runtime_error("invalid bounded shared-up NVFP4 candidate identity");
        }
        artifact::Binder candidate(*nvfp4_reader);
        const auto handle=candidate.require_tensor("weight",artifact::NumericFormat::NVFP4,
            artifact::StorageLayout::BlockScaleK16M128x4V1,std::array<std::uint64_t,2>{I,H});
        const auto input_scale=candidate.require_tensor("input_scale_divisor",artifact::NumericFormat::FP32,
            artifact::StorageLayout::ContiguousLeV1,{});
        candidate.materialize_on_device(handle);candidate.validate_only(input_scale);
        const auto raw=nvfp4_reader->payload("weight").data;
        constexpr std::size_t code_bytes=std::size_t(I)*H/2,scale_bytes=std::size_t(I)*H/16;
        if(raw.size()!=code_bytes+scale_bytes+4) throw std::runtime_error("invalid shared-up NVFP4 extent");
        const float divisor=word(raw,code_bytes+scale_bytes);
        const float activation_divisor=word(nvfp4_reader->payload("input_scale_divisor").data,0);
        if(!(divisor>0) || !std::isfinite(divisor) || !(activation_divisor>0) || !std::isfinite(activation_divisor))
            throw std::runtime_error("invalid shared-up NVFP4 divisors");
        // Independent logical decoder: signed E2M1 times exact E4M3 scale divided
        // by the stored FP32 matrix divisor; no private staging/rounding casts.
        constexpr double values[]{0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6};
        for(int row=0;row<I;++row) for(int column=0;column<H;++column) {
            const unsigned byte=std::to_integer<unsigned>(raw[std::size_t(row)*H/2+column/2]);
            const unsigned code=(byte>>(4*(column%2)))&15;
            const int group=column/16;
            const auto scale_index=((((row/128)*(H/64)+group/4)*32+row%32)*4+(row%128)/32)*4+group%4;
            const unsigned scale_word=std::to_integer<unsigned>(raw[code_bytes+scale_index]);
            if(scale_word>=127) throw std::runtime_error("invalid shared-up NVFP4 block scale");
            su[std::size_t(row)*H+column]=values[code]*fp8(scale_word)/double(divisor);
        }
        nvfp4_weight=artifact::materialize(*nvfp4_reader,candidate.finish(),device);
        const auto* data=static_cast<const std::byte*>(nvfp4_weight.device_data(handle));
        Weight w{};w.payload=w.qdata=data;w.payload_bytes=raw.size();w.scales=data+code_bytes;
        w.qtype=QType::NVFP4;w.layout=QuantLayout::BlockScaleK16M128x4;w.scale_dtype=DType::FP8_E4M3FN;
        w.group=w.group_size=16;w.ndim=2;w.n=w.shape[0]=w.padded_shape[0]=I;
        w.k=w.shape[1]=w.padded_shape[1]=H;w.weight_scale_divisor=divisor;
        w.input_scale_divisor=activation_divisor;weights.shared_up=w;
    }
    const int sequence_width = sequence_input ? static_cast<int>(sequence_input->actual.size() / H) : 0;
    const int patterns = sequence_input ? 2 * sequence_width : 2;
    const bool identical_inputs=sequence_input && sequence_input->actual==sequence_input->reference;
    std::vector<std::vector<double>> inputs(patterns), references(patterns), probabilities(patterns);
    std::vector<std::vector<double>> original_references(patterns);
    std::vector<std::vector<int>> routes(patterns);
    std::vector<double> cutoff_margins(patterns);
    for (int pattern = 0; pattern < patterns; ++pattern) {
        if(identical_inputs && pattern>=sequence_width) {
            const int earlier=pattern-sequence_width;
            inputs[pattern]=inputs[earlier];references[pattern]=references[earlier];
            probabilities[pattern]=probabilities[earlier];routes[pattern]=routes[earlier];
            cutoff_margins[pattern]=cutoff_margins[earlier];original_references[pattern]=original_references[earlier];
            continue;
        }
        inputs[pattern].resize(H);
        for (int i = 0; i < H; ++i) {
            if (sequence_input) {
                const auto& values = pattern < sequence_width ? sequence_input->actual : sequence_input->reference;
                inputs[pattern][i] = values[(pattern % sequence_width) * H + i];
            } else {
                inputs[pattern][i] = bf16_to_f32(f32_to_bf16(
                    float(std::sin((i + 1) * (0.13 + pattern * 0.037)) * 0.5)));
            }
        }
        int duplicate = -1;
        for (int previous = 0; previous < pattern; ++previous) {
            if (inputs[previous] == inputs[pattern]) { duplicate = previous; break; }
        }
        if (duplicate >= 0) {
            references[pattern] = references[duplicate];
            original_references[pattern] = original_references[duplicate];
            probabilities[pattern] = probabilities[duplicate];
            routes[pattern] = routes[duplicate];
            cutoff_margins[pattern] = cutoff_margins[duplicate];
            continue;
        }
        auto logits = project(router, inputs[pattern]);
        auto& order = routes[pattern];
        order.resize(E); std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return logits[a] > logits[b]; });
        cutoff_margins[pattern] = logits[order[R - 1]] - logits[order[R]];
        if (!sequence_input) {
            std::cout << "layer=" << layer << " pattern=" << pattern << " cutoff_margin="
                      << logits[order[R - 1]] - logits[order[R]] << '\n';
        }
        order.resize(R);
        auto& p = probabilities[pattern]; p.resize(R);
        double sum = 0;
        for (int r = 0; r < R; ++r) { p[r] = std::exp(logits[order[r]] - logits[order[0]]); sum += p[r]; }
        for (auto& value : p) { value /= sum; }
        auto shared = project(sg, inputs[pattern]); swiglu(shared, project(su, inputs[pattern]));
        auto& result = references[pattern]; result = project(sd, shared);
        const double shared_factor = 1 / (1 + std::exp(-project(selector, inputs[pattern])[0]));
        for (auto& value : result) { value *= shared_factor; }
        if(calibrated_shared) {
            auto activation=project(original_sg,inputs[pattern]);
            swiglu(activation,project(original_su,inputs[pattern]));
            auto baseline=project(original_sd,activation);
            for(int i=0;i<H;++i) { baseline[i]=baseline[i]*shared_factor-result[i]; }
            original_references[pattern]=std::move(baseline);
        }
        for (int r = 0; r < R; ++r) {
            auto activation = gate.project(order[r], inputs[pattern]);
            swiglu(activation, up.project(order[r], inputs[pattern]));
            const auto expert = down.project(order[r], activation);
            for (int i = 0; i < H; ++i) { result[i] += p[r] * expert[i]; }
        }
        if(calibrated_shared) {
            for(int i=0;i<H;++i) { original_references[pattern][i]+=result[i]; }
        }
    }
    int failures = 0;
    if(calibrated_shared && !sequence_input) {
        const auto relative=[](std::span<const double> actual,std::span<const double> expected) {
            double error=0,norm=0;
            for(std::size_t i=0;i<actual.size();++i) {
                error+=(actual[i]-expected[i])*(actual[i]-expected[i]); norm+=expected[i]*expected[i];
            }
            return std::sqrt(error/std::max(norm,1e-300));
        };
        for(int pattern=0;pattern<patterns;++pattern) {
            std::cout<<"shared FP8 weight-only ideal layer="<<layer<<" pattern="<<pattern
                <<" completeMoE_relL2="<<relative(references[pattern],original_references[pattern])<<'\n';
        }
        using LP=ops::LinearPolicy;
        std::map<std::pair<int,bool>,std::vector<double>> a16_outputs;
        std::map<std::pair<int,bool>,std::vector<double>> a4_outputs;
        for(int mask=0;mask<8;++mask) {
            const ops::Qwen4SharedExpertPolicy shared_policy{
                mask&1?LP::AllowA8:LP::A16Only,mask&2?LP::AllowA8:LP::A16Only,mask&4?LP::AllowA8:LP::A16Only};
            for(auto expert_policy:{LP::A16Only,LP::AllowA4}) {
                for(int width:{1,8,9,24,25,65}) {
                    if(mask && expert_policy==LP::AllowA4 && width!=65) { continue; }
                    for(bool chunked:{false,true}) {
                        if(chunked && width!=65) { continue; }
                        std::vector<float> panel(std::size_t(width)*H);
                        for(int t=0;t<width;++t) { std::copy(inputs[t%2].begin(),inputs[t%2].end(),panel.begin()+t*H); }
                        auto dx=to_device_bf16(panel);
                        GuardedDeviceBuffer dy(panel.size()*2),di(width*R*4),dp(width*R*4);
                        Tensor x(dx.p,DType::BF16,{H,width}),y(dy.data(),DType::BF16,{H,width}),
                            ids(di.data(),DType::I32,{R,width}),probs(dp.data(),DType::FP32,{R,width});
                        for(int offset=0;offset<width;) {
                            const int count=chunked && offset==0?64:width-offset;
                            auto cx=x.slice(1,offset,count),cy=y.slice(1,offset,count),
                                ci=ids.slice(1,offset,count),cp=probs.slice(1,offset,count);
                            DeviceArena scratch(ops::qwen4_sparse_moe_resident_workspace_capacity_bytes(
                                weights,count,expert_policy,shared_policy));
                            ops::qwen4_sparse_moe_resident(cx,weights,ci,cp,cy,scratch,nullptr,expert_policy,shared_policy);
                            cuda_synchronize();
                            if(expert_policy==LP::AllowA4 && count>=64) {
                                std::array<int,E> counts{};
                                for(int expert:from_device<int>(ci.data,count*R)) { ++counts.at(expert); }
                                if(*std::max_element(counts.begin(),counts.end())<32) {
                                    throw std::runtime_error("shared calibration test did not activate routed A4");
                                }
                            }
                            if(scratch.used()!=0 || scratch.peak_used()>scratch.capacity()) {
                                throw std::runtime_error("calibrated shared workspace ownership");
                            }
                            offset+=count;
                        }
                        const auto actual=from_device_bf16(dy.data(),panel.size());
                        const auto actual_ids=from_device<int>(di.data(),width*R);
                        const auto probabilities_f32=from_device<float>(dp.data(),width*R);
                        const std::vector<double> actual_probabilities(probabilities_f32.begin(),probabilities_f32.end());
                        constexpr ReductionCriterion a4_criterion{.16,1.0/32768.0,.16};
                        const auto criterion=expert_policy==LP::AllowA4?a4_criterion:mask?shared_a8_criterion:output_criterion;
                        double worst=0;
                        const std::string label="shared calibrated layer="+std::to_string(layer)+" mask="+std::to_string(mask)+
                            " expert="+(expert_policy==LP::AllowA4?"A4":"A16")+" T="+std::to_string(width)+(chunked?" chunked":" whole");
                        for(int t=0;t<width;++t) {
                            failures+=verify_exact((label+" ids").c_str(),
                                std::vector<int>(actual_ids.begin()+t*R,actual_ids.begin()+(t+1)*R),routes[t%2]);
                            failures+=verify_pointwise(label+" probabilities",std::span(actual_probabilities).subspan(t*R,R),
                                probabilities[t%2],route_criterion);
                            failures+=verify_reduction(label,std::span(actual).subspan(t*H,H),references[t%2],criterion);
                            worst=std::max(worst,relative(std::span(actual).subspan(t*H,H),references[t%2]));
                        }
                        if(mask==0 && expert_policy==LP::A16Only) { a16_outputs[{width,chunked}]=actual; }
                        if(mask==0 && expert_policy==LP::AllowA4) { a4_outputs[{width,chunked}]=actual; }
                        std::cout<<label<<" oracle_worst_token_relL2="<<worst
                            <<" sameweights_A16_relL2="<<relative(actual,a16_outputs.at({width,chunked}))
                            <<" sameexpert_sharedA16_relL2="<<relative(actual,
                                (expert_policy==LP::AllowA4?a4_outputs:a16_outputs).at({width,chunked}))<<'\n';
                        failures+=dy.verify_guards(label+" output");
                        failures+=di.verify_guards(label+" ids"); failures+=dp.verify_guards(label+" probabilities");
                    }
                }
            }
        }
        return failures;
    }
    if (sequence_input) {
        const int width = sequence_width;
        const auto policy = allow_a4 ? ops::LinearPolicy::AllowA4 : ops::LinearPolicy::A16Only;
        auto input_device = to_device_bf16(sequence_input->actual);
        GuardedDeviceBuffer output_device(std::size_t(width) * H * 2), ids_device(width * R * 4), probs_device(width * R * 4);
        Tensor input(input_device.p, DType::BF16, {H, width}), output(output_device.data(), DType::BF16, {H, width});
        Tensor ids(ids_device.data(), DType::I32, {R, width}), probs(probs_device.data(), DType::FP32, {R, width});
        for (int offset = 0; offset < width;) {
            const int count = partitioned && offset == 0 ? std::max(1, width - 1) : width - offset;
            Tensor part_input(static_cast<std::uint16_t*>(input.data) + offset * H, DType::BF16, {H, count});
            Tensor part_output(static_cast<std::uint16_t*>(output.data) + offset * H, DType::BF16, {H, count});
            Tensor part_ids(static_cast<int*>(ids.data) + offset * R, DType::I32, {R, count});
            Tensor part_probs(static_cast<float*>(probs.data) + offset * R, DType::FP32, {R, count});
            DeviceArena workspace(ops::qwen4_sparse_moe_resident_workspace_capacity_bytes(weights, count, policy,shared_policy));
            qwen4_sequence::time_native_component("MoE layer="+std::to_string(layer)+
                " T="+std::to_string(count)+" routed-A4="+std::to_string(allow_a4)+
                " shared-FP8-mask="+std::to_string(calibrated_shared?weight_mask:0)+
                " shared-NV4-up="+std::to_string(!shared_up_nvfp4.empty())+" shared-A8-mask="+
                std::to_string((shared_policy.gate==ops::LinearPolicy::AllowA8?1:0)|
                    (shared_policy.up==ops::LinearPolicy::AllowA8?2:0)|
                    (shared_policy.down==ops::LinearPolicy::AllowA8?4:0)),[] {},[&] {
                ops::qwen4_sparse_moe_resident(part_input,weights,part_ids,part_probs,part_output,workspace,nullptr,policy,shared_policy);
            });
            if(layer==0 && !partitioned && std::getenv("NINFER_QWEN4_MOE_PROFILE")) {
                // One exact complete resident Op; excludes fixture loading and CPU oracles.
                ops::qwen4_sparse_moe_resident(part_input,weights,part_ids,part_probs,part_output,workspace,nullptr,policy,shared_policy);
                cuda_synchronize();CUDA_CHECK(cudaProfilerStart());
                ops::qwen4_sparse_moe_resident(part_input,weights,part_ids,part_probs,part_output,workspace,nullptr,policy,shared_policy);
                cuda_synchronize();CUDA_CHECK(cudaProfilerStop());
            }
            ops::qwen4_sparse_moe_resident(part_input, weights, part_ids, part_probs, part_output, workspace, nullptr, policy,shared_policy);
            cuda_synchronize();
            if (allow_a4) {
                std::array<int, E> counts{};
                for (int expert : from_device<int>(part_ids.data, count * R)) { ++counts.at(expert); }
                const int maximum = *std::max_element(counts.begin(), counts.end());
                std::cout << "A4_SEQUENCE layer=" << layer << " chunk=" << count
                          << " max_expert_occurrences=" << maximum << '\n';
                if (count >= 32 && maximum < 32) { throw std::runtime_error("A4 sequence did not activate grouped route"); }
            }
            offset += count;
        }
        const auto actual = from_device_bf16(output.data, std::size_t(width) * H);
        const auto actual_ids = from_device<int>(ids.data, width * R);
        const auto actual_p_float = from_device<float>(probs.data, width * R);
        const std::vector<double> actual_p(actual_p_float.begin(), actual_p_float.end());
        sequence_output->actual.assign(actual.begin(), actual.end());
        sequence_output->discrete_ids.assign(actual_ids.begin(), actual_ids.end());
        sequence_output->reference.reserve(actual.size());
        for (int t = 0; t < width; ++t) {
            const auto label = "native sequence layer " + std::to_string(layer) + " MoE token=" + std::to_string(t);
            sequence_output->reference_discrete_ids.insert(sequence_output->reference_discrete_ids.end(),
                routes[width+t].begin(),routes[width+t].end());
            auto actual_set = routes[t], reference_set = routes[width + t];
            std::sort(actual_set.begin(), actual_set.end());
            std::sort(reference_set.begin(), reference_set.end());
            if (actual_set != reference_set) {
                std::cout << "ROUTE_DRIFT layer=" << layer << " token=" << t
                          << " actual_input_cutoff_margin=" << cutoff_margins[t]
                          << " reference_input_cutoff_margin=" << cutoff_margins[width + t] << '\n';
            }
            failures += verify_exact((label + " ids").c_str(), std::vector<int>(actual_ids.begin()+t*R, actual_ids.begin()+(t+1)*R), routes[t]);
            failures += verify_pointwise(label + " probabilities", std::span(actual_p).subspan(t*R,R), probabilities[t], route_criterion);
            constexpr ReductionCriterion a4_criterion{0.16, 1.0 / 32768.0, 0.16};
            const bool shared_a8=shared_policy.gate==ops::LinearPolicy::AllowA8 ||
                shared_policy.up==ops::LinearPolicy::AllowA8 || shared_policy.down==ops::LinearPolicy::AllowA8;
            failures += verify_reduction(label, std::span(actual).subspan(t*H,H), references[t],
                                         allow_a4 ? a4_criterion : shared_a8?shared_a8_criterion:output_criterion);
            const auto expected = qwen4_sequence::represented(references[width + t]);
            sequence_output->reference.insert(sequence_output->reference.end(), expected.begin(), expected.end());
        }
        failures += output_device.verify_guards("native sequence MoE output");
        failures += ids_device.verify_guards("native sequence MoE ids");
        failures += probs_device.verify_guards("native sequence MoE probabilities");
        sequence_output->failures = failures;
        return failures;
    }
    for (auto policy : {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA4}) {
    for (int width : {1, 17, 63, 64, 65, 128, 129, 257}) {
        std::vector<float> panel(std::size_t(width) * H);
        for (int t = 0; t < width; ++t) { std::copy(inputs[t % 2].begin(), inputs[t % 2].end(), panel.begin() + t * H); }
        auto input_device = to_device_bf16(panel);
        GuardedDeviceBuffer output_device(std::size_t(width) * H * 2), ids_device(width * R * 4), probs_device(width * R * 4);
        Tensor input(input_device.p, DType::BF16, {H, width}), output(output_device.data(), DType::BF16, {H, width});
        Tensor ids(ids_device.data(), DType::I32, {R, width}), probs(probs_device.data(), DType::FP32, {R, width});
        DeviceArena workspace(ops::qwen4_sparse_moe_resident_workspace_capacity_bytes(weights, width, policy));
        ops::qwen4_sparse_moe_resident(input, weights, ids, probs, output, workspace, nullptr, policy);
        cuda_synchronize();
        const auto actual = from_device_bf16(output.data, std::size_t(width) * H);
        const auto actual_ids = from_device<int>(ids.data, width * R);
        std::array<int, E> counts{};
        for (const int expert : actual_ids) { ++counts.at(expert); }
        int min_count = width, max_count = 0;
        for (const int count : counts) {
            if (count != 0) { min_count = std::min(min_count, count); }
            max_count = std::max(max_count, count);
        }
        const auto actual_p_float = from_device<float>(probs.data, width * R);
        const std::vector<double> actual_p(actual_p_float.begin(), actual_p_float.end());
        double worst_relative = 0.0;
        for (int t = 0; t < width; ++t) {
            const auto label = "native layer " + std::to_string(layer) +
                (policy == ops::LinearPolicy::AllowA4 ? " allow-a4" : " a16") +
                " MoE T=" + std::to_string(width) + " token=" + std::to_string(t);
            failures += verify_exact((label + " ids").c_str(), std::vector<int>(actual_ids.begin()+t*R, actual_ids.begin()+(t+1)*R), routes[t%2]);
            failures += verify_pointwise(label + " probabilities", std::span(actual_p).subspan(t*R,R), probabilities[t%2], route_criterion);
            constexpr ReductionCriterion a4_criterion{0.16, 1.0 / 32768.0, 0.16};
            failures += verify_reduction(label, std::span(actual).subspan(t*H,H), references[t%2],
                policy == ops::LinearPolicy::AllowA4 ? a4_criterion : output_criterion);
            double error_squared = 0.0, reference_squared = 0.0;
            for (int i = 0; i < H; ++i) {
                const double reference = references[t % 2][i];
                const double error = actual[t * H + i] - reference;
                error_squared += error * error;
                reference_squared += reference * reference;
            }
            worst_relative = std::max(worst_relative, std::sqrt(error_squared / reference_squared));
        }
        std::cout << "layer=" << layer << " policy="
                  << (policy == ops::LinearPolicy::AllowA4 ? "allow-a4" : "a16")
                  << " T=" << width << " worst_token_relative_l2=" << worst_relative
                  << " live_expert_count_min=" << min_count << " max=" << max_count << '\n';
        failures += output_device.verify_guards("native MoE output");
        failures += ids_device.verify_guards("native MoE ids");
        failures += probs_device.verify_guards("native MoE probabilities");
    }
    }
    return failures;
}
// Bounded router diagnosis uses the production route launcher, not expert execution.
// Its oracle is the represented BF16 matrix/input dot in FP64, with stable ideal ranking.
int router_input(const std::string& root, const std::string& path) {
    artifact::Reader reader(root + "/qwen4-layer-3.ninfer");
    const std::string prefix="model.language_model.layers.3.mlp.";
    const auto weights=bf16_values(reader,prefix+"gate.weight");
    const auto bytes=reader.payload(prefix+"gate.weight").data;
    const auto gate=reader.payload(prefix+"shared_expert_gate.weight").data;
    const auto size=std::filesystem::file_size(path);
    if(size==0 || size%(H*4)!=0) throw std::runtime_error("invalid router input extent");
    std::vector<float> panel(size/4);
    std::ifstream input_file(path,std::ios::binary);
    input_file.read(reinterpret_cast<char*>(panel.data()),size);
    if(!input_file) throw std::runtime_error("cannot read router input");
    for(float x:panel) if(!std::isfinite(x) || bf16_to_f32(f32_to_bf16(x))!=x)
        throw std::runtime_error("router input must represent finite BF16");
    GuardedDeviceBuffer dw(bytes.size()),dg(gate.size());
    dw.copy_from_host(bytes.data(),bytes.size());dg.copy_from_host(gate.data(),gate.size());
    Weight weight{};weight.qtype=QType::BF16_CTRL;weight.qdata=dw.data();weight.n=E;weight.k=H;
    Tensor shared(dg.data(),DType::BF16,{H});
    int failures=0;
    auto check=[&](const std::vector<float>& values,const std::vector<double>& matrix,const char* label) {
        const int width=values.size()/H;
        auto dx=to_device_bf16(values);
        GuardedDeviceBuffer dl(2*E*width*4),di(R*width*4),dp(R*width*4),ds(width*4);
        Tensor x(dx.p,DType::BF16,{H,width}),logits(dl.data(),DType::FP32,{2*E,width}),
            ids(di.data(),DType::I32,{R,width}),probs(dp.data(),DType::FP32,{R,width}),
            shared_value(ds.data(),DType::FP32,{width});
        auto launch=[&] {
            if(width==1) ops::detail::qwen4_sparse_moe_resident_route_launch(x,weight,shared,logits,ids,probs,shared_value,nullptr);
            else ops::detail::qwen4_sparse_moe_resident_wide_route_launch(x,weight,shared,logits,ids,probs,shared_value,nullptr);
        };
        launch();cuda_synchronize();
        for(bool graph:{false,true}) {
            if(graph) {
                cudaStream_t stream;CUDA_CHECK(cudaStreamCreate(&stream));
                cudaGraph_t captured;cudaGraphExec_t executable;
                CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
                if(width==1) ops::detail::qwen4_sparse_moe_resident_route_launch(x,weight,shared,logits,ids,probs,shared_value,stream);
                else ops::detail::qwen4_sparse_moe_resident_wide_route_launch(x,weight,shared,logits,ids,probs,shared_value,stream);
                CUDA_CHECK(cudaStreamEndCapture(stream,&captured));
                CUDA_CHECK(cudaGraphInstantiate(&executable,captured,0));
                CUDA_CHECK(cudaGraphLaunch(executable,stream));CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_CHECK(cudaGraphExecDestroy(executable));CUDA_CHECK(cudaGraphDestroy(captured));CUDA_CHECK(cudaStreamDestroy(stream));
            }
            const auto actual=from_device<int>(ids.data,width*R);
            const auto actual_p=from_device<float>(probs.data,width*R);
            for(int t=0;t<width;++t) {
                std::vector<double> input(values.begin()+t*H,values.begin()+(t+1)*H);
                const auto score=project(matrix,input);
                std::vector<int> order(E);std::iota(order.begin(),order.end(),0);
                std::stable_sort(order.begin(),order.end(),[&](int a,int b){return score[a]>score[b];});
                order.resize(R);std::vector<double> expected_p(R);double sum=0;
                for(int rank=0;rank<R;++rank) sum+=expected_p[rank]=std::exp(score[order[rank]]-score[order[0]]);
                for(double& p:expected_p) p/=sum;
                const std::string name=std::string(label)+" token="+std::to_string(t)+(graph?" graph":" eager");
                failures+=verify_exact(name.c_str(),std::vector<int>(actual.begin()+t*R,actual.begin()+(t+1)*R),order);
                const std::vector<double> got_p(actual_p.begin()+t*R,actual_p.begin()+(t+1)*R);
                failures+=verify_pointwise(name+" probabilities",got_p,expected_p,route_criterion);
            }
        }
        if(std::getenv("NINFER_QWEN4_ROUTER_TIMING")) {
        cudaEvent_t begin,end;CUDA_CHECK(cudaEventCreate(&begin));CUDA_CHECK(cudaEventCreate(&end));
        CUDA_CHECK(cudaEventRecord(begin));for(int repeat=0;repeat<100;++repeat) launch();
        CUDA_CHECK(cudaEventRecord(end));CUDA_CHECK(cudaEventSynchronize(end));float ms=0;
        CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));CUDA_CHECK(cudaEventDestroy(begin));CUDA_CHECK(cudaEventDestroy(end));
        std::cout<<"ROUTER_ONLY "<<label<<" T="<<width<<" us="<<ms*10<<'\n';
        }
        failures+=dl.verify_guards(label)+di.verify_guards(label)+dp.verify_guards(label)+ds.verify_guards(label);
    };
    check(panel,weights,"captured native router");
    if(panel.size()>=3*H) check(std::vector<float>(panel.begin()+2*H,panel.begin()+3*H),weights,"captured token 2");
    std::vector<double> synthetic(E*H);std::vector<float> x(H,0);x[0]=x[1]=1;
    // Expert 1 must outrank 0 although both ideal logits round to FP32 1.0.
    // Experts 2/3 are an exact mathematical tie and retain ascending id order.
    synthetic[0]=synthetic[H]=1;synthetic[1]=std::ldexp(1.,-26);synthetic[H+1]=std::ldexp(1.,-25);
    synthetic[2*H]=synthetic[3*H]=.5;
    std::vector<std::uint16_t> words(synthetic.size());
    for(std::size_t i=0;i<words.size();++i) words[i]=f32_to_bf16(float(synthetic[i]));
    dw.copy_from_host(words.data(),words.size()*2);
    check(x,synthetic,"collapsed FP32 logits");
    std::vector<float> repeated(17*H);for(int t=0;t<17;++t) std::copy(x.begin(),x.end(),repeated.begin()+t*H);
    check(repeated,synthetic,"collapsed FP32 logits wide");
    // Equal ideal sums reached through opposite-sign cancellation must remain a
    // true tie, despite potentially different high/residue staging histories.
    x[2]=1;
    synthetic[2*H]=1;synthetic[2*H+1]=std::ldexp(1.,-25);synthetic[2*H+2]=-1;
    synthetic[3*H]=-1;synthetic[3*H+1]=std::ldexp(1.,-25);synthetic[3*H+2]=1;
    for(std::size_t i=0;i<words.size();++i) words[i]=f32_to_bf16(float(synthetic[i]));
    dw.copy_from_host(words.data(),words.size()*2);
    check(x,synthetic,"opposite cancellation tie");
    for(int t=0;t<17;++t) std::copy(x.begin(),x.end(),repeated.begin()+t*H);
    check(repeated,synthetic,"opposite cancellation tie wide");
    std::cout<<(failures?"FAIL":"PASS")<<" native router exact-ranking regression\n";
    return failures?1:0;
}
} // namespace

#ifdef NINFER_QWEN4_SEQUENCE_COMPONENTS
namespace ninfer::test::qwen4_sequence {
void moe_a4_calibration_coverage(const std::string& path,int layer,
                                std::span<const float> input,std::span<const int> experts) {
    if(input.size()!=H || experts.size()!=R) { throw std::invalid_argument("A4 diagnostic extent"); }
    artifact::Reader reader(path);
    const auto prefix="model.language_model.layers."+std::to_string(layer)+".mlp.experts.";
    const Bank gate{reader.payload(prefix+"gate_proj.weight").data,I,H},
        up{reader.payload(prefix+"up_proj.weight").data,I,H},
        down{reader.payload(prefix+"down_proj.weight").data,H,I};
    gate.validate();up.validate();down.validate();
    const std::vector<double> values(input.begin(),input.end());
    // Private profile attribution only. Enumerate represented codes independently;
    // these decoded inputs feed the existing complete FP64 dot, not a second oracle.
    const auto quantize=[&](float multiplier) {
        constexpr double levels[]{0,.5,1,1.5,2,3,4,6};
        std::vector<double> decoded(H);
        for(int begin=0;begin<H;begin+=16) {
            float maximum=0;
            for(int i=0;i<16;++i) { maximum=std::max(maximum,std::abs(input[begin+i])); }
            const float target=std::clamp(maximum/(6.F*multiplier),0x1p-9F,448.F);
            unsigned chosen=0;double distance=INFINITY;
            for(unsigned code=0;code<=126;++code) {
                const double delta=std::abs(fp8(code)-target);
                if(delta<distance || (delta==distance && (code&1)==0)) {
                    chosen=code;distance=delta;
                }
            }
            const float denominator=static_cast<float>(fp8(chosen))*multiplier;
            for(int i=0;i<16;++i) {
                const float normalized=input[begin+i]/denominator;
                unsigned selected=0;distance=INFINITY;
                for(unsigned code=0;code<8;++code) {
                    const double delta=std::abs(levels[code]-std::abs(normalized));
                    if(delta<distance || (delta==distance && (code&1)==0)) {
                        selected=code;distance=delta;
                    }
                }
                decoded[begin+i]=std::copysign(levels[selected],normalized)*fp8(chosen)*double(multiplier);
            }
        }
        return decoded;
    };
    const auto attribution=[&](const char* role,const Bank& bank,int expert) {
        const float multiplier=word(bank.bytes,bank.codes()+bank.scales()+4*(E+expert));
        float maximum=0;for(float value:input) { maximum=std::max(maximum,std::abs(value)); }
        const float guarded=std::max(multiplier,maximum/(6.F*448.F));
        const auto ideal=bank.project(expert,values);
        const auto original=quantize(multiplier),candidate=quantize(guarded);
        std::vector<double> clamped=values;
        for(double& value:clamped) { value=std::clamp(value,-6.*448.*multiplier,6.*448.*multiplier); }
        for(int mode=0;mode<3;++mode) {
            const auto& activation=mode==0?original:mode==1?clamped:candidate;
            const auto projected=bank.project(expert,activation);
            double error=0,norm=0,max_error=0,activation_error=0,activation_norm=0;
            for(std::size_t i=0;i<ideal.size();++i) {
                const double delta=projected[i]-ideal[i];error+=delta*delta;norm+=ideal[i]*ideal[i];
                max_error=std::max(max_error,std::abs(delta));
            }
            for(int i=0;i<H;++i) {
                const double delta=activation[i]-values[i];activation_error+=delta*delta;
                activation_norm+=values[i]*values[i];
            }
            std::cout<<"A4_ATTRIBUTION expert="<<expert<<" role="<<role
                <<" mode="<<(mode==0?"static":mode==1?"clamp-only":"guard-candidate")
                <<" activation_relative_l2="<<std::sqrt(activation_error/activation_norm)
                <<" dot_relative_l2="<<std::sqrt(error/norm)<<" dot_max_abs="<<max_error
                <<" multiplier="<<(mode==2?guarded:multiplier)<<'\n';
        }
    };
    const auto report=[&](const char* role,const Bank& bank,int expert,std::span<const double> activation) {
        const float multiplier=word(bank.bytes,bank.codes()+bank.scales()+4*(E+expert));
        const double maximum=6.0*448.0*double(multiplier);
        double maxabs=0;int clipped=0;
        for(double value:activation) { maxabs=std::max(maxabs,std::abs(value));clipped+=std::abs(value)>maximum; }
        std::cout<<"A4_CALIBRATION layer="<<layer<<" expert="<<expert<<" role="<<role
            <<" input_multiplier="<<multiplier<<" finite_limit="<<maximum
            <<" max_abs="<<maxabs<<" range_ratio="<<maxabs/maximum
            <<" outside_range="<<clipped<<"/"<<activation.size()<<'\n';
    };
    for(int expert:experts) {
        if(expert<0 || expert>=E) { throw std::invalid_argument("A4 diagnostic expert"); }
        report("gate",gate,expert,values);report("up",up,expert,values);
        attribution("gate",gate,expert);attribution("up",up,expert);
        // Supplementary calibration coverage from the existing independent mathematical
        // intermediate, not a new oracle or a claim of bit-identical private staging.
        auto intermediate=gate.project(expert,values);
        const auto projected_up=up.project(expert,values);
        swiglu(intermediate,projected_up);
        report("down-ideal-swiglu",down,expert,intermediate);
    }
}
Result moe(const std::string& path, int layer, const Result& input, bool partitioned, bool allow_a4) {
    Result result;
    run(path, layer, &input, &result, partitioned, allow_a4);
    return result;
}
Result moe_calibrated(const std::string& root, const std::string& path, int layer,
                      const Result& input, bool partitioned, int mask, bool allow_a4,
                      int weight_mask, const std::string& shared_up_nvfp4) {
    if(layer<0 || layer>3 || mask<0 || mask>7 || weight_mask<0 || weight_mask>7 ||
       (mask&~weight_mask) || (!shared_up_nvfp4.empty() && (layer!=0 || (weight_mask&2) || (mask&2)))) {
        throw std::invalid_argument("unqualified calibrated shared source/policy");
    }
    using P=ops::LinearPolicy;
    const ops::Qwen4SharedExpertPolicy policy{
        mask&1?P::AllowA8:P::A16Only,mask&2?P::AllowA8:P::A16Only,mask&4?P::AllowA8:P::A16Only};
    Result result;
    run(path,layer,&input,&result,partitioned,allow_a4,true,policy,root,weight_mask,shared_up_nvfp4);
    return result;
}
}
#else
int main(int argc, char** argv) {
    if(argc==4 && std::string_view(argv[1])=="--router-input") {
        if(require_cuda()!=0) return 1;
        try { return router_input(argv[2],argv[3]); }
        catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
    }
    const bool shared_fp8=argc==2 && std::string_view(argv[1])=="--native-shared-fp8";
    const char* root = argc == 2 && !shared_fp8 ? argv[1] : std::getenv("NINFER_QWEN4_NATIVE_LAYERS");
    if (!root) { std::cout << "Native Qwen4 layers not configured\n"; return 77; }
    if (require_cuda() != 0) { return 1; }
    try {
        int failures = 0;
        if(shared_fp8) {
            for(int layer:{0,3}) {
                failures+=run(std::string(root)+"/qwen4-layer-"+std::to_string(layer)+".ninfer",layer,
                    nullptr,nullptr,false,false,true);
            }
            return failures?1:0;
        }
        for (int layer : {0, 3}) { failures += run(std::string(root) + "/qwen4-layer-" + std::to_string(layer) + ".ninfer", layer); }
        failures += real_ple(std::string(root) + "/qwen4-ple-rows.ninfer");
        failures += real_nvfp4_ple(std::string(root) + "/qwen4-ple-nvfp4-rows.ninfer");
        for (const char* mode : {"whole", "partition"}) {
            const auto path = std::filesystem::path(root) /
                (std::string("qwen4-moe-layer3-original-") + mode + "-input.bf16");
            if (!std::filesystem::exists(path)) {
                std::cout << "Original MoE input regression fixture missing: " << path << '\n';
                continue;
            }
            constexpr std::size_t words = 17 * H;
            if (std::filesystem::file_size(path) != words * 2) {
                throw std::runtime_error("incorrect original MoE input fixture size");
            }
            std::ifstream file(path, std::ios::binary);
            qwen4_sequence::Result panel, result;
            panel.actual.resize(words);
            for (float& value : panel.actual) {
                unsigned char bytes[2]; file.read(reinterpret_cast<char*>(bytes), 2);
                if (!file) { throw std::runtime_error("cannot read original MoE input fixture"); }
                value = bf16_to_f32(static_cast<std::uint16_t>(bytes[0] | (unsigned(bytes[1]) << 8)));
                if (!std::isfinite(value)) { throw std::runtime_error("nonfinite captured MoE input"); }
            }
            if (!file) { throw std::runtime_error("cannot read original MoE input fixture"); }
            panel.reference = panel.actual;
            failures += run(std::string(root) + "/qwen4-layer-3.ninfer", 3, &panel, &result,
                            std::string_view(mode) == "partition");
            std::cout << "Original " << mode << " MoE input: public Op vs independent FP64 oracle\n";
        }
        std::cout << (failures ? "FAIL" : "PASS") << " native represented-weight MoE; not complete GDN/QSA layers\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
#endif
