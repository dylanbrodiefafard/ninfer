#include "ops/gdn_input_proj/fp8/fp8_gdn_conv_plan.h"

#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"
#include "ops/gdn_input_proj/gdn_projected_conv.h"
#include "ops/linear/fp8/fp8_config.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kChannels = 10240;
constexpr std::int32_t kZRows    = 6144;

void require_policy(LinearPolicy policy, const char* operation) {
    if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA8) {
        throw std::invalid_argument(std::string(operation) + " admits only A16 or A8");
    }
}

void require_snapshot_domain(LinearPolicy policy, std::int32_t width, std::int32_t batch_size) {
    require_policy(policy, "fp8 GDN snapshot");
    if (width != 1 || batch_size <= 0 || batch_size > kFp8GdnConvMaxBatch) {
        throw std::invalid_argument("fp8 GDN snapshot: registered only for W=1, B=1..8");
    }
}

void require_record_domain(LinearPolicy policy, std::int32_t width, std::int32_t batch_size) {
    require_policy(policy, "fp8 GDN record");
    if (width < 2 || width > kDflash2TreeMaxWidth || batch_size <= 0 ||
        batch_size > kFp8GdnConvMaxBatch) {
        throw std::invalid_argument("fp8 GDN record: invalid B/W domain");
    }
}

} // namespace

std::size_t fp8_gdn_snapshot_workspace_capacity_bytes(LinearPolicy policy, std::int32_t batch_size,
                                                      std::int32_t min_width,
                                                      std::int32_t max_width) {
    if (min_width <= 0 || max_width < min_width) {
        throw std::invalid_argument("fp8 GDN snapshot workspace: invalid width interval");
    }
    require_snapshot_domain(policy, min_width, batch_size);
    require_snapshot_domain(policy, max_width, batch_size);
    return 0;
}

std::size_t fp8_gdn_record_workspace_capacity_bytes(LinearPolicy policy, std::int32_t batch_size,
                                                    std::int32_t min_width,
                                                    std::int32_t max_width) {
    if (min_width < 2 || max_width < min_width) {
        throw std::invalid_argument("fp8 GDN record workspace: invalid width interval");
    }
    require_record_domain(policy, min_width, batch_size);
    require_record_domain(policy, max_width, batch_size);
    // A16 writes directly into the observable record; no private staging arena.
    return 0;
}

namespace {

// The record of one request: the fused SmallT launch at its fused widths, otherwise the A16
// projection into the record followed by the projected convolution.
void launch_record_row(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                       const Tensor& conv_states, const Tensor& valid_columns,
                       const Tensor& initial_slot, Tensor& conv_record, Tensor& query, Tensor& key,
                       Tensor& value, Tensor& z, cudaStream_t stream, const Tensor* parent_index) {
    const std::int32_t width = x.ne[1];
    if (fp8_gdn_fused_record_width(width, parent_index != nullptr)) {
        fp8_gdn_record_fused_launch(x, weight, conv_weight, conv_states, valid_columns,
                                    initial_slot, conv_record, query, key, value, z, stream,
                                    parent_index);
        return;
    }
    Tensor x_flat(x.data, DType::BF16, {Fp8GdnInputGeometry::kInputRows, width});
    Tensor record_flat(conv_record.data, DType::BF16, {kChannels, width});
    Tensor z_flat(z.data, DType::BF16, {kZRows, width});
    fp8_gdn_input_a16_dispatch(x_flat, weight, record_flat, z_flat, stream);
    gdn_projected_conv_record_launch(conv_record, conv_weight, conv_states, valid_columns,
                                     initial_slot, query, key, value, stream, parent_index);
}

} // namespace

void fp8_gdn_snapshot_dispatch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                               Tensor& conv_states, const Tensor& valid_columns,
                               const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                               Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                               LinearPolicy policy, cudaStream_t stream) {
    require_snapshot_domain(policy, x.ne[1], x.ne[2]);
    // The snapshot GEMV publishes one request's history per launch.
    for (std::int32_t row = 0; row < x.ne[2]; ++row) {
        Tensor q    = query.slice(2, row, 1);
        Tensor k    = key.slice(2, row, 1);
        Tensor v    = value.slice(2, row, 1);
        Tensor gate = z.slice(2, row, 1);
        fp8_gdn_snapshot_decode_launch(x.slice(2, row, 1), weight, conv_weight, conv_states,
                                       valid_columns.data ? valid_columns.slice(0, row, 1)
                                                          : Tensor{},
                                       initial_slot.slice(0, row, 1),
                                       snapshot_base_slot.slice(0, row, 1), q, k, v, gate, stream);
    }
}

void fp8_gdn_record_dispatch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                             const Tensor& conv_states, const Tensor& valid_columns,
                             const Tensor& initial_slot, Tensor& conv_record, Tensor& query,
                             Tensor& key, Tensor& value, Tensor& z, LinearPolicy policy,
                             cudaStream_t stream, const Tensor* parent_index) {
    require_record_domain(policy, x.ne[1], x.ne[2]);
    // Every request keeps its C=1 projection and convolution profile: flattening W*B would select
    // a different SmallT reduction and could replace a fused C=1 convolution with a materialized
    // BF16 projection.
    for (std::int32_t row = 0; row < x.ne[2]; ++row) {
        Tensor q      = query.slice(2, row, 1);
        Tensor k      = key.slice(2, row, 1);
        Tensor v      = value.slice(2, row, 1);
        Tensor gate   = z.slice(2, row, 1);
        Tensor record = conv_record.slice(2, row, 1);
        Tensor parents;
        if (parent_index != nullptr) {
            parents = x.ne[2] > 1 ? parent_index->slice(1, row, 1) : *parent_index;
        }
        launch_record_row(x.slice(2, row, 1), weight, conv_weight, conv_states,
                          valid_columns.data ? valid_columns.slice(0, row, 1) : Tensor{},
                          initial_slot.slice(0, row, 1), record, q, k, v, gate, stream,
                          parent_index != nullptr ? &parents : nullptr);
    }
}

} // namespace ninfer::ops::detail
