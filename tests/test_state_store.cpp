#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/device.h"
#include "core/linear_attention_state.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

namespace {

struct PlannedState {
    ninfer::LinearAttentionStatePoolLayout layout;
    std::size_t bytes = 0;
};

PlannedState plan_state(std::uint32_t layers, std::int32_t conv_channels, std::int32_t conv_width,
                        std::int32_t value_heads, std::int32_t value_head_dim,
                        std::int32_t key_head_dim, std::int32_t slot_count = 1,
                        ninfer::DType conv_dtype = ninfer::DType::BF16) {
    ninfer::LayoutBuilder builder;
    auto layout = ninfer::plan_linear_attention_state_pool(
        builder, ninfer::LinearAttentionStatePoolSpec{.layers         = layers,
                                                      .conv_channels  = conv_channels,
                                                      .conv_width     = conv_width,
                                                      .value_heads    = value_heads,
                                                      .value_head_dim = value_head_dim,
                                                      .key_head_dim   = key_head_dim,
                                                      .slot_count     = slot_count,
                                                      .conv_dtype     = conv_dtype});
    return PlannedState{std::move(layout), builder.finish(256)};
}

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int expect_size(std::size_t actual, std::size_t expected, const char* label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return 1;
}

int check_shape(const ninfer::Tensor& tensor, const std::int32_t (&expected)[4],
                const char* label) {
    int failures = 0;
    for (int i = 0; i < 4; ++i) {
        if (tensor.ne[i] != expected[i]) {
            ++failures;
            std::cerr << label << ".ne[" << i << "] expected " << expected[i] << ", got "
                      << tensor.ne[i] << '\n';
        }
    }
    return failures;
}

int expect_device_byte(const ninfer::Tensor& tensor, unsigned char expected, const char* label) {
    std::vector<unsigned char> host(tensor.bytes());
    CUDA_CHECK(cudaMemcpy(host.data(), tensor.data, host.size(), cudaMemcpyDeviceToHost));
    for (unsigned char value : host) {
        if (value != expected) {
            std::cerr << label << " expected byte 0x" << std::hex << static_cast<int>(expected)
                      << ", got 0x" << static_cast<int>(value) << std::dec << '\n';
            return 1;
        }
    }
    return 0;
}

int expect_host_fill(const std::vector<unsigned char>& host, unsigned char expected,
                     const char* label) {
    for (unsigned char value : host) {
        if (value != expected) {
            std::cerr << label << " expected byte 0x" << std::hex << static_cast<int>(expected)
                      << ", got 0x" << static_cast<int>(value) << std::dec << '\n';
            return 1;
        }
    }
    return 0;
}

void fill_gdn_slot(ninfer::LinearAttentionStatePool& pool, std::int32_t slot, unsigned char value,
                   cudaStream_t stream) {
    for (std::uint32_t layer = 0; layer < pool.layer_count(); ++layer) {
        CUDA_CHECK(cudaMemsetAsync(pool.conv_slot(layer, slot).data, value,
                                   pool.conv_slot(layer, slot).bytes(), stream));
        CUDA_CHECK(cudaMemsetAsync(pool.recurrent_slot(layer, slot).data, value,
                                   pool.recurrent_slot(layer, slot).bytes(), stream));
    }
}

int test_gdn_freeze_prefill_overlaps_d2d(ninfer::DeviceContext& ctx) {
    // Slot 0 = current, 1 = rewrite, 2 = Engine-wide staging. D2D current→staging
    // then immediately overwrite current on the compute stream (next prefill can be
    // shorter/faster than the copy). Staging and the copy-stream host pack after
    // d2d_done must still be the freeze; rewrite must stay untouched.
    constexpr unsigned char kFreeze  = 0x11;
    constexpr unsigned char kNext    = 0x22;
    constexpr unsigned char kRewrite = 0x33;
    auto race_plan                   = plan_state(16, 32, 4, 8, 16, 16, 3);
    ninfer::DeviceArena arena(race_plan.bytes);
    ninfer::LinearAttentionStatePool pool({arena.base(), arena.capacity()}, race_plan.layout);
    fill_gdn_slot(pool, 0, kFreeze, ctx.stream);
    fill_gdn_slot(pool, 1, kRewrite, ctx.stream);
    fill_gdn_slot(pool, 2, 0x00, ctx.stream);

    pool.copy_slot_2d(0, 2, ctx.stream);
    cudaEvent_t d2d_done = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&d2d_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(d2d_done, ctx.stream));
    fill_gdn_slot(pool, 0, kNext, ctx.stream);
    CUDA_CHECK(cudaStreamWaitEvent(ctx.copy_stream, d2d_done, 0));

    std::vector<unsigned char> staging_conv(pool.conv_host_image_bytes(), 0);
    std::vector<unsigned char> staging_rec(pool.recurrent_host_image_bytes(), 0);
    std::vector<unsigned char> current_conv(pool.conv_host_image_bytes(), 0);
    std::vector<unsigned char> current_rec(pool.recurrent_host_image_bytes(), 0);
    std::vector<unsigned char> rewrite_conv(pool.conv_host_image_bytes(), 0);
    std::vector<unsigned char> rewrite_rec(pool.recurrent_host_image_bytes(), 0);
    pool.pack_slot_to_host(2, staging_conv.data(), staging_rec.data(), ctx.copy_stream);
    pool.pack_slot_to_host(1, rewrite_conv.data(), rewrite_rec.data(), ctx.copy_stream);
    cudaEvent_t copies_done = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&copies_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(copies_done, ctx.copy_stream));
    ctx.synchronize_all();
    pool.pack_slot_to_host(0, current_conv.data(), current_rec.data(), ctx.copy_stream);
    ctx.synchronize_all();

    int failures = 0;
    failures += expect_device_byte(pool.conv_slot(0, 2), kFreeze, "staging conv after overlapping prefill");
    failures += expect_device_byte(pool.recurrent_slot(0, 2), kFreeze,
                                   "staging recurrent after overlapping prefill");
    failures += expect_device_byte(pool.conv_slot(15, 2), kFreeze,
                                   "staging last-layer conv after overlapping prefill");
    failures += expect_device_byte(pool.recurrent_slot(15, 2), kFreeze,
                                   "staging last-layer recurrent after overlapping prefill");
    failures += expect_device_byte(pool.conv_slot(0, 0), kNext, "current conv is next prefill");
    failures += expect_device_byte(pool.conv_slot(0, 1), kRewrite, "rewrite conv untouched by freeze D2D");
    failures += expect_host_fill(staging_conv, kFreeze, "host pack staging conv");
    failures += expect_host_fill(staging_rec, kFreeze, "host pack staging recurrent");
    failures += expect_host_fill(current_conv, kNext, "host pack current conv");
    failures += expect_host_fill(current_rec, kNext, "host pack current recurrent");
    failures += expect_host_fill(rewrite_conv, kRewrite, "host pack rewrite conv");
    failures += expect_host_fill(rewrite_rec, kRewrite, "host pack rewrite recurrent");

    pool.copy_slot_2d(2, 0, ctx.stream);
    ctx.synchronize_all();
    failures += expect_device_byte(pool.conv_slot(0, 0), kFreeze,
                                   "staging restore D2D current conv");
    failures += expect_device_byte(pool.recurrent_slot(0, 0), kFreeze,
                                   "staging restore D2D current recurrent");
    failures += expect_device_byte(pool.conv_slot(0, 2), kFreeze,
                                   "staging still holds freeze after restore D2D");
    failures += expect_device_byte(pool.conv_slot(0, 1), kRewrite,
                                   "rewrite untouched by staging restore D2D");

    CUDA_CHECK(cudaEventSynchronize(copies_done));
    fill_gdn_slot(pool, 0, kNext, ctx.stream);
    pool.copy_slot_2d(0, 2, ctx.stream);
    CUDA_CHECK(cudaEventRecord(d2d_done, ctx.stream));
    CUDA_CHECK(cudaStreamWaitEvent(ctx.copy_stream, d2d_done, 0));
    std::vector<unsigned char> second_conv(pool.conv_host_image_bytes(), 0);
    std::vector<unsigned char> second_rec(pool.recurrent_host_image_bytes(), 0);
    pool.pack_slot_to_host(2, second_conv.data(), second_rec.data(), ctx.copy_stream);
    ctx.synchronize_all();
    CUDA_CHECK(cudaEventDestroy(d2d_done));
    CUDA_CHECK(cudaEventDestroy(copies_done));
    failures += expect_host_fill(staging_conv, kFreeze, "first host GDN after second freeze");
    failures += expect_host_fill(second_conv, kNext, "second host GDN image");
    failures += expect_host_fill(second_rec, kNext, "second host GDN recurrent");
    failures += expect_device_byte(pool.conv_slot(0, 1), kRewrite,
                                   "rewrite slot survived a second staging freeze");
    return failures;
}

int test_gdn_abort_inflight_host_pack(ninfer::DeviceContext& ctx) {
    // abort_lane waits copies_done then frees host GDN. Simulate cancel after D2D/D2H are
    // queued: wait the pack event, drop the host image, and freeze again into staging.
    constexpr unsigned char kFreeze = 0x44;
    constexpr unsigned char kNext   = 0x55;
    auto plan                       = plan_state(16, 32, 4, 8, 16, 16, 3);
    ninfer::DeviceArena arena(plan.bytes);
    ninfer::LinearAttentionStatePool pool({arena.base(), arena.capacity()}, plan.layout);
    fill_gdn_slot(pool, 0, kFreeze, ctx.stream);
    fill_gdn_slot(pool, 1, 0x00, ctx.stream);
    fill_gdn_slot(pool, 2, 0x00, ctx.stream);
    pool.copy_slot_2d(0, 2, ctx.stream);
    cudaEvent_t d2d_done = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&d2d_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(d2d_done, ctx.stream));
    CUDA_CHECK(cudaStreamWaitEvent(ctx.copy_stream, d2d_done, 0));
    ninfer::PinnedHostBuffer conv_host(pool.conv_host_image_bytes());
    ninfer::PinnedHostBuffer rec_host(pool.recurrent_host_image_bytes());
    pool.pack_slot_to_host(2, conv_host.data(), rec_host.data(), ctx.copy_stream);
    cudaEvent_t copies_done = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&copies_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(copies_done, ctx.copy_stream));
    fill_gdn_slot(pool, 0, kNext, ctx.stream);
    CUDA_CHECK(cudaEventSynchronize(copies_done));
    std::vector<unsigned char> first(
        static_cast<unsigned char*>(conv_host.data()),
        static_cast<unsigned char*>(conv_host.data()) + conv_host.size());
    conv_host = ninfer::PinnedHostBuffer(1);
    rec_host  = ninfer::PinnedHostBuffer(1);
    pool.copy_slot_2d(0, 2, ctx.stream);
    CUDA_CHECK(cudaEventRecord(d2d_done, ctx.stream));
    CUDA_CHECK(cudaStreamWaitEvent(ctx.copy_stream, d2d_done, 0));
    ninfer::PinnedHostBuffer conv_second(pool.conv_host_image_bytes());
    ninfer::PinnedHostBuffer rec_second(pool.recurrent_host_image_bytes());
    pool.pack_slot_to_host(2, conv_second.data(), rec_second.data(), ctx.copy_stream);
    ctx.synchronize_all();
    CUDA_CHECK(cudaEventDestroy(d2d_done));
    CUDA_CHECK(cudaEventDestroy(copies_done));
    int failures = 0;
    failures += expect_host_fill(first, kFreeze, "aborted freeze host GDN completed");
    std::vector<unsigned char> second(
        static_cast<unsigned char*>(conv_second.data()),
        static_cast<unsigned char*>(conv_second.data()) + conv_second.size());
    failures += expect_host_fill(second, kNext, "staging reused after abort wait");
    return failures;
}

int test_gdn_c2_shared_staging(ninfer::DeviceContext& ctx) {
    // C=2: current slots 0/1, rewrite 2/3, Engine-wide staging 4. Lane 1 must not publish
    // over lane 0's in-flight D2H, and restore identity is lane-specific.
    constexpr unsigned char kLane0 = 0x60;
    constexpr unsigned char kLane1 = 0x70;
    auto plan                      = plan_state(8, 16, 4, 4, 8, 8, 5);
    ninfer::DeviceArena arena(plan.bytes);
    ninfer::LinearAttentionStatePool pool({arena.base(), arena.capacity()}, plan.layout);
    fill_gdn_slot(pool, 0, kLane0, ctx.stream);
    fill_gdn_slot(pool, 1, kLane1, ctx.stream);
    fill_gdn_slot(pool, 2, 0x11, ctx.stream);
    fill_gdn_slot(pool, 3, 0x22, ctx.stream);
    fill_gdn_slot(pool, 4, 0x00, ctx.stream);
    pool.copy_slot_2d(0, 4, ctx.stream);
    cudaEvent_t d2d_done = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&d2d_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(d2d_done, ctx.stream));
    CUDA_CHECK(cudaStreamWaitEvent(ctx.copy_stream, d2d_done, 0));
    std::vector<unsigned char> lane0_conv(pool.conv_host_image_bytes(), 0);
    std::vector<unsigned char> lane0_rec(pool.recurrent_host_image_bytes(), 0);
    pool.pack_slot_to_host(4, lane0_conv.data(), lane0_rec.data(), ctx.copy_stream);
    cudaEvent_t copies_done = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&copies_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(copies_done, ctx.copy_stream));
    CUDA_CHECK(cudaEventSynchronize(copies_done));
    pool.copy_slot_2d(1, 4, ctx.stream);
    CUDA_CHECK(cudaEventRecord(d2d_done, ctx.stream));
    CUDA_CHECK(cudaStreamWaitEvent(ctx.copy_stream, d2d_done, 0));
    std::vector<unsigned char> lane1_conv(pool.conv_host_image_bytes(), 0);
    std::vector<unsigned char> lane1_rec(pool.recurrent_host_image_bytes(), 0);
    pool.pack_slot_to_host(4, lane1_conv.data(), lane1_rec.data(), ctx.copy_stream);
    ctx.synchronize_all();
    CUDA_CHECK(cudaEventDestroy(d2d_done));
    CUDA_CHECK(cudaEventDestroy(copies_done));
    int failures = 0;
    failures += expect_host_fill(lane0_conv, kLane0, "lane 0 host GDN after lane 1 staging reuse");
    failures += expect_host_fill(lane0_rec, kLane0, "lane 0 host recurrent after lane 1 reuse");
    failures += expect_host_fill(lane1_conv, kLane1, "lane 1 staging pack");
    failures += expect_host_fill(lane1_rec, kLane1, "lane 1 staging recurrent pack");
    failures += expect_device_byte(pool.conv_slot(0, 2), 0x11, "lane 0 rewrite survived staging");
    failures += expect_device_byte(pool.conv_slot(0, 3), 0x22, "lane 1 rewrite survived staging");
    return failures;
}

int test_gdn_c3_shared_staging(ninfer::DeviceContext& ctx) {
    // C=3: current 0/1/2, rewrite 3/4/5, Engine-wide staging 6. Prefill is one lane at a
    // time, so freezes serialize on staging; each lane's host image and rewrite slot stay
    // distinct after the later lanes reuse the same staging slot.
    constexpr unsigned char kLane0 = 0x80;
    constexpr unsigned char kLane1 = 0x90;
    constexpr unsigned char kLane2 = 0xa0;
    auto plan                      = plan_state(8, 16, 4, 4, 8, 8, 7);
    ninfer::DeviceArena arena(plan.bytes);
    ninfer::LinearAttentionStatePool pool({arena.base(), arena.capacity()}, plan.layout);
    fill_gdn_slot(pool, 0, kLane0, ctx.stream);
    fill_gdn_slot(pool, 1, kLane1, ctx.stream);
    fill_gdn_slot(pool, 2, kLane2, ctx.stream);
    fill_gdn_slot(pool, 3, 0x13, ctx.stream);
    fill_gdn_slot(pool, 4, 0x14, ctx.stream);
    fill_gdn_slot(pool, 5, 0x15, ctx.stream);
    fill_gdn_slot(pool, 6, 0x00, ctx.stream);

    auto freeze_lane = [&](std::int32_t current, std::vector<unsigned char>& conv,
                           std::vector<unsigned char>& rec) {
        pool.copy_slot_2d(current, 6, ctx.stream);
        cudaEvent_t d2d_done = nullptr;
        CUDA_CHECK(cudaEventCreateWithFlags(&d2d_done, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(d2d_done, ctx.stream));
        CUDA_CHECK(cudaStreamWaitEvent(ctx.copy_stream, d2d_done, 0));
        conv.assign(pool.conv_host_image_bytes(), 0);
        rec.assign(pool.recurrent_host_image_bytes(), 0);
        pool.pack_slot_to_host(6, conv.data(), rec.data(), ctx.copy_stream);
        cudaEvent_t copies_done = nullptr;
        CUDA_CHECK(cudaEventCreateWithFlags(&copies_done, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(copies_done, ctx.copy_stream));
        CUDA_CHECK(cudaEventSynchronize(copies_done));
        CUDA_CHECK(cudaEventDestroy(copies_done));
        CUDA_CHECK(cudaEventDestroy(d2d_done));
    };

    std::vector<unsigned char> lane0_conv;
    std::vector<unsigned char> lane0_rec;
    std::vector<unsigned char> lane1_conv;
    std::vector<unsigned char> lane1_rec;
    std::vector<unsigned char> lane2_conv;
    std::vector<unsigned char> lane2_rec;
    freeze_lane(0, lane0_conv, lane0_rec);
    freeze_lane(1, lane1_conv, lane1_rec);
    freeze_lane(2, lane2_conv, lane2_rec);

    int failures = 0;
    failures += expect_host_fill(lane0_conv, kLane0, "C=3 lane 0 host GDN after later staging reuse");
    failures += expect_host_fill(lane0_rec, kLane0, "C=3 lane 0 host recurrent after later reuse");
    failures += expect_host_fill(lane1_conv, kLane1, "C=3 lane 1 host GDN after lane 2 reuse");
    failures += expect_host_fill(lane1_rec, kLane1, "C=3 lane 1 host recurrent after lane 2 reuse");
    failures += expect_host_fill(lane2_conv, kLane2, "C=3 lane 2 staging pack");
    failures += expect_host_fill(lane2_rec, kLane2, "C=3 lane 2 staging recurrent pack");
    failures += expect_device_byte(pool.conv_slot(0, 3), 0x13, "C=3 lane 0 rewrite survived staging");
    failures += expect_device_byte(pool.conv_slot(0, 4), 0x14, "C=3 lane 1 rewrite survived staging");
    failures += expect_device_byte(pool.conv_slot(0, 5), 0x15, "C=3 lane 2 rewrite survived staging");
    return failures;
}

int test_gdn_pin_source_not_rewrite_or_leftover(ninfer::DeviceContext& ctx) {
    // Pin copies current→2C then packs 2C. Rewrite and a leftover 2C pattern must not become
    // the published image. Suffix/BeforeSuffix may then mutate current; 2C stays the pin.
    constexpr unsigned char kCurrent = 0xa1;
    constexpr unsigned char kRewrite = 0xb2;
    constexpr unsigned char kLeftover = 0xc3;
    constexpr unsigned char kSuffix  = 0xd4;
    auto plan                        = plan_state(16, 32, 4, 8, 16, 16, 3);
    ninfer::DeviceArena arena(plan.bytes);
    ninfer::LinearAttentionStatePool pool({arena.base(), arena.capacity()}, plan.layout);
    fill_gdn_slot(pool, 0, kCurrent, ctx.stream);
    fill_gdn_slot(pool, 1, kRewrite, ctx.stream);
    fill_gdn_slot(pool, 2, kLeftover, ctx.stream);

    pool.copy_slot_2d(0, 2, ctx.stream);
    cudaEvent_t d2d_done = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&d2d_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(d2d_done, ctx.stream));
    CUDA_CHECK(cudaStreamWaitEvent(ctx.copy_stream, d2d_done, 0));
    std::vector<unsigned char> host_conv(pool.conv_host_image_bytes(), 0);
    std::vector<unsigned char> host_rec(pool.recurrent_host_image_bytes(), 0);
    pool.pack_slot_to_host(2, host_conv.data(), host_rec.data(), ctx.copy_stream);
    cudaEvent_t copies_done = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&copies_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(copies_done, ctx.copy_stream));
    fill_gdn_slot(pool, 0, kSuffix, ctx.stream);
    ctx.synchronize_all();

    int failures = 0;
    failures += expect_device_byte(pool.conv_slot(0, 2), kCurrent, "pin 2C conv is current not leftover");
    failures += expect_device_byte(pool.recurrent_slot(15, 2), kCurrent,
                                   "pin 2C last-layer recurrent is current");
    failures += expect_device_byte(pool.conv_slot(0, 1), kRewrite, "pin left rewrite conv");
    failures += expect_device_byte(pool.conv_slot(0, 0), kSuffix, "suffix mutated current after pin");
    failures += expect_host_fill(host_conv, kCurrent, "pin host conv is current not rewrite/leftover");
    failures += expect_host_fill(host_rec, kCurrent, "pin host recurrent is current");

    CUDA_CHECK(cudaStreamWaitEvent(ctx.stream, copies_done, 0));
    pool.copy_slot_2d(2, 0, ctx.stream);
    ctx.synchronize_all();
    failures += expect_device_byte(pool.conv_slot(0, 0), kCurrent, "restore D2D current from pin 2C");
    failures += expect_device_byte(pool.recurrent_slot(15, 0), kCurrent,
                                   "restore D2D last-layer recurrent from pin");
    failures += expect_device_byte(pool.conv_slot(0, 1), kRewrite, "restore D2D left rewrite");
    failures += expect_device_byte(pool.conv_slot(0, 2), kCurrent, "2C still holds pin after restore");
    CUDA_CHECK(cudaEventDestroy(d2d_done));
    CUDA_CHECK(cudaEventDestroy(copies_done));
    return failures;
}

void fill_cyclic_lane(ninfer::CyclicKVCache& cache, std::int32_t lane, unsigned char value,
                      cudaStream_t stream) {
    for (std::uint32_t layer = 0; layer < cache.layer_count(); ++layer) {
        const ninfer::CyclicKVCacheLayerView view = cache.layer_view(layer);
        ninfer::Tensor k = view.k.slice(3, lane, 1);
        ninfer::Tensor v = view.v.slice(3, lane, 1);
        CUDA_CHECK(cudaMemsetAsync(k.data, value, k.bytes(), stream));
        CUDA_CHECK(cudaMemsetAsync(v.data, value, v.bytes(), stream));
    }
}

int test_dflash_cyclic_pin_packs_staging_not_live(ninfer::DeviceContext& ctx) {
    // Pin D2Ds live lane → 1-lane staging, then D2H from staging on copy_stream.
    // Suffix may mutate live; the host image and staging must stay the freeze.
    constexpr unsigned char kFreeze   = 0xa1;
    constexpr unsigned char kRewrite  = 0xb2;
    constexpr unsigned char kLeftover = 0xc3;
    constexpr unsigned char kSuffix   = 0xd4;
    ninfer::LayoutBuilder builder;
    const auto local_layout   = ninfer::plan_cyclic_kv_cache(builder, 2, 16, 2, 8, 2);
    const auto staging_layout = ninfer::plan_cyclic_kv_cache(builder, 2, 16, 2, 8, 1);
    ninfer::DeviceArena arena(builder.finish(256));
    ninfer::CyclicKVCache local({arena.base(), arena.capacity()}, local_layout);
    ninfer::CyclicKVCache staging({arena.base(), arena.capacity()}, staging_layout);

    fill_cyclic_lane(local, 0, kFreeze, ctx.stream);
    fill_cyclic_lane(local, 1, kRewrite, ctx.stream);
    fill_cyclic_lane(staging, 0, kLeftover, ctx.stream);

    staging.copy_lane_from(local, 0, 0, ctx.stream);
    cudaEvent_t d2d_done = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&d2d_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(d2d_done, ctx.stream));
    CUDA_CHECK(cudaStreamWaitEvent(ctx.copy_stream, d2d_done, 0));
    std::vector<unsigned char> host(staging.lane_host_bytes(), 0);
    staging.copy_lane_to_host(0, host.data(), ctx.copy_stream);
    cudaEvent_t copies_done = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&copies_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(copies_done, ctx.copy_stream));
    fill_cyclic_lane(local, 0, kSuffix, ctx.stream);
    ctx.synchronize_all();

    int failures = 0;
    failures += expect_device_byte(staging.layer_view(0).k.slice(3, 0, 1), kFreeze,
                                   "cyclic staging K is freeze not leftover");
    failures += expect_device_byte(staging.layer_view(1).v.slice(3, 0, 1), kFreeze,
                                   "cyclic staging last-layer V is freeze");
    failures += expect_device_byte(local.layer_view(0).k.slice(3, 1, 1), kRewrite,
                                   "pin left rewrite cyclic lane");
    failures += expect_device_byte(local.layer_view(0).k.slice(3, 0, 1), kSuffix,
                                   "suffix mutated live cyclic after pin");
    failures += expect_host_fill(host, kFreeze, "cyclic host pack is freeze not live suffix");
    CUDA_CHECK(cudaEventDestroy(d2d_done));
    CUDA_CHECK(cudaEventDestroy(copies_done));
    return failures;
}

int test_dflash_cyclic_c2_shared_staging(ninfer::DeviceContext& ctx) {
    // C=2 local lanes, 1-lane Engine-wide staging. Lane 1 must wait lane 0's D2H
    // (fence_staging_copies) before clobbering staging; both host images stay distinct.
    constexpr unsigned char kLane0 = 0x60;
    constexpr unsigned char kLane1 = 0x70;
    ninfer::LayoutBuilder builder;
    const auto local_layout   = ninfer::plan_cyclic_kv_cache(builder, 2, 16, 2, 8, 2);
    const auto staging_layout = ninfer::plan_cyclic_kv_cache(builder, 2, 16, 2, 8, 1);
    ninfer::DeviceArena arena(builder.finish(256));
    ninfer::CyclicKVCache local({arena.base(), arena.capacity()}, local_layout);
    ninfer::CyclicKVCache staging({arena.base(), arena.capacity()}, staging_layout);
    fill_cyclic_lane(local, 0, kLane0, ctx.stream);
    fill_cyclic_lane(local, 1, kLane1, ctx.stream);

    staging.copy_lane_from(local, 0, 0, ctx.stream);
    cudaEvent_t d2d_done = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&d2d_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(d2d_done, ctx.stream));
    CUDA_CHECK(cudaStreamWaitEvent(ctx.copy_stream, d2d_done, 0));
    std::vector<unsigned char> lane0(staging.lane_host_bytes(), 0);
    staging.copy_lane_to_host(0, lane0.data(), ctx.copy_stream);
    cudaEvent_t copies_done = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&copies_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(copies_done, ctx.copy_stream));
    CUDA_CHECK(cudaEventSynchronize(copies_done));

    staging.copy_lane_from(local, 1, 0, ctx.stream);
    CUDA_CHECK(cudaEventRecord(d2d_done, ctx.stream));
    CUDA_CHECK(cudaStreamWaitEvent(ctx.copy_stream, d2d_done, 0));
    std::vector<unsigned char> lane1(staging.lane_host_bytes(), 0);
    staging.copy_lane_to_host(0, lane1.data(), ctx.copy_stream);
    ctx.synchronize_all();
    CUDA_CHECK(cudaEventDestroy(d2d_done));
    CUDA_CHECK(cudaEventDestroy(copies_done));

    int failures = 0;
    failures += expect_host_fill(lane0, kLane0, "cyclic lane 0 host after lane 1 staging reuse");
    failures += expect_host_fill(lane1, kLane1, "cyclic lane 1 staging pack");
    failures += expect_device_byte(local.layer_view(0).k.slice(3, 0, 1), kLane0,
                                   "cyclic lane 0 live survived staging reuse");
    failures += expect_device_byte(local.layer_view(0).k.slice(3, 1, 1), kLane1,
                                   "cyclic lane 1 live survived staging reuse");
    return failures;
}

int test_dflash_cyclic_c3_shared_staging(ninfer::DeviceContext& ctx) {
    // Prefill is one lane at a time; freezes serialize on the 1-lane cyclic staging window.
    constexpr unsigned char kLane0 = 0x80;
    constexpr unsigned char kLane1 = 0x90;
    constexpr unsigned char kLane2 = 0xa0;
    ninfer::LayoutBuilder builder;
    const auto local_layout   = ninfer::plan_cyclic_kv_cache(builder, 2, 16, 2, 8, 3);
    const auto staging_layout = ninfer::plan_cyclic_kv_cache(builder, 2, 16, 2, 8, 1);
    ninfer::DeviceArena arena(builder.finish(256));
    ninfer::CyclicKVCache local({arena.base(), arena.capacity()}, local_layout);
    ninfer::CyclicKVCache staging({arena.base(), arena.capacity()}, staging_layout);
    fill_cyclic_lane(local, 0, kLane0, ctx.stream);
    fill_cyclic_lane(local, 1, kLane1, ctx.stream);
    fill_cyclic_lane(local, 2, kLane2, ctx.stream);

    auto freeze_lane = [&](std::int32_t lane, std::vector<unsigned char>& host) {
        staging.copy_lane_from(local, lane, 0, ctx.stream);
        cudaEvent_t d2d_done = nullptr;
        CUDA_CHECK(cudaEventCreateWithFlags(&d2d_done, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(d2d_done, ctx.stream));
        CUDA_CHECK(cudaStreamWaitEvent(ctx.copy_stream, d2d_done, 0));
        host.assign(staging.lane_host_bytes(), 0);
        staging.copy_lane_to_host(0, host.data(), ctx.copy_stream);
        cudaEvent_t copies_done = nullptr;
        CUDA_CHECK(cudaEventCreateWithFlags(&copies_done, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(copies_done, ctx.copy_stream));
        CUDA_CHECK(cudaEventSynchronize(copies_done));
        CUDA_CHECK(cudaEventDestroy(copies_done));
        CUDA_CHECK(cudaEventDestroy(d2d_done));
    };

    std::vector<unsigned char> lane0;
    std::vector<unsigned char> lane1;
    std::vector<unsigned char> lane2;
    freeze_lane(0, lane0);
    freeze_lane(1, lane1);
    freeze_lane(2, lane2);

    int failures = 0;
    failures += expect_host_fill(lane0, kLane0, "cyclic C=3 lane 0 host after later staging reuse");
    failures += expect_host_fill(lane1, kLane1, "cyclic C=3 lane 1 host after lane 2 reuse");
    failures += expect_host_fill(lane2, kLane2, "cyclic C=3 lane 2 staging pack");
    return failures;
}

int test_dflash_cyclic_abort_inflight_host_pack(ninfer::DeviceContext& ctx) {
    // abort_lane destroys the host head (wait copies_done) then a later freeze reuses staging.
    constexpr unsigned char kFreeze = 0x44;
    constexpr unsigned char kNext   = 0x55;
    ninfer::LayoutBuilder builder;
    const auto local_layout   = ninfer::plan_cyclic_kv_cache(builder, 2, 16, 2, 8, 1);
    const auto staging_layout = ninfer::plan_cyclic_kv_cache(builder, 2, 16, 2, 8, 1);
    ninfer::DeviceArena arena(builder.finish(256));
    ninfer::CyclicKVCache local({arena.base(), arena.capacity()}, local_layout);
    ninfer::CyclicKVCache staging({arena.base(), arena.capacity()}, staging_layout);
    fill_cyclic_lane(local, 0, kFreeze, ctx.stream);

    staging.copy_lane_from(local, 0, 0, ctx.stream);
    cudaEvent_t d2d_done = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&d2d_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(d2d_done, ctx.stream));
    CUDA_CHECK(cudaStreamWaitEvent(ctx.copy_stream, d2d_done, 0));
    std::vector<unsigned char> first(staging.lane_host_bytes(), 0);
    staging.copy_lane_to_host(0, first.data(), ctx.copy_stream);
    cudaEvent_t copies_done = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&copies_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(copies_done, ctx.copy_stream));
    fill_cyclic_lane(local, 0, kNext, ctx.stream);
    CUDA_CHECK(cudaEventSynchronize(copies_done));

    staging.copy_lane_from(local, 0, 0, ctx.stream);
    CUDA_CHECK(cudaEventRecord(d2d_done, ctx.stream));
    CUDA_CHECK(cudaStreamWaitEvent(ctx.copy_stream, d2d_done, 0));
    std::vector<unsigned char> second(staging.lane_host_bytes(), 0);
    staging.copy_lane_to_host(0, second.data(), ctx.copy_stream);
    ctx.synchronize_all();
    CUDA_CHECK(cudaEventDestroy(d2d_done));
    CUDA_CHECK(cudaEventDestroy(copies_done));

    int failures = 0;
    failures += expect_host_fill(first, kFreeze, "aborted freeze cyclic host completed");
    failures += expect_host_fill(second, kNext, "cyclic staging reused after abort wait");
    return failures;
}

int test_gdn_freeze_borrow_reload_restore_bytes(ninfer::DeviceContext& ctx) {
    // C=2: lane 0 rollback occupies 2C. Lane 1 freeze copies its current through 2C, then
    // reload H2Ds the rollback host image back into 2C. Restore must wait copies_done
    // (reload H2D), not freeze d2d_done (that event is ladder→2C).
    constexpr unsigned char kRollback = 0x41;
    constexpr unsigned char kLadder   = 0x51;
    constexpr unsigned char kRewrite0 = 0x11;
    constexpr unsigned char kRewrite1 = 0x22;
    auto plan                         = plan_state(16, 32, 4, 8, 16, 16, 5);
    ninfer::DeviceArena arena(plan.bytes);
    ninfer::LinearAttentionStatePool pool({arena.base(), arena.capacity()}, plan.layout);
    fill_gdn_slot(pool, 0, kRollback, ctx.stream);
    fill_gdn_slot(pool, 1, kLadder, ctx.stream);
    fill_gdn_slot(pool, 2, kRewrite0, ctx.stream);
    fill_gdn_slot(pool, 3, kRewrite1, ctx.stream);
    fill_gdn_slot(pool, 4, 0x00, ctx.stream);

    pool.copy_slot_2d(0, 4, ctx.stream);
    cudaEvent_t d2d_done = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&d2d_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(d2d_done, ctx.stream));
    CUDA_CHECK(cudaStreamWaitEvent(ctx.copy_stream, d2d_done, 0));
    ninfer::PinnedHostBuffer rollback_conv(pool.conv_host_image_bytes());
    ninfer::PinnedHostBuffer rollback_rec(pool.recurrent_host_image_bytes());
    pool.pack_slot_to_host(4, rollback_conv.data(), rollback_rec.data(), ctx.copy_stream);
    cudaEvent_t occupant_copies = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&occupant_copies, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(occupant_copies, ctx.copy_stream));
    CUDA_CHECK(cudaEventSynchronize(occupant_copies));

    pool.copy_slot_2d(1, 4, ctx.stream);
    CUDA_CHECK(cudaEventRecord(d2d_done, ctx.stream));
    CUDA_CHECK(cudaStreamWaitEvent(ctx.copy_stream, d2d_done, 0));
    ninfer::PinnedHostBuffer ladder_conv(pool.conv_host_image_bytes());
    ninfer::PinnedHostBuffer ladder_rec(pool.recurrent_host_image_bytes());
    pool.pack_slot_to_host(4, ladder_conv.data(), ladder_rec.data(), ctx.copy_stream);
    cudaEvent_t freeze_copies = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&freeze_copies, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(freeze_copies, ctx.copy_stream));
    CUDA_CHECK(cudaEventSynchronize(freeze_copies));

    CUDA_CHECK(cudaStreamWaitEvent(ctx.copy_stream, d2d_done, 0));
    pool.unpack_slot_from_host(4, rollback_conv.data(), rollback_rec.data(), ctx.copy_stream);
    cudaEvent_t reload_copies = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&reload_copies, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(reload_copies, ctx.copy_stream));

    fill_gdn_slot(pool, 0, 0xee, ctx.stream);
    CUDA_CHECK(cudaStreamWaitEvent(ctx.stream, reload_copies, 0));
    pool.copy_slot_2d(4, 0, ctx.stream);
    ctx.synchronize_all();

    int failures = 0;
    failures += expect_device_byte(pool.conv_slot(0, 0), kRollback,
                                   "restore after freeze-borrow is rollback not ladder");
    failures += expect_device_byte(pool.recurrent_slot(15, 0), kRollback,
                                   "restore last-layer recurrent is rollback");
    failures += expect_device_byte(pool.conv_slot(0, 4), kRollback, "reloaded 2C is rollback");
    failures += expect_device_byte(pool.conv_slot(0, 1), kLadder, "lane 1 current survived restore");
    failures += expect_device_byte(pool.conv_slot(0, 2), kRewrite0, "lane 0 rewrite survived borrow");
    failures += expect_device_byte(pool.conv_slot(0, 3), kRewrite1, "lane 1 rewrite survived borrow");
    std::vector<unsigned char> ladder_host(
        static_cast<unsigned char*>(ladder_conv.data()),
        static_cast<unsigned char*>(ladder_conv.data()) + ladder_conv.size());
    failures += expect_host_fill(ladder_host, kLadder, "ladder host pack during borrow");
    CUDA_CHECK(cudaEventDestroy(d2d_done));
    CUDA_CHECK(cudaEventDestroy(occupant_copies));
    CUDA_CHECK(cudaEventDestroy(freeze_copies));
    CUDA_CHECK(cudaEventDestroy(reload_copies));
    return failures;
}

int test_gdn_abort_host_unpack_ignores_stale_2c(ninfer::DeviceContext& ctx) {
    // clear_lane unoccupies 2C. A later restore of the same identity must H2D the host pin,
    // not D2D leftover 2C bytes still sitting in the slot.
    constexpr unsigned char kPin   = 0x61;
    constexpr unsigned char kStale = 0x99;
    auto plan                      = plan_state(8, 16, 4, 4, 8, 8, 3);
    ninfer::DeviceArena arena(plan.bytes);
    ninfer::LinearAttentionStatePool pool({arena.base(), arena.capacity()}, plan.layout);
    fill_gdn_slot(pool, 0, kPin, ctx.stream);
    fill_gdn_slot(pool, 1, 0x22, ctx.stream);
    fill_gdn_slot(pool, 2, 0x00, ctx.stream);
    pool.copy_slot_2d(0, 2, ctx.stream);
    cudaEvent_t d2d_done = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&d2d_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(d2d_done, ctx.stream));
    CUDA_CHECK(cudaStreamWaitEvent(ctx.copy_stream, d2d_done, 0));
    ninfer::PinnedHostBuffer host_conv(pool.conv_host_image_bytes());
    ninfer::PinnedHostBuffer host_rec(pool.recurrent_host_image_bytes());
    pool.pack_slot_to_host(2, host_conv.data(), host_rec.data(), ctx.copy_stream);
    ctx.synchronize_all();

    fill_gdn_slot(pool, 2, kStale, ctx.stream);
    fill_gdn_slot(pool, 0, 0x00, ctx.stream);
    pool.unpack_slot_from_host(0, host_conv.data(), host_rec.data(), ctx.copy_stream);
    ctx.synchronize_all();
    CUDA_CHECK(cudaEventDestroy(d2d_done));

    int failures = 0;
    failures += expect_device_byte(pool.conv_slot(0, 0), kPin, "abort restore current from host pin");
    failures += expect_device_byte(pool.recurrent_slot(0, 0), kPin,
                                   "abort restore recurrent from host pin");
    failures += expect_device_byte(pool.conv_slot(0, 2), kStale, "stale 2C was not the restore source");
    failures += expect_device_byte(pool.conv_slot(0, 1), 0x22, "abort restore left rewrite");
    return failures;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err)) {
        std::cerr << "FAIL: no usable CUDA device\n";
        return 1;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }
    if (count == 0) {
        std::cerr << "FAIL: no usable CUDA device\n";
        return 1;
    }

    int failures = 0;
    ninfer::DeviceContext ctx(0);
    auto state_plan = plan_state(3, 10, 3, 4, 5, 6);
    ninfer::DeviceArena state_arena(state_plan.bytes);
    CUDA_CHECK(cudaMemset(state_arena.base(), 0x4a, state_arena.capacity()));
    ninfer::LinearAttentionStatePool state({state_arena.base(), state_arena.capacity()},
                                           state_plan.layout);

    failures += expect_size(state.layer_count(), 3, "state.layer_count");
    failures += expect_size(state.slot_count(), 1, "state.slot_count");
    failures += expect_size(state.conv.size(), 3, "state.conv.size");
    failures += expect_size(state.recurrent.size(), 3, "state.recurrent.size");
    failures += expect_size(state.spec.conv_width, 3, "state.conv_width");
    failures += expect_size(state.conv_slot_stride_elements(), 30, "state.conv slot stride");
    failures +=
        expect_size(state.recurrent_slot_stride_elements(), 120, "state recurrent slot stride");
    for (std::size_t layer = 0; layer < state.layer_count(); ++layer) {
        failures += check_shape(state.conv[layer], {10, 3, 1, 1}, "state.conv");
        failures += check_shape(state.recurrent[layer], {6, 5, 4, 1}, "state.recurrent");
        failures += check_shape(state.conv_slot(static_cast<std::uint32_t>(layer), 0),
                                {10, 3, 1, 1}, "state.conv_slot");
        failures += check_shape(state.recurrent_slot(static_cast<std::uint32_t>(layer), 0),
                                {6, 5, 4, 1}, "state.recurrent_slot");
        if (state.conv[layer].dtype != ninfer::DType::BF16) {
            ++failures;
            std::cerr << "conv dtype is not BF16\n";
        }
        if (state.recurrent[layer].dtype != ninfer::DType::FP32) {
            ++failures;
            std::cerr << "recurrent dtype is not FP32\n";
        }
        if (state.conv[layer].data == state.recurrent[layer].data) {
            ++failures;
            std::cerr << "conv/recurrent alias for layer " << layer << '\n';
        }
        failures += expect_device_byte(state.conv[layer], 0x4a, "constructor-mutated conv");
        failures +=
            expect_device_byte(state.recurrent[layer], 0x4a, "constructor-mutated recurrent");
    }
    if (state.conv[0].data == state.conv[1].data ||
        state.recurrent[0].data == state.recurrent[1].data) {
        ++failures;
        std::cerr << "state layers alias\n";
    }

    state.zero_slot(0, ctx.stream);
    ctx.synchronize();
    failures += expect_device_byte(state.conv[0], 0, "zeroed conv");
    failures += expect_device_byte(state.recurrent[1], 0, "zeroed recurrent");

    auto slotted_plan = plan_state(2, 10, 3, 4, 5, 6, 3);
    ninfer::DeviceArena slotted_arena(slotted_plan.bytes);
    CUDA_CHECK(cudaMemsetAsync(slotted_arena.base(), 0, slotted_arena.capacity(), ctx.stream));
    ninfer::LinearAttentionStatePool slotted({slotted_arena.base(), slotted_arena.capacity()},
                                             slotted_plan.layout);
    failures += expect_size(slotted.slot_count(), 3, "slotted.slot_count");
    failures += check_shape(slotted.conv[0], {10, 3, 3, 1}, "slotted.conv");
    failures += check_shape(slotted.recurrent[0], {6, 5, 4, 3}, "slotted.recurrent");
    failures += check_shape(slotted.conv_slot(0, 2), {10, 3, 1, 1}, "slotted.conv_slot");
    failures += check_shape(slotted.recurrent_slot(0, 2), {6, 5, 4, 1}, "slotted.recurrent_slot");

    ninfer::Tensor conv0             = slotted.conv_slot(0, 0);
    ninfer::Tensor conv1             = slotted.conv_slot(0, 1);
    ninfer::Tensor recurrent0        = slotted.recurrent_slot(0, 0);
    ninfer::Tensor recurrent1        = slotted.recurrent_slot(0, 1);
    ninfer::Tensor conv1_layer1      = slotted.conv_slot(1, 1);
    ninfer::Tensor recurrent1_layer1 = slotted.recurrent_slot(1, 1);
    CUDA_CHECK(cudaMemsetAsync(conv0.data, 0x7a, conv0.bytes(), ctx.stream));
    CUDA_CHECK(cudaMemsetAsync(conv1.data, 0x6b, conv1.bytes(), ctx.stream));
    CUDA_CHECK(cudaMemsetAsync(recurrent0.data, 0x5c, recurrent0.bytes(), ctx.stream));
    CUDA_CHECK(cudaMemsetAsync(recurrent1.data, 0x4d, recurrent1.bytes(), ctx.stream));
    CUDA_CHECK(cudaMemsetAsync(conv1_layer1.data, 0x3c, conv1_layer1.bytes(), ctx.stream));
    CUDA_CHECK(cudaMemsetAsync(recurrent1_layer1.data, 0x2d, recurrent1_layer1.bytes(), ctx.stream));

    slotted.copy_slot(1, 2, ctx.stream);
    ctx.synchronize();
    failures += expect_device_byte(slotted.conv_slot(0, 2), 0x6b, "copied conv slot");
    failures += expect_device_byte(slotted.recurrent_slot(0, 2), 0x4d, "copied recurrent slot");
    failures += expect_device_byte(slotted.conv_slot(1, 2), 0x3c, "copied conv layer1");
    failures += expect_device_byte(slotted.recurrent_slot(1, 2), 0x2d, "copied recurrent layer1");

    CUDA_CHECK(cudaMemsetAsync(slotted.conv_slot(0, 0).data, 0xaa, slotted.conv_slot(0, 0).bytes(),
                               ctx.stream));
    CUDA_CHECK(cudaMemsetAsync(slotted.recurrent_slot(0, 0).data, 0xbb,
                               slotted.recurrent_slot(0, 0).bytes(), ctx.stream));
    CUDA_CHECK(cudaMemsetAsync(slotted.conv_slot(1, 0).data, 0xcc, slotted.conv_slot(1, 0).bytes(),
                               ctx.stream));
    CUDA_CHECK(cudaMemsetAsync(slotted.recurrent_slot(1, 0).data, 0xdd,
                               slotted.recurrent_slot(1, 0).bytes(), ctx.stream));
    slotted.copy_slot_2d(0, 1, ctx.stream);
    ctx.synchronize();
    failures += expect_device_byte(slotted.conv_slot(0, 0), 0xaa, "2d copy left source conv");
    failures += expect_device_byte(slotted.recurrent_slot(0, 0), 0xbb,
                                   "2d copy left source recurrent");
    failures += expect_device_byte(slotted.conv_slot(1, 0), 0xcc, "2d copy left source conv layer1");
    failures += expect_device_byte(slotted.recurrent_slot(1, 0), 0xdd,
                                   "2d copy left source recurrent layer1");
    failures += expect_device_byte(slotted.conv_slot(0, 1), 0xaa, "2d copied conv slot");
    failures += expect_device_byte(slotted.recurrent_slot(0, 1), 0xbb, "2d copied recurrent slot");
    failures += expect_device_byte(slotted.conv_slot(1, 1), 0xcc, "2d copied conv layer1");
    failures += expect_device_byte(slotted.recurrent_slot(1, 1), 0xdd, "2d copied recurrent layer1");
    failures += expect_device_byte(slotted.conv_slot(0, 2), 0x6b, "2d copy kept other conv slot");
    failures += expect_device_byte(slotted.recurrent_slot(0, 2), 0x4d,
                                   "2d copy kept other recurrent slot");
    failures += expect_device_byte(slotted.conv_slot(1, 2), 0x3c, "2d copy kept other conv layer1");
    failures += expect_device_byte(slotted.recurrent_slot(1, 2), 0x2d,
                                   "2d copy kept other recurrent layer1");

    slotted.zero_slot(0, ctx.stream);
    ctx.synchronize();
    failures += expect_device_byte(slotted.conv_slot(0, 0), 0, "zeroed conv slot0");
    failures += expect_device_byte(slotted.recurrent_slot(0, 0), 0, "zeroed recurrent slot0");
    failures += expect_device_byte(slotted.conv_slot(0, 1), 0xaa, "zero kept 2d conv dest");
    failures += expect_device_byte(slotted.recurrent_slot(0, 1), 0xbb, "zero kept 2d recurrent dest");
    failures += expect_device_byte(slotted.conv_slot(0, 2), 0x6b, "zero kept conv slot2");
    failures += expect_device_byte(slotted.recurrent_slot(0, 2), 0x4d, "zero kept recurrent slot2");

    auto fp32_conv_plan = plan_state(1, 7, 2, 2, 3, 4, 2, ninfer::DType::FP32);
    ninfer::DeviceArena fp32_conv_arena(fp32_conv_plan.bytes);
    ninfer::LinearAttentionStatePool fp32_conv({fp32_conv_arena.base(), fp32_conv_arena.capacity()},
                                               fp32_conv_plan.layout);
    if (fp32_conv.conv[0].dtype != ninfer::DType::FP32) {
        ++failures;
        std::cerr << "FP32 conv geometry did not retain its dtype\n";
    }
    failures += check_shape(fp32_conv.conv[0], {7, 2, 2, 1}, "fp32_conv.conv");
    failures += check_shape(fp32_conv.recurrent[0], {4, 3, 2, 2}, "fp32_conv.recurrent");
    failures += test_gdn_freeze_prefill_overlaps_d2d(ctx);
    failures += test_gdn_abort_inflight_host_pack(ctx);
    failures += test_gdn_c2_shared_staging(ctx);
    failures += test_gdn_c3_shared_staging(ctx);
    failures += test_gdn_pin_source_not_rewrite_or_leftover(ctx);
    failures += test_dflash_cyclic_pin_packs_staging_not_live(ctx);
    failures += test_dflash_cyclic_c2_shared_staging(ctx);
    failures += test_dflash_cyclic_c3_shared_staging(ctx);
    failures += test_dflash_cyclic_abort_inflight_host_pack(ctx);
    failures += test_gdn_freeze_borrow_reload_restore_bytes(ctx);
    failures += test_gdn_abort_host_unpack_ignores_stale_2c(ctx);

    return failures == 0 ? 0 : fail("linear attention state pool test failed");
}
