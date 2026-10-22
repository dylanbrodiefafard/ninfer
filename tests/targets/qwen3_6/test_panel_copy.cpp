#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "targets/qwen3_6/impl/runtime/panel_copy.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct RowCopy {
    std::size_t destination_offset;
    std::size_t source_offset;
    std::size_t destination_pitch;
    std::size_t source_pitch;
    std::size_t row_bytes;
    std::size_t rows;
};

// The byte oracle copies only logical row payloads; all padding and the inactive profile's
// destination retain their sentinel. Source and destination are independent represented bytes.
void apply_oracle(std::vector<std::byte>& destination, const std::vector<std::byte>& source,
                  const std::vector<RowCopy>& copies) {
    for (const auto& copy : copies) {
        for (std::size_t row = 0; row < copy.rows; ++row) {
            for (std::size_t byte = 0; byte < copy.row_bytes; ++byte) {
                destination.at(copy.destination_offset + row * copy.destination_pitch + byte) =
                    source.at(copy.source_offset + row * copy.source_pitch + byte);
            }
        }
    }
}

template <class Enqueue>
void check_updates(ninfer::DeviceContext& device, const char* label,
                   const std::array<std::vector<RowCopy>, 2>& profiles, Enqueue enqueue) {
    std::size_t source_bytes = 0, destination_bytes = 0;
    for (const auto& copies : profiles) {
        for (const auto& copy : copies) {
            source_bytes =
                std::max(source_bytes, copy.source_offset + copy.rows * copy.source_pitch);
            destination_bytes = std::max(destination_bytes, copy.destination_offset +
                                                                copy.rows * copy.destination_pitch);
        }
    }
    source_bytes += 256;
    destination_bytes += 256;
    std::vector<std::byte> source(source_bytes);
    for (std::size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<std::byte>((i * 17 + i / 251) % 251);
    }
    std::vector<std::byte> expected(destination_bytes, std::byte{0xa5});
    std::vector<std::byte> actual(destination_bytes);
    ninfer::DeviceBuffer source_device(source_bytes);
    ninfer::DeviceBuffer destination_device(destination_bytes);
    source_device.copy_from_host(source.data(), source.size());
    destination_device.copy_from_host(expected.data(), expected.size());
    const auto verify = [&] {
        device.synchronize();
        destination_device.copy_to_host(actual.data(), actual.size());
        if (actual != expected) { throw std::runtime_error(std::string(label) + " byte mismatch"); }
    };
    // Qualify the uncaptured transfer, then capture the same production call sites.
    enqueue(0, destination_device.p, source_device.p);
    apply_oracle(expected, source, profiles[0]);
    verify();
    std::fill(expected.begin(), expected.end(), std::byte{0xa5});
    destination_device.copy_from_host(expected.data(), expected.size());
    std::array<ninfer::DecodeGraphDefinition, 2> definitions;
    for (std::size_t i = 0; i < definitions.size(); ++i) {
        definitions[i].capture(device.stream,
                               [&] { enqueue(i, destination_device.p, source_device.p); });
    }
    ninfer::DecodeGraphExecutable executable;
    executable.instantiate(definitions[0]);
    executable.upload(device.stream);
    verify();
    for (const std::size_t profile : {0U, 1U, 0U}) {
        executable.update(definitions[profile]);
        executable.upload(device.stream);
        verify();
        executable.launch(device.stream);
        apply_oracle(expected, source, profiles[profile]);
        verify();
    }
    std::vector<std::byte> source_after(source_bytes);
    source_device.copy_to_host(source_after.data(), source_after.size());
    if (source_after != source) {
        throw std::runtime_error(std::string(label) + " changed source");
    }
}

void check_panels(ninfer::DeviceContext& device, std::int32_t batch) {
    namespace family = ninfer::targets::qwen3_6;
    for (const bool integers : {true, false}) {
        const std::int32_t channels = integers ? 1 : 5120;
        const std::size_t elem      = integers ? 4 : 2;
        const std::array<std::int32_t, 2> source_widths{integers ? 3 : 4, 12};
        const std::array<std::int32_t, 2> destination_widths{12, 8};
        std::array<std::vector<RowCopy>, 2> copies;
        constexpr std::size_t second_offset = 2 * 1024 * 1024;
        for (std::size_t i = 0; i < copies.size(); ++i) {
            const auto source_pitch = static_cast<std::size_t>(channels) * source_widths[i] * elem;
            const auto destination_pitch =
                static_cast<std::size_t>(channels) * destination_widths[i] * elem;
            const auto row_bytes =
                integers ? source_pitch : std::min(source_pitch, destination_pitch);
            // Integer panels copy the entire source width, so both destination panels are wider.
            copies[i].push_back({i * second_offset, i * second_offset,
                                 integers ? 16 * elem : destination_pitch, source_pitch, row_bytes,
                                 static_cast<std::size_t>(batch)});
        }
        check_updates(
            device, integers ? "integer panel" : "width panel", copies,
            [&](std::size_t i, void* dst, void* src) {
                auto* destination = static_cast<std::byte*>(dst) + i * second_offset;
                auto* source      = static_cast<std::byte*>(src) + i * second_offset;
                if (integers) {
                    family::copy_i32_panel(
                        ninfer::Tensor(destination, ninfer::DType::I32, {16, batch}),
                        ninfer::Tensor(source, ninfer::DType::I32, {source_widths[i], batch}),
                        device.stream);
                } else {
                    family::copy_strided_width_panel(
                        ninfer::Tensor(destination, ninfer::DType::BF16,
                                       {channels, destination_widths[i], batch}),
                        ninfer::Tensor(source, ninfer::DType::BF16,
                                       {channels, source_widths[i], batch}),
                        device.stream);
                }
            });
    }
}

void check_replay_records(ninfer::DeviceContext& device, std::int32_t batch) {
    constexpr std::size_t second_offset = 4 * 1024 * 1024;
    std::array<ninfer::GdnReplayRecordLayout, 2> source_layouts;
    std::array<ninfer::GdnReplayRecordLayout, 2> destination_layouts;
    std::array<std::vector<RowCopy>, 2> copies;
    for (std::size_t i = 0; i < copies.size(); ++i) {
        ninfer::GdnReplayRecordSpec spec{.layers          = 1,
                                         .record_capacity = batch,
                                         .width           = i == 0 ? 4 : 8,
                                         .conv_channels   = 10240,
                                         .qk_heads        = 16,
                                         .value_heads     = 48,
                                         .key_dim         = 128,
                                         .value_dim       = 128};
        ninfer::LayoutBuilder source_builder;
        source_layouts[i]       = ninfer::plan_gdn_replay_records(source_builder, spec);
        const auto source_width = static_cast<std::size_t>(spec.width);
        spec.width              = 12;
        ninfer::LayoutBuilder destination_builder;
        destination_layouts[i] = ninfer::plan_gdn_replay_records(destination_builder, spec);
        const auto add_plane = [&](const ninfer::TensorRegion& dst, const ninfer::TensorRegion& src,
                                   std::size_t column_bytes) {
            copies[i].push_back({i * second_offset + dst.region.offset,
                                 i * second_offset + src.region.offset, 12 * column_bytes,
                                 source_width * column_bytes, source_width * column_bytes,
                                 static_cast<std::size_t>(batch)});
        };
        add_plane(destination_layouts[i].conv, source_layouts[i].conv, 10240 * 2);
        add_plane(destination_layouts[i].key, source_layouts[i].key, 128 * 16 * 2);
        add_plane(destination_layouts[i].value, source_layouts[i].value, 128 * 48 * 2);
        // B=3, W=4 reproduces the reported gate node: width1536, source pitch1536,
        // destination pitch4608, height3. The second profile changes workspace addresses/width.
        add_plane(destination_layouts[i].gate, source_layouts[i].gate, 2 * 48 * 4);
    }
    check_updates(device, "GDN replay records", copies, [&](std::size_t i, void* dst, void* src) {
        const auto& s = source_layouts[i];
        const auto& d = destination_layouts[i];
        ninfer::GdnReplayRecords source(
            {static_cast<std::byte*>(src) + i * second_offset, s.payload_bytes()}, s);
        ninfer::GdnReplayRecords destination(
            {static_cast<std::byte*>(dst) + i * second_offset, d.payload_bytes()}, d);
        ninfer::targets::qwen3_6::pack_replay_record_layer(destination.layer(0, batch),
                                                           source.layer(0, batch), device.stream);
    });
}

void check_fused_qkv(ninfer::DeviceContext& device, std::size_t columns) {
    constexpr std::size_t second_offset = 4 * 1024 * 1024;
    std::array<std::vector<RowCopy>, 2> copies;
    for (std::size_t i = 0; i < copies.size(); ++i) {
        const auto base = i * second_offset;
        copies[i]       = {{base, base, 8192, 12288, 8192, columns},
                           {base + columns * 8192, base + 8192, 2048, 12288, 2048, columns},
                           {base + columns * 10240, base + 10240, 2048, 12288, 2048, columns}};
    }
    check_updates(device, "fused QKV split", copies, [&](std::size_t i, void* dst, void* src) {
        for (const auto& copy : copies[i]) {
            ninfer::copy_device_rows_async(
                static_cast<std::byte*>(dst) + copy.destination_offset, copy.destination_pitch,
                static_cast<std::byte*>(src) + copy.source_offset, copy.source_pitch,
                copy.row_bytes, copy.rows, device.stream);
        }
    });
}

} // namespace

int main() {
    try {
        ninfer::DeviceContext device;
        for (std::int32_t batch = 1; batch <= 8; ++batch) {
            check_panels(device, batch);
            check_replay_records(device, batch);
            check_fused_qkv(device, 4 * static_cast<std::size_t>(batch));
            check_fused_qkv(device, 12 * static_cast<std::size_t>(batch));
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "panel copy graph update test failed: " << error.what() << '\n';
        return 1;
    }
}
