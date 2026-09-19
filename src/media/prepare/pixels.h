#pragma once
#include "media/decode/decode.h"
#include <span>
#include <vector>

namespace ninfer::media::prepare {
// Shared antialiased Keys bicubic RGB8 preparation. Target policy owns resized dimensions.
// Exact existing uint8 horizontal/vertical rounding behavior is retained.
decode::Image resize_bicubic(const decode::Image&,int height,int width);
// Source Torchvision RGB8 profile: FP64 coefficient construction, signed16bit coefficient
// quantization and integer half-up clipping after each separable pass. Qwen4's pinned
// processor selects this profile; the existing family retains its qualified floating profile.
decode::Image resize_bicubic_uint8(const decode::Image&,int height,int width);
// Two already resized RGB8 temporal frames, normalized with mean/std0.5 after rescale1/255.
// Appends source [channel,temporal,y,x] for one16x16patch. Target owns merge-grid traversal.
void append_temporal_patch16(std::span<const decode::Image* const> frames,int grid_y,int grid_x,
                             std::vector<float>& output);
} // namespace ninfer::media::prepare
