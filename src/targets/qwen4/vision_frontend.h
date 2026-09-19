#pragma once
#include "targets/qwen4/vision.h"
#include "media/decode/decode.h"
#include <span>
#include <vector>

namespace ninfer::targets::qwen4 {
struct VisionPixels {
    VisionGrid grid;
    std::vector<float> patches; // [P,1536], source FP32 normalization, cast to BF16 by consumer.
    std::vector<double> timestamps; // One value per temporal video patch; empty for images.
};
// Exact pinned source pixel limits. Inputs are decoded owned RGB8 data; acquisition/decoding
// remain separate. Video frames must already be sampled with the source policy (fps2,min4,max768).
[[nodiscard]] VisionPixels prepare_vision_image(const media::decode::Image&);
[[nodiscard]] VisionPixels prepare_vision_video(const media::decode::Video&);
[[nodiscard]] VisionPixels prepare_vision_image(std::span<const std::uint8_t> bytes,
                                                const media::decode::Policy&);
[[nodiscard]] VisionPixels prepare_vision_video(std::span<const std::uint8_t> bytes,
                                                const media::decode::Policy&);

struct MultimodalPositions {
    std::vector<std::int32_t> values; // [4,L]: causal, temporal, height, width.
    std::int32_t rope_delta = 0;
    std::vector<std::int32_t> image_columns, video_columns; // Separate source feature scatter maps.
};
// Source token-type runs:0=text,1=image,2=video. Video grids expand into one run per temporal
// patch; surrounding timestamp/vision-boundary text belongs to type0. Exact run lengths must
// equal merged grid sizes. This does not tokenize/render templates or invent special-token ids.
[[nodiscard]] MultimodalPositions prepare_multimodal_positions(
    std::span<const std::uint8_t> token_types,std::span<const VisionGrid> images,
    std::span<const VisionGrid> videos);
} // namespace ninfer::targets::qwen4
