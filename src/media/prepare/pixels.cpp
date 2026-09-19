#include "media/prepare/pixels.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ninfer::media::prepare {
namespace {
constexpr int kPatch=16, kTemporal=2;
struct Size { int h,w; };
int round_even(double value) { return static_cast<int>(std::nearbyint(value)); }
double cubic(double x) {
    // Torchvision's antialiased bicubic path uses the Keys/Pillow coefficient.
    constexpr double a = -0.5;
    x                  = std::abs(x);
    if (x < 1.0) { return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0; }
    if (x < 2.0) { return (((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a); }
    return 0.0;
}

template<class Scalar=float> struct Coefficients {
    std::vector<int> starts;
    std::vector<int> offsets;
    std::vector<Scalar> weights;
};

template<class Scalar=float> Coefficients<Scalar> coefficients(int input, int output) {
    Coefficients<Scalar> out;
    out.starts.resize(static_cast<std::size_t>(output));
    out.offsets.resize(static_cast<std::size_t>(output + 1));
    const double scale    = static_cast<double>(input) / output;
    const double invscale = scale >= 1.0 ? 1.0 / scale : 1.0;
    const double support  = 2.0 * (scale >= 1.0 ? scale : 1.0);
    for (int dst = 0; dst < output; ++dst) {
        const double center = scale * (dst + 0.5);
        const int begin     = std::max(static_cast<int>(center - support + 0.5), 0);
        const int size      = std::min(static_cast<int>(center + support + 0.5), input) - begin;
        out.starts[static_cast<std::size_t>(dst)]  = begin;
        out.offsets[static_cast<std::size_t>(dst)] = static_cast<int>(out.weights.size());
        double sum                                 = 0.0;
        for (int j = 0; j < size; ++j) {
            const double weight = cubic((j + begin - center + 0.5) * invscale);
            out.weights.push_back(static_cast<Scalar>(weight));
            sum += weight;
        }
        if (sum == 0.0) { throw std::runtime_error("bicubic resize produced zero weights"); }
        const int first = out.offsets[static_cast<std::size_t>(dst)];
        for (std::size_t i = static_cast<std::size_t>(first); i < out.weights.size(); ++i) {
            out.weights[i] = static_cast<Scalar>(out.weights[i] / sum);
        }
    }
    out.offsets[static_cast<std::size_t>(output)] = static_cast<int>(out.weights.size());
    return out;
}

} // namespace

media::decode::Image resize_bicubic(const media::decode::Image& input, int height, int width) {
    const Size size{height,width};
    if (input.width == size.w && input.height == size.h) { return input; }
    const auto horizontal = coefficients(input.width, size.w);
    const auto vertical   = coefficients(input.height, size.h);
    std::vector<std::uint8_t> temp(static_cast<std::size_t>(input.height) * size.w * 3);
    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < size.w; ++x) {
            const int first = horizontal.offsets[static_cast<std::size_t>(x)];
            const int last  = horizontal.offsets[static_cast<std::size_t>(x + 1)];
            for (int c = 0; c < 3; ++c) {
                float value = 0.0f;
                for (int i = first; i < last; ++i) {
                    const int source =
                        std::clamp(horizontal.starts[static_cast<std::size_t>(x)] + (i - first), 0,
                                   input.width - 1);
                    value +=
                        horizontal.weights[static_cast<std::size_t>(i)] *
                        input.rgb[(static_cast<std::size_t>(y) * input.width + source) * 3 + c];
                }
                temp[(static_cast<std::size_t>(y) * size.w + x) * 3 + c] =
                    static_cast<std::uint8_t>(std::clamp(round_even(value), 0, 255));
            }
        }
    }

    media::decode::Image out;
    out.width  = size.w;
    out.height = size.h;
    out.rgb.resize(static_cast<std::size_t>(size.h) * size.w * 3);
    for (int y = 0; y < size.h; ++y) {
        const int first = vertical.offsets[static_cast<std::size_t>(y)];
        const int last  = vertical.offsets[static_cast<std::size_t>(y + 1)];
        for (int x = 0; x < size.w; ++x) {
            for (int c = 0; c < 3; ++c) {
                float value = 0.0f;
                for (int i = first; i < last; ++i) {
                    const int source =
                        std::clamp(vertical.starts[static_cast<std::size_t>(y)] + (i - first), 0,
                                   input.height - 1);
                    value += vertical.weights[static_cast<std::size_t>(i)] *
                             temp[(static_cast<std::size_t>(source) * size.w + x) * 3 + c];
                }
                out.rgb[(static_cast<std::size_t>(y) * size.w + x) * 3 + c] =
                    static_cast<std::uint8_t>(std::clamp(round_even(value), 0, 255));
            }
        }
    }
    return out;
}

media::decode::Image resize_bicubic_uint8(const media::decode::Image& input,int height,int width) {
    if(input.height==height && input.width==width) return input;
    auto horizontal=coefficients<double>(input.width,width),vertical=coefficients<double>(input.height,height);
    // Each axis chooses the largest fractional precision whose positive coefficients fit i16.
    auto quantize=[](const auto& c) {
        const double maximum=*std::max_element(c.weights.begin(),c.weights.end());
        int precision=0;
        while(precision<22 && int(.5+maximum*double(1u<<(precision+1)))<32768) ++precision;
        std::vector<std::int16_t> weights; weights.reserve(c.weights.size());
        for(double value:c.weights) {
            const double scaled=value*double(1u<<precision);
            weights.push_back(static_cast<std::int16_t>(scaled+(scaled<0?-.5:.5)));
        }
        return std::pair{std::move(weights),precision};
    };
    const auto [hx,hbits]=quantize(horizontal); const auto [vy,vbits]=quantize(vertical);
    std::vector<std::uint8_t> intermediate(std::size_t(input.height)*width*3);
    for(int y=0;y<input.height;++y) for(int x=0;x<width;++x) for(int c=0;c<3;++c) {
        std::int64_t sum=std::int64_t{1}<<(hbits-1);
        const int first=horizontal.offsets[x],last=horizontal.offsets[x+1];
        for(int j=first;j<last;++j)
            sum+=std::int64_t(hx[j])*input.rgb[(std::size_t(y)*input.width+horizontal.starts[x]+j-first)*3+c];
        intermediate[(std::size_t(y)*width+x)*3+c]=static_cast<std::uint8_t>(std::clamp<std::int64_t>(sum>>hbits,0,255));
    }
    media::decode::Image out{width,height,std::vector<std::uint8_t>(std::size_t(height)*width*3)};
    for(int y=0;y<height;++y) for(int x=0;x<width;++x) for(int c=0;c<3;++c) {
        std::int64_t sum=std::int64_t{1}<<(vbits-1);
        const int first=vertical.offsets[y],last=vertical.offsets[y+1];
        for(int j=first;j<last;++j)
            sum+=std::int64_t(vy[j])*intermediate[(std::size_t(vertical.starts[y]+j-first)*width+x)*3+c];
        out.rgb[(std::size_t(y)*width+x)*3+c]=static_cast<std::uint8_t>(std::clamp<std::int64_t>(sum>>vbits,0,255));
    }
    return out;
}

float normalized(const media::decode::Image& image, int y, int x, int channel) {
    return static_cast<float>(
               image.rgb[(static_cast<std::size_t>(y) * image.width + x) * 3 + channel]) /
               127.5f -
           1.0f;
}

void append_temporal_patch16(std::span<const media::decode::Image* const> frames, int grid_y, int grid_x,
                  std::vector<float>& out) {
    for (int channel = 0; channel < 3; ++channel) {
        for (int temporal = 0; temporal < kTemporal; ++temporal) {
            const media::decode::Image& frame = *frames[static_cast<std::size_t>(temporal)];
            for (int y = 0; y < kPatch; ++y) {
                for (int x = 0; x < kPatch; ++x) {
                    out.push_back(
                        normalized(frame, grid_y * kPatch + y, grid_x * kPatch + x, channel));
                }
            }
        }
    }
}

} // namespace ninfer::media::prepare
