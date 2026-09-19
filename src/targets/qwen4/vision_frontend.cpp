#include "targets/qwen4/vision_frontend.h"
#include "media/prepare/pixels.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen4 {
namespace {
void validate(const media::decode::Image& image) {
    if(image.width<=0 || image.height<=0 || image.rgb.size()!=std::size_t(image.width)*image.height*3)
        throw std::invalid_argument("Qwen4 Vision requires complete decoded RGB8 pixels");
}
struct Size { int height,width; };
void validate_grid(const VisionGrid& grid) {
    if(grid.temporal<=0 || grid.height<=0 || grid.width<=0 || grid.height%2 || grid.width%2 ||
       std::int64_t(grid.height)*grid.width>std::numeric_limits<int>::max())
        throw std::invalid_argument("Qwen4 multimodal grid is invalid");
}
Size resize_shape(int height,int width,int frames,bool video) {
    if(height<=0 || width<=0 || frames<=0) throw std::invalid_argument("invalid Qwen4 media dimensions");
    if(video && (height<32 || width<32)) {
        const double scale=std::max(32.0/height,32.0/width);
        height=static_cast<int>(height*scale); width=static_cast<int>(width*scale);
    }
    if(double(std::max(height,width))/std::min(height,width)>200.)
        throw std::invalid_argument("Qwen4 media aspect ratio exceeds200");
    int h=static_cast<int>(std::nearbyint(height/32.))*32;
    int w=static_cast<int>(std::nearbyint(width/32.))*32;
    const int t=video?static_cast<int>(std::nearbyint(frames/2.))*2:1;
    const double min=video?4096.:65536.,max=video?25165824.:16777216.;
    const double volume=double(t)*h*w;
    if(volume>max) {
        const double beta=std::sqrt(double(frames)*height*width/max);
        h=std::max(32,static_cast<int>(std::floor(height/beta/32.))*32);
        w=std::max(32,static_cast<int>(std::floor(width/beta/32.))*32);
    } else if(volume<min) {
        const double beta=std::sqrt(min/(double(frames)*height*width));
        h=static_cast<int>(std::ceil(height*beta/32.))*32;
        w=static_cast<int>(std::ceil(width*beta/32.))*32;
    }
    if(h<=0 || w<=0) throw std::invalid_argument("Qwen4 resize produced empty image");
    return {h,w};
}
void append_frame_pair(const std::array<const media::decode::Image*,2>& pair,VisionPixels& result) {
    for(int y=0;y<result.grid.height;y+=2) for(int x=0;x<result.grid.width;x+=2)
        for(int dy=0;dy<2;++dy) for(int dx=0;dx<2;++dx)
            media::prepare::append_temporal_patch16(pair,y+dy,x+dx,result.patches);
}
}

VisionPixels prepare_vision_image(const media::decode::Image& image) {
    validate(image); const auto shape=resize_shape(image.height,image.width,1,false);
    auto resized=media::prepare::resize_bicubic_uint8(image,shape.height,shape.width);
    VisionPixels result{{1,shape.height/16,shape.width/16},{},{}};
    result.patches.reserve(std::size_t(result.grid.height)*result.grid.width*1536);
    append_frame_pair({&resized,&resized},result); return result;
}

VisionPixels prepare_vision_video(const media::decode::Video& video) {
    if(video.frames.size()<2 || video.frames.size()>768 || video.indices.size()!=video.frames.size() ||
       !(video.fps>0) || !std::isfinite(video.fps))
        throw std::invalid_argument("Qwen4 video requires sampled frames and source timestamp metadata");
    for(const auto& frame:video.frames) {
        validate(frame);
        if(frame.width!=video.width || frame.height!=video.height)
            throw std::invalid_argument("Qwen4 video frame dimensions differ");
    }
    const auto shape=resize_shape(video.height,video.width,static_cast<int>(video.frames.size()),true);
    VisionPixels result{{static_cast<int>((video.frames.size()+1)/2),shape.height/16,shape.width/16},{},{}};
    result.patches.reserve(std::size_t(result.grid.temporal)*result.grid.height*result.grid.width*1536);
    for(std::size_t t=0;t<video.frames.size();t+=2) {
        const auto last=std::min(t+1,video.frames.size()-1);
        auto first_frame=media::prepare::resize_bicubic_uint8(video.frames[t],shape.height,shape.width);
        auto second_frame=media::prepare::resize_bicubic_uint8(video.frames[last],shape.height,shape.width);
        append_frame_pair({&first_frame,&second_frame},result);
        result.timestamps.push_back((double(video.indices[t])+video.indices[last])/(2.*video.fps));
    }
    return result;
}

VisionPixels prepare_vision_image(std::span<const std::uint8_t> bytes,const media::decode::Policy& policy) {
    return prepare_vision_image(media::decode::decode_image(bytes,policy));
}
VisionPixels prepare_vision_video(std::span<const std::uint8_t> bytes,const media::decode::Policy& policy) {
    return prepare_vision_video(media::decode::decode_video(bytes,policy,2.,4,768));
}

MultimodalPositions prepare_multimodal_positions(std::span<const std::uint8_t> types,
        std::span<const VisionGrid> images,std::span<const VisionGrid> videos) {
    if(types.empty() || types.size()>std::size_t(std::numeric_limits<int>::max()/4))
        throw std::invalid_argument("Qwen4 position sequence extent invalid");
    std::vector<VisionGrid> video_frames;
    for(auto grid:videos) {
        validate_grid(grid);
        if(std::size_t(grid.temporal)>types.size())
            throw std::invalid_argument("Qwen4 video has more temporal grids than tokens");
        for(int t=0;t<grid.temporal;++t) video_frames.push_back({1,grid.height,grid.width});
    }
    for(auto grid:images) {
        if(grid.temporal!=1) throw std::invalid_argument("Qwen4 image grid temporal extent must be1");
        validate_grid(grid);
    }
    const int length=static_cast<int>(types.size());
    MultimodalPositions result; result.values.resize(std::size_t(length)*4);
    std::size_t image=0,video=0; int current=0,maximum=0;
    for(int i=0;i<length;++i) result.values[i]=i;
    for(int begin=0;begin<length;) {
        int end=begin+1; while(end<length && types[end]==types[begin]) ++end;
        if(types[begin]==0) {
            for(int i=begin;i<end;++i) for(int axis=1;axis<4;++axis)
                result.values[axis*length+i]=current+i-begin;
            current+=end-begin; maximum=std::max(maximum,current-1);
        } else {
            VisionGrid grid{};
            if(types[begin]==1 && image<images.size()) grid=images[image++];
            else if(types[begin]==2 && video<video_frames.size()) grid=video_frames[video++];
            else throw std::invalid_argument("Qwen4 multimodal type run has no matching grid");
            const int h=grid.height/2,w=grid.width/2;
            if(end-begin!=h*w) throw std::invalid_argument("Qwen4 visual token count differs from merged grid");
            for(int y=0;y<h;++y) for(int x=0;x<w;++x) {
                const int i=begin+y*w+x;
                (types[begin]==1?result.image_columns:result.video_columns).push_back(i);
                result.values[length+i]=current;
                result.values[2*length+i]=current+y;
                result.values[3*length+i]=current+x;
            }
            current+=std::max(h,w); maximum=std::max(maximum,current-1);
        }
        begin=end;
    }
    if(image!=images.size() || video!=video_frames.size())
        throw std::invalid_argument("Qwen4 unconsumed multimodal grids");
    result.rope_delta=maximum+1-length;
    return result;
}
} // namespace ninfer::targets::qwen4
