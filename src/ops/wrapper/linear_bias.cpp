#include "ninfer/ops/linear_bias.h"
#include "ops/common/projection.h"
#include "ops/launcher/linear_bias.h"
#include <array>
#include <stdexcept>

namespace ninfer::ops {
void linear_bias(const Tensor& x,const Weight& w,const Tensor& bias,Tensor& out,cudaStream_t stream) {
    const auto shape=std::pair{w.n,w.k};
    constexpr std::array admitted{std::pair{1152,1536},std::pair{3456,1152},std::pair{1152,1152},
        std::pair{4304,1152},std::pair{1152,4304},std::pair{4608,4608},std::pair{2560,4608}};
    bool match=false; for(auto candidate:admitted) match|=candidate==shape;
    if(!match || w.qtype!=QType::BF16_CTRL || x.ne[1]<=0)
        throw std::invalid_argument("linear_bias: unsupported exact BF16 profile");
    detail::validate_native_projection(w,w.n,w.k,"linear_bias");
    auto require=[](const Tensor& t,int rows,int cols) {
        if(t.dtype!=DType::BF16 || !t.data || !t.is_contiguous() || t.ne[0]!=rows ||
           t.ne[1]!=cols || t.ne[2]!=1 || t.ne[3]!=1 || (reinterpret_cast<std::uintptr_t>(t.data)&15))
            throw std::invalid_argument("linear_bias: invalid tensor");
    };
    require(x,w.k,x.ne[1]); require(bias,w.n,1); require(out,w.n,x.ne[1]);
    const std::array regions{std::pair{x.data,x.bytes()},std::pair{bias.data,bias.bytes()},
        std::pair{out.data,out.bytes()},std::pair{const_cast<void*>(w.payload),std::size_t(w.payload_bytes)}};
    for(std::size_t i=0;i<regions.size();++i) for(std::size_t j=0;j<i;++j) {
        const auto a=reinterpret_cast<std::uintptr_t>(regions[i].first),b=reinterpret_cast<std::uintptr_t>(regions[j].first);
        if(a<b+regions[j].second && b<a+regions[i].second)
            throw std::invalid_argument("linear_bias: operands overlap");
    }
    detail::linear_bias_launch(x,w,bias,out,stream);
}
} // namespace ninfer::ops
