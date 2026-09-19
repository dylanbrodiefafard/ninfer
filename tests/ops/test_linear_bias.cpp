#include "ninfer/ops/linear_bias.h"
#include "ops/direct_bf16_weight.h"
#include <iostream>

using namespace ninfer;
using namespace ninfer::test;
int main() {
    int failures=0;
    for(auto [n,k]:std::array<std::pair<int,int>,7>{{{1152,1536},{3456,1152},{1152,1152},
            {4304,1152},{1152,4304},{4608,4608},{2560,4608}}}) {
        auto host=direct_bf16_weight::make_patterned(n,k,974);
        direct_bf16_weight::DeviceWeight weight(std::move(host));
        std::vector<float> bias(n); fill_uniform(bias,837,-2.F,2.F);round_to_bf16(bias);
        auto db=to_device_bf16(bias);Tensor tb(db.p,DType::BF16,{n});
        for(int tokens:{1,5}) {
            std::vector<float> x(k*tokens);fill_uniform(x,115+tokens,-1.F,1.F);round_to_bf16(x);
            auto dx=to_device_bf16(x);GuardedDeviceBuffer out(std::size_t(n)*tokens*2);
            Tensor tx(dx.p,DType::BF16,{k,tokens}),ty(out.data(),DType::BF16,{n,tokens});
            ops::linear_bias(tx,weight.view(),tb,ty,nullptr);
            std::vector<double> expected(std::size_t(n)*tokens);
            for(int t=0;t<tokens;++t) for(int row=0;row<n;++row)
                expected[t*n+row]=direct_bf16_weight::dot_fp64(weight.host,row,
                    std::span(x.data()+t*k,k))+bias[row];
            failures+=verify_reduction("LinearBias complete FP64 oracle",from_device_bf16(out.data(),expected.size()),
                expected,ReductionCriterion{1./256,1./256,2./256});
            failures+=out.verify_guards("LinearBias guards");
        }
        // Full-tile and one-column tail routes use distinct exact power-of-two column scales.
        // Linearity computes the complete FP64 formula once per output row, not a GPU reference.
        std::vector<float> base(k);fill_uniform(base,817,-1.F,1.F);round_to_bf16(base);
        std::vector<double> product(n);
        for(int row=0;row<n;++row) product[row]=direct_bf16_weight::dot_fp64(weight.host,row,base);
        constexpr std::array<float,7> scale{1.F,-.5F,2.F,0.F,-1.F,.5F,-2.F};
        for(int tokens:{128,129}) {
            std::vector<float> x(std::size_t(k)*tokens);
            std::vector<double> expected(std::size_t(n)*tokens);
            for(int t=0;t<tokens;++t) {
                for(int col=0;col<k;++col) x[t*k+col]=base[col]*scale[t%7];
                for(int row=0;row<n;++row) expected[t*n+row]=product[row]*scale[t%7]+bias[row];
            }
            auto dx=to_device_bf16(x);GuardedDeviceBuffer out(std::size_t(n)*tokens*2);
            Tensor tx(dx.p,DType::BF16,{k,tokens}),ty(out.data(),DType::BF16,{n,tokens});
            ops::linear_bias(tx,weight.view(),tb,ty,nullptr);
            failures+=verify_reduction("LinearBias full/tail FP64 oracle",from_device_bf16(out.data(),expected.size()),
                expected,ReductionCriterion{1./256,1./256,2./256});
            failures+=out.verify_guards("LinearBias full/tail guards");
        }
        failures+=weight.verify_preserved("LinearBias source unchanged");
    }
    std::cout<<(failures?"FAIL":"OK")<<" LinearBias\n";
    return failures?1:0;
}
