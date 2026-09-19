#pragma once

#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

namespace ninfer::test::qwen4_sequence {
inline bool native_timing_enabled() {
    const char* value=std::getenv("NINFER_QWEN4_NATIVE_TIMING");
    return value && std::string_view(value)=="1";
}

// Opt-in measurement on the same real fixture/input as the independent oracle test.
// Weights are resident. State restore is enqueued before, never inside, each interval.
// Restore once more before returning so the ordinary qualified call is unchanged.
template<class Restore,class Invoke>
void time_native_component(const std::string& label,Restore&& restore,Invoke&& invoke) {
    if(!native_timing_enabled()) { return; }
    cudaEvent_t begin,end;
    cuda_check(cudaEventCreate(&begin),"native timing create");
    cuda_check(cudaEventCreate(&end),"native timing create");
    std::array<float,11> samples{};
    for(int repetition=-3;repetition<int(samples.size());++repetition) {
        restore();
        cuda_check(cudaEventRecord(begin),"native timing begin");
        invoke();
        cuda_check(cudaEventRecord(end),"native timing end");
        cuda_check(cudaEventSynchronize(end),"native timing synchronize");
        float ms=0;
        cuda_check(cudaEventElapsedTime(&ms,begin,end),"native timing elapsed");
        if(repetition>=0) { samples[repetition]=1000.F*ms; }
    }
    restore();
    cuda_check(cudaEventDestroy(end),"native timing destroy");
    cuda_check(cudaEventDestroy(begin),"native timing destroy");
    std::sort(samples.begin(),samples.end());
    std::cout<<"NATIVE_COMPONENT "<<label<<" median_us="<<samples[5]
             <<" min_us="<<samples.front()<<" max_us="<<samples.back()<<'\n';
}
}
