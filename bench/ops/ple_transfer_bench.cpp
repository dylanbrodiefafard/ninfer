// Public native PLE gather -> H2D -> GPU decode, with a RAM-locked random-access table.
#include "ninfer/ops/ple.h"
#include "core/arena.h"
#include "core/device.h"

#include <sys/mman.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
namespace {
using Clock = std::chrono::steady_clock;
double micros(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end-start).count();
}
struct ResidentTable {
    void* data = MAP_FAILED;
    std::size_t bytes;
    explicit ResidentTable(std::size_t count) : bytes(count) {
        data = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (data == MAP_FAILED) { throw std::runtime_error("PLE benchmark mmap failed"); }
        // Eagerly populate, then lock: successful construction admits no demand-paged table.
        std::memset(data, 0x11, bytes);
        if (mlock(data, bytes) != 0) {
            munmap(data, bytes); data = MAP_FAILED;
            throw std::runtime_error("PLE benchmark requires adequate RAM and memlock allowance");
        }
    }
    ~ResidentTable() { if (data != MAP_FAILED) { munlock(data, bytes); munmap(data, bytes); } }
    ResidentTable(const ResidentTable&) = delete;
    ResidentTable& operator=(const ResidentTable&) = delete;
};
struct Events {
    cudaEvent_t copied{}, ready{};
    Events() { CUDA_CHECK(cudaEventCreate(&copied)); CUDA_CHECK(cudaEventCreate(&ready)); }
    ~Events() { cudaEventDestroy(copied); cudaEventDestroy(ready); }
};
double percentile(std::vector<double> values, double quantile) {
    std::sort(values.begin(), values.end());
    return values[static_cast<std::size_t>((values.size()-1)*quantile)];
}
void measure(bool nvfp4, int width, int repetitions, std::size_t table_mib) {
    constexpr std::uint64_t partitions = 128;
    const std::size_t row_bytes = nvfp4 ? ops::kPleNvfp4RowBytes : ops::kPleRowWidth;
    const std::uint64_t rows_per_partition = (table_mib*1024*1024) / (partitions*row_bytes);
    const std::uint64_t rows = partitions*rows_per_partition;
    const std::size_t table_bytes = rows*row_bytes + (nvfp4 ? partitions*4 : 0);
    ResidentTable table(table_bytes);
    if (nvfp4) {
        for (std::uint64_t p=0;p<partitions;++p) {
            const float scale = (p&1) ? 0x1p-15F : 0x1p-16F;
            std::memcpy(static_cast<std::uint8_t*>(table.data)+rows*row_bytes+4*p,&scale,4);
        }
    }
    // Every iteration drains ready before slot reuse; the table outlives the device context.
    DeviceContext device(0);
    Events events;
    const int staged_row = nvfp4 ? ops::kPleNvfp4StagedRowBytes : ops::kPleRowWidth;
    PinnedHostBuffer pinned(static_cast<std::size_t>(staged_row)*16*width);
    DeviceBuffer packed(pinned.size()), decoded(static_cast<std::size_t>(2560)*width*2);
    Tensor bytes(packed.p, DType::U8, {staged_row,16,width});
    Tensor embedding(decoded.p, DType::BF16, {160,16,width});
    const ops::PleResidentFp8Table fp8{static_cast<const std::uint8_t*>(table.data),rows,rows*row_bytes};
    const ops::PleResidentNvfp4Table fp4{static_cast<const std::uint8_t*>(table.data),partitions,rows_per_partition,table_bytes};
    std::mt19937 random(19473);
    std::uniform_int_distribution<std::int32_t> index(0,static_cast<std::int32_t>(rows-1));
    std::vector<std::int32_t> ids(16*width);
    std::vector<double> submit, ready_wall, post_submit_wait, decode;
    for (int iteration=-10;iteration<repetitions;++iteration) {
        for (auto& id:ids) { id=index(random); }
        const auto begin=Clock::now();
        if (nvfp4) {
            ops::ple_nvfp4_stage_rows_batch(fp4,ids,width,pinned.data(),pinned.size(),bytes,device.copy_stream);
        } else {
            ops::ple_fp8_stage_rows_batch(fp8,ids,width,pinned.data(),pinned.size(),bytes,device.copy_stream);
        }
        const auto submitted=Clock::now();
        CUDA_CHECK(cudaEventRecord(events.copied,device.copy_stream));
        CUDA_CHECK(cudaStreamWaitEvent(device.stream,events.copied,0));
        if (nvfp4) { ops::ple_nvfp4_decode_rows(bytes,embedding,device.stream); }
        else { ops::ple_fp8_decode_rows(bytes,0x3951,embedding,device.stream); }
        CUDA_CHECK(cudaEventRecord(events.ready,device.stream));
        CUDA_CHECK(cudaEventSynchronize(events.ready));
        const auto end=Clock::now();
        float elapsed_ms=0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms,events.copied,events.ready));
        if (iteration>=0) {
            submit.push_back(micros(begin,submitted));
            ready_wall.push_back(micros(begin,end));
            post_submit_wait.push_back(micros(submitted,end));
            decode.push_back(elapsed_ms*1000.0);
        }
    }
    // This is a timing harness; exact codec/injection correctness lives in the independent tests.
    std::printf("PLE_TRANSFER format=%s T=%d resident_bytes=%zu transfer_bytes=%zu samples=%d "
                "submit_median_us=%.3f ready_median_us=%.3f ready_p95_us=%.3f "
                "post_submit_median_us=%.3f copy_ready_to_decode_ready_median_us=%.3f\n",
        nvfp4?"nvfp4":"fp8",width,table_bytes,pinned.size(),repetitions,
        percentile(submit,.5),percentile(ready_wall,.5),percentile(ready_wall,.95),
        percentile(post_submit_wait,.5),percentile(decode,.5));
}
}
int main(int argc,char** argv) {
    try {
        int width=1,repetitions=100,table_mib=512;
        for(int i=1;i<argc;++i) {
            const std::string option=argv[i];
            if(option=="--help") {
                std::cout<<"--t 1..4096 --repetitions 1..10000 --table-mib 1..1024\n"
                    <<"Both native codecs; eagerly RAM-locked synthetic table, changing random rows.\n"
                    <<"Timings include public validation and host gather; no first-layer overlap, "
                    <<"full-table capacity or model tok/s claim. Adequate memlock is mandatory.\n";
                return 0;
            }
            if(i+1==argc) { throw std::invalid_argument("missing benchmark option value"); }
            const int value=std::stoi(argv[++i]);
            if(option=="--t") { width=value; }
            else if(option=="--repetitions") { repetitions=value; }
            else if(option=="--table-mib") { table_mib=value; }
            else { throw std::invalid_argument("unknown benchmark option"); }
        }
        if(width<1||width>4096||repetitions<1||repetitions>10000||table_mib<1||table_mib>1024) {
            throw std::invalid_argument("benchmark option out of range");
        }
        for(bool nvfp4:{false,true}) { measure(nvfp4,width,repetitions,table_mib); }
        return 0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
