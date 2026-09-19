#include "targets/qwen4/native_ple_fetch.h"
#include "core/nvtx.h"

namespace ninfer::targets::qwen4 {

NativePleFetch::NativePleFetch() {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_,cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&ready_,cudaEventDisableTiming));
}
NativePleFetch::~NativePleFetch() {
    // Stream destruction alone does not wait for pinned/table/decoded storage consumers.
    (void)cudaStreamSynchronize(stream_);
    (void)cudaEventDestroy(ready_);
    (void)cudaStreamDestroy(stream_);
}
void NativePleFetch::synchronize() const {CUDA_CHECK(cudaStreamSynchronize(stream_));}
void NativePleFetch::enqueue(const NativePleTable& table,std::span<const std::int32_t> row_ids,
    int columns,void* pinned,std::size_t pinned_bytes,Tensor packed,Tensor decoded) {
    nvtx::ScopedRange range(nvtx::Name::Qwen4PleFetch,nvtx::Category::Control,columns);
    if(table.format==NativePleFormat::Nvfp4) {
        ops::ple_nvfp4_stage_rows_batch(table.nvfp4,row_ids,columns,pinned,pinned_bytes,packed,stream_);
        ops::ple_nvfp4_decode_rows(packed,decoded,stream_);
    } else {
        ops::ple_fp8_stage_rows_batch(table.fp8,row_ids,columns,pinned,pinned_bytes,packed,stream_);
        ops::ple_fp8_decode_rows(packed,table.fp8_scale_bits,decoded,stream_);
    }
    CUDA_CHECK(cudaEventRecord(ready_,stream_));
}

} // namespace ninfer::targets::qwen4
