#pragma once

#include "targets/qwen4/native_artifact.h"

namespace ninfer::targets::qwen4 {

// One startup-owned transfer lane for the native runtime's single compact batch.
// The caller owns the complete locked table and bounded pinned/device slots, and must
// drain BOTH this lane and the decoder before reusing or releasing those slots.
// No floating-point model work runs on the host: gathering copies represented bytes;
// the existing exact codec Op expands them on this lane's GPU stream.
class NativePleFetch {
public:
    NativePleFetch();
    ~NativePleFetch();
    NativePleFetch(const NativePleFetch&)=delete;
    NativePleFetch& operator=(const NativePleFetch&)=delete;

    void enqueue(const NativePleTable& table,std::span<const std::int32_t> row_ids,
        int columns,void* pinned,std::size_t pinned_bytes,Tensor packed,Tensor decoded);
    void synchronize() const;
    cudaEvent_t ready() const noexcept {return ready_;}
    // Borrowed for ordering diagnostics/event profiling only; this owner retains the stream.
    cudaStream_t stream() const noexcept {return stream_;}

private:
    cudaStream_t stream_=nullptr;
    cudaEvent_t ready_=nullptr;
};

} // namespace ninfer::targets::qwen4
