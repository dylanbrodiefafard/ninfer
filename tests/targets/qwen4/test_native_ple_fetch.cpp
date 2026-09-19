#include "targets/qwen4/native_ple_fetch.h"
#include "ops/ple_nvfp4_oracle.h"
#include "ops/ple_fp8_oracle.h"

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <mutex>
#include <sys/mman.h>

using namespace ninfer;
using namespace ninfer::test;
namespace q4=ninfer::targets::qwen4;
namespace {
constexpr int partitions=128,partition_rows=3,rows=partitions*partition_rows,max_columns=4096;
struct ResidentTable {
    std::uint8_t* data=nullptr;std::size_t bytes;bool nvfp4;
    explicit ResidentTable(bool nv):bytes(nv?rows*90+partitions*4:rows*160),nvfp4(nv) {
        data=static_cast<std::uint8_t*>(mmap(nullptr,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0));
        if(data==MAP_FAILED) throw std::runtime_error("PLE fetch test mapping failed");
        if(mlock(data,bytes)) {munmap(data,bytes);throw std::runtime_error("PLE fetch test requires eager resident table");}
        if(nv) {
            constexpr std::uint32_t multipliers[]{0x37b30c31,0x379f3cf3,0x3f808000,0x00400000,0x70000000};
            for(int p=0;p<partitions;++p) for(int b=0;b<4;++b) data[rows*90+p*4+b]=multipliers[p%5]>>(8*b);
            for(int r=0;r<rows;++r) {
                for(int j=0;j<80;++j) data[r*90+j]=((j+r)%16)|(((3*j+r)%16)<<4);
                for(int j=0;j<10;++j) data[r*90+80+j]=(r*10+j)%127;
            }
        } else for(std::size_t j=0;j<bytes;++j) {
            const auto ordinal=j%254;data[j]=ordinal<127?ordinal:ordinal+1;
        }
    }
    ~ResidentTable() {munlock(data,bytes);munmap(data,bytes);}
    q4::NativePleTable view() const {
        q4::NativePleTable v;v.format=nvfp4?q4::NativePleFormat::Nvfp4:q4::NativePleFormat::Fp8;
        if(nvfp4) v.nvfp4={data,partitions,partition_rows,bytes};
        else {v.fp8={data,rows,bytes};v.fp8_scale_bits=0x4001;}
        return v;
    }
};
struct Expected {
    std::vector<std::uint8_t> packed;
    std::vector<std::uint16_t> decoded;
};
Expected oracle(const ResidentTable& table,std::span<const int> ids) {
    const int row_bytes=table.nvfp4?94:160;Expected e;
    e.packed.resize(ids.size()*row_bytes);e.decoded.resize(ids.size()*160);
    for(std::size_t i=0;i<ids.size();++i) {
        auto* record=e.packed.data()+i*row_bytes;const int row=ids[i];
        if(table.nvfp4) {
            std::memcpy(record,table.data+row*90,90);
            std::memcpy(record+90,table.data+rows*90+(row/partition_rows)*4,4);
            for(int d=0;d<160;++d) e.decoded[i*160+d]=ple_nvfp4_oracle(record,d);
        } else {
            std::memcpy(record,table.data+row*160,160);
            for(int d=0;d<160;++d) e.decoded[i*160+d]=fp8_ple_oracle(record[d],table.view().fp8_scale_bits);
        }
    }
    return e;
}
// No CUDA API is called from a host callback. A controlled condition-variable gate makes
// the newly recorded ready generation incomplete while the consumer graph is launched.
struct TransferGate {
    cudaStream_t transfer_stream,consumer_stream;
    std::mutex mutex;std::condition_variable cv;bool entered=false,released=false,consumed=false;
    TransferGate(cudaStream_t transfer,cudaStream_t consumer):transfer_stream(transfer),consumer_stream(consumer) {}
    static void CUDART_CB producer(void* pointer) {
        auto& g=*static_cast<TransferGate*>(pointer);std::unique_lock lock(g.mutex);
        g.entered=true;g.cv.notify_all();g.cv.wait(lock,[&]{return g.released;});
    }
    static void CUDART_CB consumer(void* pointer) {
        auto& g=*static_cast<TransferGate*>(pointer);std::lock_guard lock(g.mutex);
        g.consumed=true;g.cv.notify_all();
    }
    void release() {std::lock_guard lock(mutex);released=true;cv.notify_all();}
    ~TransferGate() {
        release();
        // Keep callback-owned mutex/condition storage alive even when a check throws.
        (void)cudaStreamSynchronize(transfer_stream);(void)cudaStreamSynchronize(consumer_stream);
    }
};
struct Graph {
    cudaGraph_t definition=nullptr;cudaGraphExec_t executable=nullptr;
    ~Graph() {if(executable) cudaGraphExecDestroy(executable);if(definition) cudaGraphDestroy(definition);}
};
int run(bool nvfp4) {
    DeviceContext consumer;ResidentTable table(nvfp4);const int row_bytes=nvfp4?94:160;
    const auto packed_capacity=std::size_t(max_columns)*16*row_bytes,output_capacity=std::size_t(max_columns)*2560*2;
    PinnedHostBuffer pinned(packed_capacity+256);
    GuardedDeviceBuffer packed(packed_capacity),decoded(output_capacity),observed(output_capacity);
    int failures=0;
    // First-use CUDA module loading may synchronize the context. Warm the exact codec
    // before intentionally blocking a transfer stream in a host callback. This is direct
    // Op warmup, not fetch preparation: ready does not even exist yet, so the first graph
    // below still captures its wait before the first ready-event record.
    {
        const auto warm=oracle(table,std::vector<int>(16,0));
        CUDA_CHECK(cudaMemcpyAsync(packed.data(),warm.packed.data(),warm.packed.size(),cudaMemcpyHostToDevice,consumer.stream));
        Tensor encoded(packed.data(),DType::U8,{row_bytes,16,1});
        Tensor output(decoded.data(),DType::BF16,{160,16,1});
        if(nvfp4) ops::ple_nvfp4_decode_rows(encoded,output,consumer.stream);
        else ops::ple_fp8_decode_rows(encoded,table.view().fp8_scale_bits,output,consumer.stream);
        consumer.synchronize();
    }
    // Both buffers and the locked table outlive the fetch owner, including unconsumed teardown.
    {
        q4::NativePleFetch fetch;
        for(int batch=1;batch<=4;++batch) for(int width:{1,3,16}) {
            const int columns=batch*width;const std::size_t bytes=std::size_t(columns)*2560*2;
            Tensor encoded(packed.data(),DType::U8,{row_bytes,16,columns});
            Tensor output(decoded.data(),DType::BF16,{160,16,columns});
            Graph graph;
            CUDA_CHECK(cudaStreamBeginCapture(consumer.stream,cudaStreamCaptureModeThreadLocal));
            // The very first graph is captured before fetch.ready has ever been recorded.
            // External wait nodes must remain live across records from later generations.
            CUDA_CHECK(cudaStreamWaitEvent(consumer.stream,fetch.ready(),cudaEventWaitExternal));
            CUDA_CHECK(cudaMemcpyAsync(observed.data(),output.data,bytes,cudaMemcpyDeviceToDevice,consumer.stream));
            CUDA_CHECK(cudaStreamEndCapture(consumer.stream,&graph.definition));
            CUDA_CHECK(cudaGraphInstantiate(&graph.executable,graph.definition,nullptr,nullptr,0));
            for(int round=0;round<3;++round) {
                std::vector<int> ids(columns*16);
                for(std::size_t i=0;i<ids.size();++i) ids[i]=(i*17+round*101+i/16)%rows;
                ids[0]=round?rows-1:0;ids[1]=partition_rows-1;ids[2]=partition_rows;
                const auto expected=oracle(table,ids);
                std::memset(static_cast<std::byte*>(pinned.data())+expected.packed.size(),0x5a,256);
                TransferGate gate(fetch.stream(),consumer.stream);const bool delayed=batch==1 && width==1 && round<2;
                if(delayed) CUDA_CHECK(cudaLaunchHostFunc(fetch.stream(),TransferGate::producer,&gate));
                fetch.enqueue(table.view(),ids,columns,pinned.data(),expected.packed.size(),encoded,output);
                if(round==2) {
                    CUDA_CHECK(cudaStreamWaitEvent(consumer.stream,fetch.ready(),0));
                    CUDA_CHECK(cudaMemcpyAsync(observed.data(),output.data,bytes,cudaMemcpyDeviceToDevice,consumer.stream));
                } else CUDA_CHECK(cudaGraphLaunch(graph.executable,consumer.stream));
                if(delayed) {
                    CUDA_CHECK(cudaLaunchHostFunc(consumer.stream,TransferGate::consumer,&gate));
                    {
                        std::unique_lock lock(gate.mutex);
                        if(!gate.cv.wait_for(lock,std::chrono::seconds(5),[&]{return gate.entered;}))
                            throw std::runtime_error("PLE producer ordering callback did not start");
                        // A stale/omitted wait lets the consumer finish with old decoded rows.
                        // The timeout bounds this intentional incomplete producer, not GPU work.
                        if(gate.cv.wait_for(lock,std::chrono::milliseconds(50),[&]{return gate.consumed;})) ++failures;
                    }
                    gate.release();
                }
                consumer.synchronize();fetch.synchronize();
                failures+=verify_exact("native PLE fetch exact packed source bytes",
                    from_device<std::uint8_t>(packed.data(),expected.packed.size()),expected.packed);
                failures+=verify_exact("native PLE fetch independent codec across eager/graph generations",
                    from_device<std::uint16_t>(observed.data(),expected.decoded.size()),expected.decoded);
                for(int j=0;j<256;++j) if(static_cast<std::uint8_t*>(pinned.data())[expected.packed.size()+j]!=0x5a) {++failures;break;}
            }
        }
        // Maximum legal C1 envelope uses the same fixed bounded slot, visiting every
        // partition edge. Preparation may be cancelled without launching a consumer.
        std::vector<int> ids(max_columns*16);for(std::size_t i=0;i<ids.size();++i) ids[i]=i%rows;
        Tensor encoded(packed.data(),DType::U8,{row_bytes,16,max_columns});
        Tensor output(decoded.data(),DType::BF16,{160,16,max_columns});
        fetch.enqueue(table.view(),ids,max_columns,pinned.data(),packed_capacity,encoded,output);
        fetch.synchronize(); // cancelled prepared batch; safe bounded-slot reuse
        for(auto& id:ids) id=rows-1-id;
        fetch.enqueue(table.view(),ids,max_columns,pinned.data(),packed_capacity,encoded,output);
        // No synchronize here: destruction must drain the prepared, unconsumed transfer.
    }
    std::vector<int> final_ids(max_columns*16);for(std::size_t i=0;i<final_ids.size();++i) final_ids[i]=rows-1-i%rows;
    const auto expected=oracle(table,final_ids);
    failures+=verify_exact("native PLE unconsumed fetch destruction drains exact output",
        from_device<std::uint16_t>(decoded.data(),expected.decoded.size()),expected.decoded);
    failures+=packed.verify_guards("native PLE packed slot");
    failures+=decoded.verify_guards("native PLE decoded slot");
    failures+=observed.verify_guards("native PLE consumer slot");
    std::cout<<(nvfp4?"NVFP4":"FP8")<<" asynchronous PLE fetch failures="<<failures<<'\n';return failures;
}
}
int main() {
    if(const int unavailable=require_cuda()) return unavailable;
    try {const int failures=run(true)+run(false);return failures?1:0;}
    catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
