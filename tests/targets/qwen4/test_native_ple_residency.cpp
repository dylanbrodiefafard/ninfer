// Opt-in, real 28.8 GB host payload. Never part of ordinary resource-light tests.
#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/device.h"
#include "ninfer/ops/ple.h"
#include "ops/op_tester.h"

#include <array>
#include <fstream>
#include <sstream>
#include <sys/mman.h>
#include <unistd.h>

using namespace ninfer;
using namespace ninfer::test;
namespace {
constexpr std::uint64_t partitions=128, rows_per_partition=2500012;
constexpr std::uint64_t payload_bytes=partitions*rows_per_partition*90+partitions*4;

std::uint64_t proc_bytes(const char* file,const char* key) {
    std::ifstream input(file);
    std::string line;
    while(std::getline(input,line)) {
        if(line.starts_with(key)) return std::stoull(line.substr(std::strlen(key)))*1024;
    }
    throw std::runtime_error("missing Linux memory accounting");
}
std::uint64_t locked() { return proc_bytes("/proc/self/status","VmLck:"); }

std::uint64_t available_capacity() {
    const auto available=proc_bytes("/proc/meminfo","MemAvailable:");
    std::uint64_t reclaimable=0;
    // Explicit local qualification opt-in, never production memory admission.
    // OpenZFS 2.4.1 arc_evictable_memory/arc_shrinker_scan establish these data
    // counters as reclaimable. Discount half and preserve arc_c_min; count no
    // metadata, ghost history, dirty or otherwise unevictable ARC bytes.
    if(const auto* setting=std::getenv("NINFER_QWEN4_PLE_ZFS_ADMISSION"); setting && std::string(setting)=="1") {
        std::ifstream input("/proc/spl/kstat/zfs/arcstats");
        std::uint64_t mru=0,mfu=0,size=0,minimum=0;
        int found=0;
        std::string line;
        while(std::getline(input,line)) {
            std::istringstream fields(line);
            std::string name;int type;std::uint64_t value;
            if(!(fields>>name>>type>>value)) continue;
            if(name=="mru_evictable_data") {mru=value;found|=1;}
            if(name=="mfu_evictable_data") {mfu=value;found|=2;}
            if(name=="size") {size=value;found|=4;}
            if(name=="c_min") {minimum=value;found|=8;}
        }
        if(found!=15) throw std::runtime_error("required OpenZFS ARC accounting is unavailable");
        reclaimable=std::min((mru+mfu)/2,size>minimum?size-minimum:0);
        std::cout<<"ARC data mru="<<mru<<" mfu="<<mfu<<" size="<<size<<" minimum="<<minimum<<'\n';
    }
    std::cout<<"Memory admission: MemAvailable="<<available<<" discounted_ARC="<<reclaimable
             <<" required="<<payload_bytes+16ULL*1024*1024*1024<<std::endl;
    return available+reclaimable;
}

template<class T> std::vector<T> read(const artifact::Reader& reader,const char* name) {
    const auto bytes=reader.payload(name).data;
    std::vector<T> result(bytes.size()/sizeof(T));
    std::memcpy(result.data(),bytes.data(),bytes.size());
    return result;
}

void resident(std::span<const std::byte> bytes,std::size_t locked_bytes) {
    const auto page=static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    const auto address=reinterpret_cast<std::uintptr_t>(bytes.data());
    const auto base=address/page*page;
    const auto extent=(address-base+bytes.size()+page-1)/page*page;
    if(extent!=locked_bytes) throw std::runtime_error("resident lock extent differs");
    std::vector<unsigned char> flags(extent/page);
    if(::mincore(reinterpret_cast<void*>(base),extent,flags.data())!=0 ||
       !std::all_of(flags.begin(),flags.end(),[](auto value){return (value&1)!=0;}))
        throw std::runtime_error("complete PLE payload is not resident");
    std::cout<<"All "<<flags.size()<<" payload pages resident and "<<extent<<" bytes locked\n";
}

int run(const std::string& root) {
    // Hard guard, not a fallback: leave at least 16 GiB of host RAM available.
    constexpr std::uint64_t margin=16ULL*1024*1024*1024;
    if(available_capacity()<payload_bytes+margin)
        throw std::runtime_error("full PLE qualification needs payload plus 16 GiB available RAM");
    DeviceContext device;
    constexpr int max_width=16;
    PinnedHostBuffer pinned(94*16*max_width);
    DeviceBuffer packed(94*16*max_width), decoded(160*16*max_width*2);
    const auto baseline=locked();
    artifact::MaterializedArtifact loaded;
    artifact::ObjectHandle handle;
    {
        artifact::Reader reader(root+"/qwen4-ple-nvfp4.ninfer");
        if(reader.identity()!=artifact::ArtifactIdentity{"qwen4/native-ple-qualification",
                                                        "primitive-nvfp4-complete-table"})
            throw std::runtime_error("wrong complete native PLE fixture identity");
        artifact::Binder binder(reader);
        const std::array<std::uint64_t,3> shape{partitions,rows_per_partition,160};
        handle=binder.require_tensor("ple.table",artifact::NumericFormat::NVFP4_PARTITION_F32M,
                                    artifact::StorageLayout::PartitionedRowBlockScaleK16V1,shape);
        binder.map_tensor_on_host(handle,true);
        loaded=artifact::materialize(reader,binder.finish(),device);
    } // Reader destruction must not release the resident materialization owner.
    auto owner=std::move(loaded);
    const auto stats=owner.stats();
    if(stats.resident_tensor_bytes!=payload_bytes || stats.device_capacity_bytes!=0 ||
       stats.h2d_bytes!=0 || locked()!=baseline+stats.resident_locked_bytes)
        throw std::runtime_error("full PLE materialization residency accounting differs");
    const auto bytes=owner.mapped_tensor_bytes(handle);
    resident(bytes,stats.resident_locked_bytes);
    artifact::Reader reference(root+"/qwen4-ple-nvfp4-boundary-reference.ninfer");
    if(reference.identity()!=artifact::ArtifactIdentity{"qwen4/native-ple-reference","source-scalar-bf16"})
        throw std::runtime_error("wrong boundary source oracle identity");
    const auto source_ids=read<std::int32_t>(reference,"row_ids");
    const auto source_expected=read<std::uint16_t>(reference,"embedding");
    if(source_ids.size()!=256 || source_expected.size()!=256*160)
        throw std::runtime_error("incomplete boundary oracle");
    for(int part=0;part<128;++part) {
        if(source_ids[2*part]!=part*rows_per_partition ||
           source_ids[2*part+1]!=(part+1)*rows_per_partition-1)
            throw std::runtime_error("oracle must cover both ends of all 128 partitions");
    }
    const ops::PleResidentNvfp4Table table{
        reinterpret_cast<const std::uint8_t*>(bytes.data()),partitions,rows_per_partition,bytes.size()};
    int failures=0;
    for(int width:{16,3,1}) {
        std::vector<std::int32_t> ids(width*16);
        std::vector<std::uint16_t> expected(width*16*160);
        for(int i=0;i<width*16;++i) {
            const int source=width==16?i:(i*97+255)%256;
            ids[i]=source_ids[source];
            std::copy_n(source_expected.data()+source*160,160,expected.data()+i*160);
        }
        Tensor input(packed.p,DType::U8,{94,16,width});
        Tensor output(decoded.p,DType::BF16,{160,16,width});
        ops::ple_nvfp4_stage_rows_batch(table,ids,width,pinned.data(),pinned.size(),input,device.stream);
        ops::ple_nvfp4_decode_rows(input,output,device.stream);
        device.synchronize(); // owner and pinned staging retained through the consumer.
        std::vector<std::uint16_t> actual(expected.size());
        decoded.copy_to_host(actual.data(),actual.size()*2);
        failures+=verify_exact("complete resident PLE source boundary BF16 words",actual,expected);
    }
    if(locked()!=baseline+stats.resident_locked_bytes)
        throw std::runtime_error("residency lost during consumers");
    owner=artifact::MaterializedArtifact{};
    if(locked()!=baseline) throw std::runtime_error("PLE teardown leaked locked pages");
    if(!failures) std::cout<<"PASS full PLE materialization, moved-owner lifetime, bounded asynchronous H2D, "
                "all 128 partition edges, and lock teardown; payload_bytes="<<payload_bytes<<'\n';
    return failures;
}
}
int main() {
    const char* root=std::getenv("NINFER_QWEN4_FULL_PLE");
    if(!root) {std::cout<<"SKIP: opt-in complete 28.8 GB PLE qualification\n";return 77;}
    try {return run(root);} catch(const std::exception& error) {
        std::cerr<<"FAIL: "<<error.what()<<'\n';return 1;
    }
}
