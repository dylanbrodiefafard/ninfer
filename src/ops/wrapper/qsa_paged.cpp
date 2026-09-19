#include "ninfer/ops/qsa.h"
#include "ops/launcher/qsa_paged.h"
#include "ops/wrapper/qsa_validation.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::ops {
namespace {
void tensor(const Tensor& t, DType type, int a, int b, int c, int d,
            const char* name, int alignment=4) {
    if (!t.data || t.dtype!=type || !t.is_contiguous() || t.ne[0]!=a ||
        t.ne[1]!=b || t.ne[2]!=c || t.ne[3]!=d ||
        reinterpret_cast<std::uintptr_t>(t.data)%alignment) {
        throw std::invalid_argument(std::string("qsa native: invalid ")+name);
    }
}
// All primitive operands, including read-only controls, are deliberately disjoint.
void disjoint(const QsaPagedStateView& s, const QsaBatchControls& c,
              std::initializer_list<const Tensor*> operands) {
    std::vector<detail::QsaAddressRange> ranges;
    for (const Tensor* t : {&s.k,&s.v,&s.k_scales,&s.v_scales,&s.raw_index_keys,
                           &s.positions,&s.block_tables,&c.table_rows,&c.valid_columns,
                           &c.frontiers,&c.positions}) {
        if(t->data) ranges.push_back(detail::qsa_address_range(*t,"qsa native","state/control"));
    }
    for(const Tensor* t:operands)
        ranges.push_back(detail::qsa_address_range(*t,"qsa native","operand"));
    for(std::size_t i=0;i<ranges.size();++i) for(std::size_t j=0;j<i;++j)
        if(ranges[i].begin<ranges[j].end && ranges[j].begin<ranges[i].end)
            throw std::invalid_argument("qsa native: overlapping storage");
}
}
namespace detail {
void qsa_validate_batch(int w,int b) {
    if(b<1 || b>4 || w<1 || w>(b==1?4096:16))
        throw std::invalid_argument("qsa native: B=1 W<=4096 or B=2..4 W<=16 required");
}
void qsa_validate_paged(const QsaPagedStateView& s,const QsaBatchControls& c,
    int w,int b,int maximum,const char*) {
    qsa_validate_batch(w,b);
    const bool bf=s.format==QsaKvFormat::BF16;
    if(!bf && s.format!=QsaKvFormat::NVFP4G16)
        throw std::invalid_argument("qsa native: invalid format");
    const int pages=s.k.ne[3], logical=s.block_tables.ne[0], slots=s.block_tables.ne[1];
    if(pages<1 || logical<1 || logical>4096 || slots<b || slots>4 ||
        maximum<1 || maximum>logical*64)
        throw std::invalid_argument("qsa native: invalid page geometry/envelope");
    tensor(s.k,bf?DType::BF16:DType::U8,bf?256:128,64,2,pages,"k");
    tensor(s.v,bf?DType::BF16:DType::U8,bf?256:128,64,2,pages,"v");
    if(bf) {
        if(s.k_scales.data || s.v_scales.data)
            throw std::invalid_argument("qsa native: BF16 scales must be absent");
    } else {
        tensor(s.k_scales,DType::FP8_E4M3FN,16,64,2,pages,"k scales",1);
        tensor(s.v_scales,DType::FP8_E4M3FN,16,64,2,pages,"v scales",1);
    }
    tensor(s.raw_index_keys,DType::BF16,128,64,1,pages,"raw keys",2);
    tensor(s.positions,DType::I32,3,64,1,pages,"state positions");
    tensor(s.block_tables,DType::I32,logical,slots,1,1,"tables");
    tensor(c.table_rows,DType::I32,b,1,1,1,"table rows");
    tensor(c.valid_columns,DType::I32,b,1,1,1,"valid columns");
    tensor(c.frontiers,DType::I32,b,1,1,1,"frontiers");
    tensor(c.positions,DType::I32,3,w,b,1,"positions");
    disjoint(s,c,{});
}
}
std::size_t qsa_index_select_workspace_bytes(int w,int b) {
    detail::qsa_validate_batch(w,b);
    return static_cast<std::size_t>(w)*b*detail::kQsaPagedSelectorWords*sizeof(float);
}
void qsa_state_append(const Tensor& k,const Tensor& v,const Tensor& raw,
    const QsaBatchControls& c,QsaPagedStateView s,int maximum,cudaStream_t stream) {
    const int w=k.ne[2],b=k.ne[3];
    detail::qsa_validate_paged(s,c,w,b,maximum,"qsa_state_append");
    tensor(k,DType::BF16,256,2,w,b,"k input",16);
    tensor(v,DType::BF16,256,2,w,b,"v input",16);
    tensor(raw,DType::BF16,128,w,b,1,"raw input",2);
    disjoint(s,c,{&k,&v,&raw});
    detail::qsa_paged_append_launch(k,v,raw,c,s,stream);
}
void qsa_index_select(const Tensor& q,const QsaPagedStateView& s,
    const QsaBatchControls& c,int maximum,const Tensor& qw,const Tensor& kw,
    Tensor& ids,Tensor& counts,Tensor& scratch,cudaStream_t stream) {
    const int w=q.ne[2],b=q.ne[3];
    detail::qsa_validate_paged(s,c,w,b,maximum,"qsa_index_select");
    tensor(q,DType::BF16,128,4,w,b,"query",2);
    tensor(qw,DType::FP32,128,1,1,1,"query norm");
    tensor(kw,DType::FP32,128,1,1,1,"key norm");
    tensor(ids,DType::I32,2051,w,b,1,"selection");
    tensor(counts,DType::I32,w,b,1,1,"counts");
    tensor(scratch,DType::U8,scratch.ne[0],1,1,1,"workspace",256);
    if(scratch.bytes()<qsa_index_select_workspace_bytes(w,b))
        throw std::invalid_argument("qsa native: selector workspace too small");
    disjoint(s,c,{&q,&qw,&kw,&ids,&counts,&scratch});
    detail::qsa_paged_select_launch(q,s,c,maximum,qw,kw,ids,counts,scratch,stream);
}
void qsa_selected_attention(const Tensor& q,const Tensor& ids,const Tensor& counts,
    const QsaPagedStateView& s,const QsaBatchControls& c,int maximum,
    Tensor& out,cudaStream_t stream) {
    const int w=q.ne[2],b=q.ne[3],bound=ids.ne[0];
    detail::qsa_validate_paged(s,c,w,b,maximum,"qsa_selected_attention");
    if(bound<1 || bound>2051) throw std::invalid_argument("qsa native: selected bound");
    tensor(q,DType::BF16,256,24,w,b,"query",2);
    tensor(ids,DType::I32,bound,w,b,1,"selection");
    tensor(counts,DType::I32,w,b,1,1,"counts");
    tensor(out,DType::BF16,256,24,w,b,"out",2);
    disjoint(s,c,{&q,&ids,&counts,&out});
    detail::qsa_paged_attention_launch(q,ids,counts,s,c,out,stream);
}
}
