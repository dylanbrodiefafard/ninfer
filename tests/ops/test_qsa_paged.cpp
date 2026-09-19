#include "ninfer/ops/qsa.h"
#include "ops/op_tester.h"

#include <array>
#include <numeric>

using namespace ninfer;
using namespace ninfer::test;
namespace {
constexpr ReductionCriterion kAttentionCriterion{1.0/256,1e-6,1.0/128};
struct State {
    int logical,slots,pages;
    bool bf;
    std::vector<int> tables;
    DeviceBuffer k,v,ks,vs,raw,pos,table;
    State(int capacity,int slots,bool bf,int mapping)
        :logical((capacity+63)/64),slots(slots),pages(logical*slots+3),bf(bf),
         tables(logical*slots),k(std::size_t(bf?512:128)*64*2*pages),v(k.bytes),
         ks(std::size_t(16)*64*2*pages),vs(ks.bytes),raw(std::size_t(256)*64*pages),
         pos(std::size_t(12)*64*pages),table(tables.size()*4) {
        for(int i=0;i<logical*slots;++i)
            tables[i]=mapping==0?i:mapping==1?i+3:pages-1-i;
        for(auto* d:{&k,&v,&ks,&vs,&raw,&pos}) d->fill(0);
        table.copy_from_host(tables.data(),table.bytes);
    }
    ops::QsaPagedStateView view() {
        return {bf?ops::QsaKvFormat::BF16:ops::QsaKvFormat::NVFP4G16,
            Tensor(k.p,bf?DType::BF16:DType::U8,{bf?256:128,64,2,pages}),
            Tensor(v.p,bf?DType::BF16:DType::U8,{bf?256:128,64,2,pages}),
            bf?Tensor{}:Tensor(ks.p,DType::FP8_E4M3FN,{16,64,2,pages}),
            bf?Tensor{}:Tensor(vs.p,DType::FP8_E4M3FN,{16,64,2,pages}),
            Tensor(raw.p,DType::BF16,{128,64,1,pages}),Tensor(pos.p,DType::I32,{3,64,1,pages}),
            Tensor(table.p,DType::I32,{logical,slots})};
    }
    std::size_t meta(int slot,int id) const { return id%64+64ULL*tables[id/64+logical*slot]; }
    std::size_t row(int slot,int id,int head) const {
        return id%64+64ULL*(head+2ULL*tables[id/64+logical*slot]);
    }
};
struct Control {
    int w,b;
    std::vector<int> rows,valid,frontier,position;
    DeviceBuffer dr,dv,df,dp;
    Control(int w,std::vector<int> rows,std::vector<int> valid,std::vector<int> frontier)
        :w(w),b(rows.size()),rows(std::move(rows)),valid(std::move(valid)),
         frontier(std::move(frontier)),position(3*w*b),dr(b*4),dv(b*4),df(b*4),dp(3*w*b*4) {
        for(int batch=0;batch<b;++batch) for(int j=0;j<w;++j) for(int a=0;a<3;++a)
            position[a+3*(j+w*batch)]=(this->frontier[batch]+j)*(a+1)+7*a;
        upload();
    }
    void upload() {
        dr.copy_from_host(rows.data(),dr.bytes);dv.copy_from_host(valid.data(),dv.bytes);
        df.copy_from_host(frontier.data(),df.bytes);dp.copy_from_host(position.data(),dp.bytes);
    }
    ops::QsaBatchControls view() {
        return {Tensor(dr.p,DType::I32,{b}),Tensor(dv.p,DType::I32,{b}),
            Tensor(df.p,DType::I32,{b}),Tensor(dp.p,DType::I32,{3,w,b})};
    }
};
double e4(int code) {
    const int exponent=(code>>3)&15,mantissa=code&7;
    return exponent==0?std::ldexp(double(mantissa),-9):
        std::ldexp(1.+double(mantissa)/8.,exponent-7);
}
double e2(int code) {
    constexpr double mag[]{0,.5,1,1.5,2,3,4,6};
    return (code&8)?-mag[code&7]:mag[code&7];
}
template<class Decode> int nearest(double v,int count,Decode decode) {
    int result=0;double error=INFINITY;
    for(int c=0;c<count;++c) {
        const double next=std::abs(v-decode(c));
        if(next<error || (next==error && !(c&1))) { result=c;error=next; }
    }
    return result;
}
void encode(std::span<const float> input,std::size_t row,bool bf,
    std::vector<unsigned char>& data,std::vector<unsigned char>& scale) {
    if(bf) {
        for(int d=0;d<256;++d) {
            const auto bits=f32_to_bf16(input[d]);
            const auto offset=2*(d+256*row);data[offset]=bits&255;data[offset+1]=bits>>8;
        }
        return;
    }
    for(int g=0;g<16;++g) {
        float maximum=0;
        for(int d=0;d<16;++d) maximum=std::max(maximum,std::abs(input[16*g+d]));
        const int code=nearest(float(maximum/6.F),127,e4);
        scale[g+16*row]=code;
        for(int pair=0;pair<8;++pair) {
            int packed=0;
            for(int i=0;i<2;++i) {
                const float normalized=code?input[16*g+2*pair+i]/float(e4(code)):0.F;
                packed|=(nearest(std::abs(normalized),8,e2)|(std::signbit(normalized)?8:0))<<(4*i);
            }
            data[8*g+pair+128*row]=packed;
        }
    }
}
double decode(const std::vector<unsigned char>& data,const std::vector<unsigned char>& scale,
    std::size_t row,int d,bool bf) {
    if(bf) { const auto i=2*(d+256*row);return bf16_to_f32(data[i]|(data[i+1]<<8)); }
    const int packed=data[d/2+128*row];
    return e2((packed>>(4*(d%2)))&15)*e4(scale[d/16+16*row]);
}
std::array<double,128> norm_rope(std::array<double,128> x,const std::vector<float>& gamma,
    const int* position) {
    double square=0;for(double value:x) square+=value*value;
    const double inv=1/std::sqrt(square/128.+1.e-6);
    for(int d=0;d<128;++d) x[d]*=inv*gamma[d];
    const auto unrotated=x;
    for(int pair=0;pair<32;++pair) {
        const double phase=position[pair%3]*std::pow(1.e7,-double(pair)/32);
        x[pair]=unrotated[pair]*std::cos(phase)-unrotated[pair+32]*std::sin(phase);
        x[pair+32]=unrotated[pair+32]*std::cos(phase)+unrotated[pair]*std::sin(phase);
    }
    return x;
}
std::vector<int> select_oracle(const State& s,const Control& c,int token,
    const std::vector<float>& query,const std::vector<float>& qgamma,
    const std::vector<float>& kgamma,const std::vector<std::uint16_t>& raw,
    const std::vector<int>& positions) {
    const int b=token/c.w,j=token%c.w;
    if(j>=c.valid[b]) return {};
    const int visible=c.frontier[b]+j+1,blocks=visible/4;
    std::array<std::array<double,128>,4> q;
    for(int h=0;h<4;++h) {
        for(int d=0;d<128;++d) q[h][d]=query[d+128*(h+4*token)];
        q[h]=norm_rope(q[h],qgamma,c.position.data()+3*token);
    }
    std::vector<double> scores(blocks);
    for(int block=0;block<blocks;++block) {
        std::array<double,128> key{};
        for(int d=0;d<128;++d) {
            float sum=0;
            for(int r=0;r<4;++r) sum+=bf16_to_f32(raw[d+128*s.meta(c.rows[b],4*block+r)]);
            key[d]=bf16_to_f32(f32_to_bf16(sum*.25F));
        }
        key=norm_rope(key,kgamma,positions.data()+3*s.meta(c.rows[b],4*block));
        for(int h=0;h<4;++h) {
            double dot=0;for(int d=0;d<128;++d) dot+=q[h][d]*key[d];
            scores[block]+=std::max(0.,dot)/std::sqrt(128.);
        }
    }
    std::vector<int> ranks(blocks);std::iota(ranks.begin(),ranks.end(),0);
    std::sort(ranks.begin(),ranks.end(),[&](int a,int b) {
        return scores[a]>scores[b] || (scores[a]==scores[b] && a<b);
    });
    std::vector<int> result;
    for(int rank=0;rank<std::min(512,blocks);++rank)
        for(int r=0;r<4;++r) result.push_back(4*ranks[rank]+r);
    for(int id=4*blocks;id<visible;++id) result.push_back(id);
    return result;
}
int selection_case(int maximum,int w,int batch,int mapping,bool ties,bool graph) {
    State state(maximum,batch,true,mapping);
    std::vector<int> rows(batch),valid(batch,w),front(batch,maximum-w);
    for(int b=0;b<batch;++b) { rows[b]=batch-1-b;front[b]-=b*67; }
    if(batch>1) valid[1]=0;
    if(batch>2) valid[2]=w/2;
    Control control(w,rows,valid,front);
    std::vector<std::uint16_t> raw(state.raw.bytes/2);
    std::vector<int> positions(state.pos.bytes/4);
    for(int b=0;b<batch;++b) for(int id=0;id<maximum;++id) {
        const auto m=state.meta(b,id);
        for(int d=0;d<128;++d) raw[d+128*m]=ties?0:f32_to_bf16(
            .3F*std::sin(float((id/4+1)*(d+3)+b*11)*.071F)+.01F*(id%4));
        for(int a=0;a<3;++a) positions[a+3*m]=id*(a+1)+7*a;
    }
    state.raw.copy_from_host(raw.data(),state.raw.bytes);state.pos.copy_from_host(positions.data(),state.pos.bytes);
    std::vector<float> query(512*w*batch),qgamma(128),kgamma(128);
    fill_uniform(query,1729,-1,1);round_to_bf16(query);
    for(int d=0;d<128;++d) { qgamma[d]=.7F+.002F*d;kgamma[d]=1.1F-.001F*d; }
    auto dq=to_device_bf16(query),dqw=to_device(qgamma),dkw=to_device(kgamma);
    GuardedDeviceBuffer ids(2051*w*batch*4),counts(w*batch*4),
        scratch(ops::qsa_index_select_workspace_bytes(w,batch));
    Tensor qt(dq.p,DType::BF16,{128,4,w,batch}),qw(dqw.p,DType::FP32,{128}),kw(dkw.p,DType::FP32,{128}),
        it(ids.data(),DType::I32,{2051,w,batch}),ct(counts.data(),DType::I32,{w,batch}),
        wt(scratch.data(),DType::U8,{int(scratch.bytes())});
    auto run=[&](cudaStream_t stream) { ops::qsa_index_select(qt,state.view(),control.view(),maximum,qw,kw,it,ct,wt,stream); };
    cudaStream_t stream{};cudaGraph_t captured{};cudaGraphExec_t executable{};
    if(graph) {
        cuda_check(cudaStreamCreate(&stream),"stream");
        cuda_check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal),"capture");run(stream);
        cuda_check(cudaStreamEndCapture(stream,&captured),"end capture");
        cuda_check(cudaGraphInstantiate(&executable,captured,nullptr,nullptr,0),"instantiate");
    }
    int failures=0;
    for(int replay=0;replay<(graph?2:1);++replay) {
        if(replay) {
            std::reverse(control.rows.begin(),control.rows.end());
            for(int b=0;b<batch;++b) { control.frontier[b]-=65;control.valid[b]=b%2?w:0; }
            control.upload();
            // Publish a changed mapping without moving payload: the oracle consumes this mapping
            // too. This witnesses that a graph never captures the table entries themselves.
            std::reverse(state.tables.begin(),state.tables.end());
            state.table.copy_from_host(state.tables.data(),state.table.bytes);
        }
        if(graph) cuda_check(cudaGraphLaunch(executable,stream),"launch");else run(nullptr);
        cuda_synchronize();
        const auto actual=from_device<int>(ids.data(),2051*w*batch),count=from_device<int>(counts.data(),w*batch);
        for(int t=0;t<w*batch;++t) {
            const auto expected=select_oracle(state,control,t,query,qgamma,kgamma,raw,positions);
            if(count[t]!=int(expected.size())) ++failures;
            for(int i=0;i<2051;++i) if(actual[i+2051*t]!=(i<int(expected.size())?expected[i]:-1)) {
                if(failures<5) std::cerr<<"selector mismatch t="<<t<<" rank="<<i<<" actual="<<actual[i+2051*t]<<" expected="<<(i<int(expected.size())?expected[i]:-1)<<'\n';
                ++failures;
            }
        }
    }
    if(graph) { cudaGraphExecDestroy(executable);cudaGraphDestroy(captured);cudaStreamDestroy(stream); }
    failures+=ids.verify_guards("paged selector IDs")+counts.verify_guards("paged selector counts")+
        scratch.verify_guards("paged selector scratch");
    std::cout<<"paged selector max="<<maximum<<" W="<<w<<" B="<<batch<<" ties="<<ties<<" failures="<<failures<<'\n';
    return failures;
}

int append_attention_case(bool bf,int mapping,int batch,int width) {
    constexpr int maximum=2176;
    State state(maximum,4,bf,mapping);
    std::vector<int> rows(batch),valid(batch),frontier(batch);
    constexpr int fronts[]{63,127,2047,0};
    for(int b=0;b<batch;++b) { rows[b]=3-b;valid[b]=b==1?0:b==2?std::min(width,7):width;frontier[b]=fronts[b]; }
    Control control(width,rows,valid,frontier);
    const int tokens=width*batch;
    std::vector<float> key(512*tokens),value(key.size()),raw(128*tokens),query(6144*tokens);
    fill_uniform(key,321,-3,3);fill_uniform(value,654,-6,6);fill_uniform(raw,987,-1,1);
    fill_uniform(query,456,-.3,.3);
    for(auto* x:{&key,&value,&raw,&query}) round_to_bf16(*x);
    auto dk=to_device_bf16(key),dv=to_device_bf16(value),dr=to_device_bf16(raw),dq=to_device_bf16(query);
    Tensor kt(dk.p,DType::BF16,{256,2,width,batch}),vt(dv.p,DType::BF16,{256,2,width,batch}),
        rt(dr.p,DType::BF16,{128,width,batch});
    // Dense represented history makes every selected old row numerically observable;
    // the oracle decodes these bytes directly, independently of the append codec.
    std::vector<unsigned char> ek(state.k.bytes),ev(state.v.bytes),eks(state.ks.bytes),evs(state.vs.bytes);
    if(bf) {
        for(std::size_t i=0;i<ek.size()/2;++i) {
            const auto kb=f32_to_bf16(float(int((i*37)%47)-23)/32.F);
            const auto vb=f32_to_bf16(float(int((i*19)%59)-29)/16.F);
            ek[2*i]=kb&255;ek[2*i+1]=kb>>8;ev[2*i]=vb&255;ev[2*i+1]=vb>>8;
        }
    } else {
        for(std::size_t i=0;i<ek.size();++i) { ek[i]=(i*37+11)%256;ev[i]=(i*19+7)%256; }
        for(std::size_t i=0;i<eks.size();++i) { eks[i]=0x28+i%9;evs[i]=0x30+i%7; }
        state.ks.copy_from_host(eks.data(),state.ks.bytes);state.vs.copy_from_host(evs.data(),state.vs.bytes);
    }
    state.k.copy_from_host(ek.data(),state.k.bytes);state.v.copy_from_host(ev.data(),state.v.bytes);
    ops::qsa_state_append(kt,vt,rt,control.view(),state.view(),maximum,nullptr);
    cuda_synchronize();
    std::vector<std::uint16_t> er(state.raw.bytes/2);
    std::vector<int> ep(state.pos.bytes/4);
    for(int b=0;b<batch;++b) for(int j=0;j<valid[b];++j) {
        const int t=j+width*b,id=frontier[b]+j;
        for(int h=0;h<2;++h) {
            encode(std::span(key).subspan(256*(h+2*t),256),state.row(rows[b],id,h),bf,ek,eks);
            encode(std::span(value).subspan(256*(h+2*t),256),state.row(rows[b],id,h),bf,ev,evs);
        }
        for(int d=0;d<128;++d) er[d+128*state.meta(rows[b],id)]=f32_to_bf16(raw[d+128*t]);
        for(int a=0;a<3;++a) ep[a+3*state.meta(rows[b],id)]=control.position[a+3*t];
    }
    int failures=0;
    if(from_device<unsigned char>(state.k,state.k.bytes)!=ek) ++failures;
    if(from_device<unsigned char>(state.v,state.v.bytes)!=ev) ++failures;
    if(!bf && (from_device<unsigned char>(state.ks,state.ks.bytes)!=eks || from_device<unsigned char>(state.vs,state.vs.bytes)!=evs)) ++failures;
    if(from_device<std::uint16_t>(state.raw,er.size())!=er || from_device<int>(state.pos,ep.size())!=ep) ++failures;
    std::vector<int> selected(2051*tokens,-1),counts(tokens);
    for(int b=0;b<batch;++b) for(int j=0;j<valid[b];++j) {
        const int t=j+width*b,visible=frontier[b]+j+1;
        // A caller-specified frozen subset may exclude the just-appended query row.
        const int n=j%2?std::min(2051,visible-1):std::min(2051,visible);
        counts[t]=n;
        for(int r=0;r<n;++r) selected[r+2051*t]=visible-1-r-(j%2);
    }
    auto di=to_device(selected),dc=to_device(counts);
    GuardedDeviceBuffer output(6144*tokens*2);
    Tensor qt(dq.p,DType::BF16,{256,24,width,batch}),it(di.p,DType::I32,{2051,width,batch}),
        ct(dc.p,DType::I32,{width,batch}),ot(output.data(),DType::BF16,{256,24,width,batch});
    ops::qsa_selected_attention(qt,it,ct,state.view(),control.view(),maximum,ot,nullptr);
    cuda_synchronize();
    std::vector<double> expected(6144*tokens);
    for(int t=0;t<tokens;++t) for(int h=0;h<24;++h) {
        const int b=t/width,n=counts[t];
        if(n==0) continue;
        std::vector<double> scores(n);double max=-INFINITY;
        for(int r=0;r<n;++r) {
            const auto row=state.row(rows[b],selected[r+2051*t],h/12);
            for(int d=0;d<256;++d) scores[r]+=query[d+256*(h+24*t)]*decode(ek,eks,row,d,bf)/16.;
            max=std::max(max,scores[r]);
        }
        double denominator=0;for(double& score:scores) { score=std::exp(score-max);denominator+=score; }
        for(int d=0;d<256;++d) for(int r=0;r<n;++r)
            expected[d+256*(h+24*t)]+=scores[r]/denominator*
                decode(ev,evs,state.row(rows[b],selected[r+2051*t],h/12),d,bf);
    }
    failures+=verify_reduction("paged QSA represented-cache FP64 attention",from_device_bf16(output.data(),expected.size()),expected,kAttentionCriterion);
    failures+=output.verify_guards("paged attention");
    if(from_device<int>(di,selected.size())!=selected || from_device<int>(dc,counts.size())!=counts) ++failures;
    if(from_device<unsigned char>(state.k,state.k.bytes)!=ek || from_device<unsigned char>(state.v,state.v.bytes)!=ev) ++failures;
    std::cout<<"paged append/attention BF16="<<bf<<" mapping="<<mapping<<" B="<<batch<<" W="<<width<<" failures="<<failures<<'\n';
    return failures;
}

struct SparseMatrix {
    int rows,columns;
    std::vector<std::uint16_t> bits;
    DeviceBuffer device;
    SparseMatrix(int n,int k):rows(n),columns(k),bits(std::size_t(n)*k),device(bits.size()*2) {}
    void unit(int row,int column) { bits[std::size_t(row)*columns+column]=f32_to_bf16(1.F); }
    Weight weight() {
        device.copy_from_host(bits.data(),device.bytes);
        Weight result{};result.payload=result.qdata=device.p;result.payload_bytes=device.bytes;
        result.qtype=QType::BF16_CTRL;result.layout=QuantLayout::Contiguous;
        result.n=rows;result.k=columns;result.ndim=2;
        result.shape[0]=result.padded_shape[0]=rows;result.shape[1]=result.padded_shape[1]=columns;
        return result;
    }
};
std::array<double,256> core_norm(std::array<double,256> x,const int* position) {
    double sum=0;for(double v:x) sum+=v*v;
    const double inverse=1/std::sqrt(sum/256.+1.e-6);
    for(double& v:x) v*=inverse;
    const auto original=x;
    for(int pair=0;pair<32;++pair) {
        const double theta=position[pair%3]*std::pow(1.e7,-double(pair)/32.);
        x[pair]=original[pair]*std::cos(theta)-original[pair+32]*std::sin(theta);
        x[pair+32]=original[pair+32]*std::cos(theta)+original[pair]*std::sin(theta);
    }
    return x;
}
int composite_case(bool bf,bool high_positions=false) {
    constexpr int w=3,batch=4,tokens=w*batch,maximum=128;
    State state(maximum,batch,bf,2);
    Control control(w,{3,1,0,2},{3,0,2,1},{63,7,65,1});
    if(high_positions) {
        // Source MRoPE coordinates are independent of logical KV ordinals. Exercise
        // all three axes near the source ceiling without allocating a huge history.
        for(int t=0;t<tokens;++t) for(int axis=0;axis<3;++axis)
            control.position[axis+3*t]=262143-503*axis-17*(axis+1)*t;
        control.upload();
    }
    SparseMatrix iq(512,2560),ik(128,2560),qg(12288,2560),k(512,2560),v(512,2560),o(2560,6144);
    for(int h=0;h<24;++h) for(int d=0;d<256;++d) {
        qg.unit(h*512+d,(17*h+d)%2560);
        qg.unit(h*512+256+d,(29*h+d+913)%2560);
    }
    for(int d=0;d<512;++d) { k.unit(d,(d+101)%2560);v.unit(d,(d+701)%2560); }
    for(int d=0;d<2560;++d) o.unit(d,(d*37)%6144);
    auto diqnorm=to_device(std::vector<float>(128,1)),diknorm=to_device(std::vector<float>(128,1)),
         dqnorm=to_device(std::vector<float>(256,1)),dknorm=to_device(std::vector<float>(256,1));
    ops::QsaVerifierWeights weights{iq.weight(),ik.weight(),qg.weight(),k.weight(),v.weight(),o.weight(),
        Tensor(diqnorm.p,DType::FP32,{128}),Tensor(diknorm.p,DType::FP32,{128}),
        Tensor(dqnorm.p,DType::FP32,{256}),Tensor(dknorm.p,DType::FP32,{256})};
    std::vector<float> x(2560*tokens);fill_uniform(x,8923,-.5,.5);round_to_bf16(x);
    auto dx=to_device_bf16(x);
    GuardedDeviceBuffer output(x.size()*2),ids(2051*tokens*4),counts(tokens*4),
        scratch(ops::qsa_verifier_workspace_bytes(w,batch,QType::BF16_CTRL,QType::BF16_CTRL,QType::BF16_CTRL,QType::BF16_CTRL));
    Tensor xt(dx.p,DType::BF16,{2560,w,batch}),ot(output.data(),DType::BF16,{2560,w,batch}),
        it(ids.data(),DType::I32,{2051,w,batch}),ct(counts.data(),DType::I32,{w,batch}),
        wt(scratch.data(),DType::U8,{int(scratch.bytes())});
    int failures=0;
    for(int frozen=0;frozen<2;++frozen) {
        // This explicit frozen subset excludes each query's current row. Later queries
        // retain a nonzero earlier append, so a zero-output shortcut cannot pass.
        std::vector<int> chosen(2051*tokens,-1),number(tokens);
        if(frozen) {
            for(int t=0;t<tokens;++t) if(t%w<control.valid[t/w]) {
                number[t]=std::min(2,control.frontier[t/w]);
                for(int r=0;r<number[t];++r)
                    chosen[r+2051*t]=t%w>0?control.frontier[t/w]-1+r:r;
            }
            ids.copy_from_host(chosen.data(),ids.bytes());counts.copy_from_host(number.data(),counts.bytes());
            cudaStream_t stream{};cudaGraph_t graph{};cudaGraphExec_t executable{};
            cuda_check(cudaStreamCreate(&stream),"composite stream");
            cuda_check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal),"composite capture");
            ops::qsa_verifier_selected(xt,control.view(),maximum,weights,state.view(),it,ct,ot,wt,stream);
            cuda_check(cudaStreamEndCapture(stream,&graph),"composite end");
            cuda_check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0),"composite instantiate");
            cuda_check(cudaGraphLaunch(executable,stream),"composite launch");cuda_synchronize(stream);
            cudaGraphExecDestroy(executable);cudaGraphDestroy(graph);cudaStreamDestroy(stream);
        } else {
            ops::qsa_verifier(xt,control.view(),maximum,weights,state.view(),it,ct,ot,wt,nullptr);
            cuda_synchronize();
            for(int t=0;t<tokens;++t) if(t%w<control.valid[t/w]) {
                number[t]=control.frontier[t/w]+t%w+1;
                for(int r=0;r<number[t];++r) chosen[r+2051*t]=r;
            }
        }
        if(from_device<int>(ids.data(),chosen.size())!=chosen || from_device<int>(counts.data(),number.size())!=number) ++failures;
        const auto kc=from_device<unsigned char>(state.k,state.k.bytes),vc=from_device<unsigned char>(state.v,state.v.bytes),
            ks=from_device<unsigned char>(state.ks,state.ks.bytes),vs=from_device<unsigned char>(state.vs,state.vs.bytes);
        // Verify the complete persistent append separately. Its input is the represented
        // sparse linear formula, independent FP64 RMS/MRoPE and the explicit BF16/codec boundary.
        std::vector<unsigned char> expected_k(kc.size()),expected_v(vc.size()),
            expected_ks(ks.size()),expected_vs(vs.size());
        for(int t=0;t<tokens;++t) if(t%w<control.valid[t/w]) for(int h=0;h<2;++h) {
            std::array<double,256> raw_key{};std::vector<float> key(256),value(256);
            for(int d=0;d<256;++d) {
                raw_key[d]=x[(h*256+d+101)%2560+2560*t];
                value[d]=x[(h*256+d+701)%2560+2560*t];
            }
            const auto normalized=core_norm(raw_key,control.position.data()+3*t);
            for(int d=0;d<256;++d) key[d]=bf16_to_f32(f32_to_bf16(float(normalized[d])));
            const auto row=state.row(control.rows[t/w],control.frontier[t/w]+t%w,h);
            encode(key,row,bf,expected_k,expected_ks);encode(value,row,bf,expected_v,expected_vs);
        }
        if(kc!=expected_k || vc!=expected_v || (!bf && (ks!=expected_ks || vs!=expected_vs))) {
            std::cerr<<"composite persistent codec disagreement\n";++failures;
        }
        std::vector<double> expected(x.size());
        for(int t=0;t<tokens;++t) if(number[t]) {
            std::array<std::array<double,256>,24> gated{};
            for(int h=0;h<24;++h) {
                std::array<double,256> rawq{};
                for(int d=0;d<256;++d) rawq[d]=x[(17*h+d)%2560+2560*t];
                const auto query=core_norm(rawq,control.position.data()+3*t);
                std::vector<double> scores(number[t]);double maximum=-INFINITY;
                for(int r=0;r<number[t];++r) {
                    const auto row=state.row(control.rows[t/w],chosen[r+2051*t],h/12);
                    for(int d=0;d<256;++d) scores[r]+=query[d]*decode(kc,ks,row,d,bf)/16.;
                    maximum=std::max(maximum,scores[r]);
                }
                double denominator=0;for(double& score:scores) { score=std::exp(score-maximum);denominator+=score; }
                for(int d=0;d<256;++d) {
                    for(int r=0;r<number[t];++r) gated[h][d]+=scores[r]/denominator*
                        decode(vc,vs,state.row(control.rows[t/w],chosen[r+2051*t],h/12),d,bf);
                    gated[h][d]/=1+std::exp(-double(x[(29*h+d+913)%2560+2560*t]));
                }
            }
            for(int d=0;d<2560;++d) { const int column=(d*37)%6144;expected[d+2560*t]=gated[column/256][column%256]; }
        }
        failures+=verify_reduction("paged composite local FP64 formula",from_device_bf16(output.data(),expected.size()),expected,
            ReductionCriterion{0.02,2.5e-4,0.02});
    }
    failures+=output.verify_guards("composite output")+scratch.verify_guards("composite scratch");
    std::cout<<"paged composite BF16="<<bf<<" high_positions="<<high_positions<<" failures="<<failures<<'\n';
    return failures;
}
}
int main() {
    if(require_cuda()) return 1;
    try {
        int failures=0;
        for(int mapping=0;mapping<3;++mapping) {
            failures+=selection_case(2115,7,4,mapping,false,mapping==2);
            failures+=append_attention_case(true,mapping,4,16);
            failures+=append_attention_case(false,mapping,4,16);
        }
        failures+=selection_case(262144,1,1,2,true,false);
        failures+=selection_case(262144,1,1,2,false,false);
        failures+=selection_case(4101,7,1,2,false,false);
        failures+=selection_case(2051,1,2,2,true,false);
        failures+=selection_case(4099,1,3,1,true,false);
        failures+=selection_case(3,1,1,0,true,false);
        failures+=selection_case(129,129,1,2,true,false);
        failures+=append_attention_case(false,2,1,1);
        failures+=composite_case(true);
        failures+=composite_case(false);
        failures+=composite_case(true,true);
        failures+=composite_case(false,true);
        std::cout<<"paged QSA failures="<<failures<<'\n';
        return failures?1:0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
