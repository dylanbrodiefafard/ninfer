"""Train-only calibrated original-source projection experiments; no model admission.

The finite NVFP4 recipe searches K16 scales against input second moments. Its
independent FP64 decoder evaluates stored codes/scales, not encoder intermediates.
Held-out documents never choose weight scales or activation divisors.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from tools.artifact.container import Artifact, ArtifactIdentity, ArtifactWriter, TensorSpec
from tools.artifact.layouts import encode_nvfp4, decode_nvfp4_words, encode_direct
from tools.convert.common.fp8_quantize import encode_source_fp8
from tools.parity.qwen4.native_weight_candidates import decode_rows, error, lut
from tools.reference.qwen4.common import linear, silu, sigmoid, ordinary_rmsnorm, source_grouped_rmsnorm, partial_rope
from tools.reference.qwen4.gdn import causal_depthwise_convolution, actual_gguf_control_gates, repeat_grouped_query_key_heads, recurrence
from tools.reference.qwen4.qsa import source_sparse_attention


FACTORS = (1., .8, .9, 1.1, 1.2)
MAGNITUDES = (0., .5, 1., 1.5, 2., 3., 4., 6.)
SOURCE_SCREEN = (.02,.005,.02)
QSA_SCREEN = (.02,2.5e-4,.02)
CONV_SCREEN = (1/256,0.,1/128)
RECURRENT_SCREEN = (.0065,2e-5,.004)


def codes(values, *, ties_even=False):
    bounds = torch.tensor((.25, .75, 1.25, 1.75, 2.5, 3.5, 5.), dtype=values.dtype)
    magnitude = torch.bucketize(values.abs().contiguous(), bounds)
    if ties_even:
        # E2M1 RNE: odd lower code advances on an exact midpoint.
        odd_tie = (magnitude % 2 == 1) & (values.abs() == bounds[magnitude.clamp(max=6)])
        magnitude += odd_tie
    return (magnitude | ((values < 0).long() << 3)).to(torch.uint8)


def decode(packed, scales, divisor):
    words = torch.empty((packed.shape[0], packed.shape[1]*2), dtype=torch.uint8)
    words[:, ::2], words[:, 1::2] = packed & 15, packed >> 4
    signed = torch.tensor(MAGNITUDES + tuple(-x for x in MAGNITUDES), dtype=torch.float64)
    return signed[words.long()] * lut()[scales.long()].repeat_interleave(16, 1) / float(divisor)


def pack(words):
    return words[..., ::2] | (words[..., 1::2] << 4)


def encode_weight(weight, energy, calibrated):
    """Finite diagonal-Hessian scale fit; maxabs is the first eligible candidate."""
    n, k = weight.shape
    maximum=float(weight.abs().max())
    divisor = torch.tensor(2688 / maximum if maximum else 1., dtype=torch.float32)
    packed, all_scales = [], []
    for start in range(0, n, 64):
        source = weight[start:start+64].float().reshape(-1, k//16, 16)
        maximum = source.abs().amax(-1) * divisor / 6
        best_loss = torch.full(source.shape[:-1], torch.inf, dtype=torch.float64)
        best_words = torch.zeros_like(source, dtype=torch.uint8)
        best_scales = torch.zeros_like(maximum, dtype=torch.uint8)
        for factor in FACTORS if calibrated else (1.,):
            scales = (maximum*factor).clamp(0,448).to(torch.float8_e4m3fn)
            values = source*divisor / scales.float().clamp_min(2**-9).unsqueeze(-1)
            words = codes(values)
            represented = decode(pack(words).reshape(-1,k//2), scales.view(torch.uint8), divisor).reshape_as(source)
            loss = ((represented-source.double()).square()*energy.reshape(1,k//16,16)).sum(-1)
            improve = loss < best_loss
            best_words = torch.where(improve[...,None], words, best_words)
            best_scales = torch.where(improve, scales.view(torch.uint8), best_scales)
            best_loss = torch.minimum(loss,best_loss)
        packed.append(pack(best_words).reshape(-1,k//2));all_scales.append(best_scales)
    return torch.cat(packed), torch.cat(all_scales), divisor


def activation4(x, divisor):
    groups = x.float().reshape(x.shape[0],-1,16)
    scales = (groups.abs().amax(-1)*divisor/6).clamp(0,448).to(torch.float8_e4m3fn)
    normalized = groups*divisor/scales.float().clamp_min(2**-9)[...,None]
    return decode(pack(codes(normalized,ties_even=True)).reshape(x.shape[0],-1),scales.view(torch.uint8),divisor)


def activation8(x, floor=0.):
    scale = (x.float().abs().amax(-1,keepdim=True)/448).clamp_min(floor)
    words = (x.float()/scale.clamp_min(torch.finfo(torch.float32).tiny)).clamp(-448,448).to(torch.float8_e4m3fn).view(torch.uint8)
    return lut()[words.long()]*scale.double()


def metrics(y, reference, *, row_label='max_token_relative_l2'):
    result=error(y,reference)
    result[row_label]=float((torch.linalg.vector_norm(y-reference,dim=1)/torch.linalg.vector_norm(reference,dim=1).clamp_min(1e-30)).max())
    return result


def screen(y, reference, criterion, *, per_token=True):
    """Predeclared source screen, separate from represented-weight GPU error."""
    y=y.double();reference=reference.double()
    result=metrics(y.flatten(1),reference.flatten(1),row_label='max_token_relative_l2' if per_token else 'max_state_row_relative_l2')
    limit=criterion[1]+criterion[2]*float(reference.abs().max())
    passed=bool(torch.isfinite(y).all()) and result['relative_l2']<=criterion[0] and result['max_absolute']<=limit
    if per_token:
        absolute=(y-reference).flatten(1).abs().amax(1)
        row_limits=criterion[1]+criterion[2]*reference.flatten(1).abs().amax(1)
        passed &= result['max_token_relative_l2']<=criterion[0] and bool((absolute<=row_limits).all())
        result['gross_token_failures']=int((absolute>row_limits).sum())
    return {**result,'criterion':list(criterion),'gross_limit':limit,'passed':passed}


def read_inputs(manifest, captures, layer, stage, side='actual'):
    result=[]
    for panel in manifest['panels']:
        folder=captures/panel['id']
        meta=json.loads((folder/'capture.json').read_text())
        if len(meta['tokens'])!=int(panel['tokens']):raise ValueError('capture count mismatch')
        with Artifact(Path(panel['panel'])) as artifact:
            if artifact.identity!=ArtifactIdentity('qwen4/native-text-qualification','nvidia-source-corpus'):
                raise ValueError('original source corpus panel required')
            tokens=torch.frombuffer(bytearray(artifact.payload('token.ids')),dtype=torch.int32).tolist()
        if tokens!=meta['tokens']:raise ValueError('capture token identity mismatch')
        if meta['positions']!='t+axis' or meta['dtype']!='little-endian-f32; BF16 public activations/conv, FP32 recurrent state':raise ValueError('capture representation mismatch')
        path=folder/f'layer{layer}.{stage}.{side}.f32'
        x=torch.frombuffer(bytearray(path.read_bytes()),dtype=torch.float32).reshape(int(panel['tokens']),2560)
        if not torch.equal(x,x.bfloat16().float()):raise ValueError('capture is not represented BF16')
        result.append(({**panel,'token_ids':meta['tokens']},x.bfloat16()))
    return result


def output_operands(inputs, source, prefix, layer):
    """Derive gated operands mathematically, never from private GPU intermediates."""
    result=[]
    if layer<3:
        p=prefix+'linear_attn.'
        for panel,x in inputs:
            gate,_=gdn_core(x,linear(x,source[p+'in_proj_qkv.weight']),linear(x,source[p+'in_proj_z.weight']),source,p)
            result.append((panel,gate))
    else:
        p=prefix+'self_attn.'
        for panel,x in inputs:
            qg=linear(x,source[p+'q_proj.weight']).reshape(-1,24,512)
            k=linear(x,source[p+'k_proj.weight']).reshape(-1,2,256)
            v=linear(x,source[p+'v_proj.weight']).reshape(-1,2,256)
            positions=torch.arange(len(x))[None,:]+torch.arange(3)[:,None]
            # Every causal token fits the protected selector's 2048-token budget.
            # Order cannot change the independent mathematical softmax result.
            if len(x)>2048:raise ValueError('bounded complete-prefix QSA oracle only')
            y=source_sparse_attention(qg[:,:,:256],k,v,[torch.arange(t+1) for t in range(len(x))],
                positions,positions,source[p+'q_norm.weight'],source[p+'k_norm.weight'],
                qg[:,:,256:].flatten(-2),source[p+'o_proj.weight'],core_cache_dtype=torch.bfloat16,mrope_section=(11,11,10))
            result.append((panel,(y.core.flatten(-2)*sigmoid(qg[:,:,256:].flatten(-2))).bfloat16()))
    return result


def gdn_core(x,raw,z,source,p,*,record_states=False):
    # Complete-layer contract, not the standalone fused input-projection profile:
    # raw QKV is an explicit BF16 consumer representation before convolution.
    qkv,history=causal_depthwise_convolution(raw.bfloat16(),source[p+'conv1d.weight'].reshape(10240,4),torch.zeros(10240,3,dtype=torch.bfloat16))
    qkv=qkv.bfloat16();z=z.bfloat16().reshape(-1,48,128)
    # Native conversion owns this FP32 fold; preserve its represented control.
    folded=-torch.exp(source[p+'A_log'].float())
    decay,beta=actual_gguf_control_gates(linear(x,source[p+'in_proj_a.weight']),linear(x,source[p+'in_proj_b.weight']),folded,source[p+'dt_bias'].float())
    q=repeat_grouped_query_key_heads(qkv[:,:2048].reshape(-1,16,128),48)
    k=repeat_grouped_query_key_heads(qkv[:,2048:4096].reshape(-1,16,128),48)
    rec=recurrence(q,k,qkv[:,4096:].reshape(-1,48,128),decay.float(),beta.float(),torch.zeros(48,128,128))
    gate=(ordinary_rmsnorm(rec.output.bfloat16(),source[p+'norm.weight'])*sigmoid(z)).bfloat16().flatten(-2)
    state={'convolution':history.double(),'recurrent':rec.final_state.double()}
    if record_states:state['recurrent_records']=rec.state_records
    return gate,state


def shared_composition(inputs, source, prefix, candidates, oracle_root, layer, *, split='heldout'):
    """Complete shared formula and its contribution to the original full MoE."""
    p=prefix+'mlp.'
    weights={r:source[p+f'shared_expert.{r}_proj.weight'] for r in ('gate','up','down')}
    scale_weight=source[p+'shared_expert_gate.weight']
    def evaluate(x,w,divisors=None,a8=False,floors=None):
        def operand(value,role):
            if a8 and (floors is None or role in floors):return activation8(value.bfloat16(),floors[role] if floors else 0.)
            return activation4(value.bfloat16(),divisors[role]) if divisors and role in divisors else value
        gate=linear(operand(x,'gate'),w['gate'])
        up=linear(operand(x,'up'),w['up'])
        down=linear(operand(silu(gate)*up,'down'),w['down'])
        return down*sigmoid(linear(x,scale_weight))
    result={}
    metadata=None
    if any(p['split']==split for p,_ in inputs):
        if oracle_root is None:raise ValueError('complete same-input FP64 MoE oracle required')
        metadata=json.loads((oracle_root/f'layer{layer}.json').read_text())
        if metadata['profile']!='qwen4-native-heldout-moe-same-input-fp64' or metadata['layer']!=layer:
            raise ValueError('wrong complete MoE oracle authority')
    for panel,x in inputs:
        if panel['split']!=split:continue
        reference=evaluate(x,weights)
        original=next(p for p in metadata['panels'] if p['id']==panel['id'])
        if original['tokens']!=panel['token_ids'] or original['shape']!=[len(x),2560]:raise ValueError('complete MoE oracle token/shape mismatch')
        full=torch.frombuffer(bytearray((oracle_root/original['file']).read_bytes()),dtype=torch.float64).reshape(len(x),2560)
        profiles={}
        for label,role_candidates in candidates.items():
            if not all('shared_'+r in role_candidates for r in weights):continue
            variants={}
            if label.startswith('nvfp4'):
                for selected in (('gate',),('up',),('down',),('gate','up','down')):
                    w={**weights,**{r:role_candidates['shared_'+r][0] for r in selected}}
                    divisors={r:role_candidates['shared_'+r][1] for r in selected}
                    variants['+'.join(selected)+':a16']=evaluate(x,w)
                    variants['+'.join(selected)+':a4']=evaluate(x,w,divisors)
            else:
                for weight_mask in (range(1,8) if label=='source_tensor_fp8' else (7,)):
                    w={r:role_candidates['shared_'+r][0] if weight_mask&(1<<bit) else weights[r]
                       for bit,r in enumerate(weights)}
                    for mask in range(8):
                        if mask&~weight_mask:continue
                        floors={r:role_candidates['shared_'+r][1] or 0. for bit,r in enumerate(weights) if mask&(1<<bit)}
                        variants[f'weight_mask{weight_mask}:a8_mask{mask}']=evaluate(x,w,a8=bool(mask),floors=floors)
            profiles[label]={mode:{'shared_path_source_loss':metrics(value,reference),
                'full_moe_source_delta':screen(full+value-reference,full,SOURCE_SCREEN)} for mode,value in variants.items()}
        result[panel['id']]=profiles
    return result


def attention_composition(inputs, source, prefix, candidates, layer, *, split='heldout'):
    """Closed GDN/QSA source formula with controls and persistent dtypes protected."""
    gdn=layer<3
    roles=('gdn_qkv','gdn_z','gdn_out') if gdn else ('qsa_q','qsa_k','qsa_v','qsa_o')
    suffixes=('linear_attn.in_proj_qkv.weight','linear_attn.in_proj_z.weight','linear_attn.out_proj.weight') if gdn else tuple(f'self_attn.{r}_proj.weight' for r in ('q','k','v','o'))
    original={r:source[prefix+s] for r,s in zip(roles,suffixes)}
    def evaluate(x,weights,divisors=None,a8_roles=()):
        def projection(value,role):
            if divisors and role in divisors:value=activation4(value.bfloat16(),divisors[role])
            elif role in a8_roles:value=activation8(value.bfloat16(),a8_roles[role])
            return linear(value,weights[role])
        if gdn:
            p=prefix+'linear_attn.'
            gate,state=gdn_core(x,projection(x,'gdn_qkv'),projection(x,'gdn_z'),source,p,record_states=True)
            return projection(gate,'gdn_out'),state
        p=prefix+'self_attn.'
        qg=projection(x,'qsa_q').reshape(-1,24,512)
        k=projection(x,'qsa_k').reshape(-1,2,256);v=projection(x,'qsa_v').reshape(-1,2,256)
        positions=torch.arange(len(x))[None,:]+torch.arange(3)[:,None]
        selections=[torch.arange(t+1) for t in range(len(x))]
        attention=source_sparse_attention(qg[:,:,:256],k,v,selections,positions,positions,
            source[p+'q_norm.weight'],source[p+'k_norm.weight'],qg[:,:,256:].flatten(-2),
            weights['qsa_o'],core_cache_dtype=torch.bfloat16,mrope_section=(11,11,10))
        output=attention.output
        if divisors or a8_roles:
            output=projection(attention.core.flatten(-2)*sigmoid(qg[:,:,256:].flatten(-2)),'qsa_o')
        keys=partial_rope(source_grouped_rmsnorm(k,source[p+'k_norm.weight'],group_size=256),positions,
            rotary_dim=64,theta=1e7,mrope_section=(11,11,10)).bfloat16().double()
        return output,{'keys':keys,'values':v.bfloat16().double()}
    result={}
    for panel,x in inputs:
        if panel['split']!=split:continue
        baseline,states=evaluate(x,original)
        profiles={}
        for label,available in candidates.items():
            if label=='nvfp4_maxabs':continue
            active=[r for r in roles if r in available]
            if not active:continue
            groups=[(r,) for r in active]+([tuple(active)] if len(active)>1 else [])
            profiles[label]={}
            for selected in groups:
                weights=dict(original);weights.update({r:available[r][0] for r in selected})
                for mode in ('a16','a4' if label.startswith('nvfp4') else 'a8'):
                    divisors={r:available[r][1] for r in selected} if mode=='a4' else None
                    y,state=evaluate(x,weights,divisors,{r:available[r][1] or 0. for r in selected} if mode=='a8' else ())
                    state_errors={}
                    for name,value in state.items():
                        if name=='recurrent_records':continue
                        if gdn:
                            state_errors[name]=screen(value,states[name],CONV_SCREEN if name=='convolution' else RECURRENT_SCREEN,per_token=False)
                        else:
                            groups=value.reshape(len(x),2,16,16);original_groups=states[name].reshape_as(groups)
                            bounds=original_groups.abs().amax(-1,keepdim=True)/256+1e-4
                            violations=int(((groups-original_groups).abs()>bounds).sum())
                            state_errors[name]={**metrics(value.flatten(1),states[name].flatten(1)),
                                'criterion':'each K16 element <= original represented group max/256 + 1e-4',
                                'violations':violations,'passed':violations==0}
                    if 'recurrent_records' in state:
                        records=[screen(actual,expected,RECURRENT_SCREEN,per_token=False) for actual,expected in zip(state['recurrent_records'],states['recurrent_records'])]
                        state_errors['recurrent_transitions']={'per_token_relative_l2':[r['relative_l2'] for r in records],
                            'max_relative_l2':max(r['relative_l2'] for r in records),'max_absolute':max(r['max_absolute'] for r in records),
                            'criterion':list(RECURRENT_SCREEN),'passed':all(r['passed'] for r in records)}
                    output_error=screen(y,baseline,SOURCE_SCREEN if gdn else QSA_SCREEN)
                    profiles[label]['+'.join(selected)+':'+mode]={'output_source_loss':output_error,
                        'state_source_loss':state_errors,'source_screen_passed':output_error['passed'] and all(s['passed'] for s in state_errors.values())}
        result[panel['id']]=profiles
        print(f'composition layer{layer} {panel["id"]}',flush=True)
    return result


def run(args):
    manifest=json.loads(args.manifest.read_text())
    if args.heldout_ids:
        selected=set(args.heldout_ids.split(','))
        if not selected <= {p['id'] for p in manifest['panels'] if p['split']=='heldout'}:
            raise ValueError('heldout selection is not in the original manifest')
        manifest={**manifest,'panels':[p for p in manifest['panels'] if p['split']=='calibration' or p['id'] in selected]}
    panels=manifest['panels']
    if {p['split'] for p in panels}!={'calibration','heldout'}:raise ValueError('require disjoint calibration and heldout splits')
    if len({p['id'] for p in panels})!=len(panels) or len({p['panel'] for p in panels})!=len(panels):raise ValueError('documents reused between splits')
    calibration_ids=[p['id'] for p in panels if p['split']=='calibration']
    frozen=json.loads(args.frozen_fit.read_text()) if args.frozen_fit else None
    if frozen and (not frozen['complete'] or frozen['stage']!='calibration_fit' or frozen['calibration_ids']!=calibration_ids or frozen['layer']!=args.layer):
        raise ValueError('frozen fit does not match this layer/calibration split')
    if frozen:
        source_name=frozen['tensor_fp8_source']
        if args.tensor_fp8_source and str(args.tensor_fp8_source)!=source_name:raise ValueError('tensor FP8 source differs from frozen fit')
        args.tensor_fp8_source=Path(source_name) if source_name else None
    if args.fit_only or frozen:
        split='calibration' if args.fit_only else 'heldout'
        manifest={**manifest,'panels':[p for p in panels if p['split']==split]}
        panels=manifest['panels']
    if args.out.exists():raise FileExistsError(args.out)
    with safe_open(str(args.source_root/f'qwen4-layer-{args.layer}.safetensors'),framework='pt') as handle:
        source={name:handle.get_tensor(name) for name in handle.keys() if '.experts.' not in name}
    tensor_fp8={}
    if args.tensor_fp8_source:
        with safe_open(str(args.tensor_fp8_source),framework='pt') as handle:
            tensor_fp8={name:handle.get_tensor(name) for name in handle.keys()}
    prefix=f'model.language_model.layers.{args.layer}.'
    roles={f'shared_{r}':f'mlp.shared_expert.{r}_proj.weight' for r in ('gate','up','down')}
    roles.update({'gdn_qkv':'linear_attn.in_proj_qkv.weight','gdn_z':'linear_attn.in_proj_z.weight','gdn_out':'linear_attn.out_proj.weight'} if args.layer<3 else {f'qsa_{r}':f'self_attn.{r}_proj.weight' for r in ('q','k','v','o')})
    if args.roles:roles={r:roles[r] for r in args.roles.split(',')}
    reports={};candidate_weights={};composition={}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    for role,name in roles.items():
        weight=source[prefix+name]
        if weight.dtype!=torch.bfloat16:raise ValueError('original BF16 weights required')
        inputs=read_inputs(manifest,args.captures,args.layer,'moe_input' if role.startswith('shared') else 'attn_input')
        if role=='shared_down':
            gate=source[prefix+'mlp.shared_expert.gate_proj.weight'];up=source[prefix+'mlp.shared_expert.up_proj.weight']
            inputs=[(p,(silu(linear(x,gate))*linear(x,up)).bfloat16()) for p,x in inputs]
        if role in ('gdn_out','qsa_o'):
            inputs=output_operands(inputs,source,prefix,args.layer)
        train=None if frozen else torch.cat([x for p,x in inputs if p['split']=='calibration'])
        energy=None if frozen else train.double().square().mean(0)
        reference=[linear(x,weight) for _,x in inputs]
        variants={}
        for calibrated in (False,True):
            label='nvfp4_diagonal_calibrated' if calibrated else 'nvfp4_maxabs'
            fitted=frozen['roles'][role]['variants'][label] if frozen else None
            if fitted:
                artifact=Path(fitted['artifact'])
                with Artifact(artifact) as saved:
                    if saved.find('weight').shape!=tuple(weight.shape):raise ValueError('frozen candidate shape mismatch')
                    packed,scales,divisor=decode_nvfp4_words(saved.payload('weight'),tuple(weight.shape))
                    frozen_activation_divisor=float(torch.frombuffer(bytearray(saved.payload('input_scale_divisor')),dtype=torch.float32)[0])
            else:
                packed,scales,divisor=encode_weight(weight,energy,calibrated)
            decoded=decode(packed,scales,divisor)
            # Activation divisor is independently chosen on training projection loss.
            if fitted:
                act_divisor=frozen_activation_divisor
                if act_divisor!=fitted['activation_divisor']:raise ValueError('frozen artifact/report activation divisor differs')
            else:
                maximum=float(train.float().abs().max())
                divisors=[float(torch.tensor(2688/maximum*f,dtype=torch.float32)) for f in FACTORS]
                train_y=linear(train,decoded)
                losses=[float((linear(activation4(train,d),decoded)-train_y).square().sum()) for d in divisors]
                act_divisor=divisors[min(range(len(losses)),key=losses.__getitem__)]
            values=[]
            for (panel,x),baseline in zip(inputs,reference):
                a16=linear(x,decoded);a4=linear(activation4(x,act_divisor),decoded)
                values.append({'id':panel['id'],'split':panel['split'],'w4a16_source_loss':metrics(a16,baseline),
                    'w4a4_source_loss':metrics(a4,baseline),'a4_incremental_loss':metrics(a4,a16)})
            if not fitted:
                artifact=args.out.with_name(args.out.stem+f'.{role}.{label}.ninfer')
                with ArtifactWriter(artifact,ArtifactIdentity('qwen4/native-projection-candidate',label),[
                    TensorSpec('weight',tuple(weight.shape),'NVFP4','blockscale-k16-m128x4-v1'),
                    TensorSpec('input_scale_divisor',(),'FP32','contiguous-le-v1')]) as writer:
                    writer.write('weight',encode_nvfp4(packed,scales,divisor,tuple(weight.shape)))
                    writer.write('input_scale_divisor',encode_direct(torch.tensor(act_divisor,dtype=torch.float32),'FP32'))
            variants[label]={'weight_source_loss':error(decoded,weight),'activation_divisor':act_divisor,'artifact':str(artifact),'documents':values}
            candidate_weights.setdefault(label,{})[role]=(decoded,act_divisor)
        if args.layer in (1,2) and (role.startswith('shared') or role=='gdn_z'):
            if frozen:
                artifact=Path(frozen['roles'][role]['variants']['row_fp8']['artifact'])
                with Artifact(artifact) as saved:encoded=bytes(saved.payload('weight'))
            else:encoded=encode_source_fp8(weight)
            decoded=decode_rows(encoded,*weight.shape)
            values=[]
            for (panel,x),baseline in zip(inputs,reference):
                a16=linear(x,decoded);a8=linear(activation8(x),decoded)
                values.append({'id':panel['id'],'split':panel['split'],'w8a16_source_loss':metrics(a16,baseline),
                    'w8a8_source_loss':metrics(a8,baseline),'a8_incremental_loss':metrics(a8,a16)})
            if not frozen:
                artifact=args.out.with_name(args.out.stem+f'.{role}.row_fp8.ninfer')
                with ArtifactWriter(artifact,ArtifactIdentity('qwen4/native-projection-candidate','source-bf16-row-fp8'),
                    [TensorSpec('weight',tuple(weight.shape),'FP8_E4M3FN_ROW_BF16S','row-scale-v1')]) as writer:
                    writer.write('weight',encoded)
            variants['row_fp8']={'weight_source_loss':error(decoded,weight),'artifact':str(artifact),'documents':values}
            candidate_weights.setdefault('row_fp8',{})[role]=(decoded,None)
        if prefix+name in tensor_fp8:
            base=prefix+name.removesuffix('.weight')
            fp8=tensor_fp8[base+'.weight']
            scale=tensor_fp8[base+'.weight_scale'];floor=tensor_fp8[base+'.input_scale']
            if fp8.dtype!=torch.float8_e4m3fn or scale.dtype!=torch.float32 or floor.dtype!=torch.float32:
                raise ValueError('exact source tensor FP8 codes/FP32 scalar words required')
            decoded=lut()[fp8.view(torch.uint8).long()]*float(scale)
            values=[]
            for (panel,x),baseline in zip(inputs,reference):
                a16=linear(x,decoded);a8=linear(activation8(x,float(floor)),decoded)
                values.append({'id':panel['id'],'split':panel['split'],'w8a16_source_loss':metrics(a16,baseline),
                    'w8a8_source_loss':metrics(a8,baseline),'a8_incremental_loss':metrics(a8,a16)})
            variants['source_tensor_fp8']={'weight_source_loss':error(decoded,weight),'input_scale':float(floor),
                'weight_scale':float(scale),'source':str(args.tensor_fp8_source),'documents':values,
                'calibration':'Original producer scalar words preserved. Producer calibration corpus is not known; these are heldout from our scale fitting, not a claim about producer training/calibration overlap.'}
            candidate_weights.setdefault('source_tensor_fp8',{})[role]=(decoded,float(floor))
        reports[role]={'source_tensor':prefix+name,'shape':list(weight.shape),'variants':variants}
        if role=='shared_down' and all('shared_'+r in reports for r in ('gate','up','down')):
            composition['shared']=shared_composition(read_inputs(manifest,args.captures,args.layer,'moe_input'),source,prefix,candidate_weights,args.moe_oracles,args.layer)
        if role==list(roles)[-1] and any(r.startswith(('gdn_','qsa_')) for r in roles):
            composition['attention']=attention_composition(read_inputs(manifest,args.captures,args.layer,'attn_input'),source,prefix,candidate_weights,args.layer)
        args.out.write_text(json.dumps({'manifest':str(args.manifest),'captures':str(args.captures),
            'layer':args.layer,'stage':'calibration_fit' if args.fit_only else 'frozen_holdout',
            'complete':set(reports)==set(roles),'tensor_fp8_source':str(args.tensor_fp8_source) if args.tensor_fp8_source else None,
            'frozen_fit':str(args.frozen_fit) if frozen else None,'calibration_ids':calibration_ids,
            'source':str(args.source_root/f'qwen4-layer-{args.layer}.safetensors'),
            'weight_coordinates':'Original NVIDIA source order. GDN V-side rows/output columns require the existing native_prepare permutation before GPU binding, preserving represented codes/scales.',
            'scope':'Same represented-input projection and isolated component counterfactuals, not runtime or propagated-chain admission. Capture baseline failures are disclosed; source loss is not implementation error. Shared full-MoE comparison uses the complete independently computed original FP64 MoE from the same actual BF16 input, not the propagated capture reference.',
            'moe_oracles':str(args.moe_oracles) if args.moe_oracles else None,
            'capture_prefix_failures':{p['id']:json.loads((args.captures/p['id']/'capture.json').read_text())['prefix_failures'] for p in panels},
            'evaluated_panels':[p['id'] for p in panels],
            'boundaries':'GDN complete-layer BF16 QKV/Z, BF16 convolution history/core/gated norm, FP32 controls/recurrent transitions and converted FP32 -exp(A_log); QSA ideal projections through normalization and BF16 K/V cache. A4/A8 explicitly round pack inputs to BF16; activation loss is separate from mathematical represented-weight error.',
            'predeclared_source_screen':{'component':SOURCE_SCREEN,'qsa_per_token':QSA_SCREEN,'gdn_conv':CONV_SCREEN,'gdn_recurrent_every_transition':RECURRENT_SCREEN,'qsa_cache':'unchanged BF16 per-K16 max/256+1e-4; selector membership unchanged below 2048 causal tokens','decision':'every heldout document, per-token output tail, gross bound and state criterion must pass; no GPU arithmetic or model-quality admission implied'},
            'finite_shared_policies':'NVFP4 gate-only, up-only, down-only, or all three; A16 or A4 on selected roles. Source tensor FP8 weight masks 1..7, A8 subsets of each weight mask (26 policies); row FP8 all-three weights with A8 masks 0..7. Bits gate/up/down.',
            'recipe':'K16 E4M3 scale factor search minimizing diagonal input-energy weighted error on calibration only; original global FP32 divisor; independent FP64 code decode and dots.',
            'composition':composition,'roles':reports},indent=2)+'\n')
        print(f'completed layer{args.layer} {role}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--captures',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--layer',type=int,choices=range(4),required=True)
    parser.add_argument('--roles')
    parser.add_argument('--heldout-ids',help='optional completed heldout IDs; all calibration documents remain mandatory')
    parser.add_argument('--tensor-fp8-source',type=Path,help='exact bounded Senfu tensor-FP8 source fixture')
    parser.add_argument('--moe-oracles',type=Path,help='complete same-input FP64 MoE oracle directory, required for heldout shared assessment')
    stage=parser.add_mutually_exclusive_group(required=True)
    stage.add_argument('--fit-only',action='store_true',help='freeze weights/divisors before heldout inputs are opened')
    stage.add_argument('--frozen-fit',type=Path,help='evaluate heldout only from this completed calibration report/artifacts')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2);run(args)
