"""Real-weight FP8 fused Ops on retained loop-capture BF16 inputs, versus FP64.

This is a same-input local intervention, not a new model trajectory. Larger T
repeats captured rows to exercise production route boundaries and tiled tails.
"""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from tools.reference.qwen3_8_27b.artifact import TextArtifact, bf16


def metric(actual, oracle, limits):
    error=actual-oracle
    l2=float(np.linalg.norm(error)/max(np.linalg.norm(oracle),1e-300))
    maximum=float(np.max(np.abs(error)))
    gross=limits[1]+limits[2]*float(np.max(np.abs(oracle)))
    return dict(relative_l2=l2,max_absolute=maximum,limits=list(limits),
                pass_=bool(np.isfinite(actual).all() and np.isfinite(oracle).all()
                           and l2<=limits[0] and maximum<=gross))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact',type=Path,required=True)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bench-swiglu',action='store_true')
    p.add_argument('--check-panels',action='store_true')
    p.add_argument('--panel-policy',choices=['a16','a8'],default='a16')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    artifact=TextArtifact(args.artifact)
    results=[]
    names=[name for name,obj in artifact.objects.items()
           if obj.get('format')=='FP8_E4M3FN_ROW_BF16S']
    if args.bench_swiglu:
        names=['text/layers/62/mlp/gate_up']
    if args.check_panels:
        names=[name for name in names if name.endswith(('/gate_up','/query_key_gate_value'))]
    for name in names:
        n,k=artifact.objects[name]['shape']
        layer=int(name.split('/')[2])
        if name.endswith('/gate_up'): mode,field='swiglu','mlp_hidden'
        elif name.endswith('/down'): mode,field='add','mlp_activation'
        elif name.endswith('/attention/output'): mode,field='add','attn_gated'
        elif name.endswith('/query_key_gate_value'): mode,field='attn','attn_hidden'
        else: raise ValueError('no retained capture mapping: '+name)
        folder=args.output/(str(layer)+'-'+mode)
        folder.mkdir(exist_ok=True)
        (folder/'weight.bin').write_bytes(artifact.raw(name))
        source=np.fromfile(args.capture/f'{layer}.{field}',dtype='<u2').reshape(-1,k)
        residual_field='mlp_input' if field=='mlp_activation' else 'attn_input'
        residual=(np.fromfile(args.capture/f'{layer}.{residual_field}',dtype='<u2').reshape(-1,n)
                  if mode=='add' else None)
        out_rows=n//2 if mode=='swiglu' else n
        rows=np.unique(np.linspace(0,out_rows-1,64,dtype=int))
        weights=np.concatenate([artifact.matrix(name,int(r),1) for r in rows])
        up=(np.concatenate([artifact.matrix(name,int(r+n//2),1) for r in rows])
            if mode=='swiglu' else None)
        cases=([('a8',1),('a8',6),('a8',12),('a8',18),('a8',24),('a8',4096)] if args.bench_swiglu else
               [('a16',1),('a16',6),('a16',33),('a8',1),('a8',3),
                ('a8',5),('a8',6),('a8',22),('a8',25),('a8',65),('a8',4096)])
        if args.check_panels: cases=[(args.panel_policy,t) for t in (6,12,18,24)]
        panel=None
        for policy,t in cases:
            inputs=source[np.arange(t)%len(source)]
            inputs.tofile(folder/'input.bin')
            if residual is not None:
                residual[np.arange(t)%len(residual)].tofile(folder/'residual.bin')
            command=[str(args.binary),str(folder),str(n),str(k),str(t),policy,mode]
            if args.bench_swiglu: command.append('bench-swiglu')
            execution=subprocess.run(command,check=True,capture_output=True,text=True)
            raw=(folder/'output.bin').read_bytes()
            actual=bf16(raw).astype(np.float64)
            if mode=='attn':
                sections=np.split(actual,np.array([6144,7168,13312])*t)
                actual=np.concatenate([v.reshape(t,-1) for v in sections],axis=1)
            else: actual=actual.reshape(t,out_rows)
            ts=np.unique(np.concatenate((np.arange(min(t,6)),[t//2,t-1])))
            x=bf16(inputs[ts].tobytes()).astype(np.float64).reshape(len(ts),k)
            expected=x@weights.T
            if mode=='swiglu': expected*=np.exp(-np.logaddexp(0.,-expected))*(x@up.T)
            if mode=='add':
                r=bf16(residual[ts%len(residual)].tobytes()).astype(np.float64).reshape(len(ts),n)
                expected+=r[:,rows]
            limits=((.08,.01,.12) if mode=='swiglu' else (.04,1/256,.06)) if policy=='a8' else (
                (.0033,.005,.0063) if mode=='swiglu' else (.0029,.004,.0045) if mode=='attn' else (1/256,1/256,2/256))
            check=metric(actual[np.ix_(ts,rows)],expected,limits)
            check['all_outputs_finite']=bool(np.isfinite(actual).all())
            check['pass_'] &= check['all_outputs_finite']
            if args.check_panels:
                if t==6: panel=actual.copy()
                check['panels_exact']=bool(np.array_equal(actual.view(np.uint64),
                                                         np.tile(panel,(t//6,1)).view(np.uint64)))
                check['pass_'] &= check['panels_exact']
            results.append(dict(matrix=name,mode=mode,policy=policy,t=t,rows=rows.tolist(),
                                tokens=ts.tolist(),command=command,timing=execution.stdout.strip(),**check))
            (args.output/'results.json').write_text(json.dumps(dict(scope=__doc__,
                artifact=str(args.artifact),capture=str(args.capture),results=results,
                all_pass=all(r['pass_'] for r in results)),indent=2)+'\n')
            print(name,policy,t,check,execution.stdout.strip(),flush=True)
        # These are local scratch copies, not retained evidence or model artifacts.
        for basename in ('weight.bin','input.bin','residual.bin','output.bin'):
            (folder/basename).unlink(missing_ok=True)
    artifact.close()
    if not all(r['pass_'] for r in results): raise SystemExit(1)


if __name__=='__main__': main()
