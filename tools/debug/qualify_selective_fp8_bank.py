"""Independent exact row-scale and FP8 rounding checks on real source matrices."""
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from tools.reference.qwen3_8_27b.artifact import TextArtifact, rounded_bf16
from tools.reference.qwen3_8_27b.source_weights import SourceWeights


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    bank=TextArtifact(args.artifact)
    source=SourceWeights(args.source)
    names=[name for name,obj in bank.objects.items() if obj.get('format')=='FP8_E4M3FN_ROW_BF16S']
    assert names, 'artifact has no FP8 matrices'
    c=np.arange(127,dtype=np.uint8)
    e=(c>>3).astype(int);m=(c&7).astype(float)
    levels=np.where(e==0,np.ldexp(m,-9),np.ldexp(1+m/8,e-7))
    results=[]
    for name in names:
        obj=bank.objects[name];n,k=obj['shape']
        # Cover head and fused-parent seams in addition to interior rows.
        rows=np.unique(np.clip(np.array([0,255,256,6143,6144,7167,7168,
                                        13311,13312,n//2-1,n//2,n-1]),0,n-1))
        logical=source.logical_tensor(name)
        x=logical[torch.from_numpy(rows)].float().numpy().astype(np.float64)
        del logical
        off=bank.base+obj['offset'];soff=off+((n*k+255)//256)*256
        sw=np.frombuffer(bank.data,dtype='<u2',count=n,offset=soff)[rows]
        scale=(sw.astype(np.uint32)<<16).view(np.float32).astype(np.float64)
        expected_scale=rounded_bf16(np.max(np.abs(x),axis=1)/448)
        assert np.array_equal(scale,expected_scale),name
        magnitude=np.abs(x)/np.where(scale==0,1,scale)[:,None]
        hi=np.clip(np.searchsorted(levels,magnitude),1,126);lo=hi-1
        dl=np.abs(magnitude-levels[lo]);dh=np.abs(levels[hi]-magnitude)
        choose_hi=(dh<dl)|((dh==dl)&((hi&1)==0))
        expected=np.where(choose_hi,hi,lo).astype(np.uint8)|(np.signbit(x).astype(np.uint8)<<7)
        actual=np.stack([np.frombuffer(bank.data,dtype=np.uint8,count=k,offset=off+int(r)*k) for r in rows])
        assert np.array_equal(actual,expected),name
        reconstructed=np.stack([bank.matrix(name,int(r),1)[0] for r in rows])
        oracle=levels[expected&127]*np.where(expected&128,-1.,1.)*scale[:,None]
        assert np.array_equal(reconstructed,oracle),name
        results.append(dict(name=name,rows=rows.tolist(),checked_codes=int(actual.size),
                            scales_exact=True,codes_exact=True,decoder_exact=True))
    bank.close()
    args.output.write_text(json.dumps(dict(scope=__doc__,artifact=str(args.artifact),
        source=str(args.source),results=results),indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__': main()
