"""Independent NumPy decoding of the represented Text weights.

No NInfer kernel, runtime binder, activation quantizer, or weight decoder is used.
NVFP4 returns E2M1 * E4M3FN / stored-FP32-divisor in FP64 by default. Callers
must explicitly select a lower-precision reference arithmetic profile.
"""
from __future__ import annotations

import json
import mmap
from pathlib import Path
import struct

import numpy as np


def bf16(data: bytes | memoryview) -> np.ndarray:
    bits = np.frombuffer(data, dtype='<u2').astype(np.uint32) << 16
    return bits.view(np.float32)


def rounded_bf16(x: np.ndarray) -> np.ndarray:
    """Direct FP64-to-BF16 RNE values, without an intermediate FP32 rounding."""
    x = np.asarray(x,dtype=np.float64)
    step = np.maximum(np.frexp(x)[1]-8,-133)
    rounded = np.ldexp(np.rint(np.ldexp(x,-step)),step)
    return np.where(np.abs(rounded) >= 2.**128,np.copysign(np.inf,rounded),rounded)


def e4m3(words: np.ndarray) -> np.ndarray:
    if np.any(words >= 127):
        raise ValueError('NVFP4 scales must be nonnegative finite E4M3FN')
    exponent = (words >> 3).astype(np.int32)
    fraction = (words & 7).astype(np.float64)
    return np.where(exponent == 0, fraction / 512,
                    np.ldexp(1 + fraction / 8, exponent - 7))


class TextArtifact:
    def __init__(self, path: Path):
        self.path = path.resolve(strict=True)
        self.file = self.path.open('rb')
        self.data = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        magic, size = struct.unpack_from('<8sQ', self.data)
        if magic != b'NINFER\x00\x02':
            raise ValueError('expected NInfer v2 artifact')
        directory = json.loads(self.data[16:16+size])
        if directory['identity'] != {'model_id':'qwen3.8-27b','weights_id':'nvfp4'}:
            raise ValueError('reference requires qwen3.8-27b/nvfp4')
        self.base = ((16 + size + 4095) // 4096) * 4096
        self.objects = {obj['name']:obj for obj in directory['objects']}
        for obj in self.objects.values():
            if obj['offset'] < 0 or obj['bytes'] <= 0 or self.base+obj['offset']+obj['bytes'] > len(self.data):
                raise ValueError('artifact object outside payload')

    def close(self):
        self.data.close()
        self.file.close()

    def raw(self, name: str) -> bytes:
        obj = self.objects[name]
        start = self.base + obj['offset']
        return self.data[start:start+obj['bytes']]

    def tensor(self, name: str) -> np.ndarray:
        obj = self.objects[name]
        if obj['layout'] != 'contiguous-le-v1':
            raise ValueError('expected direct tensor')
        if obj['format'] == 'BF16':
            result = bf16(self.raw(name))
        else:
            result = np.frombuffer(self.raw(name), dtype={'FP32':'<f4','I32':'<i4'}[obj['format']]).copy()
        return result.reshape(obj['shape'])

    def matrix(self, name: str, start: int = 0, count: int | None = None,
               *, dtype=np.float64) -> np.ndarray:
        obj = self.objects[name]
        n, k = obj['shape']
        count = n-start if count is None else count
        if start < 0 or count <= 0 or start+count > n:
            raise ValueError('matrix row interval outside logical shape')
        offset = self.base + obj['offset']
        if obj['format'] == 'FP8_E4M3FN_ROW_BF16S':
            if obj['layout'] != 'row-scale-v1':
                raise ValueError('invalid row-scaled FP8 layout')
            scale_offset=((n*k+255)//256)*256
            if obj['bytes'] != scale_offset+2*n or any(self.data[offset+n*k:offset+scale_offset]):
                raise ValueError('invalid FP8 row-scale payload or padding')
            words=np.frombuffer(self.data,dtype=np.uint8,count=count*k,offset=offset+start*k).reshape(count,k)
            magnitude=words & 127
            values=e4m3(magnitude)
            values=np.where(words & 128,-values,values)
            scale_words=np.frombuffer(self.data,dtype='<u2',count=count,offset=offset+scale_offset+2*start)
            scales=(scale_words.astype(np.uint32)<<16).view(np.float32)
            if np.any(scale_words & 32768) or not np.isfinite(scales).all():
                raise ValueError('invalid BF16 row multiplier')
            if np.any((scales[:,None]==0)&(magnitude!=0)):
                raise ValueError('nonzero FP8 code with zero row multiplier')
            # Logical represented value is the exact signed code times stored scale.
            # FP64 also preserves extreme BF16 scale products without an FP32 cast.
            return (values*scales.astype(np.float64)[:,None]).astype(dtype,copy=False)
        if obj['format'] == 'NVFP4':
            if obj['layout'] != 'blockscale-k16-m128x4-v1' or n%128 or k%64:
                raise ValueError('invalid NVFP4 geometry')
            code_bytes = n*k//2
            scale_base = offset + ((code_bytes+255)//256)*256
            divisor = struct.unpack_from('<f',self.data,scale_base+n*k//16)[0]
            if not np.isfinite(divisor) or divisor <= 0:
                raise ValueError('invalid weight divisor')
            packed = np.frombuffer(self.data,dtype=np.uint8,count=count*k//2,offset=offset+start*k//2)
            codes = np.empty((count,k),dtype=np.uint8)
            codes[:,0::2] = (packed & 15).reshape(count,k//2)
            codes[:,1::2] = (packed >> 4).reshape(count,k//2)
            row = np.arange(start,start+count,dtype=np.int64)[:,None]
            group = np.arange(k//16,dtype=np.int64)[None,:]
            indices = ((row//128*(k//64)+group//4)*512 +
                       row%32*16 + (row%128)//32*4 + group%4)
            scales = np.frombuffer(self.data,dtype=np.uint8,count=n*k//16,offset=scale_base)
            multipliers = e4m3(scales[indices]) / float(divisor)
            codebook = np.array([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],dtype=np.float64)
            decoded = codebook[codes].reshape(count,k//16,16) * multipliers[:,:,None]
            return decoded.reshape(count,k).astype(dtype,copy=False)
        if obj['format'] == 'W8G32_F16S':
            if obj['layout'] != 'row-split-k128-v1':
                raise ValueError('invalid W8 layout')
            padded = ((k+127)//128)*128
            scale_base = offset + ((n*padded+255)//256)*256
            codes = np.frombuffer(self.data,dtype=np.int8,count=count*padded,
                                  offset=offset+start*padded).reshape(count,padded)
            scales = np.frombuffer(self.data,dtype='<f2',count=count*padded//32,
                                   offset=scale_base+start*padded//32*2).astype(np.float64)
            decoded = codes.reshape(count,padded//32,32).astype(np.float64) * scales.reshape(count,padded//32,1)
            return decoded.reshape(count,padded)[:,:k].astype(dtype,copy=False)
        if obj['format'] == 'BF16' and obj['layout'] == 'contiguous-le-v1':
            data = self.data[offset+start*k*2:offset+(start+count)*k*2]
            return bf16(data).reshape(count,k).astype(dtype,copy=False)
        raise ValueError(f'unsupported Text matrix: {name}: {obj["format"]}')
