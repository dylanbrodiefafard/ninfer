"""Build bounded real-Qwen4 projection fixtures, not a converted inference model.

Requires the local llama.cpp gguf-py package on PYTHONPATH. Its vectorized source
decoder is checked against this project's independent scalar format oracle.
New weights are encoded offline from explicitly rounded BF16 GGUF values. This
fixture proves represented-kernel arithmetic, not original-BF16 model quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from gguf import GGMLQuantizationType
from gguf.quants import dequantize

from tools.artifact.container import Artifact, ArtifactIdentity, ArtifactWriter, TensorSpec
from tools.convert.common.fp8_quantize import encode_source_fp8
from tools.convert.common.nvfp4_quantize import encode_nvfp4_from_bf16
from tools.reference.qwen4.ggml_k_codecs import decode_q5_k, decode_q6_k, decode_q8_0
from tools.reference.qwen4.ggml_codecs import decode_iq1_s, decode_iq4_nl

MATRICES = (
    ('blk.0.attn_qkv.weight', None), ('blk.0.attn_gate.weight', None),
    ('blk.0.ssm_out.weight', None), ('blk.2.attn_qkv.weight', None),
    ('blk.3.attn_q.weight', None), ('blk.3.attn_k.weight', None),
    ('blk.0.hc_attn_down.weight', None), ('blk.0.hc_attn_up.weight', None),
    ('blk.0.ffn_gate_exps.weight', 0), ('blk.0.ffn_gate_exps.weight', 511),
    ('blk.0.ffn_down_exps.weight', 0), ('blk.0.ffn_down_exps.weight', 511),
)
ORACLES = {'Q5_K': (176, 256, decode_q5_k), 'Q6_K': (210, 256, decode_q6_k),
           'Q8_0': (34, 32, decode_q8_0), 'IQ1_S': (50, 256, decode_iq1_s),
           'IQ4_NL': (18, 32, decode_iq4_nl)}


def build(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    rows = []
    with Artifact(source) as artifact:
        if artifact.identity.model_id != 'qwen4/verification':
            raise ValueError('requires the exact local Qwen4 verification artifact')
        planned = []
        for name, expert in MATRICES:
            obj = artifact.find(name)
            shape = obj.shape if expert is None else obj.shape[1:]
            stem = name if expert is None else f'{name}/expert-{expert}'
            for fmt, layout in [('NVFP4', 'blockscale-k16-m128x4-v1'),
                                ('FP8_E4M3FN_ROW_BF16S', 'row-scale-v1')]:
                if fmt == 'NVFP4' and shape[0] % 128:
                    continue
                planned.append((TensorSpec(f'{stem}/{fmt}', shape, fmt, layout), obj, expert))
        output.parent.mkdir(parents=True, exist_ok=True)
        with ArtifactWriter(output, ArtifactIdentity('qwen4/layer-qualification', 'requantized-gguf'),
                            [item[0] for item in planned]) as writer:
            for spec, obj, expert in planned:
                payload = artifact.payload(obj)
                if expert is not None:
                    size = obj.bytes // obj.shape[0]
                    payload = payload[expert * size:(expert + 1) * size]
                packed = np.frombuffer(payload, dtype=np.uint8)
                values = dequantize(packed, GGMLQuantizationType[obj.format]).reshape(spec.shape)
                block_bytes, block_values, oracle = ORACLES[obj.format]
                flat = values.reshape(-1)
                for block in (0, len(payload) // block_bytes // 2, len(payload) // block_bytes - 1):
                    expected = oracle(payload[block * block_bytes:(block + 1) * block_bytes]).numpy()
                    np.testing.assert_array_equal(flat[block * block_values:(block + 1) * block_values], expected)
                represented = torch.from_numpy(values).to(torch.bfloat16)
                encoded = (encode_nvfp4_from_bf16(represented, spec.shape) if spec.format == 'NVFP4'
                           else encode_source_fp8(represented))
                writer.write(spec.name, encoded)
                rows.append(dict(name=spec.name, source_format=obj.format, shape=spec.shape,
                                 bytes=len(encoded), source_decoder_samples_exact=True))
                print(spec.name, flush=True)
                del packed, payload, values, flat, represented, encoded
    Path(str(output) + '.json').write_text(json.dumps(dict(
        source=str(source.resolve()), artifact=str(output.resolve()),
        source_boundary='GGUF exact decode -> BF16 RNE -> offline native encoding',
        limitation='Not source-BF16 quantization quality or whole-model PPL evidence', matrices=rows),
        indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    build(args.source, args.out)
