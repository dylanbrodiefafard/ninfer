"""Generate the fixed 328 MiB selective-FP8 recipe from original BF16 weights.

Preserve the original NVFP4 shell, including its BF16 protections, endpoints,
norms and any draft. No source-FP8 bank, alternate checkpoint or GPU is needed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from tools.artifact.container import (Artifact, ArtifactIdentity, ArtifactWriter,
                                      ResourceObject, ResourceSpec, TensorObject, TensorSpec)
from tools.artifact.layouts import encode_fp8_row_scaled
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common.recipe import (Concat, TensorRecipe, expression_shape,
    materialize_recipe, preflight_source_reader)
from tools.convert.qwen3_6_27b import recipe as family_recipe
from tools.convert.qwen3_6_27b.convert import validate_config

RECIPE_ID = 'qwen3_8_27b_nvfp4_selective_fp8_328-v1'
ENCODER = 'BF16 row RNE(maxabs/448); E4M3FN RNE(source/stored_scale), saturate finite'
MATRICES = (
    'text/layers/11/attention/output',
    'text/layers/27/attention/query_key_gate_value',
    'text/layers/31/attention/query_key_gate_value',
    'text/layers/51/attention/query_key_gate_value',
    'text/layers/62/mlp/gate_up', 'text/layers/62/mlp/down',
    'text/layers/63/mlp/gate_up', 'text/layers/63/mlp/down',
)


def _source_recipe(name: str) -> TensorRecipe:
    if name.endswith('/attention/query_key_gate_value'):
        prefix = name.rsplit('/', 1)[0]
        # Family conversion owns the head-interleaved Q/gate split. The stored
        # fused parent is [query, key, gate, value], not HF's [Q+gate, K, V].
        return TensorRecipe(name, Concat(tuple(family_recipe.RECIPES_BY_NAME[prefix+'/'+part].expression
                                              for part in ('query_key', 'gate_value')), 0))
    return family_recipe.RECIPES_BY_NAME[name]


RECIPES = tuple(_source_recipe(name) for name in MATRICES)


def _divisor(name: str) -> str:
    prefix, role = name.rsplit('/', 1)
    return prefix+'/'+{'output':'output', 'query_key_gate_value':'input',
                       'gate_up':'gate_up', 'down':'down'}[role]+'_projection/input_scale_divisor'


def encode_source_fp8(weights: torch.Tensor) -> bytes:
    if weights.dtype != torch.bfloat16 or weights.ndim != 2:
        raise ValueError('source must be a BF16 matrix')
    values = weights.float()
    if not torch.isfinite(values).all():
        raise ValueError('nonfinite source weight')
    maximum = values.abs().amax(dim=1)
    scale = (maximum/448).to(torch.bfloat16)
    if ((maximum != 0) & (scale == 0)).any():
        raise ValueError('BF16 row-scale underflow')
    denominator = torch.where(scale == 0, torch.ones_like(scale), scale).float()
    codes = (values/denominator[:,None]).clamp(-448,448).to(torch.float8_e4m3fn).view(torch.uint8)
    return encode_fp8_row_scaled(codes, scale, tuple(weights.shape))


def _chunks(artifact, obj):
    view = artifact.payload(obj)
    for offset in range(0, len(view), 64 << 20):
        yield view[offset:offset+(64 << 20)]


def convert(base_path: Path, model_path: Path, output: Path) -> Path:
    if output.exists() or output.resolve() == base_path.resolve():
        raise FileExistsError(output)
    config = validate_config(json.loads((model_path/'config.json').read_text()))
    recipes = {r.object_name:r for r in RECIPES}
    removed = {_divisor(name) for name in MATRICES}
    with Artifact(base_path) as base, ShardReader(model_path) as source:
        if base.identity != ArtifactIdentity('qwen3.8-27b','nvfp4'):
            raise ValueError('requires the original qwen3.8-27b/nvfp4 shell')
        if any(isinstance(o,TensorObject) and o.name.startswith('text/')
               and o.format == 'FP8_E4M3FN_ROW_BF16S' for o in base.objects):
            raise ValueError('base already contains FP8; use the original NVFP4 shell')
        source_info = preflight_source_reader(source, RECIPES)
        for name,r in recipes.items():
            obj = base.find(name)
            if (not isinstance(obj,TensorObject) or obj.format != 'NVFP4'
                    or obj.shape != expression_shape(r.expression)):
                raise ValueError('unexpected base matrix: '+name)
            divisor = base.find(_divisor(name))
            if not isinstance(divisor,TensorObject) or divisor.format != 'FP32':
                raise ValueError('unexpected NVFP4 input divisor: '+name)
        specs=[]
        for obj in base.objects:
            if obj.name in removed:
                continue
            if isinstance(obj,ResourceObject):
                specs.append(ResourceSpec(obj.name,obj.encoding,obj.bytes))
            elif obj.name in recipes:
                specs.append(TensorSpec(obj.name,obj.shape,'FP8_E4M3FN_ROW_BF16S','row-scale-v1'))
            else:
                specs.append(TensorSpec(obj.name,obj.shape,obj.format,obj.layout))
        output.parent.mkdir(parents=True,exist_ok=True)
        expected={}
        with ArtifactWriter(output,base.identity,specs) as writer:
            for obj in base.objects:
                if obj.name in removed:
                    continue
                digest=hashlib.sha256()
                if obj.name in recipes:
                    weights=materialize_recipe(recipes[obj.name],source)
                    payload=encode_source_fp8(weights)
                    writer.write(obj.name,payload)
                    digest.update(payload)
                    del weights,payload
                    print('FP8',obj.name,flush=True)
                else:
                    def copied_chunks():
                        for chunk in _chunks(base,obj):
                            digest.update(chunk)
                            yield chunk
                    writer.write(obj.name,copied_chunks())
                expected[obj.name]=digest.hexdigest()
        with Artifact(output) as result:
            for obj in result.objects:
                digest=hashlib.sha256()
                for chunk in _chunks(result,obj):
                    digest.update(chunk)
                del chunk
                if digest.hexdigest()!=expected[obj.name]:
                    raise ValueError('written payload mismatch: '+obj.name)
            extra=sum(o.bytes for o in result.objects)-sum(o.bytes for o in base.objects)
        report=dict(recipe_id=RECIPE_ID,encoder=ENCODER,base_artifact=str(base_path.resolve()),
                    source_bf16=str(model_path.resolve()),config=config,source_preflight=asdict(source_info),
                    matrices=list(MATRICES),removed_divisors=sorted(removed),
                    extra_payload_bytes=extra,all_payloads_verified_exact=True,
                    output=str(output.resolve()),output_bytes=output.stat().st_size)
    report_path=Path(str(output)+'.conversion.json')
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    return report_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-artifact',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    print(convert(args.base_artifact,args.model,args.out))


if __name__=='__main__':
    main()
