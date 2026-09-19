"""Train-only expert A4 multipliers; unchanged native NVFP4 weight codes/scales.

This is a counterfactual source-activation assessment, not a production calibration
default. Down calibration sees its own post-SwiGLU operands, never gate/up inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from tools.artifact.container import Artifact
from tools.parity.qwen4.native_endpoint_corpus import metrics
from tools.parity.qwen4.native_kv_composition import nearest, e4m3_table
from tools.reference.qwen4.common import linear, silu, sigmoid
from tools.reference.qwen4.mtp import Weights

FACTORS = (1., .5, .75, 1.25, 1.5, 2.)
ROLES = ("gate_proj", "up_proj", "down_proj")


class Source:
    def __init__(self, source, layer):
        self.source, self.layer = source, layer

    def get_tensor(self, name):
        return self.source.get_tensor(name.replace("mtp.layers.0.", f"model.language_model.layers.{self.layer}.", 1))


def activation(x, multiplier):
    """Independent native source-multiplier E4M3/E2M1 RNE codec attribution."""
    x = x.bfloat16().float()
    groups = x.reshape(len(x), -1, 16)
    m = torch.tensor(multiplier, dtype=torch.float32)
    _, scale = nearest((groups.abs().amax(-1)/(6*m)).clamp(2**-9, 448), e4m3_table())
    denominator = scale.float()*m
    _, code = nearest(groups/denominator[..., None], torch.tensor(
        [0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.float64))
    return (code*scale[..., None]*m.double()).reshape_as(x)


def select(x, weight, multiplier):
    reference = linear(x, weight)
    losses = []
    for factor in FACTORS:
        value = float(torch.tensor(multiplier*factor, dtype=torch.float32))
        candidate = linear(activation(x, value), weight)
        losses.append((float(((candidate-reference)**2).sum()), value))
    # Original multiplier is first, retaining it on a tie.
    return min(enumerate(losses), key=lambda item: (item[1][0], item[0]))[1][1]


def run(manifest_path, captures, source_root, output):
    if output.exists():
        raise FileExistsError(output)
    oracle_directory=output.parent/(output.stem+"-oracles")
    oracle_directory.mkdir(parents=True,exist_ok=False)
    manifest = json.loads(manifest_path.read_text())
    result = {}
    provenance = {}
    for layer in range(4):
        with safe_open(str(source_root/f"qwen4-layer-{layer}.safetensors"), framework="pt", device="cpu") as source:
            view = Source(source, layer)
            weights = Weights(view)
            router = source.get_tensor(f"model.language_model.layers.{layer}.mlp.gate.weight").double()
            panels = []
            for entry in manifest["panels"]:
                folder = captures/entry["id"]
                meta = json.loads((folder/"capture.json").read_text())
                with Artifact(entry["panel"]) as artifact:
                    tokens = torch.frombuffer(bytearray(artifact.payload("token.ids")), dtype=torch.int32).tolist()
                if meta["tokens"] != tokens or len(tokens) != entry["tokens"]:
                    raise ValueError("capture/token identity mismatch")
                provenance[entry["id"]] = {"split":entry["split"], "tokens":len(tokens),
                    "prefix_failures":meta["prefix_failures"]}
                x = torch.frombuffer(bytearray((folder/f"layer{layer}.moe_input.actual.f32").read_bytes()),
                                     dtype=torch.float32).reshape(entry["tokens"], 2560)
                if not torch.equal(x, x.bfloat16().float()):
                    raise ValueError("unrepresented expert input")
                logits = linear(x, router)
                ids = torch.argsort(logits, dim=1, descending=True, stable=True)[:, :10]
                probability = torch.softmax(logits.gather(1, ids), dim=1)
                panels.append((entry, x.bfloat16(), ids, probability))
            train = [(x, ids) for entry, x, ids, _ in panels if entry["split"]=="calibration"]
            held = [(entry, x, ids, p) for entry, x, ids, p in panels if entry["split"]=="heldout"]
            # Finite study: experts with >=8 calibration occurrences and at least one
            # heldout document meeting the actual >=32-occurrence A4 dispatch threshold.
            eligible = []
            for expert in range(512):
                training = sum(int((ids==expert).any(1).sum()) for _, ids in train)
                if training>=8 and any(int((ids==expert).any(1).sum())>=32 for _, _, ids, _ in held):
                    eligible.append(expert)
            layer_results = {}
            shared_prefix=f"model.language_model.layers.{layer}.mlp."
            shared=[source.get_tensor(shared_prefix+f"shared_expert.{role}.weight") for role in ROLES]
            scalar=source.get_tensor(shared_prefix+"shared_expert_gate.weight")
            complete={}
            for entry,x,_,_ in held:
                value=linear(silu(linear(x,shared[0]))*linear(x,shared[1]),shared[2])*sigmoid(linear(x,scalar))
                complete[entry["id"]]={name:value.clone() for name in ("oracle","source","fitted")}
            observed=sorted(set(torch.cat([ids.flatten() for _,_,ids,_ in held]).tolist()))
            for expert in observed:
                w = [weights.expert(role, expert) for role in ROLES]
                scales = [float(view.get_tensor(f"mtp.layers.0.mlp.experts.{expert}.{role}.input_scale")) for role in ROLES]
                fitted=scales
                if expert in eligible:
                    train_x = torch.cat([x[(ids==expert).any(1)] for x, ids in train])
                    down_x = (silu(linear(train_x,w[0]))*linear(train_x,w[1])).bfloat16()
                    fitted = [select(train_x,w[0],scales[0]), select(train_x,w[1],scales[1]), select(down_x,w[2],scales[2])]
                records = {}
                for entry, x, ids, probability in held:
                    chosen = (ids==expert).any(1)
                    if not chosen.any():
                        continue
                    x = x[chosen]
                    ideal = linear(silu(linear(x,w[0]))*linear(x,w[1]),w[2])
                    def evaluate(values):
                        intermediate = (silu(linear(activation(x,values[0]),w[0]))*
                                        linear(activation(x,values[1]),w[1])).bfloat16()
                        return linear(activation(intermediate,values[2]),w[2])
                    original=evaluate(scales) if len(x)>=32 else ideal
                    candidate=evaluate(fitted) if len(x)>=32 else ideal
                    coefficient=probability[chosen][ids[chosen]==expert].reshape(-1,1)
                    for name,value in (("oracle",ideal),("source",original),("fitted",candidate)):
                        complete[entry["id"]][name][chosen]+=coefficient*value
                    if expert in eligible:
                        records[entry["id"]] = {"occurrences":len(x), "a4_dispatch_active":len(x)>=32,
                            "source_scales":metrics(original,ideal), "fitted_scales":metrics(candidate,ideal)}
                if expert in eligible:
                    layer_results[str(expert)] = {"training_occurrences":len(train_x),
                        "original_multipliers":scales,"fitted_multipliers":fitted,"heldout":records}
                    print("routed",layer,expert,json.dumps(layer_results[str(expert)]),flush=True)
                weights.expert.cache_clear()
            complete_metrics={}
            for name,values in complete.items():
                complete_metrics[name]={label:metrics(values[label],values["oracle"]) for label in ("source","fitted")}
                gpu=torch.frombuffer(bytearray((captures/name/f"layer{layer}.moe_output.actual.f32").read_bytes()),
                                     dtype=torch.float32).reshape(-1,2560)
                reference=values["oracle"]
                gpu_error=metrics(gpu,reference)
                gpu_gross=(gpu-reference).abs().amax(1) > 1/32768+(2/255)*reference.abs().amax(1)
                gpu_error["same_input_oracle_passed"]=gpu_error["max_token_relative_l2"]<=2.5/255 and not bool(gpu_gross.any())
                complete_metrics[name]["captured_a16_gpu"]=gpu_error
                if not gpu_error["same_input_oracle_passed"]:
                    raise ValueError(f"captured A16 GPU disagrees with complete same-input MoE oracle: layer{layer} {name}")
                # Existing accumulated screen, both document and token tails. This
                # source-activation screen is not a substitute for GPU Op qualification.
                reference=values["oracle"]
                for label in ("source","fitted"):
                    delta=values[label]-reference
                    gross=(delta.abs().amax(1) > .005+.02*reference.abs().amax(1))
                    item=complete_metrics[name][label]
                    item.update(gross_token_failures=int(gross.sum()),
                        source_screen_passed=item["relative_l2"]<=.02 and item["max_token_relative_l2"]<=.02 and not bool(gross.any()))
                # Reuse the complete same-input oracle for projection studies; the
                # capture's propagated reference used a DIFFERENT input and is not
                # a valid replacement for this source-loss denominator.
                filename=f"layer{layer}.{name}.f64"
                (oracle_directory/filename).write_bytes(values["oracle"].contiguous().numpy().tobytes())
            (oracle_directory/f"layer{layer}.json").write_text(json.dumps({
                "profile":"qwen4-native-heldout-moe-same-input-fp64",
                "repository":manifest["repository"],"revision":manifest["revision"],
                "layer":layer,"panels":[{"id":entry["id"],"tokens":json.loads((captures/entry["id"]/"capture.json").read_text())["tokens"],
                    "file":f"layer{layer}.{entry['id']}.f64","shape":[entry["tokens"],2560]}
                    for entry,_,_,_ in held],
                "boundary":"Complete independent FP64 MoE from each capture's represented actual BF16 input, original decoded source weights, exact same-input router and shared expert. No propagated reference or private A4 operands."},indent=2)+"\n")
            result[str(layer)] = {"experts":layer_results,"complete_moe":complete_metrics}
            print("complete MoE",layer,json.dumps(complete_metrics),flush=True)
    output.write_text(json.dumps({"manifest":str(manifest_path),"panels":provenance,"layers":result,
        "factors":FACTORS,"oracle":"Exact stored signed NVFP4 weights and independent FP64 complete expert formula; activation quantization is a candidate, not supplied to the oracle.",
        "scope":"Source multipliers versus separate train-only gate/up/down projection fits; exact routed coefficients and unchanged shared expert included in complete-MoE source-activation assessment, not GPU Op/PPL/default admission. Only >=32-occurrence document groups use candidate A4; smaller groups use the ideal A16 formula. Source weights unchanged. Public-input prefix failures remain explicit."},indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--captures",type=Path,required=True)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    run(args.manifest,args.captures,args.source,args.out)
