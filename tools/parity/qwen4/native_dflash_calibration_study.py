"""Fit on disjoint real-feature calibration documents, then evaluate frozen DFlash weights.

The captured producer is the unregistered UD-IQ1_S diagnostic, not native NVFP4.
These tools never execute a target or stream ordinary native model weights.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from tools.artifact.container import Artifact, ArtifactIdentity, ArtifactWriter, TensorSpec
from tools.artifact.layouts import encode_nvfp4
from tools.parity.qwen4.native_dflash_calibrate import Moments, PROFILE, SCALE_FACTORS, calibrate_matrix
from tools.parity.qwen4.native_dflash_fixture import SHAPES, REPOSITORY, REVISION
from tools.parity.qwen4.native_dflash_goldens import read_weights
from tools.reference.qwen4.dflash import block_inputs, decode_nvfp4_words_exact, forward

CONTEXTS = (32, 64, 128)
# Predeclared bounded source-loss screen, NOT the same-input Op gate, target PPL
# budget or a declaration that these prompts can admit a draft default.
SOURCE_SCREEN = (.02, .005, .02)


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2); stream.write("\n")


def panels(corpus, capture):
    manifest = json.loads(corpus.read_text())
    feature_report = json.loads((capture / "qwen4-dflash-corpus-features.json").read_text())
    if (feature_report["profile"] != "qwen4-ud-iq1-s-diagnostic-disjoint-corpus" or
            feature_report["taps"] != [4,16,24,36,44] or feature_report["qsa_kv"] != "NVFP4-G16" or
            feature_report["corpus_repository"] != manifest["repository"] or
            feature_report["corpus_revision"] != manifest["revision"]):
        raise ValueError("feature/corpus authority differs")
    captured = {item["id"]: item for item in feature_report["panels"]}
    if len(captured) != len(feature_report["panels"]) or set(captured) != {p["id"] for p in manifest["panels"]}:
        raise ValueError("duplicate or missing captured documents")
    seen = set(); result = []
    for item in manifest["panels"]:
        record = captured[item["id"]]
        if record["split"] != item["split"] or record["accepted_prompt_tokens"] != 128:
            raise ValueError("split/accepted extent mismatch")
        with Artifact(Path(item["panel"])) as panel:
            if panel.identity != ArtifactIdentity("qwen4/native-text-qualification", "nvidia-source-corpus"):
                raise ValueError("calibration requires independently declared corpus panels")
            tokens = torch.frombuffer(bytearray(panel.payload("token.ids")), dtype=torch.int32).tolist()
        if tokens != record["token_ids"] or len(tokens) != item["tokens"] or len(tokens) < 136:
            raise ValueError("captured token identity differs")
        fingerprint = tuple(tokens[:128])
        if fingerprint in seen:
            raise ValueError("duplicate accepted prompt across corpus documents/splits")
        seen.add(fingerprint)
        result.append(dict(item, capture=record))
    if any(sum(p["split"] == split for p in result) < 4 for split in ("calibration", "heldout")):
        raise ValueError("requires at least four calibration and four held-out documents")
    return result, feature_report


def load_panel(item, capture):
    with Artifact(Path(item["panel"])) as panel:
        anchors = torch.frombuffer(bytearray(panel.payload("token.embeddings")), dtype=torch.bfloat16).reshape(-1,2560)
    values = (capture / item["capture"]["features"]).read_bytes()
    if len(values) != 128*12800*2:
        raise ValueError("captured feature extent differs")
    features = torch.frombuffer(bytearray(values), dtype=torch.bfloat16).reshape(128,12800)
    if not torch.isfinite(features).all() or not torch.isfinite(anchors).all():
        raise ValueError("nonfinite captured public inputs")
    return features, anchors


def mask_embedding(draft):
    with Artifact(draft / "qwen4-dflash-inputs.ninfer") as inputs:
        return torch.frombuffer(bytearray(inputs.payload("mask.embedding")), dtype=torch.bfloat16).clone()


def fit(corpus, capture, draft, output):
    if output.exists():
        raise FileExistsError(output)
    items, provenance = panels(corpus, capture)
    source = read_weights(draft / "qwen4-dflash-bf16.ninfer")
    mask = mask_embedding(draft)
    moments = Moments()
    ids = []
    for item in items:
        if item["split"] != "calibration":
            continue  # Held-out feature and embedding bytes never enter fitting.
        features, anchors = load_panel(item, capture)
        ids.append(item["id"])
        for count in CONTEXTS:
            embeddings, positions = block_inputs(anchors[count], mask, count, 7)
            result = forward(features[:count], embeddings.bfloat16(), torch.arange(count), positions,
                             source, materialize_public_bf16=True)
            moments.observe(features[:count], result)
            print("calibration trace", item["id"], count, flush=True)
    output.mkdir(parents=True)
    specs = [TensorSpec(name, shape, "NVFP4" if len(shape)==2 else "BF16",
        "blockscale-k16-m128x4-v1" if len(shape)==2 else "contiguous-le-v1") for name,shape in SHAPES.items()]
    matrix_roles = {name for name, shape in SHAPES.items() if len(shape)==2}
    if set(moments.squares) != matrix_roles:
        raise AssertionError("incomplete independent DFlash activation capture")
    summaries = {}
    for profile in ("weight_mse", "input_mse"):
        path = output / (profile + ".ninfer")
        summary = {}
        with ArtifactWriter(path, ArtifactIdentity("qwen4/dflash-calibration-assessment", profile), specs) as writer:
            for spec in specs:
                weight = source[spec.name]
                if spec.format == "BF16":
                    payload = weight.view(torch.uint16).numpy().tobytes()
                else:
                    moment = moments.mean(spec.name) if profile == "input_mse" else torch.ones(weight.shape[1])
                    codes, scales, divisor, stats = calibrate_matrix(weight, moment)
                    payload = encode_nvfp4(codes, scales, divisor, spec.shape)
                    actual = decode_nvfp4_words_exact(payload, spec.shape)
                    if (not torch.equal(actual[0], codes) or not torch.equal(actual[1], scales) or
                            not torch.equal(actual[2].view(torch.int32), divisor.view(torch.int32))):
                        raise AssertionError("independent exact NVFP4 code/scale/divisor roundtrip failed")
                    summary[spec.name] = dict(stats, calibration_rows=moments.counts[spec.name])
                writer.write(spec.name, payload)
                print("fitted", profile, spec.name, flush=True)
        # The norm bytes are a protected public input, not fitted parameters.
        with Artifact(path) as artifact:
            for name, shape in SHAPES.items():
                if len(shape)==1 and bytes(artifact.payload(name)) != source[name].view(torch.uint16).numpy().tobytes():
                    raise AssertionError("protected source norm changed")
        summaries[profile] = summary
    write_json(output / "fit.json", dict(profile=PROFILE, source_repository=REPOSITORY, source_revision=REVISION,
        corpus_repository=provenance["corpus_repository"], corpus_revision=provenance["corpus_revision"],
        calibration_ids=ids, contexts=CONTEXTS, scale_factors=SCALE_FACTORS,
        source_loss_screen=SOURCE_SCREEN, summaries=summaries,
        screening_decision="All held-out final-hidden queries must satisfy the source-loss screen; head metrics are supplementary. Passing would still not admit a runtime/default.",
        objective="sum_k E_calibration[x_k^2]*(W_hat-W)^2 per K16 group; cross-column covariance omitted; weight_mse uses unit moments",
        boundary="Tool-only frozen artifacts, no runtime/default admission. All BF16 norms and original FP32 matrix divisors protected. No held-out features, labels or results used to fit."))


def errors(actual, expected):
    delta = actual.double()-expected.double()
    relative = delta.norm(dim=-1)/expected.double().norm(dim=-1).clamp_min(1e-300)
    maximum = delta.abs().amax(dim=-1)
    limit = SOURCE_SCREEN[1]+SOURCE_SCREEN[2]*expected.double().abs().amax(dim=-1)
    return dict(relative_l2=float(delta.norm()/expected.double().norm().clamp_min(1e-300)),
        per_query_relative_l2=relative.tolist(), per_query_max_abs=maximum.tolist(),
        source_screen_passed=bool(torch.isfinite(delta).all()) and
            bool((relative<=SOURCE_SCREEN[0]).all()) and bool((maximum<=limit).all()))


def head_logits(hidden, head):
    # Bound FP64 expansion of the authentic full vocabulary head; no sampling mask.
    return torch.cat([hidden.double() @ head[start:start+4096].double().T
                      for start in range(0, len(head), 4096)], dim=-1)


def evaluate(corpus, capture, draft, fitted, native, regression, output):
    if output.exists():
        raise FileExistsError(output)
    items, provenance = panels(corpus, capture)
    fitting = json.loads((fitted / "fit.json").read_text())
    if (fitting["profile"] != PROFILE or fitting["contexts"] != list(CONTEXTS) or
            fitting["scale_factors"] != list(SCALE_FACTORS) or fitting["source_loss_screen"] != list(SOURCE_SCREEN) or
            fitting["calibration_ids"] != [p["id"] for p in items if p["split"]=="calibration"] or
            fitting["corpus_repository"] != provenance["corpus_repository"] or
            fitting["corpus_revision"] != provenance["corpus_revision"]):
        raise ValueError("frozen calibration authority differs")
    source = read_weights(draft / "qwen4-dflash-bf16.ninfer")
    mask = mask_embedding(draft)
    cases = []
    for item in items:
        features, anchors = load_panel(item, capture)
        for count in CONTEXTS:
            noise, positions = block_inputs(anchors[count], mask, count, 7)
            cases.append(dict(id=item["id"], split=item["split"], count=count,
                features=features[:count], noise=noise.bfloat16(), positions=positions,
                labels=item["capture"]["token_ids"][count+1:count+8]))
    old = json.loads((regression / "qwen4-dflash-target-features.json").read_text())
    if old["profile"] != "qwen4-ud-iq1-s-diagnostic-accepted-prompt" or old["accepted_prompt_tokens"] != 24:
        raise ValueError("old failed recipe regression provenance differs")
    features = torch.frombuffer(bytearray((regression / "qwen4-dflash-target-features.bin").read_bytes()),
                                dtype=torch.bfloat16).reshape(24,12800)
    with Artifact(native / "qwen4-text-panel.ninfer") as panel:
        ids = torch.frombuffer(bytearray(panel.payload("token.ids")), dtype=torch.int32).tolist()
        if ids != old["token_ids"]:
            raise ValueError("old regression token identity differs")
        anchors = torch.frombuffer(bytearray(panel.payload("token.embeddings")), dtype=torch.bfloat16).reshape(33,2560)
    for count in (8,16,24):
        noise, positions = block_inputs(anchors[count], mask, count, 7)
        cases.append(dict(id="legacy33", split="regression", count=count, features=features[:count],
            noise=noise.bfloat16(), positions=positions, labels=ids[count+1:count+8]))
    with Artifact(native / "qwen4-endpoint.ninfer") as endpoint:
        payload = endpoint.payload("lm_head.weight")
        # Copy-free CPU mapping stays owned until all synchronous calculations finish.
        head = torch.frombuffer(payload, dtype=torch.bfloat16).reshape(248320,2560)
        for case in cases:
            reference = forward(case["features"], case["noise"], torch.arange(case["count"]),
                case["positions"], source, materialize_public_bf16=True)
            case["reference"] = reference.hidden
            case["reference_layers"] = [layer["hidden"] for layer in reference.layers]
            case["reference_logits"] = head_logits(reference.hidden, head)
        results = {}
        for profile, path in (("maxabs_regression", draft/"qwen4-dflash-nvfp4.ninfer"),
                              ("weight_mse", fitted/"weight_mse.ninfer"),
                              ("input_mse", fitted/"input_mse.ninfer")):
            weights = read_weights(path)
            records = []
            for case in cases:
                actual = forward(case["features"], case["noise"], torch.arange(case["count"]),
                    case["positions"], weights, materialize_public_bf16=True)
                logits = head_logits(actual.hidden, head)
                expected = case["reference_logits"]
                a, b = expected.log_softmax(-1), logits.log_softmax(-1)
                labels = torch.tensor(case["labels"])
                row = torch.arange(7)
                record = dict(id=case["id"], split=case["split"], context=case["count"],
                    hidden=errors(actual.hidden, case["reference"]), logits=errors(logits, expected),
                    per_layer_hidden=[errors(layer["hidden"], reference) for layer,reference in
                                      zip(actual.layers, case["reference_layers"])],
                    top1_agreement=int((logits.argmax(-1)==expected.argmax(-1)).sum()),
                    kl_bf16_to_candidate_nats=(a.exp()*(a-b)).sum(-1).tolist(),
                    label_nll_delta=(a[row,labels]-b[row,labels]).tolist())
                records.append(record)
                print("evaluate", profile, case["split"], case["id"], case["count"],
                      record["hidden"]["relative_l2"], record["top1_agreement"], flush=True)
            results[profile] = records
            del weights; gc.collect()
        del head; payload.release()
    summary = {}
    for profile, records in results.items():
        summary[profile] = {}
        for split in ("calibration", "heldout", "regression"):
            selected = [record for record in records if record["split"] == split]
            summary[profile][split] = dict(documents=len({r["id"] for r in selected}),
                overlapping_context_cases=len(selected), queries=7*len(selected),
                worst_hidden_query_relative_l2=max(max(r["hidden"]["per_query_relative_l2"]) for r in selected),
                final_hidden_source_screen_passed=all(r["hidden"]["source_screen_passed"] for r in selected),
                top1_agreement=sum(r["top1_agreement"] for r in selected),
                mean_kl_nats=sum(sum(r["kl_bf16_to_candidate_nats"]) for r in selected)/(7*len(selected)),
                mean_point_label_nll_delta=sum(sum(r["label_nll_delta"]) for r in selected)/(7*len(selected)))
    write_json(output, dict(fit=fitting, producer=provenance, results=results, summary=summary,
        source_loss_screen=SOURCE_SCREEN,
        oracle="Unchanged complete independent FP64 DFlash formula with explicit public BF16 Op/state stores; exact stored NVFP4 decode",
        head_formula="Independent FP64 dot from represented BF16 final draft hidden and source BF16 head; no final BF16 logit store",
        distribution="Full248320 vocabulary, temperature1, no extra mask; not Engine p-less/epsilon law",
        boundary="Separate documents for fitting and held-out evaluation, overlapping context lengths within each document. Teacher-forced UD-IQ1_S diagnostic features, NOT native NVFP4 features. Source-loss screen is not a kernel-correctness gate, model PPL, speculative acceptance or default admission."))
    return 0 if all(r["hidden"]["source_screen_passed"] for r in results["input_mse"]
                    if r["split"]=="heldout") else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fit", "evaluate"))
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fitted", type=Path)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--regression", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.mode == "fit":
        fit(args.corpus, args.capture, args.draft, args.output)
    else:
        if not all((args.fitted, args.native, args.regression)):
            parser.error("evaluate requires --fitted, --native and --regression")
        raise SystemExit(evaluate(args.corpus, args.capture, args.draft, args.fitted,
                                 args.native, args.regression, args.output))
