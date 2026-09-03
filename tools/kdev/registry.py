"""Data-driven Op registry for the kdev control plane.

Adding a new Op to the control plane is one entry here. Keep the argv lists in
lock-step with the Op's test/bench binaries:
  * fast_test_args  -> the cheapest single case (fast tier, seconds)
  * full_test_args  -> the representative matrix (empty = the binary's default = all)
  * fast_bench_args -> the cheap bench point
  * full_bench_args -> the representative bench point (positions/lengths/content)

``dump=True`` marks an Op whose kernel carries the op_dump.h intermediate side-band
(P2); the diff.py step becomes available for it.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Op:
    name: str
    test_target: str
    bench_target: str
    fast_test_args: list
    full_test_args: list = field(default_factory=list)
    fast_bench_args: list = field(default_factory=list)
    full_bench_args: list = field(default_factory=list)
    ncu_metrics: list = field(default_factory=list)  # optional ncu --metrics
    dump: bool = False  # op_dump.h intermediate side-band present (P2)
    sage: bool = False  # carries the SageAttention3 (nvfp4s3) oracle + FP4-P floor mode
    s3_dump: bool = False  # s3 prefill op-dump side-band (per-stage P-quant/PV dump + --s3-dump)


OPS = {
    "l2norm": Op(
        name="l2norm",
        test_target="ninfer_l2norm_test",
        bench_target="ninfer_l2norm_bench",
        fast_test_args=["--fast"],
        full_test_args=[],
        fast_bench_args=["--tokens", "1"],
        full_bench_args=["--tokens", "1024"],
        dump=True,  # l2norm_dump side-band wired (P2)
    ),
    # Default CTest is the route-boundary unit matrix. `--full` is the complete
    # Cartesian (kdev --full). Bench = the NVFP4-S3 prefill route.
    "gqa_attention": Op(
        name="gqa_attention",
        test_target="ninfer_gqa_attention_test",
        bench_target="ninfer_gqa_attention_nvfp4s3_bench",
        fast_test_args=[],
        full_test_args=["--full"],
        fast_bench_args=[],
        full_bench_args=[],
        dump=False,
        sage=True,  # --sage mode: GQA_SAGE_ONLY conformance + GQA_SAGE_FLOOR decomposition
        s3_dump=True,  # s3 prefill op-dump side-band (score/psf/p_code/v_scale/m/l/acc + diff)
    ),
}


def get(name: str) -> Op:
    if name not in OPS:
        raise KeyError(
            f"unknown op '{name}'; registered: {', '.join(sorted(OPS))}. "
            f"Add it to tools/kdev/registry.py."
        )
    return OPS[name]


def names() -> list:
    return sorted(OPS)