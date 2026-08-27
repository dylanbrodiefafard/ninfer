"""Print the Layer 0–3 kernel-iteration recipe.

The full contract is docs/maintainer/kernel-iteration.md. This command prints the
card an agent must fill before writing CUDA.

    python3 -m tools.kdev recipe
"""

from __future__ import annotations

import argparse

CARD = """\
NInfer kernel iteration — fill this card before writing CUDA.
Authority: docs/maintainer/kernel-iteration.md

Layer 0  bound (host, no GPU)
  python3 -m tools.kdev bound --preset attn_in --t <T> --qtype nvfp4 --idea <class>
  python3 -m tools.kdev bound --n N --k K --t T --qtype nvfp4 --measured-us <us> --json
  STOP if verdict=refuse or sm120=ILLEGAL.

Layer 1  legality + issue roof (once per machine, or after mma.cuh changes)
  python3 -m tools.kdev mma
  Re-run bound; it reads profiles/kdev/mma_issue.json for t_issue.
  Constraints: warp mma.sync only, smem <= 99 KiB, cluster=1, no TMEM/tcgen05.

Layer 2  parameter sweep on the public Op (or a temporary candidate grid)
  python3 -m tools.kdev <op> --fast --bench
  python3 -m tools.kdev <op> --fast --bench --profile   # one named NCU question
  Search tiles/stages/warps inside the current family. Delete losers.

Layer 3  production path
  Qualify against the independent oracle, then the public Op bench at the exact point.
  Engine A/B only if the deliverable is end-to-end tok/s.

Gate card (print answers; 'unknown' means measure, do not implement):
  1. Public Op, exact (N,K,T,qtype,policy), prefill vs decode.
  2. model_bytes, useful_flops, AI, t_mem, t_comp[, t_issue]. Bound.
  3. Current measured µs and % of the matching roof (1674.5 GB/s or 1676 TFLOP/s).
  4. Idea class. If it does not attack the bound, stop.
  5. SM120 legality: MMA atom, smem, TMA, registers. TMEM/tcgen05 => stop.
  6. Parameters inside an existing family, or why a new family is required.
  7. Microbench the public Op at that exact point. No Engine A/B until the Op wins.
  8. If it loses, delete the candidate. Do not leave a second path.

Idea classes: occupancy more_smem software_pipeline tma tile_shape epilogue_fusion
  split_k weight_replay aggregate_T force_mma_small_t flashattention_tiling new_family
  tcgen05 tmem sm100 2sm_mma cluster_multicast
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="kdev recipe", description=__doc__)
    parser.parse_args(argv)
    print(CARD.rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
