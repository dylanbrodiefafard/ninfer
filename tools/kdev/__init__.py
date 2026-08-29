# tools/kdev — kernel iteration control plane for NInfer.
#
#   python3 -m tools.kdev recipe
#   python3 -m tools.kdev recipe --preset attn_in --t 1 --idea occupancy
#   python3 -m tools.kdev bound --preset attn_in --t 1 --idea occupancy
#   python3 -m tools.kdev mma
#   python3 -m tools.kdev <op> [--fast|--full] [--bench] [--profile] [--san] [--json]
#
# Layer 0–3: docs/maintainer/kernel-iteration.md
