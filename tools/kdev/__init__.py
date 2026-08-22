# tools/kdev — fast-signal kernel dev control plane for NInfer.
#
# One command drives the low-noise loop:
#   python3 -m tools.kdev <op> [--fast|--full] [--bench] [--profile] [--san] [--json]
#
# It runs inside the shared dev (builder) container so the host stays free of the
# CUDA toolchain, builds the op test + bench incrementally, runs the independent
# oracle (with NINFER_OP_REPORT_STATS), optionally benches, profiles, and runs
# compute-sanitizer, then emits ONE JSON verdict + a JSONL trend row + a human table.
#
# The oracle must pass before a bench number is reported (no silent "fast-but-wrong").