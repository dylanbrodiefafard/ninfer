"""San: memory-axis check via compute-sanitizer (memcheck + racecheck).

The numeric oracle (oracle.py) catches value divergence; this catches the
memory-class bugs the oracle can't see (out-of-bounds, misaligned access, races,
leaks) by running the Op's fast test under compute-sanitizer. Complements the
existing GuardedDeviceBuffer canaries rather than replacing them.
"""

from . import harness


def run_sanitize(op, tier: str = "fast", tool: str = "memcheck") -> dict:
    """tool in {'memcheck','racecheck','initcheck','synccheck','leakcheck'}."""
    args = list(op.fast_test_args if tier == "fast" else op.full_test_args)
    cmd = (
        f"cd {harness.BUILD} && compute-sanitizer --tool {tool} "
        f"--print-limit 8 --exit-code 1 "
        f"{harness.test_binary(op)} " + " ".join(args)
    )
    result = harness.run(cmd, check=False)
    output = result.output
    # compute-sanitizer prints a "== XXX ==" error block per finding; a clean run
    # ends with a summary and (with --exit-code 1) exits 0 only when clean.
    errors = _count_findings(output)
    return {
        "op": op.name,
        "tier": tier,
        "tool": tool,
        "clean": result.ok and errors == 0,
        "findings": errors,
        "tail": output.strip().splitlines()[-1] if output.strip() else "",
    }


def _count_findings(text: str) -> int:
    count = 0
    for line in text.splitlines():
        if line.startswith("== ") and ("ERROR SUMMARY" not in line):
            # individual finding header lines, e.g. "==1234== Invalid write ..."
            count += 1
    return count