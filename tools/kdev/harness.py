"""Harness: run the build/test/bench/profiler steps inside the dev container.

Everything GPU-touching happens in the shared dev (builder) container so the host
stays clean. The host only orchestrates. The container has the repo bind-mounted
at /src and the shared test+bench build tree at /build (BUILD_TESTING +
NINFER_BUILD_BENCHMARKS are ON there, so op test + bench targets are always
incrementally available).

The container name is overridable via NINFER_DEV_CONTAINER (default ninfer-builder).
If it is not running, the control plane refuses to guess and points at
scripts/dev-setup.sh instead of silently starting something it shouldn't own.
"""

import os
import re
import subprocess
from dataclasses import dataclass

CONTAINER = os.environ.get("NINFER_DEV_CONTAINER", "ninfer-builder")
REPO = "/src"  # repo bind-mount inside the container
BUILD = "/build"  # shared test+bench build tree (tests+bench ON)
JOBS = os.environ.get("NINFER_DEV_JOBS", "8")


class HarnessError(RuntimeError):
    pass


@dataclass
class Result:
    ok: bool
    code: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return self.stdout + ("\n" + self.stderr if self.stderr.strip() else "")


def _container_running() -> bool:
    probe = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER],
        capture_output=True, text=True,
    )
    return probe.returncode == 0 and probe.stdout.strip() == "true"


def run(cmd: str, *, check: bool = True, env: dict | None = None,
        timeout: float = 1800.0) -> Result:
    """Run a shell command inside the dev container; return captured output."""
    if not _container_running():
        raise HarnessError(
            f"dev container '{CONTAINER}' is not running. Start it (idempotent) with "
            f"./scripts/dev-setup.sh, or point NINFER_DEV_CONTAINER elsewhere."
        )
    argv = ["docker", "exec"]
    for key, value in (env or {}).items():
        argv += ["-e", f"{key}={value}"]
    argv += [CONTAINER, "bash", "-lc", cmd]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    result = Result(proc.returncode == 0, proc.returncode, proc.stdout, proc.stderr)
    if check and proc.returncode != 0:
        raise HarnessError(
            f"command failed (exit {proc.returncode}): {cmd}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return result


def build_target(target: str) -> Result:
    """Incremental build of a CMake target in the shared tree."""
    return run(f"cd {BUILD} && cmake --build . --target {target} --parallel {JOBS}",
               check=False)


def test_binary(op) -> str:
    return f"{BUILD}/tests/{op.test_target}"


def bench_binary(op) -> str:
    return f"{BUILD}/bench/{op.bench_target}"


def repo_head() -> str:
    """Short git HEAD of the repo. The repo is bind-mounted, so git runs on the
    host (the builder container may not have git). Assumes CWD is the repo root."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=os.getcwd(),
        ).stdout.strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def build_stamp(op) -> str:
    """A cheap 'what is the code I'm about to measure' stamp: git HEAD + mtime of the
    op test/bench objects is overkill; HEAD + target names is the tracked identity."""
    return f"git@{repo_head()}/{op.test_target}+{op.bench_target}"


_OP_STATS_RE = re.compile(r"^OP_ERROR_STATS\s+(?P<kv>.*)$", re.M)
_KV_RE = re.compile(r"([a-z_]+)=([^ ]+)")
_LABEL_RE = re.compile(r"case=(.+)$")


def parse_op_stats(text: str) -> list:
    """Parse OP_ERROR_STATS lines into a list of dicts (stable, LLM-readable)."""
    records = []
    for line in _OP_STATS_RE.findall(text):
        record = {key: value for key, value in _KV_RE.findall(line)}
        label = _LABEL_RE.search(line)
        if label:
            record["case"] = label.group(1).strip()
        records.append(record)
    return records