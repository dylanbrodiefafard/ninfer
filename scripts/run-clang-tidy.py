#!/usr/bin/env python3
"""Run the pinned clang-tidy over NInfer's C++ and CUDA translation units.

clang-tidy needs a configured CMake tree (compile_commands.json) and the CUDA toolkit, so like
run-unit-tests.sh this prefers the ninfer-builder container (./scripts/dev-setup.sh) and
re-executes itself there. nvcc compile commands are rewritten into clang CUDA commands for
sm_120a, so kernels, launchers, and host code are analyzed by the same configuration
(.clang-tidy). Diagnostics from headers are reported once, not once per including TU.

Usage:
  ./scripts/run-clang-tidy.py                    # every project TU in the build tree
  ./scripts/run-clang-tidy.py --changed          # diagnostics on lines changed vs. origin/HEAD
  ./scripts/run-clang-tidy.py --changed HEAD~3   # diagnostics on lines changed since a ref
  ./scripts/run-clang-tidy.py src/serve/foo.cpp  # named TUs (headers select their includers)

--changed analyzes every TU that is or includes a changed file, and reports only diagnostics on
changed lines, so new code is held to the full check set while existing findings in untouched
code remain a visible backlog (run without --changed to list it). Exit status is 1 when any
diagnostic is reported. The clang-tidy version is pinned below (the Dockerfile build stage
installs the same version); see docs/maintainer/code-quality.md.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

CLANG_TIDY_VERSION = "22.1.8"
CUDA_ARCH = "sm_120a"
ROOT = Path(__file__).resolve().parent.parent
BUILDER = os.environ.get("NINFER_DEV_CONTAINER", "ninfer-builder")
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cu"}
HEADER_SUFFIXES = {".h", ".hpp", ".cuh"}
EXCLUDED_PREFIXES = ("third_party/",)
DIAGNOSTIC = re.compile(
    r"^(?P<path>[^\s:][^:]*):(?P<line>\d+):\d+: (?:warning|error): .*\[[\w.,-]+\]$"
)
HUNK = re.compile(r"^@@ -\S+ \+(?P<start>\d+)(?:,(?P<count>\d+))? @@")
# nvcc arguments that carry over to clang unchanged; every other nvcc flag is dropped.
NVCC_KEEP_WITH_VALUE = {"-I", "-isystem", "-D", "-U", "-include"}
NVCC_KEEP_PREFIX = ("-I", "-D", "-U", "-std=", "-O")
# Device index arithmetic is 32-bit by design and bounded by each kernel's documented shape
# contract, so the widening checks apply to host translation units only (see .clang-tidy).
CUDA_DISABLED_CHECKS = (
    "-bugprone-implicit-widening-of-multiplication-result,-bugprone-misplaced-widening-cast"
)


def in_container() -> bool:
    return Path("/.dockerenv").exists() or Path("/run/.containerenv").exists()


def builder_running() -> bool:
    if shutil.which("docker") is None:
        return False
    probe = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", BUILDER],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.stdout.strip() == "true"


def find_clang_tidy() -> str:
    candidates = [os.environ.get("NINFER_CLANG_TIDY"), "/opt/ninfer-lint/bin/clang-tidy"]
    candidates.append(shutil.which("clang-tidy"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            version = subprocess.run(
                [candidate, "--version"], capture_output=True, text=True, check=True
            ).stdout
            if f"version {CLANG_TIDY_VERSION}" in version:
                return candidate
            sys.exit(
                f"{candidate} is not clang-tidy {CLANG_TIDY_VERSION}:\n{version}"
                f"Install it with: python3 -m pip install clang-tidy=={CLANG_TIDY_VERSION}"
            )
    sys.exit(
        f"clang-tidy {CLANG_TIDY_VERSION} not found; rebuild the builder image "
        f"(NINFER_REBUILD_BUILDER=1 ./scripts/dev-setup.sh) or set NINFER_CLANG_TIDY"
    )


def default_build_dir() -> Path:
    if Path("/build/compile_commands.json").is_file():
        return Path("/build")
    return ROOT / "build"


def project_relative(path: Path) -> str | None:
    try:
        relative = path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return None
    if relative.startswith(EXCLUDED_PREFIXES):
        return None
    return relative


def nvcc_to_clang(arguments: list[str], nvcc: str, source: str) -> list[str]:
    cuda_path = Path(nvcc).resolve().parent.parent
    translated = [
        "clang++",
        "-x",
        "cuda",
        f"--cuda-gpu-arch={CUDA_ARCH}",
        f"--cuda-path={cuda_path}",
        "-nocudalib",
        "-Wno-unknown-cuda-version",
        # clang 22 cannot name CUDA 13.1 and would lower <<<>>> through the pre-9.2
        # cudaConfigureCall API; its newest known SDK selects the current launch API.
        "-Xclang",
        "-target-sdk-version=12.8",
    ]
    # nvcc adds the CCCL headers (cub, thrust, libcu++) implicitly; clang needs them explicit.
    if (cuda_path / "include" / "cccl").is_dir():
        translated += ["-isystem", str(cuda_path / "include" / "cccl")]
    index = 1
    while index < len(arguments):
        argument = arguments[index]
        if argument in NVCC_KEEP_WITH_VALUE and index + 1 < len(arguments):
            translated += [argument, arguments[index + 1]]
            index += 2
            continue
        if argument.startswith(NVCC_KEEP_PREFIX):
            translated.append(argument)
        index += 1
    translated += ["-c", source]
    return translated


def translate_database(build: Path, output: Path) -> dict[str, dict]:
    """Write a clang-compatible compile database and return its project entries by path."""
    entries = json.loads((build / "compile_commands.json").read_text())
    translated: dict[str, dict] = {}
    for entry in entries:
        source = Path(entry["directory"], entry["file"])
        relative = project_relative(source)
        if relative is None or source.suffix not in SOURCE_SUFFIXES:
            continue
        arguments = entry.get("arguments") or shlex.split(entry["command"])
        if Path(arguments[0]).name == "nvcc":
            arguments = nvcc_to_clang(arguments, arguments[0], str(source))
        else:
            # Host flags understood only by GCC (e.g. -Wduplicated-cond) are not errors here.
            arguments = [*arguments, "-Wno-unknown-warning-option"]
        translated[relative] = {
            "directory": entry["directory"],
            "file": str(source),
            "arguments": arguments,
        }
    output.mkdir(parents=True, exist_ok=True)
    (output / "compile_commands.json").write_text(json.dumps(list(translated.values()), indent=1))
    return translated


def header_includers(build: Path, headers: set[str], sources: set[str]) -> set[str]:
    """Map changed headers to the TUs that include them, from Ninja's recorded dependencies."""
    if not headers:
        return set()
    deps = subprocess.run(
        ["ninja", "-C", str(build), "-t", "deps"], capture_output=True, text=True, check=False
    )
    if deps.returncode != 0:
        sys.exit("header changes need a built tree; run the build first (ninja -t deps failed)")
    selected: set[str] = set()
    current: list[str] = []
    for line in [*deps.stdout.splitlines(), ""]:
        if line and not line.startswith(" "):
            current = []
            continue
        if line.strip():
            current.append(line.strip())
            continue
        if not current:
            continue
        paths = {project_relative(Path(build, dep)) for dep in current}
        if paths & headers:
            selected |= paths & sources
        current = []
    return selected


def changed_lines(base: str | None) -> dict[str, set[int] | None]:
    """Changed project files mapped to their added or modified lines (None: the whole file)."""
    # The default base is the remote's default branch, so a feature branch is checked as a whole.
    merge_base = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "HEAD", base or "origin/HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if merge_base.returncode != 0:
        sys.exit(
            f"cannot find a merge base with {base or 'origin/HEAD'}; pass --changed REF "
            "(or run `git remote set-head origin --auto`)"
        )
    merge_base = merge_base.stdout.strip()
    diff = subprocess.run(
        ["git", "-C", str(ROOT), "diff", "-U0", "--no-color", "--diff-filter=d", merge_base],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    changed: dict[str, set[int] | None] = {}
    current: set[int] | None = None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            name = line[len("+++ b/") :]
            current = (
                None if name.startswith(EXCLUDED_PREFIXES) else changed.setdefault(name, set())
            )
        elif current is not None and (hunk := HUNK.match(line)):
            start = int(hunk["start"])
            current.update(range(start, start + int(hunk["count"] or 1)))
    untracked = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--others", "--exclude-standard"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    for name in untracked:
        if not name.startswith(EXCLUDED_PREFIXES):
            changed[name] = None
    return changed


def run_one(clang_tidy: str, database: Path, source: str, cuda: bool, fix: bool) -> str:
    command = [clang_tidy, "--quiet", f"-p={database}", str(ROOT / source)]
    if cuda:
        command.append(f"--checks={CUDA_DISABLED_CHECKS}")
    if fix:
        command.append("--fix")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return result.stdout


def split_diagnostics(
    output: str, lines: dict[str, set[int] | None] | None
) -> list[tuple[str, str]]:
    """Split clang-tidy output into diagnostic blocks, keeping only changed lines if given."""
    blocks: list[tuple[str, list[str]]] = []
    keep = False
    for line in output.splitlines():
        if diagnostic := DIAGNOSTIC.match(line):
            keep = in_scope(diagnostic["path"], int(diagnostic["line"]), lines)
            if keep:
                blocks.append((line, [line]))
        elif keep and blocks:
            blocks[-1][1].append(line)
    return [(head, "\n".join(body)) for head, body in blocks]


def in_scope(path: str, line: int, lines: dict[str, set[int] | None] | None) -> bool:
    """Project files only (not system or third-party headers), and changed lines if given."""
    relative = project_relative(Path(path))
    if relative is None:
        return False
    if lines is None:
        return True
    if relative not in lines:
        return False
    changed = lines[relative]
    return changed is None or line in changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-p", "--build-dir", type=Path, help="CMake tree with compile_commands")
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--changed",
        nargs="?",
        const="",
        metavar="REF",
        help="report only lines changed since REF (default: origin/HEAD)",
    )
    parser.add_argument("--fix", action="store_true", help="apply clang-tidy fix-its")
    parser.add_argument("--inner", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("files", nargs="*", help="source or header paths to check")
    args = parser.parse_args()

    if not args.inner and not in_container() and builder_running():
        forwarded = [argument for argument in sys.argv[1:] if argument != "--inner"]
        command = ["docker", "exec", "-w", "/src", BUILDER, "python3"]
        command += ["/src/scripts/run-clang-tidy.py", "--inner", *forwarded]
        return subprocess.run(command, check=False).returncode

    clang_tidy = find_clang_tidy()
    build = (args.build_dir or default_build_dir()).resolve()
    if not (build / "compile_commands.json").is_file():
        sys.exit(f"{build}/compile_commands.json not found; configure the CMake tree first")
    database = build / "clang-tidy"
    entries = translate_database(build, database)
    sources = set(entries)

    lines: dict[str, set[int] | None] | None = None
    if args.changed is not None:
        lines = changed_lines(args.changed or None)
        requested = set(lines)
    elif args.files:
        requested = {project_relative(Path(name)) or name for name in args.files}
    else:
        requested = sources
    headers = {name for name in requested if Path(name).suffix in HEADER_SUFFIXES}
    selected = sorted((requested & sources) | header_includers(build, headers, sources))
    if not selected:
        print("clang-tidy: no project translation units selected")
        return 0

    print(f"clang-tidy {CLANG_TIDY_VERSION}: {len(selected)} translation units", flush=True)
    seen: set[str] = set()
    reported = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(run_one, clang_tidy, database, s, s.endswith(".cu"), args.fix)
            for s in selected
        ]
        for future in concurrent.futures.as_completed(futures):
            for head, block in split_diagnostics(future.result(), lines):
                if head in seen:
                    continue
                seen.add(head)
                reported += 1
                print(block, flush=True)
    print(f"clang-tidy: {reported} diagnostics")
    return 1 if reported else 0


if __name__ == "__main__":
    sys.exit(main())
