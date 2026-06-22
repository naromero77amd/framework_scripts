#!/usr/bin/env python3
"""Generate a running markdown analysis for a framework run_tests.py log."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import platform
import re
import subprocess
from pathlib import Path


STATUS_RE = re.compile(r"^(?:✓|✗)\s+(PASSED|SKIPPED|ERROR|FAILED|TIMEDOUT)\s+\(([\d.]+)s\)")
PROGRESS_RE = re.compile(r"^\[(\d+)/(\d+)\]\s*$")
RUNNING_RE = re.compile(r"^Running:\s+(.+)$")
SUMMARY_RE = re.compile(r"^([A-Za-z ]+):\s+(\d+)\s*$")


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def run_text(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
    except Exception:
        return ""


def python_expr(expr: str) -> str:
    code = f"import torch, triton, platform; print({expr})"
    return run_text(["python", "-c", code], cwd=Path("/tmp"))


def test_suite(node_id: str) -> str:
    return node_id.split("::", 1)[0]


def failure_signature(lines: list[str]) -> str:
    patterns = (
        "RuntimeError:",
        "AssertionError:",
        "ImportError:",
        "ModuleNotFoundError:",
        "Timeout",
        "Failed:",
        "ERROR ",
        "E   ",
    )
    for line in lines:
        stripped = line.strip()
        if any(token in stripped for token in patterns):
            return stripped[:240]
    for line in reversed(lines):
        stripped = line.strip()
        if stripped:
            return stripped[:240]
    return "(no failure detail found)"


def parse_log(log_path: Path) -> tuple[list[dict[str, object]], dict[str, int]]:
    results: list[dict[str, object]] = []
    final_summary: dict[str, int] = {}
    current: dict[str, object] | None = None
    current_lines: list[str] = []
    in_final_summary = False

    for raw_line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.rstrip("\n")
        if "TEST SUMMARY" in line:
            in_final_summary = True
            continue
        if in_final_summary:
            match = SUMMARY_RE.match(line.strip())
            if match:
                final_summary[match.group(1).strip()] = int(match.group(2))

        progress = PROGRESS_RE.match(line.strip())
        if progress:
            if current is not None and "state" not in current:
                current["state"] = "in_progress"
                current["details"] = current_lines
                results.append(current)
            current = {
                "index": int(progress.group(1)),
                "total": int(progress.group(2)),
                "name": None,
            }
            current_lines = []
            continue

        if current is not None:
            current_lines.append(line)
            running = RUNNING_RE.match(line.strip())
            if running:
                current["name"] = running.group(1)
                continue
            status = STATUS_RE.match(line.strip())
            if status:
                current["state"] = status.group(1).lower()
                current["time"] = float(status.group(2))
                current["details"] = current_lines[:]
                results.append(current)
                current = None
                current_lines = []

    if current is not None and "state" not in current:
        current["state"] = "in_progress"
        current["details"] = current_lines
        results.append(current)

    return results, final_summary


def load_checkpoint(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def format_count_table(rows: list[tuple[str, collections.Counter]]) -> str:
    header = "| Test Suite | Total | Passed | Skipped | Failed | Error | Timed Out | In Progress |\n"
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    body = []
    for suite, counter in rows:
        total = sum(counter.values())
        body.append(
            "| {suite} | {total} | {passed} | {skipped} | {failed} | {error} | {timedout} | {in_progress} |".format(
                suite=suite,
                total=total,
                passed=counter["passed"],
                skipped=counter["skipped"],
                failed=counter["failed"],
                error=counter["error"],
                timedout=counter["timedout"],
                in_progress=counter["in_progress"],
            )
        )
    return header + sep + "\n".join(body) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meta",
        default="/home/niromero/docker_workspace/pytorch/inductor_core_latest_run.env",
        help="Path to the active run metadata file.",
    )
    parser.add_argument("--output", default=None, help="Markdown output path.")
    args = parser.parse_args()

    meta_path = Path(args.meta)
    meta = read_env_file(meta_path)
    log_path = Path(meta["LOG"])
    checkpoint_path = Path(meta["CHECKPOINT"])
    output_path = Path(args.output) if args.output else log_path.with_name("inductor_core_running_analysis.md")
    pytorch_root = log_path.parent

    results, final_summary = parse_log(log_path)
    checkpoint = load_checkpoint(checkpoint_path)

    state_counts = collections.Counter(str(r.get("state")) for r in results)
    suite_counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    suite_failures: dict[str, list[dict[str, object]]] = collections.defaultdict(list)

    for result in results:
        name = str(result.get("name") or "(unknown)")
        suite = test_suite(name)
        state = str(result.get("state"))
        suite_counts[suite][state] += 1
        if state in {"failed", "error", "timedout"}:
            suite_failures[suite].append(result)

    completed = 0
    total = 0
    if checkpoint:
        last_index = checkpoint.get("last_index")
        total = int(checkpoint.get("total") or 0)
        if isinstance(last_index, int):
            completed = last_index + 1
    if not total:
        totals = [int(r.get("total") or 0) for r in results if r.get("total")]
        total = totals[-1] if totals else 0
        completed = max((int(r.get("index") or 0) for r in results), default=0)

    branch = run_text(["git", "branch", "--show-current"], cwd=pytorch_root)
    commit = run_text(["git", "rev-parse", "--short", "HEAD"], cwd=pytorch_root)
    arch = platform.machine()
    torch_version = python_expr("torch.__version__")
    torch_git = python_expr("torch.version.git_version")
    rocm_version = python_expr("torch.version.hip")
    triton_version = python_expr("triton.__version__")
    gpu_count = python_expr("torch.cuda.device_count()")
    gpu_name = python_expr("torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable'")

    now = dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")
    pct = (completed / total * 100) if total else 0.0

    lines: list[str] = []
    lines.append("# Inductor Core Running Analysis\n")
    lines.append("## Machine And Build\n")
    lines.append(f"- Updated: `{now}`")
    lines.append(f"- Machine architecture: `{arch}`")
    lines.append(f"- ROCm/HIP version: `{rocm_version}`")
    lines.append(f"- PyTorch branch: `{branch}`")
    lines.append(f"- PyTorch commit: `{commit}`")
    lines.append(f"- PyTorch version: `{torch_version}`")
    lines.append(f"- PyTorch git version: `{torch_git}`")
    lines.append(f"- Triton version: `{triton_version}`")
    lines.append(f"- GPU count: `{gpu_count}`")
    lines.append(f"- GPU model: `{gpu_name}`")
    lines.append(f"- Test log: `{log_path}`")
    lines.append(f"- Checkpoint: `{checkpoint_path}`\n")

    lines.append("## Overall Progress\n")
    lines.append(f"- Completed: `{completed} / {total}` (`{pct:.2f}%`)")
    lines.append(f"- Passed: `{state_counts['passed']}`")
    lines.append(f"- Skipped: `{state_counts['skipped']}`")
    lines.append(f"- Failed: `{state_counts['failed']}`")
    lines.append(f"- Error: `{state_counts['error']}`")
    lines.append(f"- Timed out: `{state_counts['timedout']}`")
    if checkpoint:
        lines.append(f"- Last completed test: `{checkpoint.get('last_test')}`")
        lines.append(f"- Next test: `{checkpoint.get('next_test')}`")
        lines.append(f"- Checkpoint updated: `{checkpoint.get('updated')}`")
    if final_summary:
        lines.append(f"- Final summary detected: `{final_summary}`")
    lines.append("")

    lines.append("## Suite Summary\n")
    sorted_rows = sorted(suite_counts.items(), key=lambda item: item[0])
    lines.append(format_count_table(sorted_rows))

    lines.append("## Failures And Timeouts By Suite\n")
    if not suite_failures:
        lines.append("No failures, errors, or timeouts have been observed so far.\n")
    else:
        for suite in sorted(suite_failures):
            failures = suite_failures[suite]
            signature_counts = collections.Counter(
                failure_signature(list(f.get("details") or [])) for f in failures
            )
            lines.append(f"### `{suite}`\n")
            counter = suite_counts[suite]
            lines.append(
                f"- Failed: `{counter['failed']}`, Error: `{counter['error']}`, Timed out: `{counter['timedout']}`"
            )
            lines.append("- Common signatures:")
            for signature, count in signature_counts.most_common(5):
                lines.append(f"  - `{count}x` {signature}")
            lines.append("- Affected tests:")
            for failure in failures[:25]:
                lines.append(f"  - `{failure.get('state')}` `{failure.get('name')}`")
            if len(failures) > 25:
                lines.append(f"  - ... {len(failures) - 25} more")
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(output_path)
    print(f"{completed}/{total} ({pct:.2f}%)")


if __name__ == "__main__":
    main()
