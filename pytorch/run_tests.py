#!/usr/bin/env python3
"""
Script to run PyTorch inductor unit tests from a CSV file.
"""

import ast
import copy
import csv
import io
import json
import multiprocessing
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

# Relative path to the test file (under PyTorch root); used in run_test, discover_tests, main
TEST_FILE_REL_PATH = 'test/inductor/test_torchinductor.py'
TRITON_NIGHTLY_INDUCTOR_FILES = [
    'inductor/test_torchinductor.py',
    'inductor/test_flex_attention.py',
    'inductor/test_max_autotune.py',
    'inductor/test_aot_inductor.py',
    'inductor/test_flex_decoding.py',
    'inductor/test_torchinductor_codegen_dynamic_shapes.py',
    'inductor/test_torchinductor_opinfo.py',
]

_TORCH_VERSION_CACHE = None


def _require_rocm_home():
    """
    Return ROCM_HOME from the user environment.
    Raise RuntimeError if unset/empty because inductor tests require it.
    """
    rocm_home = (subprocess.os.environ.get('ROCM_HOME') or '').strip()
    if not rocm_home:
        raise RuntimeError(
            "ROCM_HOME environment variable must be set to some value in your environment, "
            "or Inductor unit tests will fail."
        )
    return rocm_home


def _installed_torch_version():
    """
    Return installed torch.__version__ without importing torch from the caller cwd.
    Importing torch from a PyTorch checkout root can pick up the source tree instead
    of the installed extension, so query from /tmp in a subprocess.
    """
    global _TORCH_VERSION_CACHE
    if _TORCH_VERSION_CACHE is not None:
        return _TORCH_VERSION_CACHE

    result = subprocess.run(
        [sys.executable, '-c', 'import torch; print(torch.__version__)'],
        cwd='/tmp',
        capture_output=True,
        text=True,
        timeout=20,
    )
    if result.returncode != 0:
        _TORCH_VERSION_CACHE = ''
    else:
        lines = (result.stdout or '').strip().splitlines()
        _TORCH_VERSION_CACHE = lines[-1] if lines else ''
    return _TORCH_VERSION_CACHE


def _torch_version_is_before_2_13():
    version = _installed_torch_version()
    match = re.match(r'^(\d+)\.(\d+)', version)
    if not match:
        return False
    major, minor = (int(match.group(1)), int(match.group(2)))
    return (major, minor) < (2, 13)


def _build_test_env():
    """Build subprocess environment for inductor tests."""
    env = {
        **subprocess.os.environ.copy(),
        'PYTORCH_TEST_WITH_ROCM': '1',
        'HSA_FORCE_FINE_GRAIN_PCIE': '1',
        'HSA_TOOLS_DISABLE_REGISTER': '1',
        'PYTORCH_TESTING_DEVICE_ONLY_FOR': 'cuda',
    }
    env['ROCM_HOME'] = _require_rocm_home()
    if _torch_version_is_before_2_13():
        env['TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER'] = '0'
    return env


def ensure_pytest_and_timeout_installed():
    """
    Verify that pytest, pytest-timeout, pytest-rerunfailures, and expecttest are installed (same interpreter as this script).
    Abort with a clear message if any are missing. Call before using pytest for discovery or execution.
    """
    result = subprocess.run(
        [sys.executable, '-c', 'import pytest; import pytest_timeout; import pytest_rerunfailures; import expecttest'],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        err = (result.stderr or result.stdout or '').strip()
        print("Error: This script requires pytest, pytest-timeout, pytest-rerunfailures, and expecttest.")
        print("Install them with:  pip install pytest pytest-timeout pytest-rerunfailures expecttest")
        if err:
            print(f"Details: {err}")
        sys.exit(1)


class TeeOutput:
    """Helper class to write to both console and log file."""
    def __init__(self, log_file):
        self.log_file = log_file
        self.terminal = sys.stdout
        
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


# Test outcome states (each test has exactly one)
STATE_PASSED = "passed"
STATE_SKIPPED = "skipped"
STATE_XFAILED = "xfailed"
STATE_ERROR = "error"
STATE_FAILED = "failed"
STATE_TIMEDOUT = "timedout"
STATE_MISSED = "missed"
DISCOVERY_TIMEOUT_SECONDS = 1800
DEFAULT_PER_TEST_TIMEOUT_SECONDS = 1200
DEFAULT_PER_FILE_TIMEOUT_SECONDS = 43200
DEFAULT_SHARD_SIZE = 100
BATCH_MODE_FILE = "file"
BATCH_MODE_SHARD = "shard"
BATCH_MODE_TEST = "test"
DEFAULT_SHARDED_FILE_SUFFIXES = ("test/inductor/test_torchinductor_opinfo.py",)
UNKNOWN_TIMEOUT_MISS_LIMIT = 4


def _now_iso():
    """Return a local timestamp for logs and manifests."""
    return datetime.now().astimezone().isoformat(timespec='seconds')


def _is_success_state(state: str) -> bool:
    """Return True for states that should not fail the runner."""
    return state in (STATE_PASSED, STATE_SKIPPED, STATE_XFAILED)


def _signal_name_from_returncode(returncode):
    """Return a signal name for negative subprocess return codes."""
    if returncode is None or returncode >= 0:
        return None
    signal_number = -returncode
    try:
        return signal.Signals(signal_number).name
    except ValueError:
        return f"signal {signal_number}"


def _output_indicates_xfailed(stdout: str, stderr: str) -> bool:
    """True if pytest output indicates an expected failure."""
    combined = (stdout or "") + "\n" + (stderr or "")
    return " XFAIL" in combined or re.search(r"\d+\s+xfailed", combined)


def _output_indicates_xpassed(stdout: str, stderr: str) -> bool:
    """True if pytest output indicates an unexpected pass."""
    combined = (stdout or "") + "\n" + (stderr or "")
    return " XPASS" in combined or re.search(r"\d+\s+xpassed", combined)


def _output_indicates_skipped(stdout: str, stderr: str) -> bool:
    """
    Parse captured stdout/stderr to detect if the test was skipped (vs passed).
    Handles pytest (e.g. 'SKIPPED', '1 skipped') and unittest (e.g. 'OK (skipped=1)').
    """
    combined = (stdout or "") + "\n" + (stderr or "")
    # Pytest: "test_foo SKIPPED" or summary "1 passed, 1 skipped" / "1 skipped"
    if " SKIPPED" in combined or re.search(r"\d+\s+skipped", combined):
        return True
    # Unittest: "OK (skipped=1)" or "Ran 1 test ... OK (skipped=1)"
    if "skipped=" in combined and "OK" in combined:
        return True
    return False


def _output_indicates_runtime_error(stdout: str, stderr: str) -> bool:
    """True if stdout/stderr contain RuntimeError (non-zero exit → classify as ERROR)."""
    combined = (stdout or "") + "\n" + (stderr or "")
    return "RuntimeError" in combined


def _output_indicates_timeout(stdout: str, stderr: str) -> bool:
    """True if stdout/stderr indicate pytest-timeout fired (classify as TIMEDOUT)."""
    combined = (stdout or "") + "\n" + (stderr or "")
    return "TimeoutExpired" in combined or "Timeout" in combined


def run_test(test_name, pytorch_path, log_file, timeout=300, by_id=False, attempt=None, total_attempts=None):
    """
    Run a single test with the specified test name.

    Args:
        test_name: Name of the test to run. When by_id=True this is a full pytest node id
                   (e.g. path::Class::test_method or path::Class::test_method[param]); we run
                   pytest with that node id for 1:1 mapping. When by_id=False, pass as -k <test_name>.
        pytorch_path: Path to PyTorch directory
        log_file: File object to write logs to
        timeout: Timeout in seconds for this test (default 300)
        by_id: If True, test_name is a full pytest node id; run pytest <node_id> from pytorch_path.

    Returns:
        dict with success, elapsed_time, timed_out, state, returncode, and signal_name.
        state is one of STATE_PASSED, STATE_SKIPPED, STATE_XFAILED, STATE_ERROR,
        STATE_FAILED, STATE_TIMEDOUT.
    """
    test_file = Path(pytorch_path) / TEST_FILE_REL_PATH

    if by_id:
        # Full-suite/CSV node-id mode: test_name is a full pytest node id.
        # Per-test timeout is handled via pytest-timeout.
        cmd = ['pytest', '--timeout', str(timeout), test_name]
    else:
        # Legacy keyword mode: run pytest with -k and per-test timeout.
        cmd = ['pytest', TEST_FILE_REL_PATH, '-k', test_name, '--timeout', str(timeout)]

    env = _build_test_env()
    
    attempt_suffix = ""
    if attempt is not None and total_attempts is not None:
        attempt_suffix = f"\nAttempt: {attempt}/{total_attempts}"
    header = f"\n{'='*70}\nRunning: {test_name}{attempt_suffix}\n{'='*70}\n"
    print(header, end='')
    log_file.write(header)
    log_file.flush()
    
    start_time = time.time()
    # pytest-timeout should interrupt test-level hangs; the subprocess timeout is
    # a final safety net for cases where the test process does not exit cleanly.
    run_kw = dict(
        env=env,
        capture_output=True,
        text=True,
        cwd=str(pytorch_path),
        timeout=timeout + 60,
    )

    try:
        result = subprocess.run(cmd, **run_kw)
        
        elapsed_time = time.time() - start_time
        stdout_str = result.stdout or ""
        stderr_str = result.stderr or ""
        timed_out = False
        signal_name = _signal_name_from_returncode(result.returncode)

        if signal_name:
            state = STATE_FAILED
        elif _output_indicates_xpassed(stdout_str, stderr_str):
            state = STATE_FAILED
        elif result.returncode == 0:
            if _output_indicates_xfailed(stdout_str, stderr_str):
                state = STATE_XFAILED
            elif _output_indicates_skipped(stdout_str, stderr_str):
                state = STATE_SKIPPED
            else:
                state = STATE_PASSED
        else:
            if _output_indicates_timeout(stdout_str, stderr_str):
                state = STATE_TIMEDOUT
                timed_out = True
            else:
                state = STATE_ERROR if _output_indicates_runtime_error(stdout_str, stderr_str) else STATE_FAILED
        success = _is_success_state(state)

        # Write all output to log file
        if result.stdout:
            log_file.write(result.stdout)
        if result.stderr:
            log_file.write(result.stderr)
        log_file.flush()

        # Print status to console.
        status_map = {
            STATE_PASSED: "✓ PASSED",
            STATE_SKIPPED: "✓ SKIPPED",
            STATE_XFAILED: "✓ XFAILED",
            STATE_ERROR: "✗ ERROR",
            STATE_FAILED: "✗ FAILED",
            STATE_TIMEDOUT: "✗ TIMEDOUT",
            STATE_MISSED: "✗ MISSED",
        }
        status = status_map[state]
        status_msg = f"{status} ({elapsed_time:.2f}s)\n"
        if signal_name:
            status_msg = f"{status} ({elapsed_time:.2f}s, signal: {signal_name})\n"
        print(status_msg, end='')
        log_file.write(status_msg)
        log_file.flush()

        return {
            'success': success,
            'elapsed_time': elapsed_time,
            'timed_out': timed_out,
            'state': state,
            'returncode': result.returncode,
            'signal_name': signal_name,
        }

    except subprocess.TimeoutExpired as e:
        elapsed_time = time.time() - start_time
        timeout_msg = f"✗ TIMEOUT after {elapsed_time:.2f}s (limit: {timeout}s)\n"
        print(timeout_msg, end='')
        log_file.write(timeout_msg)
        
        # Write any captured output before timeout (may be bytes or str depending on Python/env)
        try:
            if e.stdout:
                out = e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout
                log_file.write(out)
            if e.stderr:
                err = e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr
                log_file.write(err)
        except (AttributeError, TypeError):
            pass
        log_file.flush()

        return {
            'success': False,
            'elapsed_time': elapsed_time,
            'timed_out': True,
            'state': STATE_TIMEDOUT,
            'returncode': None,
            'signal_name': None,
        }

    except Exception as e:
        elapsed_time = time.time() - start_time
        error_msg = f"✗ ERROR: {str(e)}\n"
        print(error_msg, end='')
        log_file.write(error_msg)
        log_file.flush()
        return {
            'success': False,
            'elapsed_time': elapsed_time,
            'timed_out': False,
            'state': STATE_ERROR,
            'returncode': None,
            'signal_name': None,
        }


def _node_file(node_id: str) -> str:
    """Return the test file portion of a pytest node id."""
    return node_id.split("::", 1)[0]


def _normalize_test_file_name(file_name):
    """Convert pytest paths to PyTorch run_test.py test names."""
    normalized = file_name.replace("\\", "/")
    if normalized.startswith("test/"):
        normalized = normalized[len("test/"):]
    if normalized.endswith(".py"):
        normalized = normalized[:-3]
    return normalized


def _group_node_ids_by_file(test_names):
    """Group pytest node ids by file while preserving discovery order."""
    groups = []
    current_file = None
    current_nodes = []
    for node_id in test_names:
        file_name = _node_file(node_id)
        if current_file is None:
            current_file = file_name
        if file_name != current_file:
            groups.append((current_file, current_nodes))
            current_file = file_name
            current_nodes = []
        current_nodes.append(node_id)
    if current_file is not None:
        groups.append((current_file, current_nodes))
    return groups


def _load_pytorch_serial_test_names(pytorch_path):
    """
    Load PyTorch's file-level serial lists from test/run_test.py.

    These lists mean the test file should not run concurrently with other test
    files. The tests inside that file may still run using that file's own batch
    mode.
    """
    pytorch_root = Path(pytorch_path)
    run_test_path = pytorch_root / "test" / "run_test.py"
    if not run_test_path.exists():
        return set()

    def collect_strings(node):
        if isinstance(node, ast.List):
            return [
                item.value
                for item in node.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            ]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return collect_strings(node.left) + collect_strings(node.right)
        return []

    tree = ast.parse(run_test_path.read_text(encoding='utf-8'))
    serial_names = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        target_names = [
            target.id for target in node.targets if isinstance(target, ast.Name)
        ]
        if any(
            target in {"CI_SERIAL_LIST", "RUN_PARALLEL_BLOCKLIST"}
            for target in target_names
        ):
            serial_names.update(collect_strings(node.value))
    return serial_names


def _split_serial_and_parallel_groups(groups, pytorch_path):
    """Split file groups using PyTorch's CI serial and run-parallel blocklists."""
    serial_names = _load_pytorch_serial_test_names(pytorch_path)
    serial_groups = []
    parallel_groups = []
    for file_name, node_ids in groups:
        normalized = _normalize_test_file_name(file_name)
        if normalized in serial_names:
            serial_groups.append((file_name, node_ids))
        else:
            parallel_groups.append((file_name, node_ids))
    return serial_groups, parallel_groups


def _chunk_node_ids(node_ids, shard_size):
    """Split pytest node ids into fixed-size shards."""
    return [node_ids[i:i + shard_size] for i in range(0, len(node_ids), shard_size)]


def _is_default_sharded_file(file_name):
    """Return True for files that are too large/flaky for one file-level subprocess."""
    normalized = file_name.replace("\\", "/")
    return any(normalized.endswith(suffix) for suffix in DEFAULT_SHARDED_FILE_SUFFIXES)


def _parse_junit_testcases(junit_path):
    """
    Return per-testcase states from pytest's JUnit XML in execution order.

    Each item is (node_hint, state, elapsed). State is one of passed, skipped,
    failed, error. node_hint is a best-effort pytest node id fragment from
    JUnit classname/name.
    """
    try:
        root = ET.parse(junit_path).getroot()
    except (ET.ParseError, OSError):
        return []

    cases = []
    for testcase in root.iter("testcase"):
        elapsed = float(testcase.attrib.get("time", "0") or 0)
        classname = testcase.attrib.get("classname", "")
        name = testcase.attrib.get("name", "")
        node_hint = _junit_node_hint(classname, name)
        skipped = testcase.find("skipped")
        if skipped is not None:
            state = _junit_skipped_state(skipped)
        elif testcase.find("error") is not None:
            state = STATE_ERROR
        elif testcase.find("failure") is not None:
            state = STATE_FAILED
        else:
            state = STATE_PASSED
        cases.append((node_hint, state, elapsed))
    return cases


def _junit_skipped_state(skipped):
    """Distinguish ordinary pytest skips from expected failures in JUnit XML."""
    skipped_type = (skipped.attrib.get("type") or "").lower()
    skipped_message = (skipped.attrib.get("message") or "").lower()
    skipped_text = (skipped.text or "").lower()
    if "xfail" in skipped_type or "xfail" in skipped_message or "xfail" in skipped_text:
        return STATE_XFAILED
    return STATE_SKIPPED


def _junit_node_hint(classname: str, name: str) -> str:
    """Convert pytest JUnit classname/name into a pytest-node-like hint."""
    if not classname:
        return name
    parts = classname.split(".")
    class_name = parts[-1]
    module_parts = parts[:-1]
    if module_parts and module_parts[0] == "test":
        file_part = "/".join(module_parts) + ".py"
        return f"{file_part}::{class_name}::{name}"
    return f"{classname}::{name}"


def _match_junit_cases_to_nodes(node_ids, testcase_states):
    """Match JUnit testcase records to discovered node ids by identity."""
    unmatched = list(testcase_states)
    matched = []
    for node_id in node_ids:
        found_index = None
        for i, (node_hint, state, elapsed) in enumerate(unmatched):
            if node_hint == node_id or node_id.endswith(node_hint) or node_hint.endswith(node_id):
                found_index = i
                break
            # JUnit names often omit the leading test/ path in classname.
            if node_id.split("/", 1)[-1].endswith(node_hint):
                found_index = i
                break
        if found_index is None:
            matched.append((node_id, None, None))
            continue
        _, state, elapsed = unmatched.pop(found_index)
        matched.append((node_id, state, elapsed))
    return matched, unmatched


def _running_nodes_from_output(output: str):
    """Best-effort extraction of pytest node ids from verbose output."""
    nodes = []
    for line in (output or "").splitlines():
        stripped = line.strip()
        if not stripped or "::" not in stripped:
            continue
        candidate = stripped.split()[0]
        if _NODE_ID_LINE_RE.match(candidate):
            nodes.append(candidate)
    return nodes


def _parse_verbose_node_results(output: str):
    """Parse completed pytest node states from -vv output."""
    results = []
    state_map = {
        "PASSED": STATE_PASSED,
        "SKIPPED": STATE_SKIPPED,
        "FAILED": STATE_FAILED,
        "ERROR": STATE_ERROR,
        "XFAIL": STATE_XFAILED,
        "XPASS": STATE_FAILED,
    }
    pattern = re.compile(
        r"^(?P<node>[\w/.-]+::\S+)\s+"
        r"(?P<status>PASSED|SKIPPED|FAILED|ERROR|XFAIL|XPASS)"
        r"(?:\s+\[(?P<time>[\d.]+)s\])?"
    )
    for line in (output or "").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        node = match.group("node")
        elapsed = float(match.group("time") or 0)
        state = state_map[match.group("status")]
        results.append({
            'name': node,
            'success': _is_success_state(state),
            'time': elapsed,
            'timed_out': False,
            'state': state,
        })
    return results


def _write_result_status(log_file, state, elapsed):
    status_map = {
        STATE_PASSED: "✓ PASSED",
        STATE_SKIPPED: "✓ SKIPPED",
        STATE_XFAILED: "✓ XFAILED",
        STATE_ERROR: "✗ ERROR",
        STATE_FAILED: "✗ FAILED",
        STATE_TIMEDOUT: "✗ TIMEDOUT",
        STATE_MISSED: "✗ MISSED",
    }
    status_msg = f"{status_map[state]} ({elapsed:.2f}s)\n"
    print(status_msg, end='')
    log_file.write(status_msg)
    log_file.flush()


def _record_file_batch_result(node_id, state, elapsed, log_file, index, total):
    progress_msg = f"[{index}/{total}] "
    header = f"\n{'='*70}\nRunning: {node_id}\n{'='*70}\n"
    print(progress_msg, end='')
    print(header, end='')
    log_file.write(progress_msg)
    log_file.write(header)
    _write_result_status(log_file, state, elapsed)
    return {
        'name': node_id,
        'success': _is_success_state(state),
        'time': elapsed,
        'timed_out': state == STATE_TIMEDOUT,
        'state': state,
        'attempts': 1,
        'flaky': False,
        'consistent_failure': state in (STATE_FAILED, STATE_ERROR),
        'signal_name': None,
    }


def _read_log_tail_from(log_file, start_offset, max_bytes=1_000_000):
    """
    Read at most max_bytes from log_file after start_offset.

    File-batch pytest output is streamed directly to the log to avoid keeping
    large verbose pytest output in memory. Timeout recovery only needs recent
    output to identify the running node, so read a bounded tail from disk.
    """
    try:
        log_file.flush()
        path = getattr(log_file, "name", None)
        if not path:
            return ""
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            end_offset = f.tell()
            read_start = max(start_offset, end_offset - max_bytes)
            f.seek(read_start)
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _new_stepcurrent_key():
    """Return a unique key for PyTorch's pytest step-current plugin."""
    return f"framework_scripts_{os.getpid()}_{time.time_ns()}"


def _read_stepcurrent_node(pytorch_path, key):
    """Return the node most recently started by pytest, if it was recorded."""
    path = (
        Path(pytorch_path)
        / ".pytest_cache"
        / "v"
        / "cache"
        / "stepcurrent"
        / key
        / "lastrun"
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, str) else None


def _run_file_batch(file_name, node_ids, args, log_file):
    """
    Run one file (or a filtered set of nodes from one file) in one pytest process.
    Returns (results, reason, elapsed, problem_node).
    """
    env = _build_test_env()
    targets = node_ids
    with tempfile.TemporaryDirectory(prefix="run_tests_junit_") as tmpdir:
        junit_path = Path(tmpdir) / "pytest.xml"
        stepcurrent_key = _new_stepcurrent_key()
        cmd = [
            'pytest',
            '-vv',
            '-x',
            '--timeout',
            str(args.per_test_timeout),
        ] + targets + [
            '--junitxml',
            str(junit_path),
            f'--sc={stepcurrent_key}',
        ]
        if args.retry_attempts > 0:
            cmd.extend(['--reruns', str(args.retry_attempts)])
        header = (
            f"\n{'='*70}\n"
            f"Running file batch: {file_name} ({len(node_ids)} collected test(s))\n"
            f"{'='*70}\n"
        )
        print(header, end='')
        log_file.write(header)
        log_file.flush()
        output_start_offset = log_file.tell()

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(args.pytorch_path),
                timeout=args.per_file_timeout,
            )
            elapsed = time.time() - start_time
        except subprocess.TimeoutExpired as e:
            elapsed = time.time() - start_time
            msg = f"✗ FILE TIMEDOUT ({file_name}) after {elapsed:.2f}s (limit: {args.per_file_timeout}s)\n"
            print(msg, end='')
            log_file.write(msg)
            log_file.flush()
            output = _read_log_tail_from(log_file, output_start_offset)
            running_nodes = _running_nodes_from_output(output)
            timed_out_node = (
                _read_stepcurrent_node(args.pytorch_path, stepcurrent_key)
                or (running_nodes[-1] if running_nodes else None)
            )
            partial_results, _, _, _ = _build_file_results(
                node_ids, _parse_junit_testcases(junit_path),
                file_name, log_file, elapsed, allow_partial=True
            )
            if not partial_results:
                partial_results = _parse_verbose_node_results(output)
            return partial_results, "timeout", elapsed, timed_out_node

        log_file.flush()

        testcase_states = _parse_junit_testcases(junit_path)
        if result.returncode != 0:
            output = _read_log_tail_from(log_file, output_start_offset)
            if testcase_states:
                msg = f"✗ FILE FAILED ({file_name}) with exit code {result.returncode} ({elapsed:.2f}s)\n"
                print(msg, end='')
                log_file.write(msg)
                log_file.flush()
                partial_results, _, _, _ = _build_file_results(
                    node_ids,
                    testcase_states,
                    file_name,
                    log_file,
                    elapsed,
                    allow_partial=True,
                )
                failed_node = next(
                    (
                        item['name']
                        for item in reversed(partial_results)
                        if not item['success']
                    ),
                    None,
                )
                failed_node = failed_node or _read_stepcurrent_node(
                    args.pytorch_path, stepcurrent_key
                )
                return partial_results, "failure", elapsed, failed_node
            msg = f"✗ FILE FAILED ({file_name}) with exit code {result.returncode} ({elapsed:.2f}s); no JUnit results available.\n"
            print(msg, end='')
            log_file.write(msg)
            log_file.flush()
            partial_results = _parse_verbose_node_results(output)
            failed_node = _read_stepcurrent_node(
                args.pytorch_path, stepcurrent_key
            )
            if failed_node is None:
                running_nodes = _running_nodes_from_output(output)
                failed_node = running_nodes[-1] if running_nodes else None
            return partial_results, "crash", elapsed, failed_node

        return _build_file_results(node_ids, testcase_states, file_name, log_file, elapsed)


def _build_file_results(node_ids, testcase_states, file_name, log_file, elapsed, allow_partial=False):
    """Build result dictionaries from JUnit testcase states."""
    matched, extra_cases = _match_junit_cases_to_nodes(node_ids, testcase_states)
    missing_count = sum(1 for _, state, _ in matched if state is None)
    if (missing_count or extra_cases) and not allow_partial:
        msg = f"JUnit result mismatch ({file_name}): "
        msg += f"{missing_count} discovered node(s) missing, {len(extra_cases)} extra testcase(s).\n"
        print(msg, end='')
        log_file.write(msg)
        log_file.flush()

    results = []
    for node_id, state, test_elapsed in matched:
        if state is None:
            if allow_partial:
                continue
            state = STATE_MISSED
            test_elapsed = 0.0
        results.append({
            'name': node_id,
            'success': _is_success_state(state),
            'time': test_elapsed,
            'timed_out': False,
            'state': state,
        })
    return results, None, elapsed, None


# Pytest node ID line (from --collect-only -q): path::Class::test_method or path::test_method[param]
# Path can be relative (e.g. test/inductor/test_torchinductor.py). Param part may contain - and other chars.
_NODE_ID_LINE_RE = re.compile(r'^[\w/.-]+::.+$')


# Regex to parse "collected N items" from pytest --collect-only output
_COLLECTED_RE = re.compile(r'collected\s+(\d+)\s+item', re.IGNORECASE)
# Regex for <UnitTestCase ...> where the group name is the second token (e.g. <UnitTestCase TestClass> or <UnitTestCase 'TestClass'>)
_UNIT_TEST_CASE_RE = re.compile(r'<UnitTestCase\s+[\'"]?([^\'">\s]+)[\'"]?>')
# Lines that represent a test item (leaf node) under a UnitTestCase
_TEST_ITEM_RE = re.compile(r'<\s*(?:TestCase)?Function\s+')


def _parse_collect_only_hierarchy(combined_output: str):
    """
    Parse pytest --collect-only (non-quiet) output for hierarchical breakdown by UnitTestCase.
    Returns a dict mapping group name (test class name) to count of test items under it.
    """
    breakdown = {}
    current_group = None
    for line in (combined_output or '').splitlines():
        unit_match = _UNIT_TEST_CASE_RE.search(line)
        if unit_match:
            current_group = unit_match.group(1)
            if current_group not in breakdown:
                breakdown[current_group] = 0
            continue
        if current_group is not None and _TEST_ITEM_RE.search(line):
            breakdown[current_group] += 1
    return breakdown


def _run_collect_only_and_parse(pytorch_path, cmd):
    """
    Run pytest with the given command (must include --collect-only, without -q to get hierarchy),
    capture output, and parse for total collected count and hierarchical breakdown by UnitTestCase.
    Uses the same env as discover_tests/run_test for consistency.
    Returns (total: int, breakdown: dict[str, int]). On failure or parse error, returns (0, {}).
    """
    env = _build_test_env()
    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=str(pytorch_path),
            capture_output=True,
            text=True,
            timeout=120,
        )
        combined = (result.stdout or '') + '\n' + (result.stderr or '')
        total = 0
        m = _COLLECTED_RE.search(combined)
        if m:
            total = int(m.group(1))
        breakdown = _parse_collect_only_hierarchy(combined)
        return total, breakdown
    except (subprocess.TimeoutExpired, ValueError, OSError):
        return 0, {}


def _do_collect_only_and_exit(pytorch_path, test_names, by_id):
    """
    Build the appropriate pytest --collect-only command, run it, print total and hierarchical
    breakdown by test class (UnitTestCase), then exit 0. No log file is used.
    by_id: if True, test_names are pytest node ids; else they are -k keywords and we use TEST_FILE_REL_PATH.
    """
    if by_id:
        cmd = ['pytest', '--collect-only'] + test_names
    else:
        cmd = ['pytest', TEST_FILE_REL_PATH, '-k', ' or '.join(test_names), '--collect-only']
    total, breakdown = _run_collect_only_and_parse(pytorch_path, cmd)
    print(f"Total tests: {total}")
    if breakdown:
        for group_name in sorted(breakdown.keys()):
            print(f"  {group_name}: {breakdown[group_name]}")
    sys.exit(0)


def _parse_pytest_collect_only_quiet(combined_output: str):
    """
    Parse pytest --collect-only -q output: one full node id per line.
    Returns a list of node ids (e.g. path::Class::test_method or path::Class::test_method[param])
    for 1:1 mapping: each collected item is run exactly once.
    Skips non-node lines (e.g. "collected N items", "Running N items in this shard").
    """
    lines = (combined_output or '').strip().splitlines()
    node_ids = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.lower().startswith('collected ') or s.lower().startswith('running '):
            continue
        if _NODE_ID_LINE_RE.match(s):
            node_ids.append(s)
    return node_ids


def _format_discovery_failure(cmd, returncode, stdout, stderr):
    """Format all useful discovery failure output for early diagnosis."""
    msg = f"Discovery failed (exit {returncode}).\n"
    msg += f"Command: {' '.join(cmd)}\n"
    msg += "\n--- discovery stdout ---\n"
    msg += stdout or "(none)\n"
    if not msg.endswith("\n"):
        msg += "\n"
    msg += "\n--- discovery stderr ---\n"
    msg += stderr or "(none)\n"
    if not msg.endswith("\n"):
        msg += "\n"
    msg += "--- end discovery output ---\n"
    return msg


def discover_tests(pytorch_path, log_file, test_file_rel_paths=None):
    """
    Discover tests by running pytest --collect-only -q (one node id per line).
    Returns full pytest node ids for 1:1 mapping: each collected item is run
    exactly once (including each parametrized variant).

    Args:
        pytorch_path: Path to PyTorch directory.
        log_file: File object to write logs to.
        test_file_rel_paths: Optional list of relative paths (under pytorch_path) for test files.
                             If None, uses [TEST_FILE_REL_PATH].

    Returns:
        list: List of full pytest node ids (e.g. path::Class::test_method or path::Class::test_method[param])
    """
    if test_file_rel_paths is None:
        test_file_rel_paths = [TEST_FILE_REL_PATH]
    test_paths = [str(Path(pytorch_path) / p) for p in test_file_rel_paths]
    cmd = ['pytest'] + test_paths + ['--collect-only', '-q']
    env = _build_test_env()
    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=str(pytorch_path),
            capture_output=True,
            text=True,
            timeout=DISCOVERY_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            msg = _format_discovery_failure(
                cmd, result.returncode, result.stdout or "", result.stderr or ""
            )
            print(msg, end='')
            log_file.write(msg)
            log_file.flush()
            return []
        combined = (result.stdout or '') + '\n' + (result.stderr or '')
        node_ids = _parse_pytest_collect_only_quiet(combined)
        return node_ids
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        msg = _format_discovery_failure(cmd, "timeout", stdout, stderr)
        print(msg, end='')
        log_file.write(msg)
        log_file.flush()
        return []
    except Exception as e:
        msg = f"Discovery error: {e}\nCommand: {' '.join(cmd)}\n"
        print(msg, end='')
        log_file.write(msg)
        log_file.flush()
        return []


def checkpoint_path(log_file_path):
    """Return path to checkpoint file for the given log file (on by default)."""
    return str(Path(log_file_path).with_suffix(Path(log_file_path).suffix + '.checkpoint'))


def write_checkpoint(log_file_path, last_test, next_test, last_index, total, mode, csv_file=None, pytorch_path=None):
    """Write checkpoint after each test so runs can be resumed. On by default."""
    path = checkpoint_path(log_file_path)
    data = {
        'last_test': last_test,
        'next_test': next_test,
        'last_index': last_index,
        'total': total,
        'mode': mode,
        'csv_file': csv_file,
        'pytorch_path': pytorch_path,
        'updated': datetime.now().isoformat(),
    }
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass


def read_checkpoint(log_file_path):
    """Read checkpoint if it exists. Returns dict or None."""
    path = checkpoint_path(log_file_path)
    try:
        if Path(path).exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except (OSError, json.JSONDecodeError):
        pass
    return None


def read_tests_from_csv(csv_file):
    """
    Read pytest node IDs from CSV file.
    Rows whose test_name starts with '#' are treated as comments and skipped.
    
    Args:
        csv_file: Path to CSV file
        
    Returns:
        list: List of pytest node IDs
    """
    test_names = []
    invalid_rows = []
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            if 'test_name' not in reader.fieldnames:
                print(f"Error: CSV file must have a 'test_name' column")
                print(f"Found columns: {reader.fieldnames}")
                sys.exit(1)
            
            for row_number, row in enumerate(reader, start=2):
                test_name = row['test_name'].strip()
                if test_name and not test_name.startswith('#'):  # Skip empty rows and comments
                    if not _is_pytest_node_id(test_name):
                        invalid_rows.append((row_number, test_name))
                    else:
                        test_names.append(test_name)
            
            if invalid_rows:
                print("Error: CSV test_name values must be full pytest node IDs.")
                print("Expected format: path/to/test_file.py::TestClass::test_method")
                print("Example: test/inductor/test_flex_attention.py::TestFlexAttentionCUDA::test_block_mask_non_divisible_cuda")
                print("Invalid rows:")
                for row_number, test_name in invalid_rows[:10]:
                    print(f"  row {row_number}: {test_name}")
                if len(invalid_rows) > 10:
                    print(f"  ... and {len(invalid_rows) - 10} more")
                sys.exit(1)
        
        return test_names
        
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found")
        sys.exit(1)
        
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        sys.exit(1)


def _is_pytest_node_id(test_name):
    """
    Return True if test_name looks like a full pytest node ID.

    This intentionally rejects the legacy CSV keyword format, because CSV mode
    now passes test_name directly to pytest for exact test selection.
    """
    return '.py::' in test_name


def discover_inductor_all_files(pytorch_path):
    """
    Return every test/ relative file path under PyTorch's test/inductor suite.
    """
    pytorch_root = Path(pytorch_path)
    sys.path.insert(0, str(pytorch_root))
    try:
        from tools.testing.discover_tests import TESTS
    finally:
        try:
            sys.path.remove(str(pytorch_root))
        except ValueError:
            pass

    files = [
        f'{test_name}.py'
        for test_name in sorted(TESTS)
        if test_name.startswith('inductor/')
    ]

    if not files:
        raise RuntimeError("No inductor test files were discovered.")
    return files


def discover_inductor_core_files(pytorch_path):
    """Backward-compatible alias for the all-Inductor suite selector."""
    return discover_inductor_all_files(pytorch_path)


def triton_nightly_inductor_files():
    """Return the torch-triton-nightly Inductor validation file set."""
    return TRITON_NIGHTLY_INDUCTOR_FILES[:]


# Pattern for "Mode: full_suite" or "Mode: csv" at start of line (after strip)
MODE_LINE_RE = re.compile(r'^Mode:\s*(full_suite|csv)\s*$')
# Pattern for "  - test_name (12.34s)" in Failed tests section
FAILED_TEST_LINE_RE = re.compile(r'^\s+-\s+(.+?)\s+\(\d+\.\d+s\)\s*$')


def parse_log_for_rerun(log_path):
    """
    Parse a log file from a previous run to extract failed test names, timed-out test names, and mode.

    Looks for a line "Mode: full_suite" or "Mode: csv" to determine how to run tests.
    Looks for "Failed tests:" and "Timed out tests:" sections; parses lines "  - test_name (X.XXs)".
    Logs from older runs may only have "Failed tests:" (all non-passed grouped there); in that case
    timeout_tests will be empty.

    Returns:
        tuple: (failed_test_names: list[str], timeout_test_names: list[str], mode: str or None).
        mode is 'full_suite' or 'csv', or None if not found. On read error, returns (None, None, None).
    """
    try:
        content = Path(log_path).read_text(encoding='utf-8')
    except OSError:
        return None, None, None

    mode = None
    for line in content.splitlines():
        m = MODE_LINE_RE.match(line.strip())
        if m:
            mode = m.group(1)
            break

    failed_tests = []
    timeout_tests = []
    in_failed_section = False
    in_timeout_section = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped == 'Failed tests:':
            in_failed_section = True
            in_timeout_section = False
            continue
        if stripped == 'Timed out tests:':
            in_failed_section = False
            in_timeout_section = True
            continue
        if stripped.startswith('====='):
            in_failed_section = False
            in_timeout_section = False
            continue
        match = FAILED_TEST_LINE_RE.match(line)
        if match:
            name = match.group(1).strip()
            if in_failed_section:
                failed_tests.append(name)
            elif in_timeout_section:
                timeout_tests.append(name)
    return failed_tests, timeout_tests, mode


def _resolve_start_index(test_names, log_file_path, resume, no_checkpoint, log_file, not_in_list_msg):
    """
    Handle resume/checkpoint logic and return the index at which to start running tests.
    Writes messages to console and log_file.
    """
    start_index = 0
    if resume:
        cp = read_checkpoint(log_file_path)
        if not cp:
            msg = "No checkpoint found, starting from first test.\n\n"
            print(msg, end='')
            log_file.write(msg)
            log_file.flush()
        elif cp.get('next_test') is None:
            msg = "Checkpoint shows previous run completed (no next test). Starting from first test.\n\n"
            print(msg, end='')
            log_file.write(msg)
            log_file.flush()
        else:
            next_t = cp['next_test']
            if next_t in test_names:
                start_index = test_names.index(next_t)
                msg = f"Resuming from test: {next_t} [{start_index + 1}/{len(test_names)}]\n\n"
                print(msg, end='')
                log_file.write(msg)
                log_file.flush()
            else:
                msg = f"Checkpoint next_test {not_in_list_msg}, starting from first test.\n\n"
                print(msg, end='')
                log_file.write(msg)
                log_file.flush()
    elif not no_checkpoint and read_checkpoint(log_file_path):
        cp = read_checkpoint(log_file_path)
        msg = f"Checkpoint from previous run: last test = {cp.get('last_test', '?')}, next test = {cp.get('next_test', '?')}. Use --resume to continue from next test.\n\n"
        print(msg, end='')
        log_file.write(msg)
        log_file.flush()
        try:
            Path(checkpoint_path(log_file_path)).unlink(missing_ok=True)
        except OSError:
            pass
    if not no_checkpoint and start_index == 0:
        try:
            Path(checkpoint_path(log_file_path)).unlink(missing_ok=True)
        except OSError:
            pass
    return start_index


def _write_run_summary(results, start_time, log_file, summary_title):
    total_time = time.time() - start_time
    passed = sum(1 for r in results if r.get('state') == STATE_PASSED)
    skipped_count = sum(1 for r in results if r.get('state') == STATE_SKIPPED)
    xfailed_count = sum(1 for r in results if r.get('state') == STATE_XFAILED)
    error_count = sum(1 for r in results if r.get('state') == STATE_ERROR)
    failed_count = sum(1 for r in results if r.get('state') == STATE_FAILED)
    timedout_count = sum(1 for r in results if r.get('state') == STATE_TIMEDOUT)
    missed_count = sum(1 for r in results if r.get('state') == STATE_MISSED)
    flaky_count = sum(1 for r in results if r.get('flaky'))
    consistent_failure_count = sum(1 for r in results if r.get('consistent_failure'))
    summary = f"\n{'='*70}\n{summary_title}\n{'='*70}\n"
    summary += f"Total tests run: {len(results)}\n"
    summary += f"Passed: {passed}\n"
    summary += f"Skipped: {skipped_count}\n"
    summary += f"Xfailed: {xfailed_count}\n"
    summary += f"Error: {error_count}\n"
    summary += f"Failed: {failed_count}\n"
    summary += f"Timed out: {timedout_count}\n"
    summary += f"Missed: {missed_count}\n"
    summary += f"Recovered flaky: {flaky_count}\n"
    summary += f"Consistent failures: {consistent_failure_count}\n"
    summary += f"Total time: {total_time:.2f}s\n"
    flaky_tests = [r for r in results if r.get('flaky')]
    if flaky_tests:
        summary += "\nRecovered flaky tests:\n"
        for r in flaky_tests:
            summary += f"  - {r['name']} ({r['time']:.2f}s, attempts={r.get('attempts', 1)})\n"
    consistent_failures = [r for r in results if r.get('consistent_failure')]
    if consistent_failures:
        summary += "\nConsistent failure tests:\n"
        for r in consistent_failures:
            signal_name = r.get('signal_name')
            signal_part = f", signal={signal_name}" if signal_name else ""
            summary += f"  - {r['name']} ({r['time']:.2f}s, attempts={r.get('attempts', 1)}{signal_part})\n"
    for state, label in [
        (STATE_PASSED, "Passed"),
        (STATE_SKIPPED, "Skipped"),
        (STATE_XFAILED, "Xfailed"),
        (STATE_ERROR, "Error"),
        (STATE_FAILED, "Failed"),
        (STATE_TIMEDOUT, "Timed out"),
        (STATE_MISSED, "Missed"),
    ]:
        subset = [r for r in results if r.get('state') == state]
        if subset:
            summary += f"\n{label} tests:\n"
            for r in subset:
                signal_name = r.get('signal_name')
                signal_part = f", signal={signal_name}" if signal_name else ""
                summary += f"  - {r['name']} ({r['time']:.2f}s{signal_part})\n"
    summary += f"{'='*70}\n\n"
    print(summary, end='')
    log_file.write(summary)
    log_file.flush()
    any_bad = error_count > 0 or failed_count > 0 or timedout_count > 0 or missed_count > 0
    return 0 if not any_bad else 1


def _run_one_test_with_progress(test_name, index, total, args, log_file, by_id, timeout):
    progress_msg = f"[{index}/{total}] "
    print(progress_msg, end='')
    log_file.write(progress_msg)
    log_file.flush()

    total_attempts = args.retry_attempts + 1
    attempts = []
    for attempt in range(1, total_attempts + 1):
        try:
            attempt_result = run_test(
                test_name, args.pytorch_path, log_file,
                timeout=timeout,
                by_id=by_id,
                attempt=attempt,
                total_attempts=total_attempts,
            )
        except subprocess.TimeoutExpired:
            attempt_result = {
                'success': False,
                'elapsed_time': float(timeout),
                'timed_out': True,
                'state': STATE_TIMEDOUT,
                'returncode': None,
                'signal_name': None,
            }
            timeout_msg = f"✗ TIMEOUT (uncaught in run_test) after {timeout:.2f}s\n"
            print(timeout_msg, end='')
            log_file.write(timeout_msg)
            log_file.flush()
        except Exception as e:
            attempt_result = {
                'success': False,
                'elapsed_time': 0.0,
                'timed_out': False,
                'state': STATE_ERROR,
                'returncode': None,
                'signal_name': None,
            }
            err_msg = f"✗ ERROR (uncaught): {type(e).__name__}: {e}\n"
            print(err_msg, end='')
            log_file.write(err_msg)
            log_file.flush()

        attempts.append(attempt_result)
        if attempt_result['success']:
            break
        if attempt < total_attempts:
            retry_msg = (
                f"Retrying {test_name} after {attempt_result['state']} "
                f"(attempt {attempt + 1}/{total_attempts})\n"
            )
            print(retry_msg, end='')
            log_file.write(retry_msg)
            log_file.flush()

    final_result = attempts[-1]
    success = final_result['success']
    state = final_result['state']
    elapsed = sum(result.get('elapsed_time', 0.0) for result in attempts)
    timed_out = any(result.get('timed_out') for result in attempts)
    flaky = len(attempts) > 1 and success
    consistent_failure = not success and state in (STATE_FAILED, STATE_ERROR)
    if flaky:
        msg = f"Recovered flaky test after {len(attempts)} attempts: {test_name}\n"
        print(msg, end='')
        log_file.write(msg)
        log_file.flush()
    elif consistent_failure and len(attempts) > 1:
        msg = f"Consistent failure after {len(attempts)} attempts: {test_name}\n"
        print(msg, end='')
        log_file.write(msg)
        log_file.flush()

    return {
        'name': test_name, 'success': success, 'time': elapsed,
        'timed_out': timed_out,
        'state': state,
        'attempts': len(attempts),
        'flaky': flaky,
        'consistent_failure': consistent_failure,
        'signal_name': final_result.get('signal_name'),
        'returncode': final_result.get('returncode'),
    }


def _run_test_batch(test_names, start_index, args, log_file, mode, by_id, count_msg, summary_title):
    """
    Run tests from start_index to end, write checkpoints, print summary.
    Returns exit code (0 if all passed, 1 otherwise).
    """
    msg = f"{count_msg}\n\n"
    print(msg, end='')
    log_file.write(msg)
    log_file.flush()

    results = []
    start_time = time.time()
    timeout = args.per_test_timeout or DEFAULT_PER_TEST_TIMEOUT_SECONDS
    for i in range(start_index, len(test_names)):
        test_name = test_names[i]
        result = _run_one_test_with_progress(
            test_name, i + 1, len(test_names), args, log_file, by_id, timeout
        )
        results.append(result)
        if not args.no_checkpoint:
            next_test = test_names[i + 1] if i + 1 < len(test_names) else None
            write_checkpoint(
                args.log_file, test_name, next_test, i, len(test_names), mode,
                csv_file=args.csv_file, pytorch_path=args.pytorch_path
            )
        if not result['success'] and args.stop_on_failure:
            stop_msg = f"\nStopping due to test failure: {test_name}\n"
            print(stop_msg, end='')
            log_file.write(stop_msg)
            log_file.flush()
            break

    return _write_run_summary(results, start_time, log_file, summary_title)


def _run_file_node_group(file_name, node_ids, args, log_file, mode, test_names, node_index, results):
    """
    Run one file-scoped group of node ids with the existing file-batch recovery.

    The group may be a whole file or one shard from that file.
    Returns (node_index, stop_requested).
    """
    remaining_nodes = node_ids[:]
    unknown_timeout_misses = 0
    file_done = False
    while remaining_nodes and not file_done:
        file_results, reason, elapsed, problem_node = _run_file_batch(
            file_name, remaining_nodes, args, log_file
        )

        if (
            reason in ("failure", "crash")
            and problem_node in remaining_nodes
        ):
            problem_position = remaining_nodes.index(problem_node)
            result_by_name = {item['name']: item for item in file_results}

            # Record every node completed before the process-ending failure.
            for completed_node in remaining_nodes[:problem_position]:
                item = result_by_name.get(completed_node)
                state = item['state'] if item else STATE_MISSED
                test_elapsed = item['time'] if item else 0.0
                node_index += 1
                recorded = _record_file_batch_result(
                    completed_node,
                    state,
                    test_elapsed,
                    log_file,
                    node_index,
                    len(test_names),
                )
                results.append(recorded)

            msg = (
                f"Restarting failed test in a fresh pytest process: "
                f"{problem_node}\n"
            )
            print(msg, end='')
            log_file.write(msg)
            log_file.flush()

            retry_results, _retry_reason, retry_elapsed, _ = _run_file_batch(
                file_name, [problem_node], args, log_file
            )
            retry_item = next(
                (
                    item
                    for item in retry_results
                    if item['name'] == problem_node
                ),
                None,
            )
            original_item = result_by_name.get(problem_node)
            if retry_item is not None:
                final_state = retry_item['state']
                final_elapsed = retry_item['time']
            elif original_item is not None:
                final_state = original_item['state']
                final_elapsed = original_item['time']
            else:
                final_state = STATE_ERROR
                final_elapsed = retry_elapsed

            node_index += 1
            recorded = _record_file_batch_result(
                problem_node,
                final_state,
                final_elapsed,
                log_file,
                node_index,
                len(test_names),
            )
            if (
                retry_item is not None
                and retry_item['success']
                and (original_item is None or not original_item['success'])
            ):
                recorded['attempts'] = 2
                recorded['flaky'] = True
                recorded['consistent_failure'] = False
                msg = (
                    f"Recovered in a fresh pytest process: {problem_node}\n"
                )
                print(msg, end='')
                log_file.write(msg)
                log_file.flush()
            results.append(recorded)

            remaining_nodes = remaining_nodes[problem_position + 1:]
            unknown_timeout_misses = 0
            if args.stop_on_failure and not recorded['success']:
                file_done = True
            continue

        recorded_names = set()
        for result in file_results:
            node_index += 1
            recorded = _record_file_batch_result(
                result['name'], result['state'], result['time'],
                log_file, node_index, len(test_names)
            )
            results.append(recorded)
            recorded_names.add(result['name'])

        if reason is None:
            file_done = True
            continue
        if reason == "timeout" and problem_node and problem_node in remaining_nodes:
            timeout_position = remaining_nodes.index(problem_node)
            for node_id in remaining_nodes[:timeout_position]:
                if node_id in recorded_names:
                    continue
                node_index += 1
                recorded = _record_file_batch_result(
                    node_id, STATE_MISSED, 0.0, log_file, node_index, len(test_names)
                )
                results.append(recorded)

            if problem_node not in recorded_names:
                node_index += 1
                recorded = _record_file_batch_result(
                    problem_node, STATE_TIMEDOUT, elapsed, log_file, node_index, len(test_names)
                )
                results.append(recorded)
            else:
                for result in reversed(results):
                    if result['name'] == problem_node:
                        result['state'] = STATE_TIMEDOUT
                        result['success'] = False
                        result['timed_out'] = True
                        break
            msg = f"Skipping timed-out test and continuing: {problem_node}\n"
            print(msg, end='')
            log_file.write(msg)
            log_file.flush()
            remaining_nodes = remaining_nodes[timeout_position + 1:]
            unknown_timeout_misses = 0
            continue

        if reason == "timeout" and remaining_nodes:
            missed_node = remaining_nodes[0]
            unknown_timeout_misses += 1
            node_index += 1
            recorded = _record_file_batch_result(
                missed_node, STATE_MISSED, elapsed, log_file, node_index, len(test_names)
            )
            results.append(recorded)
            msg = (
                f"Could not identify timed-out test in {file_name}; "
                f"marking next unresolved test as missed ({unknown_timeout_misses}/"
                f"{UNKNOWN_TIMEOUT_MISS_LIMIT}): {missed_node}\n"
            )
            print(msg, end='')
            log_file.write(msg)
            log_file.flush()
            remaining_nodes = remaining_nodes[1:]
            if unknown_timeout_misses >= UNKNOWN_TIMEOUT_MISS_LIMIT and remaining_nodes:
                msg = (
                    f"Reached {UNKNOWN_TIMEOUT_MISS_LIMIT} consecutive unidentified timeouts "
                    f"in {file_name}; recording remaining tests as missed.\n"
                )
                print(msg, end='')
                log_file.write(msg)
                log_file.flush()
                for node_id in remaining_nodes:
                    node_index += 1
                    recorded = _record_file_batch_result(
                        node_id, STATE_MISSED, 0.0, log_file, node_index, len(test_names)
                    )
                    results.append(recorded)
                remaining_nodes = []
                file_done = True
            continue

        msg = (
            f"Unable to continue {file_name} after file batch issue ({reason}). "
            "Recording remaining tests as missed.\n"
        )
        print(msg, end='')
        log_file.write(msg)
        log_file.flush()
        for node_id in remaining_nodes:
            if node_id in recorded_names:
                continue
            node_index += 1
            recorded = _record_file_batch_result(
                node_id, STATE_MISSED, 0.0, log_file, node_index, len(test_names)
            )
            results.append(recorded)
        file_done = True

    if not remaining_nodes and not file_done:
        file_done = True

    checkpoint_index = node_index - 1
    if not args.no_checkpoint and checkpoint_index >= 0:
        last_test = test_names[checkpoint_index]
        next_test = test_names[node_index] if node_index < len(test_names) else None
        write_checkpoint(
            args.log_file, last_test, next_test, checkpoint_index, len(test_names), mode,
            csv_file=args.csv_file, pytorch_path=args.pytorch_path
        )
    stop_requested = args.stop_on_failure and any(not r['success'] for r in results[-len(node_ids):])
    return node_index, stop_requested


def _run_grouped_file_batch_mode(
    test_names, start_index, args, log_file, mode, count_msg, summary_title,
    shard_all=False, shard_default_files=False,
):
    """
    Run discovered pytest node ids grouped by file.

    When sharding is enabled, each file group is split into fixed-size chunks.
    Each chunk still uses _run_file_batch so JUnit parsing, timeout recovery,
    missed handling, and checkpointing stay consistent with file mode.
    """
    msg = f"{count_msg}\n\n"
    print(msg, end='')
    log_file.write(msg)
    log_file.flush()

    results = []
    start_time = time.time()
    groups = _group_node_ids_by_file(test_names[start_index:])
    node_index = start_index
    stop_requested = False
    for file_name, node_ids in groups:
        should_shard = shard_all or (shard_default_files and _is_default_sharded_file(file_name))
        chunks = _chunk_node_ids(node_ids, args.shard_size) if should_shard else [node_ids]
        if should_shard:
            msg = (
                f"Running {file_name} in {len(chunks)} shard(s) "
                f"of up to {args.shard_size} test(s).\n"
            )
            print(msg, end='')
            log_file.write(msg)
            log_file.flush()
        for shard_index, chunk in enumerate(chunks, start=1):
            if should_shard:
                msg = (
                    f"Shard {shard_index}/{len(chunks)} for {file_name}: "
                    f"{len(chunk)} test(s)\n"
                )
                print(msg, end='')
                log_file.write(msg)
                log_file.flush()
            node_index, stop_requested = _run_file_node_group(
                file_name, chunk, args, log_file, mode, test_names, node_index, results
            )
            if stop_requested:
                break
        if stop_requested:
            break

    return _write_run_summary(results, start_time, log_file, summary_title)


def _run_file_batch_mode(test_names, start_index, args, log_file, mode, count_msg, summary_title):
    """
    Run discovered pytest node ids grouped by file. Known oversized files are
    automatically sharded to avoid one very large pytest subprocess.
    """
    return _run_grouped_file_batch_mode(
        test_names, start_index, args, log_file, mode, count_msg, summary_title,
        shard_all=False, shard_default_files=True,
    )


def _run_shard_batch_mode(test_names, start_index, args, log_file, mode, count_msg, summary_title):
    """Run discovered pytest node ids in fixed-size shards within each file."""
    return _run_grouped_file_batch_mode(
        test_names, start_index, args, log_file, mode, count_msg, summary_title,
        shard_all=True, shard_default_files=False,
    )


def _result_state_counts(results):
    """Return summary counts for result dictionaries."""
    return {
        'total': len(results),
        STATE_PASSED: sum(1 for r in results if r.get('state') == STATE_PASSED),
        STATE_SKIPPED: sum(1 for r in results if r.get('state') == STATE_SKIPPED),
        STATE_XFAILED: sum(1 for r in results if r.get('state') == STATE_XFAILED),
        STATE_ERROR: sum(1 for r in results if r.get('state') == STATE_ERROR),
        STATE_FAILED: sum(1 for r in results if r.get('state') == STATE_FAILED),
        STATE_TIMEDOUT: sum(1 for r in results if r.get('state') == STATE_TIMEDOUT),
        STATE_MISSED: sum(1 for r in results if r.get('state') == STATE_MISSED),
    }


def _counts_have_bad_results(counts):
    return (
        counts.get(STATE_ERROR, 0) > 0
        or counts.get(STATE_FAILED, 0) > 0
        or counts.get(STATE_TIMEDOUT, 0) > 0
        or counts.get(STATE_MISSED, 0) > 0
    )


def _manifest_path_for_log(log_file_path):
    return str(Path(log_file_path).with_suffix(Path(log_file_path).suffix + '.manifest.json'))


def _worker_log_path(log_file_path, worker_id):
    return str(Path(log_file_path).with_suffix(Path(log_file_path).suffix + f'.worker{worker_id}'))


def _serial_log_path(log_file_path):
    return str(Path(log_file_path).with_suffix(Path(log_file_path).suffix + '.serial'))


def _write_manifest(path, manifest):
    try:
        Path(path).write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    except OSError:
        pass


def _assign_suite_groups_to_workers(groups, num_workers):
    """Greedily assign suite/file groups to workers by discovered node count."""
    assignments = [
        {'worker_id': worker_id, 'groups': [], 'total_tests': 0}
        for worker_id in range(num_workers)
    ]
    for file_name, node_ids in sorted(groups, key=lambda item: len(item[1]), reverse=True):
        target = min(assignments, key=lambda item: item['total_tests'])
        target['groups'].append((file_name, node_ids))
        target['total_tests'] += len(node_ids)
    return assignments


def _run_worker_assigned_suites(assigned_groups, args, log_file, mode, worker_test_names):
    """Run one worker's assigned suites and write a single worker summary."""
    results = []
    start_time = time.time()
    node_index = 0
    stop_requested = False

    for file_name, node_ids in assigned_groups:
        if args.batch_mode == BATCH_MODE_TEST:
            for node_id in node_ids:
                result = _run_one_test_with_progress(
                    node_id, node_index + 1, len(worker_test_names),
                    args, log_file, by_id=True,
                    timeout=args.per_test_timeout or DEFAULT_PER_TEST_TIMEOUT_SECONDS,
                )
                results.append(result)
                node_index += 1
                if not args.no_checkpoint:
                    next_test = worker_test_names[node_index] if node_index < len(worker_test_names) else None
                    write_checkpoint(
                        args.log_file, node_id, next_test, node_index - 1, len(worker_test_names), mode,
                        csv_file=args.csv_file, pytorch_path=args.pytorch_path,
                    )
                if not result['success'] and args.stop_on_failure:
                    stop_msg = f"\nStopping due to test failure: {node_id}\n"
                    print(stop_msg, end='')
                    log_file.write(stop_msg)
                    log_file.flush()
                    stop_requested = True
                    break
        else:
            should_shard = (
                args.batch_mode == BATCH_MODE_SHARD
                or (args.batch_mode == BATCH_MODE_FILE and _is_default_sharded_file(file_name))
            )
            chunks = _chunk_node_ids(node_ids, args.shard_size) if should_shard else [node_ids]
            if should_shard:
                msg = (
                    f"Running {file_name} in {len(chunks)} shard(s) "
                    f"of up to {args.shard_size} test(s).\n"
                )
                print(msg, end='')
                log_file.write(msg)
                log_file.flush()
            for shard_index, chunk in enumerate(chunks, start=1):
                if should_shard:
                    msg = (
                        f"Shard {shard_index}/{len(chunks)} for {file_name}: "
                        f"{len(chunk)} test(s)\n"
                    )
                    print(msg, end='')
                    log_file.write(msg)
                    log_file.flush()
                node_index, stop_requested = _run_file_node_group(
                    file_name, chunk, args, log_file, mode, worker_test_names, node_index, results
                )
                if stop_requested:
                    break
        if stop_requested:
            break

    exit_code = _write_run_summary(
        results, start_time, log_file,
        f"TEST SUMMARY (worker {getattr(args, 'worker_id', '?')})",
    )
    return exit_code, _result_state_counts(results)


def _concurrent_worker_main(worker_spec, args_dict, result_queue):
    """Worker process entry point for suite-level GPU concurrency."""
    worker_id = worker_spec['worker_id']
    gpu_id = str(worker_spec['gpu_id'])
    worker_log_path = worker_spec['log']
    assigned_groups = [
        (item['file_name'], item['node_ids'])
        for item in worker_spec['suites']
    ]
    worker_test_names = [
        node_id
        for _, node_ids in assigned_groups
        for node_id in node_ids
    ]

    subprocess.os.environ['HIP_VISIBLE_DEVICES'] = gpu_id
    subprocess.os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    args = SimpleNamespace(**args_dict)
    args.log_file = worker_log_path
    args.worker_id = worker_id

    stdout_devnull = open(os.devnull, 'w', encoding='utf-8')
    original_stdout = sys.stdout
    sys.stdout = stdout_devnull
    start_epoch = time.time()
    start_time = _now_iso()
    exit_code = 1
    counts = _result_state_counts([])
    error = None

    try:
        with open(worker_log_path, 'w', encoding='utf-8') as log_file:
            log_file.write(f"Worker: {worker_id}\n")
            log_file.write(f"GPU: {gpu_id}\n")
            log_file.write(f"Mode: full_suite\n")
            log_file.write(f"Batch mode: {args.batch_mode}\n")
            log_file.write(f"Retry attempts: {args.retry_attempts}\n")
            log_file.write(f"Worker start: {start_time}\n")
            log_file.write(f"Assigned suites: {len(assigned_groups)}\n")
            log_file.write(f"Assigned tests: {len(worker_test_names)}\n\n")
            log_file.flush()
            exit_code, counts = _run_worker_assigned_suites(
                assigned_groups, args, log_file, 'full_suite', worker_test_names
            )
            end_epoch = time.time()
            log_file.write(f"Worker end: {_now_iso()}\n")
            log_file.write(f"Worker elapsed seconds: {end_epoch - start_epoch:.2f}\n")
            log_file.flush()
    except Exception:
        error = traceback.format_exc()
        end_epoch = time.time()
        try:
            with open(worker_log_path, 'a', encoding='utf-8') as log_file:
                log_file.write("\nWorker error:\n")
                log_file.write(error)
                log_file.write("\n")
        except OSError:
            pass
    finally:
        sys.stdout = original_stdout
        stdout_devnull.close()

    if error:
        exit_code = 1
    end_time = _now_iso()
    result_queue.put({
        'worker_id': worker_id,
        'gpu_id': gpu_id,
        'log': worker_log_path,
        'checkpoint': checkpoint_path(worker_log_path),
        'start_time': start_time,
        'end_time': end_time,
        'elapsed_seconds': round(end_epoch - start_epoch, 2),
        'exit_code': exit_code,
        'counts': counts,
        'error': error,
    })


def _run_concurrent_full_suite_batch(test_names, args, log_file, mode, count_prefix):
    """Run full-suite file groups concurrently across GPUs."""
    groups = _group_node_ids_by_file(test_names)
    serial_groups, parallel_groups = _split_serial_and_parallel_groups(
        groups, args.pytorch_path
    )
    num_workers = min(args.num_gpus, len(parallel_groups))
    assignments = (
        _assign_suite_groups_to_workers(parallel_groups, num_workers)
        if num_workers
        else []
    )
    manifest_path = _manifest_path_for_log(args.log_file)
    start_epoch = time.time()
    start_time = _now_iso()

    serial_spec = None
    if serial_groups:
        serial_log = _serial_log_path(args.log_file)
        serial_spec = {
            'log': serial_log,
            'checkpoint': checkpoint_path(serial_log),
            'total_tests': sum(len(node_ids) for _, node_ids in serial_groups),
            'suites': [
                {
                    'file_name': file_name,
                    'test_count': len(node_ids),
                    'node_ids': node_ids,
                }
                for file_name, node_ids in serial_groups
            ],
        }

    worker_specs = []
    for assignment in assignments:
        worker_id = assignment['worker_id']
        worker_log = _worker_log_path(args.log_file, worker_id)
        worker_specs.append({
            'worker_id': worker_id,
            'gpu_id': worker_id,
            'log': worker_log,
            'checkpoint': checkpoint_path(worker_log),
            'total_tests': assignment['total_tests'],
            'suites': [
                {
                    'file_name': file_name,
                    'test_count': len(node_ids),
                    'node_ids': node_ids,
                }
                for file_name, node_ids in assignment['groups']
            ],
        })

    manifest = {
        'mode': mode,
        'batch_mode': args.batch_mode,
        'retry_attempts': args.retry_attempts,
        'num_gpus': args.num_gpus,
        'num_workers': num_workers,
        'serial_suites': serial_spec,
        'total_tests': len(test_names),
        'parent_log': args.log_file,
        'manifest': manifest_path,
        'start_time': start_time,
        'workers': worker_specs,
    }
    _write_manifest(manifest_path, manifest)

    msg = (
        f"{count_prefix} Running with suite-level GPU concurrency. "
        f"Serial suites: {len(serial_groups)}. "
        f"Workers: {num_workers}"
        f"{f', GPUs: 0-{num_workers - 1}' if num_workers else ''}. "
        f"Retry attempts: {args.retry_attempts}. "
        f"Manifest: {manifest_path}\n\n"
    )
    print(msg, end='')
    log_file.write(msg)
    log_file.write("Mode: full_suite\n")
    log_file.write(f"Batch mode: {args.batch_mode}\n")
    log_file.write(f"Retry attempts: {args.retry_attempts}\n")
    log_file.write(f"Serial suites: {len(serial_groups)}\n")
    log_file.write(f"Num GPUs: {args.num_gpus}\n")
    log_file.write(f"Concurrent manifest: {manifest_path}\n")
    log_file.write(f"Concurrent start: {start_time}\n")
    log_file.flush()

    serial_result = None
    if serial_groups and serial_spec:
        serial_test_names = [
            node_id for _, node_ids in serial_groups for node_id in node_ids
        ]
        serial_args = copy.deepcopy(args)
        serial_args.log_file = serial_spec['log']
        serial_args.worker_id = 'serial'
        serial_start_epoch = time.time()
        serial_start_time = _now_iso()
        serial_error = None
        serial_exit_code = 1
        serial_counts = _result_state_counts([])
        try:
            with open(serial_spec['log'], 'w', encoding='utf-8') as serial_log:
                serial_log.write("Worker: serial\n")
                serial_log.write("GPU: inherited\n")
                serial_log.write("Mode: full_suite\n")
                serial_log.write(f"Batch mode: {args.batch_mode}\n")
                serial_log.write(f"Retry attempts: {args.retry_attempts}\n")
                serial_log.write(f"Worker start: {serial_start_time}\n")
                serial_log.write(f"Assigned suites: {len(serial_groups)}\n")
                serial_log.write(f"Assigned tests: {len(serial_test_names)}\n\n")
                serial_log.flush()
                serial_exit_code, serial_counts = _run_worker_assigned_suites(
                    serial_groups,
                    serial_args,
                    serial_log,
                    'full_suite',
                    serial_test_names,
                )
                serial_end_epoch = time.time()
                serial_log.write(f"Worker end: {_now_iso()}\n")
                serial_log.write(
                    f"Worker elapsed seconds: {serial_end_epoch - serial_start_epoch:.2f}\n"
                )
                serial_log.flush()
        except Exception:
            serial_error = traceback.format_exc()
            serial_end_epoch = time.time()
            try:
                with open(serial_spec['log'], 'a', encoding='utf-8') as serial_log:
                    serial_log.write("\nWorker error:\n")
                    serial_log.write(serial_error)
                    serial_log.write("\n")
            except OSError:
                pass
        if serial_error:
            serial_exit_code = 1
        serial_result = {
            'log': serial_spec['log'],
            'checkpoint': serial_spec['checkpoint'],
            'start_time': serial_start_time,
            'end_time': _now_iso(),
            'elapsed_seconds': round(serial_end_epoch - serial_start_epoch, 2),
            'exit_code': serial_exit_code,
            'counts': serial_counts,
            'error': serial_error,
        }
        manifest['serial_suites'].update(serial_result)
        _write_manifest(manifest_path, manifest)

        if serial_exit_code != 0 and args.stop_on_failure:
            num_workers = 0
            worker_specs = []

    result_queue = multiprocessing.Queue()
    args_dict = vars(args).copy()
    processes = []
    for worker_spec in worker_specs:
        process = multiprocessing.Process(
            target=_concurrent_worker_main,
            args=(worker_spec, args_dict, result_queue),
        )
        process.start()
        processes.append(process)

    worker_results = []
    for _ in processes:
        worker_results.append(result_queue.get())
    for process in processes:
        process.join()

    end_epoch = time.time()
    end_time = _now_iso()
    aggregate_counts = _result_state_counts([])
    if serial_result:
        counts = serial_result.get('counts') or {}
        aggregate_counts['total'] += counts.get('total', 0)
        for state in [
            STATE_PASSED, STATE_SKIPPED, STATE_XFAILED, STATE_ERROR,
            STATE_FAILED, STATE_TIMEDOUT, STATE_MISSED,
        ]:
            aggregate_counts[state] += counts.get(state, 0)
    for result in worker_results:
        counts = result.get('counts') or {}
        aggregate_counts['total'] += counts.get('total', 0)
        for state in [
            STATE_PASSED, STATE_SKIPPED, STATE_XFAILED, STATE_ERROR,
            STATE_FAILED, STATE_TIMEDOUT, STATE_MISSED,
        ]:
            aggregate_counts[state] += counts.get(state, 0)
        for worker in manifest['workers']:
            if worker['worker_id'] == result['worker_id']:
                worker.update(result)
                break

    manifest['end_time'] = end_time
    manifest['elapsed_seconds'] = round(end_epoch - start_epoch, 2)
    manifest['counts'] = aggregate_counts
    manifest['exit_code'] = 1 if _counts_have_bad_results(aggregate_counts) else 0
    _write_manifest(manifest_path, manifest)

    summary = f"\n{'='*70}\nTEST SUMMARY (full suite concurrent)\n{'='*70}\n"
    summary += f"Total tests run: {aggregate_counts['total']}\n"
    summary += f"Passed: {aggregate_counts[STATE_PASSED]}\n"
    summary += f"Skipped: {aggregate_counts[STATE_SKIPPED]}\n"
    summary += f"Xfailed: {aggregate_counts[STATE_XFAILED]}\n"
    summary += f"Error: {aggregate_counts[STATE_ERROR]}\n"
    summary += f"Failed: {aggregate_counts[STATE_FAILED]}\n"
    summary += f"Timed out: {aggregate_counts[STATE_TIMEDOUT]}\n"
    summary += f"Missed: {aggregate_counts[STATE_MISSED]}\n"
    summary += f"Total time: {end_epoch - start_epoch:.2f}s\n"
    summary += f"Concurrent manifest: {manifest_path}\n"
    summary += f"{'='*70}\n\n"
    print(summary, end='')
    log_file.write(summary)
    log_file.flush()
    return manifest['exit_code']


def _run_full_suite_batch(test_names, start_index, args, log_file, mode, count_prefix):
    """Dispatch full-suite execution to the selected batch mode."""
    if args.num_gpus > 1:
        if start_index != 0:
            msg = "--num-gpus does not support resume/start offsets yet. Start a fresh full-suite run.\n"
            print(msg, end='')
            log_file.write(msg)
            log_file.flush()
            return 1
        return _run_concurrent_full_suite_batch(test_names, args, log_file, mode, count_prefix)

    if args.batch_mode == BATCH_MODE_FILE:
        count_msg = (
            f"{count_prefix} Running in file batches. "
            f"Per-file timeout: {args.per_file_timeout}s. "
            f"Retry attempts: {args.retry_attempts}"
        )
        return _run_file_batch_mode(
            test_names, start_index, args, log_file, mode,
            count_msg=count_msg, summary_title="TEST SUMMARY (full suite)"
        )
    if args.batch_mode == BATCH_MODE_SHARD:
        count_msg = (
            f"{count_prefix} Running in shards of up to {args.shard_size} test(s). "
            f"Per-file timeout: {args.per_file_timeout}s. "
            f"Retry attempts: {args.retry_attempts}"
        )
        return _run_shard_batch_mode(
            test_names, start_index, args, log_file, mode,
            count_msg=count_msg, summary_title="TEST SUMMARY (full suite)"
        )

    count_msg = f"{count_prefix} Per-test timeout: {args.per_test_timeout}s. Retry attempts: {args.retry_attempts}"
    return _run_test_batch(
        test_names, start_index, args, log_file, mode, by_id=True,
        count_msg=count_msg, summary_title="TEST SUMMARY (full suite)"
    )


def main():
    """Main function to run all tests."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run PyTorch inductor tests from a CSV file, the full test suite, or re-run failed tests from a log file'
    )
    parser.add_argument(
        'csv_file',
        nargs='?',
        default=None,
        help='Path to CSV file containing pytest node IDs in a test_name column (required unless --all-tests or --rerun-failed is used)'
    )
    parser.add_argument(
        '--all-tests',
        action='store_true',
        help='Run all tests in the inductor test suite (no CSV file needed)'
    )
    parser.add_argument(
        '--include-inductor-all-tests',
        action='store_true',
        dest='include_inductor_all_tests',
        help='Add every PyTorch test/inductor test file from PYTORCH_PATH; implies --all-tests'
    )
    parser.add_argument(
        '--include-triton-nightly-inductor-tests',
        action='store_true',
        help='Add the ROCm torch-triton-nightly Inductor validation test files; implies --all-tests'
    )
    parser.add_argument(
        '--rerun-failed',
        metavar='LOG_FILE',
        default=None,
        help='Re-run tests that failed (and optionally timed out); LOG_FILE is the log from a previous run'
    )
    parser.add_argument(
        '--rerun-include-timeouts',
        action='store_true',
        help='With --rerun-failed, also re-run tests that timed out (default: only re-run failed tests)'
    )
    parser.add_argument(
        '--pytorch-path',
        required=True,
        help='Path to PyTorch directory (containing test/inductor/test_cuda_repro.py)'
    )
    parser.add_argument(
        '--log-file',
        help='Path to log file (default: test_results_TIMESTAMP.log)',
        default=None
    )
    parser.add_argument(
        '--stop-on-failure',
        action='store_true',
        help='Stop running tests after the first failure (default: continue on failure)'
    )
    parser.add_argument(
        '--retry-attempts',
        type=int,
        default=2,
        metavar='N',
        help='Number of times to retry a failed test before recording final failure (default: 2; use 0 for no retries)'
    )
    # Backward-compatible alias for the original option name. Keep it hidden so
    # the CLI distinguishes retry attempts from --rerun-failed mode.
    parser.add_argument(
        '--reruns',
        dest='retry_attempts',
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--per-test-timeout',
        type=int,
        default=None,
        metavar='SECONDS',
        help='Per-test pytest-timeout value in seconds for --batch-mode test, CSV, and rerun modes (default: 1200)'
    )
    parser.add_argument(
        '--batch-mode',
        choices=[BATCH_MODE_FILE, BATCH_MODE_SHARD, BATCH_MODE_TEST],
        default=BATCH_MODE_FILE,
        help='Full-suite execution granularity: file batches by default, fixed-size shards, or one subprocess per pytest node with test'
    )
    parser.add_argument(
        '--per-file-timeout',
        type=int,
        default=DEFAULT_PER_FILE_TIMEOUT_SECONDS,
        metavar='SECONDS',
        help='Per-file subprocess timeout in seconds for --batch-mode file/shard (default: 43200)'
    )
    parser.add_argument(
        '--shard-size',
        type=int,
        default=DEFAULT_SHARD_SIZE,
        metavar='N',
        help='Number of pytest node ids per shard for --batch-mode shard and default opinfo sharding (default: 100)'
    )
    parser.add_argument(
        '--num-gpus',
        type=int,
        default=1,
        metavar='N',
        help='Full-suite only: run up to N test suites concurrently, one worker per GPU (default: 1)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from the next test after the last run (uses checkpoint for this log file)'
    )
    parser.add_argument(
        '--no-checkpoint',
        action='store_true',
        help='Disable checkpointing (default: checkpoint after each test for resume)'
    )
    parser.add_argument(
        '--regex',
        metavar='PATTERN',
        default=None,
        help='In full-suite mode (--all-tests), only run tests whose name matches PATTERN (regex). E.g. --regex GPUTest runs tests containing "GPUTest"'
    )
    parser.add_argument(
        '-i', '--input-files',
        nargs='+',
        default=None,
        metavar='FILE',
        help='In full-suite mode (--all-tests): run these test files under PYTORCH_PATH/test/ (e.g. -i test_ops.py test_nn.py). Default: test/inductor/test_torchinductor.py'
    )
    parser.add_argument(
        '--collect-only',
        action='store_true',
        help='Only count tests (no execution). Use with CSV file, --all-tests, or --rerun-failed. Prints total tests and number skipped due to decorators.'
    )

    args = parser.parse_args()

    if args.include_inductor_all_tests:
        args.all_tests = True
        try:
            inductor_files = discover_inductor_all_files(args.pytorch_path)
        except RuntimeError as e:
            print(f"Error: {e}")
            sys.exit(1)
        args.input_files = list(dict.fromkeys((args.input_files or []) + inductor_files))
        print(f"Added {len(inductor_files)} inductor test file(s).")

    if args.include_triton_nightly_inductor_tests:
        args.all_tests = True
        triton_files = triton_nightly_inductor_files()
        args.input_files = list(dict.fromkeys((args.input_files or []) + triton_files))
        print(f"Added {len(triton_files)} torch-triton-nightly inductor test file(s).")

    # -i only applies to full-suite mode
    if args.input_files is not None and not args.all_tests:
        print("Error: -i/--input-files can only be used with --all-tests")
        sys.exit(1)

    # Verify PyTorch path exists
    pytorch_path = Path(args.pytorch_path)

    if not pytorch_path.exists():
        print(f"Error: PyTorch path does not exist: {args.pytorch_path}")
        sys.exit(1)

    # Resolve full-suite test file(s) to check: -i list (under test/) or default single file.
    # CSV and rerun-failed modes pass pytest node IDs directly, so pytest validates those paths.
    if args.all_tests:
        if args.input_files:
            test_file_rel_paths = [str(Path('test') / f) for f in args.input_files]
            for rel in test_file_rel_paths:
                if not (pytorch_path / rel).exists():
                    print(f"Error: Test file not found at: {pytorch_path / rel}")
                    print("Please verify -i filenames and PyTorch path.")
                    sys.exit(1)
        else:
            test_file = pytorch_path / TEST_FILE_REL_PATH
            if not test_file.exists():
                print(f"Error: Test file not found at: {test_file}")
                print(f"Please verify the PyTorch path is correct.")
                sys.exit(1)
    
    # Require exactly one of: CSV file, --all-tests, or --rerun-failed
    modes_set = sum([bool(args.csv_file), args.all_tests, bool(args.rerun_failed)])
    if modes_set == 0:
        print("Error: Either provide a CSV file, use --all-tests, or use --rerun-failed LOG_FILE")
        sys.exit(1)
    if modes_set > 1:
        print("Error: Use only one of: CSV file, --all-tests, or --rerun-failed")
        sys.exit(1)
    if args.regex and not args.all_tests:
        print("Warning: --regex only applies to full-suite mode (--all-tests); ignoring --regex.\n")
    if args.per_test_timeout is None:
        args.per_test_timeout = DEFAULT_PER_TEST_TIMEOUT_SECONDS
    if args.retry_attempts < 0:
        print("Error: --retry-attempts must be a non-negative integer")
        sys.exit(1)
    if args.shard_size <= 0:
        print("Error: --shard-size must be a positive integer")
        sys.exit(1)
    if args.num_gpus <= 0:
        print("Error: --num-gpus must be a positive integer")
        sys.exit(1)
    if args.num_gpus > 1 and not args.all_tests:
        print("Error: --num-gpus > 1 is only supported with full-suite mode (--all-tests)")
        sys.exit(1)
    if args.num_gpus > 1 and args.resume:
        print("Error: --resume is not supported with --num-gpus > 1 yet. Start a fresh full-suite run.")
        sys.exit(1)

    # Inductor ROCm tests require ROCM_HOME to be set in the user environment.
    try:
        _require_rocm_home()
    except RuntimeError as e:
        print(f"Error: {e}")
        print("Example: export ROCM_HOME=/opt/rocm")
        sys.exit(1)
    
    # Generate log file name if not provided (only used when not --collect-only)
    if args.log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if args.rerun_failed:
            args.log_file = f"{Path(args.rerun_failed).stem}.rerun_{timestamp}.log"
        else:
            args.log_file = f'test_results_{timestamp}.log'

    # In collect-only mode we don't create a log file (use in-memory buffer for any internal writes).
    # Resume appends so previously completed results remain available for final analysis.
    log_mode = 'a' if args.resume else 'w'
    log_file = io.StringIO() if args.collect_only else open(args.log_file, log_mode, encoding='utf-8')

    try:
        if not args.collect_only:
            msg = f"PyTorch path: {args.pytorch_path}\n"
            msg += f"Logging to: {args.log_file}\n\n"
            print(msg, end='')
            log_file.write(msg)
            log_file.flush()

        exit_code = 1  # default if we never run the batch

        if args.rerun_failed:
            # Rerun mode: parse log for failed tests, timed-out tests, and mode
            failed_tests, timeout_tests, mode = parse_log_for_rerun(args.rerun_failed)
            if failed_tests is None:
                print(f"Error: Log file not found or could not be read: {args.rerun_failed}")
                sys.exit(1)
            if mode is None:
                print("Error: Log file must contain 'Mode: full_suite' or 'Mode: csv' (from a previous run of this script).")
                sys.exit(1)
            tests_to_rerun = failed_tests
            if args.rerun_include_timeouts:
                tests_to_rerun = failed_tests + timeout_tests
            if not tests_to_rerun:
                msg = "No failed tests to re-run."
                if timeout_tests and not args.rerun_include_timeouts:
                    msg += " (Use --rerun-include-timeouts to also re-run timed out tests.)"
                msg += "\n"
                print(msg, end='')
                log_file.write(msg)
                log_file.flush()
                exit_code = 0
            else:
                ensure_pytest_and_timeout_installed()
                if args.collect_only:
                    _do_collect_only_and_exit(args.pytorch_path, tests_to_rerun, by_id=True)
                msg = f"Re-running failed tests from: {args.rerun_failed}\n"
                if args.rerun_include_timeouts and timeout_tests:
                    msg += f"Including {len(timeout_tests)} timed out test(s).\n"
                print(msg, end='')
                log_file.write(msg)
                log_file.write(f"Mode: {mode}\n")
                log_file.flush()
                count_msg = (
                    f"Re-running {len(tests_to_rerun)} test(s). "
                    f"Per-test timeout: {args.per_test_timeout}s. "
                    f"Retry attempts: {args.retry_attempts}"
                )
                exit_code = _run_test_batch(
                    tests_to_rerun, 0, args, log_file, mode, by_id=True,
                    count_msg=count_msg, summary_title="TEST SUMMARY (rerun failed)"
                )
        elif args.all_tests:
            ensure_pytest_and_timeout_installed()
            # Full-suite mode: discover tests, then run each with per-test timeout (pytest-timeout)
            msg = "Discovering tests (pytest --collect-only)...\n"
            print(msg, end='')
            log_file.write(msg)
            log_file.write("Mode: full_suite\n")
            log_file.flush()
            test_file_rel_paths = None
            if args.input_files:
                test_file_rel_paths = [str(Path('test') / f) for f in args.input_files]
            test_names = discover_tests(args.pytorch_path, log_file, test_file_rel_paths=test_file_rel_paths)
            if not test_names:
                msg = "No tests discovered. Check that pytest can collect from the test file.\n"
                print(msg, end='')
                log_file.write(msg)
                log_file.flush()
            else:
                # Apply --regex filter if given (only in full-suite mode)
                if args.regex:
                    try:
                        pattern = re.compile(args.regex)
                    except re.error as e:
                        msg = f"Error: invalid regex pattern {args.regex!r}: {e}\n"
                        print(msg, end='')
                        log_file.write(msg)
                        log_file.flush()
                        sys.exit(1)
                    original_count = len(test_names)
                    test_names = [n for n in test_names if pattern.search(n)]
                    msg = f"Filter --regex {args.regex!r}: {len(test_names)} test(s) match (from {original_count} discovered).\n"
                    print(msg, end='')
                    log_file.write(msg)
                    log_file.flush()
                    if not test_names:
                        msg = "No tests match the regex. Nothing to run.\n"
                        print(msg, end='')
                        log_file.write(msg)
                        log_file.flush()
                        exit_code = 0
                    else:
                        if args.collect_only:
                            _do_collect_only_and_exit(args.pytorch_path, test_names, by_id=True)
                        mode = 'full_suite'
                        start_index = _resolve_start_index(
                            test_names, args.log_file, args.resume, args.no_checkpoint, log_file,
                            "not in discovered list"
                        )
                        exit_code = _run_full_suite_batch(
                            test_names, start_index, args, log_file, mode,
                            count_prefix=f"Running {len(test_names)} test(s) (filtered)."
                        )
                else:
                    if args.collect_only:
                        _do_collect_only_and_exit(args.pytorch_path, test_names, by_id=True)
                    mode = 'full_suite'
                    start_index = _resolve_start_index(
                        test_names, args.log_file, args.resume, args.no_checkpoint, log_file,
                        "not in discovered list"
                    )
                    exit_code = _run_full_suite_batch(
                        test_names, start_index, args, log_file, mode,
                        count_prefix=f"Found {len(test_names)} test(s)."
                    )
        else:
            # CSV mode: read pytest node IDs and run each one exactly.
            ensure_pytest_and_timeout_installed()
            msg = f"Reading tests from: {args.csv_file}\n"
            print(msg, end='')
            log_file.write(msg)
            log_file.write("Mode: csv\n")
            log_file.flush()
            test_names = read_tests_from_csv(args.csv_file)
            if not test_names:
                msg = "No tests found in CSV file.\n"
                print(msg, end='')
                log_file.write(msg)
                log_file.close()
                sys.exit(1)
            if args.collect_only:
                _do_collect_only_and_exit(args.pytorch_path, test_names, by_id=True)
            mode = 'csv'
            start_index = _resolve_start_index(
                test_names, args.log_file, args.resume, args.no_checkpoint, log_file,
                "not in CSV list"
            )
            count_msg = f"Found {len(test_names)} test(s) to run"
            exit_code = _run_test_batch(
                test_names, start_index, args, log_file, mode, by_id=True,
                count_msg=count_msg, summary_title="TEST SUMMARY"
            )

    finally:
        log_file.close()
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
