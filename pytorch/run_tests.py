#!/usr/bin/env python3
"""
Script to run PyTorch inductor unit tests from a CSV file.
"""

import csv
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Relative path to the test file (under PyTorch root); used in run_test, discover_tests, main
TEST_FILE_REL_PATH = 'test/inductor/test_torchinductor.py'


def ensure_pytest_and_timeout_installed():
    """
    Verify that pytest and pytest-timeout are installed (same interpreter as this script).
    Abort with a clear message if either is missing. Call before using pytest for discovery or execution.
    """
    result = subprocess.run(
        [sys.executable, '-c', 'import pytest; import pytest_timeout'],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        err = (result.stderr or result.stdout or '').strip()
        print("Error: Full-suite mode requires pytest and pytest-timeout.")
        print("Install them with:  pip install pytest pytest-timeout")
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
STATE_ERROR = "error"
STATE_FAILED = "failed"
STATE_TIMEDOUT = "timedout"


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


def run_test(test_name, pytorch_path, log_file, timeout=300, by_id=False):
    """
    Run a single test with the specified test name.

    Args:
        test_name: Name of the test to run. When by_id=True this is a full pytest node id
                   (e.g. path::Class::test_method or path::Class::test_method[param]); we run
                   pytest with that node id for 1:1 mapping. When by_id=False, pass as -k <test_name> (CSV/keyword mode).
        pytorch_path: Path to PyTorch directory
        log_file: File object to write logs to
        timeout: Timeout in seconds for this test (default 300)
        by_id: If True, test_name is a full pytest node id; run pytest <node_id> from pytorch_path.

    Returns:
        tuple: (success: bool, elapsed_time: float, timed_out: bool, state: str)
        state is one of STATE_PASSED, STATE_SKIPPED, STATE_ERROR, STATE_FAILED, STATE_TIMEDOUT.
    """
    test_file = Path(pytorch_path) / TEST_FILE_REL_PATH

    if by_id:
        # Full-suite: test_name is a full pytest node id. Per-test timeout via pytest-timeout.
        cmd = ['pytest', '--timeout', str(timeout), test_name]
    else:
        # CSV: run pytest with -k (keyword match) and per-test timeout.
        cmd = ['pytest', TEST_FILE_REL_PATH, '-k', test_name, '--timeout', str(timeout)]

    env = {
        **subprocess.os.environ.copy(),
        'PYTORCH_TEST_WITH_ROCM': '1',
        'HSA_FORCE_FINE_GRAIN_PCIE': '1',
        'PYTORCH_TESTING_DEVICE_ONLY_FOR': 'cuda',
    }
    
    header = f"\n{'='*70}\nRunning: {test_name}\n{'='*70}\n"
    print(header, end='')
    log_file.write(header)
    log_file.flush()
    
    start_time = time.time()
    run_kw = dict(env=env, capture_output=True, text=True, cwd=str(pytorch_path))
    # No subprocess timeout; pytest-timeout handles per-test timeout for both modes.

    try:
        result = subprocess.run(cmd, **run_kw)
        
        elapsed_time = time.time() - start_time
        success = result.returncode == 0
        stdout_str = result.stdout or ""
        stderr_str = result.stderr or ""
        timed_out = False

        if success:
            state = STATE_SKIPPED if _output_indicates_skipped(stdout_str, stderr_str) else STATE_PASSED
        else:
            if _output_indicates_timeout(stdout_str, stderr_str):
                state = STATE_TIMEDOUT
                timed_out = True
            else:
                state = STATE_ERROR if _output_indicates_runtime_error(stdout_str, stderr_str) else STATE_FAILED

        # Write all output to log file
        if result.stdout:
            log_file.write(result.stdout)
        if result.stderr:
            log_file.write(result.stderr)
        log_file.flush()

        # Print status to console (five states: PASSED, SKIPPED, ERROR, FAILED, TIMEDOUT)
        status_map = {
            STATE_PASSED: "✓ PASSED",
            STATE_SKIPPED: "✓ SKIPPED",
            STATE_ERROR: "✗ ERROR",
            STATE_FAILED: "✗ FAILED",
            STATE_TIMEDOUT: "✗ TIMEDOUT",
        }
        status = status_map[state]
        status_msg = f"{status} ({elapsed_time:.2f}s)\n"
        print(status_msg, end='')
        log_file.write(status_msg)
        log_file.flush()

        return success, elapsed_time, timed_out, state

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

        return False, elapsed_time, True, STATE_TIMEDOUT

    except Exception as e:
        elapsed_time = time.time() - start_time
        error_msg = f"✗ ERROR: {str(e)}\n"
        print(error_msg, end='')
        log_file.write(error_msg)
        log_file.flush()
        return False, elapsed_time, False, STATE_ERROR


# Pytest node ID line (from --collect-only -q): path::Class::test_method or path::test_method[param]
# Path can be relative (e.g. test/inductor/test_torchinductor.py). Param part may contain - and other chars.
_NODE_ID_LINE_RE = re.compile(r'^[\w/.-]+::.+$')


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
    env = {
        **subprocess.os.environ.copy(),
        'PYTORCH_TEST_WITH_ROCM': '1',
        'HSA_FORCE_FINE_GRAIN_PCIE': '1',
        'PYTORCH_TESTING_DEVICE_ONLY_FOR': 'cuda',
    }
    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=str(pytorch_path),
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode != 0:
            msg = f"Discovery failed (exit {result.returncode}). stderr: {result.stderr or '(none)'}\n"
            print(msg, end='')
            log_file.write(msg)
            log_file.flush()
            return []
        combined = (result.stdout or '') + '\n' + (result.stderr or '')
        node_ids = _parse_pytest_collect_only_quiet(combined)
        return node_ids
    except subprocess.TimeoutExpired:
        msg = "Discovery timed out.\n"
        print(msg, end='')
        log_file.write(msg)
        log_file.flush()
        return []
    except Exception as e:
        msg = f"Discovery error: {e}\n"
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
    Read test names from CSV file.
    
    Args:
        csv_file: Path to CSV file
        
    Returns:
        list: List of test names
    """
    test_names = []
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            if 'test_name' not in reader.fieldnames:
                print(f"Error: CSV file must have a 'test_name' column")
                print(f"Found columns: {reader.fieldnames}")
                sys.exit(1)
            
            for row in reader:
                test_name = row['test_name'].strip()
                if test_name:  # Skip empty rows
                    test_names.append(test_name)
        
        return test_names
        
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found")
        sys.exit(1)
        
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        sys.exit(1)


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
    for i in range(start_index, len(test_names)):
        test_name = test_names[i]
        idx_display = i + 1
        progress_msg = f"[{idx_display}/{len(test_names)}] "
        print(progress_msg, end='')
        log_file.write(progress_msg)
        log_file.flush()
        timed_out = False
        state = STATE_FAILED
        try:
            success, elapsed, timed_out, state = run_test(
                test_name, args.pytorch_path, log_file,
                timeout=args.per_test_timeout,
                by_id=by_id
            )
        except subprocess.TimeoutExpired:
            elapsed = float(args.per_test_timeout)
            timeout_msg = f"✗ TIMEOUT (uncaught in run_test) after {elapsed:.2f}s\n"
            print(timeout_msg, end='')
            log_file.write(timeout_msg)
            log_file.flush()
            success, elapsed, state = False, elapsed, STATE_TIMEDOUT
            timed_out = True
        except Exception as e:
            err_msg = f"✗ ERROR (uncaught): {type(e).__name__}: {e}\n"
            print(err_msg, end='')
            log_file.write(err_msg)
            log_file.flush()
            success, elapsed, state = False, 0.0, STATE_ERROR
        results.append({
            'name': test_name, 'success': success, 'time': elapsed,
            'timed_out': timed_out, 'state': state,
        })
        if not args.no_checkpoint:
            next_test = test_names[i + 1] if i + 1 < len(test_names) else None
            write_checkpoint(
                args.log_file, test_name, next_test, i, len(test_names), mode,
                csv_file=args.csv_file, pytorch_path=args.pytorch_path
            )
        if not success and args.stop_on_failure:
            stop_msg = f"\nStopping due to test failure: {test_name}\n"
            print(stop_msg, end='')
            log_file.write(stop_msg)
            log_file.flush()
            break

    total_time = time.time() - start_time
    passed = sum(1 for r in results if r.get('state') == STATE_PASSED)
    skipped_count = sum(1 for r in results if r.get('state') == STATE_SKIPPED)
    error_count = sum(1 for r in results if r.get('state') == STATE_ERROR)
    failed_count = sum(1 for r in results if r.get('state') == STATE_FAILED)
    timedout_count = sum(1 for r in results if r.get('state') == STATE_TIMEDOUT)
    summary = f"\n{'='*70}\n{summary_title}\n{'='*70}\n"
    summary += f"Total tests run: {len(results)}\n"
    summary += f"Passed: {passed}\n"
    summary += f"Skipped: {skipped_count}\n"
    summary += f"Error: {error_count}\n"
    summary += f"Failed: {failed_count}\n"
    summary += f"Timed out: {timedout_count}\n"
    summary += f"Total time: {total_time:.2f}s\n"
    for state, label in [
        (STATE_PASSED, "Passed"),
        (STATE_SKIPPED, "Skipped"),
        (STATE_ERROR, "Error"),
        (STATE_FAILED, "Failed"),
        (STATE_TIMEDOUT, "Timed out"),
    ]:
        subset = [r for r in results if r.get('state') == state]
        if subset:
            summary += f"\n{label} tests:\n"
            for r in subset:
                summary += f"  - {r['name']} ({r['time']:.2f}s)\n"
    summary += f"{'='*70}\n\n"
    print(summary, end='')
    log_file.write(summary)
    log_file.flush()
    any_bad = error_count > 0 or failed_count > 0 or timedout_count > 0
    return 0 if not any_bad else 1


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
        help='Path to CSV file containing test names (required unless --all-tests or --rerun-failed is used)'
    )
    parser.add_argument(
        '--all-tests',
        action='store_true',
        help='Run all tests in the inductor test suite (no CSV file needed)'
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
        '--per-test-timeout',
        type=int,
        default=300,
        metavar='SECONDS',
        help='Per-test timeout in seconds for CSV and full-suite mode (default: 300)'
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

    args = parser.parse_args()

    # -i only applies to full-suite mode
    if args.input_files is not None and not args.all_tests:
        print("Error: -i/--input-files can only be used with --all-tests")
        sys.exit(1)

    # Verify PyTorch path exists
    pytorch_path = Path(args.pytorch_path)

    if not pytorch_path.exists():
        print(f"Error: PyTorch path does not exist: {args.pytorch_path}")
        sys.exit(1)

    # Resolve test file(s) to check: -i list (under test/) or default single file
    if args.all_tests and args.input_files:
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
    
    # Generate log file name if not provided
    if args.log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if args.rerun_failed:
            args.log_file = f"{Path(args.rerun_failed).stem}.rerun_{timestamp}.log"
        else:
            args.log_file = f'test_results_{timestamp}.log'
    
    # Open log file
    log_file = open(args.log_file, 'w', encoding='utf-8')
    
    try:
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
                msg = f"Re-running failed tests from: {args.rerun_failed}\n"
                if args.rerun_include_timeouts and timeout_tests:
                    msg += f"Including {len(timeout_tests)} timed out test(s).\n"
                print(msg, end='')
                log_file.write(msg)
                log_file.write(f"Mode: {mode}\n")
                log_file.flush()
                count_msg = f"Re-running {len(tests_to_rerun)} test(s). Per-test timeout: {args.per_test_timeout}s"
                exit_code = _run_test_batch(
                    tests_to_rerun, 0, args, log_file, mode, by_id=(mode == 'full_suite'),
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
                        mode = 'full_suite'
                        start_index = _resolve_start_index(
                            test_names, args.log_file, args.resume, args.no_checkpoint, log_file,
                            "not in discovered list"
                        )
                        count_msg = f"Running {len(test_names)} test(s) (filtered). Per-test timeout: {args.per_test_timeout}s"
                        exit_code = _run_test_batch(
                            test_names, start_index, args, log_file, mode, by_id=True,
                            count_msg=count_msg, summary_title="TEST SUMMARY (full suite)"
                        )
                else:
                    mode = 'full_suite'
                    start_index = _resolve_start_index(
                        test_names, args.log_file, args.resume, args.no_checkpoint, log_file,
                        "not in discovered list"
                    )
                    count_msg = f"Found {len(test_names)} test(s). Per-test timeout: {args.per_test_timeout}s"
                    exit_code = _run_test_batch(
                        test_names, start_index, args, log_file, mode, by_id=True,
                        count_msg=count_msg, summary_title="TEST SUMMARY (full suite)"
                    )
        else:
            # CSV mode: read test names and run each one (pytest -k, same as full-suite for execution)
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
            mode = 'csv'
            start_index = _resolve_start_index(
                test_names, args.log_file, args.resume, args.no_checkpoint, log_file,
                "not in CSV list"
            )
            count_msg = f"Found {len(test_names)} test(s) to run"
            exit_code = _run_test_batch(
                test_names, start_index, args, log_file, mode, by_id=False,
                count_msg=count_msg, summary_title="TEST SUMMARY"
            )

    finally:
        log_file.close()
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
