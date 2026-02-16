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
TEST_FILE_REL_PATH = 'test/inductor/test_cuda_repro.py'


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


def run_test(test_name, pytorch_path, log_file, timeout=300, by_id=False):
    """
    Run a single test with the specified test name.

    Args:
        test_name: Name of the test to run. When by_id=True this is a unittest id
                   (e.g. __main__.CudaReproTests.test_foo); we run with -k <method_name>.
                   When by_id=False, pass as -k <test_name> (CSV/keyword mode).
        pytorch_path: Path to PyTorch directory
        log_file: File object to write logs to
        timeout: Timeout in seconds for this test (default 300)
        by_id: If True, test_name is a unittest id; extract method name and use -k
               (script run as __main__ can't take full id as positional). If False, use -k as-is.

    Returns:
        tuple: (success: bool, elapsed_time: float, timed_out: bool)
    """
    test_file = Path(pytorch_path) / TEST_FILE_REL_PATH

    if by_id:
        # Full-suite: test_name is e.g. __main__.CudaReproTests.test_3d_tiling.
        # Passing that as positional breaks (unittest resolves __main__ wrongly). Use -k with method name.
        method_name = test_name.split('.')[-1] if '.' in test_name else test_name
        cmd = ['python', str(test_file), '-k', method_name]
    else:
        # CSV/single: run by -k (keyword match)
        cmd = ['python', str(test_file), '-k', test_name]

    env = {
        **subprocess.os.environ.copy(),
        'PYTORCH_TEST_WITH_ROCM': '1'
    }
    
    header = f"\n{'='*70}\nRunning: {test_name}\n{'='*70}\n"
    print(header, end='')
    log_file.write(header)
    log_file.flush()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        elapsed_time = time.time() - start_time
        success = result.returncode == 0
        
        # Write all output to log file
        if result.stdout:
            log_file.write(result.stdout)
        if result.stderr:
            log_file.write(result.stderr)
        log_file.flush()
        
        # Print status to console
        status = "✓ PASSED" if success else "✗ FAILED"
        status_msg = f"{status} ({elapsed_time:.2f}s)\n"
        print(status_msg, end='')
        log_file.write(status_msg)
        log_file.flush()
        
        return success, elapsed_time, False
        
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
        
        return False, elapsed_time, True
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        error_msg = f"✗ ERROR: {str(e)}\n"
        print(error_msg, end='')
        log_file.write(error_msg)
        log_file.flush()
        return False, elapsed_time, False


def discover_tests(pytorch_path, log_file):
    """
    Discover test names by running the test file with --discover-tests.

    Parses stdout: skips unittest suite repr lines (e.g. "<unittest.suite.TestSuite ...>").
    For lines like "test_method (__main__.Class.test_method)", extracts the id in
    parentheses so we run by unittest id. Returns list of run ids to pass to the runner.

    Returns:
        list: List of test run ids (unittest id or plain name per line)
    """
    test_file = Path(pytorch_path) / TEST_FILE_REL_PATH
    cmd = ['python', str(test_file), '--discover-tests']
    env = {
        **subprocess.os.environ.copy(),
        'PYTORCH_TEST_WITH_ROCM': '1'
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
        lines = (result.stdout or '').strip().splitlines()
        run_ids = []
        for line in lines:
            s = line.strip()
            if not s:
                continue
            # Skip unittest suite repr (one long line listing all tests)
            if s.startswith('<') or 'TestSuite tests=' in s:
                continue
            # "test_method (__main__.CudaReproTests.test_method)" -> use id in parens
            match = re.match(r'.*\s+\((.+)\)\s*$', s)
            if match:
                run_ids.append(match.group(1).strip())
            else:
                run_ids.append(s)
        return run_ids
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
        try:
            success, elapsed, timed_out = run_test(
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
            success, elapsed = False, elapsed
            timed_out = True
        except Exception as e:
            err_msg = f"✗ ERROR (uncaught): {type(e).__name__}: {e}\n"
            print(err_msg, end='')
            log_file.write(err_msg)
            log_file.flush()
            success, elapsed = False, 0.0
        results.append({'name': test_name, 'success': success, 'time': elapsed, 'timed_out': timed_out})
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
    passed = sum(1 for r in results if r['success'])
    failed = len(results) - passed
    summary = f"\n{'='*70}\n{summary_title}\n{'='*70}\n"
    summary += f"Total tests run: {len(results)}\n"
    summary += f"Passed: {passed}\n"
    summary += f"Failed: {failed}\n"
    summary += f"Total time: {total_time:.2f}s\n"
    if failed > 0:
        failed_only = [r for r in results if not r['success'] and not r.get('timed_out')]
        timed_out_only = [r for r in results if not r['success'] and r.get('timed_out')]
        if failed_only:
            summary += f"\nFailed tests:\n"
            for r in failed_only:
                summary += f"  - {r['name']} ({r['time']:.2f}s)\n"
        if timed_out_only:
            summary += f"\nTimed out tests:\n"
            for r in timed_out_only:
                summary += f"  - {r['name']} ({r['time']:.2f}s)\n"
    summary += f"{'='*70}\n\n"
    print(summary, end='')
    log_file.write(summary)
    log_file.flush()
    return 0 if failed == 0 else 1


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

    args = parser.parse_args()
    
    # Verify PyTorch path exists
    pytorch_path = Path(args.pytorch_path)
    test_file = pytorch_path / TEST_FILE_REL_PATH
    
    if not pytorch_path.exists():
        print(f"Error: PyTorch path does not exist: {args.pytorch_path}")
        sys.exit(1)
    
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
            # Full-suite mode: discover tests via --discover-tests, then run each with per-test timeout
            msg = "Discovering tests (--discover-tests)...\n"
            print(msg, end='')
            log_file.write(msg)
            log_file.write("Mode: full_suite\n")
            log_file.flush()
            test_names = discover_tests(args.pytorch_path, log_file)
            if not test_names:
                msg = "No tests discovered. Check that the test file supports --discover-tests.\n"
                print(msg, end='')
                log_file.write(msg)
                log_file.flush()
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
            # CSV mode: read test names and run each one
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
