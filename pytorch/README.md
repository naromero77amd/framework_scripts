# PyTorch Unit Test Runner

`run_tests.py` runs PyTorch unit tests from a single test file. It supports three modes: running tests from a CSV file, running the full discovered test suite, or re-running only tests that failed or timed out in a previous run. The test file path is defined in the script (`TEST_FILE_REL_PATH`); change it to target a different test module.

## Requirements

- **PyTorch path**: A directory containing the PyTorch source tree. The tree must contain the test file configured in the script (see `TEST_FILE_REL_PATH` in `run_tests.py`). Pass the PyTorch root with `--pytorch-path` (required for all modes).
- The script sets `PYTORCH_TEST_WITH_ROCM=1` when invoking tests.

## Modes

You must use exactly one of the following:

| Mode | How to invoke | Description |
|------|----------------|-------------|
| **CSV** | Pass a CSV file path as a positional argument | Run only the tests listed in the CSV (column `test_name`). |
| **Full suite** | `--all-tests` | Discover all tests via the test file’s `--discover-tests` and run each one. |
| **Rerun failed** | `--rerun-failed LOG_FILE` | Parse a previous run’s log file and re-run only tests that failed or timed out. |

Examples:

```bash
# CSV mode
python run_tests.py my_tests.csv --pytorch-path /path/to/pytorch

# Full-suite mode
python run_tests.py --pytorch-path /path/to/pytorch --all-tests

# Rerun failed tests from a previous log
python run_tests.py --pytorch-path /path/to/pytorch --rerun-failed test_results_20250216_120000.log
```

---

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `csv_file` | One of CSV / all-tests / rerun-failed | Path to CSV with a `test_name` column. Omit when using `--all-tests` or `--rerun-failed`. |
| `--all-tests` | One of the three modes | Discover and run all tests in the configured test file. |
| `--rerun-failed LOG_FILE` | One of the three modes | Re-run tests that failed or timed out; `LOG_FILE` is the log from a previous run. |
| `--pytorch-path PATH` | Yes | Path to PyTorch directory (must contain the test file defined in the script). |
| `--log-file PATH` | No | Where to write the run log. Default: `test_results_YYYYMMDD_HHMMSS.log`, or for rerun `{input_stem}.rerun_{timestamp}.log`. |
| `--stop-on-failure` | No | Stop after the first failing test (default: continue). |
| `--per-test-timeout SECONDS` | No | Timeout per test in seconds (default: 300). Applies to CSV, full-suite, and rerun modes. |
| `--resume` | No | Resume from the next test after the last run using the checkpoint for the given log file. |
| `--no-checkpoint` | No | Disable writing checkpoints (default: checkpoint after each test for resume). |

---

## CSV mode

- The CSV must have a column named **`test_name`**. Each row’s value is used as a test name (keyword match with the test file’s `-k`).
- Empty `test_name` rows are skipped.
- Tests are run in the order they appear in the CSV.

---

## Full-suite mode (`--all-tests`)

- Runs the test file with `--discover-tests` to get the full list of test identifiers (unittest-style ids).
- Each test is run with a per-test timeout. Test names are passed by method name (e.g. `-k test_foo`) so that `__main__` resolution works correctly.

---

## Rerun-failed mode (`--rerun-failed LOG_FILE`)

- **Input**: Path to a log file produced by a previous run of this script (CSV or full-suite).
- **Parsing**: The script looks for:
  - **Mode**: A line `Mode: full_suite` or `Mode: csv` (written by this script) so it knows whether to run tests by unittest id or by keyword.
  - **Failed tests**: The “Failed tests:” summary section and lines like `  - test_name (X.XXs)` (failed or timed-out tests).
- **Output**: A **new** log file is created (e.g. `test_results_20250216_120000.rerun_20250216_130000.log`), so the original log is not overwritten.
- If the log contains no failed tests, the script prints “No failed tests found in log. Nothing to re-run.” and exits with code 0.
- If the log file is missing or does not contain a `Mode:` line, the script exits with an error.

---

## Checkpointing and resume

- By default, after each test the script writes a **checkpoint** file next to the log file: `{log_file_path}.checkpoint`.
- The checkpoint stores the last test run, the next test to run, indices, mode, and (when applicable) CSV path and PyTorch path.
- Use **`--resume`** with the **same log file path** (and, if you let the script choose the log name, the same generated name) to continue from the next test. If the run had already completed (no “next” test), the script starts from the first test again.
- If a checkpoint exists and you run **without** `--resume`, the script reports the checkpoint state and deletes it, then starts from the first test.
- Use **`--no-checkpoint`** to disable writing and using checkpoints.

---

## Log file format

- Every run logs: PyTorch path, log file path, and (for rerun) “Re-running failed tests from: …”.
- **Mode line**: CSV and full-suite runs write `Mode: csv` or `Mode: full_suite` near the top so that `--rerun-failed` can parse the log correctly.
- For each test: a “Running: &lt;test_name&gt;” header, then test output, then a status line: `✓ PASSED`, `✗ FAILED`, or `✗ TIMEOUT`.
- At the end: a summary with total run, passed, failed, total time, and (if any) a “Failed tests:” list used by `--rerun-failed`.

---

## Exit codes

- **0**: All tests that were run passed (or nothing to run, e.g. rerun with no failures in the log).
- **1**: At least one test failed, or an error (e.g. invalid arguments, missing log/mode for rerun, discovery failure).

---

## Examples

```bash
# Run a subset of tests from CSV, 120s timeout per test
python run_tests.py tests.csv --pytorch-path /path/to/pytorch --per-test-timeout 120

# Run full suite, stop on first failure, custom log path
python run_tests.py --all-tests --pytorch-path /path/to/pytorch --stop-on-failure --log-file full_run.log

# Resume a previous run (same log file path)
python run_tests.py --all-tests --pytorch-path /path/to/pytorch --log-file full_run.log --resume

# Re-run only the tests that failed in full_run.log; output to a new .rerun_*.log file
python run_tests.py --pytorch-path /path/to/pytorch --rerun-failed full_run.log
```
