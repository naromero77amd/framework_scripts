# PyTorch Unit Test Runner

`run_tests.py` runs PyTorch unit tests from a single test file. It supports three modes: running tests from a CSV file, running the full discovered test suite, or re-running only tests that failed (and optionally timed out) from a previous run. The test file path is defined in the script (`TEST_FILE_REL_PATH`); change it to target a different test module.

## Requirements

- **PyTorch path**: A directory containing the PyTorch source tree. The tree must contain the test file configured in the script (see `TEST_FILE_REL_PATH` in `run_tests.py`). Pass the PyTorch root with `--pytorch-path` (required for all modes).
- The script sets `PYTORCH_TEST_WITH_ROCM=1` when invoking tests.
- **pytest and pytest-timeout**: All modes run tests via pytest with per-test timeout from the pytest-timeout plugin. Install with `pip install pytest pytest-timeout`. The script checks that both are installed and aborts with a clear message if not.

## Modes

You must use exactly one of the following:

| Mode | How to invoke | Description |
|------|----------------|-------------|
| **CSV** | Pass a CSV file path as a positional argument | Run only the tests listed in the CSV (column `test_name`). |
| **Full suite** | `--all-tests` | Discover all tests via the test file’s `pytest <test_file> --collect-only` and run each one. Optionally filter by name with `--regex PATTERN`. |
| **Rerun failed** | `--rerun-failed LOG_FILE` | Parse a previous run’s log and re-run only tests that **failed** (not timeouts). Add `--rerun-include-timeouts` to also re-run timed-out tests. |

Examples:

```bash
# CSV mode
python run_tests.py my_tests.csv --pytorch-path /path/to/pytorch

# Full-suite mode (optionally filter by test name)
python run_tests.py --pytorch-path /path/to/pytorch --all-tests
python run_tests.py --pytorch-path /path/to/pytorch --all-tests --regex GPUTests

# Rerun failed tests from a previous log
python run_tests.py --pytorch-path /path/to/pytorch --rerun-failed test_results_20250216_120000.log
```

---

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `csv_file` | One of CSV / all-tests / rerun-failed | Path to CSV with a `test_name` column. Omit when using `--all-tests` or `--rerun-failed`. |
| `--all-tests` | One of the three modes | Discover and run all tests in the configured test file. |
| `--rerun-failed LOG_FILE` | One of the three modes | Re-run tests that failed; `LOG_FILE` is the log from a previous run. By default timed-out tests are excluded; use `--rerun-include-timeouts` to include them. |
| `--rerun-include-timeouts` | No | With `--rerun-failed`, also re-run tests that timed out (default: only re-run failed tests). |
| `--pytorch-path PATH` | Yes | Path to PyTorch directory (must contain the test file defined in the script). |
| `--log-file PATH` | No | Where to write the run log. Default: `test_results_YYYYMMDD_HHMMSS.log`, or for rerun `{input_stem}.rerun_{timestamp}.log`. |
| `--stop-on-failure` | No | Stop after the first failing test (default: continue). |
| `--per-test-timeout SECONDS` | No | Timeout per test in seconds (default: 300). Applies to CSV, full-suite, and rerun modes. |
| `--resume` | No | Resume from the next test after the last run using the checkpoint for the given log file. |
| `--no-checkpoint` | No | Disable writing checkpoints (default: checkpoint after each test for resume). |
| `--regex PATTERN` | No | **Full-suite only.** Only run tests whose full name (e.g. `ClassName.test_method`) matches the regex. E.g. `--regex GPUTests` runs tests containing “GPUTests”. Ignored in CSV and rerun modes. |

---

## CSV mode

- The CSV must have a column named **`test_name`**. Each row’s value is used as a pytest keyword expression (`-k`). Tests are run with `pytest <test_file> -k <test_name> --timeout <seconds>` from the PyTorch path.
- Empty `test_name` rows are skipped.
- Tests are run in the order they appear in the CSV.

---

## Full-suite mode (`--all-tests`)

- Runs `pytest <test_file> --collect-only -q` to get one full pytest node id per line (e.g. `path::Class::test_method` or `path::Class::test_method[param]`). This gives a **1:1 mapping**: each collected item (including each parametrized variant) is run exactly once.
- Each test is run with `pytest --timeout <seconds> <node_id>` from the PyTorch path. The timeout is enforced by the **pytest-timeout** plugin (see Requirements).
- **`--regex PATTERN`**: When given, only tests whose full id matches the regex are run (e.g. `--regex GPUTests` to run only tests whose name contains “GPUTests”). The script reports how many tests match and how many were discovered before filtering. If no tests match, it exits successfully without running any tests.

---

## Rerun-failed mode (`--rerun-failed LOG_FILE`)

- **Input**: Path to a log file produced by a previous run of this script (CSV or full-suite).
- **Default behavior**: Only tests that **failed** (assertion/error) are re-run. Tests that **timed out** are listed in the log but are not re-run unless you pass `--rerun-include-timeouts`.
- **Parsing**: The script looks for:
  - **Mode**: A line `Mode: full_suite` or `Mode: csv` so it knows whether to run tests by unittest id or by keyword.
  - **Failed tests**: The “Failed tests:” summary section (lines `  - test_name (X.XXs)`).
  - **Timed out tests**: The “Timed out tests:” summary section (same line format). These are only re-run with `--rerun-include-timeouts`.
- **Output**: A **new** log file is created (e.g. `test_results_20250216_120000.rerun_20250216_130000.log`), so the original log is not overwritten.
- If there are no tests to re-run (no failures, or no failures and no timeouts with `--rerun-include-timeouts`), the script prints a message and exits with code 0.
- If the log file is missing or does not contain a `Mode:` line, the script exits with an error.

---

## Checkpointing and resume

- By default, after each test the script writes a **checkpoint** file next to the log file: `{log_file_path}.checkpoint`.
- The checkpoint stores the last test run, the next test to run, indices, mode, and (when applicable) CSV path and PyTorch path.
- Use **`--resume`** with the **same log file path** (and, if you let the script choose the log name, the same generated name) to continue from the next test. If the run had already completed (no “next” test), the script starts from the first test again.
- If a checkpoint exists and you run **without** `--resume`, the script reports the checkpoint state and deletes it, then starts from the first test.
- Use **`--no-checkpoint`** to disable writing and using checkpoints.

---

## Test outcome states

Each test is classified into exactly one of five states by parsing the subprocess return code and captured stdout/stderr:

| State      | When |
|------------|------|
| **PASSED** | Exit code 0 and output does not indicate a skip. |
| **SKIPPED** | Exit code 0 and output indicates the test was skipped (e.g. pytest “SKIPPED” / “1 skipped”, or unittest “OK (skipped=1)”). |
| **ERROR**  | Non-zero exit and **RuntimeError** appears in stdout or stderr. |
| **FAILED** | Non-zero exit and no RuntimeError in output (e.g. assertion failure). |
| **TIMEDOUT** | The test hit the per-test timeout (`--per-test-timeout`). |

The script reports each state in the per-test status line and in the final summary (counts and optional lists per state).

---

## Log file format

- Every run logs: PyTorch path, log file path, and (for rerun) “Re-running failed tests from: …”.
- **Mode line**: CSV and full-suite runs write `Mode: csv` or `Mode: full_suite` near the top so that `--rerun-failed` can parse the log correctly.
- For each test: a “Running: &lt;test_name&gt;” header, then test output, then a status line: `✓ PASSED`, `✓ SKIPPED`, `✗ ERROR`, `✗ FAILED`, or `✗ TIMEDOUT`.
- At the end: a summary with total run, counts for Passed / Skipped / Error / Failed / Timed out, total time, and (for each non-empty state) a section listing those tests. The “Failed tests:” and “Timed out tests:” sections are used by `--rerun-failed` (and `--rerun-include-timeouts`).

---

## Exit codes

- **0**: No test ended in ERROR, FAILED, or TIMEDOUT (only PASSED and/or SKIPPED), or nothing to run (e.g. rerun with no failures in the log).
- **1**: At least one test was ERROR, FAILED, or TIMEDOUT, or a script error (e.g. invalid arguments, missing log/mode for rerun, discovery failure).

---

## Examples

```bash
# Run a subset of tests from CSV, 120s timeout per test
python run_tests.py tests.csv --pytorch-path /path/to/pytorch --per-test-timeout 120

# Run full suite, stop on first failure, custom log path
python run_tests.py --all-tests --pytorch-path /path/to/pytorch --stop-on-failure --log-file full_run.log

# Run only tests whose name contains GPUTests (full-suite mode)
python run_tests.py --all-tests --pytorch-path /path/to/pytorch --regex GPUTests

# Resume a previous run (same log file path)
python run_tests.py --all-tests --pytorch-path /path/to/pytorch --log-file full_run.log --resume

# Re-run only the tests that failed in full_run.log; output to a new .rerun_*.log file
python run_tests.py --pytorch-path /path/to/pytorch --rerun-failed full_run.log

# Re-run failed and timed-out tests from full_run.log
python run_tests.py --pytorch-path /path/to/pytorch --rerun-failed full_run.log --rerun-include-timeouts
```
