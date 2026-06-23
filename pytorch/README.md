# PyTorch Unit Test Runner

`run_tests.py` runs PyTorch unit tests. It supports three modes: running tests from a CSV file, running the full discovered test suite (from one or more test files), or re-running only tests that failed (and optionally timed out) from a previous run. By default, full-suite mode uses the inductor test file (`test/inductor/test_torchinductor.py`) and runs test files in file batches. Use **`-i`** in full-suite mode to run other test files under `PYTORCH_PATH/test/` (e.g. `-i test_ops.py test_nn.py`). Use **`--include-inductor-all-tests`** to add the PyTorch CI `inductor_core` file set automatically, or **`--include-triton-nightly-inductor-tests`** to add the smaller ROCm torch-triton-nightly Inductor subset.

## Requirements

- **PyTorch path**: A directory containing the PyTorch source tree. In full-suite mode without `-i`, the tree must contain the default inductor test file; with `-i`, it must contain each specified file under `test/`. Pass the PyTorch root with `--pytorch-path` (required for all modes).
- The script sets `PYTORCH_TEST_WITH_ROCM=1`, `HSA_FORCE_FINE_GRAIN_PCIE=1`, and `PYTORCH_TESTING_DEVICE_ONLY_FOR=cuda` when invoking tests.
- **pytest, pytest-timeout, and expecttest**: All modes run tests via pytest with per-test timeout from the pytest-timeout plugin. The script checks that pytest, pytest-timeout, and expecttest are installed and aborts with a clear message if any are missing. Install with `pip install pytest pytest-timeout expecttest`.

## Modes

You must use exactly one of the following:

| Mode | How to invoke | Description |
|------|----------------|-------------|
| **CSV** | Pass a CSV file path as a positional argument | Run only the tests listed in the CSV (column `test_name`). |
| **Full suite** | `--all-tests` | Discover all tests via `pytest <test_file(s)> --collect-only`, then run by file by default. Use **`-i FILE [FILE ...]`** to specify test files under `PYTORCH_PATH/test/` (e.g. `-i test_ops.py test_nn.py`); default is the inductor test file. Optionally filter by name with `--regex PATTERN`. |
| **Inductor all** | `--include-inductor-all-tests` | Full-suite shortcut that adds the CI-derived `inductor_core` test files from the PyTorch checkout. This implies `--all-tests` and can be combined with `-i` to include additional files. |
| **Triton nightly Inductor** | `--include-triton-nightly-inductor-tests` | Full-suite shortcut that adds the smaller ROCm torch-triton-nightly Inductor validation subset. This implies `--all-tests` and can be combined with `-i` to include additional files. |
| **Rerun failed** | `--rerun-failed LOG_FILE` | Parse a previous run’s log and re-run only tests that **failed** (not timeouts). Add `--rerun-include-timeouts` to also re-run timed-out tests. |

Examples:

```bash
# CSV mode
python run_tests.py my_tests.csv --pytorch-path /path/to/pytorch

# Full-suite mode (default inductor test file; optionally filter by test name)
python run_tests.py --pytorch-path /path/to/pytorch --all-tests
python run_tests.py --pytorch-path /path/to/pytorch --all-tests --regex GPUTests

# Full-suite mode with specific test files under PYTORCH_PATH/test/
python run_tests.py --all-tests -i test_ops.py test_nn.py --pytorch-path /path/to/pytorch

# PyTorch CI inductor_core file set
python run_tests.py --include-inductor-all-tests --pytorch-path /path/to/pytorch

# ROCm torch-triton-nightly Inductor validation subset
python run_tests.py --include-triton-nightly-inductor-tests --pytorch-path /path/to/pytorch

# Rerun failed tests from a previous log
python run_tests.py --pytorch-path /path/to/pytorch --rerun-failed test_results_20250216_120000.log
```

---

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `csv_file` | One of CSV / all-tests / rerun-failed | Path to CSV with a `test_name` column. Omit when using `--all-tests` or `--rerun-failed`. |
| `--all-tests` | One of the three modes | Discover and run all tests in the configured test file. |
| `--include-inductor-all-tests` | No | Add PyTorch CI `inductor_core` test files from the checkout at `--pytorch-path`; implies `--all-tests`. The file list is derived from `tools/testing/discover_tests.py` and `.ci/pytorch/test.sh::test_inductor_core`. |
| `--include-inductor-tests` | No | Backward-compatible alias for `--include-inductor-all-tests`. |
| `--include-triton-nightly-inductor-tests` | No | Add the ROCm torch-triton-nightly Inductor validation files; implies `--all-tests`. The file list mirrors `pytorch-ci-scripts/torch-triton-nightly/inductor-tests.py`. |
| `--rerun-failed LOG_FILE` | One of the three modes | Re-run tests that failed; `LOG_FILE` is the log from a previous run. By default timed-out tests are excluded; use `--rerun-include-timeouts` to include them. |
| `--rerun-include-timeouts` | No | With `--rerun-failed`, also re-run tests that timed out (default: only re-run failed tests). |
| `--pytorch-path PATH` | Yes | Path to PyTorch directory (must contain the default test file or, with `-i`, each specified file under `test/`). |
| `--log-file PATH` | No | Where to write the run log. Default: `test_results_YYYYMMDD_HHMMSS.log`, or for rerun `{input_stem}.rerun_{timestamp}.log`. |
| `--stop-on-failure` | No | Stop after the first failing test (default: continue). |
| `--batch-mode {file,test}` | No | **Full-suite only.** `file` runs one pytest subprocess per test file and is the default. `test` runs one pytest subprocess per collected pytest node, matching the historical behavior. |
| `--per-file-timeout SECONDS` | No | Timeout per file subprocess in `--batch-mode file` (default: 1800). |
| `--per-test-timeout SECONDS` | No | Timeout per test in seconds for `--batch-mode test`, CSV, and rerun modes (default: 300). Cannot be used with `--batch-mode file`. |
| `--resume` | No | Resume from the next test after the last run using the checkpoint for the given log file. |
| `--no-checkpoint` | No | Disable writing checkpoints (default: checkpoint after each test for resume). |
| `--regex PATTERN` | No | **Full-suite only.** Only run tests whose full name (e.g. `ClassName.test_method`) matches the regex. E.g. `--regex GPUTests` runs tests containing “GPUTests”. Ignored in CSV and rerun modes. |
| `-i`, `--input-files FILE [FILE ...]` | No | **Full-suite only.** Run these test files under `PYTORCH_PATH/test/`. E.g. `-i test_ops.py test_nn.py` runs `test/test_ops.py` and `test/test_nn.py`. Default: inductor test file (`test/inductor/test_torchinductor.py`). Cannot be used without `--all-tests`. |
| `--collect-only` | No | **Count-only mode.** Use with CSV file, `--all-tests`, or `--rerun-failed`. Does not run any tests; prints total test count and a hierarchical breakdown by test class (from `pytest --collect-only`). No log file is created; output goes to the screen only. |

---

## CSV mode

- The CSV must have a column named **`test_name`**. Each row’s value is used as a pytest keyword expression (`-k`). Tests are run with `pytest <test_file> -k <test_name> --timeout <seconds>` from the PyTorch path.
- Empty `test_name` rows are skipped. Rows whose **`test_name`** starts with **`#`** are treated as comments and skipped (e.g. `#test_foo` or `# Optional test`).
- Tests are run in the order they appear in the CSV.

---

## Full-suite mode (`--all-tests`)

- **Test file(s)**: By default the script uses the inductor test file (`test/inductor/test_torchinductor.py`). Use **`-i FILE [FILE ...]`** to run one or more other test files under `PYTORCH_PATH/test/`. Each argument is a filename or path relative to `test/` (e.g. `-i test_ops.py test_nn.py` runs `test/test_ops.py` and `test/test_nn.py`). `-i` can only be used with `--all-tests`.
- **Inductor all shortcut**: **`--include-inductor-all-tests`** derives the same file set as PyTorch CI’s `inductor_core` configuration by reading `tools/testing/discover_tests.py` and `.ci/pytorch/test.sh::test_inductor_core` from `PYTORCH_PATH`. It implies `--all-tests` and appends those files to any files passed with `-i`, de-duplicating the final list. **`--include-inductor-tests`** remains as a backward-compatible alias.
- **Triton nightly Inductor shortcut**: **`--include-triton-nightly-inductor-tests`** adds the seven test files used by ROCm’s `pytorch-ci-scripts/torch-triton-nightly/inductor-tests.py`: `inductor/test_torchinductor.py`, `inductor/test_flex_attention.py`, `inductor/test_max_autotune.py`, `inductor/test_aot_inductor.py`, `inductor/test_flex_decoding.py`, `inductor/test_torchinductor_codegen_dynamic_shapes.py`, and `inductor/test_torchinductor_opinfo.py`.
- Runs `pytest <test_file(s)> --collect-only -q` to get one full pytest node id per line (e.g. `path::Class::test_method` or `path::Class::test_method[param]`). This gives a discovered test list for filtering, reporting, fallback, and resume.
- If collection fails, the log includes the collect command, stdout, and stderr so import-time errors are visible.
- **Default file mode**: `--batch-mode file` runs one pytest subprocess per test file with `--per-file-timeout` enforced by the outer subprocess timeout. Passing files are parsed with JUnit XML so per-test pass/skip/fail/error counts remain available.
- **Fallback in file mode**: If a file subprocess fails or times out, that file’s discovered node ids are rerun individually using the existing per-test path to identify exact failing or timed-out tests.
- **Per-test mode**: `--batch-mode test` runs each discovered pytest node with `pytest --timeout <seconds> <node_id>` from the PyTorch path. The timeout is enforced by the **pytest-timeout** plugin (see Requirements), with an outer subprocess timeout as a safety net if the pytest process does not exit cleanly.
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

## Collect-only mode (`--collect-only`)

- **Use with any mode**: Pass **`--collect-only`** together with a CSV file, **`--all-tests`**, or **`--rerun-failed LOG_FILE`**. The script resolves the same test list as it would for a real run, then runs **`pytest --collect-only`** (without `-q`) on that set and prints counts only; no tests are executed.
- **No log file**: In collect-only mode no log file is created or written. All output goes to the screen.
- **Output**: Prints **Total tests: N** and a **hierarchical breakdown** by test class. The breakdown is parsed from pytest’s collect-only output: each **`<UnitTestCase ClassName>`** group is listed with its test count (e.g. `  CommonTemplate: 45`). Counts include parametrized variants (each variant is one collected item).
- **Typical use**: To see how many tests would run and how they are grouped by class before doing a full run.

Example:

```bash
# Count tests from CSV
python run_tests.py my_tests.csv --pytorch-path /path/to/pytorch --collect-only

# Count tests in full suite (optionally with --regex)
python run_tests.py --all-tests --pytorch-path /path/to/pytorch --collect-only
python run_tests.py --all-tests --pytorch-path /path/to/pytorch --regex GPUTest --collect-only

# Count tests that would be re-run from a previous log
python run_tests.py --rerun-failed previous.log --pytorch-path /path/to/pytorch --collect-only
```

---

## Checkpointing and resume

- By default, after each test the script writes a **checkpoint** file next to the log file: `{log_file_path}.checkpoint`.
- The checkpoint stores the last test run, the next test to run, indices, mode, and (when applicable) CSV path and PyTorch path.
- Use **`--resume`** with the **same log file path** (and, if you let the script choose the log name, the same generated name) to continue from the next test. Resume appends to the existing log so previous results remain available for final analysis. If the run had already completed (no “next” test), the script starts from the first test again.
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

# Run full suite with historical per-test-node subprocess behavior
python run_tests.py --all-tests --batch-mode test --pytorch-path /path/to/pytorch

# Run only tests whose name contains GPUTests (full-suite mode)
python run_tests.py --all-tests --pytorch-path /path/to/pytorch --regex GPUTests

# Run full suite from specific test files under test/
python run_tests.py --all-tests -i test_ops.py test_nn.py --pytorch-path /path/to/pytorch

# Run the PyTorch CI inductor_core file set
python run_tests.py --include-inductor-all-tests --pytorch-path /path/to/pytorch

# Run the ROCm torch-triton-nightly Inductor validation subset
python run_tests.py --include-triton-nightly-inductor-tests --pytorch-path /path/to/pytorch

# Resume a previous run (same log file path)
python run_tests.py --all-tests --pytorch-path /path/to/pytorch --log-file full_run.log --resume

# Re-run only the tests that failed in full_run.log; output to a new .rerun_*.log file
python run_tests.py --pytorch-path /path/to/pytorch --rerun-failed full_run.log

# Re-run failed and timed-out tests from full_run.log
python run_tests.py --pytorch-path /path/to/pytorch --rerun-failed full_run.log --rerun-include-timeouts

# Count tests only (no run, no log file); works with CSV, --all-tests, or --rerun-failed
python run_tests.py --all-tests --pytorch-path /path/to/pytorch --collect-only
python run_tests.py tests.csv --pytorch-path /path/to/pytorch --collect-only
```
