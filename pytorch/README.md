# PyTorch Unit Test Runner

`run_tests.py` runs PyTorch unit tests. It supports three run modes:

- CSV mode: run tests listed in a CSV file.
- Full-suite mode: discover tests from one or more PyTorch test files.
- Rerun-failed mode: rerun failures, and optionally timeouts, from a previous log.

Full-suite mode defaults to `test/inductor/test_torchinductor.py` and runs one pytest process per test file for lower overhead. Use `--batch-mode test` to use the historical one-pytest-process-per-test-node behavior.

## Requirements

- **PyTorch path**: Pass the PyTorch checkout root with `--pytorch-path`.
- The script sets `PYTORCH_TEST_WITH_ROCM=1`, `HSA_FORCE_FINE_GRAIN_PCIE=1`, and `PYTORCH_TESTING_DEVICE_ONLY_FOR=cuda` when invoking tests.
- **pytest, pytest-timeout, and expecttest**: The script checks these imports and aborts with a clear message if any are missing. Install with `pip install pytest pytest-timeout expecttest`.
- Timeout behavior depends on execution strategy:
  - `--per-test-timeout` is passed to pytest-timeout and is a per-test timeout.
  - Full-suite `--batch-mode file` also uses `--per-file-timeout` as an outer safety timeout for the file subprocess.

## Run Modes

Use exactly one of these modes:

| Mode | How to invoke | Description |
|------|---------------|-------------|
| CSV | `csv_file` positional argument | Run tests listed in a CSV file with a `test_name` column. |
| Full suite | `--all-tests` | Discover tests from one or more files under `PYTORCH_PATH/test/`, then run them with the selected batch mode. |
| Rerun failed | `--rerun-failed LOG_FILE` | Parse a previous log and rerun failed tests. Add `--rerun-include-timeouts` to include timed-out tests. |

Examples:

```bash
# CSV mode
python run_tests.py my_tests.csv --pytorch-path /path/to/pytorch

# Full-suite mode with the default inductor test file
python run_tests.py --all-tests --pytorch-path /path/to/pytorch

# Full-suite mode with specific test files under PYTORCH_PATH/test/
python run_tests.py --all-tests -i test_ops.py inductor/test_config.py --pytorch-path /path/to/pytorch

# Rerun failed tests from a previous log
python run_tests.py --pytorch-path /path/to/pytorch --rerun-failed test_results_20250216_120000.log
```

## CSV Mode

- The CSV must have a column named `test_name`.
- Each row is used as a pytest keyword expression with `-k`.
- Empty `test_name` rows are skipped.
- Rows whose `test_name` starts with `#` are treated as comments.
- CSV tests run in the order they appear in the file.
- CSV mode uses the default inductor test file (`test/inductor/test_torchinductor.py`).

## Full-Suite Mode

Full-suite mode discovers pytest node IDs with:

```bash
pytest <test_file(s)> --collect-only -q
```

The discovered node IDs are used for filtering, reporting, fallback, and resume.

If collection fails, the log includes the collect command, stdout, and stderr so import-time errors are visible.

## Full-Suite Test Selection

Full-suite mode starts from a list of files under `PYTORCH_PATH/test/`.

- **Default**: Without `-i` or a shortcut, the script uses `test/inductor/test_torchinductor.py`.
- **Explicit files**: Use `-i FILE [FILE ...]` to provide files relative to `test/`, such as `-i test_ops.py inductor/test_config.py`.
- **Inductor all shortcut**: `--include-inductor-all-tests` derives the same file set as PyTorch CI's `inductor_core` configuration by reading `tools/testing/discover_tests.py` and `.ci/pytorch/test.sh::test_inductor_core` from `PYTORCH_PATH`. It implies `--all-tests` and appends those files to any files passed with `-i`, de-duplicating the final list.
- **Backward-compatible alias**: `--include-inductor-tests` is an alias for `--include-inductor-all-tests`.
- **Triton nightly Inductor shortcut**: `--include-triton-nightly-inductor-tests` adds the seven files used by ROCm's `pytorch-ci-scripts/torch-triton-nightly/inductor-tests.py`:
  - `inductor/test_torchinductor.py`
  - `inductor/test_flex_attention.py`
  - `inductor/test_max_autotune.py`
  - `inductor/test_aot_inductor.py`
  - `inductor/test_flex_decoding.py`
  - `inductor/test_torchinductor_codegen_dynamic_shapes.py`
  - `inductor/test_torchinductor_opinfo.py`
- **Regex filter**: Use `--regex PATTERN` to filter discovered pytest node IDs by full test ID.

## Full-Suite Execution Strategy

### File mode: `--batch-mode file`

File mode is the default. It runs one pytest subprocess per test file:

```bash
pytest test/inductor/test_config.py --junitxml <tempfile>
```

- Timeout is controlled by `--per-file-timeout` (default: 1800 seconds).
- Per-test pytest-timeout is still enabled inside the file subprocess via `--per-test-timeout` (default: 300 seconds).
- Passing files are parsed from pytest's JUnit XML output so per-test pass/skip/fail/error counts remain available.
- This is the fastest mode because PyTorch and pytest startup costs are paid once per file instead of once per pytest node.

### Test mode: `--batch-mode test`

Test mode preserves the historical behavior: each discovered pytest node runs in its own pytest subprocess:

```bash
pytest --timeout 300 test/inductor/test_config.py::TestInductorConfig::test_set
```

- Timeout is controlled by `--per-test-timeout` (default: 300 seconds).
- The pytest-timeout plugin enforces the test timeout.
- The script also applies an outer subprocess timeout of timeout + 60 seconds as a safety net for cases where pytest-timeout does not terminate a stuck process cleanly.

## File-Mode Fallback Behavior

`--batch-mode file` is optimized for speed, but it still tries to keep the run moving when a test hangs.

- If a file subprocess passes, JUnit XML is parsed and each testcase is recorded as passed, skipped, failed, or error.
- If a file subprocess returns a non-zero exit code but writes JUnit XML, the script records the individual failures/errors from XML. It does not rerun failures in test mode.
- If a file subprocess exceeds `--per-file-timeout`, the script parses verbose pytest output to identify the currently running pytest node, records that node as timed out, skips it, and restarts file-mode execution for the remaining nodes in that file.
- If the timed-out node cannot be identified, remaining nodes in that file are recorded as missed rather than repeatedly rerunning an unknown hang.
- Checkpoints are written after file batches and timeout recovery so interrupted runs can continue from the next discovered test.

This gives file mode most of the speed benefit of file-level execution while recording individual failures and skipping timed-out tests so the rest of the file can continue.

## Rerun-Failed Mode

- Input is a log file produced by a previous CSV or full-suite run.
- By default, only failed tests are rerun.
- Add `--rerun-include-timeouts` to also rerun timed-out tests.
- The script looks for:
  - `Mode: full_suite` or `Mode: csv`
  - `Failed tests:` summary section
  - `Timed out tests:` summary section
- Rerun mode creates a new log file named like `{input_stem}.rerun_{timestamp}.log`.
- If there are no tests to rerun, the script exits successfully.

## Collect-Only Mode

Use `--collect-only` with CSV mode, full-suite mode, or rerun-failed mode.

- No tests are executed.
- No log file is created.
- The script prints `Total tests: N`.
- When possible, it also prints a hierarchical count by test class.

Examples:

```bash
python run_tests.py my_tests.csv --pytorch-path /path/to/pytorch --collect-only
python run_tests.py --all-tests --pytorch-path /path/to/pytorch --collect-only
python run_tests.py --rerun-failed previous.log --pytorch-path /path/to/pytorch --collect-only
```

## Checkpointing And Resume

- By default, after each test or file batch, the script writes a checkpoint next to the log: `{log_file_path}.checkpoint`.
- The checkpoint stores the last test run, the next test to run, indices, mode, and source paths.
- Use `--resume` with the same log file path to continue from the next test.
- Resume appends to the existing log so previous results remain available for final analysis.
- If a checkpoint exists and you run without `--resume`, the script reports the checkpoint state, deletes it, and starts from the first test.
- Use `--no-checkpoint` to disable checkpoints.

## Test Outcome States

Each test is classified into exactly one state:

| State | Meaning |
|-------|---------|
| PASSED | Exit code 0 and output does not indicate a skip. |
| SKIPPED | Exit code 0 and output indicates the test was skipped. |
| ERROR | Non-zero exit and `RuntimeError` appears in stdout or stderr. |
| FAILED | Non-zero exit and no `RuntimeError` appears in output. |
| TIMEDOUT | The test or fallback test hit its timeout. |
| MISSED | File mode hit an outer timeout and could not identify the currently running test, so remaining nodes in that file were not run. |

## Log File Format

- Every run logs the PyTorch path and log file path.
- CSV and full-suite runs write `Mode: csv` or `Mode: full_suite`.
- Each recorded test has:
  - a progress line: `[N/TOTAL]`
  - a `Running: <test_name>` header
  - test output, when available
  - a status line: `PASSED`, `SKIPPED`, `ERROR`, `FAILED`, `TIMEDOUT`, or `MISSED`
- The final summary includes total run count, pass/skip/error/fail/timeout/missed counts, total time, and per-state test lists.
- `--rerun-failed` uses the `Failed tests:` and `Timed out tests:` summary sections.

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `csv_file` | One of CSV / full-suite / rerun-failed | Path to CSV with a `test_name` column. Omit when using `--all-tests` or `--rerun-failed`. |
| `--all-tests` | One of CSV / full-suite / rerun-failed | Discover and run tests in the configured full-suite file list. |
| `--include-inductor-all-tests` | No | Add PyTorch CI `inductor_core` test files from `--pytorch-path`; implies `--all-tests`. |
| `--include-inductor-tests` | No | Backward-compatible alias for `--include-inductor-all-tests`. |
| `--include-triton-nightly-inductor-tests` | No | Add ROCm torch-triton-nightly Inductor validation files; implies `--all-tests`. |
| `--rerun-failed LOG_FILE` | One of CSV / full-suite / rerun-failed | Rerun failed tests from a previous log. |
| `--rerun-include-timeouts` | No | With `--rerun-failed`, also rerun timed-out tests. |
| `--pytorch-path PATH` | Yes | Path to the PyTorch checkout. |
| `--log-file PATH` | No | Path for the run log. |
| `--stop-on-failure` | No | Stop after first failing test or fallback failure. |
| `--batch-mode {file,test}` | No | Full-suite execution granularity. Default: `file`. |
| `--per-file-timeout SECONDS` | No | Outer timeout for file subprocesses in `--batch-mode file`. Default: 1800. |
| `--per-test-timeout SECONDS` | No | Pytest-timeout per-test timeout. Default: 300. |
| `--resume` | No | Resume from the next test after the last checkpoint. |
| `--no-checkpoint` | No | Disable checkpoint writing and resume handling. |
| `--regex PATTERN` | No | Full-suite only. Filter discovered pytest node IDs by regex. |
| `-i`, `--input-files FILE [FILE ...]` | No | Full-suite only. Add files under `PYTORCH_PATH/test/`. |
| `--collect-only` | No | Count tests only; do not execute tests or write a log. |

## Examples

```bash
# Run a subset of tests from CSV with 120s timeout per test
python run_tests.py tests.csv --pytorch-path /path/to/pytorch --per-test-timeout 120

# Run full suite in default file mode
python run_tests.py --all-tests --pytorch-path /path/to/pytorch

# Run full suite with historical per-test-node subprocess behavior
python run_tests.py --all-tests --batch-mode test --pytorch-path /path/to/pytorch

# Run only tests whose full node id contains GPUTests
python run_tests.py --all-tests --pytorch-path /path/to/pytorch --regex GPUTests

# Run full suite from specific test files under test/
python run_tests.py --all-tests -i test_ops.py inductor/test_config.py --pytorch-path /path/to/pytorch

# Run the PyTorch CI inductor_core file set
python run_tests.py --include-inductor-all-tests --pytorch-path /path/to/pytorch

# Run the ROCm torch-triton-nightly Inductor validation subset
python run_tests.py --include-triton-nightly-inductor-tests --pytorch-path /path/to/pytorch

# Resume a previous full-suite run
python run_tests.py --all-tests --pytorch-path /path/to/pytorch --log-file full_run.log --resume

# Rerun only failed tests from a previous log
python run_tests.py --pytorch-path /path/to/pytorch --rerun-failed full_run.log

# Rerun failed and timed-out tests from a previous log
python run_tests.py --pytorch-path /path/to/pytorch --rerun-failed full_run.log --rerun-include-timeouts

# Count tests only
python run_tests.py --all-tests --pytorch-path /path/to/pytorch --collect-only
python run_tests.py tests.csv --pytorch-path /path/to/pytorch --collect-only
```
