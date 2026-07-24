# Run All Inductor Tests on New AMD GPU

## Goal

Run every registered PyTorch Inductor unit test from `/workspace/pytorch` on an
idle AMD GPU. Keep the run resilient in tmux, report the collected test total
as soon as discovery finishes, publish a preliminary ETA after ten minutes of
test execution, and maintain rolling progress in
[framework_scripts issue #5](https://github.com/naromero77amd/framework_scripts/issues/5).

## Fail-closed preflight

1. Verify `tmux`. If absent, install it with `apt-get` using root or `sudo`, then
   verify `tmux -V`. If installation is unavailable or denied, stop and prompt
   the user.
2. Verify the official `gh` CLI with `gh --version` and credentials with
   `gh auth status --hostname github.com`. If either check fails, stop and ask
   the user to install or authenticate it.
3. Validate read and comment access to issue #5. Create one initial progress
   comment and retain its comment ID so later checkpoints amend that comment.
   Never post tokens, environment dumps, or unrestricted raw logs.
4. Require the PyTorch checkout at `/workspace/pytorch` and the runner at
   `/home/niromero/docker_workspace/framework_scripts/pytorch/run_tests.py`.
   If either is absent, stop and prompt the user. Verify the runner's Python
   dependencies and ROCm configuration; do not install anything except the
   explicitly authorized `tmux` without asking.
5. Enumerate visible AMD devices and map the selected device to its AMD render
   node through `/sys/class/drm`. Inspect open file descriptors on `/dev/kfd`
   and the selected `/dev/dri/renderD*` with `fuser` or `/proc/*/fd`, not
   `rocm-smi`. Treat found PIDs, incomplete permission, ambiguous inspection,
   or a matching active tmux session as busy and stop with process details.
6. From `/tmp`, using the same Python environment and GPU visibility intended
   for the suite, run a HIP-backed PyTorch smoke test. Confirm
   `torch.version.hip`, `torch.cuda.is_available()`, and the GPU
   name/architecture; allocate two tensors on `cuda:0`, add them, synchronize,
   and verify the result. Stop and prompt the user on any failure.
7. Repeat the GPU-process check immediately before launching the suite.

## Tmux test run

- Create a uniquely named detached tmux session with `remain-on-exit` enabled
  and separate `tests` and `progress` windows. Never replace or kill an existing
  session.
- Preserve the selected GPU visibility variables for the smoke test and suite.
- In the `tests` window, run:

  ```bash
  python3 /home/niromero/docker_workspace/framework_scripts/pytorch/run_tests.py \
    --include-inductor-all-tests \
    --pytorch-path /workspace/pytorch \
    --batch-mode file \
    --num-gpus 1 \
    --retry-attempts 2 \
    --per-test-timeout 1200 \
    --per-file-timeout 43200 \
    --log-file <timestamped-log>
  ```

- `--include-inductor-all-tests` is required because plain `--all-tests` covers
  only the default Inductor file.
- Store timestamped log, checkpoint, metadata, analysis, monitor log, and exit
  status directly under `/workspace/pytorch`, allowing
  `framework_scripts/pytorch/analyze_inductor_run.py` to resolve the checkout.
- Record individual failures and continue the suite. Only infrastructure or
  preflight failures stop the run.

## Test count, ETA, and progress

1. Start the progress monitor before discovery. As soon as the runner emits
   `Found N test(s)`, prominently print the exact collected pytest-node count
   in the tmux progress window and amend issue #5. Do not substitute a test-file
   count.
2. Ten minutes after the first test begins, publish a preliminary ETA in the
   terminal and rolling GitHub comment. Base it on deduplicated completed
   pytest nodes from the live verbose log, falling back to checkpoint/analyzer
   counts and compatible prior full-suite timing data. Include estimated
   completion time, remaining duration, observed rate, sample size, and
   confidence. If only a timeout-derived bound is available, label it as a
   low-confidence range.
3. Every 1,800 seconds thereafter, print and post timestamp, elapsed time,
   completed/total count, percentage, state counts, current or next test, an
   updated ETA, and artifact paths. Use the runner log/checkpoint and
   `analyze_inductor_run.py`, supplementing its live file-batch view with
   deduplicated `pytest -vv` node results when needed.
4. Amend one rolling issue comment rather than creating repeated checkpoint
   comments. Sanitize and cap its body. Log transient GitHub failures and retry
   at the next checkpoint without terminating tests.
5. When the test wrapper writes its exit status, immediately publish the final
   terminal and GitHub update. Include final counts, duration, exit code,
   failed/error/timed-out/missed tests, resume information, tmux session, and
   local artifact paths.

## Execution checklist

- [ ] Plan committed and pushed.
- [ ] Preflight gates passed.
- [ ] Exact collected test total reported.
- [ ] Full Inductor suite running in tmux.
- [ ] Ten-minute ETA reported.
- [ ] Thirty-minute rolling checkpoints active.
- [ ] Final summary and artifact paths reported.
