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
  preflight failures and the mandatory safety conditions below stop the run.

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

## Safety monitoring and mandatory stop policy

Run a dedicated safety monitor beside every full-suite or targeted runner. The
runner must start in its own process group with `setsid`; the monitor remains
outside that group so it can terminate the complete runner/pytest/compile-worker
tree. The wrapper must treat a nonzero monitor exit as a mandatory stop, record
runner exit `5`, and require GPU-health investigation before continuation.

### Log and state handling

1. Tail only bytes appended after the current launch offset. On resume, record
   the existing log size before starting the runner so old crash text cannot
   retrigger a stop.
2. Track the active pytest node from the verbose log, but update adjacency only
   from the runner's final per-node result records (`Running: <node>` followed
   by passed, skipped, xfailed, failed, error, timed-out, or missed status).
3. Persist monitor state atomically after every detected event and completed
   result. Keep the policy version, current adjacent-SIGABRT nodes, pending
   SIGABRT nodes, current adjacent-timeout nodes, pending timeout nodes, event
   history, and last completed test. Use one state file across suite boundaries
   because a wrapper boundary is not an executed test.
4. Write a separate mandatory-stop JSON artifact containing the reason,
   triggering detail, configured limits, current nodes, log path, and runner
   PID. Preserve it with the runner log and checkpoint.
5. Ignore configuration/header text when matching errors. In particular, do
   not treat a line beginning `Stop message after current shard:` as an
   invalid-device failure merely because it quotes the configured stop text.

### Five-adjacent-test SIGABRT rule

1. Detect SIGABRT from `Fatal Python error: Aborted` or process exit code `-6`
   and associate it with the active logical pytest node.
2. An initial batch crash and all fresh-process retry aborts for the same node
   count as one test, not multiple adjacent tests. Mark the node as pending
   when the signal is seen and add it to the streak only when the runner emits
   that node's final result.
3. A node counts as a SIGABRT test if any of its attempts emitted SIGABRT, even
   if a fresh-process retry later produced a terminal result.
4. Increment the streak only when the next completed logical test is a
   SIGABRT-producing test. Any intervening completed test without SIGABRT
   resets the streak to zero, regardless of whether its outcome is passed,
   skipped, xfailed, normally failed, error, timed out, or missed.
5. Stop immediately when the fifth adjacent executed test is finalized as a
   SIGABRT-producing test. This is not a cumulative count across the run.
   Version/reset state created by older event-based policies so their counts
   cannot carry into this adjacency rule.

### Three-adjacent-test timeout rule

1. Detect timeout-producing tests from the runner's file/test timeout markers
   and associate each marker with the active logical pytest node.
2. Treat repeated timeout markers and fresh-process retries for one node as one
   timeout-producing test. Mark the node pending when detected and add it to
   the timeout streak only when its final per-node result is emitted.
3. Increment the timeout streak only when the next completed logical test
   produced a timeout. Any intervening completed non-timeout test resets the
   timeout streak to zero, regardless of its terminal outcome.
4. Stop immediately when the third adjacent executed test is finalized as a
   timeout-producing test. This is not a cumulative distinct-node count across
   the run. Version/reset timeout state created by the older event-based policy
   so it cannot carry into this adjacency rule.

### Other mandatory stops

- Stop immediately on `CUDA error: invalid device function` or
  `hipErrorInvalidDeviceFunction`.
- Poll relevant runner, pytest, and compile processes for uninterruptible
  `D` state. Stop if a matching process remains in `D` state for 60 seconds.
- Check AMD GPU visibility at least every 30 seconds, with a bounded command
  timeout. Stop if `amd-smi list --json` fails or reports no GPU.
- If the monitor itself fails, the wrapper must stop instead of allowing an
  unmonitored test run to continue.

### Stop and recovery procedure

1. On a mandatory stop, atomically write the stop artifact, send `SIGTERM` to
   the runner process group, wait up to five seconds, then send `SIGKILL` to
   the same group if it still exists.
2. Confirm no runner, pytest, or compile-worker process survived. Inspect
   `/dev/kfd` and the selected render node for users and verify that AMD tooling
   still sees the selected GPU.
3. Run the same tensor-add correctness smoke test used in preflight. Do not
   resume unless the GPU is visible, idle, and the smoke test passes.
4. Resume from the persisted runner checkpoint/state and a new log offset;
   never discard already committed per-node outcomes.

## Known hard-hang exclusion

- Never rerun
  `test/inductor/test_mix_order_reduction.py::MixOrderReductionTest::test_layer_norm_bwd_with_dynamic_shape_dynamic_dims2`
  on this `gfx1250` system. On 2026-07-29 it wedged the GPU/kernel path; neither
  `SIGTERM` nor `SIGKILL` could terminate the pytest process or its D-state
  worker.
- Never rerun
  `test/inductor/test_flex_attention.py::TestFlexAttentionCUDA::test_builtin_score_mods_different_block_size_score_mod6_BLOCK_SIZE3_cuda_float16`
  on this system. It was active when the GPU disappeared and required a host
  driver reload on 2026-07-30.
- Apply the exclusion to primary-run resumes and missed-test reruns with
  `PYTEST_ADDOPTS=--deselect=<node-id>` (preserving any existing
  `PYTEST_ADDOPTS`). Record the node as an intentional unresolved exclusion in
  the final report, not as a test to retry.
- The run-local exclusion manifest is
  `/workspace/pytorch/inductor_all_gfx1250_20260724_210412.skip_nodes.txt`.

## Mandatory missed-test rerun

1. Let the complete primary suite finish. Then parse its log and checkpoint,
   extract every test whose final state is `MISSED`, and deduplicate the list by
   exact pytest node ID. These are attribution gaps from interrupted file
   batches, not confirmed test failures.
2. Rerun only those node IDs in file-scoped shards of at most `50` tests. Use
   the existing runner's full-suite shard path by grouping missed IDs by source
   file and invoking the runner with `--all-tests`, `-i <file>`,
   `--regex <anchored-exact-node-regex>`, `--batch-mode shard`, and
   `--shard-size 50`. Split very large exact-node regexes into
   multiple invocations if necessary to remain below command-line limits.
3. Do not use `--rerun-failed` for this step: the current runner does not select
   `MISSED` entries in that mode, and rerun mode ignores shard batching.
4. Keep separate timestamped rerun logs, checkpoints, metadata, and analyses.
   Preserve the original GPU visibility, environment, retry count, and timeout
   settings.
5. If a shard rerun still produces missed nodes, repeat the shard-size-50 pass
   for the remaining exact node IDs until no misses remain or a complete round
   makes no progress. If no progress is possible, stop the rerun loop and
   clearly report the unresolved nodes and their crash/process signatures.
6. Merge rerun outcomes over the primary results by exact node ID, with the
   latest completed rerun result taking precedence. Do not publish the final
   suite totals or issue description until this merge is complete.
7. Remove every node in the known hard-hang exclusion manifest from all rerun
   rounds.

## Final GitHub issue description

After the complete run and mandatory missed-test reruns finish, replace the
description of [framework_scripts issue #5](https://github.com/naromero77amd/framework_scripts/issues/5)
with a durable report modeled on
[ROCm/frameworks-internal issue #17237](https://github.com/ROCm/frameworks-internal/issues/17237#issue-4850446855):

1. Add an `inductor-suite-summary` section, delimited by HTML comments, with one
   row per Inductor test file and a final total row. Include total, passed,
   skipped, xfailed, failed, error, timed out, missed, and total recorded time.
2. Add a `repro-instructions` section containing the exact container/workspace
   setup, PyTorch/Triton/ROCm versions and commits, GPU architecture, runner
   commit, any uncommitted runner diff used by this run, environment overrides,
   and the exact test command.
3. Add an `inductor-suite-notes` section that explains the interpretation of
   pre-existing failures, all reruns or interrupted work, the causes of
   `MISSED` entries, and common failure patterns grouped by suite and signature.
   Distinguish correctness failures, unsupported `gfx1250` behavior,
   crashes/signals, timeouts, and test-infrastructure failures.
4. Add the note that issue comments below the description are intermediate
   Cursor checkpoints and can be ignored.
5. Generate the report from the completed log, checkpoint, analysis, and any
   rerun artifacts. Deduplicate tests by pytest node ID and let the latest rerun
   result override an earlier result. Confirm every discovered test is
   represented exactly once or explicitly classified as missed.
6. Update the issue description, not a comment, using `gh issue edit --body-file`
   or the GitHub API. Preserve any content outside generated HTML markers, avoid
   secrets and unrestricted raw logs, then read the issue back to verify the
   published body and totals.

## Execution checklist

- [ ] Plan committed and pushed.
- [ ] Preflight gates passed.
- [ ] Exact collected test total reported.
- [ ] Full Inductor suite running in tmux.
- [ ] Ten-minute ETA reported.
- [ ] Thirty-minute rolling checkpoints active.
- [ ] Dedicated safety monitor active with five-adjacent-SIGABRT and
      three-adjacent-timeout policies.
- [ ] Mandatory-stop state and post-stop GPU-health procedure verified.
- [ ] All missed tests rerun in shards of 50 and merged into final results.
- [ ] Final summary and artifact paths reported.
- [ ] Final suite report published and verified in the issue description.
