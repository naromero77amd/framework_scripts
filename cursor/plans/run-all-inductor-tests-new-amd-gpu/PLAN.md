# Run All Inductor Tests on New AMD GPU

## Goal

Run every registered PyTorch Inductor unit test from `/workspace/pytorch` on an
idle AMD GPU against an immutable exact-node manifest. Keep the run resilient
behind one fail-closed wrapper, use tmux only as the durable user interface,
report the collected test total as soon as discovery finishes, publish
confidence-bounded progress, and maintain one rolling update in
the user-selected GitHub results issue.

## Outcome classification

- `P1`: `TIMED OUT` or `MISSED`. This is a reporting and triage label only.
- `P2`: `FAILED` or `ERROR`. This is the second-priority reporting and
  investigation label only.
- A suite inherits its highest contained priority: any timeout/miss makes it
  P1; otherwise any failure/error makes it P2. A suite with no unresolved
  outcome is left unprioritized.
- Non-failing outcomes are `PASSED`, `SKIPPED`, and `XFAILED`. Unresolved
  outcomes are `FAILED`, `ERROR`, `TIMED OUT`, and `MISSED`.
- Execute each included exact pytest node at most once. Preserve its single
  primary outcome; no P1, P2, timeout, miss, failure, or error reruns are part
  of this plan.

## Fail-closed preflight

1. Require `tmux`, the official `gh` CLI, and authenticated GitHub access.
   Do not install or authenticate tools without explicit user approval.
2. Before inspecting, installing, replacing, or building Triton, ask the user
   this blocking question: **Use the Docker image's default Triton, or use a
   special Triton version?** Do not infer a choice from the current environment
   or continue until the user answers.
   - For the Docker-image default, do not modify Triton. Record the imported
     package path, version, commit, and LLVM hash from `/opt/venv/bin/python`,
     and verify they match the untouched image installation.
   - For a special version, ask for its exact source (wheel, checkout, build,
     or image), ref/commit, expected LLVM hash, activation or installation
     instructions, patches, and required environment variables. Obtain explicit
     approval before mutating the environment and fail closed if provenance is
     incomplete or the installed result does not match the requested version.
   - The canonical Triton source-build helper is
     `/home/niromero/docker_workspace/framework_scripts/triton/build.sh`.
     Never invoke it for the Docker-image default. If the approved special
     version requires a source build, require this script, record its SHA-256,
     run it only from the user-approved Triton checkout/ref, and make
     `/opt/venv/bin` first in `PATH` because the script calls unqualified
     `python` and `pip`. The script uninstalls existing Triton packages, so
     obtain explicit mutation approval immediately before execution.
   - After any special build, verify `pip show triton`, the imported module
     path/version/commit, and LLVM hash with `/opt/venv/bin/python` before
     continuing.
   - Persist `TRITON_SELECTION=default|special` and all resolved provenance in
     the environment fingerprint, preflight artifact, and final report.
3. Ask the user this blocking question: **What GitHub issue should store this
   run's progress and final results?** Require a full issue URL or an explicit
   `<owner>/<repository>#<number>` value. Do not assume issue #5 or reuse a
   destination from an earlier campaign. Resolve and record the canonical URL,
   repository, issue number, title, and state as `RESULTS_ISSUE_URL` metadata.
   If the issue is closed, require explicit confirmation before using it.
4. Ask the user this blocking question: **Should tests run continuously 24/7,
   or only daily from midnight through 10:00 AM Central Time?** Do not infer a
   schedule. Persist the answer as `RUN_SCHEDULE=continuous` or
   `RUN_SCHEDULE=central_midnight_to_10am`.
   - Interpret Central Time with the IANA zone `America/Chicago`, including
     daylight-saving transitions; do not use a fixed UTC offset.
   - For the daily window, permit test processes only while local time is
     `00:00 <= time < 10:00`. The tmux wrapper may remain idle outside the
     window, but runner, pytest, and compile-worker processes may not.
5. Require the PyTorch checkout at `/workspace/pytorch`, the runner at
   `/home/niromero/docker_workspace/framework_scripts/pytorch/run_tests.py`,
   and `/opt/venv/bin/python`. Record the exact Python, PyTorch, ROCm/HIP,
   Triton, and optional-package versions. Validate required optional imports,
   including Transformers model classes used by `test_deterministic.py`;
   never rely on a test to install packages during execution.
6. Record the PyTorch and Triton commits, dirty-worktree state, Triton LLVM
   hash, GPU architecture, and effective scheduler settings. Require and record
   `HSA_HOTSWAP_ENABLE=1`. Fail if any value differs from the approved run
   configuration.
7. Create a unique run ID and artifact directory outside the PyTorch checkout.
   Refuse to overwrite an existing state, log, metadata, exit, or cache path.
   Use separate empty preflight and execution `TORCHINDUCTOR_CACHE_DIR` and
   `TRITON_CACHE_DIR` paths.
8. Enumerate visible AMD devices and map the selected device to its AMD render
   node through `/sys/class/drm`. Inspect open file descriptors on `/dev/kfd`
   and the selected `/dev/dri/renderD*` with `fuser` or `/proc/*/fd`, not
   `rocm-smi`. Treat found PIDs, incomplete permission, ambiguous inspection,
   or a matching active tmux session as busy and stop with process details.
9. Discover every registered Inductor node before execution using the same
   checkout, Python environment, visibility, and pytest configuration as the
   primary run. Write a CSV manifest containing exact node ID and source suite;
   reject duplicate IDs, record the collected count and SHA-256, and preserve
   the manifest unchanged for execution and resume. Obtain exact IDs from the
   runner's `discover_tests()` path; its current `--collect-only` output reports
   counts but is not itself an exact-node manifest.
10. From `/tmp`, using the same Python environment and GPU visibility intended
   for the suite, run a HIP-backed PyTorch smoke test. Confirm
   `torch.version.hip`, `torch.cuda.is_available()`, and the GPU
   name/architecture; allocate two tensors on `cuda:0`, add them, synchronize,
   and verify the result. Stop and prompt the user on any failure.
11. Write an atomic preflight artifact containing explicit pass/fail records for
   manifest, collection, environment, optional dependencies, cache isolation,
   started-node journaling, single-attempt resume guards, and GPU smoke checks.
12. Validate read, comment, and description-edit access to the selected results
    issue only after local preflight passes. Confirm its canonical URL still
    matches `RESULTS_ISSUE_URL`. Create one rolling progress comment after the
    runner starts and retain its comment ID. Never post tokens, environment
    dumps, or unrestricted raw logs.
13. Repeat the GPU-process check immediately before launching the suite.

## Wrapper and tmux test run

- Create a uniquely named detached tmux session with `remain-on-exit` enabled
  and separate `tests` and `progress` windows. Never replace or kill an existing
  session. Tmux is only the durable interface; one wrapper owns runner launch,
  monitoring, progress, metadata, and final exit classification.
- Preserve the complete validated environment from manifest collection and GPU
  smoke through execution. Record unset variables explicitly in metadata.
- Export and verify `HSA_HOTSWAP_ENABLE=1` before manifest collection, GPU
  smoke, the primary `run_tests.py` invocation, and every resume. Never rely on
  an inherited shell value without recording the effective value.
- The wrapper must launch the runner in its own process group with `setsid` and
  keep the safety monitor outside that group.
- In the `tests` window, the wrapper runs:

  ```bash
  export HSA_HOTSWAP_ENABLE=1

  /opt/venv/bin/python \
    /home/niromero/docker_workspace/framework_scripts/pytorch/run_tests.py \
    --include-inductor-all-tests \
    --pytorch-path /workspace/pytorch \
    --batch-mode file \
    --num-gpus 1 \
    --retry-attempts 0 \
    --per-file-timeout 43200 \
    --log-file <timestamped-log>
  ```

- Omit `--per-test-timeout` from the primary command so the runner's current
  default applies. Record the effective default in metadata. Every node receives
  one primary attempt with no automatic retries.
- `--include-inductor-all-tests` is required because plain `--all-tests` covers
  only the default Inductor file.
- File batching remains the primary throughput mode. During a long file batch,
  live progress comes from deduplicated `pytest -vv` node records because the
  runner's final per-node records can lag until a batch exits.
- Store the manifest, preflight log, launcher log, runner state, console log,
  checkpoint, metadata, monitor log/state, mandatory-stop JSON, exit JSON,
  analysis, and generated summaries under the dedicated run directory.
- Before launch, extend `framework_scripts/pytorch/analyze_inductor_run.py` to
  accept the PyTorch root explicitly (and explicit state/metadata/log/checkpoint
  paths where present), or make it honor equivalent metadata keys. Its current
  `log_path.parent` checkout inference is incompatible with the required
  external artifact directory; treat that mismatch as a preflight failure.
- Record individual failures and continue the suite. Only infrastructure or
  preflight failures and the mandatory safety conditions below stop the run.
- Treat runner exit `1` with monitor exit `0` and a passing post-run smoke test
  as `completed_with_test_failures`, not an infrastructure failure.
- Resume only with `--resume`, the same log path and flags, and matching
  manifest/environment hashes. Without `--resume`, the runner can truncate the
  log and discard the checkpoint. Final analysis after resume must parse the
  full accumulated log/state rather than trusting the final process segment's
  summary or exit code.

### Schedule enforcement

- For `RUN_SCHEDULE=continuous`, launch immediately after preflight and continue
  until completion or a mandatory health stop.
- For `RUN_SCHEDULE=central_midnight_to_10am`, the wrapper must derive every
  boundary with `America/Chicago`, wait outside the daily window without test or
  compile processes, and launch or resume only after midnight Central.
- At 10:00 AM Central, pause the campaign at a recoverable boundary and classify
  the wrapper outcome as `scheduled_pause`, not completion or infrastructure
  failure. Persist the active-node attribution, checkpoint, log offset, and
  pause metadata, then update the rolling GitHub comment.
- Validate deadline-aware pause and resume before the real run. A blind process
  kill that loses completed-node attribution or causes automatic re-execution
  does not satisfy the one-attempt primary policy.
- At the next midnight boundary, repeat GPU-idle inspection, environment and
  manifest hash checks, and the GPU smoke test before resuming with the same
  flags and log path plus `--resume`.

## Test count, ETA, and progress

1. Prominently print the immutable manifest's exact pytest-node count before
   launch and confirm that the runner's `Found N test(s)` count matches it. Do
   not substitute a test-file count.
2. Ten minutes after the first test begins, publish only a low-confidence ETA
   range. Base live progress on deduplicated `pytest -vv` node results, falling
   back to atomic runner state/checkpoint and compatible historical suite
   timing.
3. Do not publish a point ETA until at least 30 minutes have elapsed and either
   three suites or 500 exact nodes have completed. Use a suite-weighted model,
   not a flat early-run rate. Report:
   - a fast estimate based on observed non-timeout durations;
   - a conservative estimate that prices pending timeout candidates at their
     configured timeout;
   - estimated completion time, remaining duration, sample size, and confidence.
   For the Central-time window schedule, report both remaining active test time
   and wall-clock completion time after accounting for the daily off-hours.
4. Every 1,800 seconds thereafter, print and post timestamp, elapsed time,
   completed/total count, percentage, state counts, current or next test, an
   updated ETA range, schedule state/next boundary, and artifact paths. Report
   active-test and wall-clock elapsed time separately. Persist each prediction
   so it can be compared with the actual completion time for future calibration.
5. Amend one rolling issue comment rather than creating repeated checkpoint
   comments. Sanitize and cap its body. Log transient GitHub failures and retry
   at the next checkpoint without terminating tests.
6. When the test wrapper writes its exit status, immediately publish the final
   terminal and rolling-comment update. Include completion classification,
   counts, duration, runner/monitor exit codes, post-run GPU smoke result,
   resume information, tmux session, and local artifact paths. Update the issue
   description only after the single primary pass and exact-node reconciliation
   complete.

## Safety monitoring and mandatory stop policy

Run a dedicated safety monitor beside the full-suite runner. The
runner must start in its own process group with `setsid`; the monitor remains
outside that group so it can terminate the complete runner/pytest/compile-worker
tree. The wrapper must treat a nonzero monitor exit as a mandatory stop, record
wrapper exit `5`, and require GPU-health investigation before continuation.
Use a monitor configuration or implementation that enforces only the mandatory
health stops listed below, and verify that behavior before launch.

### Log and state handling

1. Tail only bytes appended after the current launch offset. On resume, record
   the existing log size before starting the runner so old crash text cannot
   retrigger a stop.
2. Track the active pytest node and final per-node result for stop attribution
   and reporting.
3. Persist monitor state atomically after every detected event and completed
   result. Keep the policy version, active node, health-event history, and last
   completed test.
4. Write a separate mandatory-stop JSON artifact containing the reason,
   triggering detail, configured health thresholds, active node, log path, and
   runner PID. Preserve it with the runner log and checkpoint.
5. Ignore configuration/header text when matching errors. In particular, do
   not treat a line beginning `Stop message after current shard:` as an
   invalid-device failure merely because it quotes the configured stop text.

### Mandatory stops

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

## Architecture-specific hard-hang quarantine

- Begin every campaign with an empty hard-hang quarantine. Do not import,
  inherit, or automatically apply any unit-test quarantine from a prior run,
  host, or GPU architecture.
- Add a node only when the current campaign produces direct evidence of an
  unrecoverable hard hang or GPU loss, and require explicit user approval
  before excluding it from a resume. Record the exact node ID, architecture,
  timestamp, process state, recovery actions, and approval in run metadata.
- Apply a current-campaign approved exclusion to primary-run resumes with
  `PYTEST_ADDOPTS=--deselect=<node-id>` (preserving any existing
  `PYTEST_ADDOPTS`). Record the node as an intentional unresolved exclusion in
  the final report. Do not execute it again.
- Keep approved exclusions only in a run-local manifest under the artifact
  directory. Never depend on a historical manifest path.

## Single-pass completion policy

1. Execute the immutable full-suite manifest once. Each exact pytest node may
   be started at most once during this campaign.
2. Keep `--retry-attempts 0`. Do not invoke `--rerun-failed`, create targeted
   rerun manifests, or launch a second attempt for a passed, skipped, xfailed,
   failed, errored, timed-out, missed, interrupted, or quarantined node.
3. File-batch recovery may continue with exact nodes that have never started,
   but it must not repeat a node from the interrupted batch. Persist both
   started-node and terminal-result journals so this invariant can be proved.
4. A schedule pause, mandatory health stop, or process interruption may resume
   only with never-started exact nodes. If the wrapper cannot prove that a node
   never started, classify it as `MISSED` and skip it rather than execute it
   again.
5. After the primary pass, build one exact-node result map. Classify
   `TIMED OUT`/`MISSED` outcomes as P1 and `FAILED`/`ERROR` outcomes as P2 for
   reporting only. Preserve those outcomes without further execution.
6. Publish final totals only after every included manifest node has exactly one
   terminal outcome or an explicit `MISSED`/approved-exclusion classification.

## Final GitHub issue description

After the single full-suite pass and exact-node reconciliation finish, replace
the description of the user-selected `RESULTS_ISSUE_URL` with a durable report
modeled on
[the current framework_scripts issue #5 description](https://github.com/naromero77amd/framework_scripts/issues/5#issue-4971129324).
Use that description's presentation and section structure as the template, but
regenerate every fact from this campaign. Do not copy its stale cross-GPU
comparison rules or reintroduce a P3 category.

1. Wrap generated sections in stable HTML markers. Start with a dated
   `<gpu-architecture> PyTorch Inductor Outcome` heading and an
   `> [!IMPORTANT]` completeness callout. State completed/planned/pending
   coverage and all intentional exclusions. If work remains, label the report
   as a partial single-pass outcome rather than a completed final result.
2. Add `### Overall Result` with:
   - discovered, excluded, and included exact-node counts;
   - explicit `PASSED` outcomes;
   - latest non-failing rate,
     `(passed + skipped + xfailed) / included`;
   - unresolved rate,
     `(failed + error + timedout + missed) / included`; and
   - the rule that each exact node retains its one primary outcome.
   Include a baseline rate or `(was ...)` comparison only when its scope and
   denominator are exactly comparable. The latest non-failing and unresolved
   rates must reconcile to 100%.
3. Add `### Suite Summary` with one row per Inductor file and a final included
   total row. Include priority, total, passed, skipped, xfailed, failed, error,
   timed out, and missed columns. Keep intentional exclusions visible as
   struck-through rows but omit them from included totals. Use the reference
   description's durable markers and bold suite names: 🔴 **P1** for any suite
   containing timeout/miss, 🟡 **P2** for any suite containing failure/error,
   and blank for a clean suite.
4. Add `## Execution, Improvement, and Provenance Notes` with subsections for
   the completed single primary run, any exactly comparable improvements or
   regressions, exact result provenance, and unresolved-outcome interpretation.
   Include the primary manifest hash, coverage and pending counts,
   batching/default-timeout/zero-retry settings, source artifacts, transition
   counts, scheduled pauses/resumes, active-test and wall-clock durations,
   final stop reason, and post-run process/GPU health. Do not imply causality
   when multiple software or environment variables changed.
5. Add `## Environment` with the container/image identity, GPU model and
   architecture, Python, PyTorch and ROCm/HIP versions, optional dependency
   versions, selected run schedule and timezone, scheduler and visibility
   overrides, and cache policy. Report the user-selected Triton mode explicitly
   as **Docker-image default** or **special**, followed by the imported package
   path, version, commit, LLVM hash, source/build provenance, patches, and
   relevant environment variables. If the Triton build helper was used, include
   its path, SHA-256, working checkout/ref, and verification result.
6. Add `## Failure Clusters and Current Triage`. Group current P1 timeout/miss
   nodes by signature, explain `MISSED` attribution, and place the complete
   exact P1 node list in a collapsed `<details>` block. Follow it with a
   `### Priority 2 queue` listing current failed/error suites and counts. Use
   only the current campaign's P1/P2 definitions; do not add P3 or cross-device
   filtering.
7. Add reproducibility details containing the exact workspace/container setup,
   runner commit and uncommitted diff, selected Triton provenance, environment
   overrides, any Triton build-helper invocation, the primary command,
   exclusions, resume history, and local artifact paths. Avoid secrets and
   unrestricted raw logs.
8. End with the prominent note from the reference description that comments
   below the description are intermediate Cursor checkpoints and can be
   ignored.
9. Generate the report from the primary manifest, logs, checkpoints, state,
   metadata, and analysis. Confirm from the started-node and terminal-result
   journals that no exact node had more than one attempt. Ensure every
   discovered node is represented exactly once or explicitly classified as
   missed/excluded, then validate every suite row, priority, transition, and
   aggregate formula. Do not hardcode expected counts.
10. Update the issue description, not a comment, with
    `gh issue edit <number> --repo <owner>/<repository> --body-file <path>` or
    the GitHub API, using only the destination resolved from
    `RESULTS_ISSUE_URL`. Preserve content outside the generated markers, then
    read the issue back and verify its canonical URL, marker boundaries,
    headings, totals, formulas, priorities, links, and Triton provenance.

## Execution checklist

- [ ] Plan committed and pushed.
- [ ] User selected the GitHub results issue; canonical URL and edit access
      recorded and verified.
- [ ] User selected Docker-image default or special Triton; exact provenance
      recorded and verified.
- [ ] If a special source build was selected, the canonical Triton build script,
      checksum, checkout/ref, mutation approval, and installed result verified.
- [ ] User selected continuous 24/7 or midnight-to-10:00-AM Central execution;
      schedule and `America/Chicago` boundaries recorded and verified.
- [ ] `HSA_HOTSWAP_ENABLE=1` exported and verified for discovery, smoke tests,
      primary execution, and resumes.
- [ ] Preflight gates passed.
- [ ] Immutable exact-node manifest, count, and SHA-256 recorded.
- [ ] Full Inductor suite running in tmux.
- [ ] Ten-minute ETA reported.
- [ ] Thirty-minute rolling checkpoints active.
- [ ] Dedicated safety monitor active for invalid-device errors, persistent
      D-state, GPU disappearance, and monitor failure.
- [ ] Mandatory-stop state and post-stop GPU-health procedure verified.
- [ ] Primary run completed with zero automatic retries.
- [ ] Scheduled pauses, if selected, preserved exact-node attribution and
      resumed only never-started nodes after next-window preflight.
- [ ] Started-node journal proves every included node had at most one attempt;
      no rerun command or targeted rerun manifest was created.
- [ ] P1 timeout/miss and P2 failure/error queues generated from primary
      outcomes for reporting only.
- [ ] Final summary and artifact paths reported.
- [ ] Final outcome includes completeness, reconciled rate formulas, suite
      totals, P1/P2 labels, provenance, stop reason, GPU health, and unresolved
      clusters.
- [ ] Final suite report follows the reference issue #5 description structure,
      includes Triton selection/provenance, and is published to the selected
      results issue and read-back verified.
