---
name: PAAS Priority Optimization
overview: Use the tested PAAS fused-kernel pipeline to tune the 72-kernel representative subset listed in inductor/kernels/priority/priority_representative_kernel_paths.txt on the host’s gfx1250 GPU. Preserve the full 1,037-kernel source corpus, retarget only the staged subset, use one timing replica, and keep path-aware results.
todos:
  - id: campaign-control
    content: Start the 72-kernel subset campaign and initialize timing, logs, and progress telemetry
    status: pending
  - id: paas-portability
    content: Add and test gfx1250-aware PAAS hardware-limit handling
    status: completed
  - id: campaign-inputs
    content: Stage the exact 72 paths and prepare a target-specific manifested copy
    status: pending
  - id: pilot
    content: Validate the combined PAAS branch on pointwise, reduction, and persistent-reduction subset kernels
    status: pending
  - id: baseline-subset
    content: Benchmark the 72 selected inputs and capture gfx1250 launch parameters
    status: pending
  - id: tune-subset
    content: Run resumable one-replica tuning for every eligible selected kernel
    status: pending
  - id: report
    content: Generate path-aware winner, coverage, provenance, and failure reports
    status: pending
isProject: false
---

# Optimize the Representative Priority-Kernel Subset with PAAS

## Confirmed understanding and scope

PAAS lowers this corpus through five stages: benchmark the original Inductor wrapper, capture its launch parameters, generate bare `@triton.jit` tune scripts, search target-legal block/launch configurations under the selected policy, and summarize the valid rows. The relevant implementation is in [bench_kernels.py](/home/niromero/docker_workspace/inductor-triton-hacks/paas/processing/bench_kernels.py), [create_standalone_kernel.py](/home/niromero/docker_workspace/inductor-triton-hacks/paas/kernel_generator/create_standalone_kernel.py), [manager.py](/home/niromero/docker_workspace/inductor-triton-hacks/paas/tuner/simple/manager.py), and [organizer.py](/home/niromero/docker_workspace/inductor-triton-hacks/paas/downstream/organizer.py).

The source tree contains 1,037 runnable fused kernels: 624 pointwise, 284 reduction, and 129 persistent-reduction instances. This campaign will tune only the 72 exact paths in [priority_representative_kernel_paths.txt](kernels/priority/priority_representative_kernel_paths.txt): 27 pointwise, 20 reduction, and 25 persistent-reduction kernels. The list was selected for body, memory-access, reduction-regime, and geometry diversity; model-balanced duplicates are intentionally excluded. There are no templated GEMM/conv/flex kernels in the selected set.

```mermaid
flowchart LR
  Inputs["1,037-kernel source corpus"] --> Select["Stage 72 exact paths"]
  Select --> Normalize["PAAS target copy for gfx1250"]
  Normalize --> Baseline["PAAS Inductor baseline and launch capture"]
  Baseline --> Generate["Generate selected tune scripts by populated leaf"]
  Generate --> Search["One-replica capped search"]
  Search --> LeafReports["Per-leaf PAAS summaries"]
  LeafReports --> GlobalReport["Path-aware campaign report"]
```

## 1. Pin the environment and isolate the campaign

- Start all implementation and tuning work inside a named tmux session, `paas-priority-gfx1250`, before installing PAAS or creating campaign artifacts. Use a driver window for the orchestration process and a monitor window for progress; pipe both panes to durable logs. The run must remain resumable after an IDE, shell, SSH, or agent interruption, and `tmux attach -t paas-priority-gfx1250` must restore the live terminal view.
- Record `campaign_started_at` immediately when the tmux driver begins and `campaign_finished_at` only after final coverage/report generation. Use a monotonic timer for elapsed durations and UTC timestamps for auditability. Track total end-to-end wall time, each stage's time, and each model's tuning time in a machine-readable `campaign_state.json`.
- Record the clean repository revisions: `inductor-triton-hacks` branch `paas-target-portability-block-cap` (latest `origin/main` plus the reduction-cap and target-portability commits) and `Triton_Conv_Development/gemm-hf-branch` at `2d7f07e5...`.
- Install the local internal PAAS checkout editable; do not install the public `paas` URL from [requirements.txt](/home/niromero/docker_workspace/Triton_Conv_Development/requirements.txt).
- Verify the active stack before spending GPU time: PyTorch `2.13.0+rocm10.1...`, Triton `3.8.0`, one gfx1250 GPU, and the previously selected pinned prebuilt LLVM rather than `llvm-project-gfx1250`.
- Create a new resumable subset output root outside both repositories, for example `/home/niromero/docker_workspace/paas_runs/priority-representative-gfx1250/`. Do not reuse the earlier 1,037-kernel campaign output.
- Stage only the exact paths in [priority_representative_kernel_paths.txt](kernels/priority/priority_representative_kernel_paths.txt), preserving the model/category hierarchy:

```bash
SOURCE=/home/niromero/docker_workspace/Triton_Conv_Development/kernels/priority
LIST=/home/niromero/docker_workspace/framework_scripts/inductor/kernels/priority/priority_representative_kernel_paths.txt
STAGED=/home/niromero/docker_workspace/paas_runs/priority-representative-gfx1250/foreign-target

python -m paas.helpers.search_kernels \
  --dir "$SOURCE" \
  --out "$STAGED" \
  --names "$LIST"
```

- Assert that the staged identity set exactly equals the list: 72 unique files, split 27 pointwise / 20 reduction / 25 persistent reduction. An absent path, unlisted kernel, duplicate path, or flattened filename is a hard failure.
- Use a campaign-specific `TORCHINDUCTOR_CACHE_DIR`, set `HIP_VISIBLE_DEVICES=0` explicitly because [manager.py](/home/niromero/docker_workspace/inductor-triton-hacks/paas/tuner/simple/manager.py) otherwise exposes no tuner GPU, and disable GPU core dumps without using blanket `killall` commands.
- Emit an hourly progress line in the tmux monitor window and `progress.log` containing the current stage/model/kernel, completed/failed/total kernels, tested/estimated configurations, total elapsed time, processing rate, and ETA. Surface the same hourly snapshot as a progress update while the campaign is actively monitored.

## 2. Make the run valid for gfx1250

The captures embed gfx942 properties (`80` CUs, wave64), while this host is gfx1250 (`256` CUs, wave32). Unmodified Inductor uses the embedded `cc` as the Triton compile target, so the raw wrappers are not valid inputs on this host.

- Use the target-portability support in the combined PAAS branch rather than a campaign-local rewrite. `paas-inductor --retarget-output-dir` must create a non-destructive target copy, replace the complete embedded `DeviceProperties(...)` object from logical GPU 0, omit stale launch sidecars, and record source/target hashes in `paas_portability_manifest.json`.
- Require the prepared manifest and generated derivative provenance to agree with the active gfx1250/wave32 device before tuning. PAAS must pass the live wave size and thread limits through `genConfigs()` and `KernelOracle.validate_config()`; do not fall back to the captured wave64 assumptions.
- This target-portability change intentionally does not patch PyTorch/Inductor API-version differences. Record such kernels as explicit compatibility failures rather than modifying their Triton bodies in this campaign.
- Use the correct test layers:
  - Extend [test_paas-tuning-fixes.py](/home/niromero/docker_workspace/inductor-triton-hacks/test/test_paas-tuning-fixes.py), which directly tests `compat.genConfigs()`, `KernelOracle.validate_config()`, `_check_max_grid_x()`, compiled-config inclusion, pinning, and runner failure/restart behavior. The filename is `test_paas-tuning-fixes.py`, not `test_pass-tuning-fixes.py`.
  - Update/run [test_paas-optimizer.py](/home/niromero/docker_workspace/inductor-triton-hacks/test/test_paas-optimizer.py), the optimizer test explicitly documented by [README.md](/home/niromero/docker_workspace/inductor-triton-hacks/README.md), to cover the `paas-simple-full` wiring after device properties are added.
  - Run [test_paas-inductor.py](/home/niromero/docker_workspace/inductor-triton-hacks/test/test_paas-inductor.py) and [test_paas-make-standalone.py](/home/niromero/docker_workspace/inductor-triton-hacks/test/test_paas-make-standalone.py) as the integration gates for baseline/launch capture and tune-script generation.
- Do not use `paas-full`: README places it in the outdated/abandonware section. The campaign remains on the documented `paas-simple-full` path.

## 3. Run an architecture and workflow pilot

- Select one pointwise, one reduction, and one persistent-reduction kernel from the 72-path list for a workflow pilot; do not add pilot kernels outside the selected set.
- For each pilot kernel, run the normalized raw harness, then `paas-inductor --run-types=autotune`, and verify that compilation targets gfx1250, execution succeeds, and an `.autotune.launch_params` sidecar is produced.
- Generate `_tune.py` files with `paas-make-standalone --mode tune --launch-params-suffix=.autotune.launch_params`.
- Run `paas-simple-full --just-list`, followed by the actual one-replica sweep, and require:
  - a usable Inductor baseline;
  - at least one legal candidate;
  - numerical validation against the baseline (`status=ok`);
  - no gfx942 binary-load errors or missing launch metadata.
- Use the pilot’s measured configuration rate and the selected kernels’ `--just-list` counts to calculate the 72-kernel ETA. Do not reuse the earlier full-corpus estimate.

## 4. Capture baselines and launch parameters for the subset

- Run `paas-inductor` once on the 72-file staged tree with `--retarget-output-dir` pointing at a separate gfx1250 tree:

```bash
STAGED=/home/niromero/docker_workspace/paas_runs/priority-representative-gfx1250/foreign-target
TARGET=/home/niromero/docker_workspace/paas_runs/priority-representative-gfx1250/kernels

HIP_VISIBLE_DEVICES=0 paas-inductor \
  --dir "$STAGED" \
  --retarget-output-dir "$TARGET" \
  --distributed \
  --proc-per-gpu=1 \
  --run-types=autotune
```

- Preserve `baseline_perf.csv`, `autotune_perf.csv`, `benchmark_failures.csv`, and every `.autotune.launch_params` sidecar.
- Reconcile outputs against the 72-entry selection and portability manifest. Retry failures individually with focused logs and a larger timeout where appropriate; classify persistent compile, launch, API-compatibility, OOM, timeout, and harness failures rather than silently dropping them.
- Do not proceed to configuration tuning for a kernel until its gfx1250 launch parameters are present and its baseline succeeds.

## 5. Generate and tune all eligible selected instances

- Enumerate only populated model/category directories under `$TARGET`. `paas-make-standalone` is one-directory-only, and `paas-simple-full` only scans the current directory, so invoke both once per populated selected leaf rather than flattening files with colliding names.
- Generate exactly one `_tune.py` per baseline-successful kernel and run `--just-list` first to record each kernel’s legal candidate count.
- Execute `paas-simple-full` with `PAAS_TIMING_REPLICAS=1`, as requested. Each candidate still receives PAAS’s numerical output validation; only rows with case-insensitive `status=ok` are eligible to win, and the `1000 ms` failure sentinel is never ranked.
- Apply the combined branch’s search policy:
  - pointwise kernels remain otherwise uncapped;
  - reduction and persistent-reduction kernels use `max(16,384, 2 × baseline block product)`;
  - the exact Inductor baseline is retained only when it passes active-target hard limits; a rejected baseline is never reintroduced and disables the soft cap for that kernel;
  - the installed Triton tensor-size hard limit still applies;
  - use `--reduction-block-product-cap=0` only for an explicitly requested uncapped comparison.
- Do **not** run a separate second pass that arbitrarily varies `num_stages`. Preserve the current policy documented in [CHANGELOG.md](/home/niromero/docker_workspace/inductor-triton-hacks/CHANGELOG.md):
  - the default fused-kernel candidate set remains `num_stages=[1]`;
  - if Inductor's captured launch configuration used another value, that compiled value is unioned into the primary search so the baseline remains reproducible;
  - an internal `tl.range(..., num_stages=N)` baked into a reduction kernel body is not varied by changing the launch-time knob;
  - the removed `--augment-stages` behavior is not reintroduced. Any forced body-regeneration experiment would be a separate future campaign, not part of this run.
- Drive leaves through a checkpointed campaign runner that records pending/running/completed/failed state and writes one log per leaf. On interruption, use `--restart-from` to skip completed kernels and `--append` to retain already measured configurations within a partially completed kernel.
- Process category leaves model-by-model. After all leaves for one model finish, print a completion summary to the live tmux terminal and `progress.log`: model name, kernels completed/failed, configurations tested, best and median measured speedup ratios, model elapsed time, total elapsed time, processing rate, and campaign ETA. Ratios remain explicitly noise-unqualified because this is a one-replica run.
- Monitor free disk, cache growth, GPU health, and progress counts. Continue through every one of the 72 selected identities; persistent per-kernel failures remain explicit coverage failures rather than aborting the rest of the subset.

## 6. Produce collision-safe reports and verify coverage

- Run `paas-csv-summarize` separately in each leaf directory. Do not run it once over the entire tree: [organizer.py](/home/niromero/docker_workspace/inductor-triton-hacks/paas/downstream/organizer.py) keys results by bare filename stem and would collapse repeated names across models.
- Build a path-aware global aggregate keyed by `model/category/source-file`, joining against `baseline_perf.csv` by `relative_dir` and kernel name.
- Produce:
  - `best_configs.csv` with baseline config/time, best numerically valid config/time, measured ratio, candidate counts, and status counts;
  - `coverage.json` accounting for all 72 selected inputs at each stage and proving that no unlisted kernel entered the campaign;
  - `campaign_summary.md` grouped by model and category;
  - `timing.json` and a timing section in `campaign_summary.md` with total end-to-end time plus setup, portability-test, pilot, baseline, generation, tuning, and reporting durations;
  - `progress.log` containing the hourly and per-model terminal reports;
  - raw per-kernel CSVs, per-leaf summaries, logs, manifests, and environment provenance.
- Label every speedup as **single-sample and noise-unqualified**. The selected one-replica policy supports candidate ranking and numerical correctness, but it does not justify statistically validated performance claims.
- Finish by verifying both source repositories are clean and unchanged. No winning configuration will be written back into the source kernels or integrated into Inductor.
