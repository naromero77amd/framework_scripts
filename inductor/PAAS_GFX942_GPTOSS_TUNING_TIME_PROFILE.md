# Where PAAS tuning time goes on gfx942

## Quick answer

**One gpt-oss reduction kernel took 1h 12m 26.9s to tune from an empty Triton and Inductor cache. PAAS tested 500 configurations: 481 completed successfully and 19 timed out.**

The kernel measurement itself is not the reason tuning is slow. Across all 481 successful configurations, Triton's timing work consumed only 13.9s, or 0.32% of the complete run.

Most time went to four places:

1. Starting a fresh Python process and importing Python, PyTorch, and Triton: **22m 48.1s (31.5%)**.
2. Creating a fresh reference, compiling/loading and running the candidate, and checking correctness: **20m 38.0s (28.5%)**.
3. Waiting for 19 configurations to time out and terminate: **19m 38.4s (27.1%)**.
4. Waiting for completed child processes to shut down and for the scheduler to notice them: **7m 27.5s (10.3%)**.

Together, those four categories account for 97.4% of the run.

## What was run

- Kernel: `triton_red_fused_mul_sum_0` from gpt-oss-20b.
- PAAS trunk commit: `ad6d4416e180783745bcc69944b08b00cda3699a`.
- GPU: one AMD Instinct MI308X (`gfx942`), physical GPU 0.
- PAAS defaults were unchanged: one fresh process and fresh numerical reference per candidate, 20/50 ms Triton timing, one sweep measurement, three baseline measurements, and four finalists checked three times each.
- The timed interval includes parameter discovery, baselines, the 500-config sweep, and default winner confirmation. Kernel copying and tune-script generation are excluded.

## Cache verification

- Immediately before timing, the dedicated Inductor cache did not exist and contained 0 files.
- Immediately before timing, the dedicated Triton cache did not exist and contained 0 files.
- After tuning, the Triton cache contained 3,921 files using 207.0 MiB.
- The standalone tune script did not populate the dedicated Inductor cache.

## Complete time breakdown

- **Python, PyTorch, and Triton startup/import:** 22m 48.1s (31.47%). 481 completed sweep processes.
- **Reference creation, candidate compile/run, and comparison:** 20m 38.0s (28.48%). 480 validated sweep processes.
- **Nineteen timed-out configurations:** 19m 38.4s (27.11%). Includes the 60-second wait and 2-second termination grace.
- **Process shutdown and scheduler detection:** 7m 27.5s (10.30%). 481 completed sweep processes.
- **Default winner confirmation:** 51.5s (1.19%). 4 configurations, 3 fresh processes each.
- **Input tensor allocation:** 30.8s (0.71%). 481 completed sweep processes.
- **Triton estimate and do_bench timing:** 13.9s (0.32%). 481 completed sweep processes.
- **Three baseline measurements:** 11.7s (0.27%). Default baseline replication.
- **Parameter discovery:** 3.6s (0.08%). One fresh process.
- **Other parent-process overhead:** 2.8s (0.06%). Manager import, file setup, and final teardown.
- **Other work inside completed candidate processes:** 0.6s (0.01%). Argument parsing, grid construction, and result output.
- **Generate the 500-configuration list:** 0.0s (0.00%). Parent process.

## What happens inside a successful configuration

A successful configuration took a median of 4.504s and a p95 of 16.507s.

- Fresh process startup/import: median 2.562s; p95 4.105s.
- Input allocation: median 57.0 ms.
- Numerical validation: median 0.991s; p95 12.916s.
- Triton estimate and do_bench: median 17.5 ms; only 13.9s summed across the successful sweep.
- Process shutdown plus scheduler detection: median 0.870s.

The validation time itself breaks down as follows:

- Candidate compile/load and launch: 12m 53.0s.
- Fresh reference compile/load and launch: 6m 21.6s.
- FP64 comparison and restoring inputs: 1m 22.9s.
- Tensor snapshots and reference copying together took less than one second.

The compile/load-and-launch labels include the very short kernel execution. Because the kernel itself runs in microseconds, these wall times are dominated by compilation or loading compiled code.

## Timeout cost

- Exactly 19 configurations timed out.
- Each consumed about 62.018s: the 60-second timeout plus the two-second termination grace.
- Together they consumed 19m 38.4s, or 27.1% of the entire tuning run.
- Every timeout used a large block product and occurred at XBLOCK 2048, 4096, or 8192.

## What can be done

### 1. Filter oversized reduction configurations

For this kernel, a 16,384 block-product cap would skip 108 configurations. Those configurations consumed 38m 30.9s, including all 19 timeouts. The winner's block product was only 1024, so it remains inside that search.

This is the highest-confidence improvement for this kernel. The earlier A/B experiment independently found that the cap retained the winner while cutting about 38 minutes.

### 2. Split one kernel's configurations across the available GPUs

PAAS currently parallelizes separate kernel files, not configurations within one kernel. This host has eight GPUs, but this run used only one. Assigning the observed configuration times greedily across eight GPUs gives an idealized end-to-end lower bound of about 10m 6.8s, before cache, host, and I/O contention. This can preserve one fresh process and reference per candidate.

### 3. Replace the half-second polling loop

The combined child-shutdown and scheduler-detection bucket was 7m 27.5s. Polling can account for at most 4m 0.5s of that amount, and roughly half that bound is a realistic expectation. Event-driven process completion would not remove Python/HIP shutdown time.

### 4. Do not start by shortening Triton's measurement

The entire successful sweep spent only 13.9s in iteration estimation and do_bench. Even eliminating that work completely would save only 0.32% of this run.

### 5. Preserve the correctness boundary

A persistent worker or cached numerical reference could target the large import and reference costs, but it would no longer provide one fresh process and fresh reference per candidate. Treat that as a separate validation-policy experiment, not a drop-in optimization.

## Measurement quality and artifacts

- 7,064 JSON profile events were parsed successfully.
- A local 10,000-event probe measured about 21 microseconds per JSONL write; the estimated profiling write overhead for this run is about 0.15 seconds.
- The first instrumentation commit was `0285069d5daace116e1b60c56474ca1ea208a5a7`. Confirmation processes were labeled as sweep processes in that run; they were reclassified using the exact recorded sweep-end timestamp. Commit `f70917f5` fixes the label for future runs.
- Machine-readable analysis: `/home/niromero/docker_workspace/paas_runs/paas-gptoss-main-time-profile/profile_analysis.json`.
- Raw profile: `/home/niromero/docker_workspace/paas_runs/paas-gptoss-main-time-profile/run/wall_profile.jsonl`.
- Raw tuning log and CSV files: `/home/niromero/docker_workspace/paas_runs/paas-gptoss-main-time-profile/run`.
