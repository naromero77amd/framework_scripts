# PAAS default revision comparison on gfx942

Generated: 2026-09-17T07:20:07.432526+00:00

## What was compared

Two unique kernels were tuned with two different versions of PAAS:

1. DeepSeek-V4-Flash sigmoid-backward reduction.
2. gpt-oss-20b mul-sum reduction.

Each PAAS revision performed one end-to-end tuning run containing both kernels. Therefore, this comparison contains two PAAS runs and four kernel-tuning jobs.

- Tip revision: `7079b429a90f0ce90c915b898bcb47ddef5cd7e8`.
- Jack Taylor revision: `c41317e8e4009c7205e7611b5bfd27fd7d76150c`.

No timing, reduction-cap, replica, or winner-confirmation option was overridden. Each revision used exactly its own defaults. Both runs used physical GPU 0 and started with separate empty Triton and Inductor caches.

## Direct comparison

- Tip end-to-end tuning time: **1h 13m 25.8s**.
- Jack Taylor commit end-to-end tuning time: **1h 11m 8.8s**.
- The faster revision was **Jack Taylor commit c41317e**, which completed 1.03x faster than Tip of inductor-triton-hacks.
- Put another way, the tip took 2m 17.0s longer, or 3.21% more time.
- Tip generated and swept 556 configurations.
- Jack Taylor commit generated and swept 556 configurations.
- The preflight showed that both revisions generated the same 52 DeepSeek configurations and the same 504 gpt-oss configurations, in the same order.

The tip also performs default rechecks that the older commit does not: it measures the two baselines 6 times in total and performs 24 additional winner-check executions after the sweep. In total, the tip launched 586 baseline/configuration processes while the older commit launched 558. Those operations are included in the end-to-end times.

## Defaults used by each revision

### Tip

- Timing: 20 ms warmup / 50 ms measurement, at most 10,000 repeats.
- Correctness: Numerical validation enabled.
- Winner handling: Best 4 configurations rechecked 3 times each.

### Jack Taylor commit

- Timing: 100 ms warmup / 1,000 ms measurement.
- Correctness: No numerical comparison against the baseline.
- Winner handling: No separate winner confirmation.

These defaults are intentionally different. This experiment compares the complete default behavior of the two revisions; it does not isolate one individual code change.

The result labels also are not directly equivalent: the tip checks numerical correctness and found six wrong DeepSeek configurations. The older commit does not perform that comparison, so it reported all 52 DeepSeek configurations as OK.

## Results by revision and kernel

### Tip of inductor-triton-hacks

- End-to-end time for both kernels: 1h 13m 25.8s.
- Total configurations swept: 556.
- DeepSeek-V4-Flash sigmoid-backward reduction: 52 configurations swept (46 ok, 6 wrong); 12 additional winner-check executions.
- gpt-oss-20b mul-sum reduction: 504 configurations swept (19 fail, 481 ok, 4 unvalidated); 12 additional winner-check executions.

### Jack Taylor commit c41317e

- End-to-end time for both kernels: 1h 11m 8.8s.
- Total configurations swept: 556.
- DeepSeek-V4-Flash sigmoid-backward reduction: 52 configurations swept (52 ok); 0 additional winner-check executions.
- gpt-oss-20b mul-sum reduction: 504 configurations swept (4 error, 19 fail, 481 ok); 0 additional winner-check executions.

## Counting rules

- A configuration swept means one distinct row produced during the main search.
- Baseline measurements are listed separately and are not counted as searched configurations.
- Repeated winner checks are listed separately and are not counted as new configurations.
- End-to-end time starts immediately before PAAS begins processing the two generated tune scripts and ends when PAAS exits after all of its default work.
- Kernel copying, tune-script generation, candidate counting, and the 30-second between-run GPU cooldown are not included in either tuning time.

## Technical record

- GPU: AMD Instinct MI308X (gfx942:sramecc+:xnack-).
- PyTorch: 2.13.0+rocm10.1.0a20260813.
- Triton: 3.8.0.
- Full logs, CSV files, generated scripts, manifest, and machine-readable results are stored in `/home/niromero/docker_workspace/paas_runs/paas-default-revision-comparison`.
