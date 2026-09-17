# PAAS tuning experiment on gfx942

Generated: 2026-09-17T04:35:57.798255+00:00

## Quick answer

- **Use the 25 ms warmup / 100 ms measurement setting.** The longer 100 ms / 1,000 ms setting did not find a reliably better result. It increased total tuning time by 3.38%.
- **Keep the hybrid reduction cap enabled.** On the kernel where the cap actually removed configurations, it saved about 38 minutes per tuning run and did not remove the winning configuration.
- These conclusions apply to the two kernels tested here on an MI308X (`gfx942`).

## How many kernels were tuned?

**Two unique kernels were tuned.**

1. The DeepSeek-V4-Flash sigmoid-backward reduction kernel from the original investigation.
2. A gpt-oss-20b mul-sum reduction kernel, added because its search is large enough for the reduction cap to have an observable effect.

Each kernel was tuned four times: once for every combination of the two timing settings and cap on/off. That produced **eight tuning runs**, but still only **two kernels**.

Across those eight runs, PAAS tried 1,992 kernel configurations: 1,922 passed, 32 produced an incorrect result, and 38 timed out.

## What the two settings mean

- The **timing setting** controls how long Triton warms up and measures each possible kernel configuration. We compared 25/100 ms with 100/1,000 ms.
- The **hybrid reduction cap** skips unusually large block combinations before tuning. It does not change the kernel; it reduces the number of configurations PAAS tries.
- A **tuning run** below means tuning one kernel once with one timing setting and one cap setting.

## Did the longer timing setting help?

**No reliable benefit was observed.**

- With the cap enabled, the longer timing added 2.7 seconds for DeepSeek and 1 minute 32.6 seconds for gpt-oss.
- With the cap disabled, the longer timing added 8.9 seconds for DeepSeek and 1 minute 45.3 seconds for gpt-oss.
- Across all runs, the longer setting increased tuning time by 3.38%.
- The longer and shorter runs sometimes chose different top configurations. We then remeasured every chosen configuration five times using the same strict method. Their measurement ranges overlapped, so the different choices were not proven performance differences.
- The longer setting costs much less than ten times as much overall because process startup, compilation, and correctness checking dominate these tuning runs.

## Did the hybrid reduction cap help?

**Yes for the larger gpt-oss search; it made no difference to the small DeepSeek search.**

- DeepSeek had 52 possible configurations with or without the cap. Nothing was removed, so its capped and uncapped run times were effectively the same.
- gpt-oss had 500 possible configurations without the cap and 392 with it. The cap skipped 108 configurations, or 21.6% of the search.
- Without the cap, the same 19 large configurations timed out in both timing tests. Every one of those timeout configurations was among the configurations skipped by the cap.
- With 25/100 ms timing, the cap reduced gpt-oss tuning time from 1h 6m 41.8s to 28m 52.8s, saving 37m 49.0s.
- With 100/1,000 ms timing, the cap reduced gpt-oss tuning time from 1h 8m 27.1s to 30m 25.4s, saving 38m 1.7s.
- The best configuration found in either uncapped run was also present in the capped search. The cap therefore saved time without removing the winner.

## Results for each kernel

### DeepSeek-V4-Flash sigmoid-backward reduction

- **25/100 ms timing, reduction cap enabled:** tried 52 configurations in 3m 54.8s; 44 passed, 8 were incorrect. PAAS chose `R0_BLOCK=2048, XBLOCK=1, num_stages=1, num_warps=1`.
- **100/1,000 ms timing, reduction cap enabled:** tried 52 configurations in 3m 57.5s; 44 passed, 8 were incorrect. PAAS chose `R0_BLOCK=1024, XBLOCK=1, num_stages=1, num_warps=2`.
- **25/100 ms timing, reduction cap disabled:** tried 52 configurations in 3m 55.9s; 44 passed, 8 were incorrect. PAAS chose `R0_BLOCK=1024, XBLOCK=1, num_stages=1, num_warps=4`.
- **100/1,000 ms timing, reduction cap disabled:** tried 52 configurations in 4m 4.8s; 44 passed, 8 were incorrect. PAAS chose `R0_BLOCK=2048, XBLOCK=1, num_stages=1, num_warps=8`.

### gpt-oss-20b mul-sum reduction

- **25/100 ms timing, reduction cap enabled:** tried 392 configurations in 28m 52.8s; 392 passed. PAAS chose `R0_BLOCK=16, XBLOCK=64, num_stages=1, num_warps=2`.
- **100/1,000 ms timing, reduction cap enabled:** tried 392 configurations in 30m 25.4s; 392 passed. PAAS chose `R0_BLOCK=16, XBLOCK=16, num_stages=1, num_warps=4`.
- **25/100 ms timing, reduction cap disabled:** tried 500 configurations in 1h 6m 41.8s; 481 passed, 19 timed out. PAAS chose `R0_BLOCK=16, XBLOCK=64, num_stages=1, num_warps=2`.
- **100/1,000 ms timing, reduction cap disabled:** tried 500 configurations in 1h 8m 27.1s; 481 passed, 19 timed out. PAAS chose `R0_BLOCK=8, XBLOCK=64, num_stages=1, num_warps=8`.

## Final checked results

### DeepSeek-V4-Flash sigmoid-backward reduction

- Original configuration: 0.007497 ms.
- Best configuration chosen by the tuning runs: `R0_BLOCK=1024, XBLOCK=1, num_stages=1, num_warps=4`.
- Checked result: 0.007458 ms (observed range 0.007417–0.007537 ms).

### gpt-oss-20b mul-sum reduction

- Original configuration: 0.019164 ms.
- Best configuration chosen by the tuning runs: `R0_BLOCK=16, XBLOCK=64, num_stages=1, num_warps=2`.
- Checked result: 0.014514 ms (observed range 0.013511–0.016158 ms).

For DeepSeek, the original and tuned measurement ranges overlap. The apparent 0.5% improvement is measurement noise, not a demonstrated speedup.

For gpt-oss, the original median was 0.019164 ms and the tuned median was 0.014514 ms. That is 1.3204x the throughput, or about 24.3% less execution time. The tuned result was faster than the original in all five checking runs.

## Time spent

- Preparing the kernels: 43.9s.
- Counting and checking the search spaces: 22.1s.
- Running the eight tuning tests: 3h 30m 20.2s.
- Cooling the GPU between tests: 3m 30.0s.
- Rechecking all selected results five times: 3m 32.4s.
- Total measured experiment time: 3h 38m 28.7s.

## How the results were checked

- Every configuration ran in a new process and created a new reference result.
- Every tuning run used the same generated input data (seed 0).
- Every tuning run used separate empty Triton and Inductor cache directories.
- PAAS rejected numerically incorrect configurations during tuning.
- Every selected configuration was checked again using FP64 comparison and five new processes under the same 100/1,000 ms measurement setting.
- All 189 focused PAAS tests passed before tuning started.

## Technical record

- Experiment PAAS commit: `602b04ef77dcbebb9146cb62e136a0d5fffe6476`.
- Corpus commit: `2d7f07e5b80c425d8cb9af4a5e06f5f5fa1683d8`.
- The uncapped gpt-oss runs encountered the same 19 timeout configurations.
- Detailed values remain available in `results.json`, `timing.json`, and the raw per-run CSV files in `/home/niromero/docker_workspace/paas_runs/paas-gfx942-reduction-factorial/`.
- Conclusions are specific to these kernels, this software stack, and gfx942.
