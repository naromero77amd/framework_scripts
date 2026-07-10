# framework_scripts
Various scripts for building PyTorch and supporting libraries.

## PyTorch Inductor Test Runner

`pytorch/run_tests.py` can run targeted CSV test lists, full Inductor suites, and the ROCm torch-triton-nightly Inductor validation set.

By default, failed single-test executions are retried up to two times, matching PyTorch's normal rerun count. Use `--reruns 0` to disable retries entirely. Recovered tests are reported separately as flaky, while tests that still fail after all attempts are reported as consistent failures.

Signal-based exits, such as `SIGKILL` or `SIGSEGV`, are categorized as failed tests and include the signal name in the per-test log and summary.

Example 8-GPU torch-triton-nightly Inductor run:

```bash
python pytorch/run_tests.py \
  --all-tests \
  --include-triton-nightly-inductor-tests \
  --pytorch-path /path/to/pytorch \
  --num-gpus=8 \
  --log-file triton_nightly_inductor.log
```
