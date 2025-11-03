# Workflow for Analyzing Performance Improvement on HF suite due to Inductor PR

1. Pick a machine that does not have significant performance variation.

2. Pick a recent docker image based on ROCm 7. For example: `docker pull rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0`

3. Pick a version of Triton. Build and install using this script in `triton/build.sh`.
   NOTE: If you are using the PyTorch ROCm docker images from docker hub as a starting point, make sure that the `pytorch-torch-rocm` package is uninstalled and that you don't have *two* Triton packages. Aforementioned build script will take care of uninstalling older Triton packages.

4. Pick a working upstream PyTorch main commit. Save this branch so you don't loose track, e.g. `git checkout -b meta-upstream-pr-baseline`

5. Build branch with `pytorch/build.sh`.

6. Run the HF test suite using the script in `inductor/run_hf.sh`.
   This should take about 30 minutes to execute on a MI350.  
   NOTE: This script has several environment variables set including AMDGCN_USE_BUFFER_OPS=1. The Inductor cache is tarred up and saved.

7. Extract kernels using https://github.com/ROCm/inductor-triton-hacks, for example:
   `paas-organize --inductor-cache=/tmp/torchinductor_root --output-dir=./organized`
   The reason we do this is because we don't know for these models what the fraction of time is spent in Inductor/Triton kernels. Potentially, it can be very small and therefore difficult to see what the performance improvement could be.
   NOTE: You will need to make a modification to inductor-triton-hacks due to recent changes upstream to the `return mode`. Also, the `reps` and `warmup` can be changed to
   100 and 10, respectively with no impact.

```
diff --git a/processing/organize_kernels.py b/processing/organize_kernels.py
index 64518ce..b816218 100644
--- a/processing/organize_kernels.py
+++ b/processing/organize_kernels.py
@@ -140,7 +140,7 @@ class FileProcessor:
     
     def __init__(self, reps: int = 1000, warmup: int = 100, replace_device_id: bool = True):
         self.benchmark_pattern = re.compile(r'rep\s*=\s*\d+')
-        self.benchmark_replacement = f'rep={reps}, warmup={warmup}, return_mode="median"'
+        self.benchmark_replacement = f'rep={reps}, warmup={warmup}'
         self.replace_device_id = replace_device_id
         
         # Device ID replacement patterns
```

   There is a script in `inductor/modify_inductor_bench.sh` that will modify that line in all kernels that are recursively found in a subdirectory.

8. Run the kernels using the script `inductor/run_triton.sh <dir>`.

   This script is very similar to the behavior of:
   `paas-inductor --dir=./organized --run-types=autotune`

   There are two main differences:
   1. `run_triton.sh` script is serial, there is no --distributed option. This was done for simplicity mostly.
   2.  Inductor kernel naming scheme is complex and in general can change with any code changes. This can occur between different architectures (ROCm vs. CUDA) and even on single architecture when a PR is applied.
       `run_triton.sh` will create a CSV file similar to `paas-inductor` but the kernel name will have the filename with two extra concataned strings.

       For example, the kernel in `triton_per_fused_add_native_layer_norm_backward_select_16.py`
       will have an entry in the CSV file that looks something like:

       persistent_reduction,meta_pr/hf_with_per_pr/kernels-hf/persistent_reduction/triton_per_fused_add_native_layer_norm_backward_select_16.py,triton_per_fused_add_native_layer_norm_backward_select_16_x8192_r6,0.006,36.07

       where the kernel name has the basefilename with an extra string, "x<numel>_r<r0_numel>"

       This will help distinguish kernels with different dynamic shapes.

       NOTE: This script is not general enough to work in some cases:
       For example, the first integer after the kernel name, `_16` in this example, may not necessary be the same when comparing very different Inductor caches.

       In spite of these limitations, this script seems to be sufficient for the use case of a single architecture with and without the PR change.

       For 2D or 3D kernels, it is possible that we would need to add additional string to distinguish the kernel. For example, for a 2D string "y<numel>_r<r1_numel>".

       A more robust solution would be to compare the entire kernel body.

9. Repeat Steps. 4 - 8. for the baseline branch with the PR applied.

10. `kernel_perf_simply` reference.csv optimized.csv
    This last script does a comparison of the kernels and prints a geomean value.
