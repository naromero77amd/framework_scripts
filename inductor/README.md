# Workflow for Analyzing Performance Improvement on HF suite due to Inductor PR

1. Pick a machine that does not have significant performance variation
2. Pick a recent docker image based on ROCm 7. For example: `docker pull rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0`

3. Pick a version of Triton. Build and install using this script in triton/build.sh.
   NOTE: If you are using the PyTorch ROCm docker images from docker hub make sure that the `pytorch-torch-rocm` package is uninstalled and that you don't have *two* Triton packages. Aforementioned build script will take care of this.

4. Pick a working upstream PyTorch main commit. Save this branch, e.g. `git checkout -b meta-upstream-pr-baseline`
5. Build branch with `pytorch/build.sh`.
6. Run the HF test suite using the script in `inductor/run_hf.sh`.
   This should take about 30 minutes to execute on a MI350.  
   NOTE: This script has several environment variables set including AMDGCN_USE_BUFFER_OPS=1
7. Extract kernels using https://github.com/ROCm/inductor-triton-hacks, for example:
   paas-organize --inductor-cache=/tmp/torchinductor_root --output-dir=./organized
   The reason we do this is because we don't know for these models what fraction of time is spend in Inductor/Triton kernels. Potentially it can be very small and therefore difficult to see what the performance improvement could be.  
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
8. Run the kernels using the script `inductor/run_triton.sh`.

   This script runs the kernels sequentially because 