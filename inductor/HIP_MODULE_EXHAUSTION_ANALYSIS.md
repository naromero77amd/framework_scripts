# ROCm Triton "no kernel image" Bugs — Findings

There are (at least) **two distinct bugs**, both producing `hipErrorNoBinaryForGpu`.

---

## Bug 1 — Crash at compilation `[0/4]` (FIXED)

### Summary

When using PyTorch Inductor with `EXHAUSTIVE` GEMM autotuning on ROCm, a
`CUDA error: no kernel image is available for execution on the device`
(`hipErrorNoBinaryForGpu`) crash occurred during the 5th compilation (`[0/4]`)
of a compiled model. The error originated in `select_algorithm.py` during
autotuning benchmark setup, indicating that a Triton kernel was compiled for
the wrong GPU architecture after repeated recompilations.

### Status: FIXED

`repro_tiny.py` (which triggers 5 compilations) now completes successfully
with `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE=EXHAUSTIVE`.

### Conditions Required to Trigger (historical)

All of the following had to be true simultaneously:

1. **`TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE=EXHAUSTIVE`**
2. **`dynamic=False`** — forces a full recompilation for every new input shape.
3. **Varying batch sizes** — at least 4 different batch sizes after warmup, so
   that the 5th compilation (`[0/4]`) was reached.
4. **Sufficient model complexity** — the compiled graph must contain enough
   GEMM operations and fusion opportunities.
5. **`mode="max-autotune-no-cudagraphs"`**
6. **Training (forward + backward)** — inference-only did not trigger it.

### Reproducer (no longer crashes)

```
rm -rf /tmp/torchinductor_root
TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE=EXHAUSTIVE python repro_tiny.py
```

---

## Bug 2 — HIP Module Exhaustion at compilation `[0/7]` (ROOT-CAUSED)

### Status: ROOT-CAUSED — patches available

### Root Cause

This is the **same class of bug** addressed by upstream Triton PR
[triton-lang/triton#9444](https://github.com/triton-lang/triton/pull/9444)
(cherry-picked as [ROCm/triton#928](https://github.com/ROCm/triton/pull/928)):
**loaded HIP/CUDA modules are never unloaded**, exhausting the GPU driver's
module table.

PR #9444 fixed the leak in Triton's `CompiledKernel.__del__` (calling
`hipModuleUnload`). However, the fix depends on Python's garbage collector
actually running to trigger `__del__`. During EXHAUSTIVE autotuning:

- Each GEMM operation benchmarks **~145 Triton kernel variants**.
- Each variant creates a `CompiledKernel` with a loaded HIP module.
- Only the winner is kept; the ~144 losers become garbage.
- With 32 compile threads (the default) and rapid allocation, **Python's GC
  does not run frequently enough** to collect the stale objects before the
  next autotuning round creates 145 more.

After ~17,500 kernel compilations across 8 recompilations (~20,000 HSACO
files), the HIP driver's module table is exhausted, and any subsequent HIP
call fails with `hipErrorNoBinaryForGpu` (error 209).

Additionally, PyTorch Inductor loads HIP modules through a **second code path**
(`StaticallyLaunchedTritonKernel` in `static_triton_launcher.py`) that is
**not covered by PR #9444 at all**:

- `_StaticCudaLauncher._load_kernel()` loads HSACO binaries via
  `hipModuleLoadDataEx` in C++
- It returns only `(function, n_regs, n_spills)` — the **module handle is
  discarded** and never returned to Python
- There is no `__del__`, no `_unload_kernel`, no cleanup of any kind
- These modules accumulate indefinitely

### Why Triton PR #9444 / ROCm #928 Was Insufficient

PR #9444 added `CompiledKernel.__del__` which calls
`driver.active.utils.unload_module(self.module)` (i.e. `hipModuleUnload` /
`cuModuleUnload`) when a `CompiledKernel` object is garbage-collected. This
was the correct fix for the module leak in Triton's own kernel loading path.
However, it is **insufficient** for PyTorch Inductor workloads for two
independent reasons:

**Reason 1: Python GC timing mismatch**

Python's cyclic garbage collector runs based on object allocation thresholds
(default: generation 0 triggers after 700 allocations). During EXHAUSTIVE
autotuning, Inductor creates and discards `CompiledKernel` objects much faster
than GC can collect them:

- A single GEMM autotuning round creates ~145 `CompiledKernel` objects, each
  loading a HIP module via `hipModuleLoad`.
- Only the fastest kernel wins; the other ~144 are unreferenced immediately.
- But `__del__` only fires when GC actually collects the dead objects.
- With 32 compilation threads generating kernels in parallel, the rate of
  module loading far outpaces the rate of GC-driven unloading.
- By the time GC runs, thousands of stale modules have accumulated.

The `gc.collect()` workaround (patch 1) forces immediate collection after each
autotuning round, ensuring the stale `CompiledKernel.__del__` methods fire
before the next round begins. This addresses the timing mismatch.

**Reason 2: PyTorch's `_StaticCudaLauncher` bypasses Triton's cleanup entirely**

PyTorch Inductor has its own kernel loading path
(`StaticallyLaunchedTritonKernel` in `static_triton_launcher.py`) that loads
`.hsaco`/`.cubin` files directly via C++ (`_StaticCudaLauncher._load_kernel`
in `torch/csrc/inductor/static_launcher/cuda.cpp`). This code path:

1. Calls `hipModuleLoad(&mod, ...)` to load the binary into a `CUmodule`
2. Extracts `CUfunction func` from the module
3. **Returns only `(func, n_regs, n_spills)` to Python** — the `CUmodule`
   handle is a local variable that goes out of scope
4. Has **no `__del__`** method and **no unload mechanism** whatsoever

Since the module handle is discarded in C++, there is no way for Python to
ever unload it — not even with `gc.collect()`. Triton's `CompiledKernel.__del__`
is irrelevant here because these kernels are loaded through a completely
separate code path.

The C++ patch (patch 2) fixes this by returning the module handle to Python
and adding `_unload_kernel()` / `__del__` to `StaticallyLaunchedTritonKernel`.

**Summary of all three fixes and their dependencies:**

| Fix | What it does | Standalone? |
|---|---|---|
| **Triton PR #9444** | Adds `CompiledKernel.__del__` → `hipModuleUnload` / `cuModuleUnload` | Necessary but not sufficient (GC fires too late) |
| **Patch 1** (`gc_module_unload.patch`) | Calls `gc.collect()` after autotuning to force timely `__del__` | **Requires PR #9444** — without it, `CompiledKernel` has no `__del__`, so `gc.collect()` reclaims the objects but no modules are unloaded |
| **Patch 2** (`static_launcher_module_unload.patch`) | Returns module handle from C++, adds `__del__` to `StaticallyLaunchedTritonKernel` | Self-contained (adds its own cleanup, independent of Triton) |

**Which fixes cover which module loading path:**

| Module loading path | PR #9444 needed? | Patch 1 (gc.collect) needed? | Patch 2 (C++ fix) needed? |
|---|---|---|---|
| Triton `CompiledKernel` (autotuning benchmarks) | **Yes** (provides `__del__`) | **Yes** (forces timely `__del__`) | No (different path) |
| PyTorch `_StaticCudaLauncher` (winner kernel loading) | No (irrelevant path) | No (no `__del__` to trigger) | **Yes** (adds handle + `__del__`) |

**Practical impact assessment:**

The autotuning benchmark path (row 1) is the **dominant** source of module
leaks — each GEMM autotuning round creates ~145 `CompiledKernel` objects, of
which ~144 become garbage. Across 8 recompilations with many GEMMs, this
accumulates ~17,500 leaked modules and is what causes the crash.

The winner kernel path (row 2) leaks far fewer modules: only the selected
winner kernels (~50 per recompilation × 8 recompilations ≈ 400 modules).
This is orders of magnitude below the crash threshold.

**PR #9444 + Patch 1 are sufficient to fix the crash in practice.** Together
they clean up the massive autotuning leaks that exhaust the module table.
Patch 2 is a correctness fix that plugs the smaller winner-kernel leak, but
in realistic workloads (≤ tens of recompilations) those ~400 modules alone
would not exhaust the driver. Patch 2 would become necessary in a
hypothetical scenario with hundreds of recompilations, where winner-kernel
leaks alone could accumulate enough to hit the limit.

### Evidence: GC objects collected per iteration

Running `gc.collect()` between training iterations shows tens of thousands of
stale objects accumulating:

```
iter 0 [0/1]: 20,577 objects collected
iter 1 [0/2]: 44,698 objects collected
iter 2 [0/3]: 14,230 objects collected
iter 3 [0/4]: 51,786 objects collected
iter 4 [0/5]: 43,896 objects collected
iter 5 [0/6]: 36,753 objects collected
iter 6 [0/7]: 10,960 objects collected   ← would crash WITHOUT gc.collect()
iter 7 [0/8]:      7 objects collected   ← cache_size_limit hit, eager fallback
```

### Is This ROCm-Specific?

**No.** The underlying leak exists on **both ROCm and CUDA**:

- Upstream PR #9444 added `hipModuleUnload` AND `cuModuleUnload` — both
  platforms had the same missing cleanup.
- The `StaticallyLaunchedTritonKernel` code path in PyTorch Inductor is
  shared across CUDA and ROCm — neither platform unloads modules loaded
  through this path.
- CUDA GPUs likely have a **higher module table limit** than ROCm, making
  the crash harder to hit in practice. But with enough EXHAUSTIVE autotuning
  rounds, CUDA would eventually exhaust its limit too.

### Patch 1: Minimal Fix (PyTorch Python-side)

Add `gc.collect()` after each autotuning benchmark round in
`torch/_inductor/select_algorithm.py`. This forces Python to run
`CompiledKernel.__del__` → `hipModuleUnload()` / `cuModuleUnload()` between
rounds, preventing module accumulation through Triton's loading path.

Patch file: `gc_module_unload.patch`

```diff
--- a/torch/_inductor/select_algorithm.py
+++ b/torch/_inductor/select_algorithm.py
@@ -2,6 +2,7 @@
 import contextlib
 import dataclasses
 import functools
+import gc
 import hashlib
 ...
@@ -3228,7 +3229,18 @@ class AlgorithmSelectorCache(PersistentCache):
         )
         # `benchmark_fn(choices)` will execute each choice ...
-        return benchmark_fn(choices)
+        result = benchmark_fn(choices)
+
+        # Collect stale CompiledKernel objects from benchmarking.  Each
+        # autotuning round compiles many Triton kernel variants (up to ~145
+        # with EXHAUSTIVE search).  Only the winner is kept; the rest become
+        # garbage.  On ROCm, CompiledKernel.__del__ calls hipModuleUnload()
+        # (triton-lang/triton#9444) to free the loaded GPU module.  Without
+        # an explicit gc.collect() here, the modules accumulate faster than
+        # Python's cyclic GC can reclaim them, eventually exhausting the HIP
+        # driver's module table and raising hipErrorNoBinaryForGpu (error 209).
+        gc.collect()
+
+        return result
```

### Patch 2: Complete Fix (PyTorch C++ + Python side)

The `gc.collect()` fix only covers the Triton `CompiledKernel` path. The
**complete** fix requires changes to PyTorch's `_StaticCudaLauncher` C++ code
(`torch/csrc/inductor/static_launcher/cuda.cpp`) and the Python wrapper
(`torch/_inductor/runtime/static_triton_launcher.py`):

1. **Change `loadKernel()`** to return `std::pair<CUmodule, CUfunction>`
   instead of just `CUfunction`.
2. **Change `load_kernel()` Python binding** to return `(module, function,
   n_regs, n_spills)` — the module handle is now exposed to Python.
3. **Add `unload_kernel()` C++ function** that accepts a module handle and
   calls `hipModuleUnload()` / `cuModuleUnload()`.
4. **Register `_unload_kernel`** in the `StaticCudaLauncherMethods` array.
5. **Store `self.module`** in `StaticallyLaunchedTritonKernel.__init__`.
6. **Add `__del__`** to `StaticallyLaunchedTritonKernel` that calls
   `self.C_impl._unload_kernel(self.module)` (mirroring Triton PR #9444).

Patch file: `static_launcher_module_unload.patch`

This covers both CUDA and ROCm (the C++ code uses `#if defined(USE_ROCM)`
guards throughout).

### Reproducer

`repro_bug2.py` — crashes in ~5 minutes at B=720:

```bash
rm -rf /tmp/torchinductor_root
TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE=EXHAUSTIVE python repro_bug2.py
```

`repro_bug2_gc.py` — same but with `gc.collect()` between iterations (passes):

```bash
rm -rf /tmp/torchinductor_root
TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE=EXHAUSTIVE python repro_bug2_gc.py
```

### Experimental Results

| Test | COMPILE_THREADS | gc.collect()? | `[0/7]` result | Total time |
|---|---|---|---|---|
| Default (32 threads) | 32 | No | **CRASH** | ~5 min |
| Single-threaded | 1 | No | Pass | ~15 min |
| Reduced threads | 4 | No | Pass | ~10 min |
| Default + GC | 32 | Yes | **Pass** | ~5 min |

The `gc.collect()` fix is the best option: it runs at full speed (32 threads)
and avoids the crash.

Note: `COMPILE_THREADS=4` also showed a race condition in Triton's
compilation cache (`FileNotFoundError` on `.source` files and
`AttributeError: 'CompiledKernel' object has no attribute 'module'` in
`__del__`) but still completed successfully. This is a separate Triton bug.

---

## Environment Notes

### `TRITON_HIP_USE_ASYNC_COPY=0` — REQUIRED

This variable **must** be set. It addresses a **separate issue** from module
exhaustion: without it, Triton emits async copy instructions that are
incompatible with gfx950, producing binaries that fail with the same
error 209 but through a different code path.

Tested: running `repro_bug2_gc.py` (with `gc.collect()`) but **without**
`TRITON_HIP_USE_ASYNC_COPY=0` still crashes at `[0/7]`:

```
iter 6: B=713 (compilation [0/7] expected)
  → coordinate_descent_tuner.py → _precompile_config → make_launcher → load_kernel
  → triton_poi_fused__to_copy_17   ← pointwise kernel, NOT a GEMM
  RuntimeError: CUDA driver error: 209
```

This is a different crash path from the module exhaustion bug:
- **Module exhaustion** (Bug 2): crash during GEMM autotuning benchmark setup
- **Async copy incompatibility**: crash during coordinate descent tuning of
  pointwise kernels, because the generated HSACO binary contains instructions
  the GPU doesn't support

**Both `TRITON_HIP_USE_ASYNC_COPY=0` and the `gc.collect()` patch are needed.**

## Workarounds

| Workaround | Change | Trade-off |
|---|---|---|
| **gc.collect() patch** (recommended) | Apply `gc_module_unload.patch` to PyTorch | Adds ~ms of GC overhead per autotuning round; no functional impact |
| Reduce compile threads | `TORCHINDUCTOR_COMPILE_THREADS=4` | ~2x slower compilation |
| Drop EXHAUSTIVE search | Remove `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE` env var | Fewer kernel candidates; may miss optimal kernel |
| Use `dynamic=True` | `torch.compile(..., dynamic=True)` | Single symbolic compilation; may have guard overhead |
| Use `dynamic=None` (default) | `torch.compile(..., dynamic=None)` | Auto-switches to dynamic after a few recompiles |
| Pad to fixed batch size | Use `--padded` flag in `workload_v4.py` | Wastes compute on padding; no recompilations |
| Limit recompilations | `torch._dynamo.config.cache_size_limit = 7` | 8th+ unique shapes fall back to eager mode |

## Tested Configurations (Bug 1, historical)

| `dynamic=` | Crashes? | Time  | Recompilations |
|---|---|---|---|
| `False`    | **Yes** at `[0/4]` | ~5 min | 5 (warmup + 4 training) |
| `None`     | No       | ~2 min | ~2, then auto-dynamic |
| `True`     | No       | ~1.5 min | 0 (symbolic shapes) |

## Related Observation

With `B=7,200,000` and `dynamic=False`, a **layout conflict warning** appears
at every compilation:

```
Layout conflict detected for buf44: template expects
FixedLayout(..., size=[32, 16], stride=[16, 1]) but layout is
frozen to FixedLayout(..., size=[32, 16], stride=[1, 32])
```

This warning is benign and does not cause a crash. It appears at every
compilation `[0/1]` through `[0/7]` but is not present with small batch sizes.
