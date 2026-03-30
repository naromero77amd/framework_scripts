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

## Bug 2 — HIP Module Exhaustion at compilation `[0/7]` (FIXED)

### Status: FIXED — Patch 2 (static launcher module unload) resolves the crash

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

The v1 patch tried to address this with `gc.collect()` alone, but
`PyCodeCache` class-level collections held strong references to the
benchmark modules, preventing GC from collecting the `CompiledKernel`
objects. The v2 patch (`gc_module_unload_v2.patch`) first evicts losing
benchmark modules from `PyCodeCache`, severing the reference chains, then
calls `gc.collect()` to ensure `CompiledKernel.__del__` fires promptly.

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

**Summary of all fixes and their dependencies:**

| Fix | What it does | Standalone? |
|---|---|---|
| **Triton PR #9444** | Adds `CompiledKernel.__del__` → `hipModuleUnload` / `cuModuleUnload` | Necessary but not sufficient (GC fires too late; strong refs prevent collection) |
| **Patch 1 v1** (`gc_module_unload.patch`) | Calls `gc.collect()` after autotuning | **Insufficient** — `PyCodeCache` holds strong refs, preventing GC from collecting `CompiledKernel` objects |
| **Patch 1 v2** (`gc_module_unload_v2.patch`) | Evicts losing benchmark modules from `PyCodeCache`, then `gc.collect()` | **Requires PR #9444** — severs reference chains so GC can collect `CompiledKernel` and trigger `hipModuleUnload` |
| **Patch 2** (`static_launcher_module_unload.patch`) | Returns module handle from C++, adds `__del__` to `StaticallyLaunchedTritonKernel` | **Self-contained and sufficient** — adds its own cleanup, independent of Triton; verified to resolve the full workload crash alone |

**Which fixes cover which module loading path:**

| Module loading path | PR #9444 needed? | Patch 1 v2 (evict + gc) needed? | Patch 2 (C++ fix) needed? |
|---|---|---|---|
| Triton `CompiledKernel` (autotuning benchmarks) | **Yes** (provides `__del__`) | **Yes** (severs refs + forces `__del__`) | No (different path) |
| PyTorch `_StaticCudaLauncher` (winner kernel loading) | No (irrelevant path) | No (no `__del__` to trigger) | **Yes** (adds handle + `__del__`) |

**Practical impact assessment:**

The autotuning benchmark path (row 1) generates the **bulk of module
churn** — each GEMM autotuning round creates ~145 `CompiledKernel` objects,
of which ~144 become garbage. However, Triton PR #9444's
`CompiledKernel.__del__` reclaims these when Python's GC eventually runs.
The temporary pressure from stale autotuning modules causes non-fatal
errors that Inductor catches ("Ignoring this choice"), but does not crash
the process.

The winner kernel path (row 2) loads far fewer modules (~50 per compilation
cycle), but **before Patch 2, these modules were permanently leaked** —
the `CUmodule` handle was discarded in C++ with no way to ever unload it.
Over 7+ compilation cycles, these ~350+ permanently leaked modules pushed
the total module count past the HIP driver's limit. Because the fatal
crash occurred in `_load_kernel()` (loading the **next** winner kernel),
**Patch 2 alone is sufficient** to resolve it — by enabling cleanup of
winner kernel modules from previous compilation cycles, the persistent
module count stays low enough for the system to survive.

**Patch 2 is both necessary and sufficient for the full workload.**
Testing confirmed that the full `workload_v4.py` with `EXHAUSTIVE`
autotuning passes with Patch 2 alone (no Patch 1 needed). Patch 1 v2
(PyCodeCache eviction + gc.collect) is a useful optimization that reduces
temporary module table pressure, but is not required for correctness when
Patch 2 is applied.

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

### Patch 1 v1: Minimal Fix (PyTorch Python-side) — SUPERSEDED

The original v1 patch simply added `gc.collect()` after `benchmark_fn()`.
This was **necessary but not sufficient**: `PyCodeCache.modules` (a
class-level list) and `PyCodeCache.modules_no_attr` (a class-level dict)
held strong references to the benchmark template modules, which in turn
held `CachingAutotuner` → `compile_results` → `CompiledKernel` reference
chains. These strong references prevented `gc.collect()` from actually
freeing the `CompiledKernel` objects.

Patch file (historical): `gc_module_unload.patch`

### Patch 1 v2: PyCodeCache Eviction + GC Fix — CURRENT

The v2 patch first **evicts losing benchmark template modules from
`PyCodeCache`** to break the strong reference chains, then calls
`gc.collect()`. This makes the `CompiledKernel` objects truly unreachable
so GC can collect them and trigger `hipModuleUnload` / `cuModuleUnload`
via `__del__`.

Patch file: `gc_module_unload_v2.patch`

```diff
--- a/torch/_inductor/select_algorithm.py
+++ b/torch/_inductor/select_algorithm.py
@@ -2,6 +2,7 @@
 import contextlib
 import dataclasses
 import functools
+import gc
 import hashlib
 import inspect
 import itertools
@@ -3338,7 +3339,30 @@ class AlgorithmSelectorCache(PersistentCache):
         )
         # `benchmark_fn(choices)` will execute each choice, and return a dict[choice, timing] which
         # maps each choice to its runtime, calculated by the specified benchmarker, in milliseconds
-        return benchmark_fn(choices)
+        result = benchmark_fn(choices)
+
+        # Evict benchmark template modules from PyCodeCache so their
+        # CachingAutotuner → compile_results → CompiledKernel chains become
+        # unreachable.  Without this, PyCodeCache.modules (a class-level list)
+        # and PyCodeCache.modules_no_attr (a class-level dict) hold strong
+        # references that prevent GC from ever collecting the CompiledKernel
+        # objects and triggering hipModuleUnload / cuModuleUnload via __del__.
+        evict_paths = set()
+        for choice in choices:
+            bmreq = getattr(choice, "bmreq", None)
+            if bmreq is not None:
+                path = getattr(bmreq, "module_path", None)
+                if path is not None:
+                    evict_paths.add(path)
+        if evict_paths:
+            for path in evict_paths:
+                PyCodeCache.modules_no_attr.pop(path, None)
+            PyCodeCache.modules[:] = [
+                m for m in PyCodeCache.modules
+                if getattr(m, "__file__", None) not in evict_paths
+            ]
+
+        gc.collect()
+
+        return result
 
     def autotune(
         self,
```

**Why v2 is needed over v1:**

| Issue | v1 (`gc.collect()` only) | v2 (evict + `gc.collect()`) |
|---|---|---|
| Breaks `PyCodeCache` strong refs | No — modules remain in `PyCodeCache.modules` and `modules_no_attr` | **Yes** — evicts losing benchmark modules |
| `CompiledKernel` objects collectible | No — still reachable via `PyCodeCache` → module → `CachingAutotuner` → `compile_results` → `CompiledKernel` | **Yes** — reference chain severed |
| `hipModuleUnload` actually fires | Partial — only collects objects not held by `PyCodeCache` | **Yes** — all stale `CompiledKernel` objects collected |

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

| Test | Patch | COMPILE_THREADS | `[0/7]` result | Total time |
|---|---|---|---|---|
| Default (32 threads) | None | 32 | **CRASH** | ~5 min |
| Single-threaded | None | 1 | Pass | ~15 min |
| Reduced threads | None | 4 | Pass | ~10 min |
| Default + GC v1 | `gc_module_unload.patch` (gc.collect only) | 32 | **Pass** | ~5 min |
| Default + GC v1 (`repro_bug2_gc.py`) | `gc_module_unload.patch` (gc.collect in repro script) | 32 | **Pass** | ~5 min |
| Default + GC v2 (`workload_v4.py`) | `gc_module_unload_v2.patch` (evict + gc.collect) | 32 | **CRASH** | ~58 min |
| **Patch 2 only** (`workload_v4.py`) | Static launcher module unload (commit `4406fa22`) | 32 | **Pass** | ~52 min |
| **Patch 2 + GC v1** (`workload_v4.py`) | Static launcher + `gc_module_unload.patch` | 32 | **Pass** | ~57 min |

**v1 caveat:** The v1 "Pass" result was on the smaller `repro_bug2_gc.py`
reproducer, which manually calls `gc.collect()` between training iterations.
The full workload (`workload_v4.py`) was never tested with v1.

**v2 full-workload test result (`test_patch_v2.log`):**

The v2 patch **reduced the leak rate** but did **not eliminate the crash**.
The process survived through `[0/6]` but at `[0/7]`:

1. **220 autotuning errors** — individual Triton kernel benchmarks failed
   with error 209 during the `[0/7]` backward autotuning. Inductor caught
   these ("Ignoring this choice") and continued, but by the end **all**
   remaining kernel variants showed `inf ms` (all failed).

2. **Fatal crash in `_StaticCudaLauncher._load_kernel()`** — after
   autotuning completed, Inductor tried to load the winner kernels via
   `static_triton_launcher.py:144 → self.C_impl._load_kernel()`. By this
   point the HIP module table was fully exhausted and the load failed
   fatally with `CUDA driver error: 209`.

Stack trace (crash point):
```
triton_heuristics.py:503, in precompile → self._make_launchers()
triton_heuristics.py:659, in _make_launchers → result.make_launcher()
triton_heuristics.py:1795, in make_launcher → self.kernel.load_kernel(device)
static_triton_launcher.py:144, in load_kernel
    (self.function, self.n_regs, self.n_spills) = self.C_impl._load_kernel(...)
InductorError: RuntimeError: CUDA driver error: 209
```

**Why v2 was insufficient:**

The `PyCodeCache` eviction + `gc.collect()` breaks one reference chain, but
there are likely **additional strong reference holders** keeping stale
`CompiledKernel` objects alive across autotuning rounds. Possible sources:

- **`CachingAutotuner.compile_results`** or other caches internal to
  `triton_heuristics.py` that accumulate across compilations
- **`async_compile` futures** holding references to compiled kernels
- **`_StaticCudaLauncher`** winner kernel modules accumulating (this path
  has no unload mechanism at all — addressed by Patch 2)
- Other `PyCodeCache` entries loaded outside the `benchmark_fn()` path
  (e.g. the final compiled graph module loaded via `load_by_key_path`)

### Why Patch 2 Alone Resolves the Problem

Testing confirmed that Patch 2 (commit `4406fa22`, the static launcher
module unload fix) is **sufficient on its own** to resolve the crash.
The full `workload_v4.py` workload with `EXHAUSTIVE` autotuning completes
successfully with Patch 2 alone, with **no errors** in the log. Adding
Patch 1 (gc.collect) on top of Patch 2 also passes, but is not required.

The reason Patch 2 alone works comes down to **which code path actually
causes the fatal crash**:

1. **Autotuning benchmarks** (Triton `CompiledKernel` path) generate the
   **bulk** of the module churn — ~145 kernel variants per GEMM, ~144 of
   which become garbage each round. However, Triton PR #9444 already added
   `CompiledKernel.__del__` with `hipModuleUnload`. Even though Python's GC
   doesn't run frequently enough to keep the module table completely clean,
   the GC **does** eventually run between compilation cycles, reclaiming
   enough modules to stay below the driver's limit. Autotuning errors from
   temporary table pressure are caught by Inductor ("Ignoring this choice")
   and are non-fatal.

2. **Winner kernel loading** (`StaticallyLaunchedTritonKernel` path) loads
   far fewer modules (~50 per compilation cycle), but **before Patch 2,
   these modules were never unloaded**. The C++ `loadKernel()` function
   discarded the `CUmodule` handle, making cleanup impossible. Over 7+
   compilation cycles, these permanently leaked modules accumulated on top
   of whatever temporary pressure existed from autotuning. When the
   combined total exceeded the HIP driver's module table capacity, the
   **next winner kernel load** failed fatally — this was the crash point
   observed in the v2 test (`static_triton_launcher.py:144`).

Patch 2 fixes this by:
- Returning the `CUmodule` handle from `loadKernel()` back to Python
- Storing it on `StaticallyLaunchedTritonKernel.module`
- Adding `__del__` to call `hipModuleUnload`/`cuModuleUnload` when the
  kernel object is garbage-collected (e.g., when a new compilation cycle
  replaces the old compiled graph)

With Patch 2, winner kernel modules from previous compilation cycles are
properly unloaded when they become unreachable. This keeps the persistent
module count low enough that even without explicit `gc.collect()` calls in
the autotuning path, the HIP driver's module table never fills up.

```bash
# v2 test (crashed at [0/7]):
rm -rf /tmp/torchinductor_root
HIP_VISIBLE_DEVICES=7 TRITON_HIP_USE_ASYNC_COPY=0 \
  TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE=EXHAUSTIVE \
  python workload_v4.py --compile --inductor --no-cudagraphs --dtypes bfloat16
```

Log: `test_patch_v2.log`

```bash
# Patch 2 verification (passed — run via run_autotune_launcher_patch.sh):
rm -rf /tmp/torchinductor_root
HIP_VISIBLE_DEVICES=6 TRITON_HIP_USE_ASYNC_COPY=0 \
  TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE=EXHAUSTIVE \
  python workload_v4.py --compile --inductor --no-cudagraphs --dtypes bfloat16
```

Log: `test_with_launcher_patched.log`

**Verification protocol:** The full workload was run twice with Patch 2
compiled into PyTorch — once with Patch 1 (gc.collect) also applied, and
once without. Both runs completed with zero errors, confirming Patch 2
alone is sufficient.

Note: `COMPILE_THREADS=4` also showed a race condition in Triton's
compilation cache (`FileNotFoundError` on `.source` files and
`AttributeError: 'CompiledKernel' object has no attribute 'module'` in
`__del__`) but still completed successfully. This is a separate Triton bug.

---

## Environment Notes

### `TRITON_HIP_USE_ASYNC_COPY` — historical requirement, now conditional

Historical baseline evidence showed a separate non-module-exhaustion failure
mode when async copy was enabled on gfx950. In that baseline, omitting
`TRITON_HIP_USE_ASYNC_COPY=0` could crash during coordinate-descent tuning of
pointwise kernels with the same driver error code (`209`) but a different path.

However, the latest matrix with the current patch stack revalidated this
assumption and produced a different outcome (see Runs V/W below):

- `TRITON_HIP_USE_ASYNC_COPY` explicitly **unset**
- Static launcher unload fix present (PR #176988 / commit `4406fa22`)
- Python v3 benchmark-module cleanup present (`select_algorithm.py`)
- `waves_per_eu=0` forced in ROCm GEMM heuristics
- `WORKLOAD_NETWORKS="large extralarge"`
- Both `STATIC_LAUNCHER=1` and `STATIC_LAUNCHER=0` passed to completion
  with no `209`/`no kernel image` failures.

Current conclusion:

- For the **current patched stack and reproducer**, forcing
  `TRITON_HIP_USE_ASYNC_COPY=0` is **not required**.
- Keep it as a troubleshooting knob for older baselines or different kernels /
  workloads where async-copy instruction compatibility may still regress.

## Workarounds

| Workaround | Change | Trade-off |
|---|---|---|
| **Static launcher module unload patch** (recommended) | Apply Patch 2 — commit `4406fa22` to PyTorch (requires C++ rebuild) | No runtime overhead; proper resource lifecycle management |
| PyCodeCache eviction + gc.collect() patch | Apply `gc_module_unload_v2.patch` to PyTorch (Python-only, no rebuild) | Adds ~ms of GC overhead per autotuning round; insufficient alone for full workload |
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

---

## 2026-03-17 Restart Investigation (PR #9444 only) — In Progress

The investigation was restarted from a baseline with **Triton PR #9444 applied**
and no additional local fixes.

### Fixed run constraints for restart

- `TRITON_HIP_USE_ASYNC_COPY=0` (kept fixed)
- `--no-cudagraphs` (kept fixed)
- `--padded` (kept fixed)
- `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE=EXHAUSTIVE` (kept fixed)
- Added `--dynamic false` to increase recompilation pressure where applicable

### Script used

All tests run through `run_workload.sh`. The script was updated to:

- Accept `STATIC_LAUNCHER` and optional `TORCH_COMPILE_THREADS_VALUE`
- Keep the required fixed flags/envs above
- Record periodic CPU memory snapshots to `*.cpu-mem.log`
- Track Python process peak RSS (`peak_rss_kb`)

### Run A (first pass): static launcher enabled

Command:

```bash
STATIC_LAUNCHER=1 LOG_PREFIX=restart-investigation-launcher1 bash run_workload.sh
```

Artifacts:

- `restart-investigation-launcher1.log`
- `restart-investigation-launcher1.cpu-mem.log`

Result: **FAILED** (reproduced `hipErrorNoBinaryForGpu` / error 209)

Runtime and memory:

- Total elapsed: ~48 minutes (`small` + `medium` completed; failure at start of next compile phase)
- Peak Python RSS: `14305152` KB (~13.64 GiB)
- Host memory remained far from exhaustion (hundreds of GB still available)

Observed progression:

- Long compile/autotune phase with exhaustive GEMM search spaces
- Non-fatal Triton resource rejections first appeared:
  - `No valid triton configs. OutOfMemoryError: out of resource: triton_mm`
  - `Required: 196608 Hardware limit: 163840`
  - Logged as `Ignoring this choice`, run continued
- Later, repeated runtime autotune errors appeared:
  - `CUDA driver error: 209`
  - `CUDA error: no kernel image is available for execution on the device`
  - `hipErrorNoBinaryForGpu` hint in log

Fatal crash point:

- Exception raised as:
  - `torch._inductor.exc.InductorError: AcceleratorError: CUDA error: no kernel image is available for execution on the device`
- Failure surfaced while benchmarking fused Triton nodes (`scheduler.py` -> `triton.py` benchmark path), during `rand_strided(..., device='cuda:0', dtype=torch.bfloat16)` argument creation for a generated autotune module in `/tmp/torchinductor_test/...`.

Notes:

- `run_workload.sh` had a logging-only quoting bug in the `peak_rss_gb` summary line after Run A completed; this was fixed immediately after Run A. It did not affect the workload execution itself.

### Run B (next step): static launcher disabled

Per test plan after reproducing with launcher enabled, the next run uses:

```bash
STATIC_LAUNCHER=0 LOG_PREFIX=restart-investigation-launcher0 bash run_workload.sh
```

Artifacts:

- `restart-investigation-launcher0.log`
- `restart-investigation-launcher0.cpu-mem.log`

Result so far: **NO hipErrorNoBinaryForGpu observed in equivalent window**

- Ran past the launcher=1 failure window (~48 min) without `CUDA error: no kernel image`
- Continued through heavy large-network autotune phases with no 209/no-kernel-image errors
- CPU RSS rose significantly over time:
  - ~14.6 GiB at ~42 min
  - ~24.8 GiB peak (`peak_rss_kb_so_far=25973596`) at ~56 min
- Host memory still had large headroom (no host OOM pressure)

Run B was manually stopped after collecting comparison evidence (to proceed to the
`TORCH_COMPILE_THREADS=1` launcher=1 run requested in the test plan).

### Run C (requested): launcher enabled + single compile thread

Command:

```bash
STATIC_LAUNCHER=1 TORCH_COMPILE_THREADS_VALUE=1 \
  LOG_PREFIX=restart-investigation-launcher1-threads1 \
  bash run_workload.sh
```

Artifacts:

- `restart-investigation-launcher1-threads1.log`
- `restart-investigation-launcher1-threads1.cpu-mem.log`

Result: **FAILED** (reproduced `hipErrorNoBinaryForGpu` / error 209)

Runtime and memory:

- Total elapsed: ~48 minutes (`exit_code=1`)
- Peak Python RSS: `14279332` KB (`peak_rss_gb=13.618`)
- Host memory remained far from exhaustion

Observed progression:

- `small` network compiled/trained, then `medium` completed
  (`(torch.bfloat16, medium)` timing lines present)
- Same warning pattern seen before fatal failure:
  - `No valid triton configs. OutOfMemoryError: out of resource: triton_mm`
  - `Required: 196608 Hardware limit: 163840`
  - Logged as `Ignoring this choice`
- Then repeated runtime autotune errors:
  - `CUDA driver error: 209`
  - `CUDA error: no kernel image is available for execution on the device`

Fatal crash point:

- `torch._inductor.exc.InductorError: AcceleratorError: CUDA error: no kernel image is available for execution on the device`
- Raised in autotune input generation path while creating benchmark tensors
  (`select_algorithm.py` -> `rand_strided` -> `torch.randn`)

Conclusion from Run C:

- Setting `TORCH_COMPILE_THREADS=1` did **not** prevent the failure.
- Failure mode remains consistent with launcher=1 runs.

### Run D (requested follow-up): static launcher disabled to completion

Per latest request, rerun with static launcher disabled and allow it to run until
natural completion (no manual stop), while tracking CPU memory throughout.

Command:

```bash
STATIC_LAUNCHER=0 LOG_PREFIX=restart-investigation-launcher0-complete bash run_workload.sh
```

Artifacts:

- `restart-investigation-launcher0-complete.log`
- `restart-investigation-launcher0-complete.cpu-mem.log`

Result: **FAILED** (reproduced `hipErrorNoBinaryForGpu` / error 209)

Runtime and memory:

- Total elapsed: ~79.8 minutes (`elapsed_ms=4788067`, `exit_code=1`)
- Peak Python RSS: `27010140` KB (`peak_rss_gb=25.759`)
- Host memory remained far from exhaustion (large free memory headroom)

Observed progression:

- Run advanced much farther than the earlier launcher=0 comparison run
  (which had been manually stopped at ~56 minutes).
- Non-fatal autotune warnings persisted during search:
  - `OutOfResources: out of resource: shared memory, Required: 196608, Hardware limit: 163840`
  - Logged as `Ignoring this choice`
- Near failure window (`~00:47:45`), repeated autotune runtime errors appeared:
  - `Triton Error [HIP]: Code: 209, Messsage: no kernel image is available for execution on the device`
  - `CUDA error: no kernel image is available for execution on the device`
  - `hipErrorNoBinaryForGpu`

Fatal crash point:

- Final exception:
  `torch._inductor.exc.InductorError: AcceleratorError: CUDA error: no kernel image is available for execution on the device`
- Raised during autotune input generation (`select_algorithm.py` ->
  `benchmark_example_value` -> `rand_strided` -> `torch.randn`), while compiling
  backward graph.

Conclusion from Run D:

- Disabling static launcher (`STATIC_LAUNCHER=0`) does **not** eliminate the
  failure under longer runtime with `--dynamic false`.
- It appears to delay onset relative to launcher=1 runs, but the same terminal
  error still occurs eventually.

### Run E (narrowing): `--networks large extralarge`, static launcher enabled

Per narrowing request, rerun with only `large` and `extralarge` networks while
keeping all fixed constraints.

Command:

```bash
STATIC_LAUNCHER=1 WORKLOAD_NETWORKS="large extralarge" \
  LOG_PREFIX=narrow-large-extralarge-launcher1 \
  bash run_workload.sh
```

Artifacts:

- `narrow-large-extralarge-launcher1.log`
- `narrow-large-extralarge-launcher1.cpu-mem.log`

Result: **FAILED** (reproduced `hipErrorNoBinaryForGpu` / error 209)

Runtime and memory:

- Total elapsed: ~41.6 minutes (`elapsed_ms=2498300`, `exit_code=1`)
- Peak Python RSS: `15656908` KB (`peak_rss_gb=14.932`)
- Host memory remained far from exhaustion

Observed progression:

- `large` completed its timing iterations successfully.
- During subsequent compile/autotune work, the run failed with:
  - `CUDA error: no kernel image is available for execution on the device`
  - `hipErrorNoBinaryForGpu`

Fatal crash point:

- Trace terminates in:
  - `torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device`

### Run F (narrowing): `--networks large extralarge`, static launcher disabled

Follow-up run with static launcher disabled under the same narrowed network set.

Command:

```bash
STATIC_LAUNCHER=0 WORKLOAD_NETWORKS="large extralarge" \
  LOG_PREFIX=narrow-large-extralarge-launcher0 \
  bash run_workload.sh
```

Artifacts:

- `narrow-large-extralarge-launcher0.log`
- `narrow-large-extralarge-launcher0.cpu-mem.log`

Result: **FAILED** (reproduced `hipErrorNoBinaryForGpu` / error 209)

Runtime and memory:

- Total elapsed: ~51.1 minutes (`elapsed_ms=3066823`, `exit_code=1`)
- Peak Python RSS: `33476464` KB (`peak_rss_gb=31.926`)
- Host memory remained far from exhaustion

Observed progression:

- Repeated autotune-time runtime failures appeared first:
  - `Triton Error [HIP]: Code: 209, Messsage: no kernel image is available for execution on the device`
  - `CUDA error: no kernel image is available for execution on the device`
  - `hipErrorNoBinaryForGpu`
- These were followed by a fatal inductor exception.

Fatal crash point:

- Final exception:
  - `torch._inductor.exc.InductorError: AcceleratorError: CUDA error: no kernel image is available for execution on the device`

Conclusion from Runs E/F:

- Narrowing to `large + extralarge` is sufficient to reproduce the failure for
  **both** `STATIC_LAUNCHER=1` and `STATIC_LAUNCHER=0`.
- `STATIC_LAUNCHER=0` again delays failure but does not eliminate it, and
  reached a much higher peak CPU RSS in this narrowed setup.

### Run G (narrowing): `--networks extralarge`, static launcher enabled

Next narrowing step with only `extralarge` under launcher enabled.

Command:

```bash
STATIC_LAUNCHER=1 WORKLOAD_NETWORKS="extralarge" \
  LOG_PREFIX=narrow-extralarge-launcher1 \
  bash run_workload.sh
```

Artifacts:

- `narrow-extralarge-launcher1.log`
- `narrow-extralarge-launcher1.cpu-mem.log`

Result: **FAILED**

Runtime and memory:

- Total elapsed: ~25.0 minutes (`elapsed_ms=1499015`, `exit_code=1`)
- Peak Python RSS: `12380332` KB (`peak_rss_gb=11.807`)
- Host memory remained far from exhaustion

Observed progression:

- Repeated autotune-time runtime failures appeared:
  - `CUDA driver error: 209`
  - `CUDA error: no kernel image is available for execution on the device`
  - `hipErrorNoBinaryForGpu`
- After the 209/no-kernel-image phase, additional runtime errors appeared:
  - `CUDA driver error: 1`
  - `CUDA error: invalid argument` (`hipErrorInvalidValue`)

Fatal crash point:

- Final exception:
  - `torch._inductor.exc.InductorError: AcceleratorError: CUDA error: invalid argument`
- Raised during autotune benchmark argument creation
  (`torch/_dynamo/testing.py::rand_strided` -> `torch.randn`) while benchmarking
  fused Triton nodes.

Interpretation:

- `extralarge` alone is sufficient to trigger the same instability with launcher
  enabled, including explicit 209/no-kernel-image errors before terminal failure.

### Run H (narrowing): `--networks extralarge`, static launcher disabled

Follow-up run with launcher disabled on `extralarge` only.

Command:

```bash
STATIC_LAUNCHER=0 WORKLOAD_NETWORKS="extralarge" \
  LOG_PREFIX=narrow-extralarge-launcher0 \
  bash run_workload.sh
```

Artifacts:

- `narrow-extralarge-launcher0.log`
- `narrow-extralarge-launcher0.cpu-mem.log`

Result: **PASSED** (no `hipErrorNoBinaryForGpu`, no traceback)

Runtime and memory:

- Total elapsed: ~32.9 minutes (`elapsed_ms=1975349`, `exit_code=0`)
- Peak Python RSS: `26180828` KB (`peak_rss_gb=24.968`)
- Host memory remained far from exhaustion

Observed progression:

- Run completed and printed final results table:
  - `extralarge bfloat16 median = 0.049732 s`
- Non-fatal autotune pruning warnings were still present:
  - `OutOfResources: out of resource: shared memory ... Ignoring this choice`
- No `CUDA driver error: 209`, no `no kernel image`, and no
  `InductorError` were observed in this run.

Conclusion from Runs G/H (`extralarge` only):

- `STATIC_LAUNCHER=1`: **FAILED** with explicit 209/no-kernel-image before
  terminal failure.
- `STATIC_LAUNCHER=0`: **PASSED** to completion in this run.
- This creates a sharper split for `extralarge`-only than for
  `large+extralarge`, where both launcher settings failed.

### Run I (further isolation): `--networks large`, static launcher enabled

After confirming `extralarge` split behavior, run `large` only with launcher
enabled.

Command:

```bash
STATIC_LAUNCHER=1 WORKLOAD_NETWORKS="large" \
  LOG_PREFIX=narrow-large-launcher1 \
  bash run_workload.sh
```

Artifacts:

- `narrow-large-launcher1.log`
- `narrow-large-launcher1.cpu-mem.log`

Result: **FAILED** (reproduced `hipErrorNoBinaryForGpu` / no-kernel-image)

Runtime and memory:

- Total elapsed: ~42.2 minutes (`elapsed_ms=2529696`, `exit_code=1`)
- Peak Python RSS: `15673640` KB (`peak_rss_gb=14.948`)
- Host memory remained far from exhaustion

Observed progression:

- Long autotune/compile phase ran for ~40+ minutes before failure.
- Near the failure window, traceback terminates with:
  - `torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device`
  - `hipErrorNoBinaryForGpu`

Conclusion from Run I:

- `large` alone is sufficient to reproduce the no-kernel-image failure with
  launcher enabled.

### Run J (further isolation): `--networks large`, static launcher disabled

Companion run completed with launcher disabled.

Command:

```bash
STATIC_LAUNCHER=0 WORKLOAD_NETWORKS="large" \
  LOG_PREFIX=narrow-large-launcher0 \
  bash run_workload.sh
```

Artifacts:

- `narrow-large-launcher0.log`
- `narrow-large-launcher0.cpu-mem.log`

Result: **PASSED** (no `hipErrorNoBinaryForGpu`, no traceback)

Runtime and memory:

- Total elapsed: ~42.3 minutes (`elapsed_ms=2535809`, `exit_code=0`)
- Peak Python RSS: `21403744` KB (`peak_rss_gb=20.412`)
- Host memory remained far from exhaustion

Observed progression:

- Run crossed the launcher=1 failure window (~42.2 min) and completed normally.
- Final results table was printed:
  - `large bfloat16 median = 0.037159 s`
- Non-fatal autotune pruning warnings still appeared:
  - `OutOfResources: out of resource: shared memory ... Ignoring this choice`
- No `CUDA driver error: 209`, no `no kernel image`, and no
  `InductorError` were observed.

Conclusion from Runs I/J (`large` only):

- `STATIC_LAUNCHER=1`: **FAILED** with no-kernel-image (`hipErrorNoBinaryForGpu`)
- `STATIC_LAUNCHER=0`: **PASSED** to completion in this run.

### Run K (order sensitivity): `--networks extralarge large`, static launcher disabled

To test whether the launcher=0 failure on multi-network runs depends on network
ordering, rerun with reversed order.

Command:

```bash
STATIC_LAUNCHER=0 WORKLOAD_NETWORKS="extralarge large" \
  LOG_PREFIX=narrow-extralarge-large-launcher0 \
  bash run_workload.sh
```

Artifacts:

- `narrow-extralarge-large-launcher0.log`
- `narrow-extralarge-large-launcher0.cpu-mem.log`

Result: **FAILED** (reproduced `hipErrorNoBinaryForGpu` / no-kernel-image)

Runtime and memory:

- Total elapsed: ~35.6 minutes (`elapsed_ms=2135214`, `exit_code=1`)
- Peak Python RSS: `28564076` KB (`peak_rss_gb=27.241`)
- Host memory remained far from exhaustion

Observed progression:

- `extralarge` completed its timing loop first:
  - `(torch.bfloat16, extralarge) took 0.04939..0.05098 s`
- During the subsequent compile/autotune phase (`[0/1]`), repeated runtime
  errors appeared:
  - `Triton Error [HIP]: Code: 209, Messsage: no kernel image is available for execution on the device`
  - `CUDA error: no kernel image is available for execution on the device`
  - `hipErrorNoBinaryForGpu`
- Final exception:
  - `torch._inductor.exc.InductorError: AcceleratorError: CUDA error: no kernel image is available for execution on the device`
  - Raised while creating autotune benchmark arguments (`rand_strided` ->
    `torch.randn`) for a generated module in `/tmp/torchinductor_test/...`.

Conclusion from launcher=0 multi-network runs:

- `large + extralarge` (`Run F`): **FAILED**
- `extralarge + large` (`Run K`): **FAILED**
- Failure occurs in both orders under launcher=0; reversing order changed
  time-to-failure (Run K failed earlier than Run F), but did not eliminate the
  no-kernel-image failure.

### Run L (launcher=0 + single compile thread): `--networks extralarge large`

To test whether compile concurrency changes the launcher=0 multi-network
failure mode, rerun Run K with `TORCH_COMPILE_THREADS=1`.

Command:

```bash
STATIC_LAUNCHER=0 TORCH_COMPILE_THREADS_VALUE=1 \
  WORKLOAD_NETWORKS="extralarge large" \
  LOG_PREFIX=narrow-extralarge-large-launcher0-threads1 \
  bash run_workload.sh
```

Artifacts:

- `narrow-extralarge-large-launcher0-threads1.log`
- `narrow-extralarge-large-launcher0-threads1.cpu-mem.log`

Result: **FAILED**

Runtime and memory:

- Total elapsed: ~35.8 minutes (`elapsed_ms=2147718`, `exit_code=1`)
- Peak Python RSS: `28544076` KB (`peak_rss_gb=27.222`)
- Host memory remained far from exhaustion

Observed progression:

- Repeated autotune runtime errors appeared at `[0/1]`:
  - `Triton Error [HIP]: Code: 209, Messsage: no kernel image is available for execution on the device`
  - `CUDA error: no kernel image is available for execution on the device`
  - `hipErrorNoBinaryForGpu`
- `extralarge` timing iterations still completed first:
  - `(torch.bfloat16, extralarge) took 0.04942..0.05150 s`
- Terminal exception in this run was:
  - `torch._inductor.exc.InductorError: MemoryError`
  - Raised from `torch/_inductor/codecache.py::update_local_cache` during
    `json.dumps({"system": self.system, "cache": local_cache}, indent=4)`.

Conclusion from Runs K/L (`extralarge + large`, launcher=0):

- `TORCH_COMPILE_THREADS` default (Run K): **FAILED** with terminal
  no-kernel-image `InductorError`.
- `TORCH_COMPILE_THREADS=1` (Run L): **FAILED** with terminal `MemoryError`,
  but only after the same repeated `209/no-kernel-image` sequence.
- Lowering compile threads did not prevent the launcher=0 multi-network failure
  and did not materially change time-to-failure or peak RSS in this setup.

### Run M (launcher=0 + single compile thread): `--networks large extralarge`

Symmetric check for compile-thread impact with original network order
(`large` then `extralarge`).

Command:

```bash
STATIC_LAUNCHER=0 TORCH_COMPILE_THREADS_VALUE=1 \
  WORKLOAD_NETWORKS="large extralarge" \
  LOG_PREFIX=narrow-large-extralarge-launcher0-threads1 \
  bash run_workload.sh
```

Artifacts:

- `narrow-large-extralarge-launcher0-threads1.log`
- `narrow-large-extralarge-launcher0-threads1.cpu-mem.log`

Result: **FAILED** (reproduced `hipErrorNoBinaryForGpu` / no-kernel-image)

Runtime and memory:

- Total elapsed: ~51.5 minutes (`elapsed_ms=3088589`, `exit_code=1`)
- Peak Python RSS: `33483064` KB (`peak_rss_gb=31.932`)
- Host memory remained far from exhaustion

Observed progression:

- `large` timing iterations completed first:
  - `(torch.bfloat16, large) took 0.03695..0.03699 s`
- During subsequent compile/autotune work (`[0/1]`), repeated runtime errors
  appeared:
  - `Triton Error [HIP]: Code: 209, Messsage: no kernel image is available for execution on the device`
  - `CUDA error: no kernel image is available for execution on the device`
  - `hipErrorNoBinaryForGpu`
- Final exception:
  - `torch._inductor.exc.InductorError: AcceleratorError: CUDA error: no kernel image is available for execution on the device`
  - Raised in backward compile/autotune path while creating benchmark tensors
    (`rand_strided((960000, 256), ...)` -> `torch.randn`).

Conclusion from Runs F/M (`large + extralarge`, launcher=0):

- `TORCH_COMPILE_THREADS` default (Run F): **FAILED**
- `TORCH_COMPILE_THREADS=1` (Run M): **FAILED**
- Lowering compile threads did not prevent launcher=0 failure for this
  multi-network order, and produced similar time-to-failure / peak RSS.

### Run N (cumulative-control): `--networks large large`, launcher disabled

To test whether launcher=0 failures are caused by simply running two large
iterations in one process (vs. running two different large models), rerun with
the same network repeated.

Command:

```bash
STATIC_LAUNCHER=0 WORKLOAD_NETWORKS="large large" \
  LOG_PREFIX=narrow-large-large-launcher0 \
  bash run_workload.sh
```

Artifacts:

- `narrow-large-large-launcher0.log`
- `narrow-large-large-launcher0.cpu-mem.log`

Result: **PASSED** (`exit_code=0`)

Runtime and memory:

- Total elapsed: ~42.0 minutes (`elapsed_ms=2520108`)
- Peak Python RSS: `21428556` KB (`peak_rss_gb=20.436`)
- Host memory remained far from exhaustion

Observed progression:

- Both `large` entries completed and final results table was printed:
  - `large 2400000 0.037183`
  - `large 2400000 0.037183`
- No `CUDA driver error: 209`, no `no kernel image`, and no
  `InductorError` were observed.
- RSS stayed comparatively modest through most of the run, with a late jump to
  ~21.4 GiB near completion.

Conclusion from Run N:

- Repeating the same `large` network twice under launcher=0 did **not**
  reproduce the failure.
- Current evidence points more strongly to cross-network compile churn from
  `large + extralarge` (in either order), rather than merely executing two
  `large` passes in one process.

### Run O (cumulative-control): `--networks extralarge extralarge`, launcher disabled

To mirror Run N with the other suspected network, repeat `extralarge` twice
under launcher=0.

Command:

```bash
STATIC_LAUNCHER=0 WORKLOAD_NETWORKS="extralarge extralarge" \
  LOG_PREFIX=narrow-extralarge-extralarge-launcher0 \
  bash run_workload.sh
```

Artifacts:

- `narrow-extralarge-extralarge-launcher0.log`
- `narrow-extralarge-extralarge-launcher0.cpu-mem.log`

Result: **PASSED** (`exit_code=0`)

Runtime and memory:

- Total elapsed: ~32.3 minutes (`elapsed_ms=1935350`)
- Peak Python RSS: `26177632` KB (`peak_rss_gb=24.965`)

Observed progression:

- Final results table printed both repeated entries:
  - `extralarge 960000 0.049433`
  - `extralarge 960000 0.049433`
- No `CUDA driver error: 209`, no `no kernel image`, and no
  `InductorError` were observed.
- Peak RSS is nearly identical to prior launcher=0 single-`extralarge`
  passing run (Run H: ~24.968 GiB), rather than increasing toward mixed-network
  failure peaks.

Conclusion from Run O:

- Repeating `extralarge` alone twice under launcher=0 did **not** reproduce the
  failure.
- Together with Run N (`large large` pass), this strengthens the hypothesis
  that launcher=0 failures are tied to **mixed-network compile churn**
  (`large` + `extralarge` in one process), not simple same-network repetition.
- Caveat: repeated identical networks likely benefit from compile/cache reuse,
  so they may not stress the same path as mixed-network runs.

## Code-path drilldown for launcher=0 failures

This section focuses on why `STATIC_LAUNCHER=0` can still fail with
`hipErrorNoBinaryForGpu` even with Triton explicit unload support from PR #928.

### What the failure stack says

Across launcher=0 mixed-network failures (Runs F/K/M), the terminal traceback is
consistently in Inductor autotune/benchmark compilation, not in steady-state
model execution:

- `scheduler.py` fusion benchmarking
- `codegen/triton.py::benchmark_codegened_module`
- generated module `get_args()` -> `rand_strided(...)` -> `torch.randn(...)`

This indicates an already-corrupted GPU module load state by the time benchmark
argument generation runs, after repeated autotune runtime `209/no kernel image`
errors.

### Why PR #928 may be insufficient in this path

PR #928 behavior (present in this environment):

- `triton.compiler.CompiledKernel.__del__` unloads module via
  `driver.active.utils.unload_module(self.module)`.

However, in the Inductor runtime path used by launcher=0:

1. `CachingAutotuner._make_launchers()` builds launchers for **all**
   precompiled configs (often hundreds to >1000 per kernel with EXHAUSTIVE).
2. Each `TritonCompileResult.make_launcher()` places the compiled binary in
   launcher scope (`scope["bin"] = binary`) and calls `binary._init_handles()`,
   which loads the GPU module.
3. `autotune_to_one_config()` narrows `self.launchers` to one winner, but
   `self.compile_results` remains populated with all compiled candidates.
4. `PyCodeCache` keeps loaded Python modules strongly referenced in
   `PyCodeCache.modules`, and those modules keep `CachingAutotuner` instances
   alive.

Net effect:

- `CompiledKernel` objects often stay live for long process lifetimes.
- `CompiledKernel.__del__` is not called promptly, so module unload does not
  happen fast enough to prevent cumulative module-table growth.

### Why this matches observed behavior

- Single-network launcher=0 passes (`large`, `extralarge`, `large large`,
  `extralarge extralarge`) imply reuse/caching can stay below failure threshold.
- Mixed-network launcher=0 failures (`large + extralarge`, either order) imply
  additional unique compile/autotune churn pushes process lifetime module count
  over threshold.
- `TORCH_COMPILE_THREADS=1` does not fix this because the issue appears tied to
  object/module lifetime retention, not compile-thread concurrency.

### Separate static-launcher note

Current `torch/_inductor/runtime/static_triton_launcher.py` in this tree does
not currently expose a Python-side destructor unloading path, and static
launcher C++ bindings here expose `_load_kernel` / `_launch_kernel` but no
companion unload entrypoint. This is consistent with launcher=1 failing faster,
but distinct from the launcher=0 analysis above.

## 2026-03-19 Matrix: minimal-invasive sequence (in progress)

Per latest plan, the sequence is:

1. Test autotune subprocess mode with no local patch.
2. Disable autotune subprocess and add code patches incrementally (`v1` -> `v2` -> `v3`) until pass.
3. For each passing config from (1)/(2), retest with `STATIC_LAUNCHER=1`.

Updated objective:

- Find at least one configuration that passes with both `STATIC_LAUNCHER=0` and
  `STATIC_LAUNCHER=1`.
- Prefer a non-subprocess configuration (lower overhead).
- If no patch-based non-subprocess config succeeds, run no-patch/no-subprocess
  diagnostics with `AMD_LOG_LEVEL=3`.

### Run P (step 1): subprocess autotune, no local patch

Command:

```bash
STATIC_LAUNCHER=0 WORKLOAD_NETWORKS="large extralarge" \
  TORCHINDUCTOR_RELEASE_UNUSED_TRITON_KERNELS=0 \
  TORCHINDUCTOR_RELEASE_UNUSED_TRITON_KERNELS_GC=0 \
  TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC=1 \
  LOG_PREFIX=matrix-subproc1-release0-launcher0 \
  bash run_workload.sh
```

Artifacts:

- `matrix-subproc1-release0-launcher0.log`
- `matrix-subproc1-release0-launcher0.cpu-mem.log`

Result: **PASSED** (`exit_code=0`)

Runtime and memory:

- Total elapsed: ~75.96 minutes (`elapsed_ms=4557799`)
- Peak Python RSS: `19503604` KB (`peak_rss_gb=18.600`)

Observed behavior:

- No `CUDA driver error: 209`
- No `CUDA error: no kernel image is available for execution on the device`
- No `hipErrorNoBinaryForGpu`
- No traceback
- Final timing lines printed for both requested networks:
  - `(torch.bfloat16, large)` timing block present
  - `(torch.bfloat16, extralarge)` timing block present

Interpretation:

- Autotune subprocess mode alone is a strong candidate for a minimally invasive mitigation for the launcher=0 mixed-network failure mode.
- Trade-off is substantially longer compile/autotune wall clock (expected from subprocess benchmarking and process handoff overhead).

### Run Q (step 2 start): no subprocess + patch v1 (`gc.collect()` only)

Patch applied:

- `gc_module_unload.patch` semantics in
  `torch/_inductor/select_algorithm.py`:
  - `import gc`
  - after `result = benchmark_fn(choices)`, call `gc.collect()`

Command:

```bash
STATIC_LAUNCHER=0 WORKLOAD_NETWORKS="large extralarge" \
  TORCHINDUCTOR_RELEASE_UNUSED_TRITON_KERNELS=0 \
  TORCHINDUCTOR_RELEASE_UNUSED_TRITON_KERNELS_GC=0 \
  TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC=0 \
  LOG_PREFIX=matrix-nosubproc-v1-launcher0 \
  bash run_workload.sh
```

Result: **FAILED** (`exit_code=1`)

Runtime and memory:

- Total elapsed: ~56.17 minutes (`elapsed_ms=3370080`)
- Peak Python RSS: `33523780` KB (`peak_rss_gb=31.971`)

Observed behavior:

- Run remained clean through ~50 minutes, then emitted repeated autotune-time:
  - `Triton Error [HIP]: Code: 209`
  - `CUDA error: no kernel image is available for execution on the device`
  - `hipErrorNoBinaryForGpu`
- `large` timing iterations completed before failure.
- Terminal traceback ended with:
  - `torch._inductor.exc.InductorError: AcceleratorError: CUDA error: no kernel image is available for execution on the device`

Conclusion:

- `v1` (`gc.collect()` only) is insufficient for launcher=0 mixed-network case
  under no-subprocess autotuning.

### Run R (step 2): no subprocess + patch v2 (PyCodeCache eviction + `gc.collect()`)

Patch applied:

- `gc_module_unload_v2.patch` semantics in
  `torch/_inductor/select_algorithm.py`:
  - keep `result = benchmark_fn(choices)`
  - collect `choice.bmreq.module_path` entries
  - evict those paths from `PyCodeCache.modules_no_attr` and `PyCodeCache.modules`
  - call `gc.collect()`

Command:

```bash
STATIC_LAUNCHER=0 WORKLOAD_NETWORKS="large extralarge" \
  TORCHINDUCTOR_RELEASE_UNUSED_TRITON_KERNELS=0 \
  TORCHINDUCTOR_RELEASE_UNUSED_TRITON_KERNELS_GC=0 \
  TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC=0 \
  LOG_PREFIX=matrix-nosubproc-v2-launcher0 \
  bash run_workload.sh
```

Result: **FAILED** (`exit_code=1`)

Runtime and memory:

- Total elapsed: ~54.45 minutes (`elapsed_ms=3267201`)
- Peak Python RSS: `34043236` KB (`peak_rss_gb=32.466`)

Observed behavior:

- Same late-window failure signature as v1:
  - repeated `Triton Error [HIP]: Code: 209`
  - repeated `CUDA error: no kernel image is available for execution on the device`
  - repeated `hipErrorNoBinaryForGpu`
- `large` timing iterations completed before failure.
- Terminal traceback ended with:
  - `torch._inductor.exc.InductorError: AcceleratorError: CUDA error: no kernel image is available for execution on the device`

Conclusion:

- `v2` (PyCodeCache eviction + `gc.collect()`) is also insufficient for
  launcher=0 mixed-network case under no-subprocess autotuning.

### Run S (step 2): no subprocess + patch v3

Patch applied:

- `gc_module_unload_v3.patch` semantics in
  `torch/_inductor/select_algorithm.py`:
  - all v2 behavior (module path eviction from PyCodeCache containers)
  - drop matching module names from `sys.modules`
  - call `self.precompile_cache.clear()`
  - call `gc.collect()`

Command:

```bash
STATIC_LAUNCHER=0 WORKLOAD_NETWORKS="large extralarge" \
  TORCHINDUCTOR_RELEASE_UNUSED_TRITON_KERNELS=0 \
  TORCHINDUCTOR_RELEASE_UNUSED_TRITON_KERNELS_GC=0 \
  TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC=0 \
  LOG_PREFIX=matrix-nosubproc-v3-launcher0 \
  bash run_workload.sh
```

Result: **PASSED** (`exit_code=0`)

Runtime and memory:

- Start: `2026-03-19T07:39:31+00:00`
- End: `2026-03-19T09:00:16+00:00`
- Peak Python RSS: `peak_rss_gb=32.482`

Observed behavior:

- `TRITON_HIP_USE_ASYNC_COPY=0`
- No `hipErrorNoBinaryForGpu`
- No `CUDA error: no kernel image is available for execution on the device`
- No `Triton Error [HIP]: Code: 209`
- No traceback

Conclusion:

- v3 (`select_algorithm.py` benchmark-module eviction + `sys.modules` cleanup +
  `precompile_cache.clear()` + `gc.collect()`) is sufficient to make the
  launcher=0 no-subprocess mixed-network run pass in this environment.

### Run V (validation): no subprocess + v3 + launcher=1 with async-copy unset

Objective:

- Determine whether `TRITON_HIP_USE_ASYNC_COPY=0` is still required after the
  current fix stack.

Command:

```bash
env -u TRITON_HIP_USE_ASYNC_COPY \
  STATIC_LAUNCHER=1 WORKLOAD_NETWORKS="large extralarge" \
  TORCHINDUCTOR_RELEASE_UNUSED_TRITON_KERNELS=0 \
  TORCHINDUCTOR_RELEASE_UNUSED_TRITON_KERNELS_GC=0 \
  TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC=0 \
  LOG_PREFIX=matrix-noasynccopy-v3-wpeu0-launcher1 \
  bash run_workload.sh
```

Artifacts:

- `matrix-noasynccopy-v3-wpeu0-launcher1.log`
- `matrix-noasynccopy-v3-wpeu0-launcher1.cpu-mem.log`

Result: **PASSED** (`exit_code=0`)

Runtime and memory:

- Start: `2026-03-20T00:57:30+00:00`
- End: `2026-03-20T02:16:04+00:00`
- Peak Python RSS: `peak_rss_gb=24.719`

Observed behavior:

- `TRITON_HIP_USE_ASYNC_COPY=unset` confirmed in run metadata.
- No `hipErrorNoBinaryForGpu`
- No `CUDA error: no kernel image is available for execution on the device`
- No `Triton Error [HIP]: Code: 209`
- No traceback

### Run W (validation): no subprocess + v3 + launcher=0 with async-copy unset

Companion run for launcher disabled under the same no-async-copy condition.

Command:

```bash
env -u TRITON_HIP_USE_ASYNC_COPY \
  STATIC_LAUNCHER=0 WORKLOAD_NETWORKS="large extralarge" \
  TORCHINDUCTOR_RELEASE_UNUSED_TRITON_KERNELS=0 \
  TORCHINDUCTOR_RELEASE_UNUSED_TRITON_KERNELS_GC=0 \
  TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC=0 \
  LOG_PREFIX=matrix-noasynccopy-v3-wpeu0-launcher0 \
  bash run_workload.sh
```

Artifacts:

- `matrix-noasynccopy-v3-wpeu0-launcher0.log`
- `matrix-noasynccopy-v3-wpeu0-launcher0.cpu-mem.log`

Result: **PASSED** (`exit_code=0`)

Runtime and memory:

- Start: `2026-03-20T02:17:00+00:00`
- End: `2026-03-20T03:36:15+00:00`
- Peak Python RSS: `peak_rss_gb=32.478`

Observed behavior:

- `TRITON_HIP_USE_ASYNC_COPY=unset` confirmed in run metadata.
- No `hipErrorNoBinaryForGpu`
- No `CUDA error: no kernel image is available for execution on the device`
- No `Triton Error [HIP]: Code: 209`
- No traceback

Conclusion from Runs V/W:

- With the current patch stack, `TRITON_HIP_USE_ASYNC_COPY=0` is not required
  for the `large + extralarge` reproducer, for either launcher mode.

### Run X (DDP check): `torchrun` reproducer with no extra environment variables

Objective:

- Validate that the DDP/Triton reproducer now runs without the historical
  no-kernel-image failure and without requiring extra env vars.

Command (exact, no extra env vars):

```bash
torchrun --nproc_per_node=1 reproducer_ddp_triton_error.py --compile
```

#### Run X1 (first attempt)

Result: **FAILED early** due to missing Python dependency, not a Triton/HIP failure.

- Error:
  - `ModuleNotFoundError: No module named 'einops'`
- Exit:
  - `exit_code=1`
  - `elapsed_ms=5087`

Remediation:

```bash
python -m pip install einops
```

#### Run X2 (rerun after dependency install)

Command (same exact command as above, still no extra env vars):

```bash
torchrun --nproc_per_node=1 reproducer_ddp_triton_error.py --compile
```

Result: **PASSED**

- End marker:
  - `=== Test Completed ===`
- Exit:
  - `exit_code=0`
  - `elapsed_ms=1232167` (~20.5 min)
  - `ended_at=2026-03-20T15:32:17.728Z`

Observed behavior:

- No `hipErrorNoBinaryForGpu`
- No `CUDA error: no kernel image is available for execution on the device`
- No `Triton Error [HIP]: Code: 209`
- No traceback on the successful run

Conclusion from Run X:

- For this DDP reproducer, no additional environment variables are needed.
- The only blocker encountered was a missing Python package dependency
  (`einops`), after which the command passed cleanly.

## 2026-03-24 Clarifications from code review discussion

### Unload cadence (what "aggressive" means)

The benchmark cleanup runs in `AlgorithmSelectorCache.benchmark()` **after**
`benchmark_fn(choices)` returns for one autotune choice set (one kernel site).
This means:

- It is **not** per-candidate.
- It is **not** delayed until all kernels in the model finish autotuning.
- It is once per autotune round for each site.

### Benchmark template modules vs runtime winner kernels

Two different lifetimes are involved:

1. **Benchmark template modules** (autotune artifacts)
   - Loaded through `TritonBenchmarkRequest.make_run_fn()` via
     `PyCodeCache.load_by_key_path(...)` for timing candidate choices.
   - After timing, v3 cleanup evicts benchmark modules from:
     - `PyCodeCache.modules_no_attr`
     - `PyCodeCache.modules`
     - `sys.modules`
   - Then `precompile_cache.clear()` + `gc.collect()` allow stale benchmark
     `CompiledKernel` objects to be collected and unloaded.

2. **Runtime winner kernels** (execution artifacts)
   - The kernel object used by compiled graph execution is kept alive by runtime
     launchers and normal graph/object ownership.
   - For static launcher, module unload occurs in
     `StaticallyLaunchedTritonKernel.__del__` when the owner is unreachable.

Practical implication: the benchmark-side instance of a "winner" may be
collected after autotune, and the runtime-side winner is then (re)loaded as the
execution object/handle.

### Runtime/performance impact

- There can be compile-time overhead from reloading winner kernels for runtime
  after benchmark artifact eviction.
- There is **no per-launch hot-path unload/reload** behavior introduced.
- Tradeoff is intentional: small compile-time overhead vs avoiding fatal
  driver module-table exhaustion.

### Reviewer concern: "one user of modules we are unloading"

Reviewer question: "How are we ensuring there is only one user of the modules
we are unloading?"

Clarified ownership model:

- Module handles are owned per kernel object (not a shared global pool).
- Generated launchers hold `runner=self.kernel.run` (bound method), which keeps
  the owner object alive while launchers are reachable.
- Unload happens from owner `__del__`, after the object becomes unreachable.

Validation added:

- Unit test:
  `test/inductor/test_static_triton_launcher.py::TestStaticTritonLauncher::test_launcher_keeps_module_owner_alive_until_release`
- Commit:
  `d1c4d96a4ae`
- Focus:
  verifies launcher-held `runner` keeps owner alive; unload occurs only after
  launcher release.

## 2026-03-27 Review-comment patch set (final validated form)

### Motivation

The earlier experimental `v3` shape fixed the reproducer by evicting
benchmark-only modules from `PyCodeCache`, removing their names from
`sys.modules`, clearing `precompile_cache`, and then forcing
`gc.collect()`. Review feedback asked for a narrower ownership model:

- avoid publishing benchmark-only modules globally when they are not needed for
  runtime imports
- avoid clearing unrelated `precompile_cache` entries that belong to other
  autotune sites

The follow-up patch set keeps the original lifetime goal, but narrows how the
references are created and released.

### Patch summary

1. **`PyCodeCache` gets optional `sys.modules` registration control**
   - `PyCodeCache.load()` and `PyCodeCache.load_by_key_path()` now accept
     `set_sys_modules: bool | None`.
   - Default behavior is unchanged for normal top-level/runtime loads:
     `None` still follows `in_toplevel_process()`.
   - Benchmark-only callers can pass `False`, which allows the module to be
     cached in `PyCodeCache` without publishing it into `sys.modules`.
   - If a later cache hit comes from a caller that does want registration, the
     cached module is inserted into `sys.modules` at that time so runtime
     behavior remains correct.

2. **Benchmark/autotune-only loads stop creating global module roots**
   - `TritonBenchmarkRequest.make_run_fn()`
   - `TritonBenchmarkRequest.precompile()`
   - `CuteDSLBenchmarkRequest.make_run_fn()`
   - `TritonTemplate` benchmark template loads
   now use `set_sys_modules=False`.
   - This means benchmark artifacts are no longer kept alive by
     `sys.modules`; after the autotune round, the remaining strong references
     are the `PyCodeCache` containers and the site-local precompile closure.

3. **Post-benchmark cleanup is narrowed to the active autotune site**
   - `AlgorithmSelectorCache.benchmark()` still evicts matching benchmark module
     paths from `PyCodeCache.modules_no_attr` and `PyCodeCache.modules`.
   - It no longer removes names from `sys.modules`, because benchmark-only
     loads are no longer registered there.
   - It no longer calls `self.precompile_cache.clear()`.
   - Instead, the precompile closure now carries its `precompile_key`, that key
     is threaded through the benchmark/lookup flow, and cleanup does only:
     `self.precompile_cache.pop(precompile_key, None)`.
   - The final selected form does not call explicit `gc.collect()` after that
     cleanup; once benchmark-only ownership was narrowed, the validated
     reproducer sequence stayed green without forcing collection.

4. **Only minimal regression coverage was kept**
   - `test/inductor/test_codecache.py`
     - verifies `load_by_key_path(..., set_sys_modules=False)` can defer
      registration, and that a later normal load still registers the cached
      module
   - `test/inductor/test_select_algorithm.py`
     - verifies benchmark cleanup removes only the matching
       `precompile_cache` entry

### Why this is better than the experimental `v3` form

- It preserves the original module-lifetime fix: benchmark-only
  `CompiledKernel` objects can still become unreachable after autotune so their
  CUDA/HIP modules can unload promptly.
- It avoids broad `sys.modules` churn by not registering benchmark-only modules
  there in the first place.
- It avoids clearing unrelated `precompile_cache` entries, so other autotune
  sites can keep sharing their cached precompile work.
- It also avoids forced full-GC churn in the final selected form; the narrowed
  ownership cleanup alone was sufficient under the validated reproducers.
- It keeps normal runtime module-registration behavior intact for non-benchmark
  loads.

### Selected final patch stack

- This supersedes the earlier mixed code+UT commit `51dfbd9a67e`.
- Code change:
  `0697c9fde94` `[inductor] Avoid global refs for benchmark codecache modules`
- Minimal UTs:
  `55aa3ce3ce1` `[inductor] Add minimal coverage for autotune cleanup ownership`
- Final follow-up:
  `11b8adb56c3` `[inductor] Drop explicit autotune cleanup GC`
- Selected runtime shape:
  - benchmark-only module loads use `set_sys_modules=False`
  - cleanup evicts benchmark paths from `PyCodeCache` and pops only the active
    `precompile_key`
  - there is no explicit post-benchmark `gc.collect()`
- Validation sequence on the selected stack:
  1. `STATIC_LAUNCHER=0` with `WORKLOAD_NETWORKS="large extralarge"`:
     passed in about `75.7 min`
  2. `STATIC_LAUNCHER=1` with `WORKLOAD_NETWORKS="large extralarge"`:
     passed in about `75.0 min`
  3. `torchrun --nproc_per_node=1 reproducer_ddp_triton_error.py --compile`:
     passed in about `10.2 min`
- Across the final validation sequence there was no
  `hipErrorNoBinaryForGpu`, no `no kernel image`, no Triton HIP 209 signature,
  and no traceback.
