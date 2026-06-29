# PyTorch CUDA vs ROCm/HIP Feature Gap Report

**Version analyzed:** PyTorch `v2.13.0-rc12` (commit `0bdbc268e08fba9debac8f35a8a12e9c008ec0fd`, branch `release/2.13`)  
**Date:** June 29, 2026  
**Scope:** Evidence-driven comparison of CUDA backend capabilities versus ROCm/HIP support at the PyTorch 2.13 release-candidate line. Focus is on functional gaps, optimized-path gaps, and `torch.compile` / Inductor parity.

---

## Methodology

This report was built from scratch against the `v2.13.0-rc12` source tree (shallow clone at `/home/mnicosia/src/framework_scripts/pytorch/.pytorch-src`).

1. **Source code analysis** — scanned `USE_ROCM` / `torch.version.hip` guards, backend-library gates (CK, hipBLASLt, rocBLAS, MIOpen, AOTriton, MSLK, cuDNN, CUTLASS, CuTe, Triton), Inductor template wiring, and arch helpers (`gfx90a`, `gfx942`, `gfx950`, `gfx11*`, `gfx12*`).
2. **Test-suite analysis** — inventoried `skipIfRocm`, `skipCUDAIfRocm`, and platform helpers such as `PLATFORM_SUPPORTS_*` across `test/`.
3. **GitHub issue/PR status** — spot-checked open ROCm-labeled issues for live relevance.
4. **Release baseline** — compared against the v2.9-era gap landscape to identify closures retained in 2.13.

Classification rules used throughout:

- CK / CK-Tile is treated as the ROCm counterpart to CUTLASS, not “CUTLASS missing” by itself.
- Eager ATen support is distinguished from Inductor / `torch.compile` support.
- Functional fallback is distinguished from optimized template / autotune support.
- “Hardware limitation” is used only when the missing behavior depends on NVIDIA-only primitives with no AMD equivalent exposed in the current stack.

---

## Notable Gaps Closed Since v2.9

Progress retained on the 2.13 line:

- **Grouped GEMM (eager ATen):** ROCm has a fallback path and an opt-in CK fast path via `ROCM_ALLOW_GROUP_GEMM_CK=1` when built with `USE_ROCM_CK_GEMM` (`GroupedBlas.cpp`).
- **Scaled grouped GEMM (rowwise FP8):** ROCm-specific `_f8_f8_bf16_rowwise_grouped_mm_rocm` exists for MI300-class hardware; CUDA still owns CUTLASS/MSLK fast paths.
- **Scaled MM v2:** ROCm dispatch remains in-tree alongside CUDA.
- **SymmetricMemory (basic):** ROCm ships a separate `rocshmem_extension.cu` implementation; NVSHMEM tiled ops are not ported.
- **AOTriton SDPA / flash:** ROCm flash path is wired through AOTriton with optional experimental arch enablement (`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`).
- **CK SDPA:** Available when built with `USE_ROCM_CK_SDPA`; exposed via `PLATFORM_SUPPORTS_CK_SDPA` and `torch.backends.cuda.is_ck_sdpa_available()`.
- **Eager CK dense GEMM on RDNA:** `ck_gemm_half.hip` / `ck_gemm_bfloat16.hip` list `gfx1100`, `gfx1101`, `gfx1102`, `gfx1150`, `gfx1151`, `gfx1200`, `gfx1201`.
- **MX GEMM on gfx950:** `evaluate_platform_supports_mx_gemm()` enables ROCm when `ROCM_VERSION >= (7, 0)` and the device is `gfx950`.

---

## Top Feature Gaps

### 1. Inductor GEMM Template Coverage (CUTLASS / NVGEMM vs CK)

| Field | Value |
| ----- | ----- |
| Severity | High |
| Root cause | Compiler/template gap; backend library coverage gap (CK arch scope) |
| Source | Source analysis |
| Key files | `torch/_inductor/utils.py`, `torch/_inductor/kernel/mm.py`, `torch/_inductor/kernel/bmm.py`, `torch/_inductor/config.py` |
| Key tests | `test/inductor/test_max_autotune.py`, `test/inductor/test_torchinductor.py` |

On ROCm, `use_cutlass_template()` returns `False` when `torch.version.hip` is set, so CUDA CUTLASS and NVIDIA Universal GEMM templates never participate in autotune. The ROCm equivalent is CK via `use_ck_template()`, but Inductor limits CK to CDNA arches in config:

```2572:2576:pytorch/.pytorch-src/torch/_inductor/config.py
    ck_supported_arch: list[Literal["gfx90a", "gfx942", "gfx950"]] = [
        "gfx90a",
        "gfx942",
        "gfx950",
    ]
```
> Nick Notes:
> re, restriction of Inductor to CDNA. Inductor implements torch.compile() either in eager mode, you can use a number of backends for Inductor, you can use MIopen for convolutions, hipBLASlt for GEMMS, and CK is a muddle, it's for GEMMS and SDPA (aka flash attention)
> we do not allow a CK backend for any architecture - just gfx90a (MI200s), fgx942 (MI300s), gfx950 (MI350s)
> Is this a gap? CUTLASS probably handles all archictures, AMD only handles these three.

**Inductor template mapping at 2.13:**

| Op | CUDA optimized templates | ROCm optimized templates |
| --- | --- | --- |
| `mm`, `addmm`, `bmm` | CUTLASS3x (+ Triton) | CK (+ Triton) on supported CDNA only |
| `_scaled_mm` | CUTLASS / cuBLASLt paths | CK when enabled |
| `_int_mm` | CUTLASS3x (+ Triton) | Triton + ATen only — no CK template |
| `_sparse_semi_structured_mm` | CUTLASS2x sparse | ATen extern only — no CK sparse template |
| Grouped / block-scaled GEMM | CUTLASS / CuTeDSL / NVGEMM | No equivalent Inductor templates |

> Nick Notes:
> [TODO] The claim _scaled_mm is supported by CK when enabled needs to be checked. Is _scaled_mm even a part of the public API?
> Line `_int_mm` is a gap but is low priority. Double check if this can be done by ATen (ATen is _eager mode_)
> `_sparse_semi_structured_mm` is a real gap; Ramya(?) has landed a PR to start this journey. It is not supported in Eager Mode. This is covered by AIPYTORCH-16.
> Grouped and block-scaled GEMMs - clear gap, these are used by MoE models. Nick asserts this is a priority.

`use_nv_universal_gemm_template()` additionally requires `not torch.version.hip`.

> [TODO] what is `use_nv_universal_gemm_template`? do we care about this? #NeedsExplore

**Impact:** Dense GEMM autotune on MI200/MI300 can be competitive when CK is installed, but RDNA systems do not get Inductor CK templates even though eager CK GEMM lists gfx11/gfx12 arches. Integer GEMM and 2:4 sparse GEMM lack ROCm optimized Inductor templates entirely.

> [TODO] Eager CK GEMM says we support gfx11/gfx12? #NeedsExplore seems like the code is actually saying it's ok to use CK, including for Strix and Strix Halo.
> [TODO] Integer GEMM and 2:4 sparse lack Inductor templates? This is probably the same as the notes about `int_mm` and `_sparse_semi_structured_mm`

**Status / next steps:** Extend CK / CK-Tile coverage for `_int_mm` and structured sparsity; widen `ck_supported_arch` or decouple eager vs Inductor arch gates; evaluate CK-Tile parity for NVGEMM-class fused GEMMs.

> Nick notes: Could re-write this as: for missing aten operations support, support them in aten. once that's done, support these operations in Inductor. the point is to support the missing pieces in Inductor (of course) but you need the support in ATen first.

---

### 2. 2:4 Semi-Structured Sparsity

| Field | Value |
| ----- | ----- |
| Severity | High |
| Root cause | Backend library coverage gap; software integration gap |
| Source | Source analysis |
| Key files | `aten/src/ATen/native/sparse/cuda/SparseSemiStructuredOps.cu`, `aten/src/ATen/native/sparse/cuda/SparseSemiStructuredApply*.cu`, `torch/_inductor/kernel/mm.py` |
| Key tests | `test/test_sparse_csr.py` (`@skipCUDAIfRocm`), `test/test_matmul_cuda.py` |

All CUTLASS-based semi-structured kernels are compiled out on ROCm. Representative guards:

- `SparseSemiStructuredOps.cu`: `TORCH_CHECK(false, __func__, " : CUTLASS not supported")` under `USE_ROCM`.
- `SparseSemiStructuredApplyDense.cu`: `TORCH_CHECK(false, "_sparse_semi_structured_apply_dense: not supported")`.
- `SparseSemiStructuredTile.cu`: `TORCH_CHECK(false, "_sparse_semi_structured_tile: not supported")`.

Inductor lowering for `aten._sparse_semi_structured_mm` adds only `CUTLASS2xGemmTemplate` choices when `use_cutlass_template()` is true (CUDA only). There is no CK sparse template path.

**Impact:** 2:4 structured sparsity workflows (pruning, sparse training, sparse inference) are unavailable on ROCm in both eager and compiled modes.

**Status / next steps:** Needs verification whether hipSPARSELt or CK sparse kernels can cover the same layouts; until then this remains a hard functional gap, not just a performance gap.

> [todo] this covered above, see Ramya's epic AIPYTORCH-16

---

### 3. cuDNN SDPA and Fused-Attention Feature Breadth

| Field | Value |
| ----- | ----- |
| Severity | High |
| Root cause | Software integration gap (hipDNN not wired); test coverage gap |
| Source | Source analysis, tests |
| Key files | `aten/src/ATen/native/transformers/cuda/sdp_utils.cpp`, `torch/testing/_internal/common_cuda.py` |
| Key tests | `test/test_transformers.py`, `test/test_varlen_attention.py` |

`can_use_cudnn_attention()` unconditionally returns `false` on `USE_ROCM` builds. Test helpers encode the same split:

- `PLATFORM_SUPPORTS_CUDNN_ATTENTION` requires `not TEST_WITH_ROCM`.
- `PLATFORM_SUPPORTS_FUSED_SDPA` is `TEST_CUDA and not TEST_WITH_ROCM`.

ROCm SDPA instead routes through AOTriton flash, memory-efficient attention (with arch gating), and optional CK SDPA. Mem-efficient attention on ROCm applies stricter head-dim / GQA constraints than the CUDA cuDNN path.

**Impact:** Features that depend on the cuDNN attention backend — deterministic algorithms, some nested-tensor cases, broader head-dimension support — remain CUDA-only. Performance has improved with AOTriton/CK, but feature breadth still lags.

**Status / next steps:** hipDNN integration is the expected long-term fix for cuDNN-parity SDPA features. Until then, document backend-specific SDPA limitations explicitly in user-facing APIs.

---

### 4. Symmetric Memory Multicast, Multimem, and Advanced NVSHMEM Ops

| Field | Value |
| ----- | ----- |
| Severity | High |
| Root cause | Hardware limitation (NVIDIA multicast); software integration gap (rocSHMEM subset) |
| Source | Source analysis |
| Key files | `torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory.cu`, `torch/csrc/distributed/c10d/symm_mem/rocshmem_extension.cu`, `torch/distributed/_symmetric_memory/__init__.py` |
| Key tests | `torch/testing/_internal/common_distributed.py` (`requires_multicast_support`) |

CUDA multicast support is implemented against the CUDA driver API (`cuMulticastCreate`). Tests gate multicast features with `_SymmetricMemory.has_multicast_support(DeviceType.CUDA, 0)`.

ROCm has a separate `rocshmem_extension.cu`, but its header documents missing features:

- No rocSHMEM equivalent of `nvshmemx_collective_launch` / grid-wide sync semantics.
- No tiled communication (`nvshmemx::Tensor`, tile reduce ops).
- Offset writeback requires a separate kernel due to missing grid-wide synchronization.

Python helpers such as `_should_use_multimem_all_gather_matmul` and `_multimem_all_gather_matmul` depend on multicast support that ROCm does not expose.

**Impact:** One-shot all-reduce, multicast-backed symmetric memory, and multimem fused distributed GEMM paths remain NVIDIA-specific. Basic symmetric memory works on ROCm via rocSHMEM, but not the newest CUDA-only collectives.

**Status / next steps:** Track rocSHMEM feature growth; avoid labeling multicast-dependent APIs as cross-vendor without runtime guards.

---

### 5. Tensor Memory Accelerator (TMA) and CUDA-Specific Codegen

| Field | Value |
| ----- | ----- |
| Severity | Medium |
| Root cause | Hardware limitation |
| Source | Source analysis |
| Key files | `torch/_inductor/utils.py`, `torch/_inductor/kernel/mm_grouped.py`, `torch/_inductor/kernel/flex/`, `torch/_vendor/quack/` |
| Key tests | `test/inductor/test_nv_universal_gemm.py`, `test/inductor/test_cutedsl_grouped_mm.py` |

`can_use_tma()` requires `has_triton_tma_device()` and documents NVIDIA cuTensorMap constraints. Actual TMA descriptors are not available on AMD hardware.

Inductor's `use_triton_tma_template()` short-circuits on HIP:

```2269:2272:pytorch/.pytorch-src/torch/_inductor/utils.py
    # On AMD (HIP), TMA is not available but we still use non-TMA persistent
    # kernels, so skip the TMA compatibility checks.
    if torch.version.hip is not None:
        return True
```

This means ROCm may take TMA-named code paths in templates but without true TMA hardware support — functional but not equivalent to CUDA TMA performance semantics. CuTeDSL, CUTLASS TMA pipelines, PDL, and Blackwell-specific templates remain CUDA-only.

**Impact:** Latest NVIDIA kernel codegen advantages (TMA loads/stores, persistent TMA pipelines, CuTeDSL block-scaled GEMM) do not translate directly to ROCm. This is expected hardware divergence, not a simple wiring bug.

**Status / next steps:** Classify TMA-dependent optimizations as CUDA-only in docs; pursue AMD-specific memory hierarchy optimizations (buffer loads, LDS tiling) as the ROCm counterpart.

---

### 6. FP4 / NVFP4 / MXFP4 and Advanced Low-Precision GEMM

| Field | Value |
| ----- | ----- |
| Severity | High |
| Root cause | Backend library coverage gap; hardware limitation (partial) |
| Source | Source analysis, tests |
| Key files | `aten/src/ATen/native/cuda/GroupedBlas.cpp`, `aten/src/ATen/native/cuda/ScaledGroupMM.cu`, `torch/testing/_internal/common_cuda.py` |
| Key tests | `test/test_scaled_matmul_cuda.py`, `test/inductor/test_fp8.py` |

Platform helpers show the split clearly:

| Capability | CUDA | ROCm |
| --- | --- | --- |
| FP8 (`PLATFORM_SUPPORTS_FP8`) | SM90+ / SM89 | `gfx94`, `gfx120*`, `gfx95` (version-gated) |
| FP8 sparse | cuSPARSELt-gated | `gfx950` only |
| FP8 grouped GEMM | SM90 (not SM100) | `gfx942`/`gfx950` with `USE_MSLK` build |
| MX GEMM | SM100+ | `gfx950` with ROCm ≥ 7.0 |
| MXFP8 grouped GEMM | SM100 + MSLK | **Not supported** (`evaluate_platform_supports_mxfp8_grouped_gemm` returns false on HIP) |

`ScaledGroupMM.cu` and `GroupMM.cu` CUTLASS kernels are wrapped in `#if !defined(USE_ROCM)`. `_mx8_mx8_bf16_grouped_mm_mslk` is CUDA-only (`#if defined(USE_MSLK) and !defined(USE_ROCM)`).

NVFP4/MXFP4 dtype support (`float4_e2m1fn_x2`) exists at the type level, but performant scaled GEMM paths are CUDA CUTLASS / MSLK / CuTeDSL implementations.

**Impact:** Cutting-edge sub-byte training/inference kernels targeting Blackwell/Hopper are effectively CUDA-only. MI300/MI350 have growing FP8 support but not full MXFP4/NVFP4 parity.

**Status / next steps:** Needs verification on whether AITER/MSLK ROCm paths will cover MXFP4 grouped GEMM; track `gfx950` MX GEMM maturity separately from NVFP4.

---

### 7. Grouped GEMM and Scaled Grouped GEMM Fast Paths

| Field | Value |
| ----- | ----- |
| Severity | Medium |
| Root cause | Compiler/template gap; software integration gap |
| Source | Source analysis |
| Key files | `aten/src/ATen/native/cuda/GroupedBlas.cpp`, `aten/src/ATen/native/hip/ck_group_gemm.hip`, `aten/src/ATen/native/nested/cuda/NestedTensorMatmul.cu` |
| Key tests | `test/test_nestedtensor.py`, `test/test_scaled_matmul_cuda.py` |

**Eager `_grouped_mm` on ROCm:** Defaults to fallback; CK fast path requires both `USE_ROCM_CK_GEMM` build flag and runtime `ROCM_ALLOW_GROUP_GEMM_CK=1`, and is limited to `gfx942`, `gfx950`, `gfx90a`.

**Eager `_scaled_grouped_mm` on ROCm:** Rowwise FP8 path exists (`_f8_f8_bf16_rowwise_grouped_mm_rocm`); CUTLASS SM90/SM100 kernels are CUDA-only. MXFP8 grouped path is not implemented on ROCm builds without MSLK.

**NestedTensor matmul:** CUDA CUTLASS grouped GEMM in `NestedTensorMatmul.cu` is disabled on ROCm (`#if !defined(USE_ROCM)` around `build_grouped_gemm`).

**Inductor:** No CK grouped-GEMM templates equivalent to CUDA CUTLASS / CuTeDSL / NVGEMM grouped templates.

**Impact:** MoE and variable-length batched GEMM workloads may run functionally on ROCm but miss the fastest fused paths available on Hopper/Blackwell.

**Status / next steps:** Enable CK grouped GEMM by default once performance is validated; add Inductor grouped templates on ROCm.

---

### 8. Consumer AMD GPU (RDNA3 / RDNA4) Parity

| Field | Value |
| ----- | ----- |
| Severity | Medium |
| Root cause | Software integration gap; test coverage gap |
| Source | Source analysis, tests |
| Key files | `torch/_inductor/config.py`, `aten/src/ATen/native/hip/ck_gemm_*.hip`, `torch/testing/_internal/common_utils.py` |
| Key tests | `test/inductor/test_torchinductor.py` (`@skipIfRocmArch(NAVI_ARCH)`), `test/test_nestedtensor.py` |

There is a deliberate split between eager and compiled support:

- **Eager CK GEMM** lists RDNA arches (`gfx1100`, `gfx1101`, `gfx1102`, `gfx1150`, `gfx1151`, `gfx1200`, `gfx1201`).
- **Inductor CK** supports only `gfx90a`, `gfx942`, `gfx950`.
- **AOTriton flash** on RDNA requires `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` for experimentally supported arches.

Test skips on `NAVI_ARCH` and MI200/MI350 arch predicates indicate validation is still concentrated on datacenter GPUs.

**Impact:** RDNA users get partial eager support but weaker `torch.compile` / Inductor autotune coverage and fewer validated attention paths.

**Status / next steps:** Expand Inductor `ck_supported_arch`; reduce NAVI-specific test skips as coverage matures.

---

### 9. FlexAttention and Inductor Fusion Gaps

| Field | Value |
| ----- | ----- |
| Severity | Medium |
| Root cause | Compiler/template gap; test coverage gap |
| Source | Source analysis, tests |
| Key files | `torch/_inductor/kernel/flex/`, `test/inductor/test_flex_attention.py`, `test/inductor/test_combo_kernels.py` |
| Key tests | `test/inductor/test_combo_kernels.py` (6+ `@skipIfRocm` markers), `test/inductor/test_flex_attention.py` |

FlexAttention is not globally disabled on ROCm — platform detection uses `torch.cuda.get_device_capability() >= (8, 0)`, which maps to AMD arch major versions. However:

- Flex templates include `USE_TMA` branches that are CUDA-oriented.
- `test/inductor/test_combo_kernels.py` has numerous ROCm skips tied to open issues (#180017–#180026 range).
- `test/inductor/test_flex_attention.py` skips some cases on MI200 arch.

**Impact:** FlexAttention may work on ROCm for common cases, but fusion / combo-kernel optimizations and TMA-specialized paths lag CUDA validation.

**Status / next steps:** Triage ROCm combo-kernel issues; confirm which skips indicate missing functionality vs. test infra gaps.

---

### 10. CUDA-Only Runtime, Allocator, and Tooling Features

| Field | Value |
| ----- | ----- |
| Severity | Low–Medium |
| Root cause | Hardware limitation; software integration gap |
| Source | Source analysis, tests |
| Key files | `torch/testing/_internal/common_cuda.py`, `test/test_cuda.py`, `test/inductor/test_static_triton_launcher.py` |
| Key tests | `test/test_cuda.py`, `test/test_cuda_graph_debug.py`, `test/profiler/test_profiler.py` |

Representative CUDA-only or ROCm-degraded areas at 2.13:

| Feature | ROCm status |
| --- | --- |
| `PLATFORM_SUPPORTS_GREEN_CONTEXT` | CUDA 12.8+ / driver 570+ only |
| `PLATFORM_SUPPORTS_WORKQUEUE_CONFIG` | CUDA 13.1+ / driver 590+ only |
| Expandable segments allocator option | CUDA caching allocator feature; tests gate on CUDA allocator option |
| CUDA graph input liveness debugging | Explicitly CUDA-only in tests |
| `cudaMallocManaged` / UVM tests | Skipped on ROCm |
| Static Triton launcher fast path | `@skipIfRocm` in Inductor tests |
| Inductor `layout_opt_default` | Disabled on HIP (`"0"` vs `"1"`) |
| PTX inline asm tests | CUDA-specific codegen |

**Impact:** Mostly affects debugging, profiling, and bleeding-edge CUDA 12.x/13.x runtime features rather than core tensor math — but matters for production parity on memory management and compilation tooling.

**Status / next steps:** Document ROCm equivalents where they exist (`hipMallocAsync` mapping); avoid implying feature parity for green context / workqueue config.

---

## Honorable Mentions

- **`_int_mm`:** Eager HIP path exists via hipified `int8_gemm` in `Blas.cpp`; Inductor lacks CK/CUTLASS templates on ROCm (Triton + ATen only). Not a hard eager gap.
- **Helion DSL:** Optional external package (`torch.utils._helion`); not ROCm-specific but also not part of core Inductor — parity depends on Helion project support. **Needs verification** for ROCm backend status.
- **AITER:** Referenced in build scripts (`USE_CK` / MSLK gates) but not exhaustively audited in this pass.
- **Windows ROCm:** Multiple `_MSC_VER` guards mirror ROCm in excluding CUTLASS sparse paths.

---

## Summary Table

| # | Gap | Severity | Eager ATen | Inductor | Primary root cause |
| --- | --- | --- | --- | --- | --- |
| 1 | GEMM template coverage (CK vs CUTLASS/NVGEMM) | High | Partial | Gap | Compiler/template + CK arch scope |
| 2 | 2:4 semi-structured sparsity | High | Missing | Missing | Backend library |
| 3 | cuDNN SDPA features | High | Missing | N/A | Software integration |
| 4 | Symmetric memory multicast / multimem | High | Missing | N/A | Hardware + software |
| 5 | TMA / CuTeDSL codegen | Medium | N/A | CUDA-only paths | Hardware |
| 6 | FP4 / NVFP4 / MXFP4 advanced GEMM | High | Partial | CUDA-only | Backend library |
| 7 | Grouped / scaled grouped GEMM fast paths | Medium | Partial | Gap | Software + compiler |
| 8 | RDNA consumer GPU parity | Medium | Partial | Gap | Software integration |
| 9 | FlexAttention / combo kernels | Medium | Partial | Partial | Compiler + tests |
| 10 | CUDA runtime / allocator tooling | Low–Med | Partial | Partial | Hardware + software |

---

## Appendix A: Test Coverage

**Inventory (v2.13.0-rc12):**

- **270** `skipIfRocm` / `skipCUDAIfRocm` markers across **68** test files under `test/`.
- Inductor-heavy files with multiple ROCm skips: `test/inductor/test_combo_kernels.py`, `test/inductor/test_cuda_repro.py`, `test/inductor/test_torchinductor.py`, `test/inductor/test_fp8.py`, `test/inductor/test_max_autotune.py`.

**Platform helpers in `torch/testing/_internal/common_cuda.py`:**

| Helper | ROCm behavior |
| --- | --- |
| `PLATFORM_SUPPORTS_CUDNN_ATTENTION` | Always false |
| `PLATFORM_SUPPORTS_FUSED_SDPA` | Always false |
| `PLATFORM_SUPPORTS_CK_SDPA` | True when CK SDPA built |
| `PLATFORM_SUPPORTS_MEM_EFF_ATTENTION` | Arch-list gated (`gfx90a`, `gfx942`, `gfx1100`, `gfx1201`, `gfx950`, + experimental arches) |
| `PLATFORM_SUPPORTS_FP8` | gfx94 / gfx120 / gfx95 gated |
| `PLATFORM_SUPPORTS_FP8_GROUPED_GEMM` | gfx942 / gfx950 + `USE_MSLK` |
| `PLATFORM_SUPPORTS_MX_GEMM` | gfx950 + ROCm ≥ 7.0 |
| `PLATFORM_SUPPORTS_MXFP8_GROUPED_GEMM` | false |
| `PLATFORM_SUPPORTS_FP8_SPARSE` | gfx950 only |

**Interpretation guidance:** A ROCm skip does not always mean missing functionality — some skips document numerics differences (e.g., denormal handling) or CUDA-specific debug tooling. Each skip should be triaged into functional gap vs. validation gap vs. intentional CUDA-only feature.

---

## Appendix B: Issue / PR References

Spot-checked open ROCm-labeled issues (June 2026):

| Issue | State | Relevance |
| --- | --- | --- |
| [#169378](https://github.com/pytorch/pytorch/issues/169378) | Open | ROCm Triton 3.6 dynamo benchmark accuracy |
| [#162606](https://github.com/pytorch/pytorch/issues/162606) | Open | TensorPipe ROCm support |
| [#154017](https://github.com/pytorch/pytorch/issues/154017) | Open | Per-operator device capability registry RFC |
| [#146848](https://github.com/pytorch/pytorch/issues/146848) | Open | Mem-efficient SDPA shape mismatch (eager/meta) |

Referenced in Inductor ROCm skips (spot check, still open at analysis time):

- [#163765](https://github.com/pytorch/pytorch/issues/163765), [#163701](https://github.com/pytorch/pytorch/issues/163701), [#163689](https://github.com/pytorch/pytorch/issues/163689) — `test/inductor/test_cuda_repro.py`
- [#180017](https://github.com/pytorch/pytorch/issues/180017)–[#180026](https://github.com/pytorch/pytorch/issues/180026) — `test/inductor/test_combo_kernels.py`
- [#164271](https://github.com/pytorch/pytorch/issues/164271), [#175482](https://github.com/pytorch/pytorch/issues/175482) — scaled matmul / nested tensor

**Note:** Issue state should be re-verified before using this report for release planning; titles and states were checked via GitHub API at report generation time.
