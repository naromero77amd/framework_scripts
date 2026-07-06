# PyTorch CUDA vs ROCm/HIP Feature Gap Report

**Version analyzed:** PyTorch `main` (commit `605845d4891`)  
**Date:** March 19, 2026 (refreshed)  
**Scope:** Top 11 high-level feature gaps present in the CUDA backend but missing, incomplete, or degraded in the HIP/ROCm backend.

---

## Table of Contents

- [Methodology](#methodology)
- [Top 11 Feature Gaps](#top-11-feature-gaps)
  1. [Inductor GEMM Template and Semi-Structured Sparsity Coverage (CUTLASS vs CK)](#1-inductor-gemm-template-and-semi-structured-sparsity-coverage-cutlass-vs-ck)
  2. [cuDNN-Based Scaled Dot-Product Attention (SDPA)](#2-cudnn-based-scaled-dot-product-attention-sdpa)
  3. [Symmetric Memory Multicast, One-Shot All-Reduce, and Async Distributed MatMul](#3-symmetric-memory-multicast-one-shot-all-reduce-and-async-distributed-matmul)
  4. [Tensor Memory Accelerator (TMA)](#4-tensor-memory-accelerator-tma)
  5. [FP4 / NVFP4 / MXFP4 Support](#5-fp4--nvfp4--mxfp4-support)
  6. [Mixture of Experts (MoE) Communication and Compute](#6-mixture-of-experts-moe-communication-and-compute)
  7. [Consumer AMD GPU Support (RDNA3 / RDNA4)](#7-consumer-amd-gpu-support-rdna3--rdna4)
  8. [FlexAttention Limitations](#8-flexattention-limitations)
  9. [Expandable Segments and Memory Allocator Features](#9-expandable-segments-and-memory-allocator-features)
  10. [Torch.compile / Inductor Codegen Gaps](#10-torchcompile--inductor-codegen-gaps)
  11. [Helion DSL — Backend and Feature Parity](#11-helion-dsl--backend-and-feature-parity)
- [Honorable Mentions](#honorable-mentions)
- [Summary Table](#summary-table)
- [Appendix A: Test Coverage](#appendix-a-test-coverage)
- [Appendix B: Open ROCm-Labeled Issues](#appendix-b-open-rocm-labeled-issues-as-of-march-2026)

---

## Methodology

This report was compiled from three sources:

1. **Source code analysis** of current PyTorch `main` — scanning for `USE_ROCM` preprocessor guards, disabled code paths, `skipIfRocm` test markers, and CUDA-only library dependencies.
2. **GitHub issues** — open, non-stale (opened within the last 6 months) user-reported bugs and feature requests related to ROCm/HIP limitations.
3. **PR/issue status checks** — validating whether referenced fixes are merged, reverted, or still open.

### Notable Gaps Closed Since v2.9

Before diving into the current gaps, it's worth noting progress made in v2.10-v2.11 and retained on current `main`:

- **Grouped GEMM (baseline path)**: ROCm fallback support landed ([#162419](https://github.com/pytorch/pytorch/pull/162419)). Follow-up CK enablement in [#166334](https://github.com/pytorch/pytorch/pull/166334) was later reverted, so fast-path availability remains conditional in current main.
- **Scaled MM v2**: Added for ROCm ([#165528](https://github.com/pytorch/pytorch/pull/165528)).
- **SymmetricMemory (basic)**: Ported to ROCm ([#150580](https://github.com/pytorch/pytorch/pull/150580)), though multicast/multimem remain CUDA-only.
- **AOTriton SDPA**: ROCm SDPA support exists in-tree, but [#162330](https://github.com/pytorch/pytorch/pull/162330) (Windows ROCm AOTriton enablement) was later reverted, so this area is still evolving.
- **GFX1150/GFX1151 (RDNA 3.5)**: Added to hipBLASLt GEMM lists ([#164744](https://github.com/pytorch/pytorch/pull/164744)).
- **torch.cuda._compile_kernel / load_inline on ROCm**: Inline kernel compilation now works ([#162510](https://github.com/pytorch/pytorch/pull/162510), [#162577](https://github.com/pytorch/pytorch/pull/162577)).

---

## Top 11 Feature Gaps

### 1. Inductor GEMM Template and Semi-Structured Sparsity Coverage (CUTLASS vs CK)


|               |                                                                                                                                                               |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Severity**  | High                                                                                                                                                          |
| **Source**    | Source code analysis, v2.10 release notes                                                                                                                     |
| **Key files** | `torch/_inductor/utils.py`, `torch/_inductor/kernel/mm.py`, `torch/_inductor/kernel/bmm.py`, `aten/src/ATen/native/sparse/cuda/SparseSemiStructured*.cu`, `aten/src/ATen/native/nested/cuda/NestedTensorMatmul.cu` |


CUTLASS is NVIDIA's open-source CUDA kernel library; it has no ROCm port and never will — the AMD equivalent is **Composable Kernel (CK)** and **CK-Tile**. On ROCm, CK is available for dense GEMM in Inductor, but several CUTLASS-specific paths remain unmatched:

- **Inductor GEMM template coverage (exact CUTLASS vs CK mapping, including 2:4 sparsity)**: `use_cutlass_template()` returns `False` on ROCm, so CUDA CUTLASS templates are not used at runtime on HIP. In current Inductor wiring, the template mapping is:
  - CUTLASS-backed in Inductor: `mm` (CUTLASS3x), `addmm` (CUTLASS3x), `bmm` (CUTLASS3x), `_int_mm` (CUTLASS3x), `_sparse_semi_structured_mm` (CUTLASS2x sparse).
  - CK-backed in Inductor: `mm`, `addmm`, `bmm`, `_scaled_mm`.
  - Missing CK equivalents relative to existing CUTLASS-template callsites: `_int_mm` and `_sparse_semi_structured_mm`.
  For semi-structured 2:4 specifically, Inductor lowering for `aten._sparse_semi_structured_mm` adds only CUTLASS sparse template choices (`CUTLASS2xGemmTemplate`) and has no CK sparse template path. In ATen, ROCm codepaths in `SparseSemiStructured*.cu` raise `TORCH_CHECK(false, ... "CUTLASS not supported")`, so 2:4 sparse MM is currently unavailable on ROCm.
- **NestedTensor matmul**: NestedTensor matmul/bmm is functional on ROCm via fallback implementations, but the CUDA CUTLASS grouped-GEMM acceleration in `NestedTensorMatmul.cu` is compiled out on ROCm and no CK replacement is wired up for that fast path.
- **Memory-efficient attention**: CUTLASS-based ME attention kernels are CUDA-only. On ROCm, the mem-efficient backend path in `sdp_utils.cpp` calls `check_head_dim_size_flash<true>` and `check_batch_size_and_num_heads_dense<false /*supports_grouped_query_attention=*/>`. This keeps dense mem-efficient attention on stricter ROCm flash-style constraints (equal q/k/v head dims, grouped-query-attention disabled in that dense path), with ROCm/AOTriton head-dim limits also varying by arch/version (for example gfx11 under AOTriton 0.11).

**Impact:** ROCm has CK template support for dense GEMM, but CUDA CUTLASS-specific capabilities — especially 2:4 structured sparsity and some specialized fast paths — still lack ROCm-equivalent coverage.

---

### 2. cuDNN-Based Scaled Dot-Product Attention (SDPA) -- Known GAP and will be fixed with hipDNN


|               |                                                                                                               |
| ------------- | ------------------------------------------------------------------------------------------------------------- |
| **Severity**  | High                                                                                                          |
| **Source**    | Source code analysis                                                                                          |
| **Key files** | `aten/src/ATen/native/transformers/cuda/sdp_utils.cpp`, `aten/src/ATen/native/transformers/cuda/attention.cu` |


The cuDNN attention backend (`_scaled_dot_product_cudnn_attention`) remains unconditionally disabled on ROCm — `can_use_cudnn_attention()` always returns `false` when `USE_ROCM` is defined. While ROCm gained AOTriton SDPA in v2.10, the cuDNN path provides additional capabilities:

- Deterministic attention algorithms
- Nested tensor support in SDPA
- Broader head dimension support (head_dim != head_dim_v)

The `PLATFORM_SUPPORTS_FUSED_SDPA` flag is `False` on ROCm, and numerous transformer tests remain skipped via `skipIfRocm`.

**Impact:** Transformer workloads may see reduced attention kernel feature coverage on ROCm. The performance gap has narrowed with AOTriton but feature parity is not achieved.

---

### 3. Symmetric Memory Multicast, One-Shot All-Reduce, and Async Distributed MatMul


|                         |                                                                                                                                                                                                                                                                                                                                                                          |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Severity**            | High                                                                                                                                                                                                                                                                                                                                                                     |
| **Source**              | Source code analysis, v2.9/v2.10 release notes                                                                                                                                                                                                                                                                                                                           |
| **Hardware dependency** | Partial — multicast/multimem paths depend on CUDA multicast capability (NVLink/NVSwitch + CUDA driver/runtime support). AMD does not currently expose an equivalent multicast path in this stack, while non-multicast symmetric-memory reductions can still run on ROCm. |
| **Key files**           | `torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryOps.cu`, `torch/csrc/distributed/c10d/symm_mem/CudaDMAConnectivity.cpp`, `torch/csrc/distributed/c10d/symm_mem/intra_node_comm.cpp`, `torch/csrc/distributed/c10d/symm_mem/nvshmem_extension.cu`, `torch/csrc/distributed/c10d/cuda/AsyncMM.cu`, `caffe2/CMakeLists.txt` |


While basic `SymmetricMemory` was ported to ROCm in v2.10, higher-performance paths are split between (a) CUDA-only multicast paths, (b) ROCm-supported but differently integrated non-multicast paths, and (c) CUDA-only fused communication+compute paths:

- **Multicast / multimem operations**: Guarded by `#if !defined(USE_ROCM)` in `symm_mem/CUDASymmetricMemoryOps.cu` — require NVLink/NVSwitch multicast hardware. No equivalent multicast path exists in the current ROCm stack.
- **MoE dispatch/combine extension ops**: `all_to_all_vdev_2d` and `all_to_all_vdev_2d_offset` live in `symm_mem/nvshmem_extension.cu` and are tied to CUDA+NVSHMEM build wiring (`torch_nvshmem` in `caffe2/CMakeLists.txt`), with no equivalent ROCm build path today.
- **One-shot all-reduce**: `symm_mem::one_shot_all_reduce` / `two_shot_all_reduce` kernels are implemented and registered for ROCm builds (`TORCH_LIBRARY_IMPL(symm_mem, CUDA, ...)`), so the op itself is not CUDA-only. In current main, `isIntraNodeCommSupported()` is not hard-disabled by `USE_ROCM`; ROCm uses `amdsmi`-based topology checks in `symm_mem/intra_node_comm.cpp`. The remaining gap is reduced optimization coverage vs CUDA fast paths.
- **DMA connectivity**: `symm_mem/CudaDMAConnectivity.cpp` is excluded on ROCm (`#if !defined(USE_ROCM) ...`). This component is a topology detector used for topology-aware fastpath decisions; the impact is loss of topology-driven optimization rather than loss of baseline collective correctness.
- **Async distributed MatMul (fused all-gather + matmul fast path)**: This refers to `symm_mem::_async_input_mm`, used by the native fused all-gather-matmul path in tensor-parallel workloads. This fast path currently has no ROCm implementation in PyTorch, so ROCm falls back to decomposition/native alternatives. Implementation details are consolidated in [Gap 4](#4-tensor-memory-accelerator-tma).

`SymmetricMemory` support on ROCm is provided by PyTorch's in-tree HIP symmetric-memory allocator/rendezvous path (e.g., `empty_strided_p2p` + `rendezvous`), and does not require NVSHMEM/rocSHMEM.

**Impact:** On ROCm, baseline symmetric-memory reductions exist, but the highest-performance CUDA-specific paths (multicast/multimem, async distributed matmul fast path, and some topology-aware fastpaths) are missing or reduced. This creates a performance/feature gap for advanced in-node multi-GPU communication and fused communication+compute patterns rather than a complete absence of symmetric-memory collectives.

---

### 4. Tensor Memory Accelerator (TMA) - CUDA-only in current PyTorch/Triton paths


|                         |                                                                                                                                                                                                                                                                                                                                                   |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Severity**            | High                                                                                                                                                                                                                                                                                                                                              |
| **Source**              | Source code analysis                                                                                                                                                                                                                                                                                                                              |
| **Hardware dependency** | Yes — TMA is a hardware unit on NVIDIA Hopper/Blackwell GPUs with no direct AMD equivalent on MI350/CDNA4. CDNA4 has a Data Movement Engine (DME) and improved `GLOBAL_LOAD_LDS` (128-bit/lane, up from 32-bit), but these do not replicate TMA's hardware-driven multi-dimensional descriptor-based transfers with automatic address generation. |
| **Key files**           | `torch/utils/_triton.py`, `torch/_inductor/codegen/cuda/device_op_overrides.py`, `torch/csrc/distributed/c10d/cuda/AsyncMM.cu`, `torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryOps.cu` |


TMA — a Hopper/Blackwell hardware unit for efficient asynchronous multi-dimensional tensor data copies between global memory and shared memory — has no direct equivalent on AMD MI350/CDNA4. It is fully disabled on ROCm:

- On HIP/ROCm, `has_triton_tma()` and the CUDA TMA branch of `has_triton_tma_device()` are gated off via `not torch.version.hip`, so CUDA TMA descriptor entry points are unavailable.
- Host-side TMA descriptors raise `RuntimeError("Host-side TMA descriptors not supported on HIP")`.

CDNA4's Data Movement Engine (DME) and async copy instructions (`GLOBAL_LOAD_LDS`, `ds_read_b64_tr_b16` transposed loads) provide some asynchronous data movement capability, but lack TMA's hardware-level multi-dimensional descriptor support and automatic address generation.

To clarify "similar support" vs "same support":

- **What is similar**: Both NVIDIA Hopper/Blackwell and AMD CDNA-class GPUs provide hardware mechanisms to overlap data movement with compute and reduce explicit copy overhead in kernels.
- **Key hardware model difference**: NVIDIA TMA is descriptor-driven (`CUtensorMap` style metadata for shape/stride/swizzle/OOB handling) and offloads multi-dimensional address generation to dedicated hardware copy logic. AMD currently exposes lower-level async movement primitives in this stack, so more indexing/scheduling behavior stays in kernel/software logic.
- **Software surface in PyTorch/Triton**: CUDA has explicit TMA descriptor entry points (`has_triton_tma*`, host/device descriptor creation, descriptor helper codepaths). HIP path is explicitly gated off (`not torch.version.hip`, runtime error for host-side descriptors), so those codepaths are unavailable on ROCm today.
- **Why this matters in practice**: CK is the AMD-side equivalent of CUTLASS for dense GEMM, but PyTorch does not currently have a CK/ROCm equivalent of the TMA-based persistent async scheduler path used by CUDA kernels such as `_async_input_mm`.

TMA absence also explains TMA-dependent gaps that were previously called out in other sections:

- **Distributed fused all-gather + matmul (`symm_mem::_async_input_mm`)**: In `torch/csrc/distributed/c10d/cuda/AsyncMM.cu`, `BUILD_ASYNC_MM_KERNEL` is only enabled for non-ROCm CUDA 12+ builds. The kernel uses CUTLASS/CuTe `PersistentAsyncInputScheduler` with TMA warp-specialized scheduling (`KernelTmaWarpSpecializedCooperative`). On ROCm, this path is compiled out and runtime falls back (`TORCH_CHECK(false, "async_input_mm is not currently supported on your device")`), so higher-level code in `torch.distributed._symmetric_memory` uses alternative decomposition/native paths.
- **Helion backend limitations linked to descriptor-dependent optimizations**: Helion AMD path currently skips `tensor_descriptor` and eviction-policy cases ([helion#1349](https://github.com/pytorch/helion/pull/1349)); TMA-specific optimizations are therefore unavailable there as well.

**Impact:** TMA-related gaps are not limited to Triton descriptor APIs: they also affect distributed fused communication+compute fast paths and Helion optimization coverage. This is a hardware limitation on current AMD generations — software alternatives can mitigate but not fully reproduce Hopper/Blackwell TMA behavior.

---

### 5. FP4 / NVFP4 / MXFP4 Support


|               |                                                                                                |
| ------------- | ---------------------------------------------------------------------------------------------- |
| **Severity**  | High                                                                                           |
| **Source**    | Source code analysis, v2.10 release notes                                                      |
| **Key files** | `aten/src/ATen/native/cuda/ScaledBlas.cpp`, `torch/_inductor/template_heuristics/triton.py`, `test/test_scaled_matmul_cuda.py` |


FP4 quantization, a major feature for inference efficiency, has divergent support:

- **Important terminology distinction**:
  - `NVFP4` is NVIDIA-specific.
  - The open, vendor-neutral equivalent is OCP `MXFP4`.
- **ROCm does have open-source `MXFP4` support** in parts of the stack (hardware/software gated), including ROCm-focused PyTorch support work ([#151360](https://github.com/pytorch/pytorch/pull/151360)) and Triton AMD `mxfp4` paths (for example [triton#5985](https://github.com/triton-lang/triton/pull/5985) on gfx950).
- **ATen scaled GEMM behavior**: In `ScaledBlas.cpp`, NVFP4 recipes are explicitly unsupported on ROCm, while MXFP4/MXFP8 paths are ROCm/arch gated (for example gfx950 with newer ROCm).
- **NVFP4 test expectations**: Tests explicitly skip NVFP4 on ROCm (`skipIfRocm` in `test_scaled_matmul_cuda.py`), consistent with the ATen-level gating.
- **Inductor template split is broader than NVFP4**: Many CUDA-specific template heuristic registrations (for example Blackwell/TMA families) are guarded by `torch.version.hip is None`; ROCm uses separate heuristic sets. This is a general CUDA/HIP template split, not a dedicated NVFP4 switch.

**Impact:** The practical gap is not "no FP4 at all" on ROCm; it is lack of NVIDIA-specific `NVFP4` parity plus narrower/arch-gated `MXFP4` enablement compared with CUDA's NVFP4-oriented paths.

---

### 6. Mixture of Experts (MoE) Communication and Compute


|                |                                                                                       |
| -------------- | ------------------------------------------------------------------------------------- |
| **Severity**   | High                                                                                  |
| **Source**     | GitHub issues, source code analysis                                                   |
| **Key issues** | [#166807](https://github.com/pytorch/pytorch/issues/166807)                           |
| **Key files**  | `aten/src/ATen/native/cuda/GroupedBlas.cpp`, `torch/_inductor/kernel/mm_grouped.py`, `torch/csrc/distributed/c10d/symm_mem/nvshmem_extension.cu` |


MoE (Mixture of Experts) is the dominant architecture for modern large language models (Mixtral, DeepSeek, Qwen MoE, etc.). On ROCm, the communication side is still missing key APIs, while compute support is present in some grouped-GEMM paths but does not yet match CUDA fast-path coverage/performance:

**Communication (dispatch/combine):**

- `all_to_all_vdev_2d` and `all_to_all_vdev_2d_offset` — the core operations for MoE dispatch and combine — are only implemented on CUDA ([#166807](https://github.com/pytorch/pytorch/issues/166807)). MoE dispatch/combine accounts for 60–70% of communication volume in MoE workloads. Without these operations, MoE training and inference on ROCm is extremely slow. Issue is assigned and in progress.

**Compute (grouped GEMM):**

- Clarification: grouped GEMM functionality exists on ROCm (`_grouped_mm_cuda` and `_scaled_grouped_mm_cuda`), but Inductor's Triton grouped fast-template path is CUDA-only.
- Why Inductor is mentioned here: MoE training/inference stacks usually run with `torch.compile`, so grouped-GEMM performance depends on Inductor lowering/autotuning path selection, not only eager execution.
- **Subset gap for MoE compute**: In `torch/_inductor/kernel/mm_grouped.py`, the Triton grouped kernel path is gated by `can_use_triton_kernel(...)`, which requires `torch.cuda.get_device_capability() >= (9, 0)` and `not torch.version.hip`. HIP therefore cannot use this persistent Triton path.
- **CK caveat**: ATen `_grouped_mm_cuda` has an optional CK fast path on ROCm, but it is conditional (`USE_ROCM_CK_GEMM`, `ROCM_ALLOW_GROUP_GEMM_CK=1`, and arch gating such as `gfx942/gfx950/gfx90a`) rather than universal parity with CUDA fast paths.
- Net effect: ROCm has grouped-GEMM functionality, but MoE grouped-GEMM still has coverage/performance asymmetry versus CUDA's highest-performance path.

**Impact:** End-to-end MoE on ROCm is constrained by (1) missing dispatch/combine communication APIs and (2) grouped-GEMM compute path asymmetry (functional support exists, but fast-path parity with CUDA is incomplete). In practice this often leaves ROCm behind CUDA on MoE throughput and scaling efficiency.

---

### 7. Consumer AMD GPU Support (RDNA3 / RDNA4)


|                |                                                                                                                          |
| -------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Severity**   | High                                                                                                                     |
| **Source**     | GitHub issues                                                                                                            |
| **Key issues** | [#165141](https://github.com/pytorch/pytorch/issues/165141), [#175989](https://github.com/pytorch/pytorch/issues/175989) |


Consumer AMD GPU support has improved but significant issues remain:

- **MIOpen Conv3d crashes on RDNA3**: `nn.Conv3d` with bfloat16 on RX 7900 XTX causes `miopenStatusUnknownError` or ~1400x slowdown with large batches ([#165141](https://github.com/pytorch/pytorch/issues/165141)). Assigned to MIOpen team; CK support for Radeon planned.
- **gfx11/gfx12 backend asymmetry in GEMM backend selection**: For float GEMM under `BlasBackend::Ck`, current code in `CUDABlas.cpp` routes `gfx11/gfx12` to hipBLASLt fallback (`// no CK GEMM version`) instead of CK.
- **Open gfx11 FP8 codegen/test failures**: `test/inductor/test_fp8.py::TestCvtE8M0Rceil` fails on ROCm with `error: couldn't allocate output register for constraint 'h'` ([#175989](https://github.com/pytorch/pytorch/issues/175989)).

**Impact:** Consumer AMD GPU users still face a weaker experience than NVIDIA counterparts in some workloads, due to runtime errors, backend fallback asymmetry, and open compiler/test instability on gfx11-class hardware.

---

### 8. FlexAttention Limitations


|               |                                                           |
| ------------- | --------------------------------------------------------- |
| **Severity**  | Medium–High                                               |
| **Source**    | Source code analysis                                      |
| **Key files** | `torch/_inductor/kernel/flex/flex_attention.py` |


FlexAttention on ROCm has one key limitation compared to CUDA:

- FP32 matmul precision is forced to `'ieee'` on HIP (via `get_float32_precision()`), so FlexAttention does not use the TF32 path. On NVIDIA CUDA, FlexAttention can use `'tf32'` unless global float32 matmul precision is explicitly set to `"highest"`.

**Impact:** FP32 FlexAttention workloads on ROCm lose the TF32 performance path available on NVIDIA CUDA.

---

### 9. Expandable Segments and Memory Allocator Features


|                         |                                                                                                                                                                                                              |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Severity**            | Medium                                                                                                                                                                                                       |
| **Source**              | Source code analysis                                                                                                                                                                                         |
| **Hardware dependency** | No — HIP provides equivalent virtual memory management APIs (`hipMemCreate`, `hipMemAddressReserve`, `hipMemMap`, `hipMemSetAccess`) that could support expandable segments. This is a software porting gap. |
| **Key files**           | `c10/cuda/CUDACachingAllocator.cpp`, `test/test_cuda.py`, `test/distributed/test_p2p_ipc.py` |


The expandable segments feature in CUDA's caching allocator — which allows more efficient GPU memory management by dynamically growing allocations — is explicitly disabled on ROCm:

- `#if !defined(USE_ROCM)` guards around expandable segment logic in the allocator.
- Tests explicitly skip expandable segments on ROCm (`skipIfRocm`).
- PyTorch's `c10::cuda::DriverAPI` wraps CUDA's virtual memory APIs (`cuMemCreate`, `cuMemMap`, etc.) and is CUDA-only, but HIP exposes equivalent APIs (`hipMemCreate`, `hipMemMap`) that are not yet wired up.
- P2P IPC tests for expandable segments are also skipped.

**Impact:** ROCm users may experience less efficient GPU memory utilization, particularly for workloads with dynamic tensor sizes. The underlying HIP APIs exist, so this gap is addressable without hardware changes.

---

### 10. Torch.compile / Inductor Codegen Gaps


|               |                                                                                                                                 |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **Severity**  | Medium                                                                                                                          |
| **Source**    | Source code analysis                                                                                                            |
| **Key files** | `torch/_inductor/config.py`, `torch/_inductor/utils.py`, `torch/_inductor/kernel/mm_grouped.py`, `torch/_inductor/template_heuristics/triton.py` |


Several `torch.compile` / Inductor optimizations remain disabled or degraded on ROCm:


| Feature                          | CUDA                                         | ROCm                                                                                                                                    |
| -------------------------------- | -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Layout optimization              | On by default (`"1"`)                        | Off by default (`"0"`)                                                                                                                  |
| Multi-arch kernel emission       | Enabled by default in AOT standalone mode    | Disabled by default in AOT standalone mode                                                                                              |
| Decompose K splits               | 10                                           | 0 (disabled)                                                                                                                            |
| CUTLASS / CK template coverage   | CUTLASS templates widely available           | CUTLASS unavailable; CK/CKTile are available for selected dense ops, but parity gaps remain (`_int_mm`, sparse 2:4, specialized paths) |
| Multi-kernel default             | Off by default (`TORCHINDUCTOR_MULTI_KERNEL=0`) | Off by default (`TORCHINDUCTOR_MULTI_KERNEL=0`)                                                                                      |
| FP8 inductor register allocation | No matching issue called out in this report | Open gfx11 issue [#175989](https://github.com/pytorch/pytorch/issues/175989) (`test_fp8.py::TestCvtE8M0Rceil`)                        |


Note: the multi-arch row refers specifically to AOT standalone config patching (`aot_inductor_mode.compile_standalone`) in current Inductor utilities.
Note: Grouped matmul and scaled grouped MM (MoE compute kernels) are covered in [Gap 6](#6-mixture-of-experts-moe-communication-and-compute).

**Impact:** `torch.compile` throughput on ROCm lags behind CUDA due to multiple disabled optimization paths, even on high-end datacenter GPUs like MI300X.

---

### 11. Helion DSL — Backend and Feature Parity


|               |                                                                         |
| ------------- | ----------------------------------------------------------------------- |
| **Severity**  | Medium                                                                  |
| **Source**    | Source code analysis (PyTorch main + `pytorch/helion` repo)             |
| **Key files** | `torch/utils/_helion.py`, `torch/_inductor/select_algorithm.py`, `test/inductor/test_helion_kernels.py`, `pytorch/helion` release notes |


[Helion](https://github.com/pytorch/helion) is a Python-embedded DSL for writing fast, portable ML kernels with minimal boilerplate. In current PyTorch main, optional Helion integration hooks already exist (for example `torch/utils/_helion.py` and `ExternalTritonTemplateKernel` in `torch/_inductor/select_algorithm.py`), with test coverage gated by `requires_helion`. While Helion's design goal is cross-vendor portability, backend coverage is still uneven:

- **Triton backend** (primary): Works on ROCm via Triton's AMD backend. Active AMD contributions are landing — `num_warps`/`num_stages` ranges tuned for AMD ([helion#1368](https://github.com/pytorch/helion/pull/1368)), and previously-skipped ROCm tests are being re-enabled ([helion#1340](https://github.com/pytorch/helion/pull/1340), [helion#1341](https://github.com/pytorch/helion/pull/1341)).
- **CuTe backend** (new in v0.3.0): CUDA-only. CuTe is part of the NVIDIA CUTLASS ecosystem and has no ROCm equivalent. This backend enables a separate code generation path that ROCm users cannot access.
- **Pallas backend** (new in v0.3.0): Targets JAX/XLA; not directly relevant to ROCm-via-HIP but expands Helion's reach to other accelerators.
- **AMD-specific limitations**: `tensor_descriptor` and eviction policy are skipped on AMD ([helion#1349](https://github.com/pytorch/helion/pull/1349)). (Related low-level movement/descriptor details are consolidated in [Gap 4](#4-tensor-memory-accelerator-tma).)
- **NVFP4 GEMM**: Added to Helion ([helion#1403](https://github.com/pytorch/helion/pull/1403)) — CUDA-only, mirroring Gap 5 above.

**Impact:** Helion's primary Triton backend is functional on ROCm, but the CuTe backend and hardware-specific features (including NVFP4) create an expanding CUDA-only surface area. As Helion gains adoption for custom kernels and `torch.compile` fusion, the backend asymmetry will increasingly matter.

---

## Honorable Mentions


| Gap                                | Severity | Notes                                                                                                                                                             |
| ---------------------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **cuSPARSELt Float/Float8 dtypes** | Medium   | Float32 SpMMA and FP8 disabled on ROCm; hipSparseLt limited to gfx942/gfx950                                                                                      |
| **LSTM/RNN performance**           | Medium   | MIOpen RNN ~4–5x slower than cuDNN; AMP with CudnnLSTM broken on ROCm                                                                                             |
| **Windows ROCm: no training**      | Medium   | Only inference supported; no `torch.distributed` ([AMD docs](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.0.2/docs/limitations/limitationsrad.html)) |
| **ROCm not on PyPI**               | Low      | Requires custom index URL; no official PyPI package                                                                                                               |


---

## Summary Table


| #   | Feature Gap                                                        | Severity    | HW Gap? | Evidence Source              |
| --- | ------------------------------------------------------------------ | ----------- | ------- | ---------------------------- |
| 1   | Inductor GEMM Templates / Semi-Structured Sparsity (CUTLASS vs CK) | High        | No      | Source code, release notes   |
| 2   | cuDNN SDPA Attention                                               | High        | No      | Source code                  |
| 3   | Symmetric Memory Multicast / Async Distributed MatMul              | High        | **Yes** | Source code, release notes   |
| 4   | TMA Descriptors                                                    | High        | **Yes** | Source code                  |
| 5   | FP4 / NVFP4 / MXFP4 Support                                        | High        | No      | Source code, release notes   |
| 6   | MoE Communication and Compute                                      | High        | No      | GitHub issues, source code   |
| 7   | Consumer GPU Support (RDNA3/RDNA4)                                 | High        | No      | GitHub issues                |
| 8   | FlexAttention Limitations                                          | Medium–High | No      | Source code                  |
| 9   | Expandable Segments / Memory Allocator                             | Medium      | No      | Source code                  |
| 10  | Torch.compile / Inductor Gaps                                      | Medium      | No      | Source code                  |
| 11  | Helion DSL — Backend and Feature Parity                            | Medium      | No      | Source code + pytorch/helion |


---

## Appendix A: Test Coverage

The PyTorch test suite contains many uses of `skipIfRocm` and related decorators on current main. Key areas:


| Area                           | Skip Patterns                                             |
| ------------------------------ | --------------------------------------------------------- |
| **Sparse semi-structured**     | Most tests skipped on ROCm                                |
| **CUTLASS (no CK equivalent)** | All CUTLASS-related tests skipped                         |
| **Scaled matmul / FP4**        | NVFP4 skipped; MXFP4 constraints differ                   |
| **Transformers / attention**   | cuDNN attention, deterministic algorithms, nested tensors |
| **Inductor**                   | Grouped MM, PDL, scaled_grouped_mm                        |
| **Distributed**                | Symmetric memory fastpaths, async_input_mm, expandable segments, P2P IPC |
| **Linalg**                     | MAGMA, eigenvalue, SVD operations                         |
| **Nested tensors**             | Several nested tensor tests skipped                       |


## Appendix B: Open ROCm-Labeled Issues (as of March 2026)


| Issue                                                       | Description                                                |
| ----------------------------------------------------------- | ---------------------------------------------------------- |
| [#166807](https://github.com/pytorch/pytorch/issues/166807) | MoE Dispatch & MoE Combine not supported on ROCm (single-node and multi-node) |
| [#175989](https://github.com/pytorch/pytorch/issues/175989) | `test_fp8.py::TestCvtE8M0Rceil` register allocation failure (`constraint 'h'`) |
| [#175482](https://github.com/pytorch/pytorch/issues/175482) | DISABLED `test_index_put_error_cuda` (`TestNestedTensorSubclassCUDA`) |
| [#174913](https://github.com/pytorch/pytorch/issues/174913) | Test: `TestLinalg.test_tensorinv` |
| [#174313](https://github.com/pytorch/pytorch/issues/174313) | `[release 2.11][triton]` ROCm trunk test failures |
| [#175211](https://github.com/pytorch/pytorch/issues/175211) | Replace `get_device_capability()` checks with feature queries in accelerator tests |
| [#173761](https://github.com/pytorch/pytorch/issues/173761) | DISABLED `test_two_layer_fully_shard_cudagraph` (`TestFullyShardCudaGraph`) |


---

*Report generated from PyTorch main (commit 605845d4891). For the latest status on any gap, check the linked GitHub issues and the [PyTorch ROCm documentation](https://pytorch.org/docs/stable/notes/hip.html).*
