# PyTorch Inductor Code Analysis Summary

**Target Directory:** `/home/niromero/docker_workspace/pytorch/torch/_inductor`

**Analysis Date:** February 2, 2026

**Scope:** Recursive analysis of 29 subdirectories, 291 Python files, and 14 Jinja templates

---

## Table of Contents

1. [Hard-coded Warp Sizes](#1-hard-coded-warp-sizes)
2. [matrix_instr_nonkdim Usage](#2-matrix_instr_nonkdim-usage)
3. [TMA (Tensor Memory Access) Usage](#3-tma-tensor-memory-access-usage)
4. [Search Statistics](#4-search-statistics)
5. [Recommendations](#5-recommendations)

---

## 1. Hard-coded Warp Sizes

### Summary
Found **15+ instances** of hard-coded warp sizes (32 or 64) that should ideally be queried at runtime. Additionally, found 10+ warp-related calculations using hard-coded values.

### Critical Findings

#### 1.1 `runtime/triton_heuristics.py`

**Lines 2277-2279** - Has TODO indicating should query warp size:
```python
warp_size = (
    64 if torch.version.hip else 32
)  # TODO: query warp size once #129663 is merged
```
- **Status:** Hard-coded fallback (32 for CUDA, 64 for HIP)
- **Priority:** HIGH - Has explicit TODO to fix

**Line 171** - Fallback warp size:
```python
device_props.warp_size if device_props.warp_size else 32
```
- **Status:** Hard-coded fallback to 32

**Line 493** - Autotuning fallback:
```python
warp_size = device_prop.warp_size or 32
```
- **Status:** Hard-coded fallback to 32

**Line 1173** - XPU backend fallback:
```python
params["threads_per_warp"] = getattr(
    launcher.bin.metadata, "threads_per_warp", 32
)
```
- **Status:** Hard-coded default of 32 (XPU can be 16)

**Line 2659** - Hard-coded elements per warp:
```python
size_hints, bs // 2, num_elements_per_warp=64
```
- **Status:** Hard-coded 64

**Lines 2707-2708** - Config values:
```python
triton_config_with_settings(size_hints, 32, 32),
triton_config_with_settings(size_hints, 64, 64),  # ~8% better for fp16
```
- **Status:** Hard-coded config values

**Line 2487** - Max warps threshold:
```python
max_num_warps = 16 if r <= 8192 else 32
```
- **Status:** Hard-coded threshold of 32

**Line 3493** - Warp comparison:
```python
if c.num_warps < 32:
```
- **Status:** Hard-coded threshold

#### 1.2 `runtime/hints.py`

**Line 105** - Module constant:
```python
_NUM_THREADS_PER_WARP = 32
```
- **Status:** Hard-coded module-level constant
- **Priority:** HIGH - Used throughout the codebase

**Line 174** - Default warp size:
```python
warp_size=getattr(props, "warp_size", 32 if device_type != "cpu" else None),
```
- **Status:** Hard-coded default of 32 for non-CPU

#### 1.3 `runtime/triton_compat.py`

**Lines 141-149** - Function with hard-coded returns:
```python
def cc_warp_size(cc: str | int) -> int:
    if torch.version.hip:
        cc_str = str(cc)
        if "gfx10" in cc_str or "gfx11" in cc_str:
            return 32
        else:
            return 64
    else:
        return 32
```
- **Status:** Hard-coded returns based on architecture
- **Priority:** HIGH - Central function for warp size determination

#### 1.4 `codegen/cuda/device_op_overrides.py`

**Line 123** - Hard-coded in kernel launch template:
```python
func, gridX, gridY, gridZ, 32*numWarps, 1, 1, sharedMemBytes, stream, args, nullptr
```
- **Status:** Hard-coded 32 in template
- **Note:** Line 131 replaces this for HIP with actual warp_size

**Line 175** - Block size check:
```python
if (elementSize * blockDim < 32) {
```
- **Status:** Not a warp size (minimum transaction size for TMA)

#### 1.5 `utils.py`

**Line 2693** - Fallback warp size:
```python
warp_size = 32
```
- **Context:** Used in `get_max_numwarps()` when CUDA unavailable
- **Status:** Hard-coded fallback

#### 1.6 `codegen/rocm/rocm_template.py`

**Line 33** - ROCm class constant:
```python
gfx9_threads_per_warp = 64
```
- **Status:** Hard-coded to 64 (AMD GFX9 wavefront size)
- **Note:** Architecture-specific, but could query device

#### 1.7 `codegen/rocm/ck_universal_gemm_template.py`

**Line 537** - Method parameter default:
```python
def _prefetch_stages(self, op, a_dtype_size, b_dtype_size, warp_size: int = 64):
```
- **Status:** Hard-coded default of 64
- **Note:** Line 510 passes actual device warp_size when called

#### 1.8 `codegen/rocm/ck_tile_universal_gemm_template.py`

**Lines 136, 176, 214** - Warp tile dimensions:
```python
for (warp_tile_m, warp_tile_n, warp_tile_k) in [(32, 32, 16)]
```
- **Status:** Hard-coded tile dimensions (warp-related, not warp size itself)

#### 1.9 `choices.py`

**Line 437** - Thread calculation:
```python
num_threads = 32 * num_warps
```
- **Status:** Assumes 32 threads per warp

**Line 402** - Threshold calculation:
```python
threshold *= 32 // min(
    V.graph.sizevars.size_hint_or_throw(features.numel), 32
)
```
- **Status:** Uses 32 in calculation

**Line 431** - Min elements per thread:
```python
min_elements_per_thread = 32
```
- **Status:** Hard-coded value (may relate to warp size)

#### 1.10 `template_heuristics/triton.py`

**Line 706** - Register calculation:
```python
gemm_config.block_m * gemm_config.block_n / (gemm_config.num_warps * 32)
```
- **Status:** Hard-coded 32 in warp calculation

**Line 2379** - Block size calculation:
```python
min_elem_per_thread * _NUM_THREADS_PER_WARP * num_warps,
```
- **Status:** Uses imported constant (32 from hints.py)

### False Positives (Not Warp Sizes)

These use 32/64 but are NOT warp sizes:
- **Swizzle sizes** in `codegen/cuda/device_op_overrides.py:223-226` (memory alignment thresholds)
- **Config block sizes** that happen to be 32 or 64

---

## 2. matrix_instr_nonkdim Usage

### Summary
**AMD/ROCm-specific parameter** for controlling MFMA (Matrix-Fused-Multiply-Add) instruction shapes. Found **31 occurrences across 7 files**.

### Purpose
Controls the shape of MFMA instructions on AMD GPUs. Values:
- `0` = No specific MFMA constraint
- `16` = Use 16x16 MFMA tiles

### Key Locations

#### 2.1 Definition - `template_heuristics/triton.py`

**Config Classes (Lines 146, 157, 168, 179, 190):**
- `ROCmGemmConfig` - default: `16`
- `ROCmConvConfig` - default: `16`
- `ROCmFlexConfig` - default: `0`
- `ROCmFlexBwDConfig` - default: `0`
- `ROCmFlexDecodeConfig` - default: `0`

**Line 1196** - Autotuning config generation:
```python
for matrix_instr_nonkdim in [0, 16]
```
- **Purpose:** Explores both values during autotuning

#### 2.2 Validation - `template_heuristics/triton.py`

**Lines 1322-1341** - In `_finalize_mm_configs()`:
```python
matrix_instr_nonkdim = getattr(config, "matrix_instr_nonkdim", 16)
# ... validation logic ...
if matrix_instr_nonkdim != 0:
    if block_m % matrix_instr_nonkdim != 0 or block_n % matrix_instr_nonkdim != 0:
        # Invalid config - block dimensions must be multiples
```
- **Purpose:** Ensures block dimensions are multiples of `matrix_instr_nonkdim`

**Line 1292** - Config pruning:
```python
# Filter problematic configs: matrix_instr_nonkdim==2 and kpack==2 causes AMD crashes
```

#### 2.3 Compilation - `runtime/triton_heuristics.py`

**Lines 697, 759-760** - Compile options:
```python
if self.device_props.type == "hip":
    if "matrix_instr_nonkdim" in compile_meta:
        options["matrix_instr_nonkdim"] = compile_meta["matrix_instr_nonkdim"]
```
- **Purpose:** Passes to Triton compiler for HIP/ROCm

**Lines 2396-2397** - Config creation:
```python
if torch.version.hip:
    if matrix_instr is not None:
        config.kwargs["matrix_instr_nonkdim"] = matrix_instr
```

#### 2.4 Algorithm Selection - `select_algorithm.py`

**Lines 652, 655-656** - Meta extraction:
```python
matrix_instr_nonkdim = self.meta.get("matrix_instr_nonkdim")
if matrix_instr_nonkdim is not None:
    triton_meta["matrix_instr_nonkdim"] = matrix_instr_nonkdim
```

**Lines 2106, 2133** - Template caller:
```python
# Passed to TritonTemplateCaller with default 0
# Included in log info for debugging
```

**Line 2720** - Tunable key:
- Listed as one of the tunable keys for algorithm selection cache

#### 2.5 Autotuning - `autotune_process.py`

**Lines 612, 623** - TritonTemplateCaller:
```python
def __init__(self, ..., matrix_instr_nonkdim: int = 0, ...):
    # only used for hip to choose the shape of mfma instruction
    self.matrix_instr_nonkdim = matrix_instr_nonkdim
```

#### 2.6 Flex Kernels - `kernel/flex/`

**`flex_attention.py:408` and `flex_decoding.py:335`:**
```python
for attrib in ["kpack", "matrix_instr_nonkdim", "waves_per_eu"]:
    if hasattr(conf, attrib):
        cur_kernel_options[attrib] = getattr(conf, attrib)
```
- **Purpose:** Extracts ROCm-specific attributes from config

#### 2.7 Code Generation - `codegen/wrapper.py`

**Line 274** - Grid generation:
```python
if kwarg not in [
    "matrix_instr_nonkdim",
    "waves_per_eu",
    "kpack",
]:
```
- **Purpose:** Excludes from guard conditions to simplify generated code

### Code Flow Summary

```
Definition (ROCm config classes)
    ↓
Autotuning (explores [0, 16])
    ↓
Validation (block dims must be multiples)
    ↓
Compilation (passed to Triton for HIP)
    ↓
Execution (stored in TritonTemplateCaller)
    ↓
Code Generation (excluded from guards)
```

---

## 3. TMA (Tensor Memory Access) Usage

### Summary
**TMA is extensively used** throughout Inductor for NVIDIA GPUs with compute capability >= 9.0. Found **93 TMA matches across 18 files**, plus 209 total "TMA" occurrences.

### What is TMA?

**Tensor Memory Access** - NVIDIA GPU feature for efficient tensor memory operations introduced in Hopper (H100) and newer architectures.

### Requirements (from `config.py:1607-1638`)

- NVIDIA GPUs with compute capability >= 9.0 (Hopper/H100+)
- Innermost stride must be 1
- Block shape must load/store at least 16 bytes in innermost dimension
- Tensors must be 16-byte aligned
- Requires Triton >= 3.4.0 with stable TMA API support

### Key Components

#### 3.1 Core IR Nodes - `ir.py:6793-6913`

**Base class (Line 6793):**
```python
class TMADescriptor(ExternKernel):
    """An IR node representing a generic host-side TMA descriptor in the Triton API"""
```

**API variants:**
- `TMADescriptorExperimental` (Line 6857) - Old Triton TMA API
- `TMADescriptorStable` (Line 6897) - New Triton TMA API

**Line 7118** - Factory method:
```python
t = TMADescriptor.create(t, tma_descriptor_metadata[k])
```

#### 3.2 Constants - `utils.py:149-150`

```python
TMA_ALIGNMENT = 16
TMA_DESCRIPTOR_SIZE = 128
```

#### 3.3 Compatibility Checking - `codegen/triton.py:2238-2473`

**Class: `TMACompatibilityChecker`**
```python
class TMACompatibilityChecker:
    """Checks if the TMA API can be used for load / store triton operations."""
```

**Key validation methods:**
- `can_use_tma()` - Main compatibility check
- `are_block_parameters_compatible()` - Block parameter validation

**Restrictions enforced:**
- Innermost stride must be 1 (Line 2307)
- Outer strides must be 16-byte aligned (Line 2320)
- Dtype requirements for element size (Line 2390)

#### 3.4 Utility Functions - `utils.py`

**Lines 1713-1724** - `get_tma_workspace_arg()`:
```python
def get_tma_workspace_arg(...):
    """Builds and returns a WorkspaceArg for the device side TMA workspace buffer."""
    size = num_programs * num_tma_descriptors * TMA_DESCRIPTOR_SIZE
```

**Lines 1794-1912** - `can_use_tma()`:
```python
def can_use_tma(...):
    """Return True iff *all* supplied tensors satisfy the CUDA-12.9 TMA constraints"""
```

**Line 1905** - `use_triton_tma_template()`
**Line 1916** - `use_triton_blackwell_tma_template()`

#### 3.5 Code Generation

##### 3.5.1 Python Wrapper - `codegen/wrapper.py:1611-1658`

**Methods:**
- `_generate_tma_descriptor_call_experimental()` (Line 1611)
- `_generate_tma_descriptor_call_stable()` (Line 1633)
- `_generate_tma_descriptor_call()` (Line 1646) - Dispatcher
- `generate_tma_descriptor()` (Line 1655) - Main entry point

**Line 1628** - API call generation:
```python
fn = f"{prefix}.create_{desc.rank}d_tma_descriptor"
```

##### 3.5.2 C++ Wrapper (AOTI) - `codegen/cpp_wrapper_gpu.py`

**Line 437** - `write_tma_descriptor_helpers_once()`
**Line 539** - `generate_tma_descriptor()`
**Line 548** - `_generate_experimental_tma_descriptor()`
**Line 574** - `_generate_stable_tma_descriptor()`

**Line 658** - Note about AOTI TMA handling:
```python
# [Note: AOTI TMA Stable handling]
# TMA descriptors, a single python arg turns into 1 + 2 * N args
```

##### 3.5.3 CUDA Helpers - `codegen/cuda/device_op_overrides.py:135-321`

**Line 135** - `tma_descriptor_helpers()`:
```python
"""CUDA helper functions for initializing TMA Descriptors on host side"""
```

**Host-side initialization functions:**
- `init1DTMADescriptor()` (Line 149) - 1D descriptors
- `init2DTMADescriptor()` (Line 188) - 2D descriptors
- `initTMADescriptor()` (Line 242) - Generic (new API)

**Line 314** - Struct definition:
```python
struct StableTMADescriptor {
    void* data;
    uint64_t* shape;
    uint64_t* stride;
    int32_t rank;
};
```

**Line 140** - HIP restriction:
```python
# Host-side TMA descriptors not supported on HIP
```

#### 3.6 Kernel Arguments - `codegen/common.py:306-311`

```python
@dataclass
class TMADescriptorArg:
    name: str
    api_type: str  # "experimental" or "stable"
    block_shape: Optional[list[sympy.Expr]]
    dtype: Optional[torch.dtype]
```

Used in: `KernelArgType = Union[WorkspaceArg, TensorArg, SizeArg, TMADescriptorArg, ConstexprArg]`

#### 3.7 Template Usage

##### 3.7.1 Matrix Multiplication - `kernel/mm.py:95-119`

**TMA-enabled templates:**
- `persistent_tma_mm_template` (Line 95)
- `scaled_mm_device_tma_epilogue_scaling_template` (Line 102)
- `scaled_mm_device_tma_main_loop_scaling_template` (Line 109)
- `blackwell_ws_persistent_device_tma_mm_template` (Line 115)

##### 3.7.2 Template Heuristics - `template_heuristics/triton.py`

**Line 1752** - `class TMAWorkspaceMixin`:
```python
"""Small mixin to ensure that the workspace arg is correct for TMA
and TMA specific filtering can happen"""
```

**Line 1779** - `class TMATemplateConfigMixin`:
```python
"""TMA-specific mixin that uses persistent configs and adds TMA-specific options"""
```

**Line 1815** - `class BlackwellTMATemplateConfigMixin`:
- Blackwell (B200/B100) specific TMA support

**TMA config options (Line 1802-1803):**
```python
"TMA_SIZE": TMA_DESCRIPTOR_SIZE
"TMA_EXPERIMENTAL_API": not has_triton_stable_tma_api()
```

**TMA-specific heuristics:**
- `CUDAPersistentTMATemplateConfigHeuristic` (Line 2116)
- `CUDABlackwellPersistentTMATemplateConfigHeuristic` (Line 2132)
- `CUDAAddmmPersistentTMATemplateConfigHeuristic` (Line 2148)
- `CUDABlackwellAddmmPersistentTMATemplateConfigHeuristic` (Line 2160)
- `CUDAScaledTMAEpilogueScalingTemplateConfigHeuristic` (Line 2197)
- `CUDAScaledTMAMainLoopScalingTemplateConfigHeuristic` (Line 2214)
- `CUDAScaledBlackwellTMATemplateConfigHeuristic` (Line 2280)
- `XPUPersistentTMATemplateConfigHeuristic` (Line 2525)
- `XPUAddmmPersistentTMATemplateConfigHeuristic` (Line 2537)

**Line 2003** - `class ScaledTMAConfigMixin`:
```python
"""Scaled TMA-specific mixin that extends BaseScaledMMConfigMixin with TMA functionality
This is for scaled MM templates that use device TMA"""
```

**TMA filtering (Lines 2014-2018):**
```python
# TMA specific filtering:
# - num_warps=2 not safe for TMA
# - block_k >= 32 required for TMA (requires inner-most dimension >= 32)
configs = [c for c in configs if c.num_warps != 2 and c.block_k >= 32]
```

##### 3.7.3 Flex Attention - `kernel/flex/`

**Files:**
- `flex_attention.py` - TMA loads for Q, K, V tensors
- `flex_decoding.py` - TMA for decoding
- `templates/common.py.jinja` - TMA template code
- `templates/flex_attention.py.jinja` - TMA setup
- `templates/flex_decode.py.jinja` - TMA decode kernels

**`USE_TMA` flags throughout for conditional TMA usage**

#### 3.8 Runtime Heuristics - `runtime/triton_heuristics.py`

**Lines 2575-2612** - `_maybe_filter_configs_for_tma_restrictions()`:
```python
def _maybe_filter_configs_for_tma_restrictions(
    configs: list[Any],
    tma_min_block_sizes: dict[str, int],
    ...
):
```
- **Purpose:** Filters kernel configs based on TMA API restrictions
- **Log message:** "Filtering configs for TMA API restrictions"

#### 3.9 Algorithm Selection - `select_algorithm.py`

**Line 66** - Import:
```python
from .codegen.triton import TMACompatibilityChecker
```

**Line 392** - Parameter:
```python
tma_store=False
```

**Line 407** - Validation:
```python
# Error: "TMA store only supported for 2D with templates"
```

**Line 1448** - Checker instance:
```python
tma_compatibility_checker: Optional[TMACompatibilityChecker] = None
```

**Line 2718** - Tunable key:
```python
"USE_TMA"
```

#### 3.10 Configuration - `config.py`

**Line 1607-1638** - Documentation comment:
```python
# [Note: TMA API Restrictions] Currently the TMA API requires the following:
# - For Nvidia GPUs, the compute capability should be >= 9.0
# - The innermost stride of a descriptor should be 1
# - The size of the block shape in the innermost dimension should load / store at least 16 bytes
# - Tensors are 16 byte aligned
#
# TMA descriptors are only going to be generated if the above conditions
# ... with a version of triton new enough to support TMA
```

**Configuration flags:**
- `enable_persistent_tma_matmul` (Line 1633) - Enable persistent TMA matmul
- `enable_template_tma_store` (Line 1638) - Enable TMA store from templates
- `cutlass_tma_only` (Line 1917) - Only use TMA-compatible CUTLASS kernels

**Line 1937** - CUTLASS epilogue:
```python
# Set this to 'warpspecialized_cooperative_epi_tma' to enable only SM90 TMA Cutlass Kernels
```

#### 3.11 CUDA Integration - `codegen/cuda/`

**`gemm_template.py`** - TMA epilogue support checks
**`cuda_template.py`** - TMA template handling
**`cuda_cpp_scheduling.py`** - TMA kernel scheduling

**CUTLASS TMA integration** - Selects TMA-compatible GEMM kernels

#### 3.12 Scheduler - `scheduler.py`

**Line 3926** - Note:
```python
# Currently, persistent+TMA Triton template does not due to the TMA-based loads
```

### TMA Usage Patterns

1. **Host-side TMA descriptors** - Created on CPU, describe tensor layout
2. **Device-side TMA workspace** - GPU memory for TMA operations
3. **Persistent kernels** - Long-running kernels that benefit from TMA
4. **Matrix multiplication** - Primary use case for TMA
5. **Flex attention** - Q, K, V tensor loading with TMA
6. **Scaled MM** - Scaled matrix multiplication with TMA

### Files with TMA References (18 files)

1. `ir.py` (5 matches)
2. `utils.py` (2 matches)
3. `config.py` (7 matches)
4. `scheduler.py` (1 match)
5. `select_algorithm.py` (4 matches)
6. `codegen/triton.py` (13 matches)
7. `codegen/wrapper.py` (1 match)
8. `codegen/cpp_wrapper_gpu.py` (6 matches)
9. `codegen/cuda/device_op_overrides.py` (4 matches)
10. `codegen/cuda/gemm_template.py` (3 matches)
11. `codegen/cuda/cuda_template.py` (1 match)
12. `codegen/cuda/cuda_cpp_scheduling.py` (1 match)
13. `runtime/triton_heuristics.py` (1 match)
14. `template_heuristics/triton.py` (37 matches)
15. `kernel/mm.py` (1 match)
16. `kernel/flex/flex_attention.py` (1 match)
17. `kernel/flex/templates/common.py.jinja` (4 matches)
18. `kernel/flex/templates/flex_attention.py.jinja` (1 match)

---

## 4. Search Statistics

### File Coverage
- **Total directories searched:** 29 (recursive)
- **Total Python files:** 291
- **Total Jinja templates:** 14
- **Total files:** 312

### Pattern Matches

| Pattern | Matches | Files |
|---------|---------|-------|
| `\b(32\|64)\b` | 707 | 64 |
| `warp_size` (case-insensitive) | 27 | 8 |
| `threads_per_warp` | 4 | 4 |
| `num_warps` | 216 | 17 |
| `matrix_instr_nonkdim` | 31 | 7 |
| `\bTMA\b` | 93 | 18 |

### Directory Coverage

All subdirectories were recursively searched:
```
_inductor/
├── analysis/
├── autoheuristic/artifacts/
├── codegen/
│   ├── aoti_runtime/
│   ├── cuda/
│   │   └── cutlass_lib_extensions/cutlass_mock_imports/{cuda,pydot,scipy}/
│   ├── cutedsl/
│   ├── mtia/
│   ├── rocm/
│   └── xpu/
├── compile_worker/
├── fx_passes/serialized_patterns/
├── kernel/
│   ├── flex/templates/
│   ├── templates/
│   └── vendored_templates/
├── lookup_table/
├── package/
├── runtime/caching/
└── template_heuristics/
```

---

## 5. Recommendations

### 5.1 Hard-coded Warp Sizes

**Priority Actions:**

1. **HIGH:** Address the TODO in `runtime/triton_heuristics.py:2277-2279`
   - Replace hard-coded conditional with runtime query
   - Reference: Issue #129663

2. **HIGH:** Make `_NUM_THREADS_PER_WARP` constant dynamic in `runtime/hints.py:105`
   - Consider making it a device property instead of module constant
   - Update all references to query device at runtime

3. **HIGH:** Refactor `cc_warp_size()` in `runtime/triton_compat.py:141-149`
   - Query device properties instead of architecture string matching
   - Support future GPU architectures automatically

4. **MEDIUM:** Update kernel launch template in `codegen/cuda/device_op_overrides.py:123`
   - Make warp size a template parameter
   - Currently only HIP gets dynamic replacement (line 131)

5. **LOW:** Review fallback values throughout codebase
   - Many fallbacks to 32 are reasonable for CUDA
   - Consider if they should error instead of silently defaulting

### 5.2 matrix_instr_nonkdim

**Status:** ✅ **Well-implemented**

- Properly configured as ROCm-specific parameter
- Good validation and error handling
- Autotuning explores appropriate values [0, 16]
- No action required

### 5.3 TMA (Tensor Memory Access)

**Status:** ✅ **Well-implemented**

- Comprehensive TMA support for H100+ GPUs
- Proper abstraction layers (IR nodes, compatibility checking)
- Good documentation of requirements and restrictions
- Configuration flags for controlling TMA usage

**Potential Improvements:**

1. **Documentation:** Expand user-facing documentation on when TMA is used
2. **Error messages:** Improve error messages when TMA requirements not met
3. **Testing:** Ensure robust fallback when TMA unavailable
4. **Performance:** Continue expanding TMA usage to more kernel types

### 5.4 General Code Quality

**Strengths:**
- Good separation of concerns (ROCm vs CUDA vs XPU)
- Architecture-specific code well-organized
- Template system flexible and extensible

**Areas for Improvement:**
- Reduce hard-coded architecture assumptions
- More runtime queries for device properties
- Better support for emerging GPU architectures

---

## Appendix: Key Files by Category

### Warp Size Management
- `runtime/hints.py` - Core constants and device properties
- `runtime/triton_compat.py` - Warp size helper functions
- `runtime/triton_heuristics.py` - Heuristics using warp size
- `codegen/cuda/device_op_overrides.py` - CUDA kernel launch
- `codegen/rocm/rocm_template.py` - ROCm warp/wavefront handling

### ROCm/AMD Specific
- `template_heuristics/triton.py` - ROCm config classes
- `codegen/rocm/ck_universal_gemm_template.py` - CK library integration
- `codegen/rocm/ck_tile_universal_gemm_template.py` - CK tile GEMM
- `kernel/flex/flex_attention.py` - Flex attention with ROCm support

### TMA Infrastructure
- `ir.py` - TMA IR nodes
- `utils.py` - TMA utilities and compatibility
- `codegen/triton.py` - TMA compatibility checker
- `codegen/wrapper.py` - TMA descriptor code generation (Python)
- `codegen/cpp_wrapper_gpu.py` - TMA descriptor code generation (C++)
- `codegen/cuda/device_op_overrides.py` - CUDA TMA helpers
- `template_heuristics/triton.py` - TMA template heuristics
- `runtime/triton_heuristics.py` - TMA config filtering

### Templates and Kernels
- `kernel/mm.py` - Matrix multiplication templates
- `kernel/flex/flex_attention.py` - Flex attention kernels
- `kernel/flex/flex_decoding.py` - Flex decoding kernels
- `kernel/flex/templates/*.jinja` - Jinja templates for flex operations

---

**End of Analysis Summary**
