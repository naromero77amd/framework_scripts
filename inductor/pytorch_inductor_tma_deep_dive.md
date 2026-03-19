# PyTorch Inductor TMA Deep Dive

This document focuses on how TMA support is currently implemented in Inductor, what depends on Triton, and what would likely be needed for the AMD TDM analog.

## Scope and Constraints

- Source tree analyzed: `/home/niromero/pytorch`
- Runtime validation: not performed (no local NVIDIA GPU required/available)
- Method: static source analysis + upstream Triton references

## Short Answers First

1. **Is Inductor TMA support currently Gluon-based?**
   - **No** in this checkout.
   - There are no Gluon/TTGL references in `torch/_inductor` right now.
   - The implementation uses Triton APIs and template code paths directly (currently with `experimental_*` API names in this tree).

2. **If not Gluon-based, how does it work?**
   - Inductor explicitly selects TMA-capable templates, generates TMA-specific Triton source, allocates workspace for descriptors, and plumbs descriptor argument types/signatures.
   - Triton then compiles those explicit TMA operations; this is not "automatic discovery" from generic loads.

3. **For AMD, is the TMA-like feature TDM, and would Gluon help?**
   - Potentially yes, because it provides backend-oriented ops and lowering hooks.
   - But Inductor would still need backend-specific capability checks, legality rules, autotuning integration, and fallbacks.

4. **Why not only generate generic Triton and let Triton decide TMA/TDM?**
   - Inductor must decide legality and profitability, allocate extra resources, choose schedules/templates, and keep safe fallbacks.
   - Compiler lowering is only one part of the pipeline.

5. **For AMD TDM, do we need Inductor to emit `amdg.*` dialect ops directly?**
   - Usually no.
   - Inductor should emit Triton-level APIs/templates; Triton lowering should emit `amdg.*`.
   - Purely generic Triton ops can work in some cases, but TDM-specific templates are preferred for predictable legality/perf.

---

## 1) TMA in Inductor Is Not Just a Config Flag

There is a config gate:

- `torch/_inductor/config.py`:
  - `config.triton.enable_persistent_tma_matmul` (env `ENABLE_PERSISTENT_TMA_MATMUL`)

But this gate only enables selection. Real support required multiple non-config changes:

- Capability checks in `torch/utils/_triton.py`
- Compatibility filters in `torch/_inductor/utils.py`
- New template code paths in `torch/_inductor/kernel/mm.py` and `torch/_inductor/kernel/mm_scaled_grouped.py`
- Workspace descriptor plumbing via `WorkspaceArg`
- Descriptor IR and signature typing (`TMADescriptor`, `TMADescriptorArg`, `nvTmaDesc`)
- Wrapper/AOT descriptor generation paths
- Scheduler and launcher caveats

---

## 2) How Current TMA Support Works (Non-Gluon Path)

### 2.1 Capability and legality gating

`torch/utils/_triton.py` gates TMA on CUDA + architecture + API availability:

```python
if (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability() >= (9, 0)
    and not torch.version.hip
):
    from triton.language.extra.cuda import (
        experimental_device_tensormap_create1d,
        experimental_device_tensormap_create2d,
    )
```

`torch/_inductor/utils.py` then enforces template-level legality:

```python
def use_triton_tma_template(*matrices: IRNode) -> bool:
    ...
    return (
        config.triton.enable_persistent_tma_matmul
        and has_triton_tma_device()
        and all(_is_tma_compatible(m) for m in matrices)
    )
```

and checks alignment/shape/type constraints (e.g., 16-byte inner alignment).

### 2.2 Template selection and workspace

In `torch/_inductor/kernel/mm.py`, Inductor appends TMA template choices only when predicate passes:

```python
if use_triton_tma_template(mat1, mat2):
    persistent_tma_mm_template.maybe_append_choice(
        choices,
        ...,
        workspace_arg=get_tma_workspace_arg(
            num_tma_descriptors=2,
            device=mat1.get_device(),
        ),
        ...
    )
```

Workspace size is explicit in `torch/_inductor/utils.py`:

```python
size = num_programs * num_tma_descriptors * TMA_DESCRIPTOR_SIZE
```

### 2.3 TMA-specific Triton code generation (device-side)

`torch/_inductor/kernel/mm.py` emits explicit TMA operations:

```python
triton.language.extra.cuda.experimental_device_tensormap_create2d(...)
tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(a_desc_ptr)
a = tl._experimental_descriptor_load(...)
```

The scaled grouped MM path does the same in `torch/_inductor/kernel/mm_scaled_grouped.py`.

### 2.4 Host-side descriptor path for user-defined Triton kernels

For `triton_kernel_wrap` flows:

- metadata originates in `torch/_higher_order_ops/triton_kernel_wrap.py` via `tma_descriptor_metadata`
- lowered to `ir.UserDefinedTritonKernel(...)` in `torch/_inductor/lowering.py`
- converted to `TMADescriptor` IR in `torch/_inductor/ir.py`
- wrapper emits host descriptor calls in `torch/_inductor/codegen/wrapper.py`:

```python
prefix = "triton.tools.experimental_descriptor"
fn = f"{prefix}.create_{desc.rank}d_tma_descriptor"
```

### 2.5 Signature and launcher implications

`torch/_inductor/codegen/triton_utils.py` maps descriptor args to Triton signature type:

```python
if isinstance(arg, TMADescriptorArg):
    return "nvTmaDesc"
```

Static CUDA launcher currently does not support this:

```python
elif ty == "nvTmaDesc":
    raise NotImplementedError("nvTmaDesc kernels are not yet supported")
```

So some TMA forms are effectively JIT-path dependent.

---

## 3) What Changed for TMA Support (Concrete Checklist)

Inductor TMA support required:

- **Selection logic**
  - `use_triton_tma_template(...)` predicate for capability + legality.
- **Kernel templates**
  - new/extended templates that emit TMA-specific Triton ops.
- **Workspace management**
  - descriptor storage via `WorkspaceArg` and wrapper allocation/deallocation.
- **IR model**
  - `TMADescriptor` IR node for host-side descriptor path.
- **Argument model**
  - `TMADescriptorArg` and `nvTmaDesc` signature emission.
- **Wrapper codegen**
  - Python and C++ helper paths to materialize descriptors.
- **Runtime/scheduler behavior**
  - prologue fusion caveats for persistent+TMA templates.
- **Launcher behavior**
  - known static launcher gap for `nvTmaDesc`.

---

## 4) Triton Compiler Side: Dependencies and Expectations

### 4.1 Practical dependencies visible in PyTorch code

- CUDA device present
- Compute capability >= 9.0 (Hopper+)
- Not HIP/ROCm for current CUDA TMA path
- Triton package exposes required symbols (import-probe style checks)
- TMA legality constraints (alignment/layout/type) pass in Inductor predicates
- For C++ host-side helper code: `CUDA_VERSION >= 12000` in generated helper block

### 4.2 What Triton is expected to provide

From this checkout, Inductor expects these APIs to exist:

- `triton.language.extra.cuda.experimental_device_tensormap_create{1d,2d}`
- `tl.extra.cuda.experimental_tensormap_fenceproxy_acquire`
- `tl._experimental_descriptor_load`
- `triton.tools.experimental_descriptor.create_{1d,2d}_tma_descriptor`

If these symbols are missing, TMA paths are naturally disabled (or break where directly called).

### 4.3 API evolution risk and version pin sensitivity

Upstream Triton has changed descriptor/TMA APIs over time. In particular:

- [triton-lang/triton#6488](https://github.com/triton-lang/triton/pull/6488) removed/renamed older experimental descriptor surfaces.

This means:

- your effective PyTorch+Triton pin matters,
- API naming in the local checkout may lag or rely on compatibility behavior,
- "works on one Triton build" does not guarantee "works on another" without pin alignment.

---

## 5) FAQ (Basic Questions)

### Q: Is current Inductor TMA support Gluon-based?

**No.** In this checkout, TMA support is built around explicit Triton CUDA descriptor/tensormap APIs in Inductor templates and wrappers.

### Q: If it is not Gluon-based, how does it still work?

Inductor does four major things:

1. selects TMA templates (`use_triton_tma_template`),
2. emits TMA-specific Triton source (`experimental_device_tensormap_create2d`, descriptor loads),
3. allocates/plumbs descriptor workspace and descriptor arg types,
4. delegates lowering/codegen to Triton compiler backends.

### Q: If Gluon is used, does AMD new-arch support become easier?

Usually easier, yes. But still requires Inductor work for:

- backend feature detection,
- backend legality checks,
- template choice/autotune integration,
- scheduler/fusion constraints,
- fallback and testing behavior.

### Q: Why not only generate generic Triton and let Triton compiler figure out TMA details?

Because compiler lowering cannot replace frontend policy decisions:

- **Legality**: alignment/layout/dtype constraints must be enforced before choosing a path.
- **Profitability**: autotune/template selection happens in Inductor.
- **Resource plumbing**: workspace and descriptor args are explicit runtime artifacts.
- **Safety/fallback**: Inductor must preserve non-TMA fallbacks when constraints fail.

So the split is:

- **Inductor** decides *whether/where* to use TMA and emits the right primitives.
- **Triton** decides *how* to lower those primitives to target instructions.

### Q: For AMD TDM specifically, does Inductor need to emit AMD dialect ops directly?

Usually **no**, and with today’s architecture it should not.

- Inductor currently generates Triton kernels/source (Python-level Triton DSL), not Triton MLIR dialect IR directly.
- AMD dialect ops (for example, `amdg.async_tdm_*`) are Triton compiler internal/lowering-level representations.
- The normal integration model is:
  1. Inductor emits Triton-level operations/APIs that express async descriptor/data-movement intent.
  2. Triton lowers those to backend dialect ops (`amdg.*` for AMD).

What this means in practice:

- You generally do **not** add direct `amdg.*` dialect emission in Inductor.
- You **do** add/extend Inductor predicates + templates so generated Triton code is TDM-friendly and reliably triggers Triton’s AMD lowering path.
- Relying only on fully generic `tl.load`/`tl.store` may yield partial compiler optimizations, but it usually provides less control and weaker guarantees than explicit async/descriptor-style templates.

### Q: Inductor supports custom Triton kernels. Does it support custom Gluon kernels?

Not as a first-class API in this checkout.

- The custom-kernel path in Inductor is Triton-specific via `triton_kernel_wrapper_*`
  in `torch/_higher_order_ops/triton_kernel_wrap.py`.
- That wrapper path expects Triton runtime kernel objects (`JITFunction`/`Autotuner`)
  and lowers through `ir.UserDefinedTritonKernel` in `torch/_inductor/lowering.py`.
- Dynamo and descriptor reconstruction hooks are also Triton-specific (e.g.
  `triton.tools.experimental_descriptor` in `torch/_dynamo/variables/*`).
- There is no parallel, first-class custom-`gluon` wrapper/operator path in
  `torch/_inductor` or `torch/_dynamo` in this tree.

Practical interpretation:

- You can provide custom Triton kernels to Inductor.
- If Triton internally lowers those kernels through Gluon/TTNG paths, that is an internal Triton backend/lowering detail, not an Inductor-level custom Gluon API.

---

## 6) AMD TDM Support from Inductor: What Would Be Needed

Current upstream signals suggest AMD-side TDM and async tensor-movement building blocks exist in Triton/Gluon:

- [triton-lang/triton#7220](https://github.com/triton-lang/triton/pull/7220)
- [triton-lang/triton#7880](https://github.com/triton-lang/triton/pull/7880)
- [triton-lang/triton#8333](https://github.com/triton-lang/triton/pull/8333)

In Triton docs, AMD dialect operations include forms like:

- `amdg.async_wait`
- `amdg.async_tdm_copy_global_to_local`
- `amdg.async_tdm_wait`

### 6.1 Minimum Inductor work items for an AMD analog

1. **Capability probes**
   - Add explicit checks in `torch/utils/_triton.py` for AMD/Gluon APIs.

2. **Legality predicate**
   - Add an AMD-specific predicate analogous to `use_triton_tma_template(...)`.

3. **Template additions**
   - Add AMD-specific template variants in MM paths (`mm.py`, `mm_scaled_grouped.py`), with backend-appropriate async/tensor-move ops.

4. **Option/symbol plumb-through**
   - Add any required template constants, descriptor sizes, barrier semantics, or argument encodings.

5. **Wrapper/AOT handling**
   - If AMD path uses host descriptors, add wrapper/C++ helper support.
   - If descriptor-in-kernel only, focus on workspace/runtime plumbing.

6. **Scheduler + autotune integration**
   - Add/adjust fusion and autotune constraints similar to current TMA paths.

7. **Fallback and tests**
   - Preserve non-AMD-async fallback paths and add targeted correctness/perf tests.

### 6.2 Should Inductor support AMD dialect ops directly?

Recommended approach:

- Keep Inductor at Triton API/template level.
- Do not directly model `amdg.*` dialect operations in Inductor.
- Add Triton-level TDM-oriented template code and capability gates, then rely on Triton lowering to generate AMD dialect ops.

Only consider direct dialect emission if Inductor architecture changes to generate Triton MLIR directly (not the current model).

Can we just use regular Triton ops?

- Sometimes yes: backend passes may still optimize/load-store patterns.
- But for consistent TDM behavior, explicit TDM-oriented templates and gating in Inductor are more reliable than hoping generic `tl.load`/`tl.store` patterns are recognized in all cases.
- Recommended strategy:
  1. keep Inductor at Triton API level,
  2. add explicit TDM-friendly templates where performance/behavior matters,
  3. let Triton lower to AMD dialect ops.

### 6.3 Illustrative pseudo-API shape (design sketch)

```python
def use_triton_amd_async_template(*matrices):
    from torch.utils._triton import has_triton_amd_async_device
    return (
        config.triton.enable_persistent_amd_async_matmul
        and has_triton_amd_async_device()
        and all(_is_amd_async_compatible(m) for m in matrices)
    )
```

```python
# Pseudo-template intent (names illustrative):
# - create/update descriptor or async transfer state
# - async global->shared transfers
# - explicit wait/barrier points
# - compute on local/shared tiles
```

---

## 7) Upstream Triton gfx1250 TDM Evolution (Chronological)

This section summarizes additional upstream work after initial gfx1250 TDM enablement.

### 7.1 Initial bring-up

- [#8333](https://github.com/triton-lang/triton/pull/8333): initial TDM support on gfx1250 (2D scope, descriptor-in-kernel focus).
- [#8392](https://github.com/triton-lang/triton/pull/8392): TDM store support on gfx1250.
- [#8479](https://github.com/triton-lang/triton/pull/8479): skinny block support for TDM.
- [#8510](https://github.com/triton-lang/triton/pull/8510): `ttg.async_wait` support on gfx1250 path.

### 7.2 Descriptor model expansion

- [#8722](https://github.com/triton-lang/triton/pull/8722): initial host-side TDM descriptor exposure in Gluon.
- [#8743](https://github.com/triton-lang/triton/pull/8743): TDM load/store support for 1D-5D.
- [#8977](https://github.com/triton-lang/triton/pull/8977): host TDM descriptor support for 1D-5D on gfx1250.
- [#9730](https://github.com/triton-lang/triton/pull/9730): col-major support for device-side TDM descriptors.

### 7.3 Data movement feature growth (gather/scatter/prefetch)

- [#9086](https://github.com/triton-lang/triton/pull/9086): TDM L2 prefetch support.
- [#9299](https://github.com/triton-lang/triton/pull/9299): tensor async scatter support (gfx1250/gluon).
- [#9313](https://github.com/triton-lang/triton/pull/9313): tensor async gather support using TDM.
- [#9369](https://github.com/triton-lang/triton/pull/9369): `PaddedSharedLayout` support in TDM gather.
- [#9774](https://github.com/triton-lang/triton/pull/9774) (open): predicate support in TDM gather for gfx1250.

### 7.4 Correctness and robustness fixes

- [#9371](https://github.com/triton-lang/triton/pull/9371): OOB handling fixes for TDM scatter/gather.
- [#9496](https://github.com/triton-lang/triton/pull/9496): tensordesc index fix after kernel launch changes.
- [#9720](https://github.com/triton-lang/triton/pull/9720): buffer race fix in pipelined loops with TDM loads.
- [#9725](https://github.com/triton-lang/triton/pull/9725): TDM assert typo fix.

### 7.5 Scheduling/perf heuristics and pipeline work

- [#9302](https://github.com/triton-lang/triton/pull/9302): software pipelining support with TDM on gfx1250.
- [#9741](https://github.com/triton-lang/triton/pull/9741): improved LDS padding heuristic for gfx1250 TDM dot loads.
- [#9747](https://github.com/triton-lang/triton/pull/9747): unified padded layout heuristic across async copy and TDM paths.

### 7.6 Enablement and test hardening

- [#9250](https://github.com/triton-lang/triton/pull/9250): TDM enabled by default (AMD path).
- [#9718](https://github.com/triton-lang/triton/pull/9718): tensor descriptor mode coverage added to `test_matmul.py`.
- [#8680](https://github.com/triton-lang/triton/pull/8680): gfx1250 Gluon test updates.

### 7.7 Adjacent ongoing infrastructure

- [#9717](https://github.com/triton-lang/triton/pull/9717) (open): AMD backend support for Triton-to-Gluon translator.

Interpretation:

- The trajectory is clear: initial TDM functionality was followed by descriptor-generalization, gather/scatter expansion, wait/pipeline integration, and a long tail of correctness/perf fixes.
- This reinforces that "support exists" is only phase 1; production-quality backend support needs sustained follow-up in compiler + tests + heuristics.

---

## 8) Current TMA Data Flow in Inductor

```mermaid
flowchart TD
cfgGate["config.triton.enable_persistent_tma_matmul"] --> legalityGate["use_triton_tma_template()"]
legalityGate --> choiceAppend["persistent_tma_mm_template.maybe_append_choice(...)"]
choiceAppend --> wsAlloc["get_tma_workspace_arg()"]
choiceAppend --> tmaKernelSrc["mm.py emits tensormap_create + descriptor_load"]
tmaKernelSrc --> tritonCompile["runtime/triton_heuristics.py -> triton.compile(...)"]

userKernelPath["triton_kernel_wrap tma_descriptor_metadata"] --> tmaIR["ir.TMADescriptor"]
tmaIR --> wrapperEmit["wrapper.py create_{1d,2d}_tma_descriptor"]
wrapperEmit --> sigType["triton_utils: TMADescriptorArg -> nvTmaDesc"]
sigType --> staticGap["static_cuda_launcher: nvTmaDesc not supported"]
```

---

## 9) Practical Notes for Your Environment

- You can fully reason about architecture and codegen structure without an NVIDIA GPU.
- What you cannot validate locally is runtime performance and generated PTX/SASS behavior for TMA kernels.
- For source-level understanding and design work, this static analysis is sufficient.

---

## 10) Bottom Line

- Current Inductor TMA support is **not Gluon-based**.
- It is **not just config**; it required explicit changes across selection, templates, IR, wrappers, args/signatures, runtime plumbing, and launcher behavior.
- Relying only on generic Triton code is insufficient because Inductor must make policy and resource decisions before Triton lowering.
- For AMD, the equivalent path is TDM; support is feasible, but still requires substantial Inductor-side integration.
- Inductor should generally target Triton-level APIs/templates and rely on Triton to lower into AMD dialect ops, rather than emitting AMD dialect ops directly.
