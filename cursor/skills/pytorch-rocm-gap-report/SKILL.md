---
name: pytorch-rocm-gap-report
description: Generate or refresh a PyTorch CUDA-vs-ROCm feature gap report. Use when the user asks for a PyTorch feature gap report, ROCm parity report, CUDA vs HIP comparison, or analysis of PyTorch ROCm missing features.
---

# PyTorch ROCm Gap Report

## Purpose

Create or refresh a Markdown report comparing PyTorch CUDA backend capabilities with ROCm/HIP support. The report should be evidence-driven, current to the requested PyTorch version, and explicit about whether each gap is caused by hardware limits, software integration gaps, backend library coverage, test-only coverage, or unclear status.

## Startup Questions

Before analysis, ask the user:

1. Which PyTorch version should be analyzed?
   - `main`
   - latest PyTorch release
   - specific tag, branch, commit, or local checkout
2. Should the agent update the last existing report or create a new report from scratch?
   - update last report
   - create new report

If the user says "latest release", identify the latest local or upstream PyTorch release tag and compare it to the current checkout. If the user says "update", locate the prior Markdown report and preserve useful structure while correcting stale claims.

## Required Methodology

The methodology section must include:

- Source code analysis of the selected PyTorch revision.
- Test-suite analysis, especially tests skipped or xfailed on ROCm/HIP.
- GitHub issue/PR status checks when live status matters.
- Release/tag comparison when the report is tied to a release window.

Analyze at least these code patterns:

- `USE_ROCM`, `torch.version.hip`, `TEST_WITH_ROCM`, `skipIfRocm`, `xfail`, `onlyCUDA`, `onlyAccelerator`.
- Backend-library gates for CK, CK-Tile, hipBLASLt, rocBLAS, MIOpen, AOTriton, MSLK, AITER, composable kernel, cuDNN, cuBLASLt, CUTLASS, CuTe, and Triton.
- Inductor choices: `ExternKernelChoice`, `TritonTemplate`, CUTLASS templates, CK/CK-Tile templates, template heuristics, `autotune_select_algorithm`, and fallback lowerings.
- Platform support helpers such as `PLATFORM_SUPPORTS_*`, `isGPUArch(...)`, `get_device_capability()`, and ROCm arch checks like `gfx90a`, `gfx942`, `gfx950`, `gfx11*`, `gfx12*`.

## Domain Rules

- Treat **Composable Kernel (CK) / CK-Tile as the ROCm equivalent of CUTLASS**. Do not frame "CUTLASS missing on ROCm" as the gap by itself. The gap is whether CK/CK-Tile or another ROCm backend provides equivalent coverage.
- Distinguish eager ATen support from `torch.compile` / Inductor support.
- Distinguish functional fallback from optimized template/autotune support.
- Distinguish a CUDA-only hardware feature from a portable feature that ROCm has not wired up yet.
- Do not claim a gap is hardware-related unless the evidence shows the missing behavior depends on NVIDIA-only hardware semantics with no AMD equivalent exposed in the current stack.
- Mark uncertain claims as "Needs verification" instead of overstating.

## Gap Classification

For every reported gap, classify the root cause as one or more of:

- **Hardware limitation**: depends on NVIDIA-only hardware behavior with no current AMD equivalent exposed.
- **Software integration gap**: ROCm/HIP has plausible primitives, but PyTorch has not wired them into the relevant path.
- **Backend library coverage gap**: the ROCm equivalent library lacks the needed operation, dtype, layout, or arch support.
- **Compiler/template gap**: eager support exists, but Inductor/Triton/CK/CK-Tile template or autotune support is missing.
- **Test coverage gap**: tests are skipped or disabled, but source support may exist.
- **Unclear / needs verification**: evidence is incomplete or conflicting.

## Workflow

1. Identify the target PyTorch tree and revision:
   - Record path, branch, commit, tag, and date.
   - If comparing to a release, compute tag relationship and commit distance.
2. Locate or create the report:
   - If updating, read the full existing report first.
   - Preserve deliberate notes unless the user asks to remove them.
3. Build an evidence inventory:
   - Search source for ROCm guards and backend gates.
   - Search tests for ROCm skips, xfails, disabled tests, and platform support helpers.
   - Check linked GitHub issues/PRs for title, state, and relevance.
4. Validate each candidate gap:
   - Determine whether eager ATen works.
   - Determine whether Inductor lowers it.
   - Determine whether optimized templates exist for CUDA and ROCm.
   - Identify fallback behavior and performance implications.
5. Write or refresh the report with concise evidence and links.
6. Re-read the report for contradictions, stale paths, and overbroad hardware claims.

## Report Structure

Use this structure by default:

```markdown
# PyTorch CUDA vs ROCm/HIP Feature Gap Report

**Version analyzed:** ...
**Date:** ...
**Scope:** ...

## Methodology

## Notable Gaps Closed Since [baseline]

## Top Feature Gaps

### 1. [Gap Name]

| Field | Value |
| ----- | ----- |
| Severity | High/Medium/Low |
| Root cause | Hardware / software / backend / compiler-template / test / unclear |
| Source | Source analysis / tests / issues / release notes |
| Key files | `path`, `path` |
| Key tests | `test_name`, `test_name` |

Explanation with evidence.

**Impact:** ...
**Status / next steps:** ...

## Honorable Mentions

## Summary Table

## Appendix A: Test Coverage

## Appendix B: Issue / PR References
```

## Evidence Standards

- Prefer current source and tests over stale issues.
- Cite file paths and symbol names. Include short code excerpts only when they clarify the claim.
- For GitHub issues/PRs, verify current title and state before using them as evidence.
- If a test is skipped on ROCm, determine whether it indicates missing functionality, missing validation, or a CUDA-only feature.
- If a path falls back to ATen, explain whether that fallback is functional, optimized, or likely slow.

## Common Checks

- Sparse semi-structured MM: check both ATen support and Inductor template choices; CUDA may use CUTLASS sparse templates while ROCm may lack CK/CK-Tile equivalent support.
- Grouped MM / scaled grouped MM: check ATen fallback, CK env gates such as `ROCM_ALLOW_GROUP_GEMM_CK`, MSLK build gates, and Inductor template support separately.
- `_int_mm`: check ATen backend and Inductor template heuristics separately.
- TMA / PDL: classify carefully as CUDA/NVIDIA-specific codegen unless AMD exposes an equivalent primitive in the current stack.
- SDPA / attention: distinguish cuDNN, AOTriton, CK SDPA, memory-efficient attention, FlashAttention, and FlexAttention.

## Final Verification

Before finalizing:

- Confirm the report names the analyzed commit/tag.
- Confirm every "hardware gap" label is justified.
- Confirm CK/CK-Tile is used as the ROCm counterpart to CUTLASS where appropriate.
- Confirm Appendix A includes test-suite skip/xfail evidence.
- Confirm stale file paths, line references, and issue descriptions are updated.
