# Priority kernel diversity set

This document describes the representative fused-kernel set in
`inductor/kernels/priority/priority_representative_kernel_paths.txt`.

Source corpus:
`/home/niromero/docker_workspace/Triton_Conv_Development/kernels/priority`

The selection is based on the generated Triton body and benchmark inputs, not
just the filename or exact source identity. The goal is to spend optimization
time on different execution mechanisms rather than on every model occurrence
of the same mechanism.

## Scope and method

All 1,037 kernels in the three requested categories parsed successfully:

- 624 pointwise kernels
- 284 non-persistent reduction kernels
- 129 persistent-reduction kernels
- 15 model captures in total; 14 contain reductions and persistent reductions

For every file, the analysis used:

- the `triton_heuristics` category and `ReductionHint`
- the `@triton.jit` body, including operation order, loop depth, masks,
  accumulator operations, pointer expressions, indirect loads/stores, and
  multi-output epilogues
- `triton_meta` and `inductor_meta`, including pointer dtypes, load/store and
  reduction counts, mutation, and grid type
- every `rand_strided` input/output in `get_args()`, including shape, stride,
  dtype, and scalar numel arguments
- actual `xnumel`, `r0_numel`, and fixed/padded `R0_BLOCK` values in the body

Three levels of similarity were distinguished:

1. Exact generated bodies, including shape constants.
2. Shape-normalized bodies, with graph ordinals, temporary names, and numeric
   shape constants normalized.
3. Optimization families, which additionally group bodies when they exercise
   the same operation DAG, memory-access topology, reduction regime, masks,
   accumulator type, and IO contract.

A new optimization slot is justified by a changed algorithm, indexing or
layout topology, reduction hint, loop depth, indirect/atomic behavior,
materially different IO fan-in/fan-out, or a geometry boundary such as
`R0_BLOCK > r0_numel`. Model name, graph position, hash suffix, and an isolated
change in `xnumel` are not enough.

## Corpus-wide redundancy

Filename deduplication substantially overstates diversity:

- The 624 pointwise files have 157 graph-position-independent names, 397 exact
  bodies, and about 195 structural fingerprints.
- The 284 reduction files have 75 graph-position-independent names, 210 exact
  bodies, and about 165 structural fingerprints.
- The 129 persistent-reduction files have 59 graph-position-independent names,
  97 exact bodies, 55 structural fingerprints, and about 25 broad
  optimization families.
- `_to_copy` alone contributes 248 pointwise files. Across the broader simple
  cast family, 279 files reduce mostly to fp32-to-bf16, bf16-to-fp32, and
  tail-masked forms.
- 131 corpus files have hash-suffixed filenames while sharing a kernel
  function name with another file.

The model captures also repeat whole groups of kernels:

- GPT-OSS 20B and 120B have 100% shape-normalized body overlap in all three
  categories.
- DeepSeek V4 Flash and Pro overlap by about 96% for pointwise, 69% for
  reduction, and 79% for persistent reduction.
- Wan 2.1 and Wan 2.2 overlap by about 81% for pointwise and 78% for reduction.

Consequently, model-balanced sampling would repeatedly optimize the same
families.

## Pointwise patterns

All 624 pointwise kernels use `Grid1D` and a flattened `xnumel`. There are 441
unconditional/full-mask bodies and 183 tail-masked bodies. No RNG kernels were
found.

The dominant pattern is linear cast/copy. It is followed by non-contiguous
layout materialization, causal/bit-mask generation, cat/split/permute fusions,
SiLU/GELU forward and backward, binary arithmetic, and multi-input or
multi-output fusions.

The important optimization distinctions are:

- linear bandwidth kernels versus div/mod layout decomposition
- downcast versus upcast store/load byte ratios
- direct versus gathered loads and indirect scatter stores
- tail masking and conditional regions
- one-output chains versus shared-compute fan-out
- activation/transcendental work (`exp`, `tanh`, `sigmoid`, `erf`, `rsqrt`)
- RoPE table generation versus RoPE application
- register-pressure outliers with dozens or hundreds of input pointers
- in-place mutation and attention/norm backward epilogues

The representative list keeps 27 pointwise kernels spanning these mechanisms.
It does not keep one cast or activation per model.

## Non-persistent reduction patterns

All 284 reduction kernels reduce one `r0` dimension. There are no `r1` or true
two-dimensional reductions.

Reduction hints are:

- 112 `INNER`
- 89 `DEFAULT`
- 82 `OUTER`
- 1 `OUTER_TINY`

Body-derived reduction algorithms are dominated by 224 sum-based kernels,
followed by 46 Welford LayerNorm kernels, 10 RMSNorm sum-of-squares kernels,
2 online-softmax kernels, and 2 sum-plus-atomic embedding-backward kernels.
Loop depth ranges from one to four passes.

The important optimization distinctions are:

- coalesced inner reductions versus strided outer reductions
- short outer reductions versus long default reductions
- plain sum, sum-of-squares/rsqrt, Welford, online softmax, and atomic scatter
- single-pass reductions versus multi-pass reduction-plus-epilogue bodies
- compact inputs versus high-fan-in fused gradients
- contiguous rows versus slice, transpose, convolution, and embedding
  addressing
- fp32 accumulation with bf16 inputs and multi-output normalization epilogues

The representative list keeps 20 reduction kernels, including the unique
online-softmax, atomic, `OUTER_TINY`, four-loop, and 79-input cases.

## Persistent-reduction patterns

The 129 persistent reductions divide into about 25 broad optimization
families. Their hints are:

- 76 `INNER`
- 29 `OUTER`
- 12 `OUTER_TINY`
- 12 `DEFAULT`

For 99 files the fixed persistent block equals the logical reduction length.
For 30 files it is padded above the logical length. The padded cases are
important, especially softmax widths 129/130/161 with `R0_BLOCK=256` and
LayerNorm backward with `r0_numel=3072`, `R0_BLOCK=4096`.

Most files are simple sum-derived variants, but the meaningful families also
include full online softmax, causal softmax, RMSNorm, affine LayerNorm,
normalization backward, gather/index reductions, masked ReLU sums, and
activation-gradient reductions. No persistent `tl.dot`, atomic reduction,
Welford, or log-softmax body appears in this corpus.

The representative list keeps 25 persistent reductions. Similar-looking sum
skeletons are retained only when they cross a scheduling boundary such as
`OUTER` versus `OUTER_TINY`, equal versus padded `R0_BLOCK`, output dtype, or
strided input layout.

## Selected set

The exact-path list contains 72 existing files:

- 27 pointwise
- 20 reduction
- 25 persistent reduction

All 72 paths are unique and all resolve in the source corpus. Their exact
Triton bodies are also all unique. They form 69 shape-normalized skeletons;
the three additional slots are deliberate reduction-regime boundary cases,
not model duplicates.

The list is path-based because `paas-organize --target-kernels-file` accepts
kernel function names rather than paths. Function names are not unique in this
corpus, so a name-only set cannot reliably select a particular body and input
shape.

## Stage the set for PAAS

From the PAAS repository:

```bash
SOURCE=/home/niromero/docker_workspace/Triton_Conv_Development/kernels/priority
DEST=/tmp/priority-representative
LIST=/home/niromero/docker_workspace/framework_scripts/inductor/kernels/priority/priority_representative_kernel_paths.txt

rm -rf "$DEST"
python -m paas.helpers.search_kernels \
  --dir "$SOURCE" \
  --out "$DEST" \
  --names "$LIST"
```

`search_kernels` preserves the model/category relative paths, so duplicate
basenames cannot overwrite one another.

PAAS discovers the staged files recursively:

```bash
paas-inductor \
  --dir "$DEST" \
  --distributed \
  --run-types=autotune
```

Do not apply the test-suite path regex
`^(pointwise|reduction|persistent_reduction)/` to this tree: the first path
component is the model. No pattern is needed after staging because the
directory contains only selected kernels.

`paas-make-standalone` is non-recursive, so run it once for every populated
model/category directory:

```bash
for category_dir in "$DEST"/*/{pointwise,reduction,persistent_reduction}; do
  [ -d "$category_dir" ] || continue
  paas-make-standalone \
    --mode tune \
    --launch-params-suffix=.autotune.launch_params \
    "$category_dir"
done
```

`paas-simple-full` also searches only its current directory for `*_tune.py`.
Run it per populated category directory, or use the recursive
`paas-inductor` stage alone.

## Known PAAS caveats

- The source priority tree is already categorized; `paas-organize` is not
  required.
- The initial capture has no `.autotune.launch_params` sidecars.
  `paas-inductor --run-types=autotune` creates them before standalone tuning.
- `paas-inductor` discovery is recursive, but `paas-make-standalone` and
  `paas-simple-full` are directory-local.
- `paas-pick-important` is currently unusable because its search helper raises
  from a leftover debug statement. The exact-path `search_kernels` helper is
  the safe existing staging path.

## Fingerprinting recommendation

Future selections should store a stable fingerprint with these fields:

```text
category
normalized_body_hash
operation_DAG_class
input/output pointer dtypes and roles
load/store/reduction counts
indexing class: linear, div-mod, broadcast, gather, indirect-store, atomic
mask class
mutation flag
ReductionHint
xnumel and r0_numel size buckets
R0_BLOCK relation: equal or padded
loop count and accumulator dtype
```

Cluster first on the algorithm and access fields. Use size buckets only to
retain boundary cases within a cluster. This prevents graph names, model
names, and ordinary shape changes from recreating the current duplication.
