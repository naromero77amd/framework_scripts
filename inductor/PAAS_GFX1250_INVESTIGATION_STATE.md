# PAAS gfx1250 investigation state

Saved on 2026-09-16 for transfer to another computer.

## Important measurement-policy boundary

- The 107-kernel A/B cohort was tuned with PAAS commit
  `5fa811d60f0e1e3e4ac2fa3dc09946012e4deb29`. That version used one persistent,
  warmed validation worker per kernel.
- The current local campaign branch is `paas-gfx1250-campaign` at
  `50b7aeb774a76f8c32aeffc9db7d4012fecaf1c4`. It restores one fresh process and
  one fresh reference result per candidate.
- The 107 kernels have **not** been rerun under the current fresh-process policy.
- One additional reduction kernel was fully tuned under the current policy in
  both fast/native and rigorous/FP64 modes.

There are therefore 108 unique kernels with completed full tuning work recorded
below: 107 under the cached-worker A/B policy and one under the current
fresh-process policy.

## Artifact locations

- Frozen old 107-kernel cohort:
  `attempts/ab107-old-20260914/`
- Cached-worker retune of the same 107 kernels:
  `attempts/ab107-new-20260915/`
- Exact machine-readable 107-kernel selection:
  `attempts/ab107-new-20260915/selection.json`
- 107-kernel comparison summary:
  `attempts/ab107-new-20260915/ab_summary.md`
- Deterministic fresh-process reduction benchmark:
  `attempts/single-reduction-validation-20260916-1230-seed0/`
- Deterministic benchmark raw results:
  `attempts/single-reduction-validation-20260916-1230-seed0/results.json`
- Benchmark driver:
  `single_reduction_validation_benchmark.py`

The 107-kernel cached-worker retune completed 10,325 candidates in 4h27m55s.
The archived old run contained 11,889 candidates and took 51h56m, for an
observed 11.63x wall-time improvement. That comparison bundled the persistent
worker, native comparison, shorter Triton timing budget, and reduced search
space; it does not isolate any one change.

The deterministic fresh-process reduction experiment searched 52 configurations
per mode with seed 0 and separate empty caches:

- Fast/native: 565.291 seconds total; 428.024 seconds in the validation window.
- Rigorous/FP64: 600.746 seconds total; 461.014 seconds in the validation window.
- Both modes produced 44 OK and 8 wrong candidates, with the same eight
  configurations rejected.

## Code state

- Current local campaign branch: `paas-gfx1250-campaign`
- Current local campaign commit:
  `50b7aeb774a76f8c32aeffc9db7d4012fecaf1c4`
- The campaign branch is local and has no configured upstream branch.
- Validation draft PR:
  <https://github.com/AMD-ROCm-Internal/inductor-triton-hacks/pull/27>
- Pushed PR branch: `niromero_cached_native_validation`
- Pushed PR branch commit:
  `a575cb2bd6f7191a81c3bfe9b3d61bda929fc9d0`
- Deduplication is not present on the campaign branch.

The Markdown file records the state but does not transfer the CSVs, logs,
caches, benchmark script, or local-only campaign commits. Copy the campaign
directory or create a Git bundle/push the campaign branch before leaving this
machine.

## Existing priority plan status

`/root/.cursor/plans/paas_priority_optimization_66ef93cd.plan.md` is **not up to
date**. Its completed setup, portability, pilot, and baseline tasks remain
useful, and its full-campaign tuning/report tasks are still pending. However, it
predates the current search and validation policy:

- It does not record the hard 1,048,576-element Triton tensor filter.
- It does not record the hybrid reduction cap
  `max(16,384, 2 × baseline block product)`.
- It does not record the 25 ms warmup / 100 ms measurement `do_bench` policy.
- It does not record the fast/native and rigorous/FP64 validation modes.
- It does not record the persistent-worker experiment, its rollback, or the
  current fresh-process behavior.
- Its unconditional `--append` guidance is unsafe across the policy boundary.
  Current PAAS refuses to append fresh-process results to a CSV containing
  persistent-worker rows.
- It does not record the completed 107-kernel A/B cohort or the deterministic
  single-reduction benchmark.

Use this investigation-state document for the current handoff unless that plan
is explicitly revised.

## 107-kernel A/B cohort

Paths are relative to the cohort's `tune/` directory.

1. `MiniMaxAI_MiniMax-M3_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_1_tune.py`
2. `MiniMaxAI_MiniMax-M3_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_view_0_tune.py`
3. `MiniMaxAI_MiniMax-M3_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_view_2_tune.py`
4. `MiniMaxAI_MiniMax-M3_bs8_amp_bf16/pointwise/triton_poi_fused_clone_0_tune.py`
5. `MiniMaxAI_MiniMax-M3_bs8_amp_bf16/pointwise/triton_poi_fused_embedding_0_tune.py`
6. `Qwen_Qwen3-235B-A22B_bs8_amp_bf16/persistent_reduction/triton_per_fused__softmax__to_copy_prepare_softmax_online_2_tune.py`
7. `Qwen_Qwen3-235B-A22B_bs8_amp_bf16/persistent_reduction/triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_view_2_tune.py`
8. `Qwen_Qwen3-235B-A22B_bs8_amp_bf16/persistent_reduction/triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_view_5_tune.py`
9. `Qwen_Qwen3-235B-A22B_bs8_amp_bf16/persistent_reduction/triton_per_fused__to_copy_div_sum_3_tune.py`
10. `Qwen_Qwen3-235B-A22B_bs8_amp_bf16/persistent_reduction/triton_per_fused_mul_sum_1_tune.py`
11. `Qwen_Qwen3-235B-A22B_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_0.9d1bf481_tune.py`
12. `Qwen_Qwen3-235B-A22B_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_0.9f276c1b_tune.py`
13. `Qwen_Qwen3-235B-A22B_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_1.94fb6af0_tune.py`
14. `Qwen_Qwen3-235B-A22B_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_1.cb1c46ac_tune.py`
15. `Qwen_Qwen3-235B-A22B_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_4_tune.py`
16. `Qwen_Qwen3-235B-A22B_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_9_tune.py`
17. `Qwen_Qwen3-235B-A22B_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy__unsafe_view_add_cat_clone_expand_mul_neg_slice_transpose_unsqueeze_6_tune.py`
18. `Qwen_Qwen3-235B-A22B_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_add_arange_bitwise_and_index_le_new_ones_unsqueeze_0_tune.py`
19. `Qwen_Qwen3-235B-A22B_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_add_cat_mul_neg_slice_transpose_unsqueeze_3_tune.py`
20. `Qwen_Qwen3-235B-A22B_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_bmm_cat_expand_mul_transpose_unsqueeze_0_tune.py`
21. `Qwen_Qwen3-235B-A22B_bs8_amp_bf16/pointwise/triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_7_tune.py`
22. `Qwen_Qwen3-235B-A22B_bs8_amp_bf16/pointwise/triton_poi_fused_mul_silu_split_0_tune.py`
23. `Qwen_Qwen3-235B-A22B_bs8_amp_bf16/pointwise/triton_poi_fused_scalar_tensor_where_8_tune.py`
24. `Qwen_Qwen3-235B-A22B_bs8_amp_bf16/reduction/triton_red_fused_add_div_expand_mul_pow_sum_2_tune.py`
25. `Qwen_Qwen3-235B-A22B_bs8_amp_bf16/reduction/triton_red_fused_add_mean_mul_pow_rsqrt_0_tune.py`
26. `Qwen_Qwen3-235B-A22B_bs8_amp_bf16/reduction/triton_red_fused_mul_sum_0_tune.py`
27. `google_gemma-4-26B-A4B-it_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_0.c1892792_tune.py`
28. `google_gemma-4-26B-A4B-it_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_0.fa5ea906_tune.py`
29. `google_gemma-4-26B-A4B-it_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_1.2c502da9_tune.py`
30. `google_gemma-4-26B-A4B-it_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_1.347b496d_tune.py`
31. `google_gemma-4-26B-A4B-it_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_1.7239157b_tune.py`
32. `google_gemma-4-26B-A4B-it_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_1.de4810f7_tune.py`
33. `google_gemma-4-26B-A4B-it_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_10_tune.py`
34. `google_gemma-4-26B-A4B-it_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_11_tune.py`
35. `google_gemma-4-26B-A4B-it_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_3_tune.py`
36. `google_gemma-4-26B-A4B-it_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_4.1d53ac61_tune.py`
37. `google_gemma-4-26B-A4B-it_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_4.d12dec98_tune.py`
38. `google_gemma-4-26B-A4B-it_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy__unsafe_view_clone_expand_transpose_unsqueeze_7.575bd401_tune.py`
39. `google_gemma-4-26B-A4B-it_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy__unsafe_view_clone_expand_transpose_unsqueeze_7.635a2942_tune.py`
40. `google_gemma-4-26B-A4B-it_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_add_arange_bitwise_and_gt_index_le_new_ones_sub_unsqueeze_0_tune.py`
41. `google_gemma-4-26B-A4B-it_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_add_cat_mul_neg_slice_transpose_unsqueeze_3.185a9445_tune.py`
42. `google_gemma-4-26B-A4B-it_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_add_cat_mul_neg_slice_transpose_unsqueeze_3.4198281b_tune.py`
43. `google_gemma-4-26B-A4B-it_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_add_cat_mul_neg_slice_unsqueeze_6.31f33891_tune.py`
44. `google_gemma-4-26B-A4B-it_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_add_cat_mul_neg_slice_unsqueeze_6.32a026a6_tune.py`
45. `google_gemma-4-26B-A4B-it_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_bmm_cat_expand_mul_transpose_unsqueeze_0.1fbb21f0_tune.py`
46. `google_gemma-4-26B-A4B-it_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_bmm_cat_expand_mul_transpose_unsqueeze_0.a61c3ae9_tune.py`
47. `google_gemma-4-26B-A4B-it_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_clone_expand_mul_transpose_unsqueeze_8.1d700990_tune.py`
48. `google_gemma-4-26B-A4B-it_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_clone_expand_mul_transpose_unsqueeze_8.d4da357b_tune.py`
49. `google_gemma-4-26B-A4B-it_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_clone_expand_mul_transpose_unsqueeze_9.115d94f4_tune.py`
50. `google_gemma-4-26B-A4B-it_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_clone_expand_mul_transpose_unsqueeze_9.c12828f5_tune.py`
51. `google_gemma-4-26B-A4B-it_bs8_amp_bf16/pointwise/triton_poi_fused__unsafe_view_gelu_mul_2_tune.py`
52. `google_gemma-4-26B-A4B-it_bs8_amp_bf16/pointwise/triton_poi_fused_embedding_mul_0_tune.py`
53. `google_gemma-4-26B-A4B-it_bs8_amp_bf16/pointwise/triton_poi_fused_eq_0_tune.py`
54. `google_gemma-4-26B-A4B-it_bs8_amp_bf16/pointwise/triton_poi_fused_gelu_mul_split_0_tune.py`
55. `google_gemma-4-26B-A4B-it_bs8_amp_bf16/pointwise/triton_poi_fused_scalar_tensor_where_10_tune.py`
56. `google_gemma-4-26B-A4B-it_bs8_amp_bf16/pointwise/triton_poi_fused_scalar_tensor_where_9_tune.py`
57. `google_gemma-4-26B-A4B-it_bs8_amp_bf16/reduction/triton_red_fused_add_div_expand_mul_pow_sum_2.a7be7e93_tune.py`
58. `openai_gpt-oss-120b_bs8_amp_bf16/persistent_reduction/triton_per_fused__softmax_add_cat_expand_max_mul_prepare_softmax_online_sub_view_7_tune.py`
59. `openai_gpt-oss-120b_bs8_amp_bf16/persistent_reduction/triton_per_fused_mul_sum_1_tune.py`
60. `openai_gpt-oss-120b_bs8_amp_bf16/pointwise/triton_poi_fused__softmax_prepare_softmax_online_3_tune.py`
61. `openai_gpt-oss-120b_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_0.55e907ec_tune.py`
62. `openai_gpt-oss-120b_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_0.d5c1cfa7_tune.py`
63. `openai_gpt-oss-120b_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_1.849b03a0_tune.py`
64. `openai_gpt-oss-120b_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_1.a2e25905_tune.py`
65. `openai_gpt-oss-120b_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_11_tune.py`
66. `openai_gpt-oss-120b_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_12_tune.py`
67. `openai_gpt-oss-120b_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_2.35eb6e59_tune.py`
68. `openai_gpt-oss-120b_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_2.ab64ba43_tune.py`
69. `openai_gpt-oss-120b_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_3_tune.py`
70. `openai_gpt-oss-120b_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_4_tune.py`
71. `openai_gpt-oss-120b_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy__unsafe_view_add_cat_clone_expand_mul_split_sub_transpose_unsqueeze_view_6_tune.py`
72. `openai_gpt-oss-120b_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_add_arange_bitwise_and_expand_gt_index_le_new_ones_scalar_tensor_sub_unsqueeze_where_0_tune.py`
73. `openai_gpt-oss-120b_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_add_arange_bitwise_and_expand_index_le_new_ones_scalar_tensor_unsqueeze_where_0_tune.py`
74. `openai_gpt-oss-120b_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_add_cat_mul_split_sub_transpose_unsqueeze_view_5_tune.py`
75. `openai_gpt-oss-120b_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_bmm_cos_expand_mul_sin_transpose_unsqueeze_0_tune.py`
76. `openai_gpt-oss-120b_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_slice_8_tune.py`
77. `openai_gpt-oss-120b_bs8_amp_bf16/pointwise/triton_poi_fused_add_clamp_mul_sigmoid_slice_0_tune.py`
78. `openai_gpt-oss-120b_bs8_amp_bf16/pointwise/triton_poi_fused_clone_expand_transpose_unsqueeze_view_9_tune.py`
79. `openai_gpt-oss-120b_bs8_amp_bf16/pointwise/triton_poi_fused_clone_transpose_view_10_tune.py`
80. `openai_gpt-oss-120b_bs8_amp_bf16/reduction/triton_red_fused_add_div_expand_mul_pow_sum_2_tune.py`
81. `openai_gpt-oss-120b_bs8_amp_bf16/reduction/triton_red_fused_add_mean_mul_pow_rsqrt_0_tune.py`
82. `openai_gpt-oss-120b_bs8_amp_bf16/reduction/triton_red_fused_mul_sum_0_tune.py`
83. `openai_gpt-oss-20b_bs8_amp_bf16/persistent_reduction/triton_per_fused__softmax_add_cat_expand_max_mul_prepare_softmax_online_sub_view_7_tune.py`
84. `openai_gpt-oss-20b_bs8_amp_bf16/persistent_reduction/triton_per_fused_mul_sum_1_tune.py`
85. `openai_gpt-oss-20b_bs8_amp_bf16/pointwise/triton_poi_fused__softmax_prepare_softmax_online_3_tune.py`
86. `openai_gpt-oss-20b_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_0.55e907ec_tune.py`
87. `openai_gpt-oss-20b_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_0.d5c1cfa7_tune.py`
88. `openai_gpt-oss-20b_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_1.82e60436_tune.py`
89. `openai_gpt-oss-20b_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_1.a2e25905_tune.py`
90. `openai_gpt-oss-20b_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_11_tune.py`
91. `openai_gpt-oss-20b_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_12_tune.py`
92. `openai_gpt-oss-20b_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_2.35eb6e59_tune.py`
93. `openai_gpt-oss-20b_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_2.7c34a564_tune.py`
94. `openai_gpt-oss-20b_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_3_tune.py`
95. `openai_gpt-oss-20b_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_4_tune.py`
96. `openai_gpt-oss-20b_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy__unsafe_view_add_cat_clone_expand_mul_split_sub_transpose_unsqueeze_view_6_tune.py`
97. `openai_gpt-oss-20b_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_add_arange_bitwise_and_expand_gt_index_le_new_ones_scalar_tensor_sub_unsqueeze_where_0_tune.py`
98. `openai_gpt-oss-20b_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_add_arange_bitwise_and_expand_index_le_new_ones_scalar_tensor_unsqueeze_where_0_tune.py`
99. `openai_gpt-oss-20b_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_add_cat_mul_split_sub_transpose_unsqueeze_view_5_tune.py`
100. `openai_gpt-oss-20b_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_bmm_cos_expand_mul_sin_transpose_unsqueeze_0_tune.py`
101. `openai_gpt-oss-20b_bs8_amp_bf16/pointwise/triton_poi_fused__to_copy_slice_8_tune.py`
102. `openai_gpt-oss-20b_bs8_amp_bf16/pointwise/triton_poi_fused_add_clamp_mul_sigmoid_slice_0_tune.py`
103. `openai_gpt-oss-20b_bs8_amp_bf16/pointwise/triton_poi_fused_clone_expand_transpose_unsqueeze_view_9_tune.py`
104. `openai_gpt-oss-20b_bs8_amp_bf16/pointwise/triton_poi_fused_clone_transpose_view_10_tune.py`
105. `openai_gpt-oss-20b_bs8_amp_bf16/reduction/triton_red_fused_add_div_expand_mul_pow_sum_2_tune.py`
106. `openai_gpt-oss-20b_bs8_amp_bf16/reduction/triton_red_fused_add_mean_mul_pow_rsqrt_0_tune.py`
107. `openai_gpt-oss-20b_bs8_amp_bf16/reduction/triton_red_fused_mul_sum_0_tune.py`

## Additional fresh-process kernel

108. `deepseek-ai_DeepSeek-V4-Flash_bs8_amp_bf16/reduction/triton_red_fused__to_copy_mul_sigmoid_backward_squeeze_sum_1_tune.py`

This kernel was tuned from scratch twice under the current implementation:
fast/native first, then rigorous/FP64. Both runs used seed 0, separate empty
Triton and Inductor cache roots, a 30-second cooldown, Triton's 25/100 ms
timing budget, and one timing replica.
