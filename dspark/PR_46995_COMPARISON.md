# DSpark — Comparison: Our Approach vs. Upstream PR #46995

> **Date:** 2026-06-29
> **PR:** [vLLM #46995 "[Spec Decode] DSpark"](https://github.com/vllm-project/vllm/pull/46995)
>   by Benjamin Chislett (author of the DFlash speculator). 19 files, +2075/-44,
>   single commit `8b82d11`, base `main`, MERGEABLE.
> **Verdict:** Our weight anatomy was right; our *integration architecture* was
>   wrong on two axes that generated nearly all our errors. **Pivot to the PR**
>   (see `dspark/MIGRATION_PLAN.md`).

## TL;DR

The PR and our branch reach the **same weight anatomy** (our archaeology was
correct), but diverge on **two architectural decisions that explain every error
we've chased**:

1. **Context plumbing.** We invented a custom `_dspark_context_buffer` + getter +
   manual `model_runner` injection. The PR reuses vLLM's **existing EAGLE3
   `aux_hidden_states` mechanism** (same path DFlash/Eagle3 use). Our custom
   buffer is exactly the thing that is `None` → `propose()` early-returns → 0%.
2. **Non-causal attention.** We were hand-rolling cross-attention / bidirectional
   masking ("fix C", still pending). The PR **reuses Sparse-MLA kernels** by
   expanding each query's top-k index list to include the trailing window **plus
   all block tokens (future included)** — index-driven, no causal mask, exactly
   correct, zero new attention kernels.

Meta-insight: the PR is a **thin subclass of DFlash** (`DSparkSpeculator(
DFlashSpeculator)`, `Qwen3DSparkModel(DFlashQwen3Model)`). DFlash already solved
parallel drafting, context-KV precompute, the prepare-inputs kernel, CUDA-graph
capture, and aux-hidden plumbing. We built from `SpecDecodeBaseProposer` and
re-derived all of it — getting subtle pieces wrong. That is the structural root
of "error after error."

## 1. What we MISSED

### 1a. 🔴 Context = EAGLE3 aux-hidden-state path, not a custom buffer (our 0% blocker)
- Target `DeepseekV4Model` inherits `EagleModelMixin` + `SupportsEagle3`; in
  `forward()` captures `aux_recon = mhc_post_tilelang(...)` at `idx+1 in
  aux_hidden_state_layers`, appends `aux_recon.mean(dim=1)`, returns
  `(hidden_states, aux_hidden_states)`.
- `eagle3_utils.get_eagle3_aux_layers_from_config` reads `dspark_target_layer_ids`
  (`i+1`).
- `model_runner.py`: one-line add of `"dspark"` to `use_aux_hidden_state_outputs`.
- Speculator `propose()` receives `aux_hidden_states` **as an argument**, asserts
  non-None, `combine_hidden_states(torch.cat(aux_hidden_states, -1))`.
→ Our bespoke buffer/getter/injection is what silently fails. Fix = delete it,
implement `SupportsEagle3`.

### 1b. 🔴 Non-causal attention via Sparse-MLA index expansion (the key idea)
`_compute_dspark_noncausal_swa_indices_kernel` fills each query's top-k list with
`[max(prefix-window,0) .. seq_len)` — trailing window + the full block incl.
future positions. Sparse-MLA attends exactly over the index list with no causal
mask ⇒ exact non-causal attention reusing existing kernels. Width padded to a
multiple of 128. Proven by a 529-line test vs. dense SDPA (and vs. a causal
reference, to prove future tokens are really attended). **We never found this.**

### 1c. 🔴 Context enters via KV-cache precompute, not embedding addition
`precompute_and_store_context_kv` derives each draft layer's context KV from
projected `main_x` (`wkv`+`kv_norm`+RoPE+quant) and inserts it into that layer's
SWA cache; the block then cross-attends. Our interim "add `main_x` to anchor
embedding" is not how DSpark works — a leading 0% suspect even after context
arrives.

### 1d. Anchor-as-first-prediction: N query tokens, not 1+N
`num_query_per_req = num_speculative_steps`; anchor at offset 0 is the bonus
token; sample at every position; `sample_pos = query_pos + 1`. Scheduler:
`use_dspark() → num_lookahead_tokens = num_spec_tokens`. If we inherited the 1+N
convention, anchors/positions misalign — consistent with our 0/795 *exact*
mismatch.

### 1e. Sequential Markov sampling + FULL CUDA graph
`_sample_sequential`: left-to-right `markov_embed(prev) → bias → base+bias →
gumbel/argmax → prev`. Whole step (backbone + sequential sampling) in one FULL
graph via reused `DFlashCudaGraphManager`. We run eager, no DSpark graphs.

### 1f. Wrong base class
We: `DSparkProposer(SpecDecodeBaseProposer)` + `DSparkSpeculator(
DraftModelSpeculator)`. PR: `DSparkSpeculator(DFlashSpeculator)` — inherits the
whole machinery. **DFlash is the correct foundation.**

## 2. What we GOT RIGHT (PR confirms)

| Our finding | PR confirms |
|---|---|
| Fix A: context proj = `mtp.0.main_proj` + `main_norm`; drop fabricated `fc` | ✓ `combine_hidden_states = main_norm(main_proj(concat))` |
| Fix B: no `enorm/hnorm/e_proj/h_proj` | ✓ embeddings feed blocks directly, expanded to `hc_mult` |
| Fix D: capture = `mhc_post` + mean over hc_mult | ✓ `aux_recon.mean(dim=1)` |
| Fix E/F: shared embed/head; `mtp.last.norm` pre-head | ✓ `dspark_shares_target_embeddings`, alias embed/lm_head |
| hc_head needs genuine 4-stream 3D `[T,hc,H]` | ✓ `unsqueeze(-2).repeat(1,hc_mult,1)` → `hc_head_fused_kernel_tilelang` |
| markov head low-rank w1(V×r)/w2(r×V); confidence head no-bias (DSV4) | ✓ identical `DSparkMarkovHead`/`DSparkConfidenceHead(bias=False)` |
| `markov_w1` collides with `"w1"` stacked substring → load head params directly | ✓ PR comments this exact gotcha |
| expert loader, `shared_experts.w2→down_proj`, `gate.bias→e_score_correction_bias` | ✓ all present |
| `main_proj` fp8/quant_config; `attn_sink` head-sliced buffer | ✓ |

Our completeness-assertion guardrail was the right instinct and caught real bugs.

## 3. Insights for GB10 (sm_121)

1. **Kernel risk.** PR's non-causal trick needs a Sparse-MLA backend: FlashMLA
   (SM90/SM100) or FlashInfer TRTLLM MLA Sparse (SM100/SM120). **GB10 = sm_121.**
   Validate which backend runs there *before* trusting the attention path; run
   `test_dspark_noncausal_sparse_mla.py` on the GB10 box. (Highest priority.)
2. **PR not fully done either.** Probabilistic drafting degrades into junk/loops
   (suspected rejection-sampling bug); Qwen3 DSpark has a CUDA-graph IMA. **Greedy
   is the validated path** — matches our Tier-1 plan.
3. **Flash vs Pro.** PR targets DSV4-**Pro**-DSpark + Qwen3-8B-DSpark; we run
   **Flash**. Confirm Flash config field names: `n_mtp_layers` (PR reads this for
   backbone depth; we hardcoded 3), `dspark_target_layer_ids`,
   `dspark_markov_rank`, `dspark_noise_token_id`, `hc_mult`, `hc_eps`.
4. `use_v2_model_runner` is forced True for dspark; `num_lookahead_tokens = N`.
   We're already on the V2 runner; the lookahead change is a correctness fix we lack.
5. **Perf bar:** BS1/7-draft on 8×B300 — verify ~11–13ms, backbone 0.6ms,
   sampling 0.6ms, E2E ~14ms, AL~5, >350 TPS. Our 0%/6.9 t/s is the broken-context
   symptom, not a ceiling.

## Recommendation
Adopt the PR (merge `benchislett:dspark` @ `8b82d11`), re-apply our GB10
cooperative_topk fallback, validate sparse-MLA on sm_121. Full steps:
`dspark/MIGRATION_PLAN.md`.
