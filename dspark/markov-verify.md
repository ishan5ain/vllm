# Markov Head (W₁W₂) Verification Report

> **File**: `vllm/models/deepseek_v4/nvidia/dspark.py` — `forward_dspark_block()`  
> **Reference**: `dspark/ALGORITHM_REFERENCE.md` (DSpark paper, Sections 3, 5)  
> **Skeleton**: `dspark/dspark_model_skeleton.py` (design reference)

---

## 1. Location of Markov Head Code

The Markov head parameters are defined in `DSparkInnerModel.__init__()` (lines 277–289):

```python
self.markov_w1 = VocabParallelEmbedding(config.vocab_size, self.markov_rank, ...)
self.markov_w2 = ColumnParallelLinear(self.markov_rank, config.vocab_size, bias=False, ...)
```

These are used in two methods:

| Method | Lines | Purpose |
|--------|-------|---------|
| `markov_bias()` | 389–399 | Standalone helper: W₂(W₁[prev]) |
| `forward_dspark_block()` | 504–529 | Full block generation with sequential Markov sampling loop |

---

## 2. Trace of Markov W₁W₂ in `forward_dspark_block` (Lines 504–529)

```python
prev = anchor_token_ids.long()                          # [B]

for k in range(gamma):                                   # k = 0, 1, 2, 3, 4
    prev_emb = self.markov_w1(prev)          # [B, rank=256]     ← W₁ lookup
    bias     = self.markov_w2(prev_emb)      # [B, V]             ← W₁[prev] · W₂
    step_logits = base_logits[:, k, :] + bias# [B, V]             ← Uₖ + Bₖ
    # … sample next_token from step_logits …
    prev = next_token                                      # feed back for next step
```

### Step-by-step trace (γ=5):

| k | `prev`             | `prev_emb` [B,256] | `bias` [B,V] | `base_logits` slice | `step_logits` |
|---|---------------------|---------------------|--------------|---------------------|---------------|
| 0 | anchor_token_ids    | W₁[anchor]          | W₂(W₁[anchor]) | `[:, 0, :]`       | U₀ + B(anchor→·) |
| 1 | draft_token₀        | W₁[draft_token₀]    | W₂(W₁[draft_token₀]) | `[:, 1, :]` | U₁ + B(token₀→·) |
| 2 | draft_token₁        | W₁[draft_token₁]    | W₂(W₁[draft_token₁]) | `[:, 2, :]` | U₂ + B(token₁→·) |
| 3 | draft_token₂        | W₁[draft_token₂]    | W₂(W₁[draft_token₂]) | `[:, 3, :]` | U₃ + B(token₂→·) |
| 4 | draft_token₃        | W₁[draft_token₃]    | W₂(W₁[draft_token₃]) | `[:, 4, :]` | U₄ + B(token₃→·) |

---

## 3. Comparison with Paper Equations

### Eq 5 — Markov Head

```
B(x_{k-1}, ·) = W₁[x_{k-1}] · W₂    ∈ ℝ^V
```

| Component | Paper | Code | Match? |
|-----------|-------|------|--------|
| Previous token embedding | `W₁[x_{k-1}]` | `self.markov_w1(prev)` → [B, r] | ✅ |
| Logit bias | `W₁[x_{k-1}] · W₂` | `self.markov_w2(prev_emb)` → [B, V] | ✅ |
| Combined logits | `Uₖ + Bₖ` | `base_logits[:, k, :] + bias` | ✅ |

**Paper Eq 5 is a direct matrix formulation.** The code's `markov_w1` is `VocabParallelEmbedding(V, r)` — an embedding lookup of shape (V, r). The `markov_w2` is `ColumnParallelLinear(r, V)` — a linear transform computing `x @ W.T` with weight shape `(V_tp, r)`. The combined operation computes `(W₁[prev]) @ (W₂_weight.T)` which is algebraically identical to `W₁[prev] · W₂` ∈ ℝ^V.

### Eq 6 — RNN Head

The RNN head (`zₖ`, GRU-like `sₖ`, `W_o zₖ`) is **not implemented**. Production uses the Markov head only, consistent with Section 5.1 of the paper. ✅

### Semi-Autoregressive Factorization (Section 3.1)

```
P(x | x₀) = ∏ₖ₌₁ᵞ pₖ(xₖ | x₀, x_{<k})
pₖ(ν | x₀, x_{<k}) = exp(Uₖ(ν) + Bₖ(…)) / Σ exp(Uₖ(·) + Bₖ(…))
```

The loop structure implements this factorization correctly:
- Each step `k` samples `xₖ` given `x₀` (via backbone `Uₖ`) and `x_{<k}` (via Markov bias from `x_{k-1}`).
- The softmax is implicit in argmax (temperature=0) or explicit in multinomial (temperature>0).

### Eq 7 — Confidence Head

```
cₖ = σ(wᵀ [hₖ; W₁[x_{k-1}]])
```

| Component | Code | Match? |
|-----------|------|--------|
| Feature concatenation | `torch.cat([h_k, prev_emb], dim=-1)` → [B, D+r] | ✅ |
| Linear projection | `self.confidence_proj(features)` → [B, 1] | ✅ |
| Sigmoid | **NOT applied** — returns raw logits | ⚠️ Design choice (see §6) |

---

## 4. Semi-Autoregressive Sampling Order

The sampling loop runs `for k in range(gamma)` with `gamma = self.block_size = 5`:

```
k=0: anchor → token₀
k=1: token₀ → token₁
k=2: token₁ → token₂
k=3: token₂ → token₃
k=4: token₃ → token₄
```

This is **left-to-right sequential**, consistent with the paper's semi-autoregressive generation (Section 3.1) and the skeleton's `DSparkMarkovHead.sample_block()`.

Final assembly:
```python
draft_tokens = torch.stack(draft_tokens_list, dim=1)   # [B, γ] — sampled token IDs
draft_logits = torch.stack(draft_logits_list, dim=1)   # [B, γ, V] — corrected logits
confidence  = torch.stack(confidence_list,  dim=1)      # [B, γ] — raw confidence logits
```

---

## 5. Shape Verification

Given: `B = batch_size`, `γ = 5`, `D = hidden_size (4096)`, `r = markov_rank (256)`, `V = vocab_size (129280)`

| Step | Variable | Shape | Verified |
|------|----------|-------|----------|
| Input | `anchor_token_ids` | `[B]` | ✅ |
| Markov lookup | `markov_w1(prev)` | `[B, r]` | ✅ |
| Markov projection | `markov_w2(prev_emb)` | `[B, V]` | ✅ |
| Base logit slice | `base_logits[:, k, :]` | `[B, V]` | ✅ |
| Combined logits | `step_logits` | `[B, V]` | ✅ |
| Sampled token | `next_token` | `[B]` | ✅ |
| Hidden state slice | `hidden_3d[:, k, :]` | `[B, D]` | ✅ |
| Confidence features | `cat(h_k, prev_emb)` | `[B, D+r]` | ✅ |
| Confidence logit | `c_k` (squeezed) | `[B]` | ✅ |
| Feed-back | `prev = next_token` | `[B]` | ✅ |
| Final draft tokens | `stack(draft_tokens_list, dim=1)` | `[B, γ]` | ✅ |
| Final draft logits | `stack(draft_logits_list, dim=1)` | `[B, γ, V]` | ✅ |
| Final confidence | `stack(confidence_list, dim=1)` | `[B, γ]` | ✅ |

**All shapes are correct.** No dimension mismatches in the Markov sampling loop.

---

## 6. Discrepancies and Observations

### 6.1 Missing Sigmoid in Confidence Output (Minor — by Design)

**Observation**: `confidence_score()` and `forward_dspark_block()` return raw logits, not sigmoid-probabilities. The paper's Eq 7 includes `σ(·)`.

**Resolution**: This is **intentional**, confirmed by:
- `dspark_model_skeleton.py` line 98: `"Output: confidence_logits [B, γ] (pre-sigmoid)"`
- `dspark/sts_calibration.py` line 83: `"confidence_logits: raw logits from the confidence head before sigmoid"`
- The STS calibration pipeline operates on logits directly (divides by temperature before sigmoid).
- The confidence head returns logits; downstream code applies sigmoid + temperature scaling.

**Status**: Not a bug. Architectural separation of concerns.

### 6.2 Draft Input Construction

**Observation**: The skeleton creates draft input as `[anchor, noise, noise, noise, noise]` using `self.noise_token_id`. The implementation's `forward_dspark_block` accepts arbitrary `draft_input_ids` without enforcing mask tokens.

**Analysis**: The speculator (`vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:389`) passes the input buffer directly. The contents depend on how the buffer is initialized. There's no explicit mask-token construction in the current code paths.

**Status**: Design choice — caller is responsible for providing correct draft inputs. The `noise_token_id` config field exists but is unused in the current implementation. Worth wiring in Phase 2.

### 6.3 Backbone Layer Input Dimensionality (Potential Issue)

**Observation**: `forward_dspark_block` passes 2D `[T, D]` hidden states through each backbone `DeepseekV4DecoderLayer`. In the target model, the same decoder layers receive 3D `[T, hc_mult, D]` (after `unsqueeze(-2).repeat(1, hc_mult, 1)` at line ~1075 of `model.py`).

If `config.hc_mult > 1`, the `mhc_pre_tilelang` kernel inside `DeepseekV4DecoderLayer.forward` infers `hc_mult = residual.shape[-2]`. With 2D input, this would be `T` (number of tokens), not `config.hc_mult`. The assertion `fn.shape[1] == hc_mult * hidden_size` would fail for `T != config.hc_mult`.

**Mitigation**: The standard DSpark per-step forward (via `DeepSeekV4DSparkLayer.forward`) also passes 2D input to `mtp_block` for non-output layers (line ~175: `hidden_states = self.h_proj(...) + self.e_proj(...)`). If the per-step path works, then `DeepseekV4DecoderLayer` must handle 2D input correctly (likely because `hc_mult = 1` for the MTP layers in the checkpoint, or there's internal reshaping logic).

**Status**: Potential concern for the output layer (mtp.2) which in the standard flow expects 3D input. The `forward_dspark_block` method runs it in 2D mode and separately applies hc_head via `repeat(1, hc_mult, 1)` afterward. Functional equivalence depends on whether the output layer's `mtp_block` behaves identically with 2D vs 3D input.

**Recommendation**: Verify `hc_mult` value from checkpoint used with DSpark. If `hc_mult > 1`, add input reshaping to 3D before the output layer's `mtp_block`, or use the standard `layer.forward()` path.

### 6.4 Equivalence with Skeleton Model

| Skeleton (`dspark_model_skeleton.py`) | Implementation (`dspark.py`) | Match? |
|---------------------------------------|------------------------------|--------|
| `DSparkMarkovHead(w1=Embedding, w2=Linear)` | `markov_w1=VocabParallelEmbedding`, `markov_w2=ColumnParallelLinear` | ✅ TP-aware equivalents |
| `sample_block()` loop: `bias = w2(w1(prev))` | Same: `bias = markov_w2(markov_w1(prev))` | ✅ |
| Temperature sampling: argmax at T=0, multinomial at T>0 | Identical logic | ✅ |
| Confidence: batched after sampling with `prev_tokens` | Per-step inside loop with `prev` | ✅ Algebraically equivalent |
| Draft input: `[anchor, noise×4]` | Caller-provided | ⚠️ Different (see 6.2) |

---

## 7. Summary

The Markov head (W₁W₂) sampling implementation in `forward_dspark_block()` **correctly implements** the DSpark paper's semi-autoregressive generation (Section 3, Eq 5). The core logic:

1. W₁ embedding lookup → `markov_w1(prev)` — matches Eq 5 ✓
2. W₂ logit projection → `markov_w2(prev_emb)` — matches Eq 5 ✓
3. Add to base logits → `base_logits[:, k, :] + bias` — matches Section 3.1 ✓
4. Left-to-right sequential order (0→1→...→γ-1) — correct ✓
5. All shapes verified — correct ✓

**No fundamental discrepancies were found in the Markov W₁W₂ logic itself.** The identified observations (unsigmoided confidence, draft input construction, backbone dimensionality) are architectural concerns outside the scope of the Markov head verification.

---

## Acceptance Report

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Verified the Markov head (W₁W₂) sampling implementation in forward_dspark_block() against the DSpark paper (Eq 5), the ALGORITHM_REFERENCE.md, and the dspark_model_skeleton.py. All four sub-tasks completed: (1) traced markov_w1/markov_w2 usage through the sequential loop, (2) compared against paper equations 5-6, (3) confirmed left-to-right semi-autoregressive sampling order, (4) verified shapes at each step. No fundamental discrepancies found in the Markov head logic."
    }
  ],
  "changedFiles": [],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "Read and analyzed vllm/models/deepseek_v4/nvidia/dspark.py (full file, 866 lines)",
      "result": "passed",
      "summary": "Traced forward_dspark_block method (lines 419-538), markov_bias (389-399), confidence_score (401-415)"
    },
    {
      "command": "Read and analyzed dspark/ALGORITHM_REFERENCE.md",
      "result": "passed",
      "summary": "Extracted paper equations 5-7, Section 3.1 semi-autoregressive factorization, inference flow"
    },
    {
      "command": "Read and analyzed dspark/dspark_model_skeleton.py (full file, 374 lines)",
      "result": "passed",
      "summary": "Compared DSparkMarkovHead.sample_block() and DSparkConfidenceHead with implementation"
    },
    {
      "command": "Read dspark/sts_calibration.py and vllm/v1/worker/gpu/spec_decode/dspark/speculator.py",
      "result": "passed",
      "summary": "Confirmed confidence logits are intentionally pre-sigmoid; speculator passes input buffer directly"
    },
    {
      "command": "Read vllm/model_executor/kernels/mhc/tilelang.py (mhc_pre_tilelang shape contract) and vllm/models/deepseek_v4/nvidia/model.py (target model 3D hidden state flow)",
      "result": "passed",
      "summary": "Identified potential backbone 2D vs 3D dimensionality concern (not a Markov head issue)"
    }
  ],
  "validationOutput": [
    "Markov W₁ lookup: markov_w1(prev) → [B, 256] — matches W₁[x_{k-1}]",
    "Markov W₂ projection: markov_w2(prev_emb) → [B, V] — matches W₁[x_{k-1}] · W₂",
    "Logit combination: base_logits[:, k, :] + bias → [B, V] — matches Uₖ + Bₖ",
    "Sampling order: k=0→1→2→3→4, left-to-right sequential — matches paper Section 3.1",
    "All shapes verified: no dimension mismatches in the Markov loop",
    "Confidence output is pre-sigmoid (logits) — by design, consistent with skeleton and STS calibration",
    "Skeleton model DSparkMarkovHead.sample_block() has identical logic to forward_dspark_block loop"
  ],
  "residualRisks": [
    "Backbone layer 2D vs 3D input dimensionality: if hc_mult > 1 in checkpoint, the output layer's mtp_block in forward_dspark_block may produce incorrect hidden states because it receives 2D [T, D] instead of 3D [T, hc_mult, D] as in the target model. This would affect base_logits quality but not the Markov W₁W₂ logic itself.",
    "Draft input construction: caller must provide noise/mask tokens at positions 1-γ; the noise_token_id config is unused. If arbitrary token IDs are passed, the backbone may produce incorrect base logits.",
    "No runtime test coverage for forward_dspark_block — this is Phase 2 code with no observed test invocation."
  ],
  "noStagedFiles": true,
  "diffSummary": "No code changes made. Analysis-only verification task.",
  "reviewFindings": [
    "no blockers: Markov W₁W₂ implementation correctly implements paper Eq 5 with correct shapes and sampling order",
    "info: dspark.py:525 — confidence_score returns raw logits (pre-sigmoid), consistent with skeleton design and STS calibration pipeline",
    "info: dspark.py:504 — draft input IDs accepted as parameter without mask/noise enforcement; caller responsibility",
    "potential: dspark.py:477-501 — backbone layers receive 2D [T,D] hidden states; if config.hc_mult > 1, output layer may need 3D input for correct behavior"
  ],
  "manualNotes": "The Markov W₁W₂ logic is sound. The primary area needing attention before Phase 2 completion is the backbone layer forwarding (2D vs 3D, bidirectional attention wiring, target context injection) — these are outside the scope of this Markov head verification but were noted during analysis."
}
```
