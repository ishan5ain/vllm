# DSpark Algorithm Reference

> From "DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation"
> DeepSeek-AI & Peking University, 2026

## Core Equations (from Paper)

### Semi-Autoregressive Generation (Section 3.1)

The draft block distribution factorizes autoregressively over the sequential head:

```
P(X | x₀) = ∏ₖ₌₁ᵞ pₖ(xₖ | x₀, x_{<k})

pₖ(ν | x₀, x_{<k}) = exp(Uₖ(ν) + Bₖ(x₀, x_{<k}, ν)) / Σᵤ exp(Uₖ(u) + Bₖ(...))
```

where:
- `x₀` = anchor token from previous verification cycle
- `Uₖ` = base logits from parallel backbone at position k
- `Bₖ` = transition bias from sequential head

### Markov Head (Eq 5)

Simplest incarnation; bias depends only on immediately preceding token:

```
B(x_{k-1}, ·) = W₁[x_{k-1}] · W₂    ∈ ℝ^V

W₁ ∈ ℝ^{V × r}    # Embedding lookup (r = 256)
W₂ ∈ ℝ^{r × V}    # Logit projection
```

### RNN Head (Eq 6)

Full prefix history via GRU-like recurrent state:

```
zₖ = [s_{k-1}; W₁[x_{k-1}]; hₖ]    # Concat state, prev embedding, hidden

sₖ = σ(W_g zₖ) ⊙ s_{k-1} + (1 - σ(W_g zₖ)) ⊙ tanh(W_c zₖ)

Bₖ = W₂ᵀ · tanh(W_o zₖ)
```

(Production uses Markov head, not RNN.)

### Confidence Head (Eq 7)

```
cₖ = σ(wᵀ [hₖ; W₁[x_{k-1}]])

w ∈ ℝ^{d+r}    # Linear projection
hₖ ∈ ℝᵈ        # Backbone hidden state
W₁[x_{k-1}]    # Markov embedding of previous token
```

### Training Supervision (Eq 8)

The ground-truth confidence label is the analytical acceptance rate:

```
cₖ* = 1 - ½‖pₖᵈ - pₖᵗ‖₁
```

### Loss Function (Eq 12)

```
L = 0.1·Lce + 0.9·Ltv + 1.0·Lconf

Lce  = -Σ wₖ log pₖᵈ(xₖ*)              # Cross-entropy
Ltv  =  Σ wₖ‖pₖᵈ - pₖᵗ‖₁             # Distribution matching
Lconf = -Σ wₖ[cₖ* log cₖ + (1-cₖ*)log(1-cₖ)]  # BCE confidence

wₖ = exp(-(k-1)/γ)   # Position weighting (earlier = higher weight)
```

## Inference-Time Flow

```
1. TARGET FORWARD (once per cycle):
   target_model(prefix) → hidden_states_{40,41,42}[last_position]
   Also produces the anchor token if this is after verification.

2. CONTEXT PROJECTION:
   ctx = RMSNorm(W_c · concat(hidden_states_{40,41,42}))

3. DRAFT BACKBONE (single parallel pass, bidirectional):
   inputs = [anchor_emb, mask_emb, mask_emb, mask_emb, mask_emb]
   positions = [anchor_pos, anchor_pos+1, ..., anchor_pos+4]
   hidden = backbone(ctx, inputs, positions, is_causal=False)

4. BASE LOGITS:
   base_logits = hc_head(hidden) → shared_head → [B, 5, V]

5. MARKOV SAMPLING (sequential, left-to-right):
   prev = anchor_token
   for k in range(5):
       bias = W₂(W₁[prev])                    # [B, V]
       logits[k] = base_logits[k] + bias       # [B, V]
       c[k] = σ(w · [hidden[k]; W₁[prev]])     # [B, 1]
       token[k] = sample(logits[k])
       prev = token[k]

6. VERIFICATION (standard speculative decoding):
   target_model(prefix + draft_tokens) → accept/reject via rejection sampling
```

## Verification Scheduling (Section 3.2.2, Algorithm 1)

```
Given: R active requests, each with confidence c_{r,1}...c_{r,γ}
       Profiled SPS(B) curve (steps/sec at batch size B)

1. Compute cumulative survival probabilities:
   a_{r,j} = ∏ᵢ₌₁ʲ c_{r,i}

2. Sort all candidates (r,j) globally by a_{r,j} descending

3. Greedy admission with early stopping:
   - Start with ℓ_r = 0 for all r, B = R
   - For each (r,j) in sorted order:
       ℓ_r = j, B += 1, τ* += a_{r,j}
       Θ = τ* · SPS(B)
       if Θ > Θ_best: keep
       else: break  # early stop
   - Return ℓ* = best lengths

This is LOSSESS because early-stopping ensures decisions depend only on
prefix information, not future tokens (see Appendix A for proof).
```

## Production Adaptations (Section 5.2)

1. **Asynchronous scheduling**: Uses confidence from 2 steps prior to
   determine truncation length K, avoiding ZOS pipeline stalls.
2. **Unconstrained global search**: Remove early-stopping break (necessitated
   by jagged SPS curves), compensated by the 2-step-old prediction barrier.
3. **Variable-length routing**: Flat tensor + marker-based sparse attention
   to handle per-request variable verification lengths.

## STS Calibration (Section 3.2.1)

```
Per k = 1..γ (sequential, left to right):
  Find temperature Tₖ that minimizes ECE of:
    ĉ_{1..k} = σ(c₁/T₁) · σ(c₂/T₂) · ... · σ(cₖ/Tₖ)
  vs. empirical acceptance rates at position k
  Keep T₁...T_{k-1} fixed from previous iterations

Result: Calibrated scores maintain ranking (order-preserving) but
        match empirical survival probabilities.
```

## Production Configuration (Section 5.1)

- γ = 5 (maximum draft length)
- 3 MoE layers in draft backbone
- Markov head (vanilla, r=256)
- Sliding window attention: 128
- mHC (manifold-constrained hyper-connections)
- Target layers for context: [40, 41, 42] (last 3 V4 layers)
- Confidence head trained end-to-end, STS-calibrated post-hoc

## Key Distinction from Standard MTP

Standard MTP generates tokens one at a time:
```
Step 0: target → mtp.0 → token₁
Step 1: target → mtp.0 → token₂  (or mtp.1 if num_layers > 1)
...
```

DSpark generates all γ tokens in one pass:
```
Step 0: target → backbone(mtp.0, mtp.1, mtp.2) → all γ logits
        → markov sequential loop → all γ tokens
```
