# DSpark vLLM Integration — Implementation Plan

> **Target model:** `deepseek-ai/DeepSeek-V4-Flash-DSpark`
> **Paper:** [DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation](https://arxiv.org/abs/2606.XXXXX)
> **Reference implementation:** [github.com/deepseek-ai/DeepSpec](https://github.com/deepseek-ai/DeepSpec)

## Overview

DSpark is DeepSeek's production speculative decoding framework. It extends the DFlash (parallel) draft model with:
1. **A semi-autoregressive head** (Markov head) that introduces inter-token dependencies within a parallel draft block
2. **A confidence head** that predicts per-position acceptance probabilities
3. **A hardware-aware prefix scheduler** that dynamically trims verification length

The `DeepSeek-V4-Flash-DSpark` checkpoint is the same V4-Flash base model with an additional speculative decoding module (3 MTP-like layers + Markov head + confidence head weights).

## Architecture Summary

```
                                 ┌─────────────────────┐
                                 │   Target Model (43L) │
                                 │   frozen             │
                                 └────────┬────────────┘
                                          │ hidden states from layers 40,41,42
                                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        DSpark Draft Model                           │
│                                                                     │
│  Input: [anchor_token] [mask] [mask] [mask] [mask]  (γ=5 tokens)   │
│                                                                     │
│  ┌──────────────────────────────────────────────┐                   │
│  │       Parallel Backbone (2 MoE layers)        │                   │
│  │  mtp.0: attn + ffn (bidirectional over block) │                   │
│  │  mtp.1: attn + ffn (bidirectional over block) │                   │
│  └──────────────────────┬───────────────────────┘                   │
│                         │ hidden states h₁...h₅                     │
│                         ▼                                           │
│  ┌──────────────────────────────────────────────┐                   │
│  │       Output Layer (mtp.2)                    │                   │
│  │  attn + ffn → hc_head → base logits U₁...U₅ │                   │
│  └──────────────────────┬───────────────────────┘                   │
│                         │                                           │
│  ┌──────────────────────┴───────────────────────┐                   │
│  │         Markov Head (sequential)              │                   │
│  │  for k = 1..5:                                │                   │
│  │    bias = W₂(W₁[prev])     # 256-rank        │                   │
│  │    logits[k] = U[k] + bias                    │                   │
│  │    c[k] = σ(w·[h[k]; W₁[prev]])              │                   │
│  │    prev = sample(logits[k])                   │                   │
│  └──────────────────────────────────────────────┘                   │
│                                                                     │
│  Output: draft_tokens[5], confidence[5], draft_probs[5, V]          │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Differences from Existing vLLM MTP

| Aspect | Current MTP | DSpark |
|---|---|---|
| Draft attention | Causal (token-by-token) | **Bidirectional** within block |
| Draft input | Single token embedding | **Anchor + γ-1 mask tokens** |
| Block generation | Per-step: 1 layer → 1 token | **Per-block**: all layers → all γ tokens |
| Sequential bias | None | **Markov W₁W₂** applied per position |
| Confidence scores | None | **Linear→Sigmoid** per position |
| Verification | Fixed-length γ | Variable (confidence-gated, optional) |
| MTP layers loaded | 1 (`num_nextn_predict_layers=1`) | **3** (mtp.0, mtp.1, mtp.2) |

## Implementation Steps (Priority Order)

### Phase 1: Model Loading (Critical)

**Goal:** Load the DSpark checkpoint weights correctly.

**Current state:** The existing `DeepSeekV4MTP.load_weights()` only loads 1 MTP layer
(because `num_nextn_predict_layers=1` in config). The DSpark checkpoint has 3 layers
(mtp.0, mtp.1, mtp.2), plus markov/confidence head weights on mtp.2.

**Required changes:**

1. **Add a DSpark-specific model class** `vllm/models/deepseek_v4/nvidia/dspark.py`:

```python
class DeepSeekV4DSparkLayer(nn.Module):
    """Single DSpark backbone layer (parallel, bidirectional)."""
    def __init__(self, ...):
        self.enorm = RMSNorm(...)
        self.hnorm = RMSNorm(...)
        self.e_proj = ReplicatedLinear(hidden_size, hidden_size)
        self.h_proj = ReplicatedLinear(hidden_size, hidden_size)
        self.mtp_block = DeepseekV4DecoderLayer(...)  # reuses existing

    def forward(self, target_hidden_states, inputs_embeds, positions, ...):
        # Project target context & draft embeddings
        ctx = self.h_proj(target_hidden_states)
        emb = self.e_proj(inputs_embeds)
        # Run transformer layer (bidirectional within block)
        hidden = self.mtp_block(positions, ctx + emb)
        return hidden


class DeepSeekV4DSparkModel(nn.Module):
    """Full DSpark draft model with Markov + confidence heads."""

    def __init__(self, ...):
        # Load all 3 MTP layers from checkpoint
        self.backbone_layers = nn.ModuleList([
            DeepSeekV4DSparkLayer(...) for _ in range(3)
        ])
        # Markov head (Eq 5 from paper)
        self.markov_w1 = nn.Embedding(vocab_size, 256)
        self.markov_w2 = nn.Linear(256, vocab_size, bias=False)
        # Confidence head (Eq 7 from paper)
        self.confidence_head = nn.Linear(hidden_size + 256, 1)
        # HC head for logits (V4-specific)
        self.hc_head_fn = nn.Parameter(...)
        self.hc_head_base = nn.Parameter(...)
        self.hc_head_scale = nn.Parameter(...)
        self.shared_head = SharedHead(...)
        self.embed_tokens = VocabParallelEmbedding(...)
        self.logits_processor = LogitsProcessor(vocab_size)

    def forward(self, input_ids, positions, target_hidden_states, ...):
        """One decoding cycle: produce all γ draft tokens."""
        # 1. Embed anchor + mask tokens
        embeds = self.embed_tokens(input_ids)
        # 2. Project target context from layers 40,41,42
        ctx = self._project_context(target_hidden_states)
        # 3. Run parallel backbone (bidirectional)
        hidden = embeds
        for layer in self.backbone_layers:
            hidden = layer(ctx, hidden, positions)
        # 4. Compute base logits via hc_head
        base_logits = self._compute_base_logits(hidden)
        # 5. Run Markov head sequentially
        draft_tokens, draft_logits, confidence = self._sample_block(
            base_logits, hidden, anchor_token
        )
        return draft_tokens, draft_logits, confidence
```

2. **Weight loading** — New `load_weights()` that handles:
   - `mtp.2.markov_head.markov_w1.weight` → `model.markov_w1.weight`
   - `mtp.2.markov_head.markov_w2.weight` → `model.markov_w2.weight`
   - `mtp.2.confidence_head.proj.weight` → `model.confidence_head.weight`
   - Standard MTP layer weights for all 3 layers (mtp.0, mtp.1, mtp.2)

### Phase 2: Draft Generation (Core)

**Goal:** Implement the semi-autoregressive drafting loop.

**Critical details:**
- Bidirectional attention: `is_causal=False` within the draft block
- Input: `[anchor_token_id, mask_token_id, ..., mask_token_id]` (γ=5 tokens total)
- The anchor position feeds into position 0, allowing "γ input tokens yield γ draft logits" (Section 3.1)
- Markov head runs left-to-right within the block after backbone forward

### Phase 3: Integration with Speculative Decode Runner

**Goal:** Hook DSpark into vLLM's speculative decoding pipeline.

Two approaches:

**A. Use existing MTP-style step-based API (simpler, less efficient):**
- Run one DSpark cycle per verification
- Returns γ draft tokens at once
- Requires the speculative runner to handle multi-token proposals

**B. Custom proposer (optimal, more work):**
- Create a `DSparkProposer` that wraps the `DeepSeekV4DSparkModel`
- Override `propose()` to run the full block draft
- Integrate with vLLM's `MultiStepWorker` or create a new worker

### Phase 4: Confidence-Based Pruning (Optional v1)

**Goal:** Use confidence scores to dynamically trim verification length.

```python
def confident_prefix_length(confidence_logits, threshold=0.5):
    """From DeepSpec: deepspec/eval/dspark/draft_ops.py"""
    probs = confidence_logits.sigmoid()
    below = probs < threshold
    if not below.any():
        return len(probs)
    return below.nonzero()[0].item()  # first position below threshold
```

### Phase 5: STS Calibration (Optional)

**Goal:** Calibrate confidence scores for accurate throughput estimation.

The paper's STS (Sequential Temperature Scaling) temperatures are **not included** in the checkpoint.

Options:
1. Skip calibration — raw confidence has 3-8% ECE (disk vs. reality)
2. Recalibrate on a small held-out validation set
3. Use a fixed temperature of 1.0 (no scaling)

## Checkpoint Anatomy

See [checkpoint_anatomy.md](checkpoint_anatomy.md) for the full weight structure.

## Config Fields (from `config.json`)

```
dspark_block_size: 5              # γ = maximum draft length
dspark_noise_token_id: 128799     # Mask token ID for draft input
dspark_target_layer_ids: [40, 41, 42]  # Which target layers provide context
dspark_markov_rank: 256           # Low-rank dimension for W₁, W₂
num_nextn_predict_layers: 1       # Must be overridden to 3 for DSpark
hidden_size: 4096                 # V4-Flash hidden dimension
vocab_size: 129280                # Vocabulary size
num_hidden_layers: 43             # Target model depth (layers 0-42)
n_routed_experts: 256             # MoE experts per layer
num_experts_per_tok: 6            # Top-K routing
```

## DSpark Weight Shapes (from checkpoint)

| Weight | Shape | Notes |
|---|---|---|
| `markov_w1.weight` | `[129280, 256]` | W₁ embedding (paper Eq 5) |
| `markov_w2.weight` | `[256, 129280]` | W₂ projection (paper Eq 5) |
| `confidence_head.proj.weight` | `[1, 4352]` | 4096 (hidden) + 256 (markov) → 1 |
| `mtp.{0,1,2}.attn.*` | Standard V4 MLA | Q-lora, KV-lora, attn_sink |
| `mtp.{0,1,2}.ffn.*` | MoE (256 experts) | FP4 expert weights |
| `mtp.{0,1,2}.hc_attn_*` | mHC attention residual | gc_attn_fn, gc_attn_scale, gc_attn_base |
| `mtp.{0,1,2}.hc_ffn_*` | mHC FFN residual | gc_ffn_fn, gc_ffn_scale, gc_ffn_base |
| `mtp.2.hc_head_*` | Hypercompressed LM head | 4× compression |
| `mtp.2.shared_head.*` | Final vocab projection | head.weight, norm.weight |

## Reference Files in DeepSpec

| File | Purpose |
|---|---|
| `deepspec/modeling/dspark/markov_head.py` | VanillaMarkov, GatedMarkovHead, RNNHead |
| `deepspec/modeling/dspark/common.py` | DSparkForwardOutput, AcceptRatePredictor, context extraction, attention mask |
| `deepspec/modeling/dspark/loss.py` | Training loss (CE + TV + BCE confidence) |
| `deepspec/eval/dspark/draft_ops.py` | Inference-time `forward_dspark_draft_block()`, `build_dspark_proposal()` |
| `deepspec/eval/dspark/evaluator.py` | Full evaluation orchestration |
| `deepspec/eval/dspark/confidence_head.py` | ConfidenceHeadRecorder, calibration metrics |
| `config/dspark/dspark_qwen3_4b.py` | Training config (Qwen3 baseline) |

## Open Questions

1. **STS calibration temperatures** — Not in checkpoint. Can we get these from DeepSeek, or must we recalibrate?
2. **num_nextn_predict_layers override** — Should we add `dspark_num_layers: 3` to config, or repurpose `num_nextn_predict_layers`?
3. **vLLM speculative runner changes** — DSpark produces all γ tokens in one draft pass. Does the existing multi-step runner support this, or do we need a specialized `DSparkWorker`?
4. **Bidirectional block attention** — vLLM's attention backends may need a new code path or flag. Is `is_causal=False` sufficient for all backends (FlashInfer, FlashAttn, etc.)?
5. **Memory** — The DSpark checkpoint adds ~2 extra safetensors (~8GB). The 3 MTP layers with 256 experts each will have substantial memory overhead.
