# DSpark Backend Compatibility Matrix

> **Date:** 2026-06-27
> **Purpose:** Track which vLLM backends (attention, MoE, linear, mHC) support
> DSpark's requirements.
>
> **Target:** Standard recipe (`vllm-node` container, Ray executor, default backends).
> The Chthonic b12x stack is performance-optimized but out of scope for initial
> integration — revisit if profiling shows bottlenecks on the standard backend.

## DSpark Requirements by Component

| Component | Requirement | Why |
|---|---|---|
| Draft attention | Bidirectional (`is_causal=False`) within γ=5 block | DSpark draft backbone is a non-causal parallel pass |
| Draft MoE (256 experts) | Same as target model MoE | DSpark's 3 MTP layers each have 256 experts |
| Draft mHC (hyperconnections) | 4× HC dimension, same pattern as target | DSpark layers use mHC resnet structure |
| Draft hc_head | Hypercompressed LM head (4× compression) | mtp.2 output layer produces base logits |
| Draft shared_head | Vocab projection | Final logit projection to vocab_size=129280 |
| Markov head | 2 small linear ops (W₁[129280×256], W₂[256×129280]) | No special backend needed |
| Confidence head | Linear[4352→1] | No special backend needed |
| Context projection | Linear[12288→4096] + RMSNorm | No special backend needed |
| CUDA graphs | γ=5 block forward pass (Tier 1); variable-length (Tier 3+) | Memory and capture size scale with γ |

## Production Backend Stack (Chthonic)

```
Image: aidendle94/sparkrun-vllm-ds4-gb10:production-2.9
Architecture: SM121 (Blackwell GB10)
```

| Backend | Env/Flag | DSpark Support | Notes |
|---|---|---|---|
| MLA Attention | `ATTN_BACKEND: B12X_MLA_SPARSE` | **UNKNOWN** | Must test `is_causal=False`. If unsupported, fall back to FLASHINFER for draft layers. |
| MoE | `MOE_BACKEND: flashinfer_cutlass` | Likely OK | Same expert structure as target model. Different layer instances, same backend. |
| Linear | `VLLM_USE_B12X_*` (various) | **UNKNOWN** | b12x linear covers WO projection, FP8 GEMM, sparse indexer. Draft layers use same patterns. |
| mHC | `VLLM_USE_B12X_MHC: 1` | **UNKNOWN** | May assume exact layer count or hidden dims from target model. Draft has 3 layers vs 43. |
| AOT Compile | `VLLM_USE_AOT_COMPILE: 1` | Likely OK | New compute graph. Cold-boot time increase expected; persistent JIT cache mitigates. |
| CUDA Graphs | `FULL_AND_PIECEWISE` | Likely OK for Tier 1 (fixed γ=5) | Tier 2+ (variable lengths) needs breakable CUDA graphs or varlen support. |
| Sampler | `VLLM_USE_FLASHINFER_SAMPLER: 1` | OK | Draft token sampling is standard (argmax/multinomial). |
| NCCL/RoCE | Custom NCCL 2.30.4 | N/A | Draft model runs TP=1 per rank. No NCCL needed for draft. |

## Standard Backend Stack (vllm-node)

```
Image: vllm-node
Orchestration: Ray
```

| Backend | Flag | DSpark Support | Notes |
|---|---|---|---|
| MLA Attention | Default (FLASHINFER/FLASH_ATTN) | **LIKELY OK** | FlashInfer supports `is_causal=False` for non-causal attention. FlashAttn may require causal=False flag. |
| MoE | Default (flashinfer_cutlass) | Likely OK | Same as Chthonic. |
| Linear | Default | OK | Standard PyTorch ops, no custom backend. |
| mHC | Default (Python) | OK | Python fallback is available. |
| CUDA Graphs | Default | Likely OK for Tier 1 | May differ from Chthonic's FULL_AND_PIECEWISE. |

## Fallback Strategy

If any Chthonic b12x backend doesn't support DSpark requirements:

| Failing Backend | Fallback | Impact |
|---|---|---|
| B12X_MLA_SPARSE (bidirectional) | Set `ATTN_BACKEND=FLASHINFER` for draft layers | Minor — draft attention is small (γ=5 seq length, 4096 hidden) |
| b12x mHC | Disable `VLLM_USE_B12X_MHC` for draft layers (use Python fallback) | Minor — mHC is per-layer, small compute |
| b12x Linear | Fall back to default linear for draft layers | Minor — draft linear ops are small |
| AOT Compile (mega artifact) | Accept longer cold-boot time | Acceptable — one-time cost per deployment |

### Implementation approach for per-layer backend selection

If backends need to differ between target and draft layers:

```python
# In dspark.py or the speculative decode proposer:
# Use context managers or config overrides to switch backends per-model
with override_backend(ATTN_BACKEND="FLASHINFER"):
    draft_output = self.dspark_model.forward_backbone(...)
```

Or, more likely, configure at model construction time:
```python
# DSpark draft model uses a different VllmConfig with own backend settings
dspark_model = DeepSeekV4DSparkModel(
    vllm_config=draft_vllm_config,  # has its own attention/moe/linear backends
    prefix="draft_model.dspark",
)
```

## Verification Checklist

- [ ] B12X_MLA_SPARSE with `is_causal=False` — test on GB10 node
- [ ] flashinfer_cutlass MoE with draft model shards — compare outputs with eager mode
- [ ] b12x mHC with 3-layer draft — verify no shape errors
- [ ] DSpark compute graph compiles with AOT (`VLLM_USE_AOT_COMPILE=1`)
- [ ] CUDA graph capture with γ=5 — measure memory increase vs γ=2
- [ ] Draft TP=1 + target TP=2 — verify proposer doesn't reject
- [ ] FP8 KV cache compatibility — draft attention KV format
