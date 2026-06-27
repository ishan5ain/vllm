# DSpark vLLM Integration — Research Index

## Files in this directory

| File | Purpose |
|---|---|
| `IMPLEMENTATION_PLAN.md` | Full implementation roadmap, phases 1-5, open questions |
| `ALGORITHM_REFERENCE.md` | Paper equations, inference flow, production adaptations |
| `checkpoint_anatomy.md` | Exact weight structure, shapes, layer mapping |
| `dspark_model_skeleton.py` | Python skeleton with class stubs and weight loading map |

## External References

- **Paper PDF:** `/Users/ishansain/Downloads/DSpark_paper.pdf`
- **OCR'd paper:** `/Users/ishansain/Downloads/DSpark_paper_full.md`
- **DeepSpec repo (cloned):** `/tmp/pi-github-repos/deepseek-ai/DeepSpec`
- **HuggingFace model:** `deepseek-ai/DeepSeek-V4-Flash-DSpark`
- **vLLM MTP code:** `vllm/models/deepseek_v4/nvidia/mtp.py`
- **vLLM PR #40860:** [Feat] DeepSeek V4 Rebased (merged)

## Configuration Metadata

Fetched from `https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/raw/main/config.json`:

```json
{
  "dspark_block_size": 5,
  "dspark_noise_token_id": 128799,
  "dspark_target_layer_ids": [40, 41, 42],
  "dspark_markov_rank": 256,
  "num_nextn_predict_layers": 1,
  "hidden_size": 4096,
  "vocab_size": 129280,
  "num_hidden_layers": 43,
  "n_routed_experts": 256,
  "num_experts_per_tok": 6,
  "expert_dtype": "fp4"
}
```

## Key Design Decisions for Implementation

1. **Override `num_nextn_predict_layers`**: The config says 1 but the checkpoint has 3 MTP layers.
   Options:
   - (A) Add new config field `dspark_num_backbone_layers: 3`
   - (B) Use `num_nextn_predict_layers` from DSpark-specific config and override at load time
   - Recommend: (A) is cleaner, avoids confusing existing MTP logic.

2. **Bidirectional block attention**: DSpark requires `is_causal=False` within the draft block.
   The existing MTP code uses causal attention. This needs a new code path.

3. **Integration point**: Two options:
   - (A) Extend `DeepSeekV4MTP` — add markov/confidence heads, conditional bidirectional path
   - (B) Create `DeepSeekV4DSpark` as a parallel class — cleaner separation, less risk
   - Recommend: (B) to avoid breaking existing MTP. Share weight loading helpers where possible.

4. **Confidence scheduling**: Phase 4 (optional). Start with static `γ=5` verification, add
   confidence-based pruning later.
