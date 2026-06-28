# Speculative Decoding Acceptance Rate Metrics in vLLM

## 1. Prometheus Metrics Exposed on `/metrics`

Three Prometheus Counter metrics are registered and exposed via the `/metrics` HTTP
endpoint. They are all per-engine counters (label `engine` for multi-engine/DP setups).

| Metric Name | Description |
|---|---|
| `vllm:spec_decode_num_drafts_total` | Number of speculative decoding draft batches processed |
| `vllm:spec_decode_num_draft_tokens_total` | Total number of draft tokens proposed |
| `vllm:spec_decode_num_accepted_tokens_total` | Total number of accepted draft tokens |
| `vllm:spec_decode_num_accepted_tokens_per_pos_total` | Accepted tokens broken down by draft position (labeled `position="N"`) |

**Registration:**
`vllm/v1/spec_decode/metrics.py` (lines 226–263)

The counters are created in `SpecDecodingProm.__init__` and observed in `SpecDecodingProm.observe()` (lines 266–281).

**Endpoint wiring:**
- `/metrics` is mounted via `vllm/entrypoints/serve/instrumentator/metrics.py` (lines 67–81) using `prometheus_fastapi_instrumentator` and `prometheus_client.make_asgi_app`.
- The registry is obtained from `vllm/v1/metrics/prometheus.py:get_prometheus_registry()` (line 39), which supports both single-process (`REGISTRY`) and multiprocess modes.

**Programmatic access (no HTTP):**
- `LLM.get_metrics()` → `vllm/v1/engine/llm_engine.py` (line 378) → `vllm/v1/metrics/reader.py:get_metrics_snapshot()` (line 70).
- Returns typed `Counter`, `Vector`, `Gauge`, `Histogram` objects.
- The per-position metric `vllm:spec_decode_num_accepted_tokens_per_pos` is specially handled as a `Vector` (lines 100–117 in reader.py).

## 2. Accepted/Rejected Count Flow

### 2a. Rejection Sampler (GPU)

**File:** `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`

`RejectionSampler.__call__()` (lines 98–158):
1. Takes `logits` (target), `draft_logits` (optional), `draft_sampled` (tokens proposed).
2. Calls `rejection_sample()` from `rejection_sampler_utils.py` to run Triton kernels.
3. Gets `sampled` (tensor of shape `[num_reqs, num_spec_steps+1]`) and `num_sampled` (per-request count of accepted + 1 bonus).
4. Calls `get_num_sampled_and_rejected()` from `vllm/v1/worker/gpu/input_batch.py` (line 432) to compute per-request `num_sampled` and `num_rejected`.
5. Returns `SamplerOutput(sampled_token_ids=sampled, num_sampled=..., num_rejected=...)`.

**Rejection logic** (in `rejection_sampler_utils.py`, `_rejection_kernel`, lines ~134–280):
- Iterates over draft tokens for each request.
- Each draft token is compared against target distribution:
  - **Greedy** (`temp == 0.0`): accept if `draft_token == target_argmax`.
  - **Non-greedy**: accept if `exp(target_log_prob) / max(exp(draft_log_prob), ε) > u` where `u ~ Uniform(0,1)` (log-space implementation).
  - **Synthetic mode**: accept if `u < synthetic_conditional_rate[i]`.
- On first rejection, the rejected token and bonus token are **resampled** from the residual distribution.
- `num_sampled` = number of tokens accepted + 1 (the bonus/resampled token).

### 2b. Scheduler Stats Aggregation

**File:** `vllm/v1/core/sched/scheduler.py` (lines 1573–1598)

Per request:
```python
num_draft_tokens = len(scheduled_spec_token_ids)       # tokens proposed
num_sampled = self.num_sampled_tokens_per_step         # always 1
num_accepted = max(len(generated_token_ids) - num_sampled, 0)
num_rejected = num_draft_tokens - num_accepted
```

Calls `self.make_spec_decoding_stats()` (lines 2290–2307) which creates/updates a `SpecDecodingStats` instance via `observe_draft(num_draft_tokens, num_accepted)`.

### 2c. SpecDecodingStats (Dataclass)

**File:** `vllm/v1/spec_decode/metrics.py` (lines 18–49)

```python
@dataclass
class SpecDecodingStats:
    num_spec_tokens: int            # config: num_speculative_tokens
    num_drafts: int = 0             # number of draft batches
    num_draft_tokens: int = 0       # total draft tokens proposed
    num_accepted_tokens: int = 0    # total accepted tokens
    num_accepted_tokens_per_pos: list[int]  # per-position histogram
```

`observe_draft(num_draft_tokens, num_accepted_tokens)`:
- Increments `num_drafts`, `num_draft_tokens`, `num_accepted_tokens`.
- Increments `num_accepted_tokens_per_pos[i]` for each accepted position.

The stats flow through:
- `SchedulerStats.spec_decoding_stats` → `vllm/v1/metrics/stats.py` (line 190)
- Observed by `SpecDecodingLogging` (lines 74–79) and `SpecDecodingProm` (lines 266–281) in `metrics/loggers.py` (lines 182–183 and 1103–1105).

## 3. Benchmark Script for Acceptance Rate

**File:** `vllm/benchmarks/serve.py`

### 3a. Fetching metrics before/after benchmark (lines 189–240, 957, 1050–1089)

`fetch_spec_decode_metrics(base_url, session)`:
- GETs `{base_url}/metrics` (Prometheus text format).
- Parses lines starting with `vllm:spec_decode` (all `_total` suffixes).
- Collects: `num_drafts`, `num_draft_tokens`, `num_accepted_tokens`, `accepted_per_pos` (dictionary of position → count).
- Returns `SpecDecodeMetrics` dataclass (lines 183–187).

### 3b. Computing acceptance stats (lines 1051–1089)

After benchmark completes, computes deltas:
```python
delta_drafts = after.num_drafts - before.num_drafts
delta_draft_tokens = after.num_draft_tokens - before.num_draft_tokens
delta_accepted = after.num_accepted_tokens - before.num_accepted_tokens

acceptance_rate = (delta_accepted / delta_draft_tokens) * 100           # percentage
acceptance_length = 1 + delta_accepted / delta_drafts                    # includes bonus token
per_position_acceptance_rates = [delta_pos / delta_drafts for each position]
```

### 3c. Output display (lines 1229–1344)

Reports: `spec_decode_acceptance_rate`, `spec_decode_acceptance_length`, `spec_decode_num_drafts`, `spec_decode_draft_tokens`, `spec_decode_accepted_tokens`, `spec_decode_per_position_acceptance_rates`.

Text output shows:
```
----------------Speculative Decoding-----------------
Acceptance rate (%):                       XX.XX
Acceptance length:                         X.XX
Drafts:                                    N
Draft tokens:                              N
Accepted tokens:                           N
Per-position acceptance (%):
  Position 0:                              XX.XX
  Position 1:                              XX.XX
```

## 4. E2E Test Utilities for Acceptance Rate

**File:** `tests/v1/e2e/spec_decode/test_spec_decode.py`

`compute_acceptance_rate(metrics, prev_metrics=None)` (lines 1218–1234):
```python
acceptance_rate = n_accepted_tokens / n_draft_tokens
```

`compute_acceptance_len(metrics, prev_metrics=None)` (lines 1237–1254):
```python
acceptance_length = 1 + (n_accepted_tokens / n_drafts)
```

These read from `llm.get_metrics()` (via the `Metric`/`Counter`/`Vector` typed API in `vllm/v1/metrics/reader.py`).

## 5. PromQL Queries for Monitoring

From `vllm/v1/spec_decode/metrics.py` comments (lines 180–195):

```promql
# Acceptance rate (fraction of draft tokens that were accepted):
rate(vllm:spec_decode_num_accepted_tokens_total[$interval]) /
rate(vllm:spec_decode_num_draft_tokens_total[$interval])

# Mean acceptance length (including bonus token):
1 + (
  rate(vllm:spec_decode_num_accepted_tokens_total[$interval]) /
  rate(vllm:spec_decode_num_drafts_total[$interval])
)

# Per-position acceptance rate vector:
vllm:spec_decode_num_accepted_tokens_per_pos_total[$interval] /
vllm:spec_decode_num_drafts_total[$interval]
```

## 6. DSpark-Specific Notes

The DSpark speculator (`vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`) generates all γ draft tokens in one forward pass. It uses the same `RejectionSampler`, `SpecDecodingStats`, `SpecDecodingProm`, and metrics pipeline as all other spec-decode methods. There are no DSpark-specific metric names — the same Prometheus counters apply.

## 7. Key Files Summary

| File | Role |
|---|---|
| `vllm/v1/spec_decode/metrics.py` | `SpecDecodingStats`, `SpecDecodingLogging`, `SpecDecodingProm` — metric dataclass, logging, and Prometheus counter registration |
| `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py` | GPU rejection sampling orchestrator; returns `num_sampled`/`num_rejected` |
| `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py` | Triton kernels for rejection sampling (`_rejection_kernel`, `_resample_kernel`, `_insert_resampled_kernel`) |
| `vllm/v1/worker/gpu/input_batch.py` (lines 404–454) | `get_num_sampled_and_rejected()` — per-request sampled/rejected counts from GPU tensors |
| `vllm/v1/core/sched/scheduler.py` (lines 1573–1598, 2290–2307) | Scheduler computes `num_accepted`/`num_rejected` from token IDs, calls `make_spec_decoding_stats()` |
| `vllm/v1/metrics/stats.py` (line 190) | `SchedulerStats.spec_decoding_stats` field |
| `vllm/v1/metrics/loggers.py` (lines 182–183, 406–, 1095–1105) | `PrometheusStatLogger` and `AggregateStatLoggerBase` observe spec decode stats |
| `vllm/v1/metrics/reader.py` | `get_metrics_snapshot()` — programmatic metric access |
| `vllm/entrypoints/serve/instrumentator/metrics.py` | FastAPI `/metrics` endpoint wiring |
| `vllm/v1/metrics/prometheus.py` | Prometheus multiprocess setup, registry |
| `vllm/benchmarks/serve.py` (lines 183–240, 957, 1050–1344) | Benchmark script that fetches/parses/deltas spec decode metrics |
| `tests/v1/e2e/spec_decode/test_spec_decode.py` (lines 1218–1254) | Test helpers `compute_acceptance_rate` and `compute_acceptance_len` |
| `vllm/v1/worker/gpu/spec_decode/dspark/` | DSpark speculator — uses same metrics pipeline |
