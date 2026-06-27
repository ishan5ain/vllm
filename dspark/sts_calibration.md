# Sequential Temperature Scaling (STS) for DSpark Confidence Calibration

## Overview

The DSpark speculative decoding framework uses a **confidence head** that outputs per-position scalar estimates \\(c_k \\in (0,1)\\) modeling the conditional probability that a draft token at position \\(k\\) survives target verification, given all preceding tokens in the block are accepted.

However, neural networks tend to be **overconfident** (Guo et al., 2017): the predicted probabilities are systematically higher than observed acceptance rates. This overconfidence compounds across positions in the cumulative product \\(\\prod_{i \\leq k} c_i\\), leading to inaccurate throughput estimates and suboptimal scheduling decisions.

**Sequential Temperature Scaling (STS)** is a post-hoc calibration procedure from the DSpark paper (Section 3.2.1) that corrects this overconfidence while preserving the relative ranking of draft tokens.

## How STS Works

Standard temperature scaling (Guo et al., 2017) learns a **single global temperature** \\(T\\) — divide all logits by \\(T\\) before sigmoid. This works well for classification but is insufficient for DSpark's per-position confidence scores, because:

1. Each position \\(k\\) has different overconfidence characteristics.
2. The cumulative nature means errors at early positions propagate to later ones.

STS extends temperature scaling by learning **per-position temperatures** \\(T_1, \\ldots, T_\\gamma\\), calibrated sequentially from left to right:

```
For k = 1 to γ:
    1. Try candidate temperatures T_k in a predefined range [0.1, 5.0]
    2. For each candidate:
       a. Compute p_i = σ(logit_i / T_i) for i = 1..k
          (T_1..T_{k-1} are fixed from previous steps)
       b. Compute cumulative survival: ∏_{i≤k} p_i
       c. Compute ECE between cumulative predictions and observed prefix acceptance
    3. Select T_k that minimizes ECE at position k
```

Key properties:
- **Order-preserving**: Temperature scaling is monotonic — it changes probability magnitudes without disrupting the relative ranking of draft tokens.
- **Sequential**: Each position's calibration accounts for the cumulative effects of prior positions.
- **ECE-optimized**: Uses Expected Calibration Error (Naeini et al., 2015) as the optimization objective, directly targeting calibration quality.

## Expected Calibration Error (ECE)

ECE measures the mismatch between predicted probabilities and observed frequencies:

\\[\\text{ECE} = \\sum_{m=1}^{M} \\frac{|B_m|}{N} \\left| \\text{acc}(B_m) - \\text{conf}(B_m) \\right|\\]

Where:
- Predictions are split into \\(M\\) bins (typically 10-15)
- \\(\\text{acc}(B_m)\\) = average observed frequency in bin \\(m\\)
- \\(\\text{conf}(B_m)\\) = average predicted probability in bin \\(m\\)
- Weights are proportional to bin size

Lower ECE = better calibration. 0 = perfect calibration.

The STS module uses **equal-frequency binning** (each bin has ~same number of samples), which has lower statistical bias than equal-width binning (Roelofs et al., 2022).

## Expected Results

Based on the DSpark paper:

| State | ECE |
|-------|-----|
| Uncalibrated (raw sigmoid) | 3–8% |
| After STS calibration | ~1% |

The improvement is most dramatic at later positions (k=4,5) where cumulative overconfidence compounds. Early positions (k=1,2) typically need smaller corrections.

## Generating Calibration Data

STS requires a held-out calibration dataset of draft blocks and their verification outcomes:

### Step 1: Collect Draft-Verify Pairs

Run the DSpark draft model + target model verification on ~1000–5000 prompts:

```python
# Pseudo-code for data collection
calibration_logits = []
calibration_labels = []

for prompt in calibration_prompts:
    # Run DSpark draft model
    draft_output = dspark_model.generate_draft(prompt)
    
    for draft_block, confidence_logits in draft_output:
        # Target model verification
        verification = target_model.verify(draft_block)
        
        # Collect per-position results
        calibration_logits.append(confidence_logits)  # [γ] raw logits
        # Per-position acceptance: 1 if accepted, 0 if rejected
        calibration_labels.append(verification.accept_mask)  # [γ]
```

### Step 2: Format Data

Stack all collected blocks into [N, γ] arrays:

```python
import numpy as np

# confidence_logits: [N, γ] — raw logits before sigmoid
# accept_labels: [N, γ] — per-position 0/1 acceptance flags
logits = np.stack(calibration_logits)
labels = np.stack(calibration_labels)
```

### Step 3: Run STS Calibration

```python
from dspark.sts_calibration import calibrate_sts

temperatures = calibrate_sts(logits, labels, gamma=5)
# Example output: [1.2, 1.5, 1.8, 2.3, 3.0]
```

## Integrating with vLLM

### 1. Store Calibrated Temperatures

After running STS offline, store the temperatures in the model config or a separate calibration file:

```python
# Save temperatures
import json
with open("dspark_sts_temperatures.json", "w") as f:
    json.dump({"temperatures": temperatures, "gamma": len(temperatures)}, f)
```

### 2. Apply During Inference

In the vLLM DSpark implementation, apply temperatures when computing confidence scores:

```python
from dspark.sts_calibration import apply_temperatures, compute_cumulative_survival

# Load precomputed temperatures
temperatures = [1.2, 1.5, 1.8, 2.3, 3.0]  # from calibration

# During draft generation
hidden_states = self.draft_backbone(...)
confidence_logits = self.confidence_head(hidden_states)  # [B, γ] raw logits

# Apply STS calibration
calibrated_probs = apply_temperatures(confidence_logits, temperatures)

# Compute cumulative survival for prefix scheduler
cumulative_survival = compute_cumulative_survival(confidence_logits, temperatures)
```

### 3. Integration Point in DSpark Pipeline

The STS temperatures should be applied **after** the confidence head produces raw logits and **before** the Hardware-Aware Prefix Scheduler consumes the cumulative survival probabilities:

```
DSpark Backbone → Confidence Head (raw logits)
    → STS Temperature Scaling → Cumulative Survival Probs
        → Hardware-Aware Prefix Scheduler → Optimal verification lengths
```

### 4. Fallback

When no calibration data is available, use identity temperatures:

```python
from dspark.sts_calibration import compute_default_temperatures
temperatures = compute_default_temperatures(gamma=5)
# [1.0, 1.0, 1.0, 1.0, 1.0] — same as raw sigmoid
```

## Relationship to Standard Temperature Scaling

| Aspect | Standard TS (Guo 2017) | DSpark STS |
|--------|----------------------|------------|
| Parameters | 1 global T | γ per-position T_k |
| Optimization | NLL on validation set | ECE per position (grid search) |
| Calibration target | Multi-class softmax outputs | Cumulative product of per-step sigmoids |
| Inference cost | O(1) per forward pass | O(γ) per forward pass |
| Monotonic? | Yes | Yes (per position) |
| Preserves ranking? | Yes | Yes |

Standard temperature scaling uses NLL optimization because log-likelihood is a proper scoring rule that decomposes into calibration + refinement. STS uses **grid search over ECE directly** because the cumulative product structure means NLL optimization would need to account for the chain-rule dependencies across positions — grid search is simpler and equally effective for this low-dimensional problem.

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Calibration data needed | 1000–5000 draft blocks |
| Calibration runtime | ~milliseconds on CPU |
| Inference overhead | O(γ) ~5 divisions + sigmoids |
| Memory overhead | γ floats (~40 bytes) |
| ECE improvement | 3-8% → ~1% |

## References

- **Guo et al. (2017)** — "On Calibration of Modern Neural Networks", ICML. Introduced temperature scaling; showed modern DNNs are systematically overconfident.
- **Naeini et al. (2015)** — "Obtaining Well Calibrated Probabilities Using Bayesian Binning", AAAI. Introduced ECE metric.
- **Platt (1999)** — "Probabilistic Outputs for Support Vector Machines". Original Platt scaling (2-parameter logistic calibration).
- **Roelofs et al. (2022)** — "Mitigating Bias in Calibration Error Estimation". Showed equal-mass binning reduces ECE estimation bias.
- **DSpark Paper** — "DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation". DeepSeek-AI, 2026.
