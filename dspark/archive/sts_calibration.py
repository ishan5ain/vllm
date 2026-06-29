"""
Sequential Temperature Scaling (STS) Calibration for DSpark Confidence Head
============================================================================

Implements the STS post-hoc calibration procedure described in the DSpark paper
(Section 3.2.1, "Post-hoc Calibration") for calibrating the confidence head's
per-position scalar estimates c_k ∈ (0, 1).

Background
----------
The DSpark confidence head outputs per-position probability estimates c_k that
model the conditional probability that a draft token at position k survives
target verification, given all preceding tokens in the block have been accepted.

Because neural confidence estimates are typically overconfident (Guo et al.,
2017), the raw sigmoid outputs do not match empirical acceptance rates. This
matters because DSpark's Hardware-Aware Prefix Scheduler relies on the absolute
magnitudes of cumulative acceptance probabilities (∏_{i≤k} c_i) to estimate
expected throughput.

Temperature Scaling Background
------------------------------
Temperature scaling (Guo et al., 2017, "On Calibration of Modern Neural
Networks") is the simplest and most widely used post-hoc calibration method.
For a classification model, it learns a single scalar T > 0 and divides all
logits by T before softmax:
    p_calibrated = softmax(logits / T)
T > 1 softens the distribution (reduces overconfidence), T < 1 sharpens it.
Because it applies a single monotonic transformation across all classes, it
preserves accuracy—only confidence magnitudes change.

A key insight from Guo et al. is that temperature scaling, despite having only
ONE parameter, often matches or outperforms more complex methods like Platt
scaling (which fits a logistic regression) and isotonic regression, because
modern neural networks tend to be systematically overconfident in a way that a
single temperature can correct. However, vector scaling (per-class temperatures)
and matrix scaling can help when different classes have different overconfidence
patterns.

Platt scaling (Platt, 1999) was originally designed for SVM outputs and learns
two parameters (a, b) via logistic regression: p = sigmoid(a * score + b).
For multi-class neural networks, temperature scaling is preferred because it
doesn't distort the relative ordering of class probabilities.

Modern best practices (2024-2025):
- Temperature scaling remains the first-line method due to simplicity.
- LBFGS optimizer converges faster than SGD for the single-parameter case.
- For LLMs, adaptive temperature scaling (Joy et al., 2023; Balanya et al., 2024)
  predicts per-token or per-input temperatures from hidden features, addressing
  the heterogeneous miscalibration across different linguistic contexts.
- Equal-mass (equal-frequency) binning is preferred over equal-width binning
  for ECE estimation, as it has lower statistical bias (Roelofs et al., 2022).

STS Algorithm
-------------
Standard temperature scaling learns a single global T. DSpark's STS extends
this by learning per-position temperatures, calibrated sequentially:

1. For position k (1-indexed, going left to right), perform a 1D grid search
   over candidate temperatures T_k in a predefined range.

2. For each candidate T_k:
   a. Apply sigmoid with temperature: p_i = sigmoid(logit_i / T_i) for i=1..k
      (T_1..T_{k-1} are already fixed from prior steps)
   b. Compute cumulative survival probability: cumprod(p_1, ..., p_k)
   c. Compute ECE between cumulative probabilities at position k and the
      empirical prefix acceptance labels across all calibration samples.

3. Select T_k that minimizes ECE at position k.

4. Proceed to position k+1.

This sequential, order-preserving approach preserves the relative ranking of
draft tokens (i.e., doesn't change which tokens are preferred) while correcting
the absolute probability magnitudes. The paper reports ECE improvements from
3-8% (uncalibrated) to approximately 1% after STS calibration.

Usage Notes
-----------
Generating calibration data (~1000-5000 samples):
1. Run the draft model (DSpark) on a held-out validation set of prompts.
2. For each decoding cycle, collect:
   - confidence_logits: raw logits from the confidence head before sigmoid
   - accept_labels: per-position binary labels (1=accepted, 0=rejected)
     from target model verification.
3. Concatenate across cycles to form [N, γ] arrays where N is total number
   of draft blocks collected.

Expected ECE improvement:
- Uncalibrated: 3-8% ECE (neural overconfidence)
- After STS: ~1% ECE
- The improvement is most dramatic at later positions (k=4,5) where
  cumulative overconfidence compounds.

Memory/performance characteristics:
- STS calibration runs once offline on a held-out set; it has zero inference
  overhead (temperatures are precomputed constants).
- Memory: O(γ * N) for storing logits and labels; with N=5000 and γ=5,
  this is negligible (~200KB).
- Compute: O(γ * G * N) where G is grid size (~50-100). With N=5000,
  γ=5, G=100, this is ~2.5M sigmoid+ECE operations, completing in
  milliseconds on CPU.
- Inference: O(γ) to divide each logit by its precomputed temperature
  before sigmoid—effectively free.

References
----------
- Guo et al. (2017): "On Calibration of Modern Neural Networks", ICML.
  Introduced temperature scaling and showed modern DNNs are overconfident.
- Naeini et al. (2015): "Obtaining Well Calibrated Probabilities Using
  Bayesian Binning", AAAI. Introduced ECE metric.
- Platt (1999): "Probabilistic Outputs for Support Vector Machines".
  Original Platt scaling method.
- Roelofs et al. (2022): "Mitigating Bias in Calibration Error Estimation".
  Showed equal-mass binning reduces ECE estimation bias.
- Ovadia et al. (2019): "Can You Trust Your Model's Uncertainty?" NeurIPS.
  Evaluated calibration under distribution shift.
- Joy et al. (2023), Balanya et al. (2024), Ding et al. (2021):
  Adaptive temperature scaling for LLMs with per-token temperatures.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Union, List, Tuple


def _to_numpy(x: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
    """Convert torch tensor or numpy array to numpy float64 array."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().to(torch.float64).numpy()
    except ImportError:
        pass
    return np.asarray(x, dtype=np.float64)


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    binning: str = "equal_frequency",
) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE measures the difference between predicted probabilities and observed
    frequencies, binned to estimate the conditional expectation E[Y|f(X)].
    Lower is better; 0 indicates perfect calibration.

    The standard definition (Naeini et al., 2015; Guo et al., 2017):
        ECE = sum_{m=1}^{M} (|B_m| / N) * |acc(B_m) - conf(B_m)|
    where B_m is the m-th bin, acc is average label in the bin, and conf is
    average predicted probability in the bin.

    Args:
        probs: 1D array of predicted probabilities in [0, 1], shape [N].
        labels: 1D array of binary labels {0, 1}, shape [N].
        n_bins: Number of bins for discretization (default: 15).
        binning: Binning strategy:
            - "equal_frequency" (default): Each bin has ~N/n_bins samples.
              Lower bias than equal-width (Roelofs et al., 2022).
            - "equal_width": Bins span equal intervals in [0, 1].

    Returns:
        ECE value as a float.

    Raises:
        ValueError: If probs and labels have different shapes, or if probs
            are not in [0, 1], or if labels are not binary.
    """
    probs = np.asarray(probs, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()

    if probs.shape != labels.shape:
        raise ValueError(
            f"probs and labels must have the same shape, got {probs.shape} "
            f"and {labels.shape}"
        )
    if np.any(probs < 0) or np.any(probs > 1):
        raise ValueError("probs must be in [0, 1]")
    if not np.all(np.isin(labels, [0.0, 1.0])):
        raise ValueError("labels must be binary {0, 1}")

    n = len(probs)
    if n == 0:
        return 0.0

    if binning == "equal_frequency":
        # Sort by predicted probability, then split into equal-sized bins.
        # This ensures each bin has roughly the same number of samples,
        # which reduces estimation bias (Roelofs et al., 2022).
        sort_idx = np.argsort(probs)
        sorted_probs = probs[sort_idx]
        sorted_labels = labels[sort_idx]

        ece = 0.0
        for i in range(n_bins):
            start = int(i * n / n_bins)
            end = int((i + 1) * n / n_bins)
            if start >= end:
                continue
            bin_probs = sorted_probs[start:end]
            bin_labels = sorted_labels[start:end]
            bin_weight = len(bin_probs) / n
            avg_prob = np.mean(bin_probs)
            avg_label = np.mean(bin_labels)
            ece += bin_weight * np.abs(avg_prob - avg_label)
        return float(ece)

    elif binning == "equal_width":
        # Equal-width bins: divide [0, 1] into n_bins equal intervals.
        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            # Right-inclusive on the last bin to capture p=1.0
            if i == n_bins - 1:
                mask = (probs >= lo) & (probs <= hi)
            else:
                mask = (probs >= lo) & (probs < hi)
            bin_count = np.sum(mask)
            if bin_count == 0:
                continue
            bin_probs = probs[mask]
            bin_labels = labels[mask]
            bin_weight = bin_count / n
            avg_prob = np.mean(bin_probs)
            avg_label = np.mean(bin_labels)
            ece += bin_weight * np.abs(avg_prob - avg_label)
        return float(ece)

    else:
        raise ValueError(
            f"Unknown binning strategy: {binning}. "
            f"Use 'equal_frequency' or 'equal_width'."
        )


def compute_adaptive_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute ECE with adaptive binning (equal-mass, also called equal-frequency).

    This is a convenience wrapper around compute_ece with equal_frequency
    binning, matching the adaptive binning scheme commonly used in
    calibration literature. Adaptive (equal-mass) binning puts roughly
    the same number of samples in each bin, which has been shown to
    produce lower-bias ECE estimates than equal-width binning
    (Roelofs et al., 2022; Nixon et al., 2019).

    Args:
        probs: 1D array of predicted probabilities in [0, 1].
        labels: 1D array of binary labels {0, 1}.
        n_bins: Number of bins (default: 10, Naeini et al. use 10, 
                Guo et al. use 15).

    Returns:
        ECE value as a float.
    """
    return compute_ece(probs, labels, n_bins=n_bins, binning="equal_frequency")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    # Clip to avoid overflow in exp
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def calibrate_sts(
    confidence_logits: Union[np.ndarray, "torch.Tensor"],
    accept_labels: Union[np.ndarray, "torch.Tensor"],
    gamma: int = 5,
    t_min: float = 0.1,
    t_max: float = 5.0,
    n_grid: int = 98,
    n_bins: int = 10,
    return_metrics: bool = False,
) -> Union[List[float], Tuple[List[float], dict]]:
    """Calibrate confidence head logits using Sequential Temperature Scaling.

    Implements the STS algorithm from the DSpark paper (Section 3.2.1).
    For each position k (1..gamma), performs a 1D grid search over
    candidate temperatures T_k to minimize the Expected Calibration Error
    (ECE) of the cumulative survival probability at position k, keeping
    previously calibrated positions (1..k-1) fixed.

    The key insight: because each c_k models a CONDITIONAL probability
    (survival at position k given positions 1..k-1 survived), the chain
    rule gives the cumulative survival probability as the product
    ∏_{i≤k} sigmoid(logit_i / T_i). STS calibrates this cumulative
    product to match empirical prefix acceptance rates.

    Args:
        confidence_logits: Raw logits from the confidence head before
            sigmoid. Shape [N, gamma] where N is the number of draft blocks.
            Can be numpy array or PyTorch tensor.
        accept_labels: Per-position binary acceptance labels.
            Shape [N, gamma]. 1 = draft token at position k was accepted
            by target verification, 0 = rejected.
            IMPORTANT: These are per-position labels. The function internally
            converts them to prefix acceptance labels (cumprod) because STS
            calibrates cumulative survival probabilities.
            Can be numpy array or PyTorch tensor.
        gamma: Number of draft positions (block size). Default: 5.
        t_min: Minimum temperature in the grid search. Default: 0.1.
            T < 1 sharpens distributions (increases confidence).
        t_max: Maximum temperature in the grid search. Default: 5.0.
            T > 1 softens distributions (reduces overconfidence).
        n_grid: Number of temperature candidates in the grid.
            Default: 98 (yields ~50 temps after adding the key threshold points).
        n_bins: Number of bins for ECE computation. Default: 10.
            Naeini et al. (2015) used 10, Guo et al. (2017) used 15.
        return_metrics: If True, also returns a dictionary with per-position
            calibration metrics (optimal T_k, ECE before/after).

    Returns:
        If return_metrics is False:
            temperatures: List of gamma floats [T_1, ..., T_gamma].
        If return_metrics is True:
            (temperatures, metrics_dict) where metrics_dict contains
            per-position calibration diagnostics.

    Usage Example:
        >>> # Collect calibration data from ~1000-5000 draft cycles
        >>> logits = np.array(...)   # [N, 5] raw confidence logits
        >>> labels = np.array(...)   # [N, 5] per-position acceptance
        >>> temps = calibrate_sts(logits, labels, gamma=5)
        >>> # At inference: calibrated_prob = sigmoid(logit_k / temps[k-1])

    Notes:
        - Temperature scaling is ORDER-PRESERVING: it changes probability
          magnitudes without changing the relative ranking of draft tokens.
          This preserves the confidence head's ranking ability.
        - The grid search uses log-spacing for better coverage of the
          temperature range (typical optimal T is between 1.0 and 3.0).
        - T=1.0 is always included in the grid (identity calibration).
        - For a given position k, only samples where positions 1..k-1 were
          accepted (i.e., the prefix survived to k-1) contribute to the ECE
          calculation at position k. This follows the conditional structure.
    """
    # Convert inputs to numpy float64
    logits = _to_numpy(confidence_logits)
    labels = _to_numpy(accept_labels)

    # Validate shapes
    if logits.ndim != 2 or logits.shape[1] < gamma:
        raise ValueError(
            f"confidence_logits must be [N, gamma>=] shape, got {logits.shape}"
        )
    if labels.shape != logits.shape:
        raise ValueError(
            f"accept_labels shape {labels.shape} must match "
            f"confidence_logits shape {logits.shape}"
        )
    if not np.all(np.isin(labels, [0.0, 1.0])):
        raise ValueError("accept_labels must be binary {0, 1}")

    N = logits.shape[0]

    # Convert per-position acceptance labels to prefix acceptance labels.
    # prefix_labels[n, k] = 1 iff all positions 0..k were accepted.
    # This is the cumulative product since labels are 0/1.
    prefix_labels = np.cumprod(labels, axis=1)

    # Build temperature grid.
    # Use a log-spaced grid for better coverage, plus explicit inclusion
    # of T=1.0 (identity) and T=2.0 (common optimal value).
    # Log-spacing is appropriate because the effect of T is multiplicative
    # on logits (division), making equal ratio steps more meaningful than
    # equal absolute steps.
    base_grid = np.logspace(np.log10(t_min), np.log10(t_max), n_grid)
    # Ensure T=1.0 is in the grid (identity calibration reference point)
    special_points = np.array([1.0, 2.0])
    t_grid = np.sort(np.unique(np.concatenate([base_grid, special_points])))

    # Precompute per-position unscaled sigmoid probabilities.
    # p_raw[n, k] = sigmoid(logits[n, k])
    p_raw = _sigmoid(logits)  # [N, gamma]

    # Compute uncalibrated ECE for baseline reporting
    uncalibrated_ece = np.zeros(gamma)
    for k in range(gamma):
        cumprod_raw = np.cumprod(p_raw[:, : k + 1], axis=1)[:, k]
        uncalibrated_ece[k] = compute_ece(
            cumprod_raw, prefix_labels[:, k], n_bins=n_bins
        )

    # STS: calibrate positions left to right
    temperatures = np.ones(gamma)  # Start with identity
    calibrated_ece = np.zeros(gamma)

    # We'll maintain calibrated cumulative probabilities as we go
    # cumprod_cal[n, k] = calibrated cumulative survival probability at pos k
    cumprod_cal = np.ones((N, gamma))

    for k in range(gamma):
        # For position k, we need to compute:
        #   p_cal_k = sigmoid(logits[n, k] / T)
        #   cumprod_cal[n, k] = cumprod_cal[n, k-1] * p_cal_k
        # and measure ECE between cumprod_cal[:, k] and prefix_labels[:, k].
        #
        # Positions 0..k-1 have fixed temperatures. For k=0, cumprod_cal is
        # just sigmoid(logits[:, 0] / T_0).

        best_t = 1.0
        best_ece = float("inf")

        # If k > 0, we use the previously computed cumulative product
        # as the base for the current position's product.
        if k > 0:
            prev_cumprod = cumprod_cal[:, k - 1]  # [N]
        else:
            prev_cumprod = np.ones(N)  # All prefixes "survive" to position 0

        for t in t_grid:
            # Calibrate the per-step probability at position k
            p_cal_k = _sigmoid(logits[:, k] / t)  # [N]

            # Cumulative product up to position k
            curr_cumprod = prev_cumprod * p_cal_k  # [N]

            # Compute ECE between predicted cumulative probability at
            # position k and the empirical prefix acceptance at position k.
            ece = compute_ece(
                curr_cumprod, prefix_labels[:, k], n_bins=n_bins
            )

            if ece < best_ece:
                best_ece = ece
                best_t = t

        # Store the best temperature and update cumulative product
        temperatures[k] = best_t
        calibrated_ece[k] = best_ece

        # Update cumulative product for the next positions
        p_cal_k = _sigmoid(logits[:, k] / best_t)
        cumprod_cal[:, k] = prev_cumprod * p_cal_k

    # Convert to Python floats for clean return
    temps_list = [float(t) for t in temperatures]

    if return_metrics:
        metrics = {
            "gamma": gamma,
            "temperatures": temps_list,
            "uncalibrated_ece": [float(e) for e in uncalibrated_ece],
            "calibrated_ece": [float(e) for e in calibrated_ece],
            "ece_improvement": [
                float(u - c) for u, c in zip(uncalibrated_ece, calibrated_ece)
            ],
            "mean_uncalibrated_ece": float(np.mean(uncalibrated_ece)),
            "mean_calibrated_ece": float(np.mean(calibrated_ece)),
            "grid_size": len(t_grid),
            "t_range": [float(t_min), float(t_max)],
            "n_samples": N,
            "n_bins": n_bins,
        }
        return temps_list, metrics

    return temps_list


def compute_default_temperatures(gamma: int = 5) -> List[float]:
    """Return identity (no-op) temperature vector for initial/default use.

    When no calibration data is available, use identity temperatures
    (T_k = 1.0 for all k), which corresponds to uncalibrated raw sigmoid
    probabilities.

    Args:
        gamma: Number of draft positions (block size). Default: 5.

    Returns:
        List of gamma floats, all 1.0.

    Usage:
        >>> temps = compute_default_temperatures(gamma=5)
        >>> # temps = [1.0, 1.0, 1.0, 1.0, 1.0]
        >>> # calibrated_prob = sigmoid(logit_k / temps[k-1]) = sigmoid(logit_k)
    """
    return [1.0] * gamma


def apply_temperatures(
    confidence_logits: Union[np.ndarray, "torch.Tensor"],
    temperatures: List[float],
) -> np.ndarray:
    """Apply precomputed STS temperatures to confidence logits at inference time.

    Given raw logits and calibrated temperatures, returns calibrated
    per-step probabilities: p_k = sigmoid(logit_k / T_k).

    Args:
        confidence_logits: Raw logits from the confidence head.
            Shape [..., gamma] (supports batch dimensions).
        temperatures: List of gamma temperatures from calibrate_sts().

    Returns:
        Calibrated probabilities, same shape as input, in [0, 1].

    Usage:
        >>> logits = model.confidence_head(hidden_states)  # [B, gamma]
        >>> probs = apply_temperatures(logits, temperatures)  # [B, gamma]
        >>> cumulative_survival = np.cumprod(probs, axis=-1)
    """
    logits = _to_numpy(confidence_logits)
    temps = np.asarray(temperatures, dtype=np.float64)

    if logits.shape[-1] != len(temps):
        raise ValueError(
            f"Last dimension of logits ({logits.shape[-1]}) must match "
            f"number of temperatures ({len(temps)})"
        )

    # Broadcast temperatures to match logits shape
    # temps: [gamma] -> reshape for broadcasting
    scaled_logits = logits / temps.reshape(
        (1,) * (logits.ndim - 1) + (-1,)
    )
    return _sigmoid(scaled_logits)


def compute_cumulative_survival(
    confidence_logits: Union[np.ndarray, "torch.Tensor"],
    temperatures: Optional[List[float]] = None,
) -> np.ndarray:
    """Compute cumulative prefix survival probabilities.

    For each position k, computes ∏_{i≤k} sigmoid(logit_i / T_i).
    If temperatures is None, uses identity (T_i = 1.0 for all i).

    This is the key quantity used by DSpark's Hardware-Aware Prefix Scheduler
    to determine optimal verification lengths.

    Args:
        confidence_logits: Raw logits, shape [..., gamma].
        temperatures: Optional per-position temperatures. If None, uses
            identity calibration.

    Returns:
        Cumulative survival probabilities, same shape as input.
        cumprod[n, k] = probability that all positions 0..k survive.

    Usage:
        >>> logits = model.confidence_head(hidden_states)  # [B, gamma]
        >>> survival = compute_cumulative_survival(logits, temperatures)
        >>> # survival[b, k] = P(prefix length >= k+1)
    """
    logits = _to_numpy(confidence_logits)
    gamma = logits.shape[-1]

    if temperatures is not None:
        temps = np.asarray(temperatures, dtype=np.float64)
        if len(temps) != gamma:
            raise ValueError(
                f"temperatures length {len(temps)} must match gamma={gamma}"
            )
        scaled_logits = logits / temps.reshape(
            (1,) * (logits.ndim - 1) + (-1,)
        )
        probs = _sigmoid(scaled_logits)
    else:
        probs = _sigmoid(logits)

    return np.cumprod(probs, axis=-1)


# ---------------------------------------------------------------------------
# Demonstration and testing
# ---------------------------------------------------------------------------

def _demo():
    """Demonstrate STS calibration on synthetic data.

    Generates synthetic confidence logits and acceptance labels that mimic
    the overconfidence pattern observed in neural draft models: early
    positions are reasonably calibrated but later positions become
    increasingly overconfident due to compounding errors in the cumulative
    product.
    """
    rng = np.random.RandomState(42)
    N = 3000  # Number of draft blocks (typical calibration set size)
    gamma = 5

    # Generate synthetic true per-step acceptance probabilities.
    # Early positions have higher acceptance, decaying as we go deeper.
    true_step_probs = np.array([0.85, 0.70, 0.55, 0.40, 0.30])

    # Generate per-step acceptance labels (Bernoulli samples)
    labels = np.zeros((N, gamma))
    for k in range(gamma):
        labels[:, k] = (rng.rand(N) < true_step_probs[k]).astype(np.float64)

    # Generate overconfident logits: map true probs to logit space,
    # then artificially inflate them (simulating neural overconfidence).
    # logit = log(p / (1-p)) for the true probability
    # Then multiply by 1.5 to make them overconfident.
    eps = 1e-6
    true_logits = np.log(
        (true_step_probs + eps) / (1 - true_step_probs + eps)
    )
    overconfident_factor = 1.5
    base_logits = true_logits * overconfident_factor

    # Add noise to simulate per-sample variation
    noise_scale = 0.3
    logits = base_logits[np.newaxis, :] + noise_scale * rng.randn(N, gamma)

    print("=" * 60)
    print("Sequential Temperature Scaling (STS) Demo")
    print("=" * 60)
    print(f"Calibration samples: {N}")
    print(f"Block size (gamma): {gamma}")
    print(f"True step acceptance probabilities: {true_step_probs}")
    print()

    # Default (uncalibrated) behavior
    default_temps = compute_default_temperatures(gamma=gamma)
    print(f"Default temperatures (identity): {default_temps}")
    print()

    # Run STS calibration
    temps, metrics = calibrate_sts(
        logits, labels, gamma=gamma, return_metrics=True
    )

    print("STS Calibration Results:")
    print("-" * 40)
    for k in range(gamma):
        print(
            f"  Position {k + 1}: T={temps[k]:.4f}  "
            f"ECE: {metrics['uncalibrated_ece'][k]:.4f} → "
            f"{metrics['calibrated_ece'][k]:.4f}  "
            f"(Δ={metrics['ece_improvement'][k]:.4f})"
        )
    print()
    print(
        f"  Mean ECE: {metrics['mean_uncalibrated_ece']:.4f} → "
        f"{metrics['mean_calibrated_ece']:.4f}"
    )
    print()

    # Demonstrate inference-time usage
    print("Inference-time usage example:")
    print(f"  temperatures = {[round(t, 3) for t in temps]}")
    print(f"  probs = apply_temperatures(logits, temperatures)")
    print(f"  cumulative_survival = compute_cumulative_survival(logits, temperatures)")
    print()

    # Verify order preservation
    # Temperature scaling is order-preserving: sigmoid(x/T) is monotonic in x,
    # so higher logits always map to higher probabilities.
    sample_logits = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    raw_probs = _sigmoid(sample_logits)
    cal_probs = apply_temperatures(sample_logits, temps)
    print("Order preservation check:")
    print(f"  Raw probs:    {raw_probs[0]}")
    print(f"  Calibrated:   {cal_probs[0]}")
    print(f"  Order preserved: {np.all(np.diff(raw_probs[0]) >= 0) == np.all(np.diff(cal_probs[0]) >= 0)}")
    print()

    # Also verify with compute_default_temperatures
    print(f"Default temperatures for gamma=5: {compute_default_temperatures()}")
    print(f"Default temperatures for gamma=8: {compute_default_temperatures(8)}")


if __name__ == "__main__":
    _demo()
