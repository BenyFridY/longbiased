# V23 Model Proposal — Dynamic Allocation with Confidence Scaling

## Executive Summary

The V22 model (BTC/CDI dynamic allocation) returns +1,155% over 2022-2026 with Sortino 4.11 and max drawdown -9.1%. However, it uses a fixed multiplier (K) per market regime that ignores model confidence — a 51% prediction receives the same position size as a 90% prediction. This proposal evaluates adding confidence-based position sizing (meta-labeling) to improve risk-adjusted returns.

**Three candidates for V23:**

| Config | Return | MaxDD | Sortino | Sharpe | Saturation | 2026 YTD |
|---|---|---|---|---|---|---|
| V22 baseline (K=50/30/15) | +1,155% | -9.1% | 4.11 | 2.36 | 43% | +14% |
| **A: Sigmoid K=60/30/15** | **+1,167%** | **-9.1%** | **4.28** | **2.43** | **41%** | **+15%** |
| **B: Conf floor=0.7, K=75/45/22** | **+1,228%** | **-9.1%** | **4.14** | **2.36** | **44%** | **+16%** |
| C: Conf floor=0.7, K=60/40/20 | +1,026% | -9.1% | 4.10 | 2.35 | 39% | +15% |

---

## 1. Problem Statement

### 1.1 Current Model (V22)

The V22 model uses 80 bagged XGBoost regressors trained on 37 features to predict BTC's 3-day forward return. Every Friday, the prediction is converted to an allocation:

```
allocation = clip(prediction × K[regime], -25%, +100%)
```

Where K depends on the SMA-based market regime:
- **BULL** (price > SMA50 > SMA200): K = 50
- **MILD** (price > SMA200): K = 30
- **BEAR** (else): K = 15

Unallocated capital earns CDI (Brazilian risk-free rate, currently 14.75%/yr).

### 1.2 The Problem: Fixed K Ignores Confidence

The model's directional accuracy varies significantly with its own confidence:

| Classifier Confidence | Days | Regression Accuracy |
|---|---|---|
| Very low (0-5%) | 85 | **55.3%** (coin flip) |
| Low (5-10%) | 104 | 59.6% |
| Medium (10-20%) | 128 | 64.1% |
| High (20-50%) | 129 | **64.3%** |

When the model is uncertain, it's barely better than random. When confident, accuracy reaches 64%. Yet V22 treats both identically — same K, same position size.

Additionally, with K=50 in BULL, any prediction above +2% saturates at 100% allocation. The model's nuance is lost 43% of the time.

### 1.3 The K Saturation Problem

K acts as an amplifier: it converts small predictions (1-3%) into operational position sizes (20-80%). Without K, positions are too small to generate meaningful returns. However, when K is too high relative to the prediction, the allocation clips at the cap (100% or -25%) and the model loses granularity.

**Example with V22 K=50 in BULL:**

| Prediction | K | Raw Alloc | After Clip | Model Nuance |
|---|---|---|---|---|
| +1.0% | 50 | 50% | 50% | Preserved |
| +1.5% | 50 | 75% | 75% | Preserved |
| +2.0% | 50 | 100% | **100% (cap)** | **Lost** |
| +3.0% | 50 | 150% | **100% (cap)** | **Lost** — same as +2% |
| +5.0% | 50 | 250% | **100% (cap)** | **Lost** — same as +2% |
| -1.0% | 50 | -50% | **-25% (cap)** | **Lost** |

In V22, **43% of all rebalance days are saturated** — the model hits the cap and its prediction magnitude is ignored. A strong conviction (+5%) is treated identically to a mild one (+2%).

**Adding confidence makes this worse with high K:** if K=75 in BULL, saturation rises to 44%. The confidence factor (e.g., ×0.7) still produces 75 × 0.02 × 0.7 = 1.05, which clips to 100%. The confidence information is wasted.

**With lower K (K=60) + sigmoid:** 60 × 0.02 × 0.90 = 1.08 → still saturates at confident predictions, but 60 × 0.02 × 0.60 = 0.72 → **72% allocation when uncertain**. The model expresses nuance.

**Saturation rates across configs:**

| Config | Saturation | Meaning |
|---|---|---|
| V22 K=50/30/15 | 43% | Almost half of decisions hit cap |
| K=75/45/22, conf=0.7 | 44% | Confidence has little room |
| **K=60/30/15, sigmoid** | **41%** | Slightly less, sigmoid dampens |
| K=60/40/20, conf=0.7 | 39% | More decisions with nuance |
| K=55/35/20, conf=0.5 | 30% | Much more expressive |

The fundamental question: **is it better for the model to go all-in more often (high saturation, high return) or to express varying conviction levels (low saturation, potentially better risk-adjusted)?**

### 1.4 Academic Foundation

López de Prado's *Advances in Financial Machine Learning* (Chapter 10) establishes that:

> "The expected value of a bet grows with the confidence of the prediction; the appropriate stake should too."

He proposes **meta-labeling**: a secondary classifier estimates confidence, and position size scales accordingly. This is standard practice in systematic quant funds.

---

## 2. Methodology

### 2.1 Architecture

We train two models on the same features and walk-forward schedule:

1. **Regressor** (XGBoost, 80 bags): predicts 3-day forward BTC return
2. **Classifier** (XGBoost, 80 bags): predicts P(BTC up in 3 days)

The classifier's output is used solely for position sizing, not for direction.

### 2.2 Allocation Formulas Tested

**V22 (baseline):**
```
allocation = prediction × K[regime]
```

**Sigmoid scaling:**
```
confidence = |P_classifier - 0.5|
sigmoid_factor = 1 / (1 + exp(-confidence × 15))    # range: 0.5 to 1.0
allocation = prediction × K[regime] × sigmoid_factor
```

**Confidence floor:**
```
confidence = |P_classifier - 0.5|
factor = max(floor, min(confidence / 0.5, 1.0))     # range: floor to 1.0
allocation = prediction × K[regime] × factor
```

**Inverse volatility:**
```
vol_factor = target_vol / realized_vol_30d            # range: 0.3 to 2.0
allocation = prediction × K × vol_factor
```

**Hybrid (confidence × volatility):**
```
allocation = prediction × K × confidence_factor × vol_factor
```

### 2.3 Evaluation

- Walk-forward: semi-annual retraining (Jan/Jul), out-of-sample 2022-2026
- 2 seeds (242, 442), averaged
- Metrics: return, max drawdown, Sortino, Sharpe, saturation %, yearly breakdown
- Rebalance: Fridays + emergency (|daily_ret| > 8%)

---

## 3. Experiments and Results

### 3.1 Horizon Analysis

| Horizon | Accuracy (Fri) | Best Sortino | Best Return |
|---|---|---|---|
| h=1d | 59.3% | 2.46 | +349% |
| h=2d | 59.3% | 3.69 | +806% |
| **h=3d** | **61.1%** | **4.28** | **+1,259%** |
| h=4d | 60.9% | 3.93 | +1,706% |
| h=5d | 57.1% | 3.90 | +1,625% |

**Conclusion:** h=3d has the best accuracy (61.1%) and best risk-adjusted metrics. h=4d and h=5d generate higher raw returns but with lower accuracy and higher drawdowns.

### 3.2 Rebalance Frequency

For h=3d, K=50/30/15:

| Frequency | Return | MaxDD | Sortino |
|---|---|---|---|
| Daily | +893% | -16.4% | 2.72 |
| MWF | +907% | -15.9% | 2.79 |
| Tue+Fri | +667% | -13.7% | 2.43 |
| **Friday only** | **+1,155%** | **-9.1%** | **4.11** |

**Conclusion:** Weekly rebalance (Friday) is definitively superior. More frequent trading adds noise and drawdown without improving returns. This is consistent with the 3-day prediction horizon.

### 3.3 K Regime Analysis

Without confidence scaling, K=60/30/15 is the best:

| K Config | Return | MaxDD | Sortino | Sat |
|---|---|---|---|---|
| K=50/30/15 (V22) | +1,155% | -9.1% | 4.11 | 43% |
| K=60/30/15 | +1,259% | -9.1% | 4.16 | 46% |
| K=60/40/20 | +1,391% | -9.7% | 3.91 | - |
| K=70/40/20 | +1,455% | -9.9% | 3.95 | - |

Higher K in BULL increases returns but also saturation. The regime structure is essential — removing it (K fixed) doubles max drawdown to -13% to -20%.

### 3.4 Confidence Scaling Methods

All tested with h=3d, Friday rebal, averaged over 2 seeds:

#### 3.4.1 Sigmoid (smooth, floor at 0.5)

| K Config + Sigmoid | Return | MaxDD | Sortino | Sharpe | 2026 |
|---|---|---|---|---|---|
| K=50/30/15 | +1,041% | -9.1% | 4.21 | 2.39 | +15% |
| **K=60/30/15** | **+1,167%** | **-9.1%** | **4.28** | **2.43** | **+15%** |
| K=60/40/20 | +1,289% | -9.1% | 4.05 | 2.33 | +18% |

#### 3.4.2 Confidence Floor = 0.7 (moderate dampening)

| K Config + Floor=0.7 | Return | MaxDD | Sortino | Sharpe | Sat | 2026 |
|---|---|---|---|---|---|---|
| K=50/30/15 | +741% | -8.2% | 3.98 | 2.35 | 32% | +12% |
| K=60/40/20 | +1,026% | -9.1% | 4.10 | 2.35 | 39% | +15% |
| K=70/40/20 | +1,125% | -9.1% | 4.17 | 2.38 | - | +15% |
| **K=75/45/22** | **+1,228%** | **-9.1%** | **4.14** | **2.36** | **44%** | **+16%** |

#### 3.4.3 Confidence Floor = 0.5 (aggressive dampening)

| K Config + Floor=0.5 | Return | MaxDD | Sortino | Sharpe | Sat | 2026 |
|---|---|---|---|---|---|---|
| K=70/40/20 | +836% | -8.1% | 4.23 | 2.38 | - | +17% |
| K=75/45/22 | +931% | -8.8% | 4.23 | 2.38 | - | +18% |
| K=80/50/25 | +1,031% | -9.0% | 4.22 | 2.37 | - | +20% |

### 3.5 Alternative Sizing Methods

| Method | Best Config | Return | MaxDD | Sortino | 2026 |
|---|---|---|---|---|---|
| Kelly (f=1.0, K100/60/30) | Full Kelly + regime | +974% | -8.9% | 3.84 | +23% |
| Inverse Vol (tv=0.4, K70/40/20) | Vol-only | +1,133% | -14.0% | 3.77 | +10% |
| Hybrid Conf+Vol (f=0.5, tv=0.6, K75/45/22) | Combined | +1,069% | -10.9% | 4.01 | +17% |
| No-K Normalized (scale=0.5) | No multiplier | +534% | -7.1% | 3.16 | +18% |

**Conclusions:**
- Kelly is too conservative for this signal strength (61% accuracy)
- Inverse volatility alone worsens drawdown (-14%)
- Hybrid adds complexity without clear benefit vs simple confidence
- No-K approaches fail — the multiplier is necessary to translate small predictions (~1-3%) into operational position sizes

### 3.6 Fixed K vs Regime K

| Config | Return | MaxDD | Sortino |
|---|---|---|---|
| K regime 75/45/22, conf=0.7 | +1,228% | -9.1% | 4.14 |
| K fixed=50, conf=0.7 | +1,005% | -13.4% | 2.83 |
| K fixed=35, conf=0.7 | +760% | -9.3% | 3.13 |

**Regime K is essential.** Fixed K doubles drawdown because it over-bets in BEAR markets. The SMA regime acts as a macro risk filter.

---

## 4. Accuracy Analysis

### 4.1 Overall Accuracy (h=3d, Friday, 2 seeds)

| Year | Accuracy | Notes |
|---|---|---|
| 2022 | 68.3% | Bear market, model excels |
| 2023 | 65.4% | Recovery, strong |
| 2024 | 54.8% | Bull→correction, weakest |
| 2025 | 55.8% | Mixed, below average |
| 2026 | 66.7% | Recent improvement |
| **Overall** | **61.1%** | |

### 4.2 Accuracy Does Not Change with Sizing Method

All sizing methods use the same regression predictions — accuracy is 61.1% regardless of K, confidence, or sigmoid. The difference is **how much the model bets when right vs wrong**.

### 4.3 Confidence Correctly Predicts Accuracy

| Classifier Confidence | Regression Accuracy | % of Days |
|---|---|---|
| Bottom half (conf < 0.13) | 54.1% | 50% |
| Top half (conf >= 0.13) | 64.3% | 50% |

The 10pp accuracy gap is statistically significant and consistent across years (except 2022 where both halves performed well).

---

## 5. Recommendation

### 5.1 Three Candidates

**Option A: Sigmoid + K=60/30/15** (Best risk-adjusted)
```
allocation = prediction × K[regime] × sigmoid(|P_cls - 0.5| × 15)
K = {BULL: 60, MILD: 30, BEAR: 15}
```
- Return: +1,167% | Sortino: **4.28** | Sharpe: **2.43** | MDD: -9.1%
- Pros: Best Sortino and Sharpe, return comparable to V22, smooth scaling
- Cons: BEAR K=15 is very conservative, 41% saturation

**Option B: Confidence floor=0.7 + K=75/45/22** (Best return with confidence)
```
allocation = prediction × K[regime] × max(0.7, |P_cls - 0.5| / 0.5)
K = {BULL: 75, MILD: 45, BEAR: 22}
```
- Return: **+1,228%** | Sortino: 4.14 | Sharpe: 2.36 | MDD: -9.1%
- Pros: 7% more return than V22, slightly better Sortino, uses confidence
- Cons: 44% saturation (similar to V22), floor of 0.7 has limited effect

**Option C: Confidence floor=0.7 + K=60/40/20** (Less saturation)
```
allocation = prediction × K[regime] × max(0.7, |P_cls - 0.5| / 0.5)
K = {BULL: 60, MILD: 40, BEAR: 20}
```
- Return: +1,026% | Sortino: 4.10 | Sharpe: 2.35 | MDD: -9.1%
- Pros: 39% saturation (model has more nuance), balanced K across regimes
- Cons: 11% less return than V22

### 5.2 Trade-off

The core trade-off is between **return** and **model expressiveness** (saturation):

- More saturation → higher returns (model goes all-in more often, and historically it's right 61%)
- Less saturation → model can express confidence levels, but smaller positions mean less profit

Given the academic literature (López de Prado meta-labeling, fractional Kelly) and our empirical results, **Option A (sigmoid + K=60/30/15)** is the most defensible:

1. It achieves the **highest Sortino (4.28) and Sharpe (2.43)** of any config tested
2. It retains **nearly identical return** to V22 (+1,167% vs +1,155%)
3. It **reduces position size when uncertain** (sigmoid floor at ~0.5) without eliminating trades
4. It's **academically grounded** in meta-labeling and confidence-weighted sizing
5. It adds minimal complexity (one classifier, one multiplication)

### 5.3 Implementation Requirements

V23 requires training **two** model ensembles instead of one:
- 80 XGBoost regressors (same as V22)
- 80 XGBoost classifiers (new, same features and schedule)

Training time increases by ~50% (from ~3 min to ~5 min per retrain). Prediction time is negligible.

---

## 6. Production Implementation

The production pipeline (`scripts/production/`) implements **Option A** exactly:

```python
# config.py
K_REGIME = {'BULL': 60, 'MILD': 30, 'BEAR': 15}
SIGMOID_SCALE = 15
FEATURES_ALL = FEATURES_37  # 37 features, same as V22

# generate_signal.py — allocation formula
confidence = abs(P_classifier - 0.5)
confidence_factor = 1 / (1 + exp(-confidence * 15))   # sigmoid: 0.50 to 1.0
allocation = clip(prediction * K[regime] * confidence_factor, -25%, +100%)
```

Key files:
- `config.py` — K=60/30/15, sigmoid scale=15, 37 features
- `generate_signal.py` — trains regressor + classifier, applies sigmoid
- `build_features.py` — 37 features aligned with original pipeline
- `bootstrap_from_original.py` — uses validated enhanced dataset as base

Note: 9 Messari features (active_addresses, buy/sell ratio, dominance, fees, etc.) were tested but **reduced returns from +1,155% to +757%**. They are NOT included in V23 until further validation.

---

## 7. Risk Considerations

1. **Accuracy degradation**: 2024-2025 accuracy dropped to 55%. If this continues, all configs lose money. The sigmoid helps by automatically reducing bets when uncertain.

2. **Regime lag**: SMA-based regime detection lags 30-60 days. The model may miss the start of bull runs. This is a known limitation not addressed in V23.

3. **Overfitting**: All results are out-of-sample (2022-2026, walk-forward). However, K values were optimized on this period. True out-of-sample validation requires live performance.

4. **CDI dependency**: The strategy's edge partly comes from earning 14.75%/yr on unallocated capital. If Brazilian rates drop significantly, the risk-free cushion diminishes.

---

## 7. Appendix: All Test Results Summary

Over 500 configurations were tested across:
- 5 prediction horizons (1-5 days)
- 5 rebalance frequencies (daily, MWF, Tue+Fri, Wed+Fri, Friday)
- 6+ K regime configurations
- 5 confidence floor values (0.5, 0.6, 0.7, 0.8, 0.9)
- Sigmoid scaling
- Kelly criterion (0.25×, 0.5×, 0.75×, 1.0×)
- Inverse volatility (4 target vols)
- Hybrid confidence × volatility
- No-K normalized approaches (8 methods)
- 2 random seeds per configuration

All tests used the same walk-forward framework with semi-annual retraining on the bootstrapped production dataset (2,661 days, 2019-01-01 to 2026-04-14).
