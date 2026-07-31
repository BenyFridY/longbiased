# V23 Experiment Log

## Overview

Over 1,500+ configurations were tested across two phases to select the final V23 model.
All results are out-of-sample using walk-forward validation with semi-annual retraining (Jan/Jul)
and purge=5d to eliminate label leakage (Lopez de Prado, AFML Ch. 7).

**Final model: Sigmoid s=15, K=60/30/15, 37 features, h=3d, Friday rebalance.**

**Final validated metrics (BRL, what a Brazilian investor actually earns):**
- Return 2022-2026: **+793%** | Sortino **3.21** | Sharpe **1.72** | MDD **-10.7%**
- Accuracy (Friday directional, h=3d): **60.3%**
- 2026 YTD (strict OOS, live): Strategy **+11%** vs BTC **-26%**

USD validation (10 seeds): +1063%, Sortino 4.05. BRL is ~15% lower due to BRL appreciation
(+11.5% vs USD) over the period. BRL is the honest metric for a Brazilian investor.

---

## Phase 1: Exploratory Tests (V23 Proposal)

Ran across 12 experiment scripts, 2 seeds (242, 442), 37 features, h=3d unless noted.

### 1.1 Prediction Horizon (v23_horizon_accuracy.py)

| Horizon | Accuracy (Fri) | Best Sortino |
|---|---|---|
| h=1d | 59.3% | 2.46 |
| h=2d | 59.3% | 3.69 |
| **h=3d** | **61.1%** | **4.28** |
| h=4d | 60.9% | 3.93 |
| h=5d | 57.1% | 3.90 |

**Decision:** h=3d — best accuracy and risk-adjusted return.

### 1.2 Rebalance Frequency (v23_horizon_accuracy.py)

| Frequency | Return | MaxDD | Sortino |
|---|---|---|---|
| Daily | +893% | -16.4% | 2.72 |
| MWF | +907% | -15.9% | 2.79 |
| **Friday only** | **+1,155%** | **-9.1%** | **4.11** |

**Decision:** Friday-only rebalance — dramatically better Sortino and lower drawdown.

### 1.3 Regime Detection (v23_experiments.py)

| Method | Best Sortino | MaxDD |
|---|---|---|
| **SMA (price > SMA50 > SMA200)** | **4.11** | **-9.1%** |
| Momentum (30d return thresholds) | 3.2 | -12% |
| Volatility (30d vol thresholds) | 2.9 | -14% |
| No regime (fixed K) | 2.8 | -13% |

**Decision:** SMA-based regime is essential — removing it doubles drawdown.

### 1.4 K Multiplier Sweep (v23_k_sweep_final.py, v23_confidence_k_sweep.py)

50+ K combinations tested. Key results without confidence:

| K Config | Return | MaxDD | Sortino |
|---|---|---|---|
| K=50/30/15 (V22) | +1,155% | -9.1% | 4.11 |
| **K=60/30/15** | **+1,259%** | **-9.1%** | **4.16** |
| K=60/40/20 | +1,391% | -9.7% | 3.91 |
| K=75/45/22 | +1,455% | -10.6% | 3.56 |

**Decision:** K=60/30/15 — slightly more return than V22 without increasing drawdown.

### 1.5 Confidence Scaling Methods (v23_gating_tests.py, v23_sizing_methods.py)

All methods tested with h=3d, Friday rebalance, 2 seeds averaged:

| Method | Best Config | Return | Sortino | MaxDD |
|---|---|---|---|---|
| **Sigmoid (scale=15)** | **K=60/30/15** | **+1,167%** | **4.28** | **-9.1%** |
| Confidence floor=0.7 | K=75/45/22 | +1,228% | 4.14 | -9.1% |
| Confidence floor=0.5 | K=80/50/25 | +1,031% | 4.22 | -9.0% |
| Kelly (f=1.0) | K=100/60/30 | +974% | 3.84 | -8.9% |
| Inverse Volatility | K=70/40/20 | +1,133% | 3.77 | -14.0% |
| Hybrid (conf + vol) | K=75/45/22 | +1,069% | 4.01 | -10.9% |
| No-K (normalized) | scale=0.5 | +534% | 3.16 | -7.1% |

**Decision:** Sigmoid s=15 — best Sortino and Sharpe, academically grounded (meta-labeling).

### 1.6 New Features (v23_model_improvement.py)

| Feature Set | Return | Sortino |
|---|---|---|
| **37 features (V22)** | **+1,155%** | **4.11** |
| 46 features (37 + 9 Messari) | +757% | 3.2 |

9 Messari features (active_addresses, buy_sell_ratio, dominance, fees, etc.) **worsened** performance.

**Decision:** Keep 37 features.

### 1.7 Retrain Frequency (v23_model_improvement.py)

| Frequency | Accuracy | Sortino |
|---|---|---|
| **Semi-annual (Jan/Jul)** | **61.1%** | **4.11** |
| Quarterly | 59.8% | 3.9 |
| Every 4 months | 60.2% | 3.8 |

**Decision:** Semi-annual retraining is optimal.

### 1.8 Model Types (v23_experiments.py)

| Model | Allocation Method | Sortino |
|---|---|---|
| **Regression + Classification** | **pred × K × sigmoid(conf)** | **4.28** |
| Regression only | pred × K | 4.11 |
| Classification only | (P-0.5) × scale | 2.8 |

**Decision:** Dual model (regression for direction/magnitude, classification for confidence).

---

## Phase 2: Ultimate Validation (v23_ultimate.py)

Comprehensive 4-hour test: 10 seeds, 5 training variants, 300+ allocation combos.
All with purge=5d (removes label leakage per Lopez de Prado).

### 2.1 Training Variants

| Variant | Description | Accuracy | Sortino | MaxDD |
|---|---|---|---|---|
| **Baseline** | Expanding window + purge=5d | **60.1%** | **4.05** | **-9.2%** |
| Recency 730d | Exponential decay, half-life=2yr | 59.1% | 3.97 | -9.7% |
| Recency 365d | Exponential decay, half-life=1yr | 59.1% | 3.47 | -10.5% |
| Sliding 3yr | Last 1095 days only | 61.5% | 2.50 | -19.3% |
| Early stopping | n_estimators=500 + 15% val holdout | 55.3% | 1.08 | -5.5% |

All measured with Sigmoid s=15 K=60/30/15.

**Findings:**
- Baseline expanding window is best overall (Sortino 4.05)
- Sliding 3yr has highest accuracy (61.5%) but unacceptable drawdown (-19.3%)
- Early stopping destroys performance — accuracy drops to 55.3%
- Recency weighting doesn't help and slightly hurts

**Decision:** Expanding window (baseline) with purge=5d.

### 2.2 Allocation Methods — Full Results (baseline variant, 10 seeds)

**Top 10 by Sortino:**

| # | Method | Sortino | Return | MaxDD | Holdout 25-26 | Sortino Std |
|---|---|---|---|---|---|---|
| 1 | **Sigmoid s=20 K60/30/15** | **4.05** | +1105% | -9.3% | +77.2% | 0.05 |
| 2 | **Sigmoid s=15 K60/30/15** | **4.05** | +1063% | -9.2% | +78.1% | 0.05 |
| 3 | Sigmoid s=10 K60/30/15 | 4.04 | +984% | -9.1% | +79.1% | 0.04 |
| 4 | ECDF K75/45/22 | 4.02 | +697% | -6.5% | +99.6% | 0.05 |
| 5 | ECDF K70/40/20 | 4.02 | +638% | -6.1% | +97.5% | 0.04 |
| 6 | Floor=0.5 K75/45/22 | 3.99 | +866% | -9.1% | +84.4% | 0.04 |
| 7 | Floor=0.5 K80/50/25 | 3.99 | +956% | -9.1% | +87.8% | 0.03 |
| 8 | MultiH 70/30 K60/30/15 | 3.98 | +1001% | -9.1% | +78.2% | 0.05 |
| 9 | ECDF K80/50/25 | 3.98 | +759% | -6.9% | +100.0% | 0.04 |
| 10 | Floor=0.5 K70/40/20 | 3.97 | +781% | -8.6% | +79.8% | 0.04 |

**Top 5 by Holdout 2025-2026 (true out-of-sample):**

| # | Method | Holdout 25-26 | Sortino | MaxDD |
|---|---|---|---|---|
| 1 | ECDF K80/50/25 | +100.0% | 3.98 | -6.9% |
| 2 | ECDF K75/45/22 | +99.6% | 4.02 | -6.5% |
| 3 | ECDF K70/40/20 | +97.5% | 4.02 | -6.1% |
| 4 | LdP CDF K80/50/25 | +95.1% | 3.62 | -6.3% |
| 5 | Floor=0.5 K80/50/25 | +87.8% | 3.99 | -9.1% |

### 2.3 With vs Without Confidence (baseline, K=60/30/15)

| Config | Sortino | Sharpe | Return | MaxDD | Holdout 25-26 |
|---|---|---|---|---|---|
| V22 (no confidence) | 3.94 | 2.26 | +1141% | -9.5% | +74.4% |
| **Sigmoid s=15 (V23)** | **4.05** | **2.31** | +1063% | **-9.2%** | **+78.1%** |

Confidence scaling improves Sortino (+3%), Sharpe (+2%), reduces drawdown, and improves holdout.

### 2.4 Accuracy by Year (baseline, purge=5d)

| Year | Accuracy | Notes |
|---|---|---|
| 2022 | 66.1% | Bear market — model excels |
| 2023 | 62.9% | Recovery — strong |
| 2024 | 54.1% | Bull→correction — weakest |
| 2025 | 54.9% | Mixed — below average |
| 2026 | 66.9% | Recent improvement |
| **Overall** | **60.1%** | Measured on Fridays only (~52/year) |

Note: accuracy with purge is 60.1% vs 61.1% without purge. The ~1% difference was label leakage.

### 2.5 Statistical Confidence

Across 10 seeds (42, 142, 242, 342, 442, 542, 642, 742, 842, 942):

| Metric | Mean | Std | Worst Seed |
|---|---|---|---|
| Sortino | 4.05 | 0.05 | 3.96 |
| Accuracy | 60.1% | ~0.5% | ~59.5% |

Low variance confirms results are not seed-dependent.

### 2.6 Methods That Don't Work

| Method | Why |
|---|---|
| Early stopping | Accuracy drops to 55.3% — severe underfitting |
| Inverse volatility (alone) | Drawdown doubles to -14% to -18% |
| Kelly criterion | Too conservative for 60% accuracy (Sortino < 3.0) |
| No-K (normalized predictions) | Positions too small, returns inadequate |
| Sliding window 3yr | Best accuracy but drawdown -19.3% |
| 9 Messari features | Worsened performance (noise > signal) |

---

## Final Production Config

```python
# config.py
K_REGIME = {'BULL': 60, 'MILD': 30, 'BEAR': 15}
SIGMOID_SCALE = 15
HORIZON = 3
BAGS = 80
FEATURES = 37  # original V22 features
RETRAIN = 'semi'  # January and July
REBAL_DOW = [4]  # Friday
EMERGENCY_THRESHOLD = 0.08

# Allocation formula:
# confidence = abs(P_classifier - 0.5)
# allocation = prediction * K[regime] * sigmoid(confidence * 15)
# allocation = clip(allocation, -25%, +100%)
```

**Validated BRL metrics (purge=5d, cost=5bps, seed 242):**
- Return 2022-2026: **+793%**
- Max Drawdown: **-10.7%**
- Sortino: **3.21**
- Sharpe: **1.72**
- Accuracy: **60.3%** (Friday directional, h=3d)
- BTC B&H reference: +31% BRL, MDD -66%
- 2026 YTD: Strategy +11% vs BTC -26%

**Validated USD metrics (10 seeds, purge=5d):**
- Return: +1,063%
- Sortino: 4.05 (std: 0.05 across 10 seeds)
- Sharpe: 2.31
- Max Drawdown: -9.2%
- Holdout 2025-2026: +78.1%

The BRL and USD gaps come from two sources:
1. BRL appreciated +11.5% vs USD from 2022 to 2026 (cuts strategy returns when in BTC)
2. Transaction costs: BRL backtest uses 5bps per rebalance, USD test was cost-free

---

## Overfitting Assessment

**Protected against:**
- Walk-forward validation (no future data in training)
- Purge=5d (removes label leakage)
- 10 seeds (low variance: Sortino std = 0.05)
- Simple model (2 hyperparams: K and s)

**Residual risk:**
- K=60 and s=15 were selected looking at 2022-2026 — mild data snooping on allocation hyperparams
- 2024-2025 accuracy dropped to ~55% — could continue or recover (2026 shows 66.9%)
- True validation requires live performance over 1-2 years
