# Next Steps V14 — Smart Allocation & New Approaches

**Date**: 2026-03-05
**Status**: V14 running (Tracks A-D done, E-F pending)

---

## Context

V13 achieved +1313% / S=3.22 on old data (Jan 31).
After CoinMetrics revised exchange flows (btc3.csv columns K,L,M,N,AA,AB), performance dropped to +912% / S=2.39.
Removing the broken `exchange_netflow_ma7` feature recovered to +1028% / S=2.49.
V14 found XGB pure Bag60 K=27 as best model: **+1041% / S=2.63 / 27pp spread**.

### The Real Problem

The model predicts 3-day raw return. Weekly accuracy is ~63%. But:

- **Correlation with actual returns is weak**: 0.40 (best year) to -0.09 (worst year)
- **In 2026 the model predicted UP every single week** during a -23% bear market
- **The allocation formula has bullish bias**: pred=0 gives 37.5% BTC allocation
- **No concept of risk**: +2% prediction during high vol is treated same as during low vol
- **No risk management**: if BTC falls 15% mid-week, we only react next Friday
- **MaxDD of -30%** is unacceptable for institutional use

### What V14/V15 Already Tested

| Pipeline | What | Best Result |
|----------|------|-------------|
| V14 Track A | Feature re-selection (drop exchange_netflow_ma7, screen 30 new) | base+3 feats: S=2.59, 50pp |
| V14 Track B | K re-sweep + allocation formulas + thresholds | K=27: S=2.55, 79pp |
| V14 Track C | Model architecture (hybrid ratios, pure XGB, pure LGB) | **XGB60: S=2.63, 27pp** |
| V14 Track D | Adaptive K + regularization | adaptive 17/27/27: S=2.63, 40pp |
| V15 Track 1 | Multi-timeframe signals (blend 3d+7d+14d) | 3d+7d: S=2.44, 62pp |
| V15 Track 2 | Stacking meta-learner | **FAILED** (S=1.36) |
| V15 Track 3 | Regime-aware allocation (K varies by SMA regime) | **bear20/neut30/bull35: S=2.55, 74pp** |
| V15 Track 4 | Ensemble weighting (median, trimmed mean) | median: S=2.52, 95pp |
| V15 Track 5 | Classification target (binary UP/DOWN) | **FAILED** (S=1.32, but 9pp spread) |

### Weekly Accuracy by Year (XGB60 K=27, seed=42)

| Year | Weeks | Correct | Wrong | Accuracy | Notes |
|------|-------|---------|-------|----------|-------|
| 2022 | 52 | 34 | 18 | 65% | Bear market — model went short, big alpha |
| 2023 | 52 | 36 | 16 | 69% | Bull market — best accuracy |
| 2024 | 52 | 30 | 22 | 58% | Mild bull — correlation dropped to -0.09 |
| 2025 | 52 | 32 | 20 | 62% | Choppy — decent |
| 2026 | 9 | 4 | 5 | 44% | Bear — predicted UP every week, trapped |

---

## Root Cause Analysis

### 1. The model predicts raw return, not risk-adjusted return

The model doesn't know that +2% return during 50% annualized volatility is noise, while +2% during 10% volatility is a strong signal. It treats both the same.

### 2. The allocation formula has bullish bias

```
alloc = clip(pred * K + 0.375, -0.25, 1.0)
```

When pred = 0 (the model has no opinion), allocation is already **37.5% BTC** — that's bullish! For the model to be neutral (0% BTC), it needs to predict **-1.4%**. For max short (-25%), it needs **-2.3%**. The asymmetry means any noise in the positive direction creates long exposure.

### 3. K amplifies tiny signals into extreme positions

Predictions range from -0.03 to +0.03 (tiny). K=27 amplifies: pred=+0.023 → 100% allocation. The model doesn't need to be "right" — it just needs to be slightly positive to go fully long. This works in bull markets but is catastrophic in bears.

### 4. No risk management within the week

We rebalance every Friday. If BTC crashes Monday-Thursday, we're fully exposed until Friday. The Jan 30 week saw BTC drop -13.2% while the model held 83% allocation.

### 5. The model can't detect regime changes

In 2026, `price_percentile_1y` was at 0th percentile (BTC at yearly low) and `basis_pct` was at -4.8% (extreme backwardation). These are screaming "bear market" but the model still predicted UP because it learned patterns from 2019-2025 that don't apply.

---

## Proposed Experiments

### Level 1: Change What the Model Learns

#### Experiment 1: Risk-Adjusted Target

**Instead of**: `target = (price[t+3] - price[t]) / price[t]`
**Use**: `target = return_3d / volatility_7d`

The model learns to predict risk-adjusted return (a Sharpe-like ratio). A +2% return during high volatility produces a small target value, teaching the model not to signal strongly. A +2% return during low volatility produces a large target value.

**Why this matters**: The model would have given weaker signals during the high-vol crash of Jan-Feb 2026, naturally reducing exposure.

**Estimated impact**: Could fix the MaxDD problem while maintaining Sortino.

#### Experiment 2: Two-Model Pipeline (Classifier + Regressor)

Train two models:
- **Model A (Classifier)**: Binary UP/DOWN, 63% accuracy
- **Model B (Regressor)**: Magnitude prediction

**Rule**: Only act when both agree. If classifier says DOWN but regressor says +0.01, stay flat.

**Why this matters**: The classifier has genuine edge (63% accuracy). Using it as a filter eliminates the weak positive predictions that trapped us in 2026. In 2026, even if the regressor said UP, the classifier might have caught some DOWN weeks.

#### Experiment 3: Quantile Regression

Instead of predicting mean return, predict the 10th, 50th, and 90th percentiles.

- **Spread** (p90 - p10) = natural uncertainty measure
- When spread is wide → reduce position (model is uncertain)
- When spread is narrow → increase position (model is confident)

**Why this matters**: We get confidence for free, without relying on ensemble std.

### Level 2: Change How We Size Positions

#### Experiment 4: Target Volatility (Risk Parity)

Standard institutional approach. Instead of fixed K:

```python
target_vol = 0.15  # 15% annualized
position = target_vol / (realized_vol * sqrt(365))
alloc = pred_sign * min(position, 1.0)
```

When BTC vol is 80% annualized (crisis), position = 0.19.
When BTC vol is 30% annualized (calm), position = 0.50.

**Why this matters**: Automatically reduces exposure during high-vol periods like Jan-Feb 2026. This is how most quant funds actually size positions.

#### Experiment 5: Kelly Criterion

Mathematically optimal position sizing:

```python
# Rolling 52-week stats
win_rate = rolling_accuracy  # ~0.63
avg_win = mean(returns when correct)
avg_loss = mean(abs(returns when wrong))
kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
alloc = kelly_fraction * pred_sign
```

**Why this matters**: If model accuracy drops (as it did in 2024/2026), Kelly automatically reduces position. It's self-correcting.

#### Experiment 6: Centered Allocation + Trend Filter

Two simple fixes combined:

1. **Center at 0**: `alloc = clip(pred * K, -0.25, 1.0)` — removes the bullish bias
2. **Trend filter**: If price < SMA200, cap long at 0%. If price < SMA50, cap at 30%.

**Why this matters**: In 2026, BTC dropped below SMA200 in February. The trend filter would have forced the strategy to 0% or short, avoiding most of the drawdown.

### Level 3: Risk Management

#### Experiment 7: Drawdown Budget

Set maximum acceptable drawdown (e.g., -15%). Track running drawdown and scale position:

```python
if running_drawdown < -0.10:
    scale = max(0.1, 1.0 - abs(running_drawdown) / 0.15)
    alloc *= scale
```

**Why this matters**: Hard caps the maximum loss. Once strategy hits -10%, it starts reducing. At -15%, position is minimal.

#### Experiment 8: Multi-Horizon Veto

Train models on 3d, 7d, and 14d targets. Voting rule:
- ALL say UP → go long (full signal)
- Mixed → stay flat (0% BTC)
- ALL say DOWN → go short

**Why this matters**: The 7d and 14d models would likely have detected the February 2026 downtrend even when the 3d model was noisy. The veto prevents acting on weak signals.

#### Experiment 9: Prediction Z-Score

Normalize predictions relative to recent history:

```python
pred_z = (pred - rolling_mean(pred, 20)) / (rolling_std(pred, 20) + 1e-8)
```

A prediction of +0.02 when recent predictions are all +0.02 = noise (z=0).
A prediction of +0.02 when recent predictions were -0.02 = strong signal (z=2).

**Why this matters**: In 2026, predictions were persistently +0.01 to +0.03. The z-score would have been near zero (no signal), keeping us flat.

#### Experiment 10: Bayesian Shrinkage

Shrink predictions toward zero based on model's historical accuracy:

```python
shrinkage = 1.0 - (1.0 / (1.0 + rolling_accuracy_zscore))
adjusted_pred = pred * shrinkage
```

When rolling accuracy is high → keep prediction as-is.
When rolling accuracy is low → shrink toward zero (don't trade).

**Why this matters**: In 2024 (accuracy 58%) and 2026 (accuracy 44%), predictions would be shrunk significantly, reducing exposure.

---

## Priority Ranking

| Priority | Experiment | Expected Impact | Complexity | Time |
|----------|-----------|----------------|------------|------|
| **1** | **#4 Target Volatility** | Fix MaxDD from -30% to ~-15% | Low | 1h |
| **2** | **#1 Risk-Adjusted Target** | Fundamentally better model | Medium | 2h |
| **3** | **#6 Centered + Trend Filter** | Fix bullish bias + bear protection | Low | 1h |
| **4** | **#8 Multi-Horizon Veto** | Only approach that prevents 2026 | Medium | 2h |
| **5** | **#5 Kelly Criterion** | Self-correcting position sizing | Low | 1h |
| **6** | **#7 Drawdown Budget** | Hard cap on losses | Low | 0.5h |
| **7** | **#9 Prediction Z-Score** | Normalize signal strength | Low | 0.5h |
| **8** | **#2 Two-Model Pipeline** | Classifier as filter | Medium | 2h |
| **9** | **#3 Quantile Regression** | Natural uncertainty measure | High | 3h |
| **10** | **#10 Bayesian Shrinkage** | Reduce noise in weak periods | Low | 0.5h |

**Recommended sequence**: Run experiments 4, 1, 6, 8, 5 first (~7h). If any shows clear improvement, combine with best from V14/V15 for final config.

---

## Success Criteria

A new config must improve on XGB60 K=27 baseline on **at least 2 of 4**:

| Metric | Current (XGB60 K=27) | Target |
|--------|---------------------|--------|
| Return | +1041% | >= +800% |
| Sortino | 2.63 | >= 2.50 |
| Spread | 27pp | <= 50pp |
| **MaxDD** | **-30.3%** | **<= -20%** |

The key unlock is **MaxDD**. Going from -30% to -15-20% while maintaining Sortino > 2.5 would be a major improvement.

---

## Files

| File | Description |
|------|-------------|
| `scripts/optimization/pipeline_v14_reoptimize.py` | V14 re-optimization (running) |
| `scripts/optimization/pipeline_v15_new_approaches.py` | V15 new approaches (complete) |
| `scripts/optimization/v14_smart_sizing.py` | Smart sizing tests (draft, needs update) |
| `outputs/results/pipeline_v14_reoptimize.json` | V14 results (partial) |
| `outputs/results/pipeline_v15_new_approaches.json` | V15 results (complete) |
