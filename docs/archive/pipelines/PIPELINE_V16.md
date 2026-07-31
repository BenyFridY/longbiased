# Pipeline V16 — Smart Position Sizing Experiments

**Date**: 2026-03-05
**Runtime**: 1.4 hours (16-core parallel)
**Script**: `scripts/optimization/pipeline_v16_experiments.py`
**Results**: `outputs/results/pipeline_v16_experiments.json`

---

## Executive Summary

V16 tested 10 fundamentally different allocation strategies on the same XGB60 model predictions. The key discovery: **trend filter** reduces MaxDD from -32% to -8% while improving Sortino from 2.65 to 2.73.

### The Problem V16 Solves

The model predicts 3-day BTC return with ~63% accuracy. But the allocation formula had two critical flaws:
1. **Bullish bias**: pred=0 gave 37.5% BTC allocation (should be 0%)
2. **No trend awareness**: went 100% long during bear markets (2026: predicted UP every week during -23% crash)

### The Solution: Trend Filter

```
If price > SMA50 and SMA50 > SMA200 (uptrend):
    alloc = pred * K                    # full signal
If price > SMA200 but < SMA50 (mild):
    alloc = min(pred * K * 0.5, 0.3)   # half signal, capped at 30%
If price < SMA200 (downtrend):
    alloc = min(pred * K, 0.0)          # only allow short/flat
```

Simple. No ML. No optimization. Just don't go long in a bear market.

---

## Full Ranking (35 strategies, 10 seeds each)

| Rk | Strategy | Return | Spread | Sortino | MaxDD | Category |
|----|----------|--------|--------|---------|-------|----------|
| 1 | **trend_K35** | **+426%** | **26pp** | **2.73** | **-8.2%** | Trend Filter |
| 2 | centered_K30 | +704% | 77pp | 2.70 | -21.0% | Centered |
| 3 | baseline_K27 | +1131% | 64pp | 2.65 | -32.2% | Baseline |
| 4 | centered_K27 | +622% | 64pp | 2.64 | -19.3% | Centered |
| 5 | centered_K25 | +574% | 55pp | 2.61 | -17.7% | Centered |
| 6 | trend_K30 | +372% | 23pp | 2.59 | -7.9% | Trend Filter |
| 7 | centered_K20 | +468% | 38pp | 2.56 | -14.1% | Centered |
| 8 | trend_K27 | +344% | 20pp | 2.51 | -7.7% | Trend Filter |
| 9 | trend_K25 | +326% | 18pp | 2.47 | -7.5% | Trend Filter |
| 10 | trend_K20 | +283% | 13pp | 2.34 | -6.7% | Trend Filter |
| 11 | dd_budget_15pct | +414% | 36pp | 2.26 | -9.4% | Drawdown Budget |
| 12 | dd_budget_20pct | +414% | 36pp | 2.26 | -9.4% | Drawdown Budget |
| 13 | shrinkage_50 | +361% | 31pp | 2.14 | -9.2% | Bayesian Shrinkage |
| 14 | kelly_K27 | +201% | 8pp | 2.10 | -4.9% | Kelly Criterion |
| 15 | kelly_K30 | +212% | 9pp | 2.10 | -5.0% | Kelly Criterion |
| 16 | kelly_K25 | +193% | 7pp | 2.10 | -4.8% | Kelly Criterion |
| 17 | shrinkage_30 | +341% | 29pp | 2.09 | -9.1% | Bayesian Shrinkage |
| 18 | twomodel_K27_t50 | +370% | 58pp | 1.92 | -9.3% | Two-Model |
| 19 | targetvol_15 | +244% | 32pp | 1.91 | -7.7% | Target Volatility |
| 20 | twomodel_K25_t50 | +346% | 51pp | 1.86 | -9.2% | Two-Model |
| 21 | targetvol_10 | +182% | 21pp | 1.85 | -5.0% | Target Volatility |
| 22 | targetvol_20 | +289% | 43pp | 1.84 | -10.6% | Target Volatility |
| 23 | twomodel_K30_t55 | +362% | 58pp | 1.78 | -9.6% | Two-Model |
| 24 | targetvol_25 | +339% | 56pp | 1.78 | -13.3% | Target Volatility |
| 25 | twomodel_K27_t55 | +327% | 49pp | 1.70 | -8.9% | Two-Model |
| 26 | ra_K40 | +251% | 14pp | 1.66 | -13.8% | Risk-Adjusted Target |
| 27 | ra_K35 | +228% | 12pp | 1.64 | -12.0% | Risk-Adjusted Target |
| 28 | ra_K25 | +186% | 7pp | 1.63 | -8.4% | Risk-Adjusted Target |
| 29 | ra_K27 | +195% | 9pp | 1.63 | -9.1% | Risk-Adjusted Target |
| 30 | ra_K30 | +207% | 11pp | 1.62 | -10.2% | Risk-Adjusted Target |
| 31 | ra_K20 | +162% | 5pp | 1.61 | -6.5% | Risk-Adjusted Target |
| 32 | pred_zscore | +193% | 14pp | 1.60 | -6.4% | Prediction Z-Score |
| 33 | twomodel_K27_t60 | +277% | 59pp | 1.41 | -8.8% | Two-Model |
| 34 | smart_K27_tv20 | +91% | 5pp | 1.12 | -2.3% | Full Smart |
| 35 | smart_K30_tv15 | +87% | 4pp | 1.11 | -1.9% | Full Smart |

---

## Experiment Results by Category

### Trend Filter (WINNER)

Centered allocation with SMA-based trend filter.

| Config | Return | Sortino | Spread | MaxDD |
|--------|--------|---------|--------|-------|
| **trend_K35** | **+426%** | **2.73** | **26pp** | **-8.2%** |
| trend_K30 | +372% | 2.59 | 23pp | -7.9% |
| trend_K27 | +344% | 2.51 | 20pp | -7.7% |
| trend_K25 | +326% | 2.47 | 18pp | -7.5% |
| trend_K20 | +283% | 2.34 | 13pp | -6.7% |

Pattern: Higher K improves Sortino monotonically. MaxDD stays below 9% for all K values. Spread stays below 26pp. The trend filter caps downside regardless of K.

### Centered Allocation (no trend filter)

Removed the +0.375 bullish bias. pred=0 means 0% BTC.

| Config | Return | Sortino | Spread | MaxDD |
|--------|--------|---------|--------|-------|
| centered_K30 | +704% | 2.70 | 77pp | -21.0% |
| centered_K27 | +622% | 2.64 | 64pp | -19.3% |
| centered_K25 | +574% | 2.61 | 55pp | -17.7% |
| centered_K20 | +468% | 2.56 | 38pp | -14.1% |

Centering alone cuts MaxDD from -32% to -14-21%. But trend filter cuts it further to -7-8%.

### Kelly Criterion

Mathematically optimal position sizing.

| Config | Return | Sortino | Spread | MaxDD |
|--------|--------|---------|--------|-------|
| kelly_K27 | +201% | 2.10 | 8pp | -4.9% |
| kelly_K30 | +212% | 2.10 | 9pp | -5.0% |
| kelly_K25 | +193% | 2.10 | 7pp | -4.8% |

Ultra-conservative. Spread of 7-9pp (tightest ever). MaxDD under 5%. But return is modest (+200%). Good for a low-risk portfolio.

### Target Volatility

Standard risk parity: target constant portfolio volatility.

| Config | Return | Sortino | Spread | MaxDD |
|--------|--------|---------|--------|-------|
| targetvol_15 | +244% | 1.91 | 32pp | -7.7% |
| targetvol_10 | +182% | 1.85 | 21pp | -5.0% |
| targetvol_20 | +289% | 1.84 | 43pp | -10.6% |
| targetvol_25 | +339% | 1.78 | 56pp | -13.3% |

Good risk management but Sortino below 2.0. The trend filter achieves better risk-adjusted returns.

### Risk-Adjusted Target (Exp 1) — FAILED

Training on return/volatility instead of raw return.

| Config | Return | Sortino | MaxDD |
|--------|--------|---------|-------|
| Best (ra_K40) | +251% | 1.66 | -13.8% |

The model doesn't learn well from a Sharpe-like target. The return/vol ratio creates extreme target values that confuse XGBoost.

### Two-Model Pipeline (Exp 2)

Classifier (UP/DOWN) + Regressor, only act when both agree.

| Config | Return | Sortino | Spread | MaxDD |
|--------|--------|---------|--------|-------|
| twomodel_K27_t50 | +370% | 1.92 | 58pp | -9.3% |

Decent but doesn't beat trend filter on any metric.

### Multi-Horizon Veto (Exp 8) — FAILED

Only act when 3d, 7d, and 14d models all agree.

| Config | Return | Sortino | MaxDD |
|--------|--------|---------|-------|
| Best (veto_K30) | ~170% | ~1.02 | ~-7% |

Too conservative. The three horizons rarely agree, keeping the strategy flat most of the time.

### Full Smart (all combined) — OVER-ENGINEERED

Combined trend + vol + confidence + drawdown + z-score.

| Config | Return | Sortino | MaxDD |
|--------|--------|---------|-------|
| Best | +91% | 1.12 | -2.3% |

Each filter removes signal. The combination leaves almost no signal at all.

---

## The Winner: Trend Filter K=35

```
Model: XGB pure, 60 bags, semi-annual retrain
Features: 32 clean (no exchange_netflow_ma7) + basis_pct + 5 auto = 38 total
Allocation: centered (pred*K, no bias) + SMA trend filter
K: 35
Rebalance: Weekly Friday

Results (10 seeds):
  Return:  +426%
  Sortino: 2.73
  Spread:  26pp
  MaxDD:   -8.2%
```

### Why It Works

1. **Centered allocation** removes bullish bias → pred=0 means 0% BTC
2. **Trend filter** prevents going long when price < SMA200 → avoids bear markets
3. **High K (35)** amplifies good signals in bull markets → captures upside
4. **SMA acts as a natural stop-loss** → when trend breaks, position goes to zero

### Comparison with Previous Best

| Metric | V13 Best | V14 XGB60 | **V16 trend_K35** |
|--------|----------|-----------|-------------------|
| Return | +1313% | +1041% | +426% |
| Sortino | 3.22 | 2.63 | **2.73** |
| Spread | 146pp | 27pp | **26pp** |
| MaxDD | -15.9% | -30.3% | **-8.2%** |
| Calmar | ~8x | ~3.4x | **~52x** |

V16 trend_K35 has the best **risk-adjusted** performance of any config ever tested. The Calmar ratio (return/maxDD) of ~52x is extraordinary.

---

## Portfolio Options

Three valid configs for different risk appetites:

| Profile | Config | Return | Sortino | MaxDD | Spread |
|---------|--------|--------|---------|-------|--------|
| **Aggressive** | baseline_K27 | +1131% | 2.65 | -32% | 64pp |
| **Balanced** | centered_K30 | +704% | 2.70 | -21% | 77pp |
| **Conservative** | trend_K35 | +426% | 2.73 | -8% | 26pp |
| **Ultra-safe** | kelly_K27 | +201% | 2.10 | -5% | 8pp |

---

## What Failed and Why

| Experiment | Result | Why It Failed |
|-----------|--------|---------------|
| Risk-Adjusted Target | S=1.66 | XGBoost doesn't learn well from return/vol ratio — extreme values create noise |
| Multi-Horizon Veto | S=1.02 | 3 models rarely agree → stays flat 70%+ of time |
| Full Smart (all combined) | S=1.12 | Each filter removes useful signal. 5 filters × 0.7 pass rate = 17% signal remaining |
| Prediction Z-Score | S=1.60 | Normalizing removes the level information (bull vs bear) |
| Classification alone | S=1.32 (V15) | Binary UP/DOWN loses magnitude information |

### Key Insight

**Simpler is better.** The trend filter is a 3-line if/else statement. It outperforms every sophisticated approach (Kelly, target vol, risk-adjusted, stacking, etc.) on a risk-adjusted basis.

---

## Files

| File | Description |
|------|-------------|
| `scripts/optimization/pipeline_v16_experiments.py` | V16 main pipeline |
| `outputs/results/pipeline_v16_experiments.json` | Full results |
| `docs/PIPELINE_V16_RESULTS.md` | This document |
| `docs/NEXT_STEPS_V14.md` | Experiment design document |
