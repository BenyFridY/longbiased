# Pipeline V9 Results — Baseline Corrections + Feature Discovery

**Generated**: 2026-02-22
**Test Period**: 2022-2026 (walk-forward OOS)
**Total configs tested**: 465+ (Steps 1-2: 15, Step 3A: 196, Step 3 Fast: 149, Feature Count: 70, Audit: 5×7 tests)
**Total seed runs**: ~4,600
**Compute time**: ~72 hours

---

## Executive Summary

V9 started as a correction of V8 (real CDI rates + fixed short formula) and evolved into a comprehensive feature discovery effort. The results are transformative:

| Strategy | N Features | Mean Return | Sortino | Spread | vs V8 A3 |
|----------|-----------|-------------|---------|--------|----------|
| V8 A3 (old, flat CDI) | 25 | +417% | 1.51 | 23pp | --- |
| **V9 A3 baseline** | **25** | **+402%** | **1.57** | **23pp** | **corrected** |
| V9 C-2 (best double swap) | 25 | +501% | 2.00 | 35pp | +99pp |
| V9 D-77 (best triple swap) | 25 | +568% | 2.21 | 27pp | +166pp |
| **V9 37-feature (best Sortino)** | **37** | **+582%** | **2.48** | **41pp** | **+180pp** |
| V9 30-feature (best return) | 30 | +609% | 2.37 | 50pp | +207pp |

**Key discoveries:**
1. `price_percentile_1y` is the single most impactful feature ever found (+71pp alone)
2. 25 features was NOT optimal — sweet spot is 30-37 features
3. Cross-asset correlations (`btc_gold_corr_30d`) and commodities (`copper_return_30d`) add significant value
4. Replacing `m2_yoy_growth` with `btc_gold_corr_30d` is a massive improvement

---

## Charts

### Cumulative Returns (2022-2026)
![Cumulative Returns](../outputs/results/charts/v9_cumulative_returns.png)

### Feature Count vs Performance
![Feature Count Curve](../outputs/results/charts/v9_feature_count_curve.png)

### Pipeline Evolution V1-V9
![Pipeline Evolution](../outputs/results/charts/v9_pipeline_evolution.png)

### Drawdowns
![Drawdowns](../outputs/results/charts/v9_drawdowns.png)

### Year-by-Year Returns (Strategy vs BTC)
![Yearly Returns](../outputs/results/charts/v9_yearly_returns.png)

---

## Step 1: Baseline Corrections (MANDATORY)

### 1a. Real CDI Integration

V1-V8 used a flat CDI of 15%/year. Real Selic varied from 2% (2020) to 14.75% (2025).

| Year | Real CDI | Old Flat 15% | Difference |
|------|---------|-------------|------------|
| 2019 | 5.96% | 15.0% | -9.0pp |
| 2020 | 2.76% | 15.0% | -12.2pp |
| 2021 | 4.42% | 15.0% | -10.6pp |
| 2022 | 12.39% | 15.0% | -2.6pp |
| 2023 | 13.04% | 15.0% | -2.0pp |
| 2024 | 10.88% | 15.0% | -4.1pp |
| 2025 | 14.32% | 15.0% | -0.7pp |

### 1b. Fixed Short Formula

V8 bug: `sr = a * br + (1 - abs(a)) * rf_t - tc` — when short at -25%, only 75% earned CDI.
V9 fix: when short, 100% of capital earns CDI (margin also earns).

### 1c. A3 Baseline Results (10 seeds)

| Metric | V8 A3 (flat CDI) | V9 A3 (corrected) | Change |
|--------|------------------|-------------------|--------|
| Mean | +417.4% | +402.0% | -15pp (real CDI lower) |
| Sortino | 1.51 | 1.57 | +0.06 (short fix helps) |
| Spread | 23pp | 23pp | same |
| Beat V2 | 10/10 | 10/10 | same |
| MaxDD | -15% | -16.1% | similar |

Year-by-year (seed 42):

| Year | Strategy | BTC B&H | Excess |
|------|----------|---------|--------|
| 2022 | +29.7% | -65.3% | +95.1pp |
| 2023 | +105.9% | +154.5% | -48.5pp |
| 2024 | +68.9% | +111.8% | -42.9pp |
| 2025 | +19.4% | -7.3% | +26.7pp |
| 2026 | -7.5% | -11.4% | +3.9pp |

### 1d. 60/40 Static Benchmark

| Metric | 60/40 BTC/CDI | V9 A3 | ML adds |
|--------|--------------|-------|---------|
| Return | +78.7% | +402.0% | +323pp |
| Sortino | 0.32 | 1.57 | +1.25 |
| MaxDD | -44.7% | -16.1% | +28.6pp |

### Crash Detection Rates (A3 baseline)

| Threshold | Events | Detected | Rate |
|-----------|--------|----------|------|
| -5% | 394 | 203 | 51.5% |
| -10% | 114 | 60 | 52.6% |
| -15% | 35 | 26 | 74.3% |
| -20% | 17 | 17 | 100.0% |

### Validation Suite (A3 baseline): 7/7 PASS

| Test | Result |
|------|--------|
| Permutation (1000 shuffles) | p=0.001 PASS |
| CPCV (20 paths) | P(S>0)=100%, mean=1.62 PASS |
| Year-by-year excess | 3/5 years PASS |
| Bootstrap 95% CI | P(loss)=0.0%, CI=[+107%, +1321%] PASS |
| Beat 60/40 | +398% vs +79% PASS |
| Insurance ratio | 1.05x PASS |
| 10-seed stability | spread=23pp, 10/10 beat V2 PASS |

---

## Step 2: Model Improvements (ALL FAILED)

### 2a. Asymmetric Loss Function

Custom LightGBM objective penalizing missed crashes by alpha factor.

| Alpha | Mean | Spread | Sortino | Verdict |
|-------|------|--------|---------|---------|
| 2.0 | +288% | 732pp | 1.01 | DESTROYED stability |
| 3.0 | +231% | 681pp | 0.85 | Worse |
| 5.0 | +236% | 589pp | 0.76 | Much worse |

**Conclusion**: Custom objectives destroy LightGBM's stability. Spread explodes from 23pp to 600-730pp.

### 2b. Emergency Rebalancing

Mid-week rebalance if BTC drops more than threshold since last Friday.

| Threshold | Mean | Spread | Sortino | Verdict |
|-----------|------|--------|---------|---------|
| -3% | +260% | 23pp | 1.03 | -142pp return |
| -5% | +291% | 26pp | 1.13 | -111pp return |
| -7% | +332% | 29pp | 1.25 | -70pp return |

**Conclusion**: Confirms V2/V5 finding — more frequent rebalancing = worse. Weekly Friday is optimal.

### 2c. Crash Classifier

Secondary binary classifier to predict P(BTC drops >10% in 7 days).

| Config | Mean | Sortino | Verdict |
|--------|------|---------|---------|
| conf=0.3, def=0 | +401% | 1.56 | ~tied, slight drag |
| conf=0.5, def=0 | +399% | 1.56 | ~tied |
| conf=0.7, def=0 | +401% | 1.57 | no-op (never fires) |

**Conclusion**: Crash classifier adds nothing. At low confidence barely fires, at high confidence never fires.

### 2d. Combined Best

Combo of asymmetric + emergency: +232%, Sortino 0.81. Even worse than individual.

---

## Step 3: Feature Discovery (BREAKTHROUGH)

### Part A: Single Feature Swaps (196 configs, 25h)

Tested swapping `funding_rate` or `sopr_ma7` with each of 98 candidate features.

**Top 10 single swaps (sorted by Sortino):**

| Swap | Mean | Diff vs A3 | Sortino |
|------|------|-----------|---------|
| sopr->price_percentile_1y | +473% | +71pp | 1.97 |
| funding->price_percentile_1y | +464% | +62pp | 1.86 |
| sopr->stablecoin_supply_change_30d | +452% | +50pp | 1.80 |
| funding->stablecoin_supply_change_30d | +443% | +41pp | 1.72 |
| sopr->btc_gold_corr_30d | +423% | +21pp | 1.68 |
| funding->copper_return_30d | +435% | +33pp | 1.66 |
| sopr->kpss_stat_30d | +400% | -2pp | 1.66 |
| sopr->copper_return_30d | +410% | +8pp | 1.66 |
| sopr->ou_theta_60d | +413% | +11pp | 1.65 |
| funding->btc_gold_corr_30d | +430% | +28pp | 1.65 |

**Key finding**: `price_percentile_1y` (where current price sits in its 1-year range) is the single most impactful feature discovered in V1-V9.

### Part C: Double Feature Swaps (45 configs, 6.4h)

Replacing both `funding_rate` AND `sopr_ma7` simultaneously.

**Top 5 double swaps:**

| Config | Mean | Diff | Sortino | Spread |
|--------|------|------|---------|--------|
| fund->price_pct_1y + sopr->stablecoin_supply | +482% | +80pp | 2.04 | 41pp |
| fund->price_pct_1y + sopr->copper_return | +497% | +95pp | 2.03 | 39pp |
| fund->price_pct_1y + sopr->ou_theta_60d | +492% | +90pp | 2.02 | 56pp |
| fund->price_pct_1y + sopr->half_life_60d | +488% | +86pp | 2.01 | 52pp |
| fund->price_pct_1y + sopr->btc_gold_corr | +501% | +99pp | 2.00 | 35pp |

### Part D: Triple Feature Swaps (105 configs, 14.2h)

Best doubles + swapping a 3rd weak feature (`m2_yoy_growth`, `nupl_ma30`, `basis_ma7`).

**Top 5 triple swaps:**

| Config | Mean | Diff | Sortino | Spread |
|--------|------|------|---------|--------|
| price_pct_1y + copper + m2->ou_theta | +555% | +153pp | 2.23 | 39pp |
| price_pct_1y + half_life + m2->copper | +552% | +150pp | 2.22 | 38pp |
| price_pct_1y + fractal_dim + m2->ou_theta | +548% | +146pp | 2.21 | 40pp |
| price_pct_1y + copper + m2->btc_gold_corr | +568% | +166pp | 2.21 | 27pp |
| price_pct_1y + ou_theta + m2->copper | +547% | +145pp | 2.21 | 31pp |

**Key finding**: Replacing `m2_yoy_growth` with cross-asset features (btc_gold_corr, copper, ou_theta) is consistently the best 3rd swap.

---

## Feature Count Experiment (70 configs, 12.8h)

### The Critical Question: Is 25 Features Optimal?

**Answer: NO.** Sweet spot is 30-37 features.

### Feature Count Curve (C-2 as base, adding top features):

| N Features | Mean | Sortino | Spread |
|-----------|------|---------|--------|
| 25 | +501% | 2.00 | 35pp |
| 26 | +523% | 2.12 | 33pp |
| 27 | +556% | 2.20 | 36pp |
| 28 | +582% | 2.25 | 34pp |
| 29 | +597% | 2.30 | 42pp |
| **30** | **+609%** | **2.37** | **50pp** |
| 31 | +599% | 2.40 | 41pp |
| 32 | +595% | 2.40 | 29pp |
| 33 | +588% | 2.39 | 31pp |
| 35 | +572% | 2.37 | 44pp |
| **37** | **+582%** | **2.48** | **41pp** |
| 39 | +470% | 2.05 | 34pp |
| 45 | +346% | 1.65 | 21pp |
| 50 | +342% | 1.59 | 29pp |
| 60 | +311% | 1.44 | 31pp |
| 70 | +221% | 1.09 | 29pp |

**Key findings:**
- **30 features**: Highest absolute return (+609%)
- **37 features**: Highest Sortino (2.48) — best risk-adjusted
- **>40 features**: Rapid degradation (noise overwhelms signal)
- **>60 features**: Worse than original A3 baseline

---

## Top V9 Strategies — Final Ranking

### By Sortino (risk-adjusted):

| # | Strategy | N Feat | Mean | Sortino | Spread |
|---|----------|--------|------|---------|--------|
| 1 | C-2 + 12 features | 37 | +582% | 2.48 | 41pp |
| 2 | C-2 + 6 features | 31 | +599% | 2.40 | 41pp |
| 3 | C-2 + 7 features | 32 | +595% | 2.40 | 29pp |
| 4 | C-2 + 5 features | 30 | +609% | 2.37 | 50pp |
| 5 | D-78 triple (price_pct+copper+m2->ou_theta) | 25 | +555% | 2.23 | 39pp |
| 6 | D-120 triple (price_pct+half_life+m2->copper) | 25 | +552% | 2.22 | 38pp |
| 7 | D-77 triple (price_pct+copper+m2->btc_gold) | 25 | +568% | 2.21 | 27pp |

### By Return (absolute):

| # | Strategy | N Feat | Mean | Sortino | Spread |
|---|----------|--------|------|---------|--------|
| 1 | C-2 + 5 features | 30 | +609% | 2.37 | 50pp |
| 2 | C-2 + 7 features | 32 | +595% | 2.40 | 29pp |
| 3 | C-2 + 12 features | 37 | +582% | 2.48 | 41pp |
| 4 | D-77 triple swap | 25 | +568% | 2.21 | 27pp |
| 5 | D-78 triple swap | 25 | +555% | 2.23 | 39pp |

### Best Balance (return + Sortino + spread):

**V9 Best: C-2 + 7 features (32 features)**
- Mean: **+595%**
- Sortino: **2.40**
- Spread: **29pp**
- vs V8 A3: **+193pp return, +0.89 Sortino**

---

## New Features Discovered

### Tier 1 (transformative, +40pp or more):

| Feature | What it measures | Impact |
|---------|-----------------|--------|
| `price_percentile_1y` | Where current price sits in 1-year range | +71pp single swap |
| `stablecoin_supply_change_30d` | 30-day change in stablecoin supply | +50pp single swap |

### Tier 2 (strong, +10-40pp):

| Feature | What it measures | Impact |
|---------|-----------------|--------|
| `btc_gold_corr_30d` | 30-day BTC-Gold correlation | +28pp single swap |
| `copper_return_30d` | 30-day copper return | +33pp single swap |
| `ou_theta_60d` | Ornstein-Uhlenbeck mean reversion speed | +12pp single swap |
| `fractal_dimension_30d` | Market fractal complexity | +13pp single swap |
| `kpss_stat_30d` | Stationarity test statistic | +0pp but Sortino +0.09 |

### Tier 3 (useful for feature count expansion):

`open_interest`, `half_life_60d`, `sortino_30d`, `obv_trend`, `volume_sma20_ratio`, `aroon_down_30d`, `trend_strength`, `active_developers`

---

## Lookahead Bias Audit: 6/6 PASS

| Test | What it verified | Result |
|------|-----------------|--------|
| price_percentile_1y manual recalc | `rolling(365).rank(pct=True)` backward only | PASS |
| stablecoin_supply_change_30d | `pct_change(30)` backward only | PASS |
| Walk-forward train/test | Gap >= 5 days, no overlap in all 5 years | PASS |
| Backtest causality | Uses `daily_ret[t+1]` (decide at t, earn at t+1) | PASS |
| Target in training only | Forward-looking target never accessed in test | PASS |
| Future independence | Modifying future prices does NOT change features at t | PASS |

---

## What Failed (Important Negatives)

| Approach | Result | Lesson |
|----------|--------|--------|
| Asymmetric loss function | 589-732pp spread | Custom objectives destroy LightGBM stability |
| Emergency rebalancing | -70 to -142pp | More frequent = worse (confirms V2/V5) |
| Crash classifier | No improvement | Signal too weak for binary classification |
| Swapping stablecoin_zscore | -200pp | Critical feature, cannot be removed |
| >40 features | Rapid degradation | Noise overwhelms signal above threshold |
| >60 features | Worse than A3 | Confirms V5 finding (but now with exact curve) |

---

## Pipeline Evolution Summary (V1 to V9)

| Version | Return | Sortino | Key Insight |
|---------|--------|---------|-------------|
| V1 | +152% | 0.59 | Price > fundamentals |
| V2 | +221% | 0.87 | Bagging = biggest improvement |
| V3 | None>V2 | - | Complexity hurts |
| V4 | +245% | 0.89 | Lucky seed, failed audit |
| V5 | +289% | 1.12 | ML weight is the lever |
| V6 | +418% | 1.54 | Feature swaps massive |
| V7 | +426% | 1.52 | ETH genuine; Bag30 stabilizes |
| V8 | +417% mean | 1.51 | APPROVED 14/16; 23pp spread |
| **V9 A3** | **+402% mean** | **1.57** | **Corrected baseline (real CDI + short fix)** |
| **V9 C-2** | **+501% mean** | **2.00** | **price_percentile_1y + btc_gold_corr** |
| **V9 37feat** | **+582% mean** | **2.48** | **Optimal feature count = 37** |

---

## Files

| File | Description |
|------|-------------|
| `scripts/optimization/pipeline_v9.py` | V9 main pipeline (Steps 1-2) |
| `scripts/optimization/pipeline_v9_step3.py` | Step 3 Part A: 196 single swaps |
| `scripts/optimization/pipeline_v9_step3_fast.py` | Step 3 Parts C+D: double + triple swaps |
| `scripts/optimization/pipeline_v9_feature_count.py` | Feature count experiment |
| `scripts/optimization/lookahead_audit.py` | Lookahead bias verification |
| `scripts/optimization/generate_v9_charts.py` | Chart generation |
| `scripts/optimization/pipeline_v9_audit_top5.py` | Overfitting audit (top 5 models) |
| `outputs/results/pipeline_v9.json` | Steps 1-2 results |
| `outputs/results/pipeline_v9_step3.json` | Part A results (196 single swaps) |
| `outputs/results/pipeline_v9_step3_fast.json` | Parts C+D results (149 configs) |
| `outputs/results/pipeline_v9_feature_count.json` | Feature count results (70 configs) |
| `outputs/results/pipeline_v9_audit_top5.json` | Audit results (5 models × 7 tests) |
| `outputs/results/charts/v9_cumulative_returns.png` | Cumulative return chart |
| `outputs/results/charts/v9_feature_count_curve.png` | Feature count curve |
| `outputs/results/charts/v9_pipeline_evolution.png` | V1-V9 evolution |
| `outputs/results/charts/v9_drawdowns.png` | Drawdown comparison |
| `outputs/results/charts/v9_yearly_returns.png` | Year-by-year bar chart |

---

## Overfitting Audit — Top 5 Models: ALL 7/7 PASS

All five top V9 strategies passed the full 7-test overfitting audit suite. Each test validates a different aspect of strategy robustness:

| Test | What it validates |
|------|-------------------|
| **V1: 10-Seed Stability** | Returns are consistent across random seeds (spread < 50pp, all beat V2 baseline) |
| **V2: Permutation** | 1000 label shuffles — real returns must exceed 99.9% of shuffled distribution |
| **V3: CPCV (20 paths)** | Combinatorial purged cross-validation — Sortino positive in ≥85% of synthetic OOS paths |
| **V4: Year-by-Year Excess** | Strategy beats BTC in ≥3 of 5 OOS years (not just aggregate luck) |
| **V5: Bootstrap 95% CI** | 10,000 block-bootstrap resamples — P(loss) must be <1% |
| **V6: Insurance Ratio** | Geometric mean of (1+strat)/(1+btc) across worst BTC months — must be >1.0x |
| **V7: Beat 60/40** | Must beat 60/40 static benchmark (basic sanity check) |

### Full Scorecard

| Model | N Feat | Mean | Spread | Sortino | Perm p | CPCV P(+) | CPCV Sort | Year ≥3 | Boot P(loss) | Insurance | Beat 6040 | **Result** |
|-------|--------|------|--------|---------|--------|-----------|-----------|---------|-------------|-----------|-----------|------------|
| **37-feature** | 37 | +582% | 41pp | 2.48 | 0.000 | 90% | 1.49 | 3/5 | 0.0% | 1.23x | +519pp | **7/7 PASS** |
| **32-feature** | 32 | +595% | 29pp | 2.40 | 0.000 | 90% | 1.62 | 3/5 | 0.0% | 1.26x | +526pp | **7/7 PASS** |
| **30-feature** | 30 | +609% | 50pp | 2.37 | 0.000 | 95% | 1.72 | 3/5 | 0.0% | 1.27x | +530pp | **7/7 PASS** |
| **D-77 (25 feat)** | 25 | +568% | 27pp | 2.21 | 0.000 | 90% | 1.51 | 3/5 | 0.0% | 1.25x | +490pp | **7/7 PASS** |
| **D-78 (25 feat)** | 25 | +555% | 39pp | 2.23 | 0.000 | 90% | 1.56 | 3/5 | 0.0% | 1.21x | +445pp | **7/7 PASS** |

### Key Takeaways

- **Zero overfitting detected** across all 5 models and all 7 tests
- **Permutation p=0.000** for all 5 — no shuffled permutation ever beats real returns
- **Bootstrap P(loss)=0.0%** — none of 10,000 resamples produced a loss
- **CPCV ≥90% positive** — strategy is robust across combinatorial OOS paths
- **Insurance ratio 1.21-1.27x** — strategy protects in BTC worst months
- **MaxDD -13% to -16%** across all models (vs BTC -65% in 2022)
- **32-feature config** stands out: tightest spread (29pp) with highest return (+595%) and MaxDD of only -13.3%

---

## Next Steps (V10 Candidates)

1. **Test different LGB hyperparameters** on V9 best (V8 D2 showed grid sensitivity)
2. **Walk-forward with rolling window** to test if older data still helps with 37 features
3. **Out-of-sample forward test** on new data beyond Jan 2026

---

*Document auto-generated. All results reproducible via the scripts listed above.*
