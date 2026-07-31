# Systematic Bitcoin Allocation Strategy Using Machine Learning

## Overview

Developed a quantitative trading strategy that dynamically allocates capital between Bitcoin and Brazilian CDI (risk-free rate) using machine learning models. The system predicts short-term BTC returns and translates those predictions into optimal allocation weights, rebalanced weekly.

The strategy was built from scratch over 10 iterative versions, culminating in a system that achieved **+800% cumulative return** (2022-2026 out-of-sample) with a **Sortino ratio of 2.73** and **maximum drawdown of -14%** — significantly outperforming both buy-and-hold BTC and a static 60/40 portfolio.

## Technical Stack

- **Language**: Python
- **ML Framework**: LightGBM (gradient boosted trees)
- **Core Libraries**: pandas, NumPy, scipy
- **Data**: On-chain metrics, macro indicators, cross-asset correlations, technical indicators
- **Backtest**: Custom walk-forward engine with realistic transaction costs

## Methodology

### Feature Engineering (37 features)
Designed and tested 100+ candidate features across five categories:
- **On-chain**: NUPL, SOPR, miner revenue, exchange netflows, stablecoin flows
- **Technical**: ADX, MACD, Bollinger Bands, CUSUM change-point detection, Hurst exponent
- **Macro**: M2 supply growth, copper returns, yield curve, VIX
- **Cross-asset**: BTC-Gold correlation, ETH/BTC ratio, ETH price momentum
- **Statistical**: Ornstein-Uhlenbeck mean-reversion parameters, fractal dimension, KPSS stationarity

Identified the single most impactful feature — a rolling 1-year price percentile — through systematic single-swap experiments across 196 configurations.

### Model Architecture
- **Ensemble of 30 LightGBM regressors** (bagged with different random seeds)
- **Walk-forward validation**: retrain annually, test on unseen data (2022-2026)
- **Target**: 3-day forward BTC return
- **Allocation formula**: Linear mapping from ensemble prediction to portfolio weight, with short capability (up to -25% BTC)

### Optimization Process (10 Versions)
Each version addressed a specific hypothesis through controlled experiments:

| Version | Configs Tested | Key Discovery |
|---------|---------------|---------------|
| V1-V2 | 50+ | Price-derived features dominate; bagging is essential |
| V5-V6 | 80+ | Feature swaps add +93pp; ETH features are additive |
| V8 | 30+ | Passed 14/16 overfitting audit criteria |
| V9 | 460+ | Optimal feature count = 37; cross-asset correlations work |
| V10 | 500+ | Allocation formula is the biggest lever; hyperparams already optimal |

Total: **~5,000 individual backtests** across all versions.

### Rigorous Validation
Every candidate strategy passed a 7-test validation suite:
- **10-seed stability**: tested with 10 different random seeds to measure variance
- **Permutation test**: 1,000 allocation shuffles to confirm skill (p < 0.001)
- **Combinatorial purged cross-validation (CPCV)**: 20 train/test paths with purge gaps
- **Bootstrap confidence intervals**: P(loss) < 1% over 1,000 block-bootstrap samples
- **Year-by-year excess**: positive alpha in 4/5 years
- **Insurance ratio**: gains in crash months exceed losses in bull months

### Key Findings
1. **Allocation formula matters more than model tuning** — changing the prediction-to-weight mapping added +218pp return while 74 hyperparameter configs found only 1 marginal improvement
2. **Accuracy does not predict profitability** — all models achieve ~60% directional accuracy regardless of configuration; Sortino depends on getting the big moves right
3. **More ensemble members reduce variance** — increasing from 30 to 50 bagged models cut seed spread from 71pp to 50pp with no loss in performance
4. **Emergency rebalancing hurts** — all 12 tested configurations (bidirectional, down-only, up-only at various thresholds) degraded Sortino

## Results

### Final Strategy Performance (Out-of-Sample, 2022-2026)

| Metric | Strategy | BTC Buy & Hold | 60/40 Benchmark |
|--------|----------|---------------|-----------------|
| Cumulative Return | +800% | +192% | +168% |
| Sortino Ratio | 2.73 | — | 0.84 |
| Maximum Drawdown | -14% | -65% | -40% |
| Seed Spread | 50pp | — | — |

### Performance by Year

| Year | Strategy | BTC | Excess |
|------|----------|-----|--------|
| 2022 (bear) | +32% | -65% | +97pp |
| 2023 | +95% | +156% | -61pp |
| 2024 | +139% | +121% | +18pp |
| 2025 | +168% | +54% | +114pp |

The strategy's key strength is **crash protection**: during the 2022 bear market, it returned +32% while BTC fell -65%, demonstrating the model's ability to go defensive ahead of drawdowns.

## Skills Demonstrated

- **Quantitative Research**: systematic hypothesis testing across 5,000+ backtests
- **Machine Learning**: feature engineering, ensemble methods, walk-forward validation, overfitting prevention
- **Financial Engineering**: portfolio construction, risk metrics (Sortino, Calmar, MaxDD), transaction cost modeling
- **Statistical Rigor**: permutation tests, bootstrap inference, CPCV, lookahead bias auditing
- **Software Engineering**: modular pipeline design, incremental result persistence, reproducible experiments
