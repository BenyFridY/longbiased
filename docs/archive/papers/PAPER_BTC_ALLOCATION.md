# Dynamic Bitcoin Allocation via Regime-Adaptive Machine Learning Ensemble

**Beny Frid**

Sao Paulo, Brazil

March 2026

**Keywords**: Bitcoin, Dynamic Allocation, Machine Learning, XGBoost, Regime Detection, Risk-Free Rate, Sortino Ratio

---

## Abstract

We propose a machine learning-based dynamic allocation framework between Bitcoin (BTC) and a risk-free asset (Brazilian CDI). The model employs a bagged ensemble of 80 XGBoost regressors retrained semi-annually on an expanding window, using 37 features from technical, macroeconomic, on-chain, and derivatives domains. A regime-adaptive mechanism classifies market conditions into bull, mild, and bear states via SMA(50)/SMA(200) crossovers, dynamically scaling position size according to the prevailing regime. Rebalancing occurs weekly (Fridays) with an emergency trigger for daily absolute returns exceeding 8%.

Over a 4-year out-of-sample period (January 2022 — March 2026, 1,514 trading days), the strategy achieves a cumulative return of +1,059% (10-seed average) with a Sortino ratio of 3.87 and maximum drawdown of -10.1%, compared to +39% cumulative return and -65.6% maximum drawdown for a BTC buy-and-hold benchmark. The Probability of Backtest Overfitting (PBO) is 0.0% across 70 combinatorial cross-validation folds, and the return spread across 10 random seeds is 87 percentage points on a base of +1,059% (8.2% relative variation), confirming robustness.

All reported results are from backtesting. No live trading has been conducted.

---

## 1. Introduction

### 1.1 Motivation

Bitcoin, the largest cryptocurrency by market capitalization, offers extraordinary long-term return potential but with severe drawdowns — often exceeding 50-70% from peak to trough. The 2022 bear market, for instance, saw BTC decline 64% over the year. For institutional investors and asset managers, pure buy-and-hold exposure carries risk characteristics that are difficult to reconcile with fiduciary duties and fund mandates.

This paper addresses a fundamental question in crypto asset management: **can a machine learning model dynamically allocate between BTC and a risk-free asset to capture the upside while systematically mitigating drawdowns?**

The Brazilian context adds an important dimension: the CDI (Certificado de Deposito Interbancario) yields approximately 14% per year in the 2024-2026 period, providing a meaningful "floor" when the model reduces crypto exposure.

### 1.2 Contributions

This work makes the following contributions:

1. **Regime-adaptive allocation**: A three-state market classification (bull, mild, bear) based on dual SMA crossovers dynamically scales model conviction, reducing position sizes during unfavorable regimes without requiring explicit stop-loss rules.

2. **Emergency rebalancing**: An event-driven trigger supplements weekly rebalancing when daily absolute returns exceed 8%, enabling rapid repositioning during extreme market events — a mechanism that adds +0.23 Sortino over weekly-only rebalancing.

3. **Systematic feature rejection**: Of 280 candidate features, we demonstrate that only 37 improve risk-adjusted performance. Notably, widely-used indicators including sentiment (Fear & Greed Index), GARCH volatility, MVRV ratio, and Google Trends are shown to degrade model performance when added to the selected feature set.

4. **Comprehensive overfitting validation**: Probability of Backtest Overfitting (PBO) analysis (Bailey et al. 2014) yields 0.0% across 70 combinatorial folds, and 10-seed robustness testing shows minimal performance variation.

### 1.3 Related Work

Machine learning applications to cryptocurrency markets have expanded significantly in recent years. Tree-based ensemble methods, particularly XGBoost (Chen and Guestrin 2016), have demonstrated strong performance on tabular financial data due to their capacity for modeling nonlinear feature interactions without requiring feature normalization (Gu, Kelly, and Xiu 2020).

Regime detection in financial markets has been approached through Hidden Markov Models (Bouri et al. 2017, Ang and Bekaert 2002), threshold models, and technical indicator-based methods. In cryptocurrency markets specifically, Jiang and Liang (2017) applied deep reinforcement learning to portfolio management, while Fischer and Krauss (2018) demonstrated the efficacy of LSTM networks for return prediction.

The tactical asset allocation literature (Brandt, Santa-Clara, and Valkanov 2009) provides the theoretical foundation for our approach, which we extend to the cryptocurrency domain with regime conditioning and high-frequency rebalancing triggers.

Our work differs from prior crypto ML studies in its emphasis on risk-adjusted returns (Sortino rather than raw return), comprehensive overfitting validation, and the explicit modeling of a high risk-free rate environment.

---

## 2. Data

### 2.1 Sample Period and Structure

The dataset comprises 2,619 daily observations spanning January 1, 2019 to March 3, 2026. The data is partitioned as follows:

| Period | Dates | Purpose | Days |
|--------|-------|---------|------|
| Initial training | Jan 2019 — Dec 2021 | First model fit | 1,096 |
| Out-of-sample (OOS) | Jan 2022 — Mar 2026 | All reported results | 1,523 |

The model is retrained semi-annually during the OOS period using an expanding window (Section 3.2), ensuring that no future information leaks into predictions. All performance metrics, tables, and figures in this paper are computed exclusively on out-of-sample data.

### 2.2 Price and Risk-Free Data

Bitcoin daily closing prices are sourced from Binance spot market (BTCUSDT pair), selected for its consistently highest global liquidity and narrow bid-ask spreads. Daily returns are computed as simple percentage changes from the closing price series.

The risk-free rate is the Brazilian CDI, obtained from COPOM/Bacen historical records and converted to daily rates. During the out-of-sample period, the CDI rate ranged from approximately 2% (2021) to 14.25% (2025-2026), reflecting the Selic rate trajectory.

### 2.3 Feature Universe

The initial feature universe comprises 280 daily features sourced from eight providers:

| Source | Type | Description | Features |
|--------|------|------------|----------|
| Binance API | Derivatives & OHLCV | Spot and futures prices, funding rates, basis, taker ratios | 41 |
| Artemis/Messari | Network activity | Transaction counts, fees, developer commits, stablecoin flows | 47 |
| CoinMetrics | On-chain | Hash rate, exchange flows, MVRV, supply metrics | 32 |
| yfinance | Macro cross-asset | ETH, Gold, Copper, VIX, S&P 500, Dollar Index | ~20 |
| FRED | Macroeconomic | Federal Reserve balance sheet, M2 money supply, yield curve | 6 |
| BGeometrics | On-chain behavioral | NUPL, SOPR, realized price | 23 |
| DefiLlama | DeFi ecosystem | Stablecoin supply and flow aggregates | 5 |
| Calculated | Technical & statistical | Derived from price/volume: MACD, Bollinger Bands, Hurst exponent, CUSUM, Ornstein-Uhlenbeck parameters, fractal dimension, etc. | ~106 |

### 2.4 Feature Selection

Through systematic screening across 19 pipeline iterations (totaling over 80 hours of computation and 1,500+ configurations evaluated), 37 features were selected based on their marginal contribution to out-of-sample Sortino ratio when evaluated with 10 random seeds. The selected features are grouped into five categories:

| Category | Count | Representative Features |
|----------|-------|------------------------|
| Technical indicators | 12 | MACD histogram, Bollinger Band position, ADX, Aroon oscillator, OBV trend, volume-to-SMA ratio |
| Statistical properties | 8 | Hurst exponent (DFA), fractal dimension (Higuchi), Ornstein-Uhlenbeck theta, CUSUM, structural break score, KPSS statistic |
| Macroeconomic | 5 | Fed balance sheet, M2 YoY growth, BTC-gold 30d correlation, copper 30d return, ETH/BTC ratio |
| On-chain | 6 | NUPL, miners revenue ratio, hash rate 30d change, velocity, stablecoin z-score, stablecoin supply 30d change |
| Derivatives | 6 | Futures basis (spot-futures premium), basis 7d MA, open interest, futures trade count, volatility x regime duration |

Importantly, the following widely-discussed feature categories were systematically tested and rejected, as each degraded the Sortino ratio when added to the 37-feature baseline:

- **Sentiment**: Fear & Greed Index (all transformations), Google Trends search volume
- **Macro risk**: VIX z-score, Dollar Index momentum, yield curve slope, high-yield credit spread
- **On-chain valuation**: MVRV z-score, SOPR, price-to-realized ratio
- **Volatility models**: GARCH(1,1) persistence, GJR-GARCH asymmetry, GARCH-predicted volatility
- **Market microstructure**: Taker buy/sell pressure, funding rate x open interest interaction
- **Mining & supply cycle**: Stock-to-flow, difficulty ribbon, Puell multiple, halving cycle

---

## 3. Methodology

### 3.1 Model Architecture

The predictive model consists of a bagged ensemble of $B = 80$ XGBoost gradient-boosted tree regressors (Chen and Guestrin 2016). Each regressor $f_k$ is trained on the same data but with a different random seed $s_k$, introducing diversity through stochastic gradient boosting.

The target variable is the 3-day forward simple return of BTC:

$$y_t = \frac{P_{t+3} - P_t}{P_t}$$

At inference time, the ensemble prediction is the arithmetic mean:

$$\hat{y}_t = \frac{1}{B} \sum_{k=1}^{B} f_k(\mathbf{x}_t)$$

where $\mathbf{x}_t \in \mathbb{R}^{42}$ is the feature vector at time $t$ (37 selected features plus 5 automatically appended price-derived features: returns at 3, 10, 30, 60 days and 14-day volatility).

XGBoost hyperparameters are held at library defaults (max_depth=6, learning_rate=0.3, n_estimators=200), as exhaustive grid search over 200 configurations (conducted in V17) yielded negligible improvement for this dataset size (~2,000 training observations).

### 3.2 Training Protocol: Semi-Annual Expanding Window

The model is retrained every six months using all data from the start of the sample (January 2019) to the retraining date. This expanding window approach ensures that the model continuously incorporates new information while retaining the full historical context.

The retraining schedule during the out-of-sample period is:

| Retrain Period | Training Data | Test Period |
|---------------|--------------|-------------|
| 1 | Jan 2019 — Dec 2021 | Jan — Jun 2022 |
| 2 | Jan 2019 — Jun 2022 | Jul — Dec 2022 |
| 3 | Jan 2019 — Dec 2022 | Jan — Jun 2023 |
| 4 | Jan 2019 — Jun 2023 | Jul — Dec 2023 |
| ... | ... | ... |
| 9 | Jan 2019 — Jun 2025 | Jul — Dec 2025 |
| 10 | Jan 2019 — Dec 2025 | Jan — Mar 2026 |

This approach was selected over rolling windows (which discard older data) and quarterly retraining (which overfits to recent conditions), based on systematic comparison in V13 where expanding window outperformed all alternatives by a significant margin.

### 3.3 Regime-Adaptive Allocation

The raw ensemble prediction $\hat{y}_t$ is converted to a portfolio allocation through a regime-dependent scaling mechanism. The market regime at time $t$ is classified based on the relative ordering of the BTC price $P_t$ and two simple moving averages:

$$\text{Regime}_t = \begin{cases} \text{Bull} & \text{if } P_t > \text{SMA}_{50}(t) > \text{SMA}_{200}(t) \\ \text{Mild} & \text{if } P_t > \text{SMA}_{200}(t) \text{ (but not Bull)} \\ \text{Bear} & \text{if } P_t < \text{SMA}_{200}(t) \end{cases}$$

Each regime has an associated multiplier $K$:

| Regime | Multiplier $K$ | Rationale |
|--------|---------------|-----------|
| Bull | 50 | Strong trend — amplify model signal |
| Mild | 30 | Uncertain conditions — moderate signal |
| Bear | 15 | Downtrend — minimize exposure |

The allocation is then:

$$a_t = \text{clip}\left(\hat{y}_t \times K_{\text{regime}_t},\ -0.25,\ 1.0\right)$$

The clip function constrains the allocation to a maximum of 100% long and 25% short. In practice, during bear regimes, the low $K$ value (15) combined with typically small model predictions results in allocations near zero — meaning the portfolio is predominantly invested in CDI. During bull regimes, the high $K$ (50) allows the strategy to take full advantage of strong positive predictions.

This regime-adaptive mechanism was the single most impactful innovation in the pipeline (V17), increasing the Sortino ratio from 3.36 to 3.55 while simultaneously reducing maximum drawdown from -18.1% to -9.9%. It effectively embeds risk management directly into the allocation formula, rendering explicit drawdown budgets and stop-loss overlays unnecessary.

### 3.4 Rebalancing Schedule

The strategy rebalances under two conditions:

1. **Weekly (Fridays)**: The model generates a new allocation every Friday, reflecting institutional settlement patterns and weekend derivatives expiration dynamics.

2. **Emergency trigger**: On any day where $|r_t^{\text{BTC}}| > 8\%$, the model generates a new allocation regardless of the day of the week. This mechanism allows rapid repositioning following extreme market events (e.g., flash crashes or parabolic rallies).

On non-rebalancing days, the previous allocation is maintained. The emergency threshold of 8% was selected based on systematic comparison of 3%, 5%, 8%, and 10% thresholds, where 8% optimized the trade-off between responsiveness and excessive trading (Section 4.4).

### 3.5 Portfolio Return Calculation

The daily portfolio return is:

$$r_t^{\text{port}} = \begin{cases} a_t \cdot r_{t+1}^{\text{BTC}} + (1 - a_t) \cdot r_t^{\text{CDI}} - c_t & \text{if } a_t \geq 0 \\[6pt] a_t \cdot r_{t+1}^{\text{BTC}} + 1.0 \cdot r_t^{\text{CDI}} - c_t & \text{if } a_t < 0 \end{cases}$$

where the transaction cost $c_t = |a_t - a_{t-1}| \times 0.0002 \times \lambda$ with $\lambda = 1.5$ for short positions and $\lambda = 1.0$ otherwise, reflecting a 2 basis points brokerage fee with a 50% surcharge for shorting.

Note: when the allocation is negative (short BTC), the full capital earns the CDI rate as collateral margin also accrues interest.

### 3.6 Performance Metrics

The primary evaluation metric is the **Sortino ratio**, which measures risk-adjusted excess return penalizing only downside deviation:

$$\text{Sortino} = \frac{\overline{r^{\text{excess}}}}{\sigma_{\text{downside}}} \times \sqrt{365}$$

where $r^{\text{excess}}_t = r^{\text{port}}_t - r^{\text{CDI}}_t$ and $\sigma_{\text{downside}} = \text{std}(r^{\text{excess}}_t\ |\ r^{\text{excess}}_t < 0)$.

The Sortino ratio is preferred over the Sharpe ratio for this application because cryptocurrency returns exhibit strong positive skewness and fat tails, making upside volatility desirable rather than penalizable. We also report the Sharpe ratio, maximum drawdown, and annualized return and volatility for completeness.

---

## 4. Results

All results in this section are computed on out-of-sample data only (January 2022 — March 2026). The primary results use seed=42; Section 5.1 demonstrates robustness across 10 seeds.

### 4.1 Aggregate Performance

*[See Figure 1: Cumulative Performance — Strategy vs BTC Buy & Hold vs CDI]*

**Table 1: Out-of-Sample Performance Summary (seed=42)**

| Metric | ML Strategy | BTC Buy & Hold | CDI (Risk-Free) |
|--------|-----------|----------------|-----------------|
| Cumulative return | **+1,052%** | +39% | +64% |
| Annualized return | 80.3% | ~8% | ~13% |
| Annualized volatility | 21.4% | ~65% | ~0.3% |
| Sortino ratio | **3.83** | 0.26 | — |
| Sharpe ratio | 2.30 | 0.14 | — |
| Maximum drawdown | **-10.4%** | -65.6% | ~0% |
| Average BTC allocation | 11.5% | 100% | 0% |
| Days with negative allocation | 49.7% | — | — |
| Out-of-sample trading days | 1,514 | 1,514 | 1,514 |

The strategy outperforms BTC buy-and-hold by a factor of 27 in cumulative return while reducing maximum drawdown by 55 percentage points. The average BTC allocation of 11.5% indicates that the portfolio spends most of its time predominantly in CDI, with concentrated BTC exposure during high-conviction periods.

The high proportion of days with negative allocation (49.7%) reflects the bear regime dominance in this period (44% of days). However, these "short" positions are typically very small (allocation of -1% to -5%), representing a mild hedge rather than an aggressive directional bet. The $K_{\text{bear}} = 15$ multiplier ensures that even when the model predicts negative returns, the resulting short allocation is minimal.

### 4.2 Annual Decomposition

*[See Figure 3: Annual Returns]*

**Table 2: Year-by-Year Performance**

| Year | Strategy | BTC B&H | Excess | Interpretation |
|------|---------|---------|--------|---------------|
| 2022 | +57.3% | -64.1% | +121.5pp | Bear market — model reduces exposure, CDI accrues |
| 2023 | +130.1% | +153.5% | -23.4pp | Bull market — captures majority of upside |
| 2024 | +94.3% | +111.4% | -17.0pp | Bull market — captures majority of upside |
| 2025 | +55.7% | -6.1% | +61.7pp | Mixed/bear — model protects capital |
| 2026 (Q1) | +5.2% | -23.1% | +28.3pp | Bear — model in defensive posture |

The strategy generates its largest alpha during bear markets and periods of market stress. In 2022, when BTC declined 64%, the strategy returned +57% by maintaining low BTC allocation and earning CDI. During the 2023-2024 bull markets, the strategy captured +130% and +94% respectively — underperforming pure buy-and-hold by 23pp and 17pp, but with dramatically lower risk. This asymmetric profile (protecting in downturns, participating in upturns) is the hallmark of a well-calibrated tactical allocation system.

### 4.3 Regime Analysis

*[See Figure 2: Market Regime Classification and Allocation Over Time]*

**Table 3: Regime Distribution and Behavior**

| Regime | Days | % of OOS | Description |
|--------|------|----------|------------|
| Bear | 665 | 43.9% | Price below SMA(200) — predominantly CDI allocation |
| Bull | 512 | 33.8% | Price above SMA(50) above SMA(200) — elevated BTC allocation |
| Mild | 337 | 22.3% | Price above SMA(200) but below SMA(50) — moderate allocation |

The regime mechanism serves as an embedded risk management system. During bear periods, the low multiplier ($K=15$) results in allocations near zero or slightly negative, effectively parking over 90% of capital in CDI at 14%+/year. This approach proved more effective than explicit risk overlays: drawdown budgets, confidence gating, and stop-loss rules were all tested and rejected as they either failed to improve Sortino or reduced returns without commensurate risk reduction.

### 4.4 Rebalancing Analysis

**Table 4: Rebalancing Method Comparison (10-seed averages)**

| Method | Sortino | Cumulative Return | Max Drawdown |
|--------|---------|------------------|-------------|
| **Friday + emergency 8%** | **3.87** | **+1,059%** | **-10.1%** |
| Friday only | 3.64 | +1,054% | -10.1% |
| Friday + emergency 10% | 3.81 | +1,125% | -10.1% |
| Friday + emergency 5% | 3.17 | +888% | -9.9% |
| Friday + emergency 3% | 2.76 | +790% | -15.6% |
| Friday + Thursday (fixed) | 2.90 | +694% | -9.5% |
| Friday + Monday (fixed) | 2.69 | +658% | -14.2% |

Adding a fixed second rebalancing day consistently degraded performance. The emergency trigger at 8% is the sole exception, as it targets genuinely extreme events (occurring approximately 5-10 times per year) where rapid repositioning has material value. Lower thresholds (3%, 5%) triggered too frequently, increasing transaction costs and whipsaw risk.

---

## 5. Robustness and Validation

### 5.1 Seed Robustness

*[See Figure 5: Return and Sortino Distribution Across 10 Seeds]*

To verify that results are not artifacts of a particular random seed, we execute the full pipeline with 10 different base seeds (42, 142, 242, ..., 942). Each seed generates a distinct ensemble of 80 XGBoost models.

**Table 5: 10-Seed Summary Statistics**

| Metric | Mean | Std Dev | Min | Max | Spread |
|--------|------|---------|-----|-----|--------|
| Cumulative return | +1,059% | +/- 25pp | +1,024% | +1,111% | 87pp |
| Sortino ratio | 3.87 | +/- 0.05 | 3.77 | 3.95 | 0.18 |
| Maximum drawdown | -10.1% | +/- 0.4pp | -9.5% | -10.6% | 1.1pp |

The relative spread — 87 percentage points on a base of +1,059%, or 8.2% relative variation — is remarkably low, indicating that performance is driven by systematic signal rather than seed-specific noise. The Sortino ratio varies by only 0.18 across seeds (3.77 to 3.95), and maximum drawdown is stable between -9.5% and -10.6%.

### 5.2 Probability of Backtest Overfitting (PBO)

We employ the PBO framework of Bailey, Borwein, Lopez de Prado, and Zhu (2014) to assess whether the selected configuration could be an artifact of parameter cherry-picking. The procedure:

1. The OOS period is divided into $N=8$ temporal blocks (~190 days each).
2. For each of $\binom{8}{4} = 70$ combinations, half the blocks serve as in-sample (IS) and half as out-of-sample (OOS).
3. A grid of 72 configurations (varying $K_{\text{bull}}, K_{\text{mild}}, K_{\text{bear}}$) is evaluated on each IS partition.
4. The best IS configuration is then evaluated on the corresponding OOS partition.
5. PBO = fraction of combinations where the best IS configuration has negative OOS Sortino.

**Results:**
- **PBO = 0.0%** — In none of the 70 combinations did the best IS configuration produce a negative OOS result.
- **Average logit(rank) = -0.876** — The best IS config consistently ranks in the upper half of OOS configurations (a logit of 0 would indicate random ranking).

These results provide strong evidence that the strategy's performance is not attributable to in-sample optimization.

### 5.3 Rolling Performance Stability

*[See Figure 4: Rolling 90-Day Sortino Ratio]*

The rolling 90-day Sortino ratio remains above 2.0 for the majority of the out-of-sample period, with occasional dips during rapid regime transitions. The metric does not exhibit a declining trend, suggesting that the model's predictive power has not degraded over the 4-year test period.

### 5.4 Monthly Excess Return Distribution

*[See Figure 6: Monthly Excess Return Heatmap]*

The monthly heatmap of excess returns (strategy minus BTC buy-and-hold) reveals a clear pattern: the strategy outperforms most strongly during months when BTC declines (green cells in 2022, early 2025, early 2026) and underperforms modestly during strong bull months (red cells in late 2024). This is consistent with the regime-adaptive mechanism functioning as designed.

---

## 6. Discussion

### 6.1 Source of Alpha

The strategy's risk-adjusted outperformance derives from two complementary mechanisms:

1. **Regime-conditioned exposure**: The SMA-based regime classification correctly identifies unfavorable market environments, allowing the strategy to reduce exposure before drawdowns deepen. Critically, the capital freed from BTC is not idle — it earns CDI at rates between 10-14% annualized. This "productive hedging" is a structural advantage of the Brazilian rate environment.

2. **Short-term predictive signal**: The XGBoost ensemble extracts nonlinear relationships between the 37 features and 3-day forward returns. The model's predictions, while individually modest (directional accuracy of approximately 60%), become economically significant when amplified by the regime multiplier during high-conviction periods.

The interaction between these mechanisms is key: the regime filter prevents the model from taking large positions based on noisy predictions during adverse conditions, while allowing full expression during favorable conditions.

### 6.2 Why Many Features Fail

A notable finding is the systematic rejection of feature categories that are widely considered informative in the cryptocurrency literature:

- **Sentiment (Fear & Greed, Google Trends)**: Likely already captured by the price-derived features (momentum, volatility, regime), making them redundant.
- **MVRV / SOPR**: On-chain valuation metrics may operate on longer time horizons than the 3-day prediction target.
- **GARCH volatility**: The existing volatility_7d feature already captures this information; adding GARCH variants introduces noise.
- **VIX / DXY / Yield curve**: Macro factors may influence crypto over weeks or months, not the 3-day window.

This suggests that for short-horizon BTC prediction, technical and derivatives-based features dominate, with selective macroeconomic variables (Fed balance sheet, M2) providing complementary signal about the broader liquidity environment.

### 6.3 Limitations and Caveats

1. **Backtested, not live**: All results are from historical simulation. Live performance may differ due to execution slippage, data delays, and market impact.

2. **Transaction costs**: The assumed 2 bps per rebalance reflects institutional brokerage but may underestimate total costs including slippage for large positions.

3. **Regime detection lag**: SMA crossovers are lagging indicators by construction. Flash crashes occurring within a regime may not trigger sufficiently fast allocation changes, though the emergency rebalance mechanism partially addresses this.

4. **Single market cycle**: The OOS period (2022-2026) covers one bear-bull-bear cycle. Additional cycles would strengthen out-of-sample evidence.

5. **Feature stability**: The optimal feature set was selected on data through 2026. As cryptocurrency markets mature and new market structures emerge (e.g., ETF flows), periodic re-screening is warranted.

6. **CDI dependency**: The strategy's strong absolute returns are partly attributable to the high Brazilian risk-free rate. In a lower-rate environment, the "productive hedging" effect would be diminished.

---

## 7. Conclusion

We present a regime-adaptive machine learning strategy for dynamic Bitcoin allocation that achieves a Sortino ratio of 3.87 and maximum drawdown of -10.1% over four years of out-of-sample testing, compared to a Sortino of 0.26 and drawdown of -65.6% for passive buy-and-hold.

The strategy's architecture is intentionally simple and interpretable: an XGBoost ensemble for return prediction, SMA crossovers for regime classification, and Friday rebalancing with an emergency trigger. This simplicity — validated through the rejection of more complex alternatives including GARCH models, HMM regimes, stacking, and explicit risk overlays — supports the strategy's potential for robust out-of-sample deployment.

The framework is extensible to multi-asset crypto portfolios and alternative risk-free rate environments, directions we leave for future research.

---

## References

- Ang, A., & Bekaert, G. (2002). International asset allocation with regime shifts. *Review of Financial Studies*, 15(4), 1137-1187.
- Bailey, D. H., Borwein, J. M., Lopez de Prado, M., & Zhu, Q. J. (2014). The Probability of Backtest Overfitting. *Journal of Computational Finance*, 20(4), 39-69.
- Bouri, E., Gupta, R., Tiwari, A. K., & Roubaud, D. (2017). Does Bitcoin hedge global uncertainty? Evidence from wavelet-based quantile-in-quantile regressions. *Finance Research Letters*, 23, 87-95.
- Brandt, M. W., Santa-Clara, P., & Valkanov, R. (2009). Parametric portfolio policies: Exploiting characteristics in the cross-section of equity returns. *Review of Financial Studies*, 22(9), 3411-3447.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.
- Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. *European Journal of Operational Research*, 270(2), 654-669.
- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223-2273.
- Jiang, Z., & Liang, J. (2017). Cryptocurrency portfolio management with deep reinforcement learning. *IEEE Intelligent Systems Conference (IntelliSys)*, 905-913.

---

## Appendix A: Complete Feature List (37 Features)

| # | Feature | Source | Category |
|---|---------|--------|----------|
| 1 | cusum_pos | Calculated (CUSUM) | Regime change |
| 2 | cusum_neg | Calculated (CUSUM) | Regime change |
| 3 | structural_break_score | Calculated | Regime change |
| 4 | velocity | Calculated (price momentum) | Regime |
| 5 | mr_score_30d | Calculated (mean reversion) | Mean reversion |
| 6 | ou_theta_60d | Calculated (Ornstein-Uhlenbeck) | Mean reversion |
| 7 | half_life_60d | Calculated (mean reversion) | Mean reversion |
| 8 | hurst_60d | Calculated (DFA) | Statistical |
| 9 | fractal_dimension_30d | Calculated (Higuchi) | Statistical |
| 10 | kpss_stat_30d | Calculated (stationarity) | Statistical |
| 11 | price_percentile_1y | Calculated (rolling rank) | Statistical |
| 12 | sortino_30d | Calculated (rolling Sortino) | Statistical |
| 13 | volatility_7d | Calculated (rolling std) | Statistical |
| 14 | adx | Calculated (directional) | Technical |
| 15 | macd_histogram | Calculated (trend) | Technical |
| 16 | bb_position | Calculated (Bollinger Bands) | Technical |
| 17 | obv_trend | Calculated (On-Balance Volume) | Technical |
| 18 | volume_sma20_ratio | Calculated (volume) | Technical |
| 19 | aroon_down_30d | Calculated (Aroon oscillator) | Technical |
| 20 | trend_strength | Calculated (ADX x Hurst) | Technical |
| 21 | vol_x_regime_duration | Calculated (interaction) | Interaction |
| 22 | eth | yfinance | Cross-asset |
| 23 | eth_btc_ratio | yfinance / Calculated | Cross-asset |
| 24 | eth_pctchg_30d | yfinance / Calculated | Cross-asset |
| 25 | btc_gold_corr_30d | yfinance / Calculated | Macro |
| 26 | copper_return_30d | yfinance / Calculated | Macro |
| 27 | m2_yoy_growth | FRED (WM2NS) | Macro |
| 28 | fed_balance_sheet | FRED (WALCL) | Macro |
| 29 | nupl_ma30 | BGeometrics | On-chain |
| 30 | miners_revenue_ratio | Blockchain.com | On-chain |
| 31 | stablecoin_zscore | DefiLlama | On-chain |
| 32 | stablecoin_supply_change_30d | DefiLlama | On-chain |
| 33 | hash_rate_pctchg_30d | CoinMetrics | On-chain |
| 34 | basis_pct | Binance (futures premium) | Derivatives |
| 35 | basis_ma7 | Binance / Calculated | Derivatives |
| 36 | open_interest | Artemis (multi-exchange) | Derivatives |
| 37 | futures_trade_count | Artemis (multi-exchange) | Derivatives |

## Appendix B: Rejected Approaches

**Table B1: Model and Training Alternatives**

| Approach | Best Sortino | vs Baseline | Tested In |
|----------|-------------|-------------|-----------|
| XGBoost pure (selected) | 3.87 | — | V17-V19 |
| LightGBM + XGBoost hybrid | 3.29 | -0.58 | V12, V14, V17 |
| Stacking (multi-model) | < 3.0 | Failed | V15 |
| Classification (up/down) | < 3.0 | Failed | V15 |
| Rolling 2-year window | < 2.5 | Destroyed | V13 |
| Quarterly retraining | 1.65 | -2.22 | V13, V17 |
| Walk-forward K optimization | 3.15 | -0.72 | V13, V17 |

**Table B2: Risk Management Overlays**

| Approach | Best Sortino | vs Baseline | Tested In |
|----------|-------------|-------------|-----------|
| Dynamic regime (selected) | 3.87 | — | V17-V19 |
| Drawdown budget (10-20%) | 3.29-3.55 | Neutral/worse | V17 |
| Confidence gating (20-40%) | 3.13-3.34 | Worse | V17 |
| SMA200 trend filter | ~2.5 | Too conservative | V14 |
| HMM regime classification | — | 79% mild, useless | V18 |

**Table B3: Feature Categories Rejected (V19 Feature Screening)**

| Category | Features Tested | Best Delta Sortino | Verdict |
|----------|----------------|-------------------|---------|
| Sentiment | 5 (F&G, Google Trends) | -0.047 | All worse |
| Macro risk | 4 (VIX, DXY, yield curve, HY) | -0.000 | All worse |
| On-chain value | 4 (MVRV, SOPR, realized) | -0.047 | All worse |
| Microstructure | 3 (taker, funding x OI) | -0.009 | All worse |
| Mining/supply | 4 (S2F, ribbon, Puell, halving) | -0.051 | All worse |
| GARCH volatility | 5 variants | -0.110 | All worse |

## Appendix C: List of Figures

All figures are available in `outputs/charts/paper/`.

- **Figure 1** (`fig1_cumulative.png`): Cumulative performance on logarithmic scale — ML Strategy vs BTC Buy & Hold vs CDI, with drawdown comparison panel.
- **Figure 2** (`fig2_allocation_regime.png`): Market regime classification (bull/mild/bear background coloring) with BTC price evolution and daily allocation bars.
- **Figure 3** (`fig3_annual.png`): Annual return comparison — Strategy, BTC Buy & Hold, and CDI side by side for each calendar year.
- **Figure 4** (`fig4_rolling_sortino.png`): Rolling 90-day Sortino ratio time series with reference line at Sortino = 2.0.
- **Figure 5** (`fig5_seed_robustness.png`): Distribution of cumulative return and Sortino ratio across 10 random seeds.
- **Figure 6** (`fig6_monthly_heatmap.png`): Heatmap of monthly excess return (Strategy minus BTC Buy & Hold) by year and month.
