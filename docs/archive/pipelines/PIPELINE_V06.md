# Pipeline V6 Comprehensive — Feature Optimization + Stability

**Generated:** 2026-02-12 18:36:21
**Test Period:** 2022-2026 OOS (walk-forward)
**Cost Model:** 2 bps (1.5x for shorts)

## Baselines

| Strategy | Return | Sortino | MaxDD |
|----------|--------|---------|-------|
| BTC Buy & Hold | +53.4% | 0.27 | -66.7% |
| V2 Baseline (50/50) | +221.2% | 0.87 | -20.9% |
| B4 Baseline (60/40) | +236.5% | 0.97 | -20.2% |
| B7 Baseline (100/0 ret_3d) | +284.9% | 1.10 | -21.5% |

## Part A: Feature Optimization

### A2: Individual Swaps

| Swap | Return | Delta | Verdict |
|------|--------|-------|---------|
| swap_puell_multiple_with_eth | +377.8% | +92.9pp | BETTER |
| swap_rsi_14d_with_eth | +362.3% | +77.4pp | BETTER |
| swap_rsi_14d_with_momentum_accel_7d | +355.7% | +70.8pp | BETTER |
| swap_rsi_14d_with_sharpe_3y_artemis | +331.5% | +46.6pp | BETTER |
| swap_rsi_14d_with_active_developers | +328.8% | +43.9pp | BETTER |
| swap_sopr_ma7_with_eth | +323.2% | +38.3pp | BETTER |
| swap_sopr_ma7_with_sharpe_3y_artemis | +314.6% | +29.7pp | BETTER |
| swap_sopr_ma7_with_active_developers | +312.1% | +27.1pp | BETTER |
| swap_funding_rate_with_eth | +309.9% | +25.0pp | BETTER |
| swap_puell_multiple_with_momentum_accel_7d | +308.5% | +23.6pp | BETTER |
| swap_funding_rate_with_sharpe_3y_artemis | +304.4% | +19.5pp | BETTER |
| swap_sopr_ma7_with_momentum_accel_7d | +303.0% | +18.1pp | BETTER |
| swap_funding_rate_with_active_developers | +301.2% | +16.3pp | BETTER |
| swap_puell_multiple_with_sharpe_3y_artemis | +297.5% | +12.6pp | BETTER |
| swap_funding_rate_with_momentum_accel_7d | +293.1% | +8.1pp | BETTER |
| swap_puell_multiple_with_active_developers | +291.7% | +6.8pp | BETTER |
| swap_rsi_14d_with_acceleration | +282.0% | -3.0pp | similar |
| swap_puell_multiple_with_acceleration | +273.5% | -11.4pp | worse |
| swap_funding_rate_with_acceleration | +254.1% | -30.8pp | worse |
| swap_sopr_ma7_with_acceleration | +241.4% | -43.5pp | worse |

### A3: Combo Swaps

| Combo | B7 Return | B4 Return |
|-------|-----------|-----------|
| top2_combo | +418.0% | +306.2% |
| top3_combo | +415.9% | +306.2% |
| all4_combo | +376.9% | +287.5% |
| conservative_bottom2 | +305.4% | +250.0% |

## Part B: V6 Tier 1 Ideas

### B1: Bag Size Stability

| Bags | Mean Return | Spread | Verdict |
|------|------------|--------|---------|
| Bag10 | +293.0% | 51pp | MODERATE |
| Bag20 | +289.3% | 41pp | MODERATE |
| Bag30 | +294.8% | 37pp | MODERATE |

### B2: Ensemble Weights

| Scheme | Return | Sortino | MaxDD |
|--------|--------|---------|-------|
| equal | +259.3% | 1.05 | -20.7% |
| favor_B7 | +273.0% | 1.09 | -21.0% |
| favor_B6 | +262.4% | 1.06 | -20.7% |
| B5_B6_only | +256.2% | 1.04 | -20.6% |
| B4_B7_extremes | +262.4% | 1.06 | -20.7% |
| inverse_variance | +257.1% | 1.05 | -20.6% |
| sharpe_weighted | +260.1% | 1.06 | -20.7% |
| bootstrap_optimal | +260.6% | 1.06 | -20.7% |

### B3: Wider Short Range

| Config | Return | MaxDD |
|--------|--------|-------|
| B7_clip[-0.5,1.0] | +284.9% | -21.5% |
| V2_clip[-0.5,1.0] | +203.4% | -25.4% |
| B7_clip[-0.75,1.0] | +284.9% | -21.5% |
| V2_clip[-0.75,1.0] | +209.1% | -25.3% |
| B7_clip[-1.0,1.0] | +284.9% | -21.5% |
| V2_clip[-1.0,1.0] | +223.7% | -25.3% |

### B4: Regime-Conditional ML Weight

| Regime | Return | Sortino | MaxDD |
|--------|--------|---------|-------|
| regime_vol_ratio | +259.0% | 1.05 | -20.5% |
| regime_hurst | +260.9% | 1.04 | -21.3% |
| regime_kpss | +219.0% | 0.88 | -19.8% |
| regime_combined | +275.6% | 1.10 | -21.1% |

## Winner

**A3_top2_combo**
- Config: `{'swap_dict': {'puell_multiple': 'eth', 'rsi_14d': 'eth'}, 'ml_weight': 1.0, 'n_bags': 5, 'target_horizon': 3}`
- Seed stability: UNSTABLE (mean=+405.9%, spread=80pp)
- Bootstrap 95% CI: [+77%, +1497%]
- P(return > 0): 99.9%

### Year-by-Year

| Year | Return | BTC | Excess |
|------|--------|-----|--------|
| 2022 | +35.0% | -65.3% | +100.4% |
| 2023 | +100.2% | +154.5% | -54.2% |
| 2024 | +68.8% | +111.8% | -43.0% |
| 2025 | +22.2% | -7.3% | +29.5% |
| 2026 | -7.1% | -11.4% | +4.3% |