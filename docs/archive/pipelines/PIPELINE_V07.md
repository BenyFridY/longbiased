# Pipeline V7 Validation — Feature Swap Robustness Check

**Generated**: 2026-02-12 19:45
**Test Period**: 2022-2026 (walk-forward OOS)
**Baseline**: B7 Pure ML + ret_3d + Bag5 + Weekly = +284.9%

---

## Executive Summary

### Is ETH genuine?
- **D1** (add eth): +357.0% (+72.1pp vs baseline)
- **D2** (remove 4 weak): +332.3% (+47.4pp vs baseline)
- **D4** (swap puell->eth): +377.8% (+92.9pp vs baseline)
- **D5** (non-ETH replacements): +335.7% (+50.8pp vs baseline)

### Cross-Model Validation
- **E1_top2_RF**: +207.8% (Sortino 0.90)
- **E2_top2_CatBoost**: +146.6% (Sortino 0.60)
- **E3_top2_LGB_RF**: +308.2% (Sortino 1.27)
- **E4_D5mixed_RF**: +189.7% (Sortino 0.79)
- **E5_D5mixed_CatBoost**: +154.8% (Sortino 0.62)
- **E6_D5mixed_LGB_RF**: +268.0% (Sortino 1.08)

### Seed Stability
- **F1_top2_Bag5**: mean=+405.9%, spread=80pp, beat V2: 10/10
- **F2_top2_Bag30**: mean=+412.0%, spread=39pp, beat V2: 10/10
- **F3_D5mixed_Bag5**: mean=+336.3%, spread=53pp, beat V2: 10/10
- **F4_D6mixed_Bag5**: mean=+424.9%, spread=109pp, beat V2: 10/10

---

## Group D: ETH Ablation

| Test | Config | Features | Return | Delta | Sortino | MaxDD |
|------|--------|----------|--------|-------|---------|-------|
| Baseline | B7 ret_3d | 25 | +284.9% | --- | 1.10 | -21.5% |
| D1_add_eth | Add eth as 26th feature, no removal | 26 | +357.0% | +72.1pp | 1.38 | -16.3% |
| D2_remove_4weak | Remove 4 weak features, no replacement ( | 21 | +332.3% | +47.4pp | 1.29 | -21.9% |
| D3_remove_puell_rsi | Remove puell+rsi only, no replacement (2 | 23 | +310.0% | +25.1pp | 1.20 | -21.2% |
| D4_swap_puell_eth | Swap puell->eth only (keep sopr, funding | 25 | +377.8% | +92.9pp | 1.40 | -16.4% |
| D5_mixed_no_eth | puell+rsi->momentum_accel_7d, sopr+fundi | 25 | +335.7% | +50.8pp | 1.23 | -20.6% |
| D6_mixed_1eth | puell->eth, rsi->momentum_accel_7d | 25 | +425.8% | +140.9pp | 1.52 | -18.8% |

## Group E: Cross-Model Validation

| Test | Model | Swap | Return | Sortino | MaxDD |
|------|-------|------|--------|---------|-------|
| E1_top2_RF | RandomForest | top2_eth | +207.8% | 0.90 | -23.8% |
| E2_top2_CatBoost | CatBoost | top2_eth | +146.6% | 0.60 | -37.3% |
| E3_top2_LGB_RF | LGB+RF | top2_eth | +308.2% | 1.27 | -16.9% |
| E4_D5mixed_RF | RandomForest | D5_mixed | +189.7% | 0.79 | -34.1% |
| E5_D5mixed_CatBoost | CatBoost | D5_mixed | +154.8% | 0.62 | -39.7% |
| E6_D5mixed_LGB_RF | LGB+RF | D5_mixed | +268.0% | 1.08 | -19.9% |

## Group F: Seed Stability

| Strategy | Mean | Median | Spread | Min | Max | Beat V2 |
|----------|------|--------|--------|-----|-----|---------|
| F1_top2_Bag5 | +405.9% | +417.6% | 80pp | +357.4% | +437.1% | 10/10 |
| F2_top2_Bag30 | +412.0% | +413.9% | 39pp | +390.4% | +429.0% | 10/10 |
| F3_D5mixed_Bag5 | +336.3% | +339.8% | 53pp | +310.0% | +363.1% | 10/10 |
| F4_D6mixed_Bag5 | +424.9% | +426.0% | 109pp | +353.6% | +462.7% | 10/10 |

### F1_top2_Bag5 Bootstrap 95% CI
- Return CI: [+77%, +1497%]
- P(return > 0): 99.9%

### F3_D5mixed_Bag5 Bootstrap 95% CI
- Return CI: [+29%, +1453%]
- P(return > 0): 99.5%

## Group G: Untested High-Rank Features

| Test | Weak | Strong | Rank | Return | Delta | Sortino |
|------|------|--------|------|--------|-------|---------|
| G8 | rsi_14d | eth_pctchg_30d | 29 | +340.0% | +55.0pp | 1.28 |
| G6 | rsi_14d | plus_di | 15 | +335.9% | +51.0pp | 1.27 |
| G4 | puell_multiple | eth_pctchg_30d | 29 | +330.6% | +45.7pp | 1.24 |
| G5 | rsi_14d | kpss_stat_30d | 13 | +310.9% | +26.0pp | 1.23 |
| G7 | rsi_14d | miners_revenue_usd | 17 | +296.6% | +11.7pp | 1.17 |
| G3 | puell_multiple | miners_revenue_usd | 17 | +291.3% | +6.4pp | 1.14 |
| G2 | puell_multiple | plus_di | 15 | +282.4% | -2.5pp | 1.10 |
| G1 | puell_multiple | kpss_stat_30d | 13 | +281.7% | -3.2pp | 1.14 |

---

## Key Conclusions

1. **Is ETH genuine?** — See Group D results above
2. **Do non-ETH combos work?** — See D5 result + F3 stability
3. **Model-robust?** — See Group E (>+50pp improvement across models = real)
4. **Most stable strategy?** — See F1-F4 spreads
5. **Better features than ETH?** — See Group G results