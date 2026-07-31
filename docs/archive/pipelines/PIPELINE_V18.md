# Pipeline V18 — Modelo Final

**Data**: 2026-03-12 | **Compute total**: ~16h (bear fix 15h + PBO 0.5h)

---

## Resumo Executivo

O V18 partiu do V17 (37 features, S=3.55) e testou melhorias em bear markets e GARCH.

**Bear Market Fix (9 fases)**: Testou K_bear negativo, prediction offset, regime retrain, trend features, K assimetrico. **Nenhum bear fix melhorou.** Porem, K_bull=50 (mais agressivo em bull) e Bag80 (mais estabilidade) melhoraram.

**GARCH (9 testes)**: Testou garch_vol, gjr_vol, garch_persistence, gjr_asymmetry, vol_ratio. **Nenhuma feature GARCH melhorou** (resultado inicial positivo foi bug no script de teste, confirmado em rerun).

**PBO Test**: Probability of Backtest Overfitting = **0.0%** (zero overfitting).

### Configuracao Final V18

```
Modelo:     XGBoost puro, 80 bags, 16 cores paralelos
Features:   37 (32 base + 5 V17)
Target:     ret_3d (retorno 3 dias a frente)
Retrain:    Semi-anual, expanding window
Alocacao:   Dynamic Regime — K_bull=50, K_mild=30, K_bear=15
Rebalance:  Sexta-feira, sempre
PBO:        0.0% (zero overfitting)
```

### Evolucao V17 -> V18

| Metrica | V17 | V18 |
|---------|-----|-----|
| **Sortino** | 3.55 | **3.64** |
| **Retorno** | +930% | **+1054%** |
| **Spread** | 74pp | 78pp |
| **MaxDD** | -9.9% | -10.1% |

---

## Perfis V18

| Perfil | Retorno | Sortino | Spread | MaxDD |
|--------|---------|---------|--------|-------|
| Conservador | +746% | 3.37 | 57pp | -9.0% |
| **Balanceado** | **+1054%** | **3.64** | **78pp** | **-10.1%** |
| Agressivo | +1117% | 3.63 | 73pp | -10.7% |
| Ultra-Agressivo | +1149% | 3.56 | 78pp | -11.6% |

---

## PBO Test — Probability of Backtest Overfitting

**PBO = 0.0%** (zero overfitting em 70 combinacoes CSCV).

Metodo: divide OOS em 8 blocos, testa 72 configs em cada combinacao de 4 blocos IS / 4 blocos OOS. Em nenhuma das 70 combinacoes a melhor config IS teve retorno negativo OOS.

---

## Bear Market Fix — Resultados (todos falharam)

| Teste | Sortino | MaxDD |
|-------|---------|-------|
| **K_bear=15 (baseline)** | **3.55** | **-9.9%** |
| K_bear=0 (flat) | 2.11 | -9.8% |
| K_bear=-5 (short) | 1.82 | -19.1% |
| K_bear=-10 | 0.98 | -41.0% |
| Pred offset 0.01 | 3.36 | -10.3% |
| Regime retrain | 3.09 | -9.2% |
| Trend features | 2.52 | -9.5% |
| K assimetrico | 3.29 | -9.8% |

---

## GARCH — Resultados (todos falharam)

Resultado inicial de S=3.74 com garch_persistence foi bug no script de teste. Rerun confirmou S=3.02 — pior que baseline.

| Feature | Sortino | vs Baseline |
|---------|---------|-------------|
| V18 baseline (sem GARCH) | **3.64** | — |
| +garch_persistence (rerun) | 3.02 | **-0.62** |
| +garch_vol | 3.53 | -0.11 |
| +gjr_vol | 3.52 | -0.12 |
| +all_garch (5 feat) | 3.32 | -0.32 |

**Conclusao**: GARCH nao adiciona valor ao modelo XGBoost que ja tem volatility_7d como feature.

---

## 37 Features do Modelo Final

| # | Feature | Fonte | Categoria |
|---|---------|-------|-----------|
| 1-6 | cusum_pos/neg, miners_revenue_ratio, mr_score_30d, adx, structural_break_score | Calc/CM | Regime/On-chain |
| 7-11 | macd_histogram, eth_btc_ratio, m2_yoy_growth, volatility_7d, basis_ma7 | Calc/Ext | Tecnica/Macro |
| 12-16 | nupl_ma30, hurst_60d, eth, bb_position, eth_pctchg_30d | BG/Calc/yf | On-chain/Tecnica |
| 17-21 | price_percentile_1y, stablecoin_zscore, btc_gold_corr_30d, stablecoin_supply_change_30d, copper_return_30d | Calc/CM/yf | Stat/Macro |
| 22-26 | ou_theta_60d, fractal_dimension_30d, kpss_stat_30d, open_interest, half_life_60d | Calc/Artemis | Stat/Derivativos |
| 27-32 | sortino_30d, obv_trend, volume_sma20_ratio, aroon_down_30d, trend_strength, basis_pct | Calc/Binance | Tecnica/Derivativos |
| 33-37 | fed_balance_sheet, futures_trade_count, vol_x_regime_duration, velocity, hash_rate_pctchg_30d | FRED/Artemis/CM | Macro/On-chain |

---

## Arquivos

| Arquivo | Descricao |
|---------|-----------|
| `scripts/optimization/pipeline_v18.py` | Script V18 (modelo + charts + PBO) |
| `outputs/results/pipeline_v18_bear_fix.json` | Resultados bear fix |
| `outputs/results/v18_garch_test.json` | Resultados GARCH (todos falharam) |
| `outputs/results/v18_final_pbo.json` | PBO test (ignorar Sortino — bug GARCH) |
| `outputs/charts/v18_*.png` | 7 graficos PNG |
| `docs/PIPELINE_V18.md` | Este documento |
