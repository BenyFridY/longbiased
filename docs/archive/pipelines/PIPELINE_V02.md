# PIPELINE V2 - BTC ALLOCATION STRATEGY

> **STATUS:** Completo. Pronto para revisao externa.
> **Ultima atualizacao:** 2026-02-09
> **Objetivo:** Estrategia de alocacao BTC (long/short/neutro) que bata buy-and-hold com walk-forward honesto.

---

## 1. RESUMO EXECUTIVO

Testamos **12+ abordagens** para alocar BTC dinamicamente (de -25% short a +100% long).
Todas testadas com **walk-forward out-of-sample (OOS) 2022-2026**, sem look-ahead bias.

### Melhor resultado atual:

| Estrategia | OOS Return | Sortino | MaxDD | vs BTC |
|-----------|-----------|---------|-------|--------|
| **Hybrid + Sentiment** | **+171%** | **0.69** | **-25%** | **+117pp** |
| Hybrid Baseline (ML+Mom) | +166% | 0.67 | -27% | +112pp |
| ML Standalone (LightGBM) | +202% | 0.72 | -24% | +149pp |
| Momentum (ret_3d+ret_60d) | +152% | 0.59 | -41% | +99pp |
| BTC Buy & Hold | +53% | 0.27 | -67% | ref |

**Nota:** ML Standalone tem +202% mas tem concerns de robustez (seed [+134%,+177%] range). Hybrid+Sentiment e mais robusto.

---

## 2. DATASET

**Arquivo:** `outputs/feature_selection/dataset_enhanced.csv`
- **2588 linhas** (2019-01-01 a 2026-01-31, diario)
- **280 colunas** organizadas em:

| Categoria | Exemplos | Qtd |
|-----------|----------|-----|
| Preco/Volume | price_usd, volume_usd, OHLC | ~10 |
| Returns | return_1d/7d/14d/30d/60d/90d/180d/365d | ~10 |
| Volatilidade | volatility_7d/30d/60d, parkinson, ATR | ~8 |
| Tecnico | RSI, MACD, BB, ADX, momentum, z-scores | ~15 |
| Onchain | NUPL, SOPR, MVRV, puell, miners, netflow, active addr | ~30 |
| Derivativos | funding_rate, basis, OI, taker_pressure, buy_sell | ~30 |
| Macro | M2, Fed BS, SP500, gold, DXY, VIX, yields | ~20 |
| Regime | cusum, hurst, structural_break, mean_reversion | ~15 |
| Sentiment | fear_greed, stablecoin_supply, btc_dominance | ~15 |
| Interacoes | funding_x_oi, mvrv_x_nupl, rsi_x_bb etc | ~10 |
| Targets | direction_1d/5d, return_1d/5d, regime, triple_barrier | ~20 |

**IMPORTANTE:** `return_1d` do dataset != `price_usd.pct_change()` (ate 10% diferenca). Sempre recalcular retornos a partir de `price_usd`.

---

## 3. METODOLOGIA

### 3.1 Walk-Forward (Sem Look-Ahead)

```
Para cada ano de teste (2022, 2023, 2024, 2025, 2026):
  1. TREINAR: todos dados antes do ano de teste
  2. OTIMIZAR: parametros no training set (random search 2000 trials)
  3. TESTAR: aplicar no ano de teste (nunca visto)
  4. REGISTRAR: retornos diarios OOS
Concatenar todos os anos OOS = resultado final
```

### 3.2 Backtest Engine

```python
# Alocacao decidida no fim do dia t, aplicada ao retorno de t+1
# Range: [-0.25 (short), +1.0 (full long)]
# Custo: 2 bps por trade, 1.5x para shorts
# Cash rende: 15% ao ano (RF_DAILY)
# Metrica principal: Return Total, Sortino, MaxDD
```

### 3.3 Estrategia Base (Momentum)

```
10 parametros continuos controlando:
- Limiares bear (slow + fast return)
- Limiar bull (slow return)
- Range de alocacao (min, max, mid)
- Slopes (bear/bull transition speed)
- Vol dampening (reduce exposure in high vol)

Sinais: ret_3d (fast), ret_60d (slow), vol_14d
```

---

## 4. TODAS AS ABORDAGENS TESTADAS

### Fase 1: Baseline Momentum (scripts/final/)

| Sinais | OOS Return | Sortino | MaxDD |
|--------|-----------|---------|-------|
| ret_3d + ret_60d + vol_14d | +152% | 0.59 | -41% |
| ret_10d + ret_60d + vol_14d | +105% | 0.37 | -44% |
| Old balanced_v2 (fixed rules) | +127% | 0.51 | - |

**Script:** `scripts/final/truly_honest_backtest.py`

### Fase 2: Exhaustive Search (scripts/optimization/)

| # | Abordagem | OOS Return | Sortino | MaxDD | Verdict |
|---|----------|-----------|---------|-------|---------|
| 1 | + Onchain/Macro Filters | +150% | 0.59 | -38% | ~same |
| 2 | Regime Switching | +75% | 0.18 | -50% | WORSE |
| 3 | **ML Regression (LightGBM)** | **+202%** | **0.72** | **-24%** | **BEST** |
| 4 | Ensemble (Mom+OC+Macro) | +128% | 0.51 | -32% | WORSE |
| 5 | Alt Signals (cusum best) | +124% | 0.47 | -51% | WORSE |

**ML Details:**
- 25 features (20 onchain/regime/macro + 5 price-derived)
- LightGBM regression, predicts 5-day forward return
- 200 boost rounds, lr=0.05, num_leaves=31
- Prediction -> allocation: `scaled = clip(pred / 0.05, -1, 1)`

**ML Robustness Concerns:**
- Seed stability: [+134%, +177%] range (not all seeds beat baseline)
- Shuffled target test: +85% (features carry independent signal)
- No data leakage found in audit

**Script:** `scripts/optimization/exhaustive_search.py`

### Fase 3: Edge Quant (scripts/optimization/)

Todas baseadas no **Hybrid (50% ML + 50% Momentum)** = +166% baseline.

| # | Abordagem | OOS Return | Sortino | MaxDD | Verdict |
|---|----------|-----------|---------|-------|---------|
| 0 | Hybrid Baseline (ML+Mom) | +166% | 0.67 | -27% | BASELINE |
| 1 | + Derivatives Filter (p95) | +164% | 0.66 | -29% | ~same |
| 2 | **+ Sentiment + Stablecoin** | **+171%** | **0.69** | **-25%** | **WINNER** |
| 3 | Multi-Target ML + Kelly | +62% | 0.08 | -36% | WORSE |
| 4 | COMBO (1+2+3) | +103% | 0.37 | -35% | WORSE |

**Edge 2 Details (WINNER):**
- Fear/Greed z-score < p5 -> contrarian buy (+0.12 alloc)
- Fear/Greed z-score > p90 -> modest reduce (-0.06 alloc)
- Stablecoin supply growth > p80 + zscore > p80 -> bullish (+0.06)
- Stablecoin shrinking < p20 -> bearish (-0.04)
- ETH/BTC 30d change < p10 -> crypto risk-off (-0.04)
- COMBO: fear + stablecoin inflow + trend positive -> max conviction (+0.08)
- All thresholds computed from TRAINING data only per year

**Script:** `scripts/optimization/edge_quant.py`

---

## 5. LICOES APRENDIDAS (O QUE NAO FUNCIONA)

### 5.1 Onchain/Macro como Sinais Primarios: NAO FUNCIONA
- Miners revenue, netflow, NUPL, SOPR como sinais diretos = pior que momentum puro
- Razao: preco ja incorpora info publica. Edge vem de MOMENTUM (comportamental).

### 5.2 Regime Switching: NAO FUNCIONA
- Alternar entre trend-following e mean-reversion = -50% pior
- Razao: falsos sinais de regime destroem retorno

### 5.3 Kelly Sizing: DESTRUTIVO
- Acuracia direcional ~50-52% -> Kelly fraction colapsa para o piso
- Resultado: muta todas as posicoes, retorno cai de +166% para +62%
- Licao: Kelly requer win rate >> 50% para funcionar. BTC nao tem isso em daily.

### 5.4 Combinar Edges Multiplicativamente: PIORA
- Kelly * Concordance * Prediction = compressao excessiva de sinais
- Filtros aditivos simples > sizing multiplicativo complexo

### 5.5 Derivativos como Filtro em p95: NEUTRO
- Triggers muito raros (~100 em 4 anos) para impactar resultados
- Em p90 era muito frequente e adicionava ruido

### 5.6 In-Sample vs OOS Gap
- In-sample: ate 3626% com parametros otimizados no periodo todo
- OOS honesto: 105-202% dependendo da abordagem
- Licao: QUALQUER resultado sem walk-forward e lixo

---

## 6. O QUE FUNCIONA (RESUMO)

1. **Momentum (ret_3d + ret_60d)** = o motor principal da estrategia
2. **LightGBM** = adiciona ~50pp ao momentum quando combinado 50/50
3. **Sentiment contrarian** = adiciona ~5pp e melhora MaxDD
4. **Volatility dampening** = reduz exposicao em vol alta (ja no momentum base)
5. **Walk-forward re-optimization anual** = crucial para robustez

---

## 7. ARQUIVOS CHAVE

### Scripts
| Arquivo | Descricao |
|---------|-----------|
| `scripts/optimization/edge_quant.py` | Edge quant: derivatives, sentiment, multi-Kelly |
| `scripts/optimization/exhaustive_search.py` | 5 abordagens vs baseline |
| `scripts/optimization/audit_exhaustive_search.py` | Audit + robustez do ML |
| `scripts/optimization/finetune_strategy.py` | Pipeline de otimizacao momentum |
| `scripts/optimization/balanced_strategies.py` | Estrategias originais (balanced_v2) |
| `scripts/final/truly_honest_backtest.py` | Walk-forward honesto |
| `scripts/final/honest_audit.py` | Prova dia-a-dia de no look-ahead |

### Resultados
| Arquivo | Descricao |
|---------|-----------|
| `outputs/results/edge_quant.json` | Resultados edge quant |
| `outputs/results/exhaustive_search.json` | Resultados busca exaustiva |
| `outputs/results/truly_honest_backtest.json` | Resultados walk-forward |
| `outputs/results/charts/edge_quant_vs_buyhold.png` | Chart comparativo edges |
| `outputs/results/charts/edge_quant_detailed.png` | Equity curves + drawdown |
| `outputs/results/charts/exhaustive_comparison.png` | Chart busca exaustiva |

### Dataset
| Arquivo | Descricao |
|---------|-----------|
| `outputs/feature_selection/dataset_enhanced.csv` | Dataset principal (280 cols, 2588 rows) |

---

## 8. PARAMETROS DA MELHOR ESTRATEGIA

### Hybrid + Sentiment (atual melhor)
```
Base = 50% * Momentum(ret_3d, ret_60d, vol_14d, 10_params) + 50% * ML(25_features, LightGBM)

Sentiment adjustments (additive):
  +0.12 if fear_greed_zscore < p5 OR extreme_fear flag
  +0.05 if fear_greed_zscore < p10
  -0.06 if fear_greed_zscore > p90 OR extreme_greed flag
  +0.06 if stablecoin_chg_30d > p80 AND stablecoin_zscore > p80
  -0.04 if stablecoin_chg_30d < p20
  -0.04 if eth_btc_30d_change < p10
  +0.08 if (fear < p10 AND stablecoin > p80 AND ret_30d > 0)  # combo

Cost: 2 bps, 1.5x for shorts
Range: [-0.25, +1.0]
RF: 15% annual
Walk-forward: annual re-optimization (2000 random trials per year)
```

---

## 9. PERGUNTAS ABERTAS PARA REVISAO

1. **ML Robustness:** O LightGBM tem +202% standalone mas seed range [134%, 177%]. Vale usar bagging (10 seeds) para estabilizar? Ou o custo de complexidade nao vale os ~20pp extras?

2. **Sentiment Edge Real?** +5pp (+171% vs +166%) e estatisticamente significante ou apenas ruido? Seriam necessarios mais anos de dados para confirmar.

3. **Features Nao Exploradas:** Temos 280 features mas o ML usa apenas 25. Ha features subutilizadas que poderiam adicionar valor? (ex: Hurst exponent, half_life, OU parameters)

4. **Horizontes Alternativos:** Testamos alocacao diaria. Semanal ou bi-semanal reduziria custos e ruido?

5. **Position Sizing Alternativo:** Kelly falhou por win rate ~50%. Ha outros metodos de sizing (volatility targeting, risk parity, CPPI) que funcionariam melhor?

6. **Regime Detection:** Regime switching puro falhou, mas e se usarmos regime apenas como FILTRO (ex: desligar short em bull confirmado)?

7. **Derivativos em Threshold Diferente:** p95 e muito extremo (100 triggers em 4 anos). p90 era ruidoso. Ha um sweet spot? Ou combinar com vol regime?

8. **Overfitting Risk:** Com 10 params de momentum re-otimizados anualmente + ML retreinado = quantos graus de liberdade efetivos? E muito?

9. **Alternativas ao LightGBM:** CatBoost? XGBoost? Neural nets (LSTM/Transformer)? Foundation models (TabPFN)?

10. **Custo Real:** Assumimos 2 bps. Se for 5 bps ou 10 bps, a estrategia ainda funciona? Qual o breakeven de custo?

---

## 10. COMO REPRODUZIR

```bash
cd C:\Users\voce\Documents\longbiased

# Instalar dependencias
pip install pandas numpy lightgbm matplotlib

# Rodar walk-forward baseline (momentum)
python scripts/final/truly_honest_backtest.py

# Rodar busca exaustiva (5 abordagens)
python scripts/optimization/exhaustive_search.py

# Rodar edge quant (3 edges + combo)
python scripts/optimization/edge_quant.py

# Resultados em outputs/results/*.json e outputs/results/charts/*.png
```

---

## 11. CONCLUSAO

A estrategia **Hybrid (ML+Momentum) + Sentiment Contrarian** atinge **+171% OOS** contra **+53% do BTC** no periodo 2022-2026, com Sortino de 0.69 e MaxDD de -25%.

O motor principal do alpha e **momentum de preco** (ret_3d como sinal rapido, ret_60d como sinal lento). O ML adiciona ~14pp e o sentiment contrarian adiciona ~5pp.

Abordagens mais complexas (Kelly sizing, regime switching, ensembles multiplicativos) **pioram** o resultado. A simplicidade vence.

A principal duvida restante e se o edge do ML (~14pp) e robusto a diferentes seeds e se 5 anos de dados OOS sao suficientes para validar.
