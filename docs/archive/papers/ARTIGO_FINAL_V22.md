# De Estagiario a Modelo Quant: 22 Versoes de um Sistema de Alocacao BTC/CDI

**Beny Frid** | **Marco 2026**

---

## Resumo

Este documento descreve o desenvolvimento de um sistema de alocacao dinamica entre Bitcoin e CDI usando Machine Learning, iterado ao longo de 22 versoes, 1500+ configuracoes testadas e 300+ horas de compute. O modelo final (V22) alcanca Sortino 4.72, Sharpe 2.35, retorno de +1046% vs +39% do BTC buy & hold, com drawdown maximo de apenas -10.0%, no periodo out-of-sample 2022-2026. Analise CAPM confirma alpha anual de +48.7% com beta de 0.107.

---

## 1. Motivacao

Bitcoin tem retorno historico excepcional mas volatilidade extrema (MaxDD de -77% em 2022). A pergunta central eh: **eh possivel capturar o upside do BTC evitando o downside, usando um modelo quantitativo?**

O benchmark eh simples: superar BTC buy & hold em retorno ajustado ao risco (Sortino), mantendo drawdown controlado. O "safe haven" eh CDI, que rendeu ~65% no periodo (12-15% ao ano).

---

## 2. Arquitetura do Modelo

### 2.1 Visao Geral

```
37 features → XGBoost (80 bags) → predicao ret_3d → regime × K → alocacao
```

O modelo preve o **retorno do BTC nos proximos 3 dias**. A predicao eh multiplicada por um fator K que depende do regime de mercado (BULL/MILD/BEAR), gerando uma alocacao entre -25% e +100%.

### 2.2 Features (37)

As 37 features vem de 9 fontes de dados:

**Tecnicas (18)**: cusum_pos/neg, adx, macd_histogram, bb_position, volatility_7d, price_percentile_1y, sortino_30d, obv_trend, volume_sma20_ratio, aroon_down, hurst_60d, fractal_dimension, kpss_stat, ou_theta, half_life, mr_score, structural_break_score, trend_strength

**Macro (4)**: m2_yoy_growth, fed_balance_sheet, copper_return_30d, btc_gold_corr_30d

**On-chain (4)**: nupl_ma30, miners_revenue_ratio, hash_rate_pctchg_30d, velocity

**Derivativos (5)**: basis_pct, basis_ma7, open_interest, futures_trade_count, vol_x_regime_duration

**Cross-asset (4)**: eth, eth_btc_ratio, eth_pctchg_30d, stablecoin_zscore

**Stablecoins (2)**: stablecoin_supply_change_30d, stablecoin_zscore

### 2.3 Regime e Alocacao

O regime eh classificado pela relacao entre preco, SMA50 e SMA200:

| Regime | Condicao | K | Logica |
|--------|----------|---|--------|
| BULL | price > SMA50 > SMA200 | 50 | Tendencia confirmada, aposta mais |
| MILD | price > SMA200 | 30 | Incerto, moderado |
| BEAR | price < SMA200 | 15 | Adverso, conservador |

Formula: `alocacao = clip(predicao × K, -25%, 100%)`

Exemplo: predicao +0.7% em BULL → 0.7% × 50 = 35% BTC, 65% CDI.

### 2.4 Rebalance

- **Sexta-feira**: rebalance semanal. Escolhido empiricamente (V19) — sexta eh 2x melhor que o segundo melhor dia (quinta)
- **Emergency**: se |retorno diario| > 8%, rebalanceia imediatamente. Captura crashes/pumps extremos

### 2.5 Treino

- **Expanding window**: usa todos os dados desde o inicio ate o corte
- **Semi-anual**: retreina Jan e Jul. Trimestral overfita, anual adapta devagar
- **80 bags**: cada modelo treina com seed diferente, predicao = media do ensemble

---

## 3. Jornada de 22 Versoes

### 3.1 Fase 1: Fundamentos (V02-V05)

| Versao | Sortino | Insight |
|--------|---------|---------|
| V02 | 0.69 | ML+Momentum hybrid funciona. Regime switching (Markov) **falha** |
| V03 | 0.87 | **Bagging** = maior melhoria da historia do pipeline |
| V04 | 0.89 | Modelos assimetricos bull/bear sao seed-dependent |
| V05 | 1.12 | Peso 100% ML > 50/50. **ML weight eh o lever** |

**Licoes**: Bagging transforma um modelo mediocre em bom. Simplicidade vence complexidade. Markov regime switching nao funciona com BTC.

### 3.2 Fase 2: Features e Formula (V06-V12)

| Versao | Sortino | Insight |
|--------|---------|---------|
| V06 | 1.54 | price_percentile_1y = feature mais impactante (+71pp) |
| V08 | 1.51 | Aprovado 14/16 testes de overfitting |
| V09 | 2.48 | 37 features eh o ponto otimo |
| V10 | 2.73 | Formula linear_direct K=25 |
| V12 | 2.91 | Hybrid LGB+XGB no ponto otimo |

**Licoes**: A formula de alocacao importa mais que o modelo. 37 features — nem mais, nem menos. Feature selection tem retorno decrescente rapido.

### 3.3 Fase 3: Regime e Escala (V13-V19)

| Versao | Sortino | Insight |
|--------|---------|---------|
| V13 | 3.22 | Semi-anual retrain (+0.10 Sortino) + basis_pct |
| V17 | 3.55 | **Dynamic regime** (K por bull/mild/bear) = game-changer |
| V18 | 3.64 | K_bull=50 + Bag80. HMM testado e descartado |
| V19 | 3.87 | Emergency 8% rebalance |

**Licoes**: Regime detection via SMA eh simples e robusto. HMM classifica 79% dos dias como "mild" (inutil). Emergency rebalance captura eventos extremos raros.

### 3.4 Fase 4: Validacao e Auditoria (V20-V22)

| Versao | Sortino | Insight |
|--------|---------|---------|
| V20 | 3.96 | 126 combos testados, baseline venceu |
| V21 | 4.00 | colsample_bytree=0.5 (fine-tuning) |
| V22 | 4.72 | **Sortino corrigido** (Sortino & Price 1994) |

**Licoes**: Chegou no ponto de retornos marginais. 126 combinacoes testadas e nenhuma superou o baseline de forma robusta. A formula do Sortino importa — `std(negativos)` inflava 24% (V21 reportava 4.00, correto eh 4.72 com formula padrao).

---

## 4. O Que Nao Funcionou (e Por Que)

### 4.1 Modelos Descartados

| O que | Por que falhou |
|-------|---------------|
| HMM / Markov regime | 79% dos dias em um unico estado |
| GARCH features | Nao reproduzivel entre bibliotecas |
| LSTM / Deep Learning | XGBoost puro venceu com menos dados |
| Stacking / Meta-learners | Complexidade sem ganho |
| Early stopping XGBoost | Underfitting severo (para em 10-20 arvores) |

### 4.2 Features Descartadas

| Feature | Resultado |
|---------|-----------|
| Fear & Greed Index | -0.39 Sortino |
| Google Trends | Sem sinal |
| MVRV Z-Score | Piorou |
| SOPR, STH-SOPR | Sem melhoria |
| GARCH volatility | Inconsistente |
| 20 features extras (V19) | Nenhuma das 20 melhorou |

### 4.3 Estrategias Descartadas

| Estrategia | Resultado |
|------------|-----------|
| Kelly criterion | Accuracy ~53% torna Kelly destrutivo |
| Stop loss 5% | Dispara toda semana em BTC |
| Rebalance diario | Sortino 2.04 vs 4.72 |
| Retrain mensal | Overfitting a dados recentes |
| Segundo dia fixo (Mon+Fri) | Sempre piora |

---

## 5. Validacao

### 5.1 PBO (Probability of Backtest Overfitting)

PBO = 0.0% com 70 combinacoes CSCV. O melhor config in-sample tambem eh o melhor out-of-sample em 100% dos testes.

### 5.2 CAPM Alpha/Beta

```
R_strat - CDI = alpha + beta × (R_btc - CDI)
```

| Metrica | Valor |
|---------|-------|
| Alpha (anual) | **+48.7%** |
| Beta | 0.107 |
| Up capture | 17% |
| Down capture | 2% |
| Assimetria | **6.77x** |
| Rolling alpha | Positivo em **11/11 periodos** |

O retorno nao vem de exposicao ao BTC (beta = 0.1). Vem da decisao de **quando estar dentro e fora**.

### 5.3 Comparacao vs Trend Following em Producao

Periodo: outubro 2025 a marco 2026 (dados reais, nao backtest).

| Metrica | Trend Following | V22 |
|---------|----------------|-----|
| Return | -10.8% | **+9.3%** |
| MaxDD | -18.9% | **-12.7%** |
| Meses vencidos | 1/6 | **5/6** |

### 5.4 Timing de Execucao

Testado close (00:00 UTC) vs open (manha): **diferenca zero**. Slippage medio de 0.004% por rebalance. Pode executar a qualquer hora do dia.

---

## 6. Configuracao Final (V22)

```
Modelo:         XGBoost puro
Bags:           80
Features:       37
Target:         Retorno 3 dias a frente
Retrain:        Semi-anual (Jan/Jul), expanding window
Regime:         SMA50/SMA200 → K_bull=50, K_mild=30, K_bear=15
Alocacao:       clip(pred × K, -25%, 100%)
Rebalance:      Sexta-feira + emergency (|daily_ret| > 8%)
Custo:          2 bps por rebalance

Hyperparameters XGBoost:
  learning_rate=0.05, n_estimators=200, max_leaves=31
  min_child_weight=12, colsample_bytree=0.5, subsample=0.8
  grow_policy=lossguide, tree_method=hist
```

### Resultado (5 seeds, 2022-2026 OOS)

| Metrica | Valor |
|---------|-------|
| Sortino | 4.72 +/- 0.07 |
| Sharpe | 2.35 +/- 0.02 |
| Retorno | +1046% +/- 18% |
| MaxDD | -10.0% +/- 0.3% |
| BTC buy & hold | +39% |
| CDI acumulado | +65% |
| PBO | 0.0% |
| Alpha (CAPM) | +48.7% anual |

---

## 7. Pipeline de Producao

O modelo opera com um comando:

```bash
python scripts/production/generate_signal.py
```

Faz tudo automaticamente: detecta dados desatualizados, busca 9 APIs, calcula 37 features, gera sinal de alocacao.

### Fontes de Dados

| Fonte | Dados | Custo |
|-------|-------|-------|
| Binance API | OHLCV, basis, funding | Gratis |
| BigQuery Messari | OI, futures trade count | Interno |
| yfinance | ETH, Gold, Copper | Gratis |
| FRED API | M2, Fed BS | Gratis |
| CoinMetrics | Hash rate | Gratis |
| BGeometrics | NUPL | Gratis |
| Blockchain.com | Miners revenue | Gratis |
| DefiLlama | Stablecoin supply | Gratis |
| BCB API | CDI real | Gratis |

---

## 8. Insights Finais

1. **Simplicidade vence.** Em cada decisao de design — horizonte, retrain, modelo, formula — a opcao mais simples tende a superar variantes complexas fora da amostra.

2. **Bagging eh o multiplicador.** Mover de 1 modelo para 80 bags foi a maior melhoria isolada de toda a historia do pipeline.

3. **A formula importa mais que o modelo.** Mesmas predicoes com formulas de alocacao diferentes geram resultados dramaticamente diferentes.

4. **Regime detection via SMA eh imbativel.** HMM, RSI, vol percentile, composite — todos piores que duas medias moveis.

5. **CDI eh o hedge perfeito.** Em bear market, a estrategia nao precisa shortar agressivamente — basta ir para CDI que rende 12-15% ao ano. O "productive hedging" brasileiro eh uma vantagem estrutural.

6. **O modelo sabe quando NAO sabe.** Em regime MILD (accuracy 54%), a alocacao eh automaticamente moderada (K=30). Nao precisa de regras adicionais.

7. **22 versoes > 1 versao perfeita.** O processo iterativo — testar, medir, descartar — eh mais valioso que qualquer insight individual.

---

## Referencias

- Sortino, F. A., & Price, L. N. (1994). Performance Measurement in a Downside Risk Framework. *Journal of Investing*.
- Bailey, D. H., et al. (2014). The Probability of Backtest Overfitting. *Journal of Computational Finance*.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.
- Hamilton, J. D. (1989). A New Approach to the Economic Analysis of Nonstationary Time Series. *Econometrica*.
