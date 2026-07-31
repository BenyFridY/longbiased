# Pipeline V19 — Feature Screening + Rebalance Optimization

**Data**: 2026-03-17 | **Compute total**: ~7h (feature screening ~2h + rebalance ~5h)

---

## Resumo Executivo

O V19 partiu do V18 (37 features, S=3.64) e testou em duas frentes:

**Frente 1 — Feature Screening (20 features)**: Testou sentiment (Fear & Greed, Google Trends), macro (VIX, DXY, yield curve), on-chain (MVRV, SOPR), microstructure (taker pressure) e mining (S2F, difficulty ribbon). **Nenhuma feature melhorou o baseline.** O modelo com 37 features ja captura toda a informacao necessaria.

**Frente 2 — Rebalance Optimization (13 configs)**: Testou todos os combos Fri+outro dia, e emergency triggers com diferentes thresholds. **Fri+emergency 8% (all regimes) melhorou**: S=3.87 (+0.23 vs baseline).

### Configuracao Final V19

```
Modelo:     XGBoost puro, 80 bags, 16 cores paralelos
Features:   37 (32 base + 5 V17) — SEM garch_persistence
Target:     ret_3d (retorno 3 dias a frente)
Retrain:    Semi-anual, expanding window
Alocacao:   Dynamic Regime — K_bull=50, K_mild=30, K_bear=15
Rebalance:  Sexta-feira + emergency (|daily_ret| > 8%)
```

### Evolucao V18 -> V19

| Metrica | V18 | V19 | Delta |
|---------|-----|-----|-------|
| **Sortino** | 3.64 | **3.87** | **+0.23** |
| **Retorno** | +1054% | **+1059%** | +5pp |
| **Spread** | 78pp | 87pp | +9pp (pior) |
| **MaxDD** | -10.1% | **-10.1%** | igual |

---

## Frente 1: Feature Screening

### Metodologia

Base: V18 com 37 features (sem garch_persistence — removido porque arch 7.2.0 produz valores diferentes da versao original, causando degradacao de S=3.63 para S=3.02).

Testou 20 features candidatas individualmente (1 seed cada), adicionando ao baseline de 37:

### Resultados (todas pioraram)

| Feature | Sortino | Delta | Grupo |
|---------|---------|-------|-------|
| BASELINE | 3.63 | — | — |
| fg_extreme_signal | 3.58 | -0.047 | Sentiment |
| sth_capitulation | 3.59 | -0.047 | On-chain |
| puell_multiple | 3.58 | -0.051 | Mining |
| basis_zscore | 3.56 | -0.075 | Microstructure |
| funding_x_oi | 3.55 | -0.085 | Microstructure |
| dxy_pctchg_30d | 3.54 | -0.094 | Macro |
| taker_pressure | 3.53 | -0.098 | Microstructure |
| mvrv_zscore | 3.52 | -0.111 | On-chain |
| price_vs_realized | 3.52 | -0.111 | On-chain |
| fear_greed_zscore | 3.51 | -0.121 | Sentiment |
| high_yield_spread | 3.49 | -0.147 | Macro |
| vix_zscore | 3.39 | -0.247 | Macro |
| yield_curve_2s10s | 3.38 | -0.249 | Macro |
| fg_x_vix | 3.31 | -0.326 | Sentiment |
| fear_greed_ma7 | 3.24 | -0.388 | Sentiment |
| sopr_ma7 | 2.94 | -0.690 | On-chain |
| difficulty_ribbon | 2.89 | -0.740 | Mining |
| days_since_halving | 2.04 | -0.975 | Mining |
| google_trend_btc | 1.50 | -2.135 | Sentiment |

### Conclusoes Feature Screening

1. **Nenhuma das 20 features melhorou o baseline** — todas tiveram delta negativo
2. **Sentiment e catastrofico**: Google Trends destruiu (S=1.50, DD=-34.6%), Fear & Greed piorou consistentemente
3. **Macro risk nao ajuda**: VIX, DXY, yield curve — todos negativos
4. **On-chain value nao ajuda**: MVRV, SOPR, realized price — redundantes com features existentes
5. **O modelo com 37 features ja esta saturado** — informacao adicional vira ruido

### Nota sobre garch_persistence

O V18 original reportou S=3.74 com garch_persistence. Porem, ao re-rodar com arch 7.2.0 (versao atual), o resultado cai para S=3.02 — pior que o baseline sem garch (S=3.63). Os valores de persistence do GARCH(1,1) dependem do optimizer da biblioteca `arch`, que mudou entre versoes. **garch_persistence nao eh reprodutivel entre versoes da biblioteca e portanto nao deve ser usado.**

---

## Frente 2: Rebalance Optimization

### Metodologia

Testou 13 configuracoes de rebalance com 10 seeds cada:
- 1 baseline (sexta only)
- 4 combos de sexta + outro dia fixo (Mon, Tue, Wed, Thu)
- 4 emergency triggers em todos os regimes (3%, 5%, 8%, 10%)
- 4 emergency triggers apenas em bear (3%, 5%, 8%, 10%)

Emergency trigger: rebalanceia em qualquer dia da semana se |daily_ret| > threshold.

### Parte 1: Single Day (confirmacao)

| Dia | Sortino | Retorno | MaxDD |
|-----|---------|---------|-------|
| **Sexta** | **3.64** | **+1054%** | **-10.1%** |
| Quinta | 1.89 | +357% | -11.0% |
| Segunda | 1.31 | +236% | -16.0% |
| Quarta | 0.92 | +178% | -22.8% |
| Terca | 0.64 | +140% | -20.0% |

Sexta confirmada como melhor dia — 2x melhor que quinta (segundo lugar).

### Parte 2: Fri + Outro Dia Fixo

| Config | Sortino | Retorno | MaxDD |
|--------|---------|---------|-------|
| Fri only | 3.64 | +1054% | -10.1% |
| Fri+Thu | 2.90 | +694% | -9.5% |
| Fri+Mon | 2.69 | +658% | -14.2% |
| Fri+Wed | 2.60 | +736% | -14.1% |
| Fri+Tue | 2.05 | +531% | -14.2% |

**Adicionar um segundo dia fixo SEMPRE piora.** Rebalancear mais de 1x/semana em dias fixos adiciona ruido e custos de transacao.

### Parte 3: Fri + Emergency (todos os regimes)

| Threshold | Sortino | Retorno | Spread | MaxDD |
|-----------|---------|---------|--------|-------|
| **8%** | **3.87** | **+1059%** | **87pp** | **-10.1%** |
| 10% | 3.81 | +1125% | 87pp | -10.1% |
| 5% | 3.17 | +888% | 56pp | -9.9% |
| 3% | 2.76 | +790% | 73pp | -15.6% |

**8% eh o threshold otimo.** 3% rebalanceia demais (DD sobe para -15.6%). 5% ainda rebalanceia demais. 10% eh quase igual a 8%.

### Parte 4: Fri + Emergency (bear only)

| Threshold | Sortino | Retorno | Spread | MaxDD |
|-----------|---------|---------|--------|-------|
| 10% | 3.81 | +1155% | 91pp | -10.1% |
| 5% | 3.54 | +1180% | 84pp | -10.1% |
| 8% | 3.38 | +1024% | 95pp | -10.1% |
| 3% | 3.12 | +1118% | 101pp | -15.6% |

Bear-only emergency nao eh melhor que all-regimes para 8% threshold.

### Ranking Final Completo

| # | Config | Sortino | Retorno | Spread | MaxDD |
|---|--------|---------|---------|--------|-------|
| 1 | **Fri+emerg_8%_all** | **3.87** | **+1059%** | **87pp** | **-10.1%** |
| 2 | Fri+emerg_10%_bear | 3.81 | +1155% | 91pp | -10.1% |
| 3 | Fri+emerg_10%_all | 3.81 | +1125% | 87pp | -10.1% |
| 4 | Fri only (baseline) | 3.64 | +1054% | 78pp | -10.1% |
| 5 | Fri+emerg_5%_bear | 3.54 | +1180% | 84pp | -10.1% |
| 6 | Fri+emerg_8%_bear | 3.38 | +1024% | 95pp | -10.1% |
| 7 | Fri+emerg_5%_all | 3.17 | +888% | 56pp | -9.9% |
| 8 | Fri+emerg_3%_bear | 3.12 | +1118% | 101pp | -15.6% |
| 9 | Fri+Thu | 2.90 | +694% | 55pp | -9.5% |
| 10 | Fri+emerg_3%_all | 2.76 | +790% | 73pp | -15.6% |
| 11 | Fri+Mon | 2.69 | +658% | 52pp | -14.2% |
| 12 | Fri+Wed | 2.60 | +736% | 63pp | -14.1% |
| 13 | Fri+Tue | 2.05 | +531% | 54pp | -14.2% |

---

## Insights

### 1. Emergency Rebalance e o Unico Upgrade
Adicionar um segundo dia fixo destroi Sortino. Mas emergency trigger (rebalancear em dias de movimento >8%) melhora porque captura oportunidades raras mas significativas — crashes e pumps extremos onde reposicionar imediatamente tem valor.

### 2. O Modelo com 37 Features Esta Saturado
Das 20 features testadas (sentiment, macro, on-chain, microstructure, mining), nenhuma melhorou. O XGBoost com 37 features + dynamic regime ja extrai toda informacao util dos dados disponiveis.

### 3. Sentiment Nao Funciona
Fear & Greed Index, Google Trends — todos pioraram significativamente. O modelo ML ja captura sentiment indiretamente via features de preco e volume.

### 4. garch_persistence Nao Eh Reprodutivel
Depende da versao da biblioteca `arch`. Nao deve ser usado em producao.

---

## Arquivos

| Arquivo | Descricao |
|---------|-----------|
| `scripts/optimization/pipeline_v19.py` | Feature screening (20 features) |
| `scripts/optimization/v19_rebal_focused.py` | Rebalance optimization (13 configs) |
| `outputs/results/pipeline_v19.json` | Resultados feature screening |
| `outputs/results/v19_rebal_focused.json` | Resultados rebalance |
| `docs/PIPELINE_V19.md` | Este documento |

---

## Ruled Out (V19)

Adicionar a lista de "nunca testar de novo":
- Todas as 20 features testadas (sentiment, macro risk, on-chain value, microstructure, mining/supply)
- Google Trends como feature (30% NaN, destroi modelo)
- garch_persistence (nao reprodutivel entre versoes de arch)
- Segundo dia fixo de rebalance (Mon/Tue/Wed/Thu junto com Fri)
- Emergency 3% threshold (rebalanceia demais)

---

## Apendice: V19B — Teste de Features de Opcoes e ETF (2026-03-19)

Testamos 14 features novas de opcoes (ATM IV, DVOL), ETF flows e Fear & Greed. **Nenhuma melhorou o modelo. A maioria destruiu a performance.**

| Feature | Sortino | Delta vs V19 | MaxDD |
|---------|---------|--------------|-------|
| **BASELINE (V19)** | **3.83** | **—** | **-10.4%** |
| Fear & Greed Z-score | 3.51 | -0.32 | -9.7% |
| Fear & Greed 7d MA | 3.40 | -0.43 | -10.4% |
| ATM IV 7d | 1.72 | -2.11 | -28.9% |
| IV Term Structure | 1.64 | -2.19 | -31.8% |
| DVOL Close | 1.60 | -2.23 | -36.6% |
| ETF Flow 7d MA | 1.57 | -2.26 | -12.7% |
| ETF Total Net Flow | 1.16 | -2.67 | -13.2% |

**Por que nao funcionou:**
- **Opcoes (IV, DVOL)**: historico incompleto (comeca 2020/2021). Periodos NaN confundem o modelo. DD saltou de -10% para -30%.
- **ETF Flows**: so existem desde jan/2024 (555 dias). Muito pouco para treinar.
- **Fear & Greed**: redundante com features de preco/momentum ja existentes.

Substituir volatility_7d por ATM IV 30d tambem destruiu: S=1.59, DD=-30.6%.

**Nota futura**: re-testar IV quando houver 5+ anos de historico (~2025+). ETF flows a partir de ~2028.

Features descartadas: ATM IV (7/30/90/180d), IV term structure/zscore, DVOL (close/zscore/pctchg), ETF flows (total/IBIT/MA/cumsum), Fear & Greed (MA7/zscore).
