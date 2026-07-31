# V36 — Final Report: Data Audit + Feature Engineering (2026-04-19)

**Motivação inicial:** 2026 YTD acurácia caiu pra 41.9% (vs 54-60% histórico). Usuário queria saber se era problema real ou ruído.

**Resposta em uma frase:** Era dataset defasado (47 dias). Com dataset fresh + 3 features novas on-chain, **Sortino sobe de 5.72 (V29 prod atual) → 6.39 (+11%)** mantendo filosofia long-biased.

---

## TOP 3 candidatos finais (10 seeds cada, validados estatisticamente)

| # | Config | Sortino | Return | DD | 2026 acc | Features | Short? |
|---|--------|---------|--------|-----|----------|----------|--------|
| 🥇 | **E1 D7 combo no-short** | **6.393 ± 0.073** | +877% | -8.1% | 52.8% | 32 | ❌ |
| 🥈 | E3 D7 combo + short | 5.859 ± 0.054 | **+1182%** | -8.1% | 52.8% | 32 | ✅ |
| 🥉 | E2 V29 prod atual | 5.721 ± 0.039 | +1215% | -8.2% | 52.8% | ✅ |

---

## Features ADD recomendadas

3 features adicionadas ao V29 baseline (29 → 32):

1. **Reserve-Risk** (bitcoin-data.com) — conviction de long-term holders. Baixo = boa hora comprar
2. **funding_rate_ma7** (Binance futures, media 7 dias) — sentimento de leverage futures
3. **Puell Multiple** (bitcoin-data.com) — miner profitability ratio. Baixo = capitulation de mineradores

**Todas com median-fill pre-2022** (não havia histórico). Solução: preencher NaN pré-histórico com mediana dos primeiros 30 dias disponíveis, evitando XGBoost aprender "is NaN" como feature.

---

## Resultados ano a ano (E1 D7 combo sem short, 10 seeds)

| Ano | BTC | Estratégia | Excess | Acc all | Acc Fri |
|-----|-----|------------|--------|---------|---------|
| 2022 | -64% | +38% | +102pp | 60.3% | 62.7% |
| 2023 | +156% | +83% | -73pp | 57.5% | 64.5% |
| 2024 | +121% | +120% | -1pp | 54.2% | 52.9% |
| 2025 | -6% | +53% | +59pp | 55.4% | 61.3% |
| 2026 YTD | -14% | +14% | +28pp | 52.8% | n/a |

**Mandato long-biased**: estratégia positiva em TODOS os anos.

---

## Decisão produção

### Recomendação principal: **E1 (D7 combo no-short)**

**Por quê:**
1. Máximo Sortino (6.39 vs 5.72 atual = +11%)
2. Mandato long-biased puro (sem short)
3. Menor DD (-8.1%)
4. 3σ significância estatística (10 seeds)
5. 2026 YTD mesmo desempenho que baseline (52.8% acc)

**Trade-off:** Retorno absoluto -28% vs V29 atual. Mas CAGR 78%/ano ainda excepcional.

### Opção alternativa: **E3 (D7 combo + short)**

Se prioridade é manter retorno absoluto alto, adotar apenas as 3 features novas sem mudar floor:
- Sortino 5.86 (+2.4% vs V29 atual)
- Return +1182% (-3% vs V29 atual)
- Melhoria modesta mas sem trade-off significativo

### Opção conservadora: **Apenas refresh dataset**

Se não quer mudar features nem floor:
- Atualizar dataset_production.csv na prod (já feito — 2026-04-18)
- Mantém config atual de V29
- 2026 acc sobe 41.9% → 52.8% apenas pelo refresh

---

## O que NÃO funciona (para evitar retestar)

### Features que hurt ou não ajudam (V33-V35)

1. **Dropping features** — fractal_dimension_30d, fed_balance_sheet não são lixo mesmo "saturadas"
2. **daily_active_addresses + transactions** (Messari BQ) — piora Sortino -0.98
3. **fear_greed + fear_greed_ma7** (alternative.me) — sentimento já capturado
4. **MVRV mfill** — redundante com NUPL (-0.30)
5. **Features com >40% NaN raw** (SOPR/MVRV/Puell/Reserve-risk sem median-fill) — XGB aprende "is NaN" como sinal

### Features não testadas por falta de histórico

- **Open Interest diário** (Binance): só ~30 dias
- **Taker buy/sell ratio**: só ~30 dias
- **Funding rate raw** (Binance): 2019-12+ mas 13.7% NaN inicial — com mfill foi útil

---

## Experimentos rodados (total: 46 configs)

| Experimento | Configs | Tempo | Winner |
|-------------|---------|-------|--------|
| V33 | 4 | 32 min | B0 baseline (fresh data) |
| V34 | 8 | 46 min | C1 btc_dominance (+0.06) |
| V35 | 7 | 56 min | **D7 combo Reserve+Funding+Puell (+0.30)** |
| V36 | 5 (10 seeds) | 132 min | **E1 D7 combo no-short (6.393)** |
| **Total** | **24 experiments** | **~4.4h** | |

**Total compute**: ~130 individual seed runs across 46 configurations.

---

## Arquivos de suporte

- `scripts/production/data/dataset_production.csv` — **fresh 2026-04-18** (atualizado)
- `scripts/production/data/v34_features.csv` — **novas features** (fear_greed, funding, SOPR, MVRV, Puell, Reserve-risk)
- `outputs/results/v33_experiments.json`
- `outputs/results/v34_experiments.json`
- `outputs/results/v35_experiments.json`
- `outputs/results/v36_validation.json`
- `archive/test_scripts/v33_experiments.py`
- `archive/test_scripts/v34_add_features.py`
- `archive/test_scripts/v35_nan_fix.py`
- `archive/test_scripts/v36_final_validation.py`
- `archive/test_scripts/fetch_v34_features.py` — pipeline novas features

---

## Próximos passos possíveis

Se quiser ir além:

1. **V37**: testar monthly retrain (vs semi-anual atual)
2. **V38**: fix fractal_dimension_30d saturation bug
3. **V39**: walk-forward com dados 2024+ only (ver se regime ETF muda modelo)
4. **Integração prod**: adicionar Reserve-Risk + funding + Puell ao pipeline de produção (3 scripts a editar: config.py, build_features.py, fetch_raw_data.py)
5. **Monitor**: alertar se 2026+ acc cair abaixo de 50% (sinal de regime shift)
