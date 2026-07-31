# V33 — Data Audit & Incremental Improvements (2026-04-19)

**Motivação**: 2026 YTD acurácia caiu pra 41.9% (vs 54-60% histórico). Investigar causas e corrigir.

**Baseline antes de mexer**: V31.7 / baseline_floor0 — Sortino 6.19 ± 0.07 (10 seeds), Ret +824%.

---

## Investigação inicial (2026-04-19 13:20)

### Dataset status
- `dataset_enhanced.csv`: último dia **2026-03-03** (47 dias defasado)
- `dataset_production.csv`: último dia **2026-04-18** (1 dia defasado — aceitável)

### Bugs identificados

1. **Dias faltando em raw data**: Apr 15 (Wed), Apr 17 (Fri) — bug em `fetch_raw_data.py`
2. **`fractal_dimension_30d` saturado em 2.000** por 20+ dias — cálculo Higuchi saturando
3. **`fed_balance_sheet` release lag** — FRED só atualiza quartas, gera staleness artificial

### Features com baixa variação em 2026 YTD

| Feature | 2026_std / 2024_std | Status |
|---------|---------------------|--------|
| fed_balance_sheet | 0.15 | low info real (Fed hold) |
| fed_fracdiff_05 | 0.21 | derivada de fed_bs |
| basis_ma7 | 0.10 | real (contango compressed) |
| basis_pct | 0.18 | real |
| eth_btc_ratio | 0.28 | real |
| fractal_dimension_30d | N/A | **BUG — frozen 2.0** |

### BQ tables novas descobertas

| Tabela | Histórico | Features utilizáveis |
|--------|-----------|---------------------|
| `messari_total_marketcap_and_coin_dominance` | 2015+ (4125d) | btc_dominance |
| `messari_network_activity` | 2009+ (6336d) | daily_active_addresses, daily_transactions |
| `messari_asset_futures_funding_rate` | 2026+ (133d) | funding_rate — MUITO CURTO |
| `messari_asset_futures_volume` | ? | volume buy/sell taker agg |

---

## Experimentos

Serão incrementais. Cada mudança testada com 3 seeds no **dataset atualizado 2026-04-18** (106 dias 2026 YTD vs 62 antes).

| # | Experimento | Sortino 10s | Ret | DD | 2026 Acc | 2026 Ret | Decisão |
|---|-------------|-------------|-----|-----|----------|----------|---------|
| 0 | V31.7 baseline (stale data) | 6.194 | +824% | -8.2% | 41.9% | +6.7% | ref |
| B0 | **Fresh data (29 feats)** | 6.120 | +891% | -8.3% | **52.8%** | +13.8% | ✅ **+10.9pp acc só com refresh** |
| B1 | Drop fractal+fed+fed_fracdiff (26 feats) | 5.577 | +689% | -8.6% | 50.9% | +13.5% | ❌ -0.54 Sortino |
| B2 | B1 + btc_dominance (27 feats) | 5.675 | +699% | -9.0% | 50.0% | +12.8% | ❌ -0.45 Sortino |
| B3 | B2 + network_activity (29 feats) | 5.138 | +682% | -9.1% | 50.9% | +13.0% | ❌ -0.98 Sortino |

## Conclusões V33 (32.3min)

1. **Refresh dataset = +10.9pp 2026 acc** (de 41.9% → 52.8%). Maior ganho single-intervention
2. **"Features staladas" NÃO são lixo** — dropping fractal/fed features custou 0.54 Sortino
3. **Network activity prejudica** — daily_active_addresses + transactions adicionaram ruído
4. **btc_dominance neutro** quando combinado com drops (ainda pior que B0)

Next: V34 adiciona features ON TOP de B0 (mantém 29 originais).

## V34 Resultados (46.4min)

| # | Config | Sortino | Ret | 2026 acc | vs B0 |
|---|--------|---------|-----|----------|-------|
| 🥇 | **C1 btc_dominance** | **6.175** | +897% | 50.9% | **+0.06** ✅ |
| - | C2 fear_greed (+ma7) | 5.77 | +870% | 51.9% | -0.35 |
| - | C3 funding_rate (+ma7) | 5.64 | +774% | 50.9% | -0.48 |
| - | C5 MVRV | 2.26 | +171% | 50.9% | -3.86 (NaN) |
| - | C4 SOPR | 1.97 | +156% | 51.9% | -4.15 (NaN) |
| - | C7 Reserve-Risk | 1.93 | +150% | 50.9% | -4.19 (NaN) |
| - | C6 Puell Multiple | 1.75 | +142% | 50.9% | -4.37 (NaN) |
| - | C8 ALL 8 new | 1.67 | +131% | 50.9% | -4.45 (compounded NaN) |

## V34 Conclusões

1. **btc_dominance é o único que ajuda** — +0.06 Sortino (marginal mas real)
2. **fear_greed não ajuda mesmo com 0% NaN** — sentimento já capturado via outras features
3. **funding_rate piora** mesmo com só 13.7% NaN
4. **SOPR/MVRV/Puell/Reserve-risk CATASTRÓFICO** com 45% NaN — XGBoost aprende "se NaN → era cedo" e não generaliza
5. **C8 combo piora compostamente** — NaN compounds

## Next: V35 — Median-fill NaN features

Preencher NaN pre-histórico com mediana dos primeiros 30 dias disponíveis. Evita "NaN signal" que XGBoost aprende.

## V35 Resultados (56min)

Todos baseados em B0 + feature(s) adicionada(s), median-filled onde necessário.

| # | Config | Sortino | Ret | 2026 acc | all acc | Δ vs B0 |
|---|--------|---------|-----|----------|---------|---------|
| 🥇🥇 | **D7 combo (Reserve+Funding+Puell)** | **6.423** | +872% | 52.8% | **56.80%** | **+0.30** ✅✅ |
| 🥇 | D4 Reserve-Risk mfill | 6.232 | +907% | 51.9% | 55.97% | +0.11 ✅ |
| 🥈 | D5 funding_rate_ma7 mfill | 6.212 | +859% | 52.8% | 56.92% | +0.09 ✅ |
| 🥉 | D3 Puell mfill | 6.193 | +883% | 50.9% | 55.90% | +0.07 ✅ |
| - | D6 btc_dominance | 6.175 | +897% | 50.9% | 56.16% | +0.06 ✅ |
| - | D1 SOPR mfill | 6.151 | +879% | 50.9% | 55.97% | +0.03 ✅ |
| - | B0 reference | 6.120 | +891% | 52.8% | 56.29% | baseline |
| - | D2 MVRV mfill | 5.815 | +806% | 51.9% | 56.22% | -0.30 ❌ |

## Conclusões V35

1. **Median-fill resolve NaN catastrophe** — SOPR passou de 1.97 → 6.15 só com isso
2. **Reserve-Risk é a melhor feature nova individual** (+0.11) — mede conviction dos long-term holders
3. **Combo de 3 features (Reserve+Funding+Puell) tem +0.30 Sortino** — synergy real
4. **MVRV é a única que NÃO ajuda** (-0.30), já captura info similar a NUPL
5. **Return do combo -2% vs baseline** (+872% vs +891%) — trade-off mínimo

## Próximo: V36 — 10-seed validation

Validar com rigor estatístico:
- E0: B0 baseline (ref)
- E1: D7 combo (vencedor V35)
- E2: V29 with short (comparação prod atual)
- E3: D7 combo COM SHORT (ver se short ainda prejudica)
- E4: B0 + btc_dominance

## Novas features pesquisadas (2026-04-19)

### APIs abertas testadas

| API | Endpoint | Feature | Histórico | Status |
|-----|----------|---------|-----------|--------|
| alternative.me | `/fng/` | fear_greed | 2018-01-31+ (2996d) | ✅ Funciona — last 27 (Fear) |
| Binance Futures | `/fapi/v1/fundingRate` | funding_rate | 2019-12-31+ (2302d) | ✅ 8h bars -> daily mean |
| BigQuery Messari | `messari_total_marketcap_and_coin_dominance` | btc_dominance | 2015+ (4125d) | ✅ V33 B2 already testing |
| BigQuery Messari | `messari_network_activity` | daily_active_addresses, daily_transactions | 2009+ (6336d) | ✅ V33 B3 already testing |
| bitcoin-data.com | `/v1/sopr`, `/mvrv`, `/puell-multiple` | SOPR, MVRV, Puell | 2022+ (1461d) | ⚠️ 502 error (retentar) |
| Binance Futures | `/futures/data/openInterestHist` | OI daily | ~30d | ❌ Muito curto |
| Binance Futures | `/futures/data/takerlongshortRatio` | taker buy/sell | ~30d | ❌ Muito curto |

### Contexto atual (2026-04-18)

- Fear&Greed: **27 (Fear)** — últimos 10 dias entre 12-27 (Extreme Fear). Historicamente contrarian bullish
- Funding rate: **-0.012% / 8h** — negativo persistente. Shorts pagando longs = contrarian bullish
- Ambos sinais apontam pra bottom, mas modelo atual não vê
