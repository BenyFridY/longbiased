# Pipeline de Producao — Manual de Operacao

## Modelo Atual: E1 D7 + H1 (no-short, com risk controls)

Este pipeline gera diariamente o sinal de alocacao BTC/CDI usando o modelo
**E1 D7 (32 features) + H1 K-config + 3 risk controls**, em producao desde
**2026-04-22** (apos auditoria de overfit que motivou H2 -> H1).

**Configuracao (em `config.py`):**
- 32 features (29 V29 originais + 3 V36 on-chain: reserveRisk, funding_rate_ma7, puellMultiple)
- K_REGIME = **{BULL: 60, MILD: 30, BEAR: 15}**  *(H1, antes era H2 100/50/20)*
- Floor = 0 (no-short, long-biased mandate)
- Sigmoid confidence scale = 15
- Horizon = 3 dias
- Bags = **160 por ensemble** (160 reg + 160 cls = **320 modelos**) — atualizado 2026-04-29 (era 80, i.e. 80+80=160)
- Rebalance = sexta-feira (ou emergencia se |daily_ret| > 8%)
- Retrain = semi-anual (Jan + Jul)

**Risk controls** (em `risk_management.py`, aplicados automaticamente em
`generate_signal.py`):

| Controle | Regra | Acao |
|----------|-------|------|
| Kill switch | DD acumulado <= -12% | Cap alloc em 15% |
| Acc de-risk | Rolling 12w acc < 48% **E** conf média 12w > 80% | alloc x 0.5 |
| PSI monitor | PSI > 3.0 em 3+ features | Warn (sem auto-adjust) |

**Metricas validadas** (10-seed, Ultra 9, XGBoost 3.2.0, BAGS=160, BCB CDI, 4.28y OOS, gross — 4 bps ~ -0.6pp CAGR):
- CAGR: **+57.3% ± 0.3%**
- Sortino daily: **3.53 ± 0.06**
- Sharpe excess (daily): **2.47 ± 0.01**
- Max DD daily: **-7.14% ± 0.25%**

**Live realista (apos deflation 38 trials):** CAGR 25-40%, Sortino 1.5-2.5,
DD daily -15 a -25%.

**2026 YTD live OOS (105 dias, 17 rebals)**: Estrategia **+19.67%** vs BTC
**-14.36%** vs CDI **+4.09%** — modelo treinado em Jan/2026, strict OOS.

Spec completa: [`../../docs/MODEL_FINAL.md`](../../docs/MODEL_FINAL.md).
Auditoria de overfit: [`../../docs/OVERFIT_TESTS_2026-04-22.md`](../../docs/OVERFIT_TESTS_2026-04-22.md).

---

## Fluxo Diario

**Todo dia apos 00:00 UTC** (quando os candles diarios fecham):

```bash
python scripts/production/run_daily.py
```

Executa automaticamente:
1. `fetch_raw_data.py` — busca dados de **12+ fontes** (incremental)
2. `bootstrap_from_original.py` — hybrid: enhanced base + build_features para dias novos
3. `generate_signal.py` — gera alocacao do dia + aplica risk controls

Tempo: ~2-3 minutos.

### Opcoes:

```bash
python scripts/production/run_daily.py              # normal
python scripts/production/run_daily.py --retrain    # forca retrain do modelo
python scripts/production/run_daily.py --full       # rebuild completo dos dados
```

---

## O Que o Sinal Mostra

```
=================================================================
  SIGNAL — 2026-04-24 (Fri)  [E1-D7 K=60/30/15]
=================================================================
  BTC Price:      $XX,XXX
  Daily Return:   +X.XX%
  Regime:         BULL (K_base=60)
  Prediction:     +X.XXX% (3d return)
  P(up):          XX.X% (confidence: XX%)
  K effective:    XX (base 60 x 0.XX)

  >> Action:      REBALANCE (Friday)
  >> Allocation:  +XX.X% BTC / XX.X% CDI
  Rolling acc 12w: XX.X% (threshold 48%)
  Current DD:     -X.XX% (kill at -12%)

  Last rebalance: 2026-04-17
  Model trained:  2026-01-01
  Next retrain:   2026-07-01 (64 days)
  Is Friday:      >> YES — REBALANCE DAY
  Emergency:      no (threshold: >8%)
=================================================================
```

Quando algum risk control dispara, aparece um bloco extra:

```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  RISK MANAGEMENT CONTROLS ACTIVATED
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  >> ACCURACY DE-RISK: 45.2% < 48%. Halving alloc (0.5x).
  Original alloc: +XX.X%
  Adjusted alloc: +YY.Y%
```

---

## Quando Rebalancear

1. **Sexta-feira** — rebalance semanal programado
2. **Emergency** — quando |retorno diario do BTC| > 8%

Nos outros dias: HOLD (manter ultima sexta).

**Executar o rebalance (comprar/vender) eh manual**. O pipeline so gera o sinal.

---

## Quando Retreinar

Automaticamente **semi-anual** (Janeiro e Julho). Forcar:
```bash
python scripts/production/run_daily.py --retrain
```

**Importante**: nao retreinar fora do schedule semi-anual sem motivo —
quebra a comparabilidade com walk-forward backtest.

---

## Arquitetura do Dataset

O `dataset_production.csv` eh construido em 2 camadas:

1. **Base historica** (2019-01-01 a 2026-03-03): `outputs/feature_selection/dataset_enhanced.csv`
   - Dataset original validado (267 features, 2619 dias)
   - As features V29 (29) sao extraidas daqui
   - **Nunca modificado** — ground truth

2. **Dias novos** (pos 2026-03-03): calculados por `build_features.py` a partir de `raw_data.csv`
   - Features alinhadas com o pipeline original

3. **V36 NEW features** (reserveRisk, funding_rate_ma7, puellMultiple):
   - Backfilled no historico inteiro via `build_features.py`
   - **Median-fill pre-2022-04-19** (data start), entao predicoes pre-2022 usam fill, nao real

`bootstrap_from_original.py` combina as 3 camadas.

---

## Arquivos na Pasta

```
scripts/production/
+-- run_daily.py                  # Entry point unico
+-- config.py                     # Parametros + 32 features (K=H1)
+-- fetch_raw_data.py             # Busca dados de 12+ fontes
+-- bootstrap_from_original.py    # Builder do dataset hybrid
+-- build_features.py             # Calcula 32 features
+-- generate_signal.py            # Gera sinal diario + chama risk controls
+-- risk_management.py            # Kill switch + acc derisk + PSI monitor
+-- training.py                   # Helpers de treino
+-- walkforward_backtest.py       # Walk-forward OOS validator (re-validacao)
+-- INSTRUCTIONS.md               # Este arquivo
+-- data/
|   +-- raw_data.csv              # Dados brutos das fontes
|   +-- dataset_production.csv    # 32 features, hybrid (bootstrap)
|   +-- cached_models.pkl         # 160 reg + 160 cls XGBoost
|   +-- signal_history.csv        # Historico de sinais (alimenta DD/acc)
+-- archive/                      # Backups antigos + experiments
    +-- experiments/              # overfit_test_*, deflated_sharpe, audits
```

---

## Fontes de Dados (12+ fontes)

| # | Fonte | Dados | Desde |
|---|-------|-------|-------|
| 1 | Binance Spot | OHLCV BTC/USDT | 2019 |
| 2 | Binance Futures | Futures close (basis) | 2019-09 |
| 3 | Binance Futures | **Funding rate (V36)** | 2019-12 |
| 4 | BQ Messari OHLCV | volume_usd (agregado) | 2019 |
| 5 | BQ Messari Financial | miners_revenue_usd | 2019 |
| 6 | BQ Messari Futures | open_interest, futures_trade_count | 2019 |
| 7 | yfinance | ETH, Gold, Copper | 2019 |
| 8 | FRED | M2, Fed Balance Sheet | 2019 |
| 9 | BGeometrics | NUPL | 2019 |
| 10 | bitcoin-data.com | **Reserve-Risk (V36)** | 2022 |
| 11 | bitcoin-data.com | **Puell Multiple (V36)** | 2022 |
| 12 | DefiLlama | Stablecoin supply | 2019 |
| 13 | CoinMetrics | Hash rate | 2019 |
| 14 | Blockchain.com | Miners revenue (fallback) | 2019 |

**Features V36 (2022-)**: pre-historia preenchida com mediana dos primeiros 30
dias disponiveis (median-fill).

**Dependencias**: `FRED_API_KEY` no `.env`, `bq` CLI autenticado para BigQuery.

---

## Re-validar / Auditar (rapido)

```bash
# Re-validar walk-forward OOS (~30min, compara H1 vs H2)
python scripts/production/walkforward_backtest.py --compare

# Audit completo com daily DD (rapido, usa preds existentes)
python scripts/production/archive/experiments/final_audit_daily_dd.py

# Deflated Sharpe (Bailey-Prado)
python scripts/production/archive/experiments/deflated_sharpe.py

# Smoke test do risk_management (kill switch / acc / PSI)
python scripts/production/risk_management.py
```

---

## Rollback pra V29 (com short)

Se precisar voltar pro modelo antigo (anterior a V36/E1): os backups `.pre_e1`
**nao existem mais** no repo — use o historico git. Ache o ultimo commit antes
do V36/E1 e restaure os 3 arquivos:

```bash
git log --oneline -- scripts/production/config.py        # achar o commit pre-V36/E1
git checkout <commit> -- scripts/production/config.py \
    scripts/production/build_features.py scripts/production/fetch_raw_data.py
python scripts/production/run_daily.py --full --retrain
```

---

## Rebuild Completo

Se dados corromperem:
```bash
rm scripts/production/data/dataset_production.csv
python scripts/production/run_daily.py --full --retrain
```

---

## Troubleshooting

| Problema | Solucao |
|----------|---------|
| FRED vazio | Verificar `FRED_API_KEY` no `.env` |
| BigQuery falha | `gcloud auth login` + `bq ls` |
| bitcoin-data.com 429 (rate limit) | Esperar 30s e tentar |
| BGeometrics vazio | Rate-limited, tentar mais tarde |
| Model stale | Rodar com `--retrain` |
| Dataset corrompido | `rm data/dataset_production.csv && run_daily.py --full` |
| Features V36 missing | Conferir se `reserveRisk`, `puellMultiple`, `funding_rate_ma7` estao em `raw_data.csv` |
| Kill switch ativo | DD <= -12%; investigar regime shift, aguardar recuperacao |
| Acc derisk ativo | Rolling 12w acc < 48%; checar se features migraram (PSI) |

---

## Historico de Versoes

| Versao | Data | Mudanca | Sortino | DD daily |
|--------|------|---------|---------|----------|
| V22 | 2026-02 | Baseline 37 features | 4.05 | -13% |
| V23 | 2026-03 | + K=60/30/15, sigmoid=15 | 4.70 | -10.7% |
| V25 | 2026-04-15 | + YoY fixes, 37 features | 4.95 | -10% |
| V29 | 2026-04-18 | Pruning 37->29, fracdiff d=0.3 | 5.84 | -10% |
| V31.7 | 2026-04-19 (AM) | V29 + floor=0 (no short) | 6.19 | -8% |
| E1 D7 (V36) | 2026-04-19 (PM) | + 3 on-chain (reserveRisk, funding_rate_ma7, puellMultiple) | 6.39 | -8.1% |
| **H2** | 2026-04-20 | E1 D7 + K=100/50/20 | 5.61 (4.3y) | -4.0% (weekly) / -9.1% (daily) |
| **H1 (atual)** | **2026-04-22** | **K=60/30/15 + risk controls (kill switch, acc derisk, PSI)** | **3.53 (daily)** / 7.79 (weekly) | **-7.14% (daily)** / -2.46% (weekly) |

Sortino/DD na linha H1 reportados em **daily** (canonico, MODEL_FINAL.md).
Linhas anteriores misturam daily e weekly (historico das docs por versao).

Detalhes da migracao H2 -> H1: [`../../docs/OVERFIT_TESTS_2026-04-22.md`](../../docs/OVERFIT_TESTS_2026-04-22.md).
