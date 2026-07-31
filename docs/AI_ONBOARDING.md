# AI Onboarding — Long-Biased BTC Strategy

**Indice para navegar o projeto rapidamente.**
Last updated: 2026-06-09

---

## O que e este projeto em 3 linhas

Sistema de alocacao dinamica BTC/CDI usando XGBoost ensemble (160+160 bags).
Modelo prediz retorno BTC 3 dias a frente, alocacao derivada de prediction +
regime detection (SMA50/200) + confidence scaling. Rebalance semanal (sexta) +
emergency quando \|daily_ret\| > 8%. Universo: BTC spot (long only),
CDI como risk-free. **MODELO FINAL FECHADO 2026-04-29 — pronto pra paper trade.**

---

## Ordem de leitura

1. **[`MODEL_FINAL.md`](./MODEL_FINAL.md)** — spec do modelo em producao ⭐
2. **[`OVERFIT_TESTS_2026-04-22.md`](./OVERFIT_TESTS_2026-04-22.md)** — auditoria de overfit (7 testes)
3. Este doc (`AI_ONBOARDING.md`) — indice e quickstart

Tudo antigo (pipelines V02-V38, auditorias pre-H1, papers) esta em `archive/`.

---

## Config atual (H1, aplicado 2026-04-22)

```python
# scripts/production/config.py
K_REGIME = {'BULL': 60, 'MILD': 30, 'BEAR': 15}  # H1 — robusto em frozen train
ALLOC_MIN = 0.0      # no-short
ALLOC_MAX = 1.0
SIGMOID_SCALE = 15
HORIZON = 3
BAGS = 160               # 2026-04-29: 80 → 160 (estabilidade)
REBAL_DOW = [4]      # Friday
EMERGENCY_THRESHOLD = 0.08
RETRAIN = 'semi'     # Jan + Jul
```

**32 features** (V29 base 29 + V36 on-chain 3: reserveRisk, funding_rate_ma7, puellMultiple).

**Risk controls ativos** (`scripts/production/risk_management.py`):
- Kill switch: DD <= -12% -> cap alloc at 15%
- Acc derisk: rolling 12w acc < 48% -> alloc * 0.5
- PSI monitor: informational only

---

## Metricas validadas (Ultra 9, XGBoost 3.2.0, BAGS=160, BRL + 4 bps cost, 4.28y OOS)

- CAGR: **+57.3% ± 0.3%** (10-seed std)
- Sortino daily: **3.53 ± 0.06**  (variance menor que 80 bags: std 0.10 -> 0.06)
- Sharpe excess (daily): **2.47 ± 0.01**
- Max DD daily: **-7.14% ± 0.25%**

**2026 YTD live OOS (105 dias)**: estrategia **+19.67%** vs BTC **-14.36%**

Expectativa live (post-deflation): CAGR 25-40%, Sortino 1.5-2.5, DD -15 a -25%.

---

## Estrutura do repo

```
longbiased-beny/
|- scripts/production/           # CODIGO ATIVO
|  |- run_daily.py               # Entry point diario
|  |- generate_signal.py         # Pipeline V23 + risk controls
|  |- risk_management.py         # NOVO: kill switch + derisk
|  |- config.py                  # K=H1, 32 features
|  |- fetch_raw_data.py
|  |- bootstrap_from_original.py
|  |- build_features.py
|  |- training.py
|  |- walkforward_backtest.py
|  |- INSTRUCTIONS.md
|  |- data/
|  |  |- dataset_production.csv
|  |  |- raw_data.csv
|  |  |- cached_models.pkl
|  |  |- signal_history.csv
|  |- archive/
|     |- experiments/            # Scripts de teste (overfit_*, final_audit, deflated_sharpe, etc)
|
|- docs/                         # SO 3 docs ativos
|  |- MODEL_FINAL.md             # Spec do modelo ⭐
|  |- OVERFIT_TESTS_2026-04-22.md # 7 testes
|  |- AI_ONBOARDING.md           # Este arquivo
|  |- archive/                   # Historia
|     |- pipelines/              # V02-V38 progressao
|     |- legacy/                 # Audits antigos, STATUS, V23, SESSION
|     |- papers/                 # Artigos, PROJECT_DESCRIPTION
|
|- outputs/results/
|  |- overfit_tests/             # 6 CSVs com resultados dos testes
|  |- walkforward_backtest.csv
|  |- horizon_ablation_4y.csv    # 248 rebals 2022-2026
|  |- retrain_parallel_*.csv     # Retrain frequency experiments
|  |- logs/
|
|- archive/                      # Historia antiga (V02-V22 pipelines)
```

---

## Como rodar

```bash
# Ver sinal do dia (com risk controls)
python scripts/production/run_daily.py

# Ver sinal sem refetch (modo debug)
python scripts/production/generate_signal.py

# Forcar retrain (so em 01-Jan ou 01-Jul, alinhado com schedule)
python scripts/production/run_daily.py --retrain

# Walk-forward backtest OOS (compare H1 vs H2)
python scripts/production/walkforward_backtest.py --compare

# Auditoria completa com DD diario (usa preds existentes, rapido)
python scripts/production/archive/experiments/final_audit_daily_dd.py

# Deflated Sharpe (Bailey-Prado)
python scripts/production/archive/experiments/deflated_sharpe.py
```

---

## Mudancas recentes

### Sessao 2026-04-22/23 (audit + H1 deploy)
1. K: H2 (100/50/20) -> H1 (60/30/15)
2. Risk controls integrados (kill switch, acc derisk, PSI)
3. 7 testes overfit, edge validado (0/100 shuffles, p<0.01)

### Sessao 2026-04-28/29 (final validation)
1. **Reconciliação de números**: descobertos numeros antigos eram single-seed
   em maquina antiga (CPU-dependent). 10-seed em Ultra 9 + XGBoost 3.2.0 pinado
   eh o canonical agora. Numeros antigos NAO reproduziveis (CPU non-determinism).
2. **Confidence-weighted acc derisk**: gate adicional - so derisk se modelo
   estava confiante e errado (acc<48% AND avg conf>80%). Halves false positives.
3. **BAGS 80 -> 160**: 10-seed validation mostrou variance menor (Sortino std
   0.10 -> 0.06), mean ~igual. Modelo MUITO mais previsivel em live.
4. **Cost: 4 bps** (era 8 bps assumption, ajustado pra realidade BRL institucional)
5. **10 alternativas testadas, 2 aceitas**:
   ✅ conf-weighted derisk, ✅ bags=160
   ❌ Huber loss, drop V36, vol overlay, rampa graduada, quantile regression,
      K_BULL=75, rolling vs expanding, regime-triggered retrain, emergency thresh
6. Modelo confirmado em local optimum no backtest landscape.

Ver [`MODEL_FINAL.md`](./MODEL_FINAL.md) para detalhes.

---

## O que NAO fazer

1. Nao retreine fora de `run_daily.py --retrain` (quebra walk-forward).
2. Nao mude `FEATURES_37` sem retrain — modelos ficam incompativeis.
3. Nao mude `ALLOC_MIN` para negativo — sair do mandato no-short.
4. Nao tire risk controls — foram adicionados apos audit por razoes especificas.
5. Consulte `archive/pipelines/` se quiser historia de experimentos V02-V38.

---

## Troubleshooting rapido

| Sintoma | Causa provavel | Fix |
|---|---|---|
| Sinal nao atualiza | dataset stale | `python run_daily.py --full` |
| Features missing | bootstrap nao rodou | `rm dataset_production.csv && run_daily.py --full` |
| Sortino caiu | dataset/feature stale | Verificar ultima data + PSI |
| Alloc sempre 0 | pred negativo (normal) | Modelo esta em modo defensivo — OK |
| Kill switch ativo | DD <= -12% | Investigar regime shift, aguardar recuperacao |

---

## Contato

- Repo: `longbiased`
- Deploy: H1 + 160 bags + risk controls, desde 2026-04-29
- Proximo rebal: sextas (semanal) + emergency (>8%)
- Proximo retrain: 2026-07-01 (semi-annual)
- Status: PAPER TRADE phase ate Q3/2026

## Nota de reprodutibilidade CRITICA

XGBoost training NAO eh deterministico entre CPUs diferentes. Cache atual
(scripts/production/data/cached_models.pkl) foi treinado em **Intel Ultra 9
em 2026-04-29 com XGBoost 3.2.0**. Se for trocar maquina ou XGBoost version,
modelos vao produzir predicoes diferentes mesmo com mesma seed.

Solucao: pinar XGBoost (`requirements.txt: xgboost==3.2.0`) e retreinar em
maquina alvo antes de live trading.
