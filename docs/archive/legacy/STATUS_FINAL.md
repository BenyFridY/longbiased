# Status Final (2026-04-23 — atualizado apos overfit audit)

**Estado atual:** **H1 em produção** (trocado de H2 em 2026-04-22 apos auditoria), risk controls integrados, docs consolidados.

**Docs na ordem:**
1. [`MODEL_FINAL.md`](./MODEL_FINAL.md) — spec completo do modelo final ⭐
2. [`OVERFIT_TESTS_2026-04-22.md`](./OVERFIT_TESTS_2026-04-22.md) — 7 testes de overfit
3. [`MODEL_AUDIT_2026-04-20.md`](./MODEL_AUDIT_2026-04-20.md) — auditoria H2 anterior

---

## Modelo em produção

**E1 D7 combo + H1 balanced (no-short) + risk controls** — 32 features, XGBoost ensemble 80+80 bags.

### Config (`scripts/production/config.py`)

```python
K_REGIME = {'BULL': 60, 'MILD': 30, 'BEAR': 15}   # H1 — robusto em frozen train
ALLOC_MIN = 0.0          # no-short (V31.7)
ALLOC_MAX = 1.0
SIGMOID_SCALE = 15
HORIZON = 3
BAGS = 80
REBAL_DOW = [4]          # Friday
EMERGENCY_THRESHOLD = 0.08
RETRAIN = 'semi'         # Jan + Jul
```

**Risk controls ativos** (`scripts/production/risk_management.py`):
- Kill switch: DD total <= -12% -> alloc <= 15%
- Acc derisk: rolling 12w acc < 48% -> alloc x 0.5
- PSI monitor: informacional (nao auto-adjusta)

32 features = 19 base (V29) + 9 TOP_ADD (V22) + 1 EXTRA + 3 V36 NEW (reserveRisk, funding_rate_ma7, puellMultiple).

---

## Performance

### Walk-forward OOS (2025-10-03 → 2026-04-19, 30 rebals)

| | Estratégia | BTC |
|---|---|---|
| Return | **+29.79%** | −38.83% |
| Excess | **+68.62pp** | — |
| Max DD | **−4.01%** | −44.22% |

### Backtest 4 anos (V36 10-seed validation)

| Metric | H1 (alternativa) | **H2 (atual)** |
|---|---|---|
| Sortino | **6.43** | 5.72 |
| Return | +893% | **+1228%** |
| DD | **−8.0%** | −9.2% |
| CAGR | 79% | 95% |

### Ano a ano (H2)

| Ano | Strat | BTC | Excess |
|---|---|---|---|
| 2022 bear | +42% | −64% | **+106pp** |
| 2023 bull | +99% | +156% | −57pp |
| 2024 bull | +135% | +121% | +14pp |
| 2025 flat | +65% | −6% | **+71pp** |
| **2026 YTD** | **+18%** | **−17%** | **+35pp** |

Positiva em todos os 5 anos.

---

## Decisão aberta: H1 vs H2

| | **H1** (K=60/30/15) | **H2** (K=100/50/20) |
|---|---|---|
| Filosofia | Max Sortino | Max Return |
| Sortino | **6.43** | 5.72 |
| Return 4y | +893% | **+1228%** |
| DD | **−8.0%** | −9.2% |

**H2 ativo desde 2026-04-20.** Troca: 1 linha em `config.py:49`, sem retrain.

---

## Pipeline produção

```bash
python scripts/production/run_daily.py                    # rebal diário
python scripts/production/run_daily.py --retrain          # force retrain (só Jan/Jul)
python scripts/production/walkforward_backtest.py         # audit OOS
python scripts/production/walkforward_backtest.py --compare   # H1 vs H2
```

Execução:
1. `fetch_raw_data.py` — 12+ fontes (Binance, yfinance, FRED, bitcoin-data.com, Messari BQ)
2. `bootstrap_from_original.py` — hybrid dataset + V36 backfill + median-fill
3. `generate_signal.py` — sinal diário (semi-annual retrain schedule)

**Dataset status (2026-04-19):** 2666 rows, 32 features, 0 NaN, última data 2026-04-19.

---

## Rebal history resumo

29 rebals em signal_history.csv (Oct 2025 → Apr 2026) + 1 backfilled walk-forward (2026-04-17).

**Últimos 4 rebals (H2 walk-forward):**

| Data | Regime | Pred 3d | P_up | Alloc | BTC wk | Strat wk |
|---|---|---|---|---|---|---|
| 03-27 | BEAR | +1.94% | 81% | 38% | +0.84% | +0.46% |
| 04-03 | BEAR | +3.51% | 91% | 70% | +8.96% | +6.36% |
| 04-10 | BEAR | +3.50% | 87% | 70% | +5.63% | +4.01% |
| 04-17 | BEAR | +1.84% | 74% | 36% | −2.98% | −1.07% |

**Sinal atual (2026-04-19 close, domingo):** HOLD 54.5% (rebal 04-10 live). Próxima sexta 04-24 aplica config H2 novo.

---

## Garantias de walk-forward

Pipeline produção é walk-forward safe:
- Retrain só em Jan/Jul (`RETRAIN_MONTHS=[1,7]`)
- Gap=5d no target previne leakage
- Entre retrains, cache reutilizado
- Prediction no `X[-1:]` com modelo treinado antes

**Nunca use `--retrain` fora do schedule + backtest retroativo.** Use `walkforward_backtest.py` pra audits históricos.

---

## Próximos passos (2026-04-24 em diante)

1. **Sexta 04-24**: rodar `run_daily.py` → primeiro rebal real com H2 ativo.
2. **Monitorar 2026 acc**: se cair <50% por 8+ semanas, revisitar features.
3. **Jul 2026**: retrain semi-annual automático (data 2019 → 2026-06-30).
4. **Eventual**: Cost modeling (5-10bps), backtests em stress windows (Mar 2020, Nov 2022).

---

## Memória persistente (AI bot)

Em `.claude/projects/.../memory/`:

- `project_h2_deployed.md` — decisão H2, OOS results
- `project_v36_winners.md` — top 3 candidatos V36
- `project_v32_battery.md` — 46 técnicas testadas Dia 3
- `project_v29_v31_results.md` — Dia 1-2 pipeline
- `feedback_not_retest.md` — 36 técnicas a não retestar
- `reference_bq_signals.md` — BQ signals table
