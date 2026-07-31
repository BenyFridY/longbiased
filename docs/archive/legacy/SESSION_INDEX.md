# Session Index — V29 to V37 (2026-04-17 to 2026-04-19)

Guia de navegacao pra toda a sessao longa de ~3 dias rodando experimentos.

## Modelo em producao (atual)

**E1 D7 Combo (no-short)** — Sortino 6.44, Return +860%, DD -8.1%, 2026 YTD +14.4%
- Config: `scripts/production/config.py` (ALLOC_MIN=0, 32 features)
- Manual: `scripts/production/INSTRUCTIONS.md`
- Relatorio final: `docs/V36_FINAL_REPORT.md`

---

## Cronologia da sessao

### Dia 1 (2026-04-17) — V24-V28 (V23 baseline + tecnicas basicas)

| Versao | Script | Resultado |
|--------|--------|-----------|
| V24 | `v24_backtest.py`, `v24_accuracy.py`, `v24_dow_sweep.py`, `v24_final.py`, `v24_prepare_features.py` | Adicoes de features novas — maioria piorou |
| V25 | `v25_backtest.py`, `v25_apply_fixes.py`, `v25_isolate_fixes.py`, `v25_final.py`, `v25_apply_to_production.py` | Fixes bugs YoY (m2_yoy, fed_bs, m2_3m) + remove OI/FTC |
| V26 | `v26_sweep.py`, `v26_techniques.py` | 9 tecnicas ML (sample weighting, ensemble, etc.) — todas falharam |
| V27 | `v27_deep_techniques.py` | 9 tecnicas mais (reg_alpha, meta-labeling, per-DOW, horizons) — falharam |
| V28 | `v28_advanced.py` | Feature pruning bot-10 + combos — **vencedor: pruning** |

### Dia 2 (2026-04-18) — V29-V31 (refinement + validation)

| Versao | Script | Resultado |
|--------|--------|-----------|
| V29 | `v29_validate.py`, `v29_combos.py`, `v29_oos_test.py`, `v29_apply_to_production.py`, `v29_recompute_fracdiff.py` | **Pruning 37->29 + fracdiff d=0.3. Sortino 5.84 aplicado em prod** |
| V30 | `v30_refinements.py` | Fracdiff d granular — plateau, sem ganho |
| V31 | `v31_v23_pipeline.py`, `v31_extras.py` | V23 pipeline completo + sigmoid/floor/alpha sweeps — **floor=0 vencedor +0.4 Sortino** |

### Dia 3 (2026-04-19) — V32-V38 (novel techniques + data audit + final)

| Versao | Script | Resultado |
|--------|--------|-----------|
| V32 | `v32_exploration.py`, `v32_yoy_analysis.py` | 46 configs novel techniques (Huber, quantile, Kelly, etc.) — **nada bate baseline** |
| V33 | `v33_experiments.py` | Data audit — **refresh dataset = 2026 acc 41.9%->52.8%** |
| V34 | `v34_fetch_features.py`, `v34_add_features.py` | Adicionar fear_greed, funding_rate, SOPR/MVRV/Puell/Reserve-Risk — raw NaN destroi XGBoost |
| V35 | `v35_nan_fix.py` | Median-fill NaN resolve — **D7 combo (Reserve+Funding+Puell) +0.30 Sortino** |
| V36 | `v36_final_validation.py`, `v36_e1_backtest.py` | 10-seed validation — **E1 D7 combo no-short: Sortino 6.39** |
| V37 | `v37_k_and_halving.py` | K sweep + days_since_halving — **E1 K=60/30/15 permanece otimo** |
| V38 | `v38_selic_features.py` | Selic/CDI como feature — **marginal/negativo, descartado** |

---

## Arquivos de documentacao por versao

| Doc | Conteudo |
|-----|----------|
| `V29_FINAL.md` | Decisao V29 + aplicacao em prod |
| `V31_CURRENT_STATE.md` | Snapshot V31 state (V29 em prod + top 5) |
| `V31_FINAL_SESSION.md` | Relatorio V31 completo |
| `V32_FINAL_RESULTS.md` | V32 novel techniques battery (46 configs) |
| `V33_DATA_AUDIT.md` | Audit V33-V35 (data quality + feature additions) |
| **`V36_FINAL_REPORT.md`** | **Relatorio final consolidado (E1 D7 combo wins)** |
| `V37_K_TUNING.md` | K regime sweep + halving feature (E1 wins) |
| `V38_SELIC_TEST.md` | Selic/CDI feature test (nao ajuda) |
| `SESSION_INDEX.md` | Este arquivo (navegacao) |
| `ARTIGO_FINAL_V22.md` | Artigo V22 (pre-sessao) |

---

## Resultados JSON (`outputs/results/`)

| Arquivo | Conteudo |
|---------|----------|
| `v26_techniques.json` | 9 tests V26 |
| `v27_deep.json` | 9 tests V27 |
| `v28_advanced.json` | Pruning + combos V28 |
| `v29_validate.json`, `v29_combos.json` | V29 validation |
| `v30_refinements.json` | V30 fracdiff granular |
| `v31_v23_pipeline.json`, `v31_extras.json` | V31 + V23 pipeline |
| `v32_exploration.json`, `v32_yoy.json` | V32 novel + YoY |
| `v33_experiments.json` | V33 data audit |
| `v34_experiments.json` | V34 feature additions |
| `v35_experiments.json` | V35 NaN-fix |
| `v36_validation.json` | V36 10-seed validation |
| `e1_final_backtest.json` | E1 final backtest (on prod dataset) |
| `v37_experiments.json` | V37 K + halving (em progresso) |

---

## Memoria persistente (`.claude/projects/.../memory/`)

| Arquivo | Tipo | Conteudo |
|---------|------|----------|
| `MEMORY.md` | index | Indice geral |
| `project_alpha_beta_analysis.md` | project | V22 alpha/beta CAPM |
| `project_v23_ultimate_results.md` | project | V23 ultimate test |
| `project_v29_v31_results.md` | project | V29 prod + V31 top 5 |
| `project_v32_battery.md` | project | V32 novel techniques |
| **`project_v36_winners.md`** | project | **V36 final winners** |
| `feedback_not_retest.md` | feedback | 36 tecnicas que falharam (nao retestar) |
| `reference_bq_signals.md` | reference | Tabelas BQ |

---

## Pipeline de producao (`scripts/production/`)

```
scripts/production/
├── run_daily.py                  # Entry point (1 comando)
├── config.py                     # 32 features + E1 config (floor=0)
├── fetch_raw_data.py             # 12+ fontes
├── bootstrap_from_original.py    # Hybrid dataset builder
├── build_features.py             # Feature engineering
├── generate_signal.py            # Sinal diario
├── training.py                   # Helpers
├── rebuild_signal_history.py     # Backfill sinais antigos
├── INSTRUCTIONS.md               # Manual
├── data/
│   ├── raw_data.csv              # Dados brutos
│   ├── dataset_production.csv    # 32 features (2019-2026)
│   ├── cached_models.pkl         # 80+80 XGBoost
│   └── signal_history.csv        # Sinais historicos
└── archive/                      # Backups antigos
```

**Rodar diario:**
```bash
python scripts/production/run_daily.py
```

---

## Estatisticas da sessao

- **Total experimentos rodados**: 135+ configuracoes
- **Tempo total compute**: ~12 horas
- **Testes no arquivo**: 24 scripts .py
- **Docs gerados**: 7 arquivos .md
- **Features finais**: 32 (29 V29 + 3 V36 novas on-chain)
- **Floor decision**: 0.0 (long-biased, sem short)
- **Sortino final**: **6.44** (10 seeds)
- **Return final**: **+860%** (2022-2026), +14.4% 2026 YTD
- **DD final**: -8.1%

---

## Comandos uteis

```bash
# Rodar pipeline diario
python scripts/production/run_daily.py

# Forcar retrain
python scripts/production/run_daily.py --retrain

# Rebuild total
rm scripts/production/data/dataset_production.csv
python scripts/production/run_daily.py --full --retrain

# Rollback pra V29 (com short)
cp scripts/production/archive/config.py.pre_e1 scripts/production/config.py
cp scripts/production/archive/build_features.py.pre_e1 scripts/production/build_features.py
cp scripts/production/archive/fetch_raw_data.py.pre_e1 scripts/production/fetch_raw_data.py
python scripts/production/run_daily.py --full --retrain

# Re-validar modelo (10 seeds)
python archive/test_scripts/v36_final_validation.py
```
