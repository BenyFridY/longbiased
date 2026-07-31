# V31 — Estado Atual Completo (Snapshot)

**Data**: 2026-04-18/19 (sessão noturna longa)
**Status**: V29 em produção, V31 completo, V31 extras rodando, V32 planejado

---

## 🎯 EM PRODUÇÃO AGORA

**V29**: 29 features + fracdiff d=0.3 + pipeline V23 (regressor + classifier + sigmoid 15)

### Config atual (`scripts/production/config.py`):

```python
FEATURES_37 = [  # (name retained but now 29 features)
    # C2_BASE (removed: macd_histogram, price_percentile_1y, trend_strength)
    'cusum_pos', 'mr_score_30d', 'adx',
    'cusum_neg', 'structural_break_score',
    'eth_btc_ratio', 'm2_yoy_growth', 'volatility_7d', 'basis_ma7',
    'nupl_ma30', 'hurst_60d', 'eth', 'bb_position', 'eth_pctchg_30d',
    'stablecoin_zscore', 'btc_gold_corr_30d',
    # TOP_ADD (removed: ou_theta_60d, obv_trend, miners_revenue_ratio)
    'stablecoin_supply_change_30d', 'copper_return_30d',
    'fractal_dimension_30d', 'kpss_stat_30d',
    'half_life_60d', 'sortino_30d', 'volume_sma20_ratio',
    'aroon_down_30d',
    # V21
    'basis_pct',
    # EXTRA (removed: hash_rate_pctchg_30d, vol_x_regime_duration)
    'fed_balance_sheet', 'velocity',
    # V25 NEW
    'price_fracdiff_05',  # fracdiff d=0.3 (changed from 0.5 in V29)
    'fed_fracdiff_05',    # fracdiff d=0.3
]
# Total: 29 features (was 37 in V25)

XGB_PARAMS = {
    'max_leaves': 31,
    'grow_policy': 'lossguide',
    'tree_method': 'hist',
    'colsample_bytree': 0.5,
    'subsample': 0.8,
    'learning_rate': 0.05,
    'min_child_weight': 12,
    'n_estimators': 200,
}

K_REGIME = {'BULL': 60, 'MILD': 30, 'BEAR': 15}
ALLOC_MIN = -0.25  # max 25% short
ALLOC_MAX = 1.0    # max 100% long
BAGS = 80
HORIZON = 3
REBAL_DOW = [4]              # Friday
EMERGENCY_THRESHOLD = 0.08
SIGMOID_SCALE = 15           # confidence scaling
USE_CONFIDENCE_SCALING = True
```

### Fracdiff em `build_features.py`:
```python
df['price_fracdiff_05'] = fractional_diff(np.log(prices), d=0.3)       # was 0.5
df['fed_fracdiff_05']   = fractional_diff(np.log(fed_bs), d=0.3)       # was 0.5
```

### Dataset: `outputs/feature_selection/dataset_enhanced.csv`
- 2619 rows (2019-2026-03-03) + incremental
- fracdiff recalculated at d=0.3
- YoY corrections from V25 (m2_yoy_growth, fed_bs_yoy_change, m2_3m_growth)

### Backups (rollback):
- `scripts/production/config.py.pre_v29`
- `scripts/production/build_features.py.pre_v29`

---

## 🏆 TOP 5 CONFIGS (V31 V23 pipeline, 3 seeds each)

| # | Name | Sortino | Ret | DD | Fri Acc | Config delta vs V29 prod |
|---|------|---------|-----|-----|---------|--------------------------|
| 🥇 | **V31.7 NO SHORT** | **6.24 ± 0.06** | +830% | -8.3% | 59.9% | `ALLOC_MIN=0.0` (sem short) |
| 🥈 | **V31.3 sigmoid=10** | 5.86 ± 0.03 | +1061% | -8.1% | 59.9% | `SIGMOID_SCALE=10` |
| 🥉 | **V31.0 sigmoid=15 (PROD)** | 5.84 ± 0.03 | **+1146%** | -8.3% | 59.9% | (current) |
| 4 | V31.4 sigmoid=20 | 5.78 ± 0.03 | +1176% | -8.4% | 59.9% | `SIGMOID_SCALE=20` |
| 5 | V31.5 sigmoid=25 | 5.74 ± 0.03 | +1191% | -8.6% | 59.9% | `SIGMOID_SCALE=25` |

### Por critério profissional:

| Critério | Vencedor |
|----------|----------|
| **Max Sortino** (robustez) | V31.7 NO SHORT (6.24) |
| **Max Return** (crescimento) | V31.1 no_sigmoid (+1251%) ou V31.5 (+1191%) |
| **Balanced (Sortino + Ret)** ⭐ | V31.0 ou V31.3 (sigmoid 15 ou 10) |
| **Min DD** (proteção) | V31.8 ceil=0.75 (-6.6%) mas Sortino ruim |

### Accuracy per DOW (invariante em quase todas as configs):

- Mon: 52-55%
- Tue: 48-54%
- Wed: 49-53%
- Thu: 55-58%
- **Fri: 59-63%** (sweet spot — structural, NOT ML-dependent)
- Sat: 53-55%
- Sun: 52-56%

---

## ⏳ V31 EXTRAS (em execução agora)

Rodando em background (PID via b5q2o1016):

| Test | Config | Expected |
|------|--------|----------|
| V31.A | V29 + sigmoid 15 + **mcw=6** | more aggressive trees, possibly +Ret |
| V31.B | V29 + sigmoid 15 + **fracdiff d=0.25** | V30 marginal winner, validate |
| V31.C | **V25 full (37 features)** + sigmoid | isolate pruning effect |

Time estimate: ~55 min.

Results json: `outputs/results/v31_extras.json`

---

## 📚 HISTÓRICO COMPLETO DE TESTES

### Sessão overnight 2026-04-17/18 (V26-V31):

| Version | Tests | Winner / Key Finding |
|---------|-------|---------------------|
| V26 | 13 | fracdiff d=0.3 marginal; sample weighting, ensemble, multi-day all FAIL |
| V27 | 9 | meta-labeling = baseline; reg α>=0.1 FAIL; rolling window FAIL |
| V28 | 7 | **Feature pruning bot-10 wins +0.14 Sortino** |
| V29 | 5 (10 seeds) | **Pruned + d=0.3 wins +0.31 Sortino (5σ)**; applied to prod |
| V30 | 10 | d=0.25-0.35 plateau; no marginal gain |
| V31 | 10 (V23 pipe) | **+0.50 Sortino from sigmoid; V31.7 no-short best by Sortino** |
| V31 extras | 3 (in progress) | pending |

**Total: 57 backtest configurations tested (~12h compute)**

---

## ❌ TÉCNICAS QUE NÃO FUNCIONARAM (exaustivo)

Para evitar retestar:

1. **Sample weighting** (AFML Ch.4) — labels clustering não prejudica
2. **Ensemble XGB + LightGBM** — XGB bagged já é ensemble
3. **Triple-barrier target** (as regression) — regressor no {-1,0,1} falha
4. **Vol-normalized target** — K_REGIME não calibrado pra vol-scale
5. **Multi-day rebalance** (Thu+Fri, ThuFriMon, MonWedFri) — TODOS degradam
6. **Regularization α=0.1+** — piora Sortino (XGBoost já regulariza)
7. **Rolling window 3y** — histórico antigo (2019-2020) é informativo
8. **Classifier binário puro** — perde magnitude
9. **Per-DOW ensemble** — ~500 samples/dia insuficiente
10. **Top-N features (<30)** — perde info demais
11. **Adicionar fracdiff em ETH/vol** — ruído, não sinal
12. **Remove mais que 8 features** — começa a degradar
13. **Meta-labeling formal** — sigmoid atual já faz isso
14. **mcw<6 ou >20** — mcw=12 é ótimo
15. **Horizon 2d/5d/7d** — 3d é ótimo (V21)
16. **K_REGIME variações** — 60/30/15 é ótimo (V23 ultimate test)

---

## 🎯 PRÓXIMOS TESTES A RODAR (V32 — enquanto user dorme)

**Objetivo**: explorar novas dimensões SEM repetir o que já falhou.

Tempo disponível: ~5h = 300min

### Plano V32 (17 tests, ~5h):

#### SET A — Validação TOP 3 com 10 seeds (1h total = 60min)
- V32.A1 = V31.0 (prod sigmoid=15) × 10 seeds
- V32.A2 = V31.7 (no short) × 10 seeds
- V32.A3 = V31.3 (sigmoid=10) × 10 seeds

Total: 3 × ~20min × 10 seeds = 3 × 20 = 60min (rate: ~2min/seed)

Objetivo: confirmar ranking TOP 3 com rigor estatístico.

#### SET B — K regime asymmetric (ensaio novo, ~60min)
Normalmente K BULL/MILD/BEAR = 60/30/15 (simétrico por "otimismo").
Testar:
- V32.B1: K = {70, 30, 10} (mais long em BULL, menos short em BEAR)
- V32.B2: K = {50, 30, 25} (menos long, mais short)
- V32.B3: K = {80, 20, 0} (extremely binary)

#### SET C — Features não testadas ainda (~80min)
- V32.C1: V29 + add `days_since_halving` (cyclic feature)
- V32.C2: V29 + add `dxy_pctchg_30d` (USD strength macro)
- V32.C3: V29 + add `mvrv_ratio` (using realized_price)
- V32.C4: V29 + add `funding_rate_ma7` (derivatives sentiment, if exists)

#### SET D — Combos (~40min)
- V32.D1: V31.7 (no short) + sigmoid 10
- V32.D2: V29 + K assimétrico vencedor + sigmoid 10

#### SET E — Stability test (~60min)
- V32.E1: V29 rolling window 5y (between 3y fail and expanding)
- V32.E2: V29 expanding until train_end=2024, test only 2025-2026

**Execution**: Will be launched in background with `v32_exploration.py` after V31 extras completes.

---

## 📁 ARQUIVOS CHAVE

### Scripts de teste (arquivados em `archive/test_scripts/`):
- `v26_techniques.py` - V26 technique battery
- `v27_deep_techniques.py` - V27 deeper techniques
- `v28_advanced.py` - V28 pruning + combos
- `v29_validate.py`, `v29_combos.py` - V29 10-seed validation
- `v29_oos_test.py` - OOS split (2022-2024 vs 2025+)
- `v29_apply_to_production.py` - Apply V29 to prod
- `v29_recompute_fracdiff.py` - Recalc fracdiff d=0.3
- `v30_refinements.py` - V30 d granular + extras
- `v31_v23_pipeline.py` - V31 V23 full pipeline (10 tests)
- `v31_extras.py` - V31 extras (3 tests, running now)

### Resultados JSON (em `outputs/results/`):
- `v26_techniques.json`
- `v27_deep.json`
- `v28_advanced.json`
- `v29_validate.json`, `v29_combos.json`
- `v30_refinements.json`
- `v31_v23_pipeline.json`
- `v31_extras.json` (pending)

### Docs:
- `docs/V29_FINAL.md` — V29 decision and application
- `docs/V31_FINAL_SESSION.md` — Complete session report
- **`docs/V31_CURRENT_STATE.md`** — this file, state snapshot
- `docs/PIPELINE_V*.md` — historical versions V02-V22

---

## ⚡ COMO RECUPERAR CONTEXTO SE A CONVERSA FOR COMPACTADA

1. Read this file: `docs/V31_CURRENT_STATE.md`
2. Current prod = V29 (29 features, fracdiff d=0.3, sigmoid 15)
3. Top 5 alternatives identified in V31 (see table above)
4. V31 extras running now, V32 planned after
5. Rollback available via `.pre_v29` backups
6. User sleeping, wants V32 battery run during the night

---

## 💾 MEMORY NOTES

Save to `.claude/projects/.../memory/`:
- V29 configuration and validation
- Top 5 model rankings
- Techniques that don't work (to avoid retesting)
- V32 plan for overnight execution