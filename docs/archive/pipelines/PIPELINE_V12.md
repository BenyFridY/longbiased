# V12 Roadmap — Hybrid vs XGB-Alone Optimization

**Last updated**: 2026-02-25
**V11 completed**: 2026-02-25 (12.8h, 47 configs)

---

## V11 Final Results

| Rank | Config | Return | Sortino | Spread | MaxDD | Beats V10 |
|------|--------|--------|---------|--------|-------|-----------|
| 1 | **37f Hybrid 15+15 K=25** | **+925%** | **2.92** | **45pp** | -13.9% | **3/3** |
| 2 | 37f Hybrid 25+25 K=25 | +918% | 2.92 | 46pp | -13.7% | 3/3 |
| 3 | 37f XGB alone K=25 | +992% | 2.91 | 83pp | -14.5% | 2/3 |
| 4 | 37f LGB Bag100 K=25 | +817% | 2.76 | 53pp | -13.9% | 2/3 |
| — | V10 best (reference) | +802% | 2.74 | 50pp | -14.1% | — |

### V11 Ruled Out
- **Prediction smoothing**: ALL alpha < 1.0 hurt. alpha=0.3 kills 290pp return.
- **ret_5d**: High return (+924%) but 109pp spread, S=2.50. Terrible tradeoff.
- **ret_4d**: Worse Sortino than ret_3d in all configs.
- **ret_2d**: Tightest spread (20pp for 32f) but too low return (+648%).
- **K > 25 for LGB**: K=27/30 worse Sortino despite higher return.
- **Bag60-100 for LGB**: Marginal Sortino gain (+0.02), spread doesn't improve vs Bag50.

---

## V12 Goal: Determine Final Production Config

Two competing paths:

| Path | Current Best | Strength | Weakness | What's Needed |
|------|-------------|----------|----------|---------------|
| **Hybrid** | +925%, S=2.92, 45pp | Balanced, tight spread | May be ceiling | K tuning, bag ratios |
| **XGB alone** | +992%, S=2.91, 83pp | Highest return ceiling | 83pp spread | Hyperparams, features, spread fix |

V12 will optimize both paths and pick the winner.

---

## V12 Phases

### Phase A: Validation of V11 Hybrid (~1h)

Run V9 validation suite on hybrid 15+15 (seed=42):
1. Permutation test (1000 block-shuffles), p < 0.05
2. Bootstrap CI (1000 block-bootstrap), P(loss) < 5%
3. Year-by-year excess: positive in >= 3/5 years

**Decision gate**: Must pass. If fails, investigate before proceeding.

### Phase B: Hybrid K Sweep (~3h, 10 seeds x 7 values)

K=25 was tuned for LGB predictions. Hybrid averages LGB+XGB, different scale.

- K = 20, 22, 23, 25, 27, 30, 33
- Hybrid 15+15, 37feat, min_data_in_leaf=12

### Phase C: Hybrid Bag Ratios (~3h, 10 seeds x 6 ratios)

15+15 was arbitrary. XGB is stronger on return, LGB is more stable.

- 10+10 = 20 total (lightest)
- 10+20 = 30 total (XGB-heavy)
- 15+15 = 30 total (baseline)
- 20+10 = 30 total (LGB-heavy)
- 15+25 = 40 total (XGB-heavy, more models)
- 25+15 = 40 total (LGB-heavy, more models)

Use best K from Phase B.

### Phase D: XGB Hyperparameter Screen (~2h, 1 seed x ~50 configs)

XGB params are LGB defaults mapped over — never tuned. Screen one-at-a-time with 1 seed.

Sweep (XGB alone, 37feat, K=25, Bag50):
- `max_leaves`: [15, 23, 31, 47, 63]
- `min_child_weight`: [5, 8, 12, 20, 30]
- `learning_rate`: [0.02, 0.03, 0.05, 0.08, 0.10]
- `n_estimators`: [100, 150, 200, 300, 400]
- `max_depth`: [4, 6, 8, 10, 0]
- `colsample_bytree`: [0.5, 0.6, 0.7, 0.8, 0.9]
- `subsample`: [0.6, 0.7, 0.8, 0.9, 1.0]
- `reg_alpha`: [0, 0.01, 0.1, 1.0]
- `reg_lambda`: [0, 0.1, 1.0, 5.0]

### Phase E: XGB Best Params (10 seeds, ~2h)

Take top 5 param combos from Phase D, run 10 seeds each.
Also apply best XGB params to hybrid (Phase D winner inside hybrid 15+15).

### Phase F: XGB Spread Fix (~3h, 10 seeds x 8 configs)

Using best XGB hyperparams from Phase E:
- XGB alone Bag70, Bag100 (more bags = less variance)
- XGB alone K=20, K=22 (less aggressive = tighter spread)
- Hybrid with tuned XGB params (does hybrid still beat XGB-alone?)

### Phase G: Feature Swaps (~4h, 10 seeds x ~12 configs)

Test rejected LGB features in XGB:
- Features dropped in V6-V9: puell_multiple, rsi_14d, funding_rate, sopr_ma7
- Add each one (38feat XGB), swap one (replace weakest)
- Test in both hybrid (XGB half only) and XGB-alone

### Phase H: Final Showdown (~1h, 10 seeds x 3-4 configs)

Best hybrid config vs best XGB-alone config vs V11 baseline.
Full metrics comparison. Declare winner.

---

## Priority & Estimated Timeline

| Phase | Time | Priority | Description |
|-------|------|----------|-------------|
| A | 1h | MUST DO | Validation gate |
| B | 3h | HIGH | Hybrid K tuning |
| C | 3h | HIGH | Hybrid bag ratios |
| D | 2h | HIGH | XGB hyperparam screen |
| E | 2h | HIGH | XGB best params (10 seeds) |
| F | 3h | MEDIUM | XGB spread fix |
| G | 4h | MEDIUM | Feature swaps |
| H | 1h | HIGH | Final comparison |

**Total: ~19h**

---

## Decision Framework

V12 config beats V11 if **at least 2 of 3**:
- Sortino >= 2.92
- Spread <= 45pp
- Return >= +925%

If nothing beats V11 hybrid → hybrid 15+15 is production config.
If XGB-alone beats hybrid → need full validation before switching.

---

## Files

| File | Description |
|------|-------------|
| `scripts/optimization/pipeline_v11_xgb.py` | V11 pipeline (complete) |
| `scripts/optimization/pipeline_v12_hybrid.py` | V12 pipeline |
| `outputs/results/pipeline_v11_xgb.json` | V11 results |
| `outputs/results/pipeline_v12_hybrid.json` | V12 results |
| `docs/NEXT_STEPS_V12.md` | This document |
