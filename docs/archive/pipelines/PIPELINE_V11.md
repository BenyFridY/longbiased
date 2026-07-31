# Next Steps — V11 Roadmap

**Last updated**: 2026-02-23
**Current best**: 37feat + K25 + Bag50 → +802%, Sortino 2.74, 50pp spread

---

## V11 Candidates (Ordered by Expected Impact)

### 1. XGBoost Comparison (HIGH PRIORITY)

LightGBM was chosen early and never questioned. XGBoost is the closest alternative — same gradient boosting concept, different tree growth strategy.

**What to test:**
- Drop-in replacement: same features, same walk-forward, same allocation formula
- XGBoost equivalents for our LGB params:
  - `num_leaves=31` → `max_leaves=31` (with `grow_policy='lossguide'`)
  - `feature_fraction=0.7` → `colsample_bytree=0.7`
  - `bagging_fraction=0.8` → `subsample=0.8`
  - `min_data_in_leaf=12` → `min_child_weight` (needs tuning)
  - `learning_rate=0.05` → `eta=0.05`
- Test with both 32feat and 37feat, 10 seeds each
- Compare: if Sortino within ±0.05 of LGB, likely not worth switching

**Why it might help:**
- XGBoost regularizes differently (level-wise vs leaf-wise growth)
- Could produce more diverse models in the ensemble → tighter spread
- XGBoost's native `reg:squarederror` might behave differently from LGB's `regression`

**Why it might not:**
- V10 showed LGB defaults are near-optimal → model choice may not matter
- CatBoost and RandomForest were much worse (V9)
- The allocation formula is the real lever, not the model

**Estimated time:** ~2h (2 models x 10 seeds x 2 frameworks)

---

### 2. Hybrid Ensemble: LightGBM + XGBoost (MEDIUM PRIORITY)

If XGBoost alone is competitive, blend predictions from both:
- Train Bag25 LGB + Bag25 XGB = 50 total models
- More model diversity → potentially tighter spread than 50 models from same framework
- This is the real opportunity: ensemble diversity, not model superiority

**Estimated time:** ~3h

---

### 3. Bag Count Optimization (LOW-MEDIUM)

V10 spread experiment showed:
- Bag30 → Bag50: spread drops ~20pp
- Bag50 → Bag70: likely another ~10pp (testing in progress)
- Diminishing returns but free performance

**Test:** Bag50, 60, 70, 80, 100 — find the point where spread stops improving.
Bag70 caused OOM before but with gc.collect() between seeds it works.

**Estimated time:** ~4h

---

### 4. K Fine-Tuning Around 25 (LOW)

We tested K = 5, 10, 15, 20, 25. K=25 won. But we never tested K=22, 23, 28, 30.
Quick sweep of K = 18, 20, 22, 23, 25, 27, 30 with Bag50.

**Estimated time:** ~2h

---

### 5. Prediction Smoothing (EXPERIMENTAL)

Instead of using raw model prediction each Friday, use EMA of last N predictions:
- `smoothed_pred = alpha * new_pred + (1-alpha) * prev_smoothed`
- Could reduce allocation whipsaws and tighten spread
- Risk: introduces lag, might miss fast regime changes

**Estimated time:** ~2h

---

### 6. Walk-Forward Window Experiments (EXPERIMENTAL)

Currently: train on ALL data before test year (expanding window).
Alternative: rolling window (e.g., only last 2-3 years).
- Rolling might adapt faster to regime changes
- But less data = noisier models

**Estimated time:** ~3h

---

### 7. Target Horizon Sweep (LOW)

We use `ret_3d` as target. Never tested `ret_2d`, `ret_4d`, `ret_5d` with the new allocation formula.
With linear_K25, the optimal target might shift.

**Estimated time:** ~2h

---

### 8. Out-of-Sample Extension (VALIDATION)

Current test period: 2022-2026. If we can get data through Feb 2026 (live), run the best config on truly unseen data. This is the ultimate validation.

---

## Priority Order

1. **XGBoost comparison** — quick, high information value
2. **Hybrid ensemble** — only if XGBoost is competitive
3. **Bag count optimization** — easy wins
4. **K fine-tuning** — marginal gains
5. **Everything else** — only if above don't yield improvements

## Decision Framework

For V11, a config must beat the current best on **at least 2 of 3 criteria**:
- Sortino ≥ 2.74
- Spread ≤ 50pp
- Return ≥ +800%

If no V11 config beats this, the pipeline is done — focus shifts to production deployment and live monitoring.
