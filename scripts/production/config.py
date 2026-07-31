"""
Production config — E1 D7 + H1, no-short, with risk controls (state 2026-04-22).

Model: 32 features (V29 base 29 + V36 on-chain 3), K_REGIME=H1 (60/30/15),
ALLOC_MIN=0.0 (no-short), SIGMOID_SCALE=15, HORIZON=3, BAGS=160, SEMI retrain.
Risk controls live in risk_management.py, applied by generate_signal.py:
kill switch ACTIVE, PSI monitor informational, acc-derisk DISABLED 2026-06-09
(M2 — 10-seed validation: the rule cost +2.2pp CAGR / +0.08 Sortino_d with
max DD unchanged; acc/conf still reported informationally).

Validated M1 = current config (OOS walk-forward, daily DD, BAGS=160,
SIGMOID_SCALE=5, NO acc-derisk, 10-seed mean, BCB CDI, GROSS — 4 bps ~
-0.6pp CAGR). OFFICIAL reporting currency: consistent BRL (BTC*USDBRL + CDI):
  canonical window 2022-01-07 -> 2026-04-17 (BRL):
    CAGR +50.5% +/- 0.4%, Sortino daily 3.84 +/- 0.05,
    Sharpe excess daily 2.35 +/- 0.01, Max DD daily -5.34% +/- 0.30%
  full window -> 2026-05-29 (BRL): CAGR +48.2%, Sortino_d 3.73, 2026 YTD +9.3%
  hybrid reference (BTC USD + CDI BRL): CAGR 50.2%, Sortino_d 3.96, DD -5.55%
  (sources: m1_brl_canonical_2026_06_09.json; multiseed_eval_2026_06_09.json)
EMERGENCY OPS (2026-06-09): detect AND execute right after the daily candle
close (00:00 UTC) — do NOT trade intraday at the -8% crossing; the intraday
convention costs -5.0pp CAGR / -0.45 Sortino_d vs post-close execution
(variant_grid_2026_06_09). Requires the pipeline to run daily.

Live expectation post-deflation (38 trials): CAGR 25-40%, Sortino 1.5-2.5,
DD daily -15 to -25%. See docs/MODEL_FINAL.md for the canonical spec and
docs/OVERFIT_TESTS_2026-04-22.md for the audit that motivated H2 -> H1.

History (for archaeology only — production state is the config below):
  V22: K=regime-aware, floor=-0.25, 37 features.
  V23: +sigmoid confidence, K=60/30/15.
  V25: feature fixes (m2/fed yoy), +fracdiff.
  V29: pruned to 29 features (low-gain removed).
  V31.7: floor=0 (no-short).
  V36/E1 D7: +3 on-chain (reserveRisk, funding_rate_ma7, puellMultiple) = 32.
  H2 (2026-04-20): K=100/50/20 (max backtest return).
  H1 (2026-04-22): K=60/30/15 + risk controls (38-44% more robust
                   in frozen-train tests, -14% DD vs H2).
  M2 (2026-06-09, current): H1 with acc-derisk DISABLED + emergency executed
                            post-close (10-seed variant grid validation).
"""
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data"
PRODUCTION_DIR = ROOT / "scripts" / "production"
DATASET_PATH = PRODUCTION_DIR / "data" / "dataset_production.csv"

# BigQuery
BQ_PROJECT = os.environ.get("BQ_PROJECT", "")  # projeto GCP com tabelas Messari (opcional)
BQ_DATASET = os.environ.get("BQ_DATASET", "refined")

# Model params (same as V22)
XGB_PARAMS = {
    'max_leaves': 31,
    'grow_policy': 'lossguide',
    'tree_method': 'hist',
    'colsample_bytree': 0.5,
    'subsample': 0.8,
    'learning_rate': 0.05,
    'min_child_weight': 12,
    'n_estimators': 200,
    'objective': 'reg:squarederror',
    'verbosity': 0,
    'nthread': 1,
}

# K_REGIME: H1 conservative (set 2026-04-22 after overfit audit).
# Validation:
#   - Walk-forward 4.3y OOS: Sortino 7.00 (vs H2's 5.61), DD -2.9% (vs -4.0%)
#   - Frozen train 2023-2026 (no retrain): Sortino 6.54 H1 vs 4.54 H2  (+44%)
#   - Frozen train 2024-2026 (no retrain): Sortino 6.55 H1 vs 4.74 H2  (+38%)
#   - H1 dominates H2 in EVERY frozen-train test → more robust to regime shift.
#   - H2 won walk-forward because K=100 amplifies "killer weeks" (overfit to
#     specific 2022-2026 bull capture events, per overfit_test_2_frozen_train.py).
# Rationale: H2 optimizes backtest return (+1228%) but depends on retrain doing
# 66% of the Sortino work. H1 has structural edge that survives no retrain.
# Reversible: change this line only (K is post-prediction multiplier).
K_REGIME = {'BULL': 60, 'MILD': 30, 'BEAR': 15}
# Alternatives:
#   H2 aggressive  {BULL:100, MILD:50, BEAR:20}  — max backtest return (old prod)
#   Conservative   {BULL:40,  MILD:20, BEAR:10}  — lowest DD, smallest alpha
# ALLOC_MIN=0.0: no-short (long-biased mandate, universal +0.4 Sortino vs floor=-0.25).
ALLOC_MIN = 0.0
ALLOC_MAX = 1.0
BAGS = 160                # bags PER ENSEMBLE: trains 160 regressors + 160
                          # classifiers = 320 models total (generate_signal.py
                          # loops range(BAGS) for each head). 80 -> 160 (2026-04-29):
                          # Sortino std 0.07 -> 0.01, 7x lower seed variance,
                          # mean Sortino +0.05. Cost: 2x training time (~10min
                          # retrain). Predict unchanged.
HORIZON = 3
RETRAIN = 'semi'
REBAL_DOW = [4]  # Friday
EMERGENCY_THRESHOLD = 0.08
WORKERS = 16

# Sigmoid confidence scaling: allocation = pred * K[regime] * sigmoid(|P_cls - 0.5| * SCALE)
# P~0.5 (uncertain): factor ~0.50 (halves bet). Never 0.
# SCALE 15 -> 5 (2026-06-09, M1): 10-seed paired validation — sigmoid=5 gives
# +0.34 Sortino_d (t=58), -1.55pp DD_d, Sharpe equal (2.43 vs 2.42), cost
# -6.1pp CAGR. Mandate is Sortino-driven -> adopted. Post-prediction dial:
# model cache unaffected. See multiseed_eval_2026_06_09.json.
USE_CONFIDENCE_SCALING = True
SIGMOID_SCALE = 5

# ═══════════════════════════════════════════════════════════════
# THE 32 FEATURES (E1 D7 combo: V29 base 29 + V36 on-chain 3)
# Variable name FEATURES_37 retained as legacy (~50 archive scripts import it).
# ═══════════════════════════════════════════════════════════════
# Note: 9 Messari features were tested (active_addresses, buy_sell_ratio,
# dominance, fees, etc.) but WORSENED performance (return dropped from
# +1155% to +757%). V36 trio adopted after +0.30 Sortino validation.

FEATURES_37 = [
    # V29 base (29 features, pruned from V25's 37 by removing 8 low-gain).
    # C2_BASE (16)
    'cusum_pos', 'mr_score_30d', 'adx',
    'cusum_neg', 'structural_break_score', 'eth_btc_ratio', 'm2_yoy_growth', 'volatility_7d', 'basis_ma7',
    'nupl_ma30', 'hurst_60d', 'eth', 'bb_position', 'eth_pctchg_30d',
    'stablecoin_zscore', 'btc_gold_corr_30d',
    # TOP_ADD (9)
    'stablecoin_supply_change_30d', 'copper_return_30d', 'fractal_dimension_30d', 'kpss_stat_30d',
    'half_life_60d', 'sortino_30d', 'volume_sma20_ratio',
    'aroon_down_30d',
    'basis_pct',
    # EXTRA (4)
    'fed_balance_sheet',
    'velocity',
    'price_fracdiff_05', 'fed_fracdiff_05',  # fractional diff (Lopez de Prado, AFML Ch. 5)
    # V36/E1 D7 on-chain (3, 2026-04-19): +0.30 Sortino vs 29-feat baseline.
    # Median-filled pre-2022-04-19 (data start).
    'reserveRisk',        # bitcoin-data.com /v1/reserve-risk - LTH conviction
    'funding_rate_ma7',   # Binance futures funding rate, 7d MA - leverage sentiment
    'puellMultiple',      # bitcoin-data.com /v1/puell-multiple - miner profitability
]

FEATURES_ALL = FEATURES_37  # alias
