"""
Generate daily allocation signal — E1 D7 + H1, no-short, with risk controls.

Pipeline:
  - 32 features (V29 base + V36 on-chain D7 trio)
  - Regression (sizing) + Classifier (confidence), 160 bagged XGBoost each (BAGS=160 -> 320 models total)
  - allocation = clip(pred * K[regime] * sigmoid(|P_cls - 0.5| * SIGMOID_SCALE), 0, 1)
  - K_REGIME (H1): BULL=60, MILD=30, BEAR=15
  - Rebalance Friday + emergency when |daily_ret| > 8%
  - Retrain semi-annual: Jan 1 + Jul 1
  - Risk controls (kill switch, acc derisk, PSI) applied via risk_management.py

Validated 4.28y OOS (daily DD, BAGS=160, with acc-derisk, 10-seed, BCB CDI, gross):
  CAGR +57.3% +/- 0.3%, Sortino 3.53 +/- 0.06, Sharpe 2.47 +/- 0.01, DD -7.14% +/- 0.25%.
See docs/MODEL_FINAL.md for the canonical spec.

Usage:
    python scripts/production/generate_signal.py              # daily check
    python scripts/production/generate_signal.py --retrain    # force retrain
"""
import sys, pickle, argparse, logging, hashlib, json
import numpy as np, pandas as pd
import xgboost as xgb
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.production.config import (
    DATASET_PATH, FEATURES_ALL, XGB_PARAMS,
    K_REGIME, ALLOC_MIN, ALLOC_MAX, BAGS, HORIZON, WORKERS,
    EMERGENCY_THRESHOLD, REBAL_DOW, USE_CONFIDENCE_SCALING, SIGMOID_SCALE,
)
from scripts.production.training import _train_one_xgb

SIGNAL_LOG = Path(__file__).parent / "data" / "signal_history.csv"
MODEL_CACHE = Path(__file__).parent / "data" / "cached_models.pkl"

RETRAIN_MONTHS = [1, 7]


def _config_fingerprint() -> str:
    """Short hash of the config that determines the trained models (features,
    BAGS, HORIZON, XGB_PARAMS). If it changes, a cached model is stale and must
    be retrained — guards against silently serving predictions from a model
    trained on a different config without --retrain."""
    payload = {
        'features': list(FEATURES_ALL),
        'bags': BAGS,
        'horizon': HORIZON,
        'xgb': {k: XGB_PARAMS[k] for k in sorted(XGB_PARAMS)},
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


# Health monitoring
HEALTH_ACCURACY_MIN = 0.48
HEALTH_ACCURACY_WINDOW = 60
HEALTH_DRAWDOWN_MAX = -0.15
HEALTH_FEATURE_ZSCORE_MAX = 4.0


def _run_data_pipeline():
    """Run fetch + bootstrap to update the dataset."""
    import subprocess
    prod_dir = Path(__file__).parent
    log.info("  [1/2] Fetching raw data...")
    subprocess.run([sys.executable, str(prod_dir / "fetch_raw_data.py")],
                   cwd=str(ROOT), timeout=300)
    log.info("  [2/2] Bootstrapping dataset (enhanced base + new days)...")
    subprocess.run([sys.executable, str(prod_dir / "bootstrap_from_original.py")],
                   cwd=str(ROOT), timeout=300)


def get_regime(price, sma50, sma200):
    if np.isnan(sma50) or np.isnan(sma200):
        return 'MILD'
    if price > sma50 and sma50 > sma200:
        return 'BULL'
    elif price > sma200:
        return 'MILD'
    return 'BEAR'


def next_retrain_date(today):
    for month in RETRAIN_MONTHS:
        candidate = today.replace(month=month, day=1)
        if candidate > today:
            return candidate
    return today.replace(year=today.year + 1, month=RETRAIN_MONTHS[0], day=1)


def last_retrain_date(today):
    candidates = []
    for month in RETRAIN_MONTHS:
        c = today.replace(month=month, day=1)
        if c <= today:
            candidates.append(c)
        c_prev = today.replace(year=today.year - 1, month=month, day=1)
        candidates.append(c_prev)
    return max(c for c in candidates if c <= today)


def train_regression_ensemble(X_train, y_train, seed=242):
    """Train bagged XGBoost regressors (sizing)."""
    bag_seeds = [seed + i * 7 for i in range(BAGS)]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        models = list(ex.map(_train_one_xgb,
            [(s, X_train, y_train, XGB_PARAMS) for s in bag_seeds]))
    return models


def train_classifier_ensemble(X_train, y_train, seed=242):
    """Train bagged XGBoost classifiers for confidence estimation."""
    models = []
    for i in range(BAGS):
        s = seed + i * 7
        rng = np.random.RandomState(s)
        idx = rng.choice(len(X_train), size=len(X_train), replace=True)
        m = xgb.XGBClassifier(
            max_leaves=XGB_PARAMS['max_leaves'],
            grow_policy=XGB_PARAMS['grow_policy'],
            tree_method='hist',
            colsample_bytree=XGB_PARAMS['colsample_bytree'],
            subsample=XGB_PARAMS['subsample'],
            learning_rate=XGB_PARAMS['learning_rate'],
            min_child_weight=XGB_PARAMS['min_child_weight'],
            n_estimators=XGB_PARAMS['n_estimators'],
            objective='binary:logistic',
            eval_metric='logloss',
            n_jobs=1, verbosity=0, random_state=s)
        m.fit(X_train[idx], y_train[idx])
        models.append(m)
    return models


def main():
    parser = argparse.ArgumentParser(description='Generate E1 D7 H1 allocation signal')
    parser.add_argument('--retrain', action='store_true', help='Force model retrain')
    parser.add_argument('--seed', type=int, default=242, help='Random seed')
    args = parser.parse_args()

    # ── Auto-update data if stale ──
    actual_today = pd.Timestamp.now().normalize()

    if not DATASET_PATH.exists():
        log.info("No dataset found — running full pipeline...")
        _run_data_pipeline()

    df = pd.read_csv(DATASET_PATH)
    df['date'] = pd.to_datetime(df['date'])
    last_data_date = df['date'].iloc[-1]

    yesterday = actual_today - pd.Timedelta(days=1)
    if last_data_date < yesterday:
        log.info(f"Data is stale ({last_data_date.date()} < {yesterday.date()}) — updating...")
        _run_data_pipeline()
        df = pd.read_csv(DATASET_PATH)
        df['date'] = pd.to_datetime(df['date'])

    # ── Load dataset ──
    prices = df['price_usd'].values
    n = len(prices)
    today_date = df['date'].iloc[-1]
    today_dow = today_date.dayofweek

    log.info(f"Dataset: {len(df)} rows, {df['date'].min().date()} to {today_date.date()}")

    # Warn loudly when calendar today is newer than the last closed candle.
    # The signal follows the candle (backtest convention): to act on Friday's
    # close, the Friday candle must have closed (Saturday 00:00 UTC).
    days_behind = (actual_today.date() - today_date.date()).days
    if days_behind >= 1:
        cal_name = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][actual_today.dayofweek]
        log.warning(
            f"Calendar today is {actual_today.date()} ({cal_name}) but last closed candle "
            f"is {today_date.date()} — signal reflects that candle, not today. "
            f"Run again after 00:00 UTC tomorrow to pick up today's close."
        )

    # ── Rebalance logic ──
    is_friday = today_dow in REBAL_DOW
    daily_ret = (prices[-1] - prices[-2]) / prices[-2] if n > 1 else 0
    is_emergency = abs(daily_ret) > EMERGENCY_THRESHOLD
    is_rebalance_day = is_friday or is_emergency

    last_rebal_date = None
    last_allocation = None
    if SIGNAL_LOG.exists():
        history = pd.read_csv(SIGNAL_LOG)
        if len(history) > 0:
            last_rebal_date = history['date'].iloc[-1]
            last_allocation = history['allocation'].iloc[-1]

    # ── Retrain logic ──
    last_retrain = last_retrain_date(today_date.date())
    next_retrain = next_retrain_date(today_date.date())
    days_to_retrain = (next_retrain - today_date.date()).days

    needs_retrain = False
    if MODEL_CACHE.exists() and not args.retrain:
        with open(MODEL_CACHE, 'rb') as f:
            cache = pickle.load(f)
        cache_date = cache.get('trained_date')
        if cache_date:
            cache_date = pd.to_datetime(cache_date).date()
            if cache_date < last_retrain:
                needs_retrain = True
                log.info(f"Retrain due: model from {cache_date}, last retrain window: {last_retrain}")
        cached_fp = cache.get('config_fp')
        if cached_fp is None:
            log.info("Cached model has no config fingerprint (pre-fingerprint cache) — not forcing retrain")
        elif cached_fp != _config_fingerprint():
            needs_retrain = True
            log.warning(f"Config changed (fingerprint {cached_fp} -> {_config_fingerprint()}) — forcing retrain")
    else:
        needs_retrain = True

    if args.retrain:
        needs_retrain = True

    # ── Build features (32 features: V29 base + V36 on-chain) ──
    feature_cols = [f for f in FEATURES_ALL if f in df.columns]
    X_all = df[feature_cols].values.astype(float)
    X_all = np.nan_to_num(X_all, nan=0.0)

    log.info(f"Features: {len(feature_cols)}/{len(FEATURES_ALL)}")

    # ── Train or load models ──
    if needs_retrain:
        log.info("Training ensemble (regression + classifier)...")

        # Regression target
        target_reg = np.zeros(n)
        for i in range(n - HORIZON):
            target_reg[i] = (prices[i + HORIZON] - prices[i]) / prices[i]

        # Classification target
        target_cls = np.zeros(n)
        for i in range(n - HORIZON):
            target_cls[i] = 1.0 if prices[i + HORIZON] > prices[i] else 0.0

        gap = max(HORIZON, 5)
        train_end = n - gap
        train_idx = np.arange(60, train_end + 1)
        valid = ~np.any(np.isnan(X_all[train_idx]), axis=1)
        train_idx = train_idx[valid]

        reg_models = train_regression_ensemble(X_all[train_idx], target_reg[train_idx], args.seed)
        cls_models = train_classifier_ensemble(X_all[train_idx], target_cls[train_idx], args.seed)

        with open(MODEL_CACHE, 'wb') as f:
            pickle.dump({
                'reg_models': reg_models,
                'cls_models': cls_models,
                'train_end': train_end,
                'trained_date': today_date.strftime('%Y-%m-%d'),
                'seed': args.seed,
                'version': 'E1-D7-H1',
                'config_fp': _config_fingerprint(),
            }, f)
        log.info(f"Models trained and cached ({len(train_idx)} samples)")
    else:
        with open(MODEL_CACHE, 'rb') as f:
            cache = pickle.load(f)
        reg_models = cache.get('reg_models', cache.get('models', []))
        cls_models = cache.get('cls_models', [])
        log.info(f"Loaded cached models (trained {cache.get('trained_date', '?')}, version {cache.get('version', '?')})")

    # ── Generate prediction ──
    x_today = np.nan_to_num(X_all[-1:], nan=0.0)
    prediction = float(np.mean([m.predict(x_today)[0] for m in reg_models]))

    # ── Confidence: sigmoid(|P - 0.5| * SIGMOID_SCALE) ──
    # With SIGMOID_SCALE=5: ~0.50 (uncertain P~0.5) → ~0.92 (confident P~1). Never zero.
    if cls_models and USE_CONFIDENCE_SCALING:
        p_up = float(np.mean([m.predict_proba(x_today)[0, 1] for m in cls_models]))
        confidence = abs(p_up - 0.5)
        confidence_factor = float(1.0 / (1.0 + np.exp(-confidence * SIGMOID_SCALE)))
    else:
        p_up = None
        confidence_factor = 1.0

    # ── Regime ──
    sma50 = pd.Series(prices).rolling(50).mean().iloc[-1]
    sma200 = pd.Series(prices).rolling(200).mean().iloc[-1]
    regime = get_regime(prices[-1], sma50, sma200)
    K = K_REGIME[regime]

    # ── Allocation: confidence-scaled, clipped to [ALLOC_MIN, ALLOC_MAX] ──
    raw_alloc = prediction * K
    suggested_alloc = float(np.clip(raw_alloc * confidence_factor, ALLOC_MIN, ALLOC_MAX))

    # ── Risk controls: drawdown kill-switch + rolling-accuracy de-risk + PSI drift monitor ──
    from scripts.production.risk_management import apply_risk_controls
    history_for_risk = None
    if SIGNAL_LOG.exists():
        history_for_risk = pd.read_csv(SIGNAL_LOG)
    suggested_alloc_pre_risk = suggested_alloc
    suggested_alloc, risk_status = apply_risk_controls(
        suggested_alloc=suggested_alloc,
        signal_history=history_for_risk,
        dataset=df,
        feature_cols=feature_cols,
        verbose=False,  # we'll print summary below
    )

    if is_rebalance_day:
        allocation = suggested_alloc
        action = "REBALANCE (Friday)" if is_friday else f"EMERGENCY REBALANCE (daily ret {daily_ret*100:+.1f}%)"
    else:
        allocation = last_allocation if last_allocation is not None else suggested_alloc
        action = f"HOLD — keeping last rebalance ({last_rebal_date})" if last_rebal_date else "HOLD — no prior rebalance, using model suggestion"

    # ── Output ──
    day_name = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][today_dow]
    trained_date = cache.get('trained_date', 'just now') if not needs_retrain else today_date.date()

    log.info("")
    k_tag = f'K={K_REGIME["BULL"]}/{K_REGIME["MILD"]}/{K_REGIME["BEAR"]}'
    log.info("=" * 65)
    log.info(f"  SIGNAL — {today_date.date()} ({day_name})  [E1-D7 {k_tag}]")
    log.info("=" * 65)
    log.info(f"  BTC Price:      ${prices[-1]:,.0f}")
    log.info(f"  Daily Return:   {daily_ret*100:+.2f}%")
    log.info(f"  Regime:         {regime} (K_base={K})")
    log.info(f"  Prediction:     {prediction*100:+.3f}% (3d return)")
    if p_up is not None:
        log.info(f"  P(up):          {p_up*100:.1f}% (confidence: {confidence_factor*100:.0f}%)")
        log.info(f"  K effective:    {K * confidence_factor:.0f} (base {K} x {confidence_factor:.2f})")
    log.info(f"")
    log.info(f"  >> Action:      {action}")
    log.info(f"  >> Allocation:  {allocation*100:+.1f}% BTC / {max(0, (1-max(allocation,0)))*100:.1f}% CDI")
    if suggested_alloc_pre_risk != suggested_alloc:
        log.info(f"     (risk controls adjusted: {suggested_alloc_pre_risk*100:+.1f}% -> {suggested_alloc*100:+.1f}%)")
    if risk_status.get('warnings'):
        for w in risk_status['warnings']:
            log.warning(f"  !! {w}")
    if risk_status.get('rolling_acc') is not None:
        log.info(f"  Rolling acc 12w: {risk_status['rolling_acc']*100:.1f}% (threshold 48%)")
    if risk_status.get('current_dd', 0) < -0.01:
        log.info(f"  Current DD:     {risk_status['current_dd']*100:.2f}% (kill at -12%)")
    if not is_rebalance_day:
        log.info(f"     (model suggests {suggested_alloc*100:+.1f}% — will apply on next Friday)")
    log.info(f"")
    log.info(f"  Last rebalance: {last_rebal_date or 'none yet'}")
    log.info(f"  Model trained:  {trained_date}")
    log.info(f"  Next retrain:   {next_retrain} ({days_to_retrain} days)")
    log.info(f"  Is Friday:      {'>> YES — REBALANCE DAY' if is_friday else 'no'}")
    log.info(f"  Emergency:      {'>> YES — REBALANCE NOW' if is_emergency else 'no'} (threshold: >{EMERGENCY_THRESHOLD*100:.0f}%)")
    log.info("=" * 65)

    # ── Fill FORWARD retornos on the PREVIOUS rebal row ──
    # retorno_btc[T]  = BTC return from rebal T to next rebal (known only now)
    # retorno_strat[T] = strat return for that same window (alloc[T] × btc + (1−alloc[T]) × CDI)
    # Only filled when a new rebal fires; the current row's retornos stay NaN until next rebal.
    prev_retorno_btc = None
    prev_retorno_strat = None
    if is_rebalance_day and last_rebal_date is not None and SIGNAL_LOG.exists():
        try:
            hist_prev = pd.read_csv(SIGNAL_LOG)
            if len(hist_prev) > 0:
                prev_row = hist_prev.iloc[-1]
                prev_date = pd.to_datetime(prev_row['date'])
                prev_price = float(prev_row['price_usd'])
                prev_alloc = float(prev_row['allocation'])
                prev_retorno_btc = float(prices[-1] / prev_price - 1)
                from src.features.macro.cdi_rates import build_rf_daily
                wk_dates = pd.date_range(prev_date, today_date, freq='D')
                cdi_series = build_rf_daily(wk_dates)
                cdi_acc = float(np.prod(1 + cdi_series[1:]) - 1) if len(cdi_series) > 1 else 0.0
                prev_retorno_strat = float(prev_alloc * prev_retorno_btc + (1 - prev_alloc) * cdi_acc)
        except Exception as e:
            log.warning(f"Could not compute forward retornos for previous row: {e}")

    # ── Save to signal history (only rebalance days) ──
    # Round to 4 decimal places for readability (preds are pct so 4 dec = 0.0001 = 0.01%)
    DEC = 4
    signal = {
        'date': today_date.strftime('%Y-%m-%d'),
        'day': day_name,
        'price_usd': round(float(prices[-1]), 2),
        'regime': regime,
        'previsao': round(float(prediction), DEC),
        'p_up': round(float(p_up), DEC) if p_up is not None else None,
        'confidence_factor': round(float(confidence_factor), DEC),
        'allocation': round(float(allocation), DEC),
        'K_base': K,
        'K_effective': round(float(K * confidence_factor), 2),
        'is_emergency': is_emergency,
        'retorno_btc': None,     # filled on next rebal
        'retorno_strat': None,   # filled on next rebal
        'action': action,
    }

    if not is_rebalance_day:
        log.info(f"  Not a rebalance day — signal NOT saved to history")
        return signal

    if SIGNAL_LOG.exists():
        history = pd.read_csv(SIGNAL_LOG)
        date_str = today_date.strftime('%Y-%m-%d')
        # Fill forward retornos on the previous row (the rebal that just ended)
        if prev_retorno_btc is not None and len(history) > 0:
            last_idx = history.index[-1]
            history.at[last_idx, 'retorno_btc'] = round(prev_retorno_btc, DEC)
            history.at[last_idx, 'retorno_strat'] = round(prev_retorno_strat, DEC)
        if date_str in history['date'].values:
            history = history[history['date'] != date_str]
        history = pd.concat([history, pd.DataFrame([signal])], ignore_index=True)
    else:
        history = pd.DataFrame([signal])

    history.to_csv(SIGNAL_LOG, index=False)
    log.info(f"  Signal saved to {SIGNAL_LOG}")
    if prev_retorno_btc is not None:
        prev_row = history.iloc[-2]
        prev_prev = float(prev_row['previsao'])
        log.info(f"  Rebal anterior ({prev_row['date']}) fechado: "
                 f"previsao {prev_prev*100:+.2f}% | BTC real {prev_retorno_btc*100:+.2f}% | "
                 f"Strat real {prev_retorno_strat*100:+.2f}%")

    # ── Health check ──
    _run_health_check(history, df, feature_cols, today_date)

    return signal


def _run_health_check(history, df, feature_cols, today_date):
    """Monitor model health."""
    alerts = []

    if len(history) >= 10 and 'previsao' in history.columns:
        h = history.copy()
        h['date'] = pd.to_datetime(h['date'])
        h = h.sort_values('date')
        if 'price_usd' in h.columns and len(h) >= 5:
            h['actual_ret_1d'] = h['price_usd'].pct_change().shift(-1)
            h['correct'] = ((h['previsao'] > 0) & (h['actual_ret_1d'] > 0)) | \
                           ((h['previsao'] < 0) & (h['actual_ret_1d'] < 0))
            recent = h.tail(min(HEALTH_ACCURACY_WINDOW, len(h)))
            valid = recent['correct'].dropna()
            if len(valid) >= 10:
                rolling_acc = valid.mean()
                if rolling_acc < HEALTH_ACCURACY_MIN:
                    alerts.append(f"ACCURACY DRIFT: {rolling_acc*100:.1f}% (threshold: {HEALTH_ACCURACY_MIN*100:.0f}%)")

    if len(df) > 200:
        for feat in feature_cols[:10]:
            if feat not in df.columns: continue
            col = df[feat].dropna()
            if len(col) < 100: continue
            historical_mean = col.iloc[:-30].mean()
            historical_std = col.iloc[:-30].std()
            if historical_std == 0: continue
            zscore = abs((col.iloc[-1] - historical_mean) / historical_std)
            if zscore > HEALTH_FEATURE_ZSCORE_MAX:
                alerts.append(f"FEATURE ANOMALY: {feat} z={zscore:.1f}")

    if alerts:
        log.warning("")
        log.warning("!" * 65)
        log.warning("  HEALTH ALERTS")
        log.warning("!" * 65)
        for alert in alerts:
            log.warning(f"  >> {alert}")
        log.warning("!" * 65)
    else:
        log.info("  Health check: OK")


if __name__ == '__main__':
    main()
