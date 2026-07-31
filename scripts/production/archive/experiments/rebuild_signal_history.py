"""
Rebuild signal_history.csv from Oct 2025 to today.
Simulates what generate_signal.py would have produced each day.

Usage:
    python scripts/production/rebuild_signal_history.py
"""
import sys, pickle, logging
import numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.production.config import (
    DATASET_PATH, FEATURES_37, XGB_PARAMS,
    K_REGIME, ALLOC_MIN, ALLOC_MAX, BAGS, HORIZON,
    EMERGENCY_THRESHOLD, REBAL_DOW, WORKERS,
)
from scripts.production.training import _get_retrain_periods, _train_one_xgb

SIGNAL_LOG = Path(__file__).parent / "data" / "signal_history.csv"
MODEL_CACHE = Path(__file__).parent / "data" / "cached_models.pkl"

SEED = 242
START_DATE = '2025-10-01'


def get_regime(price, sma50, sma200):
    if np.isnan(sma50) or np.isnan(sma200):
        return 'MILD'
    if price > sma50 and sma50 > sma200:
        return 'BULL'
    elif price > sma200:
        return 'MILD'
    return 'BEAR'


def main():
    log.info("Rebuilding signal history from scratch...")

    # Load dataset
    df = pd.read_csv(DATASET_PATH)
    df['date'] = pd.to_datetime(df['date'])
    prices = df['price_usd'].values
    dates = df['date'].values
    n = len(prices)

    log.info(f"Dataset: {n} rows, {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}")

    # Build features matrix
    feature_cols = [f for f in FEATURES_37 if f in df.columns]
    X_all = df[feature_cols].values.astype(float)
    X_all = np.nan_to_num(X_all, nan=0.0)

    # Build target
    target = np.zeros(n)
    for i in range(n - HORIZON):
        target[i] = (prices[i + HORIZON] - prices[i]) / prices[i]

    # Daily returns
    daily_ret = np.zeros(n)
    for i in range(1, n):
        daily_ret[i] = (prices[i] - prices[i-1]) / prices[i-1]

    # SMAs for regime
    sma50 = pd.Series(prices).rolling(50).mean().values
    sma200 = pd.Series(prices).rolling(200).mean().values

    # Day of week
    dow = pd.to_datetime(dates).dayofweek

    # Retrain periods (semi-annual)
    periods = _get_retrain_periods(dates, n, 'semi')

    # Find start index
    start_idx = df[df['date'] >= START_DATE].index[0]
    log.info(f"Simulating from {START_DATE} (index {start_idx}) to {df['date'].iloc[-1].date()}")

    # Train models for each period that covers our range
    signals = []
    prev_alloc = 0.0
    last_rebal_date = None
    current_models = None

    for te, ts, tend in periods:
        # Skip periods before our start
        if tend < start_idx:
            continue

        # Train models for this period
        gap = max(HORIZON, 5)
        tm = np.arange(60, te - gap + 1)
        valid = ~np.any(np.isnan(X_all[tm]), axis=1)
        idx = tm[valid]
        if len(idx) < 100:
            continue

        Xt = np.nan_to_num(X_all[idx], nan=0.0)
        yt = target[idx]

        log.info(f"Training ensemble for period {pd.Timestamp(dates[ts]).date()} to {pd.Timestamp(dates[min(tend, n-1)]).date()}...")
        bag_seeds = [SEED + i * 7 for i in range(BAGS)]
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            current_models = list(ex.map(_train_one_xgb,
                [(s, Xt, yt, XGB_PARAMS) for s in bag_seeds]))

        # Simulate each day in this period
        sim_start = max(ts, start_idx)
        sim_end = min(tend, n - 1)

        for t in range(sim_start, sim_end + 1):
            date = pd.Timestamp(dates[t])
            day_name = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][dow[t]]

            is_friday = dow[t] in REBAL_DOW
            is_emergency = t > 0 and abs(daily_ret[t]) > EMERGENCY_THRESHOLD
            is_rebalance = is_friday or is_emergency

            # Predict
            x_t = np.nan_to_num(X_all[t:t+1], nan=0.0)
            prediction = float(np.mean([m.predict(x_t)[0] for m in current_models]))

            # Regime
            regime = get_regime(prices[t], sma50[t], sma200[t])
            K = K_REGIME[regime]

            # Allocation
            suggested_alloc = float(np.clip(prediction * K, ALLOC_MIN, ALLOC_MAX))

            if is_rebalance:
                allocation = suggested_alloc
                if is_emergency:
                    action = f"EMERGENCY REBALANCE (daily ret {daily_ret[t]*100:+.1f}%)"
                else:
                    action = "REBALANCE (Friday)"
                last_rebal_date = date.strftime('%Y-%m-%d')
            else:
                allocation = prev_alloc
                action = f"HOLD (last rebal: {last_rebal_date})" if last_rebal_date else "HOLD"

            prev_alloc = allocation

            # Only save rebalance days
            if is_rebalance:
                signals.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'day': day_name,
                    'price_usd': float(prices[t]),
                    'daily_ret': float(daily_ret[t]),
                    'regime': regime,
                    'prediction': float(prediction),
                    'allocation': float(allocation),
                    'K': K,
                    'is_emergency': is_emergency,
                    'action': action,
                })

        del current_models

    # Save
    history = pd.DataFrame(signals)
    history.to_csv(SIGNAL_LOG, index=False)
    log.info(f"\nSaved {len(history)} signals to {SIGNAL_LOG}")
    log.info(f"Date range: {history['date'].iloc[0]} to {history['date'].iloc[-1]}")

    # Summary stats
    log.info(f"Rebalance days: {len(history)}")
    log.info(f"Emergency rebalances: {history['is_emergency'].sum()}")

    # Also save fresh model cache
    # Retrain on full data for current use
    log.info("\nTraining fresh model cache on full data...")
    gap = max(HORIZON, 5)
    train_end = n - gap
    train_idx = np.arange(60, train_end + 1)
    valid = ~np.any(np.isnan(X_all[train_idx]), axis=1)
    train_idx = train_idx[valid]

    bag_seeds = [SEED + i * 7 for i in range(BAGS)]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        models = list(ex.map(_train_one_xgb,
            [(s, X_all[train_idx], target[train_idx], XGB_PARAMS) for s in bag_seeds]))

    with open(MODEL_CACHE, 'wb') as f:
        pickle.dump({
            'models': models,
            'train_end': train_end,
            'trained_date': df['date'].iloc[-1].strftime('%Y-%m-%d'),
            'seed': SEED,
        }, f)
    log.info(f"Saved model cache ({len(train_idx)} training samples)")


if __name__ == '__main__':
    main()
