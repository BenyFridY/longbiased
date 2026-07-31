"""
Replay the signal that WOULD HAVE BEEN generated on 2026-05-01 (Friday)
using only data available up to and including that date — no look-ahead.

Uses the same cached model (trained 2026-04-27, semi-annual retrain cadence)
that was live on that day.
"""
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.production.config import (
    FEATURES_ALL, K_REGIME, ALLOC_MIN, ALLOC_MAX,
    SIGMOID_SCALE, EMERGENCY_THRESHOLD,
)

DATASET = ROOT / 'scripts' / 'production' / 'data' / 'dataset_production.csv'
CACHE = ROOT / 'scripts' / 'production' / 'data' / 'cached_models.pkl'
TARGET_DATE = pd.Timestamp('2026-05-01')


def get_regime(price, sma50, sma200):
    if np.isnan(sma50) or np.isnan(sma200):
        return 'MILD'
    if price > sma50 and sma50 > sma200:
        return 'BULL'
    elif price > sma200:
        return 'MILD'
    return 'BEAR'


def main():
    # Load dataset, filter to data available AS OF target date (no look-ahead)
    ds = pd.read_csv(DATASET, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    ds = ds[ds['date'] <= TARGET_DATE].reset_index(drop=True)
    print(f'Dataset filtered to <= {TARGET_DATE.date()}: {len(ds)} rows')
    print(f'Last 3 rows:')
    print(ds[['date', 'price_usd']].tail(3).to_string(index=False))

    target_row = ds[ds['date'] == TARGET_DATE]
    if target_row.empty:
        print(f'\nWARN: {TARGET_DATE.date()} not in dataset. Using last row as proxy.')
        target_row = ds.tail(1)
    target_idx = target_row.index[0]

    # Confirm it is a Friday
    dow = target_row.iloc[0]['date'].day_name()
    print(f'\nTarget date {TARGET_DATE.date()} is a {dow}')

    # Extract features at target date
    feats = target_row[FEATURES_ALL].iloc[0].values
    if np.isnan(feats).any():
        missing = [f for f, v in zip(FEATURES_ALL, feats) if pd.isna(v)]
        print(f'WARN: missing features: {missing}')

    # Regime
    price = target_row.iloc[0]['price_usd']
    sma50 = target_row.iloc[0].get('sma50', np.nan)
    sma200 = target_row.iloc[0].get('sma200', np.nan)
    if pd.isna(sma50):
        # Compute on the fly
        sma50 = ds['price_usd'].iloc[max(0, target_idx-49):target_idx+1].mean()
        sma200 = ds['price_usd'].iloc[max(0, target_idx-199):target_idx+1].mean()
    regime = get_regime(price, sma50, sma200)
    K_base = K_REGIME[regime]

    # Load cached models (trained 2026-04-27, was live on 2026-05-01)
    with open(CACHE, 'rb') as f:
        cache = pickle.load(f)
    print(f'\nCached models version: {cache.get("version", "?")}')
    print(f'Cached models trained: {cache.get("trained_date", "?")}')
    reg_models = cache['reg_models']
    cls_models = cache['cls_models']
    print(f'Regression bags: {len(reg_models)}')
    print(f'Classifier bags: {len(cls_models)}')

    # Predict — bagged ensemble
    X = feats.reshape(1, -1)
    reg_preds = np.array([m.predict(X)[0] for m in reg_models])
    cls_preds = np.array([m.predict_proba(X)[0, 1] for m in cls_models])
    pred = float(reg_preds.mean())
    p_up = float(cls_preds.mean())

    # Confidence factor
    conf = 1.0 / (1.0 + np.exp(-abs(p_up - 0.5) * SIGMOID_SCALE))
    K_eff = K_base * conf

    # Allocation
    raw_alloc = pred * K_base * conf
    alloc = float(np.clip(raw_alloc, ALLOC_MIN, ALLOC_MAX))

    # Emergency check
    daily_ret = target_row.iloc[0].get('price_usd') / ds.iloc[target_idx-1]['price_usd'] - 1
    is_emergency = abs(daily_ret) > EMERGENCY_THRESHOLD

    print()
    print('=' * 65)
    print(f'  REPLAYED SIGNAL — {TARGET_DATE.date()} ({dow}) — NO LOOK-AHEAD')
    print('=' * 65)
    print(f'  BTC price (USD):           ${price:>12,.2f}')
    print(f'  Daily return:              {daily_ret*100:>+11.2f}%')
    print(f'  Regime:                    {regime}  (K_base={K_base})')
    print(f'  Model prediction (3d):     {pred*100:>+11.3f}%')
    print(f'  P(up) classifier:          {p_up*100:>11.1f}%')
    print(f'  Confidence factor:         {conf:>11.4f}')
    print(f'  K effective:               {K_eff:>11.2f}')
    print()
    print(f'  Raw alloc (pred*K*conf):   {raw_alloc:>11.4f}')
    print(f'  Final alloc (clipped):     {alloc*100:>+10.1f}% BTC  /  {(1-alloc)*100:.1f}% CDI')
    print()
    print(f'  Is Friday rebalance:       {dow == "Friday"}')
    print(f'  Emergency triggered:       {is_emergency} (threshold {EMERGENCY_THRESHOLD*100}%)')
    print(f'  Previous rebal alloc:      22.8% (from 2026-04-24)')
    print(f'  Change vs previous:        {(alloc - 0.2284)*100:>+10.1f} pp')
    print('=' * 65)


if __name__ == '__main__':
    main()
