"""
Align cached_models.pkl with the semi-annual retrain schedule.

When --retrain is forced off-schedule (e.g., after adding new features mid-cycle),
the cache drifts from the walk-forward discipline. This script trains a model
exactly at the last scheduled cutoff (Jan 1 or Jul 1, whichever was most recent)
using data strictly prior to that date + gap=5d, then saves it as the active cache.

This makes the production pipeline use the model it SHOULD have been using if the
retrain had happened on the proper schedule date.

Usage:
    python scripts/production/align_cache_to_schedule.py            # use last scheduled cutoff
    python scripts/production/align_cache_to_schedule.py --cutoff 2026-01-01
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'production'))

from config import FEATURES_37 as FEATURES_ALL, HORIZON
from generate_signal import (
    train_regression_ensemble, train_classifier_ensemble,
    RETRAIN_MONTHS, last_retrain_date,
)

DATA_DIR = ROOT / 'scripts' / 'production' / 'data'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cutoff', default=None, help='YYYY-MM-DD (default: most recent scheduled)')
    ap.add_argument('--seed', type=int, default=242)
    args = ap.parse_args()

    ds = pd.read_csv(DATA_DIR / 'dataset_production.csv', parse_dates=['date']).sort_values('date').reset_index(drop=True)
    today = ds['date'].iloc[-1].date()

    if args.cutoff:
        cutoff = pd.Timestamp(args.cutoff)
    else:
        cutoff = pd.Timestamp(last_retrain_date(today))
    print(f"Today (last close): {today}")
    print(f"Cutoff for aligned retrain: {cutoff.date()}")

    mask = ds['date'] < cutoff
    n = int(mask.sum())
    print(f"Training samples from data < {cutoff.date()}: n={n}")

    sub = ds.iloc[:n].reset_index(drop=True)
    X = np.nan_to_num(sub[FEATURES_ALL].values.astype(float), nan=0.0)
    prices = sub['price_usd'].values

    target_reg = np.zeros(n)
    target_cls = np.zeros(n)
    for i in range(n - HORIZON):
        target_reg[i] = (prices[i + HORIZON] - prices[i]) / prices[i]
        target_cls[i] = 1.0 if prices[i + HORIZON] > prices[i] else 0.0

    gap = max(HORIZON, 5)
    train_end = n - gap
    train_idx = np.arange(60, train_end + 1)
    train_idx = train_idx[~np.any(np.isnan(X[train_idx]), axis=1)]
    print(f"Valid training samples: {len(train_idx)}")
    print(f"Features used: {len(FEATURES_ALL)}")

    print("\nTraining regressor ensemble (80 bags)...")
    reg_models = train_regression_ensemble(X[train_idx], target_reg[train_idx], args.seed)
    print("Training classifier ensemble (80 bags)...")
    cls_models = train_classifier_ensemble(X[train_idx], target_cls[train_idx], args.seed)

    out_path = DATA_DIR / 'cached_models.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump({
            'reg_models': reg_models,
            'cls_models': cls_models,
            'train_end': train_end,
            'trained_date': cutoff.strftime('%Y-%m-%d'),
            'seed': args.seed,
            'version': 'V23',
            'aligned_to_schedule': True,
            'data_cutoff': cutoff.strftime('%Y-%m-%d'),
        }, f)
    print(f"\nCache saved: {out_path}")
    print(f"  trained_date={cutoff.date()}, train_end={train_end}, aligned_to_schedule=True")


if __name__ == '__main__':
    main()
