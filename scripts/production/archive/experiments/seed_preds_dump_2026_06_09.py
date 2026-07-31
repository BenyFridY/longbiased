"""Dump per-seed walkforward predictions to disk (2026-06-09).

Trains the 10-seed ensemble ONCE (reusing seeds_validation_2026_04_28.run_seed)
over the full current dataset (2022-01-07 -> last Friday) and saves each seed's
rebal-level prediction dataframe (pred, p_up, conf_factor, regime, alloc_raw,
btc_fwd, cdi_period) to outputs/results/seed_preds_2026_06_09/seed_<s>.csv.

Purpose: decision-layer variants (sigmoid, derisk, emergency-exec convention,
confidence-head variants) are POST-prediction — with these dumps they can be
evaluated across all 10 seeds without ever retraining. Incremental: skips seeds
already dumped, so the job is resumable.

Run: python scripts/production/archive/experiments/seed_preds_dump_2026_06_09.py
"""
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/production'))
sys.path.insert(0, str(Path(__file__).parent))

import importlib
sv = importlib.import_module('seeds_validation_2026_04_28')
from config import REBAL_DOW, EMERGENCY_THRESHOLD
from src.features.macro.cdi_rates import build_rf_daily

DATASET = ROOT / 'scripts/production/data/dataset_production.csv'
OUT = ROOT / 'outputs/results/seed_preds_2026_06_09'


def main():
    OUT.mkdir(exist_ok=True)
    ds = pd.read_csv(DATASET, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    start = pd.Timestamp('2022-01-07')
    sub_all = ds[ds['date'] >= start].copy()
    sub_all['dow'] = sub_all['date'].dt.dayofweek
    end = sub_all.loc[sub_all['dow'].isin(REBAL_DOW), 'date'].max()  # last Friday in data
    sub = ds[(ds['date'] >= start) & (ds['date'] <= end)].copy()
    sub['daily_ret'] = sub['price_usd'].pct_change()
    sub['dow'] = sub['date'].dt.dayofweek
    fridays = set(sub.loc[sub['dow'].isin(REBAL_DOW), 'date'])
    emerg = set(sub.loc[sub['daily_ret'].abs() > EMERGENCY_THRESHOLD, 'date'])
    rebals = sorted(fridays | emerg)
    rf = pd.Series(
        build_rf_daily(pd.date_range(start - pd.Timedelta(days=10), end + pd.Timedelta(days=10), freq='D')),
        index=pd.date_range(start - pd.Timedelta(days=10), end + pd.Timedelta(days=10), freq='D'),
    )
    print(f"Dataset to {ds['date'].max().date()} | window {start.date()} -> {end.date()} | "
          f"rebals {len(rebals)} | seeds {sv.SEEDS}", flush=True)

    for i, seed in enumerate(sv.SEEDS):
        out_path = OUT / f'seed_{seed}.csv'
        if out_path.exists():
            print(f"  seed {seed} already dumped, skipping ({i+1}/{len(sv.SEEDS)})", flush=True)
            continue
        t0 = time.time()
        df = sv.run_seed(ds, seed, rebals, rf)
        df.to_csv(out_path, index=False)
        print(f"  seed {seed} done: {len(df)} rebals, {time.time()-t0:.0f}s "
              f"({i+1}/{len(sv.SEEDS)}) -> {out_path.name}", flush=True)

    print("ALL SEEDS DUMPED.", flush=True)


if __name__ == '__main__':
    main()
