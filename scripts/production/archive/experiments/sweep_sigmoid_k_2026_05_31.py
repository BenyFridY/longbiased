"""
Sigmoid + K-config sweep + per-year robustness (2026-05-31).

K[regime] and sigmoid are POST-prediction multipliers, so we train the 10-seed
ensemble ONCE (reusing seeds_validation_2026_04_28.run_seed) and sweep configs
for free — no retraining per config. Runs on the current (CUSUM-fixed) dataset.

GROSS (no cost) for clean cross-config comparison; cost is ~uniform across configs.

Answers in one run:
  - sigmoid=5 vs current 15 (docs flagged sigmoid=5 as worth testing)
  - K-config sensitivity (H1 vs Conservative vs H2)
  - per-year robustness (2022 bear / 2023 bull / 2024 / 2025 / 2026) at baseline

OVERFIT NOTE: picking the max-Sortino config off the full backtest is overfitting.
Read this as sensitivity + robustness, not "switch to the winner". Any change must
survive per-year consistency + add to the multiple-testing deflation count.

Run: python scripts/production/archive/experiments/sweep_sigmoid_k_2026_05_31.py
"""
import sys
from pathlib import Path

import numpy as np
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

SIGMOIDS = [1, 5, 10, 15, 25]          # at K = H1
K_CONFIGS = {
    'H1 (60/30/15, atual)': {'BULL': 60, 'MILD': 30, 'BEAR': 15},
    'Conservative (40/20/10)': {'BULL': 40, 'MILD': 20, 'BEAR': 10},
    'H2 (100/50/20)': {'BULL': 100, 'MILD': 50, 'BEAR': 20},
}


def alloc_for(df, K_map, sigmoid):
    """Recompute allocation from stored pred/p_up/regime for a (K, sigmoid) config."""
    conf = 1.0 / (1.0 + np.exp(-np.abs(df['p_up'].values - 0.5) * sigmoid))
    K = df['regime'].map(K_map).values
    return np.clip(df['pred'].values * K * conf, 0.0, 1.0)


def eval_config(seed_dfs, ds, rf, K_map, sigmoid):
    """Aggregate metrics across seeds for one (K, sigmoid) config (gross)."""
    rows = []
    for df in seed_dfs:
        d = df.copy()
        d['alloc_cfg'] = alloc_for(d, K_map, sigmoid)
        weekly = (d['alloc_cfg'] * d['btc_fwd'] + (1 - d['alloc_cfg']) * d['cdi_period']).values
        daily, daily_cdi = sv.expand_to_daily(d, 'alloc_cfg', ds, rf)
        rows.append(sv.metrics_from_returns(weekly, daily, d['cdi_period'].values, daily_cdi))
    agg = lambda k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows], ddof=1)))
    return {k: agg(k) for k in ['cum_ret', 'cagr', 'sortino_d', 'sortino_w', 'sharpe_excess_d', 'max_dd_d']}


def per_year(seed_dfs, ds, rf, K_map, sigmoid):
    """Per-year cum + daily Sortino at the given config (mean across seeds)."""
    out = {}
    for yr in [2022, 2023, 2024, 2025, 2026]:
        cums, sorts = [], []
        for df in seed_dfs:
            d = df[df['date'].dt.year == yr].copy()
            if len(d) < 3:
                continue
            d['alloc_cfg'] = alloc_for(d, K_map, sigmoid)
            weekly = (d['alloc_cfg'] * d['btc_fwd'] + (1 - d['alloc_cfg']) * d['cdi_period']).values
            daily, daily_cdi = sv.expand_to_daily(d, 'alloc_cfg', ds, rf)
            m = sv.metrics_from_returns(weekly, daily, d['cdi_period'].values, daily_cdi)
            cums.append(m['cum_ret']); sorts.append(m['sortino_d'])
        if cums:
            out[yr] = (float(np.mean(cums)), float(np.mean(sorts)), len(cums))
    return out


def main():
    ds = pd.read_csv(DATASET, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    start, end = pd.Timestamp('2022-01-07'), pd.Timestamp('2026-04-17')
    sub = ds[(ds['date'] >= start) & (ds['date'] <= end)].copy()
    sub['daily_ret'] = sub['price_usd'].pct_change()
    sub['dow'] = sub['date'].dt.dayofweek
    rebals = sorted(set(sub.loc[sub['dow'].isin(REBAL_DOW), 'date']) |
                    set(sub.loc[sub['daily_ret'].abs() > EMERGENCY_THRESHOLD, 'date']))
    rf = pd.Series(
        build_rf_daily(pd.date_range(start - pd.Timedelta(days=10), end + pd.Timedelta(days=10), freq='D')),
        index=pd.date_range(start - pd.Timedelta(days=10), end + pd.Timedelta(days=10), freq='D'),
    )
    print(f"Dataset to {ds['date'].max().date()} | rebals {len(rebals)} | seeds {sv.SEEDS}")
    print("Training 10-seed ensemble once (then sweeping configs post-prediction)...")

    seed_dfs = []
    for i, seed in enumerate(sv.SEEDS):
        seed_dfs.append(sv.run_seed(ds, seed, rebals, rf))
        print(f"  seed {seed} done ({i+1}/{len(sv.SEEDS)})", flush=True)

    def fmt(m):
        return (f"CAGR {m['cagr'][0]*100:+5.1f}% | Sort_d {m['sortino_d'][0]:.2f}±{m['sortino_d'][1]:.2f} | "
                f"Shp_d {m['sharpe_excess_d'][0]:.2f} | DD_d {m['max_dd_d'][0]*100:.2f}% | cum {m['cum_ret'][0]*100:+.0f}%")

    print("\n" + "=" * 78)
    print("SIGMOID SWEEP (K = H1 60/30/15), gross, 10-seed mean")
    print("=" * 78)
    for sig in SIGMOIDS:
        m = eval_config(seed_dfs, ds, rf, K_CONFIGS['H1 (60/30/15, atual)'], sig)
        tag = ' <- atual' if sig == 15 else ''
        print(f"  sigmoid={sig:>3} | {fmt(m)}{tag}")

    print("\n" + "=" * 78)
    print("K-CONFIG SWEEP (sigmoid = 15), gross, 10-seed mean")
    print("=" * 78)
    for name, K in K_CONFIGS.items():
        m = eval_config(seed_dfs, ds, rf, K, 15)
        print(f"  {name:<24} | {fmt(m)}")

    print("\n" + "=" * 78)
    print("PER-YEAR ROBUSTNESS (baseline H1, sigmoid=15), 10-seed mean")
    print("=" * 78)
    py = per_year(seed_dfs, ds, rf, K_CONFIGS['H1 (60/30/15, atual)'], 15)
    for yr, (c, s, n) in py.items():
        print(f"  {yr}: cum {c*100:+7.1f}%  Sortino_d {s:5.2f}  (n_seeds={n})")


if __name__ == '__main__':
    main()
