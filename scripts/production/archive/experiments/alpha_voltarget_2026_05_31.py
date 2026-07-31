"""
ALPHA experiment: volatility-targeting overlay (2026-05-31).

Goal: try to RAISE the Sharpe frontier (not just slide along it like sigmoid/K).
Vol-targeting (Moreira-Muir "Volatility-Managed Portfolios", AQR) scales exposure
inversely to realized vol -> can raise Sharpe by stabilizing risk. It's a
POST-prediction overlay (uses vol30 known at decision time + the model's alloc),
so no retraining: train the 10-seed ensemble ONCE, then sweep overlays.

DISCIPLINE (anti-overfit):
  - Metric = daily Sharpe (frontier metric; sigmoid/K leave it flat ~2.55).
  - Pick the best overlay on DEV (2022-2024), judge on HOLDOUT (2025-2026).
    A config that only wins on the full backtest but not the holdout = overfit.
  - Small grid (few variants) to limit multiple-testing.
  - Prior: the model already sizes by regime+confidence (vol-correlated), so the
    docs found a discrete "vol regime overlay" MARGINAL. Expect modest/none.

Run: python scripts/production/archive/experiments/alpha_voltarget_2026_05_31.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts/production'))
sys.path.insert(0, str(Path(__file__).parent))
import importlib
sv = importlib.import_module('seeds_validation_2026_04_28')
from config import REBAL_DOW, EMERGENCY_THRESHOLD

DATASET = ROOT / 'scripts/production/data/dataset_production.csv'
DEV_END = pd.Timestamp('2025-01-01')   # dev = before this; holdout = on/after


def metrics_on(df, ds, rf, alloc_col, mask):
    """Daily-MtM metrics for the rebals selected by `mask` (a boolean over df rows)."""
    d = df[mask].copy()
    if len(d) < 5:
        return None
    weekly = (d[alloc_col] * d['btc_fwd'] + (1 - d[alloc_col]) * d['cdi_period']).values
    daily, daily_cdi = sv.expand_to_daily(d, alloc_col, ds, rf)
    return sv.metrics_from_returns(weekly, daily, d['cdi_period'].values, daily_cdi)


def main():
    ds = pd.read_csv(DATASET, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    # Realized 30d vol from price (backward-looking; the slim dataset doesn't keep volatility_30d)
    _ret = np.log(ds['price_usd'] / ds['price_usd'].shift(1))
    ds['_vol30'] = _ret.rolling(30).std() * np.sqrt(365)
    vol_by_date = ds.set_index('date')['_vol30']
    med_vol = float(ds['_vol30'].median())
    start, end = pd.Timestamp('2022-01-07'), pd.Timestamp('2026-04-17')
    sub = ds[(ds['date'] >= start) & (ds['date'] <= end)].copy()
    sub['daily_ret'] = sub['price_usd'].pct_change(); sub['dow'] = sub['date'].dt.dayofweek
    rebals = sorted(set(sub.loc[sub['dow'].isin(REBAL_DOW), 'date']) |
                    set(sub.loc[sub['daily_ret'].abs() > EMERGENCY_THRESHOLD, 'date']))
    rf = pd.Series(sv.build_rf_daily(pd.date_range(start - pd.Timedelta(days=10), end + pd.Timedelta(days=10), freq='D')),
                   index=pd.date_range(start - pd.Timedelta(days=10), end + pd.Timedelta(days=10), freq='D'))
    print(f"median vol30 = {med_vol:.3f} | rebals {len(rebals)} | seeds {sv.SEEDS}")
    print("Training 10-seed ensemble once...")
    seed_dfs = []
    for i, seed in enumerate(sv.SEEDS):
        df = sv.run_seed(ds, seed, rebals, rf)
        df['vol30'] = df['date'].map(vol_by_date).fillna(med_vol)
        seed_dfs.append(df)
        print(f"  seed {seed} ({i+1}/{len(sv.SEEDS)})", flush=True)

    # Overlay configs: alloc_ov = clip(alloc_raw * clip(Vref/vol30, lo, hi), 0, 1)
    # baseline = no overlay.
    configs = {'baseline (no overlay)': None}
    for vref in [med_vol, 0.50, 0.65]:
        for (lo, hi) in [(0.5, 1.5), (0.5, 2.0)]:
            configs[f'voltgt Vref={vref:.2f} cap[{lo},{hi}]'] = (vref, lo, hi)

    def make_alloc(df, cfg):
        if cfg is None:
            return df['alloc_raw'].values
        vref, lo, hi = cfg
        scale = np.clip(vref / df['vol30'].values, lo, hi)
        return np.clip(df['alloc_raw'].values * scale, 0.0, 1.0)

    def agg_over_seeds(cfg, mask_fn):
        rows = []
        for df in seed_dfs:
            df = df.copy(); df['alloc_ov'] = make_alloc(df, cfg)
            m = metrics_on(df, ds, rf, 'alloc_ov', mask_fn(df))
            if m: rows.append(m)
        keys = ['cagr', 'sortino_d', 'sharpe_excess_d', 'max_dd_d', 'cum_ret']
        return {k: float(np.mean([r[k] for r in rows])) for k in keys}

    dev_mask = lambda df: df['date'] < DEV_END
    hold_mask = lambda df: df['date'] >= DEV_END
    full_mask = lambda df: df['date'] == df['date']

    print("\n" + "=" * 96)
    print(f"VOL-TARGET OVERLAY — daily Sharpe is the frontier metric. dev=<{DEV_END.date()}, holdout>=.")
    print("=" * 96)
    print(f"{'config':<30} | {'DEV Shp':>8} {'DEV Sort':>8} | {'HOLD Shp':>9} {'HOLD Sort':>9} {'HOLD DD':>8} | {'FULL Shp':>8} {'FULL CAGR':>9}")
    print("-" * 96)
    base_hold = None
    for name, cfg in configs.items():
        dev = agg_over_seeds(cfg, dev_mask)
        hold = agg_over_seeds(cfg, hold_mask)
        full = agg_over_seeds(cfg, full_mask)
        if name.startswith('baseline'):
            base_hold = hold['sharpe_excess_d']
        delta = ''
        if base_hold is not None and not name.startswith('baseline'):
            delta = f"  (holdout Shp {hold['sharpe_excess_d']-base_hold:+.3f} vs base)"
        print(f"{name:<30} | {dev['sharpe_excess_d']:8.3f} {dev['sortino_d']:8.2f} | "
              f"{hold['sharpe_excess_d']:9.3f} {hold['sortino_d']:9.2f} {hold['max_dd_d']*100:7.2f}% | "
              f"{full['sharpe_excess_d']:8.3f} {full['cagr']*100:8.1f}%{delta}")
    print("-" * 96)
    print("VERDICT RULE: an overlay only 'raises the frontier' if its HOLDOUT Sharpe beats")
    print("baseline holdout Sharpe. Winning only on FULL/DEV but not HOLDOUT = overfit.")


if __name__ == '__main__':
    main()
