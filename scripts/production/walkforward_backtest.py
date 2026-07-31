"""
Walk-forward backtest (strict out-of-sample, no look-ahead).

Simulates the raw allocation the model WOULD have signaled on each Friday (and
emergency) in a date range, using models retrained semi-annually per the
RETRAIN_MONTHS schedule in config.py.

NOTE: this reports the GROSS allocation = clip(pred*K*conf, 0, 1), WITHOUT the
live risk controls (kill switch / acc-derisk / PSI in risk_management.py) and
WITHOUT transaction cost — so it does NOT exactly reproduce the live equity
curve. For the canonical with-acc-derisk, multi-seed headline metrics use
scripts/production/archive/experiments/seeds_validation_2026_04_28.py.

Default range: start of signal_history.csv -> last close in dataset.
H1 vs H2 comparison available via --compare.

Usage:
    python scripts/production/walkforward_backtest.py
    python scripts/production/walkforward_backtest.py --start 2025-10-03
    python scripts/production/walkforward_backtest.py --compare
    python scripts/production/walkforward_backtest.py --save-missed  # append missed Fridays to signal_history.csv
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'production'))
sys.path.insert(0, str(ROOT))

from config import (
    FEATURES_37 as FEATURES_ALL,
    K_REGIME, ALLOC_MIN, ALLOC_MAX, SIGMOID_SCALE,
    HORIZON, REBAL_DOW, EMERGENCY_THRESHOLD,
)
from generate_signal import (
    train_regression_ensemble, train_classifier_ensemble, get_regime,
    RETRAIN_MONTHS,
)
from src.features.macro.cdi_rates import build_rf_daily

DATA_DIR = ROOT / 'scripts' / 'production' / 'data'
OUT_DIR = ROOT / 'outputs' / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def retrain_cutoffs(start, end):
    """Semi-annual retrain dates (1st of each RETRAIN_MONTHS) in [start, end]."""
    cutoffs = []
    y = start.year - 1
    while y <= end.year + 1:
        for m in RETRAIN_MONTHS:
            d = pd.Timestamp(year=y, month=m, day=1)
            if d <= end:
                cutoffs.append(d)
        y += 1
    return sorted(set(cutoffs))


def build_targets(prices, n, horizon=HORIZON):
    treg = np.zeros(n); tcls = np.zeros(n)
    for i in range(n - horizon):
        treg[i] = (prices[i + horizon] - prices[i]) / prices[i]
        tcls[i] = 1.0 if prices[i + horizon] > prices[i] else 0.0
    return treg, tcls


def train_at_cutoff(ds, cutoff, seed=242, verbose=True):
    mask = ds['date'] < cutoff
    n = int(mask.sum())
    sub = ds.iloc[:n].reset_index(drop=True)
    X = np.nan_to_num(sub[FEATURES_ALL].values.astype(float), nan=0.0)
    prices = sub['price_usd'].values
    treg, tcls = build_targets(prices, n)
    gap = max(HORIZON, 5)
    train_end = n - gap
    train_idx = np.arange(60, train_end + 1)
    train_idx = train_idx[~np.any(np.isnan(X[train_idx]), axis=1)]
    if verbose:
        print(f"  Cutoff {cutoff.date()}: n={n}, train_samples={len(train_idx)}", flush=True)
    t0 = time.time()
    reg = train_regression_ensemble(X[train_idx], treg[train_idx], seed)
    cls = train_classifier_ensemble(X[train_idx], tcls[train_idx], seed)
    if verbose:
        print(f"    trained in {time.time()-t0:.0f}s", flush=True)
    return reg, cls


def predict_one(reg_models, cls_models, x_row):
    pred = float(np.mean([m.predict(x_row)[0] for m in reg_models]))
    p_up = float(np.mean([m.predict_proba(x_row)[0, 1] for m in cls_models]))
    conf = float(1 / (1 + np.exp(-abs(p_up - 0.5) * SIGMOID_SCALE)))
    return pred, p_up, conf


def pick_models(cutoff_models, d0):
    """Return the model trained at the most recent cutoff <= d0."""
    applicable = [(c, m) for c, m in cutoff_models.items() if c <= d0]
    if not applicable:
        raise ValueError(f"No walk-forward model available for {d0.date()}")
    c = max(a[0] for a in applicable)
    return c, cutoff_models[c]


def rebalance_dates(ds, start, end):
    """All Fridays in [start, end] + emergency days (|daily_ret| > threshold) between them."""
    sub = ds[(ds['date'] >= start) & (ds['date'] <= end)].copy()
    sub['daily_ret'] = sub['price_usd'].pct_change()
    sub['dow'] = sub['date'].dt.dayofweek
    fridays = set(sub.loc[sub['dow'].isin(REBAL_DOW), 'date'])
    emerg = set(sub.loc[sub['daily_ret'].abs() > EMERGENCY_THRESHOLD, 'date'])
    all_rebal = sorted(fridays | emerg)
    return all_rebal


def run_backtest(ds, start, end, K_map, cutoff_models, rf_series):
    rebals = rebalance_dates(ds, start, end)
    if not rebals:
        return pd.DataFrame(), 1.0

    rows = []
    cum = 1.0
    cum_btc = 1.0
    ds_by_date = ds.set_index('date')
    for i, d0 in enumerate(rebals):
        d1 = rebals[i + 1] if i + 1 < len(rebals) else end
        if d0 not in ds_by_date.index or d1 not in ds_by_date.index:
            continue
        cutoff, (reg, cls) = pick_models(cutoff_models, d0)
        idx = ds[ds['date'] == d0].index[0]
        X_row = np.nan_to_num(
            ds.iloc[idx][FEATURES_ALL].values.astype(float).reshape(1, -1), nan=0.0
        )
        pred, p_up, conf = predict_one(reg, cls, X_row)
        prices = ds['price_usd'].values[:idx + 1]
        s50 = pd.Series(prices).rolling(50).mean().iloc[-1]
        s200 = pd.Series(prices).rolling(200).mean().iloc[-1]
        regime = get_regime(prices[-1], s50, s200)
        alloc = float(np.clip(pred * K_map[regime] * conf, ALLOC_MIN, ALLOC_MAX))
        daily_ret = float(ds.iloc[idx]['price_usd'] / ds.iloc[idx - 1]['price_usd'] - 1) if idx > 0 else 0.0
        is_emergency = abs(daily_ret) > EMERGENCY_THRESHOLD
        p0 = float(ds_by_date.loc[d0, 'price_usd'])
        p1 = float(ds_by_date.loc[d1, 'price_usd'])
        btc_ret = p1 / p0 - 1
        mask = (rf_series.index > d0) & (rf_series.index <= d1)
        cdi_ret = float((1 + rf_series[mask]).prod() - 1)
        strat_ret = alloc * btc_ret + (1 - alloc) * cdi_ret
        cum *= (1 + strat_ret)
        cum_btc *= (1 + btc_ret)
        rows.append({
            'date': d0.strftime('%Y-%m-%d'),
            'dow': d0.strftime('%a'),
            'price_usd': round(p0, 2),
            'daily_ret': round(daily_ret, 5),
            'regime': regime,
            'model_cutoff': cutoff.strftime('%Y-%m-%d'),
            'prediction_3d': round(pred, 6),
            'p_up': round(p_up, 4),
            'conf': round(conf, 4),
            'allocation': round(alloc, 4),
            'K_used': K_map[regime],
            'is_emergency': is_emergency,
            'to_date': d1.strftime('%Y-%m-%d'),
            'btc_ret_period': round(btc_ret, 5),
            'cdi_ret_period': round(cdi_ret, 6),
            'strat_ret_period': round(strat_ret, 5),
            'cum_strat': round(cum - 1, 5),
            'cum_btc': round(cum_btc - 1, 5),
        })
    return pd.DataFrame(rows), cum - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default=None, help='YYYY-MM-DD (default: first signal_history date)')
    ap.add_argument('--end', default=None, help='YYYY-MM-DD (default: last close in dataset)')
    ap.add_argument('--compare', action='store_true', help='Also run H1 and H0=V29 legacy')
    ap.add_argument('--save-missed', action='store_true',
                    help='Append rebals that are missing from signal_history.csv')
    args = ap.parse_args()

    ds = pd.read_csv(DATA_DIR / 'dataset_production.csv', parse_dates=['date']).sort_values('date').reset_index(drop=True)

    sig_path = DATA_DIR / 'signal_history.csv'
    sig = pd.read_csv(sig_path, parse_dates=['date']).sort_values('date').reset_index(drop=True)

    start = pd.Timestamp(args.start) if args.start else sig['date'].iloc[0]
    end = pd.Timestamp(args.end) if args.end else ds['date'].iloc[-1]
    print(f"Backtest range: {start.date()} -> {end.date()}", flush=True)
    print(f"Config K_REGIME: {K_REGIME}, ALLOC_MIN={ALLOC_MIN}", flush=True)

    cutoffs = retrain_cutoffs(start, end)
    cutoffs = [c for c in cutoffs if c >= pd.Timestamp('2022-01-01')]
    print(f"\nWalk-forward retrain cutoffs: {[c.date() for c in cutoffs]}")
    print("Training models...", flush=True)
    cutoff_models = {c: train_at_cutoff(ds, c) for c in cutoffs}

    rf = pd.Series(
        build_rf_daily(pd.date_range(start, end, freq='D')),
        index=pd.date_range(start, end, freq='D'),
    )

    df_prod, cum_prod = run_backtest(ds, start, end, K_REGIME, cutoff_models, rf)

    out_csv = OUT_DIR / 'walkforward_backtest.csv'
    df_prod.to_csv(out_csv, index=False)

    cum_btc = df_prod['cum_btc'].iloc[-1] if len(df_prod) else 0.0
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD BACKTEST (current config: K={K_REGIME}, floor={ALLOC_MIN})")
    print(f"{'='*60}")
    print(f"Range:       {start.date()} -> {end.date()}")
    print(f"Rebalances:  {len(df_prod)}")
    print(f"BTC:         {cum_btc*100:+.2f}%")
    print(f"Strategy:    {cum_prod*100:+.2f}%")
    print(f"Excess:      {(cum_prod - cum_btc)*100:+.2f}pp")
    print(f"NOTE: GROSS alloc, no risk controls / no cost. Canonical with-acc-derisk")
    print(f"      multi-seed metrics: seeds_validation_2026_04_28.py")
    print(f"Saved: {out_csv}")

    if args.compare:
        print("\nRunning comparison: H1 (60/30/15)...")
        df_h1, cum_h1 = run_backtest(ds, start, end, {'BULL': 60, 'MILD': 30, 'BEAR': 15},
                                      cutoff_models, rf)
        print(f"\nH1 (60/30/15):  {cum_h1*100:+.2f}%")
        print(f"H2 (100/50/20): {cum_prod*100:+.2f}%")
        print(f"H2 - H1:        {(cum_prod - cum_h1)*100:+.2f}pp")

    if args.save_missed:
        # Identify rebals in df_prod not yet in signal_history.csv
        existing = set(pd.to_datetime(sig['date']).dt.strftime('%Y-%m-%d'))
        new_rows = df_prod[~df_prod['date'].isin(existing)].copy()
        if len(new_rows) == 0:
            print("\nNo missed rebals to append.")
        else:
            print(f"\n{len(new_rows)} missed rebals detected:")
            for _, r in new_rows.iterrows():
                print(f"  {r['date']} ({r['dow']}) alloc={r['allocation']*100:.1f}% regime={r['regime']}")
            append_df = pd.DataFrame({
                'date': pd.to_datetime(new_rows['date']),
                'day': new_rows['dow'],
                'price_usd': new_rows['price_usd'],
                'daily_ret': new_rows['daily_ret'],
                'regime': new_rows['regime'],
                'prediction': new_rows['prediction_3d'],
                'allocation': new_rows['allocation'],
                'K': new_rows['K_used'],
                'is_emergency': new_rows['is_emergency'],
                'action': new_rows.apply(
                    lambda r: f"EMERGENCY REBALANCE (daily ret {r['daily_ret']*100:+.1f}%)"
                    if r['is_emergency'] else "REBALANCE (Friday) [backfill walk-forward]",
                    axis=1,
                ),
            })
            sig['date'] = pd.to_datetime(sig['date'])
            merged = pd.concat([sig, append_df], ignore_index=True).sort_values('date').reset_index(drop=True)
            merged['date'] = merged['date'].dt.strftime('%Y-%m-%d')
            merged.to_csv(sig_path, index=False)
            print(f"Appended to {sig_path}")


if __name__ == '__main__':
    main()
