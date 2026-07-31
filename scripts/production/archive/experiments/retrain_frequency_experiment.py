"""
Retrain frequency experiment.

Tests how retrain cadence affects OOS performance on a 4-year walk-forward backtest.
Compares:
  - Annual (every 12m)
  - Semi (every 6m, current schedule)
  - Quarterly (every 3m)
  - Monthly (every 1m)
  - Conditional (retrain if rolling acc drops below threshold)

All configs use the current production config.py (K_REGIME, features, etc.).
No look-ahead: each retrain only uses data prior to the cutoff, gap=5d.

Usage:
    python scripts/production/retrain_frequency_experiment.py
    python scripts/production/retrain_frequency_experiment.py --start 2022-01-01
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
)
from src.features.macro.cdi_rates import build_rf_daily

DATA_DIR = ROOT / 'scripts' / 'production' / 'data'
OUT_DIR = ROOT / 'outputs' / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_targets(prices, n, horizon=HORIZON):
    treg = np.zeros(n); tcls = np.zeros(n)
    for i in range(n - horizon):
        treg[i] = (prices[i + horizon] - prices[i]) / prices[i]
        tcls[i] = 1.0 if prices[i + horizon] > prices[i] else 0.0
    return treg, tcls


def train_at_cutoff(ds, cutoff, seed=242):
    mask = ds['date'] < cutoff
    n = int(mask.sum())
    if n < 500:
        return None, None
    sub = ds.iloc[:n].reset_index(drop=True)
    X = np.nan_to_num(sub[FEATURES_ALL].values.astype(float), nan=0.0)
    prices = sub['price_usd'].values
    treg, tcls = build_targets(prices, n)
    gap = max(HORIZON, 5)
    train_end = n - gap
    train_idx = np.arange(60, train_end + 1)
    train_idx = train_idx[~np.any(np.isnan(X[train_idx]), axis=1)]
    reg = train_regression_ensemble(X[train_idx], treg[train_idx], seed)
    cls = train_classifier_ensemble(X[train_idx], tcls[train_idx], seed)
    return reg, cls


def generate_cutoffs(start, end, freq):
    """Generate retrain cutoff dates within [start, end] for the given frequency."""
    cutoffs = []
    if freq == 'annual':
        months = [1]
    elif freq == 'semi':
        months = [1, 7]
    elif freq == 'quarterly':
        months = [1, 4, 7, 10]
    elif freq == 'monthly':
        months = list(range(1, 13))
    else:
        raise ValueError(f"Unknown freq {freq}")
    y = start.year - 2
    while y <= end.year + 1:
        for m in months:
            d = pd.Timestamp(year=y, month=m, day=1)
            cutoffs.append(d)
        y += 1
    return sorted(set(cutoffs))


def pick_model(cutoff_models, d0):
    applicable = [c for c in cutoff_models if c <= d0 and cutoff_models[c] is not None and cutoff_models[c][0] is not None]
    if not applicable:
        return None
    c = max(applicable)
    return c, cutoff_models[c]


def backtest_freq(ds, start, end, freq, rf_series, seed=242):
    """Run walk-forward backtest with the given retrain frequency."""
    print(f"\n{'='*60}\nFreq: {freq}\n{'='*60}", flush=True)
    cutoffs = generate_cutoffs(start, end, freq)
    cutoffs = [c for c in cutoffs if c >= pd.Timestamp('2020-07-01') and c <= end]
    print(f"Retrain cutoffs: {len(cutoffs)}  [{[c.strftime('%Y-%m') for c in cutoffs[:3]]} ... {[c.strftime('%Y-%m') for c in cutoffs[-3:]]}]")

    cutoff_models = {}
    t0 = time.time()
    for i, c in enumerate(cutoffs):
        reg, cls = train_at_cutoff(ds, c, seed=seed)
        cutoff_models[c] = (reg, cls)
        if (i+1) % 5 == 0 or i == len(cutoffs)-1:
            print(f"  trained {i+1}/{len(cutoffs)} in {time.time()-t0:.0f}s", flush=True)
    print(f"All trained in {time.time()-t0:.0f}s")

    sub = ds[(ds['date'] >= start) & (ds['date'] <= end)].reset_index(drop=True)
    ds_by_date = ds.set_index('date')
    prices = ds['price_usd'].values

    cum_strat = 1.0; cum_btc = 1.0
    prev_alloc = 0.0
    daily_strat = []
    daily_btc = []
    daily_cdi = []
    preds_record = []
    for _, row in sub.iterrows():
        d0 = row['date']
        idx = ds[ds['date'] == d0].index[0]
        btc_ret = ds.iloc[idx]['price_usd'] / ds.iloc[idx-1]['price_usd'] - 1 if idx > 0 else 0.0
        # FIX: today's btc_ret was governed by PREVIOUS alloc.
        # Rebal at today's close updates alloc, but that new alloc applies FROM tomorrow.
        applied_alloc = prev_alloc
        is_friday = d0.dayofweek in REBAL_DOW
        is_emergency = abs(btc_ret) > EMERGENCY_THRESHOLD
        if is_friday or is_emergency:
            picked = pick_model(cutoff_models, d0)
            if picked is None:
                new_alloc = prev_alloc
            else:
                _, (reg, cls) = picked
                X = np.nan_to_num(ds.iloc[idx][FEATURES_ALL].values.astype(float).reshape(1, -1), nan=0.0)
                pred = float(np.mean([m.predict(X)[0] for m in reg]))
                p_up = float(np.mean([m.predict_proba(X)[0, 1] for m in cls]))
                conf = float(1 / (1 + np.exp(-abs(p_up - 0.5) * SIGMOID_SCALE)))
                hist = prices[:idx+1]
                s50 = pd.Series(hist).rolling(50).mean().iloc[-1]
                s200 = pd.Series(hist).rolling(200).mean().iloc[-1]
                regime = get_regime(prices[idx], s50, s200)
                new_alloc = float(np.clip(pred * K_REGIME[regime] * conf, ALLOC_MIN, ALLOC_MAX))
                preds_record.append({'date': d0, 'pred': pred, 'actual_future': None})
            prev_alloc = new_alloc  # applies from tomorrow onward
        # CDI daily
        cdi = rf_series.loc[d0] if d0 in rf_series.index else 0.0
        strat = applied_alloc * btc_ret + (1 - applied_alloc) * cdi
        cum_strat *= (1 + strat)
        cum_btc *= (1 + btc_ret)
        daily_strat.append(strat)
        daily_btc.append(btc_ret)
        daily_cdi.append(float(cdi))

    # Sortino & Price 1994 (V22/V36 convention: excess over CDI, sqrt(365))
    strat_arr = np.array(daily_strat)
    cdi_arr = np.array(daily_cdi)
    excess = strat_arr - cdi_arr
    downside = np.minimum(excess, 0.0)
    dd_std = float(np.sqrt(np.mean(downside ** 2)))
    sortino = float(excess.mean() / (dd_std + 1e-10) * np.sqrt(365))
    sharpe = float(excess.mean() / (strat_arr.std() + 1e-10) * np.sqrt(365)) if strat_arr.std() > 0 else np.nan
    cum = np.cumprod(1 + strat_arr)
    peak = np.maximum.accumulate(cum)
    dd = float((cum / peak - 1).min())
    n_years = (end - start).days / 365.25
    cagr = float(cum_strat**(1/n_years) - 1) if n_years > 0 else np.nan
    return {
        'freq': freq,
        'n_retrains': len(cutoffs),
        'cum_strat': cum_strat - 1,
        'cum_btc': cum_btc - 1,
        'cagr': cagr,
        'sortino': sortino,
        'sharpe': sharpe,
        'max_dd': dd,
        'days': len(daily_strat),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2022-01-01')
    ap.add_argument('--end', default=None)
    ap.add_argument('--freqs', default='annual,semi,quarterly,monthly')
    ap.add_argument('--seed', type=int, default=242)
    args = ap.parse_args()

    ds = pd.read_csv(DATA_DIR / 'dataset_production.csv', parse_dates=['date']).sort_values('date').reset_index(drop=True)
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else ds['date'].iloc[-1]

    full_dates = pd.date_range(start, end, freq='D')
    rf = pd.Series(build_rf_daily(full_dates), index=full_dates)

    results = []
    for freq in args.freqs.split(','):
        r = backtest_freq(ds, start, end, freq.strip(), rf, seed=args.seed)
        results.append(r)

    df = pd.DataFrame(results)
    df['cum_strat_%'] = (df['cum_strat']*100).round(2)
    df['cum_btc_%'] = (df['cum_btc']*100).round(2)
    df['cagr_%'] = (df['cagr']*100).round(2)
    df['max_dd_%'] = (df['max_dd']*100).round(2)
    out = OUT_DIR / 'retrain_frequency_results.csv'
    df.to_csv(out, index=False)
    print(f"\n{'='*70}")
    print(f"RESULTS ({start.date()} -> {end.date()}, {(end-start).days} days)")
    print('='*70)
    display = df[['freq','n_retrains','cum_strat_%','cum_btc_%','cagr_%','sortino','sharpe','max_dd_%']]
    print(display.to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
