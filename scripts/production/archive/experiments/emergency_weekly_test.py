"""
A/B test: emergency rebal trigger — daily return vs weekly (7d) return.

Current production rule:
    emergency if |daily_ret| > 8%

Alternative under test:
    emergency if |7d cumulative ret| > THRESHOLD

Train semi-annual walk-forward models ONCE (shared across all rules), then run
the backtest under each rebal rule. Compute DAILY Sortino + DAILY max DD by
expanding rebal-period allocations to a daily equity curve (matches README
reporting: "Sortino 3.41 daily, DD -6.68% daily").

3 seeds for stability — same methodology as emergency_threshold_sweep.py.

Usage:
    python scripts/production/archive/experiments/emergency_weekly_test.py
    python scripts/production/archive/experiments/emergency_weekly_test.py \\
        --weekly-thresholds 0.08,0.10,0.12,0.15 --seeds 242,251,263

NOTE: this is a research experiment. Model is FECHADO since 2026-04-29 — not a
production change. Goal: quantify whether weekly trigger would degrade returns.
"""
import sys
import time
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/production'))

from config import (
    FEATURES_37, XGB_PARAMS, ALLOC_MIN, ALLOC_MAX, SIGMOID_SCALE,
    REBAL_DOW, BAGS, HORIZON, WORKERS,
)
from src.features.macro.cdi_rates import build_rf_daily

DATASET = ROOT / 'scripts/production/data/dataset_production.csv'
OUT = ROOT / 'outputs/results'
RETRAIN_MONTHS = [1, 7]
K_H1 = {'BULL': 60, 'MILD': 30, 'BEAR': 15}


def get_regime(price, sma50, sma200):
    if np.isnan(sma50) or np.isnan(sma200):
        return 'MILD'
    if price > sma50 and sma50 > sma200:
        return 'BULL'
    elif price > sma200:
        return 'MILD'
    return 'BEAR'


def retrain_cutoffs(start, end):
    cuts = []
    y = start.year - 1
    while y <= end.year + 1:
        for m in RETRAIN_MONTHS:
            d = pd.Timestamp(year=y, month=m, day=1)
            if d <= end:
                cuts.append(d)
        y += 1
    return sorted(set(c for c in cuts if c >= pd.Timestamp('2022-01-01')))


def _train_one_reg(args):
    s, X, y = args
    m = xgb.XGBRegressor(**XGB_PARAMS, random_state=s)
    rng = np.random.RandomState(s)
    idx = rng.choice(len(X), size=len(X), replace=True)
    m.fit(X[idx], y[idx])
    return m


def _train_one_cls(args):
    s, X, y = args
    cls_params = {**{k: v for k, v in XGB_PARAMS.items() if k != 'objective'},
                  'objective': 'binary:logistic', 'eval_metric': 'logloss'}
    m = xgb.XGBClassifier(**cls_params, random_state=s)
    rng = np.random.RandomState(s)
    idx = rng.choice(len(X), size=len(X), replace=True)
    m.fit(X[idx], y[idx])
    return m


def train_at_cutoff(ds, cutoff, base_seed):
    mask = ds['date'] < cutoff
    n = int(mask.sum())
    sub = ds.iloc[:n].reset_index(drop=True)
    X = np.nan_to_num(sub[FEATURES_37].values.astype(float), nan=0.0)
    prices = sub['price_usd'].values
    treg = np.zeros(n); tcls = np.zeros(n)
    for i in range(n - HORIZON):
        treg[i] = (prices[i + HORIZON] - prices[i]) / prices[i]
        tcls[i] = 1.0 if prices[i + HORIZON] > prices[i] else 0.0
    gap = max(HORIZON, 5)
    train_end = n - gap
    train_idx = np.arange(60, train_end + 1)
    train_idx = train_idx[~np.any(np.isnan(X[train_idx]), axis=1)]
    seeds = [base_seed + i * 7 for i in range(BAGS)]
    Xtr, ytr_reg, ytr_cls = X[train_idx], treg[train_idx], tcls[train_idx]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        regs = list(ex.map(_train_one_reg, [(s, Xtr, ytr_reg) for s in seeds]))
        clss = list(ex.map(_train_one_cls, [(s, Xtr, ytr_cls) for s in seeds]))
    return regs, clss


def build_rebals_daily(ds, start, end, threshold):
    """Fridays + days where |1d ret| > threshold (current production rule)."""
    sub = ds[(ds['date'] >= start) & (ds['date'] <= end)].copy()
    sub['ret'] = sub['price_usd'].pct_change()
    sub['dow'] = sub['date'].dt.dayofweek
    fridays = set(sub.loc[sub['dow'].isin(REBAL_DOW), 'date'])
    emerg = set(sub.loc[sub['ret'].abs() > threshold, 'date'])
    return sorted(fridays | emerg), len(emerg - fridays)


def build_rebals_weekly(ds, start, end, threshold):
    """Fridays + days where |7d cum ret| > threshold (proposed rule)."""
    sub = ds[(ds['date'] >= start) & (ds['date'] <= end)].copy()
    sub['ret_7d'] = sub['price_usd'].pct_change(7)
    sub['dow'] = sub['date'].dt.dayofweek
    fridays = set(sub.loc[sub['dow'].isin(REBAL_DOW), 'date'])
    emerg = set(sub.loc[sub['ret_7d'].abs() > threshold, 'date'])
    return sorted(fridays | emerg), len(emerg - fridays)


def predict_rebals(ds, cutoff_models, rebals):
    """Run walk-forward predictions at each rebal date. Returns per-rebal DataFrame."""
    ds_by_date = ds.set_index('date')
    rows = []
    for d0 in rebals:
        if d0 not in ds_by_date.index:
            continue
        applicable = [(c, m) for c, m in cutoff_models.items() if c <= d0]
        if not applicable:
            continue
        c = max(a[0] for a in applicable)
        regs, clss = cutoff_models[c]
        idx = ds[ds['date'] == d0].index[0]
        X_row = np.nan_to_num(
            ds.iloc[idx][FEATURES_37].values.astype(float).reshape(1, -1), nan=0.0
        )
        pred = float(np.mean([m.predict(X_row)[0] for m in regs]))
        p_up = float(np.mean([m.predict_proba(X_row)[0, 1] for m in clss]))
        conf = float(1 / (1 + np.exp(-abs(p_up - 0.5) * SIGMOID_SCALE)))
        prices = ds['price_usd'].values[:idx + 1]
        s50 = pd.Series(prices).rolling(50).mean().iloc[-1]
        s200 = pd.Series(prices).rolling(200).mean().iloc[-1]
        regime = get_regime(prices[-1], s50, s200)
        K = K_H1[regime]
        alloc = float(np.clip(pred * K * conf, ALLOC_MIN, ALLOC_MAX))
        rows.append({'date': d0, 'alloc': alloc})
    return pd.DataFrame(rows)


def daily_equity(rebal_df, ds, rf_daily, start, end):
    """Expand piecewise-constant allocations to daily portfolio returns.

    Between rebal d0 and next rebal, alloc is fixed. Compute:
        r_port_daily = alloc * btc_daily + (1 - alloc) * cdi_daily
    """
    cal = pd.date_range(start, end, freq='D')
    px = ds.set_index('date')['price_usd'].reindex(cal).ffill()
    btc_daily = px.pct_change().fillna(0.0)
    cdi_daily = rf_daily.reindex(cal).fillna(0.0)

    # Step-fill alloc onto daily calendar
    alloc_series = pd.Series(np.nan, index=cal, dtype=float)
    for _, r in rebal_df.iterrows():
        if r['date'] in alloc_series.index:
            alloc_series.loc[r['date']] = r['alloc']
    alloc_series = alloc_series.ffill().fillna(0.0)

    port_daily = alloc_series * btc_daily + (1 - alloc_series) * cdi_daily
    return port_daily.values, btc_daily.values, cdi_daily.values


def metrics_daily(r_daily: np.ndarray, ppy_daily: int = 365):
    """Daily-frequency Sortino, max DD, CAGR — matches README reporting basis."""
    r = np.asarray(r_daily, dtype=float)
    if len(r) == 0:
        return {'cagr': 0.0, 'sortino_d': 0.0, 'max_dd_d': 0.0}
    eq = np.cumprod(1.0 + r)
    cum = float(eq[-1] - 1.0)
    years = len(r) / ppy_daily
    cagr = (1 + cum) ** (1 / years) - 1 if years > 0 else 0.0
    neg = r[r < 0]
    dev = float(np.sqrt(np.mean(neg ** 2))) if len(neg) > 0 else 1e-12
    sortino = float(np.mean(r) / dev * np.sqrt(ppy_daily)) if dev > 0 else 0.0
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))
    full = np.concatenate([[1.0], eq])
    maxdd = float(((full - peak) / peak).min())
    return {'cagr': float(cagr), 'sortino_d': sortino, 'max_dd_d': maxdd}


def run_rule(label, rebal_builder, threshold, ds, cutoff_models, rf_daily, start, end):
    rebals, n_emerg_nonfri = rebal_builder(ds, start, end, threshold)
    df = predict_rebals(ds, cutoff_models, rebals)
    port_daily, btc_daily, cdi_daily = daily_equity(df, ds, rf_daily, start, end)
    m = metrics_daily(port_daily)
    m['n_rebals'] = len(df)
    m['n_emerg_nonfri'] = n_emerg_nonfri
    m['label'] = label
    m['threshold'] = threshold
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='242,251,263',
                    help='Comma-separated seeds (default: 3 seeds)')
    ap.add_argument('--weekly-thresholds', default='0.08,0.10,0.12,0.15',
                    help='Thresholds for weekly emergency rule')
    ap.add_argument('--start', default='2022-01-07')
    ap.add_argument('--end', default='2026-04-17')
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    weekly_thrs = [float(t) for t in args.weekly_thresholds.split(',')]

    ds = pd.read_csv(DATASET, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    cuts = retrain_cutoffs(start, end)
    cal = pd.date_range(start - pd.Timedelta(days=10), end + pd.Timedelta(days=10), freq='D')
    rf_daily = pd.Series(build_rf_daily(cal), index=cal)

    print(f"Emergency trigger A/B test | seeds={seeds} | weekly thresholds={weekly_thrs}")
    print(f"Range: {start.date()} -> {end.date()} | BAGS={BAGS} | K_H1={K_H1}\n")

    all_seed_results = {seed: {} for seed in seeds}
    t_start = time.time()

    for seed in seeds:
        t0 = time.time()
        cutoff_models = {c: train_at_cutoff(ds, c, seed) for c in cuts}
        print(f"Seed {seed}: trained {len(cuts)} cutoffs in {time.time()-t0:.0f}s", flush=True)

        print(f"  {'rule':<20s} {'rebals':>7s} {'emerg':>7s} {'CAGR':>9s} {'Sort_d':>8s} {'DD_d':>9s}")

        # Baseline: daily 8% (current production)
        m = run_rule('daily_8pct', build_rebals_daily, 0.08, ds, cutoff_models, rf_daily, start, end)
        all_seed_results[seed]['daily_8pct'] = m
        print(f"  {'daily 8% (ATUAL)':<20s} {m['n_rebals']:>5d}   {m['n_emerg_nonfri']:>5d}  "
              f"{m['cagr']*100:+7.2f}% {m['sortino_d']:7.2f}  {m['max_dd_d']*100:7.2f}%", flush=True)

        # Weekly thresholds
        for thr in weekly_thrs:
            key = f'weekly_{thr:.3f}'
            m = run_rule(key, build_rebals_weekly, thr, ds, cutoff_models, rf_daily, start, end)
            all_seed_results[seed][key] = m
            print(f"  weekly {thr*100:4.1f}%         {m['n_rebals']:>5d}   {m['n_emerg_nonfri']:>5d}  "
                  f"{m['cagr']*100:+7.2f}% {m['sortino_d']:7.2f}  {m['max_dd_d']*100:7.2f}%", flush=True)

    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")

    # Aggregate across seeds
    rule_keys = ['daily_8pct'] + [f'weekly_{thr:.3f}' for thr in weekly_thrs]
    print(f"\n{'='*88}")
    print(f"AGGREGATED ACROSS {len(seeds)} SEEDS  (daily-frequency metrics)")
    print(f"{'='*88}")
    print(f"  {'rule':<22s} {'rebals':>7s} {'CAGR mean ± std':>18s} {'Sort_d mean ± std':>20s} "
          f"{'DD_d mean':>11s}")
    print(f"  {'-'*86}")

    summary = {}
    baseline_cagr = None
    baseline_sort = None
    baseline_dd = None
    nseed = len(seeds)

    def _std(a):
        return float(a.std(ddof=1)) if nseed >= 2 else 0.0

    for key in rule_keys:
        cagrs = np.array([all_seed_results[s][key]['cagr'] for s in seeds])
        sorts = np.array([all_seed_results[s][key]['sortino_d'] for s in seeds])
        dds = np.array([all_seed_results[s][key]['max_dd_d'] for s in seeds])
        n_rebals = all_seed_results[seeds[0]][key]['n_rebals']
        n_emerg = all_seed_results[seeds[0]][key]['n_emerg_nonfri']

        if key == 'daily_8pct':
            baseline_cagr = cagrs.mean()
            baseline_sort = sorts.mean()
            baseline_dd = dds.mean()
            delta = ''
        else:
            d_cagr = (cagrs.mean() - baseline_cagr) * 100
            d_sort = sorts.mean() - baseline_sort
            d_dd = (dds.mean() - baseline_dd) * 100
            delta = f'  delta {d_cagr:+5.2f}pp CAGR / {d_sort:+.2f} Sort / {d_dd:+5.2f}pp DD'

        marker = ' <-- atual' if key == 'daily_8pct' else ''
        print(f"  {key:<22s} {n_rebals:>5d}    "
              f"{cagrs.mean()*100:+7.2f}% +/- {_std(cagrs)*100:4.2f}%   "
              f"{sorts.mean():6.2f} +/- {_std(sorts):.2f}        "
              f"{dds.mean()*100:7.2f}%{marker}{delta}")

        summary[key] = {
            'n_rebals': int(n_rebals),
            'n_emerg_nonfri': int(n_emerg),
            'cagr_mean': float(cagrs.mean()), 'cagr_std': _std(cagrs),
            'sortino_d_mean': float(sorts.mean()), 'sortino_d_std': _std(sorts),
            'max_dd_d_mean': float(dds.mean()), 'max_dd_d_std': _std(dds),
        }

    out_path = OUT / 'emergency_weekly_test.json'
    with open(out_path, 'w') as f:
        json.dump({
            'config': {
                'seeds': seeds, 'weekly_thresholds': weekly_thrs,
                'start': str(start.date()), 'end': str(end.date()),
                'BAGS': BAGS, 'K_REGIME': K_H1,
            },
            'results': summary,
        }, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
