"""
Rolling vs Expanding window test — methodological question (NOT another tweak).

Current pipeline uses EXPANDING window (all data from 2019 to cutoff).
This test compares against ROLLING windows of 3, 4, 5 years.

Hypothesis: Crypto has regime shifts; older data may anchor model in irrelevant
regimes. Rolling 4-5y might match or beat expanding by adapting faster.

Setup:
  - 5 seeds (242, 251, 263, 277, 281)
  - 4 variants: Expanding 7y, Rolling 3y, Rolling 4y, Rolling 5y
  - Same H1 (60/30/15), same 32 features, same retrain semi-annual
  - Period: 2022-01-07 → 2026-04-17 (248 rebals)

Expected runtime: ~15 min on Ultra 9.
"""
import sys
import time
import json
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
    REBAL_DOW, EMERGENCY_THRESHOLD, BAGS, HORIZON, WORKERS,
)
from src.features.macro.cdi_rates import build_rf_daily

DATASET = ROOT / 'scripts/production/data/dataset_production.csv'
OUT = ROOT / 'outputs/results'
SEEDS = [242, 251, 263, 277, 281]
RETRAIN_MONTHS = [1, 7]
K_H1 = {'BULL': 60, 'MILD': 30, 'BEAR': 15}

# Window variants — None means expanding (use all data from start)
WINDOWS = [
    ('expanding',   None),   # current production
    ('rolling_3y',  3 * 365),
    ('rolling_4y',  4 * 365),
    ('rolling_5y',  5 * 365),
]


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


def train_at_cutoff(ds, cutoff, base_seed, window_days):
    """Train at cutoff using either expanding (window_days=None) or rolling window."""
    if window_days is None:
        mask = ds['date'] < cutoff
    else:
        window_start = cutoff - pd.Timedelta(days=window_days)
        mask = (ds['date'] >= window_start) & (ds['date'] < cutoff)
    n_total = int(mask.sum())
    sub = ds[mask].reset_index(drop=True)
    if len(sub) < 60:
        raise ValueError(f"Window too small: {len(sub)} days")

    X = np.nan_to_num(sub[FEATURES_37].values.astype(float), nan=0.0)
    prices = sub['price_usd'].values
    n = len(prices)
    treg = np.zeros(n); tcls = np.zeros(n)
    for i in range(n - HORIZON):
        treg[i] = (prices[i + HORIZON] - prices[i]) / prices[i]
        tcls[i] = 1.0 if prices[i + HORIZON] > prices[i] else 0.0
    gap = max(HORIZON, 5)
    train_end = n - gap
    train_idx = np.arange(min(60, n // 4), train_end + 1)
    train_idx = train_idx[~np.any(np.isnan(X[train_idx]), axis=1)]

    seeds = [base_seed + i * 7 for i in range(BAGS)]
    Xtr, ytr_reg, ytr_cls = X[train_idx], treg[train_idx], tcls[train_idx]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        regs = list(ex.map(_train_one_reg, [(s, Xtr, ytr_reg) for s in seeds]))
        clss = list(ex.map(_train_one_cls, [(s, Xtr, ytr_cls) for s in seeds]))
    return regs, clss, len(train_idx)


def run_seed_window(ds, seed, window_days, rebals, rf):
    """One seed, one window variant. Returns df of rebal-level results."""
    cuts = retrain_cutoffs(rebals[0], rebals[-1])
    cutoff_models = {}
    train_sizes = []
    for c in cuts:
        regs, clss, n_tr = train_at_cutoff(ds, c, seed, window_days)
        cutoff_models[c] = (regs, clss)
        train_sizes.append(n_tr)

    ds_by_date = ds.set_index('date')
    rows = []
    for i, d0 in enumerate(rebals):
        d1 = rebals[i + 1] if i + 1 < len(rebals) else rebals[-1] + pd.Timedelta(days=7)
        if d0 not in ds_by_date.index or d1 not in ds_by_date.index:
            continue
        applicable = [(c, m) for c, m in cutoff_models.items() if c <= d0]
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
        p0 = float(ds_by_date.loc[d0, 'price_usd'])
        p1 = float(ds_by_date.loc[d1, 'price_usd'])
        btc_ret = p1 / p0 - 1
        mask_cdi = (rf.index > d0) & (rf.index <= d1)
        cdi_ret = float((1 + rf[mask_cdi]).prod() - 1)
        strat_ret = alloc * btc_ret + (1 - alloc) * cdi_ret
        rows.append({'date': d0, 'pred': pred, 'p_up': p_up, 'regime': regime,
                     'alloc': alloc, 'btc_fwd': btc_ret, 'cdi': cdi_ret,
                     'strat': strat_ret})
    return pd.DataFrame(rows), train_sizes


def metrics(strat: np.ndarray, cdi: np.ndarray):
    cum = float(np.cumprod(1 + strat)[-1] - 1)
    neg = strat[strat < 0]
    dev = float(np.sqrt(np.mean(neg ** 2))) if len(neg) > 0 else 1e-9
    sortino = float(np.mean(strat) / dev * np.sqrt(52)) if dev > 0 else 0.0
    excess = strat - cdi
    sd_e = float(np.std(excess, ddof=0))
    sharpe_x = float(np.mean(excess) / sd_e * np.sqrt(52)) if sd_e > 0 else 0.0
    eq = np.concatenate([[1.0], np.cumprod(1 + strat)])
    peak = np.maximum.accumulate(eq)
    maxdd = float(((eq - peak) / peak).min())
    return {'cum': cum, 'sortino': sortino, 'sharpe_x': sharpe_x, 'max_dd': maxdd}


def main():
    ds = pd.read_csv(DATASET, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    start = pd.Timestamp('2022-01-07')
    end = pd.Timestamp('2026-04-17')
    sub = ds[(ds['date'] >= start) & (ds['date'] <= end)].copy()
    sub['daily_ret'] = sub['price_usd'].pct_change()
    sub['dow'] = sub['date'].dt.dayofweek
    fridays = set(sub.loc[sub['dow'].isin(REBAL_DOW), 'date'])
    emerg = set(sub.loc[sub['daily_ret'].abs() > EMERGENCY_THRESHOLD, 'date'])
    rebals = sorted(fridays | emerg)
    rf = pd.Series(
        build_rf_daily(pd.date_range(start - pd.Timedelta(days=10),
                                     end + pd.Timedelta(days=10), freq='D')),
        index=pd.date_range(start - pd.Timedelta(days=10),
                            end + pd.Timedelta(days=10), freq='D'),
    )

    print(f"Rolling vs Expanding | seeds={SEEDS} | {len(rebals)} rebals")
    print(f"Variants: {[w[0] for w in WINDOWS]}")
    print(f"\n{'='*100}")

    all_results = {w[0]: [] for w in WINDOWS}
    train_size_log = {w[0]: [] for w in WINDOWS}
    t_start = time.time()
    for w_name, w_days in WINDOWS:
        print(f"\n--- Variant: {w_name} (window={w_days}d) ---", flush=True)
        for seed in SEEDS:
            t0 = time.time()
            df, train_sizes = run_seed_window(ds, seed, w_days, rebals, rf)
            train_size_log[w_name].append(train_sizes)
            m = metrics(df['strat'].values, df['cdi'].values)
            m['seed'] = seed
            m['avg_alloc'] = float(df['alloc'].mean())
            all_results[w_name].append(m)
            print(f"  seed {seed}: cum={m['cum']*100:+7.1f}%  "
                  f"Sortino={m['sortino']:5.2f}  Shp_x={m['sharpe_x']:5.2f}  "
                  f"DD={m['max_dd']*100:6.2f}%  "
                  f"avg_train_size={np.mean(train_sizes):.0f}d  ({time.time()-t0:.0f}s)",
                  flush=True)

    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")

    # Aggregate
    def agg(lst, key):
        v = np.array([r[key] for r in lst])
        return float(v.mean()), float(v.std(ddof=1))

    print(f"\n{'='*100}")
    print(f"AGGREGATED ({len(SEEDS)} seeds each)")
    print(f"{'='*100}")
    print(f"{'Variant':<14s} {'cum':>18s} {'CAGR':>11s} {'Sortino':>13s} {'Shp_x':>11s} "
          f"{'DD':>14s} {'avg_train':>10s}")
    print(f"{'-'*100}")
    summary = {}
    for w_name, _ in WINDOWS:
        c_mean, c_std = agg(all_results[w_name], 'cum')
        s_mean, s_std = agg(all_results[w_name], 'sortino')
        sx_mean, sx_std = agg(all_results[w_name], 'sharpe_x')
        dd_mean, dd_std = agg(all_results[w_name], 'max_dd')
        cagr_mean = (1 + c_mean) ** (1/4.28) - 1
        avg_train = np.mean([np.mean(ts) for ts in train_size_log[w_name]])
        print(f"{w_name:<14s} {c_mean*100:+9.1f}% ± {c_std*100:5.1f}%  "
              f"{cagr_mean*100:+9.1f}%  "
              f"{s_mean:6.2f} ± {s_std:.2f}  "
              f"{sx_mean:5.2f} ± {sx_std:.2f}  "
              f"{dd_mean*100:6.2f}% ± {dd_std*100:.2f}%  "
              f"{avg_train:>8.0f}d")
        summary[w_name] = {
            'cum_mean': c_mean, 'cum_std': c_std,
            'sortino_mean': s_mean, 'sortino_std': s_std,
            'sharpe_x_mean': sx_mean, 'sharpe_x_std': sx_std,
            'max_dd_mean': dd_mean, 'max_dd_std': dd_std,
            'avg_train_days': avg_train,
        }

    # Pairwise comparison vs expanding (the baseline)
    print(f"\n{'='*100}")
    print(f"PAIRED COMPARISON vs Expanding (paired by seed)")
    print(f"{'='*100}")
    base = all_results['expanding']
    print(f"{'Variant':<14s} {'Δcum mean':>13s} {'Δcum std':>10s} "
          f"{'ΔSortino mean':>15s} {'ΔSharpe mean':>14s} {'ΔDD mean':>11s} {'wins/5':>10s}")
    for w_name, _ in WINDOWS:
        if w_name == 'expanding':
            continue
        var = all_results[w_name]
        d_cum = [v['cum'] - b['cum'] for v, b in zip(var, base)]
        d_sortino = [v['sortino'] - b['sortino'] for v, b in zip(var, base)]
        d_sharpe = [v['sharpe_x'] - b['sharpe_x'] for v, b in zip(var, base)]
        d_dd = [v['max_dd'] - b['max_dd'] for v, b in zip(var, base)]
        wins_cum = sum(1 for d in d_cum if d > 0)
        wins_sortino = sum(1 for d in d_sortino if d > 0)
        print(f"{w_name:<14s} {np.mean(d_cum)*100:+10.1f}pp  ± {np.std(d_cum, ddof=1)*100:5.1f}pp  "
              f"{np.mean(d_sortino):+10.3f}     {np.mean(d_sharpe):+10.3f}    "
              f"{np.mean(d_dd)*100:+7.2f}pp  cum:{wins_cum}/5 srt:{wins_sortino}/5")

    # T-tests
    try:
        from scipy import stats as scipy_stats
        print(f"\nPaired t-tests (vs expanding):")
        for w_name, _ in WINDOWS:
            if w_name == 'expanding':
                continue
            var = all_results[w_name]
            diffs = np.array([v['sortino'] - b['sortino'] for v, b in zip(var, base)])
            t, p = scipy_stats.ttest_1samp(diffs, 0)
            sig = '*** SIGNIFICANT ***' if p < 0.05 else '(noise)'
            print(f"  {w_name}: ΔSortino mean {np.mean(diffs):+.3f}  t={t:+.2f}  p={p:.3f}  {sig}")
    except ImportError:
        pass

    out_path = OUT / 'rolling_vs_expanding_test.json'
    with open(out_path, 'w') as f:
        json.dump({'config': 'H1, 32 features, 5 seeds',
                   'period': '2022-01-07 → 2026-04-17 (248 rebals)',
                   'seeds': SEEDS,
                   'summary': summary,
                   'per_seed': all_results}, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
