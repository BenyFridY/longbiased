"""
Emergency rebalance threshold sweep — does 8% capture the right tail?

Currently EMERGENCY_THRESHOLD = 0.08. Test 5%, 6%, 7%, 7.5%, 8%, 8.5%, 9%, 10%, 12%, 15%.

For each threshold:
  - Build rebal set (Fridays + days where |daily_ret| > threshold)
  - Run walkforward with H1 K=60/30/15
  - Compare cum_return, Sortino, Sharpe, DD, # rebals

3 seeds for paired comparison (cheap test, single threshold per seed run).
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
    REBAL_DOW, BAGS, HORIZON, WORKERS,
)
from src.features.macro.cdi_rates import build_rf_daily

DATASET = ROOT / 'scripts/production/data/dataset_production.csv'
OUT = ROOT / 'outputs/results'
SEEDS = [242, 251, 263]
THRESHOLDS = [0.05, 0.06, 0.07, 0.075, 0.08, 0.085, 0.09, 0.10, 0.12, 0.15]
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


def build_rebals(ds, start, end, threshold):
    sub = ds[(ds['date'] >= start) & (ds['date'] <= end)].copy()
    sub['daily_ret'] = sub['price_usd'].pct_change()
    sub['dow'] = sub['date'].dt.dayofweek
    fridays = set(sub.loc[sub['dow'].isin(REBAL_DOW), 'date'])
    emerg = set(sub.loc[sub['daily_ret'].abs() > threshold, 'date'])
    return sorted(fridays | emerg), len(fridays), len(emerg - fridays)


def predict_rebals(ds, cutoff_models, rebals, rf):
    """Run walkforward with given rebal set. Returns df with strat returns."""
    ds_by_date = ds.set_index('date')
    rows = []
    for i, d0 in enumerate(rebals):
        d1 = rebals[i + 1] if i + 1 < len(rebals) else rebals[-1] + pd.Timedelta(days=7)
        if d0 not in ds_by_date.index or d1 not in ds_by_date.index:
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
        p0 = float(ds_by_date.loc[d0, 'price_usd'])
        p1 = float(ds_by_date.loc[d1, 'price_usd'])
        btc_ret = p1 / p0 - 1
        mask = (rf.index > d0) & (rf.index <= d1)
        cdi_ret = float((1 + rf[mask]).prod() - 1)
        strat_ret = alloc * btc_ret + (1 - alloc) * cdi_ret
        rows.append({'strat': strat_ret, 'btc': btc_ret, 'cdi': cdi_ret,
                     'alloc': alloc, 'date': d0})
    return pd.DataFrame(rows)


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

    cuts = retrain_cutoffs(start, end)
    rf = pd.Series(
        build_rf_daily(pd.date_range(start - pd.Timedelta(days=10),
                                     end + pd.Timedelta(days=10), freq='D')),
        index=pd.date_range(start - pd.Timedelta(days=10),
                            end + pd.Timedelta(days=10), freq='D'),
    )

    print(f"Emergency threshold sweep | seeds={SEEDS} | thresholds={THRESHOLDS}")

    # Train cutoffs PER SEED (since seed differs)
    all_seed_results = {seed: {} for seed in SEEDS}
    t_start = time.time()

    for seed in SEEDS:
        t0 = time.time()
        cutoff_models = {c: train_at_cutoff(ds, c, seed) for c in cuts}
        train_time = time.time() - t0
        print(f"\nSeed {seed}: trained {len(cuts)} cutoffs in {train_time:.0f}s", flush=True)
        print(f"  {'thresh':<8s} {'rebals':>7s} {'emerg-Fri':>11s} {'cum':>9s} {'Sortino':>8s} "
              f"{'Shp_x':>7s} {'DD':>8s}")

        for thr in THRESHOLDS:
            rebals, n_fri, n_emerg_nonfri = build_rebals(ds, start, end, thr)
            df = predict_rebals(ds, cutoff_models, rebals, rf)
            m = metrics(df['strat'].values, df['cdi'].values)
            m['n_rebals'] = len(df)
            m['n_emerg_nonfri'] = n_emerg_nonfri
            all_seed_results[seed][thr] = m
            marker = '   <-- atual' if abs(thr - 0.08) < 1e-9 else ''
            print(f"  {thr*100:5.1f}%   {len(df):>5d}    {n_emerg_nonfri:>5d}      "
                  f"{m['cum']*100:+8.1f}%  {m['sortino']:7.2f} {m['sharpe_x']:6.2f} "
                  f"{m['max_dd']*100:7.2f}%{marker}", flush=True)

    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")

    # Aggregate across seeds
    print(f"\n{'='*100}")
    print(f"AGGREGATED ACROSS {len(SEEDS)} SEEDS")
    print(f"{'='*100}")
    print(f"  {'thresh':<8s} {'rebals':>7s} {'cum mean ± std':>20s} {'Sortino mean ± std':>22s} "
          f"{'Shp_x mean':>12s} {'DD mean ± std':>16s}")
    print(f"  {'-'*100}")
    summary = {}
    for thr in THRESHOLDS:
        cums = np.array([all_seed_results[s][thr]['cum'] for s in SEEDS])
        sorts = np.array([all_seed_results[s][thr]['sortino'] for s in SEEDS])
        shps = np.array([all_seed_results[s][thr]['sharpe_x'] for s in SEEDS])
        dds = np.array([all_seed_results[s][thr]['max_dd'] for s in SEEDS])
        n_rebals = all_seed_results[SEEDS[0]][thr]['n_rebals']
        n_emerg = all_seed_results[SEEDS[0]][thr]['n_emerg_nonfri']
        marker = ' <-- atual' if abs(thr - 0.08) < 1e-9 else ''
        print(f"  {thr*100:5.1f}%   {n_rebals:>5d}   "
              f"{cums.mean()*100:+8.1f}% ± {cums.std(ddof=1)*100:5.1f}%   "
              f"{sorts.mean():6.2f} ± {sorts.std(ddof=1):.2f}        "
              f"{shps.mean():6.2f}     "
              f"{dds.mean()*100:6.2f}% ± {dds.std(ddof=1)*100:.2f}%{marker}")
        summary[f'{thr:.3f}'] = {
            'n_rebals': n_rebals, 'n_emerg_nonfri': n_emerg,
            'cum_mean': float(cums.mean()), 'cum_std': float(cums.std(ddof=1)),
            'sortino_mean': float(sorts.mean()), 'sortino_std': float(sorts.std(ddof=1)),
            'sharpe_x_mean': float(shps.mean()),
            'max_dd_mean': float(dds.mean()), 'max_dd_std': float(dds.std(ddof=1)),
        }

    # Save
    out_path = OUT / 'emergency_threshold_sweep.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
