"""Regen baseline.csv with 160 bags (single seed 242) for chart updates."""
import sys
import time
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
SEED = 242
K_H1 = {'BULL': 60, 'MILD': 30, 'BEAR': 15}
RETRAIN_MONTHS = [1, 7]


def get_regime(price, sma50, sma200):
    if np.isnan(sma50) or np.isnan(sma200):
        return 'MILD'
    if price > sma50 and sma50 > sma200:
        return 'BULL'
    elif price > sma200:
        return 'MILD'
    return 'BEAR'


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


def main():
    print(f"BAGS = {BAGS}")
    ds = pd.read_csv(DATASET, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    start = pd.Timestamp('2022-01-07')
    end = pd.Timestamp('2026-04-17')

    cuts = []
    y = start.year - 1
    while y <= end.year + 1:
        for m in RETRAIN_MONTHS:
            d = pd.Timestamp(year=y, month=m, day=1)
            if d <= end:
                cuts.append(d)
        y += 1
    cuts = sorted(set(c for c in cuts if c >= pd.Timestamp('2022-01-01')))

    cutoff_models = {}
    t0 = time.time()
    for c in cuts:
        ts = time.time()
        mask = ds['date'] < c
        n = int(mask.sum())
        sub = ds.iloc[:n].reset_index(drop=True)
        X = np.nan_to_num(sub[FEATURES_37].values.astype(float), nan=0.0)
        prices = sub['price_usd'].values
        treg = np.zeros(n); tcls = np.zeros(n)
        for i in range(n - HORIZON):
            treg[i] = (prices[i + HORIZON] - prices[i]) / prices[i]
            tcls[i] = 1.0 if prices[i + HORIZON] > prices[i] else 0.0
        train_idx = np.arange(60, n - max(HORIZON, 5) + 1)
        train_idx = train_idx[~np.any(np.isnan(X[train_idx]), axis=1)]
        seeds = [SEED + i * 7 for i in range(BAGS)]
        Xtr = X[train_idx]
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            regs = list(ex.map(_train_one_reg, [(s, Xtr, treg[train_idx]) for s in seeds]))
            clss = list(ex.map(_train_one_cls, [(s, Xtr, tcls[train_idx]) for s in seeds]))
        cutoff_models[c] = (regs, clss)
        print(f"  {c.date()}: {time.time()-ts:.0f}s", flush=True)

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
    ds_by_date = ds.set_index('date')

    rows = []
    for i, d0 in enumerate(rebals):
        d1 = rebals[i + 1] if i + 1 < len(rebals) else end
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
                     'alloc': alloc, 'btc_fwd': btc_ret, 'strat': strat_ret,
                     'variant': f'baseline_H1_32feat_{BAGS}bags'})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'experiments_2026_04_28_baseline.csv', index=False)
    print(f"\nTotal: {time.time()-t0:.0f}s, {len(df)} rebals")
    print(f"Saved baseline with {BAGS} bags")


if __name__ == '__main__':
    main()
