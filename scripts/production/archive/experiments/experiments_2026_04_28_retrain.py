"""
Retrain experiments 2026-04-28: Huber loss + Drop-V36.

Walk-forward backtest with H1 (60/30/15), 248 rebals 2022-01 to 2026-04.
Two configs to compare against baseline:
  1. Huber: XGB objective 'reg:pseudohubererror' (robust to fat tails)
  2. Drop V36: remove 3 on-chain features, train with 29 features

Run:
    python scripts/production/archive/experiments/experiments_2026_04_28_retrain.py
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
OUT.mkdir(parents=True, exist_ok=True)

K_H1 = {'BULL': 60, 'MILD': 30, 'BEAR': 15}
RETRAIN_MONTHS = [1, 7]
V36_FEATURES = ['reserveRisk', 'funding_rate_ma7', 'puellMultiple']


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
    s, X, y, params = args
    m = xgb.XGBRegressor(**params, random_state=s)
    rng = np.random.RandomState(s)
    idx = rng.choice(len(X), size=len(X), replace=True)
    m.fit(X[idx], y[idx])
    return m


def _train_one_cls(args):
    s, X, y, params = args
    m = xgb.XGBClassifier(**params, random_state=s)
    rng = np.random.RandomState(s)
    idx = rng.choice(len(X), size=len(X), replace=True)
    m.fit(X[idx], y[idx])
    return m


def train_at_cutoff(ds, cutoff, features, xgb_params_reg, xgb_params_cls, seed=242):
    mask = ds['date'] < cutoff
    n = int(mask.sum())
    sub = ds.iloc[:n].reset_index(drop=True)
    X = np.nan_to_num(sub[features].values.astype(float), nan=0.0)
    prices = sub['price_usd'].values
    treg = np.zeros(n); tcls = np.zeros(n)
    for i in range(n - HORIZON):
        treg[i] = (prices[i + HORIZON] - prices[i]) / prices[i]
        tcls[i] = 1.0 if prices[i + HORIZON] > prices[i] else 0.0
    gap = max(HORIZON, 5)
    train_end = n - gap
    train_idx = np.arange(60, train_end + 1)
    train_idx = train_idx[~np.any(np.isnan(X[train_idx]), axis=1)]

    seeds = [seed + i * 7 for i in range(BAGS)]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        regs = list(ex.map(_train_one_reg,
            [(s, X[train_idx], treg[train_idx], xgb_params_reg) for s in seeds]))
        clss = list(ex.map(_train_one_cls,
            [(s, X[train_idx], tcls[train_idx], xgb_params_cls) for s in seeds]))
    return regs, clss


def run_walkforward(ds, features, xgb_params_reg, xgb_params_cls, label):
    print(f"\n{'='*70}")
    print(f"  Running: {label}  ({len(features)} features)")
    print(f"{'='*70}", flush=True)

    start = pd.Timestamp('2022-01-07')
    end = pd.Timestamp('2026-04-17')

    cuts = retrain_cutoffs(start, end)
    print(f"  Cutoffs: {[c.date() for c in cuts]}", flush=True)

    cutoff_models = {}
    for c in cuts:
        t0 = time.time()
        regs, clss = train_at_cutoff(ds, c, features, xgb_params_reg, xgb_params_cls)
        cutoff_models[c] = (regs, clss)
        print(f"  Trained @ {c.date()}: {time.time()-t0:.0f}s", flush=True)

    # Build rebal dates
    sub = ds[(ds['date'] >= start) & (ds['date'] <= end)].copy()
    sub['daily_ret'] = sub['price_usd'].pct_change()
    sub['dow'] = sub['date'].dt.dayofweek
    fridays = set(sub.loc[sub['dow'].isin(REBAL_DOW), 'date'])
    emerg = set(sub.loc[sub['daily_ret'].abs() > EMERGENCY_THRESHOLD, 'date'])
    rebals = sorted(fridays | emerg)
    print(f"  Rebals: {len(rebals)}", flush=True)

    # CDI series
    rf = pd.Series(
        build_rf_daily(pd.date_range(start, end, freq='D')),
        index=pd.date_range(start, end, freq='D'),
    )
    ds_by_date = ds.set_index('date')

    rows = []
    for i, d0 in enumerate(rebals):
        d1 = rebals[i + 1] if i + 1 < len(rebals) else end
        if d0 not in ds_by_date.index or d1 not in ds_by_date.index:
            continue
        applicable = [(c, m) for c, m in cutoff_models.items() if c <= d0]
        if not applicable:
            continue
        c = max(a[0] for a in applicable)
        regs, clss = cutoff_models[c]

        idx = ds[ds['date'] == d0].index[0]
        X_row = np.nan_to_num(
            ds.iloc[idx][features].values.astype(float).reshape(1, -1), nan=0.0
        )
        pred = float(np.mean([m.predict(X_row)[0] for m in regs]))
        p_up = float(np.mean([m.predict_proba(X_row)[0, 1] for m in clss]))
        conf = float(1 / (1 + np.exp(-abs(p_up - 0.5) * SIGMOID_SCALE)))

        prices = ds['price_usd'].values[:idx + 1]
        s50 = pd.Series(prices).rolling(50).mean().iloc[-1]
        s200 = pd.Series(prices).rolling(200).mean().iloc[-1]
        regime = get_regime(prices[-1], s50, s200)
        alloc = float(np.clip(pred * K_H1[regime] * conf, ALLOC_MIN, ALLOC_MAX))

        p0 = float(ds_by_date.loc[d0, 'price_usd'])
        p1 = float(ds_by_date.loc[d1, 'price_usd'])
        btc_ret = p1 / p0 - 1
        mask = (rf.index > d0) & (rf.index <= d1)
        cdi_ret = float((1 + rf[mask]).prod() - 1)
        strat_ret = alloc * btc_ret + (1 - alloc) * cdi_ret

        rows.append({
            'date': d0.strftime('%Y-%m-%d'),
            'pred': pred, 'p_up': p_up, 'regime': regime,
            'alloc': alloc, 'btc_fwd': btc_ret, 'strat': strat_ret,
            'variant': label,
        })

    return pd.DataFrame(rows)


def metrics(strat: np.ndarray) -> dict:
    cum = float(np.cumprod(1 + strat)[-1] - 1)
    neg = strat[strat < 0]
    dev = float(np.sqrt(np.mean(neg ** 2))) if len(neg) > 0 else 1e-9
    sortino = float(np.mean(strat) / dev * np.sqrt(52)) if dev > 0 else 0.0
    sd = float(np.std(strat))
    sharpe = float(np.mean(strat) / sd * np.sqrt(52)) if sd > 0 else 0.0
    eq = np.concatenate([[1.0], np.cumprod(1 + strat)])
    peak = np.maximum.accumulate(eq)
    maxdd = float(((eq - peak) / peak).min())
    return {'cum_ret': cum, 'sortino': sortino, 'sharpe': sharpe,
            'max_dd': maxdd, 'n': int(len(strat))}


def main():
    ds = pd.read_csv(DATASET, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    print(f"Dataset: {len(ds)} rows, {ds['date'].min().date()} to {ds['date'].max().date()}")

    base_reg = dict(XGB_PARAMS)  # reg:squarederror by default
    base_cls = {**{k: v for k, v in XGB_PARAMS.items() if k not in ('objective',)},
                'objective': 'binary:logistic', 'eval_metric': 'logloss'}

    all_results = {}

    # ── BASELINE: H1 + 32 features + squared error ──
    df_base = run_walkforward(ds, FEATURES_37, base_reg, base_cls,
                               label='baseline_H1_32feat_squared')
    m_base = metrics(df_base['strat'].values)
    m_base['avg_alloc'] = float(df_base['alloc'].mean())
    print(f"\n  BASELINE: cum={m_base['cum_ret']*100:+.1f}% Sortino={m_base['sortino']:.2f} "
          f"Sharpe={m_base['sharpe']:.2f} DD={m_base['max_dd']*100:.2f}%")
    all_results['baseline'] = m_base
    df_base.to_csv(OUT / 'experiments_2026_04_28_baseline.csv', index=False)

    # ── TEST 1: HUBER LOSS ──
    huber_reg = dict(base_reg)
    huber_reg['objective'] = 'reg:pseudohubererror'
    df_huber = run_walkforward(ds, FEATURES_37, huber_reg, base_cls,
                                label='huber_loss_H1_32feat')
    m_huber = metrics(df_huber['strat'].values)
    m_huber['avg_alloc'] = float(df_huber['alloc'].mean())
    print(f"\n  HUBER:    cum={m_huber['cum_ret']*100:+.1f}% Sortino={m_huber['sortino']:.2f} "
          f"Sharpe={m_huber['sharpe']:.2f} DD={m_huber['max_dd']*100:.2f}%")
    print(f"  vs base:  delta_cum={(m_huber['cum_ret']-m_base['cum_ret'])*100:+.1f}pp "
          f"delta_Sortino={m_huber['sortino']-m_base['sortino']:+.2f}")
    all_results['huber'] = m_huber
    df_huber.to_csv(OUT / 'experiments_2026_04_28_huber.csv', index=False)

    # ── TEST 5: DROP V36 ──
    feat_29 = [f for f in FEATURES_37 if f not in V36_FEATURES]
    df_29 = run_walkforward(ds, feat_29, base_reg, base_cls,
                             label='drop_V36_H1_29feat_squared')
    m_29 = metrics(df_29['strat'].values)
    m_29['avg_alloc'] = float(df_29['alloc'].mean())
    print(f"\n  DROP V36: cum={m_29['cum_ret']*100:+.1f}% Sortino={m_29['sortino']:.2f} "
          f"Sharpe={m_29['sharpe']:.2f} DD={m_29['max_dd']*100:.2f}%")
    print(f"  vs base:  delta_cum={(m_29['cum_ret']-m_base['cum_ret'])*100:+.1f}pp "
          f"delta_Sortino={m_29['sortino']-m_base['sortino']:+.2f}")
    all_results['drop_v36'] = m_29
    df_29.to_csv(OUT / 'experiments_2026_04_28_drop_v36.csv', index=False)

    # ── BONUS: Huber + drop V36 (combo) ──
    df_combo = run_walkforward(ds, feat_29, huber_reg, base_cls,
                                label='huber_drop_V36_H1_29feat')
    m_combo = metrics(df_combo['strat'].values)
    m_combo['avg_alloc'] = float(df_combo['alloc'].mean())
    print(f"\n  COMBO:    cum={m_combo['cum_ret']*100:+.1f}% Sortino={m_combo['sortino']:.2f} "
          f"Sharpe={m_combo['sharpe']:.2f} DD={m_combo['max_dd']*100:.2f}%")
    print(f"  vs base:  delta_cum={(m_combo['cum_ret']-m_base['cum_ret'])*100:+.1f}pp "
          f"delta_Sortino={m_combo['sortino']-m_base['sortino']:+.2f}")
    all_results['huber_drop_v36'] = m_combo
    df_combo.to_csv(OUT / 'experiments_2026_04_28_combo.csv', index=False)

    # Save summary
    with open(OUT / 'experiments_2026_04_28_retrain_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  {'config':25s} {'cum':>10s} {'Sortino':>9s} {'Sharpe':>8s} {'DD':>8s}")
    for k, v in all_results.items():
        print(f"  {k:25s} {v['cum_ret']*100:+9.1f}% {v['sortino']:9.2f} "
              f"{v['sharpe']:8.2f} {v['max_dd']*100:7.2f}%")


if __name__ == '__main__':
    main()
