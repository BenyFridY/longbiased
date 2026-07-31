"""
Quantile regression for sizing — Test 4 (deferred from earlier audit).

Theory: BTC returns have skew +3.63 and excess kurtosis +16.4. Squared-error
regression predicts MEAN, dominated by right-tail outliers. Quantile regression
predicts arbitrary percentiles directly:
  - q10: 10th percentile of 3d return distribution (downside floor)
  - q50: median (robust central tendency)
  - q90: 90th percentile (upside potential)

Then size based on risk-aware ratios instead of raw mean.

Sizing variants to compare against baseline (mean * K * conf):
  V1. Pure q50 (median) — robust to right-tail
  V2. Risk-adjusted: q50 / max(|q10|, eps) * fraction (Kelly-style tail)
  V3. Conditional: alloc only if q10 > 0 (even tail positive)
  V4. Spread-scaled: K reduced when (q90-q10) is wide (high uncertainty)
  V5. Hybrid: q50 if q10 > 0 else 0 (binary safety)

Walk-forward 248 rebals 2022-01 to 2026-04, H1 K=60/30/15, single seed=242
(baseline run for comparison; can extend to 10-seed if winner emerges).

Run:
    python scripts/production/archive/experiments/quantile_regression_2026_04_28.py
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


def _train_one_reg_mean(args):
    s, X, y = args
    m = xgb.XGBRegressor(**XGB_PARAMS, random_state=s)
    rng = np.random.RandomState(s)
    idx = rng.choice(len(X), size=len(X), replace=True)
    m.fit(X[idx], y[idx])
    return m


def _train_one_reg_quantile(args):
    s, X, y, alpha = args
    # XGBoost 3.x quantile regression
    params = {k: v for k, v in XGB_PARAMS.items() if k != 'objective'}
    m = xgb.XGBRegressor(
        **params,
        objective='reg:quantileerror',
        quantile_alpha=alpha,
        random_state=s,
    )
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
    Xtr, ytr = X[train_idx], treg[train_idx]

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        # Mean (baseline reproduction)
        regs_mean = list(ex.map(_train_one_reg_mean,
            [(s, Xtr, ytr) for s in seeds]))
        # Quantile q10
        regs_q10 = list(ex.map(_train_one_reg_quantile,
            [(s, Xtr, ytr, 0.10) for s in seeds]))
        # Quantile q50 (median)
        regs_q50 = list(ex.map(_train_one_reg_quantile,
            [(s, Xtr, ytr, 0.50) for s in seeds]))
        # Quantile q90
        regs_q90 = list(ex.map(_train_one_reg_quantile,
            [(s, Xtr, ytr, 0.90) for s in seeds]))
        # Classifier
        ytr_cls = tcls[train_idx]
        clss = list(ex.map(_train_one_cls,
            [(s, Xtr, ytr_cls) for s in seeds]))
    return {'mean': regs_mean, 'q10': regs_q10, 'q50': regs_q50,
            'q90': regs_q90, 'cls': clss}


def predict_all(models, X_row):
    return {
        'mean': float(np.mean([m.predict(X_row)[0] for m in models['mean']])),
        'q10':  float(np.mean([m.predict(X_row)[0] for m in models['q10']])),
        'q50':  float(np.mean([m.predict(X_row)[0] for m in models['q50']])),
        'q90':  float(np.mean([m.predict(X_row)[0] for m in models['q90']])),
        'p_up': float(np.mean([m.predict_proba(X_row)[0, 1] for m in models['cls']])),
    }


def metrics(strat: np.ndarray, cdi: np.ndarray = None) -> dict:
    cum = float(np.cumprod(1 + strat)[-1] - 1)
    neg = strat[strat < 0]
    dev = float(np.sqrt(np.mean(neg ** 2))) if len(neg) > 0 else 1e-9
    sortino = float(np.mean(strat) / dev * np.sqrt(52)) if dev > 0 else 0.0
    sd = float(np.std(strat, ddof=0))
    sharpe_abs = float(np.mean(strat) / sd * np.sqrt(52)) if sd > 0 else 0.0
    sharpe_x = None
    if cdi is not None:
        excess = strat - cdi
        sd_e = float(np.std(excess, ddof=0))
        sharpe_x = float(np.mean(excess) / sd_e * np.sqrt(52)) if sd_e > 0 else 0.0
    eq = np.concatenate([[1.0], np.cumprod(1 + strat)])
    peak = np.maximum.accumulate(eq)
    maxdd = float(((eq - peak) / peak).min())
    return {
        'cum_ret': cum, 'sortino': sortino,
        'sharpe_abs': sharpe_abs, 'sharpe_excess': sharpe_x,
        'max_dd': maxdd,
    }


def main():
    print(f"XGBoost: {xgb.__version__}")
    print(f"Seed: {SEED}, K=H1, sigmoid=15")

    ds = pd.read_csv(DATASET, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    start = pd.Timestamp('2022-01-07')
    end = pd.Timestamp('2026-04-17')

    cuts = retrain_cutoffs(start, end)
    print(f"Cutoffs: {len(cuts)}")

    cutoff_models = {}
    t0_total = time.time()
    for c in cuts:
        t0 = time.time()
        cutoff_models[c] = train_at_cutoff(ds, c, SEED)
        print(f"  Trained @ {c.date()}: {time.time()-t0:.0f}s "
              f"(mean+q10+q50+q90+cls = {sum(len(v) for v in cutoff_models[c].values())} models)",
              flush=True)
    print(f"  Total training: {time.time()-t0_total:.0f}s", flush=True)

    # Build rebals
    sub = ds[(ds['date'] >= start) & (ds['date'] <= end)].copy()
    sub['daily_ret'] = sub['price_usd'].pct_change()
    sub['dow'] = sub['date'].dt.dayofweek
    fridays = set(sub.loc[sub['dow'].isin(REBAL_DOW), 'date'])
    emerg = set(sub.loc[sub['daily_ret'].abs() > EMERGENCY_THRESHOLD, 'date'])
    rebals = sorted(fridays | emerg)
    print(f"Rebals: {len(rebals)}", flush=True)

    rf = pd.Series(
        build_rf_daily(pd.date_range(start, end + pd.Timedelta(days=10), freq='D')),
        index=pd.date_range(start, end + pd.Timedelta(days=10), freq='D'),
    )
    ds_by_date = ds.set_index('date')

    # Generate predictions for all rebals
    rows = []
    for i, d0 in enumerate(rebals):
        d1 = rebals[i + 1] if i + 1 < len(rebals) else end
        if d0 not in ds_by_date.index or d1 not in ds_by_date.index:
            continue
        applicable = [(c, m) for c, m in cutoff_models.items() if c <= d0]
        c = max(a[0] for a in applicable)
        models = cutoff_models[c]
        idx = ds[ds['date'] == d0].index[0]
        X_row = np.nan_to_num(
            ds.iloc[idx][FEATURES_37].values.astype(float).reshape(1, -1), nan=0.0
        )
        preds = predict_all(models, X_row)
        prices = ds['price_usd'].values[:idx + 1]
        s50 = pd.Series(prices).rolling(50).mean().iloc[-1]
        s200 = pd.Series(prices).rolling(200).mean().iloc[-1]
        regime = get_regime(prices[-1], s50, s200)
        p0 = float(ds_by_date.loc[d0, 'price_usd'])
        p1 = float(ds_by_date.loc[d1, 'price_usd'])
        btc_ret = p1 / p0 - 1
        mask = (rf.index > d0) & (rf.index <= d1)
        cdi_ret = float((1 + rf[mask]).prod() - 1)
        conf = float(1 / (1 + np.exp(-abs(preds['p_up'] - 0.5) * SIGMOID_SCALE)))
        rows.append({
            'date': d0, 'regime': regime, 'K_base': K_H1[regime], 'conf': conf,
            'pred_mean': preds['mean'], 'pred_q10': preds['q10'],
            'pred_q50': preds['q50'], 'pred_q90': preds['q90'],
            'p_up': preds['p_up'], 'btc_fwd': btc_ret, 'cdi': cdi_ret,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'quantile_regression_predictions.csv', index=False)
    print(f"Saved predictions: {OUT / 'quantile_regression_predictions.csv'}", flush=True)

    # === Compute alloc + strat for each sizing variant ===
    K_arr = df['K_base'].values
    conf = df['conf'].values
    btc = df['btc_fwd'].values
    cdi = df['cdi'].values
    eps = 1e-4

    variants = {}

    # BASELINE: mean * K * conf
    alloc = np.clip(df['pred_mean'].values * K_arr * conf, 0, 1)
    variants['baseline_mean'] = (alloc, alloc * btc + (1 - alloc) * cdi)

    # V1: Pure q50 (median replaces mean)
    alloc = np.clip(df['pred_q50'].values * K_arr * conf, 0, 1)
    variants['V1_q50'] = (alloc, alloc * btc + (1 - alloc) * cdi)

    # V2: Risk-adjusted (Kelly tail). alloc ~ q50 / max(|q10|, eps) * f
    # Use small fraction since this can blow up
    for f in [0.05, 0.10, 0.15, 0.20]:
        q50 = df['pred_q50'].values
        q10 = df['pred_q10'].values
        risk = np.maximum(np.abs(q10), eps)
        a = np.clip(q50 / risk * f * conf, 0, 1)
        # Only allocate when q50 > 0
        a = np.where(q50 > 0, a, 0.0)
        variants[f'V2_kelly_q50_q10_f{int(f*100)}'] = (a, a * btc + (1 - a) * cdi)

    # V3: Conditional — only allocate if q10 > 0
    pred = df['pred_mean'].values
    a = np.clip(pred * K_arr * conf, 0, 1)
    a = np.where(df['pred_q10'].values > 0, a, 0.0)
    variants['V3_only_if_q10_pos'] = (a, a * btc + (1 - a) * cdi)

    # V4: Spread-scaled K (reduce K when q90-q10 is wide)
    # Scale = 1 if narrow spread, < 1 if wide
    spread = df['pred_q90'].values - df['pred_q10'].values
    # Normalize: spread relative to typical (use 75th percentile as reference)
    spread_ref = np.percentile(spread, 75)
    scale = np.clip(spread_ref / np.maximum(spread, eps), 0.3, 1.0)
    a = np.clip(df['pred_mean'].values * K_arr * scale * conf, 0, 1)
    variants['V4_spread_scaled'] = (a, a * btc + (1 - a) * cdi)

    # V5: Hybrid binary safety — q50 if q10>0 else 0
    a = np.where(df['pred_q10'].values > 0,
                 np.clip(df['pred_q50'].values * K_arr * conf, 0, 1),
                 0.0)
    variants['V5_hybrid_q50_safe'] = (a, a * btc + (1 - a) * cdi)

    # === Report ===
    print(f"\n{'='*100}")
    print(f"{'Variant':<32s} {'cum':>10s} {'Sortino':>9s} {'Sharpe_abs':>11s} {'Shp_x':>7s} {'DD_w':>8s} {'avg_alloc':>10s}")
    print(f"{'-'*100}")
    results = {}
    for name, (alloc, strat) in variants.items():
        m = metrics(strat, cdi)
        m['avg_alloc'] = float(alloc.mean())
        m['n_active'] = int((alloc > 0).sum())
        results[name] = m
        print(f"{name:<32s} {m['cum_ret']*100:+9.1f}% {m['sortino']:9.2f} "
              f"{m['sharpe_abs']:11.2f} {m['sharpe_excess']:7.2f} "
              f"{m['max_dd']*100:7.2f}% {m['avg_alloc']*100:9.1f}%")

    # Save summary
    with open(OUT / 'quantile_regression_summary.json', 'w') as f:
        json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
                  f, indent=2, default=str)
    print(f"\nSaved: {OUT / 'quantile_regression_summary.json'}")

    # === Picks ===
    print(f"\n{'='*100}")
    print("WINNERS")
    print(f"{'='*100}")
    base = results['baseline_mean']
    print(f"\nBaseline:        cum={base['cum_ret']*100:+.1f}%  Sortino={base['sortino']:.2f}  "
          f"Sharpe={base['sharpe_excess']:.2f}  DD={base['max_dd']*100:.2f}%")
    by_sortino = sorted(results.items(), key=lambda x: -x[1]['sortino'])
    print(f"\nTop 3 by Sortino:")
    for name, m in by_sortino[:3]:
        delta = m['sortino'] - base['sortino']
        print(f"  {m['sortino']:5.2f} ({delta:+.2f})  {name:30s}  cum={m['cum_ret']*100:+.1f}%")
    by_sharpe = sorted(results.items(),
                       key=lambda x: -x[1]['sharpe_excess'] if x[1]['sharpe_excess'] is not None else -99)
    print(f"\nTop 3 by Sharpe excess:")
    for name, m in by_sharpe[:3]:
        delta = m['sharpe_excess'] - base['sharpe_excess']
        print(f"  {m['sharpe_excess']:5.2f} ({delta:+.2f})  {name:30s}  cum={m['cum_ret']*100:+.1f}%")


if __name__ == '__main__':
    main()
