"""
Final validation — 2 paired tests with 5 seeds in BRL + 4 bps:

1. DROP V36: 32 features → 29 features (remove reserveRisk, funding_rate_ma7,
   puellMultiple). Earlier 1-seed test was ambiguous: return up, Sortino down.

2. 160 BAGS vs 80 BAGS: V22-era concluded 80 was optimal. Re-test on Ultra 9
   to confirm with current canonical setup.

Variants:
  A. baseline             32 feat, 80 bags  (atual)
  B. drop_v36             29 feat, 80 bags
  C. bags160              32 feat, 160 bags
  D. drop_v36_bags160     29 feat, 160 bags

5 seeds × 4 variants × 9 cutoffs. Estimated ~25-30 min on Ultra 9.
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
    REBAL_DOW, EMERGENCY_THRESHOLD, HORIZON, WORKERS,
)
from src.features.macro.cdi_rates import build_rf_daily

DATASET = ROOT / 'scripts/production/data/dataset_production.csv'
OUT = ROOT / 'outputs/results'
SEEDS = [242, 251, 263, 277, 281]
RETRAIN_MONTHS = [1, 7]
K_H1 = {'BULL': 60, 'MILD': 30, 'BEAR': 15}
V36_FEATURES = ['reserveRisk', 'funding_rate_ma7', 'puellMultiple']

VARIANTS = [
    ('A. baseline (32f, 80b)',          FEATURES_37,                              80),
    ('B. drop_v36 (29f, 80b)',          [f for f in FEATURES_37 if f not in V36_FEATURES], 80),
    ('C. bags160 (32f, 160b)',          FEATURES_37,                              160),
    ('D. drop_v36 + bags160 (29f, 160b)', [f for f in FEATURES_37 if f not in V36_FEATURES], 160),
]

# BRL FX
fx_raw = pd.read_csv(OUT / 'usd_brl.csv', skiprows=[1, 2])
fx_raw.columns = ['date', 'close', 'high', 'low', 'open', 'volume']
fx_raw['date'] = pd.to_datetime(fx_raw['date'])
fx = fx_raw[['date', 'close']].rename(columns={'close': 'usdbrl'})


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


def train_at_cutoff(ds, cutoff, base_seed, features, bags):
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
    seeds = [base_seed + i * 7 for i in range(bags)]
    Xtr, ytr_reg, ytr_cls = X[train_idx], treg[train_idx], tcls[train_idx]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        regs = list(ex.map(_train_one_reg, [(s, Xtr, ytr_reg) for s in seeds]))
        clss = list(ex.map(_train_one_cls, [(s, Xtr, ytr_cls) for s in seeds]))
    return regs, clss


def run_seed_variant(ds, seed, features, bags, rebals, rf):
    cuts = retrain_cutoffs(rebals[0], rebals[-1])
    cutoff_models = {c: train_at_cutoff(ds, c, seed, features, bags) for c in cuts}
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
            ds.iloc[idx][features].values.astype(float).reshape(1, -1), nan=0.0
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
        u0 = float(ds_by_date.loc[d0, 'usdbrl'])
        u1 = float(ds_by_date.loc[d1, 'usdbrl'])
        btc_brl_ret = (p1 * u1) / (p0 * u0) - 1
        mask_cdi = (rf.index > d0) & (rf.index <= d1)
        cdi_ret = float((1 + rf[mask_cdi]).prod() - 1)
        rows.append({'date': d0, 'alloc': alloc, 'btc_fwd': btc_brl_ret,
                     'cdi': cdi_ret, 'regime': regime})
    return pd.DataFrame(rows)


def daily_metrics(rebal_df, ds, start, end, cost_bps=4):
    daily_ds = ds[(ds['date'] >= start) & (ds['date'] <= end)].copy()
    daily_ds['btc_ret_brl'] = (daily_ds['price_usd'] * daily_ds['usdbrl']).pct_change().fillna(0)
    cdi_daily = (1.13)**(1/365) - 1
    rebal_dates = rebal_df['date'].values
    rebal_alloc = rebal_df['alloc'].values
    rets = []
    prev = 0
    for _, row in daily_ds.iterrows():
        mask = rebal_dates < row['date'].to_numpy()
        new_alloc = rebal_alloc[mask][-1] if mask.any() else 0
        cost = abs(new_alloc - prev) * cost_bps / 10000 if new_alloc != prev else 0
        rets.append(new_alloc * row['btc_ret_brl'] + (1 - new_alloc) * cdi_daily - cost)
        if new_alloc != prev:
            prev = new_alloc
    rets = np.array(rets)
    cum = float(np.cumprod(1 + rets)[-1] - 1)
    cagr = (1 + cum) ** (365 / len(rets)) - 1
    neg = rets[rets < 0]
    dev = float(np.sqrt(np.mean(neg ** 2))) if len(neg) > 0 else 1e-9
    sortino = float(np.mean(rets) / dev * np.sqrt(365)) if dev > 0 else 0
    excess = rets - cdi_daily
    sd_e = float(np.std(excess, ddof=0))
    sharpe_x = float(np.mean(excess) / sd_e * np.sqrt(365)) if sd_e > 0 else 0
    eq = np.concatenate([[1.0], np.cumprod(1 + rets)])
    peak = np.maximum.accumulate(eq)
    maxdd = float(((eq - peak) / peak).min())

    daily_ds = daily_ds.reset_index(drop=True)
    yearly = {}
    for y in [2022, 2023, 2024, 2025, 2026]:
        if y == 2022:
            mask = (daily_ds['date'] >= '2022-01-07') & (daily_ds['date'] <= '2022-12-31')
        elif y == 2026:
            mask = (daily_ds['date'] >= '2026-01-01') & (daily_ds['date'] <= '2026-04-17')
        else:
            mask = (daily_ds['date'] >= f'{y}-01-01') & (daily_ds['date'] <= f'{y}-12-31')
        if mask.any():
            yearly[y] = float(np.cumprod(1 + rets[mask.values])[-1] - 1)
    return {'cum': cum, 'cagr': cagr, 'sortino': sortino,
            'sharpe_x': sharpe_x, 'max_dd': maxdd, 'yearly': yearly}


def main():
    ds = pd.read_csv(DATASET, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    ds = ds.merge(fx, on='date', how='left')
    ds['usdbrl'] = ds['usdbrl'].ffill().bfill()
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

    print(f"Final validation: {len(VARIANTS)} variants × {len(SEEDS)} seeds, BRL + 4 bps")
    print()

    all_results = {v[0]: [] for v in VARIANTS}
    t_total = time.time()

    for v_name, features, bags in VARIANTS:
        print(f"\n=== {v_name}  ({len(features)} feat, {bags} bags) ===", flush=True)
        for seed in SEEDS:
            t0 = time.time()
            df = run_seed_variant(ds, seed, features, bags, rebals, rf)
            m = daily_metrics(df, ds, start, end)
            m['seed'] = seed
            all_results[v_name].append(m)
            print(f"  seed {seed}: cum={m['cum']*100:+7.1f}%  CAGR={m['cagr']*100:+5.1f}%  "
                  f"Sortino={m['sortino']:5.2f}  Shp_x={m['sharpe_x']:5.2f}  "
                  f"DD={m['max_dd']*100:6.2f}%  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\nTotal time: {(time.time()-t_total)/60:.1f} min")

    # Aggregate
    def agg(lst, k):
        v = np.array([r[k] for r in lst])
        return float(v.mean()), float(v.std(ddof=1))

    print(f"\n{'='*120}")
    print(f"AGGREGATED ({len(SEEDS)} seeds, BRL + 4 bps)")
    print(f"{'='*120}")
    print(f"{'Variant':<32s} {'CAGR':>15s} {'Sortino':>15s} {'Sharpe_x':>15s} {'DD':>15s}")
    print(f"{'-'*120}")
    summary = {}
    for v_name in [v[0] for v in VARIANTS]:
        cm, cs = agg(all_results[v_name], 'cagr')
        sm, ss = agg(all_results[v_name], 'sortino')
        sxm, sxs = agg(all_results[v_name], 'sharpe_x')
        dm, dds = agg(all_results[v_name], 'max_dd')
        print(f"{v_name:<32s} {cm*100:+8.1f}% ± {cs*100:.1f}%  "
              f"{sm:6.2f} ± {ss:.2f}      {sxm:.2f} ± {sxs:.2f}      "
              f"{dm*100:6.2f}% ± {dds*100:.2f}%")
        summary[v_name] = {'cagr_mean': cm, 'cagr_std': cs,
                           'sortino_mean': sm, 'sortino_std': ss,
                           'sharpe_x_mean': sxm, 'sharpe_x_std': sxs,
                           'max_dd_mean': dm, 'max_dd_std': dds}

    # Paired comparisons vs baseline
    print(f"\n{'='*120}")
    print("PAIRED vs A. baseline")
    print(f"{'='*120}")
    base = all_results['A. baseline (32f, 80b)']
    for v_name in [v[0] for v in VARIANTS]:
        if 'baseline' in v_name:
            continue
        var = all_results[v_name]
        d_cagr = [v['cagr'] - b['cagr'] for v, b in zip(var, base)]
        d_sort = [v['sortino'] - b['sortino'] for v, b in zip(var, base)]
        d_sx = [v['sharpe_x'] - b['sharpe_x'] for v, b in zip(var, base)]
        d_dd = [v['max_dd'] - b['max_dd'] for v, b in zip(var, base)]
        wins_cagr = sum(1 for d in d_cagr if d > 0)
        wins_sort = sum(1 for d in d_sort if d > 0)
        print(f"{v_name:<32s} ΔCAGR={np.mean(d_cagr)*100:+5.1f}pp  ΔSortino={np.mean(d_sort):+5.2f}  "
              f"ΔSharpe_x={np.mean(d_sx):+5.2f}  ΔDD={np.mean(d_dd)*100:+5.2f}pp  "
              f"wins_cagr:{wins_cagr}/5  wins_sortino:{wins_sort}/5")

    # T-tests
    try:
        from scipy import stats as scipy_stats
        print(f"\nPaired t-tests (vs baseline):")
        for v_name in [v[0] for v in VARIANTS]:
            if 'baseline' in v_name:
                continue
            var = all_results[v_name]
            for metric, key in [('CAGR', 'cagr'), ('Sortino', 'sortino'),
                                 ('Sharpe_x', 'sharpe_x'), ('DD', 'max_dd')]:
                arr = np.array([v[key] - b[key] for v, b in zip(var, base)])
                t, p = scipy_stats.ttest_1samp(arr, 0)
                sig = '*' if p < 0.05 else ' '
                print(f"  {v_name[:30]:<30s} {metric:<10s}: mean {arr.mean():+7.4f}  "
                      f"p={p:.4f} {sig}")
    except ImportError:
        pass

    # Save
    with open(OUT / 'final_validation_features_bags.json', 'w') as f:
        json.dump({'config': 'BRL + 4 bps', 'seeds': SEEDS,
                   'summary': summary, 'all_results': all_results}, f, indent=2, default=str)
    print(f"\nSaved: {OUT / 'final_validation_features_bags.json'}")


if __name__ == '__main__':
    main()
