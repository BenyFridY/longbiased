"""
Statistical validation: K_BULL=75 vs H1 across 10 seeds.

Question: Is the Pareto improvement of K_BULL=75 over H1 (seed=242) real, or
just coincidence of one seed?

Method: Re-use the seed predictions from seeds_validation_2026_04_28
(if cached) OR re-train. Apply BOTH H1 (60/30/15) and K_BULL75 (75/30/15)
post-hoc to each seed's predictions, compute metrics, compare.

Since K is post-prediction, we can apply both K configs to the SAME seed run,
making this a paired comparison (eliminates seed noise from the diff).
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
SEEDS = [242, 251, 263, 277, 281, 293, 307, 311, 317, 331]
RETRAIN_MONTHS = [1, 7]

K_H1   = {'BULL': 60, 'MILD': 30, 'BEAR': 15}
K_BULL75 = {'BULL': 75, 'MILD': 30, 'BEAR': 15}


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


def predict_seed(ds, seed, rebals, rf):
    cuts = retrain_cutoffs(rebals[0], rebals[-1])
    cutoff_models = {c: train_at_cutoff(ds, c, seed) for c in cuts}
    ds_by_date = ds.set_index('date')
    rows = []
    for i, d0 in enumerate(rebals):
        d1 = rebals[i + 1] if i + 1 < len(rebals) else rebals[-1] + pd.Timedelta(days=7)
        if d0 not in ds_by_date.index:
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
        if d1 not in ds_by_date.index:
            continue
        p0 = float(ds_by_date.loc[d0, 'price_usd'])
        p1 = float(ds_by_date.loc[d1, 'price_usd'])
        btc_ret = p1 / p0 - 1
        mask = (rf.index > d0) & (rf.index <= d1)
        cdi_ret = float((1 + rf[mask]).prod() - 1)
        rows.append({'date': d0, 'pred': pred, 'p_up': p_up, 'conf': conf,
                     'regime': regime, 'btc_fwd': btc_ret, 'cdi': cdi_ret})
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
    cagr = float((1 + cum) ** (52 / len(strat)) - 1) if len(strat) > 0 else 0.0
    return {'cum': cum, 'sortino': sortino, 'sharpe_x': sharpe_x,
            'max_dd': maxdd, 'cagr_w': cagr}


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

    print(f"Validating K_BULL=75 vs H1 across {len(SEEDS)} seeds")
    print(f"Rebals: {len(rebals)}, period 2022-01-07 → 2026-04-17")
    print(f"\n{'='*120}")
    print(f"{'seed':>5s} | {'H1 (60/30/15)':^45s} | {'K_BULL75 (75/30/15)':^45s} | "
          f"{'Δreturn':>9s} {'ΔSortino':>10s} {'ΔDD':>8s}")
    print(f"{'-'*120}")

    h1_results = []
    k75_results = []
    diffs = []
    for seed in SEEDS:
        t0 = time.time()
        df = predict_seed(ds, seed, rebals, rf)
        # H1
        K_arr = np.array([K_H1[r] for r in df['regime']])
        alloc_h1 = np.clip(df['pred'] * K_arr * df['conf'], 0, 1)
        strat_h1 = alloc_h1 * df['btc_fwd'] + (1 - alloc_h1) * df['cdi']
        m_h1 = metrics(strat_h1.values, df['cdi'].values)
        # K_BULL75
        K_arr75 = np.array([K_BULL75[r] for r in df['regime']])
        alloc_k75 = np.clip(df['pred'] * K_arr75 * df['conf'], 0, 1)
        strat_k75 = alloc_k75 * df['btc_fwd'] + (1 - alloc_k75) * df['cdi']
        m_k75 = metrics(strat_k75.values, df['cdi'].values)

        d_ret = (m_k75['cum'] - m_h1['cum']) * 100
        d_sortino = m_k75['sortino'] - m_h1['sortino']
        d_dd = (m_k75['max_dd'] - m_h1['max_dd']) * 100
        h1_results.append(m_h1)
        k75_results.append(m_k75)
        diffs.append({'d_ret': d_ret, 'd_sortino': d_sortino, 'd_dd': d_dd,
                       'd_sharpe': m_k75['sharpe_x'] - m_h1['sharpe_x']})

        elapsed = time.time() - t0
        print(f"{seed:>5d} | "
              f"{m_h1['cum']*100:+7.1f}% S={m_h1['sortino']:.2f} Shp={m_h1['sharpe_x']:.2f} DD={m_h1['max_dd']*100:.2f}% | "
              f"{m_k75['cum']*100:+7.1f}% S={m_k75['sortino']:.2f} Shp={m_k75['sharpe_x']:.2f} DD={m_k75['max_dd']*100:.2f}% | "
              f"{d_ret:+8.1f}pp {d_sortino:+9.3f} {d_dd:+7.2f}pp ({elapsed:.0f}s)", flush=True)

    print(f"{'='*120}")

    def stats(lst, key):
        v = np.array([r[key] for r in lst])
        return float(v.mean()), float(v.std(ddof=1)), float(v.min()), float(v.max())

    print(f"\nAGGREGATED ({len(SEEDS)} seeds):")
    print(f"{'='*120}")
    print(f"  {'Metric':<20s} {'H1 mean ± std':>22s} {'K_BULL75 mean ± std':>26s} "
          f"{'Diff (paired)':>22s} {'Win rate':>10s}")
    print(f"  {'-'*100}")
    for key, label, fmt in [
        ('cum',      'cum_return',   '{:+7.1f}%'),
        ('sortino',  'Sortino weekly', '{:7.2f}'),
        ('sharpe_x', 'Sharpe excess', '{:7.2f}'),
        ('max_dd',   'Max DD weekly', '{:7.2f}%'),
    ]:
        h1_mean, h1_std, _, _ = stats(h1_results, key)
        k75_mean, k75_std, _, _ = stats(k75_results, key)
        # Paired diff
        diff_key = {'cum': 'd_ret', 'sortino': 'd_sortino',
                    'sharpe_x': 'd_sharpe', 'max_dd': 'd_dd'}[key]
        diff_arr = np.array([d[diff_key] for d in diffs])
        d_mean = float(diff_arr.mean())
        d_std = float(diff_arr.std(ddof=1))
        win_rate = float((diff_arr > 0).mean()) if key != 'max_dd' else float((diff_arr >= 0).mean())
        scale = 1
        if key in ('cum',):
            h1_str = fmt.format(h1_mean*100); k75_str = fmt.format(k75_mean*100)
            d_str = f"{d_mean:+6.1f}pp ± {d_std:.1f}pp"
        elif key == 'max_dd':
            h1_str = fmt.format(h1_mean*100); k75_str = fmt.format(k75_mean*100)
            d_str = f"{d_mean:+5.2f}pp ± {d_std:.2f}pp"
        else:
            h1_str = fmt.format(h1_mean); k75_str = fmt.format(k75_mean)
            d_str = f"{d_mean:+6.3f} ± {d_std:.3f}"
        print(f"  {label:<20s} {h1_str:>13s} ± {h1_std:.2f}{'%' if key in ('cum','max_dd') else '  ':<2s} "
              f"{k75_str:>14s} ± {k75_std:.2f}{'%' if key in ('cum','max_dd') else '  ':<2s} "
              f"{d_str:>22s} {win_rate*100:>8.0f}%")

    # T-test on paired diffs (do K_BULL75 - H1 distributions differ from 0?)
    from scipy import stats as scipy_stats
    print(f"\nPaired t-test (does K_BULL75 differ from H1?):")
    for key, label in [('d_ret', 'Δreturn'), ('d_sortino', 'ΔSortino'),
                       ('d_sharpe', 'ΔSharpe'), ('d_dd', 'ΔDD')]:
        arr = np.array([d[key] for d in diffs])
        t_stat, p_val = scipy_stats.ttest_1samp(arr, 0)
        sig = '*** SIGNIFICANT ***' if p_val < 0.05 else '(not significant, p>=0.05)'
        print(f"  {label:<10s} mean={arr.mean():+7.3f}  t={t_stat:+5.2f}  p={p_val:.4f}  {sig}")

    out = {
        'config': 'K_BULL=75 vs K_BULL=60 (H1), MILD/BEAR same (30/15)',
        'seeds': SEEDS,
        'h1_results': h1_results,
        'k75_results': k75_results,
        'diffs': diffs,
    }
    with open(OUT / 'k_bull75_validation_10seeds.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {OUT / 'k_bull75_validation_10seeds.json'}")


if __name__ == '__main__':
    main()
