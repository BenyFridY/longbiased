"""
Multi-year HORIZON=3 vs HORIZON=7 walk-forward ablation.

Same walk-forward semantics as walkforward_backtest.py:
  - Retrain at each semi-annual cutoff (Jan/Jul 1, 2022-2026)
  - Walk through all Fridays + emergencies
  - Apply K_REGIME × sigmoid confidence, clip [0,1]

Two K configs for H=7:
  - Same K (100/50/20)  -> magnitudes saturate, unfair to H=3
  - Scaled K (3/7 of H=3) -> magnitude-neutral

Run:
    python scripts/production/archive/experiments/horizon_ablation_4y.py
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts' / 'production'))

from config import (
    FEATURES_37, K_REGIME, ALLOC_MIN, ALLOC_MAX, SIGMOID_SCALE,
    REBAL_DOW, EMERGENCY_THRESHOLD,
)
from generate_signal import train_regression_ensemble, train_classifier_ensemble, get_regime
from src.features.macro.cdi_rates import build_rf_daily


RETRAIN_MONTHS = [1, 7]


def retrain_cutoffs(start, end):
    cutoffs = []
    y = start.year - 1
    while y <= end.year + 1:
        for m in RETRAIN_MONTHS:
            d = pd.Timestamp(year=y, month=m, day=1)
            if d <= end:
                cutoffs.append(d)
        y += 1
    return sorted(set(c for c in cutoffs if c >= pd.Timestamp('2022-01-01')))


def train_at_cutoff(ds, cutoff, horizon):
    mask = ds['date'] < cutoff
    n = int(mask.sum())
    sub = ds.iloc[:n].reset_index(drop=True)
    X = np.nan_to_num(sub[FEATURES_37].values.astype(float), nan=0.0)
    prices = sub['price_usd'].values
    treg = np.zeros(n); tcls = np.zeros(n)
    for i in range(n - horizon):
        treg[i] = (prices[i + horizon] - prices[i]) / prices[i]
        tcls[i] = 1.0 if prices[i + horizon] > prices[i] else 0.0
    gap = max(horizon, 5)
    train_end = n - gap
    train_idx = np.arange(60, train_end + 1)
    train_idx = train_idx[~np.any(np.isnan(X[train_idx]), axis=1)]
    reg = train_regression_ensemble(X[train_idx], treg[train_idx])
    cls = train_classifier_ensemble(X[train_idx], tcls[train_idx])
    return reg, cls


def pick_models(cutoff_models, d0):
    applicable = [(c, m) for c, m in cutoff_models.items() if c <= d0]
    c = max(a[0] for a in applicable)
    return c, cutoff_models[c]


def rebalance_dates(ds, start, end):
    sub = ds[(ds['date'] >= start) & (ds['date'] <= end)].copy()
    sub['daily_ret'] = sub['price_usd'].pct_change()
    sub['dow'] = sub['date'].dt.dayofweek
    fri = set(sub.loc[sub['dow'].isin(REBAL_DOW), 'date'])
    emg = set(sub.loc[sub['daily_ret'].abs() > EMERGENCY_THRESHOLD, 'date'])
    return sorted(fri | emg)


def run(ds, start, end, cutoff_models, K_map, rf_series):
    rebals = rebalance_dates(ds, start, end)
    ds_by = ds.set_index('date')
    cum = 1.0; cum_btc = 1.0
    rows = []
    for i, d0 in enumerate(rebals):
        d1 = rebals[i + 1] if i + 1 < len(rebals) else end
        if d0 not in ds_by.index or d1 not in ds_by.index:
            continue
        cutoff, (reg, cls) = pick_models(cutoff_models, d0)
        idx = ds[ds['date'] == d0].index[0]
        X_row = np.nan_to_num(
            ds.iloc[idx][FEATURES_37].values.astype(float).reshape(1, -1), nan=0.0
        )
        pred = float(np.mean([m.predict(X_row)[0] for m in reg]))
        p_up = float(np.mean([m.predict_proba(X_row)[0, 1] for m in cls]))
        conf = float(1 / (1 + np.exp(-abs(p_up - 0.5) * SIGMOID_SCALE)))
        prices = ds['price_usd'].values[:idx + 1]
        s50 = pd.Series(prices).rolling(50).mean().iloc[-1]
        s200 = pd.Series(prices).rolling(200).mean().iloc[-1]
        regime = get_regime(prices[-1], s50, s200)
        alloc = float(np.clip(pred * K_map[regime] * conf, ALLOC_MIN, ALLOC_MAX))
        p0 = float(ds_by.loc[d0, 'price_usd'])
        p1 = float(ds_by.loc[d1, 'price_usd'])
        btc_ret = p1 / p0 - 1
        mask = (rf_series.index > d0) & (rf_series.index <= d1)
        cdi_ret = float((1 + rf_series[mask]).prod() - 1)
        strat_ret = alloc * btc_ret + (1 - alloc) * cdi_ret
        cum *= (1 + strat_ret)
        cum_btc *= (1 + btc_ret)
        rows.append({
            'date': d0, 'pred': pred, 'p_up': p_up, 'alloc': alloc,
            'regime': regime, 'btc_fwd': btc_ret, 'strat': strat_ret,
            'dir_match': (pred > 0) == (btc_ret > 0),
        })
    return pd.DataFrame(rows), cum - 1, cum_btc - 1


def sortino(returns, ppy=52):
    r = np.array(returns)
    down = r[r < 0]
    if len(down) == 0:
        return float('inf')
    return r.mean() * ppy / (np.sqrt((down ** 2).mean()) * np.sqrt(ppy))


def max_drawdown(returns):
    cum = (1 + np.array(returns)).cumprod()
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    return dd.min()


def report(df, cum, cum_btc, label):
    acc = df['dir_match'].sum()
    ann = (1 + cum) ** (1 / 4.3) - 1
    print(f'\n{"=" * 70}')
    print(f'  {label}')
    print(f'{"=" * 70}')
    print(f'Period:           {df["date"].iloc[0].date()} -> {df["date"].iloc[-1].date()}')
    print(f'N rebals:         {len(df)}')
    print(f'Return total:     {cum*100:+.2f}%')
    print(f'CAGR:             {ann*100:.1f}%/yr')
    print(f'BTC total:        {cum_btc*100:+.2f}%')
    print(f'Alpha:            {(cum-cum_btc)*100:+.2f}pp')
    print(f'Sortino:          {sortino(df["strat"]):.2f}')
    print(f'Max DD:           {max_drawdown(df["strat"])*100:.2f}%')
    print(f'Dir 7d accuracy:  {acc}/{len(df)} = {acc/len(df)*100:.1f}%')
    print(f'corr(pred, fwd):  {df["pred"].corr(df["btc_fwd"]):+.3f}')
    print(f'Mean alloc:       {df["alloc"].mean()*100:.1f}%')


def main():
    ds = pd.read_csv(
        ROOT / 'scripts/production/data/dataset_production.csv',
        parse_dates=['date']
    ).sort_values('date').reset_index(drop=True)

    start = pd.Timestamp('2022-01-01')
    end = ds['date'].iloc[-1]

    cutoffs = retrain_cutoffs(start, end)
    print(f'Cutoffs: {[c.date() for c in cutoffs]}', flush=True)
    print(f'Range: {start.date()} -> {end.date()}', flush=True)

    rf = pd.Series(
        build_rf_daily(pd.date_range(start, end, freq='D')),
        index=pd.date_range(start, end, freq='D'),
    )

    print('\nTraining H=3 models...', flush=True)
    t0 = time.time()
    m3 = {}
    for c in cutoffs:
        t1 = time.time()
        m3[c] = train_at_cutoff(ds, c, horizon=3)
        print(f'  {c.date()} trained in {time.time()-t1:.0f}s', flush=True)
    print(f'H=3 total: {time.time()-t0:.0f}s')

    print('\nTraining H=7 models...', flush=True)
    t0 = time.time()
    m7 = {}
    for c in cutoffs:
        t1 = time.time()
        m7[c] = train_at_cutoff(ds, c, horizon=7)
        print(f'  {c.date()} trained in {time.time()-t1:.0f}s', flush=True)
    print(f'H=7 total: {time.time()-t0:.0f}s')

    print('\nRunning walk-forwards...', flush=True)
    df3, cum3, cum_btc = run(ds, start, end, m3, K_REGIME, rf)
    df7, cum7, _ = run(ds, start, end, m7, K_REGIME, rf)
    K_scaled = {k: v * 3/7 for k, v in K_REGIME.items()}
    df7s, cum7s, _ = run(ds, start, end, m7, K_scaled, rf)

    report(df3, cum3, cum_btc, 'H=3 (PRODUCTION) | K=100/50/20')
    report(df7, cum7, cum_btc, 'H=7 same K=100/50/20 (saturates)')
    report(df7s, cum7s, cum_btc, 'H=7 scaled K=43/21/9 (magnitude-neutral)')

    print(f'\n{"=" * 70}')
    print('SIDE-BY-SIDE')
    print('=' * 70)
    print(f'{"Metric":<20} {"H=3":>12} {"H=7 sameK":>14} {"H=7 scaledK":>14}')
    print(f'{"Return":<20} {cum3*100:+11.2f}% {cum7*100:+13.2f}% {cum7s*100:+13.2f}%')
    print(f'{"Sortino":<20} {sortino(df3["strat"]):>12.2f} {sortino(df7["strat"]):>14.2f} {sortino(df7s["strat"]):>14.2f}')
    print(f'{"Max DD":<20} {max_drawdown(df3["strat"])*100:>11.2f}% {max_drawdown(df7["strat"])*100:>13.2f}% {max_drawdown(df7s["strat"])*100:>13.2f}%')
    print(f'{"Dir accuracy":<20} {df3["dir_match"].mean()*100:>11.1f}% {df7["dir_match"].mean()*100:>13.1f}% {df7s["dir_match"].mean()*100:>13.1f}%')
    print(f'{"corr(pred,fwd)":<20} {df3["pred"].corr(df3["btc_fwd"]):>12.3f} {df7["pred"].corr(df7["btc_fwd"]):>14.3f} {df7s["pred"].corr(df7s["btc_fwd"]):>14.3f}')

    out = ROOT / 'outputs/results/horizon_ablation_4y.csv'
    combined = pd.concat([
        df3.assign(variant='H=3'),
        df7.assign(variant='H=7_sameK'),
        df7s.assign(variant='H=7_scaledK'),
    ], ignore_index=True)
    combined.to_csv(out, index=False)
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
