"""
Ablation: HORIZON=3 vs HORIZON=7 on 2026 YTD.

Train at 2026-01-01 cutoff, same features, same K, same sigmoid.
Walk through 2026 YTD rebals (Friday + emergency).
Report Return / Sortino / direction accuracy / alloc stats.

Run:
    python scripts/production/archive/experiments/horizon_ablation.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts' / 'production'))

from config import (
    FEATURES_37, K_REGIME, ALLOC_MIN, ALLOC_MAX, SIGMOID_SCALE,
    BAGS, REBAL_DOW, EMERGENCY_THRESHOLD,
)
from generate_signal import train_regression_ensemble, train_classifier_ensemble, get_regime
from src.features.macro.cdi_rates import build_rf_daily


ds = pd.read_csv(ROOT / 'scripts/production/data/dataset_production.csv',
                 parse_dates=['date']).sort_values('date').reset_index(drop=True)
CUTOFF = pd.Timestamp('2026-01-01')
END = ds['date'].iloc[-1]


def train_h(horizon):
    mask = ds['date'] < CUTOFF
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
    print(f'  H={horizon}: n={n} train_samples={len(train_idx)}', flush=True)
    reg = train_regression_ensemble(X[train_idx], treg[train_idx])
    cls = train_classifier_ensemble(X[train_idx], tcls[train_idx])
    return reg, cls


def rebal_dates():
    sub = ds[(ds['date'] >= CUTOFF) & (ds['date'] <= END)].copy()
    sub['daily_ret'] = sub['price_usd'].pct_change()
    sub['dow'] = sub['date'].dt.dayofweek
    fri = set(sub.loc[sub['dow'].isin(REBAL_DOW), 'date'])
    emg = set(sub.loc[sub['daily_ret'].abs() > EMERGENCY_THRESHOLD, 'date'])
    return sorted(fri | emg)


def backtest(reg, cls, K_scale=1.0, label=''):
    K = {k: v * K_scale for k, v in K_REGIME.items()}
    rebals = rebal_dates()
    ds_by = ds.set_index('date')
    rf = pd.Series(
        build_rf_daily(pd.date_range(CUTOFF, END, freq='D')),
        index=pd.date_range(CUTOFF, END, freq='D'),
    )
    cum = 1.0; cum_btc = 1.0
    rows = []
    for i, d0 in enumerate(rebals):
        d1 = rebals[i + 1] if i + 1 < len(rebals) else END
        if d0 not in ds_by.index or d1 not in ds_by.index:
            continue
        idx = ds[ds['date'] == d0].index[0]
        X_row = np.nan_to_num(
            ds.iloc[idx][FEATURES_37].values.astype(float).reshape(1, -1), nan=0.0
        )
        pred = float(np.mean([m.predict(X_row)[0] for m in reg]))
        p_up = float(np.mean([m.predict_proba(X_row)[0, 1] for m in cls]))
        conf = float(1 / (1 + np.exp(-abs(p_up - 0.5) * SIGMOID_SCALE)))
        prices_upto = ds['price_usd'].values[:idx + 1]
        s50 = pd.Series(prices_upto).rolling(50).mean().iloc[-1]
        s200 = pd.Series(prices_upto).rolling(200).mean().iloc[-1]
        regime = get_regime(prices_upto[-1], s50, s200)
        alloc = float(np.clip(pred * K[regime] * conf, ALLOC_MIN, ALLOC_MAX))
        p0 = float(ds_by.loc[d0, 'price_usd'])
        p1 = float(ds_by.loc[d1, 'price_usd'])
        btc_ret = p1 / p0 - 1
        mask = (rf.index > d0) & (rf.index <= d1)
        cdi_ret = float((1 + rf[mask]).prod() - 1)
        strat_ret = alloc * btc_ret + (1 - alloc) * cdi_ret
        cum *= (1 + strat_ret)
        cum_btc *= (1 + btc_ret)
        rows.append({
            'date': d0, 'pred': pred, 'p_up': p_up, 'alloc': alloc,
            'btc_fwd': btc_ret, 'strat': strat_ret,
            'dir_match': (pred > 0) == (btc_ret > 0),
        })
    return pd.DataFrame(rows), cum - 1, cum_btc - 1


def sortino(returns, ppy=52):
    r = np.array(returns)
    down = r[r < 0]
    if len(down) == 0:
        return float('inf')
    ann_ret = r.mean() * ppy
    ann_down = np.sqrt((down ** 2).mean()) * np.sqrt(ppy)
    return ann_ret / ann_down if ann_down > 0 else float('inf')


def report(df, cum, cum_btc, label):
    acc = df['dir_match'].sum()
    corr_pred_fwd = df['pred'].corr(df['btc_fwd'])
    corr_alloc_fwd = df['alloc'].corr(df['btc_fwd'])
    print(f'\n{"="*60}')
    print(f'  {label}')
    print(f'{"="*60}')
    print(f'Return YTD:       {cum*100:+.2f}%')
    print(f'BTC YTD:          {cum_btc*100:+.2f}%')
    print(f'Alpha:            {(cum-cum_btc)*100:+.2f}pp')
    print(f'Sortino:          {sortino(df["strat"]):.2f}')
    print(f'Dir 7d accuracy:  {acc}/{len(df)} = {acc/len(df)*100:.1f}%')
    print(f'corr(pred, 7dfw): {corr_pred_fwd:+.3f}')
    print(f'corr(alloc, 7dfw):{corr_alloc_fwd:+.3f}')
    print(f'Mean alloc:       {df["alloc"].mean()*100:.1f}%')
    print(f'Max alloc:        {df["alloc"].max()*100:.1f}%')
    print(f'% rebals alloc=0: {(df["alloc"]==0).sum()}/{len(df)}')
    print(f'% rebals alloc>=70%: {(df["alloc"]>=0.7).sum()}/{len(df)}')


def main():
    print('Training both models at cutoff 2026-01-01...', flush=True)
    reg3, cls3 = train_h(3)
    reg7, cls7 = train_h(7)

    df3, cum3, cum_btc = backtest(reg3, cls3, K_scale=1.0)
    df7, cum7, _       = backtest(reg7, cls7, K_scale=1.0)
    df7s, cum7s, _     = backtest(reg7, cls7, K_scale=3/7)

    report(df3, cum3, cum_btc, 'H=3 (PRODUCTION) — K=100/50/20')
    report(df7, cum7, cum_btc, 'H=7 (same K=100/50/20) — unfair, magnitude saturates')
    report(df7s, cum7s, cum_btc, 'H=7 (K scaled 3/7 = 43/21/9) — magnitude-neutral')

    out = ROOT / 'outputs/results/horizon_ablation.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([
        df3.assign(variant='H=3'),
        df7.assign(variant='H=7_sameK'),
        df7s.assign(variant='H=7_scaledK'),
    ], ignore_index=True)
    combined.to_csv(out, index=False)
    print(f'\nSaved detailed rows: {out}')


if __name__ == '__main__':
    main()
