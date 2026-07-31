"""
TESTE 2: FROZEN TRAIN — Modelo treinado UMA VEZ em 2019-2021, sem retrain.

Objetivo: O modelo de producao retreina semi-anualmente (9 retrains em 4.3y).
Isso significa que cada modelo nunca esta muito longe do presente. Se o edge
vem de FEATURES GENUINAMENTE PREDITIVAS, um modelo frozen em 2022-01-01
deveria ainda ter alpha (talvez menor). Se o edge vem do retrain capturar
drift recente, frozen vai collapsar.

Teste honesto: Mesmos features, mesmo K, mesmo sigmoid — so congelado.

Output: outputs/results/overfit_tests/test2_frozen_train.csv
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
    HORIZON, REBAL_DOW, EMERGENCY_THRESHOLD,
)
from generate_signal import train_regression_ensemble, train_classifier_ensemble, get_regime
from src.features.macro.cdi_rates import build_rf_daily

OUT = ROOT / 'outputs' / 'results' / 'overfit_tests' / 'test2_frozen_train.csv'
OUT.parent.mkdir(parents=True, exist_ok=True)


def train_at_date(ds, freeze_date, horizon=HORIZON):
    """Train models using ONLY data strictly before freeze_date."""
    mask = ds['date'] < freeze_date
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
    print(f'  Training on {len(train_idx)} samples, date range: {sub.date.iloc[60].date()} to {sub.date.iloc[train_end].date()}')
    t0 = time.time()
    reg = train_regression_ensemble(X[train_idx], treg[train_idx])
    cls = train_classifier_ensemble(X[train_idx], tcls[train_idx])
    print(f'  Trained in {time.time()-t0:.0f}s')
    return reg, cls


def predict(reg, cls, x):
    pred = float(np.mean([m.predict(x)[0] for m in reg]))
    p_up = float(np.mean([m.predict_proba(x)[0, 1] for m in cls]))
    conf = float(1 / (1 + np.exp(-abs(p_up - 0.5) * SIGMOID_SCALE)))
    return pred, p_up, conf


def rebalance_dates(ds, start, end):
    sub = ds[(ds['date'] >= start) & (ds['date'] <= end)].copy()
    sub['daily_ret'] = sub['price_usd'].pct_change()
    sub['dow'] = sub['date'].dt.dayofweek
    fri = set(sub.loc[sub['dow'].isin(REBAL_DOW), 'date'])
    emg = set(sub.loc[sub['daily_ret'].abs() > EMERGENCY_THRESHOLD, 'date'])
    return sorted(fri | emg)


def run_frozen_backtest(ds, reg, cls, start, end, K_map, floor=ALLOC_MIN, ceil=ALLOC_MAX):
    rebals = rebalance_dates(ds, start, end)
    print(f'  Running {len(rebals)} rebals {start.date()} to {end.date()}')
    ds_idx = ds.set_index('date')
    rf = pd.Series(build_rf_daily(pd.date_range(start, end, freq='D')),
                   index=pd.date_range(start, end, freq='D'))
    rows = []
    for i, d0 in enumerate(rebals):
        d1 = rebals[i+1] if i+1 < len(rebals) else end
        if d0 not in ds_idx.index or d1 not in ds_idx.index:
            continue
        idx = ds[ds['date'] == d0].index[0]
        x = np.nan_to_num(ds.iloc[idx][FEATURES_37].values.astype(float).reshape(1, -1), nan=0.0)
        pred, p_up, conf = predict(reg, cls, x)
        prices = ds['price_usd'].values[:idx+1]
        s50 = pd.Series(prices).rolling(50).mean().iloc[-1]
        s200 = pd.Series(prices).rolling(200).mean().iloc[-1]
        regime = get_regime(prices[-1], s50, s200)
        alloc = float(np.clip(pred * K_map[regime] * conf, floor, ceil))
        p0 = float(ds_idx.loc[d0, 'price_usd'])
        p1 = float(ds_idx.loc[d1, 'price_usd'])
        btc_ret = p1/p0 - 1
        mask = (rf.index > d0) & (rf.index <= d1)
        cdi_ret = float((1 + rf[mask]).prod() - 1)
        strat = alloc * btc_ret + (1 - alloc) * cdi_ret
        rows.append({
            'date': d0, 'regime': regime, 'pred': pred, 'p_up': p_up, 'conf': conf,
            'alloc': alloc, 'btc_fwd': btc_ret, 'cdi_ret': cdi_ret, 'strat': strat
        })
    return pd.DataFrame(rows)


def metrics(arr, ppy=52):
    arr = np.array(arr)
    cum = (1 + arr).prod() - 1
    ann = arr.mean() * ppy
    vol = arr.std() * np.sqrt(ppy)
    down = arr[arr < 0]
    sortino = (arr.mean() / np.sqrt((down**2).mean())) * np.sqrt(ppy) if len(down) > 0 else float('inf')
    sharpe = ((arr.mean() - 0.0021) / arr.std()) * np.sqrt(ppy) if arr.std() > 0 else 0
    cum_series = (1+arr).cumprod()
    dd = ((cum_series - np.maximum.accumulate(cum_series)) / np.maximum.accumulate(cum_series)).min()
    return {
        'cum_%': round(cum*100, 1), 'ann_%': round(ann*100, 1),
        'sortino': round(sortino, 2), 'sharpe': round(sharpe, 2),
        'max_dd_%': round(dd*100, 2)
    }


def main():
    DATA = ROOT / 'scripts' / 'production' / 'data' / 'dataset_production.csv'
    ds = pd.read_csv(DATA, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    print(f'Dataset: {len(ds)} rows, {ds.date.min().date()} to {ds.date.max().date()}')
    print()

    test_end = pd.Timestamp('2026-04-17')

    # Test A: Frozen train @ 2022-01-01 (2019-2021 only, 3 years training)
    # Test B: Frozen train @ 2023-01-01 (2019-2022, 4 years training)
    # Test C: Frozen train @ 2024-01-01 (2019-2023, 5 years training)

    freeze_dates = [
        ('frozen_2022', pd.Timestamp('2022-01-01')),
        ('frozen_2023', pd.Timestamp('2023-01-01')),
        ('frozen_2024', pd.Timestamp('2024-01-01')),
    ]

    all_results = []
    for label, freeze_date in freeze_dates:
        print(f'\n=== {label} (train end: {freeze_date.date()}) ===')
        reg, cls = train_at_date(ds, freeze_date)

        # Run from freeze date to end
        df_strat = run_frozen_backtest(ds, reg, cls, freeze_date, test_end, K_REGIME)
        print(f'  {len(df_strat)} rebals')

        # Compute across multiple K configs
        for k_label, K_map in [
            ('H2 K=100/50/20', {'BULL': 100, 'MILD': 50, 'BEAR': 20}),
            ('H1 K=60/30/15',  {'BULL': 60,  'MILD': 30, 'BEAR': 15}),
            ('K=40/20/10',     {'BULL': 40,  'MILD': 20, 'BEAR': 10}),
            ('K=20/10/5',      {'BULL': 20,  'MILD': 10, 'BEAR': 5}),
        ]:
            df_recompute = df_strat.copy()
            df_recompute['alloc_new'] = df_recompute.apply(
                lambda r: float(np.clip(r['pred'] * K_map[r['regime']] * r['conf'], 0, 1)),
                axis=1
            )
            df_recompute['strat_new'] = df_recompute['alloc_new']*df_recompute['btc_fwd'] + (1-df_recompute['alloc_new'])*df_recompute['cdi_ret']
            m = metrics(df_recompute['strat_new'].values)
            m['config'] = k_label
            m['freeze'] = label
            m['train_years'] = (freeze_date - pd.Timestamp('2019-01-01')).days / 365.25
            m['n_rebals'] = len(df_recompute)
            m['avg_alloc_%'] = round(df_recompute['alloc_new'].mean()*100, 1)
            all_results.append(m)
            print(f'    {k_label}: Sortino {m["sortino"]}, Sharpe {m["sharpe"]}, Ret {m["cum_%"]}%, DD {m["max_dd_%"]}%')

    res = pd.DataFrame(all_results)
    cols = ['freeze', 'config', 'train_years', 'n_rebals', 'cum_%', 'ann_%',
            'sortino', 'sharpe', 'max_dd_%', 'avg_alloc_%']
    res = res[cols]
    res.to_csv(OUT, index=False)
    print()
    print('=' * 80)
    print('TESTE 2: FROZEN TRAIN — resumo')
    print('=' * 80)
    print(res.to_string(index=False))
    print()
    print(f'Saved: {OUT}')


if __name__ == '__main__':
    main()
