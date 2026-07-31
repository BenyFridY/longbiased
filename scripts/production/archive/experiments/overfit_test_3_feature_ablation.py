"""
TESTE 3: Feature Ablation — remover V36 on-chain features (reserveRisk, puellMultiple, funding_rate_ma7)

Objetivo: V36 features foram adicionadas em 2026-04-19, DEPOIS de centenas de
experimentos. Por isso, seu "+0.30 Sortino" pode ser overfit a 2026 recent.
Este teste remove essas 3 features e re-treina walk-forward.

Tambem testa:
  - Remover top-5 features (crash test — edge sobrevive sem features principais?)
  - Modelo V29 original (29 features, pre-V36)
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

OUT = ROOT / 'outputs' / 'results' / 'overfit_tests' / 'test3_feature_ablation.csv'
OUT.parent.mkdir(parents=True, exist_ok=True)

RETRAIN_MONTHS = [1, 7]
V36_FEATURES = ['reserveRisk', 'puellMultiple', 'funding_rate_ma7']
TOP5 = ['cusum_pos', 'nupl_ma30', 'bb_position', 'eth_pctchg_30d', 'm2_yoy_growth']


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


def train_at_cutoff(ds, cutoff, features):
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
    reg = train_regression_ensemble(X[train_idx], treg[train_idx])
    cls = train_classifier_ensemble(X[train_idx], tcls[train_idx])
    return reg, cls


def rebalance_dates(ds, start, end):
    sub = ds[(ds['date'] >= start) & (ds['date'] <= end)].copy()
    sub['daily_ret'] = sub['price_usd'].pct_change()
    sub['dow'] = sub['date'].dt.dayofweek
    fri = set(sub.loc[sub['dow'].isin(REBAL_DOW), 'date'])
    emg = set(sub.loc[sub['daily_ret'].abs() > EMERGENCY_THRESHOLD, 'date'])
    return sorted(fri | emg)


def run_walkforward(ds, features, start, end, K_map):
    cutoffs = retrain_cutoffs(start, end)
    print(f'  Retrain cutoffs: {[c.date() for c in cutoffs]}')
    cutoff_models = {}
    for c in cutoffs:
        t0 = time.time()
        cutoff_models[c] = train_at_cutoff(ds, c, features)
        print(f'  Cutoff {c.date()}: trained in {time.time()-t0:.0f}s')

    rebals = rebalance_dates(ds, start, end)
    rf = pd.Series(build_rf_daily(pd.date_range(start, end, freq='D')),
                   index=pd.date_range(start, end, freq='D'))
    ds_idx = ds.set_index('date')
    rows = []
    for i, d0 in enumerate(rebals):
        d1 = rebals[i+1] if i+1 < len(rebals) else end
        if d0 not in ds_idx.index or d1 not in ds_idx.index:
            continue
        applicable = [c for c in cutoff_models if c <= d0]
        c = max(applicable)
        reg, cls = cutoff_models[c]
        idx = ds[ds['date'] == d0].index[0]
        x = np.nan_to_num(ds.iloc[idx][features].values.astype(float).reshape(1, -1), nan=0.0)
        pred = float(np.mean([m.predict(x)[0] for m in reg]))
        p_up = float(np.mean([m.predict_proba(x)[0, 1] for m in cls]))
        conf = float(1 / (1 + np.exp(-abs(p_up - 0.5) * SIGMOID_SCALE)))
        prices = ds['price_usd'].values[:idx+1]
        s50 = pd.Series(prices).rolling(50).mean().iloc[-1]
        s200 = pd.Series(prices).rolling(200).mean().iloc[-1]
        regime = get_regime(prices[-1], s50, s200)
        alloc = float(np.clip(pred * K_map[regime] * conf, 0, 1))
        p0 = float(ds_idx.loc[d0, 'price_usd']); p1 = float(ds_idx.loc[d1, 'price_usd'])
        btc_ret = p1/p0 - 1
        mask = (rf.index > d0) & (rf.index <= d1)
        cdi_ret = float((1 + rf[mask]).prod() - 1)
        strat = alloc * btc_ret + (1 - alloc) * cdi_ret
        rows.append({'date': d0, 'regime': regime, 'pred': pred, 'p_up': p_up, 'conf': conf,
                     'alloc': alloc, 'btc_fwd': btc_ret, 'cdi_ret': cdi_ret, 'strat': strat})
    return pd.DataFrame(rows)


def metrics(arr, ppy=52):
    arr = np.array(arr)
    cum = (1 + arr).prod() - 1
    ann = arr.mean() * ppy
    down = arr[arr < 0]
    sortino = (arr.mean() / np.sqrt((down**2).mean())) * np.sqrt(ppy) if len(down)>0 and (down**2).mean()>0 else float('inf')
    sharpe = ((arr.mean() - 0.0021) / arr.std()) * np.sqrt(ppy) if arr.std()>0 else 0
    c = (1+arr).cumprod()
    dd = ((c - np.maximum.accumulate(c)) / np.maximum.accumulate(c)).min()
    return dict(cum_=round(cum*100,1), ann_=round(ann*100,1), sortino=round(sortino,2), sharpe=round(sharpe,2), dd_=round(dd*100,2))


def main():
    DATA = ROOT / 'scripts' / 'production' / 'data' / 'dataset_production.csv'
    ds = pd.read_csv(DATA, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    start = pd.Timestamp('2022-01-01')
    end = pd.Timestamp('2026-04-17')

    results = []

    # Baseline: full 32 features (current config)
    print('\n=== BASELINE: 32 features (current production) ===')
    df_base = run_walkforward(ds, FEATURES_37, start, end, K_REGIME)
    m_base = metrics(df_base['strat'].values)
    print(f'  H2: Sortino {m_base["sortino"]}, Ret {m_base["cum_"]}%, DD {m_base["dd_"]}%')
    results.append({'test': 'baseline_32feat', 'n_feat': len(FEATURES_37), 'config': 'H2', **m_base})

    # Also H1 on same preds
    df_base['alloc_H1'] = df_base.apply(lambda r: float(np.clip(r['pred'] * {'BULL':60,'MILD':30,'BEAR':15}[r['regime']] * r['conf'], 0, 1)), axis=1)
    df_base['strat_H1'] = df_base['alloc_H1']*df_base['btc_fwd'] + (1-df_base['alloc_H1'])*df_base['cdi_ret']
    m_h1 = metrics(df_base['strat_H1'].values)
    results.append({'test': 'baseline_32feat', 'n_feat': len(FEATURES_37), 'config': 'H1', **m_h1})
    print(f'  H1: Sortino {m_h1["sortino"]}, Ret {m_h1["cum_"]}%, DD {m_h1["dd_"]}%')

    # TEST A: Remove V36 features (3 features added late in April 2026)
    feat_no_v36 = [f for f in FEATURES_37 if f not in V36_FEATURES]
    print(f'\n=== TEST A: 29 features (remove V36 on-chain: {V36_FEATURES}) ===')
    df_a = run_walkforward(ds, feat_no_v36, start, end, K_REGIME)
    m_a = metrics(df_a['strat'].values)
    print(f'  H2: Sortino {m_a["sortino"]}, Ret {m_a["cum_"]}%, DD {m_a["dd_"]}%')
    results.append({'test': 'no_v36', 'n_feat': len(feat_no_v36), 'config': 'H2', **m_a})
    df_a['alloc_H1'] = df_a.apply(lambda r: float(np.clip(r['pred'] * {'BULL':60,'MILD':30,'BEAR':15}[r['regime']] * r['conf'], 0, 1)), axis=1)
    df_a['strat_H1'] = df_a['alloc_H1']*df_a['btc_fwd'] + (1-df_a['alloc_H1'])*df_a['cdi_ret']
    m_a_h1 = metrics(df_a['strat_H1'].values)
    results.append({'test': 'no_v36', 'n_feat': len(feat_no_v36), 'config': 'H1', **m_a_h1})
    print(f'  H1: Sortino {m_a_h1["sortino"]}, Ret {m_a_h1["cum_"]}%, DD {m_a_h1["dd_"]}%')

    # TEST B: Remove top-5 "most important" features
    feat_no_top5 = [f for f in FEATURES_37 if f not in TOP5]
    print(f'\n=== TEST B: {len(feat_no_top5)} features (remove top-5: {TOP5}) ===')
    df_b = run_walkforward(ds, feat_no_top5, start, end, K_REGIME)
    m_b = metrics(df_b['strat'].values)
    print(f'  H2: Sortino {m_b["sortino"]}, Ret {m_b["cum_"]}%, DD {m_b["dd_"]}%')
    results.append({'test': 'no_top5', 'n_feat': len(feat_no_top5), 'config': 'H2', **m_b})

    # TEST C: Only top-5 features
    print(f'\n=== TEST C: top-5 features only ===')
    df_c = run_walkforward(ds, TOP5, start, end, K_REGIME)
    m_c = metrics(df_c['strat'].values)
    print(f'  H2: Sortino {m_c["sortino"]}, Ret {m_c["cum_"]}%, DD {m_c["dd_"]}%')
    results.append({'test': 'only_top5', 'n_feat': 5, 'config': 'H2', **m_c})

    # TEST D: No macro features (remove m2, fed, velocity, copper, gold_corr)
    macro = ['m2_yoy_growth', 'fed_balance_sheet', 'velocity', 'copper_return_30d', 'btc_gold_corr_30d', 'fed_fracdiff_05']
    feat_no_macro = [f for f in FEATURES_37 if f not in macro]
    print(f'\n=== TEST D: {len(feat_no_macro)} features (no macro) ===')
    df_d = run_walkforward(ds, feat_no_macro, start, end, K_REGIME)
    m_d = metrics(df_d['strat'].values)
    print(f'  H2: Sortino {m_d["sortino"]}, Ret {m_d["cum_"]}%, DD {m_d["dd_"]}%')
    results.append({'test': 'no_macro', 'n_feat': len(feat_no_macro), 'config': 'H2', **m_d})

    res = pd.DataFrame(results)
    res.to_csv(OUT, index=False)
    print()
    print('=' * 90)
    print('TESTE 3: FEATURE ABLATION — resumo')
    print('=' * 90)
    print(res.to_string(index=False))
    print(f'\nSaved: {OUT}')


if __name__ == '__main__':
    main()
