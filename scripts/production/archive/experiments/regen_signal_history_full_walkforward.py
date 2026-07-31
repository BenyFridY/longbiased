"""Full walkforward signal_history 2022-2026 with H1 + 160 bags, semi-annual retrain.
Each rebal uses model trained at most recent cutoff <= rebal_date (no leak)."""
import sys, time
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/production'))

from config import (FEATURES_37, XGB_PARAMS, K_REGIME, ALLOC_MIN, ALLOC_MAX,
                    SIGMOID_SCALE, REBAL_DOW, EMERGENCY_THRESHOLD, BAGS, HORIZON, WORKERS)
from src.features.macro.cdi_rates import build_rf_daily

DATASET = ROOT / 'scripts/production/data/dataset_production.csv'
SIG_PATH = ROOT / 'scripts/production/data/signal_history.csv'

ds = pd.read_csv(DATASET, parse_dates=['date']).sort_values('date').reset_index(drop=True)

RETRAIN_MONTHS = [1, 7]
end_data = ds['date'].iloc[-1]
cuts = []
for y in range(2021, end_data.year + 2):
    for m in RETRAIN_MONTHS:
        d = pd.Timestamp(year=y, month=m, day=1)
        if d <= end_data:
            cuts.append(d)
cuts = sorted(c for c in cuts if c >= pd.Timestamp('2022-01-01'))
print(f"Cutoffs: {[c.date() for c in cuts]}")
print(f"H1 K={K_REGIME}, BAGS={BAGS}")
print()


def _train_reg(args):
    s, X, y = args
    m = xgb.XGBRegressor(**XGB_PARAMS, random_state=s)
    rng = np.random.RandomState(s)
    idx = rng.choice(len(X), size=len(X), replace=True)
    m.fit(X[idx], y[idx])
    return m


def _train_cls(args):
    s, X, y = args
    cls_params = {**{k: v for k, v in XGB_PARAMS.items() if k != 'objective'},
                  'objective': 'binary:logistic', 'eval_metric': 'logloss'}
    m = xgb.XGBClassifier(**cls_params, random_state=s)
    rng = np.random.RandomState(s)
    idx = rng.choice(len(X), size=len(X), replace=True)
    m.fit(X[idx], y[idx])
    return m


cutoff_models = {}
t_start = time.time()
for c in cuts:
    t0 = time.time()
    mask = ds['date'] < c
    n = int(mask.sum())
    sub = ds.iloc[:n].reset_index(drop=True)
    X = np.nan_to_num(sub[FEATURES_37].values.astype(float), nan=0.0)
    prices_train = sub['price_usd'].values
    treg = np.zeros(n); tcls = np.zeros(n)
    for i in range(n - HORIZON):
        treg[i] = (prices_train[i + HORIZON] - prices_train[i]) / prices_train[i]
        tcls[i] = 1.0 if prices_train[i + HORIZON] > prices_train[i] else 0.0
    train_idx = np.arange(60, n - max(HORIZON, 5) + 1)
    train_idx = train_idx[~np.any(np.isnan(X[train_idx]), axis=1)]
    seeds = [242 + i * 7 for i in range(BAGS)]
    Xtr = X[train_idx]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        regs = list(ex.map(_train_reg, [(s, Xtr, treg[train_idx]) for s in seeds]))
        clss = list(ex.map(_train_cls, [(s, Xtr, tcls[train_idx]) for s in seeds]))
    cutoff_models[c] = (regs, clss)
    print(f"  Cutoff {c.date()}: {time.time()-t0:.0f}s ({len(train_idx)} train)", flush=True)
print(f"Total training: {time.time()-t_start:.0f}s\n")

start = pd.Timestamp('2022-01-07')
sub2 = ds[(ds['date'] >= start) & (ds['date'] <= end_data)].copy()
sub2['daily_ret'] = sub2['price_usd'].pct_change()
sub2['dow'] = sub2['date'].dt.dayofweek
fridays = set(sub2.loc[sub2['dow'].isin(REBAL_DOW), 'date'])
emerg = set(sub2.loc[sub2['daily_ret'].abs() > EMERGENCY_THRESHOLD, 'date'])
rebals = sorted(fridays | emerg)
print(f"Total rebals: {len(rebals)}")


def get_regime(price, sma50, sma200):
    if np.isnan(sma50) or np.isnan(sma200):
        return 'MILD'
    if price > sma50 and sma50 > sma200:
        return 'BULL'
    elif price > sma200:
        return 'MILD'
    return 'BEAR'


rf = pd.Series(
    build_rf_daily(pd.date_range(start, end_data + pd.Timedelta(days=10), freq='D')),
    index=pd.date_range(start, end_data + pd.Timedelta(days=10), freq='D'),
)
ds_by_date = ds.set_index('date')

DEC = 4
rows = []
for d0 in rebals:
    if d0 not in ds_by_date.index:
        continue
    applicable = [c for c in cutoff_models if c <= d0]
    if not applicable:
        continue
    c = max(applicable)
    reg_models, cls_models = cutoff_models[c]
    idx = ds[ds['date'] == d0].index[0]
    X_row = np.nan_to_num(
        ds.iloc[idx][FEATURES_37].values.astype(float).reshape(1, -1), nan=0.0
    )
    pred = float(np.mean([m.predict(X_row)[0] for m in reg_models]))
    p_up = float(np.mean([m.predict_proba(X_row)[0, 1] for m in cls_models]))
    conf = float(1 / (1 + np.exp(-abs(p_up - 0.5) * SIGMOID_SCALE)))
    prices = ds['price_usd'].values[:idx + 1]
    s50 = pd.Series(prices).rolling(50).mean().iloc[-1]
    s200 = pd.Series(prices).rolling(200).mean().iloc[-1]
    regime = get_regime(prices[-1], s50, s200)
    K = K_REGIME[regime]
    alloc = float(np.clip(pred * K * conf, ALLOC_MIN, ALLOC_MAX))
    daily_ret = (prices[-1] - prices[-2]) / prices[-2] if idx > 0 else 0.0
    is_em = d0 in emerg and d0 not in fridays
    day_name = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][d0.dayofweek]
    if is_em:
        action = f'EMERGENCY (ret {daily_ret*100:+.1f}%) [H1 BAGS=160 cut {c.date()}]'
    else:
        action = f'REBALANCE (Fri) [H1 BAGS=160 cut {c.date()}]'
    rows.append({
        'date': d0.strftime('%Y-%m-%d'), 'day': day_name,
        'price_usd': round(float(prices[-1]), 2), 'regime': regime,
        'previsao': round(pred, DEC), 'p_up': round(p_up, DEC),
        'confidence_factor': round(conf, DEC), 'allocation': round(alloc, DEC),
        'K_base': K, 'K_effective': round(K * conf, 2),
        'is_emergency': is_em, 'retorno_btc': None, 'retorno_strat': None,
        'action': action,
    })

df = pd.DataFrame(rows)
for i in range(len(df) - 1):
    d0 = pd.Timestamp(df['date'].iloc[i])
    d1 = pd.Timestamp(df['date'].iloc[i + 1])
    p0 = float(df['price_usd'].iloc[i])
    p1 = float(df['price_usd'].iloc[i + 1])
    alloc = float(df['allocation'].iloc[i])
    btc_ret = p1 / p0 - 1
    mask = (rf.index > d0) & (rf.index <= d1)
    cdi_ret = float((1 + rf[mask]).prod() - 1)
    strat_ret = alloc * btc_ret + (1 - alloc) * cdi_ret
    df.at[i, 'retorno_btc'] = round(btc_ret, DEC)
    df.at[i, 'retorno_strat'] = round(strat_ret, DEC)

df.to_csv(SIG_PATH, index=False)
print(f"\nSaved {len(df)} rebals from {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
print(f"  K_base: {sorted(df['K_base'].unique())}")
print(f"  Negative preds: {(df['previsao']<0).sum()}/{len(df)} ({(df['previsao']<0).mean()*100:.0f}%)")
print(f"  Alloc=0: {(df['allocation']==0).sum()}/{len(df)} ({(df['allocation']==0).mean()*100:.0f}%)")
print(f"  Alloc=1: {(df['allocation']==1).sum()}/{len(df)} ({(df['allocation']==1).mean()*100:.0f}%)")

df_closed = df.dropna(subset=['retorno_strat']).copy()
df_closed['year'] = pd.to_datetime(df_closed['date']).dt.year
print(f"\nPer-year:")
for y in sorted(df_closed['year'].unique()):
    sub = df_closed[df_closed['year'] == y]
    s = float(np.prod(1 + sub['retorno_strat']) - 1)
    b = float(np.prod(1 + sub['retorno_btc']) - 1)
    print(f"  {y}: {len(sub)} rebals  strat {s*100:+7.1f}%  vs BTC {b*100:+7.1f}%  "
          f"alloc avg {sub['allocation'].mean()*100:5.1f}%  alloc=0: {(sub['allocation']==0).sum()}")

cum_strat = float(np.prod(1 + df_closed['retorno_strat'].values) - 1)
cum_btc = float(np.prod(1 + df_closed['retorno_btc'].values) - 1)
print(f"\nFull walkforward {df['date'].iloc[0]} to {df['date'].iloc[-2]}:")
print(f"  Strategy: {cum_strat*100:+.1f}%")
print(f"  BTC B&H:  {cum_btc*100:+.1f}%")
print(f"  Excess:   {(cum_strat - cum_btc)*100:+.1f}pp")
