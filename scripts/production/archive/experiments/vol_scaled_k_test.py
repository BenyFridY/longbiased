"""
Teste: K fixo vs K escalado por volatilidade (Moskowitz/Pedersen TSMOM).

Fórmula:
  vol_realized = log_returns.rolling(20).std() * sqrt(365)
  K_scale      = vol_target / vol_realized
  K_dinamico   = K_REGIME[regime] * K_scale
  alloc        = clip(pred * K_dinamico * confidence, 0, 1)

Variantes testadas:
  1. Baseline: K fixo (H=3 produção, 100/50/20)
  2. Vol-scaled K, target=0.70 (media historica BTC), sem clamp
  3. Vol-scaled K, target=0.70, clamp [0.5, 2.0] (conservador)
  4. Vol-scaled K, target=0.60, clamp [0.5, 2.0] (ainda mais conservador)

Mesma walk-forward que horizon_ablation_4y.py, mesmas features, mesmos cutoffs.
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
HORIZON = 3


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


def train_at_cutoff(ds, cutoff):
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


def realized_vol(prices_upto, window=20):
    """Annualized log-return vol over last `window` days."""
    if len(prices_upto) < window + 1:
        return 0.70  # default to target if not enough data
    log_rets = np.diff(np.log(prices_upto[-window - 1:]))
    vol = np.std(log_rets) * np.sqrt(365)
    return max(vol, 0.10)  # floor to avoid division blow-up


def run_variant(ds, start, end, cutoff_models, rf_series,
                vol_scale=False, vol_target=0.70, vol_clamp=None):
    """vol_scale=False => K fixo. True => K ajustado por vol."""
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

        K_base = K_REGIME[regime]
        if vol_scale:
            vol_now = realized_vol(prices, window=20)
            scale = vol_target / vol_now
            if vol_clamp is not None:
                scale = np.clip(scale, vol_clamp[0], vol_clamp[1])
            K_used = K_base * scale
        else:
            K_used = K_base
            vol_now = realized_vol(prices, window=20)  # for logging
            scale = 1.0

        alloc = float(np.clip(pred * K_used * conf, ALLOC_MIN, ALLOC_MAX))
        p0 = float(ds_by.loc[d0, 'price_usd'])
        p1 = float(ds_by.loc[d1, 'price_usd'])
        btc_ret = p1 / p0 - 1
        mask = (rf_series.index > d0) & (rf_series.index <= d1)
        cdi_ret = float((1 + rf_series[mask]).prod() - 1)
        strat_ret = alloc * btc_ret + (1 - alloc) * cdi_ret
        cum *= (1 + strat_ret)
        cum_btc *= (1 + btc_ret)
        rows.append({
            'date': d0, 'regime': regime, 'pred': pred, 'p_up': p_up, 'conf': conf,
            'vol_20d': vol_now, 'vol_scale': scale, 'K_used': K_used,
            'alloc': alloc, 'btc_fwd': btc_ret, 'strat': strat_ret,
            'dir_match': (pred > 0) == (btc_ret > 0),
        })
    return pd.DataFrame(rows), cum - 1, cum_btc - 1


def sortino(returns, ppy=52):
    r = np.array(returns)
    down = r[r < 0]
    if len(down) == 0:
        return float('inf')
    return r.mean() * ppy / (np.sqrt((down ** 2).mean()) * np.sqrt(ppy))


def max_dd(returns):
    cum = (1 + np.array(returns)).cumprod()
    peak = np.maximum.accumulate(cum)
    return ((cum - peak) / peak).min()


def report(df, cum, cum_btc, label):
    acc = df['dir_match'].sum()
    dd = max_dd(df['strat'])
    s = sortino(df['strat'])
    ann = (1 + cum) ** (1 / 4.3) - 1
    print(f'\n{"=" * 72}')
    print(f'  {label}')
    print(f'{"=" * 72}')
    print(f'N rebals:          {len(df)}')
    print(f'Return total:      {cum*100:+.2f}%')
    print(f'CAGR:              {ann*100:.1f}%/yr')
    print(f'Sortino:           {s:.2f}')
    print(f'Max DD:            {dd*100:+.2f}%')
    print(f'Dir accuracy:      {acc}/{len(df)} = {acc/len(df)*100:.1f}%')
    print(f'Mean alloc:        {df["alloc"].mean()*100:.1f}%')
    print(f'Max alloc:         {df["alloc"].max()*100:.1f}%')
    print(f'% alloc=0:         {(df["alloc"]==0).sum()}/{len(df)}')
    print(f'% alloc>=70%:      {(df["alloc"]>=0.7).sum()}/{len(df)}')
    if 'vol_scale' in df.columns:
        print(f'vol_scale range:   [{df["vol_scale"].min():.2f}, {df["vol_scale"].max():.2f}]')
        print(f'vol_scale mean:    {df["vol_scale"].mean():.2f}')
        print(f'K_used range:      [{df["K_used"].min():.0f}, {df["K_used"].max():.0f}]')


def main():
    ds = pd.read_csv(
        ROOT / 'scripts/production/data/dataset_production.csv',
        parse_dates=['date']
    ).sort_values('date').reset_index(drop=True)

    start = pd.Timestamp('2022-01-01')
    end = ds['date'].iloc[-1]

    cutoffs = retrain_cutoffs(start, end)
    print(f'Cutoffs: {[c.date() for c in cutoffs]}', flush=True)

    rf = pd.Series(
        build_rf_daily(pd.date_range(start, end, freq='D')),
        index=pd.date_range(start, end, freq='D'),
    )

    print('\nTraining H=3 models...', flush=True)
    t0 = time.time()
    models = {}
    for c in cutoffs:
        t1 = time.time()
        models[c] = train_at_cutoff(ds, c)
        print(f'  {c.date()} trained in {time.time()-t1:.0f}s', flush=True)
    print(f'Total train: {time.time()-t0:.0f}s')

    # Variant 1: baseline K fixo
    df1, cum1, cum_btc = run_variant(ds, start, end, models, rf, vol_scale=False)

    # Variant 2: vol-scaled, target 0.70, sem clamp
    df2, cum2, _ = run_variant(ds, start, end, models, rf,
                                vol_scale=True, vol_target=0.70, vol_clamp=None)

    # Variant 3: vol-scaled, target 0.70, clamp [0.5, 2.0]
    df3, cum3, _ = run_variant(ds, start, end, models, rf,
                                vol_scale=True, vol_target=0.70, vol_clamp=(0.5, 2.0))

    # Variant 4: vol-scaled, target 0.60, clamp [0.5, 2.0]
    df4, cum4, _ = run_variant(ds, start, end, models, rf,
                                vol_scale=True, vol_target=0.60, vol_clamp=(0.5, 2.0))

    report(df1, cum1, cum_btc, '[1] BASELINE - K fixo 100/50/20')
    report(df2, cum2, cum_btc, '[2] Vol-scaled K, target=0.70, SEM clamp')
    report(df3, cum3, cum_btc, '[3] Vol-scaled K, target=0.70, clamp [0.5, 2.0]')
    report(df4, cum4, cum_btc, '[4] Vol-scaled K, target=0.60, clamp [0.5, 2.0]')

    # Side-by-side
    print(f'\n{"=" * 72}')
    print('SIDE-BY-SIDE (BTC reference: {:+.2f}%)'.format(cum_btc * 100))
    print('=' * 72)
    fmt = '{:<30} {:>12} {:>10} {:>10} {:>10}'
    print(fmt.format('Variant', 'Return', 'Sortino', 'Max DD', 'Dir acc'))
    print('-' * 72)
    for lbl, d, c in [('[1] Baseline K fixo',       df1, cum1),
                       ('[2] Vol-scaled target 0.70', df2, cum2),
                       ('[3] Vol-scaled 0.70 clamp',  df3, cum3),
                       ('[4] Vol-scaled 0.60 clamp',  df4, cum4)]:
        print(fmt.format(lbl, f'{c*100:+.2f}%',
                         f'{sortino(d["strat"]):.2f}',
                         f'{max_dd(d["strat"])*100:+.2f}%',
                         f'{d["dir_match"].mean()*100:.1f}%'))

    out = ROOT / 'outputs/results/vol_scaled_k_test.csv'
    combined = pd.concat([
        df1.assign(variant='baseline'),
        df2.assign(variant='vol070_noclamp'),
        df3.assign(variant='vol070_clamp'),
        df4.assign(variant='vol060_clamp'),
    ], ignore_index=True)
    combined.to_csv(out, index=False)
    print(f'\nSalvou: {out}')


if __name__ == '__main__':
    main()
