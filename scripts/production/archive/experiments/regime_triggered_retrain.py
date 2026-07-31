"""
Regime-shift triggered retrain — does adapting faster fix 2023's miss?

Hypothesis: 2023 lost -78pp because Jan-Jun model was trained on 2022 bear,
too defensive when BTC entered recovery. Semi-annual retrain (Jul/2023)
came too late to catch Q1-Q2 rally.

Solution test: trigger an EXTRA retrain when regime stays in a NEW state
for N+ consecutive rebals since last training.

Variants:
  A. BASELINE       semi-annual only (current production, 9 retrains)
  B. TRIG_4         semi-annual + trigger after 4+ rebals in new regime
  C. TRIG_6         semi-annual + trigger after 6+ rebals in new regime
  D. TRIG_8         semi-annual + trigger after 8+ rebals in new regime
  E. TRIG_4_ONLY    triggered only (no semi-annual baseline)

Variants tested in BRL puro, single seed=242 first.
If 2023 improves significantly, extend to multi-seed.
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
MIN_DAYS_BETWEEN_RETRAIN = 30  # don't retrain too often

# Load BRL FX
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


def semi_annual_cutoffs(start, end):
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


def train_at_date(ds, train_end_date, base_seed):
    """Train using all data strictly before train_end_date."""
    mask = ds['date'] < train_end_date
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


def predict_one_rebal(ds, models, d0, idx):
    """Predict for one rebal date."""
    regs, clss = models
    X_row = np.nan_to_num(
        ds.iloc[idx][FEATURES_37].values.astype(float).reshape(1, -1), nan=0.0
    )
    pred = float(np.mean([m.predict(X_row)[0] for m in regs]))
    p_up = float(np.mean([m.predict_proba(X_row)[0, 1] for m in clss]))
    conf = float(1 / (1 + np.exp(-abs(p_up - 0.5) * SIGMOID_SCALE)))
    return pred, p_up, conf


def get_regime_at(ds, idx):
    prices = ds['price_usd'].values[:idx + 1]
    s50 = pd.Series(prices).rolling(50).mean().iloc[-1]
    s200 = pd.Series(prices).rolling(200).mean().iloc[-1]
    return get_regime(prices[-1], s50, s200)


def run_strategy(ds, rebals, rf, base_seed, semi_annual=True, trigger_n=None):
    """Run walk-forward with optional regime-shift triggered retrain.

    Returns (df with rebal results, list of retrain dates)
    """
    ds_by_date = ds.set_index('date')
    retrain_dates = []  # list of (type, date) tuples
    cutoff_models = {}

    # Build initial set of semi-annual cutoffs
    if semi_annual:
        semi_cuts = semi_annual_cutoffs(rebals[0], rebals[-1])
        for c in semi_cuts:
            cutoff_models[c] = train_at_date(ds, c, base_seed)
            retrain_dates.append(('semi', c))
    else:
        first_cut = pd.Timestamp('2022-01-01')
        cutoff_models[first_cut] = train_at_date(ds, first_cut, base_seed)
        retrain_dates.append(('initial', first_cut))

    # Track regime tracking state — RESET WITH EACH RETRAIN
    last_seen_train_date = None  # last cutoff date that affected our state
    last_train_regime = None
    rebal_count_in_new_regime = 0

    def regime_30d_before(date):
        train_idx = ds[ds['date'] < date].index[-30:]
        regimes = [get_regime_at(ds, ti) for ti in train_idx]
        if not regimes:
            return 'MILD'
        return pd.Series(regimes).mode().iloc[0]

    rows = []
    for i, d0 in enumerate(rebals):
        d1 = rebals[i + 1] if i + 1 < len(rebals) else rebals[-1] + pd.Timedelta(days=7)
        if d0 not in ds_by_date.index or d1 not in ds_by_date.index:
            continue
        idx = ds[ds['date'] == d0].index[0]
        regime_now = get_regime_at(ds, idx)

        # Find most recent retrain (semi or triggered) <= d0
        applicable_retrains = [d for t, d in retrain_dates if d <= d0]
        most_recent_retrain = max(applicable_retrains) if applicable_retrains else None

        # If we crossed a NEW retrain since last loop, reset tracking
        if most_recent_retrain != last_seen_train_date:
            last_seen_train_date = most_recent_retrain
            last_train_regime = regime_30d_before(most_recent_retrain) if most_recent_retrain else 'MILD'
            rebal_count_in_new_regime = 0

        # Triggered retrain check
        triggered = False
        if trigger_n is not None:
            if regime_now != last_train_regime:
                rebal_count_in_new_regime += 1
                days_since_last_retrain = (d0 - most_recent_retrain).days if most_recent_retrain else 999
                if (rebal_count_in_new_regime >= trigger_n and
                    days_since_last_retrain >= MIN_DAYS_BETWEEN_RETRAIN):
                    # Trigger retrain at d0
                    cutoff_models[d0] = train_at_date(ds, d0, base_seed)
                    retrain_dates.append(('triggered', d0))
                    last_seen_train_date = d0
                    last_train_regime = regime_now
                    rebal_count_in_new_regime = 0
                    triggered = True
            else:
                rebal_count_in_new_regime = 0

        # Pick most recent applicable model
        applicable = [(c, m) for c, m in cutoff_models.items() if c <= d0]
        c = max(a[0] for a in applicable)
        models = cutoff_models[c]

        pred, p_up, conf = predict_one_rebal(ds, models, d0, idx)
        K = K_H1[regime_now]
        alloc = float(np.clip(pred * K * conf, ALLOC_MIN, ALLOC_MAX))

        p0 = float(ds_by_date.loc[d0, 'price_usd'])
        p1 = float(ds_by_date.loc[d1, 'price_usd'])
        usd_brl_0 = float(ds_by_date.loc[d0, 'usdbrl'])
        usd_brl_1 = float(ds_by_date.loc[d1, 'usdbrl'])
        # BRL return = BTC USD return × FX impact
        btc_brl_ret = (p1 * usd_brl_1) / (p0 * usd_brl_0) - 1
        mask_cdi = (rf.index > d0) & (rf.index <= d1)
        cdi_ret = float((1 + rf[mask_cdi]).prod() - 1)
        strat_ret = alloc * btc_brl_ret + (1 - alloc) * cdi_ret

        rows.append({
            'date': d0, 'pred': pred, 'p_up': p_up, 'regime': regime_now,
            'alloc': alloc, 'btc_brl_fwd': btc_brl_ret, 'cdi': cdi_ret,
            'strat': strat_ret, 'model_cutoff': c, 'triggered': triggered,
        })

    return pd.DataFrame(rows), retrain_dates


def metrics(df, ds, daily_start, daily_end, rf):
    """Daily MtM metrics in BRL."""
    daily_ds = ds[(ds['date'] >= daily_start) & (ds['date'] <= daily_end)].copy()
    daily_ds['btc_ret_brl'] = (daily_ds['price_usd'] * daily_ds['usdbrl']).pct_change().fillna(0)
    cdi_daily = (1.13)**(1/365) - 1

    rebal_dates_arr = df['date'].values
    rebal_alloc_arr = df['alloc'].values

    rets = []
    prev = 0
    for _, row in daily_ds.iterrows():
        mask = rebal_dates_arr < row['date'].to_numpy()
        new_alloc = rebal_alloc_arr[mask][-1] if mask.any() else 0
        cost = abs(new_alloc - prev) * 8 / 10000 if new_alloc != prev else 0
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

    # Per year
    daily_ds = daily_ds.reset_index(drop=True)
    daily_ds['_strat'] = rets
    daily_ds['_year'] = daily_ds['date'].dt.year
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

    print(f"Regime-triggered retrain test, seed={SEED}, BRL")
    print(f"Rebals: {len(rebals)}, period 2022-01-07 → 2026-04-17")
    print()

    variants = [
        ('A. BASELINE (semi only)',  True,  None),
        ('B. SEMI + TRIG_4',         True,  4),
        ('C. SEMI + TRIG_6',         True,  6),
        ('D. SEMI + TRIG_8',         True,  8),
        ('E. TRIG_4 ONLY (no semi)', False, 4),
    ]
    results = {}
    for name, semi, trig_n in variants:
        t0 = time.time()
        df, retrain_dates = run_strategy(ds, rebals, rf, SEED, semi, trig_n)
        m = metrics(df, ds, start, end, rf)
        m['n_retrains'] = len(retrain_dates)
        m['retrain_dates'] = [(t, d.strftime('%Y-%m-%d')) for t, d in retrain_dates]
        results[name] = m
        triggered_count = sum(1 for t, _ in retrain_dates if t == 'triggered')
        print(f"{name:<30s}: {time.time()-t0:.0f}s  "
              f"retrains={len(retrain_dates)} ({triggered_count} triggered)", flush=True)

    print(f"\n{'='*120}")
    print(f"{'Variant':<28s} {'CAGR':>9s} {'Sortino':>8s} {'Shp_x':>7s} {'DD':>9s} | "
          f"{'2022':>9s} {'2023':>10s} {'2024':>10s} {'2025':>10s} {'2026':>9s}")
    print(f"{'-'*120}")
    for name, m in results.items():
        y = m['yearly']
        print(f"{name:<28s} {m['cagr']*100:+8.1f}% {m['sortino']:7.2f} "
              f"{m['sharpe_x']:6.2f} {m['max_dd']*100:+7.2f}% | "
              f"{y.get(2022,0)*100:+8.1f}% {y.get(2023,0)*100:+9.1f}% "
              f"{y.get(2024,0)*100:+9.1f}% {y.get(2025,0)*100:+9.1f}% {y.get(2026,0)*100:+8.1f}%")

    # Show baseline retrains
    print(f"\nRetrain dates per variant:")
    for name, m in results.items():
        print(f"  {name}: {[d for t, d in m['retrain_dates']]}")

    # Save
    with open(OUT / 'regime_triggered_retrain.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {OUT / 'regime_triggered_retrain.json'}")


if __name__ == '__main__':
    main()
