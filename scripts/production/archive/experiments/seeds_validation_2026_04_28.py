"""
10-seed validation of current production config (H1 + 32 features).

Goal: replace single-seed numbers (which vary 10-15% between runs due to XGBoost
parallel non-determinism) with mean+/-std across 10 seeds.

For each seed runs full walk-forward (9 cutoffs Jan/Jul 2022-2026, 248 rebals)
and reports:
  - cum return
  - Sortino weekly (rebal-level returns)
  - Sortino daily (mark-to-market)
  - Sharpe excess weekly (CDI subtracted)
  - Max DD weekly (rebal close)
  - Max DD daily (mark-to-market intraweek)

Both with and WITHOUT acc derisk control applied.

Run:
    python scripts/production/archive/experiments/seeds_validation_2026_04_28.py
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

K_H1 = {'BULL': 60, 'MILD': 30, 'BEAR': 15}
SEEDS = [242, 251, 263, 277, 281, 293, 307, 311, 317, 331]  # primes spaced
RETRAIN_MONTHS = [1, 7]
ROLLING_WINDOW = 12
ACC_THRESHOLD = 0.48
ACC_MULT = 0.5
# NOTE: this validator reports GROSS (pre-cost) returns. Applying a per-rebal
# turnover cost as a single daily "spike" was tried (4 bps) but it dilutes the
# Sortino downside-RMS and spuriously RAISES daily Sortino. The cost-stress
# table (docs/MODEL_FINAL.md) is the authority on cost: 4 bps ~ -0.6pp CAGR and
# ~ -0.01 Sortino. Keep this run gross; subtract that documented delta if a
# net figure is needed.


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
    cls_params = {**{k: v for k, v in XGB_PARAMS.items() if k not in ('objective',)},
                  'objective': 'binary:logistic', 'eval_metric': 'logloss'}
    m = xgb.XGBClassifier(**cls_params, random_state=s)
    rng = np.random.RandomState(s)
    idx = rng.choice(len(X), size=len(X), replace=True)
    m.fit(X[idx], y[idx])
    return m


def train_at_cutoff(ds, cutoff, features, base_seed):
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
    seeds = [base_seed + i * 7 for i in range(BAGS)]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        regs = list(ex.map(_train_one_reg,
            [(s, X[train_idx], treg[train_idx]) for s in seeds]))
        clss = list(ex.map(_train_one_cls,
            [(s, X[train_idx], tcls[train_idx]) for s in seeds]))
    return regs, clss


def run_seed(ds, base_seed, rebals, rf):
    """Run full walk-forward for a single seed. Returns dataframe of rebals."""
    cuts = retrain_cutoffs(rebals[0], rebals[-1])
    cutoff_models = {}
    for c in cuts:
        cutoff_models[c] = train_at_cutoff(ds, c, FEATURES_37, base_seed)

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
        alloc_raw = float(np.clip(pred * K_H1[regime] * conf, ALLOC_MIN, ALLOC_MAX))
        if d1 not in ds_by_date.index:
            continue
        p0 = float(ds_by_date.loc[d0, 'price_usd'])
        p1 = float(ds_by_date.loc[d1, 'price_usd'])
        btc_ret = p1 / p0 - 1
        mask = (rf.index > d0) & (rf.index <= d1)
        cdi_ret = float((1 + rf[mask]).prod() - 1)
        rows.append({
            'date': d0, 'd1': d1, 'pred': pred, 'p_up': p_up, 'conf_factor': conf,
            'regime': regime, 'alloc_raw': alloc_raw, 'btc_fwd': btc_ret,
            'cdi_period': cdi_ret,
        })
    return pd.DataFrame(rows)


def apply_acc_derisk(df: pd.DataFrame) -> np.ndarray:
    """Apply the current production binary 48% rule to alloc_raw."""
    correct = ((df['pred'] > 0) & (df['btc_fwd'] > 0)) | \
              ((df['pred'] < 0) & (df['btc_fwd'] < 0))
    n = len(df)
    rolling_acc = np.full(n, np.nan)
    for i in range(n):
        if i >= ROLLING_WINDOW:
            rolling_acc[i] = correct.iloc[i - ROLLING_WINDOW:i].mean()
    mult = np.where((rolling_acc < ACC_THRESHOLD) & (~np.isnan(rolling_acc)),
                    ACC_MULT, 1.0)
    return np.clip(df['alloc_raw'].values * mult, 0.0, 1.0)


def expand_to_daily(df: pd.DataFrame, alloc_col: str, ds: pd.DataFrame, rf: pd.Series):
    """Expand rebal-level allocations to daily MtM returns.

    Returns (strat_daily, cdi_daily) so callers can compute a daily EXCESS
    Sharpe (over the daily CDI), matching the daily Sharpe convention used in
    the comparison tables.
    """
    ds = ds.copy()
    ds['btc_ret_daily'] = ds['price_usd'].pct_change()
    out_strat = []
    out_cdi = []
    for i in range(len(df)):
        d_start = df['date'].iloc[i]
        alloc = float(df[alloc_col].iloc[i])
        d_end = df['date'].iloc[i + 1] if i + 1 < len(df) else df['d1'].iloc[i]
        days = ds[(ds['date'] > d_start) & (ds['date'] <= d_end)]
        for _, row in days.iterrows():
            d = row['date']
            btc_d = row['btc_ret_daily'] if not pd.isna(row['btc_ret_daily']) else 0.0
            cdi_d = float(rf.get(d, 0.0))
            out_strat.append(alloc * btc_d + (1 - alloc) * cdi_d)
            out_cdi.append(cdi_d)
    return np.array(out_strat), np.array(out_cdi)


def metrics_from_returns(weekly: np.ndarray, daily: np.ndarray, cdi_weekly: np.ndarray,
                         daily_cdi: np.ndarray = None):
    cum = float(np.cumprod(1 + weekly)[-1] - 1)
    # Weekly Sortino
    neg_w = weekly[weekly < 0]
    dev_w = float(np.sqrt(np.mean(neg_w ** 2))) if len(neg_w) > 0 else 1e-9
    sortino_w = float(np.mean(weekly) / dev_w * np.sqrt(52)) if dev_w > 0 else 0.0
    # Excess Sharpe weekly
    excess = weekly - cdi_weekly
    sd_e = float(np.std(excess, ddof=0))
    sharpe_excess = float(np.mean(excess) / sd_e * np.sqrt(52)) if sd_e > 0 else 0.0
    # Max DD weekly
    eq_w = np.concatenate([[1.0], np.cumprod(1 + weekly)])
    peak_w = np.maximum.accumulate(eq_w)
    maxdd_w = float(((eq_w - peak_w) / peak_w).min())
    # Daily Sortino + DD
    neg_d = daily[daily < 0]
    dev_d = float(np.sqrt(np.mean(neg_d ** 2))) if len(neg_d) > 0 else 1e-9
    sortino_d = float(np.mean(daily) / dev_d * np.sqrt(365)) if dev_d > 0 else 0.0
    eq_d = np.concatenate([[1.0], np.cumprod(1 + daily)])
    peak_d = np.maximum.accumulate(eq_d)
    maxdd_d = float(((eq_d - peak_d) / peak_d).min())
    cagr = float((1 + cum) ** (365 / len(daily)) - 1) if len(daily) > 0 else 0.0
    # Daily excess Sharpe (over daily CDI) — matches the comparison-table convention
    if daily_cdi is not None and len(daily_cdi) == len(daily):
        ex_d = daily - daily_cdi
        sd_ed = float(np.std(ex_d, ddof=0))
        sharpe_excess_d = float(np.mean(ex_d) / sd_ed * np.sqrt(365)) if sd_ed > 0 else 0.0
    else:
        sharpe_excess_d = 0.0
    return {
        'cum_ret': cum, 'cagr': cagr,
        'sortino_w': sortino_w, 'sortino_d': sortino_d,
        'sharpe_excess_w': sharpe_excess, 'sharpe_excess_d': sharpe_excess_d,
        'max_dd_w': maxdd_w, 'max_dd_d': maxdd_d,
    }


def main():
    ds = pd.read_csv(DATASET, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    print(f"Dataset: {len(ds)} rows {ds['date'].min().date()} -> {ds['date'].max().date()}")
    print(f"Seeds: {SEEDS} (n={len(SEEDS)})")
    print(f"Config: H1 (60/30/15), 32 features, sigmoid=15, BAGS={BAGS}")

    # Build rebal dates once (independent of seed)
    start = pd.Timestamp('2022-01-07')
    end = pd.Timestamp('2026-04-17')
    sub = ds[(ds['date'] >= start) & (ds['date'] <= end)].copy()
    sub['daily_ret'] = sub['price_usd'].pct_change()
    sub['dow'] = sub['date'].dt.dayofweek
    fridays = set(sub.loc[sub['dow'].isin(REBAL_DOW), 'date'])
    emerg = set(sub.loc[sub['daily_ret'].abs() > EMERGENCY_THRESHOLD, 'date'])
    rebals = sorted(fridays | emerg)
    print(f"Rebals: {len(rebals)}")

    # CDI series
    rf = pd.Series(
        build_rf_daily(pd.date_range(start - pd.Timedelta(days=10), end + pd.Timedelta(days=10), freq='D')),
        index=pd.date_range(start - pd.Timedelta(days=10), end + pd.Timedelta(days=10), freq='D'),
    )

    all_results_no_ctrl = []
    all_results_with_ctrl = []

    print(f"\n{'='*100}")
    print(f"{'seed':>5s} | {'NO control':^45s} | {'WITH acc derisk (current 48%)':^45s}")
    print(f"{'':>5s} | {'cum':>8s} {'Sort_w':>7s} {'Sort_d':>7s} {'Shp_x':>6s} {'DDw':>6s} {'DDd':>7s} | "
          f"{'cum':>8s} {'Sort_w':>7s} {'Sort_d':>7s} {'Shp_x':>6s} {'DDw':>6s} {'DDd':>7s}")
    print(f"{'-'*100}")

    for s_idx, seed in enumerate(SEEDS):
        t0 = time.time()
        df = run_seed(ds, seed, rebals, rf)

        # No control (gross)
        weekly_nc = (df['alloc_raw'] * df['btc_fwd'] +
                     (1 - df['alloc_raw']) * df['cdi_period']).values
        daily_nc, daily_cdi_nc = expand_to_daily(df, 'alloc_raw', ds, rf)
        m_nc = metrics_from_returns(weekly_nc, daily_nc, df['cdi_period'].values, daily_cdi_nc)

        # With acc derisk (gross)
        df['alloc_with_ctrl'] = apply_acc_derisk(df)
        weekly_wc = (df['alloc_with_ctrl'] * df['btc_fwd'] +
                     (1 - df['alloc_with_ctrl']) * df['cdi_period']).values
        daily_wc, daily_cdi_wc = expand_to_daily(df, 'alloc_with_ctrl', ds, rf)
        m_wc = metrics_from_returns(weekly_wc, daily_wc, df['cdi_period'].values, daily_cdi_wc)

        m_nc['seed'] = seed
        m_wc['seed'] = seed
        all_results_no_ctrl.append(m_nc)
        all_results_with_ctrl.append(m_wc)

        elapsed = time.time() - t0
        print(f"{seed:>5d} | "
              f"{m_nc['cum_ret']*100:+7.1f}% {m_nc['sortino_w']:7.2f} {m_nc['sortino_d']:7.2f} "
              f"{m_nc['sharpe_excess_w']:6.2f} {m_nc['max_dd_w']*100:5.2f}% {m_nc['max_dd_d']*100:6.2f}% | "
              f"{m_wc['cum_ret']*100:+7.1f}% {m_wc['sortino_w']:7.2f} {m_wc['sortino_d']:7.2f} "
              f"{m_wc['sharpe_excess_w']:6.2f} {m_wc['max_dd_w']*100:5.2f}% {m_wc['max_dd_d']*100:6.2f}% | "
              f"{elapsed:.0f}s", flush=True)

    print(f"{'='*100}")

    def agg(lst, key):
        v = np.array([r[key] for r in lst])
        return float(v.mean()), float(v.std(ddof=1))

    print(f"\nAGGREGATED ACROSS {len(SEEDS)} SEEDS:")
    print(f"{'='*100}")
    print(f"{'metric':<22s} | {'NO control (mean +/- std)':^35s} | {'WITH derisk (mean +/- std)':^35s}")
    print(f"{'-'*100}")
    for key, label, scale, fmt in [
        ('cum_ret',           'cum_ret',     100, '{:+7.1f}%'),
        ('cagr',              'CAGR',        100, '{:+7.1f}%'),
        ('sortino_w',         'Sortino weekly',1, '{:7.2f} '),
        ('sortino_d',         'Sortino daily', 1, '{:7.2f} '),
        ('sharpe_excess_w',   'Sharpe excess (w)', 1, '{:7.2f} '),
        ('sharpe_excess_d',   'Sharpe excess (d)', 1, '{:7.2f} '),
        ('max_dd_w',          'Max DD weekly', 100, '{:7.2f}%'),
        ('max_dd_d',          'Max DD daily',  100, '{:7.2f}%'),
    ]:
        m_nc, s_nc = agg(all_results_no_ctrl, key)
        m_wc, s_wc = agg(all_results_with_ctrl, key)
        print(f"  {label:<20s} | {fmt.format(m_nc*scale)} +/- {fmt.format(s_nc*scale).strip()} "
              f"({s_nc/abs(m_nc)*100 if m_nc!=0 else 0:5.1f}%)  | "
              f"{fmt.format(m_wc*scale)} +/- {fmt.format(s_wc*scale).strip()} "
              f"({s_wc/abs(m_wc)*100 if m_wc!=0 else 0:5.1f}%)")

    # Save
    out = {
        'config': f'H1 (60/30/15), 32 features, sigmoid=15, BAGS={BAGS}, GROSS (pre-cost), BCB CDI, no kill switch active in this run',
        'period': f"{start.date()} -> {end.date()} ({len(rebals)} rebals)",
        'seeds': SEEDS,
        'no_control': all_results_no_ctrl,
        'with_acc_derisk': all_results_with_ctrl,
        'aggregated_no_control': {k: list(agg(all_results_no_ctrl, k))
                                   for k in ['cum_ret','cagr','sortino_w','sortino_d',
                                             'sharpe_excess_w','sharpe_excess_d','max_dd_w','max_dd_d']},
        'aggregated_with_derisk': {k: list(agg(all_results_with_ctrl, k))
                                    for k in ['cum_ret','cagr','sortino_w','sortino_d',
                                              'sharpe_excess_w','max_dd_w','max_dd_d']},
    }
    out_path = OUT / 'seeds_validation_2026_04_28.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
