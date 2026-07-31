"""
Conditional retrain experiment.

Tests adaptive retrain strategies triggered by market/model conditions.
Based on standard ML/finance concepts:

1. BASELINE (semi) — retrain every 6 months (current schedule)
2. DDT (Drawdown-Triggered) — retrain when strategy DD since last retrain > 5%
   Rationale: DD can indicate model decay; refresh on adversity
3. VRC (Vol Regime Change) — retrain when 30d BTC realized vol > 1.5x its level at last retrain
   Rationale: K multiplier is calibrated to historical vol; big vol shift invalidates it
4. ACC (Accuracy Decay) — retrain when 8-week rolling direction acc < 50%
   Rationale: classic concept drift detection (Gama et al. 2004 "Learning with drift detection")
5. BMT (Big Move Trigger) — retrain after BTC daily |move| > 10%
   Rationale: large moves often signal regime change (e.g., Feb 2026 -14% crash)
6. HYBRID (SEMI + ACC) — base semi-annual + forced retrain if ACC trigger fires
   Rationale: best of both worlds — time-based safety net + responsive adaptation
7. HYBRID2 (SEMI + DDT) — base semi-annual + forced retrain if DD trigger fires

All variants enforce a MIN_GAP (30 days) between retrains to prevent thrashing.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'production'))
sys.path.insert(0, str(ROOT))

from config import (
    FEATURES_37 as FEATURES_ALL,
    K_REGIME, ALLOC_MIN, ALLOC_MAX, SIGMOID_SCALE,
    HORIZON, REBAL_DOW, EMERGENCY_THRESHOLD,
)
from generate_signal import (
    train_regression_ensemble, train_classifier_ensemble, get_regime,
)
from src.features.macro.cdi_rates import build_rf_daily

DATA_DIR = ROOT / 'scripts' / 'production' / 'data'
OUT_DIR = ROOT / 'outputs' / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_GAP_DAYS = 30  # anti-thrashing: minimum days between any two retrains


def build_targets(prices, n, horizon=HORIZON):
    treg = np.zeros(n); tcls = np.zeros(n)
    for i in range(n - horizon):
        treg[i] = (prices[i + horizon] - prices[i]) / prices[i]
        tcls[i] = 1.0 if prices[i + horizon] > prices[i] else 0.0
    return treg, tcls


def train_at_idx(ds, end_idx, seed=242):
    """Train using data ds.iloc[:end_idx]."""
    sub = ds.iloc[:end_idx].reset_index(drop=True)
    X = np.nan_to_num(sub[FEATURES_ALL].values.astype(float), nan=0.0)
    prices = sub['price_usd'].values
    n = len(sub)
    treg, tcls = build_targets(prices, n)
    gap = max(HORIZON, 5)
    train_end = n - gap
    train_idx = np.arange(60, train_end + 1)
    train_idx = train_idx[~np.any(np.isnan(X[train_idx]), axis=1)]
    if len(train_idx) < 200:
        return None, None
    reg = train_regression_ensemble(X[train_idx], treg[train_idx], seed)
    cls = train_classifier_ensemble(X[train_idx], tcls[train_idx], seed)
    return reg, cls


def predict(reg, cls, x_row):
    pred = float(np.mean([m.predict(x_row)[0] for m in reg]))
    p_up = float(np.mean([m.predict_proba(x_row)[0, 1] for m in cls]))
    conf = float(1 / (1 + np.exp(-abs(p_up - 0.5) * SIGMOID_SCALE)))
    return pred, p_up, conf


def compute_alloc(pred, conf, regime):
    return float(np.clip(pred * K_REGIME[regime] * conf, ALLOC_MIN, ALLOC_MAX))


def run_strategy(ds, start, end, strategy, rf_series, seed=242, verbose=True):
    """
    Run walk-forward backtest with a conditional retrain strategy.

    strategy: dict with keys 'name', 'trigger' (callable(state) -> bool), 'schedule' (list of dates)
    """
    name = strategy['name']
    sched_dates = strategy.get('schedule', [])
    trigger_fn = strategy.get('trigger')

    if verbose:
        print(f"\n{'='*60}\n[{time.strftime('%H:%M:%S')}] Strategy: {name}\n{'='*60}", flush=True)

    prices = ds['price_usd'].values
    dates = ds['date'].values

    # State during backtest
    reg_models = None
    cls_models = None
    last_retrain_idx = None
    last_retrain_date = None

    # Initial train (bootstrap): use all data before 'start'
    start_idx = ds[ds['date'] == start].index[0] if start in ds['date'].values else ds[ds['date'] >= start].index[0]
    end_idx = ds[ds['date'] == end].index[0] if end in ds['date'].values else ds[ds['date'] <= end].index[-1]

    # Always train at the beginning (using all data prior to start, with gap)
    gap = max(HORIZON, 5)
    init_train_end = start_idx - gap
    if init_train_end < 500:
        print("  Not enough pre-start data for initial training")
        return None
    t0 = time.time()
    reg_models, cls_models = train_at_idx(ds, init_train_end + 1, seed=seed)
    if reg_models is None:
        return None
    last_retrain_idx = init_train_end + 1
    last_retrain_date = ds.iloc[init_train_end]['date']
    retrain_dates = [last_retrain_date.strftime('%Y-%m-%d')]
    if verbose:
        print(f"  Initial train at {last_retrain_date.date()} in {time.time()-t0:.0f}s")

    # Backtest loop
    daily_strat = []
    daily_btc = []
    daily_cdi = []
    prev_alloc = 0.0
    recent_hits = []  # last 8 rebals: 1 for hit, 0 for miss
    recent_preds = []  # (date, pred) for tracking
    dd_since_retrain = 0.0
    peak_since_retrain = 1.0
    cum_since_retrain = 1.0
    vol_at_retrain = None

    for idx in range(start_idx, end_idx + 1):
        d0 = pd.Timestamp(ds.iloc[idx]['date'])
        btc_ret = prices[idx] / prices[idx-1] - 1 if idx > 0 else 0.0
        # FIX: today's return was governed by PREVIOUS alloc.
        applied_alloc = prev_alloc
        is_friday = d0.dayofweek in REBAL_DOW
        is_emergency = abs(btc_ret) > EMERGENCY_THRESHOLD

        # ── Determine if we need to retrain ──
        days_since_retrain = (d0 - last_retrain_date).days if last_retrain_date else 9999
        should_retrain = False
        retrain_reason = None
        if days_since_retrain >= MIN_GAP_DAYS:
            # Scheduled retrain?
            if d0 in sched_dates or any(pd.Timestamp(s) == d0 for s in sched_dates):
                should_retrain = True
                retrain_reason = 'scheduled'
            # Trigger?
            elif trigger_fn is not None:
                state = {
                    'idx': idx, 'date': d0, 'prices': prices,
                    'dd_since_retrain': dd_since_retrain,
                    'vol_at_retrain': vol_at_retrain,
                    'recent_hits': recent_hits,
                }
                t_reason = trigger_fn(state)
                if t_reason:
                    should_retrain = True
                    retrain_reason = t_reason

        if should_retrain:
            t0 = time.time()
            reg_models, cls_models = train_at_idx(ds, idx, seed=seed)
            if reg_models is None:
                continue
            last_retrain_idx = idx
            last_retrain_date = d0
            retrain_dates.append(f"{d0.date()}({retrain_reason[:4]})")
            # reset state
            dd_since_retrain = 0.0
            peak_since_retrain = 1.0
            cum_since_retrain = 1.0
            # compute vol at retrain
            vol_at_retrain = float(pd.Series(prices[max(0,idx-30):idx]).pct_change().std() * np.sqrt(365))
            if verbose:
                print(f"  Retrain {d0.date()} ({retrain_reason}) in {time.time()-t0:.0f}s")

        # ── Generate signal (updates prev_alloc for TOMORROW) ──
        if is_friday or is_emergency:
            X = np.nan_to_num(ds.iloc[idx][FEATURES_ALL].values.astype(float).reshape(1, -1), nan=0.0)
            pred, p_up, conf = predict(reg_models, cls_models, X)
            hist = prices[:idx+1]
            s50 = pd.Series(hist).rolling(50).mean().iloc[-1]
            s200 = pd.Series(hist).rolling(200).mean().iloc[-1]
            regime = get_regime(prices[idx], s50, s200)
            new_alloc = compute_alloc(pred, conf, regime)
            prev_alloc = new_alloc  # applies from tomorrow onward
            # Track hit for ACC trigger (use 3d actual)
            future_idx = min(idx + HORIZON, len(prices) - 1)
            actual_3d = prices[future_idx] / prices[idx] - 1
            hit = 1 if np.sign(pred) == np.sign(actual_3d) and pred != 0 else 0
            recent_hits.append(hit)
            if len(recent_hits) > 8:
                recent_hits = recent_hits[-8:]

        cdi = float(rf_series.loc[d0]) if d0 in rf_series.index else 0.0
        strat = applied_alloc * btc_ret + (1 - applied_alloc) * cdi
        daily_cdi.append(cdi)
        cum_since_retrain *= (1 + strat)
        peak_since_retrain = max(peak_since_retrain, cum_since_retrain)
        dd_since_retrain = min(dd_since_retrain, cum_since_retrain / peak_since_retrain - 1)
        daily_strat.append(strat)
        daily_btc.append(btc_ret)

    # Sortino & Price 1994 (V22/V36 convention: excess over CDI, sqrt(365))
    arr = np.array(daily_strat)
    cdi_arr = np.array(daily_cdi)
    excess = arr - cdi_arr
    downside = np.minimum(excess, 0.0)
    dd_std = float(np.sqrt(np.mean(downside ** 2)))
    sortino = float(excess.mean() / (dd_std + 1e-10) * np.sqrt(365))
    sharpe = float(excess.mean() / (arr.std() + 1e-10) * np.sqrt(365)) if arr.std() > 0 else np.nan
    cum = np.cumprod(1 + arr)
    peak = np.maximum.accumulate(cum)
    dd = float((cum / peak - 1).min())
    cum_strat = float(cum[-1] - 1)
    cum_btc = float(np.cumprod(1 + np.array(daily_btc))[-1] - 1)
    n_years = (end - start).days / 365.25
    cagr = float((1 + cum_strat) ** (1 / n_years) - 1) if n_years > 0 else np.nan

    return {
        'name': name,
        'n_retrains': len(retrain_dates),
        'retrain_dates': ';'.join(retrain_dates),
        'cum_strat': cum_strat,
        'cum_btc': cum_btc,
        'cagr': cagr,
        'sortino': sortino,
        'sharpe': sharpe,
        'max_dd': dd,
    }


def triggers_factory(kind, **params):
    """Returns a trigger function based on kind."""
    if kind == 'ddt':
        threshold = params.get('threshold', -0.05)
        def trig(state):
            return 'ddt_dd{:.1%}'.format(state['dd_since_retrain']) if state['dd_since_retrain'] <= threshold else None
        return trig
    elif kind == 'vrc':
        multiplier = params.get('multiplier', 1.5)
        lookback = params.get('lookback', 30)
        def trig(state):
            if state['vol_at_retrain'] is None: return None
            prices = state['prices']; idx = state['idx']
            if idx < lookback: return None
            vol_now = float(pd.Series(prices[idx-lookback:idx]).pct_change().std() * np.sqrt(365))
            ratio = vol_now / state['vol_at_retrain']
            if ratio > multiplier or ratio < 1/multiplier:
                return f'vrc_vol_ratio{ratio:.2f}'
            return None
        return trig
    elif kind == 'acc':
        threshold = params.get('threshold', 0.5)
        min_n = params.get('min_n', 8)
        def trig(state):
            hits = state['recent_hits']
            if len(hits) >= min_n:
                acc = sum(hits) / len(hits)
                if acc < threshold:
                    return f'acc_drop{acc:.2f}'
            return None
        return trig
    elif kind == 'bmt':
        threshold = params.get('threshold', 0.10)
        def trig(state):
            prices = state['prices']; idx = state['idx']
            if idx < 1: return None
            move = prices[idx]/prices[idx-1] - 1
            if abs(move) > threshold:
                return f'bmt_move{move:+.2%}'
            return None
        return trig
    elif kind == 'hybrid_acc':
        sched_months = params.get('months', [1, 7])
        acc_trig = triggers_factory('acc', threshold=params.get('threshold', 0.5))
        def trig(state):
            return acc_trig(state)
        return trig
    elif kind == 'hybrid_ddt':
        ddt_trig = triggers_factory('ddt', threshold=params.get('threshold', -0.05))
        def trig(state):
            return ddt_trig(state)
        return trig
    else:
        raise ValueError(f"Unknown trigger: {kind}")


def make_schedule(start, end, months):
    """Generate scheduled retrain dates for given months-of-year."""
    dates = []
    y = start.year - 1
    while y <= end.year + 1:
        for m in months:
            d = pd.Timestamp(year=y, month=m, day=1)
            if d > start and d < end:
                dates.append(d)
        y += 1
    return dates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2022-01-01')
    ap.add_argument('--end', default=None)
    ap.add_argument('--strategies', default='annual,semi,triennial,quarterly,ddt5,acc,bmt,annual_acc,semi_acc,semi_ddt,semi_bmt,semi_stack,tri_acc,quart_acc')
    ap.add_argument('--seed', type=int, default=242)
    args = ap.parse_args()

    ds = pd.read_csv(DATA_DIR / 'dataset_production.csv', parse_dates=['date']).sort_values('date').reset_index(drop=True)
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else ds['date'].iloc[-1]
    full_dates = pd.date_range(start, end, freq='D')
    rf = pd.Series(build_rf_daily(full_dates), index=full_dates)

    # Trigger factories
    acc50 = triggers_factory('acc', threshold=0.5)
    ddt5 = triggers_factory('ddt', threshold=-0.05)
    ddt8 = triggers_factory('ddt', threshold=-0.08)
    bmt10 = triggers_factory('bmt', threshold=0.10)
    vrc15 = triggers_factory('vrc', multiplier=1.5)

    def stack(*trigs):
        def stacked(state):
            for t in trigs:
                r = t(state)
                if r:
                    return r
            return None
        return stacked

    sched_annual = make_schedule(start, end, [1])
    sched_semi = make_schedule(start, end, [1, 7])
    sched_tri = make_schedule(start, end, [1, 5, 9])        # 3x/yr (Jan/May/Sep)
    sched_quarter = make_schedule(start, end, [1, 4, 7, 10])

    strategies_def = {
        # Pure periodic (baseline reference)
        'annual': {'name': 'ANNUAL (1x/yr)', 'schedule': sched_annual, 'trigger': None},
        'semi': {'name': 'SEMI (2x/yr, current prod)', 'schedule': sched_semi, 'trigger': None},
        'triennial': {'name': 'TRI (3x/yr, Jan/May/Sep)', 'schedule': sched_tri, 'trigger': None},
        'quarterly': {'name': 'QUART (4x/yr)', 'schedule': sched_quarter, 'trigger': None},
        # Pure triggers (no periodic base)
        'ddt5': {'name': 'DDT5 (DD<-5% only)', 'schedule': [], 'trigger': ddt5},
        'acc': {'name': 'ACC (8wk hit<50% only)', 'schedule': [], 'trigger': acc50},
        'bmt': {'name': 'BMT (|move|>10% only)', 'schedule': [], 'trigger': bmt10},
        # Hybrids (schedule + trigger)
        'annual_acc': {'name': 'ANNUAL + ACC', 'schedule': sched_annual, 'trigger': acc50},
        'semi_acc': {'name': 'SEMI + ACC', 'schedule': sched_semi, 'trigger': acc50},
        'semi_ddt': {'name': 'SEMI + DDT5', 'schedule': sched_semi, 'trigger': ddt5},
        'semi_bmt': {'name': 'SEMI + BMT', 'schedule': sched_semi, 'trigger': bmt10},
        'semi_stack': {'name': 'SEMI + ACC + DDT5 (stacked)', 'schedule': sched_semi, 'trigger': stack(acc50, ddt5)},
        'tri_acc': {'name': 'TRI + ACC', 'schedule': sched_tri, 'trigger': acc50},
        'quart_acc': {'name': 'QUART + ACC', 'schedule': sched_quarter, 'trigger': acc50},
    }

    selected = [s.strip() for s in args.strategies.split(',')]
    print(f"Strategies: {selected}")
    print(f"Range: {start.date()} -> {end.date()}")
    print(f"K_REGIME: {K_REGIME}")

    results = []
    for key in selected:
        if key not in strategies_def:
            print(f"Skipping unknown {key}")
            continue
        r = run_strategy(ds, start, end, strategies_def[key], rf, seed=args.seed)
        if r:
            results.append(r)

    df = pd.DataFrame(results)
    df['cum_strat_%'] = (df['cum_strat']*100).round(2)
    df['cum_btc_%'] = (df['cum_btc']*100).round(2)
    df['cagr_%'] = (df['cagr']*100).round(2)
    df['max_dd_%'] = (df['max_dd']*100).round(2)
    out = OUT_DIR / 'retrain_conditional_results.csv'
    df.to_csv(out, index=False)

    print(f"\n{'='*80}")
    print(f"CONDITIONAL RETRAIN RESULTS ({start.date()} -> {end.date()})")
    print('='*80)
    display = df[['name','n_retrains','cum_strat_%','cum_btc_%','cagr_%','sortino','sharpe','max_dd_%']]
    print(display.to_string(index=False))
    print(f"\nSaved: {out}")
    print(f"\nRetrain dates per strategy:")
    for _, r in df.iterrows():
        print(f"  {r['name']}: {r['retrain_dates']}")


if __name__ == '__main__':
    main()
