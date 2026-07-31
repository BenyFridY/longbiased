"""
Parallel retrain experiment.

Runs the full set of retrain strategies x multiple seeds with process-level
parallelism. Each (strategy, seed) job is an independent backtest with walk-forward
retraining. Results are averaged across seeds for final ranking.

Parallelism: ProcessPoolExecutor with N concurrent workers. Each worker uses
a reduced thread pool (XGB_WORKERS) so total CPU does not oversubscribe.

Usage:
    python scripts/production/retrain_experiment_parallel.py
    python scripts/production/retrain_experiment_parallel.py --n-proc 4 --seeds 3
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'production'))
sys.path.insert(0, str(ROOT))

# ── Strategy definitions (must be top-level for pickling) ─────────────────────

STRATEGIES = [
    # Top 6 after v1 run: semi was clear winner, keep top performers + references.
    # Dropped (underperformed in v1): triennial, quarterly, bmt, annual_acc, semi_bmt, tri_acc, quart_acc, ddt5
    # (key, label, base_months or None, trigger_spec or None)
    ('annual',     'ANNUAL (1x/yr, lower ref)',     [1],             None),
    ('semi',       'SEMI (2x/yr, current prod)',    [1, 7],          None),
    ('acc',        'ACC (trigger only)',            [],              ('acc', 0.5)),
    ('semi_acc',   'SEMI + ACC',                    [1, 7],          ('acc', 0.5)),
    ('semi_ddt',   'SEMI + DDT',                    [1, 7],          ('ddt', -0.05)),
    ('semi_stack', 'SEMI + ACC + DDT',              [1, 7],          ('stack', [('acc', 0.5), ('ddt', -0.05)])),
]

# Transaction cost (V22/V36 convention: 5bps per unit allocation change)
COST_BPS = 0.0005


def _worker_init():
    """Limit each worker's internal threads. Must run before numpy/xgboost import."""
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')


def run_one(args):
    """Run one (strategy, seed) backtest. Called in worker process."""
    strat_key, strat_label, base_months, trigger_spec, start_str, end_str, seed, workers_per_job = args

    # Limit xgboost thread usage in this subprocess
    os.environ['OMP_NUM_THREADS'] = str(workers_per_job)
    os.environ['MKL_NUM_THREADS'] = str(workers_per_job)

    # Imports inside worker to pick up env vars
    import pandas as pd
    import numpy as np
    COST = COST_BPS
    from config import (FEATURES_37 as FEATURES_ALL,
                        K_REGIME, ALLOC_MIN, ALLOC_MAX, SIGMOID_SCALE,
                        HORIZON, REBAL_DOW, EMERGENCY_THRESHOLD)
    from generate_signal import (train_regression_ensemble,
                                  train_classifier_ensemble, get_regime)
    from src.features.macro.cdi_rates import build_rf_daily

    DATA_DIR = ROOT / 'scripts' / 'production' / 'data'
    ds = pd.read_csv(DATA_DIR / 'dataset_production.csv', parse_dates=['date']).sort_values('date').reset_index(drop=True)
    start = pd.Timestamp(start_str)
    end = pd.Timestamp(end_str)

    full_dates = pd.date_range(start, end, freq='D')
    rf = pd.Series(build_rf_daily(full_dates), index=full_dates)

    def build_targets(prices, n):
        treg = np.zeros(n); tcls = np.zeros(n)
        for i in range(n - HORIZON):
            treg[i] = (prices[i + HORIZON] - prices[i]) / prices[i]
            tcls[i] = 1.0 if prices[i + HORIZON] > prices[i] else 0.0
        return treg, tcls

    def train_at_end(end_idx):
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
        # Temporarily monkey-patch WORKERS in generate_signal to workers_per_job
        import generate_signal
        generate_signal.WORKERS = workers_per_job
        reg = train_regression_ensemble(X[train_idx], treg[train_idx], seed)
        cls = train_classifier_ensemble(X[train_idx], tcls[train_idx], seed)
        return reg, cls

    def trigger_from_spec(spec):
        if spec is None:
            return None
        kind = spec[0]
        if kind == 'ddt':
            th = spec[1]
            def trig(state):
                return 'ddt' if state['dd_since_retrain'] <= th else None
            return trig
        if kind == 'acc':
            th = spec[1]
            def trig(state):
                hits = state['recent_hits']
                if len(hits) >= 8 and sum(hits)/len(hits) < th:
                    return 'acc'
                return None
            return trig
        if kind == 'bmt':
            th = spec[1]
            def trig(state):
                p = state['prices']; i = state['idx']
                if i < 1: return None
                m = p[i]/p[i-1] - 1
                if abs(m) > th: return 'bmt'
                return None
            return trig
        if kind == 'stack':
            sub_trigs = [trigger_from_spec(s) for s in spec[1]]
            def trig(state):
                for t in sub_trigs:
                    r = t(state)
                    if r: return r
                return None
            return trig
        raise ValueError(f"Unknown trigger {kind}")

    trig_fn = trigger_from_spec(trigger_spec)

    # Build schedule
    sched = set()
    if base_months is not None and len(base_months) > 0:
        for y in range(start.year - 1, end.year + 2):
            for m in base_months:
                d = pd.Timestamp(year=y, month=m, day=1)
                if start < d < end:
                    sched.add(d)

    prices = ds['price_usd'].values
    start_idx = ds[ds['date'] >= start].index[0]
    end_idx = ds[ds['date'] <= end].index[-1]

    gap = max(HORIZON, 5)
    init_train_end = start_idx - gap
    if init_train_end < 500:
        return None
    reg_models, cls_models = train_at_end(init_train_end + 1)
    if reg_models is None:
        return None
    last_retrain_date = ds.iloc[init_train_end]['date']
    n_retrains = 1

    daily_strat = []
    daily_btc = []
    daily_cdi = []
    prev_alloc = 0.0
    recent_hits = []
    dd_since_retrain = 0.0
    peak_since_retrain = 1.0
    cum_since_retrain = 1.0
    MIN_GAP = 30

    for idx in range(start_idx, end_idx + 1):
        d0 = pd.Timestamp(ds.iloc[idx]['date'])
        btc_ret = prices[idx] / prices[idx-1] - 1 if idx > 0 else 0.0
        applied_alloc = prev_alloc
        is_friday = d0.dayofweek in REBAL_DOW
        is_emergency = abs(btc_ret) > EMERGENCY_THRESHOLD

        # Retrain logic
        days_since = (d0 - last_retrain_date).days
        should_retrain = False
        if days_since >= MIN_GAP:
            if d0 in sched:
                should_retrain = True
            elif trig_fn is not None:
                state = {'idx': idx, 'prices': prices,
                         'dd_since_retrain': dd_since_retrain,
                         'recent_hits': recent_hits}
                if trig_fn(state):
                    should_retrain = True
        if should_retrain:
            reg_new, cls_new = train_at_end(idx)
            if reg_new is not None:
                reg_models, cls_models = reg_new, cls_new
                last_retrain_date = d0
                n_retrains += 1
                dd_since_retrain = 0.0
                peak_since_retrain = 1.0
                cum_since_retrain = 1.0

        # Generate signal
        if is_friday or is_emergency:
            X = np.nan_to_num(ds.iloc[idx][FEATURES_ALL].values.astype(float).reshape(1, -1), nan=0.0)
            pred = float(np.mean([m.predict(X)[0] for m in reg_models]))
            p_up = float(np.mean([m.predict_proba(X)[0, 1] for m in cls_models]))
            conf = float(1 / (1 + np.exp(-abs(p_up - 0.5) * SIGMOID_SCALE)))
            hist = prices[:idx+1]
            s50 = pd.Series(hist).rolling(50).mean().iloc[-1]
            s200 = pd.Series(hist).rolling(200).mean().iloc[-1]
            regime = get_regime(prices[idx], s50, s200)
            new_alloc = float(np.clip(pred * K_REGIME[regime] * conf, ALLOC_MIN, ALLOC_MAX))
            prev_alloc = new_alloc
            # Hit tracking
            fi = min(idx + HORIZON, len(prices) - 1)
            actual_3d = prices[fi] / prices[idx] - 1
            hit = 1 if np.sign(pred) == np.sign(actual_3d) and pred != 0 else 0
            recent_hits.append(hit)
            if len(recent_hits) > 8:
                recent_hits = recent_hits[-8:]

        cdi = float(rf.loc[d0]) if d0 in rf.index else 0.0
        # Transaction cost on rebalance days: proportional to |alloc change|
        if (is_friday or is_emergency) and applied_alloc != prev_alloc:
            tc = abs(prev_alloc - applied_alloc) * COST
        else:
            tc = 0.0
        strat = applied_alloc * btc_ret + (1 - applied_alloc) * cdi - tc
        cum_since_retrain *= (1 + strat)
        peak_since_retrain = max(peak_since_retrain, cum_since_retrain)
        dd_since_retrain = min(dd_since_retrain, cum_since_retrain / peak_since_retrain - 1)
        daily_strat.append(strat)
        daily_btc.append(btc_ret)
        daily_cdi.append(cdi)

    strat_arr = np.array(daily_strat)
    btc_arr = np.array(daily_btc)
    cdi_arr = np.array(daily_cdi)
    # Sortino & Price 1994 (same as pipeline_v22.metrics_v22)
    ex = strat_arr - cdi_arr
    downside = np.minimum(ex, 0.0)
    dd_std = float(np.sqrt(np.mean(downside ** 2)))
    sortino = float(ex.mean() / (dd_std + 1e-10) * np.sqrt(365))
    sharpe = float(ex.mean() / (strat_arr.std() + 1e-10) * np.sqrt(365)) if strat_arr.std() > 0 else np.nan
    cum = np.cumprod(1 + strat_arr)
    peak = np.maximum.accumulate(cum)
    dd = float((cum / peak - 1).min())
    cum_strat = float(cum[-1] - 1)
    cum_btc = float(np.cumprod(1 + btc_arr)[-1] - 1)
    n_years = (end - start).days / 365.25
    cagr = float((1 + cum_strat) ** (1 / n_years) - 1)

    return {
        'strategy': strat_key,
        'label': strat_label,
        'seed': seed,
        'n_retrains': n_retrains,
        'cum_strat': cum_strat,
        'cum_btc': cum_btc,
        'cagr': cagr,
        'sortino': sortino,
        'sharpe': sharpe,
        'max_dd': dd,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2022-01-01')
    ap.add_argument('--end', default=None)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--n-proc', type=int, default=4, help='parallel processes')
    ap.add_argument('--workers-per-job', type=int, default=4, help='XGB threads per job')
    args = ap.parse_args()

    DATA_DIR = ROOT / 'scripts' / 'production' / 'data'
    OUT_DIR = ROOT / 'outputs' / 'results'
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ds = pd.read_csv(DATA_DIR / 'dataset_production.csv', parse_dates=['date']).sort_values('date').reset_index(drop=True)
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else ds['date'].iloc[-1]

    seed_base = [242, 323, 451, 787, 919, 1013]
    seeds = seed_base[:args.seeds]
    jobs = []
    for key, label, base_months, trig_spec in STRATEGIES:
        for seed in seeds:
            jobs.append((key, label, base_months, trig_spec,
                         start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'),
                         seed, args.workers_per_job))

    print(f"Total jobs: {len(jobs)}  ({len(STRATEGIES)} strategies × {len(seeds)} seeds)")
    print(f"Parallel processes: {args.n_proc}  |  XGB workers per job: {args.workers_per_job}  |  Total max threads: {args.n_proc * args.workers_per_job}")
    print(f"Range: {start.date()} -> {end.date()}")
    print(f"Seeds: {seeds}")

    results = []
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.n_proc, initializer=_worker_init) as ex:
        futures = {ex.submit(run_one, job): job for job in jobs}
        for fut in as_completed(futures):
            r = fut.result()
            done += 1
            if r is not None:
                results.append(r)
                print(f"[{time.time()-t0:.0f}s] {done}/{len(jobs)} done: {r['strategy']}/seed={r['seed']}  ret={r['cum_strat']*100:+.1f}%  S={r['sortino']:.2f}  retrains={r['n_retrains']}", flush=True)
            else:
                print(f"[{time.time()-t0:.0f}s] {done}/{len(jobs)} done: FAILED", flush=True)

    df = pd.DataFrame(results)
    raw_csv = OUT_DIR / 'retrain_parallel_raw.csv'
    df.to_csv(raw_csv, index=False)

    agg = df.groupby('strategy').agg(
        label=('label', 'first'),
        n_seeds=('seed', 'count'),
        n_retrains_avg=('n_retrains', 'mean'),
        cum_strat_mean=('cum_strat', 'mean'),
        cum_strat_std=('cum_strat', 'std'),
        cum_btc_mean=('cum_btc', 'mean'),
        cagr_mean=('cagr', 'mean'),
        sortino_mean=('sortino', 'mean'),
        sortino_std=('sortino', 'std'),
        sharpe_mean=('sharpe', 'mean'),
        max_dd_mean=('max_dd', 'mean'),
    ).reset_index()
    agg['cum_strat_%'] = (agg['cum_strat_mean']*100).round(1)
    agg['cum_strat_std_%'] = (agg['cum_strat_std']*100).round(1)
    agg['cagr_%'] = (agg['cagr_mean']*100).round(1)
    agg['max_dd_%'] = (agg['max_dd_mean']*100).round(1)
    agg['sortino'] = agg['sortino_mean'].round(2)
    agg['sortino_std'] = agg['sortino_std'].round(2)
    agg['sharpe'] = agg['sharpe_mean'].round(2)
    agg = agg.sort_values('sortino_mean', ascending=False)

    out_csv = OUT_DIR / 'retrain_parallel_agg.csv'
    agg.to_csv(out_csv, index=False)

    print(f"\n{'='*85}")
    print(f"PARALLEL EXPERIMENT RESULTS ({start.date()} -> {end.date()}, {len(seeds)} seeds averaged)")
    print('='*85)
    print(agg[['label','n_retrains_avg','cum_strat_%','cum_strat_std_%','cagr_%','sortino','sortino_std','sharpe','max_dd_%']].to_string(index=False))
    print(f"\nTotal wall clock: {time.time()-t0:.0f}s")
    print(f"Saved: {raw_csv}")
    print(f"Saved: {out_csv}")


if __name__ == '__main__':
    main()
