"""M1 (sigmoid=5, no derisk) official BRL numbers + BRL benchmarks (2026-06-09).

Evaluates M1 on the 10 dumped seeds in CONSISTENT BRL (BTC*USDBRL + CDI),
both on the canonical window (2022-01-07 -> 2026-04-17) and the full window,
and computes the BRL benchmarks for the comparison tables: 100% CDI,
30% BTC + 70% CDI (weekly rebal), 100% BTC HODL.

Run: python scripts/production/archive/experiments/m1_brl_canonical_2026_06_09.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/production'))
sys.path.insert(0, str(Path(__file__).parent))

import importlib
vg = importlib.import_module('variant_grid_2026_06_09')
mse = importlib.import_module('multiseed_eval_2026_06_09')
sv = importlib.import_module('seeds_validation_2026_04_28')
from src.features.macro.cdi_rates import build_rf_daily

SEED_DIR = ROOT / 'outputs/results/seed_preds_2026_06_09'
OUT = ROOT / 'outputs/results/m1_brl_canonical_2026_06_09.json'
CANON_END = pd.Timestamp('2026-04-17')


def eval_window(wf, ds_px, fx, rf, conf_variant, end=None):
    if end is not None:
        wf = wf[wf['date'] <= end].reset_index(drop=True)
    pts = vg.build_exec_points(wf, ds_px)
    factors = vg.conf_factors(wf, conf_variant)
    allocs, _, _ = vg.sequential_allocs(wf, factors, 'none')
    w = vg.weekly_returns(wf, allocs, pts['close'], ds_px, fx, 'brl')
    dly, dly_cdi, days = vg.daily_returns(wf, allocs, pts['close'], ds_px, fx, rf, 'brl')
    m = sv.metrics_from_returns(w, dly, wf['cdi_ret_period'].values, dly_cdi)
    m['per_year'] = vg.per_year(dly, days)
    return m


def benchmarks_brl(ds_px, fx, rf, start, end):
    px = (ds_px['price_usd'] * fx).loc[start:end]
    btc_d = px.pct_change().fillna(0.0).values
    cdi_d = np.array([float(rf.get(d, 0.0)) for d in px.index])
    out = {}
    # 100% CDI
    cum = float(np.prod(1 + cdi_d) - 1)
    out['CDI'] = {'cum': cum, 'cagr': float((1 + cum) ** (365 / len(cdi_d)) - 1),
                  'sortino_d': float('inf'), 'max_dd_d': 0.0}
    # 100% BTC HODL (BRL)
    neg = btc_d[btc_d < 0]
    dev = float(np.sqrt(np.mean(neg ** 2)))
    eq = np.cumprod(1 + btc_d)
    peak = np.maximum.accumulate(eq)
    cum = float(eq[-1] - 1)
    out['BTC_HODL'] = {'cum': cum, 'cagr': float((1 + cum) ** (365 / len(btc_d)) - 1),
                       'sortino_d': float(np.mean(btc_d) / dev * np.sqrt(365)),
                       'max_dd_d': float(((eq - peak) / peak).min())}
    # 30% BTC + 70% CDI, weekly rebal (Fridays)
    dows = pd.DatetimeIndex(px.index).dayofweek
    w = 0.30
    rets = []
    bal_btc, bal_cdi = 0.30, 0.70
    for i in range(len(btc_d)):
        bal_btc *= (1 + btc_d[i])
        bal_cdi *= (1 + cdi_d[i])
        tot = bal_btc + bal_cdi
        rets.append(tot - (rets and (1 + np.array(rets)).prod() or 1.0))
        if dows[i] == 4:  # Friday: rebalance to 30/70
            bal_btc, bal_cdi = tot * w, tot * (1 - w)
    eq = 1 + np.array([])  # recompute cleanly below
    # simpler: daily portfolio return series
    bal_btc, bal_cdi = 0.30, 0.70
    drets = []
    for i in range(len(btc_d)):
        prev = bal_btc + bal_cdi
        bal_btc *= (1 + btc_d[i])
        bal_cdi *= (1 + cdi_d[i])
        tot = bal_btc + bal_cdi
        drets.append(tot / prev - 1)
        if dows[i] == 4:
            bal_btc, bal_cdi = tot * w, tot * (1 - w)
    drets = np.array(drets)
    neg = drets[drets < 0]
    dev = float(np.sqrt(np.mean(neg ** 2)))
    eq = np.cumprod(1 + drets)
    peak = np.maximum.accumulate(eq)
    cum = float(eq[-1] - 1)
    out['BTC30_CDI70'] = {'cum': cum, 'cagr': float((1 + cum) ** (365 / len(drets)) - 1),
                          'sortino_d': float(np.mean(drets) / dev * np.sqrt(365)),
                          'max_dd_d': float(((eq - peak) / peak).min())}
    return out


def main():
    files = sorted(SEED_DIR.glob('seed_*.csv'))
    ds = pd.read_csv(ROOT / 'scripts/production/data/dataset_production.csv',
                     parse_dates=['date']).sort_values('date')
    ds_px = ds.set_index('date')[['price_usd']]
    fx_raw = pd.read_csv(ROOT / 'outputs/results/usd_brl_2026_06_09.csv',
                         parse_dates=['date']).set_index('date')['usdbrl']
    fx = fx_raw.reindex(ds_px.index).ffill().bfill()

    res = {'canon': {'sig5': [], 'sig15': []}, 'full': {'sig5': [], 'sig15': []}}
    for f in files:
        df = pd.read_csv(f)
        wf = mse.to_wf(df, ds_px)
        rf_idx = pd.date_range(wf['date'].iloc[0] - pd.Timedelta(days=10),
                               wf['to_date'].iloc[-1] + pd.Timedelta(days=10), freq='D')
        rf = pd.Series(build_rf_daily(rf_idx), index=rf_idx)
        for cv in ['sig5', 'sig15']:
            res['canon'][cv].append(eval_window(wf, ds_px, fx, rf, cv, end=CANON_END))
            res['full'][cv].append(eval_window(wf, ds_px, fx, rf, cv))
        print(f"{f.stem} done", flush=True)

    def agg(lst, k):
        v = np.array([r[k] for r in lst])
        return float(v.mean()), float(v.std(ddof=1))

    summary = {}
    for win in ['canon', 'full']:
        for cv in ['sig5', 'sig15']:
            lst = res[win][cv]
            summary[f'{win}|{cv}'] = {k: agg(lst, k) for k in
                ['cum_ret', 'cagr', 'sortino_d', 'sortino_w', 'sharpe_excess_d',
                 'sharpe_excess_w', 'max_dd_d', 'max_dd_w']}
            summary[f'{win}|{cv}']['per_year'] = {
                y: float(np.mean([r['per_year'].get(y, np.nan) for r in lst]))
                for y in [2022, 2023, 2024, 2025, 2026]}

    start = pd.Timestamp('2022-01-07')
    rf_idx = pd.date_range(start - pd.Timedelta(days=10),
                           ds_px.index[-1] + pd.Timedelta(days=10), freq='D')
    rf = pd.Series(build_rf_daily(rf_idx), index=rf_idx)
    bm_canon = benchmarks_brl(ds_px, fx, rf, start, CANON_END)
    bm_full = benchmarks_brl(ds_px, fx, rf, start, ds_px.index[-1])

    with open(OUT, 'w') as fj:
        json.dump({'note': 'BRL consistent (BTC*USDBRL + CDI), GROSS, exec=close, derisk=none',
                   'summary': summary, 'benchmarks_canon': bm_canon,
                   'benchmarks_full': bm_full}, fj, indent=2, default=str)
    print(f"saved: {OUT}\n")

    for key, s in summary.items():
        py = '  '.join(f"{y}:{v*100:+5.1f}%" for y, v in s['per_year'].items() if not np.isnan(v))
        print(f"{key:<14s} CAGR {s['cagr'][0]*100:+5.1f}±{s['cagr'][1]*100:.1f}  "
              f"So_d {s['sortino_d'][0]:.2f}±{s['sortino_d'][1]:.2f}  "
              f"So_w {s['sortino_w'][0]:.2f}  Sh_d {s['sharpe_excess_d'][0]:.2f}  "
              f"Sh_w {s['sharpe_excess_w'][0]:.2f}  DDd {s['max_dd_d'][0]*100:.2f}%  "
              f"DDw {s['max_dd_w'][0]*100:.2f}%")
        print(f"               {py}")
    print("\nBenchmarks BRL (canonical window):")
    for k, v in bm_canon.items():
        print(f"  {k:<12s} cum {v['cum']*100:+7.1f}%  CAGR {v['cagr']*100:+5.1f}%  "
              f"So_d {v['sortino_d']:.2f}  DDd {v['max_dd_d']*100:.1f}%")
    print("Benchmarks BRL (full window to last data):")
    for k, v in bm_full.items():
        print(f"  {k:<12s} cum {v['cum']*100:+7.1f}%  CAGR {v['cagr']*100:+5.1f}%  "
              f"So_d {v['sortino_d']:.2f}  DDd {v['max_dd_d']*100:.1f}%")


if __name__ == '__main__':
    main()
