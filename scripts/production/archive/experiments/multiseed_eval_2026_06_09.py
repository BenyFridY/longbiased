"""Multi-seed (10) validation of the variant-grid finalists (2026-06-09).

Loads the per-seed prediction dumps (seed_preds_2026_06_09/seed_<s>.csv,
trained fresh on the current dataset) and evaluates the decision-layer
finalists PAIRED across seeds:

  conf  : sig15 | sig5 | conf1
  derisk: none | confgate (production rule)
  exec  : close | thresh | nextclose | noemerg
  curr  : hybrid | brl

Reports mean +/- std across seeds and paired per-seed deltas for the four
decisions on the table: (1) remove derisk, (2) sigmoid 15 -> 5,
(3) honest emergency execution, (4) emergency vs Fridays-only.

Run: python scripts/production/archive/experiments/multiseed_eval_2026_06_09.py
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
sv = importlib.import_module('seeds_validation_2026_04_28')
from src.features.macro.cdi_rates import build_rf_daily
from config import EMERGENCY_THRESHOLD

SEED_DIR = ROOT / 'outputs/results/seed_preds_2026_06_09'
OUT = ROOT / 'outputs/results/multiseed_eval_2026_06_09.json'

CONFS = ['sig15', 'sig5', 'conf1']
DERISKS = ['none', 'confgate']
EXECS = ['close', 'thresh', 'nextclose', 'noemerg']
CURS = ['hybrid', 'brl']


def to_wf(df, ds_px):
    """Seed dump -> wf-like frame the variant_grid functions expect."""
    wf = pd.DataFrame({
        'date': pd.to_datetime(df['date']),
        'to_date': pd.to_datetime(df['d1']),
        'prediction_3d': df['pred'].values,
        'p_up': df['p_up'].values,
        'regime': df['regime'].values,
        'btc_ret_period': df['btc_fwd'].values,
        'cdi_ret_period': df['cdi_period'].values,
    })
    dr = ds_px['price_usd'].pct_change()
    wf['daily_ret'] = wf['date'].map(dr).fillna(0.0)
    wf['is_emergency'] = wf['daily_ret'].abs() > EMERGENCY_THRESHOLD
    return wf


def main():
    files = sorted(SEED_DIR.glob('seed_*.csv'))
    if not files:
        print('no seed dumps found'); sys.exit(1)
    ds = pd.read_csv(ROOT / 'scripts/production/data/dataset_production.csv',
                     parse_dates=['date']).sort_values('date')
    ds_px = ds.set_index('date')[['price_usd']]
    fx_raw = pd.read_csv(ROOT / 'outputs/results/usd_brl_2026_06_09.csv',
                         parse_dates=['date']).set_index('date')['usdbrl']
    fx = fx_raw.reindex(ds_px.index).ffill().bfill()

    per_seed = {}   # key -> list of metric dicts (one per seed)
    for f in files:
        seed = f.stem.split('_')[1]
        df = pd.read_csv(f)
        wf = to_wf(df, ds_px)
        rf_idx = pd.date_range(wf['date'].iloc[0] - pd.Timedelta(days=10),
                               wf['to_date'].iloc[-1] + pd.Timedelta(days=10), freq='D')
        rf = pd.Series(build_rf_daily(rf_idx), index=rf_idx)
        pts = vg.build_exec_points(wf, ds_px)
        factors = {cv: vg.conf_factors(wf, cv) for cv in CONFS}
        for cv in CONFS:
            for dv in DERISKS:
                allocs, fd, fk = vg.sequential_allocs(wf, factors[cv], dv)
                for ev in EXECS:
                    a_ev = vg.carry_over_emergencies(wf, allocs) if ev == 'noemerg' else allocs
                    for cur in CURS:
                        w = vg.weekly_returns(wf, a_ev, pts[ev], ds_px, fx, cur)
                        dly, dly_cdi, days = vg.daily_returns(wf, a_ev, pts[ev], ds_px, fx, rf, cur)
                        m = sv.metrics_from_returns(w, dly, wf['cdi_ret_period'].values, dly_cdi)
                        m['y2026'] = vg.per_year(dly, days).get(2026, np.nan)
                        m['derisk_fires'] = fd
                        per_seed.setdefault(f"{cv}|{dv}|{ev}|{cur}", []).append(m)
        print(f"seed {seed} evaluated", flush=True)

    n = len(files)
    agg = {}
    for k, lst in per_seed.items():
        agg[k] = {met: (float(np.mean([r[met] for r in lst])),
                        float(np.std([r[met] for r in lst], ddof=1)) if n > 1 else 0.0)
                  for met in ['cum_ret', 'cagr', 'sortino_d', 'sortino_w',
                              'sharpe_excess_d', 'max_dd_d', 'max_dd_w', 'y2026']}
        agg[k]['derisk_fires'] = float(np.mean([r['derisk_fires'] for r in lst]))

    with open(OUT, 'w') as fjson:
        json.dump({'n_seeds': n, 'note': 'GROSS, full window 2022-01-07 -> 2026-05-29 closed',
                   'agg': agg}, fjson, indent=2)
    print(f"saved: {OUT}  (n_seeds={n})\n")

    def show(k):
        a = agg[k]
        return (f"{k:<28s} CAGR {a['cagr'][0]*100:+5.1f}±{a['cagr'][1]*100:.1f}  "
                f"So_d {a['sortino_d'][0]:5.2f}±{a['sortino_d'][1]:.2f}  "
                f"Sh_d {a['sharpe_excess_d'][0]:5.2f}  DDd {a['max_dd_d'][0]*100:6.2f}%  "
                f"DDw {a['max_dd_w'][0]*100:6.2f}%  26 {a['y2026'][0]*100:+5.1f}%")

    print("=" * 120)
    print(f"AGGREGATED ({n} seeds, GROSS, hybrid) — finalists")
    print("=" * 120)
    for k in sorted([k for k in agg if k.endswith('|hybrid')],
                    key=lambda k: -agg[k]['sortino_d'][0]):
        print(show(k))
    print()
    print("BRL finalists:")
    for k in sorted([k for k in agg if k.endswith('|brl')],
                    key=lambda k: -agg[k]['sortino_d'][0])[:8]:
        print(show(k))

    # paired per-seed deltas for the four decisions
    def paired(ka, kb, met='sortino_d'):
        da = np.array([r[met] for r in per_seed[ka]])
        db = np.array([r[met] for r in per_seed[kb]])
        d = da - db
        t = d.mean() / (d.std(ddof=1) / np.sqrt(n)) if n > 1 and d.std(ddof=1) > 0 else np.nan
        return d.mean(), d.std(ddof=1) if n > 1 else 0.0, t

    print()
    print("PAIRED PER-SEED DELTAS (A - B), hybrid:")
    comps = [
        ('remove derisk     ', 'sig15|none|close|hybrid', 'sig15|confgate|close|hybrid'),
        ('sig5 vs sig15     ', 'sig5|none|thresh|hybrid', 'sig15|none|thresh|hybrid'),
        ('thresh vs close   ', 'sig15|none|thresh|hybrid', 'sig15|none|close|hybrid'),
        ('emerg vs fridays  ', 'sig15|none|thresh|hybrid', 'sig15|none|noemerg|hybrid'),
        ('emerg vs fridays s5', 'sig5|none|thresh|hybrid', 'sig5|none|noemerg|hybrid'),
    ]
    for name, ka, kb in comps:
        for met in ['sortino_d', 'cagr', 'max_dd_d']:
            mu, sd, t = paired(ka, kb, met)
            scale = 100 if met in ('cagr', 'max_dd_d') else 1
            print(f"  {name} d{met:<12s} {mu*scale:+7.3f} ± {sd*scale:.3f}  t={t:+.1f}")
        print()


if __name__ == '__main__':
    main()
