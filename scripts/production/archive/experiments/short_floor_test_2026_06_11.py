"""Short-floor test on the final M1 config (2026-06-11).

Question: would allowing SHORT positions improve the model? Evaluates
allocation floors {0 (baseline), -10%, -25%, -50%, -100%} PAIRED across the
10 stored seed prediction dumps (seed_preds_2026_06_09/) — decision-layer
only, no retraining, everything else identical to the final M1 config
(sig5 confidence head, derisk none, exec close, kill switch on).

Short mechanics (deliberately OPTIMISTIC — if short loses even here, the
conclusion is robust):
  alloc = clip(pred * K_regime * conf, floor, 1)   -> negative alloc = short BTC
  weekly ret = a*btc + (1-a)*cdi                   -> short proceeds earn CDI
  zero borrow/funding cost
  kill switch (DD<=-12%) clips |alloc| to 15% (symmetric extension)

Architectural caveat reported, not fixed: K_regime (60/30/15) was designed
for longs — shorts come out LARGEST in BULL (K=60) and smallest in BEAR.

Run: python scripts/production/archive/experiments/short_floor_test_2026_06_11.py
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
me = importlib.import_module('multiseed_eval_2026_06_09')
sv = importlib.import_module('seeds_validation_2026_04_28')
from src.features.macro.cdi_rates import build_rf_daily

SEED_DIR = ROOT / 'outputs/results/seed_preds_2026_06_09'
OUT = ROOT / 'outputs/results/short_floor_test_2026_06_11.json'

FLOORS = [0.0, -0.10, -0.15, -0.25, -0.50, -1.00]
CURS = ['brl', 'hybrid']
CONF = 'sig5'  # final M1 head


def sequential_allocs_floor(wf, factors, floor):
    """vg.sequential_allocs with a negative floor, derisk none.

    Kill switch extended symmetrically: under DD<=-12%, |alloc| capped at 15%.
    """
    pred = wf['prediction_3d'].values
    btc_fwd = wf['btc_ret_period'].values
    cdi = wf['cdi_ret_period'].values
    K = wf['regime'].map(vg.K_H1).values
    raw = np.clip(pred * K * factors, floor, 1.0)
    n = len(wf)
    allocs = np.empty(n)
    weekly_closed = []
    fires_k = 0
    for i in range(n):
        a = raw[i]
        closed = weekly_closed[:max(0, i - 1)]      # j <= i-2
        if len(closed) >= 2:
            cum = np.cumprod(1 + np.array(closed))
            dd = cum[-1] / cum.max() - 1
            if dd <= vg.KILL_DD:
                a = float(np.clip(a, -vg.KILL_FLOOR, vg.KILL_FLOOR))
                fires_k += 1
        allocs[i] = a
        wr = a * btc_fwd[i] + (1 - a) * cdi[i] if not np.isnan(btc_fwd[i]) else 0.0
        weekly_closed.append(wr)
    return allocs, fires_k


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

    per_seed = {}    # key floor|cur -> list of metric dicts
    diag = {}        # floor -> list of per-seed diagnostics
    for f in files:
        df = pd.read_csv(f)
        wf = me.to_wf(df, ds_px)
        rf_idx = pd.date_range(wf['date'].iloc[0] - pd.Timedelta(days=10),
                               wf['to_date'].iloc[-1] + pd.Timedelta(days=10), freq='D')
        rf = pd.Series(build_rf_daily(rf_idx), index=rf_idx)
        pts = vg.build_exec_points(wf, ds_px)
        factors = vg.conf_factors(wf, CONF)
        for floor in FLOORS:
            allocs, fk = sequential_allocs_floor(wf, factors, floor)
            short = allocs < 0
            diag.setdefault(floor, []).append({
                'pct_weeks_short': float(short.mean()),
                'avg_short_when_short': float(allocs[short].mean()) if short.any() else 0.0,
                'min_alloc': float(allocs.min()),
                'kill_fires': fk,
            })
            for cur in CURS:
                w = vg.weekly_returns(wf, allocs, pts['close'], ds_px, fx, cur)
                dly, dly_cdi, days = vg.daily_returns(wf, allocs, pts['close'], ds_px, fx, rf, cur)
                m = sv.metrics_from_returns(w, dly, wf['cdi_ret_period'].values, dly_cdi)
                m['y2026'] = vg.per_year(dly, days).get(2026, np.nan)
                per_seed.setdefault(f"{floor}|{cur}", []).append(m)
        print(f"{f.stem} evaluated", flush=True)

    n = len(files)
    mets = ['cagr', 'sortino_d', 'sharpe_excess_d', 'max_dd_d', 'max_dd_w', 'y2026']
    agg = {k: {met: (float(np.mean([r[met] for r in lst])),
                     float(np.std([r[met] for r in lst], ddof=1)))
               for met in mets}
           for k, lst in per_seed.items()}
    diag_agg = {str(fl): {k: float(np.mean([d[k] for d in lst])) for k in lst[0]}
                for fl, lst in diag.items()}

    def paired(ka, kb, met):
        d = (np.array([r[met] for r in per_seed[ka]])
             - np.array([r[met] for r in per_seed[kb]]))
        t = d.mean() / (d.std(ddof=1) / np.sqrt(n)) if d.std(ddof=1) > 0 else np.nan
        return float(d.mean()), float(d.std(ddof=1)), float(t)

    deltas = {}
    for floor in FLOORS[1:]:
        for cur in CURS:
            deltas[f"{floor}|{cur}"] = {
                met: paired(f"{floor}|{cur}", f"0.0|{cur}", met)
                for met in ['sortino_d', 'cagr', 'max_dd_d', 'y2026']}

    with open(OUT, 'w') as fjson:
        json.dump({'n_seeds': n,
                   'note': 'GROSS; sig5|none|close (final M1); short leg OPTIMISTIC '
                           '(proceeds earn CDI, zero borrow/funding); kill switch '
                           'symmetric |a|<=15%; paired on seed_preds_2026_06_09',
                   'agg': agg, 'diag': diag_agg, 'paired_deltas_vs_long_only': deltas},
                  fjson, indent=2)
    print(f"saved: {OUT}  (n_seeds={n})\n")

    for cur in CURS:
        print("=" * 112)
        print(f"{cur.upper()} — floor sweep (mean ± std, {n} seeds, GROSS)")
        print("=" * 112)
        for floor in FLOORS:
            a = agg[f"{floor}|{cur}"]
            d = diag_agg[str(floor)]
            tag = 'LONG-ONLY (atual)' if floor == 0.0 else f"floor {floor:+.0%}"
            print(f"{tag:<20s} CAGR {a['cagr'][0]*100:+5.1f}±{a['cagr'][1]*100:.1f}  "
                  f"So_d {a['sortino_d'][0]:5.2f}±{a['sortino_d'][1]:.2f}  "
                  f"Sh_d {a['sharpe_excess_d'][0]:5.2f}  "
                  f"DDd {a['max_dd_d'][0]*100:6.2f}%  DDw {a['max_dd_w'][0]*100:6.2f}%  "
                  f"26 {a['y2026'][0]*100:+5.1f}%  "
                  f"short {d['pct_weeks_short']*100:4.1f}% sem "
                  f"(média {d['avg_short_when_short']*100:+5.1f}%)")
        print()

    print("PAIRED DELTAS vs long-only (mean ± std, t-stat), BRL:")
    for floor in FLOORS[1:]:
        dd = deltas[f"{floor}|brl"]
        parts = []
        for met in ['sortino_d', 'cagr', 'max_dd_d', 'y2026']:
            mu, sd, t = dd[met]
            scale = 100 if met != 'sortino_d' else 1
            parts.append(f"d{met} {mu*scale:+.2f}±{sd*scale:.2f} (t={t:+.1f})")
        print(f"  floor {floor:+.0%}: " + '  '.join(parts))


if __name__ == '__main__':
    main()
