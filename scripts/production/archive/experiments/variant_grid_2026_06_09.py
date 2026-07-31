"""Variant grid over the stored canonical walkforward (2026-06-09).

Evaluates decision-layer variants PAIRED on the same stored predictions
(outputs/results/walkforward_backtest.csv, 255 rebals 2022-01-07 -> 2026-06-05):

  exec      : how emergency rebalances execute
              close     = decide and trade at the same close (current backtest)
              thresh    = trade intraday at the +/-8% threshold price (live intent)
              nextclose = trade at the next day's close (1-day-lag pessimistic)
  derisk    : none | plain48 (acc12<48% -> x0.5) | confgate (production rule:
              acc12<48% AND avg conf12>0.80 -> x0.5). Kill switch (-12% -> cap
              15%) applied in ALL variants. Decision-time windows (j <= i-2,
              matching the audited production lag).
  conf head : sig15 (current) | sig5 | conf1 (head retired) | signed15
              (sigmoid((p_up-0.5)*sign(pred)*15), dampens disagreement) |
              recal15 (expanding OOS isotonic recalibration of p_up, min 52 obs)
  currency  : hybrid (BTC in USD + CDI BRL — current convention) | brl
              (BTC*USDBRL + CDI BRL — what a BRL investor actually sees)

All GROSS (canonical convention; 4 bps ~ -0.6pp CAGR uniform across variants).
Metrics use seeds_validation_2026_04_28.metrics_from_returns conventions.
Calibration gate: (sig15, none, close, hybrid) must reproduce the stored
allocation and weekly strat returns.

Run: python scripts/production/archive/experiments/variant_grid_2026_06_09.py
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
sv = importlib.import_module('seeds_validation_2026_04_28')
from src.features.macro.cdi_rates import build_rf_daily

WF = ROOT / 'outputs/results/walkforward_backtest.csv'
DATASET = ROOT / 'scripts/production/data/dataset_production.csv'
FX = ROOT / 'outputs/results/usd_brl_2026_06_09.csv'
OUT = ROOT / 'outputs/results/variant_grid_2026_06_09.json'

K_H1 = {'BULL': 60, 'MILD': 30, 'BEAR': 15}
SIG_CURRENT = 15
KILL_DD, KILL_FLOOR = -0.12, 0.15
ACC_W, ACC_THR, ACC_MULT, CONF_THR = 12, 0.48, 0.5, 0.80
EMERG_THR = 0.08

CONF_VARIANTS = ['sig15', 'sig5', 'conf1', 'signed15', 'recal15']
DERISK_VARIANTS = ['none', 'confgate', 'plain48']
EXEC_VARIANTS = ['close', 'thresh', 'nextclose', 'noemerg']
CURRENCIES = ['hybrid', 'brl']


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def conf_factors(wf, variant):
    p = wf['p_up'].values
    pred = wf['prediction_3d'].values
    if variant == 'sig15':
        return sigmoid(np.abs(p - 0.5) * 15)
    if variant == 'sig5':
        return sigmoid(np.abs(p - 0.5) * 5)
    if variant == 'conf1':
        return np.ones(len(p))
    if variant == 'signed15':
        s = np.where(pred >= 0, 1.0, -1.0)
        return sigmoid((p - 0.5) * s * 15)
    if variant == 'recal15':
        from sklearn.isotonic import IsotonicRegression
        y = (wf['btc_ret_period'].values > 0).astype(float)
        out = np.empty(len(p))
        for i in range(len(p)):
            past = slice(0, max(0, i - 1))  # j <= i-2 (closed at decision time)
            xp, yp = p[past], y[past]
            ok = ~np.isnan(yp)
            if ok.sum() >= 52:
                iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
                iso.fit(xp[ok], yp[ok])
                p_cal = float(iso.predict([p[i]])[0])
            else:
                p_cal = p[i]
            out[i] = sigmoid(abs(p_cal - 0.5) * 15)
        return out
    raise ValueError(variant)


def sequential_allocs(wf, factors, derisk):
    """Decision-time sequential pass: derisk + kill switch (windows j <= i-2)."""
    pred = wf['prediction_3d'].values
    btc_fwd = wf['btc_ret_period'].values
    cdi = wf['cdi_ret_period'].values
    K = wf['regime'].map(K_H1).values
    raw = np.clip(pred * K * factors, 0.0, 1.0)
    correct = ((pred > 0) & (btc_fwd > 0)) | ((pred < 0) & (btc_fwd < 0))
    n = len(wf)
    allocs = np.empty(n)
    weekly_closed = []   # close-convention weekly strat returns, by rebal index
    fires_d, fires_k = 0, 0
    for i in range(n):
        a = raw[i]
        closed = weekly_closed[:max(0, i - 1)]      # j <= i-2
        if len(closed) >= 2:
            cum = np.cumprod(1 + np.array(closed))
            dd = cum[-1] / cum.max() - 1
            if dd <= KILL_DD:
                a = min(a, KILL_FLOOR)
                fires_k += 1
        if derisk != 'none' and i - 1 >= ACC_W:
            w = slice(i - 1 - ACC_W, i - 1)         # last 12 closed (j <= i-2)
            acc = float(correct[w].mean())
            if acc < ACC_THR:
                if derisk == 'plain48':
                    a *= ACC_MULT
                    fires_d += 1
                elif derisk == 'confgate' and float(factors[w].mean()) > CONF_THR:
                    a *= ACC_MULT
                    fires_d += 1
        allocs[i] = a
        wr = a * btc_fwd[i] + (1 - a) * cdi[i] if not np.isnan(btc_fwd[i]) else 0.0
        weekly_closed.append(wr)
    return allocs, fires_d, fires_k


def build_exec_points(wf, ds_px):
    """Per rebal: exec date + USD exec price for each exec convention.

    ds_px: DataFrame indexed by date with price_usd (daily, ~calendar).
    Returns dict conv -> list of (exec_date, exec_price_usd, intraday_flag).
    """
    dates = list(ds_px.index)
    pos = {d: k for k, d in enumerate(dates)}
    out = {c: [] for c in EXEC_VARIANTS}
    for _, r in wf.iterrows():
        d0 = r['date']
        k = pos[d0]
        close0 = float(ds_px['price_usd'].iloc[k])
        out['close'].append((d0, close0, False))
        out['noemerg'].append((d0, close0, False))
        if bool(r['is_emergency']):
            sign = 1.0 if r['daily_ret'] > 0 else -1.0
            prev_close = float(ds_px['price_usd'].iloc[k - 1])
            p_thr = prev_close * (1 + sign * EMERG_THR)
            out['thresh'].append((d0, p_thr, True))
            if k + 1 < len(dates):
                out['nextclose'].append((dates[k + 1], float(ds_px['price_usd'].iloc[k + 1]), False))
            else:
                out['nextclose'].append((d0, close0, False))
        else:
            out['thresh'].append((d0, close0, False))
            out['nextclose'].append((d0, close0, False))
    return out


def carry_over_emergencies(wf, allocs):
    """'noemerg' exec: emergency rebals don't trade — carry the prior alloc."""
    out = allocs.copy()
    is_em = wf['is_emergency'].values.astype(bool)
    for i in range(len(out)):
        if is_em[i]:
            out[i] = out[i - 1] if i > 0 else 0.0
    return out


def weekly_returns(wf, allocs, points, ds_px, fx, currency):
    """Exec-adjusted weekly strat returns: chain of exec-moment price ratios."""
    n = len(wf)
    cdi = wf['cdi_ret_period'].values
    # exit of rebal i = exec moment of rebal i+1; last exits at stored to_date close
    last_exit_date = wf['to_date'].iloc[-1]
    if last_exit_date in ds_px.index:
        last_px = float(ds_px.loc[last_exit_date, 'price_usd'])
    else:
        last_px = float(ds_px['price_usd'].iloc[-1])
        last_exit_date = ds_px.index[-1]
    rets = np.empty(n)
    for i in range(n):
        d_in, px_in, _ = points[i]
        if i + 1 < n:
            d_out, px_out, _ = points[i + 1]
        else:
            d_out, px_out = last_exit_date, last_px
        if currency == 'brl':
            px_in = px_in * float(fx.loc[d_in])
            px_out = px_out * float(fx.loc[d_out])
        btc_r = px_out / px_in - 1 if px_in > 0 else 0.0
        rets[i] = allocs[i] * btc_r + (1 - allocs[i]) * cdi[i]
    return rets


def daily_returns(wf, allocs, points, ds_px, fx, rf, currency):
    """Daily MtM strat returns with exec-convention transitions."""
    px = ds_px['price_usd'].copy()
    if currency == 'brl':
        px = px * fx
    btc_d = px.pct_change()
    start = wf['date'].iloc[0]
    last_exit = wf['to_date'].iloc[-1]
    days = [d for d in ds_px.index if start < d <= min(last_exit, ds_px.index[-1])]
    # transitions sorted by (exec_date, rebal idx); same-date later idx wins
    trans = sorted(range(len(points)), key=lambda i: (points[i][0], i))
    emerg_by_date = {points[i][0]: i for i in range(len(points))
                     if points[i][2]}  # intraday (thresh) rebals by exec date
    fx_d = fx.pct_change() if currency == 'brl' else None
    out_s, out_c = [], []
    ptr, a = 0, 0.0
    for d in days:
        # allocation effective for day d: last transition with exec_date < d
        while ptr < len(trans) and points[trans[ptr]][0] < d:
            a = float(allocs[trans[ptr]])
            ptr += 1
        b = btc_d.get(d, 0.0)
        b = 0.0 if pd.isna(b) else float(b)
        c = float(rf.get(d, 0.0))
        if d in emerg_by_date:
            i = emerg_by_date[d]
            sign = 1.0 if wf['daily_ret'].iloc[i] > 0 else -1.0
            r1 = sign * EMERG_THR
            if currency == 'brl':
                f = fx_d.get(d, 0.0)
                f = 0.0 if pd.isna(f) else float(f)
                r1 = (1 + r1) * (1 + f) - 1   # FX move assigned to first leg
            r2 = (1 + b) / (1 + r1) - 1
            a_new = float(allocs[i])
            ret = (1 + a * r1) * (1 + a_new * r2) - 1 + (1 - a_new) * c
        else:
            ret = a * b + (1 - a) * c
        out_s.append(ret)
        out_c.append(c)
    return np.array(out_s), np.array(out_c), days


def per_year(daily, days):
    out = {}
    yr = pd.Series(daily, index=pd.DatetimeIndex(days))
    for y, g in yr.groupby(yr.index.year):
        out[int(y)] = float(np.prod(1 + g.values) - 1)
    return out


def main():
    wf = pd.read_csv(WF, parse_dates=['date', 'to_date'])
    ds = pd.read_csv(DATASET, parse_dates=['date']).sort_values('date')
    ds_px = ds.set_index('date')[['price_usd']]
    fx_raw = pd.read_csv(FX, parse_dates=['date']).set_index('date')['usdbrl']
    fx = fx_raw.reindex(ds_px.index).ffill().bfill()
    last_exit = wf['to_date'].iloc[-1]
    rf_idx = pd.date_range(wf['date'].iloc[0] - pd.Timedelta(days=10),
                           max(last_exit, ds_px.index[-1]) + pd.Timedelta(days=10), freq='D')
    rf = pd.Series(build_rf_daily(rf_idx), index=rf_idx)

    print(f"wf: {len(wf)} rebals {wf['date'].iloc[0].date()} -> {wf['date'].iloc[-1].date()} "
          f"| emergencies {int(wf['is_emergency'].sum())} | GROSS")

    # ── calibration gate ─────────────────────────────────────────
    f15 = conf_factors(wf, 'sig15')
    raw15 = np.clip(wf['prediction_3d'].values * wf['regime'].map(K_H1).values * f15, 0, 1)
    gate_alloc = float(np.max(np.abs(raw15 - wf['allocation'].values)))
    pts = build_exec_points(wf, ds_px)
    w_base = weekly_returns(wf, raw15, pts['close'], ds_px, fx, 'hybrid')
    stored_w = wf['strat_ret_period'].values
    ok = ~np.isnan(stored_w)
    gate_weekly = float(np.max(np.abs(w_base[ok] - stored_w[ok])))
    cum_mine = float(np.prod(1 + w_base[ok]) - 1)
    cum_stored = float(wf['cum_strat'].iloc[-1])
    print(f"GATE alloc max|diff|={gate_alloc:.4f}  weekly max|diff|={gate_weekly:.5f}  "
          f"cum mine {cum_mine*100:+.1f}% vs stored {cum_stored*100:+.1f}%")
    if gate_alloc > 0.01 or gate_weekly > 0.01:
        print("GATE FAILED — aborting"); sys.exit(1)

    factors = {cv: conf_factors(wf, cv) for cv in CONF_VARIANTS}
    results = {}
    for cv in CONF_VARIANTS:
        for dv in DERISK_VARIANTS:
            allocs, fd, fk = sequential_allocs(wf, factors[cv], dv)
            for ev in EXEC_VARIANTS:
                a_ev = carry_over_emergencies(wf, allocs) if ev == 'noemerg' else allocs
                for cur in CURRENCIES:
                    w = weekly_returns(wf, a_ev, pts[ev], ds_px, fx, cur)
                    dly, dly_cdi, days = daily_returns(wf, a_ev, pts[ev], ds_px, fx, rf, cur)
                    m = sv.metrics_from_returns(w[ok], dly, wf['cdi_ret_period'].values[ok], dly_cdi)
                    m['per_year'] = per_year(dly, days)
                    m['derisk_fires'] = fd
                    m['kill_fires'] = fk
                    m['avg_alloc'] = float(np.mean(allocs))
                    results[f"{cv}|{dv}|{ev}|{cur}"] = m

    with open(OUT, 'w') as f:
        json.dump({'note': 'GROSS, paired on stored canonical walkforward (single seed); '
                           'kill switch always on; windows j<=i-2',
                   'results': results}, f, indent=2)
    print(f"saved: {OUT}\n")

    def row(key):
        m = results[key]
        py = m['per_year']
        return (f"{key:<38s} cum {m['cum_ret']*100:+7.1f}%  CAGR {m['cagr']*100:+5.1f}%  "
                f"So_d {m['sortino_d']:5.2f}  Sh_d {m['sharpe_excess_d']:5.2f}  "
                f"DDd {m['max_dd_d']*100:6.2f}%  DDw {m['max_dd_w']*100:6.2f}%  "
                f"26: {py.get(2026, float('nan'))*100:+5.1f}%  fires {m['derisk_fires']}")

    print("=" * 140)
    print("HYBRID (current convention) — all variants sorted by Sortino daily")
    print("=" * 140)
    keys = [k for k in results if k.endswith('|hybrid')]
    for k in sorted(keys, key=lambda k: -results[k]['sortino_d']):
        print(row(k))

    print()
    print("=" * 140)
    print("BRL (BTC*FX + CDI) — same ordering")
    print("=" * 140)
    keys = [k for k in results if k.endswith('|brl')]
    for k in sorted(keys, key=lambda k: -results[k]['sortino_d']):
        print(row(k))

    print()
    print("EMERGENCY A/B under each execution convention (hybrid, sig15/none and sig5/none):")
    for cv in ['sig15', 'sig5']:
        for ev in EXEC_VARIANTS:
            print(row(f"{cv}|none|{ev}|hybrid"))

    print()
    print("PER-YEAR (hybrid) for key configs:")
    for k in ['sig15|confgate|close|hybrid', 'sig15|none|close|hybrid',
              'sig15|none|thresh|hybrid', 'sig15|none|nextclose|hybrid',
              'sig15|none|noemerg|hybrid', 'sig5|none|noemerg|hybrid',
              'sig5|none|close|hybrid', 'conf1|none|close|hybrid',
              'signed15|none|close|hybrid', 'recal15|none|close|hybrid']:
        py = results[k]['per_year']
        yrs = '  '.join(f"{y}: {v*100:+6.1f}%" for y, v in sorted(py.items()))
        print(f"  {k:<38s} {yrs}")


if __name__ == '__main__':
    main()
