"""
Test 4 alternative formulations of the rolling-acc derisk control.

Baseline: alloc *= 0.5 if rolling_12w_acc < 0.48 else 1.0   (current production)

Alternatives:
  C. Graduated ramp:   smooth multiplier between 0.40 and 0.55
  D. Conf-weighted:    only derisk if avg confidence was HIGH (model was wrong AND sure)
  H. Vol-conditional:  only derisk if rolling vol > median
  Threshold sweep:     test 45/48/50 with current binary rule

All run post-hoc on baseline retrain CSV (today's 248 rebals, H1 32feat).
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / 'outputs/results'
DATASET = ROOT / 'scripts/production/data/dataset_production.csv'
BASE_CSV = OUT / 'experiments_2026_04_28_baseline.csv'

ROLLING_WINDOW = 12


def metrics(strat_returns: np.ndarray, cdi_period: np.ndarray = None) -> dict:
    cum = float(np.cumprod(1 + strat_returns)[-1] - 1)
    neg = strat_returns[strat_returns < 0]
    dev = float(np.sqrt(np.mean(neg ** 2))) if len(neg) > 0 else 1e-9
    sortino = float(np.mean(strat_returns) / dev * np.sqrt(52)) if dev > 0 else 0.0
    sd = float(np.std(strat_returns, ddof=0))
    sharpe_abs = float(np.mean(strat_returns) / sd * np.sqrt(52)) if sd > 0 else 0.0
    sharpe_excess = None
    if cdi_period is not None:
        excess = strat_returns - cdi_period
        sd_e = float(np.std(excess, ddof=0))
        sharpe_excess = float(np.mean(excess) / sd_e * np.sqrt(52)) if sd_e > 0 else 0.0
    eq = np.concatenate([[1.0], np.cumprod(1 + strat_returns)])
    peak = np.maximum.accumulate(eq)
    maxdd = float(((eq - peak) / peak).min())
    return {
        'cum_ret': cum, 'sortino': sortino,
        'sharpe_abs': sharpe_abs, 'sharpe_excess': sharpe_excess,
        'max_dd_w': maxdd,
    }


def rolling_accuracy(preds: np.ndarray, btc_fwd: np.ndarray, window: int = 12) -> np.ndarray:
    """At each rebal i, looks back at last `window` rebals and computes direction acc."""
    correct = ((preds > 0) & (btc_fwd > 0)) | ((preds < 0) & (btc_fwd < 0))
    n = len(preds)
    out = np.full(n, np.nan)
    for i in range(n):
        if i >= window:
            out[i] = correct[i - window:i].mean()
    return out


def rolling_avg_conf(p_ups: np.ndarray, window: int = 12, sigmoid_scale: float = 15) -> np.ndarray:
    """Rolling avg of sigmoid confidence factor over last `window` rebals."""
    conf = 1.0 / (1.0 + np.exp(-np.abs(p_ups - 0.5) * sigmoid_scale))
    n = len(conf)
    out = np.full(n, np.nan)
    for i in range(n):
        if i >= window:
            out[i] = conf[i - window:i].mean()
    return out


def main():
    base = pd.read_csv(BASE_CSV, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    print(f"Loaded baseline: {len(base)} rebals, {base['date'].min().date()} -> {base['date'].max().date()}")

    # Reconstruct CDI per period (alloc + strat + btc_fwd known => cdi backed out)
    btc = base['btc_fwd'].values
    alloc_orig = base['alloc'].values
    strat_orig = base['strat'].values
    cdi_period = np.where(np.abs(1 - alloc_orig) > 1e-9,
                          (strat_orig - alloc_orig * btc) / (1 - alloc_orig),
                          0.0)
    cdi_period = np.where(np.isnan(cdi_period) | np.isinf(cdi_period), 0.0, cdi_period)

    preds = base['pred'].values
    p_ups = base['p_up'].values

    # Rolling stats
    roll_acc = rolling_accuracy(preds, btc, window=ROLLING_WINDOW)
    roll_conf = rolling_avg_conf(p_ups, window=ROLLING_WINDOW)

    # Rolling vol from dataset (30d)
    ds = pd.read_csv(DATASET, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    ds['ret'] = ds['price_usd'].pct_change()
    ds['vol_30d_ann'] = ds['ret'].rolling(30).std() * np.sqrt(365)
    vol_by_date = dict(zip(ds['date'], ds['vol_30d_ann']))
    rebal_vol = np.array([vol_by_date.get(d, np.nan) for d in base['date']])
    median_vol = np.nanmedian(rebal_vol)

    print(f"\nMedian 30d annualized BTC vol: {median_vol*100:.1f}%/y")

    results = []

    def apply_and_metric(label: str, mult: np.ndarray, info: dict = None):
        new_alloc = np.clip(alloc_orig * mult, 0.0, 1.0)
        new_strat = new_alloc * btc + (1 - new_alloc) * cdi_period
        m = metrics(new_strat, cdi_period)
        m['label'] = label
        m['triggers'] = int((mult < 1.0).sum())
        m['avg_alloc'] = float(new_alloc.mean())
        if info:
            m.update(info)
        results.append(m)
        return m

    print("\n" + "=" * 100)
    print(f"{'Config':<55s} {'cum':>9s} {'Sortino':>8s} {'Shp_x':>6s} {'DD_w':>7s} {'trig':>5s} {'alloc':>6s}")
    print("=" * 100)

    # ── 0. NO CONTROL (baseline) ──
    m = apply_and_metric("0. NO control (baseline H1 32f)",
                          np.ones(len(base)))
    print(f"{m['label']:<55s} {m['cum_ret']*100:+8.1f}% {m['sortino']:8.2f} "
          f"{m['sharpe_excess']:6.2f} {m['max_dd_w']*100:6.2f}% {m['triggers']:5d} {m['avg_alloc']*100:5.1f}%")

    # ── 1. CURRENT (binary 48% / x0.5) ──
    mult = np.where((roll_acc < 0.48) & (~np.isnan(roll_acc)), 0.5, 1.0)
    m = apply_and_metric("1. CURRENT: acc<48 -> x0.5", mult)
    print(f"{m['label']:<55s} {m['cum_ret']*100:+8.1f}% {m['sortino']:8.2f} "
          f"{m['sharpe_excess']:6.2f} {m['max_dd_w']*100:6.2f}% {m['triggers']:5d} {m['avg_alloc']*100:5.1f}%")

    # ── 2. THRESHOLD SWEEP ──
    print(f"\n  -- threshold sweep (binary x0.5) --")
    for thresh in [0.42, 0.45, 0.48, 0.50, 0.52]:
        mult = np.where((roll_acc < thresh) & (~np.isnan(roll_acc)), 0.5, 1.0)
        m = apply_and_metric(f"2.{int(thresh*100)}. threshold={thresh:.2f}", mult,
                              {'threshold': thresh})
        print(f"{m['label']:<55s} {m['cum_ret']*100:+8.1f}% {m['sortino']:8.2f} "
              f"{m['sharpe_excess']:6.2f} {m['max_dd_w']*100:6.2f}% {m['triggers']:5d} {m['avg_alloc']*100:5.1f}%")

    # ── 3. ALT C: GRADUATED RAMP ──
    print(f"\n  -- alt C: graduated ramp (no cliff) --")
    for low, high in [(0.40, 0.55), (0.42, 0.52), (0.45, 0.50), (0.40, 0.50)]:
        ramp_mult = np.ones(len(base))
        valid = ~np.isnan(roll_acc)
        a = roll_acc.copy()
        # Linear ramp from low to high; below low -> 0; above high -> 1
        with np.errstate(invalid='ignore'):
            ramp = np.clip((a - low) / (high - low), 0.0, 1.0)
        ramp_mult = np.where(valid, ramp, 1.0)
        m = apply_and_metric(f"3. C. ramp [{low:.2f},{high:.2f}] linear", ramp_mult,
                              {'low': low, 'high': high})
        print(f"{m['label']:<55s} {m['cum_ret']*100:+8.1f}% {m['sortino']:8.2f} "
              f"{m['sharpe_excess']:6.2f} {m['max_dd_w']*100:6.2f}% "
              f"{int((ramp_mult < 1.0).sum()):5d} {m['avg_alloc']*100:5.1f}%")

    # ── 4. ALT D: CONFIDENCE-WEIGHTED ──
    # Only derisk when acc<48% AND avg conf in last 12w was HIGH (>0.7)
    # i.e. the model was confident and wrong
    print(f"\n  -- alt D: confidence-weighted (only derisk if model was confident-wrong) --")
    for conf_thresh in [0.60, 0.70, 0.80]:
        mult = np.where(
            (roll_acc < 0.48) & (~np.isnan(roll_acc)) &
            (roll_conf > conf_thresh) & (~np.isnan(roll_conf)),
            0.5, 1.0
        )
        m = apply_and_metric(f"4. D. acc<48 + conf>{conf_thresh:.2f} -> x0.5", mult,
                              {'conf_thresh': conf_thresh})
        print(f"{m['label']:<55s} {m['cum_ret']*100:+8.1f}% {m['sortino']:8.2f} "
              f"{m['sharpe_excess']:6.2f} {m['max_dd_w']*100:6.2f}% {m['triggers']:5d} {m['avg_alloc']*100:5.1f}%")

    # ── 5. ALT H: VOL-CONDITIONAL ──
    print(f"\n  -- alt H: vol-conditional (only derisk in volatile markets) --")
    for vol_thresh in [median_vol, median_vol * 1.2, 0.60]:
        mult = np.where(
            (roll_acc < 0.48) & (~np.isnan(roll_acc)) &
            (rebal_vol > vol_thresh) & (~np.isnan(rebal_vol)),
            0.5, 1.0
        )
        m = apply_and_metric(f"5. H. acc<48 + vol>{vol_thresh*100:.0f}%/y -> x0.5", mult,
                              {'vol_thresh': vol_thresh})
        print(f"{m['label']:<55s} {m['cum_ret']*100:+8.1f}% {m['sortino']:8.2f} "
              f"{m['sharpe_excess']:6.2f} {m['max_dd_w']*100:6.2f}% {m['triggers']:5d} {m['avg_alloc']*100:5.1f}%")

    # Save
    out_path = OUT / 'acc_derisk_alternatives.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    # Pick winner
    print(f"\n{'='*100}")
    print("WINNERS BY METRIC")
    print(f"{'='*100}")
    by_sortino = sorted(results, key=lambda r: -r['sortino'])[:5]
    print(f"\n  Top 5 by Sortino:")
    for r in by_sortino:
        print(f"    {r['sortino']:6.2f}  {r['label']}  (cum={r['cum_ret']*100:+.1f}%, DD={r['max_dd_w']*100:.2f}%)")
    by_sharpe = sorted(results, key=lambda r: -r['sharpe_excess'] if r['sharpe_excess'] is not None else -99)[:5]
    print(f"\n  Top 5 by Sharpe (excess):")
    for r in by_sharpe:
        print(f"    {r['sharpe_excess']:5.2f}  {r['label']}  (cum={r['cum_ret']*100:+.1f}%, DD={r['max_dd_w']*100:.2f}%)")
    by_dd = sorted(results, key=lambda r: r['max_dd_w'])[:5]
    print(f"\n  Top 5 by DD (least negative):")
    for r in by_dd:
        print(f"    {r['max_dd_w']*100:6.2f}%  {r['label']}  (cum={r['cum_ret']*100:+.1f}%, Sortino={r['sortino']:.2f})")


if __name__ == '__main__':
    main()
