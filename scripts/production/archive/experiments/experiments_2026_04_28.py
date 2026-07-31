"""
Post-hoc experiments 2026-04-28: sigmoid sweep + vol regime overlay.

Reuses 4.3y walk-forward predictions (horizon_ablation_4y.csv, H=3 variant)
to test alternative sizing rules WITHOUT retraining. Baseline H1 numbers
should reproduce OVERFIT_TESTS_2026-04-22.md test 1 (Sortino 7.00).

Run:
    python scripts/production/archive/experiments/experiments_2026_04_28.py
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
PRED_PATH = OUT / 'horizon_ablation_4y.csv'

K_H1 = {'BULL': 60, 'MILD': 30, 'BEAR': 15}


def metrics(strat_returns: np.ndarray) -> dict:
    """Weekly-rebal metrics matching OVERFIT_TESTS baseline."""
    if len(strat_returns) == 0:
        return {}
    cum = np.cumprod(1 + strat_returns) - 1
    final = float(cum[-1])
    # Sortino: mean / downside dev (negatives only). Weekly periodicity ~ 52/y.
    neg = strat_returns[strat_returns < 0]
    dd_dev = float(np.sqrt(np.mean(neg ** 2))) if len(neg) > 0 else 1e-9
    sortino = float(np.mean(strat_returns) / dd_dev * np.sqrt(52)) if dd_dev > 0 else 0.0
    # Sharpe
    sd = float(np.std(strat_returns))
    sharpe = float(np.mean(strat_returns) / sd * np.sqrt(52)) if sd > 0 else 0.0
    # Max DD on weekly equity curve (rebal-to-rebal)
    eq = np.concatenate([[1.0], np.cumprod(1 + strat_returns)])
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    maxdd = float(dd.min())
    return {
        'cum_ret': final,
        'sortino': sortino,
        'sharpe': sharpe,
        'max_dd': maxdd,
        'n_rebals': len(strat_returns),
        'avg_alloc': None,  # filled by caller
    }


def apply_sizing(preds: np.ndarray, p_ups: np.ndarray, regimes: np.ndarray,
                 K_map: dict, sigmoid_scale: float,
                 alloc_floor: float = 0.0, alloc_ceil: float = 1.0) -> np.ndarray:
    """Reproduce production sizing: alloc = clip(pred*K[regime]*sigmoid, [floor, ceil])."""
    K_arr = np.array([K_map[r] for r in regimes])
    conf = 1.0 / (1.0 + np.exp(-np.abs(p_ups - 0.5) * sigmoid_scale))
    alloc = preds * K_arr * conf
    return np.clip(alloc, alloc_floor, alloc_ceil)


def compute_strat_returns(df: pd.DataFrame, alloc: np.ndarray) -> np.ndarray:
    """strat_ret = alloc * btc_fwd + (1 - alloc) * cdi_period_return."""
    btc = df['btc_fwd'].values
    if 'strat' in df.columns and 'alloc' in df.columns:
        # Back out cdi from existing strat row (strat = alloc*btc + (1-alloc)*cdi)
        # cdi = (strat - alloc*btc) / (1-alloc) when alloc<1
        old_alloc = df['alloc'].values
        old_strat = df['strat'].values
        cdi = np.where(np.abs(1 - old_alloc) > 1e-9,
                       (old_strat - old_alloc * btc) / (1 - old_alloc),
                       0.0)
    else:
        cdi = np.zeros(len(btc))
    return alloc * btc + (1 - alloc) * cdi


def main():
    print("=" * 70)
    print("EXPERIMENTS 2026-04-28: post-hoc sigmoid sweep + vol overlay")
    print("=" * 70)

    pred_df = pd.read_csv(PRED_PATH, parse_dates=['date'])
    h3 = pred_df[pred_df['variant'] == 'H=3'].copy().reset_index(drop=True)
    print(f"\nH=3 baseline: {len(h3)} rebals, "
          f"{h3['date'].min().date()} -> {h3['date'].max().date()}")

    preds = h3['pred'].values
    p_ups = h3['p_up'].values
    regimes = h3['regime'].values
    btc = h3['btc_fwd'].values

    # Back out CDI per rebal from existing strat
    old_strat = h3['strat'].values
    old_alloc = h3['alloc'].values
    cdi = np.where(np.abs(1 - old_alloc) > 1e-9,
                   (old_strat - old_alloc * btc) / (1 - old_alloc),
                   0.0)

    results = []

    # ── Baseline reproduction: H1 sigmoid=15 ──
    alloc_base = apply_sizing(preds, p_ups, regimes, K_H1, 15.0)
    strat_base = alloc_base * btc + (1 - alloc_base) * cdi
    m = metrics(strat_base)
    m['avg_alloc'] = float(alloc_base.mean())
    m['label'] = 'BASELINE H1 (K=60/30/15, sigmoid=15)'
    m['test'] = 'baseline'
    results.append(m)
    print(f"\n[baseline] H1 sigmoid=15:  cum={m['cum_ret']*100:+.1f}% "
          f"Sortino={m['sortino']:.2f} Sharpe={m['sharpe']:.2f} "
          f"DD={m['max_dd']*100:.1f}% avg_alloc={m['avg_alloc']*100:.1f}%")

    # ── TEST 2: SIGMOID SWEEP (with H1) ──
    print(f"\n[TEST 2] Sigmoid sweep (K=H1)")
    for s in [1, 3, 5, 7, 10, 15, 20, 25, 50]:
        alloc = apply_sizing(preds, p_ups, regimes, K_H1, float(s))
        strat = alloc * btc + (1 - alloc) * cdi
        m = metrics(strat)
        m['avg_alloc'] = float(alloc.mean())
        m['label'] = f'H1 sigmoid={s}'
        m['test'] = 'sigmoid_sweep'
        m['sigmoid'] = s
        results.append(m)
        marker = ' <-- current' if s == 15 else ''
        print(f"  sigmoid={s:2d}: cum={m['cum_ret']*100:+7.1f}% "
              f"Sortino={m['sortino']:.2f} Sharpe={m['sharpe']:.2f} "
              f"DD={m['max_dd']*100:6.2f}% avg_alloc={m['avg_alloc']*100:5.1f}%{marker}")

    # ── TEST 3: VOL REGIME OVERLAY ──
    # Compute rolling 30d vol on BTC daily returns (from dataset), align to rebal dates
    print(f"\n[TEST 3] Vol regime overlay (K=H1, sigmoid=15)")
    ds = pd.read_csv(DATASET, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    ds['ret'] = ds['price_usd'].pct_change()
    ds['vol_30d_ann'] = ds['ret'].rolling(30).std() * np.sqrt(365)
    vol_by_date = dict(zip(ds['date'], ds['vol_30d_ann']))
    vols = np.array([vol_by_date.get(d, np.nan) for d in h3['date']])

    # Defensive multiplier: when vol is high, derisk
    # alloc_adjusted = alloc * mult
    for vol_thresh in [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]:
        for mult in [0.5, 0.7]:
            multipliers = np.where(vols > vol_thresh, mult, 1.0)
            multipliers = np.where(np.isnan(vols), 1.0, multipliers)
            alloc = apply_sizing(preds, p_ups, regimes, K_H1, 15.0) * multipliers
            alloc = np.clip(alloc, 0.0, 1.0)
            strat = alloc * btc + (1 - alloc) * cdi
            m = metrics(strat)
            m['avg_alloc'] = float(alloc.mean())
            m['n_triggered'] = int((multipliers != 1.0).sum())
            m['label'] = f'H1 sig=15 + vol>{int(vol_thresh*100)}%/y -> x{mult}'
            m['test'] = 'vol_overlay'
            m['vol_thresh'] = vol_thresh
            m['mult'] = mult
            results.append(m)
            print(f"  vol>{int(vol_thresh*100):3d}%/y -> x{mult}: "
                  f"cum={m['cum_ret']*100:+7.1f}% "
                  f"Sortino={m['sortino']:.2f} Sharpe={m['sharpe']:.2f} "
                  f"DD={m['max_dd']*100:6.2f}% avg_alloc={m['avg_alloc']*100:5.1f}% "
                  f"trig={m['n_triggered']}/{len(alloc)}")

    # Save
    out_path = OUT / 'experiments_2026_04_28_posthoc.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: best by Sortino")
    print(f"{'='*70}")
    sigmoid_top = max([r for r in results if r['test'] == 'sigmoid_sweep'],
                       key=lambda r: r['sortino'])
    vol_top = max([r for r in results if r['test'] == 'vol_overlay'],
                   key=lambda r: r['sortino'])
    base = next(r for r in results if r['test'] == 'baseline')
    print(f"  baseline:       Sortino={base['sortino']:.2f} cum={base['cum_ret']*100:+.1f}%")
    print(f"  best sigmoid:   {sigmoid_top['label']}  Sortino={sigmoid_top['sortino']:.2f} "
          f"cum={sigmoid_top['cum_ret']*100:+.1f}%")
    print(f"  best vol overlay: {vol_top['label']}  Sortino={vol_top['sortino']:.2f} "
          f"cum={vol_top['cum_ret']*100:+.1f}%")


if __name__ == '__main__':
    main()
