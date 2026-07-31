"""
K_BULL capture test — does middle-ground K recover 2023's bull miss?

Hypothesis: H1 (K_BULL=60) was too defensive in 2023 (BTC +160%, strat +68%,
avg alloc 21%). Increasing K_BULL alone (keeping MILD/BEAR conservative)
might recover some 2023 upside while preserving 2022 protection (BEAR).

BULL-confirmation overlay: K_BULL increases when regime has been BULL stable
for N+ days. Idea: don't increase exposure on first day of BULL (whipsaw),
only after sustained bull.

Post-hoc on today's baseline predictions (seed=242, Ultra 9, XGBoost 3.2.0).
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / 'outputs/results'
SIGMOID_SCALE = 15


def metrics(strat: np.ndarray, btc: np.ndarray, cdi: np.ndarray) -> dict:
    cum = float(np.cumprod(1 + strat)[-1] - 1)
    cum_btc = float(np.cumprod(1 + btc)[-1] - 1)
    neg = strat[strat < 0]
    dev = float(np.sqrt(np.mean(neg ** 2))) if len(neg) > 0 else 1e-9
    sortino = float(np.mean(strat) / dev * np.sqrt(52)) if dev > 0 else 0.0
    sd = float(np.std(strat, ddof=0))
    excess = strat - cdi
    sd_e = float(np.std(excess, ddof=0))
    sharpe_x = float(np.mean(excess) / sd_e * np.sqrt(52)) if sd_e > 0 else 0.0
    eq = np.concatenate([[1.0], np.cumprod(1 + strat)])
    peak = np.maximum.accumulate(eq)
    maxdd = float(((eq - peak) / peak).min())
    return {
        'cum_ret': cum, 'btc_ret': cum_btc, 'sortino': sortino,
        'sharpe_excess': sharpe_x, 'max_dd_w': maxdd,
    }


def per_year(df: pd.DataFrame, alloc: np.ndarray, btc: np.ndarray, cdi: np.ndarray) -> dict:
    df = df.copy()
    df['_alloc'] = alloc
    df['_strat'] = alloc * btc + (1 - alloc) * cdi
    df['_btc'] = btc
    df['year'] = df['date'].dt.year
    out = {}
    for y in sorted(df['year'].unique()):
        sub = df[df['year'] == y]
        out[int(y)] = {
            'strat': float(np.prod(1 + sub['_strat']) - 1),
            'btc':   float(np.prod(1 + sub['_btc']) - 1),
            'avg_alloc': float(sub['_alloc'].mean()),
        }
    return out


def regime_persistence(regimes: np.ndarray, dates: pd.Series) -> np.ndarray:
    """Days that current regime has been stable up to each rebal."""
    persist = np.zeros(len(regimes), dtype=int)
    for i in range(len(regimes)):
        if i == 0:
            persist[i] = 1
        else:
            if regimes[i] == regimes[i - 1]:
                # Days between last rebal and this one — approx, use 7d default
                days = (dates.iloc[i] - dates.iloc[i - 1]).days if i > 0 else 7
                persist[i] = persist[i - 1] + days
            else:
                persist[i] = (dates.iloc[i] - dates.iloc[i - 1]).days if i > 0 else 1
    return persist


def main():
    df = pd.read_csv(OUT / 'experiments_2026_04_28_baseline.csv', parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"Loaded {len(df)} rebals from baseline H1 (seed 242)")

    pred = df['pred'].values
    p_up = df['p_up'].values
    btc = df['btc_fwd'].values
    # Back out CDI per-period from strat = alloc*btc + (1-alloc)*cdi
    old_alloc = df['alloc'].values
    old_strat = df['strat'].values
    cdi = np.where(np.abs(1 - old_alloc) > 1e-9,
                   (old_strat - old_alloc * btc) / (1 - old_alloc),
                   0.0)
    cdi = np.where(np.isnan(cdi) | np.isinf(cdi), 0.0, cdi)
    regime = df['regime'].values
    conf = 1.0 / (1.0 + np.exp(-np.abs(p_up - 0.5) * SIGMOID_SCALE))

    # === Variants ===
    K_VARIANTS = {
        # Baseline
        'H1 (60/30/15)':            {'BULL': 60, 'MILD': 30, 'BEAR': 15},
        # Middle-ground proportional
        'K70 (70/35/17)':           {'BULL': 70, 'MILD': 35, 'BEAR': 17},
        'K80 (80/40/20)':           {'BULL': 80, 'MILD': 40, 'BEAR': 20},
        # Asymmetric: BULL more aggressive, MILD/BEAR same as H1
        'K_BULL75 (75/30/15)':      {'BULL': 75, 'MILD': 30, 'BEAR': 15},
        'K_BULL90 (90/30/15)':      {'BULL': 90, 'MILD': 30, 'BEAR': 15},
        'K_BULL100 (100/30/15)':    {'BULL': 100, 'MILD': 30, 'BEAR': 15},
        'K_BULL120 (120/30/15)':    {'BULL': 120, 'MILD': 30, 'BEAR': 15},
        # H2 reference
        'H2 (100/50/20)':           {'BULL': 100, 'MILD': 50, 'BEAR': 20},
    }

    print(f"\n{'='*120}")
    print(f"{'Variant':<28s} {'TOTAL':>10s} {'Sortino':>8s} {'Shp_x':>6s} {'DD_w':>7s} | "
          f"{'2022':>10s} {'2023':>10s} {'2024':>10s} {'2025':>10s} {'2026':>9s}")
    print(f"{'='*120}")

    results_static = {}
    for name, K in K_VARIANTS.items():
        K_arr = np.array([K[r] for r in regime])
        alloc = np.clip(pred * K_arr * conf, 0.0, 1.0)
        strat = alloc * btc + (1 - alloc) * cdi
        m = metrics(strat, btc, cdi)
        py = per_year(df, alloc, btc, cdi)
        excess_str = lambda y: f"{(py.get(y,{}).get('strat',0) - py.get(y,{}).get('btc',0))*100:+5.0f}pp"
        strat_str = lambda y: f"{py.get(y,{}).get('strat',0)*100:+5.1f}%"
        print(f"{name:<28s} {m['cum_ret']*100:+9.1f}% {m['sortino']:8.2f} "
              f"{m['sharpe_excess']:6.2f} {m['max_dd_w']*100:6.2f}% | "
              f"{strat_str(2022):>10s} {strat_str(2023):>10s} {strat_str(2024):>10s} "
              f"{strat_str(2025):>10s} {strat_str(2026):>9s}")
        results_static[name] = {'overall': m, 'per_year': py}

    # === BULL-confirmation overlay variants ===
    print(f"\n{'='*120}")
    print(f"BULL-confirmation overlay (boost K_BULL when regime stable for N+ days)")
    print(f"{'='*120}")
    print(f"{'Variant':<28s} {'TOTAL':>10s} {'Sortino':>8s} {'Shp_x':>6s} {'DD_w':>7s} | "
          f"{'2022':>10s} {'2023':>10s} {'2024':>10s} {'2025':>10s} {'2026':>9s}")
    print(f"{'-'*120}")

    persistence = regime_persistence(regime, df['date'])

    OVERLAYS = [
        ('H1 + boost1.5x BULL>60d',  60, 1.5, 60),
        ('H1 + boost1.5x BULL>90d',  60, 1.5, 90),
        ('H1 + boost2.0x BULL>60d',  60, 2.0, 60),
        ('H1 + boost2.0x BULL>90d',  60, 2.0, 90),
        ('H1 + boost1.7x BULL>60d',  60, 1.7, 60),
        ('H1 + boost1.7x BULL>90d',  60, 1.7, 90),
    ]
    for name, base_bull, boost, days in OVERLAYS:
        K_bull_dyn = np.where(
            (regime == 'BULL') & (persistence >= days),
            base_bull * boost,
            np.where(regime == 'BULL', base_bull, 0)  # placeholder
        )
        # Build full K_arr
        K_arr = np.zeros(len(regime))
        for i, r in enumerate(regime):
            if r == 'BULL':
                K_arr[i] = base_bull * boost if persistence[i] >= days else base_bull
            elif r == 'MILD':
                K_arr[i] = 30
            else:
                K_arr[i] = 15
        alloc = np.clip(pred * K_arr * conf, 0.0, 1.0)
        strat = alloc * btc + (1 - alloc) * cdi
        m = metrics(strat, btc, cdi)
        py = per_year(df, alloc, btc, cdi)
        strat_str = lambda y: f"{py.get(y,{}).get('strat',0)*100:+5.1f}%"
        print(f"{name:<28s} {m['cum_ret']*100:+9.1f}% {m['sortino']:8.2f} "
              f"{m['sharpe_excess']:6.2f} {m['max_dd_w']*100:6.2f}% | "
              f"{strat_str(2022):>10s} {strat_str(2023):>10s} {strat_str(2024):>10s} "
              f"{strat_str(2025):>10s} {strat_str(2026):>9s}")

    # Reference: BTC by year
    print(f"\n{'-'*120}")
    py_btc = {}
    df['year'] = df['date'].dt.year
    for y in sorted(df['year'].unique()):
        sub = df[df['year'] == y]
        py_btc[int(y)] = float(np.prod(1 + sub['btc_fwd']) - 1)
    btc_str = ' '.join([f"{y}:{r*100:+5.1f}%" for y, r in py_btc.items()])
    print(f"{'BTC B&H per year:':<28s} {' ':>10s} {' ':>8s} {' ':>6s} {' ':>7s} | {btc_str}")

    # === Save and pick winners ===
    out_path = OUT / 'k_bull_capture_test.json'
    with open(out_path, 'w') as f:
        json.dump({k: {'overall': v['overall'], 'per_year': v['per_year']}
                   for k, v in results_static.items()}, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    # Find best by 2023 capture vs total
    print(f"\n{'='*100}")
    print(f"WINNERS (Static K only):")
    print(f"{'='*100}")
    base_2023 = results_static['H1 (60/30/15)']['per_year'][2023]['strat']
    base_total = results_static['H1 (60/30/15)']['overall']['cum_ret']
    base_dd = results_static['H1 (60/30/15)']['overall']['max_dd_w']
    base_sortino = results_static['H1 (60/30/15)']['overall']['sortino']

    print(f"\nBaseline H1: total +{base_total*100:.1f}%, 2023 +{base_2023*100:.1f}%, "
          f"Sortino {base_sortino:.2f}, DD {base_dd*100:.2f}%")
    print()
    for name, r in results_static.items():
        if name == 'H1 (60/30/15)':
            continue
        d_total = (r['overall']['cum_ret'] - base_total) * 100
        d_2023 = (r['per_year'][2023]['strat'] - base_2023) * 100
        d_dd = (r['overall']['max_dd_w'] - base_dd) * 100
        d_sort = r['overall']['sortino'] - base_sortino
        print(f"  {name:<28s} ΔTotal={d_total:+6.1f}pp  Δ2023={d_2023:+5.1f}pp  "
              f"ΔSortino={d_sort:+5.2f}  ΔDD={d_dd:+5.2f}pp")


if __name__ == '__main__':
    main()
