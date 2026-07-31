"""
Reconcile metrics across docs: weekly vs daily, period covered, train sizes.

Answers the user's questions:
  1. Why did Sharpe drop from 2.3 (config.py) to 2.0 (OVERFIT_TESTS)?
  2. Is DD weekly or daily? Verify both.
  3. Is training EXACTLY 4 years? Check cutoffs.
  4. Why return dropped from ~800% to ~700% in fresh retrain?
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/production'))

OUT = ROOT / 'outputs/results'
DATASET = ROOT / 'scripts/production/data/dataset_production.csv'

from src.features.macro.cdi_rates import build_rf_daily


def metrics_weekly(strat: np.ndarray, cdi_period: np.ndarray = None):
    """Weekly metrics — annualized with sqrt(52) since rebals are mostly weekly."""
    cum = float(np.cumprod(1 + strat)[-1] - 1)
    neg = strat[strat < 0]
    dev = float(np.sqrt(np.mean(neg ** 2))) if len(neg) > 0 else 1e-9
    sortino_abs = float(np.mean(strat) / dev * np.sqrt(52)) if dev > 0 else 0.0
    sd = float(np.std(strat, ddof=0))
    sharpe_abs = float(np.mean(strat) / sd * np.sqrt(52)) if sd > 0 else 0.0
    # Excess Sharpe (vs CDI) — the textbook one
    if cdi_period is not None:
        excess = strat - cdi_period
        sd_e = float(np.std(excess, ddof=0))
        sharpe_excess = float(np.mean(excess) / sd_e * np.sqrt(52)) if sd_e > 0 else 0.0
    else:
        sharpe_excess = None
    eq = np.concatenate([[1.0], np.cumprod(1 + strat)])
    peak = np.maximum.accumulate(eq)
    maxdd_weekly = float(((eq - peak) / peak).min())
    return {
        'cum_ret': cum,
        'sortino_abs_w': sortino_abs,
        'sharpe_abs_w': sharpe_abs,
        'sharpe_excess_w': sharpe_excess,
        'max_dd_weekly': maxdd_weekly,
        'n_rebals': int(len(strat)),
    }


def expand_to_daily(rebal_df: pd.DataFrame, ds: pd.DataFrame, rf: pd.Series):
    """Expand rebal-level allocations into a DAILY equity curve with mark-to-market.

    For each day between rebal_i and rebal_{i+1}:
      strat_daily_ret[d] = alloc[i] * btc_daily_ret[d] + (1 - alloc[i]) * cdi_daily_ret[d]
    """
    rebals = rebal_df.copy()
    rebals['date'] = pd.to_datetime(rebals['date'])
    rebals = rebals.sort_values('date').reset_index(drop=True)

    ds = ds.copy()
    ds['date'] = pd.to_datetime(ds['date'])
    ds = ds.sort_values('date').reset_index(drop=True)
    ds['btc_ret_daily'] = ds['price_usd'].pct_change()

    # Pair each day with its applicable alloc (latest rebal <= day)
    out_dates, out_strat, out_alloc = [], [], []
    for i in range(len(rebals)):
        d_start = rebals['date'].iloc[i]
        alloc = float(rebals['alloc'].iloc[i])
        d_end = rebals['date'].iloc[i + 1] if i + 1 < len(rebals) else ds['date'].iloc[-1]
        days = ds[(ds['date'] > d_start) & (ds['date'] <= d_end)]
        for _, row in days.iterrows():
            d = row['date']
            btc_d = row['btc_ret_daily']
            if pd.isna(btc_d):
                btc_d = 0.0
            cdi_d = float(rf.get(d, 0.0))
            strat_d = alloc * btc_d + (1 - alloc) * cdi_d
            out_dates.append(d)
            out_strat.append(strat_d)
            out_alloc.append(alloc)

    return pd.DataFrame({'date': out_dates, 'strat_d': out_strat, 'alloc': out_alloc})


def metrics_daily(daily_df: pd.DataFrame):
    """Daily metrics — annualized with sqrt(365) (calendar days incl weekends)."""
    s = daily_df['strat_d'].values
    cum = float(np.cumprod(1 + s)[-1] - 1)
    neg = s[s < 0]
    dev = float(np.sqrt(np.mean(neg ** 2))) if len(neg) > 0 else 1e-9
    sortino_d = float(np.mean(s) / dev * np.sqrt(365)) if dev > 0 else 0.0
    sd = float(np.std(s, ddof=0))
    sharpe_d = float(np.mean(s) / sd * np.sqrt(365)) if sd > 0 else 0.0
    eq = np.concatenate([[1.0], np.cumprod(1 + s)])
    peak = np.maximum.accumulate(eq)
    maxdd_daily = float(((eq - peak) / peak).min())
    n_days = len(s)
    cagr = float((1 + cum) ** (365 / n_days) - 1) if n_days > 0 else 0.0
    return {
        'cum_ret': cum,
        'cagr': cagr,
        'sortino_daily': sortino_d,
        'sharpe_daily': sharpe_d,
        'max_dd_daily': maxdd_daily,
        'n_days': n_days,
    }


def main():
    print("=" * 78)
    print("RECONCILIATION OF METRICS — answering user questions")
    print("=" * 78)

    ds = pd.read_csv(DATASET, parse_dates=['date']).sort_values('date').reset_index(drop=True)

    print(f"\nDataset: {len(ds)} rows")
    print(f"Date range: {ds['date'].min().date()} -> {ds['date'].max().date()}")
    print(f"Years covered: {(ds['date'].max() - ds['date'].min()).days / 365:.2f}")

    # ── 1. INVESTIGATE TRAINING PERIOD ──
    print(f"\n{'='*78}\n  Q3: Is training EXACTLY 4 years?\n{'='*78}")
    cuts = [pd.Timestamp(f'{y}-{m:02d}-01') for y in range(2022, 2027) for m in [1, 7]]
    cuts = [c for c in cuts if c <= ds['date'].max()]
    for c in cuts[:6]:
        train_size = int((ds['date'] < c).sum())
        train_start = ds['date'].iloc[0]
        train_years = (c - train_start).days / 365
        print(f"  Cutoff {c.date()}: train = {train_start.date()} -> {c.date() - pd.Timedelta(days=1)}  "
              f"({train_size} days, {train_years:.2f} years)")
    print(f"  ...")
    c = cuts[-1]
    train_size = int((ds['date'] < c).sum())
    train_years = (c - ds['date'].iloc[0]).days / 365
    print(f"  Cutoff {c.date()}: train = {ds['date'].iloc[0].date()} -> {c.date() - pd.Timedelta(days=1)}  "
          f"({train_size} days, {train_years:.2f} years)")
    print(f"\n  ANSWER: NO — training is EXPANDING WINDOW.")
    print(f"  First cutoff (2022-01) trains on ~3y, last cutoff (2026-01) trains on ~7y.")
    print(f"  This is BY DESIGN — 'expanding window' walk-forward (Lopez de Prado standard).")

    # ── 2. RECOMPUTE BASELINE METRICS BOTH WAYS ──
    print(f"\n{'='*78}\n  Q1+Q2: Recompute baseline H1 — weekly AND daily DD\n{'='*78}")

    base_path = OUT / 'experiments_2026_04_28_baseline.csv'
    base = pd.read_csv(base_path, parse_dates=['date'])
    print(f"  Source: {base_path.name}")
    print(f"  Rebals: {len(base)}, range {base['date'].min().date()} -> {base['date'].max().date()}")
    print(f"  Period: {(base['date'].max() - base['date'].min()).days / 365:.2f} years")

    # Build CDI series for weekly cdi_period
    start = base['date'].min()
    end = ds['date'].iloc[-1]
    rf_index = pd.date_range(start - pd.Timedelta(days=10), end, freq='D')
    rf = pd.Series(build_rf_daily(rf_index), index=rf_index)

    # Recover cdi_period for each rebal (between rebal i and i+1)
    base_sorted = base.sort_values('date').reset_index(drop=True)
    cdi_period = []
    for i in range(len(base_sorted)):
        d0 = base_sorted['date'].iloc[i]
        d1 = base_sorted['date'].iloc[i + 1] if i + 1 < len(base_sorted) else end
        mask = (rf.index > d0) & (rf.index <= d1)
        cdi_period.append(float((1 + rf[mask]).prod() - 1))
    cdi_period = np.array(cdi_period)

    # Weekly
    m_w = metrics_weekly(base['strat'].values, cdi_period)
    print(f"\n  WEEKLY (rebal-level returns, annualized sqrt(52)):")
    print(f"    cum_ret:          {m_w['cum_ret']*100:+8.1f}%")
    print(f"    Sortino (abs):    {m_w['sortino_abs_w']:.2f}")
    print(f"    Sharpe (abs):     {m_w['sharpe_abs_w']:.2f}")
    print(f"    Sharpe (excess):  {m_w['sharpe_excess_w']:.2f}    <-- CDI subtracted (textbook)")
    print(f"    Max DD weekly:    {m_w['max_dd_weekly']*100:.2f}%")

    # Daily
    daily = expand_to_daily(base, ds, rf)
    m_d = metrics_daily(daily)
    print(f"\n  DAILY (mark-to-market, annualized sqrt(365)):")
    print(f"    cum_ret:          {m_d['cum_ret']*100:+8.1f}%")
    print(f"    CAGR:             {m_d['cagr']*100:+.1f}%")
    print(f"    Sortino daily:    {m_d['sortino_daily']:.2f}")
    print(f"    Sharpe daily:     {m_d['sharpe_daily']:.2f}")
    print(f"    Max DD daily:     {m_d['max_dd_daily']*100:.2f}%   <-- THE NUMBER MODEL_FINAL.md REPORTS")
    print(f"    n_days:           {m_d['n_days']}")

    # ── 3. COMPARE TO REPORTED NUMBERS ──
    print(f"\n{'='*78}\n  COMPARISON WITH REPORTED NUMBERS IN DOCS\n{'='*78}")
    print(f"{'Source':<35s} {'cum':>10s} {'Sortino':>9s} {'Sharpe':>8s} {'DD':>10s}")
    print(f"{'-'*78}")
    rows = [
        ("OVERFIT_TESTS H1 baseline (weekly)", "+782%",  "7.00", "2.04", "-2.9% (w)"),
        ("OVERFIT_TESTS H1+controls (weekly)", "+667%",  "6.91", "1.99", "-2.9% (w)"),
        ("MODEL_FINAL H1 8bps (daily, w/risk)", "n/a",   "3.61", "2.48", "-7.87% (d)"),
        ("config.py docstring (old, H2/5bps)",  "+1131%","5.91", "2.39", "-9.1% (?)"),
        ("Today baseline H1 32f WEEKLY",  f"{m_w['cum_ret']*100:+.1f}%",
                                          f"{m_w['sortino_abs_w']:.2f}",
                                          f"{m_w['sharpe_abs_w']:.2f}",
                                          f"{m_w['max_dd_weekly']*100:.2f}% (w)"),
        ("Today baseline H1 32f WEEKLY excess", f"{m_w['cum_ret']*100:+.1f}%",
                                                "—",
                                                f"{m_w['sharpe_excess_w']:.2f}",
                                                "—"),
        ("Today baseline H1 32f DAILY",   f"{m_d['cum_ret']*100:+.1f}%",
                                          f"{m_d['sortino_daily']:.2f}",
                                          f"{m_d['sharpe_daily']:.2f}",
                                          f"{m_d['max_dd_daily']*100:.2f}% (d)"),
    ]
    for r in rows:
        print(f"{r[0]:<35s} {r[1]:>10s} {r[2]:>9s} {r[3]:>8s} {r[4]:>10s}")

    # ── 4. INVESTIGATE RETURN DROP (782 -> 654) ──
    print(f"\n{'='*78}\n  Q4: Why return dropped from +782% (OVERFIT_TESTS) to +654% (today)?\n{'='*78}")
    print(f"  Hypotheses:")
    print(f"   a. New dataset has more rows than horizon_ablation_4y.csv had when generated")
    print(f"   b. Same XGB seed but different XGBoost version / underlying RNG")
    print(f"   c. Different feature standardization or NaN handling")
    print(f"\n  Check (a): horizon_ablation_4y range vs today's run")
    h4y = pd.read_csv(OUT / 'horizon_ablation_4y.csv', parse_dates=['date'])
    h4y_h3 = h4y[h4y['variant'] == 'H=3']
    print(f"   horizon_ablation_4y H=3:  {h4y_h3['date'].min().date()} -> {h4y_h3['date'].max().date()}  "
          f"({len(h4y_h3)} rebals)")
    print(f"   today's baseline:          {base['date'].min().date()} -> {base['date'].max().date()}  "
          f"({len(base)} rebals)")
    print(f"   Difference: {len(base) - len(h4y_h3)} rebals")
    print(f"   Dataset today: {len(ds)} rows ending {ds['date'].iloc[-1].date()}")

    # ── 5. SAVE ──
    out_path = OUT / 'reconcile_metrics_2026_04_28.json'
    with open(out_path, 'w') as f:
        json.dump({
            'weekly': m_w,
            'daily': m_d,
            'training_window': 'expanding (3-7 years)',
            'rebals_today': len(base),
            'rebals_horizon_ablation_4y_h3': len(h4y_h3),
            'dataset_rows_today': len(ds),
        }, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
