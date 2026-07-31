"""
Attribution test — quanto do edge vem de regime, ML ou ambos?

Pergunta do usuário: "Se eu for 100% crypto em BULL e usar AI só em MILD/BEAR,
ganho a mesma coisa? O sinal de ML é fraco?"

Strategies tested (todas em BRL):
  A. CURRENT       ML pred × K[regime] × conf  (atual)
  B. USER_HYBRID   100% BTC se BULL, ML se MILD/BEAR
  C. REGIME_ONLY   regras fixas por regime (sem ML)
  D. PURE_HODL     100% BTC sempre
  E. ML_NO_REGIME  ML pred × K_const × conf (K=50 sem regime, K=H1)
  F. ML_NO_CONF    ML pred × K[regime] (sem confidence)
  G. SIGN_ONLY     0.5 se pred>0, 0 se pred<=0 (sem magnitude)
  H. ALL_BULL_AI   AI em BULL+MILD+BEAR (atual)
  I. AI_ONLY_BEAR  100% BTC em BULL+MILD, AI só em BEAR

Resultado: separa contribuição de regime vs ML magnitude vs ML sign vs conf.
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/production'))

OUT = ROOT / 'outputs/results'
OUT_CHARTS = ROOT / 'outputs/charts'

K_H1 = {'BULL': 60, 'MILD': 30, 'BEAR': 15}
SIGMOID_SCALE = 15

# Load BRL FX
fx_raw = pd.read_csv(ROOT / 'outputs/results/usd_brl.csv', skiprows=[1, 2])
fx_raw.columns = ['date', 'close', 'high', 'low', 'open', 'volume']
fx_raw['date'] = pd.to_datetime(fx_raw['date'])
fx = fx_raw[['date', 'close']].rename(columns={'close': 'usdbrl'})


# Build daily price series in BRL
ds = pd.read_csv(ROOT / 'scripts/production/data/dataset_production.csv', parse_dates=['date'])
ds = ds.sort_values('date').reset_index(drop=True)
ds = ds.merge(fx, on='date', how='left')
ds['usdbrl'] = ds['usdbrl'].ffill().bfill()
ds['price_brl'] = ds['price_usd'] * ds['usdbrl']
ds['btc_ret_brl'] = ds['price_brl'].pct_change().fillna(0)

# Load baseline predictions (seed 242, today's machine)
base = pd.read_csv(OUT / 'experiments_2026_04_28_baseline.csv', parse_dates=['date'])
base = base.sort_values('date').reset_index(drop=True)

# Compute confidence factor
base['conf'] = 1 / (1 + np.exp(-np.abs(base['p_up'] - 0.5) * SIGMOID_SCALE))


# === Strategy alloc functions ===
def alloc_current(row):
    """A. Current: ML × K[regime] × conf"""
    return float(np.clip(row['pred'] * K_H1[row['regime']] * row['conf'], 0, 1))


def alloc_user_hybrid(row):
    """B. User hypothesis: 100% if BULL, else AI"""
    if row['regime'] == 'BULL':
        return 1.0
    return float(np.clip(row['pred'] * K_H1[row['regime']] * row['conf'], 0, 1))


def alloc_regime_only(row):
    """C. Regime-only fixed: 1.0 BULL, 0.5 MILD, 0 BEAR"""
    return {'BULL': 1.0, 'MILD': 0.5, 'BEAR': 0.0}[row['regime']]


def alloc_pure_hodl(row):
    """D. 100% BTC always"""
    return 1.0


def alloc_ml_no_regime(row):
    """E. ML × K_const × conf (no regime filter, K=50 flat)"""
    return float(np.clip(row['pred'] * 50 * row['conf'], 0, 1))


def alloc_ml_no_conf(row):
    """F. ML × K[regime] without confidence sigmoid"""
    return float(np.clip(row['pred'] * K_H1[row['regime']], 0, 1))


def alloc_sign_only(row):
    """G. 50% if pred>0, 0 else (no magnitude, no regime, no conf)"""
    return 0.5 if row['pred'] > 0 else 0.0


def alloc_ai_only_bear(row):
    """I. 100% BTC se BULL+MILD, AI só em BEAR"""
    if row['regime'] == 'BULL':
        return 1.0
    elif row['regime'] == 'MILD':
        return 1.0
    else:  # BEAR
        return float(np.clip(row['pred'] * K_H1[row['regime']] * row['conf'], 0, 1))


STRATEGIES = {
    'A. CURRENT (ML×K×conf)':        alloc_current,
    'B. USER (BULL=100%, else AI)':  alloc_user_hybrid,
    'C. REGIME-ONLY (1/0.5/0)':      alloc_regime_only,
    'D. PURE HODL':                  alloc_pure_hodl,
    'E. ML, NO REGIME (K=50)':       alloc_ml_no_regime,
    'F. ML+REGIME, NO CONF':         alloc_ml_no_conf,
    'G. SIGN-ONLY (50% if pred>0)':  alloc_sign_only,
    'I. BULL+MILD=100%, AI=BEAR':    alloc_ai_only_bear,
}


# === Daily simulation in BRL ===
start = pd.Timestamp('2022-01-07')
end = pd.Timestamp('2026-04-17')
daily = ds[(ds['date'] >= start) & (ds['date'] <= end)].copy().reset_index(drop=True)
cdi_daily = (1.13)**(1/365) - 1


def daily_sim(rebal_dates, rebal_allocs, cost_bps=8):
    """Daily MtM simulation: alloc applies AFTER rebal date."""
    rets = []
    prev = 0
    for _, row in daily.iterrows():
        # Most recent rebal STRICTLY BEFORE today
        mask = rebal_dates < row['date']
        new_alloc = rebal_allocs[mask][-1] if mask.any() else 0.0
        cost = abs(new_alloc - prev) * cost_bps / 10000 if new_alloc != prev else 0
        rets.append(new_alloc * row['btc_ret_brl'] + (1 - new_alloc) * cdi_daily - cost)
        if new_alloc != prev:
            prev = new_alloc
    return np.array(rets)


# === Run all strategies ===
print(f"Attribution test — {len(STRATEGIES)} strategies, BRL puro")
print(f"Period: {start.date()} → {end.date()}, {len(daily)} days, {len(base)} rebals")
print()

dates = daily['date'].values
btc_brl_rets = daily['btc_ret_brl'].values
cdi_rets = np.full(len(daily), cdi_daily)

rebal_dates = base['date'].values

results = {}
for name, alloc_fn in STRATEGIES.items():
    allocs = np.array([alloc_fn(r) for _, r in base.iterrows()])
    strat_daily = daily_sim(rebal_dates, allocs)

    # Aggregate metrics
    cum = float(np.cumprod(1 + strat_daily)[-1] - 1)
    cum_btc = float(np.cumprod(1 + btc_brl_rets)[-1] - 1)
    n_days = len(strat_daily)
    cagr = (1 + cum) ** (365 / n_days) - 1
    neg = strat_daily[strat_daily < 0]
    dev = float(np.sqrt(np.mean(neg ** 2))) if len(neg) > 0 else 1e-9
    sortino = float(np.mean(strat_daily) / dev * np.sqrt(365)) if dev > 0 else 0.0
    excess = strat_daily - cdi_rets
    sd_e = float(np.std(excess, ddof=0))
    sharpe_x = float(np.mean(excess) / sd_e * np.sqrt(365)) if sd_e > 0 else 0.0
    eq = np.concatenate([[1.0], np.cumprod(1 + strat_daily)])
    peak = np.maximum.accumulate(eq)
    maxdd = float(((eq - peak) / peak).min())

    # Per year
    daily['_y'] = daily['date'].dt.year
    daily['_strat'] = strat_daily
    yearly = {}
    for y in [2022, 2023, 2024, 2025, 2026]:
        mask = daily['_y'] == y
        if y == 2022:
            mask = mask & (daily['date'] >= pd.Timestamp('2022-01-07'))
        if y == 2026:
            mask = mask & (daily['date'] <= pd.Timestamp('2026-04-17'))
        if mask.any():
            yearly[y] = float(np.cumprod(1 + strat_daily[mask.values])[-1] - 1)

    avg_alloc = float(allocs.mean())
    results[name] = {
        'cum_brl': cum, 'cagr_brl': cagr, 'sortino_d': sortino,
        'sharpe_x': sharpe_x, 'max_dd': maxdd, 'avg_alloc': avg_alloc,
        'yearly': yearly, 'allocs_sample': allocs[:10].tolist(),
    }


# === Print table ===
print(f"{'Strategy':<35s} {'CAGR':>9s} {'Sortino':>8s} {'Shp_x':>7s} {'DD':>9s} {'avg alloc':>10s}")
print("-" * 90)
for name, m in results.items():
    print(f"{name:<35s} {m['cagr_brl']*100:+8.1f}% {m['sortino_d']:7.2f} "
          f"{m['sharpe_x']:6.2f} {m['max_dd']*100:+7.2f}% {m['avg_alloc']*100:9.1f}%")

# Per-year breakdown
print(f"\n{'='*120}")
print(f"PER-YEAR RETURN (BRL)")
print(f"{'='*120}")
print(f"{'Strategy':<35s} {'2022':>9s} {'2023':>10s} {'2024':>10s} {'2025':>10s} {'2026':>9s}")
for name, m in results.items():
    y = m['yearly']
    print(f"{name:<35s} {y.get(2022,0)*100:+8.1f}% {y.get(2023,0)*100:+9.1f}% "
          f"{y.get(2024,0)*100:+9.1f}% {y.get(2025,0)*100:+9.1f}% {y.get(2026,0)*100:+8.1f}%")

# BTC for reference
btc_yearly = {}
daily['_y'] = daily['date'].dt.year
for y in [2022, 2023, 2024, 2025, 2026]:
    mask = daily['_y'] == y
    if y == 2022:
        mask = mask & (daily['date'] >= pd.Timestamp('2022-01-07'))
    if y == 2026:
        mask = mask & (daily['date'] <= pd.Timestamp('2026-04-17'))
    btc_yearly[y] = float(np.cumprod(1 + btc_brl_rets[mask.values])[-1] - 1)
print(f"{'BTC B&H reference':<35s} {btc_yearly[2022]*100:+8.1f}% {btc_yearly[2023]*100:+9.1f}% "
      f"{btc_yearly[2024]*100:+9.1f}% {btc_yearly[2025]*100:+9.1f}% {btc_yearly[2026]*100:+8.1f}%")


# === ATTRIBUTION ANALYSIS ===
print(f"\n{'='*100}")
print("ATTRIBUTION — Sortino decomposition")
print(f"{'='*100}")
sortino_current = results['A. CURRENT (ML×K×conf)']['sortino_d']
sortino_regime = results['C. REGIME-ONLY (1/0.5/0)']['sortino_d']
sortino_ml_no_regime = results['E. ML, NO REGIME (K=50)']['sortino_d']
sortino_sign = results['G. SIGN-ONLY (50% if pred>0)']['sortino_d']

print(f"Sign-only (random direction signal):       Sortino {sortino_sign:.2f}")
print(f"Regime-only (no ML):                       Sortino {sortino_regime:.2f}  "
      f"(+{sortino_regime-sortino_sign:.2f} from regime structure)")
print(f"ML-only no regime (K=50 flat):             Sortino {sortino_ml_no_regime:.2f}  "
      f"(+{sortino_ml_no_regime-sortino_sign:.2f} from ML magnitude alone)")
print(f"Full current (ML × regime × conf):         Sortino {sortino_current:.2f}  "
      f"(+{sortino_current-sortino_regime:.2f} ML adds on top of regime,  "
      f"+{sortino_current-sortino_ml_no_regime:.2f} regime adds on top of ML)")

# Save
with open(OUT / 'regime_vs_ml_attribution.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved: {OUT / 'regime_vs_ml_attribution.json'}")
