"""
10,000-permutation test on production signal_history.csv (H1 + BAGS=160 + 4bps).

Method:
  1. Load production rebal log (249 rebals, 2022-2026)
  2. Back out implied CDI per rebal
  3. Baseline = actual cum return + Sortino
  4. For 10,000 random shuffles of (pred, p_up):
     - Recompute allocation = clip(pred * K_regime * sigmoid_conf, 0, 1)
     - Recompute strat_return = alloc * btc_fwd + (1-alloc) * cdi
     - Compute cum return + Sortino weekly
  5. Report fraction that beats baseline (p-value)

Saves:
  outputs/results/permutation_10k.csv (every shuffle's stats)
  outputs/results/permutation_10k_summary.json (headline numbers)
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SIGNAL_PATH = ROOT / 'scripts' / 'production' / 'data' / 'signal_history.csv'
OUT_DIR = ROOT / 'outputs' / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_REGIME = {'BULL': 60, 'MILD': 30, 'BEAR': 15}
SIGMOID_SCALE = 15
N_PERM = 1_000_000
PPY_W = 52


def metrics(rets):
    rets = np.asarray(rets)
    cum = (1 + rets).prod() - 1
    downside = rets[rets < 0]
    if len(downside) > 0 and (downside ** 2).mean() > 0:
        sortino = (rets.mean() / np.sqrt((downside ** 2).mean())) * np.sqrt(PPY_W)
    else:
        sortino = float('inf')
    if rets.std() > 0:
        sharpe = (rets.mean() / rets.std()) * np.sqrt(PPY_W)
    else:
        sharpe = 0.0
    cum_series = (1 + rets).cumprod()
    dd = ((cum_series - np.maximum.accumulate(cum_series)) / np.maximum.accumulate(cum_series)).min()
    return cum, sortino, sharpe, dd


def main():
    print(f'Loading {SIGNAL_PATH}')
    df = pd.read_csv(SIGNAL_PATH, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    n0 = len(df)
    df = df.dropna(subset=['retorno_btc', 'retorno_strat']).reset_index(drop=True)
    print(f'  {len(df)} rebals (dropped {n0 - len(df)} with no forward return), {df.date.min().date()} -> {df.date.max().date()}')

    pred = df['previsao'].values.astype(float)
    p_up = df['p_up'].values.astype(float)
    regime = df['regime'].values
    btc_fwd = df['retorno_btc'].values.astype(float)
    alloc_real = df['allocation'].values.astype(float)
    strat_real = df['retorno_strat'].values.astype(float)

    cdi = np.where(
        (1 - alloc_real) > 1e-3,
        (strat_real - alloc_real * btc_fwd) / (1 - alloc_real),
        0.0021,
    )

    baseline_cum, baseline_sortino, baseline_sharpe, baseline_dd = metrics(strat_real)
    print(f'\nBASELINE (actual signal_history):')
    print(f'  cum return: {baseline_cum*100:+.1f}%')
    print(f'  Sortino  W: {baseline_sortino:.2f}')
    print(f'  Sharpe   W: {baseline_sharpe:.2f}')
    print(f'  Max DD   W: {baseline_dd*100:+.2f}%')

    K_base = np.array([K_REGIME[r] for r in regime])

    n = len(pred)
    rng_master = np.random.default_rng(20260430)

    perm_cum = np.empty(N_PERM)
    perm_sortino = np.empty(N_PERM)
    perm_sharpe = np.empty(N_PERM)
    perm_dd = np.empty(N_PERM)

    print(f'\nRunning {N_PERM:,} permutations...')
    for i in range(N_PERM):
        idx = rng_master.permutation(n)
        pred_s = pred[idx]
        p_up_s = p_up[idx]
        conf_s = 1.0 / (1.0 + np.exp(-np.abs(p_up_s - 0.5) * SIGMOID_SCALE))
        alloc_s = np.clip(pred_s * K_base * conf_s, 0.0, 1.0)
        strat_s = alloc_s * btc_fwd + (1 - alloc_s) * cdi
        c, so, sh, d = metrics(strat_s)
        perm_cum[i] = c
        perm_sortino[i] = so
        perm_sharpe[i] = sh
        perm_dd[i] = d
        if (i + 1) % 1000 == 0:
            print(f'  {i+1:>6,}/{N_PERM:,} done')

    pct_beat_cum = (perm_cum > baseline_cum).mean()
    pct_beat_sortino = (perm_sortino > baseline_sortino).mean()
    pct_beat_sharpe = (perm_sharpe > baseline_sharpe).mean()

    print(f'\nPERMUTATION RESULTS (N={N_PERM:,}):')
    print(f'  Cum return  -- mean={perm_cum.mean()*100:+.1f}%  p95={np.percentile(perm_cum,95)*100:+.1f}%  max={perm_cum.max()*100:+.1f}%')
    print(f'                 fraction beating baseline ({baseline_cum*100:+.1f}%): {pct_beat_cum*100:.3f}% ({int(pct_beat_cum*N_PERM)}/{N_PERM})')
    print(f'  Sortino W   -- mean={perm_sortino.mean():.2f}  p95={np.percentile(perm_sortino,95):.2f}  max={perm_sortino.max():.2f}')
    print(f'                 fraction beating baseline ({baseline_sortino:.2f}): {pct_beat_sortino*100:.3f}% ({int(pct_beat_sortino*N_PERM)}/{N_PERM})')
    print(f'  Sharpe W    -- fraction beating baseline ({baseline_sharpe:.2f}): {pct_beat_sharpe*100:.3f}% ({int(pct_beat_sharpe*N_PERM)}/{N_PERM})')

    perm_df = pd.DataFrame({
        'cum': perm_cum,
        'sortino_w': perm_sortino,
        'sharpe_w': perm_sharpe,
        'dd_w': perm_dd,
    })
    perm_df.to_csv(OUT_DIR / 'permutation_10k.csv', index=False)

    summary = {
        'n_permutations': N_PERM,
        'baseline': {
            'cum_pct': float(baseline_cum * 100),
            'sortino_weekly': float(baseline_sortino),
            'sharpe_weekly': float(baseline_sharpe),
            'max_dd_weekly_pct': float(baseline_dd * 100),
        },
        'permutation_distribution': {
            'cum_pct_mean': float(perm_cum.mean() * 100),
            'cum_pct_std': float(perm_cum.std() * 100),
            'cum_pct_p95': float(np.percentile(perm_cum, 95) * 100),
            'cum_pct_p99': float(np.percentile(perm_cum, 99) * 100),
            'cum_pct_max': float(perm_cum.max() * 100),
            'sortino_w_mean': float(perm_sortino.mean()),
            'sortino_w_p95': float(np.percentile(perm_sortino, 95)),
            'sortino_w_max': float(perm_sortino.max()),
        },
        'p_values': {
            'cum_return': float(pct_beat_cum),
            'sortino_weekly': float(pct_beat_sortino),
            'sharpe_weekly': float(pct_beat_sharpe),
        },
        'n_beating_baseline': {
            'cum_return': int((perm_cum > baseline_cum).sum()),
            'sortino_weekly': int((perm_sortino > baseline_sortino).sum()),
            'sharpe_weekly': int((perm_sharpe > baseline_sharpe).sum()),
        },
    }
    with open(OUT_DIR / 'permutation_10k_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nSaved:')
    print(f'  {OUT_DIR / "permutation_10k.csv"}')
    print(f'  {OUT_DIR / "permutation_10k_summary.json"}')


if __name__ == '__main__':
    main()
