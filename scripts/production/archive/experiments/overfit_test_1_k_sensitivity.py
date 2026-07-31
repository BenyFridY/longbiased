"""
TESTE 1: K sensitivity — overfit probe

Objetivo: Usar predicoes existentes (horizon_ablation_4y.csv) e simular 7
configuracoes de K distintas, mantendo o mesmo modelo (mesmas predicoes).

Tese: Se o modelo tem edge REAL, Sortino deve ser monotonicamente relacionado
a K (tradeoff return vs risco). Se Sortino pica em um K especifico e cai nas
laterais, K foi otimizado para maximizar ajuste ex-post (sinal de overfit).

Output: outputs/results/overfit_tests/test1_k_sensitivity.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
INPUT = ROOT / 'outputs' / 'results' / 'horizon_ablation_4y.csv'
OUT = ROOT / 'outputs' / 'results' / 'overfit_tests' / 'test1_k_sensitivity.csv'
OUT.parent.mkdir(parents=True, exist_ok=True)

SIGMOID_SCALE = 15  # frozen, NOT varying

def sortino(r, ppy=52):
    arr = np.array(r)
    down = arr[arr < 0]
    if len(down) == 0:
        return float('inf')
    return (arr.mean() / np.sqrt((down ** 2).mean())) * np.sqrt(ppy)

def sharpe(r, ppy=52, rf_weekly=0.0021):
    arr = np.array(r)
    if arr.std() == 0: return 0
    return ((arr.mean() - rf_weekly) / arr.std()) * np.sqrt(ppy)

def max_dd(r):
    cum = (1 + np.array(r)).cumprod()
    peak = np.maximum.accumulate(cum)
    return ((cum - peak) / peak).min()

def annual_ret(r, ppy=52):
    return np.array(r).mean() * ppy

def simulate_k(df, K_map, floor=0.0, ceil=1.0):
    """Given fixed predictions, simulate allocation with new K config."""
    conf = 1 / (1 + np.exp(-np.abs(df['p_up'] - 0.5) * SIGMOID_SCALE))
    K_vec = df['regime'].map(K_map).values
    alloc = np.clip(df['pred'].values * K_vec * conf, floor, ceil)
    # Use implied CDI to reconstruct strat return
    orig_alloc = df['alloc'].values
    orig_strat = df['strat'].values
    orig_btc = df['btc_fwd'].values
    cdi_implied = np.where(
        (1 - orig_alloc) > 0.001,
        (orig_strat - orig_alloc * orig_btc) / (1 - orig_alloc),
        0.0021  # fallback to ~11% annual
    )
    strat = alloc * orig_btc + (1 - alloc) * cdi_implied
    return alloc, strat


def main():
    df = pd.read_csv(INPUT, parse_dates=['date'])
    df = df[df['variant'] == 'H=3'].reset_index(drop=True).copy()
    print(f'Loaded {len(df)} rebals from 2022-01 to 2026-04')
    print()

    # Configurations to test (expanded to probe peak)
    configs = [
        ('K=200/100/40 (super-aggressive)', {'BULL': 200, 'MILD': 100, 'BEAR': 40}, 0.0),
        ('K=150/75/30 (aggressive)',  {'BULL': 150, 'MILD': 75, 'BEAR': 30}, 0.0),
        ('K=100/50/20 (H2 atual)',    {'BULL': 100, 'MILD': 50, 'BEAR': 20}, 0.0),
        ('K=80/40/15 (H2 ajustado)',  {'BULL': 80,  'MILD': 40, 'BEAR': 15}, 0.0),
        ('K=60/30/15 (H1 menor risco)', {'BULL': 60, 'MILD': 30, 'BEAR': 15}, 0.0),
        ('K=50/25/10 (meio termo)',   {'BULL': 50,  'MILD': 25, 'BEAR': 10}, 0.0),
        ('K=40/20/10 (conservador)',  {'BULL': 40,  'MILD': 20, 'BEAR': 10}, 0.0),
        ('K=30/15/7 (menos-agressivo)', {'BULL': 30, 'MILD': 15, 'BEAR': 7}, 0.0),
        ('K=20/10/5 (ultra-conserv)', {'BULL': 20,  'MILD': 10, 'BEAR': 5},  0.0),
        ('K=10/5/2 (minimo)',         {'BULL': 10,  'MILD': 5,  'BEAR': 2},  0.0),
    ]

    results = []
    for label, K_map, floor in configs:
        alloc, strat = simulate_k(df, K_map, floor)
        cum = (1 + pd.Series(strat)).prod() - 1
        sor = sortino(strat)
        sha = sharpe(strat)
        dd = max_dd(strat)
        ann = annual_ret(strat)
        avg_alloc = alloc.mean()
        pct_max = (alloc >= 0.99).mean()
        pct_zero = (alloc <= 0.01).mean()
        calmar = ann / abs(dd) if dd < 0 else float('inf')
        # BTC underlying return
        btc_cum = (1 + df['btc_fwd']).prod() - 1
        results.append({
            'config': label,
            'K_BULL': K_map['BULL'],
            'K_MILD': K_map['MILD'],
            'K_BEAR': K_map['BEAR'],
            'cum_return_%': round(cum*100, 1),
            'vs_btc_pp': round((cum - btc_cum)*100, 1),
            'sortino': round(sor, 2),
            'sharpe': round(sha, 2),
            'calmar': round(calmar, 1),
            'max_dd_%': round(dd*100, 2),
            'annual_ret_%': round(ann*100, 1),
            'avg_alloc_%': round(avg_alloc*100, 1),
            'pct_maxed': round(pct_max*100, 1),
            'pct_zero': round(pct_zero*100, 1),
        })

    res_df = pd.DataFrame(results)
    print('=' * 110)
    print('TESTE 1: K SENSITIVITY — Mesmo modelo, 7 configuracoes de K')
    print('=' * 110)
    print(res_df.to_string(index=False))
    print()
    res_df.to_csv(OUT, index=False)
    print(f'Saved: {OUT}')

    # Verdict
    sortino_max = res_df['sortino'].max()
    sortino_mean = res_df['sortino'].mean()
    sortino_min = res_df['sortino'].min()
    print()
    print('VEREDICTO:')
    print(f'  Sortino max: {sortino_max:.2f} (config {res_df.loc[res_df.sortino.idxmax(), "config"]})')
    print(f'  Sortino min: {sortino_min:.2f}')
    print(f'  Sortino mean: {sortino_mean:.2f}')
    print(f'  Range Sortino: {sortino_max - sortino_min:.2f}')
    # If Sortino holds across configs -> edge is robust
    # If one config pops, rest drop -> K overfit


if __name__ == '__main__':
    main()
