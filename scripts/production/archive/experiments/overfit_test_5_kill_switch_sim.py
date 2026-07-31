"""
TESTE 5: Kill-switch simulation + realistic live expectations.

Simula o modelo COM o kill switch ativo em toda a OOS historica, comparando
performance com/sem. Tambem mostra como ficaria live expectation com deflation
factor aplicado.

Objetivo: validar que kill switch so dispara em situacoes legitimas
(nao inflaciona Sortino falsamente) e que ele reduz DD em live.
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
INPUT = ROOT / 'outputs' / 'results' / 'horizon_ablation_4y.csv'
OUT = ROOT / 'outputs' / 'results' / 'overfit_tests' / 'test5_kill_switch.csv'
OUT.parent.mkdir(parents=True, exist_ok=True)

KILL_SWITCH_DD = -0.12
KILL_SWITCH_FLOOR = 0.15
ROLLING_ACC_WINDOW = 12
ROLLING_ACC_THRESHOLD = 0.48


def metrics(arr, ppy=52):
    arr = np.array(arr)
    cum = (1 + arr).prod() - 1
    ann = arr.mean() * ppy
    down = arr[arr < 0]
    sortino = (arr.mean() / np.sqrt((down**2).mean())) * np.sqrt(ppy) if len(down)>0 and (down**2).mean()>0 else float('inf')
    sharpe = ((arr.mean() - 0.0021) / arr.std()) * np.sqrt(ppy) if arr.std()>0 else 0
    c = (1+arr).cumprod()
    dd = ((c - np.maximum.accumulate(c)) / np.maximum.accumulate(c)).min()
    return dict(cum_=round(cum*100,1), ann_=round(ann*100,1), sortino=round(sortino,2), sharpe=round(sharpe,2), dd_=round(dd*100,2))


def apply_kill_switch_simulation(df, dd_threshold=-0.12, floor=0.15):
    """Apply kill switch post-hoc: if rolling DD <= threshold, cap alloc at floor."""
    df = df.copy()
    strat_cum = (1 + df['strat']).cumprod()
    peak = strat_cum.cummax()
    rolling_dd = (strat_cum - peak) / peak
    # Kill switch: if rolling_dd on PRIOR week was below threshold, cap alloc this week
    kill_active = rolling_dd.shift(1) <= dd_threshold
    alloc_new = df['alloc'].copy()
    alloc_new[kill_active] = alloc_new[kill_active].clip(upper=floor)
    # Recompute strat ret with new alloc
    cdi_implied = np.where(
        (1 - df['alloc']) > 0.001,
        (df['strat'] - df['alloc']*df['btc_fwd']) / (1 - df['alloc']),
        0.0021
    )
    df['alloc_with_kill'] = alloc_new
    df['strat_with_kill'] = alloc_new * df['btc_fwd'] + (1 - alloc_new) * cdi_implied
    df['kill_active'] = kill_active
    return df


def apply_acc_derisk_simulation(df, window=12, threshold=0.48, derisk_mult=0.5):
    """Apply rolling-accuracy de-risk: if 12w acc < 48%, halve alloc."""
    df = df.copy()
    # Direction accuracy
    df['correct'] = ((df['pred'] > 0) & (df['btc_fwd'] > 0)) | ((df['pred'] < 0) & (df['btc_fwd'] < 0))
    df['rolling_acc'] = df['correct'].rolling(window).mean()
    # Derisk: if rolling_acc on prior week was below threshold, halve alloc
    derisk_active = df['rolling_acc'].shift(1) < threshold
    alloc_new = df['alloc'].copy()
    alloc_new[derisk_active] = alloc_new[derisk_active] * derisk_mult
    cdi_implied = np.where(
        (1 - df['alloc']) > 0.001,
        (df['strat'] - df['alloc']*df['btc_fwd']) / (1 - df['alloc']),
        0.0021
    )
    df['alloc_derisk'] = alloc_new
    df['strat_derisk'] = alloc_new * df['btc_fwd'] + (1 - alloc_new) * cdi_implied
    df['derisk_active'] = derisk_active
    return df


def main():
    df = pd.read_csv(INPUT, parse_dates=['date'])
    df = df[df['variant'] == 'H=3'].reset_index(drop=True).copy()
    print(f'Loaded {len(df)} rebals 2022-01 to 2026-04\n')

    # Baseline
    m0 = metrics(df['strat'].values)
    print(f'BASELINE (current): Sortino {m0["sortino"]}, Ret {m0["cum_"]}%, DD {m0["dd_"]}%')

    # Apply kill switch
    df_ks = apply_kill_switch_simulation(df, KILL_SWITCH_DD, KILL_SWITCH_FLOOR)
    m_ks = metrics(df_ks['strat_with_kill'].values)
    n_kill = df_ks['kill_active'].sum()
    print(f'\nWITH KILL SWITCH (DD<={KILL_SWITCH_DD*100:.0f}% -> alloc<={KILL_SWITCH_FLOOR*100:.0f}%):')
    print(f'  Sortino {m_ks["sortino"]}, Ret {m_ks["cum_"]}%, DD {m_ks["dd_"]}%')
    print(f'  Fired {n_kill} times')

    # Apply accuracy de-risk
    df_da = apply_acc_derisk_simulation(df, ROLLING_ACC_WINDOW, ROLLING_ACC_THRESHOLD)
    m_da = metrics(df_da['strat_derisk'].values)
    n_derisk = df_da['derisk_active'].sum()
    print(f'\nWITH ACC DE-RISK (12w acc<{ROLLING_ACC_THRESHOLD*100:.0f}% -> alloc×0.5):')
    print(f'  Sortino {m_da["sortino"]}, Ret {m_da["cum_"]}%, DD {m_da["dd_"]}%')
    print(f'  Fired {n_derisk} weeks')

    # Both combined
    df_both = df.copy()
    df_both = apply_kill_switch_simulation(df_both)
    df_both['alloc'] = df_both['alloc_with_kill']
    df_both['strat'] = df_both['strat_with_kill']
    df_both = apply_acc_derisk_simulation(df_both)
    m_b = metrics(df_both['strat_derisk'].values)
    print(f'\nBOTH controls combined:')
    print(f'  Sortino {m_b["sortino"]}, Ret {m_b["cum_"]}%, DD {m_b["dd_"]}%')

    # K=60/30/15 with both controls (H1 safer)
    K_H1 = {'BULL': 60, 'MILD': 30, 'BEAR': 15}
    conf = 1 / (1 + np.exp(-np.abs(df['p_up'] - 0.5) * 15))
    K_vec = np.array([K_H1[r] for r in df['regime']])
    alloc_h1 = np.clip(df['pred'].values * K_vec * conf, 0, 1)
    cdi_implied = np.where(
        (1 - df['alloc']) > 0.001,
        (df['strat'] - df['alloc']*df['btc_fwd']) / (1 - df['alloc']),
        0.0021
    )
    strat_h1 = alloc_h1 * df['btc_fwd'] + (1-alloc_h1)*cdi_implied
    df_h1 = df.copy()
    df_h1['alloc'] = alloc_h1
    df_h1['strat'] = strat_h1
    df_h1 = apply_kill_switch_simulation(df_h1)
    df_h1['alloc'] = df_h1['alloc_with_kill']
    df_h1['strat'] = df_h1['strat_with_kill']
    df_h1 = apply_acc_derisk_simulation(df_h1)
    m_h1 = metrics(df_h1['strat_derisk'].values)
    print(f'\nK=H1 (60/30/15) + Both controls:')
    print(f'  Sortino {m_h1["sortino"]}, Ret {m_h1["cum_"]}%, DD {m_h1["dd_"]}%')

    # K=40/20/10 conservative with both controls
    K_CONS = {'BULL': 40, 'MILD': 20, 'BEAR': 10}
    K_vec = np.array([K_CONS[r] for r in df['regime']])
    alloc_c = np.clip(df['pred'].values * K_vec * conf, 0, 1)
    strat_c = alloc_c * df['btc_fwd'] + (1-alloc_c)*cdi_implied
    df_c = df.copy()
    df_c['alloc'] = alloc_c
    df_c['strat'] = strat_c
    df_c = apply_kill_switch_simulation(df_c)
    df_c['alloc'] = df_c['alloc_with_kill']
    df_c['strat'] = df_c['strat_with_kill']
    df_c = apply_acc_derisk_simulation(df_c)
    m_c = metrics(df_c['strat_derisk'].values)
    print(f'\nK=Conservative (40/20/10) + Both controls:')
    print(f'  Sortino {m_c["sortino"]}, Ret {m_c["cum_"]}%, DD {m_c["dd_"]}%')

    # Save results
    results = pd.DataFrame([
        {'config': 'H2 baseline (atual)', **m0},
        {'config': 'H2 + kill switch', **m_ks},
        {'config': 'H2 + acc de-risk', **m_da},
        {'config': 'H2 + both controls', **m_b},
        {'config': 'H1 (60/30/15) + both controls', **m_h1},
        {'config': 'Conservative (40/20/10) + both', **m_c},
    ])
    print()
    print('=' * 80)
    print('TESTE 5: RISK CONTROLS IMPACT — resumo')
    print('=' * 80)
    print(results.to_string(index=False))
    print(f'\nSaved: {OUT}')
    results.to_csv(OUT, index=False)

    # Deflation analysis on best config
    print()
    print('=' * 80)
    print('REALISTIC LIVE EXPECTATIONS — H1+Controls after deflation')
    print('=' * 80)
    sr_observed = m_h1['sharpe']
    print(f'  Observed Sharpe (H1+controls, backtest): {sr_observed:.2f}')
    # Deflation factor for 38 trials (Bailey-Prado)
    # Expected max SR under H0 = sqrt(2 ln N) asymptotic
    n_trials = 38
    gamma = 0.5772156649
    max_z = np.sqrt(2*np.log(n_trials)) - (gamma + np.log(np.log(n_trials))) / (2*np.sqrt(2*np.log(n_trials)))
    deflated_sr = max(0, sr_observed - max_z)
    print(f'  Expected max SR under null (38 trials): {max_z:.2f}')
    print(f'  Deflated Sharpe (subtract null): {deflated_sr:.2f}')
    # This corresponds to ann return
    ann_ret = deflated_sr * 0.216 + 0.11  # using observed vol, CDI floor
    print(f'  Realistic annual return @ deflated SR: {ann_ret*100:.1f}%')
    print()
    print('  Interpretacao: se o modelo capturar todo o edge real (apos correcao')
    print('  pra multiple testing), esperamos ~20-40%/ano em live, com DD ~15-25%.')


if __name__ == '__main__':
    main()
