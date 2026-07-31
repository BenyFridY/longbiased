"""
TESTES RAPIDOS — usam predicoes existentes (sem re-treinar):

TESTE 3: Sigmoid sensitivity — SIGMOID_SCALE=5/10/15/20/25/50/100
TESTE 4: Shuffled predictions — quebra alinhamento pred-actual, mede structural edge
TESTE 5: Random predictions — substitui preds por ruido, mede regime edge
TESTE 6: Signals only (sign) — usa so o sinal de pred, nao magnitude
TESTE 7: Constant predictions — pred fixa (+1%), mede regime edge puro

Output: outputs/results/overfit_tests/test_fast.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
INPUT = ROOT / 'outputs' / 'results' / 'horizon_ablation_4y.csv'
OUT = ROOT / 'outputs' / 'results' / 'overfit_tests' / 'test_fast.csv'
OUT.parent.mkdir(parents=True, exist_ok=True)

K_REGIME = {'BULL': 100, 'MILD': 50, 'BEAR': 20}  # H2 current
K_REGIME_H1 = {'BULL': 60, 'MILD': 30, 'BEAR': 15}  # H1 lower risk

def metrics(arr, ppy=52):
    arr = np.array(arr)
    cum = (1 + arr).prod() - 1
    ann = arr.mean() * ppy
    down = arr[arr < 0]
    sortino = (arr.mean() / np.sqrt((down**2).mean())) * np.sqrt(ppy) if len(down) > 0 and (down**2).mean() > 0 else float('inf')
    sharpe = ((arr.mean() - 0.0021) / arr.std()) * np.sqrt(ppy) if arr.std() > 0 else 0
    cum_series = (1+arr).cumprod()
    dd = ((cum_series - np.maximum.accumulate(cum_series)) / np.maximum.accumulate(cum_series)).min()
    return round(cum*100, 1), round(sortino, 2), round(sharpe, 2), round(dd*100, 2), round(ann*100, 1)


def simulate(pred, p_up, regime, btc_fwd, cdi_proxy, K_map, sigmoid_scale=15, floor=0, ceil=1):
    conf = 1 / (1 + np.exp(-np.abs(p_up - 0.5) * sigmoid_scale))
    K_vec = np.array([K_map[r] for r in regime])
    alloc = np.clip(pred * K_vec * conf, floor, ceil)
    strat = alloc * btc_fwd + (1 - alloc) * cdi_proxy
    return alloc, strat


def main():
    df = pd.read_csv(INPUT, parse_dates=['date'])
    df = df[df['variant'] == 'H=3'].reset_index(drop=True).copy()
    print(f'Loaded {len(df)} rebals\n')

    # Implied CDI per rebal
    cdi_implied = np.where(
        (1 - df['alloc']) > 0.001,
        (df['strat'] - df['alloc']*df['btc_fwd']) / (1 - df['alloc']),
        0.0021
    )

    pred = df['pred'].values.copy()
    p_up = df['p_up'].values.copy()
    regime = df['regime'].values.copy()
    btc_fwd = df['btc_fwd'].values.copy()

    results = []

    # ========================================================================
    # TESTE 3: SIGMOID SCALE SENSITIVITY (K=H2)
    # ========================================================================
    print('=== TESTE 3: SIGMOID SCALE SENSITIVITY (K=H2 100/50/20) ===')
    for scale in [1, 5, 10, 15, 20, 25, 50, 100, 1000]:
        _, strat = simulate(pred, p_up, regime, btc_fwd, cdi_implied, K_REGIME, sigmoid_scale=scale)
        cum, sor, sha, dd, ann = metrics(strat)
        results.append({'test': 'T3_sigmoid', 'config': f'scale={scale}', 'cum_%': cum, 'sortino': sor, 'sharpe': sha, 'max_dd_%': dd, 'ann_%': ann})
        print(f'  scale={scale:4d}: cum={cum:+7.1f}%  Sortino={sor:.2f}  Sharpe={sha:.2f}  DD={dd:+.2f}%')

    # ========================================================================
    # TESTE 4: SHUFFLED PREDICTIONS (100 seeds)
    # ========================================================================
    print('\n=== TESTE 4: SHUFFLED PREDICTIONS (quebra pred-actual correlation, 100 seeds) ===')
    shuffle_results = []
    for seed in range(100):
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(pred))
        pred_s = pred[idx]
        p_up_s = p_up[idx]
        # mantém regime original (alinhado com btc_fwd) — só embaralha predictions
        _, strat = simulate(pred_s, p_up_s, regime, btc_fwd, cdi_implied, K_REGIME)
        cum, sor, sha, dd, _ = metrics(strat)
        shuffle_results.append({'cum': cum, 'sortino': sor, 'sharpe': sha, 'dd': dd})
    shuffle_df = pd.DataFrame(shuffle_results)
    print(f'  N=100 seeds. Shuffled pred results:')
    print(f'  cum ret:  mean={shuffle_df.cum.mean():+.1f}%  std={shuffle_df.cum.std():.1f}  p95={shuffle_df.cum.quantile(0.95):+.1f}  max={shuffle_df.cum.max():+.1f}')
    print(f'  Sortino:  mean={shuffle_df.sortino.mean():.2f}  std={shuffle_df.sortino.std():.2f}  p95={shuffle_df.sortino.quantile(0.95):.2f}')
    print(f'  Sharpe:   mean={shuffle_df.sharpe.mean():.2f}  std={shuffle_df.sharpe.std():.2f}  p95={shuffle_df.sharpe.quantile(0.95):.2f}')
    print(f'  Actual strategy (non-shuffled): cum=+1191%  Sortino=5.61  Sharpe=2.09')
    # What fraction of shuffles beat actual?
    actual_cum = 1191.1
    actual_sortino = 5.61
    pct_beat_cum = (shuffle_df.cum > actual_cum).mean() * 100
    pct_beat_sortino = (shuffle_df.sortino > actual_sortino).mean() * 100
    print(f'  % shuffles beating actual return: {pct_beat_cum:.1f}%')
    print(f'  % shuffles beating actual Sortino: {pct_beat_sortino:.1f}%')
    results.append({'test': 'T4_shuffle', 'config': 'shuffle_mean', 'cum_%': round(shuffle_df.cum.mean(),1), 'sortino': round(shuffle_df.sortino.mean(),2), 'sharpe': round(shuffle_df.sharpe.mean(),2), 'max_dd_%': round(shuffle_df.dd.mean(),2), 'ann_%': 0})
    results.append({'test': 'T4_shuffle', 'config': 'shuffle_p95', 'cum_%': round(shuffle_df.cum.quantile(0.95),1), 'sortino': round(shuffle_df.sortino.quantile(0.95),2), 'sharpe': round(shuffle_df.sharpe.quantile(0.95),2), 'max_dd_%': round(shuffle_df.dd.quantile(0.05),2), 'ann_%': 0})

    # ========================================================================
    # TESTE 5: RANDOM PREDICTIONS (pure regime + random signal)
    # ========================================================================
    print('\n=== TESTE 5: RANDOM PREDICTIONS (100 seeds) ===')
    # Substitute pred with random N(mean=real_mean, std=real_std) and p_up uniform [0,1]
    random_results = []
    for seed in range(100):
        rng = np.random.RandomState(seed)
        pred_r = rng.normal(pred.mean(), pred.std(), len(pred))
        p_up_r = rng.uniform(0, 1, len(pred))
        _, strat = simulate(pred_r, p_up_r, regime, btc_fwd, cdi_implied, K_REGIME)
        cum, sor, sha, dd, _ = metrics(strat)
        random_results.append({'cum': cum, 'sortino': sor, 'sharpe': sha, 'dd': dd})
    rand_df = pd.DataFrame(random_results)
    print(f'  N=100 seeds of random pred (N(μ,σ)):')
    print(f'  cum ret:  mean={rand_df.cum.mean():+.1f}%  std={rand_df.cum.std():.1f}  p95={rand_df.cum.quantile(0.95):+.1f}  max={rand_df.cum.max():+.1f}')
    print(f'  Sortino:  mean={rand_df.sortino.mean():.2f}  std={rand_df.sortino.std():.2f}  p95={rand_df.sortino.quantile(0.95):.2f}')
    print(f'  Sharpe:   mean={rand_df.sharpe.mean():.2f}  std={rand_df.sharpe.std():.2f}  p95={rand_df.sharpe.quantile(0.95):.2f}')
    results.append({'test': 'T5_random', 'config': 'random_mean', 'cum_%': round(rand_df.cum.mean(),1), 'sortino': round(rand_df.sortino.mean(),2), 'sharpe': round(rand_df.sharpe.mean(),2), 'max_dd_%': round(rand_df.dd.mean(),2), 'ann_%': 0})

    # ========================================================================
    # TESTE 6: SIGN-ONLY PREDICTION (no magnitude)
    # ========================================================================
    print('\n=== TESTE 6: SIGN-ONLY PREDICTIONS (magnitude descartada) ===')
    # Replace pred with just its sign × avg magnitude
    pred_sign = np.sign(pred) * np.abs(pred).mean()
    _, strat_sign = simulate(pred_sign, p_up, regime, btc_fwd, cdi_implied, K_REGIME)
    cum, sor, sha, dd, ann = metrics(strat_sign)
    print(f'  Sign-only (×mean|pred|): cum={cum:+.1f}%  Sortino={sor}  Sharpe={sha}  DD={dd}%')
    results.append({'test': 'T6_sign_only', 'config': 'sign×avg_mag', 'cum_%': cum, 'sortino': sor, 'sharpe': sha, 'max_dd_%': dd, 'ann_%': ann})

    # ========================================================================
    # TESTE 7: CONSTANT POSITIVE PRED (+1% always)
    # ========================================================================
    print('\n=== TESTE 7: CONSTANT POSITIVE PRED (pure regime edge) ===')
    pred_const = np.ones(len(pred)) * 0.01  # always expect +1%
    p_up_const = np.ones(len(pred)) * 0.65  # high confidence
    _, strat_const = simulate(pred_const, p_up_const, regime, btc_fwd, cdi_implied, K_REGIME)
    cum, sor, sha, dd, ann = metrics(strat_const)
    print(f'  Constant +1% pred, p_up=0.65: cum={cum:+.1f}%  Sortino={sor}  Sharpe={sha}  DD={dd}%')
    results.append({'test': 'T7_const_pos', 'config': 'const_+1%', 'cum_%': cum, 'sortino': sor, 'sharpe': sha, 'max_dd_%': dd, 'ann_%': ann})

    # ========================================================================
    # TESTE 8: No regime — fixed K=50 always (no regime filter)
    # ========================================================================
    print('\n=== TESTE 8: K=50 CONSTANT (no regime) ===')
    K_flat = {'BULL': 50, 'MILD': 50, 'BEAR': 50}
    _, strat_flat = simulate(pred, p_up, regime, btc_fwd, cdi_implied, K_flat)
    cum, sor, sha, dd, ann = metrics(strat_flat)
    print(f'  K=50 always (regime ignored): cum={cum:+.1f}%  Sortino={sor}  Sharpe={sha}  DD={dd}%')
    results.append({'test': 'T8_no_regime', 'config': 'K=50_all', 'cum_%': cum, 'sortino': sor, 'sharpe': sha, 'max_dd_%': dd, 'ann_%': ann})

    # Save
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT, index=False)
    print()
    print(f'Saved: {OUT}')
    print('\n' + '='*80)
    print('RESUMO DE TODOS OS TESTES RAPIDOS')
    print('='*80)
    print(res_df.to_string(index=False))


if __name__ == '__main__':
    main()
