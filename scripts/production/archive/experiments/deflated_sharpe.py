"""
Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).

Corrige o Sharpe por:
  1. Selecao ex-post (numero de configuracoes testadas)
  2. Nao-normalidade (skewness, kurtosis dos retornos)

Tambem computa Probability of Backtest Overfitting (PBO) via CSCV
(Combinatorially Symmetric Cross-Validation).

Uso:
    python scripts/production/archive/experiments/deflated_sharpe.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent


def sharpe_ratio(returns, ppy=52, rf=0.0021):
    """Annualized Sharpe from weekly returns."""
    r = np.asarray(returns)
    if r.std() == 0:
        return 0.0
    return (r.mean() - rf) / r.std() * np.sqrt(ppy)


def probabilistic_sharpe_ratio(returns, sr_benchmark=0.0, ppy=52, rf=0.0021):
    """PSR: prob that TRUE Sharpe > benchmark, given observed.

    Accounts for non-Normality (skewness, kurtosis). Bailey/Prado 2012.
    Returns value in [0, 1] — interpret as confidence.
    """
    r = np.asarray(returns)
    n = len(r)
    if n < 3:
        return 0.5
    sr = sharpe_ratio(r, ppy, rf) / np.sqrt(ppy)  # convert to per-period
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=True)  # excess kurtosis
    denom = np.sqrt((1 - skew * sr + (kurt - 1) / 4 * sr**2) / (n - 1))
    if denom == 0:
        return 0.5
    z = (sr - sr_benchmark / np.sqrt(ppy)) / denom
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(returns, n_trials, var_sr_trials=None, ppy=52, rf=0.0021):
    """Deflated Sharpe — corrects for multiple testing.

    The expected maximum Sharpe under H0 (no skill) from N trials grows like
    sqrt(2 ln N). We deflate the PSR using this benchmark instead of 0.

    Args:
        returns: weekly strategy returns (array)
        n_trials: number of strategy variants/configs tried
        var_sr_trials: sample variance of Sharpe across trials. If None,
                       assumes the SR distribution follows theoretical.
        ppy: periods per year (52 = weekly)
        rf: weekly risk-free rate

    Returns:
        dsr: probability that the observed Sharpe is real
    """
    r = np.asarray(returns)
    # Expected max Sharpe under H0 with N trials (Bailey & Prado 2014 eq.8)
    gamma = 0.5772156649  # Euler-Mascheroni
    max_z = np.sqrt(2 * np.log(max(n_trials, 1))) - \
            (gamma + np.log(np.log(max(n_trials, 1)))) / (2 * np.sqrt(2 * np.log(max(n_trials, 1)))) \
            if n_trials > 1 else 0.0
    if var_sr_trials is None:
        # Assume theoretical variance from normal SR
        var_sr_trials = 1.0  # annualized unit variance assumption
    sr_benchmark = max_z * np.sqrt(var_sr_trials)  # annualized expected benchmark
    return probabilistic_sharpe_ratio(r, sr_benchmark, ppy, rf)


def pbo_cscv(perf_matrix, n_splits=16):
    """Probability of Backtest Overfitting (CSCV).

    perf_matrix: shape (T, N) where T = time periods, N = strategy variants.
                 cell (t, j) is variant j's return in period t.

    Returns: PBO (prob the best in-sample strategy underperforms median OOS).
    """
    from itertools import combinations
    T, N = perf_matrix.shape
    if T < n_splits * 2 or N < 2:
        return np.nan
    # Split T into n_splits (must be even)
    if n_splits % 2:
        n_splits -= 1
    split_size = T // n_splits
    if split_size < 1:
        return np.nan
    # Truncate to fit exactly
    M = perf_matrix[:split_size * n_splits]
    parts = np.array(np.vsplit(M, n_splits))  # (n_splits, split_size, N)

    half = n_splits // 2
    overfit_count = 0
    total = 0
    # All combinations of choosing `half` parts for in-sample
    for in_idx in combinations(range(n_splits), half):
        out_idx = [i for i in range(n_splits) if i not in in_idx]
        in_data = np.vstack([parts[i] for i in in_idx])
        out_data = np.vstack([parts[i] for i in out_idx])
        in_sharpes = np.array([sharpe_ratio(in_data[:, j]) for j in range(N)])
        out_sharpes = np.array([sharpe_ratio(out_data[:, j]) for j in range(N)])
        best_in = np.argmax(in_sharpes)
        # Rank of best_in in OOS
        out_rank = np.mean(out_sharpes < out_sharpes[best_in])
        if out_rank < 0.5:
            overfit_count += 1
        total += 1
    return float(overfit_count / total) if total > 0 else np.nan


def main():
    print('=' * 80)
    print('DEFLATED SHARPE RATIO — Bailey & Lopez de Prado (2014)')
    print('=' * 80)

    # Load strategy returns from horizon_ablation_4y (H=3 variant, 248 weekly rebals)
    ha = pd.read_csv(ROOT / 'outputs' / 'results' / 'horizon_ablation_4y.csv', parse_dates=['date'])
    h3 = ha[ha['variant'] == 'H=3'].reset_index(drop=True)

    returns = h3['strat'].values
    n = len(returns)
    sr = sharpe_ratio(returns)
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns, fisher=True)
    down = returns[returns < 0]
    sortino = (returns.mean() / np.sqrt((down**2).mean())) * np.sqrt(52) if len(down) > 0 else float('inf')

    print(f'\nObserved metrics (H=3, H2 config, {n} rebals):')
    print(f'  Sharpe (observed):  {sr:.3f}')
    print(f'  Sortino:            {sortino:.3f}')
    print(f'  Skewness:           {skew:+.3f}')
    print(f'  Excess kurtosis:    {kurt:+.3f}')
    print(f'  Mean weekly ret:    {returns.mean()*100:+.3f}%')
    print(f'  Vol weekly:         {returns.std()*100:.3f}%')
    print(f'  Annualized:         {returns.mean()*52*100:+.1f}% / {returns.std()*np.sqrt(52)*100:.1f}% vol')

    # Probabilistic Sharpe (no deflation)
    psr_vs_0 = probabilistic_sharpe_ratio(returns, sr_benchmark=0, ppy=52)
    print(f'\nProbabilistic Sharpe Ratio (vs SR=0):  {psr_vs_0*100:.2f}%')
    print(f'  → prob that true SR > 0')

    # PSR vs SR = 1 (modest benchmark)
    psr_vs_1 = probabilistic_sharpe_ratio(returns, sr_benchmark=1, ppy=52)
    print(f'\nProbabilistic Sharpe Ratio (vs SR=1):  {psr_vs_1*100:.2f}%')
    print(f'  → prob that true SR > 1 (modest hurdle)')

    # Deflated Sharpe — N trials estimates
    print('\n' + '=' * 80)
    print('DEFLATED SHARPE — varying trial count estimates')
    print('=' * 80)
    print(f'{"Trials N":<12} {"Expected max SR (H0)":<22} {"Deflated Sharpe (prob)":<25}')
    print('-' * 80)
    for n_trials in [1, 5, 10, 38, 100, 500, 1000]:
        gamma = 0.5772156649
        if n_trials > 1:
            max_z = np.sqrt(2*np.log(n_trials)) - (gamma + np.log(np.log(n_trials))) / (2*np.sqrt(2*np.log(n_trials)))
        else:
            max_z = 0.0
        dsr = deflated_sharpe_ratio(returns, n_trials)
        marker = '  ← repo has ~38 versions tested' if n_trials == 38 else ''
        print(f'{n_trials:<12} {max_z:<22.3f} {dsr*100:<20.2f}% {marker}')

    print('\nINTERPRETATION:')
    print('  - Deflated SR < 95%: strategy selection may be result of luck (red flag).')
    print('  - Deflated SR > 95%: strategy passes multiple-testing correction (green).')
    print('  - The repo ran V02-V39 + feature_not_retest.md (36 discarded): ~75 trials')
    print('    realistic upper bound.')


if __name__ == '__main__':
    main()
