"""
Parte 1: Cost stress BRL realista
  - OTC 4bps (usuario)
  - + Slippage 1-2bps
  - + Tracking error (exec vs model close) 1-2bps
  - + BRL/USD conversion drag (feito 1x no funding, 0 por rebal)
  - Custo e aplicado sobre |alloc[t] - alloc[t-1]| (nao sobre alloc absoluta)
  - Valida com custos escalonados: 4, 6, 8, 10, 15 bps

Parte 2: Confidence factor weight audit
  - Mostra distribuicao de confidence_factor
  - Compara strat COM confidence vs SEM confidence (factor=1)
  - Quantifica impacto no retorno e Sortino
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts' / 'production'))

from config import K_REGIME, ALLOC_MIN, ALLOC_MAX, SIGMOID_SCALE


# Carrega o walk-forward H=3 ja rodado
df = pd.read_csv(ROOT / 'outputs/results/horizon_ablation_4y.csv', parse_dates=['date'])
df = df[df['variant'] == 'H=3'].reset_index(drop=True).copy()


# =============================================================
# PARTE 1: COST STRESS REALISTA
# =============================================================
print('=' * 75)
print('  PARTE 1: COST STRESS — configuracao BRL realista')
print('=' * 75)

# delta-alloc por rebal
df['delta_alloc'] = df['alloc'].diff().abs().fillna(df['alloc'])

# strat_ret_gross ja esta sem custo no CSV (coluna "strat")
df['strat_gross'] = df['strat']

def sortino(r, ppy=52):
    arr = np.array(r)
    down = arr[arr < 0]
    if len(down) == 0: return float('inf')
    return arr.mean() * ppy / (np.sqrt((down ** 2).mean()) * np.sqrt(ppy))

def max_dd(r):
    cum = (1 + np.array(r)).cumprod()
    peak = np.maximum.accumulate(cum)
    return ((cum - peak) / peak).min()

def apply_cost(df, bps):
    cost = bps / 10000
    return df['strat_gross'] - cost * df['delta_alloc']

print('\nBreakdown de custo realista BRL:')
print('  OTC spread all-in (seu valor):              4 bps')
print('  Slippage execucao (small OTC):              1-2 bps')
print('  Tracking error (model close vs exec time):  1-2 bps')
print('  Total tipico:                               6-8 bps')
print('  Pessimista:                                 10 bps')

print('\n{:<30} {:>12} {:>10} {:>10} {:>10} {:>10}'.format(
    'Cenario', 'Return', 'CAGR', 'Sortino', 'Max DD', 'Alpha'))
print('-' * 75)

scenarios = [
    (0,  'Bruto (sem custo)'),
    (4,  'Otimista (so OTC 4bps)'),
    (6,  'Realista (OTC+slip 6bps)'),
    (8,  'Realista+ (OTC+slip+TE 8bps)'),
    (10, 'Conservador (seu teto 10bps)'),
    (15, 'Pessimista (stress 15bps)'),
]

# BTC return in same window for alpha calc
btc_cum = (1 + df['btc_fwd']).prod() - 1

for bps, label in scenarios:
    net = apply_cost(df, bps)
    cum = (1 + net).prod() - 1
    ann = (1 + cum) ** (1 / 4.3) - 1
    s = sortino(net)
    dd = max_dd(net)
    alpha = cum - btc_cum
    print(f'{label:<30} {cum*100:>+10.2f}% {ann*100:>8.1f}% {s:>10.2f} {dd*100:>+9.2f}% {alpha*100:>+9.2f}pp')

print(f'\nBTC 4.3y total: {btc_cum*100:+.2f}%')
print(f'Numero de rebals: {len(df)}')
print(f'Delta-alloc medio por rebal: {df["delta_alloc"].mean()*100:.1f}% do portfolio')
print(f'Delta-alloc total 4.3y: {df["delta_alloc"].sum()*100:.0f}% = turnover equivalente')

# Sobre BRL: CDI > T-bill, BTC price em USD x BRL volatility
print('\n' + '=' * 75)
print('  AJUSTE BRL: o que MUDA vs o numero USD?')
print('=' * 75)
print('''
  1. CDI (risk-free BRL) e MAIS ALTO que T-bill USD (~11%/ano vs ~4.5%)
     => Quando alloc baixa, CDI da mais retorno. Strategy BRL OUTPERFORMA USD na parte de cash.

  2. BTC em BRL inclui FX. Real apreciou vs USD 2022-2026 (~11.5%).
     => BTC-BRL cresceu MENOS que BTC-USD. Strat absoluta em BRL < USD.

  3. Para custo: OTC cobra spread BRL-BTC, nao USD-BTC. 4bps bate.

  Net effect: numeros absolutos em BRL sao MENORES, mas ALPHA relativa a BTC-BRL e similar.
  Doc ja cita validacao BRL: Sortino 3.21, Ret +793%, DD -10.7% com cost=5bps.
''')


# =============================================================
# PARTE 2: CONFIDENCE FACTOR AUDIT
# =============================================================
print('=' * 75)
print('  PARTE 2: QUANTO O CONFIDENCE FACTOR MEXE?')
print('=' * 75)

# Recompute confidence from p_up (should match what was saved)
df['conf_factor'] = 1 / (1 + np.exp(-np.abs(df['p_up'] - 0.5) * SIGMOID_SCALE))

print('\n[2.1] DISTRIBUICAO DO CONFIDENCE FACTOR (248 rebals):')
print('-' * 75)
print(f"  min:    {df['conf_factor'].min():.3f}  (p_up={df.loc[df['conf_factor'].idxmin(), 'p_up']:.3f})")
print(f"  10%:    {df['conf_factor'].quantile(0.10):.3f}")
print(f"  25%:    {df['conf_factor'].quantile(0.25):.3f}")
print(f"  median: {df['conf_factor'].median():.3f}")
print(f"  75%:    {df['conf_factor'].quantile(0.75):.3f}")
print(f"  90%:    {df['conf_factor'].quantile(0.90):.3f}")
print(f"  max:    {df['conf_factor'].max():.3f}  (p_up={df.loc[df['conf_factor'].idxmax(), 'p_up']:.3f})")

print('\n  Tabela exemplo: p_up => conf_factor => reducao da aposta:')
print('  ' + '-' * 55)
print('  p_up     |dist 0.5| * 15   sigmoid   reduce K by')
for pu in [0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 0.95]:
    d = abs(pu - 0.5) * SIGMOID_SCALE
    cf = 1 / (1 + np.exp(-d))
    reduce = (1 - cf) * 100
    print(f'  {pu:.2f}        {d:5.2f}          {cf:.3f}     {reduce:5.1f}%')

# ============== Ablation: strat SEM confidence ==============
print('\n[2.2] ABLATION: strat COM vs SEM confidence factor')
print('-' * 75)

# Para rodar sem confidence, precisamos recomputar alloc. Mas strat_ret no csv
# ja foi computado com conf. Precisamos recompute alloc_no_conf = clip(pred * K, 0, 1)
# e depois recompute strat_ret_no_conf com o mesmo btc_fwd e cdi proxy.
# Problema: nao temos CDI por rebal salvo. Aproxima: strat - alloc*btc ≈ (1-alloc)*cdi
# Entao cdi_effective = (strat - alloc*btc) / (1-alloc)
cdi_implied = np.where(
    (1 - df['alloc']) > 0.001,
    (df['strat_gross'] - df['alloc'] * df['btc_fwd']) / (1 - df['alloc']),
    0.0
)

# Mapa K por regime
K_map = K_REGIME  # {'BULL': 100, 'MILD': 50, 'BEAR': 20}

# Recompute alloc sem conf
df['alloc_no_conf'] = df.apply(
    lambda r: np.clip(r['pred'] * K_map[r['regime']], ALLOC_MIN, ALLOC_MAX), axis=1
)
df['cdi_implied'] = cdi_implied
df['strat_no_conf'] = df['alloc_no_conf'] * df['btc_fwd'] + (1 - df['alloc_no_conf']) * df['cdi_implied']

# Aplica custo 5bps em ambos
df['delta_alloc_no_conf'] = df['alloc_no_conf'].diff().abs().fillna(df['alloc_no_conf'])
def cum_with_cost(ret, delta, bps):
    c = bps / 10000
    net = ret - c * delta
    return (1 + net).prod() - 1, sortino(net), max_dd(net)

for bps in [5, 8, 10]:
    cum_c, s_c, dd_c = cum_with_cost(df['strat_gross'], df['delta_alloc'], bps)
    cum_nc, s_nc, dd_nc = cum_with_cost(df['strat_no_conf'], df['delta_alloc_no_conf'], bps)
    print(f'\n  @ {bps} bps cost:')
    print(f'    COM confidence:  Return {cum_c*100:+7.2f}%  Sortino {s_c:5.2f}  MaxDD {dd_c*100:+6.2f}%  mean_alloc {df["alloc"].mean()*100:5.1f}%')
    print(f'    SEM confidence:  Return {cum_nc*100:+7.2f}%  Sortino {s_nc:5.2f}  MaxDD {dd_nc*100:+6.2f}%  mean_alloc {df["alloc_no_conf"].mean()*100:5.1f}%')
    delta_ret = cum_c - cum_nc
    delta_s = s_c - s_nc
    print(f'    Delta (conf - noconf): Return {delta_ret*100:+5.2f}pp  Sortino {delta_s:+5.2f}')

# ========== Quantos rebals o confidence efetivamente mexeu? ==========
print('\n[2.3] IMPACTO DO CONFIDENCE NA ALOC TOMADA')
print('-' * 75)
df['alloc_diff_from_noconf'] = df['alloc'] - df['alloc_no_conf']
mean_reduction = (df['alloc_no_conf'] - df['alloc']).clip(lower=0).mean()
print(f'  Reducao media por rebal:        {mean_reduction*100:.1f}pp de alloc')
print(f'  Rebals com alloc reduzida > 5pp: {(df["alloc_no_conf"] - df["alloc"] > 0.05).sum()}/{len(df)}')
print(f'  Rebals com alloc reduzida > 20pp:{(df["alloc_no_conf"] - df["alloc"] > 0.20).sum()}/{len(df)}')
print()
print('  Top 5 rebals onde confidence mais reduziu alloc:')
top = df.nlargest(5, 'alloc_no_conf').copy()
top['reduction_pp'] = (top['alloc_no_conf'] - top['alloc']) * 100
print(top[['date', 'p_up', 'conf_factor', 'alloc_no_conf', 'alloc', 'reduction_pp', 'btc_fwd', 'dir_match']].to_string(index=False))

print('\n' + '=' * 75)
print('  CONCLUSAO')
print('=' * 75)
