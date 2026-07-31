"""
Deep audit: find the actual problem in the model.

Uses the H=3 walk-forward results from horizon_ablation_4y.csv (248 rebals over 4.3y)
and cross-checks against dataset + signal_history.

10 checks:
 1. Accuracy por ano/trimestre - ha drift?
 2. Accuracy por regime (BULL/MILD/BEAR)
 3. Accuracy condicionada em alocacao (sizing realmente acerta onde bate?)
 4. Binomial CI: 47% em 2026 eh estatisticamente diferente de 62% historico?
 5. Prediction magnitude calibration (modelo prev de mais? de menos?)
 6. Feature drift: z-score das features em 2026 vs 2022-2025
 7. Overfit check: accuracy diminui de 1st half para 2nd half do OOS?
 8. Retrain effect: accuracy por janela de cutoff (primeiros vs ultimos rebals)
 9. Emergency rebal contribution: sem emergencies, o modelo bate BTC?
 10. PnL sensitivity a 5bps cost (como fica com 10bps? 20bps?)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts' / 'production'))

from config import FEATURES_37


df = pd.read_csv(ROOT / 'outputs/results/horizon_ablation_4y.csv', parse_dates=['date'])
df = df[df['variant'] == 'H=3'].copy()
ds = pd.read_csv(ROOT / 'scripts/production/data/dataset_production.csv', parse_dates=['date'])

df['year'] = df['date'].dt.year
df['hit'] = df['dir_match']

print('=' * 78)
print('  DEEP AUDIT - H=3 walk-forward, 248 rebals, 2022-01-07 => 2026-04-17')
print('=' * 78)


# ==================== 1. Accuracy por ano ====================
print('\n[1] ACCURACY POR ANO (direcao 7d forward)')
print('-' * 78)
by_year = df.groupby('year').agg(
    N=('hit', 'size'),
    hits=('hit', 'sum'),
    acc=('hit', 'mean'),
    btc_mean=('btc_fwd', 'mean'),
    strat_mean=('strat', 'mean'),
    alloc_mean=('alloc', 'mean'),
    corr=('pred', lambda x: df.loc[x.index, 'pred'].corr(df.loc[x.index, 'btc_fwd'])),
)
by_year['acc'] = (by_year['acc'] * 100).round(1)
by_year['btc_mean'] = (by_year['btc_mean'] * 100).round(2)
by_year['strat_mean'] = (by_year['strat_mean'] * 100).round(2)
by_year['alloc_mean'] = (by_year['alloc_mean'] * 100).round(1)
by_year['corr'] = by_year['corr'].round(3)
print(by_year.to_string())
print('  => DRIFT? Se accuracy cai ano a ano = sinal morrendo.')


# ==================== 2. Accuracy por regime ====================
print('\n[2] ACCURACY POR REGIME')
print('-' * 78)
by_reg = df.groupby('regime').agg(
    N=('hit', 'size'),
    acc=('hit', 'mean'),
    btc_mean=('btc_fwd', 'mean'),
    alloc_mean=('alloc', 'mean'),
    strat_mean=('strat', 'mean'),
)
by_reg['acc'] = (by_reg['acc'] * 100).round(1)
by_reg['btc_mean'] = (by_reg['btc_mean'] * 100).round(2)
by_reg['alloc_mean'] = (by_reg['alloc_mean'] * 100).round(1)
by_reg['strat_mean'] = (by_reg['strat_mean'] * 100).round(2)
print(by_reg.to_string())


# ==================== 3. Accuracy condicionada em alocacao ====================
print('\n[3] ACCURACY vs TAMANHO DA ALOCACAO (sizing calibration)')
print('-' * 78)
df['alloc_bucket'] = pd.cut(df['alloc'], bins=[-0.001, 0.001, 0.2, 0.5, 0.8, 1.01],
                            labels=['=0', '0-20%', '20-50%', '50-80%', '80-100%'])
by_alloc = df.groupby('alloc_bucket', observed=True).agg(
    N=('hit', 'size'),
    acc=('hit', 'mean'),
    btc_mean=('btc_fwd', 'mean'),
    strat_mean=('strat', 'mean'),
)
by_alloc['acc'] = (by_alloc['acc'] * 100).round(1)
by_alloc['btc_mean'] = (by_alloc['btc_mean'] * 100).round(2)
by_alloc['strat_mean'] = (by_alloc['strat_mean'] * 100).round(2)
print(by_alloc.to_string())
print('  => Calibracao ideal: acc aumenta com alloc. Se nao aumenta = sizing nao funciona.')


# ==================== 4. Binomial CI ====================
print('\n[4] TESTE ESTATISTICO: 2026 YTD 47% vs historico 62%')
print('-' * 78)
y2026 = df[df['year'] == 2026]
hist = df[df['year'] < 2026]
n26, k26 = len(y2026), y2026['hit'].sum()
nh, kh = len(hist), hist['hit'].sum()
p26 = k26 / n26
ph = kh / nh
# binomial CI for 2026
ci26 = stats.binomtest(k26, n26).proportion_ci(confidence_level=0.95)
cih = stats.binomtest(kh, nh).proportion_ci(confidence_level=0.95)
# p-value testing if 2026 = historical
z = (p26 - ph) / np.sqrt(ph * (1 - ph) / n26)
p_val = 2 * (1 - stats.norm.cdf(abs(z)))
print(f'  2026 YTD:  {k26}/{n26} = {p26*100:.1f}%  95% CI [{ci26.low*100:.1f}%, {ci26.high*100:.1f}%]')
print(f'  2022-2025: {kh}/{nh} = {ph*100:.1f}%  95% CI [{cih.low*100:.1f}%, {cih.high*100:.1f}%]')
print(f'  z-score: {z:.2f}  p-value: {p_val:.3f}')
if p_val < 0.05:
    print(f'  => 2026 DIFERENTE do historico (p<0.05). DRIFT REAL.')
else:
    print(f'  => 2026 nao diferente estatisticamente (p={p_val:.2f}). Provavel RUIDO.')


# ==================== 5. Prediction magnitude calibration ====================
print('\n[5] PREDICTION MAGNITUDE CALIBRATION')
print('-' * 78)
df['pred_abs'] = df['pred'].abs()
df['fwd_abs'] = df['btc_fwd'].abs()
bins = pd.qcut(df['pred_abs'], q=5, duplicates='drop')
calib = df.groupby(bins, observed=True).agg(
    N=('pred', 'size'),
    pred_mean=('pred', 'mean'),
    fwd_mean=('btc_fwd', 'mean'),
    pred_mag=('pred_abs', 'mean'),
    fwd_mag=('fwd_abs', 'mean'),
)
calib['pred_mean'] = (calib['pred_mean'] * 100).round(2)
calib['fwd_mean'] = (calib['fwd_mean'] * 100).round(2)
calib['pred_mag'] = (calib['pred_mag'] * 100).round(2)
calib['fwd_mag'] = (calib['fwd_mag'] * 100).round(2)
print('  Predicoes agrupadas por magnitude absoluta:')
print(calib.to_string())
print('  => Se pred_mag << fwd_mag: modelo eh CONSERVADOR (subestima movimentos).')
print('  => Se pred_mag >> fwd_mag: modelo eh OVERCONFIDENT.')


# ==================== 6. Feature drift 2026 vs historico ====================
print('\n[6] FEATURE DRIFT: z-score em 2026 vs media historica')
print('-' * 78)
ds['year'] = ds['date'].dt.year
pre = ds[ds['year'] < 2026]
cur = ds[ds['year'] == 2026]
drift_rows = []
for f in FEATURES_37:
    if f not in ds.columns:
        continue
    mu, sig = pre[f].mean(), pre[f].std()
    if sig == 0 or np.isnan(sig):
        continue
    z_2026 = (cur[f].mean() - mu) / sig
    drift_rows.append({'feature': f, 'z_2026': z_2026,
                       'abs_z': abs(z_2026)})
drift = pd.DataFrame(drift_rows).sort_values('abs_z', ascending=False)
print('  Top 10 features com maior drift:')
print(drift.head(10).to_string(index=False))
extreme = drift[drift['abs_z'] > 2.0]
print(f'\n  {len(extreme)}/32 features com |z| > 2.0 (drift significativo).')


# ==================== 7. Overfit check: 1st vs 2nd half OOS ====================
print('\n[7] OVERFIT CHECK: 1st half OOS vs 2nd half OOS')
print('-' * 78)
mid = df.iloc[len(df)//2]['date']
first = df[df['date'] < mid]
second = df[df['date'] >= mid]
print(f'  1st half (N={len(first)}, 2022-{mid.year}): acc {first["hit"].mean()*100:.1f}%, corr {first["pred"].corr(first["btc_fwd"]):+.3f}')
print(f'  2nd half (N={len(second)}, {mid.year}-2026):  acc {second["hit"].mean()*100:.1f}%, corr {second["pred"].corr(second["btc_fwd"]):+.3f}')
delta = second['hit'].mean() - first['hit'].mean()
print(f'  Delta: {delta*100:+.1f}pp')


# ==================== 8. Accuracy por janela de cutoff ====================
print('\n[8] ACCURACY POR JANELA DE CUTOFF (decay entre retrains?)')
print('-' * 78)
cutoffs = ['2022-01-01', '2022-07-01', '2023-01-01', '2023-07-01',
           '2024-01-01', '2024-07-01', '2025-01-01', '2025-07-01', '2026-01-01']
cutoffs = [pd.Timestamp(c) for c in cutoffs]
for i, c in enumerate(cutoffs):
    end_c = cutoffs[i+1] if i+1 < len(cutoffs) else df['date'].max()
    window = df[(df['date'] >= c) & (df['date'] < end_c)]
    if len(window) == 0: continue
    print(f'  Cutoff {c.date()} ({len(window):3d} rebals): acc {window["hit"].mean()*100:5.1f}% | mean_alloc {window["alloc"].mean()*100:5.1f}% | strat {(window["strat"].mean())*100:+.2f}%/rebal')


# ==================== 9. Emergency contribution ====================
print('\n[9] CONTRIBUICAO DOS EMERGENCY REBALS (sem eles, o modelo bate?)')
print('-' * 78)
# Emergency = daily_ret > 8% no dia do rebal. Proxy: computar daily_ret no dataset
ds_idx = ds.set_index('date')
emergencies = []
for _, r in df.iterrows():
    d = r['date']
    if d in ds_idx.index:
        prev_idx = ds_idx.index.get_loc(d) - 1
        if prev_idx >= 0:
            p_prev = ds_idx.iloc[prev_idx]['price_usd']
            daily = (r['pred'] * 0) + (ds_idx.loc[d, 'price_usd'] / p_prev - 1)
            emergencies.append(abs(daily) > 0.08)
        else:
            emergencies.append(False)
    else:
        emergencies.append(False)
df['emergency'] = emergencies
emg = df[df['emergency']]
reg_only = df[~df['emergency']]
print(f'  Emergency rebals: {len(emg)} (acc {emg["hit"].mean()*100:.1f}%, strat medio {emg["strat"].mean()*100:+.2f}%)')
print(f'  Regular (Friday): {len(reg_only)} (acc {reg_only["hit"].mean()*100:.1f}%, strat medio {reg_only["strat"].mean()*100:+.2f}%)')

cum_all = (1 + df['strat']).prod() - 1
cum_no_emg = (1 + reg_only['strat']).prod() - 1
btc_all = (1 + df['btc_fwd']).prod() - 1
print(f'\n  Cum return COM emergencies:  {cum_all*100:+.2f}%')
print(f'  Cum return SEM emergencies:  {cum_no_emg*100:+.2f}%')
print(f'  BTC same period:             {btc_all*100:+.2f}%')
print(f'  Emergencies isoladas geram:  {((1+cum_all)/(1+cum_no_emg)-1)*100:+.2f}%')


# ==================== 10. Cost sensitivity ====================
print('\n[10] COST SENSITIVITY (bps por rebal subtraido do strat_ret)')
print('-' * 78)
for bps in [5, 10, 20, 50, 100]:
    cost = bps / 10000
    # assume cost * alloc_delta, simplify: cost*alloc
    net = df['strat'] - cost * df['alloc'].diff().abs().fillna(df['alloc'])
    cum_net = (1 + net).prod() - 1
    ann = (1 + cum_net) ** (1 / 4.3) - 1
    print(f'  {bps:3d} bps: Return {cum_net*100:+.2f}%  CAGR {ann*100:5.1f}%/yr')


print('\n' + '=' * 78)
print('  CONCLUSAO')
print('=' * 78)
print('Ver analise em cada secao acima. O principal problema provavel:')
print('  - Se [1] ou [8] mostra drift descendente: modelo expirou, precisa retrain.')
print('  - Se [6] tem muitas features com |z|>2: mercado mudou estrutura.')
print('  - Se [9] mostra que SEM emergencies o alpha some: estrategia depende de cisnes.')
print('  - Se [3] mostra acc alta=baixa: sizing nao funciona, risco mal calibrado.')
