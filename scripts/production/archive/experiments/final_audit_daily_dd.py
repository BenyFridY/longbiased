"""
Final audit — TODAS as metricas recomputadas com DD DIARIO (o que realmente importa).

1. K sensitivity com DD daily
2. Sigmoid sensitivity com DD daily
3. Confidence on/off com DD daily
4. Cost stress (5/10/15/20 bps) com DD daily
5. Comparacao H1 vs H2 vs Conservative vs baselines estaticos

Output: outputs/results/overfit_tests/final_audit_daily.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
OUT = ROOT / 'outputs' / 'results' / 'overfit_tests' / 'final_audit_daily.csv'
OUT.parent.mkdir(parents=True, exist_ok=True)

# Load data
ds = pd.read_csv(ROOT/'scripts/production/data/dataset_production.csv', parse_dates=['date']).sort_values('date').reset_index(drop=True)
ha = pd.read_csv(ROOT/'outputs/results/horizon_ablation_4y.csv', parse_dates=['date'])
h3 = ha[ha['variant']=='H=3'].reset_index(drop=True).copy()

start = pd.Timestamp('2022-01-07')
end = pd.Timestamp('2026-04-17')
daily = ds[(ds['date']>=start)&(ds['date']<=end)].copy().reset_index(drop=True)
daily['btc_ret'] = daily['price_usd'].pct_change().fillna(0)
cdi_daily = (1.13)**(1/365) - 1  # ~13%/ano CDI proxy

rebal_dates = h3['date'].values


def compute_alloc(h3_df, K_map, sigmoid_scale=15, use_confidence=True, floor=0.0, ceil=1.0):
    """Compute alloc for each rebal given K, sigmoid, confidence on/off."""
    if use_confidence:
        conf = 1/(1+np.exp(-np.abs(h3_df['p_up']-0.5)*sigmoid_scale))
    else:
        conf = np.ones(len(h3_df))
    K_vec = np.array([K_map[r] for r in h3_df['regime']])
    return np.clip(h3_df['pred'].values * K_vec * conf, floor, ceil)


def daily_sim(daily_df, rebal_dates, rebal_alloc, cost_bps=0):
    """CORRECT timing: alloc[t] applies from day AFTER the rebal."""
    rets = []
    curr_alloc = 0.0
    prev_alloc = 0.0
    for i, row in daily_df.iterrows():
        # Find latest rebal STRICTLY before today
        applicable = np.where(rebal_dates < row['date'].to_numpy())[0]
        new_alloc = rebal_alloc[applicable[-1]] if len(applicable) > 0 else 0.0
        # If alloc changed from prev_alloc, apply cost
        cost = abs(new_alloc - prev_alloc) * cost_bps / 10000 if new_alloc != prev_alloc else 0
        curr_alloc = new_alloc
        day_ret = curr_alloc * row['btc_ret'] + (1-curr_alloc) * cdi_daily - cost
        rets.append(day_ret)
        if new_alloc != prev_alloc:
            prev_alloc = new_alloc
    return np.array(rets)


def full_metrics(rets, ppy=365):
    r = np.array(rets)
    cum = (1+r).prod() - 1
    years = len(r)/ppy
    cagr = (1+cum)**(1/years) - 1
    vol = r.std() * np.sqrt(ppy)
    down = r[r<0]
    sortino = (r.mean()/np.sqrt((down**2).mean())) * np.sqrt(ppy) if len(down)>0 and (down**2).mean()>0 else float('inf')
    sharpe = ((r.mean()-cdi_daily)/r.std())*np.sqrt(ppy) if r.std()>0 else 0
    eq = (1+r).cumprod()
    peak = np.maximum.accumulate(eq)
    dd = ((eq-peak)/peak).min()
    calmar = cagr/abs(dd) if dd<0 else float('inf')
    return {'cum_%': round(cum*100,1), 'cagr_%': round(cagr*100,1), 'vol_%': round(vol*100,1),
            'sortino': round(sortino,2), 'sharpe': round(sharpe,2), 'dd_%': round(dd*100,2),
            'calmar': round(calmar,2)}


results = []

# ============================================================
# 1. K SENSITIVITY (daily DD)
# ============================================================
print('='*95)
print('1. K SENSITIVITY — DD DIARIO')
print('='*95)
print(f'{"Config":<35} {"Return":<10} {"CAGR":<8} {"Sortino":<9} {"Sharpe":<9} {"Max DD":<10} {"Calmar":<8}')
print('-'*95)
K_configs = [
    ('K=200/100/40 super-aggressive', {'BULL':200,'MILD':100,'BEAR':40}),
    ('K=150/75/30 aggressive',        {'BULL':150,'MILD':75,'BEAR':30}),
    ('K=100/50/20 H2 (antigo)',       {'BULL':100,'MILD':50,'BEAR':20}),
    ('K=80/40/15 H2-',                {'BULL':80, 'MILD':40,'BEAR':15}),
    ('K=60/30/15 H1 (NOVO)',          {'BULL':60, 'MILD':30,'BEAR':15}),
    ('K=50/25/10',                    {'BULL':50, 'MILD':25,'BEAR':10}),
    ('K=40/20/10 Conservative',       {'BULL':40, 'MILD':20,'BEAR':10}),
    ('K=30/15/7',                     {'BULL':30, 'MILD':15,'BEAR':7}),
    ('K=20/10/5 Ultra-conserv',       {'BULL':20, 'MILD':10,'BEAR':5}),
]
for label, K_map in K_configs:
    alloc = compute_alloc(h3, K_map, 15, True)
    rets = daily_sim(daily, rebal_dates, alloc, 0)
    m = full_metrics(rets)
    row = {'test':'1_K', 'config':label, **m}
    results.append(row)
    print(f'{label:<35} {m["cum_%"]:>+7.1f}%  {m["cagr_%"]:>+5.1f}%  {m["sortino"]:>+6.2f}   {m["sharpe"]:>+6.2f}   {m["dd_%"]:>+7.2f}%   {m["calmar"]:>5.2f}')

# ============================================================
# 2. SIGMOID SENSITIVITY (K=H1)
# ============================================================
print()
print('='*95)
print('2. SIGMOID SCALE SENSITIVITY — K=H1, DD DIARIO')
print('='*95)
K_H1 = {'BULL':60,'MILD':30,'BEAR':15}
for scale in [1, 5, 10, 15, 20, 25, 50, 100]:
    alloc = compute_alloc(h3, K_H1, scale, True)
    rets = daily_sim(daily, rebal_dates, alloc, 0)
    m = full_metrics(rets)
    marker = '  <- atual' if scale == 15 else ''
    row = {'test':'2_Sigmoid', 'config':f'sigmoid={scale} K=H1', **m}
    results.append(row)
    print(f'  sigmoid={scale:<4d} cum={m["cum_%"]:>+7.1f}%  CAGR={m["cagr_%"]:>+5.1f}%  Sortino={m["sortino"]:>+5.2f}  Sharpe={m["sharpe"]:>+5.2f}  DD={m["dd_%"]:>+6.2f}%{marker}')

# ============================================================
# 3. CONFIDENCE ON/OFF (K=H1, sigmoid=15)
# ============================================================
print()
print('='*95)
print('3. CONFIDENCE SCALING ON vs OFF — K=H1')
print('='*95)
for use_conf, label in [(True, 'COM confidence (sigmoid=15)'), (False, 'SEM confidence')]:
    alloc = compute_alloc(h3, K_H1, 15, use_conf)
    rets = daily_sim(daily, rebal_dates, alloc, 0)
    m = full_metrics(rets)
    row = {'test':'3_Conf', 'config':label, **m}
    results.append(row)
    avg_alloc = alloc.mean()*100
    print(f'  {label:<35} cum={m["cum_%"]:>+7.1f}%  Sortino={m["sortino"]:>+5.2f}  DD={m["dd_%"]:>+6.2f}%  avg_alloc={avg_alloc:5.1f}%')

# ============================================================
# 4. COST STRESS (K=H1, various bps)
# ============================================================
print()
print('='*95)
print('4. COST STRESS BRL — K=H1 com custos realistas')
print('='*95)
alloc_h1 = compute_alloc(h3, K_H1, 15, True)
for cost_bps in [0, 4, 8, 15, 25, 50]:
    rets = daily_sim(daily, rebal_dates, alloc_h1, cost_bps)
    m = full_metrics(rets)
    label = f'custos {cost_bps}bps/rebal'
    if cost_bps == 0: label += ' (bruto)'
    elif cost_bps == 8: label += ' (realista BRL)'
    elif cost_bps == 50: label += ' (pessimista)'
    row = {'test':'4_Costs', 'config':label, **m}
    results.append(row)
    print(f'  {label:<35} cum={m["cum_%"]:>+7.1f}%  CAGR={m["cagr_%"]:>+5.1f}%  Sortino={m["sortino"]:>+5.2f}  DD={m["dd_%"]:>+6.2f}%')

# ============================================================
# 5. H1 vs H2 vs baselines (DD diario final)
# ============================================================
print()
print('='*95)
print('5. COMPARACAO FINAL — H1 vs H2 vs baselines estaticos (com custos 8bps)')
print('='*95)

final_configs = [
    ('100% CDI (sem risco)', None, None),
    ('30% BTC estatico', 0.30, None),
    ('50% BTC estatico', 0.50, None),
    ('100% BTC HODL', 1.00, None),
    ('H2 antigo (K=100/50/20)', None, {'BULL':100,'MILD':50,'BEAR':20}),
    ('H1 NOVO (K=60/30/15)', None, {'BULL':60,'MILD':30,'BEAR':15}),
    ('Conservative (K=40/20/10)', None, {'BULL':40,'MILD':20,'BEAR':10}),
]

print(f'{"Strategy":<35} {"Return":<10} {"CAGR":<8} {"Sortino":<9} {"Sharpe":<9} {"Max DD":<10} {"Calmar":<8}')
print('-'*95)
for label, static_alloc, K_map in final_configs:
    if static_alloc is not None:
        rets = static_alloc * daily['btc_ret'].values + (1-static_alloc)*cdi_daily
    elif K_map is not None:
        alloc = compute_alloc(h3, K_map, 15, True)
        rets = daily_sim(daily, rebal_dates, alloc, 8)  # 8bps realistic BRL cost
    else:
        rets = np.array([cdi_daily]*len(daily))
    m = full_metrics(rets)
    row = {'test':'5_Final', 'config':label, **m}
    results.append(row)
    print(f'{label:<35} {m["cum_%"]:>+7.1f}%  {m["cagr_%"]:>+5.1f}%  {m["sortino"]:>+6.2f}   {m["sharpe"]:>+6.2f}   {m["dd_%"]:>+7.2f}%   {m["calmar"]:>5.2f}')

# Save
df_res = pd.DataFrame(results)
df_res.to_csv(OUT, index=False)
print()
print(f'Saved: {OUT}')
