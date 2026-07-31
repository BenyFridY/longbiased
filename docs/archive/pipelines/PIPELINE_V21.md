# V21 — Modelo Final

**Datas**: V20 Alocação (2026-03-20), Robustez 20 seeds (2026-03-22), Horizonte + Hyperparameters (2026-03-23)

---

## Modelo Final (V21)

```
XGBoost Bag80 | 37 features | horizonte 3d
Retrain: semi-anual, expanding window
Regime: SMA50/200 (BULL/MILD/BEAR)
K: BULL=50, MILD=30, BEAR=15
Alocação: clip(pred × K, -25%, 100%)
Rebalanceo: sexta-feira + emergência (|daily_ret| > 8%)
Custo: 2 bps

Hyperparameters:
  learning_rate=0.05, n_estimators=200, max_leaves=31
  min_child_weight=12, colsample_bytree=0.5, subsample=0.8
  grow_policy=lossguide, tree_method=hist

Mudança vs V19: colsample_bytree 0.7 → 0.5 (cada árvore vê 50% das features)

Resultado (10 seeds): Sortino 4.00±0.07, +1067%±23%, MaxDD -9.9%±0.3%
```

### Evolução completa do pipeline

| Ver | Return | Sortino | MaxDD | Key Insight |
|-----|--------|---------|-------|-------------|
| V1 | +152% | 0.59 | — | Price > fundamentals |
| V2 | +221% | 0.87 | — | Bagging = biggest improvement |
| V5 | +289% | 1.12 | — | ML weight is the lever |
| V8 | +417% | 1.51 | — | APPROVED 14/16 |
| V10 | +802% | 2.74 | -14.1% | linear_direct K=25 |
| V13 | +1313% | 3.22 | -15.9% | Semi-annual + basis_pct + K=30 |
| V17 | +930% | 3.55 | -9.9% | Dynamic regime + 5 features |
| V18 | +1054% | 3.64 | -10.1% | K_bull=50 + Bag80 |
| V19 | +1059% | 3.87 | -10.1% | Fri+emergency 8% rebalance |
| V20 | +1042% | 3.96 | -10.1% | 126 combos tested, baseline won |
| **V21** | **+1067%** | **4.00** | **-9.9%** | **colsample_bytree=0.5 (fine-tuning)** |

---

## O que o XGBoost faz

O modelo prevê o **retorno de BTC nos próximos 3 dias**:

```
Exemplo: pred = +0.7% ("BTC sobe 0.7% em 3 dias")
  Em BEAR  (K=15) → alocação = 0.7% × 15 = 10.5%
  Em MILD  (K=30) → alocação = 0.7% × 30 = 21.0%
  Em BULL  (K=50) → alocação = 0.7% × 50 = 35.0%
```

O K é necessário porque o output raw do modelo (~±3%) é pequeno demais para usar diretamente como % de alocação. O K por regime é um **ajuste de risco assimétrico**: aposta mais quando errar custa menos (bull), menos quando errar custa mais (bear).

### Escala das previsões

| Métrica | Target (retorno real 3d) | Previsão XGB |
|---------|--------------------------|--------------|
| Desvio padrão | 4.65% | 2.88% |
| Range | -23.8% a +20.7% | -10.7% a +10.3% |

### Acurácia

| Métrica | Valor |
|---------|-------|
| Acerto direcional (todos os dias) | ~57% |
| Win/Loss ratio | 1.72x |
| Correlação pred vs actual | ~0.19 |
| Edge por trade | ~+0.51% |

---

## Testes realizados

### 1. Alocação: 126 regime x fórmula (V20, 2026-03-20)

7 regimes × 18 fórmulas × 10 seeds = 1.260 backtests.

**Top 5:**

| # | Config | Sortino | Retorno | MaxDD |
|---|--------|---------|---------|-------|
| **1** | **sma50_200 \| baseline** | **3.94** | **+1039%** | **-10.1%** |
| 2 | sma50_200 \| winsorize | 3.93 | +1028% | -10.1% |
| 3 | rsi \| baseline | 3.65 | +922% | -11.4% |
| 4 | sma20_100 \| baseline | 3.62 | +889% | -13.3% |
| 5 | sma50_200 \| sign_mag | 3.54 | +409% | -5.9% |

A fórmula baseline venceu em **todos os 7 regimes**. SMA50/200 venceu em **todas as 18 fórmulas**.

### 2. Robustez: 20 seeds (2026-03-22)

| Config | Sortino (avg±std) | Retorno (avg±std) | MaxDD (avg±std) |
|--------|-------------------|-------------------|-----------------|
| **sma50_200 \| baseline** | **3.96 ± 0.05** | **+1042% ± 21%** | **-10.1% ± 0.3%** |
| sma50_200 \| winsorize | 3.95 ± 0.05 | +1031% ± 21% | -10.1% ± 0.3% |
| composite \| baseline | 3.88 ± 0.05 | +1144% ± 28% | -12.5% ± 0.3% |

**sma50_200|baseline ganhou 20/20 seeds** vs composite.

### 3. Horizonte de previsão — comparação justa (2026-03-23)

O modelo prevê 3 dias mas segura 7 dias. Para testar se outro horizonte tem sinal melhor, **calibramos K por horizonte** para que as alocações tenham mesma escala:

| Horizonte | K calibrado (B/M/B) | Sortino | Retorno | MaxDD |
|-----------|---------------------|---------|---------|-------|
| 2d | 70/42/21 | 3.47 | +1011% | -9.4% |
| **3d** | **50/30/15** | **3.97** | **+1050%** | **-10.3%** |
| 5d | 33/20/10 | 3.62 | +845% | -11.8% |
| 7d | 25/15/7 | 2.75 | +656% | -17.4% |

**3d vence todos.** O 7d (alinhado com o holding) é o pior — prever 7 dias é muito mais ruidoso.

O mismatch 3d-pred / 7d-hold é intencional: sinal curto é mais preciso, correlação entre ret_3d e ret_7d é alta (~0.7-0.8), e a literatura acadêmica confirma que usar horizonte mais curto que o holding period é prática padrão em quant finance.

### 4. Hyperparameters do XGBoost (2026-03-23)

**Fase 1**: 50 configs × 3 seeds (screening de todos os eixos).
**Fase 2**: 6 candidatos × 10 seeds (confirmação).

| # | Config | Sortino | ±std | Retorno | MaxDD | vs Baseline |
|---|--------|---------|------|---------|-------|-------------|
| 1 | sub=0.6+col=0.5 | 4.08 | 0.05 | +1118% | -12.5% | +0.15 S, -2.4pp DD |
| 2 | sub=0.7+col=0.5 | 4.04 | 0.05 | +1092% | -11.6% | +0.10 S, -1.5pp DD |
| 3 | subsamp=0.6 | 4.03 | 0.06 | +1128% | -12.9% | +0.09 S, -2.8pp DD |
| **4** | **colsamp=0.5** | **4.00** | **0.07** | **+1067%** | **-9.9%** | **+0.07 S, +0.2pp DD** |
| 5 | subsamp=0.7 | 3.99 | 0.06 | +1088% | -11.7% | +0.05 S, -1.7pp DD |
| 6 | BASELINE (sub=0.8, col=0.7) | 3.93 | 0.05 | +1059% | -10.1% | — |

Head-to-head vs baseline (wins out of 10 seeds):
- sub=0.6+col=0.5: **10/10** (+0.15 S, mas DD piora 2.4pp)
- sub=0.7+col=0.5: **10/10** (+0.10 S, mas DD piora 1.5pp)
- **colsamp=0.5: 8/10** (+0.07 S, DD melhora 0.2pp)

**Decisão: colsamp=0.5** — a única mudança que melhora Sortino **sem piorar DD**. Ganho "free lunch".

O que colsamp=0.5 faz: cada árvore vê 50% das features aleatórias (vs 70% antes). Isso força mais diversidade no ensemble → modelo mais robusto.

### Parâmetros finais vs baseline

| Parâmetro | Antes (V19) | Agora (V21) | Recomendado (XGBoost docs) |
|-----------|-------------|-------------|---------------------------|
| subsample | 0.8 | 0.8 | 0.5-1.0 |
| **colsample_bytree** | **0.7** | **0.5** | **0.5-1.0** |
| learning_rate | 0.05 | 0.05 | 0.01-0.3 |
| max_leaves | 31 | 31 | ~15-63 |
| min_child_weight | 12 | 12 | depende do dataset |
| n_estimators | 200 | 200 | 100-1000 |

---

## Verificação da conta de retorno

Verificação manual passo-a-passo:
- Método 1 (função backtest_allocations): +1089.8%
- Método 2 (reprodução manual): +1089.8%
- **Match: 100%**

Fórmula:
- Long (a>=0): `retorno = a × ret_BTC + (1-a) × CDI - custo`
- Short (a<0): `retorno = a × ret_BTC + 1.0 × CDI - custo`
- Custo: |delta_alloc| × 2bps × (1.5x se short)
- BTC buy&hold no período (2022-2026): **+43.2%**
- CDI acumulado no período: **+64.6%** (12.7% a.a.)
- PBO = **0.0%** (70 CSCV combos)

---

## Tudo que foi descartado

### Alocação (V20)
- Kelly criterion (todas as variantes): mais DD sem ganho de Sortino
- Tanh/sigmoid: perdem para clip linear
- Regimes alternativos (RSI, vol, momentum, percentile, composite): todos piores que SMA50/200
- Z-score/rank/normalize do sinal: todos piores
- Adaptive K por volatilidade: mais DD
- Composite regime: +100pp retorno mas -2.4pp DD

### Horizonte (V21)
- 2d, 5d, 7d: todos piores que 3d com K calibrado
- 7d é o pior apesar de alinhar com holding period

### Hyperparameters (V21)
- subsample 0.6/0.7: Sortino melhora mas DD piora demais
- sub=0.6+col=0.5, sub=0.7+col=0.5: idem (Sortino alto mas DD -11 a -12.5%)
- learning_rate diferente de 0.05: 0.01-0.03 = underfit, 0.1+ = overfit
- n_estimators diferente de 200: 100 = underfit, 400+ = sem ganho
- max_leaves 7/15/63/127: todos piores ou iguais a 31
- min_child_weight 3/5/8/20/30/50: todos piores que 12
- reg_alpha, reg_lambda: sem impacto
- max_depth (alternativa a leaves): sem melhoria

---

## Referências

- [Short term return prediction of cryptocurrency based on XGBoost (IEEE 2022)](https://ieeexplore.ieee.org/document/9758430/)
- [Short-Term Bitcoin Market Prediction via Machine Learning (2021)](https://www.sciencedirect.com/science/article/pii/S2405918821000027)
- [Time-Series Forecasting of Bitcoin Prices using ML (2020)](https://link.springer.com/article/10.1007/s00521-020-05129-6)
- [Multi-Horizon Equity Returns Predictability via ML (ECB WP)](https://www.econstor.eu/bitstream/10419/247369/1/wp2021-02.pdf)
- [XGBoost Parameter Tuning — Official Documentation](https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html)
- [Cryptocurrency Price Forecasting Using XGBoost (2024)](https://arxiv.org/html/2407.11786v1)

## Scripts

| Script | O que faz |
|--------|-----------|
| `scripts/optimization/v20_alloc_research.py` | 126 combos regime × fórmula (10 seeds) |
| `scripts/optimization/v20_robustness.py` | Top 3 × 20 seeds |
| `scripts/optimization/v20_xgb_hyperparams.py` | 50 configs de hyperparameters (3 seeds) |
| `scripts/optimization/v21_fair_horizon.py` | Horizontes com K calibrado (4 × 5 seeds) |
| `scripts/optimization/v21_hp_confirm.py` | Top 6 hyperparameters (10 seeds) |
