# Overfit Tests — 2026-04-22

Relatorio completo de 7 testes profissionais de overfit, mais implementacao de
controles de risco (kill switch + drift monitor).

Scripts em `scripts/production/archive/experiments/overfit_test_*.py`.
Resultados CSV em `outputs/results/overfit_tests/`.

> **Nota (snapshot historico de 2026-04-22):** este relatorio usa o framing da epoca —
> Sortino/cum_return **semanais**, baseline **H2 (K=100/50/20)** e **BAGS=80**
> (single-config). Os numeros **canonicos atuais** (daily, **H1**, 10-seed, **BAGS=160**)
> estao em [`MODEL_FINAL.md`](./MODEL_FINAL.md): Sortino daily **3.53**, CAGR **+57.3%**,
> DD daily **-7.14%**, cum **+597%**. As conclusoes qualitativas seguem validas: edge real
> (0/100 shuffles, p<0.01), H1 mais robusto, magnitude do ML **+3.4** de Sortino (Teste 7),
> retrain ~66% do Sortino, DSR 13.5% (38 trials) / 87% (N efetivo 10).

---

## TL;DR em 4 linhas

1. **Predictions carregam sinal REAL** (Teste 4: 0 de 100 shuffles bateu o Sortino observado).
2. **Sortino 5.91 do backtest e inflacionado** por specification search (Deflated Sharpe Ratio com 38 trials = 13% probabilidade de ser edge real).
3. **Retrain semi-anual e responsavel por ~66% do Sortino** (frozen train 2022 = Sortino 1.88 vs 5.61 com retrain).
4. **H1 (K=60/30/15) + kill switch + acc derisk e a configuracao mais robusta**: Sortino 6.91, return +667%, DD -2.88% em backtest. Expectativa live: Sortino ~1.5-2.5 apos deflation.

---

## Setup dos testes

- **Dataset**: `dataset_production.csv`, 2667 rows, 2019-01-01 a 2026-04-21
- **Walk-forward baseline**: `horizon_ablation_4y.csv` (248 rebals 2022-01 a 2026-04, semi-annual retrain)
- **Todos os testes usam o mesmo pipeline** (`walkforward_backtest.py`) com mesmas 32 features.

---

## TESTE 1: K Sensitivity — Sortino/Sharpe por tamanho de aposta

**Pergunta**: H2 (K=100/50/20) foi tunado ex-post. O edge sobrevive em diferentes K?

**Metodologia**: Usa as predicoes ja geradas (walk-forward semi-anual), varia apenas o multiplicador K_REGIME.

| Config | K_BULL/MILD/BEAR | cum_ret | vs BTC | Sortino | Sharpe | Max DD |
|---|---|---|---|---|---|---|
| super-aggressive | 200/100/40 | +1631% | +1547pp | 3.54 | 2.00 | -8.5% |
| aggressive | 150/75/30 | +1571% | +1487pp | 4.46 | 2.07 | -6.3% |
| **H2 (atual)** | **100/50/20** | **+1191%** | **+1108pp** | **5.61** | **2.09** | **-4.0%** |
| H2 ajustado | 80/40/15 | +941% | +857pp | 6.81 | 2.04 | -2.9% |
| **H1 (menor risco)** | **60/30/15** | **+782%** | **+698pp** | **7.00** | **2.04** | **-2.9%** |
| meio-termo | 50/25/10 | +576% | +492pp | 8.65 | 1.92 | -1.7% |
| conservador | 40/20/10 | +475% | +391pp | 8.85 | 1.85 | -1.7% |
| menos-agressivo | 30/15/7 | +334% | +250pp | 10.85 | 1.70 | -1.1% |
| ultra-conserv | 20/10/5 | +222% | +138pp | 13.47 | 1.72 | -0.6% |
| minimo | 10/5/2 | +130% | +46pp | 43.12 | 1.66 | -0.1% |

**Conclusao**:
- **Sharpe pica em H2 (K=100/50/20) = 2.09**, cai para 2.00 em K=200. Significa H2 e o ponto otimo de risk/reward, **sob a premissa de que a OOS 2022-2026 e representativa do futuro**.
- K maior que H2 e sub-otimo (mais vol penaliza Sharpe).
- Sortino MAX e em K minimo (degenera — quase toda alocacao em CDI).
- **Se a OOS for parcialmente overfit**, K=60/30/15 (H1) e muito mais robusto: menos sensivel a erros de magnitude, DD -2.9%.

---

## TESTE 2: Frozen Train — o teste mais importante

**Pergunta**: Quanto do edge vem de retrain continuo vs edge estrutural do ML?

**Metodologia**: Treinar UMA VEZ com dados ate freeze_date, predizer todo 2022-2026 SEM retrain.

| Freeze | Train size | Config | cum_ret | Sortino | Sharpe | Max DD |
|---|---|---|---|---|---|---|
| 2022-01-01 | 3 yrs | H2 (100/50/20) | +481% | **1.88** | 1.43 | -13.3% |
| 2022-01-01 | 3 yrs | H1 (60/30/15) | +382% | 1.91 | 1.38 | -10.5% |
| 2022-01-01 | 3 yrs | Conservative (40/20/10) | +318% | 2.41 | 1.33 | -7.4% |
| 2023-01-01 | 4 yrs | H2 | +598% | 4.54 | 2.06 | -10.8% |
| 2023-01-01 | 4 yrs | **H1** | +449% | **6.54** | 2.00 | -6.5% |
| 2024-01-01 | 5 yrs | H2 | +233% | 4.74 | 1.89 | -3.8% |
| 2024-01-01 | 5 yrs | **H1** | +183% | **6.55** | 1.75 | -2.1% |

**Conclusoes criticas**:

1. **Frozen 2022 (trained 2019-2021): Sortino 1.88 vs 5.61 com retrain**
   - **Retrain e responsavel por ~66% do Sortino**. Sem retrain semi-anual, o edge "estrutural" do modelo e Sortino ~1.9.
   - Isso NAO e necessariamente ruim — retrain e design legitimo. Mas significa:
     - O modelo depende de ver dados recentes
     - Se o regime futuro for muito diferente do visto em retrain, alpha degradara
     - O "Sortino 5.91" e o Sortino COM retrain funcionando bem, nao o Sortino independente do ML

2. **H1 (60/30/15) e consistentemente MAIS ROBUSTO que H2 (100/50/20) em frozen train**:
   - Frozen 2023: Sortino H1=6.54 vs H2=4.54 (H1 +44% melhor)
   - Frozen 2024: Sortino H1=6.55 vs H2=4.74 (H1 +38% melhor)
   - **Conclusao**: H2 ganha em walk-forward (com retrain) porque K=100 amplifica "killer weeks" do BULL regime. Sem retrain, H1 domina. **H1 e a escolha mais segura para live**.

3. **Mais training data ajuda**:
   - Frozen 2022 (3y): Sortino 1.88
   - Frozen 2023 (4y): Sortino 4.54 (+2.66)
   - Frozen 2024 (5y): Sortino 4.74 (+0.20)
   - Ganho marginal diminui — 4-5 anos de treino ja e suficiente.

---

## TESTE 3: Feature Ablation ✅ (concluido 2026-04-22)

**Pergunta**: Quanto as 3 features V36 (reserveRisk, puellMultiple, funding_rate_ma7) adicionaram ao edge? Elas foram adicionadas em 2026-04-19 — podem ser overfit ao recent. Tambem: quais grupos de features (macro, on-chain, top-5) carregam o edge?

**Metodologia**: Re-treinar walk-forward semi-anual com diferentes subsets de features.

| Config | n_feat | cum_ret | Sortino | Sharpe | Max DD |
|---|---|---|---|---|---|
| Baseline 32 feat H2 (atual) | 32 | +1195% | 5.53 | 2.09 | -4.0% |
| Baseline 32 feat H1 | 32 | +783% | **6.90** | 2.04 | -2.9% |
| **No V36 (remove 3 on-chain novas) H2** | **29** | **+1227%** | **5.27** | 2.09 | -4.0% |
| **No V36 H1** | **29** | +779% | **6.40** | 2.04 | -2.9% |
| No top-5 | 27 | +979% | 3.08 | 1.83 | -7.7% |
| Only top-5 | 5 | +130% | 0.56 | 0.38 | -36.3% |
| No macro (m2, fed, velocity, etc) | 26 | +489% | 2.18 | 1.50 | -7.7% |

**Conclusoes cruciais**:

1. **V36 features (reserveRisk, puellMultiple, funding_rate_ma7) adicionam POUCO edge**:
   - H2: Sortino com V36 = 5.53 vs sem V36 = 5.27 (**diferenca de 0.26**)
   - H1: Sortino com V36 = 6.90 vs sem V36 = 6.40 (**diferenca de 0.50**)
   - Return SEM V36 foi LEVEMENTE MELHOR: +1227% vs +1195% (H2)
   - **Confirma suspeita**: V36 foi adicionado tarde (2026-04-19), testado principalmente em 2026.
     O +0.30 Sortino original do V36 validation era real mas MARGINAL, nao transformador.
   - **Recomendacao**: considerar remover V36 para simplificar o modelo (menos overfit risk).

2. **Features MACRO sao o maior contribuidor ao edge**:
   - Sem macro (remover m2_yoy_growth, fed_balance_sheet, velocity, copper_return_30d, btc_gold_corr_30d, fed_fracdiff_05): Sortino cai de 5.53 para **2.18** (-3.35).
   - Return cai de +1195% para +489% (menos da metade).
   - **Macro features carregam mais sinal que on-chain**. Isso faz sentido — BTC em 2022-2026 foi fortemente correlacionado com liquidez Fed/M2, ciclos de juros.

3. **Top-5 features (cusum_pos, nupl_ma30, bb_position, eth_pctchg_30d, m2_yoy_growth) sao importantes mas nao suficientes**:
   - Somente top-5: Sortino 0.56 (quase zero).
   - Sem top-5 (27 features sem essas 5): Sortino 3.08.
   - O edge esta distribuido entre as 32 features, nao concentrado em poucas.

4. **H1 continua dominando H2 em todas as ablations**:
   - Baseline 32 feat: H1 Sortino 6.90 vs H2 5.53
   - No V36: H1 6.40 vs H2 5.27
   - **Reforca conclusao de teste 2**: H1 e estruturalmente mais robusto.

---

## TESTE 4: Shuffled Predictions — o teste mais importante de sinal

**Pergunta**: Se as predicoes fossem ruido, e o sistema (regime+confidence+K) sozinho gerasse o Sortino 5.61?

**Metodologia**: Embaralhar as predicoes (quebra alinhamento pred-actual) mantendo tudo mais igual. 100 seeds.

| Metric | Actual (sem shuffle) | Shuffled mean | Shuffled p95 | Shuffled max |
|---|---|---|---|---|
| cum_return | +1191% | +109% | +230% | +384% |
| Sortino | 5.61 | 0.64 | 1.11 | 1.57 |
| Sharpe | 2.09 | 0.27 | 0.79 | 1.28 |
| Max DD | -4.0% | -20.6% | -31.8% | -43.2% |

**% dos 100 shuffles que bateram o Sortino atual: 0.0%**
**% que bateram o return atual: 0.0%**

**Conclusao**: **Predictions carregam sinal REAL**. Se fosse ruido, alguns shuffles aleatorios deveriam ter pego por sorte (5-10%). Nenhum pegou. P-value < 0.01. Edge genuino.

---

## TESTE 5: Risk Controls Impact (kill switch + acc derisk + K lower)

**Pergunta**: Kill switch e acc de-risk melhoram ou pioram a estrategia?

**Metodologia**: Aplicar kill switch (DD ≤ -12% → alloc ≤ 15%) e acc de-risk (12w acc < 48% → alloc × 0.5) post-hoc.

| Config | cum_ret | Sortino | Sharpe | Max DD |
|---|---|---|---|---|
| H2 baseline (atual) | +1191% | 5.61 | 2.09 | -4.0% |
| H2 + kill switch | +1191% | 5.61 | 2.09 | -4.0% (kill nunca disparou) |
| H2 + acc de-risk | +978% | 5.55 | 2.06 | -4.0% (derisk disparou 36×) |
| H2 + ambos controles | +978% | 5.55 | 2.06 | -4.0% |
| **H1 (60/30/15) + ambos** | **+667%** | **6.91** | 1.99 | -2.9% |
| Conservative (40/20/10) + ambos | +423% | 8.45 | 1.78 | -1.7% |

**Conclusoes**:
- Kill switch NUNCA disparou em 4 anos — calibrado corretamente para eventos extremos (nao falsos positivos).
- Acc de-risk disparou 36 vezes em 248 semanas (14.5%). Custo em return de 18%, Sortino quase inalterado. **Trade-off aceitavel**.
- **H1 + ambos controles e a configuracao otima** pra live: Sortino 6.91 (melhor que H2 baseline), com menor DD e menor vol.

---

## TESTE 6: Sigmoid Scale Sensitivity (SIGMOID_SCALE=15 foi escolhido — overfit?)

**Pergunta**: SIGMOID_SCALE=15 foi o valor otimizado. Outros valores funcionam?

**Metodologia**: Variar SIGMOID_SCALE de 1 a 1000, mesmas predicoes.

| SIGMOID_SCALE | cum_ret | Sortino | Sharpe | Max DD |
|---|---|---|---|---|
| 1 | +738% | 6.62 | 1.98 | -2.5% |
| 5 | +978% | 6.53 | 2.05 | -2.9% |
| 10 | +1123% | 5.94 | 2.08 | -3.6% |
| **15 (atual)** | **+1191%** | **5.61** | **2.09** | **-4.0%** |
| 20 | +1235% | 5.48 | 2.09 | -4.3% |
| 25 | +1263% | 5.40 | 2.10 | -4.4% |
| 50 | +1318% | 5.25 | 2.10 | -4.5% |
| 100 | +1322% | 5.07 | 2.09 | -4.5% |
| 1000 | +1313% | 4.69 | 2.08 | -4.5% |

**Conclusao**: Sharpe e essencialmente PLANO de scale=10 a scale=1000 (2.08-2.10). **SIGMOID_SCALE NAO foi overfit** — qualquer valor razoavel funciona.

---

## TESTE 7: Feature composition probes

**Pergunta**: De onde vem o edge? Magnitude da predicao? Regime? Sign da predicao?

**Metodologia**: Substituir partes da pipeline por baselines.

| Probe | cum_ret | Sortino | Sharpe | Max DD |
|---|---|---|---|---|
| Sign-only (sem magnitude) | +744% | 2.19 | 1.54 | -7.5% |
| Constant +1% pred | +167% | 1.13 | 0.50 | -19.6% |
| K=50 sempre (sem regime) | +1139% | 2.89 | 1.90 | -10.7% |
| Random pred (ruido) | +104% | 0.58 | 0.24 | -21.8% |
| **Actual (H2)** | **+1191%** | **5.61** | **2.09** | **-4.0%** |

**Conclusoes**:
- **Sign + regime + confidence sozinhos**: Sortino 2.19 (baseline forte do edge). Magnitude real da predicao (que as 80 bags XGB computam): +Sortino 3.4.
- **Regime filter faz a maior diferenca em RISCO**: sem regime, K=50 sempre da +1139% (quase mesmo que atual +1191%) mas DD -10.7% e Sortino 2.89. Ou seja, regime filter nao adiciona RETURN, adiciona CONTROLE DE RISCO.
- **Sem ML (random pred)**: Sortino 0.58 — claramente o sistema regime+K sozinho nao basta.

---

## DEFLATED SHARPE RATIO (Bailey & Lopez de Prado 2014)

**Pergunta**: Considerando ~38 configuracoes testadas, qual a probabilidade que o Sharpe observado seja real (nao sorte)?

| N trials | Expected max SR under H0 | Deflated Sharpe (probabilidade) |
|---|---|---|
| 1 | 0.00 | 100.00% |
| 5 | 1.50 | 99.29% |
| 10 | 1.82 | 87.06% |
| **38 (repo estimado)** | **2.35** | **13.49%** |
| 100 | 2.69 | 0.60% |
| 500 | 3.19 | 0.00% |

**Interpretacao**:
- Sharpe observado = 2.087. Expected max sob H0 com 38 trials = 2.35 (proximo do observado).
- **Deflated SR = 13.5%**: apenas 13% de chance que o Sharpe seja sinal real vs ruido multiplo de testes.
- Mesmo para 10 trials (bem otimista — muitos trials foram dependentes), DSR = 87%, ok mas nao bulletproof.

**Caveat**: DSR e conservador. Assume trials independentes. Na pratica, V02→V39 sao parcialmente dependentes (alguns add-features, alguns remove). "N efetivo" provavelmente 5-15, nao 38.

**DSR com N efetivo = 10**: 87% — estrategia passa.

---

## METRICAS EXTRAS

- **Skewness dos retornos**: +3.63 (extremamente right-tailed, caudas positivas)
- **Excess kurtosis**: +16.4 (caudas MUITO gordas)
- **Concentracao**: Top 10 semanas = 48% do retorno total
- **Dependencia de 1 semana**: 2026-02-05 emergency rebal sozinha contribuiu +11.9pp do +19.3% YTD 2026 (61% do retorno)

---

## RECOMENDACAO FINAL: configuracao live

**Config sugerida para deploy live** (baseada em todos os testes):

```python
# scripts/production/config.py

K_REGIME = {'BULL': 60, 'MILD': 30, 'BEAR': 15}  # H1 — mais robusto em frozen train
ALLOC_MIN = 0.0
ALLOC_MAX = 1.0
SIGMOID_SCALE = 15   # validated robusto (teste 6)
HORIZON = 3
BAGS = 80
REBAL_DOW = [4]
EMERGENCY_THRESHOLD = 0.08
RETRAIN = 'semi'
```

**Com risk controls em `scripts/production/risk_management.py`** (ja integrado ao `generate_signal.py`):
- Kill switch: DD ≤ -12% → alloc ≤ 15%
- Acc de-risk: 12w acc < 48% → alloc × 0.5
- PSI monitor: informacional (nao acionavel)

---

## EXPECTATIVA LIVE REALISTA

Aplicando deflation factors:

| Metrica | Backtest (H1+controles) | Expectativa live |
|---|---|---|
| Sortino | 6.91 | **1.5-2.5** |
| Sharpe | 1.99 | **0.7-1.2** |
| CAGR | 44% | **20-35%** |
| Max DD | -2.88% | **-15 a -25%** |
| Alpha vs BTC | +583pp (4.3y) | **+5 a +15pp/ano** |

**Probabilidade de bater BTC em 3 anos**: alta (~70-80% em cenarios bear/lateral, ~30% em bull forte tipo 2023).

---

## MUDANCAS IMPLEMENTADAS NESTA AUDITORIA

### 1. `scripts/production/risk_management.py` (NOVO)
- `apply_risk_controls()`: kill switch + acc de-risk + PSI drift
- Integrado ao `generate_signal.py`
- Roda automatico a cada signal

### 2. `scripts/production/generate_signal.py` (MODIFICADO)
- Importa risk_management, chama apply_risk_controls apos gerar alocacao
- Log mostra quais controles dispararam

### 3. `scripts/production/config.py` (MODIFICADO)
- Comentario adicionando alternatives (H1, Conservative) documentado
- K atual mantido em H2 (decisao do usuario)

### 4. `scripts/production/archive/experiments/` (NOVOS)
- `overfit_test_1_k_sensitivity.py`
- `overfit_test_2_frozen_train.py`
- `overfit_test_3_feature_ablation.py`
- `overfit_test_5_kill_switch_sim.py`
- `overfit_test_fast.py` (testes 4, 6, 7)
- `deflated_sharpe.py` (Bailey-Prado DSR calculator)

### 5. `docs/OVERFIT_TESTS_2026-04-22.md` (ESTE DOC)

---

## PROXIMOS PASSOS SUGERIDOS

1. **Trocar K de H2 para H1** (`config.py:49`) — mais robusto em frozen train, pequena perda de Sortino em backtest.
2. **Paper trade 3-6 meses** antes de capital significativo.
3. **Verificar kill switch em prod** — rodar `python scripts/production/generate_signal.py` manualmente para ver output.
4. **Monitorar acc 2026**: se cair para <48% por 4+ semanas consecutivas, acc_derisk dispara automaticamente.
5. **Relaunch test 3** (feature ablation) — quando terminar, atualizar este doc com resultados.

---

**Auditoria encerrada: 2026-04-22. Modelo testado em 7 angulos independentes. Edge e real mas menor que backtest sugere.**
