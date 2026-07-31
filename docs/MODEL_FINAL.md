# Modelo Final — atualizado 2026-06-09 (M1, reporte oficial em BRL)

**Status**: Em producao (paper trade) com config **M1** = H1 + **sigmoid=5** +
**sem acc-derisk** + emergency executado **pos-close diario**. Mudancas validadas
em 10 seeds (secao 9, entrada 2026-06-09); kill switch e PSI permanecem;
acc/conf viram informacionais. **Moeda oficial de reporte: BRL consistente**
(perna BTC = preco USD × USDBRL; perna caixa = CDI).
**Metricas validadas em 10 seeds, Intel Ultra 9, XGBoost 3.2.0** (pinado em requirements.txt).
**Numeros antigos pre-2026-04-28 (single-seed em maquina diferente) NAO sao reproduziveis** —
ver historico de mudancas na secao 9.

> **HEADLINE OFICIAL M1 (BRL, 10 seeds, GROSS — 4 bps ≈ -0.6pp CAGR):**
> janela canonica 2022-01-07 → 2026-04-17:
> **CAGR +50.5% ± 0.4 | Sortino daily 3.84 ± 0.05 | Sharpe excess daily 2.35 |
> Max DD daily -5.34% ± 0.30 | cum +478%**
> Per-year: 2022 +26.9 / 2023 +51.7 / 2024 +93.8 / 2025 +37.7 / 2026(abr) +12.5 — todos positivos.
> Janela completa ate 2026-05-29: CAGR +48.2 ± 0.3, Sortino_d 3.73, DD -5.34%, 2026 YTD +9.3%.
> Benchmarks BRL (janela canonica): CDI +12.8% CAGR; BTC HODL +12.1% (DD -66.5%);
> 30/70 +15.6% (DD -20.2%). Fonte: `outputs/results/m1_brl_canonical_2026_06_09.json`.
>
> Os numeros HIBRIDOS (BTC em USD + CDI BRL) das secoes abaixo (ex.: CAGR +57.3 /
> Sortino 3.53 com derisk e sigmoid=15) ficam como referencia historica das configs
> anteriores (M2/M3) — NAO sao o headline oficial.

---

## 1. Configuracao Final (arquivo: `scripts/production/config.py`)

```python
K_REGIME = {'BULL': 60, 'MILD': 30, 'BEAR': 15}   # H1 — robusto em frozen train
ALLOC_MIN = 0.0          # no-short
ALLOC_MAX = 1.0
SIGMOID_SCALE = 5        # 15 -> 5 em 2026-06-09 (M1): +0.34 Sortino_d, -1.55pp DD_d,
                         # Sharpe igual, -6.1pp CAGR (10-seed pareado)
HORIZON = 3              # 3 dias forward
BAGS = 160               # 2026-04-29: 80 → 160 (variance 7× menor entre seeds)
REBAL_DOW = [4]          # Sexta
EMERGENCY_THRESHOLD = 0.08  # >8% move (close-to-close) — EXECUTAR POS-CLOSE, nao intraday
RETRAIN = 'semi'         # Jan + Jul
# risk_management.py: ACC_DERISK_ENABLED = False (2026-06-09); kill switch ativo
```

**32 features** (E1 D7: V29 base 29 + V36 on-chain 3).
**Risk controls ativos** (kill switch, acc derisk, PSI monitor) em `risk_management.py`.

---

## 2. Performance FINAL — 10 seeds canonicos (2026-04-28)

**Periodo**: 2022-01-07 → 2026-04-17 (4.28 anos OOS, 248 rebals)
**Metodo**: walk-forward expanding-window, retrain semi-anual, sem custo aplicado

### Headline (BRL, gross pre-cost, 160 bags, com acc-derisk, 10-seed, BCB CDI):

| Metrica | Valor | Reportar como |
|---|---|---|
| **CAGR** | **+57.3%** | +57.3% ± 0.3pp (10-seed std) |
| **Sortino daily** | **3.53** | 3.53 ± 0.06 (variance 7x menor que 80 bags) |
| **Sharpe excess (daily)** | **2.47** | 2.47 ± 0.01 (weekly: 1.94 ± 0.02) |
| **Max DD daily** | **-7.14%** | -7.14% ± 0.25% |
| cum_return | +597% | +597% ± 7pp |

Nota: BAGS aumentado de 80 → 160 em 2026-04-29. Mean metrics quase iguais,
mas variance entre seeds caiu 7× — strategy muito mais previsivel em live.

### Comparativo no mesmo periodo:

| Strategy | CAGR | Sortino daily | Sharpe excess | Max DD daily |
|---|---|---|---|---|
| 100% CDI | +13.0% | ∞ | 0.00 | 0% |
| 30% BTC + 70% CDI | +17.0% | 1.13 | 0.30 | -21.7% |
| 100% BTC HODL | +15.5% | 0.56 | 0.30 | -66.7% |
| **H1 atual (BRL, gross, 160 bags)** | **+57.3%** | **3.53** | **2.47** | **-7.14%** |

### 2026 YTD live OOS (1 jan → 17 abr, 17 rebals)

Modelo treinado em 2026-01-01 — 3.5 meses de **strict out-of-sample**:

| | Estrategia | BTC B&H | CDI |
|---|---|---|---|
| Retorno (105 dias) | **+19.67%** | -14.36% | +4.09% |
| CAGR anualizado | **+86.7%** | -41.7% | +14.9% |
| Excess vs BTC | — | +34.0pp | — |

Top 3 rebals 2026 contribuiram com 100% do retorno YTD (concentracao tipica de
strats de regime/momentum).

**Por que H1 e nao H2 ou Conservative?**
- vs H2: aceita CAGR menor em troca de +38-44% Sortino em frozen-train, -14% DD, robusto a regime shift
- vs Conservative: aceita +12% DD em troca de +31% CAGR, Sortino quase igual
- H1 e o **"sweet spot"** — Sharpe excess 1.93 proximo do pico

---

## 3. Auditoria de cada parametro — o que esta certo vs errado

> **Nota:** os numeros nesta secao (sigmoid/confidence/regime sweeps: Sortino ~3.56,
> DD ~-7.87% etc.) vem de um sweep **single-config mais antigo**, usado para
> comparar PARAMETROS entre si. NAO sao o headline canonico — esse e o 10-seed,
> BAGS=160, BCB CDI da secao 2 (Sortino daily 3.53, DD -7.14%). Use esta secao
> para os trade-offs relativos, nao para os valores absolutos.

### 🟢 K = 60/30/15 (H1) — CORRETO

**Evidencia a favor:**
- Sharpe 2.48 — perto do topo da curva (K=50 picoh em 2.55, diferenca desprezivel)
- DD -7.87% — 14% melhor que H2
- **Frozen train 2023**: Sortino H1=6.54 vs H2=4.54 (+44%)
- **Frozen train 2024**: Sortino H1=6.55 vs H2=4.74 (+38%)
- Calmar 8.28 — excelente

**O que esta "errado" (ou ao menos questionavel):**
- Ainda amplifica predicoes 60x em BULL — pred +1.67% ja clipa em 100% alloc
- H1 foi escolhido porque frozen test mostrou dominancia, mas esse teste tambem foi ex-post
- **Nao temos garantia** que K=60 e o otimo para dados OOS DE VERDADE (2026+ live)

**Veredicto**: K=60/30/15 e **a melhor escolha razoavel**. Alternativa viavel se quiser mais seguranca: K=40/20/10 (Conservative).

### 🟡 SIGMOID_SCALE = 15 — SUB-OTIMO MAS OK

**Sensitivity test (K=H1, daily DD):**
| sigmoid | CAGR | Sortino | DD |
|---|---|---|---|
| 1 | +47.7% | **4.07** | -5.65% |
| 5 | +57.7% | 3.81 | -6.73% |
| 10 | +63.8% | 3.63 | -7.46% |
| **15 (atual)** | **+66.5%** | **3.56** | **-7.82%** |
| 25 | +69.1% | 3.51 | -8.12% |
| 100 | +71.4% | 3.42 | -8.92% |

**O que esta certo:**
- Sharpe peak area (2.53 em sigmoid=15, 2.54 em sigmoid=5-10) — diferenca < noise
- Qualquer sigmoid 5-25 funciona bem

**O que esta errado/questionavel:**
- Sigmoid=15 **NAO e o optimo de Sortino** — sigmoid=1 da Sortino 4.07 (+14% melhor)
- Sigmoid foi escolhido para maximizar RETURN, nao Sortino
- Sigmoid=5 seria mais defensivo: Sortino 3.81 vs 3.56, DD -6.73% vs -7.82%

**Alternativa recomendada**: **sigmoid=5** (perde 9pp CAGR, ganha 0.25 Sortino, -1.1pp DD). Nao implementado para preservar backtest comparability, mas vale a pena testar em paper trade.

### 🟢 CONFIDENCE SCALING — CORRETO (modestamente)

**Test:**
| Config | CAGR | Sortino | DD | avg alloc |
|---|---|---|---|---|
| COM confidence (atual) | +66.5% | **3.56** | -7.82% | 15.5% |
| SEM confidence | +76.0% | 3.41 | -9.02% | 17.7% |

**O que esta certo:**
- Confidence adiciona **+0.15 Sortino** e reduz DD em 1.2pp
- Custa 9.5pp CAGR — trade-off aceitavel
- Design alinhado com Kelly fracionado (literatura quant valida)

**O que esta questionavel:**
- Sortino gain pequeno — pode ser noise
- P(up) pode nao ser bem calibrado — model outputs podem ser overconfident
- Beneficio ja foi "absorvido" nos testes ex-post

**Veredicto**: manter confidence ativo. Beneficio pequeno mas positivo, alinhado com teoria.

### 🟢 REGIME DETECTION (SMA50/200) — CORRETO (essencial)

**Evidencia:**
- Sem regime filter (K=50 flat): Sortino 2.89, DD -10.7%
- Com regime: Sortino 3.56, DD -7.87%
- **Regime adiciona +0.7 Sortino e reduz DD em 3pp**

**O que esta certo:**
- Filtro eficaz — remove exposure em BEAR regime
- Componente crucial do 57.7% tempo-em-CDI que caracteriza o modelo

**O que esta questionavel:**
- SMA50/200 tem LAG conhecido (20-50 dias)
- Whipsaws em regimes choppy (ex: Oct 2025 BULL->MILD->BEAR em 4 semanas)
- Modelo usa SOMENTE SMA50/200 — poderia complementar com volatility regime

**Veredicto**: manter, mas considerar adicionar filtro de vol como overlay futuro.

### 🟢 HORIZON = 3 — CORRETO

**Evidencia (horizon_ablation_4y.csv):**
- H=3: Sortino 5.61 (weekly)
- H=7 same K: Sortino 2.32 (weekly)
- H=7 scaled K: Sortino 3.80 (weekly)
- H=3 dominates with same data

**Veredicto**: H=3 e robusto. Nao mexer.

### 🟢 BAGS = 160 — VALIDADO 2026-04-29

V22 antigo testou 40/80/120/160 (USD, sem cost) e concluiu 80 como otimo.
Re-validado em 10 seeds (BRL, BCB CDI, gross): **160 bags ganha em
estabilidade**. Mean Sortino daily quase igual (80: 3.51 vs 160: 3.53), mas
std entre seeds cai de **0.10 para 0.06**. DD tambem mais estavel.
Custo: 2× training time por cutoff. Adotado em producao.

### 🟢 EMERGENCY_THRESHOLD = 8% — CORRETO

Emergency rebal capturou +11.9pp em 2026-02-05 sozinho. Calibracao historica valida.

### 🟡 RETRAIN = semi-anual — OK MAS DEPENDENTE

**Frozen train tests mostraram:**
- Modelo frozen em 2022: Sortino 1.88 (vs 5.61 com retrain)
- **66% do Sortino vem do retrain**

**O que esta certo:**
- Semi-anual e melhor que annual, monthly, quarterly, triggered
- Validado em retrain_parallel_agg.csv (5 seeds stability)

**O que esta questionavel:**
- Dependencia forte de retrain sugere adaptacao a regime
- Se regime mudar DRAMATICAMENTE em 2026+, semi pode ser tarde demais

**Veredicto**: manter semi-anual, mas o kill switch + acc derisk cobrem regime shift abrupto.

---

## 4. Teste de Cost Stress — H1 e ROBUSTO a custos

| Custo por rebal | CAGR | Sortino | DD |
|---|---|---|---|
| 0 bps (bruto) | +59.0% | 3.52 | -7.13% |
| **4 bps (atual realista BRL)** | **+58.4%** | **3.51** | **-7.15%** |
| 8 bps | +57.8% | 3.51 | -7.18% |
| 15 bps | +56.6% | 3.50 | -7.23% |
| 25 bps | +55.0% | 3.48 | -7.30% |
| 50 bps (pessimista) | +51.0% | 3.42 | -7.50% |

**Conclusao**: mesmo a 50 bps pessimista, CAGR 51% e Sortino 3.42. **Custos nao sao gargalo**.

Cost atual mudou de 8 bps -> 4 bps em 2026-04-29 (realismo BRL: spreads BTC/BRL
em corretoras locais ficaram menores que estimativa inicial conservadora).

---

## 5. Expectativa Live Realista (apos deflation de multiple testing)

| Metrica | Backtest H1 (10-seed) | **Live realista** | vs 30% BTC estatico | vs 100% BTC |
|---|---|---|---|---|
| CAGR | +57.3% | **+25-40%** | +17% (vence) | +15.5% (vence) |
| Sortino daily | 3.53 | **1.5-2.5** | 1.13 (vence) | 0.56 (vence) |
| Sharpe excess (w) | 1.94 | **0.7-1.0** | 0.30 (vence) | 0.30 (vence) |
| DD daily | -7.14% | **-15 a -25%** | -22% (≈ empata) | -67% (vence) |

**Bottom line**: mesmo no pior cenario live, H1 deve **bater 30% BTC estatico** em CAGR (~10pp mais), Sortino (1.5x), e DD (igual ou menor).

**Validacao 2026 YTD live**: +19.67% vs BTC -14.36% em 105 dias OOS — acima
do range esperado, mas concentrado em 3 rebals (top 3 = 100% do retorno).

---

## 6. O que pode dar errado (riscos nao eliminados)

### 🔴 Alto impacto
- **Regime shift drastico** (ex: BTC vira ativo puramente macro-driven) — retrain semi pode ser lento, kill switch cobre parcial
- **Multiple-testing overfit** — 38+ trials → Sortino real pode ser 2x menor que observado
- **Concentracao em poucas semanas** — top 10 semanas = 48% do retorno. Perder 2-3 delas = metade do alpha

### 🟡 Medio impacto
- **V36 features noise** — ablation mostrou +0.5 Sortino, dentro do ruido. Pode sumir em live.
- **Acurracy drift** — 2026 YTD caiu para 47% (abaixo de coinflip). Se persistir, acc derisk dispara automaticamente.
- **Custos BRL reais** — se > 25 bps, perde 4pp CAGR mas ainda bom

### 🟢 Baixo impacto
- API externa cair (pipeline detecta NaN, median-fill salva)
- Seed instability (10-seed validation Sortino std 0.07 — estavel)

---

## 7. Deploy checklist — Estado atual

- [x] Config H1 aplicado em `config.py:45`
- [x] Risk management integrado em `generate_signal.py` (kill switch + acc derisk + PSI)
- [x] Pipeline roda sem erro (`python scripts/production/generate_signal.py`)
- [x] Cache de modelos preservado (K e multiplicador post-prediction, nao precisa retrain)
- [x] Signal output atualizado: `[E1-D7 K=60/30/15]`
- [x] Rolling acc 12w exibido (atual 58.3%, threshold 48%)
- [x] Current DD exibido (atual 0%, kill at -12%)
- [x] PSI monitor exibido (info only, nao dispara derisk)

**Deploy APROVADO para live**, com as seguintes recomendacoes:

### Pre-deploy (ate 2026-04-25)
1. [ ] Paper trade 3-6 meses antes de capital significativo
2. [ ] Setup cron: `python scripts/production/run_daily.py` diario
3. [ ] Define alerta email se kill switch dispara

### Semi-annual (2026-07-01)
4. [ ] Primeiro retrain automatico — verificar que modelo novo bate walkforward OOS
5. [ ] Revisar PSI top features — se mudou muito, investigar

### Em caso de problema
- Se acc 12w < 48% por 4+ semanas: **acc derisk dispara automaticamente** (alloc × 0.5)
- Se DD total <= -12%: **kill switch dispara automaticamente** (alloc cap 15%)
- Manual override: editar `config.py` e restart

---

## 8. Scripts de auditoria (rodar se algo estranhar)

```bash
# Re-validar walkforward OOS (~30min)
python scripts/production/walkforward_backtest.py --compare

# Auditoria diaria completa (rapida, usa preds existentes)
python scripts/production/archive/experiments/final_audit_daily_dd.py

# Deflated Sharpe (Bailey-Prado)
python scripts/production/archive/experiments/deflated_sharpe.py

# Kill switch simulation
python scripts/production/archive/experiments/overfit_test_5_kill_switch_sim.py

# Ver sinal atual
python scripts/production/generate_signal.py
```

---

## 9. Historico de mudancas

### 2026-04-22 — H1 deploy + risk controls
1. **K_REGIME: H2 (100/50/20) -> H1 (60/30/15)** — `config.py:45`
   - Razao: frozen train tests mostraram H1 38-44% mais Sortino quando modelo e testado sem retrain recente
2. **Novo modulo `risk_management.py`** — kill switch + acc derisk + PSI monitor
3. **Integracao em `generate_signal.py`** — alloc ajustado automaticamente, log mostra controles
4. **7 scripts de overfit teste** em `scripts/production/archive/experiments/overfit_test_*.py`
5. **Documentacao em `docs/OVERFIT_TESTS_2026-04-22.md`** e este doc

### 2026-04-28 — Reconciliacao de metricas + estabilidade

1. **10-seed validation rodado** — substitui numeros single-seed antigos:
   - Antigo: CAGR +65.2%, Sortino 3.61, DD -7.87% (single-seed, maquina antiga)
   - Novo (BAGS=80): CAGR +56.8% +/- 0.3%, Sortino 3.51 +/- 0.10, DD -7.15% +/- 0.29% (10 seeds)
   - Re-validado 2026-05-30 (BAGS=160, BCB CDI, gross): CAGR +57.3% +/- 0.3%, Sortino daily 3.53 +/- 0.06,
     Sharpe daily 2.47 +/- 0.01 (weekly 1.94), DD daily -7.14% +/- 0.25% (variance menor: std 0.10 -> 0.06)
2. **XGBoost pinado** em `requirements.txt`: `xgboost==3.2.0`
   - Razao: XGBoost training nao-deterministico entre CPUs e versoes minor
   - Numeros antigos foram gerados em maquina diferente — nao reproduziveis sem hardware identico
3. **5 testes adicionais (todos rejeitados)**:
   - Huber loss: pior (-0.56 Sortino) — fat tails do BTC sao em CIMA, alpha vem do squared error
   - Drop V36 features: empate tecnico — mantem por proteção downside
   - Vol regime overlay: marginal demais
   - Acc derisk variantes (rampa, conf-weighted, vol-conditional): so conf>0.80 marginal
   - Quantile regression sizing: TODOS variants pioram (q50, Kelly, hibrido) — degrada upside
4. **Validacao 2026 YTD**: +19.67% vs BTC -14.36% em 105 dias OOS strict (modelo treinado Jan/2026)
5. **Cleanup do repo**: docs antigos arquivados, scripts obsoletos zipados, 5 zips em `archive/`

### 2026-06-09 (parte 2) — M1 adotado: sigmoid 15 → 5 + reporte oficial em BRL

Decisao do usuario no mesmo dia, apos a validacao 10-seed do grid: adotar a
versao de **maior Sharpe (2.43) e maior Sortino (3.96 hibrido / 3.84 BRL)**.

1. **`SIGMOID_SCALE = 5`** (`config.py`): pareado em 10 seeds, +0.34 Sortino_d
   (t=58), -1.55pp DD daily, Sharpe igual, custo -6.1pp CAGR. Dial pos-predicao
   — cache de modelos inalterado (fingerprint cobre so features/BAGS/HORIZON/
   XGB_PARAMS). Com sigmoid=5 o conf-gate do antigo derisk nunca passaria de
   0.80 — coerente com o desligamento do derisk (parte 1).
2. **Reporte oficial passa a ser BRL consistente** (BTC×USDBRL + CDI):
   headline M1 na janela canonica: **CAGR +50.5% ± 0.4, Sortino_d 3.84 ± 0.05,
   Sharpe_d 2.35, DD_d -5.34%**, todos os anos positivos. Hibrido (referencia
   interna/historica): CAGR 50.2%, Sortino_d 3.96, DD -5.55%.
   Script: `archive/experiments/m1_brl_canonical_2026_06_09.py`.
3. `walkforward_backtest.csv` regenerado sob M1; `signal_history.csv` NAO foi
   reescrito (registro vivo — rows antigas refletem a config vigente a epoca;
   M1 vale do proximo sinal em diante).
4. Testes: 133 verdes (test_config_consistency e test_signal_logic atualizados).
5. Expectativa live pos-deflation (BRL): CAGR 20-35%, Sortino 1.5-2.5,
   DD daily -15 a -25%.

### 2026-06-09 (parte 1) — M2: acc-derisk desligado + emergency pos-close (validacao 10-seed)

Grid de variantes pareado sobre o walkforward canonico + dump fresco de
predicoes de 10 seeds (janela completa 2022-01 → 2026-06). Scripts:
`archive/experiments/{variant_grid,multiseed_eval,seed_preds_dump}_2026_06_09.py`;
resultados em `outputs/results/{variant_grid,multiseed_eval}_2026_06_09.json`.

1. **Acc-derisk DESLIGADO** (`risk_management.py: ACC_DERISK_ENABLED=False`):
   10-seed pareado, a regra custava **+2.2pp CAGR e +0.08 Sortino_d com max DD
   IDENTICO** (t=38.8/8.4) — nunca reduziu drawdown em 4.4 anos, so cortou
   semanas boas (2026: +10.5% sem vs +6.5% com). Monitoramento de acc/conf
   continua sendo reportado (informacional). Kill switch e PSI inalterados.
2. **Emergency: detectar E executar logo apos o close do candle diario
   (00:00 UTC)** — nao operar intraday no cruzamento de -8%. Teste das 3
   convencoes de execucao: intraday-threshold custa **-5.0pp CAGR / -0.45
   Sortino_d** vs pos-close (em dias de crash o preco segue caindo apos o
   trigger; esperar o close compra o dip mais barato). Atraso de 24h (pior
   caso) custa o mesmo ~5pp — executar minutos apos o close ≈ convencao do
   backtest. **Sob execucao honesta o emergency segue positivo vs so-sextas**
   (+0.13 Sortino_d, +0.5pp CAGR, t=13.5) — mantido como controle de risco;
   o antigo "+93pp" era majoritariamente artefato de execucao.
3. **BRL consistente (BTC×USDBRL + CDI)**: modelo segura — M2 em BRL:
   CAGR 54.6% / Sortino 3.34 / DD -6.83% (~2-3pp de CAGR abaixo do hibrido;
   o real apreciou 5.57→5.20 na janela). O headline canonico e o HIBRIDO
   (BTC em USD + CDI BRL) — rotulo corrigido.
4. **Reparos do confidence head todos rejeitados**: gating assinado (3.41),
   recalibracao isotonica OOS (3.43), aposentar head (3.49/DD -8.1%) — todos
   piores que o baseline 3.56. sigmoid=5 segue validado como candidato
   defensivo (So_d 3.96 / DD -5.55% / CAGR 50.2%) para decisao no retrain de
   julho (M1).
5. Testes: 133 verdes (novo: `test_acc_derisk_disabled_by_default`).

### 2026-05-31 — Auditoria de codigo (Lote A + Lote B parcial) + suite de testes

1. **Lote A** (sem mudanca de predicao): fix de look-ahead no fetch
   (`utcnow().timestamp()` estava +3h local -> `time.time()`/UTC), dead code,
   doc consistency, headline unificado, deps pinadas, cache fingerprint.
2. **Lote B — CUSUM train/serve fix**: `build_features.py` agora calcula
   cusum_pos/cusum_neg com retornos SIMPLES (igual a base de treino
   `add_regime_features.py`), antes usava log -> corrige skew train/serve e a
   descontinuidade no seam de 2026-03-03.
   - Re-validado (10-seed, BAGS=160, BCB CDI, gross): CAGR **+57.2% +/- 0.4%**,
     Sortino daily **3.52 +/- 0.06**, Sharpe daily 2.47, DD daily -7.14%, cum +597%.
   - **Dentro do ruido** vs pre-fix (3.53) — sem regressao; valor real e a
     consistencia de feature no live. Headline 3.53 +/- 0.06 segue cobrindo.
3. **Lote B — bfill leak (source fix)**: removido `.ffill().bfill()` ->
   `.ffill()`/`.fillna(0)` em `build_dataset.py` e `add_extra_features.py` (o bfill
   preenchia NaN-lider com valor FUTURO). So afeta rebuilds futuros da base
   congelada; `dataset_enhanced.csv` atual nao foi reconstruido (impacto limitado
   ao warmup pre-2022, dentro do treino).
4. **Suite de testes** (repo nao tinha nenhuma): `tests/`, 132 testes verdes,
   incl. property-test provando que `build_features` e estritamente backward-looking.
5. **Pendente (Lote B/C)**: on-chain reporting lag (recomendado testar shift de 1d),
   kill-switch DD window e acc-horizon (defensaveis como estao), limpeza do legado.

---

## 10. Resumo Executivo em 5 linhas

1. **Modelo**: XGBoost ensemble prediz BTC 3d, regime SMA50/200 + confidence sigmoid escolhem tamanho da aposta, rebal Friday+emergency. **Edge e REAL** (0 de 100 shuffles bateu baseline em p<0.01).
2. **Config final**: K=60/30/15 (H1), sigmoid=15, 32 features, squared error loss, com risk controls (kill at -12% DD, acc derisk <48%).
3. **Backtest 4.28y canonico (10 seeds, BAGS=160, BCB CDI, gross)**: CAGR **+57.3% +/- 0.3%**, Sortino daily **3.53 +/- 0.06**, Sharpe excess daily **2.47 +/- 0.01** (weekly 1.94), DD daily **-7.14% +/- 0.25%**.
4. **2026 YTD live OOS** (105 dias, 17 rebals): **+19.67% vs BTC -14.36%** — +34pp excess.
5. **Live realista (post-deflation)**: CAGR 25-40%, Sortino 1.5-2.5, Sharpe excess 0.7-1.0, DD daily -15 a -25%.

**Proximo rebal live: Sexta 2026-05-01 (refetch dataset necessario, atrasado 7d).**
