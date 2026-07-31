# Model Audit — 2026-04-20

**Ponto de parada:** H2 deployed, walk-forward validado, docs consolidados.

Ler em ordem: esse audit → `STATUS_FINAL.md` → `AI_ONBOARDING.md`.

---

## 1. TL;DR em 5 linhas

1. Modelo em produção: **E1 D7 combo no-short + H2 balanced K**, XGBoost ensemble 80+80 bags, 32 features, rebalance semanal (Friday) + emergency >8%.
2. Walk-forward OOS validado (sem look-ahead) em **29 semanas reais** (2025-10-03 → 2026-04-19): **+29.79%** vs BTC **−38.83%** = **+68.6pp alpha**.
3. 2026 YTD: **+18.4%** vs BTC **−16.9%** = **+35.3pp** (drawdown estratégia −4.0%, drawdown BTC −44.2%).
4. **Decisão aberta no config**: H1 Conservative (K=60/30/15, Sortino max) vs **H2 Balanced (K=100/50/20, Return max — ativo desde 2026-04-20)**.
5. Pipeline de produção é walk-forward safe por design (retrain semi-annual, gap=5d no target) — sem risco de look-ahead nas predições futuras.

---

## 2. Arquitetura do modelo

```
┌─────────────────────────────────────────────────────────────────┐
│  ENTRADA: 32 features de cada dia (macro, on-chain, técnicas)    │
└─────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
     ┌──────────────────────┐    ┌──────────────────────┐
     │  XGB REGRESSOR x80   │    │  XGB CLASSIFIER x80  │
     │  (prediz BTC 3d ret) │    │  (prediz P(up 3d))   │
     └──────────────────────┘    └──────────────────────┘
                │                           │
                ▼                           ▼
           prediction                    p_up
                │                           │
                │                      conf = σ(|p_up−0.5|·15)
                │                           │
                └─────────────┬─────────────┘
                              ▼
                   regime = SMA50 vs SMA200
                              │
                              ▼
              raw_alloc = prediction × K[regime] × conf
                              │
                              ▼
               alloc = clip(raw_alloc, 0.0, 1.0)
                              │
                              ▼
       portfolio_return = alloc × BTC + (1−alloc) × CDI
```

**Configuração atual (`scripts/production/config.py`):**

| Parâmetro | Valor | Nota |
|---|---|---|
| `K_REGIME` | **{BULL: 100, MILD: 50, BEAR: 20}** | H2 balanced (set 2026-04-20) |
| `ALLOC_MIN` | 0.0 | No-short (V31.7 mandate) |
| `ALLOC_MAX` | 1.0 | Max 100% BTC |
| `SIGMOID_SCALE` | 15 | Confidence amplification |
| `HORIZON` | 3 | Predict 3d forward |
| `BAGS` | 80 | Ensemble size per model type |
| `REBAL_DOW` | [4] | Friday only |
| `EMERGENCY_THRESHOLD` | 0.08 | Rebal if \|daily_ret\| > 8% |
| `RETRAIN` | 'semi' | Jan 1 + Jul 1 |

---

## 3. Features (32)

| Grupo | N | Features |
|---|---|---|
| Regime/momentum | 6 | cusum_pos, cusum_neg, mr_score_30d, adx, structural_break_score, bb_position |
| Correlação crypto | 4 | eth, eth_btc_ratio, eth_pctchg_30d, stablecoin_zscore |
| Volatilidade/risk | 3 | volatility_7d, hurst_60d, sortino_30d |
| On-chain (V36 NEW) | 3 | **reserveRisk, funding_rate_ma7, puellMultiple** |
| On-chain legacy | 2 | nupl_ma30, stablecoin_supply_change_30d |
| Macro | 5 | m2_yoy_growth, fed_balance_sheet, velocity, copper_return_30d, btc_gold_corr_30d |
| Microstructure | 4 | basis_ma7, basis_pct, volume_sma20_ratio, aroon_down_30d |
| Futures | 1 | (dentro do basis) |
| Fractal/statistical | 3 | fractal_dimension_30d, kpss_stat_30d, half_life_60d |
| Fracdiff | 2 | price_fracdiff_05, fed_fracdiff_05 |

**Top 5 mais informativas (gain importance):** cusum_pos, nupl_ma30, bb_position, eth_pctchg_30d, m2_yoy_growth.

---

## 4. Performance validada (walk-forward, 29 semanas reais)

### Cumulative

| | H1 (K=60/30/15) | **H2 (atual, K=100/50/20)** | BTC |
|---|---|---|---|
| Return | +24.12% | **+29.79%** | −38.83% |
| Excess vs BTC | +62.95pp | **+68.62pp** | — |
| Max DD | — | **−4.01%** | −44.22% |
| H2 − H1 | — | **+5.67pp** | — |

### Performance por mês

| Mês | Rebals | Avg Alloc | Strat | BTC | Excess |
|---|---|---|---|---|---|
| 2025-10 | 5 | 14% | +4.3% | −15.5% | **+19.8pp** |
| 2025-11 | 4 | 31% | +4.2% | −13.6% | **+17.7pp** |
| 2025-12 | 4 | 4% | +0.9% | +0.7% | +0.1pp |
| 2026-01 | 5 | 17% | −2.6% | −30.1% | **+27.5pp** |
| 2026-02 | 5 | **58%** | +11.5% | +8.3% | +3.2pp |
| 2026-03 | 4 | 29% | −0.4% | −1.7% | +1.3pp |
| 2026-04 | 3 | **59%** | +9.4% | +11.7% | −2.2pp |

**Observações:**
- Modelo brilha em **quedas** (Out, Nov, Jan 2026) — fica em CDI e protege capital.
- Setembro-Outubro 2025 e Janeiro 2026 mostram o valor real do no-short: capturou ~20-28pp de alpha só por estar fora do BTC.
- Abril 2026: rallies fortes, alloc ~59% — estratégia ganhou +9.4% mas ficou −2.2pp atrás do BTC puro. H1 teria ficado ainda mais atrás.

### Accuracy ponderada (29 predições)

| Métrica | H2 |
|---|---|
| Raw direção 3d | 55% |
| Magnitude-weighted (\|BTC 3d\|) | 61% |
| Confidence-weighted (\|p_up−0.5\|) | 66% |
| **Alloc-weighted** | **79%** |
| Correlação pred vs actual 3d | +0.39 |
| Mean \|prediction\| vs \|actual\| | 1.93% vs 2.89% (modelo subestima ~33%) |

Insight: modelo acerta **4 de cada 5 vezes** que aposta pesado — é isso que sustenta o Sortino alto apesar da accuracy raw só 55%.

### Stats de alocação (H2)

| | Valor |
|---|---|
| Média | 29.4% |
| Mediana | 25.5% |
| % semanas em 0% (full CDI) | 30% |
| % semanas > 50% | 23% |
| % semanas > 80% | 7% |

Modelo é **conservador por padrão** — aloca forte só quando confidence+prediction são grandes.

---

## 5. Validação histórica 4 anos (V36, 10 seeds)

**Metodologia:** Walk-forward com retrain semi-annual (`v36_final_validation.py:126-158`). Cada período de 6 meses usa modelo treinado **apenas com dados anteriores** + gap de 5 dias no target (evita leakage forward-looking). Não há look-ahead. O que acumula entre retrains é o tamanho do training set (cada modelo vê mais histórico que o anterior), o que é **comportamento correto** — em produção o modelo também aprende com o tempo.

| Config | Sortino | Sharpe | Return | DD | CAGR | 2026 YTD |
|---|---|---|---|---|---|---|
| **H1** | **6.43** | 2.50 | +893% | **−8.0%** | 79% | +14.4% |
| **H2** | 5.72 | 2.35 | **+1228%** | −9.2% | 95% | +16.8% |

Período: 2022-01-01 → 2026-04-19 (out-of-sample, treinado 2019-2021 + retrain semi-annual).

### Performance ano-a-ano (H2)

| Ano | Estrat | BTC | Excess | Contexto |
|---|---|---|---|---|
| 2022 bear | +42% | −64% | **+106pp** | Crypto winter — modelo ficou defensivo |
| 2023 bull | +99% | +156% | −57pp | Subestimou rally pós-FTX |
| 2024 bull | +135% | +121% | +14pp | ETF approval + halving — modelo acompanhou |
| 2025 flat | +65% | −6% | **+71pp** | Lateral com spikes — sweet spot do modelo |
| 2026 YTD | +18% | −17% | **+35pp** | Recessão/queda — modelo segurou capital |

Positiva em **todos os anos** desde 2022.

---

## 6. Decisão: H1 vs H2

| Critério | **H1** (K=60/30/15) | **H2** (K=100/50/20) |
|---|---|---|
| Sortino (4y) | **6.43** | 5.72 |
| Return (4y) | +893% | **+1228%** (+38% extra) |
| DD | **−8.0%** | −9.2% |
| Filosofia | Max risk-adjusted | Max absolute return |
| Ideal para | Mandato conservador, prop desk, fund retorno estável | Retail, long-biased puro, busca de alpha absoluto |
| OOS 6.5m | +24.1% | +29.8% |

**Hoje em produção:** H2 (set 2026-04-20 por preferência do usuário por retorno absoluto).

**Como trocar (1 linha em `config.py:49`):**
```python
# H1 conservative:
K_REGIME = {'BULL': 60, 'MILD': 30, 'BEAR': 15}

# H2 balanced (ATUAL):
K_REGIME = {'BULL': 100, 'MILD': 50, 'BEAR': 20}
```

**Não precisa retreinar** — K é multiplier pós-predição. Reversível imediato.

---

## 7. Strengths e Weaknesses

### Strengths

1. **Downside protection excepcional** — DD −4% vs BTC −44% na janela OOS. No-short + confidence scaling + K regime tiram exposição nos sinais ruins.
2. **Capture dos big moves** — 91% alloc no bounce de 12% em 2026-02-05. 93% alloc no rally de 6.8% em 2025-11-21.
3. **Consistência ano-a-ano** — positiva todos os 5 anos testados.
4. **Robustez** — validado com 10 seeds (Sortino 6.39 ± 0.07), não é artefato de 1 seed.
5. **Features diversificadas** — macro + on-chain + microstructure (não só TA).

### Weaknesses

1. **Accuracy raw baixa (55%)** — pouco acima de coin flip. Depende do sizing pra converter em alpha.
2. **Subestima magnitude (~33%)** — por isso K multiplier é grande (50-100x). Se o mercado mudar regime de vol, K pode ficar mal calibrado.
3. **Conservador em bull markets puros** — 2023 e 2024 underperformou BTC por ser defensivo demais.
4. **Weekly rebal perde intra-semana** — modelo só reage em sexta ou em emergency >8%. Move de 5-7% intraweek passa sem rebal.
5. **Retrain semi-annual pode ser lento** — se regime mudar em Fev, modelo espera até Jul pra adaptar.
6. **Universo único (BTC)** — sem diversificação crypto. ETH em features é correlação, não alocação.

### Riscos

| Risco | Probabilidade | Mitigação |
|---|---|---|
| Regime shift (novo tipo de bull/bear) | Média | Retrain semi-annual ajuda, mas não instantâneo |
| Feature data source cai (ex: bitcoin-data.com) | Baixa | Pipeline detecta NaN; median-fill histórico |
| BTC muda microstructure (ex: ETF flow domina) | Média | Monitorar 2026 acc — se cair <50%, reavaliar |
| K mal calibrado (vol muda) | Média | A/B H1 vs H2 a cada 6m no walk-forward |
| Overfit (32 features / 2500 samples) | Baixa | 10-seed validation estável; feature importance consistente |

---

## 8. Garantias de walk-forward no pipeline de produção

O `run_daily.py` **não tem look-ahead** por design:

1. **Retrain schedule fixo:** `RETRAIN_MONTHS=[1,7]` em `generate_signal.py:39` — só retrain em Jan e Jul.
2. **Training gap de 5 dias:** `train_end = n - 5` em `generate_signal.py:223` — previne leakage do target forward-looking.
3. **Prediction só no `X[-1:]`:** o modelo já treinado projeta pra frente.
4. **Entre retrains, cache é reutilizado** — nunca retreina no meio do período.

**Regras pra não furar walk-forward:**
- ✅ `python scripts/production/run_daily.py` (diário, walk-forward OK)
- ✅ `python scripts/production/run_daily.py --retrain` em 1-Jan ou 1-Jul (schedule) ou após mudar features (necessário)
- ❌ NUNCA use `--retrain` e depois backtest retrocessivo em datas anteriores com essa cache. Use `walkforward_backtest.py` pra isso.
- ✅ `python scripts/production/walkforward_backtest.py` sempre que precisar validar performance histórica (treina modelos walk-forward automaticamente).

---

## 9. Estrutura de arquivos (2026-04-20)

```
longbiased-beny/
├── scripts/production/              # CÓDIGO ATIVO
│   ├── run_daily.py                 # Entry: fetch → bootstrap → signal
│   ├── walkforward_backtest.py      # NEW: audit OOS com retrain semi-annual
│   ├── config.py                    # K_REGIME (H2), 32 features
│   ├── generate_signal.py           # V23 pipeline (semi-annual retrain)
│   ├── training.py                  # Helpers XGB
│   ├── fetch_raw_data.py            # 12+ fontes
│   ├── bootstrap_from_original.py   # Hybrid dataset
│   ├── build_features.py            # Feature engineering
│   ├── rebuild_signal_history.py    # Full backfill
│   ├── INSTRUCTIONS.md              # Manual operacional
│   └── data/
│       ├── raw_data.csv             # 22 cols brutas
│       ├── dataset_production.csv   # 32 features (2019-hoje)
│       ├── cached_models.pkl        # 80+80 XGB (trained 2026-04-19)
│       └── signal_history.csv       # Sinais (30 rows, com backfill 04-17)
│
├── docs/
│   ├── MODEL_AUDIT_2026-04-20.md    # ESTE DOC (audit final)
│   ├── STATUS_FINAL.md              # Status atual
│   ├── AI_ONBOARDING.md             # Entrada pra novo AI
│   ├── PIPELINE_V02-V38.md          # Histórico V02-V38
│   ├── SESSION_INDEX.md             # Navegação sessão V29-V39
│   └── ARTIGO_FINAL_V22.md          # Artigo acadêmico
│
├── outputs/results/
│   ├── walkforward_backtest.csv     # 30 rebals detalhados
│   └── v2X-v39_*.json               # Resultados experimentos
│
└── archive/
    ├── old_pipelines/               # V02-V22 (históricos)
    └── test_scripts/                # v20-v39 (experimentos)
```

---

## 10. Como um novo AI deve começar

1. **Ler este doc** (`MODEL_AUDIT_2026-04-20.md`) — panorama completo.
2. **Ler `STATUS_FINAL.md`** — estado atual + decisões abertas.
3. **Rodar `python scripts/production/run_daily.py`** — ver se pipeline funciona.
4. **Ler `AI_ONBOARDING.md`** — contexto histórico da evolução.
5. **Rodar `python scripts/production/walkforward_backtest.py --compare`** quando questionar performance — sempre use esse (não cache direto) pra evitar look-ahead.

### Perguntas típicas e onde achar resposta

| Pergunta | Arquivo |
|---|---|
| "Qual modelo ta em prod?" | Este doc seção 2 |
| "Por que H2 e não H1?" | Seção 6 + `project_h2_deployed.md` (memória) |
| "Retorno até hoje?" | Seção 4 + rodar `walkforward_backtest.py` |
| "Quais features funcionam?" | Seção 3 + `project_v36_winners.md` |
| "O que foi testado e falhou?" | `feedback_not_retest.md` (memória) + `PIPELINE_V32.md` |
| "Como debugar sinal quebrado?" | `AI_ONBOARDING.md` seção 11 |
| "Posso adicionar feature X?" | Checa `feedback_not_retest.md` primeiro — 36 técnicas já descartadas |

---

## 11. Retrain frequency experiment (sessão 2026-04-20/22)

**Motivação:** Validar empiricamente se semi-annual é realmente ótimo ou se outras cadências/triggers melhoram.

### Setup
- 6 estratégias × 5 seeds = 30 jobs
- 4 processos paralelos (4 threads XGB cada)
- Período 2022-01-01 → 2026-04-19 (4.3 anos, walk-forward)
- Custos 5bps por mudança de alloc (convention V22/V36)
- Sortino & Price 1994 formula

### Resultados

| # | Strategy | Retrains | Return (±σ) | Sortino (±σ) | DD |
|---|---|---|---|---|---|
| 🥇 | SEMI + DDT | 13 | +1028% (±17) | **6.23** (±0.02) | −9.1% |
| 🥈 | **SEMI (atual)** | 9 | **+1131%** (±9) | 5.91 (±0.04) | −9.1% |
| 🥉 | ACC (trigger only) | 16 | +896% (±72) | 5.05 (±0.20) | −7.4% |
| 4 | SEMI + ACC + DDT | 23 | +825% (±71) | 4.90 (±0.23) | −7.3% |
| 5 | SEMI + ACC | 18 | +842% (±77) | 4.73 (±0.23) | −7.3% |
| 6 | ANNUAL | 5 | +798% (±10) | 4.69 (±0.04) | −9.1% |

### Conclusões

1. **SEMI puro é ótimo** — bate com V36 (+1131% vs +1228%, 5.91 vs 5.72). Minha recomendação.
2. **SEMI + DDT marginal gain em Sortino (+0.32) custa 104pp de return** — descartado.
3. **Mais retrains ≠ melhor** — quarterly, monthly, combos com ACC todos piores que semi.
4. **Triggers com ACC** têm alta variance entre seeds (±77% return) — não confiáveis.

### DDT (Drawdown-Triggered retrain): DESCARTADO

Pesquisa em quant funds (abril 2026) confirmou que **DDT não é best practice**:
- Quant funds usam DD como **risk management** (kill switches, sizing), NÃO como trigger de retreino.
- Concept drift é detectado por **estatística de features** (PSI, KS test, ADWIN), não por P&L.
- DDT é reativo, path-dependent, e pode overfitear ao dip (retreinar no pior momento).

### Scripts criados
- `scripts/production/retrain_frequency_experiment.py` — periodic (annual/semi/quarterly/monthly)
- `scripts/production/retrain_conditional_experiment.py` — triggers (DDT/ACC/VRC/BMT)
- `scripts/production/retrain_experiment_parallel.py` — parallel multi-seed com 5bps cost (formula V22 correta)
- `scripts/production/align_cache_to_schedule.py` — alinhar cache ao schedule semi-annual

### Bugs encontrados (documentados pra não repetir)

1. **Timing alloc**: alloc nova de sexta aplicada ao retorno Thu→Fri (que já aconteceu). Fix: `applied_alloc = prev_alloc` antes de atualizar.
2. **Sqrt(252) vs sqrt(365)**: annualizei errado. Daily returns incluem weekends → sqrt(365) correto.
3. **Sortino formula V21 vs V22**: V21 usa `std(excess[excess<0])`, V22 (Sortino & Price 1994) usa `sqrt(mean(min(excess,0)²))`. V21 subestima Sortino em ~40%.

---

## 12. Próximas validações sugeridas

1. **Cost modeling** — adicionar 5-10bps por rebal pra ver impact real.
2. **A/B H1 vs H2 contínuo** — rodar `walkforward_backtest.py --compare` a cada fim de trimestre.
3. **Stress test** — backtest em janelas com vol anormal (Mar 2020 covid, Nov 2022 FTX) — o treino cobre?
4. **Feature drift monitoring** — se acc cair <50% por 8+ semanas consecutivas, reavaliar features.
5. **Dollar-weighted Sortino** — Sortino assume retornos log-normais; metrics alternativas em caudas pesadas.

---

**Audit encerrado 2026-04-20. Próximo rebal programado: Friday 2026-04-24.**
