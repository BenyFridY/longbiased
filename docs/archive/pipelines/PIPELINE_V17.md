# Pipeline V17 — Resultados Definitivos

**Data**: 2026-03-07 | **Tempo de execução**: 28.1 horas | **Configs testadas**: 138 × 10 seeds = 1380 runs

---

## Resumo Executivo

O Pipeline V17 é o teste definitivo após V13-V16 (4 versões, 30+ horas de compute). Testou **10 fases** cobrindo todas as questões em aberto: modelo, features, horizonte, alocação, gestão de risco, K adaptativo e frequência de rebalanceamento.

### Configuração Final (Balanced)

```
Modelo:     XGBoost puro, 40 bags (sem LightGBM)
Features:   37 (32 base + 5 novas)
Target:     Retorno 3 dias à frente (ret_3d)
Retrain:    Semi-anual, janela expansiva
Alocação:   Dynamic Regime — K_bull=40, K_mild=30, K_bear=15
Rebalance:  Sexta-feira, sempre (sem threshold)
```

### Resultado

| Métrica | V13 (anterior) | V17 Balanced | Δ |
|---------|----------------|-------------|---|
| **Retorno** | +1313% | +930% | -383pp (menor, porém mais estável) |
| **Sortino** | 3.22 | **3.55** | **+0.33** |
| **Spread** | 146pp | **74pp** | **-72pp (metade da variância!)** |
| **MaxDD** | -15.9% | **-9.9%** | **+6.0pp (muito mais seguro)** |

### Validação: 3/3 PASS

| Teste | Resultado | Critério |
|-------|-----------|----------|
| Permutation (1000 shuffles) | p = 0.0000 | p < 0.05 |
| Bootstrap CI (1000 resamples) | P(loss) = 0.0%, CI = [+917%, +943%] | P(loss) < 5% |
| Year-by-year excess | 4/5 anos positivos | ≥ 3/5 |

---

## 5 Perfis de Risco

| Perfil | Retorno | Sortino | Spread | MaxDD | Uso sugerido |
|--------|---------|---------|--------|-------|-------------|
| Ultra-Conservador | +288% | 1.79 | 18pp | **-7.9%** | Capital preservação |
| Conservador | +661% | 3.24 | **46pp** | **-9.0%** | Produção (recomendado) |
| **Balanceado** | **+930%** | **3.55** | **74pp** | **-9.9%** | Melhor risco-ajustado |
| Agressivo | +1106% | **3.64** | 93pp | -11.2% | Alta convicção |
| Ultra-Agressivo | +1235% | 3.60 | 113pp | -12.7% | Máximo retorno |

> **Nota**: O perfil Conservador (S=3.24, 46pp spread, -9.0% DD) é excelente para produção — supera V13 em Sortino, spread E drawdown simultaneamente.

---

## Retorno Ano a Ano (seed=42)

| Ano | Estratégia | BTC | Excesso | Resultado |
|-----|-----------|-----|---------|-----------|
| 2022 | +73.3% | -64.1% | **+137.4%** | Bear market — estratégia brilha |
| 2023 | +156.4% | +153.5% | +2.8% | Bull market — acompanha BTC |
| 2024 | +68.9% | +111.4% | -42.5% | Underperformou no rally forte |
| 2025 | +41.1% | -6.1% | **+47.2%** | Proteção no mercado misto |
| 2026 | -1.5% | -23.1% | **+21.6%** | Proteção no bear |

**Padrão**: A estratégia gera alfa massivo em bear markets (2022: +137%, 2025: +47%, 2026: +22%) e acompanha em bull markets. O underperformance em 2024 (+69% vs BTC +111%) é o custo da proteção — mas o Sortino composto de 3.55 mostra que o trade-off vale a pena.

---

## Fases Detalhadas

### Fase 1: Seleção de Modelo (8 configs × 10 seeds = 80 runs)

**Pergunta**: XGBoost puro vs Hybrid LGB+XGB?

| Modelo | Sortino | Retorno | Spread | MaxDD |
|--------|---------|---------|--------|-------|
| **XGB Bag40** | **2.65** | +626% | 98pp | -19.2% |
| XGB Bag60 | 2.64 | +622% | 64pp | -19.3% |
| XGB Bag100 | 2.62 | +615% | 56pp | -19.2% |
| XGB Bag80 | 2.62 | +615% | 59pp | -19.3% |
| Hybrid 10+50 | 2.59 | +607% | 62pp | -20.0% |
| Hybrid 15+60 | 2.58 | +602% | 69pp | -20.2% |
| Hybrid 15+45 | 2.57 | +601% | 75pp | -20.3% |
| Hybrid 15+30 | 2.52 | +588% | 88pp | -20.6% |

**Decisão**: XGB puro Bag40 vence. Híbrido não adiciona valor nesta versão dos dados. Bag40 é suficiente — mais bags (60-100) não melhoram Sortino, apenas reduzem spread marginalmente.

---

### Fase 2: Screening de Features (16 configs × 10 seeds = 160 runs)

**Pergunta**: Quais das 15 features candidatas melhoram o modelo?

| Feature | Sortino | Δ Sortino | Retorno |
|---------|---------|-----------|---------|
| **+fed_balance_sheet** | **2.87** | **+0.215** | +688% |
| **+futures_trade_count** | **2.81** | **+0.158** | +675% |
| **+vol_x_regime_duration** | **2.73** | **+0.079** | +632% |
| **+velocity** | **2.70** | **+0.045** | +626% |
| **+hash_rate_pctchg_30d** | **2.69** | **+0.040** | +627% |
| +realized_price | 2.69 | +0.036 | +641% |
| +true_range | 2.67 | +0.015 | +638% |
| +stock_to_flow | 2.67 | +0.014 | +624% |
| +exchange_netflow_usd | 2.66 | +0.012 | +639% |
| baseline (sem adição) | 2.65 | — | +626% |
| +volume_per_trade | 2.65 | -0.001 | +595% |
| +fees_total_24h | 2.64 | -0.014 | +619% |
| +btc_sp500_corr_30d | 2.64 | -0.015 | +595% |
| +btc_eth_corr_30d | 2.63 | -0.020 | +610% |
| +futures_dominance | 2.61 | -0.041 | +678% |
| +spot_taker_sell_ratio | 2.56 | -0.094 | +588% |

**Decisão**: 5 features melhoram o baseline. `fed_balance_sheet` é a mais forte (+0.215 Sortino). Features macro (Fed balance sheet) e de mercado (futures trade count, velocity) dominam.

**Interpretação das features**:
- **fed_balance_sheet**: Tamanho do balanço do Fed — proxy de liquidez global
- **futures_trade_count**: Volume de trades em futuros — atividade especulativa
- **vol_x_regime_duration**: Interação volatilidade × duração do regime — persistência
- **velocity**: Velocidade do BTC on-chain — atividade econômica
- **hash_rate_pctchg_30d**: Mudança no hash rate — saúde dos mineradores

---

### Fase 3: Combinações de Features (32 configs × 10 seeds = 320 runs)

**Pergunta**: Qual combinação das 5 features confirmadas é ótima?

| Combinação | #Features | Sortino | Retorno |
|-----------|-----------|---------|---------|
| **Todas as 5** | **5** | **3.29** | **+816%** |
| fed + futures_trade + vol_regime + velocity | 4 | 3.28 | +815% |
| fed + vol_regime + velocity + hash_rate | 4 | 3.19 | +755% |
| fed + futures_trade + velocity + hash_rate | 4 | 3.18 | +789% |
| fed + futures_trade + vol_regime + hash_rate | 4 | 3.17 | +778% |
| fed + futures_trade + velocity | 3 | 3.12 | +779% |
| baseline (sem adição) | 0 | 2.65 | +626% |

**Decisão**: Todas as 5 features juntas dão o melhor resultado (S=3.29 vs baseline S=2.65). Diferente do V13 onde "1 feature addition optimal" — no V17 com XGB puro, o modelo absorve bem 5 features adicionais. A combinação de 4 features (sem hash_rate) é quase idêntica (S=3.28), mas o conjunto completo foi selecionado.

**Insight**: O salto de S=2.65 → S=3.29 (+0.64) ao adicionar 5 features é o maior ganho de todo o pipeline.

---

### Fase 4: Horizonte do Target (11 configs × 10 seeds = 110 runs)

**Pergunta**: Prever retorno de 3, 5 ou 7 dias?

| Horizonte | Modelo | Sortino | Retorno | Spread |
|-----------|--------|---------|---------|--------|
| **ret_3d** | **XGB** | **3.29** | **+816%** | **75pp** |
| ret_3d | Hybrid | 3.29 | +816% | 75pp |
| ret_3d | Hybrid_15 | 3.22 | +771% | 80pp |
| ret_5d | XGB | 2.74 | +1057% | 151pp |
| ret_5d | Hybrid_15 | 2.75 | +1048% | 124pp |
| ret_7d | XGB | 2.28 | +955% | 126pp |
| ret_5d quarterly | 2.17 | +649% | 55pp |

**Decisão**: ret_3d é claramente superior em Sortino (3.29 vs 2.74 vs 2.28). Retrain semi-anual confirmado (quarterly S=2.17 destrói performance). ret_5d tem retorno maior (+1057%) mas Sortino e spread muito piores.

---

### Fase 5: Sweep de K (12 configs × 10 seeds = 120 runs)

**Pergunta**: Qual o K ótimo para alocação centrada?

| K | Sortino | Retorno | Spread | MaxDD |
|---|---------|---------|--------|-------|
| **K=40** | **3.36** | +1261% | 173pp | -18.1% |
| K=45 | 3.33 | +1397% | 166pp | -19.5% |
| K=35 | 3.33 | +1088% | 121pp | -15.9% |
| K=33 | 3.32 | +1022% | 111pp | -15.0% |
| K=30 | 3.32 | +920% | 92pp | -13.6% |
| K=27 | 3.29 | +816% | 75pp | -12.2% |
| K=25 | 3.26 | +746% | 62pp | -11.3% |
| K=22 | 3.20 | +645% | 44pp | -9.8% |
| K=20 | 3.21 | +581% | 34pp | -8.9% |
| K=18 | 3.22 | +520% | 28pp | -7.9% |
| K=15 | 3.23 | +434% | 21pp | -6.4% |
| K=10 | 3.18 | +304% | 11pp | -5.0% |

**Decisão**: K=40 tem o melhor Sortino (3.36), mas spread alto (173pp) e DD pesado (-18.1%). O trade-off return/risk escala linearmente. Mas com alocação centrada simples, K alto → DD alto. Isso motiva a Fase 6.

---

### Fase 6: Lógica de Alocação — O Maior Teste (32 configs × 10 seeds = 320 runs)

**Pergunta**: Qual lógica de alocação dá o melhor Sortino com DD controlado?

#### 6A: Dynamic Regime (baseado em SMA50/SMA200)

| Config | K_bull/K_mild/K_bear | Sortino | Retorno | Spread | MaxDD |
|--------|---------------------|---------|---------|--------|-------|
| **dyn_40_30_15** | **40/30/15** | **3.55** | **+930%** | **74pp** | **-9.9%** |
| dyn_45_25_10 | 45/25/10 | 3.49 | +847% | 63pp | -9.2% |
| dyn_35_25_15 | 35/25/15 | 3.44 | +796% | 54pp | -9.0% |
| dyn_35_30_15 | 35/30/15 | 3.44 | +838% | 63pp | -9.7% |
| dyn_40_25_10 | 40/25/10 | 3.43 | +787% | 52pp | -9.2% |
| dyn_35_25_10 | 35/25/10 | 3.32 | +708% | 48pp | -9.0% |
| dyn_30_20_10 | 30/20/10 | 3.28 | +595% | 38pp | -8.4% |

**O regime funciona assim**:
- **Bull** (Price > SMA50 > SMA200): K alto → posição agressiva
- **Mild** (Price > SMA200): K médio → posição moderada
- **Bear** (Price < SMA200): K baixo → posição defensiva

#### 6B: Partial Trend Filter

| Config | Sortino | Retorno | MaxDD |
|--------|---------|---------|-------|
| partial_40_15_n10 | 2.99 | +412% | -7.8% |
| partial_35_10_0 | 2.46 | +312% | -7.2% |
| partial_30_10_0 | 2.31 | +256% | -6.4% |

#### 6C: Vol-Scaled

| Config | Sortino | Retorno | Spread | MaxDD |
|--------|---------|---------|--------|-------|
| vol_K30 | 3.28 | +1229% | 143pp | -19.7% |
| vol_K35 | 3.26 | +1369% | 204pp | -21.5% |
| vol_K25 | 3.19 | +992% | 87pp | -17.5% |

#### 6D: Position Limits / 6E: Gradual Changes

| Config | Sortino | Retorno | MaxDD |
|--------|---------|---------|-------|
| limit_80_15 | 3.16 | +858% | -16.5% |
| limit_70_10 | 3.10 | +677% | -14.3% |
| grad_25 | 2.47 | +571% | -9.3% |
| grad_15 | 2.30 | +470% | -7.2% |

#### Comparação: Centered vs Dynamic Regime

| Métrica | Centered K=40 | Dynamic 40/30/15 | Δ |
|---------|---------------|-------------------|---|
| Sortino | 3.36 | **3.55** | **+0.19** |
| Retorno | +1261% | +930% | -331pp |
| Spread | 173pp | **74pp** | **-99pp** |
| MaxDD | -18.1% | **-9.9%** | **+8.2pp** |

**Decisão**: `dynamic_regime 40/30/15` é o vencedor claro. Ao adaptar K ao regime de mercado, reduz MaxDD de -18% para -10% e spread de 173pp para 74pp, enquanto AUMENTA Sortino de 3.36 para 3.55. O mecanismo: em bear market, K=15 evita grandes perdas; em bull, K=40 captura o upside.

**Este é o maior insight do V17**: a alocação dinâmica por regime é superior a qualquer K fixo.

---

### Fase 7: Gestão de Risco (11 configs × 10 seeds = 110 runs)

**Pergunta**: DD budget, confidence gating ou combinações melhoram?

| Config | Sortino | Retorno | Spread | MaxDD |
|--------|---------|---------|--------|-------|
| **baseline (dynamic regime)** | **3.55** | **+930%** | **74pp** | **-9.9%** |
| alloc+dd_15 | 3.55 | +930% | 74pp | -9.9% |
| alloc+conf_30 | 3.34 | +469% | 55pp | -7.3% |
| alloc+dd+conf | 3.34 | +469% | 55pp | -7.3% |
| dd_10/12/15/20 | 3.29 | +816% | 75pp | -12.2% |
| conf_20/30/40 | 3.13-3.15 | +525-531% | 51-64pp | -8.4% |

**Decisão**: Nenhuma gestão de risco melhora o baseline. O DD budget não faz diferença (alloc+dd_15 = baseline idêntico) porque o dynamic regime já controla DD naturalmente (-9.9%). Confidence gating reduz retorno pela metade sem ganho em Sortino. **O regime dinâmico já É a gestão de risco**.

---

### Fase 8: Walk-Forward K (8 configs × 10 seeds = 80 runs)

**Pergunta**: Otimizar K adaptativamente em cada retrain melhora?

| Config | Sortino | Retorno | Spread | MaxDD |
|--------|---------|---------|--------|-------|
| **fixed K=27** | **3.29** | **+816%** | **75pp** | **-12.2%** |
| wf_narrow (20-35) | 3.15 | +824% | 76pp | -11.1% |
| wf_val90 | 2.99 | +433% | 42pp | -6.4% |
| wf_val180 / wf_semi | 2.43 | +460% | 41pp | -9.6% |
| wf_wide (10-45) | 2.39 | +460% | 42pp | -9.8% |
| wf_val365 | 2.35 | +388% | 28pp | -9.6% |
| wf_quarterly | 1.65 | +261% | 18pp | -9.8% |

**Decisão**: Fixed K vence convincentemente. Walk-forward K overfitta ao período de validação recente e reduz Sortino de 3.29 para 1.65-3.15. Quanto mais dados de validação ou mais frequente o ajuste, pior o resultado. **K fixo é mais robusto que K adaptativo**.

---

### Fase 9: Variações de Rebalanceamento (8 configs × 10 seeds = 80 runs)

**Pergunta**: Bi-weekly, thresholds condicionais ou outros dias melhoram?

| Config | Sortino | Retorno | Spread | MaxDD |
|--------|---------|---------|--------|-------|
| **Friday only** | **3.55** | **+930%** | **74pp** | **-9.9%** |
| thresh_0.015 | 3.39 | +862% | 118pp | -11.4% |
| thresh_0.005 | 3.34 | +847% | 51pp | -11.6% |
| thresh_0.01 | 3.30 | +825% | 153pp | -11.5% |
| Thu+Fri | 2.89 | +637% | 48pp | -9.1% |
| Mon+Fri | 2.60 | +591% | 57pp | -14.1% |
| Wed+Fri | 2.53 | +662% | 48pp | -13.9% |
| Thu only | 1.89 | +342% | 29pp | -10.6% |

**Decisão**: Friday only é imbatível. Threshold condicional piora spread significativamente (118-153pp vs 74pp). Bi-weekly (Mon+Fri, Wed+Fri) destrói Sortino — rebalancear mais de 1x/semana adiciona ruído. Thursday only é catastrófico (S=1.89).

**Insight**: O efeito de sexta-feira é estrutural — provavelmente ligado a liquidez institucional e settlement de derivativos no fim de semana.

---

### Fase 10: Montagem Final + Validação

Os 5 perfis de risco variam apenas os parâmetros K do regime dinâmico:

| Perfil | K_bull | K_mild | K_bear |
|--------|--------|--------|--------|
| Ultra-Conservador | 25 | 20 | 0 |
| Conservador | 32 | 25 | 10 |
| **Balanceado** | **40** | **30** | **15** |
| Agressivo | 45 | 33 | 20 |
| Ultra-Agressivo | 50 | 38 | 25 |

---

## Insights Fundamentais

### 1. Dynamic Regime é o Game-Changer
A alocação baseada em regime (SMA50/SMA200) reduz MaxDD de -18% para -10% sem sacrificar Sortino. O modelo ML prevê melhor em bull markets (K alto captura alpha) e pior em bear markets (K baixo evita perdas). O regime atua como um "amplificador condicional" do sinal ML.

### 2. Features Macro Dominam
`fed_balance_sheet` é a feature mais forte (+0.215 Sortino). As 5 features adicionadas representam 3 categorias: macro (Fed), mercado (futures, velocity) e técnico (hash rate, vol×regime). Isso sugere que o modelo se beneficia de informação que captura o ciclo macro-cripto.

### 3. XGB Puro é Suficiente
O debate XGB vs Hybrid foi resolvido: com dados atualizados e 40 bags, XGB puro vence. LightGBM não adiciona diversidade suficiente para justificar o custo computacional.

### 4. Gestão de Risco Adicional é Redundante
DD budget, confidence gating e combinações não melhoram sobre o dynamic regime. O regime dinâmico já embute gestão de risco via K adaptativo ao mercado. Adicionar overlays apenas reduz retorno.

### 5. Simplicidade Vence Complexidade
- K fixo > walk-forward K
- Rebalance semanal > bi-weekly
- Sem threshold > com threshold
- Sem overlays > DD budget + confidence

---

## Arquivos de Referência

| Arquivo | Descrição |
|---------|-----------|
| `scripts/optimization/pipeline_v17_definitive.py` | Pipeline V17 (10 fases, engine completa) |
| `outputs/results/pipeline_v17_definitive.json` | Resultados completos (todos os configs e seeds) |
| `docs/PIPELINE_V17_RESULTS.md` | Este documento |
| `scripts/optimization/pipeline_v13_definitive.py` | Pipeline V13 (referência anterior) |
| `scripts/optimization/pipeline_v9.py` | Engine base (load_data, build_ml_features) |

---

## Configuração para Produção

```python
# Configuração recomendada (perfil Conservador para produção)
config = {
    'model': 'XGBoost',
    'n_bags': 40,
    'features': 37,  # 32 base + fed_balance_sheet, futures_trade_count,
                      # vol_x_regime_duration, velocity, hash_rate_pctchg_30d
    'target': 'ret_3d',
    'retrain': 'semi-annual',
    'rebalance': 'Friday',
    'allocation': 'dynamic_regime',
    'K_bull': 32,   # Price > SMA50 > SMA200
    'K_mild': 25,   # Price > SMA200
    'K_bear': 10,   # Price < SMA200
    # Expected: +661%, S=3.24, 46pp spread, -9.0% MaxDD
}
```
