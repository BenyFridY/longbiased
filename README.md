# Long-Biased BTC/CDI — Alocacao Dinamica via Machine Learning

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0_(pinned)-EB5424)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-paper_trading-blue)

Sistema de ML que decide **semanalmente** quanto alocar em Bitcoin vs renda fixa
(CDI), com mandato **long-biased no-short** e otimizado para **Sortino** (retorno
ajustado ao risco de queda). Dois ensembles XGBoost (160 regressores + 160
classificadores, 32 features), regime de mercado via SMA50/200, e alocacao entre
0% e 100% escalada pela confianca do classificador.

> **Aviso**: projeto de pesquisa quantitativa em fase de paper trading.
> Nada aqui e recomendacao de investimento. Numeros de backtest vem com as
> ressalvas de praxe — e a secao [Riscos conhecidos](#riscos-conhecidos-nao-eliminados)
> lista exatamente onde isso pode quebrar.

---

## Como funciona

```
Input:     32 features (tecnicas, macro, on-chain, cross-asset)
Model:     XGBoost duplo (160 regressores + 160 classificadores, bagging)
Target:    Retorno BTC 3 dias a frente (regressao) + direcao (classificacao)
Regime:    SMA50/SMA200 -> BULL (K=60) / MILD (K=30) / BEAR (K=15)
Sizing:    pred * K[regime] * sigmoid(|P_cls - 0.5| * 5)
Alocacao:  clip(sizing, 0%, 100%)        # no-short (long-biased mandate)
Rebalance: Sexta-feira + emergencia (|ret diario| > 8%, executada pos-close)
Retrain:   Semi-anual (jan/jul), expanding window + purge 5d
Risk:      Kill switch (DD <= -12%) ativo; PSI monitor informacional
```

A intuicao: a previsao de retorno dimensiona a posicao, o regime limita o
tamanho maximo da aposta (defensivo em BEAR), e a incerteza do classificador
corta a posicao pela metade quando o modelo esta "em cima do muro". Previsao
negativa = 100% CDI.

---

## Resultados — backtest 4.28 anos out-of-sample (BRL)

Validacao com 10 seeds independentes, walk-forward com purge (Lopez de Prado),
custos e CDI reais (BCB). Janela canonica 2022-01-07 → 2026-04-17:

| Estrategia (BRL) | CAGR | Sortino diario | Sharpe excesso | DD diario |
|---|---|---|---|---|
| 100% CDI | +12.8% | — | 0.00 | 0% |
| 30% BTC + 70% CDI | +15.6% | 1.03 | — | -20.2% |
| 100% BTC HODL | +12.1% | 0.50 | — | -66.5% |
| **Modelo (160 bags)** | **+50.5% ± 0.4%** | **3.84 ± 0.05** | **2.35 ± 0.01** | **-5.34% ± 0.30%** |

Por ano (media 10 seeds): 2022 **+26.9%** · 2023 **+51.7%** · 2024 **+93.8%** ·
2025 **+37.7%** · 2026 ate abril **+12.5%** — todos positivos, incluindo o bear
market de 2022.

**Expectativa honesta para live** (apos deflation por multiple testing, 38
trials): CAGR 20-35%, Sortino 1.5-2.5, DD diario -15% a -25%. Mesmo no piso,
bate 30% BTC estatico em CAGR, Sortino e DD. Edge validado: **0/100 shuffles
das previsoes bateram o baseline** (p < 0.01).

---

## 📊 Dashboard

Dashboard local (Streamlit, tema dark) para acompanhar o sinal e — principalmente —
**entender o porque de cada alocacao**:

```bash
streamlit run scripts/production/dashboard.py
```

Abre em `http://localhost:8501`, somente leitura dos CSVs (nao roda o modelo).
Quatro abas:

- **📊 Visao geral** — sinal atual, alocacao x preco x regime (com emergencias
  marcadas), desempenho acumulado vs **CDI** (benchmark justo) e BTC, drawdown, e
  metricas de factsheet em base diaria reconstruida: CAGR, % do CDI, vol, Sharpe,
  Sortino, Calmar, capturas de alta/baixa, consistencia mensal vs CDI.
- **🧭 Por que da semana** — para cada rebalance: decomposicao da formula em
  cartoes (regime, previsao, confianca), o **principal freio** da posicao,
  cenarios contrafactuais ("e se fosse BULL?", "e sem o corte de confianca?"),
  semanas vizinhas, o resultado realizado da janela e o retrato das features na
  data (percentis historicos).
- **📅 Visao mensal** — heatmap de retornos mensais (hover compara com CDI e BTC)
  e explicacao gerada por regra de por que cada mes ficou pouco/muito alocado.
- **📄 Dados & ajuda** — tabela completa dos sinais e a explicacao do calculo.

Filtro de periodo global (presets + intervalo personalizado). Paleta validada
para daltonismo, dark/light.

---

## Quickstart

```bash
git clone https://github.com/BenyFridY/longbiased
cd longbiased
pip install -r requirements.txt

# chaves de API (a unica obrigatoria e a FRED, gratuita)
cp .env.example .env       # e preencha FRED_API_KEY

# pipeline completo: busca dados, calcula features, gera o sinal
python scripts/production/run_daily.py

# dashboard
streamlit run scripts/production/dashboard.py
```

O repositorio ja inclui `signal_history.csv` e `dataset_production.csv`, entao o
**dashboard funciona imediatamente**, sem rodar o pipeline. Versoes de pandas/
numpy/scikit-learn/XGBoost sao **pinadas** — treino de XGBoost nao e
deterministico entre versoes/CPUs (numeros validados em XGBoost 3.2.0).

### Fontes de dados (12+)

| Fonte | Dados |
|-------|-------|
| Binance Spot / Futures / Funding | OHLCV BTC, basis, funding rate |
| yfinance | ETH, ouro, cobre |
| FRED | M2, Fed balance sheet |
| BGeometrics / bitcoin-data.com | NUPL, Reserve Risk, Puell Multiple |
| DefiLlama | Stablecoin supply |
| CoinMetrics / Blockchain.com | Hash rate, miners revenue |
| BCB | CDI real |

---

## Rigor de validacao

O projeto trata overfitting como o risco #1. Auditoria com 7 testes:

| Teste | Resultado |
|-------|-----------|
| Sensibilidade de K | H2 (100/50/20) vence backtest, mas H1 e mais robusto |
| Frozen train (sem retrain) | H1 +38-44% melhor que H2 — por isso H1 em producao |
| Feature ablation | Macro carrega o edge; trio on-chain e marginal (+0.30) |
| Shuffled predictions | 0/100 shuffles batem baseline (p < 0.01) — sinal real |
| Risk controls | Kill switch nunca disparou em 4 anos de backtest |
| Sigmoid scale sweep | Plano de 10 a 1000 — parametro nao foi overfit |
| Composicao | Sem a magnitude ML, Sortino cai de ~5.6 para 2.19 |

Mais ~1500 configuracoes testadas e descartadas ao longo de 23 versoes —
incluindo HMM, GARCH, LSTM, Kelly, stop-loss, Fear & Greed, Google Trends e
9 features extras da Messari (todas pioraram). Historico completo em
[`docs/`](./docs/) e [`docs/archive/pipelines/`](./docs/archive/pipelines/).

---

## Riscos conhecidos (nao eliminados)

- **Regime shift drastico** — retrain semi-anual pode ser lento; kill switch cobre parcialmente
- **Multiple-testing** — o Sortino real deve ser ~2x menor que o observado em backtest
- **Concentracao** — top 10 semanas = 48% do retorno; perder 2-3 = metade do alpha
- **Features on-chain marginais** — podem nao se sustentar em live
- **Custos** — acima de 25 bps por rebalance, perde ~4pp de CAGR

---

## Estrutura do projeto

```
longbiased/
+-- scripts/production/          ** PIPELINE ATIVO **
|   +-- run_daily.py             Entry point unico (dados -> features -> sinal)
|   +-- generate_signal.py       Sinal com risk controls
|   +-- dashboard.py             Dashboard Streamlit (dark, benchmark CDI)
|   +-- config.py                K=60/30/15, no-short, 32 features
|   +-- fetch_raw_data.py        12+ fontes de dados
|   +-- build_features.py        Engenharia das 32 features
|   +-- risk_management.py       Kill switch + PSI monitor
|   +-- walkforward_backtest.py  Validador walk-forward OOS
|   +-- data/                    signal_history.csv, dataset_production.csv
+-- src/features/                Feature engineering reusavel (macro, regime)
+-- tests/                       132 testes (inclui property-test de look-ahead)
+-- docs/                        Spec canonica, auditorias, manuais
+-- .streamlit/config.toml       Tema dark do dashboard
```

Documentos-chave: [`docs/MODEL_FINAL.md`](./docs/MODEL_FINAL.md) (spec canonica),
[`docs/OVERFIT_TESTS_2026-04-22.md`](./docs/OVERFIT_TESTS_2026-04-22.md)
(auditoria), [`docs/RACIONAL_DECISOES.md`](./docs/RACIONAL_DECISOES.md) (por que
cada escolha), [`scripts/production/INSTRUCTIONS.md`](./scripts/production/INSTRUCTIONS.md)
(operacao).

---

*Projeto de pesquisa desenvolvido por Beny Frid. Codigo e resultados publicados
para fins educacionais — nao constitui recomendacao de investimento.*
