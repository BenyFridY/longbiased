# Arquitetura, UX e Criterios de Aceite

**Parte do entregavel avaliado #1 (Artefatos de Desenvolvimento do Produto, 35%).**
Cobre os componentes **arquitetura**, **UX** e **criterios de aceite** exigidos pela
rubrica, e **documenta** (sem construir) a abordagem de **implantacao (#6)** e
**integracao (#7)** — decisao do projeto: nesta fase de paper trade, esses dois
entregaveis sao **documentados, nao implementados**.

> Spec do modelo:
> [`MODEL_FINAL.md`](./MODEL_FINAL.md). Operacao: [`MANUAL_IMPLANTACAO.md`](./MANUAL_IMPLANTACAO.md).

---

## 1. Arquitetura

### 1.1 Visao geral

O produto e um **pipeline batch** que transforma dados de mercado em um sinal de
alocacao. Componentes (todos em `scripts/production/`, orquestrados por
`run_daily.py`):

```
  Fontes externas (12+)                  ┌─────────────────────────────┐
  Binance, BQ/Messari, FRED,             │  run_daily.py (orquestrador) │
  yfinance, on-chain, DefiLlama   ──────>│  3 etapas, abort+timeout 600s│
                                          └──────────────┬──────────────┘
                                                         v
  (1) fetch_raw_data.py  ──>  raw_data.csv
                                                         v
  (2) bootstrap_from_original.py  ──>  dataset_production.csv
        base historica enhanced (congelada) + build_features (dias novos)
        + backfill das 3 features V36
                                                         v
  (3) generate_signal.py
        ├─ training.py        : ensemble XGBoost (160 reg + 160 cls)
        ├─ sizing             : clip(pred x K[regime] x sigmoid(conf), 0, 1)
        ├─ risk_management.py : kill switch + acc de-risk + PSI
        └─ saida              : sinal formatado (CLI) + signal_history.csv
                                                         v
                                 operador executa o rebalance (manual)
```

### 1.2 Camadas e tecnologias

| Camada | Responsabilidade | Implementacao |
|---|---|---|
| Ingestao | Buscar dados de 12+ fontes (incremental) | `fetch_raw_data.py` (requests, BigQuery, yfinance, fredapi) |
| Dataset | Montar base hibrida (2 camadas) + backfill V36 | `bootstrap_from_original.py` |
| Features | Calcular as 32 features (macro/on-chain/regime) | `build_features.py`, `src/features/` |
| Modelo | Treinar/inferir ensemble | `training.py`, `generate_signal.py` (XGBoost 3.2.0) |
| Sizing | Converter previsao em alocacao 0-100% | `generate_signal.py` (regime SMA50/200 + sigmoid) |
| Risk controls | Proteger downside e detectar drift | `risk_management.py` (kill switch, acc de-risk, PSI) |
| Saida/registro | Exibir sinal + registrar historico | CLI + `data/signal_history.csv` |
| Qualidade | Garantir integridade | `tests/` (132 testes), deps pinadas |

### 1.3 Decisoes arquiteturais relevantes

- **Dataset hibrido em 2 camadas:** base historica validada **congelada** (ground
  truth) + dias novos recalculados — garante estabilidade e comparabilidade.
- **Cache fingerprint (sha1)** sobre features/bags/horizonte/params: detecta modelo
  stale e forca retrain automatico, evitando servir um modelo incompativel.
- **Paralelismo** (`ThreadPoolExecutor`, 16 workers) para treinar os 320 modelos.
- **Backward-looking garantido** por property-test (`tests/test_lookahead.py`).
- **Reprodutibilidade** por deps pinadas + retrain na maquina alvo (XGBoost nao e
  deterministico entre CPUs).

---

## 2. Implantacao (#6) e Integracao (#7) — documentadas

> **Decisao de projeto:** na fase de paper trade, deploy e integracao sao
> **documentados, nao construidos**. Nao ha containerizacao, CI/CD nem API — por
> escolha consciente (custo de infra so se justifica apos o gate de capital).

### 2.1 Implantacao (#6)

Pipeline batch agendado (cron / Task Scheduler) rodando `run_daily.py` diariamente
apos 00:00 UTC. Passo-a-passo completo, monitoramento e checklist em
[`MANUAL_IMPLANTACAO.md`](./MANUAL_IMPLANTACAO.md). Hardening (Docker, CI/CD, alta
disponibilidade) listado como **roadmap condicional** (sec 9 do manual).

### 2.2 Integracao (#7) — contrato de dados

A integracao com sistemas downstream se da pelo **contrato de dados
`signal_history.csv`** (e nao por API REST, por decisao de fase). Schema atual:

| Coluna | Tipo | Significado |
|---|---|---|
| `date` | data | Data do sinal |
| `day` | texto | Dia da semana (ex.: Fri) |
| `price_usd` | float | Preco do BTC (USD) |
| `regime` | texto | BULL / MILD / BEAR |
| `previsao` | float | Retorno previsto do BTC em 3 dias |
| `p_up` | float | Probabilidade de alta (classificador) |
| `confidence_factor` | float | Fator de confianca (sigmoid) |
| `allocation` | float | **Alocacao alvo em BTC (0-1)** |
| `K_base` | int | K do regime |
| `K_effective` | float | K efetivo (base x confianca) |
| `is_emergency` | bool | Rebalance de emergencia? |
| `retorno_btc` | float | Retorno realizado do BTC |
| `retorno_strat` | float | Retorno realizado da estrategia |
| `action` | texto | REBALANCE / HOLD + tag do modelo |

Consumo programatico: ler o CSV (ex.: `pandas.read_csv`) e usar a coluna
`allocation` da ultima linha. A skill **`sinal-semanal`** automatiza a geracao e o
registro semanal. **Roadmap (#7), nao implementado:** endpoint REST com OpenAPI
para servir o sinal a sistemas internos.

---

## 3. UX — experiencia do usuario

### 3.1 Natureza da interface (sem GUI)

O produto **nao tem interface grafica** — e uma **ferramenta operacional** para a
mesa quant. A "UX" e a experiencia de **linha de comando + saida textual
estruturada + arquivo CSV**. Isso e adequado ao publico (operador tecnico) e ao
estagio (paper trade interno); um dashboard e item de roadmap, nao requisito.

### 3.2 Persona e jornada

- **Persona:** operador/gestor da mesa quant (tecnico, le terminal e CSV).
- **Jornada semanal:**
  1. Sexta (ou emergencia): roda `run_daily.py` ou aciona a skill `sinal-semanal`.
  2. Le o sinal formatado — foca na linha `>> Allocation`.
  3. Executa o rebalance (compra/venda) **manualmente**.
  4. O sinal e registrado automaticamente em `signal_history.csv`.

### 3.3 Principios de UX aplicados na saida

- **Escaneabilidade:** blocos com cabecalho, campos rotulados, separadores.
- **Acao explicita:** `>> Action` (REBALANCE/HOLD) e `>> Allocation` em destaque.
- **Alertas de risco** em bloco proprio (`!!! RISK MANAGEMENT CONTROLS ACTIVATED`)
  quando um controle dispara, com valor original vs ajustado.
- **Contexto:** ultimo rebalance, proximo retrain, DD atual, acuracia 12w.
- **Reducao de friccao:** a skill faz catch-up de sextas perdidas se os dados
  estavam atrasados.
- **Tratamento de erro:** mensagens claras de falha/timeout e tabela de
  troubleshooting ([`MANUAL_USO.md`](./MANUAL_USO.md) sec 7).

Detalhe da interpretacao de cada campo: [`MANUAL_USO.md`](./MANUAL_USO.md) sec 2.2.

---

## 4. Criterios de aceite

### 4.1 Funcionais

| ID | Criterio | Verificacao |
|---|---|---|
| AC-1 | `run_daily.py` roda fim-a-fim sem erro e gera o sinal do dia | Execucao manual / log |
| AC-2 | Rebalance sinalizado nas sextas e em emergencia (\|ret\| > 8%); HOLD nos demais dias | Inspecao do output / `signal_history.csv` |
| AC-3 | Risk controls aplicados automaticamente (kill switch, acc de-risk, PSI) | `tests/test_risk_management.py` + output |
| AC-4 | Sinal de rebalance registrado em `signal_history.csv` com o schema da sec 2.2 | Inspecao do CSV |

### 4.2 Qualidade / engenharia

| ID | Criterio | Verificacao |
|---|---|---|
| AC-5 | Suite de testes verde (132 testes) | `pytest -q` |
| AC-6 | Zero look-ahead/leakage | `tests/test_lookahead.py` (property-test) |
| AC-7 | Reprodutibilidade: retrain na maquina alvo reproduz metricas dentro do ruido de seed (Sortino std ~0.06) | `walkforward_backtest.py --compare` |

### 4.3 Performance (aceite do modelo)

| ID | Criterio | Verificacao |
|---|---|---|
| AC-8 | No backtest/paper-trade, supera os baselines: Sortino > 30/70 estatico e DD << 100% BTC | `MODEL_FINAL.md` sec 2 |
| AC-9 | Metas SMART (K1-K10) avaliadas no gate Q3/2026; piso conservador para go/no-go de capital | plano de metas do projeto |

---

*Arquitetura detalhada do dataset: [`scripts/production/INSTRUCTIONS.md`](../scripts/production/INSTRUCTIONS.md).
Implantacao: [`MANUAL_IMPLANTACAO.md`](./MANUAL_IMPLANTACAO.md). Uso: [`MANUAL_USO.md`](./MANUAL_USO.md).*
