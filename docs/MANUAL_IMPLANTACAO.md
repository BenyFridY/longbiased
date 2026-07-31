# Manual de Implantacao — Pipeline de Sinal BTC/CDI

Projeto `longbiased-beny`
**Publico:** quem vai colocar o pipeline em operacao (producao / paper trade).
**Objetivo:** permitir que um terceiro implante e opere o pipeline **do zero, sem
suporte da equipe**.

---

## 1. Escopo e estado atual

O produto e um **pipeline batch agendado** (nao um servico web). A implantacao
consiste em: preparar o ambiente, configurar credenciais, fazer o primeiro build
do dataset/modelo, validar, e **agendar a execucao diaria**. O sinal gerado e
consumido por um operador (ver [`MANUAL_USO.md`](./MANUAL_USO.md)).

**Estado atual:** paper trade ate Q3/2026. Implantacao = uma maquina (workstation
ou VM) rodando o pipeline via agendador do SO. Hardening de producao
(containerizacao, CI/CD, API) e roadmap condicional — ver sec 9.

---

## 2. Arquitetura de execucao

```
  [agendador: cron / Task Scheduler]   (diario, apos 00:00 UTC)
                |
                v
  run_daily.py  ──>  1. fetch_raw_data.py        (12+ APIs externas, incremental)
                     2. bootstrap_from_original.py (monta dataset_production.csv)
                     3. generate_signal.py         (sinal + risk controls)
                |
                v
  data/signal_history.csv  ──>  operador executa o rebalance (manual)
                |
                +─> logs / alerta de kill switch
```

Cada etapa aborta o pipeline em erro (exit code != 0) e tem timeout de 600s
(`run_daily.py:48`), para nao travar em API lenta.

---

## 3. Requisitos de ambiente

| Item | Requisito |
|---|---|
| SO | Linux, macOS ou Windows |
| Python | 3.11+ (validado em **3.13**) |
| Hardware | CPU multi-core (treino bagged paralelo; validado em Intel Ultra 9). GPU nao necessaria |
| Rede | Acesso de saida as APIs (Binance, FRED, BigQuery, yfinance, etc.) |
| Credenciais | `FRED_API_KEY`; conta GCP autenticada para BigQuery (`bq`) |

> **Reprodutibilidade critica:** o XGBoost **nao e deterministico entre CPUs e
> versoes minor**. As metricas canonicas foram validadas em Intel Ultra 9 +
> `xgboost==3.2.0`. **Ao trocar de maquina, sempre retreine na maquina alvo**
> (sec 4, passo 5) — nao reutilize um `cached_models.pkl` treinado em outro
> hardware. Ver `MODEL_FINAL.md` sec 9 e `AI_ONBOARDING.md` (nota de
> reprodutibilidade).

---

## 4. Passo a passo de implantacao

### Passo 1 — Obter o codigo

```bash
git clone <repo-url> longbiased-beny
cd longbiased-beny
```

### Passo 2 — Ambiente Python isolado

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

As versoes core estao **pinadas** (`pandas==2.3.3`, `numpy==2.4.3`,
`scikit-learn==1.8.0`, `xgboost==3.2.0`) por reprodutibilidade — nao as altere.

### Passo 3 — Credenciais (`.env`)

Crie um arquivo `.env` na raiz com a chave do FRED (tier gratuito em
https://fred.stlouisfed.org/docs/api/api_key.html):

```
FRED_API_KEY=sua_chave_aqui
```

### Passo 4 — Autenticar o BigQuery (fonte Messari)

```bash
gcloud auth login          # login interativo
gcloud auth application-default login
bq ls                      # valida que o CLI responde
```

### Passo 5 — Primeiro build completo + treino

```bash
python scripts/production/run_daily.py --full --retrain
```

Isso reconstroi o dataset do zero e **treina o modelo na maquina alvo** (passo
obrigatorio de reprodutibilidade). Gera `data/dataset_production.csv` e
`data/cached_models.pkl`.

### Passo 6 — Validar a implantacao

```bash
# Suite de testes (132 testes; inclui property-test de look-ahead)
pytest -q

# Re-validacao walk-forward OOS (~30min; confirma metricas H1 vs H2)
python scripts/production/walkforward_backtest.py --compare
```

Implantacao OK quando os testes passam e o walk-forward reproduz as metricas do
`MODEL_FINAL.md` (Sortino diario ~3.5, DD ~-7%, dentro do ruido de seed).

### Passo 7 — Agendar a execucao diaria

Rodar **diariamente apos 00:00 UTC**.

**Linux (cron) — exemplo 00:15 UTC:**
```cron
15 0 * * * cd /caminho/longbiased-beny && /caminho/.venv/bin/python scripts/production/run_daily.py >> outputs/results/logs/daily.log 2>&1
```

**Windows (Task Scheduler):** criar tarefa diaria que executa
`.venv\Scripts\python.exe scripts\production\run_daily.py` no diretorio do repo,
redirecionando a saida para um log.

---

## 5. Monitoramento em producao

| O que monitorar | Onde | Acao |
|---|---|---|
| Sinal gerado (cobertura) | `data/signal_history.csv` | Garantir 1 linha por sexta/emergencia |
| Kill switch | bloco "RISK MANAGEMENT" na saida / log | DD <= -12%: investigar regime |
| Acc de-risk | mesma saida; `Rolling acc 12w` | < 48%: checar PSI/drift |
| PSI (drift de features) | saida do `generate_signal.py` | PSI > 3.0 em 3+ features: investigar fontes |
| Falha de etapa | log (`FAILED`/`TIMEOUT` do `run_daily.py`) | Reexecutar; ver Troubleshooting |

**Alerta recomendado:** disparar e-mail/notificacao quando o log contiver
"RISK MANAGEMENT CONTROLS ACTIVATED" ou "Pipeline aborted". (Hoje o pipeline so
loga; o alerta e configurado no agendador/wrapper — ver roadmap sec 9.)

---

## 6. Retrain semi-anual

O retrain e **semi-anual (Janeiro e Julho)**. Para executar a janela:

```bash
python scripts/production/run_daily.py --retrain
```

Apos o retrain, **revalide** com o walk-forward (passo 6) e confira o PSI das top
features. Nao retreine fora do schedule sem motivo — quebra a comparabilidade com
o backtest.

---

## 7. Rollback, rebuild e troubleshooting

```bash
# Rebuild completo (dataset corrompido)
rm scripts/production/data/dataset_production.csv
python scripts/production/run_daily.py --full --retrain

# Rollback para versao anterior (via git)
git log --oneline -- scripts/production/config.py
git checkout <commit> -- scripts/production/config.py \
    scripts/production/build_features.py scripts/production/fetch_raw_data.py
python scripts/production/run_daily.py --full --retrain
```

| Problema | Solucao |
|---|---|
| FRED vazio | Conferir `FRED_API_KEY` no `.env` |
| BigQuery falha | `gcloud auth login` + `bq ls` |
| bitcoin-data.com 429 | Rate limit: aguardar 30s e reexecutar |
| `TIMEOUT` em uma etapa | API externa lenta/fora: reexecutar |
| Model stale | `run_daily.py --retrain` |

Tabela completa: [`scripts/production/INSTRUCTIONS.md`](../scripts/production/INSTRUCTIONS.md) (Troubleshooting).

---

## 8. Checklist de implantacao

- [ ] Repo clonado, venv criado, `pip install -r requirements.txt` OK
- [ ] `.env` com `FRED_API_KEY`
- [ ] BigQuery autenticado (`bq ls` responde)
- [ ] `run_daily.py --full --retrain` concluido (dataset + modelo na maquina alvo)
- [ ] `pytest -q` verde
- [ ] `walkforward_backtest.py --compare` reproduz metricas do `MODEL_FINAL.md`
- [ ] Agendador (cron / Task Scheduler) configurado para apos 00:00 UTC
- [ ] Log + alerta de kill switch configurados

---

## 9. Roadmap de hardening (condicional, pos gate de capital)

Itens **ainda nao implementados** — entram se a decisao de capital (Q3/2026) for
positiva (ver entregaveis #6 deploy e #7 API):

- Containerizacao (`Dockerfile`) e/ou Infra como codigo
- CI/CD rodando `pytest` a cada PR
- API (ex.: endpoint REST com OpenAPI) para distribuir o sinal a sistemas internos
- Alerta automatizado nativo (hoje depende do wrapper do agendador)

---

*Operacao do dia a dia: [`MANUAL_USO.md`](./MANUAL_USO.md). Evolucao do codigo:
[`MANUAL_EVOLUCAO.md`](./MANUAL_EVOLUCAO.md).*
