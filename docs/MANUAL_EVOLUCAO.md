# Manual de Evolucao — Desenvolvimento e Manutencao

Projeto `longbiased-beny`
**Publico:** desenvolvedor que vai manter ou evoluir o modelo/codigo.
**Objetivo:** configurar o ambiente de desenvolvimento e alterar o codigo com
seguranca, **sem suporte da equipe original**.

---

## 1. Ambiente de desenvolvimento

```bash
git clone <repo-url> longbiased-beny
cd longbiased-beny
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows:     .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Python 3.11+ (validado em 3.13). O `requirements.txt` ja inclui as ferramentas de
qualidade: `pytest`, `pytest-cov`, `black`, `flake8`, `mypy`.

**Versoes pinadas (NAO alterar sem revalidar):** `pandas==2.3.3`, `numpy==2.4.3`,
`scipy==1.17.1`, `scikit-learn==1.8.0`, `xgboost==3.2.0`. Sao a base das metricas
canonicas; mudar versao muda os numeros do backtest.

---

## 2. Fonte canonica e regra de ouro

- **`docs/MODEL_FINAL.md` e a unica fonte de verdade** da config e das metricas.
- **Regra de ouro:** *toda* alteracao que afete predicao (features, parametros,
  loss, horizonte, treino) exige **retrain + re-validacao walk-forward 10-seed**
  antes de virar canonica. Numeros nao revalidados nao entram no `MODEL_FINAL.md`.

---

## 3. Estrutura do repositorio

```
longbiased-beny/
|- scripts/production/        # CODIGO ATIVO (pipeline de producao)
|  |- run_daily.py            # Entry point: orquestra as 3 etapas
|  |- config.py               # Parametros (K, sigmoid, horizonte, bags) + 32 features
|  |- fetch_raw_data.py       # Ingestao de 12+ fontes
|  |- bootstrap_from_original.py  # Monta o dataset hybrid
|  |- build_features.py       # Calcula as 32 features
|  |- generate_signal.py      # Treina/infere + aplica risk controls
|  |- training.py             # Helpers de treino XGBoost
|  |- risk_management.py       # Kill switch + acc de-risk + PSI
|  |- walkforward_backtest.py # Validador walk-forward OOS
|  |- data/                   # dataset, raw, cached_models.pkl, signal_history.csv
|  |- archive/experiments/    # Scripts de overfit/auditoria
|- src/features/              # Feature engineering reutilizavel (macro, regime)
|- tests/                     # Suite (132 testes)
|- docs/                      # MODEL_FINAL, OVERFIT_TESTS, AI_ONBOARDING, manuais, plano
|- outputs/                   # Datasets, resultados, graficos
|- archive/                   # Historico V02-V22
|- requirements.txt
```

---

## 4. Rodar os testes

```bash
pytest -q                       # suite completa (132 testes)
pytest tests/test_lookahead.py  # property-test: features estritamente backward-looking
pytest --cov=scripts --cov=src  # com cobertura
```

A suite cobre: ausencia de look-ahead/leakage (`test_lookahead.py`), determinismo
(`test_determinism.py`), paridade treino/serving (`test_train_serve.py`), risk
controls (`test_risk_management.py`), consistencia de config (`test_config_consistency.py`),
logica do sinal (`test_signal_logic.py`) e CDI (`test_cdi_rates.py`).
**Rode a suite antes e depois de qualquer mudanca.**

---

## 5. Como fazer mudancas comuns

| Quero... | Onde mexer | Passos obrigatorios depois |
|---|---|---|
| Mudar K / sigmoid / horizonte / bags | `scripts/production/config.py` | `run_daily.py --retrain` + walk-forward + `pytest` |
| Adicionar/remover feature | `config.py` (`FEATURES_37`) + `build_features.py` | retrain (modelos ficam incompativeis) + revalidar |
| Adicionar fonte de dados | `fetch_raw_data.py` (nova `fetch_*`) + `build_features.py` | conferir alinhamento train/serve + revalidar |
| Ajustar um risk control | `risk_management.py` | `pytest tests/test_risk_management.py` + backtest de impacto |
| Mudar logica de sizing | `generate_signal.py` / `walkforward_backtest.py` | manter as duas em paridade + revalidar |

> **Atencao a paridade train/serve:** features sao calculadas no treino e no
> serving por caminhos diferentes; mudancas devem manter os dois identicos (ha
> historico de bugs de skew, ex.: CUSUM log vs simples — `MODEL_FINAL.md` sec 9).

---

## 6. Re-validacao obrigatoria

```bash
# Walk-forward OOS, compara H1 vs H2 (~30min)
python scripts/production/walkforward_backtest.py --compare

# Auditoria diaria de DD (rapido, usa preds existentes)
python scripts/production/archive/experiments/final_audit_daily_dd.py

# Deflated Sharpe (Bailey-Prado)
python scripts/production/archive/experiments/deflated_sharpe.py
```

Uma mudanca so e aceita se as metricas se mantiverem dentro do ruido de seed
(Sortino diario std ~0.06) ou melhorarem de forma robusta (testar em 10 seeds,
nao 1).

---

## 7. O que NAO fazer

1. Nao retreinar fora de `run_daily.py --retrain` (quebra a comparabilidade walk-forward).
2. Nao alterar `FEATURES_37` sem retrain — modelos cacheados ficam incompativeis.
3. Nao colocar `ALLOC_MIN` negativo — viola o mandato no-short.
4. Nao remover risk controls — foram adicionados apos auditoria por razoes especificas.
5. Nao reutilizar `cached_models.pkl` em outra maquina — retreine no hardware alvo.

---

## 8. Reprodutibilidade

O XGBoost **nao e deterministico entre CPUs e versoes minor**. Por isso:
- Deps core pinadas no `requirements.txt`.
- Cache de modelo carrega um *fingerprint* (sha1 sobre features/bags/horizonte/params)
  para detectar config stale e forcar retrain automatico.
- Ao validar/operar em maquina nova: **retreine na maquina alvo** antes de confiar
  nos numeros. Metricas canonicas = Intel Ultra 9 + `xgboost==3.2.0`.

---

## 9. Workflow de contribuicao

- Branch por feature; PR para `master` (historico: PRs #1 audit, #2 experiments,
  #3 weekly-signal-tooling).
- Antes do PR: `pytest -q` verde + (se afetar predicao) walk-forward revalidado.
- Documentar a mudanca no historico do `MODEL_FINAL.md` (sec 9) quando alterar
  config/metricas canonicas.
- Formatacao/lint: `black .` e `flake8` antes do commit.

---

*Spec do modelo: [`MODEL_FINAL.md`](./MODEL_FINAL.md). Operacao:
[`MANUAL_USO.md`](./MANUAL_USO.md). Implantacao:
[`MANUAL_IMPLANTACAO.md`](./MANUAL_IMPLANTACAO.md).*
