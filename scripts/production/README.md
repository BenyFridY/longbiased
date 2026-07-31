# `scripts/production/` — Pipeline ativo

Codigo de producao do modelo E1 D7 + H1 (no-short, com risk controls). Entry point
unico: **`run_daily.py`**.

> **Manual de operacao detalhado:** [`INSTRUCTIONS.md`](./INSTRUCTIONS.md).
> Uso pelo operador: [`../../docs/MANUAL_USO.md`](../../docs/MANUAL_USO.md).
> Implantacao: [`../../docs/MANUAL_IMPLANTACAO.md`](../../docs/MANUAL_IMPLANTACAO.md).

## Rodar

```bash
python scripts/production/run_daily.py            # sinal do dia
python scripts/production/run_daily.py --retrain  # forca retrain (Jan/Jul)
python scripts/production/run_daily.py --full     # rebuild completo dos dados
```

## Modulos

| Arquivo | Proposito |
|---|---|
| `run_daily.py` | Orquestrador: roda as 3 etapas (fetch -> bootstrap -> signal) com abort + timeout 600s |
| `config.py` | Parametros (K, sigmoid, horizonte, bags) + lista das 32 features |
| `fetch_raw_data.py` | Ingestao incremental de 12+ fontes -> `data/raw_data.csv` |
| `bootstrap_from_original.py` | Monta o dataset hibrido (base congelada + dias novos + backfill V36) |
| `build_features.py` | Calcula as 32 features |
| `training.py` | Helpers de treino XGBoost (bagging paralelo) |
| `generate_signal.py` | Treina/infere o ensemble, faz o sizing e aplica risk controls |
| `risk_management.py` | Kill switch + acc de-risk + PSI monitor |
| `walkforward_backtest.py` | Validador walk-forward OOS (re-validacao H1 vs H2) |
| `backfill_signal.py` | Backfill do historico de sinais |

## Dados (`data/`)

| Arquivo | Conteudo |
|---|---|
| `raw_data.csv` | Dados brutos das fontes |
| `dataset_production.csv` | 32 features (dataset hibrido) |
| `cached_models.pkl` | Ensemble treinado (160 reg + 160 cls) |
| `signal_history.csv` | Historico de sinais (alimenta DD e acuracia rolling) |

`archive/experiments/` guarda scripts de overfit/auditoria (deflated Sharpe,
kill-switch sim, etc.).
