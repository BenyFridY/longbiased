# `tests/` — Suite de testes

Suite que protege a integridade do pipeline de producao (**132 testes**). Cobre
ausencia de look-ahead, determinismo, paridade treino/serving, risk controls,
consistencia de config e logica do sinal.

## Como rodar

```bash
pytest -q                          # suite completa
pytest tests/test_lookahead.py     # so o property-test de look-ahead
pytest --cov=scripts --cov=src     # com cobertura
```

`conftest.py` coloca a raiz do repo **e** `scripts/production/` no `sys.path`, para
os testes importarem tanto `scripts.production.*` quanto os modulos `config`/
`generate_signal` usados pelo `walkforward_backtest.py`.

## Arquivos

| Arquivo | O que valida |
|---|---|
| `test_lookahead.py` | **Property-test**: perturba a ultima linha dos dados brutos e prova que **nenhuma feature passada muda** — garante pipeline estritamente backward-looking (anti-leakage) |
| `test_signal_logic.py` | Logica do sinal: sizing, regime, confidence scaling, clip 0-100%, regra de rebalance |
| `test_risk_management.py` | Risk controls: kill switch (DD <= -12%), acc de-risk (12w < 48%), PSI monitor |
| `test_config_consistency.py` | Consistencia da config (features, parametros, invariantes) |
| `test_determinism.py` | Determinismo do treino/inferencia (mesma seed -> mesmo resultado) |
| `test_train_serve.py` | Paridade treino vs serving (features identicas nos dois caminhos) |
| `test_cdi_rates.py` | Taxa CDI (`src/features/macro/cdi_rates.py`): BCB API + fallback COPOM |

## Criterio de aceite

Suite **verde** e pre-requisito de qualquer PR e de implantacao (ver
[`docs/ARQUITETURA_UX.md`](../docs/ARQUITETURA_UX.md) AC-5/AC-6 e
[`docs/MANUAL_EVOLUCAO.md`](../docs/MANUAL_EVOLUCAO.md) sec 4).
