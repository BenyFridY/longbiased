# `src/features/` — Feature Engineering reutilizavel

Modulos de engenharia de features **reutilizaveis**, separados do pipeline de
producao (`scripts/production/`). Calculam indicadores de regime de mercado e a
taxa livre de risco (CDI), consumidos na construcao do dataset de treino/serving.

## Estrutura

```
src/features/
+-- macro/
|   +-- cdi_rates.py        # CDI/Selic diario (BCB) — taxa livre de risco
+-- regime/
    +-- hurst_features.py          # Hurst exponent (tendencia vs mean-reversion)
    +-- mean_reversion_features.py # half-life, ADF/KPSS (reversao a media)
    +-- regime_change_features.py  # CUSUM / mudanca estrutural de regime
    +-- trend_features.py          # forca e direcao de tendencia
```

## Modulos

| Modulo | Proposito |
|---|---|
| `macro/cdi_rates.py` | Taxa CDI diaria via **BCB SGS API** (serie 12), com **fallback COPOM** offline. `build_rf_daily(dates)` — dias uteis acumulam, fins de semana/feriados = 0; composicao `(1+anual)^(1/252)-1` |
| `regime/hurst_features.py` | **Hurst exponent** (R/S analysis) sobre log-returns: H > 0.5 = tendencia, H < 0.5 = mean reverting |
| `regime/mean_reversion_features.py` | **Half-life** de reversao + testes de estacionariedade (ADF/KPSS) — quando migrar de cripto para renda fixa |
| `regime/regime_change_features.py` | **CUSUM** e deteccao de mudanca estrutural — timing de entrada/saida |
| `regime/trend_features.py` | Tendencia linear e forca de tendencia — quando aumentar exposicao a cripto |

## Convencoes

- Cada arquivo expoe uma classe (`HurstFeatures`, `MeanReversionFeatures`,
  `RegimeChangeFeatures`, `TrendFeatures`) com `feature_names`.
- Sem look-ahead: features sao backward-looking (garantido por
  [`tests/test_lookahead.py`](../../tests/README.md)).

> Nota: o CUSUM em producao usa **retornos simples** (paridade com a base de treino).
> Ver `docs/MODEL_FINAL.md` sec 9 (fix de skew train/serve).
