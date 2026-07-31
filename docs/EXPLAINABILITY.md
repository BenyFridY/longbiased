# Explicabilidade do Modelo — SHAP

**Reforça o entregavel #5 (Validacao/Interpretacao) e #11 (mitigacao de opacidade etica).**
Gerado por [`scripts/production/archive/experiments/shap_explainability.py`](../scripts/production/archive/experiments/shap_explainability.py).

> Endereca a unica dimensao que separava a Validacao (#5) do nivel maximo e a
> principal critica etica de "caixa-preta" (#11): **por que o modelo preve o que
> preve**. Antes era roadmap; agora esta materializado.

---

## 1. Metodo

- **Tecnica:** SHAP (SHapley Additive exPlanations) via `TreeExplainer` — atribuicao
  exata e teoricamente fundamentada para modelos de arvore.
- **Modelo:** um regressor XGBoost **representativo**, treinado com os parametros e as
  32 features de producao (`config.XGB_PARAMS`, `FEATURES_37`) sobre todo o historico
  (2.645 amostras, target = retorno BTC 3d). O ensemble de producao faz a media de 160
  regressores; este e um modelo representativo para atribuicao **global**.
- **Saida:** beeswarm em [`../outputs/charts/fig_shap_summary.png`](../outputs/charts/fig_shap_summary.png).

---

## 2. Top features por importancia (mean |SHAP|)

| # | Feature | mean \|SHAP\| | Familia |
|---|---|---|---|
| 1 | `eth` | 0.00518 | Cross-asset (preco ETH) |
| 2 | `stablecoin_supply_change_30d` | 0.00354 | On-chain / liquidez |
| 3 | `nupl_ma30` | 0.00315 | On-chain (valuation) |
| 4 | `kpss_stat_30d` | 0.00304 | Regime (estacionariedade) |
| 5 | `fed_fracdiff_05` | 0.00300 | Macro (Fed BS, frac-diff) |
| 6 | `price_fracdiff_05` | 0.00299 | Preco (frac-diff) |
| 7 | `eth_btc_ratio` | 0.00263 | Cross-asset |
| 8 | `velocity` | 0.00255 | Macro (velocidade da moeda) |
| 9 | `basis_ma7` | 0.00235 | Derivativos (basis) |
| 10 | `eth_pctchg_30d` | 0.00234 | Cross-asset |
| 11 | `btc_gold_corr_30d` | 0.00231 | Macro / cross-asset |
| 12 | `cusum_neg` | 0.00219 | Regime (quebra estrutural) |
| 13 | `copper_return_30d` | 0.00205 | Macro |
| 14 | `basis_pct` | 0.00192 | Derivativos |
| 15 | `sortino_30d` | 0.00186 | Momentum / risco |

---

## 3. Interpretacao

- **O edge e distribuido** entre cross-asset (ETH e relacao ETH/BTC), macro
  (Fed/liquidez, velocidade, cobre, correlacao com ouro) e on-chain (stablecoins,
  NUPL) — nenhuma feature isolada domina. Isso e consistente com a tese: o alpha vem
  de combinar muitos sinais fracos, nao de um indicador unico.
- **Corrobora o teste de ablacao (OVERFIT_TESTS, Teste 3):** as features **macro e
  cross-asset carregam o edge** (sem macro, Sortino cai de 5.53 para 2.18). O SHAP
  confirma de forma independente o que a ablacao mostrou removendo grupos.
- **As features V36 on-chain** (reserveRisk, funding_rate_ma7, puellMultiple) aparecem
  com importancia menor — coerente com o achado de que sao marginais (+0.3-0.5 Sortino).

---

## 4. Caveats

- Atribuicao **global** sobre um modelo representativo (nao o ensemble inteiro nem
  walk-forward por cutoff). Serve para entender os drivers, nao para auditar uma
  predicao especifica de um dia.
- SHAP explica o **regressor de magnitude**; a alocacao final tambem depende do regime
  (SMA50/200) e da confianca do classificador, que sao deterministicos e ja documentados.

---

*Validacao: [`OVERFIT_TESTS_2026-04-22.md`](./OVERFIT_TESTS_2026-04-22.md). Spec: [`MODEL_FINAL.md`](./MODEL_FINAL.md).*
