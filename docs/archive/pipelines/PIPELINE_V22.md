# V22 — Sortino Corrigido + Auditoria

**Data**: 2026-03-26

---

## Mudancas vs V21

### 1. Sortino corrigido (Sortino & Price 1994)

V21 usava `std(negativos)` — centra na media dos retornos negativos, inflando o Sortino.

```
V21 (errado):  dd = std(excess[excess < 0])           → Sortino = 4.00
V22 (correto): dd = sqrt(1/N * sum(min(excess, 0)^2)) → Sortino = 4.72
```

A formula correta usa TODOS os dias (N total), com retornos positivos contribuindo zero ao denominador. V21 estava **subestimando** o Sortino real.

| Formula | Sortino (5 seeds) |
|---------|------------------|
| V21: std(negativos) | 4.00 ± 0.07 |
| Errado: sqrt(mean(neg²)) / N_neg | 3.23 ± 0.05 |
| **V22: sqrt(mean(min(ex,0)²)) / N_total** | **4.72 ± 0.07** |

### 2. Sharpe ratio adicionado

Sharpe = 2.35 ± 0.02 (annualizado, sqrt(365))

### 3. Early stopping testado e descartado

Testamos 7 configs de early stopping (patience 10-50, val 15-25%):

| Config | Sortino | Return | Trees |
|--------|---------|--------|-------|
| **Baseline (V21, sem ES)** | **3.21** | **+1046%** | **200** |
| ES patience=10 | 0.65 | +109% | 10 |
| ES patience=20 | 0.69 | +117% | 15 |
| ES patience=30 | 0.72 | +121% | 17 |
| ES patience=50 | 0.74 | +125% | 20 |

Early stopping para em 10-20 arvores de 500 — underfitting severo. O modelo fica basicamente CDI. Series financeiras mudam de regime entre treino e validacao, entao o erro de validacao sobe rapido e o ES corta cedo demais.

**Conclusao**: 200 arvores fixas + bagging de 80 modelos eh a regularizacao certa.

### 4. Modelo/treino/alocacoes identicos ao V21

Unica mudanca eh a metrica. Resultados de backtest sao identicos.

---

## Resultados (5 seeds, 2022-2026)

| Metrica | V22 |
|---------|-----|
| Sortino (correto) | 4.72 ± 0.07 |
| Sharpe | 2.35 ± 0.02 |
| Retorno | +1046% ± 18% |
| MaxDD | -10.0% ± 0.3% |
| BTC buy&hold | +39% |

---

## Pipeline de Producao

Criado pipeline automatizado em `scripts/production/`:

```
fetch_raw_data.py  → 9 fontes (Binance, yfinance, FRED, BQ Messari, etc.)
build_features.py  → 37 features calculadas com funcoes originais
generate_signal.py → Alocacao diaria com logica de rebalance
```

Ver `scripts/production/INSTRUCTIONS.md` para operacao.

---

## Tudo que foi descartado neste ciclo

- Early stopping (todas as variantes): underfitting
- Stop loss (5% BTC drawdown): triggers toda semana, destroi retorno
- Sortino formula `sqrt(mean(neg²))` com N_negativos: mais conservadora que o padrao
