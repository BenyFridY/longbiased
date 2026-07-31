# V37 — K Regime Sweep + days_since_halving (2026-04-19)

**Motivação:** Testar se K mais agressivo (sem short) melhora E1 baseline, e se `days_since_halving` captura ciclo institucional.

**Base:** E1 D7 combo no-short (Sortino 6.44, Ret +860%, 32 features, floor=0).

---

## Resultados (3 seeds cada)

| # | Config | K | Sortino | Return | DD | 2026 YTD | Verdict |
|---|--------|---|---------|--------|-----|----------|---------|
| 🥇 | **E1 baseline** | 60/30/15 | **6.441** | +860% | **-8.1%** | +14.4% | Max Sortino |
| 2 | F1 | 80/40/15 | 6.072 | +969% | -8.7% | +14.4% | +13% ret, -0.37 S |
| 3 | F2 | 100/50/20 | 5.775 | +1196% | -9.1% | +17.9% | +39% ret, -0.67 S |
| 4 | F3 | 120/60/30 | 5.428 | +1504% | -9.5% | +21.5% | +75% ret, -1.00 S |
| - | F4 halving | 60/30/15 | 4.253 | +435% | -7.5% | +15.3% | ❌ halving quebra |
| - | F5 K80 + halving | 80/40/15 | 4.146 | +492% | -8.5% | — | ❌ halving quebra |

## Observações

### K sweep: trade-off linear
- Cada +20 em BULL_K = +100-300pp return, -0.3 Sortino, -0.2pp DD
- F3 tem DD apenas -9.5% (ainda excelente)

### Alocação média por ano
| Config | 2022 | 2023 | 2024 | 2025 | 2026 | Média |
|--------|-----:|-----:|-----:|-----:|-----:|------:|
| E1 (prod) | 8% | 22% | 16% | 19% | 24% | 18% |
| F1 | 8% | 26% | 18% | 22% | 24% | 19% |
| F2 | 10% | 29% | 21% | 25% | 31% | 23% |
| F3 | 13% | 32% | 25% | 27% | 46% | 29% |

### days_since_halving: FALHA
- Halving 2024-04-19 resetou feature → modelo tratou como "início ciclo tipo 2020"
- 2024 foi pós-ETF approval (novo regime), não pós-COVID
- Feature precisa de múltiplos ciclos pra generalizar (só temos 1 completo)
- **DESCARTADA da lista de candidatos**

## Decisão

**Mantém E1 K=60/30/15.** K mais agressivo aumenta return mas:
- Sortino cai significativamente
- DD aumenta
- Marketing institucional valoriza Sortino
- Mandato long-biased já bem representado

## Conclusão V37

**6 configs testadas, E1 permanece winner.** K sweep produziu alternativas F1-F3 válidas para pitch diferente (max return), mas padrão institucional = E1.

---

**Arquivo:** `archive/test_scripts/v37_k_and_halving.py`
**Resultados JSON:** `outputs/results/v37_experiments.json`
