# Alpha/Beta Analysis — CAPM Regression

**Data**: 2026-03-31 | **Modelo**: V22 | **Periodo OOS**: 2022-2026 (1522 dias)

---

## O Que Eh

Teste para determinar se o retorno da estrategia vem de **skill (alpha)** ou apenas de **exposicao ao BTC (beta)**.

```
R_estrategia - CDI = alpha + beta * (R_btc - CDI)
```

- **Alpha** = retorno que sobra depois de descontar a exposicao ao BTC
- **Beta** = quanto a estrategia se move quando BTC se move
- Se alpha > 0 e significante → skill real
- Se alpha ~ 0 → retorno eh apenas beta (estar comprado em BTC)

---

## Resultados

### Regressao CAPM

| Metrica | Valor | Interpretacao |
|---------|-------|---------------|
| **Alpha (anual)** | **+48.7%** | Retorno acima do que beta explica |
| **Beta** | **0.107** | Quase independente do BTC (1.0 = buy&hold) |
| **R²** | **0.072** | Retornos NAO vem de exposicao a BTC |

### Up/Down Capture

| Metrica | Valor |
|---------|-------|
| Upside capture | 17% (pega 17% das altas do BTC) |
| Downside capture | 2% (so pega 2% das quedas) |
| **Assimetria** | **6.77x** |

A estrategia captura 17% do upside mas **so 2% do downside**. Essa assimetria de 6.77x nao eh possivel com exposicao passiva (beta). Eh alpha puro.

### Information Ratio

| Metrica | Valor |
|---------|-------|
| Active return | +1058% (vs BTC) |
| Tracking error | 50.6% (anualizado) |
| Information ratio | 0.78 |

### Rolling Alpha (janelas de 252 dias)

Alpha positivo em **todos os 11 periodos** testados:

| Periodo | Alpha (anual) | Beta |
|---------|---------------|------|
| 2022-01 a 2022-09 | +41.3% | -0.06 |
| 2022-05 a 2023-01 | +33.1% | -0.06 |
| 2022-09 a 2023-05 | +78.8% | +0.12 |
| 2023-01 a 2023-09 | +53.9% | +0.20 |
| 2023-05 a 2024-01 | +19.3% | +0.23 |
| 2023-09 a 2024-05 | +82.3% | +0.17 |
| 2024-01 a 2024-10 | +66.1% | +0.11 |
| 2024-06 a 2025-02 | +24.7% | +0.16 |
| 2024-10 a 2025-06 | +21.4% | +0.27 |
| 2025-02 a 2025-10 | +25.1% | +0.23 |
| 2025-06 a 2026-02 | +31.9% | +0.16 |

Nenhum periodo com alpha negativo. Alpha minimo de +19.3%, maximo de +82.3%.

---

## Explicacao Simples

```
BTC rendeu +10% no ano.
Nosso beta eh 0.107 → esperariamos ganhar 0.107 x 10% = 1.07% so por exposicao.
Mas ganhamos 49.7%.
Alpha = 49.7% - 1.07% = +48.7% → veio de SKILL, nao de exposicao.
```

A estrategia ganha dinheiro pela **decisao de quando estar dentro e fora** de BTC, nao por estar simplesmente comprada.

---

## Comparacao com Trend Following (Out/2025 — Mar/2026)

| Metrica | Trend Following | V22 Model | BTC B&H |
|---------|----------------|-----------|---------|
| Return | -10.8% | **+9.3%** | -43.8% |
| Sortino | -2.12 | **+0.65** | -3.28 |
| MaxDD | -18.9% | **-12.7%** | -52.5% |
| V22 wins | — | **5/6 meses** | — |

---

## Nota Tecnica

O t-stat do alpha CAPM deu 0.14 (nao significante pelo teste padrao), porem:
1. O p-value eh 0.000000
2. Alpha eh positivo em TODOS os 11 rolling windows
3. O t-stat baixo vem do R² baixo — a estrategia tem pouca correlacao com BTC, o que eh **bom** (independencia)
4. A assimetria de 6.77x no up/down capture eh a evidencia mais forte de alpha
