# V38 — Selic/CDI como Feature (2026-04-19)

**Motivação:** Usuário perguntou se modelo "sabe" da Selic (2% em 2021 vs 15% em 2026). Testar se adicionar taxa brasileira como feature ajuda.

**Base:** E1 D7 combo no-short (Sortino 6.44, 32 features, floor=0).

---

## Contexto importante

**Selic NÃO afeta predição do BTC** — BTC é ativo global, preço o mesmo em qualquer país. Selic afeta **flow** de investidor BR, que é ~2-5% do volume global.

**MAS** Selic JÁ afeta retorno do FUNDO automaticamente via cálculo de portfolio:
```python
strategy_return = allocation × BTC_return + (1 - allocation) × CDI_daily
```

Ou seja, quando Selic sobe:
- 76% do fundo (CDI portion) rende mais automaticamente
- Só 24% (BTC portion) depende do modelo
- Cliente já colhe benefício sem modelo "saber" Selic

Sortino também subtrai rf_daily (CDI) no cálculo — "excess return" reportado É alpha pura vs CDI.

## Resultados (3 seeds cada, fixed weekends = ffill Selic)

| # | Config | Feature nova | Sortino | Return | DD |
|---|--------|--------------|---------|--------|-----|
| 🥇 | **E1 baseline** | — | **6.441** | +860% | -8.1% |
| G1 | + selic_ann | Selic anualizada | 6.396 | +814% | **-7.7%** |
| G2 | + selic_change_30d | Variação 30d | 5.637 | +619% | -8.7% |
| G3 | + ambos | Nível + variação | 5.629 | +581% | -8.1% |
| G4 | + us10y | Treasury 10Y US | (rodando) | — | — |

## Interpretação

### G1 selic_ann: marginal, DD melhor
- **Sortino empata** E1 (-0.05 dentro de ruído)
- **DD melhora** (-7.7% vs -8.1%)
- Modelo aprendeu ligeiramente: reduz exposição quando Selic alta (custo de oportunidade)
- Benefício real muito pequeno — pode ser overfit de 3 seeds

### G2 selic_change_30d: FALHA
- **-0.80 Sortino** vs baseline
- Variações Selic são raras (~6 COPOMs/ano, +/-0.5pp)
- Feature quase constante = ruído
- Modelo se confunde com sinal fraco

### G3 combo: PIOR
- Adicionar selic_change + selic_ann juntas amplifica ruído
- Confirma que change é problemático

## Decisão

**NÃO adicionar Selic como feature ao E1.**

Razões:
1. Ganho marginal (0.05 Sortino, dentro de ruído)
2. Adiciona complexidade (fonte externa, BCB API)
3. CDI já beneficia retorno automaticamente
4. BTC é global, flow BR tem impacto limitado
5. Features macro US (fed_balance_sheet) já capturam monetary cycle
6. G1 ganho de DD pode ser noise de 3 seeds — precisaria 10 seeds

## Conclusão V38

**Selic NÃO é feature útil pro modelo.** User insight válido conceitualmente mas empiricamente confirmado que já está coberto indiretamente.

---

**Arquivo:** `archive/test_scripts/v38_selic_features.py`
**Resultados JSON:** `outputs/results/v38_experiments.json`
