# Pesquisa pré-registrada — floor de alocação negativo (short)

**Data do teste:** 2026-06-11 (pós-congelamento; NÃO adotado em produção)
**Status:** nota de pesquisa para discussão pós-gate · produção segue long-only (floor 0)
**Script:** `scripts/production/archive/experiments/short_floor_test_2026_06_11.py`
**Resultado:** `outputs/results/short_floor_test_2026_06_11.json`

---

## 1. Pergunta

O modelo de produção é long-only: `alloc = clip(pred × K_regime × conf, 0, 1)`.
Quando a previsão é negativa, a alocação vai a zero (100% CDI). Pergunta: permitir
que a alocação fique **negativa** (short BTC) até um piso melhora o resultado?

Short com floor negativo já havia sido testado e rejeitado em configurações
antigas do P&D. Este re-teste foi feito na **configuração final M1** (H1 60/30/15,
sigmoid 5, sem derisk, exec close, kill switch ativo).

## 2. Método

- **Pareado em 10 seeds** sobre as previsões armazenadas
  (`outputs/results/seed_preds_2026_06_09/`) — camada de decisão apenas, zero
  retreino. Janela 2022-01-07 → 2026-05-29, BRL, **bruto**.
- Mecânica de short **deliberadamente otimista** (se short perdesse mesmo assim,
  a conclusão seria robusta): retorno semanal `a·BTC + (1−a)·CDI` com `a < 0` —
  ou seja, proceeds do short rendendo CDI integral e **custo zero de
  funding/aluguel**. Kill switch estendido simetricamente (|a| ≤ 15% sob DD ≤ −12%).
- Pisos testados: 0 (baseline), −10%, −15%, −25%, −50%, −100%.

## 3. Resultados (BRL, 10 seeds, média ± std)

| Piso | CAGR | Sortino (d) | Sharpe (d) | Max DD (d) | 2026 YTD |
|---|---|---|---|---|---|
| **0% — long-only (produção)** | +48.2% ± 0.3 | 3.73 ± 0.04 | 2.23 | −5.34% | +9.3% |
| **−10%** | +53.1% ± 0.4 | **5.30 ± 0.04** | 2.39 | −5.34% | +10.9% |
| −15% | +55.2% ± 0.5 | 4.98 ± 0.02 | 2.39 | −5.36% | +10.9% |
| −25% | +58.8% ± 0.5 | 4.33 ± 0.02 | 2.32 | −6.56% | +10.9% |
| −50% | +64.0% ± 0.6 | 3.35 ± 0.02 | 2.09 | −9.23% | +10.9% |
| −100% | +64.8% ± 0.6 | 2.60 ± 0.02 | 1.77 | −15.97% | +10.9% |

> Nota de janela: o baseline deste harness (48.2%/3.73, corte 29/05) difere do
> headline canônico (50.5%/3.84, corte 05/06) só pela janela — comparações aqui
> são sempre pareadas dentro do mesmo harness.

Deltas pareados do **−10%** vs long-only: Sortino **+1.57** (t=200), CAGR
**+4.9pp** (t=133), DD diário **inalterado** (t=−0.4), 2026 **+1.6pp** (t=24).

**Forma da curva:** CAGR sobe monotonicamente com a profundidade do piso;
Sortino tem pico ≈ −10% e cai a partir daí; DD começa a deteriorar depois de
−15%. Short pequeno = defesa adicional; short grande = aposta direcional que
destrói as métricas do mandato.

**De onde vem o ganho (−10%):** anos de bear — 2022 +14pp, 2025 +5pp, 2026
+1.8pp; 2023 cede −3.4pp. O modelo ficaria short em ~53% das semanas (as
semanas hoje zeradas), quase sempre colado no piso. Turnover +18%
(custo ~0.7pp vs 0.6pp de CAGR — imaterial).

## 4. Por que NÃO foi adotado

1. **Mandato é no-short** — definição de produto (veículos spot
   long-only, defensabilidade em comitê/regulador). Floor negativo não é
   parâmetro; é outro produto (overlay long-short), que exige instrumento
   (perp/futuro/aluguel) e aprovação de risco própria.
2. **Premissas otimistas**: custo zero de funding/aluguel não é realista; o
   ganho de CAGR provavelmente sobrevive a custos razoáveis, mas o Sortino 5.30
   comprimiria — precisa ser modelado com dados reais de funding antes de
   qualquer proposta.
3. **Risco de seleção**: −10% é o pico de um sweep de 6 pisos na mesma janela
   já minerada — adotar o máximo de um sweep é o padrão clássico de overfit por
   seleção (+1 trial no DSR).
4. **Arquitetura desenhada para longs**: o K de regime (BULL 60 / MILD 30 /
   BEAR 15) faz os shorts mais agressivos saírem em regime BULL — o lugar mais
   perigoso para estar vendido. Funcionou no backtest, mas é frágil por
   construção.
5. **Disciplina de congelamento**: produção congelada para o gate Q3/2026; a
   credibilidade do paper trade depende de não mexer.

## 5. Opções na mesa (decisão pós-gate)

| Opção | O que é | Prós | Contras |
|---|---|---|---|
| **A. Shadow tracking** (recomendada) | A partir do retreino de jul/2026, registrar em paralelo a alocação que o variant −10% teria tomado (coluna/arquivo separado do `signal_history.csv`), sem tocar a produção | Custo e risco zero; gera evidência OOS viva; não quebra o congelamento nem o mandato | Resposta só após meses de shadow |
| **B. Adoção documentada** | Floor −10% entra como mudança datada, sob o protocolo do projeto (validação 10-seed feita; faltam custos reais de short e sign-off de mandato) | Captura o ganho imediatamente se for real | Quebra a tese long-only e reinicia o relógio do paper trade/gate; prematuro sem custos modelados |
| **C. Descartar** | Manter rejeição e não acompanhar | Simplicidade | Joga fora um achado com t=200 e DD intacto, validado em 10 seeds |

## 6. Critérios pré-registrados de adoção (gate da pesquisa)

Só vira proposta de mudança de produto se **todos**:

1. Gate de capital Q3/2026 do produto long-only **aprovado** (a pesquisa não
   fura a fila da decisão principal);
2. Custos reais de short modelados (funding perp/aluguel spot, com dados
   históricos) e o variant −10% mantiver **Sortino pareado superior** ao
   long-only líquido de custos;
3. **≥ 3–6 meses de shadow tracking** com Sortino superior pareado e DD não
   pior que a produção;
4. **Sign-off** na mudança de mandato (long-only → overlay
   long-short) e definição do instrumento de short.

## 7. Leitura positiva do achado

Previsões **negativas** do modelo têm valor econômico mesmo sem o modelo nunca
ter sido otimizado para shorts — evidência adicional de que o sinal subjacente
é real (consistente com a permutação 0/1.000.000), não um artefato da
construção long-only.
