# Manual de Uso — Sinal de Alocacao BTC/CDI

Projeto `longbiased`
**Publico:** operador da mesa / usuario final que consome o sinal semanal.
**Pre-requisito:** ambiente ja instalado e configurado — ver
[`MANUAL_IMPLANTACAO.md`](./MANUAL_IMPLANTACAO.md).

---

## 1. O que este produto faz

Gera um **sinal de alocacao** que diz, a cada semana, **que percentual da carteira
manter em Bitcoin (0% a 100%) e o restante em CDI**. O sinal vem de um modelo de
ML (ensemble XGBoost) que preve o retorno do BTC a 3 dias, ajustado por regime de
mercado e confianca, com risk controls automaticos. O produto **gera o sinal**; a
**execucao da compra/venda e manual** (feita pelo operador).

---

## 2. Operacao do dia a dia

### 2.1 Gerar o sinal

Na raiz do repositorio, rode:

```bash
python scripts/production/run_daily.py
```

Leva ~2-3 minutos (busca dados de 12+ fontes, recalcula features, gera o sinal).
Rode **todo dia apos 00:00 UTC** (quando os candles diarios fecham). Em producao
isso roda agendado (cron) — ver Manual de Implantacao.

> **Modo debug (sem rebuscar dados):** `python scripts/production/generate_signal.py`
> — usa o dataset ja existente. Util so para reinspecionar o ultimo sinal.

### 2.2 Ler o sinal

A saida tem este formato:

```
=================================================================
  SIGNAL — 2026-04-24 (Fri)  [E1-D7 K=60/30/15]
=================================================================
  BTC Price:      $XX,XXX
  Daily Return:   +X.XX%
  Regime:         BULL (K_base=60)
  Prediction:     +X.XXX% (3d return)
  P(up):          XX.X% (confidence: XX%)
  K effective:    XX (base 60 x 0.XX)

  >> Action:      REBALANCE (Friday)
  >> Allocation:  +XX.X% BTC / XX.X% CDI
  Rolling acc 12w: XX.X% (threshold 48%)
  Current DD:     -X.XX% (kill at -12%)

  Last rebalance: 2026-04-17
  Model trained:  2026-01-01
  Next retrain:   2026-07-01 (64 days)
  Is Friday:      >> YES — REBALANCE DAY
  Emergency:      no (threshold: >8%)
=================================================================
```

| Campo | O que significa |
|---|---|
| **Regime** | Estado do mercado por SMA50/200: BULL / MILD / BEAR (define a agressividade base K) |
| **Prediction** | Retorno previsto do BTC em 3 dias (saida do ensemble regressor) |
| **P(up) / confidence** | Probabilidade de alta (classificador) e a confianca derivada |
| **K effective** | Multiplicador de tamanho da aposta = K base do regime x confianca |
| **>> Action** | **REBALANCE** (executar) ou **HOLD** (manter a alocacao da ultima sexta) |
| **>> Allocation** | **A acao a executar:** % em BTC e % em CDI |
| **Rolling acc 12w** | Acuracia das ultimas 12 semanas (gatilho de seguranca; ver sec 4) |
| **Current DD** | Drawdown acumulado atual (gatilho do kill switch em -12%) |

**A linha que importa para agir e `>> Allocation`.**

---

## 3. Quando agir (regra de rebalance)

| Situacao | Acao |
|---|---|
| **Sexta-feira** | **REBALANCE** — ajustar a carteira para o `>> Allocation` do dia |
| **Emergencia** (\|retorno diario do BTC\| > 8% em qualquer dia) | **REBALANCE** fora de sexta |
| Qualquer outro dia | **HOLD** — manter a alocacao definida na ultima sexta |

A execucao (ordem de compra/venda de BTC) e **manual**. O pipeline nunca movimenta
capital — ele apenas calcula e registra o alvo de alocacao.

---

## 4. Quando um risk control dispara

Se um controle de risco for acionado, aparece um bloco extra na saida:

```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  RISK MANAGEMENT CONTROLS ACTIVATED
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  >> ACCURACY DE-RISK: 45.2% < 48%. Halving alloc (0.5x).
```

| Controle | Quando dispara | O que o sistema faz | O que o operador faz |
|---|---|---|---|
| **Kill switch** | DD acumulado <= -12% | Limita alocacao a 15% | Acompanhar; investigar mudanca de regime. **Nao** sobrepor manualmente sem motivo |
| **Acc de-risk** | Acuracia 12w < 48% **e** confianca media 12w > 80% | Reduz a alocacao pela metade (x0.5) | Verificar drift de features (PSI). O sistema ja se protege sozinho |
| **PSI monitor** | PSI > 3.0 em 3+ features | Apenas alerta (nao ajusta) | Investigar se as fontes de dados mudaram de distribuicao |

Os controles ja estao embutidos no `>> Allocation` exibido — voce executa o numero
final mostrado, nao precisa recalcular.

---

## 5. Sinal semanal automatizado (skill `sinal-semanal`)

Para fechar a semana sem rodar comandos na mao, existe a skill **`sinal-semanal`**
(no Claude Code), que roda o pipeline, preenche o `signal_history.csv` nos dias de
rebalance (sexta/emergencia), faz catch-up de sextas perdidas se os dados estavam
atrasados, e mostra o sinal da semana. Use no fim de cada semana ou quando quiser
"rodar o pipeline / gerar o sinal".

---

## 6. O registro de sinais (`signal_history.csv`)

Cada sinal de rebalance e gravado em
`scripts/production/data/signal_history.csv` (data, regime, previsao, P(up),
alocacao, K efetivo, acao). Esse arquivo **alimenta os calculos de drawdown e
acuracia rolling** — nao o apague nem edite a mao. E tambem a base de evidencia do
paper trade para o gate de decisao de capital.

---

## 7. Problemas comuns

| Sintoma | Causa provavel | O que fazer |
|---|---|---|
| Sinal nao atualiza / data velha | Dataset desatualizado | `python scripts/production/run_daily.py --full` |
| `Alloc` sempre 0% | Previsao negativa (modelo em modo defensivo) | Normal em BEAR/MILD — nao e bug |
| Kill switch ativo | DD <= -12% | Investigar regime; aguardar recuperacao |
| Acc de-risk ativo | Acuracia 12w < 48% | Conferir PSI (drift de features) |
| Erro de FRED / BigQuery vazio | Credenciais | Ver Manual de Implantacao (sec API keys) |

Troubleshooting completo: [`scripts/production/INSTRUCTIONS.md`](../scripts/production/INSTRUCTIONS.md).

---

*Spec do modelo: [`MODEL_FINAL.md`](./MODEL_FINAL.md). Operacao tecnica detalhada:
[`scripts/production/INSTRUCTIONS.md`](../scripts/production/INSTRUCTIONS.md).*
