# Racional das Decisoes de Design (o "porque", nao so o backtest)

Este documento explica a **intuicao economica e estatistica** por tras de cada
escolha do modelo — o raciocinio que se defende numa banca, **antes** de invocar
qualquer numero. Cada secao traz: **a escolha**, **o racional conceitual** e, no
fim, uma linha de **confirmacao empirica** (o teste apenas *confirmou* a intuicao,
nao a substituiu).

---

## 1. Por que rebalancear SEMANALMENTE (e na sexta)

**Conceito.** A frequencia de decisao deve casar com a **velocidade com que a
informacao util chega** — nem mais, nem menos. As features que movem o modelo
(macro de liquidez, on-chain, regime de tendencia) evoluem em escala de **dias a
semanas**, nao de horas. Rebalancear *diariamente* seria agir sobre **ruido**:
o sinal muda pouco de um dia para o outro, mas cada trade paga custo e expoe a
*whipsaw* (entra-sai-entra). Rebalancear *mensalmente* seria lento demais para um
ativo que troca de regime em poucas semanas. **Semanal e o ponto de equilibrio**:
rapido o bastante para capturar uma virada de regime, lento o bastante para que
custo e ruido nao corroam o alpha.

**Por que sexta, especificamente.** (i) E o **fim da semana de informacao** — a
decisao incorpora tudo que aconteceu na semana antes de fixar a posicao; (ii) define
de forma **deliberada a exposicao de fim de semana** (cripto negocia 24/7, entao
"quanto carregar no sabado/domingo" e uma decisao real, e melhor toma-la
conscientemente na sexta do que herda-la por inercia); (iii) e **operacionalmente
limpo e auditavel** — uma unica decisao por semana, em dia fixo, que a mesa executa
sem ambiguidade.

> *Confirmacao empirica:* rebal diario derruba o Sortino (~2.0 vs ~3.5); sexta vence
> marginalmente os outros dias da semana. Mas a escolha vem do conceito, nao do sweep.

---

## 2. Por que PREVER 3 dias e SEGURAR ~7 (horizonte != frequencia)

Essa e a aparente contradicao mais interessante — e e **intencional**.

**Conceito.** Sao duas decisoes diferentes:
- **Horizonte de previsao (3 dias):** e onde o modelo tem **edge real**. A
  previsibilidade de retorno **decai com o horizonte** — prever 3 dias tem
  relacao sinal/ruido muito melhor que prever 7. A 3 dias, momentum e regime ainda
  carregam informacao; a 7 dias, o ruido ja domina. Entao treinamos o modelo onde
  ele **consegue acertar**, nao onde gostariamos.
- **Frequencia de acao (semanal, ~7 dias):** e onde a execucao e **economica**.

A previsao de 3 dias funciona como uma **leitura fresca das condicoes atuais**
(um "nowcast" do regime de curto prazo): se ela e positiva e confiante, a
**condicao favoravel tende a persistir pela semana**. Nao estamos literalmente
"tradando o retorno de 3 dias" — usamos a previsao de 3 dias como o **melhor
termometro disponivel** do que vem pela frente, e seguramos a posicao a semana
porque **nao chega informacao materialmente melhor no meio da semana** que
justifique pagar custo para mexer de novo.

**Analogia:** voce olha a previsao do tempo para os proximos 3 dias (confiavel)
para decidir como se vestir esta semana — voce nao re-decide a cada 3 horas, nem
tenta prever o tempo de daqui a 7 dias (impreciso demais).

E se algo grande acontecer no meio da semana? Ai entra a **valvula de emergencia**
(secao 3): o "segurar 7 dias" nao e rigido — e "segurar a semana, *exceto* se um
evento definidor de regime ocorrer".

> *Confirmacao empirica:* horizonte 3d domina 2d/5d/7d com o K calibrado; e o ponto
> de melhor sinal/ruido. A intuicao do decaimento de previsibilidade vem primeiro.

---

## 3. Por que EMERGENCIA em 8% (e nao 5% ou 10%)

**Conceito.** O rebal semanal assume que o mundo nao muda dramaticamente dentro de
uma semana. Mas o BTC tem **cauda gorda** — dias de +-10% a +-20% que sao
**definidores de regime**, nao ruido. Esperar ate sexta para reagir a um *crash* ou
*spike* significaria absorver o pior do movimento sem agir. A emergencia e um
**seguro barato**: raramente dispara, mas captura exatamente os dias que mais
importam para o controle de drawdown.

**Por que 8% e o limiar certo (conceitualmente):** ele separa "**volatilidade
normal que a cadencia semanal absorve**" de "**eventos que exigem resposta
imediata**". O dia tipico do BTC move ~2-4%; 5% acontece com frequencia — gatilhar
ai viraria quase-semanal, devolvendo o problema do *overtrading* e do *whipsaw* que
a cadencia semanal resolve. Acima de ~10%, voce **perde** eventos importantes-mas-
nao-extremos. **8% e o ponto onde o movimento deixa de ser ruido absorvivel e passa
a ser sinal estrutural** (um evento de ~3 desvios para o BTC). E uma assimetria
deliberada: custo quase nulo (dispara pouco), valor alto (pega o que define a perda).

> *Confirmacao empirica:* 3%/5% rebalanceiam demais e pioram o DD; 10% perde
> eventos; 8% e o "sweet spot". A logica do "sinal vs ruido absorvivel" vem antes.

---

## 4. Por que LONG-ONLY (piso = 0, sem short)

**Conceito.** Para um mandato long-biased, o pior caso aceitavel e estar em **caixa
(0% BTC)** — nunca uma posicao vendida. Shortar cripto e **assimetricamente
perigoso**: perda potencialmente ilimitada, custo de aluguel, risco de *short
squeeze*, e — o mais importante — vai contra a **deriva positiva de longo prazo**
do ativo. Dado que a cauda gorda do BTC e principalmente para **cima**, o ganho que
voce capturaria shortando e pequeno frente ao risco de cauda. Piso = 0 limita o pior
caso a "ganhos perdidos", nunca a "perda catastrofica de um short".

> *Confirmacao empirica:* remover o short (floor de -25% -> 0) deu **+0.4 de Sortino
> universal**. Mas e tambem o mandato e a logica de risco assimetrico.

---

## 5. Por que REGIME por SMA50/200

**Conceito.** O filtro de regime responde "a tendencia e de alta ou de baixa?". O
cruzamento de medias **SMA50/200** (golden/death cross) e o detector de tendencia
mais **simples, robusto e universalmente compreendido** do mercado. Ele e *laggy*
(atrasado) de proposito — isso o torna **estavel**, evitando *whipsaw* em ruido.
Usa-lo para definir o multiplicador K (agressivo em BULL, defensivo em BEAR) e uma
forma de **contexto estrategico**: o ML da a visao tatica (a previsao), o regime da
o pano de fundo (apostar mais quando a tendencia apoia, menos quando nao apoia).
Escolhemos o simples e transparente em vez de algo sofisticado (HMM, regime de vol)
por **robustez e interpretabilidade** — defensavel perante um comite de risco e
menos sujeito a overfit.

> *Confirmacao empirica:* sem regime, Sortino cai ~0.7 e DD piora ~3pp; o HMM
> classificou 79% dos dias como "mild" (inutil). Simplicidade venceu.

---

## 6. Por que RETRAIN SEMI-ANUAL (Jan/Jul)

**Conceito.** O mercado evolui, entao o modelo precisa ver dados recentes — mas
retreinar **demais** (mensal) faz o modelo **perseguir ruido recente** e quebra a
comparabilidade; retreinar **de menos** (nunca) deixa o modelo **obsoleto**. Semestral
e o equilibrio: frequente o bastante para adaptar a mudancas estruturais, raro o
bastante para nao caçar ruido e manter um modelo **estavel e auditavel**. Jan/Jul
sao ancoras de calendario limpas.

> *Confirmacao empirica:* semi vence mensal/trimestral/anual; o retrain responde por
> ~66% do Sortino — por isso a escolha da frequencia e tratada com cuidado.

---

## 7. Por que CONFIDENCE SCALING (sigmoid sobre o classificador)

**Conceito.** Nem toda previsao merece a mesma aposta. Quando o modelo esta
**incerto** (P(alta) ~ 50%), aposte pouco; quando esta **confiante** (P longe de
50%), aposte mais. Isso e a intuicao do **Kelly fracionado**: dimensione a aposta
pela sua conviccao/edge. A sigmoide mapeia a confianca para um fator suave que
**nunca zera** (sempre mantem alguma exposicao quando a previsao e positiva),
evitando liga-desliga abrupto.

> *Confirmacao empirica:* confidence adiciona ~+0.15 Sortino e reduz DD; o sweep
> mostra o Sharpe plano numa faixa larga (nao foi overfit do parametro).

---

## 8. Por que K CONSERVADOR (H1 = 60/30/15, e nao H2 = 100/50/20)

**Conceito.** O multiplicador K transforma uma previsao pequena (ex.: +1%) numa
alocacao significativa. Ele e maior em BULL (inclinar quando a tendencia favorece) e
menor em BEAR (proteger). Entre o agressivo (H2) e o conservador (H1), a escolha do
**H1** e uma aposta em **robustez**: um multiplicador menor e **menos sensivel a
erros de previsao e a erros de classificacao de regime**. Em producao o modelo
**nao vai acertar exatamente** — entao e melhor **sub-apostar e ser robusto** do que
super-apostar e depender de o modelo estar certo. H2 maximiza o retorno *no backtest*;
H1 maximiza a **probabilidade de o edge sobreviver ao vivo**.

> *Confirmacao empirica:* H1 e 38-44% mais robusto em testes de "frozen train" (sem
> retrain recente) e tem DD menor. Trocamos retorno de backtest por robustez live.

---

## 9. Por que XGBOOST (e nao deep learning)

**Conceito.** O formato do dado dita a ferramenta. Temos **dados tabulares**, ~30
features **engenheiradas** e poucos milhares de observacoes. Para esse formato,
**gradient boosting domina** redes neurais: e mais **eficiente em amostra**, mais
**robusto**, mais rapido e mais **interpretavel** (importancia de feature, SHAP).
Deep learning (LSTM/Transformer) brilha em sequencias/texto/imagem cruas e com muito
mais dado — aqui seria uma marreta que **overfitta**. Usar a ferramenta certa para o
formato certo e uma decisao de engenharia, nao de moda.

> *Confirmacao empirica:* LSTM, Random Forest, stacking e meta-learners foram
> testados e perderam para o XGBoost puro.

---

## 10. Por que SORTINO como objetivo (e squared error, nao Huber)

**Conceito.** A distribuicao de retorno do BTC e **assimetrica** — os grandes
movimentos sao para **cima** (cauda direita gorda). O **Sharpe** penaliza *toda* a
volatilidade, **inclusive a de alta** (justamente os ganhos que queremos). O
**Sortino** penaliza so o **downside**. Para uma estrategia long-biased de cripto,
**nao** faz sentido punir oscilacoes de alta — queremos proteger a queda e capturar
a subida. Sortino **alinha o objetivo ao mandato**. Pela mesma logica, mantemos
*squared error* (e rejeitamos Huber loss): Huber **suaviza as caudas**, e a cauda de
alta e exatamente de onde vem o alpha — suaviza-la seria jogar fora o premio.

> *Confirmacao empirica:* Huber loss perdeu ~0.56 de Sortino; a skewness dos
> retornos e +3.63 (cauda direita), confirmando que punir/suavizar o upside e erro.

---

## Sintese

Nenhuma dessas escolhas nasceu de um sweep numerico. Cada uma vem de um **principio**
— casar frequencia de decisao com chegada de informacao (1, 2), tratar eventos de
cauda como sinal e nao ruido (3), risco assimetrico (4, 10), robustez sobre ajuste-
fino (5, 6, 8), dimensionar pela conviccao (7), e usar a ferramenta certa para o
formato do dado (9). O backtest serviu para **confirmar** que a intuicao se sustenta
nos dados — nao para descobrir a intuicao.

---

*Spec e numeros: [`MODEL_FINAL.md`](./MODEL_FINAL.md). Testes que confirmam:
[`OVERFIT_TESTS_2026-04-22.md`](./OVERFIT_TESTS_2026-04-22.md). Alternativas rejeitadas:
[`../apresentacao/13_alternativas_rejeitadas.md`](../apresentacao/13_alternativas_rejeitadas.md).*
