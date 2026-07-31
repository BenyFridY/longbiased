# V32 — Final Results (2026-04-19)

**Script**: `archive/test_scripts/v32_exploration.py`
**Results JSON**: `outputs/results/v32_exploration.json`
**Log**: `outputs/results/logs/v32.log`
**Elapsed**: 134.5 min (~2.2h)
**Configs tested**: 36 (10 novel bases + 33 decision-layer sweeps + 3 × 10-seed validation)

---

## Conclusão em uma frase

**Nenhuma das 9 técnicas novas testadas (Huber, quantile, timedecay, monotonic, log target, rank target, multi-horizon, calib, per-regime) bateu o baseline.** O ganho vem inteiramente de `floor=0` (no short), confirmando V31.7 com 10 seeds: **Sortino 6.19 ± 0.07**.

---

## TOP 10 FINAL (validated with 10 seeds where marked n=10)

| # | Config | n | Sortino | Return | DD | Fri Acc | All Acc |
|---|--------|---|---------|--------|-----|---------|---------|
| 🥇 | **baseline_floor0** | 3 | **6.243 ± 0.063** | +830% | -8.3% | 59.9% | 56.01% |
| 🥇 | **baseline_floor0_10s** | **10** | **6.194 ± 0.067** | +824% | -8.2% | 59.4% | 55.75% |
| 🥈 | log_target_floor0 | 3 | 6.123 ± 0.022 | +811% | -8.1% | **61.3%** | **56.40%** |
| 🥈 | log_target_floor0_10s | **10** | 6.094 ± 0.067 | +805% | -8.1% | 60.4% | 56.01% |
| 🥉 | huber_floor0 | 3 | 6.097 ± 0.081 | +828% | -8.4% | 59.0% | 55.88% |
| 🥉 | huber_floor0_10s | **10** | 6.052 ± 0.064 | +816% | -8.3% | 59.0% | 55.81% |
| 4 | calib_floor0 | 3 | 5.917 ± 0.051 | +874% | -8.6% | 59.9% | 56.01% |
| 5 | baseline_sig=10 | 3 | 5.856 ± 0.030 | +1061% | -8.1% | 59.9% | 56.01% |
| 6 | **baseline_default (V31.0 PROD)** | 3 | 5.837 ± 0.027 | **+1146%** | -8.3% | 59.9% | 56.01% |
| 7 | log_target_default | 3 | 5.789 ± 0.011 | +1127% | -8.1% | **61.3%** | **56.40%** |

---

## Phase 1 — Novel Bases (all x 3 seeds, default decision layer)

Ranked by Sortino:

| Base | Sortino | Return | DD | Fri Acc | Veredicto |
|------|---------|--------|-----|---------|-----------|
| baseline | 5.837 | +1146% | -8.3% | 59.9% | REFERÊNCIA (V31.0 = V29 prod) |
| huber | 5.780 | +1151% | -8.4% | 59.0% | Marginal loss (-0.06). Robust to outliers mas sem edge |
| log_target | 5.789 | +1127% | -8.1% | **61.3%** | +1.4pp Fri acc, S quase igual |
| calib | 5.649 | +1209% | -8.6% | 59.9% | Mais return mas Sortino -0.19 |
| multi_horiz | 5.433 | +974% | -9.0% | 58.5% | 7d polui 3d (confirma V21) |
| monotone | 5.287 | +941% | -7.3% | 54.4% | Menor DD -7.3% mas Sortino -0.55 |
| timedecay | 4.747 | +929% | -9.3% | 59.9% | Data antiga tem sinal (confirma V27) |
| quantile | 3.531 | +564% | -19.3% | 56.2% | Median reg underpredicts fat tail |
| rank_target | 3.065 | +1407% | -29.5% | 59.9% | Alto ret mas DD catastrófico |
| per_regime | 2.518 | +445% | -10.7% | n/a | Poucos samples por regime |

---

## Phase 2 — Decision Layer Sweep (post-training, cheap)

**Descoberta: `floor=0` vence EM TODAS as 9 bases testadas** — +0.4 Sortino universal. Isso é por construção do XGB: dados 2022-2026 têm ~68% upside com tails assimétricos. Evitar short elimina cauda ruim sem perder muita upside.

### Baseline com variantes:

| Config | Sortino | Return | DD |
|--------|---------|--------|-----|
| baseline_floor0 | **6.243** | +830% | -8.3% |
| baseline_sig=10 | 5.856 | +1061% | -8.1% |
| baseline_default (V31.0) | 5.837 | +1146% | -8.3% |
| baseline_threshold10 | 5.785 | +1129% | -8.3% |
| baseline_sig=20 | 5.778 | +1176% | -8.4% |
| baseline_K50/30/25 | 5.670 | **+1332%** | -7.9% |
| baseline_sig=25 | 5.740 | +1191% | -8.6% |
| baseline_K70/30/10 | 5.635 | +1076% | -8.4% |
| baseline_K80/20/0 | 5.520 | +1033% | -8.5% |
| baseline_half_weight | 5.287 | +859% | -7.2% |
| baseline_no_sig | 5.621 | +1251% | -9.1% |
| baseline_vol_tgt_15% | 4.521 | +621% | -9.4% |
| baseline_vol_tgt_25% | 4.832 | +782% | -10.2% |
| baseline_kelly | 3.845 | +512% | -11.5% |
| baseline_dd_aware | 4.891 | +847% | -7.8% |

**Notas:**
- Kelly sizing e vol targeting **pioraram** — features demais pra capturar via regra fixa
- `K=50/30/25` máxima retorno (+1332%) mas Sortino pior (5.67)
- threshold rebal e half weight não ajudam
- dd_aware reduz DD pra -7.8% mas Sortino cai

---

## Decisão de Produção

### Opção A — MANTER V29 atual (V31.0 / baseline_default)
- Sortino 5.84, Return +1146%, Fri 59.9%
- **Balanceado**: maximiza return com bom Sortino

### Opção B — TROCAR pra V31.7 (baseline_floor0)
- Sortino 6.19, Return +824% (-28%), Fri 59.4%
- **Max Sortino**: menor downside risk mas abre mão de ~27% return absoluto

### Opção C — log_target + floor=0
- Sortino 6.09, Return +805%, Fri **60.4%** (+1pp vs A)
- **Max accuracy**: ligeiramente melhor predição em todos os dias (56.01% vs 55.75% all-day)

### Trade-off
- De V29 → V31.7: **+0.35 Sortino** em troca de **-27% return** absoluto
- Se o objetivo é Sharpe-ratio-like metric (risco-ajustado), V31.7 ganha
- Se o objetivo é CAGR / retorno absoluto, V29 ganha

---

## Técnicas a NÃO retestar (adiciona à lista de V26-V31)

Adicionar à feedback_not_retest.md:

1. **Quantile regression (reg:quantileerror 0.5)** — median underpredicts em dist assimétrica
2. **Rank transform target** — causa scaling agressivo, DD catastrófico
3. **Per-regime models** — poucos samples (~200-500) por regime
4. **Pseudo-Huber loss** — marginal -0.06 Sortino
5. **Time decay sample weights (half-life 2y)** — antigo é informativo (confirma)
6. **Multi-horizon 3d+7d blend** — 7d polui 3d
7. **Isotonic calibration** — já incluído implicitly no sigmoid
8. **Monotonic constraints** — reduz DD mas Sortino pior
9. **Kelly sizing (pred/var)** — piora Sortino pra 3.85
10. **Vol targeting 15%/25%** — piora Sortino
11. **Threshold rebalance (0.10)** — sem diferença material
12. **Half-weight rebalance** — reduz return sem melhorar Sortino
