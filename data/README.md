# DATA - Fontes de Dados (insumos brutos)

**Ultima Atualizacao:** 2026-06-09

> **STATUS / LINHAGEM:** este diretorio guarda **CSVs de fonte bruta** e documenta o
> pipeline **legado** (`build_dataset.py`, ~280 features, export manual da Artemis). O
> **pipeline de PRODUCAO canonico** e outro e vive em `scripts/production/`:
> `fetch_raw_data.py` -> `bootstrap_from_original.py` -> `dataset_production.csv`
> (**32 features**, E1 D7 + H1). Veja
> [`scripts/production/INSTRUCTIONS.md`](../scripts/production/INSTRUCTIONS.md) e
> [`docs/ARQUITETURA_UX.md`](../docs/ARQUITETURA_UX.md). As tabelas abaixo descrevem as
> colunas das fontes brutas e seguem uteis como referencia das fontes de dados.

---

## VISAO GERAL

```
data/
├── bitcoin_all_data.csv    # Artemis - On-chain, Futures, Dev Activity
├── btc.csv                  # CoinMetrics - MVRV, Hash Rate, Flows
├── binance_data.csv         # Binance API - OHLCV, Funding, Basis
├── onchain_data.csv         # Blockchain.com + BGeometrics - Mempool, SOPR, NUPL
├── fred_cache.csv           # FRED API - Macro (cache local)
└── README.md                # Este arquivo
```

---

## 1. bitcoin_all_data.csv (Artemis)

### Fonte
- **Provider:** Artemis Terminal (https://app.artemis.xyz)
- **Como obter:** Export manual do dashboard Artemis
- **Frequencia:** Diaria

### Colunas Utilizadas (32 de 47)
| Coluna Original | Coluna no Pipeline | Descricao |
|-----------------|-------------------|-----------|
| `asset_price_close` | `price_usd` | Preco de fechamento |
| `asset_price_open/high/low` | `price_open/high/low` | OHLC (substituido por Binance) |
| `asset_price_volume` | `volume_usd` | Volume em USD |
| `asset_sharpe-ratio_*` | `sharpe_1y/3y/30d/90d_artemis` | Sharpe Ratios pre-calculados |
| `asset_volatility_*` | `volatility_1y/3y_artemis` | Volatilidade pre-calculada |
| `asset_futures-funding-rate_*` | `funding_rate`, `funding_rate_volume` | Funding rate |
| `asset_futures-open-interest_*` | `open_interest` | Open Interest |
| `asset_futures-volume_*` | `futures_volume`, `futures_buy/sell_volume` | Volume futuros |
| `asset_marketcap_*` | `btc_dominance`, `fdv` | Market Cap, Dominance |
| `network_activity_*` | `active_addresses_24h` | Addresses ativos |
| `network_ecosystem_*` | `core_commits`, `active_developers`, `ecosystem_commits` | Dev Activity |
| `network_financial_*` | `fees_7d_avg`, `fee_median_usd`, `fees_total_24h`, `network_revenue_24h`, `network_expenses_24h`, `block_rewards_24h` | Fees e Revenue |

### Como Atualizar
1. Acesse https://app.artemis.xyz
2. Va para Bitcoin > All Metrics
3. Export CSV com todas as metricas
4. Salve como `bitcoin_all_data.csv` (pode ter timestamp no nome)

### Tratamento no Pipeline
```python
# build_dataset.py faz o seguinte:
artemis_cols = {
    'asset_price_close': 'price_usd',
    'asset_price_open': 'price_open',  # Substituido por Binance
    ...
}
```

---

## 2. btc.csv (CoinMetrics)

### Fonte
- **Provider:** CoinMetrics Community API
- **Como obter:** Download do site ou API
- **URL:** https://coinmetrics.io/community-network-data/

### Colunas Principais
| Coluna Original | Coluna no Pipeline | Descricao |
|-----------------|-------------------|-----------|
| `time` | `date` | Data |
| `CapMVRVCur` | `mvrv_ratio` | MVRV Ratio (Market Value / Realized Value) |
| `CapMrktCurUSD` | `market_cap` | Market Cap em USD |
| `HashRate` | `hash_rate` | Taxa de hash da rede |
| `AdrActCnt` | `active_addresses` | Enderecos ativos |
| `AdrBalCnt` | `addresses_with_balance` | Enderecos com saldo |
| `BlkCnt` | `block_count` | Blocos minerados |
| `IssTotNtv` | `btc_issued_daily` | BTC emitidos no dia |
| `IssTotUSD` | `issuance_usd` | Valor emitido em USD |
| `TxCnt` | `tx_count` | Numero de transacoes |
| `TxTfrCnt` | `transfer_count` | Numero de transferencias |
| `FeeTotNtv` | `fees_btc` | Fees pagas em BTC |
| `FlowInExNtv` | `exchange_inflow_btc` | BTC entrando em exchanges |
| `FlowOutExNtv` | `exchange_outflow_btc` | BTC saindo de exchanges |
| `SplyExNtv` | `supply_on_exchanges` | Supply em exchanges |
| `SplyCur` | `circulating_supply` | Supply circulante |

### Como Atualizar
1. Va para https://coinmetrics.io/community-network-data/
2. Download o CSV de Bitcoin
3. Salve como `btc.csv`

**OU via API:**
```python
import requests
url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
params = {
    "assets": "btc",
    "metrics": "CapMVRVCur,HashRate,AdrActCnt,...",
    "start_time": "2019-01-01"
}
response = requests.get(url, params=params)
```

---

## 3. binance_data.csv (Binance API)

### Fonte
- **Provider:** Binance REST API (gratuita)
- **Como obter:** Rodar `scripts/fetch_binance_data.py`
- **Frequencia:** Diaria

### Colunas Principais

#### OHLCV Spot
| Coluna | Descricao |
|--------|-----------|
| `spot_open` | Preco de abertura |
| `spot_high` | Preco maximo do dia |
| `spot_low` | Preco minimo do dia |
| `spot_close` | Preco de fechamento |
| `spot_volume_btc` | Volume em BTC |
| `spot_volume_usd` | Volume em USD |
| `spot_trade_count` | Numero de trades |
| `spot_taker_buy_btc` | Volume comprador (taker) |
| `spot_taker_buy_ratio` | % volume de compra |

#### Futures
| Coluna | Descricao |
|--------|-----------|
| `futures_open/high/low/close` | OHLC futuros perpetuos |
| `futures_volume_btc/usd` | Volume futuros |
| `basis_pct` | Premium futuros vs spot (%) |
| `basis_annualized` | Basis anualizado |
| `binance_funding_daily` | Soma funding rates do dia |

#### Derivadas
| Coluna | Descricao |
|--------|-----------|
| `true_range` | True Range (ATR base) |
| `atr_14` | Average True Range 14 dias |
| `atr_pct` | ATR como % do preco |
| `gap_pct` | Gap abertura vs fechamento anterior |
| `candle_body_ratio` | Tamanho do corpo vs range |
| `has_futures_data` | Flag: tem dados futuros (0/1) |
| `has_funding_data` | Flag: tem funding (0/1) |

### Como Atualizar
```bash
cd C:\Users\voce\Documents\longbiased
python scripts/fetch_binance_data.py
```

O script:
1. Busca dados spot BTCUSDT desde 2017
2. Busca dados futuros BTCUSDT desde 2019
3. Busca funding rates desde 2020
4. Calcula features derivadas (ATR, candle patterns, basis)
5. Remove features com historico < 365 dias
6. Salva em `data/binance_data.csv`

### Endpoints Utilizados
```
GET /api/v3/klines           # Spot OHLCV
GET /fapi/v1/klines          # Futures OHLCV
GET /fapi/v1/fundingRate     # Funding rates
```

### Limitacoes
- L/S Ratios: So tem ~30 dias de historico (removidos automaticamente)
- Open Interest historico: Limitado (removido automaticamente)

---

## 4. onchain_data.csv (Blockchain.com + BGeometrics)

### Fonte
- **Blockchain.com API:** Gratuita, sem rate limit
- **BGeometrics API:** Gratuita, rate limited (~10 req/hora)
- **Como obter:** Rodar `scripts/fetch_onchain_data.py`

### Colunas
| Coluna | Fonte | Descricao |
|--------|-------|-----------|
| `mempool_tx_count` | Blockchain.com | Transacoes no mempool |
| `mempool_size_bytes` | Blockchain.com | Tamanho do mempool |
| `miners_revenue_usd` | Blockchain.com | Receita dos mineradores |
| `cost_per_tx_usd` | Blockchain.com | Custo por transacao |
| `total_fees_usd` | Blockchain.com | Total de fees |
| `n_transactions` | Blockchain.com | Numero de transacoes |
| `sopr` | BGeometrics | Spent Output Profit Ratio |
| `nupl` | BGeometrics | Net Unrealized Profit/Loss |
| `sth_sopr` | BGeometrics | Short-Term Holder SOPR |
| `realized_price` | BGeometrics | Preco medio de compra |

### Como Atualizar
```bash
# Buscar todos (pode dar rate limit no BGeometrics)
python scripts/fetch_onchain_data.py

# Apenas Blockchain.com (sempre funciona)
python scripts/fetch_onchain_data.py --skip-bgeometrics --skip-google

# Apenas BGeometrics (esperar 1h se rate limited)
python scripts/fetch_onchain_data.py --only-bgeometrics
```

### Por que essas metricas importam
- **SOPR < 1:** Holders vendendo com prejuizo (capitulacao = sinal de compra)
- **NUPL < 0:** Mercado em prejuizo nao realizado (fundo)
- **NUPL > 0.75:** Euforia (topo)
- **Mempool alto:** Rede congestionada (demanda alta)

---

## 5. fred_cache.csv (FRED API)

### Fonte
- **Provider:** Federal Reserve Economic Data (FRED)
- **Como obter:** Automatico via `fredapi` (requer API key)
- **Frequencia:** Diaria/Semanal

### Colunas
| Serie FRED | Coluna no Pipeline | Descricao |
|------------|-------------------|-----------|
| `T10Y2Y` | `yield_curve_2s10s` | Yield Curve (10Y - 2Y) |
| `BAMLH0A0HYM2` | `high_yield_spread` | High Yield Spread |
| `T10YIE` | `breakeven_10y` | Breakeven Inflation 10Y |
| `WALCL` | `fed_balance_sheet` | Fed Balance Sheet |
| `WM2NS` | `m2_supply` | M2 Money Supply |

### Como Obter API Key
1. Cadastre-se em https://fred.stlouisfed.org/
2. Va para My Account > API Keys
3. Crie uma nova API key
4. Adicione ao `.env`:
```
FRED_API_KEY=sua_api_key_aqui
```

### Como Atualizar
O pipeline atualiza automaticamente via API.
Se falhar, usa o cache local (`fred_cache.csv`).

---

## 5. Dados Obtidos Automaticamente pelo Pipeline

Alem dos CSVs, o pipeline busca dados de APIs automaticamente:

### yfinance (Gratuito)
| Ticker | Coluna | Descricao |
|--------|--------|-----------|
| `^VIX` | `vix` | Indice de Volatilidade |
| `^GSPC` | `sp500` | S&P 500 |
| `GC=F` | `gold` | Ouro |
| `CL=F` | `oil` | Petroleo |
| `HG=F` | `copper` | Cobre |
| `DX-Y.NYB` | `dxy` | Dollar Index |
| `^TNX` | `us10y` | Treasury 10Y |
| `ETH-USD` | `eth` | Ethereum |

### Fear & Greed Index (Gratuito)
```
API: https://api.alternative.me/fng/?limit=0
Coluna: fear_greed (0-100)
```

### DefiLlama Stablecoins (Gratuito)
```
API: https://stablecoins.llama.fi/stablecoincharts/all
Coluna: stablecoin_supply
```

---

## FLUXO DO PIPELINE

```
┌─────────────────┐
│ bitcoin_all_data│ ──┐
│     (Artemis)   │   │
└─────────────────┘   │
                      │
┌─────────────────┐   │    ┌───────────────┐    ┌─────────────────┐
│     btc.csv     │ ──┼───>│ build_dataset │───>│ dataset_final   │
│  (CoinMetrics)  │   │    │     .py       │    │     .csv        │
└─────────────────┘   │    └───────────────┘    └─────────────────┘
                      │            │
┌─────────────────┐   │            v
│ binance_data.csv│ ──┤    ┌───────────────┐
│    (Binance)    │   │    │add_extra_     │
└─────────────────┘   │    │features.py    │
                      │    └───────────────┘
┌─────────────────┐   │            │
│  fred_cache.csv │ ──┤            v
│     (FRED)      │   │    Features extras:
└─────────────────┘   │    - Stock-to-Flow
                      │    - Puell Multiple
┌─────────────────┐   │    - Difficulty Ribbon
│   yfinance API  │ ──┤    - Stablecoins
│  (automatico)   │   │
└─────────────────┘   │
                      │
┌─────────────────┐   │
│Fear & Greed API │ ──┘
│  (automatico)   │
└─────────────────┘
```

---

## COMO RODAR O PIPELINE

### Pre-requisitos
```bash
pip install pandas numpy yfinance fredapi requests python-dotenv
```

### Configurar .env
```
FRED_API_KEY=sua_api_key_aqui
```

### Atualizar Binance Data (opcional, ja incluido)
```bash
python scripts/fetch_binance_data.py
```

### Rodar Pipeline Completo
```bash
python scripts/build_dataset.py
```

### Output
- `outputs/dataset.csv` - Dataset final pronto para ML

---

## BOAS PRATICAS

### 1. Nao usar dados futuros (Data Leakage)
O pipeline foi corrigido para evitar data leakage:
- Rolling statistics usam `center=False` (so dados passados)
- Outlier treatment usa rolling mean/std (nao global)
- Imputation usa rolling median (nao global)

### 2. Verificar qualidade antes de treinar
```python
import pandas as pd
df = pd.read_csv('outputs/dataset.csv')

# Checar nulls
print(df.isnull().sum().sum())  # Deve ser 0

# Checar range de datas
print(df['date'].min(), df['date'].max())

# Checar target
print(df['target_direction_1d'].value_counts(normalize=True))
```

### 3. Atualizar dados regularmente
- **Artemis:** Download manual (mensal)
- **CoinMetrics:** Download manual ou API (mensal)
- **Binance:** Rodar script (automatico)
- **FRED:** Automatico via API
- **yfinance:** Automatico via API

---

## TROUBLESHOOTING

### Erro: "No Artemis data file found"
- Verifique se `bitcoin_all_data*.csv` existe em `data/`
- O arquivo pode ter timestamp no nome (ex: `bitcoin_all_data_20251204.csv`)

### Erro: "CoinMetrics file not found"
- Baixe o CSV de https://coinmetrics.io/community-network-data/
- Salve como `btc.csv`

### Erro: "FRED API error"
- Verifique se `FRED_API_KEY` esta no `.env`
- Se falhar, o pipeline usa `fred_cache.csv`

### Dados de Binance desatualizados
```bash
python scripts/fetch_binance_data.py
```

---

## CHANGELOG

### 2026-01-26 (v2)
- Adicionado SOPR, NUPL, STH-SOPR, Realized Price (BGeometrics API)
- Adicionado mempool, miners revenue (Blockchain.com API)
- Adicionado 8 colunas extras do Artemis (sharpe 30d/90d, fees, revenue, etc.)
- Dataset final: 256 features, 2530 rows, 0 nulls

### 2026-01-26 (v1)
- Corrigido `center=True` -> `center=False` no regime (evita data leakage)
- Corrigido outlier treatment para usar rolling stats
- Corrigido imputation para usar rolling median
- Criada documentacao completa

### 2026-01-23
- Adicionado Binance como fonte primaria de OHLC
- Removidos dados antes de Dez 2018
- Renomeado `bitcoin_all_data_*.csv` para aceitar nome simples

### 2026-01-20
- Pipeline inicial criado
- 186 features no dataset final
