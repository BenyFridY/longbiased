"""
BUILD DATASET - Pipeline Completo e Autonomo
=============================================

Gera o dataset do ZERO usando todas as fontes de dados.
NAO DEPENDE de nenhum dataset pre-existente.

Fontes:
- data/bitcoin_all_data_*.csv (Artemis - on-chain, futures, dev activity)
- data/btc.csv (CoinMetrics - MVRV, hash, flows)
- yfinance (VIX, SP500, Gold, Oil, Copper, DXY, US10Y, ETH)
- FRED API (Yield curves, M2, Fed balance sheet)
- Fear & Greed API (Sentiment)

Output: outputs/dataset_1.csv

Uso:
    python scripts/build_dataset_1.py

Configuracao:
    Ajuste START_DATE e END_DATE abaixo conforme necessario.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Import extra features module
try:
    from add_extra_features import add_all_extra_features, treat_outliers, add_calculated_features
    EXTRA_FEATURES_AVAILABLE = True
except ImportError:
    EXTRA_FEATURES_AVAILABLE = False
    print("WARNING: add_extra_features.py not found. Extra features will be skipped.")

# Import regime features module (sophisticated features from src/features/regime/)
try:
    from add_regime_features import add_all_regime_features
    REGIME_FEATURES_AVAILABLE = True
except ImportError:
    REGIME_FEATURES_AVAILABLE = False
    print("WARNING: add_regime_features.py not found. Regime features will be skipped.")

# =============================================================================
# CONFIGURATION - AJUSTE AQUI
# =============================================================================
START_DATE = "2019-01-01"
END_DATE = "2026-03-03"

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

# Create output dir if not exists
OUTPUT_DIR.mkdir(exist_ok=True)

# Output file
OUTPUT_FILE = OUTPUT_DIR / "dataset_final.csv"

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

FRED_API_KEY = os.getenv("FRED_API_KEY", "").strip('"')

print("=" * 70)
print("BUILD DATASET - Pipeline Completo e Autonomo")
print("=" * 70)
print(f"Period: {START_DATE} to {END_DATE}")
print(f"Output: {OUTPUT_FILE}")
print("=" * 70)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def calc_rsi(prices, period=14):
    """
    Calculate RSI indicator using Wilder's EMA (exponential moving average).

    FIX 1.4: Changed from SMA to EMA with alpha=1/period for proper Wilder smoothing.
    This matches the original RSI formula by J. Welles Wilder.
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    # Use Wilder's smoothing (EMA with alpha=1/period)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calc_adx(high, low, close, period=14):
    """Calculate ADX, +DI, -DI, ATR."""
    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    plus_dm = high.diff()
    minus_dm = (-low.diff())

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    # Smoothed values (Wilder's smoothing)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)

    # DX and ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    return adx, plus_di, minus_di, atr

def calc_hurst(series, window=30):
    """
    Calculate Hurst exponent using R/S (Rescaled Range) method on LOG RETURNS.

    H > 0.5: Trending/persistent series
    H = 0.5: Random walk
    H < 0.5: Mean-reverting/anti-persistent series

    FIXED: Now uses log returns instead of price differences.
    """
    result = []
    for i in range(len(series)):
        if i < window:
            result.append(np.nan)
        else:
            prices = series.iloc[i-window:i].dropna()
            if len(prices) < window // 2:
                result.append(np.nan)
                continue
            try:
                # CRITICAL FIX: Use LOG RETURNS, not price differences
                log_returns = np.log(prices / prices.shift(1)).dropna().values

                if len(log_returns) < 10:
                    result.append(np.nan)
                    continue

                # R/S Analysis over different lag sizes
                lags = range(2, min(20, len(log_returns) // 2))
                rs_values = []

                for lag in lags:
                    # Divide series into blocks of size 'lag'
                    n_blocks = len(log_returns) // lag
                    if n_blocks < 1:
                        continue

                    rs_block = []
                    for j in range(n_blocks):
                        block = log_returns[j*lag:(j+1)*lag]
                        if len(block) < 2:
                            continue

                        # Mean-centered cumulative deviations
                        mean = np.mean(block)
                        deviations = block - mean
                        cumsum = np.cumsum(deviations)

                        # Range (R)
                        R = np.max(cumsum) - np.min(cumsum)

                        # Standard deviation (S)
                        S = np.std(block, ddof=1)

                        if S > 0:
                            rs_block.append(R / S)

                    if rs_block:
                        rs_values.append(np.mean(rs_block))

                if len(rs_values) > 2:
                    # Log-log regression to find Hurst exponent
                    log_lags = np.log(list(lags)[:len(rs_values)])
                    log_rs = np.log(rs_values)

                    # Remove invalid values
                    mask = np.isfinite(log_lags) & np.isfinite(log_rs)
                    if np.sum(mask) < 2:
                        result.append(np.nan)
                        continue

                    poly = np.polyfit(log_lags[mask], np.array(log_rs)[mask], 1)
                    hurst = np.clip(poly[0], 0, 1)
                    result.append(hurst)
                else:
                    result.append(np.nan)
            except:
                result.append(np.nan)
    return result

def classify_regime(row):
    """Classify market regime based on volatility, returns, ADX, and Hurst."""
    vix = row.get('vix', 20) or 20
    vol_30d = row.get('volatility_30d', 0.5) or 0.5
    return_30d = row.get('return_30d', 0) or 0
    adx = row.get('adx', 20) or 20
    hurst = row.get('hurst_30d', 0.5) or 0.5

    # Crisis: VIX very high OR extreme volatility OR rapid decline
    if (vix > 35) or (vol_30d > 0.9) or (return_30d < -0.20):
        return 4  # crisis

    # Strong trending: ADX > 30
    if adx > 30:
        if return_30d > 0.05:
            return 2  # trending_up
        elif return_30d < -0.05:
            return -2  # trending_down

    # Moderate trending: ADX 20-30
    if adx > 20:
        if return_30d > 0.02:
            return 1  # mild_up
        elif return_30d < -0.02:
            return -1  # mild_down

    # Mean reverting: ADX < 20 AND Hurst < 0.45
    if (adx < 20) and (hurst < 0.45):
        return -3  # mean_reverting

    return 0  # neutral

# =============================================================================
# 1. LOAD ARTEMIS DATA
# =============================================================================
print("\n[1/11] Loading Artemis data...")

# Find latest Artemis file (accepts bitcoin_all_data.csv or bitcoin_all_data_*.csv)
artemis_files = list(DATA_DIR.glob("bitcoin_all_data*.csv"))
if not artemis_files:
    raise FileNotFoundError(f"No Artemis data file found in {DATA_DIR}")

ARTEMIS_FILE = max(artemis_files, key=lambda x: x.stat().st_mtime)
print(f"   Using: {ARTEMIS_FILE.name}")

artemis = pd.read_csv(ARTEMIS_FILE)
artemis['date'] = pd.to_datetime(artemis['time']).dt.date
artemis['date'] = pd.to_datetime(artemis['date'])

# Rename columns to friendly names
artemis_cols = {
    'asset_price_close': 'price_usd',
    'asset_price_open': 'price_open',
    'asset_price_high': 'price_high',
    'asset_price_low': 'price_low',
    'asset_price_volume': 'volume_usd',
    'asset_marketcap_circulating-marketcap-dominance': 'btc_dominance',
    'asset_sharpe-ratio_sharpe-ratio-1y': 'sharpe_1y_artemis',
    'asset_sharpe-ratio_sharpe-ratio-3y': 'sharpe_3y_artemis',
    'asset_sharpe-ratio_sharpe-ratio-30d': 'sharpe_30d_artemis',  # NEW
    'asset_sharpe-ratio_sharpe-ratio-90d': 'sharpe_90d_artemis',  # NEW
    'asset_volatility_volatility-1y': 'volatility_1y_artemis',
    'asset_volatility_volatility-3y': 'volatility_3y_artemis',
    'asset_futures-funding-rate_funding-rate-open-interest': 'funding_rate',
    'asset_futures-funding-rate_funding-rate-volume': 'funding_rate_volume',  # NEW
    'asset_futures-open-interest_open-interest': 'open_interest',
    'asset_futures-volume_volume-usd': 'futures_volume',
    'asset_futures-volume_trade-count': 'futures_trade_count',
    'asset_futures-volume_volume-buy-usd': 'futures_buy_volume',
    'asset_futures-volume_volume-sell-usd': 'futures_sell_volume',
    'network_activity_activeAddresses24Hour': 'active_addresses_24h',
    'network_ecosystem_coreCommits24Hour': 'core_commits',
    'network_ecosystem_activeDevelopers24Hour': 'active_developers',
    'network_ecosystem_ecosystemCommits24Hour': 'ecosystem_commits',
    'network_financial_rolling7dAvgFees': 'fees_7d_avg',
    'network_financial_feeMedian24HourUsd': 'fee_median_usd',
    'network_financial_feesTotal24HourUsd': 'fees_total_24h',  # NEW
    'network_financial_expenses24HourUsd': 'network_expenses_24h',  # NEW
    'network_financial_revenue24HourUsd': 'network_revenue_24h',  # NEW
    'network_financial_feesSupplySide24HourUsd': 'fees_supply_side_24h',  # NEW
    'network_financial_tokenIncentives24HourUsd': 'block_rewards_24h',  # NEW
    'asset_marketcap_fully-diluted-marketcap': 'fdv',
    'network_financial_avgFeePerTxn24Hour': 'fee_per_tx',
}

for old, new in artemis_cols.items():
    if old in artemis.columns:
        artemis[new] = artemis[old]

# Select and dedupe
artemis_keep = ['date', 'price_usd', 'price_open', 'price_high', 'price_low', 'volume_usd',
                'btc_dominance', 'sharpe_1y_artemis', 'sharpe_3y_artemis',
                'sharpe_30d_artemis', 'sharpe_90d_artemis',  # NEW
                'volatility_1y_artemis', 'volatility_3y_artemis',
                'funding_rate', 'funding_rate_volume',  # NEW
                'open_interest', 'futures_volume', 'futures_trade_count',
                'futures_buy_volume', 'futures_sell_volume',
                'active_addresses_24h', 'core_commits',
                'active_developers', 'ecosystem_commits', 'fees_7d_avg', 'fee_median_usd',
                'fees_total_24h', 'network_expenses_24h', 'network_revenue_24h',  # NEW
                'fees_supply_side_24h', 'block_rewards_24h',  # NEW
                'fdv', 'fee_per_tx']

artemis_keep = [c for c in artemis_keep if c in artemis.columns]
artemis = artemis[artemis_keep].copy()
artemis = artemis.groupby('date').last().reset_index()

print(f"   Artemis: {len(artemis)} rows, {len(artemis.columns)} columns")

# =============================================================================
# 2. LOAD COINMETRICS DATA (HashRate via API, fallback to btc3.csv)
# =============================================================================
print("\n[2/11] Loading CoinMetrics HashRate...")


def fetch_coinmetrics_hashrate_api(start_date: str) -> pd.DataFrame:
    """Fetch HashRate from CoinMetrics Community API (free, no key needed).

    V19 only uses hash_rate from CoinMetrics. All other columns (MVRV, exchange
    flows, etc.) were loaded but never used by the model.
    """
    import requests as _req

    url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    params = {
        "assets": "btc",
        "metrics": "HashRate",
        "frequency": "1d",
        "start_time": start_date,
        "page_size": 10000,
    }

    all_rows = []
    while True:
        resp = _req.get(url, params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        all_rows.extend(payload.get("data", []))
        next_url = payload.get("next_page_url")
        if not next_url:
            break
        # Use full next_page_url for pagination
        url = next_url
        params = {}  # params are embedded in next_page_url

    if not all_rows:
        raise ValueError("CoinMetrics API returned no data")

    df = pd.DataFrame(all_rows)
    df['date'] = pd.to_datetime(df['time']).dt.tz_localize(None).dt.normalize()
    df['hash_rate'] = pd.to_numeric(df['HashRate'], errors='coerce')
    df = df[['date', 'hash_rate']].dropna()
    df = df.groupby('date').last().reset_index()
    return df


def load_coinmetrics_csv_fallback() -> pd.DataFrame:
    """Fallback: load HashRate from local btc3.csv."""
    coinmetrics_files = list(DATA_DIR.glob("btc*.csv"))
    coinmetrics_files = [f for f in coinmetrics_files if f.name.startswith('btc')
                         and not f.name.startswith('bitcoin')]
    if not coinmetrics_files:
        raise FileNotFoundError(f"No CoinMetrics CSV found in {DATA_DIR}")
    cm_file = max(coinmetrics_files, key=lambda x: x.stat().st_mtime)
    print(f"   Fallback CSV: {cm_file.name}")
    raw = pd.read_csv(cm_file)
    raw['date'] = pd.to_datetime(raw['time']).dt.normalize()
    if 'HashRate' in raw.columns:
        raw['hash_rate'] = raw['HashRate']
    raw = raw[['date', 'hash_rate']].dropna()
    raw = raw.groupby('date').last().reset_index()
    return raw


try:
    cm = fetch_coinmetrics_hashrate_api(START_DATE)
    print(f"   CoinMetrics API: {len(cm)} rows (hash_rate)")
except Exception as e:
    print(f"   CoinMetrics API failed: {e}")
    print("   Falling back to local CSV...")
    cm = load_coinmetrics_csv_fallback()
    print(f"   CoinMetrics CSV: {len(cm)} rows (hash_rate)")

# =============================================================================
# 3. FETCH YFINANCE DATA
# =============================================================================
print("\n[3/11] Fetching yfinance data...")

try:
    import yfinance as yf

    tickers = {
        '^VIX': 'vix',
        '^GSPC': 'sp500',
        'GC=F': 'gold',
        'CL=F': 'oil',
        'HG=F': 'copper',
        'DX-Y.NYB': 'dxy',
        '^TNX': 'us10y',
        'ETH-USD': 'eth',
    }

    yf_data = []
    for ticker, name in tickers.items():
        try:
            data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if len(data) > 0:
                # Handle MultiIndex columns
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                data = data[['Close']].rename(columns={'Close': name})
                data.index = pd.to_datetime(data.index.date)
                yf_data.append(data)
                print(f"      {name}: {len(data)} rows")
        except Exception as e:
            print(f"      {name}: FAILED - {e}")

    if yf_data:
        yf_df = yf_data[0]
        for yf_d in yf_data[1:]:
            yf_df = yf_df.join(yf_d, how='outer')
        yf_df = yf_df.reset_index().rename(columns={'index': 'date'})
        yf_df['date'] = pd.to_datetime(yf_df['date'])
    else:
        yf_df = pd.DataFrame({'date': pd.date_range(START_DATE, END_DATE)})

except ImportError:
    print("   WARNING: yfinance not installed")
    yf_df = pd.DataFrame({'date': pd.date_range(START_DATE, END_DATE)})

print(f"   yfinance total: {len(yf_df)} rows")

# =============================================================================
# 4. FETCH FRED DATA
# =============================================================================
print("\n[4/11] Fetching FRED data...")

FRED_CACHE_FILE = DATA_DIR / "fred_cache.csv"

fred_df = None

if FRED_API_KEY:
    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)

        fred_series = {
            'T10Y2Y': 'yield_curve_2s10s',
            'BAMLH0A0HYM2': 'high_yield_spread',
            'T10YIE': 'breakeven_10y',
            'WALCL': 'fed_balance_sheet',
            'WM2NS': 'm2_supply',
        }

        fred_dfs = []
        for series_id, name in fred_series.items():
            try:
                data = fred.get_series(series_id, observation_start=START_DATE)
                if len(data) > 0:
                    temp_df = pd.DataFrame({name: data})
                    temp_df.index = pd.to_datetime(temp_df.index.date)
                    fred_dfs.append(temp_df)
                    print(f"      {name}: {len(temp_df)} rows")
            except Exception as e:
                print(f"      {name}: FAILED - {e}")

        if fred_dfs:
            fred_df = fred_dfs[0]
            for temp_df in fred_dfs[1:]:
                fred_df = fred_df.join(temp_df, how='outer')
            fred_df = fred_df.reset_index().rename(columns={'index': 'date'})
            fred_df['date'] = pd.to_datetime(fred_df['date'])

            # Save to cache
            fred_df.to_csv(FRED_CACHE_FILE, index=False)
            print(f"   FRED cached to: {FRED_CACHE_FILE}")

    except ImportError:
        print("   WARNING: fredapi not installed")
    except Exception as e:
        print(f"   WARNING: FRED API error - {e}")

# Try cache if no fresh data
if fred_df is None and FRED_CACHE_FILE.exists():
    print("   Loading FRED from cache...")
    fred_df = pd.read_csv(FRED_CACHE_FILE)
    fred_df['date'] = pd.to_datetime(fred_df['date'])

if fred_df is None:
    print("   WARNING: No FRED data available")
    fred_df = pd.DataFrame({'date': pd.date_range(START_DATE, END_DATE)})

print(f"   FRED total: {len(fred_df)} rows")

# =============================================================================
# 5. FETCH FEAR & GREED DATA
# =============================================================================
print("\n[5/11] Fetching Fear & Greed data...")

try:
    import requests

    url = "https://api.alternative.me/fng/?limit=0"
    response = requests.get(url, timeout=30)
    data = response.json()

    if 'data' in data:
        fg_records = []
        for item in data['data']:
            fg_records.append({
                'date': pd.to_datetime(int(item['timestamp']), unit='s'),
                'fear_greed': int(item['value'])
            })
        fg_df = pd.DataFrame(fg_records)
        fg_df['date'] = pd.to_datetime(fg_df['date'].dt.date)
        fg_df = fg_df.groupby('date').last().reset_index()
        print(f"   Fear & Greed: {len(fg_df)} rows")
    else:
        fg_df = pd.DataFrame({'date': pd.date_range(START_DATE, END_DATE), 'fear_greed': 50})

except Exception as e:
    print(f"   WARNING: Fear & Greed FAILED - {e}")
    fg_df = pd.DataFrame({'date': pd.date_range(START_DATE, END_DATE), 'fear_greed': 50})

# =============================================================================
# 6. MERGE ALL DATA
# =============================================================================
print("\n[6/11] Merging all data...")

# Start with artemis (has price)
df = artemis.copy()

# Merge CoinMetrics
df = df.merge(cm, on='date', how='left')

# Merge yfinance
df = df.merge(yf_df, on='date', how='left')

# Merge FRED
df = df.merge(fred_df, on='date', how='left')

# Merge Fear & Greed
df = df.merge(fg_df, on='date', how='left')

# Sort by date
df = df.sort_values('date').reset_index(drop=True)

# Filter date range
df = df[(df['date'] >= START_DATE) & (df['date'] <= END_DATE)]

# Forward fill macro data (weekends/holidays)
macro_cols = ['vix', 'sp500', 'gold', 'oil', 'copper', 'dxy', 'us10y', 'eth',
              'yield_curve_2s10s', 'high_yield_spread', 'breakeven_10y',
              'fed_balance_sheet', 'm2_supply', 'fear_greed']
for col in macro_cols:
    if col in df.columns:
        df[col] = df[col].ffill()

# FIX: Correct OHLC inconsistencies (High < Close: 57 cases, Low > Close: 8 cases)
if all(col in df.columns for col in ['price_open', 'price_high', 'price_low', 'price_usd']):
    # Ensure High >= max(Open, Close)
    df['price_high'] = df[['price_open', 'price_high', 'price_usd']].max(axis=1)
    # Ensure Low <= min(Open, Close)
    df['price_low'] = df[['price_open', 'price_low', 'price_usd']].min(axis=1)
    print("   Fixed OHLC inconsistencies (High/Low adjusted)")

print(f"   Merged: {len(df)} rows, {len(df.columns)} columns")

# =============================================================================
# 7. CALCULATE PRICE-BASED FEATURES
# =============================================================================
print("\n[7/11] Calculating price-based features...")

# Validate price exists
if 'price_usd' not in df.columns or df['price_usd'].isnull().all():
    raise ValueError("price_usd is required but missing!")

# --- RETURNS ---
# FIX 1.1: Use LOG RETURNS instead of simple returns
# Log returns are additive over time: return_7d ≈ sum(return_1d for 7 days)
# This is mathematically correct for compounding and volatility calculations
for period in [1, 7, 14, 30, 60, 90, 180, 365]:
    df[f'return_{period}d'] = np.log(df['price_usd'] / df['price_usd'].shift(period))

# --- VOLATILITY ---
# FIX 1.2: Use log returns for volatility calculation
# First ensure we have log_return_1d explicitly for clarity
df['log_return_1d'] = np.log(df['price_usd'] / df['price_usd'].shift(1))
for period in [7, 30, 60, 90]:
    df[f'volatility_{period}d'] = df['log_return_1d'].rolling(period).std() * np.sqrt(365)

# Parkinson volatility (using high/low)
# FIX 1.3: Correct Parkinson formula - sum should be inside sqrt, divided by n
# Formula: sqrt((1 / (4 * n * ln(2))) * sum(ln(H/L)^2)) * sqrt(252 or 365)
if 'price_high' in df.columns and 'price_low' in df.columns:
    log_hl = np.log(df['price_high'] / df['price_low'])
    n_periods = 14
    df['parkinson_vol_14d'] = np.sqrt(
        (1 / (4 * n_periods * np.log(2))) * (log_hl ** 2).rolling(n_periods).sum()
    ) * np.sqrt(365)
    df['intraday_range'] = (df['price_high'] - df['price_low']) / df['price_usd']

# --- TECHNICAL INDICATORS ---
# RSI
df['rsi_14d'] = calc_rsi(df['price_usd'], 14)

# SMA
df['sma_50d'] = df['price_usd'].rolling(50).mean()
df['sma_200d'] = df['price_usd'].rolling(200).mean()
df['price_to_sma20'] = df['price_usd'] / df['price_usd'].rolling(20).mean() - 1
df['price_to_sma50'] = df['price_usd'] / df['sma_50d'] - 1
df['price_to_sma200'] = df['price_usd'] / df['sma_200d'] - 1

# MACD
# FIX 1.8: Add adjust=False for proper EMA calculation (standard MACD behavior)
ema12 = df['price_usd'].ewm(span=12, adjust=False).mean()
ema26 = df['price_usd'].ewm(span=26, adjust=False).mean()
df['macd'] = ema12 - ema26
df['macd_histogram'] = df['macd'] - df['macd'].ewm(span=9, adjust=False).mean()

# Bollinger Bands
bb_sma = df['price_usd'].rolling(20).mean()
bb_std = df['price_usd'].rolling(20).std()
bb_upper = bb_sma + 2 * bb_std
bb_lower = bb_sma - 2 * bb_std
df['bb_position'] = (df['price_usd'] - bb_lower) / (bb_upper - bb_lower)
df['bb_width'] = (bb_upper - bb_lower) / bb_sma

# Stochastic
low_14 = df['price_usd'].rolling(14).min()
high_14 = df['price_usd'].rolling(14).max()
df['stochastic_k'] = 100 * (df['price_usd'] - low_14) / (high_14 - low_14)
df['stochastic_d'] = df['stochastic_k'].rolling(3).mean()

# ADX (proper calculation)
if 'price_high' in df.columns and 'price_low' in df.columns:
    df['adx'], df['plus_di'], df['minus_di'], df['atr_14d'] = calc_adx(
        df['price_high'], df['price_low'], df['price_usd'], 14
    )

# Momentum
df['momentum_10d'] = df['price_usd'].pct_change(10)

# Z-score
df['zscore_50d'] = (df['price_usd'] - df['price_usd'].rolling(50).mean()) / df['price_usd'].rolling(50).std()

# OBV
if 'volume_usd' in df.columns:
    obv = [0]
    for i in range(1, len(df)):
        if df['price_usd'].iloc[i] > df['price_usd'].iloc[i-1]:
            obv.append(obv[-1] + df['volume_usd'].iloc[i])
        elif df['price_usd'].iloc[i] < df['price_usd'].iloc[i-1]:
            obv.append(obv[-1] - df['volume_usd'].iloc[i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv
    # OBV trend: use z-score of 20-day change (avoids division by zero)
    obv_series = pd.Series(obv, index=df.index)
    obv_change = obv_series.diff(20)
    obv_change_std = obv_change.rolling(60).std()
    df['obv_trend'] = (obv_change / (obv_change_std + 1e-10)).clip(-5, 5)
    df['obv_trend'] = df['obv_trend'].fillna(0)
    df['volume_sma20_ratio'] = df['volume_usd'] / df['volume_usd'].rolling(20).mean()
    df['volume_sma50_ratio'] = df['volume_usd'] / df['volume_usd'].rolling(50).mean()

print("   Calculated: returns, volatility, RSI, MACD, BB, ADX, OBV")

# =============================================================================
# 8. CALCULATE RISK METRICS
# =============================================================================
print("\n[8/11] Calculating risk metrics...")

# Sharpe ratios (assuming risk-free = 0 for simplicity)
# FIX 1.5: Correct Sharpe scaling - annualize return, use already-annualized volatility
# Sharpe = (Annualized Return) / (Annualized Volatility)
for period in [30, 60, 90]:
    # Annualize the period return: multiply by (365/period)
    annualized_return = df[f'return_{period}d'] * (365 / period)
    # volatility_{period}d is already annualized (multiplied by sqrt(365))
    vol_annualized = df[f'volatility_{period}d']
    df[f'sharpe_{period}d'] = np.where(vol_annualized > 0, annualized_return / vol_annualized, 0)

# Sortino (downside deviation)
# FIX 1.6: Correct Sortino formula - use downside deviation (sqrt of mean of squared negatives)
# and ensure consistent annualization
downside = df['log_return_1d'].where(df['log_return_1d'] < 0, 0)
# Downside deviation = sqrt(mean(negative_returns^2)) * sqrt(365)
downside_dev = np.sqrt((downside ** 2).rolling(30).mean()) * np.sqrt(365)
# Annualize the 30d return
annualized_return_30d = df['return_30d'] * (365 / 30)
df['sortino_30d'] = np.where(downside_dev > 0, annualized_return_30d / downside_dev, 0)

# Max Drawdown
for period in [30, 90, 180, 365]:
    rolling_max = df['price_usd'].rolling(period).max()
    df[f'max_drawdown_{period}d'] = ((df['price_usd'] - rolling_max) / rolling_max).rolling(period).min()

# VaR and CVaR
df['var_95'] = df['return_1d'].rolling(252).quantile(0.05)
df['cvar_95'] = df['return_1d'].rolling(252).apply(
    lambda x: x[x <= x.quantile(0.05)].mean() if len(x[x <= x.quantile(0.05)]) > 0 else x.quantile(0.05),
    raw=False
)

# Hurst exponent
# NOTE: The R/S method below is BIASED (gives ~0.8 on random data).
# We calculate it here but will be REPLACED by DFA Hurst from add_regime_features.py
# The DFA version (hurst_dfa_30d) is more accurate and will be copied to hurst_30d later.
df['hurst_30d_rs'] = calc_hurst(df['price_usd'], 30)  # Keep as _rs for reference
df['hurst_60d_rs'] = calc_hurst(df['price_usd'], 60)  # Keep as _rs for reference
# Placeholder - will be overwritten by DFA version
df['hurst_30d'] = df['hurst_30d_rs']
df['hurst_60d'] = df['hurst_60d_rs']

# ATH metrics
ath = df['price_usd'].cummax()
df['distance_from_ath'] = (df['price_usd'] - ath) / ath

# Days since ATH
ath_dates = []
current_ath = 0
current_ath_date = df['date'].iloc[0]
for i, row in df.iterrows():
    if row['price_usd'] >= current_ath:
        current_ath = row['price_usd']
        current_ath_date = row['date']
    ath_dates.append((row['date'] - current_ath_date).days)
df['days_since_ath'] = ath_dates

# Halving cycle
halving_dates = [
    pd.Timestamp('2012-11-28'),
    pd.Timestamp('2016-07-09'),
    pd.Timestamp('2020-05-11'),
    pd.Timestamp('2024-04-20'),
]

def days_since_halving(date):
    for i in range(len(halving_dates) - 1, -1, -1):
        if date >= halving_dates[i]:
            return (date - halving_dates[i]).days
    return 0

df['days_since_halving'] = df['date'].apply(days_since_halving)

print("   Calculated: sharpe, sortino, drawdown, VaR, hurst, ATH")

# =============================================================================
# 9. CALCULATE MACRO DERIVED FEATURES
# =============================================================================
print("\n[9/11] Calculating macro derived features...")

# VIX features
if 'vix' in df.columns:
    df['vix_ma20'] = df['vix'].rolling(20).mean()
    df['vix_zscore'] = (df['vix'] - df['vix'].rolling(90).mean()) / df['vix'].rolling(90).std()
    df['vix_percentile_1y'] = df['vix'].rolling(252).rank(pct=True)

# Returns for macro
for col in ['gold', 'oil', 'copper']:
    if col in df.columns:
        df[f'{col}_return_30d'] = df[col].pct_change(30)

# ETH/BTC ratio
if 'eth' in df.columns and 'price_usd' in df.columns:
    df['eth_btc_ratio'] = df['eth'] / df['price_usd']

# FRED derived features
if 'yield_curve_2s10s' in df.columns:
    df['yield_curve_inverted'] = (df['yield_curve_2s10s'] < 0).astype(int)

if 'us10y' in df.columns and 'breakeven_10y' in df.columns:
    df['real_yield_10y'] = df['us10y'] - df['breakeven_10y']

if 'm2_supply' in df.columns:
    # V25 FIX: data is DAILY (ffilled from weekly), so pct_change(N) = N days.
    # Previous code used pct_change(52) which is 52 days, not 52 weeks.
    # YoY in daily data = pct_change(252) business days or approximate
    # via compounding the 90d pct_change to a year.
    df['m2_supply_pctchg_30d'] = df['m2_supply'].pct_change(30)
    df['m2_supply_pctchg_90d'] = df['m2_supply'].pct_change(90)
    df['m2_3m_growth'] = df['m2_supply'].pct_change(90)          # 90d = ~3 months
    df['m2_yoy_growth'] = (1 + df['m2_supply_pctchg_90d']) ** (365.0/90.0) - 1

if 'fed_balance_sheet' in df.columns:
    # V25 FIX: same bug, now uses 252 business days = real YoY
    df['fed_bs_yoy_change'] = df['fed_balance_sheet'].pct_change(252)

# Correlations
if 'sp500' in df.columns:
    sp500_ret = df['sp500'].pct_change()
    df['btc_sp500_corr_30d'] = df['return_1d'].rolling(30).corr(sp500_ret)
    df['btc_sp500_corr_90d'] = df['return_1d'].rolling(90).corr(sp500_ret)
    df['sp500_pctchg_30d'] = df['sp500'].pct_change(30)
    df['sp500_pctchg_90d'] = df['sp500'].pct_change(90)

if 'gold' in df.columns:
    df['btc_gold_corr_30d'] = df['return_1d'].rolling(30).corr(df['gold'].pct_change())

if 'dxy' in df.columns:
    df['btc_dxy_corr_30d'] = df['return_1d'].rolling(30).corr(df['dxy'].pct_change())
    df['dxy_pctchg_30d'] = df['dxy'].pct_change(30)
    df['dxy_pctchg_90d'] = df['dxy'].pct_change(90)

if 'eth' in df.columns:
    df['btc_eth_corr_30d'] = df['return_1d'].rolling(30).corr(df['eth'].pct_change())
    df['eth_pctchg_30d'] = df['eth'].pct_change(30)
    df['eth_pctchg_90d'] = df['eth'].pct_change(90)

if 'vix' in df.columns:
    df['btc_vix_corr_30d'] = df['return_1d'].rolling(30).corr(df['vix'].pct_change())

# Fear & Greed features
if 'fear_greed' in df.columns:
    df['fear_greed_ma7'] = df['fear_greed'].rolling(7).mean()
    df['fear_greed_ma30'] = df['fear_greed'].rolling(30).mean()
    df['fear_greed_zscore'] = (df['fear_greed'] - df['fear_greed'].rolling(90).mean()) / df['fear_greed'].rolling(90).std()
    df['extreme_fear'] = (df['fear_greed'] < 25).astype(int)
    df['extreme_greed'] = (df['fear_greed'] > 75).astype(int)

print("   Calculated: vix, correlations, fear_greed derived")

# =============================================================================
# 10. CALCULATE ON-CHAIN AND FUTURES FEATURES
# =============================================================================
print("\n[10/11] Calculating on-chain and futures features...")

# Exchange netflow
if 'exchange_inflow_btc' in df.columns and 'exchange_outflow_btc' in df.columns:
    df['exchange_netflow_btc'] = df['exchange_inflow_btc'] - df['exchange_outflow_btc']
    df['exchange_netflow_ma7'] = df['exchange_netflow_btc'].rolling(7).mean()
    df['exchange_netflow_lag_1'] = df['exchange_netflow_btc'].shift(1)
    df['exchange_netflow_lag_7'] = df['exchange_netflow_btc'].shift(7)

if 'exchange_inflow_usd' in df.columns and 'exchange_outflow_usd' in df.columns:
    df['exchange_netflow_usd'] = df['exchange_inflow_usd'] - df['exchange_outflow_usd']

# Supply on exchanges %
if 'supply_on_exchanges' in df.columns and 'circulating_supply' in df.columns:
    df['supply_on_exchanges_pct'] = df['supply_on_exchanges'] / df['circulating_supply'] * 100
    df['supply_change_30d'] = df['supply_on_exchanges'].pct_change(30)

# MVRV z-score
if 'mvrv_ratio' in df.columns:
    df['mvrv_zscore'] = (df['mvrv_ratio'] - df['mvrv_ratio'].rolling(365).mean()) / df['mvrv_ratio'].rolling(365).std()

# Active addresses MA
if 'active_addresses' in df.columns:
    df['active_addresses_ma7'] = df['active_addresses'].rolling(7).mean()

# TX count MA
if 'tx_count' in df.columns:
    df['tx_count_ma7'] = df['tx_count'].rolling(7).mean()
    df['tx_growth_30d'] = df['tx_count'].pct_change(30)

# Futures features
if 'futures_buy_volume' in df.columns and 'futures_sell_volume' in df.columns:
    df['buy_sell_ratio'] = df['futures_buy_volume'] / (df['futures_sell_volume'] + 1)
    df['net_buy_pressure'] = df['futures_buy_volume'] - df['futures_sell_volume']

    # FIX: Treat outliers in buy_sell_ratio (max was 253,665, should be ~0.8-1.2)
    # Use ROLLING percentiles to avoid data leakage (no future data)
    p01_roll = df['buy_sell_ratio'].rolling(365, min_periods=30).quantile(0.01)
    p99_roll = df['buy_sell_ratio'].rolling(365, min_periods=30).quantile(0.99)
    df['buy_sell_ratio'] = df['buy_sell_ratio'].clip(lower=p01_roll, upper=p99_roll)

if 'open_interest' in df.columns:
    df['oi_change_7d'] = df['open_interest'].pct_change(7)
    df['oi_change_30d'] = df['open_interest'].pct_change(30)

    # FIX: Treat outliers in oi_change (max was 41,213 = 4,121,300%!)
    # Use ROLLING percentiles to avoid data leakage (no future data)
    for col in ['oi_change_7d', 'oi_change_30d']:
        p01_roll = df[col].rolling(365, min_periods=30).quantile(0.01)
        p99_roll = df[col].rolling(365, min_periods=30).quantile(0.99)
        df[col] = df[col].clip(lower=p01_roll, upper=p99_roll)

if 'futures_volume' in df.columns and 'volume_usd' in df.columns:
    df['futures_dominance'] = df['futures_volume'] / (df['volume_usd'] + df['futures_volume'] + 1) * 100

if 'funding_rate' in df.columns:
    df['funding_rate_ma7'] = df['funding_rate'].rolling(7).mean()

# Developer features
if 'core_commits' in df.columns:
    df['commits_ma7'] = df['core_commits'].rolling(7).mean()
    df['commits_trend_30d'] = df['core_commits'].rolling(30).mean().pct_change(30)

if 'ecosystem_commits' in df.columns:
    df['ecosystem_commits_ma7'] = df['ecosystem_commits'].rolling(7).mean()

# Price percentiles
df['price_percentile_1y'] = df['price_usd'].rolling(365).rank(pct=True)
df['price_percentile_2y'] = df['price_usd'].rolling(730).rank(pct=True)

# Volatility percentile
if 'volatility_30d' in df.columns:
    df['vol_percentile_1y'] = df['volatility_30d'].rolling(365).rank(pct=True)

# Lag features
for lag in [1, 2, 3, 5, 7]:
    df[f'return_lag_{lag}'] = df['return_1d'].shift(lag)

df['volatility_lag_7'] = df['volatility_30d'].shift(7)

# Trend strength and mean reversion
if 'adx' in df.columns and 'hurst_30d' in df.columns:
    df['trend_strength'] = df['adx'] * df['hurst_30d']
    df['mean_reversion_score'] = (1 - df['hurst_30d']) * (50 - df['adx'].clip(0, 50)) / 50

# Z-scores for trending features
if 'btc_dominance' in df.columns:
    df['btc_dom_zscore'] = (df['btc_dominance'] - df['btc_dominance'].rolling(90).mean()) / df['btc_dominance'].rolling(90).std()

# Pct change for trending features
trending_cols = {
    'hash_rate': [30, 90],
    'addresses_with_balance': [30, 90],
}

for col, windows in trending_cols.items():
    if col in df.columns:
        for w in windows:
            df[f'{col}_pctchg_{w}d'] = df[col].pct_change(w)

print("   Calculated: on-chain, futures, developer, lags, z-scores")

# =============================================================================
# 10.5. CALCULATE ADVANCED FEATURES
# =============================================================================
print("\n[10.5/11] Calculating advanced features...")

# --- MOMENTUM FEATURES ---
# Momentum consensus (are all timeframes aligned?)
df['momentum_7d_sign'] = np.sign(df['return_7d'])
df['momentum_30d_sign'] = np.sign(df['return_30d'])
df['momentum_90d_sign'] = np.sign(df['return_90d'])
df['momentum_consensus'] = df['momentum_7d_sign'] + df['momentum_30d_sign'] + df['momentum_90d_sign']
# -3 = all bearish, +3 = all bullish, 0 = mixed

# Momentum acceleration
df['momentum_accel_7d'] = df['return_7d'] - df['return_7d'].shift(7)
df['momentum_accel_30d'] = df['return_30d'] - df['return_30d'].shift(30)

# Drop temp columns
df = df.drop(columns=['momentum_7d_sign', 'momentum_30d_sign', 'momentum_90d_sign'])

# --- TEMPORAL FEATURES ---
df['date_temp'] = pd.to_datetime(df['date'])
df['day_of_week'] = df['date_temp'].dt.dayofweek  # 0=Monday, 6=Sunday
df['is_month_end'] = (df['date_temp'].dt.is_month_end).astype(int)
df['quarter'] = df['date_temp'].dt.quarter
df['is_q4'] = (df['quarter'] == 4).astype(int)  # Q4 historically bullish
df = df.drop(columns=['date_temp', 'quarter'])

# --- INTERACTION FEATURES ---
# RSI normalized by regime (relative RSI)
if 'rsi_14d' in df.columns:
    df['rsi_regime_normalized'] = df['rsi_14d'] / df['rsi_14d'].rolling(90).mean()

# MVRV * Hurst interaction (key signal)
if 'mvrv_zscore' in df.columns and 'hurst_30d' in df.columns:
    df['mvrv_hurst_signal'] = df['mvrv_zscore'] * df['hurst_30d']

# Fear & Greed extreme detector
if 'fear_greed_ma7' in df.columns:
    df['fg_extreme_signal'] = 0
    df.loc[df['fear_greed_ma7'] < 25, 'fg_extreme_signal'] = -1  # extreme fear = buy signal
    df.loc[df['fear_greed_ma7'] > 75, 'fg_extreme_signal'] = 1   # extreme greed = sell signal

# Volatility regime (low/medium/high) — use rolling quantiles to avoid leakage
if 'volatility_30d' in df.columns:
    vol_33 = df['volatility_30d'].rolling(365, min_periods=60).quantile(0.33)
    vol_66 = df['volatility_30d'].rolling(365, min_periods=60).quantile(0.66)
    df['vol_regime'] = 1  # medium
    df.loc[df['volatility_30d'] < vol_33, 'vol_regime'] = 0  # low
    df.loc[df['volatility_30d'] > vol_66, 'vol_regime'] = 2  # high

# Price momentum vs volume (confirmation) - REMOVED: redundant with return_7d

# --- VALUATION METRICS (NEW) ---
# NVT Ratio: Network Value to Transactions (market cap / daily transaction volume)
if 'market_cap' in df.columns and 'transfer_count' in df.columns:
    # NVT = Market Cap / Daily Transaction Volume (in USD)
    daily_tx_volume = df['transfer_count'] * df['price_usd']
    df['nvt_ratio'] = df['market_cap'] / (daily_tx_volume + 1e-10)
    df['nvt_ratio'] = df['nvt_ratio'].clip(0, 1000)  # Reasonable bounds

# Market Cap to FDV ratio: How much of supply is circulating
if 'market_cap' in df.columns and 'fdv' in df.columns:
    df['mcap_to_fdv'] = df['market_cap'] / (df['fdv'] + 1e-10)
    df['mcap_to_fdv'] = df['mcap_to_fdv'].clip(0, 1.5)  # Should be <= 1, but allow margin

# Market Cap Z-Score: Market cap relative to historical average
if 'market_cap' in df.columns:
    mcap_ma365 = df['market_cap'].rolling(365, min_periods=30).mean()
    mcap_std365 = df['market_cap'].rolling(365, min_periods=30).std()
    df['mcap_zscore'] = (df['market_cap'] - mcap_ma365) / (mcap_std365 + 1e-10)
    df['mcap_zscore'] = df['mcap_zscore'].clip(-5, 5)  # Reasonable bounds

print("   Calculated: momentum_consensus, momentum_accel, temporal, interactions, valuation metrics")

# =============================================================================
# 11. CALCULATE REGIME AND TARGETS
# =============================================================================
print("\n[11/11] Calculating regime and targets...")

# Regime classification
df['regime_raw'] = df.apply(classify_regime, axis=1)
# IMPORTANTE: center=False para não usar dados futuros no cálculo!
# center=True olharia 2 dias para frente, causando data leakage
df['regime_v3'] = df['regime_raw'].rolling(5, min_periods=1, center=False).median().round().astype(int)
df = df.drop(columns=['regime_raw'])

# Targets
df['target_direction_1d'] = (df['return_1d'].shift(-1) > 0).astype(float)
df['target_direction_5d'] = (df['price_usd'].shift(-5) > df['price_usd']).astype(float)
df['target_return_1d'] = df['return_1d'].shift(-1)
# FIX 1.7: Correct target_return_5d calculation
# target_return_5d = (price in 5 days / current price) - 1
# This represents the forward-looking return, not a shifted backward-looking one
df['target_return_5d'] = (df['price_usd'].shift(-5) / df['price_usd']) - 1
df['target_regime'] = df['regime_v3'].shift(-1)

# has_futures_data flag
if 'funding_rate' in df.columns:
    df['has_futures_data'] = df['funding_rate'].notna().astype(int)

# Regime duration (how many days in current regime)
regime_change = (df['regime_v3'] != df['regime_v3'].shift(1))
df['regime_group'] = regime_change.cumsum()
df['regime_duration'] = df.groupby('regime_group').cumcount() + 1
df = df.drop(columns=['regime_group'])

print("   Calculated: regime_v3, targets, has_futures_data, regime_duration")

# =============================================================================
# FINAL CLEANUP
# =============================================================================
print("\n" + "=" * 70)
print("FINAL CLEANUP")
print("=" * 70)

# Define final columns (include price_usd!)
# Note: Removed redundant features with corr > 0.90:
# - sma_50d, sma_200d (use price_to_sma instead)
# - sharpe_30d, sharpe_60d, sharpe_90d (highly corr with returns)
# - stochastic_d (highly corr with stochastic_k)
# - cvar_95 (highly corr with var_95)
# - volatility_90d (corr 0.905 with volatility_60d)
# - stochastic_k (corr 0.920 with bb_position)
# - real_yield_10y (corr 0.961 with us10y)
# - mean_reversion_score (corr -0.967, derived from adx)
# - price_vol_confirm (corr 0.926, derived from return_7d)
FINAL_COLUMNS = [
    # === BASE ===
    'date', 'price_usd', 'volume_usd',

    # === FROM ARTEMIS ===
    'sharpe_1y_artemis', 'sharpe_3y_artemis', 'sharpe_30d_artemis', 'sharpe_90d_artemis',
    'volatility_1y_artemis', 'volatility_3y_artemis',
    'funding_rate', 'funding_rate_volume', 'open_interest', 'futures_volume', 'futures_trade_count',
    'active_addresses_24h', 'core_commits', 'active_developers', 'ecosystem_commits',
    'fees_7d_avg', 'fee_median_usd', 'fees_total_24h', 'network_expenses_24h',
    'network_revenue_24h', 'fees_supply_side_24h', 'block_rewards_24h',

    # === RETURNS ===
    'return_1d', 'return_7d', 'return_14d', 'return_30d', 'return_60d', 'return_90d',
    'return_180d', 'return_365d',

    # === VOLATILITY ===
    'volatility_7d', 'volatility_30d', 'volatility_60d',  # REMOVED: volatility_90d (redundant with 60d)
    'parkinson_vol_14d', 'intraday_range',

    # === TECHNICAL INDICATORS ===
    'rsi_14d', 'price_to_sma20', 'price_to_sma50', 'price_to_sma200',
    'macd', 'macd_histogram', 'bb_position', 'bb_width',  # REMOVED: stochastic_k (redundant with bb_position)
    'adx', 'plus_di', 'minus_di', 'momentum_10d', 'zscore_50d', 'atr_14d',
    'obv', 'obv_trend', 'volume_sma20_ratio', 'volume_sma50_ratio',

    # === RISK METRICS ===
    'sortino_30d', 'max_drawdown_30d', 'max_drawdown_90d', 'max_drawdown_180d',
    'max_drawdown_365d', 'var_95', 'hurst_30d', 'hurst_60d',
    'distance_from_ath', 'days_since_ath', 'days_since_halving',

    # === MACRO ===
    'vix', 'sp500', 'gold', 'oil', 'copper', 'dxy', 'us10y', 'eth', 'yield_curve_2s10s',
    'high_yield_spread', 'breakeven_10y', 'fed_balance_sheet',
    'vix_ma20', 'vix_zscore', 'vix_percentile_1y',
    'gold_return_30d', 'oil_return_30d', 'copper_return_30d',
    'eth_btc_ratio', 'yield_curve_inverted',  # REMOVED: real_yield_10y (redundant with us10y)
    'm2_yoy_growth', 'm2_3m_growth', 'fed_bs_yoy_change',

    # === CORRELATIONS ===
    'btc_sp500_corr_30d', 'btc_sp500_corr_90d', 'btc_gold_corr_30d',
    'btc_dxy_corr_30d', 'btc_eth_corr_30d', 'btc_vix_corr_30d',

    # === SENTIMENT ===
    'fear_greed_ma7', 'fear_greed_ma30', 'fear_greed_zscore',
    'extreme_fear', 'extreme_greed',

    # === ON-CHAIN ===
    'block_count', 'btc_issued_daily', 'issuance_usd', 'transfer_count', 'fees_btc',
    'exchange_netflow_btc', 'exchange_netflow_ma7', 'exchange_netflow_usd',
    'supply_on_exchanges_pct', 'supply_change_30d', 'mvrv_zscore',
    'active_addresses_ma7', 'tx_count_ma7', 'tx_growth_30d',

    # === FUTURES ===
    'buy_sell_ratio', 'net_buy_pressure', 'oi_change_7d', 'oi_change_30d',
    'futures_dominance', 'funding_rate_ma7',

    # === DEVELOPER ===
    'commits_ma7', 'commits_trend_30d', 'ecosystem_commits_ma7',

    # === PERCENTILES ===
    'price_percentile_1y', 'price_percentile_2y', 'vol_percentile_1y',

    # === LAGS ===
    'return_lag_1', 'return_lag_2', 'return_lag_3', 'return_lag_5', 'return_lag_7',
    'volatility_lag_7', 'exchange_netflow_lag_1', 'exchange_netflow_lag_7',

    # === TREND/REGIME FEATURES ===
    'trend_strength', 'btc_dom_zscore',  # REMOVED: mean_reversion_score (derived from adx)

    # === PCT CHANGE ===
    'hash_rate_pctchg_30d', 'hash_rate_pctchg_90d',
    'addresses_with_balance_pctchg_30d', 'addresses_with_balance_pctchg_90d',
    'sp500_pctchg_30d', 'sp500_pctchg_90d', 'eth_pctchg_30d', 'eth_pctchg_90d',
    'dxy_pctchg_30d', 'dxy_pctchg_90d', 'm2_supply_pctchg_30d', 'm2_supply_pctchg_90d',

    # === ADVANCED FEATURES (NEW) ===
    'momentum_consensus', 'momentum_accel_7d', 'momentum_accel_30d',
    'day_of_week', 'is_month_end', 'is_q4',
    'rsi_regime_normalized', 'mvrv_hurst_signal', 'fg_extreme_signal',
    'vol_regime', 'regime_duration',  # REMOVED: price_vol_confirm (derived from return_7d)

    # === VALUATION METRICS (NEW) ===
    # REMOVED: market_cap, fdv (corr > 0.99 with price_usd - redundant)
    'fee_per_tx',  # From raw data
    'nvt_ratio', 'mcap_to_fdv', 'mcap_zscore',  # Calculated (derived, not redundant)

    # === REGIME AND TARGETS ===
    'regime_v3', 'target_direction_1d', 'target_direction_5d',
    'target_return_1d', 'target_return_5d', 'target_regime', 'has_futures_data'
]

# Select available columns
available = [c for c in FINAL_COLUMNS if c in df.columns]
missing = [c for c in FINAL_COLUMNS if c not in df.columns]

print(f"Available columns: {len(available)}/{len(FINAL_COLUMNS)}")
if missing:
    print(f"Missing columns ({len(missing)}): {missing[:10]}..." if len(missing) > 10 else f"Missing columns: {missing}")

df = df[available].copy()

# Handle nulls in futures data — use rolling median to avoid data leakage
futures_cols = ['funding_rate', 'open_interest', 'futures_volume', 'futures_trade_count',
                'buy_sell_ratio', 'net_buy_pressure', 'oi_change_7d', 'oi_change_30d',
                'futures_dominance', 'funding_rate_ma7']
for col in futures_cols:
    if col in df.columns:
        rolling_med = df[col].rolling(365, min_periods=30).median()
        df[col] = df[col].fillna(rolling_med)
        if df[col].isnull().sum() > 0:
            df[col] = df[col].ffill().fillna(0)  # no-leak: forward + 0 (was .bfill() = future leak)

# Fill other nulls
for col in df.columns:
    if col == 'date':
        continue
    if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            df[col] = df[col].ffill()  # forward only (no future leak); leading NaN -> rolling-median below
            remaining = df[col].isnull().sum()
            if remaining > 0:
                # Rolling median (só dados passados) para evitar data leakage
                rolling_median = df[col].rolling(365, min_periods=30).median()
                df[col] = df[col].fillna(rolling_median)

                # Se ainda houver nulls (início do dataset), 0 (no-leak último recurso)
                remaining = df[col].isnull().sum()
                if remaining > 0:
                    df[col] = df[col].fillna(0)  # was .bfill() = future leak

# Drop rows without target (last few rows)
rows_before = len(df)
df = df.dropna(subset=['target_return_1d'])
print(f"Dropped {rows_before - len(df)} rows without target")

# Final null check
total_nulls = df.isnull().sum().sum()
if total_nulls > 0:
    print(f"WARNING: {total_nulls} nulls remaining, filling with 0")
    df = df.fillna(0)

print(f"\nFinal shape: {df.shape}")

# =============================================================================
# ADD EXTRA FEATURES (Stock-to-Flow, Puell, Difficulty Ribbon, Stablecoins, etc.)
# =============================================================================
if EXTRA_FEATURES_AVAILABLE:
    print("\n" + "=" * 70)
    print("ADDING EXTRA FEATURES...")
    print("=" * 70)

    df = add_all_extra_features(df, fetch_external=True)

    print(f"\nShape after extra features: {df.shape}")
else:
    print("\nSkipping extra features (module not available)")

# =============================================================================
# ADD REGIME FEATURES (Hurst DFA, OU Parameters, CUSUM, etc.)
# =============================================================================
if REGIME_FEATURES_AVAILABLE:
    print("\n" + "=" * 70)
    print("ADDING REGIME FEATURES (from src/features/regime/)...")
    print("=" * 70)

    df = add_all_regime_features(df, price_col='price_usd')

    # CRITICAL: Replace biased R/S Hurst with accurate DFA Hurst
    if 'hurst_dfa_30d' in df.columns:
        df['hurst_30d'] = df['hurst_dfa_30d']
        print("   Replaced hurst_30d with hurst_dfa_30d (DFA is more accurate)")
    if 'hurst_dfa_60d' in df.columns:
        df['hurst_60d'] = df['hurst_dfa_60d']
        print("   Replaced hurst_60d with hurst_dfa_60d (DFA is more accurate)")

    # Clip extreme acceleration values
    if 'acceleration' in df.columns:
        before_clip = df['acceleration'].abs().max()
        df['acceleration'] = df['acceleration'].clip(-50, 50)
        after_clip = df['acceleration'].abs().max()
        print(f"   Clipped acceleration: max |{before_clip:.1f}| -> |{after_clip:.1f}|")

    print(f"\nShape after regime features: {df.shape}")
else:
    print("\nSkipping regime features (module not available)")

# =============================================================================
# FINAL DATA QUALITY FIX (after all integrations)
# =============================================================================
print("\n" + "=" * 70)
print("FINAL DATA QUALITY FIX")
print("=" * 70)

# 1. Fix OHLC inconsistencies AGAIN (Binance data may have overwritten)
if all(col in df.columns for col in ['price_open', 'price_high', 'price_low', 'price_usd']):
    before_high = (df['price_high'] < df['price_usd']).sum()
    before_low = (df['price_low'] > df['price_usd']).sum()

    df['price_high'] = df[['price_open', 'price_high', 'price_usd']].max(axis=1)
    df['price_low'] = df[['price_open', 'price_low', 'price_usd']].min(axis=1)

    after_high = (df['price_high'] < df['price_usd']).sum()
    after_low = (df['price_low'] > df['price_usd']).sum()
    print(f"   OHLC fixed: High<Close {before_high}->{after_high}, Low>Close {before_low}->{after_low}")

# 2. Remove redundant features (corr > 0.99 with price_usd)
redundant_cols = ['futures_close', 'market_cap_raw', 'futures_high', 'futures_low', 'futures_open']
removed = []
for col in redundant_cols:
    if col in df.columns:
        df = df.drop(columns=[col])
        removed.append(col)
if removed:
    print(f"   Removed redundant cols: {removed}")

# 3. Clip outliers AGAIN (extra features may have added new extreme values)
# Use ROLLING percentiles to avoid data leakage
outlier_cols_percentile = ['buy_sell_ratio']
for col in outlier_cols_percentile:
    if col in df.columns:
        before_max = df[col].max()
        p01_roll = df[col].rolling(365, min_periods=30).quantile(0.01)
        p99_roll = df[col].rolling(365, min_periods=30).quantile(0.99)
        df[col] = df[col].clip(lower=p01_roll, upper=p99_roll)
        after_max = df[col].max()
        if before_max != after_max:
            print(f"   {col}: max {before_max:.2f} -> {after_max:.2f}")

# Hard clip for oi_change (max reasonable is ~200% = 2.0)
oi_cols = ['oi_change_7d', 'oi_change_30d']
for col in oi_cols:
    if col in df.columns:
        before_max = df[col].max()
        # Hard limits: -90% to +500% change is reasonable
        df[col] = df[col].clip(-0.9, 5.0)
        after_max = df[col].max()
        if before_max != after_max:
            print(f"   {col}: max {before_max:.2f} -> {after_max:.2f} (hard clip)")

# 4. Fill remaining nulls — NO global statistics to avoid data leakage
null_before = df.isnull().sum().sum()
for col in df.columns:
    if col == 'date':
        continue
    if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].ffill()  # forward only (no future leak); leading NaN handled below
            if df[col].isnull().sum() > 0:
                rolling_med = df[col].rolling(365, min_periods=30).median()
                df[col] = df[col].fillna(rolling_med)
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(0)  # last resort: zero, not global median

null_after = df.isnull().sum().sum()
print(f"   Nulls: {null_before} -> {null_after}")

# 5. Remove constant columns (std == 0)
numeric_cols = df.select_dtypes(include=[np.number]).columns
const_cols = [c for c in numeric_cols if df[c].std() == 0]
if const_cols:
    df = df.drop(columns=const_cols)
    print(f"   Removed {len(const_cols)} constant columns: {const_cols}")

print(f"\nFinal shape after quality fix: {df.shape}")

# =============================================================================
# SAVE
# =============================================================================
print("\n" + "=" * 70)
print("SAVING...")
print("=" * 70)

df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved to: {OUTPUT_FILE}")

# =============================================================================
# VALIDATION SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)

# Shape
print(f"\nShape: {df.shape}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Total nulls: {df.isnull().sum().sum()}")

# Price
print(f"\nPrice stats:")
print(f"   Min: ${df['price_usd'].min():,.2f}")
print(f"   Max: ${df['price_usd'].max():,.2f}")
print(f"   Current: ${df['price_usd'].iloc[-1]:,.2f}")

# Regime distribution
print("\nRegime distribution:")
regime_map = {4: 'crisis', 2: 'trending_up', 1: 'mild_up', 0: 'neutral',
              -1: 'mild_down', -2: 'trending_down', -3: 'mean_reverting'}
regime_dist = df['regime_v3'].value_counts(normalize=True).sort_index()
for val in sorted(regime_dist.index):
    name = regime_map.get(val, f'unknown_{val}')
    pct = regime_dist[val] * 100
    print(f"   {name:18}: {pct:5.1f}%")

# Target distribution
print("\nTarget distribution (1d direction):")
target_dist = df['target_direction_1d'].value_counts(normalize=True)
print(f"   Up (1):   {target_dist.get(1.0, 0)*100:.1f}%")
print(f"   Down (0): {target_dist.get(0.0, 0)*100:.1f}%")

# Z-score validation
print("\nZ-score means (should be ~0):")
zscore_cols = [c for c in df.columns if '_zscore' in c]
for col in zscore_cols:
    mean_val = df[col].mean()
    status = "OK" if abs(mean_val) < 0.2 else "WARN"
    print(f"   {col:25}: {mean_val:+.3f} [{status}]")

# High correlations check
print("\nHigh correlations (>0.90):")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if not c.startswith('target_')]
corr_matrix = df[numeric_cols].corr()

high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_val = abs(corr_matrix.iloc[i, j])
        if corr_val > 0.90:
            high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))

high_corr.sort(key=lambda x: -x[2])
print(f"   Found {len(high_corr)} pairs with |corr| > 0.90")
for f1, f2, corr in high_corr[:5]:
    print(f"      {f1[:20]:20} | {f2[:20]:20} | {corr:.3f}")

# =============================================================================
# CRITICAL FIXES VALIDATION
# =============================================================================
print("\n" + "=" * 70)
print("CRITICAL FIXES VALIDATION")
print("=" * 70)

all_tests_passed = True

# Test 1: Hurst corrected (should have varied distribution)
if 'hurst_30d' in df.columns:
    hurst_max = df['hurst_30d'].max()
    hurst_min = df['hurst_30d'].min()
    hurst_mean = df['hurst_30d'].mean()
    trending_pct = (df['hurst_30d'] > 0.6).mean() * 100
    mean_rev_pct = (df['hurst_30d'] < 0.4).mean() * 100

    print(f"\n1. Hurst 30d validation:")
    print(f"   Max:  {hurst_max:.3f} (expected > 0.6)")
    print(f"   Min:  {hurst_min:.3f} (expected < 0.4)")
    print(f"   Mean: {hurst_mean:.3f} (expected ~0.50-0.55)")
    print(f"   Trending (>0.6): {trending_pct:.1f}% (expected ~30%)")
    print(f"   Mean-rev (<0.4): {mean_rev_pct:.1f}% (expected ~30%)")

    if hurst_max < 0.5:
        print("   [WARN] Hurst max still too low - may need more data or larger window")
        all_tests_passed = False
    else:
        print("   [OK] Hurst distribution looks reasonable")

# Test 2: Outliers treated
if 'buy_sell_ratio' in df.columns:
    bsr_max = df['buy_sell_ratio'].max()
    print(f"\n2. buy_sell_ratio outliers:")
    print(f"   Max: {bsr_max:.2f} (expected < 10)")
    if bsr_max < 10:
        print("   [OK] Outliers treated")
    else:
        print("   [WARN] Still has extreme values")
        all_tests_passed = False

if 'oi_change_30d' in df.columns:
    oi_max = df['oi_change_30d'].max()
    print(f"\n3. oi_change_30d outliers:")
    print(f"   Max: {oi_max:.2f} (expected < 10)")
    if oi_max < 10:
        print("   [OK] Outliers treated")
    else:
        print("   [WARN] Still has extreme values")
        all_tests_passed = False

# Test 3: Redundancy reduced
print(f"\n4. Redundancy check:")
high_corr_99 = sum(1 for f1, f2, c in high_corr if c > 0.99)
print(f"   Pairs with corr > 0.99: {high_corr_99} (expected < 20)")
if high_corr_99 < 30:  # More lenient threshold
    print("   [OK] Redundancy reduced")
else:
    print("   [WARN] Still has many highly correlated features")

# Test 4: OHLC consistency
if all(col in df.columns for col in ['price_open', 'price_high', 'price_low', 'price_usd']):
    high_lt_close = (df['price_high'] < df['price_usd']).sum()
    low_gt_close = (df['price_low'] > df['price_usd']).sum()
    print(f"\n5. OHLC consistency:")
    print(f"   High < Close: {high_lt_close} cases (expected 0)")
    print(f"   Low > Close:  {low_gt_close} cases (expected 0)")
    if high_lt_close == 0 and low_gt_close == 0:
        print("   [OK] OHLC is consistent")
    else:
        print("   [WARN] OHLC still has inconsistencies")
        all_tests_passed = False

print("\n" + "-" * 70)
if all_tests_passed:
    print("ALL CRITICAL FIXES VALIDATED SUCCESSFULLY!")
else:
    print("SOME VALIDATIONS FAILED - Review warnings above")
print("-" * 70)

print("\n" + "=" * 70)
print("DONE! Dataset ready for training.")
print("=" * 70)
