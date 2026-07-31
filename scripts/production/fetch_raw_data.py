"""
Fetch raw data from all sources needed for the 37 features.
Can do full historical fetch or incremental (append new days).

Usage:
    python scripts/production/fetch_raw_data.py                # incremental
    python scripts/production/fetch_raw_data.py --full          # full rebuild
    python scripts/production/fetch_raw_data.py --start 2019-01-01  # from date
"""
import sys, argparse, time, json, logging
import numpy as np, pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

# Load .env for API keys
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
import os

from scripts.production.config import DATASET_PATH, DATA_DIR, BQ_PROJECT, BQ_DATASET

RAW_CSV = Path(__file__).parent / "data" / "raw_data.csv"


def _date_to_ms(date_str: str) -> int:
    """UTC-midnight epoch (ms) for a 'YYYY-MM-DD' string.

    Uses pd.Timestamp(..., tz='UTC') so the boundary is true UTC, not the
    machine's local midnight. datetime.strptime(s).timestamp() interprets the
    naive datetime as LOCAL time, which shifts the kline window by the local
    offset (machine-timezone-dependent off-by-N-hours).
    """
    return int(pd.Timestamp(date_str, tz='UTC').timestamp() * 1000)


def _now_ms() -> int:
    """Current UTC epoch (ms), via time.time() (true UTC).

    Do NOT use datetime.utcnow().timestamp(): utcnow() returns a NAIVE datetime
    holding the UTC wall-clock, and .timestamp() reinterprets it as LOCAL time,
    shifting it by the local offset (e.g. +3h in UTC-3). That made the
    'drop unfinished candle' filter (close_time < now_ms) admit today's
    still-open candle when run in the local evening (UTC 21:00-24:00) — a
    silent look-ahead bug in the live signal path.
    """
    return int(time.time() * 1000)


# ═══════════════════════════════════════════════════════════════
# 1. BINANCE SPOT
# ═══════════════════════════════════════════════════════════════

def fetch_binance_spot(start: str, end: str) -> pd.DataFrame:
    """Fetch BTCUSDT daily spot OHLCV from Binance."""
    import requests
    log.info(f"Binance spot: {start} → {end}")
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    start_ms = _date_to_ms(start)
    end_ms = _date_to_ms(end)
    current = start_ms

    while current < end_ms:
        resp = requests.get(url, params={
            'symbol': 'BTCUSDT', 'interval': '1d',
            'startTime': current, 'endTime': end_ms, 'limit': 1000
        }, timeout=30)
        data = resp.json()
        if not data:
            break
        all_data.extend(data)
        if len(data) < 1000:
            break
        current = data[-1][0] + 86400000
        time.sleep(0.1)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trade_count',
        'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
    ])
    # Drop unfinished candles (close_time still in the future)
    now_ms = _now_ms()
    df = df[pd.to_numeric(df['close_time']) < now_ms].copy()
    df['date'] = pd.to_datetime(df['open_time'], unit='ms').dt.strftime('%Y-%m-%d')
    for c in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
        df[c] = pd.to_numeric(df[c])
    df = df.rename(columns={
        'close': 'price_usd', 'open': 'price_open',
        'high': 'price_high', 'low': 'price_low',
        'quote_volume': 'volume_usd',
    })
    log.info(f"  {len(df)} days")
    return df[['date', 'price_usd', 'price_open', 'price_high', 'price_low', 'volume_usd']].copy()


# ═══════════════════════════════════════════════════════════════
# 2. BINANCE FUTURES (for basis)
# ═══════════════════════════════════════════════════════════════

def fetch_binance_futures(start: str, end: str) -> pd.DataFrame:
    """Fetch BTCUSDT perpetual futures daily close."""
    import requests
    log.info(f"Binance futures: {start} → {end}")
    url = "https://fapi.binance.com/fapi/v1/klines"
    all_data = []
    start_ms = _date_to_ms(start)
    end_ms = _date_to_ms(end)
    current = start_ms

    while current < end_ms:
        resp = requests.get(url, params={
            'symbol': 'BTCUSDT', 'interval': '1d',
            'startTime': current, 'endTime': end_ms, 'limit': 1000
        }, timeout=30)
        data = resp.json()
        if not data:
            break
        all_data.extend(data)
        if len(data) < 1000:
            break
        current = data[-1][0] + 86400000
        time.sleep(0.1)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    # Drop unfinished candles (close_time still in the future)
    now_ms = _now_ms()
    df = df[pd.to_numeric(df[6]) < now_ms].copy()  # col 6 = close_time
    df['date'] = pd.to_datetime(df[0], unit='ms').dt.strftime('%Y-%m-%d')
    df['futures_close'] = pd.to_numeric(df[4])
    df['futures_volume_usd'] = pd.to_numeric(df[7])  # quote_volume for futures
    log.info(f"  {len(df)} days")
    return df[['date', 'futures_close', 'futures_volume_usd']].copy()


# ═══════════════════════════════════════════════════════════════
# 3. YFINANCE (ETH, Gold, Copper)
# ═══════════════════════════════════════════════════════════════

def fetch_yfinance(start: str, end: str) -> pd.DataFrame:
    """Fetch ETH, Gold, Copper from yfinance."""
    import yfinance as yf
    log.info(f"yfinance: {start} → {end}")
    tickers = {'ETH-USD': 'eth', 'GC=F': 'gold', 'HG=F': 'copper'}
    dfs = []
    for ticker, name in tickers.items():
        data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if not data.empty:
            s = data['Close'].reset_index()
            s.columns = ['date', name]
            s['date'] = pd.to_datetime(s['date']).dt.strftime('%Y-%m-%d')
            dfs.append(s)
    if not dfs:
        return pd.DataFrame()
    result = dfs[0]
    for d in dfs[1:]:
        result = result.merge(d, on='date', how='outer')
    log.info(f"  {len(result)} days")
    return result.sort_values('date').reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
# 4. FRED (M2, Fed Balance Sheet)
# ═══════════════════════════════════════════════════════════════

def fetch_fred(start: str, end: str) -> pd.DataFrame:
    """Fetch M2 and Fed Balance Sheet from FRED."""
    import os
    log.info(f"FRED: {start} → {end}")
    api_key = os.getenv("FRED_API_KEY", "").strip('"')
    if not api_key:
        log.warning("  FRED_API_KEY not set, skipping")
        return pd.DataFrame()

    import requests
    base = "https://api.stlouisfed.org/fred/series/observations"
    result = pd.DataFrame()

    for series_id, col_name in [('M2SL', 'm2_supply'), ('WALCL', 'fed_balance_sheet')]:
        resp = requests.get(base, params={
            'series_id': series_id, 'api_key': api_key,
            'file_type': 'json', 'observation_start': start, 'observation_end': end
        }, timeout=30)
        data = resp.json().get('observations', [])
        if data:
            df = pd.DataFrame(data)[['date', 'value']]
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.rename(columns={'value': col_name})
            if result.empty:
                result = df
            else:
                result = result.merge(df, on='date', how='outer')

    if not result.empty:
        result = result.sort_values('date').reset_index(drop=True)
        log.info(f"  {len(result)} observations")
    return result


# ═══════════════════════════════════════════════════════════════
# 5. BGEOMETRICS (NUPL)
# ═══════════════════════════════════════════════════════════════

def fetch_bgeometrics() -> pd.DataFrame:
    """Fetch NUPL from BGeometrics."""
    import requests
    log.info("BGeometrics: NUPL")
    try:
        resp = requests.get("https://bitcoin-data.com/v1/nupl", timeout=30)
        data = resp.json()
        if isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data)
            # API returns 'd' for date, 'nupl' for value
            if 'd' in df.columns:
                df = df.rename(columns={'d': 'date'})
            df['nupl'] = pd.to_numeric(df['nupl'], errors='coerce')
            df = df[['date', 'nupl']].dropna()
            log.info(f"  {len(df)} days")
            return df
    except Exception as e:
        log.warning(f"  BGeometrics failed: {e}")
    return pd.DataFrame()


def fetch_bitcoindata_metric(metric: str, col: str) -> pd.DataFrame:
    """Fetch on-chain metric from bitcoin-data.com (V36/E1 new feature source).
    Used for: reserve-risk, puell-multiple."""
    import requests, time
    log.info(f"bitcoin-data.com: {metric}")
    for attempt in range(3):
        try:
            resp = requests.get(f"https://bitcoin-data.com/v1/{metric}", timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    df = pd.DataFrame(data).rename(columns={'d': 'date'})
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df = df[['date', col]].dropna()
                    log.info(f"  {len(df)} days")
                    return df
            elif resp.status_code == 429:
                time.sleep(15)  # rate limit
        except Exception as e:
            log.warning(f"  attempt {attempt+1}: {e}")
            time.sleep(5)
    return pd.DataFrame()


def fetch_binance_funding_rate(start: str, end: str) -> pd.DataFrame:
    """Fetch Binance BTCUSDT funding rate (3 bars/day, aggregated to daily mean)."""
    import requests, time
    log.info("Binance futures: funding rate BTCUSDT")
    all_data = []
    try:
        start_ts = _date_to_ms(start)
        end_ts = _date_to_ms(end)
        cur = start_ts
        while cur < end_ts:
            resp = requests.get(
                f"https://fapi.binance.com/fapi/v1/fundingRate"
                f"?symbol=BTCUSDT&startTime={cur}&limit=1000",
                timeout=30)
            data = resp.json()
            if not data: break
            all_data.extend(data)
            last_t = data[-1]['fundingTime']
            if last_t <= cur: break
            cur = last_t + 1
            time.sleep(0.2)
        if not all_data: return pd.DataFrame()
        df = pd.DataFrame(all_data)
        df['date'] = pd.to_datetime(df['fundingTime'], unit='ms').dt.strftime('%Y-%m-%d')
        df['funding_rate'] = df['fundingRate'].astype(float)
        daily = df.groupby('date')['funding_rate'].mean().reset_index()
        daily.columns = ['date', 'funding_rate_mean']
        log.info(f"  {len(daily)} days (aggregated from {len(all_data)} 8h bars)")
        return daily
    except Exception as e:
        log.warning(f"  Binance funding failed: {e}")
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
# 6. BLOCKCHAIN.COM (miners revenue)
# ═══════════════════════════════════════════════════════════════

def fetch_blockchain_miners() -> pd.DataFrame:
    """Fetch miners revenue from Blockchain.com."""
    import requests
    log.info("Blockchain.com: miners revenue")
    try:
        resp = requests.get(
            "https://api.blockchain.info/charts/miners-revenue",
            params={'timespan': '8years', 'format': 'json', 'rollingAverage': '1days'},
            timeout=30
        )
        data = resp.json().get('values', [])
        if data:
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['x'], unit='s').dt.strftime('%Y-%m-%d')
            df['miners_revenue_usd'] = df['y']
            log.info(f"  {len(df)} days")
            return df[['date', 'miners_revenue_usd']].copy()
    except Exception as e:
        log.warning(f"  Blockchain.com failed: {e}")
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
# 7. DEFILLAMA (stablecoin supply)
# ═══════════════════════════════════════════════════════════════

def fetch_defillama_stablecoins() -> pd.DataFrame:
    """Fetch total stablecoin supply from DefiLlama (ALL stablecoins, not just USDT)."""
    import requests
    log.info("DefiLlama: stablecoin supply")
    try:
        # No stablecoin filter — fetch ALL stablecoins to match add_extra_features.py
        resp = requests.get(
            "https://stablecoins.llama.fi/stablecoincharts/all",
            timeout=30
        )
        data = resp.json()
        if isinstance(data, list):
            rows = []
            for entry in data:
                ts = int(float(entry['date']))
                date = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')
                # totalCirculating.peggedUSD is the total supply
                tc = entry.get('totalCirculating', {})
                total = float(tc.get('peggedUSD', 0)) if isinstance(tc, dict) else 0
                rows.append({'date': date, 'stablecoin_supply': total})
            df = pd.DataFrame(rows)
            df = df[df['stablecoin_supply'] > 0]
            log.info(f"  {len(df)} days")
            return df
    except Exception as e:
        log.warning(f"  DefiLlama failed: {e}")
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
# 8. COINMETRICS (hash rate)
# ═══════════════════════════════════════════════════════════════

def fetch_coinmetrics_hashrate(start: str, end: str) -> pd.DataFrame:
    """Fetch BTC hash rate from CoinMetrics community API."""
    import requests
    log.info(f"CoinMetrics: hash rate {start} → {end}")
    try:
        resp = requests.get(
            "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics",
            params={
                'assets': 'btc', 'metrics': 'HashRate',
                'start_time': start, 'end_time': end,
                'frequency': '1d', 'page_size': 10000,
            },
            timeout=30
        )
        data = resp.json().get('data', [])
        if data:
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d')
            df['hash_rate'] = pd.to_numeric(df['HashRate'], errors='coerce')
            log.info(f"  {len(df)} days")
            return df[['date', 'hash_rate']].copy()
    except Exception as e:
        log.warning(f"  CoinMetrics failed: {e}")
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
# 9. BIGQUERY MESSARI (OI + futures trade count)
# ═══════════════════════════════════════════════════════════════

def _run_bq_query(bq_path, name, query):
    """Run a single BQ query and return DataFrame."""
    import subprocess
    try:
        result = subprocess.run(
            [bq_path, 'query', '--format=json', '--max_rows=100000', query],
            capture_output=True, text=True, timeout=120, shell=(sys.platform == 'win32')
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            df = pd.DataFrame(data)
            for col in df.columns:
                if col != 'date':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            log.info(f"  {name}: {len(df)} rows")
            return df
        else:
            log.warning(f"  {name} BQ error: {result.stderr[:200]}")
    except Exception as e:
        log.warning(f"  {name} failed: {e}")
    return pd.DataFrame()


def fetch_bigquery_messari(start: str, end: str) -> pd.DataFrame:
    """Fetch OI, futures_trade_count, volume, and miners_revenue from BigQuery Messari."""
    import shutil
    log.info(f"BigQuery Messari: {start} → {end}")

    bq_path = shutil.which('bq')
    if not bq_path:
        log.warning("  bq CLI not found in PATH")
        return pd.DataFrame()

    queries = {
        'OI': (
            f"SELECT reference_date as date, CAST(open_interest AS FLOAT64) as open_interest "
            f"FROM `{BQ_PROJECT}.{BQ_DATASET}.messari_asset_futures_open_interest` "
            f"WHERE slug = 'bitcoin' AND reference_date >= '{start}' AND reference_date <= '{end}' "
            f"ORDER BY reference_date"
        ),
        'FTC': (
            f"SELECT reference_date as date, CAST(futures_trade_count AS INT64) as futures_trade_count "
            f"FROM `{BQ_PROJECT}.{BQ_DATASET}.messari_asset_futures_volume` "
            f"WHERE slug = 'bitcoin' AND reference_date >= '{start}' AND reference_date <= '{end}' "
            f"ORDER BY reference_date"
        ),
        'VOL': (
            f"SELECT reference_date as date, CAST(volume_usd AS FLOAT64) as volume_usd_messari "
            f"FROM `{BQ_PROJECT}.{BQ_DATASET}.messari_asset_ohlcv_and_market_cap` "
            f"WHERE symbol = 'BTC' AND reference_date >= '{start}' AND reference_date <= '{end}' "
            f"ORDER BY reference_date LIMIT 10000"
        ),
        'MINERS': (
            f"SELECT reference_date as date, "
            f"CAST(revenue_usd AS FLOAT64) + CAST(token_incentives_usd AS FLOAT64) as miners_revenue_messari "
            f"FROM `{BQ_PROJECT}.{BQ_DATASET}.messari_network_financial` "
            f"WHERE symbol = 'BTC' AND reference_date >= '{start}' AND reference_date <= '{end}' "
            f"ORDER BY reference_date LIMIT 10000"
        ),
    }

    # Run all queries and collect results per-column
    all_cols = {}
    for name, query in queries.items():
        result_df = _run_bq_query(bq_path, name, query)
        if not result_df.empty and 'date' in result_df.columns:
            # Deduplicate by date (keep last)
            result_df = result_df.drop_duplicates(subset='date', keep='last')
            for col in result_df.columns:
                if col != 'date':
                    all_cols[col] = result_df[['date', col]].copy()

    if not all_cols:
        return pd.DataFrame()

    # Start from first result and left-join others
    merged = list(all_cols.values())[0]
    for col_df in list(all_cols.values())[1:]:
        merged = merged.merge(col_df, on='date', how='outer')

    return merged


# ═══════════════════════════════════════════════════════════════
# MAIN: fetch all and merge
# ═══════════════════════════════════════════════════════════════

def fetch_all(start: str, end: str) -> pd.DataFrame:
    """Fetch all raw data and merge on date."""
    log.info(f"\n{'='*60}\nFETCH ALL RAW DATA: {start} → {end}\n{'='*60}\n")

    # Base: Binance spot (has every calendar day)
    df = fetch_binance_spot(start, end)
    if df.empty:
        log.info("Binance spot returned 0 closed candles — nothing new to fetch yet")
        return pd.DataFrame()

    # Merge each source
    futures_start = max(start, "2019-09-01")

    for name, fetcher, kwargs in [
        ("Binance futures", fetch_binance_futures, {'start': futures_start, 'end': end}),
        ("yfinance", fetch_yfinance, {'start': start, 'end': end}),
        ("FRED", fetch_fred, {'start': start, 'end': end}),
        ("BGeometrics", fetch_bgeometrics, {}),
        ("Blockchain.com", fetch_blockchain_miners, {}),
        ("DefiLlama", fetch_defillama_stablecoins, {}),
        ("CoinMetrics", fetch_coinmetrics_hashrate, {'start': start, 'end': end}),
        ("BigQuery Messari", fetch_bigquery_messari, {'start': start, 'end': end}),
        # V36/E1 new features
        ("bitcoin-data reserve-risk", fetch_bitcoindata_metric, {'metric': 'reserve-risk', 'col': 'reserveRisk'}),
        ("bitcoin-data puell-multiple", fetch_bitcoindata_metric, {'metric': 'puell-multiple', 'col': 'puellMultiple'}),
        ("Binance funding rate", fetch_binance_funding_rate, {'start': start, 'end': end}),
    ]:
        try:
            source_df = fetcher(**kwargs)
            if not source_df.empty:
                df = df.merge(source_df, on='date', how='left')
                log.info(f"  ✓ {name}: merged ({len(source_df)} rows)")
            else:
                log.warning(f"  ✗ {name}: empty")
        except Exception as e:
            log.warning(f"  ✗ {name}: {e}")

    # Forward-fill weekly data (FRED)
    for col in ['m2_supply', 'fed_balance_sheet']:
        if col in df.columns:
            df[col] = df[col].ffill()

    # Forward-fill daily data gaps (weekends for yfinance)
    for col in ['eth', 'gold', 'copper']:
        if col in df.columns:
            df[col] = df[col].ffill()

    # Fill NaN for pre-futures era
    if 'futures_close' in df.columns:
        df['futures_close'] = df['futures_close'].fillna(df['price_usd'])

    # Use BQ Messari volume (aggregated, matches Artemis) instead of Binance spot
    if 'volume_usd_messari' in df.columns:
        df['volume_usd'] = df['volume_usd_messari'].fillna(df['volume_usd'])
        df = df.drop(columns=['volume_usd_messari'])
        log.info("  Volume: using Messari aggregate (matches Artemis)")

    # Use BQ Messari miners revenue (more complete than Blockchain.com)
    if 'miners_revenue_messari' in df.columns:
        df['miners_revenue_usd'] = df['miners_revenue_messari'].fillna(df.get('miners_revenue_usd', pd.Series()))
        df = df.drop(columns=['miners_revenue_messari'])
        log.info("  Miners revenue: using Messari (fallback: Blockchain.com)")

    df = df.sort_values('date').reset_index(drop=True)
    log.info(f"\nFinal: {len(df)} rows, {len(df.columns)} columns")
    return df


def main():
    parser = argparse.ArgumentParser(description='Fetch raw data for V22 production')
    parser.add_argument('--start', default='2019-01-01', help='Start date')
    parser.add_argument('--end', default=None, help='End date (default: today)')
    parser.add_argument('--full', action='store_true', help='Full rebuild from start')
    args = parser.parse_args()

    # end is exclusive — use tomorrow (UTC) so we can pick up today's candle if it already closed
    end = args.end or (datetime.now(timezone.utc) + timedelta(days=1)).strftime('%Y-%m-%d')

    if not args.full and RAW_CSV.exists():
        existing = pd.read_csv(RAW_CSV)
        last_date = existing['date'].max()
        start = (pd.to_datetime(last_date) + timedelta(days=1)).strftime('%Y-%m-%d')
        if start > end:
            log.info("Raw data is up to date.")
            return
        log.info(f"Incremental update: {start} → {end}")
        new_data = fetch_all(start, end)
        if not new_data.empty:
            combined = pd.concat([existing, new_data], ignore_index=True)
            combined = combined.drop_duplicates(subset='date', keep='last')
            combined = combined.sort_values('date').reset_index(drop=True)
            combined.to_csv(RAW_CSV, index=False)
            log.info(f"Saved: {RAW_CSV} ({len(combined)} rows)")
    else:
        df = fetch_all(args.start, end)
        if not df.empty:
            df.to_csv(RAW_CSV, index=False)
            log.info(f"Saved: {RAW_CSV} ({len(df)} rows)")


if __name__ == '__main__':
    main()
