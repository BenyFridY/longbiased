"""
Binance Data Fetcher
====================
Busca dados históricos da Binance API (GRÁTIS, sem API key)

Dados coletados:
- Spot OHLCV (Open, High, Low, Close, Volume)
- Trade count, Taker buy volumes
- Futures OHLCV (para calcular basis)
- Funding Rate
- Long/Short Ratio (Global e Top Traders)
- Taker Buy/Sell Ratio

Features derivadas calculadas:
- basis_pct (futures premium)
- true_range, atr
- taker_buy_ratio
- gap_pct
- candle_body_ratio, shadow_ratios

Autor: Beny Frid
Data: 2026-01
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import time
import logging
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BinanceDataFetcher:
    """
    Classe para buscar dados da Binance API

    Endpoints utilizados (todos públicos, sem API key):
    - Spot: /api/v3/klines
    - Futures: /fapi/v1/klines
    - Funding: /fapi/v1/fundingRate
    - Long/Short: /futures/data/globalLongShortAccountRatio
    - Top Traders: /futures/data/topLongShortPositionRatio
    - Taker Ratio: /futures/data/takerlongshortRatio
    - OI History: /futures/data/openInterestHist
    """

    SPOT_BASE = "https://api.binance.com"
    FUTURES_BASE = "https://fapi.binance.com"

    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.session = requests.Session()

    def _request(self, url: str, params: dict, max_retries: int = 3) -> Optional[List]:
        """Faz request com retry e rate limit handling"""
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=30)

                if response.status_code == 429:  # Rate limited
                    wait_time = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        return None

    def _ms_to_date(self, ms: int) -> str:
        """Converte timestamp ms para date string"""
        return datetime.utcfromtimestamp(ms / 1000).strftime('%Y-%m-%d')

    def _date_to_ms(self, date_str: str) -> int:
        """Converte date string para timestamp ms"""
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        return int(dt.timestamp() * 1000)

    def fetch_spot_klines(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Busca candles diários do mercado spot

        Retorna: open, high, low, close, volume, quote_volume,
                 trade_count, taker_buy_volume, taker_buy_quote_volume
        """
        logger.info(f"Fetching spot klines: {start_date} to {end_date}")

        url = f"{self.SPOT_BASE}/api/v3/klines"
        all_data = []

        start_ms = self._date_to_ms(start_date)
        end_ms = self._date_to_ms(end_date)

        current_start = start_ms

        while current_start < end_ms:
            params = {
                'symbol': self.symbol,
                'interval': '1d',
                'startTime': current_start,
                'endTime': end_ms,
                'limit': 1000  # Max allowed
            }

            data = self._request(url, params)

            if not data:
                break

            all_data.extend(data)

            if len(data) < 1000:
                break

            # Next batch starts after last candle
            current_start = data[-1][0] + 86400000  # +1 day in ms

            time.sleep(0.1)  # Be nice to the API

        if not all_data:
            logger.error("No spot klines data received")
            return pd.DataFrame()

        # Parse data
        df = pd.DataFrame(all_data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trade_count',
            'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
        ])

        # Convert types
        df['date'] = df['open_time'].apply(self._ms_to_date)

        numeric_cols = ['open', 'high', 'low', 'close', 'volume',
                       'quote_volume', 'trade_count',
                       'taker_buy_volume', 'taker_buy_quote_volume']

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Rename with prefix
        df = df[['date'] + numeric_cols].copy()
        df.columns = ['date', 'spot_open', 'spot_high', 'spot_low', 'spot_close',
                     'spot_volume_btc', 'spot_volume_usd', 'spot_trade_count',
                     'spot_taker_buy_btc', 'spot_taker_buy_usd']

        logger.info(f"  Fetched {len(df)} spot candles")
        return df

    def fetch_futures_klines(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Busca candles diários do mercado de futuros perpetual
        Usado para calcular basis (futures - spot)
        """
        logger.info(f"Fetching futures klines: {start_date} to {end_date}")

        url = f"{self.FUTURES_BASE}/fapi/v1/klines"
        all_data = []

        start_ms = self._date_to_ms(start_date)
        end_ms = self._date_to_ms(end_date)

        current_start = start_ms

        while current_start < end_ms:
            params = {
                'symbol': self.symbol,
                'interval': '1d',
                'startTime': current_start,
                'endTime': end_ms,
                'limit': 1000
            }

            data = self._request(url, params)

            if not data:
                break

            all_data.extend(data)

            if len(data) < 1000:
                break

            current_start = data[-1][0] + 86400000
            time.sleep(0.1)

        if not all_data:
            logger.warning("No futures klines data received")
            return pd.DataFrame()

        df = pd.DataFrame(all_data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trade_count',
            'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
        ])

        df['date'] = df['open_time'].apply(self._ms_to_date)

        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'quote_volume']].copy()
        df.columns = ['date', 'futures_open', 'futures_high', 'futures_low',
                     'futures_close', 'futures_volume_btc', 'futures_volume_usd']

        logger.info(f"  Fetched {len(df)} futures candles")
        return df

    def fetch_funding_rates(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Busca funding rates (a cada 8h) e agrega para diário
        """
        logger.info(f"Fetching funding rates: {start_date} to {end_date}")

        url = f"{self.FUTURES_BASE}/fapi/v1/fundingRate"
        all_data = []

        start_ms = self._date_to_ms(start_date)
        end_ms = self._date_to_ms(end_date)

        current_start = start_ms

        while current_start < end_ms:
            params = {
                'symbol': self.symbol,
                'startTime': current_start,
                'endTime': end_ms,
                'limit': 1000
            }

            data = self._request(url, params)

            if not data:
                break

            all_data.extend(data)

            if len(data) < 1000:
                break

            current_start = data[-1]['fundingTime'] + 1
            time.sleep(0.1)

        if not all_data:
            logger.warning("No funding rate data received")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df['date'] = df['fundingTime'].apply(self._ms_to_date)
        df['fundingRate'] = pd.to_numeric(df['fundingRate'], errors='coerce')

        # Aggregate to daily (sum of 3 funding periods)
        daily = df.groupby('date').agg({
            'fundingRate': ['sum', 'mean', 'max', 'min', 'count']
        }).reset_index()

        daily.columns = ['date', 'binance_funding_daily', 'binance_funding_mean',
                        'binance_funding_max', 'binance_funding_min', 'funding_count']

        logger.info(f"  Fetched {len(daily)} days of funding rates")
        return daily

    def fetch_long_short_ratio(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Busca Global Long/Short Account Ratio
        NOTA: Este endpoint só disponibiliza ~30 dias de histórico!
        Útil para produção diária, não para backtest histórico.
        """
        logger.info(f"Fetching long/short ratio (NOTE: only ~30 days available)")

        url = f"{self.FUTURES_BASE}/futures/data/globalLongShortAccountRatio"

        # Este endpoint só retorna os últimos N registros (max 500)
        params = {
            'symbol': self.symbol,
            'period': '1d',
            'limit': 500  # Max available
        }

        data = self._request(url, params)

        if not data:
            logger.warning("No long/short ratio data received")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df['date'] = df['timestamp'].apply(self._ms_to_date)

        df['longShortRatio'] = pd.to_numeric(df['longShortRatio'], errors='coerce')
        df['longAccount'] = pd.to_numeric(df['longAccount'], errors='coerce')
        df['shortAccount'] = pd.to_numeric(df['shortAccount'], errors='coerce')

        df = df[['date', 'longShortRatio', 'longAccount', 'shortAccount']].copy()
        df.columns = ['date', 'global_ls_ratio', 'global_long_pct', 'global_short_pct']

        # Filtra pelo período solicitado
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

        # Remove duplicatas (manter última)
        df = df.drop_duplicates(subset=['date'], keep='last')

        logger.info(f"  Fetched {len(df)} days of L/S ratio")
        return df

    def fetch_top_traders_ratio(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Busca Top Traders Long/Short Position Ratio
        NOTA: Este endpoint só disponibiliza ~30 dias de histórico!
        """
        logger.info(f"Fetching top traders ratio (NOTE: only ~30 days available)")

        url = f"{self.FUTURES_BASE}/futures/data/topLongShortPositionRatio"

        params = {
            'symbol': self.symbol,
            'period': '1d',
            'limit': 500
        }

        data = self._request(url, params)

        if not data:
            logger.warning("No top traders ratio data received")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df['date'] = df['timestamp'].apply(self._ms_to_date)

        df['longShortRatio'] = pd.to_numeric(df['longShortRatio'], errors='coerce')
        df['longAccount'] = pd.to_numeric(df['longAccount'], errors='coerce')
        df['shortAccount'] = pd.to_numeric(df['shortAccount'], errors='coerce')

        df = df[['date', 'longShortRatio', 'longAccount', 'shortAccount']].copy()
        df.columns = ['date', 'top_traders_ls_ratio', 'top_traders_long_pct', 'top_traders_short_pct']

        # Filtra pelo período solicitado
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

        df = df.drop_duplicates(subset=['date'], keep='last')

        logger.info(f"  Fetched {len(df)} days of top traders ratio")
        return df

    def fetch_taker_buysell_ratio(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Busca Taker Buy/Sell Ratio (futures)
        NOTA: Este endpoint só disponibiliza ~30 dias de histórico!
        """
        logger.info(f"Fetching taker buy/sell ratio (NOTE: only ~30 days available)")

        url = f"{self.FUTURES_BASE}/futures/data/takerlongshortRatio"

        params = {
            'symbol': self.symbol,
            'period': '1d',
            'limit': 500
        }

        data = self._request(url, params)

        if not data:
            logger.warning("No taker ratio data received")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df['date'] = df['timestamp'].apply(self._ms_to_date)

        df['buySellRatio'] = pd.to_numeric(df['buySellRatio'], errors='coerce')
        df['buyVol'] = pd.to_numeric(df['buyVol'], errors='coerce')
        df['sellVol'] = pd.to_numeric(df['sellVol'], errors='coerce')

        df = df[['date', 'buySellRatio', 'buyVol', 'sellVol']].copy()
        df.columns = ['date', 'futures_taker_ratio', 'futures_taker_buy_vol', 'futures_taker_sell_vol']

        # Filtra pelo período solicitado
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

        df = df.drop_duplicates(subset=['date'], keep='last')

        logger.info(f"  Fetched {len(df)} days of taker ratio")
        return df

    def fetch_oi_history(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Busca Open Interest histórico
        NOTA: Este endpoint só disponibiliza ~30 dias de histórico!
        """
        logger.info(f"Fetching OI history (NOTE: only ~30 days available)")

        url = f"{self.FUTURES_BASE}/futures/data/openInterestHist"

        params = {
            'symbol': self.symbol,
            'period': '1d',
            'limit': 500
        }

        data = self._request(url, params)

        if not data:
            logger.warning("No OI history data received")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df['date'] = df['timestamp'].apply(self._ms_to_date)

        df['sumOpenInterest'] = pd.to_numeric(df['sumOpenInterest'], errors='coerce')
        df['sumOpenInterestValue'] = pd.to_numeric(df['sumOpenInterestValue'], errors='coerce')

        df = df[['date', 'sumOpenInterest', 'sumOpenInterestValue']].copy()
        df.columns = ['date', 'binance_oi_btc', 'binance_oi_usd']

        # Filtra pelo período solicitado
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

        df = df.drop_duplicates(subset=['date'], keep='last')

        logger.info(f"  Fetched {len(df)} days of OI history")
        return df

    def calculate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula features derivadas a partir dos dados brutos
        """
        logger.info("Calculating derived features...")

        df = df.copy()

        # === OHLC-based features ===

        # Previous close for gap calculation
        df['prev_close'] = df['spot_close'].shift(1)

        # Gap (abertura vs fechamento anterior)
        df['gap_pct'] = ((df['spot_open'] - df['prev_close']) / df['prev_close'] * 100)

        # True Range (considera gaps)
        df['true_range'] = np.maximum(
            df['spot_high'] - df['spot_low'],
            np.maximum(
                np.abs(df['spot_high'] - df['prev_close']),
                np.abs(df['spot_low'] - df['prev_close'])
            )
        )

        # ATR (14 períodos)
        df['atr_14'] = df['true_range'].rolling(14).mean()

        # Normalized ATR (% do preço)
        df['atr_pct'] = df['atr_14'] / df['spot_close'] * 100

        # Intraday range
        df['intraday_range_pct'] = (df['spot_high'] - df['spot_low']) / df['spot_close'] * 100

        # === Candlestick patterns ===

        # Body size
        body = np.abs(df['spot_close'] - df['spot_open'])
        hl_range = df['spot_high'] - df['spot_low']

        # Avoid division by zero
        hl_range_safe = hl_range.replace(0, np.nan)

        df['candle_body_ratio'] = body / hl_range_safe

        # Upper shadow (rejeição de alta)
        df['upper_shadow_ratio'] = (df['spot_high'] - np.maximum(df['spot_open'], df['spot_close'])) / hl_range_safe

        # Lower shadow (rejeição de baixa)
        df['lower_shadow_ratio'] = (np.minimum(df['spot_open'], df['spot_close']) - df['spot_low']) / hl_range_safe

        # Candle direction (1 = bullish, -1 = bearish)
        df['candle_direction'] = np.sign(df['spot_close'] - df['spot_open'])

        # === Volume-based features ===

        # Taker buy ratio (spot)
        df['spot_taker_buy_ratio'] = df['spot_taker_buy_btc'] / df['spot_volume_btc']

        # Taker sell ratio (spot)
        df['spot_taker_sell_ratio'] = 1 - df['spot_taker_buy_ratio']

        # Volume per trade (liquidez média)
        df['volume_per_trade'] = df['spot_volume_btc'] / df['spot_trade_count']

        # Trade count normalized (rolling z-score)
        trade_ma = df['spot_trade_count'].rolling(30).mean()
        trade_std = df['spot_trade_count'].rolling(30).std()
        df['trade_count_zscore'] = (df['spot_trade_count'] - trade_ma) / trade_std

        # === Basis (Futures Premium) ===

        if 'futures_close' in df.columns:
            # Basis = (Futures - Spot) / Spot * 100
            df['basis_pct'] = (df['futures_close'] - df['spot_close']) / df['spot_close'] * 100

            # Annualized basis (assuming perpetual ~ 3 month equivalent)
            df['basis_annualized'] = df['basis_pct'] * 4  # Rough approximation

            # Basis MA
            df['basis_ma7'] = df['basis_pct'].rolling(7).mean()
            df['basis_ma30'] = df['basis_pct'].rolling(30).mean()

            # Basis z-score
            basis_ma = df['basis_pct'].rolling(90).mean()
            basis_std = df['basis_pct'].rolling(90).std()
            df['basis_zscore'] = (df['basis_pct'] - basis_ma) / basis_std

        # === Long/Short derived ===

        if 'global_ls_ratio' in df.columns:
            # L/S ratio change
            df['ls_ratio_change'] = df['global_ls_ratio'].pct_change()

            # L/S ratio MA
            df['ls_ratio_ma7'] = df['global_ls_ratio'].rolling(7).mean()

            # Extreme long (contrarian bearish)
            df['extreme_long'] = (df['global_long_pct'] > 0.70).astype(int)

            # Extreme short (contrarian bullish)
            df['extreme_short'] = (df['global_short_pct'] > 0.45).astype(int)

        if 'top_traders_ls_ratio' in df.columns:
            # Smart money vs retail divergence
            df['smart_retail_divergence'] = df['top_traders_ls_ratio'] - df.get('global_ls_ratio', 0)

        # === Taker ratio derived ===

        if 'futures_taker_ratio' in df.columns:
            df['taker_ratio_ma7'] = df['futures_taker_ratio'].rolling(7).mean()

            # Aggressive buying (taker ratio > 1.1)
            df['aggressive_buying'] = (df['futures_taker_ratio'] > 1.1).astype(int)

            # Aggressive selling (taker ratio < 0.9)
            df['aggressive_selling'] = (df['futures_taker_ratio'] < 0.9).astype(int)

        # === OI derived ===

        if 'binance_oi_btc' in df.columns:
            df['oi_change_1d'] = df['binance_oi_btc'].pct_change()
            df['oi_change_7d'] = df['binance_oi_btc'].pct_change(7)
            df['oi_ma7'] = df['binance_oi_btc'].rolling(7).mean()

        # Clean up temp columns
        df = df.drop(columns=['prev_close'], errors='ignore')

        logger.info(f"  Created {len([c for c in df.columns if c != 'date'])} features")

        return df

    def fetch_all(self, start_date: str = "2017-08-17",
                  end_date: str = None) -> pd.DataFrame:
        """
        Busca todos os dados e combina em um DataFrame

        Args:
            start_date: Data inicial (default: início do BTCUSDT na Binance)
            end_date: Data final (default: hoje)

        Returns:
            DataFrame com todos os dados combinados
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        logger.info(f"\n{'='*60}")
        logger.info(f"BINANCE DATA FETCHER")
        logger.info(f"{'='*60}")
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"{'='*60}\n")

        # 1. Fetch spot klines (desde 2017)
        df_spot = self.fetch_spot_klines(start_date, end_date)

        if df_spot.empty:
            logger.error("Failed to fetch spot data. Aborting.")
            return pd.DataFrame()

        # Start with spot data
        df = df_spot.copy()

        # 2. Fetch futures klines (desde 2019)
        futures_start = max(start_date, "2019-09-01")
        df_futures = self.fetch_futures_klines(futures_start, end_date)

        if not df_futures.empty:
            df = df.merge(df_futures, on='date', how='left')

        # 3. Fetch funding rates (desde 2020)
        funding_start = max(start_date, "2020-01-01")
        df_funding = self.fetch_funding_rates(funding_start, end_date)

        if not df_funding.empty:
            df = df.merge(df_funding, on='date', how='left')

        # 4. Fetch long/short ratio (desde 2020)
        df_ls = self.fetch_long_short_ratio(funding_start, end_date)

        if not df_ls.empty:
            df = df.merge(df_ls, on='date', how='left')

        # 5. Fetch top traders ratio (desde 2020)
        df_top = self.fetch_top_traders_ratio(funding_start, end_date)

        if not df_top.empty:
            df = df.merge(df_top, on='date', how='left')

        # 6. Fetch taker buy/sell ratio (desde 2020)
        df_taker = self.fetch_taker_buysell_ratio(funding_start, end_date)

        if not df_taker.empty:
            df = df.merge(df_taker, on='date', how='left')

        # 7. Fetch OI history (desde 2020)
        df_oi = self.fetch_oi_history(funding_start, end_date)

        if not df_oi.empty:
            df = df.merge(df_oi, on='date', how='left')

        # 8. Calculate derived features
        df = self.calculate_derived_features(df)

        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)

        logger.info(f"\n{'='*60}")
        logger.info(f"FETCH COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total rows: {len(df)}")
        logger.info(f"Total columns: {len(df.columns)}")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"{'='*60}\n")

        return df

    def update_existing(self, csv_path: str) -> pd.DataFrame:
        """
        Atualiza CSV existente com dados novos (incremental update)

        Args:
            csv_path: Caminho do CSV existente

        Returns:
            DataFrame atualizado
        """
        if not os.path.exists(csv_path):
            logger.info("No existing file found. Fetching all data...")
            return self.fetch_all()

        existing = pd.read_csv(csv_path)
        last_date = existing['date'].max()

        logger.info(f"Existing data until: {last_date}")

        # Fetch from last date + 1 day
        start = (datetime.strptime(last_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        end = datetime.now().strftime('%Y-%m-%d')

        if start >= end:
            logger.info("Data is already up to date!")
            return existing

        logger.info(f"Fetching new data: {start} to {end}")

        new_data = self.fetch_all(start, end)

        if new_data.empty:
            return existing

        # Combine
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=['date'], keep='last')
        combined = combined.sort_values('date').reset_index(drop=True)

        return combined


def clean_and_prepare_data(df: pd.DataFrame,
                          remove_short_history: bool = True,
                          min_history_days: int = 365) -> pd.DataFrame:
    """
    Limpa e prepara dados da Binance para ML.

    REGRAS DE TRATAMENTO (PhD ML best practices):

    1. Features com < min_history_days de dados → REMOVE
       - L/S ratios, taker ratios futures, OI history (só ~30 dias)
       - Inúteis para backtest, só para produção diária

    2. Features com dados parciais → IMPUTE com valor NEUTRO + FLAG
       - basis_pct: 0 (sem premium = neutro)
       - funding: 0 (funding neutro)
       - futures OHLCV: forward-fill do spot (aproximação)

    3. Criar FLAGS de disponibilidade
       - has_futures_data: 1 se date >= 2019-09-08
       - has_funding_data: 1 se date >= 2020-01-01

    Args:
        df: DataFrame com dados brutos da Binance
        remove_short_history: Se True, remove features com pouco histórico
        min_history_days: Mínimo de dias de histórico para manter feature

    Returns:
        DataFrame limpo e pronto para ML
    """
    logger.info("Cleaning and preparing data for ML...")

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # =========================================================================
    # 1. IDENTIFICAR E REMOVER FEATURES COM POUCO HISTÓRICO
    # =========================================================================

    if remove_short_history:
        cols_to_remove = []

        for col in df.columns:
            if col == 'date':
                continue

            # Contar dias com dados válidos (não null e não zero)
            valid_days = ((df[col].notna()) & (df[col] != 0)).sum()

            if valid_days < min_history_days:
                cols_to_remove.append(col)

        if cols_to_remove:
            logger.info(f"  Removing {len(cols_to_remove)} features with < {min_history_days} days history:")
            for col in cols_to_remove:
                logger.info(f"    - {col}")

            df = df.drop(columns=cols_to_remove)

    # =========================================================================
    # 2. CRIAR FLAGS DE DISPONIBILIDADE
    # =========================================================================

    # Flag para dados de futures (disponível desde 2019-09-08)
    futures_start = pd.Timestamp('2019-09-08')
    df['has_futures_data'] = (df['date'] >= futures_start).astype(int)

    # Flag para dados de funding (disponível desde 2020-01-01)
    funding_start = pd.Timestamp('2020-01-01')
    df['has_funding_data'] = (df['date'] >= funding_start).astype(int)

    logger.info(f"  Created availability flags")
    logger.info(f"    - has_futures_data: {df['has_futures_data'].sum()} days")
    logger.info(f"    - has_funding_data: {df['has_funding_data'].sum()} days")

    # =========================================================================
    # 3. IMPUTAÇÃO INTELIGENTE
    # =========================================================================

    # --- Basis (futures premium) ---
    # Antes de futures existirem, não há premium → 0 é valor neutro correto
    basis_cols = [c for c in df.columns if 'basis' in c.lower()]
    for col in basis_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            df[col] = df[col].fillna(0)
            logger.info(f"  {col}: filled {null_count} nulls with 0 (neutral)")

    # --- Funding rate ---
    # Antes de funding existir, funding = 0 (neutro)
    funding_cols = [c for c in df.columns if 'funding' in c.lower()]
    for col in funding_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            df[col] = df[col].fillna(0)
            logger.info(f"  {col}: filled {null_count} nulls with 0 (neutral)")

    # --- Futures OHLCV ---
    # Antes de futures, usar spot como proxy (são muito próximos)
    futures_ohlcv = ['futures_open', 'futures_high', 'futures_low', 'futures_close']
    spot_ohlcv = ['spot_open', 'spot_high', 'spot_low', 'spot_close']

    for fut_col, spot_col in zip(futures_ohlcv, spot_ohlcv):
        if fut_col in df.columns and spot_col in df.columns:
            null_mask = df[fut_col].isnull()
            null_count = null_mask.sum()
            if null_count > 0:
                df.loc[null_mask, fut_col] = df.loc[null_mask, spot_col]
                logger.info(f"  {fut_col}: filled {null_count} nulls with {spot_col}")

    # --- Futures volume ---
    # Antes de futures, volume = 0 (não existia)
    fut_vol_cols = [c for c in df.columns if 'futures_volume' in c.lower()]
    for col in fut_vol_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            df[col] = df[col].fillna(0)
            logger.info(f"  {col}: filled {null_count} nulls with 0")

    # --- Gap e True Range ---
    # Primeiro dia não tem gap (não há dia anterior)
    if 'gap_pct' in df.columns:
        df['gap_pct'] = df['gap_pct'].fillna(0)

    if 'true_range' in df.columns:
        # Para primeira linha, usar high-low como aproximação
        if df['true_range'].isnull().any():
            mask = df['true_range'].isnull()
            df.loc[mask, 'true_range'] = df.loc[mask, 'spot_high'] - df.loc[mask, 'spot_low']

    # --- ATR ---
    # Rolling features precisam de warmup, forward fill do primeiro válido
    atr_cols = [c for c in df.columns if 'atr' in c.lower()]
    for col in atr_cols:
        if col in df.columns:
            df[col] = df[col].bfill()  # Backfill para período de warmup

    # --- Shadow ratios ---
    # Podem ser NaN se candle tem range 0 (doji perfeito) → usar 0.5 (neutro)
    shadow_cols = [c for c in df.columns if 'shadow' in c.lower()]
    for col in shadow_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.5)  # Neutro = sombras iguais

    # --- Trade count zscore ---
    # Rolling, precisa warmup
    if 'trade_count_zscore' in df.columns:
        df['trade_count_zscore'] = df['trade_count_zscore'].fillna(0)  # Neutro

    # =========================================================================
    # 4. VERIFICAÇÃO FINAL
    # =========================================================================

    remaining_nulls = df.isnull().sum().sum()
    if remaining_nulls > 0:
        logger.warning(f"  WARNING: {remaining_nulls} null values remaining!")
        # Fill any remaining with 0
        for col in df.columns:
            if col != 'date' and df[col].isnull().any():
                df[col] = df[col].fillna(0)

    logger.info(f"  Final shape: {df.shape}")
    logger.info(f"  Null values: {df.isnull().sum().sum()}")

    return df


def main():
    """Main function - executa fetch completo ou atualização"""
    import argparse

    parser = argparse.ArgumentParser(description='Fetch Binance historical data')
    parser.add_argument('--start', type=str, default='2019-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None,
                       help='End date (YYYY-MM-DD), default: today')
    parser.add_argument('--output', type=str, default='data/binance_data.csv',
                       help='Output CSV path')
    parser.add_argument('--update', action='store_true',
                       help='Update existing file incrementally')
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                       help='Trading pair symbol')
    parser.add_argument('--no-clean', action='store_true',
                       help='Skip data cleaning (keep raw data)')
    parser.add_argument('--min-history', type=int, default=365,
                       help='Minimum days of history to keep feature (default: 365)')

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)

    fetcher = BinanceDataFetcher(symbol=args.symbol)

    if args.update and os.path.exists(args.output):
        df = fetcher.update_existing(args.output)
    else:
        df = fetcher.fetch_all(start_date=args.start, end_date=args.end)

    if not df.empty:
        # Clean and prepare data for ML
        if not args.no_clean:
            df = clean_and_prepare_data(
                df,
                remove_short_history=True,
                min_history_days=args.min_history
            )

        df.to_csv(args.output, index=False)
        logger.info(f"\nData saved to: {args.output}")
        logger.info(f"Shape: {df.shape}")

        # Print summary
        print("\n" + "="*60)
        print("COLUMNS SUMMARY")
        print("="*60)

        categories = {
            'Spot OHLCV': [c for c in df.columns if c.startswith('spot_')],
            'Futures': [c for c in df.columns if c.startswith('futures_') or c.startswith('basis')],
            'Funding': [c for c in df.columns if 'funding' in c.lower()],
            'Long/Short': [c for c in df.columns if 'ls_' in c or 'long' in c.lower() or 'short' in c.lower()],
            'Taker': [c for c in df.columns if 'taker' in c.lower()],
            'OI': [c for c in df.columns if 'oi' in c.lower()],
            'Candle Patterns': [c for c in df.columns if 'candle' in c or 'shadow' in c or 'gap' in c],
            'Volatility': [c for c in df.columns if 'atr' in c or 'range' in c or 'true_range' in c],
            'Other': []
        }

        # Find uncategorized
        categorized = set()
        for cols in categories.values():
            categorized.update(cols)
        categories['Other'] = [c for c in df.columns if c not in categorized and c != 'date']

        for cat, cols in categories.items():
            if cols:
                print(f"\n{cat} ({len(cols)}):")
                for col in cols[:10]:  # Show first 10
                    print(f"  - {col}")
                if len(cols) > 10:
                    print(f"  ... and {len(cols)-10} more")

        # Print null summary
        print("\n" + "="*60)
        print("NULL VALUES (expected for pre-2020 futures data)")
        print("="*60)
        nulls = df.isnull().sum()
        nulls = nulls[nulls > 0].sort_values(ascending=False)
        if len(nulls) > 0:
            print(nulls.head(15).to_string())
        else:
            print("No null values!")
    else:
        logger.error("Failed to fetch data")


if __name__ == "__main__":
    main()
