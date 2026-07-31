"""
FETCH ON-CHAIN DATA - Métricas Avançadas de Ciclo
==================================================

Busca métricas on-chain avançadas de APIs gratuitas:
1. Blockchain.com API (sem rate limit): mempool, miners revenue, fees
2. BGeometrics API (rate limited): SOPR, NUPL, STH-SOPR, Realized Price
3. Google Trends (pytrends): Bitcoin search volume

Uso:
    python scripts/fetch_onchain_data.py
    python scripts/fetch_onchain_data.py --skip-bgeometrics  # Se rate limited
    python scripts/fetch_onchain_data.py --only-bgeometrics  # Só BGeometrics

Output:
    data/onchain_data.csv
"""

import pandas as pd
import numpy as np
import requests
import time
import argparse
from pathlib import Path
from datetime import datetime

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_FILE = DATA_DIR / "onchain_data.csv"


# =============================================================================
# 1. BLOCKCHAIN.COM API (No rate limit)
# =============================================================================
class BlockchainAPI:
    """
    API gratuita do Blockchain.com - sem rate limit.
    Docs: https://www.blockchain.com/api/charts_api
    """

    BASE_URL = "https://api.blockchain.info/charts"

    METRICS = {
        'mempool-size': 'mempool_size_bytes',
        'mempool-count': 'mempool_tx_count',
        'miners-revenue': 'miners_revenue_usd',
        'cost-per-transaction': 'cost_per_tx_usd',
        'transaction-fees': 'total_fees_usd',
        'n-transactions': 'n_transactions',
    }

    def __init__(self):
        self.session = requests.Session()

    def fetch_metric(self, endpoint: str, col_name: str) -> pd.DataFrame:
        """Busca uma métrica específica."""
        url = f"{self.BASE_URL}/{endpoint}?timespan=all&format=json"

        try:
            r = self.session.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()

            values = data.get('values', [])
            if not values:
                return pd.DataFrame()

            df = pd.DataFrame(values)
            df['date'] = pd.to_datetime(df['x'], unit='s').dt.date
            df['date'] = pd.to_datetime(df['date'])
            df[col_name] = df['y']

            return df[['date', col_name]]

        except Exception as e:
            print(f"   ERROR {col_name}: {e}")
            return pd.DataFrame()

    def fetch_all(self) -> pd.DataFrame:
        """Busca todas as métricas."""
        print("\n[BLOCKCHAIN.COM] Buscando métricas...")

        all_data = None

        for endpoint, col_name in self.METRICS.items():
            df = self.fetch_metric(endpoint, col_name)

            if df.empty:
                continue

            if all_data is None:
                all_data = df
            else:
                all_data = all_data.merge(df, on='date', how='outer')

            print(f"   {col_name}: {len(df):,} rows")
            time.sleep(0.3)

        return all_data


# =============================================================================
# 2. BGEOMETRICS API (Rate limited - ~10 requests/hour)
# =============================================================================
class BGeometricsAPI:
    """
    API gratuita para SOPR, NUPL, Realized Price.
    Rate limit: ~10 requests por hora.
    Docs: https://bitcoin-data.com/api/redoc.html
    """

    BASE_URL = "https://bitcoin-data.com/v1"

    METRICS = {
        'sopr': 'sopr',
        'nupl': 'nupl',
        'sth-sopr': 'sth_sopr',
        'realized-price': 'realized_price',
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def fetch_metric(self, endpoint: str, col_name: str) -> pd.DataFrame:
        """Busca uma métrica específica."""
        url = f"{self.BASE_URL}/{endpoint}"

        try:
            r = self.session.get(url, timeout=30)

            if r.status_code == 429:
                print(f"   {col_name}: RATE LIMITED (tente novamente em 1h)")
                return pd.DataFrame()

            r.raise_for_status()
            data = r.json()

            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)

            # Get date column
            if 'd' in df.columns:
                df['date'] = pd.to_datetime(df['d'])
            elif 'theDay' in df.columns:
                df['date'] = pd.to_datetime(df['theDay'])

            # Get value column
            value_cols = [c for c in df.columns if c not in ['d', 'theDay', 'date', 'unixTs']]
            if value_cols:
                df[col_name] = pd.to_numeric(df[value_cols[0]], errors='coerce')

            return df[['date', col_name]].dropna()

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"   {col_name}: RATE LIMITED")
            else:
                print(f"   ERROR {col_name}: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"   ERROR {col_name}: {e}")
            return pd.DataFrame()

    def fetch_all(self, delay: float = 3.0) -> pd.DataFrame:
        """Busca todas as métricas (com delay para evitar rate limit)."""
        print("\n[BGEOMETRICS] Buscando métricas on-chain avançadas...")
        print("   (Rate limit: ~10 req/hora - seja paciente)")

        all_data = None

        for endpoint, col_name in self.METRICS.items():
            df = self.fetch_metric(endpoint, col_name)

            if df.empty:
                continue

            if all_data is None:
                all_data = df
            else:
                all_data = all_data.merge(df, on='date', how='outer')

            print(f"   {col_name}: {len(df):,} rows")
            time.sleep(delay)

        return all_data


# =============================================================================
# 3. GOOGLE TRENDS
# =============================================================================
def fetch_google_trends() -> pd.DataFrame:
    """Busca interesse de busca por Bitcoin."""
    print("\n[GOOGLE TRENDS] Buscando interesse de busca...")

    try:
        from pytrends.request import TrendReq
    except ImportError:
        print("   WARNING: pytrends não instalado (pip install pytrends)")
        return pd.DataFrame()

    try:
        pytrend = TrendReq(hl='en-US', tz=360)
        pytrend.build_payload(kw_list=['Bitcoin'], timeframe='today 5-y', geo='', gprop='')
        df = pytrend.interest_over_time()

        if df.empty:
            print("   WARNING: Resposta vazia")
            return pd.DataFrame()

        df = df.reset_index()
        df['date'] = pd.to_datetime(df['date'])
        df = df.rename(columns={'Bitcoin': 'google_trend_btc'})
        df = df[['date', 'google_trend_btc']]

        # Expand weekly to daily
        df = df.set_index('date').resample('D').ffill().reset_index()

        print(f"   google_trend_btc: {len(df)} rows")
        return df

    except Exception as e:
        print(f"   ERROR: {e}")
        return pd.DataFrame()


# =============================================================================
# 4. DERIVED FEATURES
# =============================================================================
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula features derivadas."""
    print("\n[DERIVED] Calculando features derivadas...")

    # SOPR features
    if 'sopr' in df.columns:
        df['sopr_ma7'] = df['sopr'].rolling(7, min_periods=1).mean()
        df['sopr_ma30'] = df['sopr'].rolling(30, min_periods=1).mean()
        df['sopr_below_1'] = (df['sopr'] < 1).astype(int)
        print("   SOPR features: OK")

    # NUPL features
    if 'nupl' in df.columns:
        df['nupl_ma30'] = df['nupl'].rolling(30, min_periods=1).mean()
        df['nupl_capitulation'] = (df['nupl'] < 0).astype(int)
        df['nupl_euphoria'] = (df['nupl'] > 0.75).astype(int)
        print("   NUPL features: OK")

    # STH-SOPR features
    if 'sth_sopr' in df.columns:
        df['sth_sopr_ma7'] = df['sth_sopr'].rolling(7, min_periods=1).mean()
        df['sth_capitulation'] = (df['sth_sopr'] < 1).astype(int)
        print("   STH-SOPR features: OK")

    # Mempool features
    if 'mempool_tx_count' in df.columns:
        df['mempool_ma7'] = df['mempool_tx_count'].rolling(7, min_periods=1).mean()
        df['mempool_congestion'] = df['mempool_tx_count'] / df['mempool_ma7']
        print("   Mempool features: OK")

    # Miners revenue features
    if 'miners_revenue_usd' in df.columns:
        df['miners_revenue_ma30'] = df['miners_revenue_usd'].rolling(30, min_periods=1).mean()
        df['miners_revenue_ratio'] = df['miners_revenue_usd'] / df['miners_revenue_ma30']
        print("   Miners features: OK")

    # Google Trends features
    if 'google_trend_btc' in df.columns:
        df['google_trend_ma4w'] = df['google_trend_btc'].rolling(28, min_periods=1).mean()
        df['google_trend_fomo'] = (df['google_trend_btc'] > 80).astype(int)
        print("   Google Trends features: OK")

    return df


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Fetch on-chain data')
    parser.add_argument('--skip-bgeometrics', action='store_true',
                        help='Skip BGeometrics (if rate limited)')
    parser.add_argument('--skip-google', action='store_true',
                        help='Skip Google Trends (if rate limited)')
    parser.add_argument('--only-bgeometrics', action='store_true',
                        help='Only fetch BGeometrics')
    args = parser.parse_args()

    print("=" * 70)
    print("FETCH ON-CHAIN DATA")
    print("=" * 70)

    all_dfs = []

    # 1. Blockchain.com (always works)
    if not args.only_bgeometrics:
        bc_api = BlockchainAPI()
        bc_df = bc_api.fetch_all()
        if bc_df is not None and not bc_df.empty:
            all_dfs.append(bc_df)

    # 2. BGeometrics (rate limited)
    if not args.skip_bgeometrics:
        bg_api = BGeometricsAPI()
        bg_df = bg_api.fetch_all()
        if bg_df is not None and not bg_df.empty:
            all_dfs.append(bg_df)

    # 3. Google Trends
    if not args.skip_google and not args.only_bgeometrics:
        gt_df = fetch_google_trends()
        if gt_df is not None and not gt_df.empty:
            all_dfs.append(gt_df)

    # Merge all
    if not all_dfs:
        print("\nERROR: Nenhum dado obtido!")
        return

    final_df = all_dfs[0]
    for df in all_dfs[1:]:
        final_df = final_df.merge(df, on='date', how='outer')

    # Add derived features
    final_df = add_derived_features(final_df)

    # Filter and sort
    final_df['date'] = pd.to_datetime(final_df['date'])
    final_df = final_df[final_df['date'] >= '2019-01-01']
    final_df = final_df.sort_values('date').reset_index(drop=True)

    # Save
    final_df.to_csv(OUTPUT_FILE, index=False)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Shape: {final_df.shape}")
    print(f"Date: {final_df['date'].min().date()} to {final_df['date'].max().date()}")
    print(f"Nulls: {final_df.isnull().sum().sum()}")

    print("\nFeatures:")
    for col in sorted(final_df.columns):
        if col != 'date':
            non_null = final_df[col].notna().sum()
            pct = non_null / len(final_df) * 100
            print(f"   {col}: {non_null:,} values ({pct:.0f}%)")

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
