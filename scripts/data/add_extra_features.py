"""
ADD EXTRA FEATURES - Novas Features para o Dataset BTC/RF
==========================================================

Este script adiciona features extras ao dataset existente:
1. Features calculadas (Stock-to-Flow, Puell Multiple, Difficulty Ribbon)
2. DefiLlama Stablecoins
3. Google Trends
4. ETF Flows (Farside)
5. Tratamento de outliers

Uso:
    python scripts/add_extra_features.py

Ou importar funcoes individuais:
    from add_extra_features import add_calculated_features, fetch_stablecoin_data
"""

import pandas as pd
import numpy as np
import requests
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

# =============================================================================
# 1. TRATAMENTO DE OUTLIERS
# =============================================================================
def treat_outliers(df, sigma=5, window=365):
    """
    Aplica winsorization em features com outliers extremos.

    IMPORTANTE: Usa ROLLING mean/std para evitar data leakage!
    - Antes: usava mean/std de TODO o dataset (incluindo dados futuros)
    - Agora: usa rolling window de 365 dias (só dados passados)

    Args:
        df: DataFrame
        sigma: Número de desvios padrão para clipping (default: 5)
        window: Janela para cálculo de rolling mean/std (default: 365 dias)
    """
    print("\n[OUTLIERS] Tratando outliers extremos (com rolling stats)...")

    outlier_cols = [
        'oi_change_30d', 'buy_sell_ratio', 'fee_median_usd',
        'oi_change_7d', 'exchange_netflow_btc', 'fees_btc',
        'intraday_range', 'oil_return_30d', 'exchange_netflow_lag_1',
        'exchange_netflow_lag_7', 'net_buy_pressure'
    ]

    treated = 0
    for col in outlier_cols:
        if col in df.columns:
            # ROLLING mean/std para não usar dados futuros
            rolling_mean = df[col].rolling(window, min_periods=30).mean()
            rolling_std = df[col].rolling(window, min_periods=30).std()

            # Para os primeiros dias sem rolling suficiente, usar expanding
            expanding_mean = df[col].expanding(min_periods=30).mean()
            expanding_std = df[col].expanding(min_periods=30).std()

            # Combinar: usar rolling onde disponível, expanding para início
            mean = rolling_mean.fillna(expanding_mean)
            std = rolling_std.fillna(expanding_std)

            if std.notna().any():
                lower = mean - sigma * std
                upper = mean + sigma * std

                # Contar outliers antes do clipping
                outliers_before = ((df[col] < lower) | (df[col] > upper)).sum()

                if outliers_before > 0:
                    # Aplicar clipping apenas onde temos bounds válidos
                    df[col] = df[col].clip(lower=lower, upper=upper)
                    treated += outliers_before
                    print(f"   {col}: {outliers_before} valores clipped")

    print(f"   Total de outliers tratados: {treated}")
    return df


# =============================================================================
# 2. FEATURES CALCULADAS
# =============================================================================
def load_coinmetrics_raw():
    """
    Carrega dados brutos do CoinMetrics para features adicionais.
    Usa o arquivo mais recente (btc.csv ou btc2.csv).
    """
    # Find latest CoinMetrics file
    cm_files = list(DATA_DIR.glob("btc*.csv"))
    cm_files = [f for f in cm_files if f.name.startswith('btc') and not f.name.startswith('bitcoin')]

    if not cm_files:
        print(f"   CoinMetrics file not found in {DATA_DIR}")
        return None

    cm_file = max(cm_files, key=lambda x: x.stat().st_mtime)
    print(f"   Loaded extra columns from CoinMetrics: {cm_file.name}")

    cm = pd.read_csv(cm_file)
    cm['date'] = pd.to_datetime(cm['time']).dt.date
    cm['date'] = pd.to_datetime(cm['date'])

    # Extrair colunas necessarias
    cols_map = {
        'SplyCur': 'circulating_supply',
        'HashRate': 'hash_rate_raw',
        'IssTotNtv': 'btc_issued_daily_raw',
        'CapMrktCurUSD': 'market_cap_raw'
    }

    for old, new in cols_map.items():
        if old in cm.columns:
            cm[new] = cm[old]

    keep_cols = ['date'] + [v for v in cols_map.values() if v in cm.columns]
    cm = cm[keep_cols].copy()
    cm = cm.groupby('date').last().reset_index()

    return cm


def add_calculated_features(df):
    """
    Adiciona features calculadas a partir dos dados existentes:
    - Stock-to-Flow e S2F Multiple
    - Puell Multiple
    - Difficulty Ribbon e Ribbon Compression
    - NVT Signal melhorado
    """
    print("\n[CALC] Calculando features derivadas...")

    # Carregar dados extras do CoinMetrics se necessario
    if 'circulating_supply' not in df.columns or 'hash_rate_raw' not in df.columns:
        cm_data = load_coinmetrics_raw()
        if cm_data is not None:
            # Merge cuidadoso para nao sobrescrever colunas existentes
            for col in cm_data.columns:
                if col != 'date' and col not in df.columns:
                    df = df.merge(cm_data[['date', col]], on='date', how='left')
            print("   Loaded extra columns from CoinMetrics")

    # Usar colunas raw se disponiveis
    circ_supply_col = 'circulating_supply' if 'circulating_supply' in df.columns else None
    hash_rate_col = 'hash_rate_raw' if 'hash_rate_raw' in df.columns else ('hash_rate' if 'hash_rate' in df.columns else None)
    btc_issued_col = 'btc_issued_daily_raw' if 'btc_issued_daily_raw' in df.columns else ('btc_issued_daily' if 'btc_issued_daily' in df.columns else None)

    # --- STOCK-TO-FLOW ---
    if circ_supply_col and btc_issued_col:
        # Evitar divisao por zero
        daily_issuance = df[btc_issued_col].replace(0, np.nan)
        annual_issuance = daily_issuance * 365

        df['stock_to_flow'] = df[circ_supply_col] / annual_issuance
        df['stock_to_flow'] = df['stock_to_flow'].clip(0, 1000)  # Limitar valores extremos

        # S2F Model Price = S2F^3 (simplificado)
        # S2F Multiple = Price / S2F_Model_Price
        s2f_model_price = (df['stock_to_flow'] ** 3)
        df['s2f_multiple'] = df['price_usd'] / s2f_model_price.replace(0, np.nan)
        df['s2f_multiple'] = df['s2f_multiple'].clip(0, 100)

        # S2F Z-score
        s2f_mean = df['s2f_multiple'].rolling(365, min_periods=30).mean()
        s2f_std = df['s2f_multiple'].rolling(365, min_periods=30).std()
        df['s2f_zscore'] = (df['s2f_multiple'] - s2f_mean) / (s2f_std + 1e-10)
        df['s2f_zscore'] = df['s2f_zscore'].clip(-5, 5)

        print("   Stock-to-Flow: OK")
    else:
        print("   Stock-to-Flow: SKIP (missing columns)")

    # --- PUELL MULTIPLE ---
    if 'issuance_usd' in df.columns:
        df['issuance_ma_365'] = df['issuance_usd'].rolling(365, min_periods=30).mean()
        df['puell_multiple'] = df['issuance_usd'] / df['issuance_ma_365'].replace(0, np.nan)
        df['puell_multiple'] = df['puell_multiple'].clip(0, 10)

        # Puell extremes
        df['puell_extreme_low'] = (df['puell_multiple'] < 0.5).astype(int)
        df['puell_extreme_high'] = (df['puell_multiple'] > 4).astype(int)

        print("   Puell Multiple: OK")
    else:
        print("   Puell Multiple: SKIP (missing issuance_usd)")

    # --- DIFFICULTY RIBBON ---
    if hash_rate_col and hash_rate_col in df.columns:
        # Calcular medias moveis do hash rate
        periods = [9, 14, 25, 40, 60, 90, 128, 200]
        for period in periods:
            df[f'hash_ma_{period}'] = df[hash_rate_col].rolling(period, min_periods=period//2).mean()

        # Difficulty Ribbon = MA curta / MA longa - 1
        df['difficulty_ribbon'] = df['hash_ma_9'] / df['hash_ma_200'].replace(0, np.nan) - 1
        df['difficulty_ribbon'] = df['difficulty_ribbon'].clip(-1, 1)

        # Ribbon Compression (todas as MAs convergindo = capitulacao de miners)
        ribbon_range = df['hash_ma_9'] - df['hash_ma_200']
        df['ribbon_compression'] = ribbon_range.abs() / df['hash_ma_200'].replace(0, np.nan)
        df['ribbon_compression'] = df['ribbon_compression'].clip(0, 2)

        # Miner capitulation signal
        df['miner_capitulation'] = (
            (df['difficulty_ribbon'] < -0.05) &
            (df['ribbon_compression'] < 0.1)
        ).astype(int)

        # Limpar colunas intermediarias
        for period in periods:
            if f'hash_ma_{period}' in df.columns:
                df = df.drop(columns=[f'hash_ma_{period}'])

        print("   Difficulty Ribbon: OK")
    else:
        print(f"   Difficulty Ribbon: SKIP (hash_rate_col={hash_rate_col})")

    # --- NVT SIGNAL MELHORADO ---
    mcap_col = 'market_cap' if 'market_cap' in df.columns else ('market_cap_raw' if 'market_cap_raw' in df.columns else None)
    if mcap_col and 'transfer_count' in df.columns:
        # NVT = Market Cap / Transaction Volume
        tx_volume_usd = df['transfer_count'] * df['price_usd']
        tx_volume_ma = tx_volume_usd.rolling(90, min_periods=7).mean()

        df['nvt_signal'] = df[mcap_col] / tx_volume_ma.replace(0, np.nan)
        df['nvt_signal'] = df['nvt_signal'].clip(0, 500)

        # NVT Golden Cross (NVT vs sua MA)
        nvt_ma = df['nvt_signal'].rolling(30).mean()
        df['nvt_golden_cross'] = (df['nvt_signal'] < nvt_ma).astype(int)

        print("   NVT Signal: OK")

    # --- REALIZED PRICE PROXY ---
    # Como nao temos realized cap, usamos MVRV para estimar
    if 'mvrv_zscore' in df.columns and 'price_usd' in df.columns:
        # Realized Price ~ Price / MVRV (aproximacao)
        # MVRV > 1 significa Price > Realized Price
        df['price_vs_realized'] = df['mvrv_zscore']  # Ja temos isso como proxy
        print("   Price vs Realized: OK (using MVRV)")

    return df


# =============================================================================
# 3. BINANCE DATA (OHLCV, Volume, Futures)
# =============================================================================
def load_binance_data():
    """
    Carrega dados da Binance (previamente baixados por fetch_binance_data.py).

    Dados incluem:
    - OHLCV spot (open, high, low, close, volume)
    - Trade count e taker buy volumes
    - Futures OHLCV (para basis)
    - Funding rates
    - Long/Short ratios (últimos 30 dias apenas)
    - Features derivadas (ATR, candle patterns, etc.)
    """
    binance_file = DATA_DIR / "binance_data.csv"

    if not binance_file.exists():
        print(f"   Binance data not found: {binance_file}")
        print("   Run: python scripts/fetch_binance_data.py")
        return None

    df = pd.read_csv(binance_file)
    df['date'] = pd.to_datetime(df['date'])

    print(f"   Binance data loaded: {len(df)} rows, {len(df.columns)} columns")
    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    return df


def add_binance_features(df, binance_df):
    """
    Integra dados da Binance ao dataset principal.

    IMPORTANTE: Binance é a fonte PRIMÁRIA de preços OHLC.
    - Substitui price_open/high/low do Artemis pelos dados Binance (mais precisos)
    - Mantém price_usd como close price principal

    Features adicionadas:
    - price_open, price_high, price_low (de Binance, substituindo Artemis)
    - spot_trade_count, spot_taker_buy_ratio
    - true_range, atr_14, atr_pct
    - gap_pct, candle_body_ratio, shadow ratios
    - basis_pct (futures premium)
    - binance_funding_daily
    """
    if binance_df is None or len(binance_df) == 0:
        print("   Binance features: SKIP (no data)")
        return df

    print("\n[BINANCE] Integrando dados da Binance...")

    # =========================================================================
    # SUBSTITUIR OHLC do Artemis pelo Binance (mais preciso)
    # =========================================================================

    # Primeiro, remover colunas OHLC do Artemis se existirem
    artemis_ohlc_cols = ['price_open', 'price_high', 'price_low']
    removed_cols = []
    for col in artemis_ohlc_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
            removed_cols.append(col)

    if removed_cols:
        print(f"   Removed Artemis OHLC: {removed_cols}")

    # Mapear colunas Binance para nomes padrão
    # IMPORTANTE: Usamos Binance como fonte UNICA de preços OHLC para consistência
    binance_to_standard = {
        'spot_open': 'price_open',
        'spot_high': 'price_high',
        'spot_low': 'price_low',
        'spot_close': 'price_close_binance',  # Será usado para substituir price_usd
    }

    # Colunas que queremos adicionar
    binance_cols_to_add = [
        # OHLC - mapear para nomes padrão (inclui close!)
        'spot_open', 'spot_high', 'spot_low', 'spot_close',

        # Volume e Trade Count
        'spot_volume_usd', 'spot_trade_count',
        'spot_taker_buy_btc', 'spot_taker_buy_usd',
        'spot_taker_buy_ratio', 'spot_taker_sell_ratio',

        # Futures OHLCV
        'futures_open', 'futures_high', 'futures_low', 'futures_close',
        'futures_volume_btc', 'futures_volume_usd',

        # Basis (Futures Premium)
        'basis_pct', 'basis_annualized', 'basis_ma7', 'basis_ma30', 'basis_zscore',

        # Funding
        'binance_funding_daily', 'binance_funding_mean',
        'binance_funding_max', 'binance_funding_min',

        # Flags de disponibilidade
        'has_futures_data', 'has_funding_data',

        # Volatility
        'true_range', 'atr_14', 'atr_pct', 'intraday_range_pct',

        # Candle Patterns
        'gap_pct', 'candle_body_ratio',
        'upper_shadow_ratio', 'lower_shadow_ratio', 'candle_direction',

        # Other
        'volume_per_trade', 'trade_count_zscore'
    ]

    # Filtrar colunas que existem no binance_df
    cols_to_merge = ['date'] + [c for c in binance_cols_to_add
                                 if c in binance_df.columns]

    # Merge
    original_rows = len(df)
    df = df.merge(binance_df[cols_to_merge], on='date', how='left')

    # Renomear colunas OHLC para nomes padrão
    for old_name, new_name in binance_to_standard.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
            print(f"   Renamed {old_name} -> {new_name}")

    added_cols = len(cols_to_merge) - 1  # -1 para 'date'
    print(f"   Added {added_cols} columns from Binance data")

    # Verificar se High/Low foram adicionados (agora são price_high/price_low)
    if 'price_high' in df.columns and 'price_low' in df.columns:
        # Recalcular Parkinson Volatility (mais precisa com High/Low reais)
        log_hl = np.log(df['price_high'] / df['price_low'])
        df['parkinson_vol_14d_binance'] = np.sqrt(
            (1 / (4 * np.log(2))) * (log_hl ** 2)
        ).rolling(14).mean()

        print("   Recalculated Parkinson volatility with real High/Low")

    # =========================================================================
    # SUBSTITUIR price_usd pelo Binance close para consistência OHLC
    # =========================================================================
    if 'price_close_binance' in df.columns and 'price_usd' in df.columns:
        # Guardar preço Artemis como referência
        df['price_usd_artemis'] = df['price_usd'].copy()

        # Contar quantos valores Binance temos
        binance_available = df['price_close_binance'].notna().sum()
        total_rows = len(df)

        # Substituir price_usd pelo Binance close onde disponível
        mask = df['price_close_binance'].notna()
        df.loc[mask, 'price_usd'] = df.loc[mask, 'price_close_binance']

        # Calcular diferença média para log
        both_available = df['price_usd_artemis'].notna() & df['price_close_binance'].notna()
        if both_available.sum() > 0:
            diff_pct = ((df.loc[both_available, 'price_close_binance'] -
                        df.loc[both_available, 'price_usd_artemis']).abs() /
                       df.loc[both_available, 'price_usd_artemis'] * 100)
            print(f"   Replaced price_usd with Binance close: {binance_available}/{total_rows} rows")
            print(f"   Price difference Artemis vs Binance: mean={diff_pct.mean():.3f}%, max={diff_pct.max():.2f}%")

        # Remover coluna temporária
        df = df.drop(columns=['price_close_binance'])
        print("   Now using Binance as SINGLE source for OHLC (consistent)")

    return df


# =============================================================================
# 4. DEFILLAMA STABLECOINS
# =============================================================================
def fetch_stablecoin_data():
    """
    Busca dados historicos de stablecoins do DefiLlama.
    API gratuita, sem autenticacao.
    """
    print("\n[DEFILLAMA] Buscando dados de stablecoins...")

    try:
        url = "https://stablecoins.llama.fi/stablecoincharts/all"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        data = response.json()

        records = []
        for item in data:
            date = pd.to_datetime(item['date'], unit='s')

            # Total stablecoin supply
            total_supply = 0
            if 'totalCirculating' in item and 'peggedUSD' in item['totalCirculating']:
                total_supply = item['totalCirculating']['peggedUSD']
            elif 'totalCirculatingUSD' in item and 'peggedUSD' in item['totalCirculatingUSD']:
                total_supply = item['totalCirculatingUSD']['peggedUSD']

            records.append({
                'date': date.date(),
                'stablecoin_supply': total_supply
            })

        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        df = df.groupby('date').last().reset_index()
        df = df.sort_values('date')

        print(f"   Stablecoin data: {len(df)} rows")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")

        return df

    except Exception as e:
        print(f"   ERROR: {e}")
        return None


def add_stablecoin_features(df, stablecoin_df):
    """
    Adiciona features de stablecoins ao dataframe principal.
    """
    if stablecoin_df is None or len(stablecoin_df) == 0:
        print("   Stablecoin features: SKIP (no data)")
        return df

    # Se já existe, skip
    if 'stablecoin_supply' in df.columns:
        print("   Stablecoin features: SKIP (already exists)")
        return df

    # Merge
    df = df.merge(stablecoin_df, on='date', how='left')

    # Forward fill para dias sem dados
    df['stablecoin_supply'] = df['stablecoin_supply'].ffill()

    # Features derivadas
    df['stablecoin_supply_change_7d'] = df['stablecoin_supply'].pct_change(7)
    df['stablecoin_supply_change_30d'] = df['stablecoin_supply'].pct_change(30)

    # Stablecoin / BTC Market Cap ratio
    mcap_col = 'market_cap' if 'market_cap' in df.columns else ('market_cap_raw' if 'market_cap_raw' in df.columns else None)
    if mcap_col:
        df['stablecoin_btc_ratio'] = df['stablecoin_supply'] / df[mcap_col].replace(0, np.nan)
        df['stablecoin_btc_ratio'] = df['stablecoin_btc_ratio'].clip(0, 10)

    # Stablecoin dominance proxy
    df['stablecoin_ma30'] = df['stablecoin_supply'].rolling(30).mean()
    df['stablecoin_zscore'] = (df['stablecoin_supply'] - df['stablecoin_ma30']) / df['stablecoin_supply'].rolling(90).std()
    df['stablecoin_zscore'] = df['stablecoin_zscore'].clip(-5, 5)

    # Limpar coluna intermediaria
    df = df.drop(columns=['stablecoin_ma30'], errors='ignore')

    print("   Stablecoin features: OK")
    return df




# =============================================================================
# 5. ON-CHAIN DATA (Blockchain.com, BGeometrics)
# =============================================================================
def load_onchain_data():
    """
    Carrega dados on-chain (mempool, miners, SOPR, NUPL).
    Gerado por scripts/fetch_onchain_data.py
    """
    onchain_file = DATA_DIR / "onchain_data.csv"

    if not onchain_file.exists():
        print(f"   On-chain data not found: {onchain_file}")
        print("   Run: python scripts/fetch_onchain_data.py")
        return None

    df = pd.read_csv(onchain_file)
    df['date'] = pd.to_datetime(df['date'])

    print(f"   On-chain data loaded: {len(df)} rows, {len(df.columns)} columns")
    return df


def add_onchain_features(df, onchain_df):
    """
    Integra dados on-chain ao dataset principal.

    Features adicionadas:
    - mempool_tx_count, mempool_congestion
    - miners_revenue_usd, miners_revenue_ratio
    - sopr, nupl (se disponível do BGeometrics)
    - google_trend_btc (se disponível)
    """
    if onchain_df is None or len(onchain_df) == 0:
        print("   On-chain features: SKIP (no data)")
        return df

    print("\n[ON-CHAIN] Integrando dados on-chain...")

    # Colunas para adicionar
    cols_to_add = [c for c in onchain_df.columns if c != 'date' and c not in df.columns]

    if not cols_to_add:
        print("   On-chain features: SKIP (already present)")
        return df

    # Merge
    merge_cols = ['date'] + cols_to_add
    df = df.merge(onchain_df[merge_cols], on='date', how='left')

    # Forward fill para preencher gaps
    for col in cols_to_add:
        if col in df.columns:
            df[col] = df[col].ffill()

    print(f"   Added {len(cols_to_add)} on-chain features")
    return df


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def add_all_extra_features(df, fetch_external=True, include_binance=True, include_onchain=True):
    """
    Pipeline completo para adicionar todas as features extras.

    Args:
        df: DataFrame com o dataset base
        fetch_external: Se True, busca dados externos (APIs)
        include_binance: Se True, integra dados da Binance (OHLCV, etc.)
        include_onchain: Se True, integra dados on-chain (mempool, SOPR, etc.)

    Returns:
        DataFrame com features extras

    Features adicionadas:
        - BINANCE: OHLCV (High/Low!), trade count, taker volumes, basis, funding
        - Stock-to-Flow, S2F Multiple, S2F Z-score
        - Puell Multiple, Puell Extreme Low
        - Difficulty Ribbon, Ribbon Compression, Miner Capitulation
        - Stablecoin Supply e derivadas (DefiLlama API)
        - NVT Signal, NVT Golden Cross
        - ON-CHAIN: mempool, miners revenue, SOPR, NUPL (se disponível)
        - Tratamento de outliers (winsorization 5 sigma)

    FIX 2.1: REORDERED to ensure Binance price is used for calculated features.
    Previous order had S2F/NVT using Artemis price instead of Binance.
    """
    print("=" * 70)
    print("ADDING EXTRA FEATURES")
    print("=" * 70)

    original_cols = len(df.columns)

    # FIX 2.1: STEP 1 - FIRST integrate Binance (replaces price_usd with Binance close)
    # This ensures all subsequent calculations use consistent Binance pricing
    if include_binance:
        binance_df = load_binance_data()
        df = add_binance_features(df, binance_df)

        # FIX 2.2: Recalculate market_cap after price change
        if 'circulating_supply' in df.columns:
            df['market_cap'] = df['circulating_supply'] * df['price_usd']
            print("   Recalculated market_cap with Binance price")

        # FIX 2.5: Recalculate issuance_usd with consistent price
        if 'btc_issued_daily' in df.columns:
            df['issuance_usd'] = df['btc_issued_daily'] * df['price_usd']
            print("   Recalculated issuance_usd with Binance price")

        # FIX: Recalculate targets with consistent Binance price
        # target_return_5d = (price in 5 days / current price) - 1
        df['target_return_5d'] = (df['price_usd'].shift(-5) / df['price_usd']) - 1
        df['target_return_1d'] = np.log(df['price_usd'] / df['price_usd'].shift(1)).shift(-1)
        df['target_direction_1d'] = (df['target_return_1d'] > 0).astype(float)
        df['target_direction_5d'] = (df['price_usd'].shift(-5) > df['price_usd']).astype(float)
        print("   Recalculated targets with Binance price")

    # FIX 2.3: Document volume sources clearly
    if 'volume_usd' in df.columns and 'spot_volume_usd' in df.columns:
        # Keep Artemis volume as reference (aggregated from multiple exchanges)
        df['volume_usd_artemis'] = df['volume_usd'].copy()
        # Binance spot volume is in spot_volume_usd
        # Note: volume_usd will be the primary volume (Artemis aggregated)
        print("   Volume sources: volume_usd=Artemis(aggregated), spot_volume_usd=Binance")

    # STEP 2 - Treat outliers (before calculated features to avoid propagation)
    df = treat_outliers(df, sigma=5)

    # STEP 3 - Calculate derived features (S2F, Puell, Difficulty Ribbon, NVT)
    # NOW these use the correct Binance price_usd
    df = add_calculated_features(df)

    # STEP 4 - External data: Stablecoins
    if fetch_external:
        stablecoin_df = fetch_stablecoin_data()
        df = add_stablecoin_features(df, stablecoin_df)

    # STEP 5 - On-chain data (mempool, miners, SOPR, NUPL)
    if include_onchain:
        onchain_df = load_onchain_data()
        df = add_onchain_features(df, onchain_df)

    # Limpar NaNs restantes
    for col in df.columns:
        if col == 'date':
            continue
        if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                df[col] = df[col].ffill()  # forward only (no future leak); leading NaN -> fillna(0) below
                remaining = df[col].isnull().sum()
                if remaining > 0:
                    df[col] = df[col].fillna(0)

    new_cols = len(df.columns) - original_cols
    print("\n" + "=" * 70)
    print(f"DONE! Added {new_cols} new features")
    print(f"Final shape: {df.shape}")
    print("=" * 70)

    return df


# =============================================================================
# SCRIPT EXECUTION
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Add extra features to BTC dataset')
    parser.add_argument('--input', type=str, default='outputs/dataset_new.csv',
                        help='Input dataset file')
    parser.add_argument('--output', type=str, default='outputs/dataset_enhanced.csv',
                        help='Output dataset file')
    parser.add_argument('--no-external', action='store_true',
                        help='Skip external API calls (DefiLlama)')
    parser.add_argument('--no-binance', action='store_true',
                        help='Skip Binance data integration')

    args = parser.parse_args()

    # Load dataset
    input_file = BASE_DIR / args.input
    print(f"Loading dataset from: {input_file}")
    df = pd.read_csv(input_file)
    df['date'] = pd.to_datetime(df['date'])
    print(f"Loaded: {df.shape}")

    # Add features
    df = add_all_extra_features(
        df,
        fetch_external=not args.no_external,
        include_binance=not args.no_binance
    )

    # Save
    output_file = BASE_DIR / args.output
    df.to_csv(output_file, index=False)
    print(f"\nSaved to: {output_file}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Nulls: {df.isnull().sum().sum()}")

    # New features check
    new_features = {
        'Binance OHLCV': ['spot_open', 'spot_high', 'spot_low', 'spot_trade_count'],
        'Binance Taker': ['spot_taker_buy_ratio', 'taker_ratio_ma7'],
        'Binance Volatility': ['true_range', 'atr_14', 'atr_pct'],
        'Binance Candle': ['gap_pct', 'candle_body_ratio', 'upper_shadow_ratio'],
        'Binance Basis': ['basis_pct', 'basis_zscore'],
        'Binance L/S': ['global_ls_ratio', 'top_traders_ls_ratio'],
        'Stock-to-Flow': ['stock_to_flow', 's2f_multiple', 's2f_zscore'],
        'Puell': ['puell_multiple', 'puell_extreme_low'],
        'Difficulty Ribbon': ['difficulty_ribbon', 'ribbon_compression', 'miner_capitulation'],
        'Stablecoin': ['stablecoin_supply', 'stablecoin_btc_ratio', 'stablecoin_zscore'],
        'NVT': ['nvt_signal', 'nvt_golden_cross']
    }

    print("\nNew features status:")
    for category, features in new_features.items():
        present = [f for f in features if f in df.columns]
        missing = [f for f in features if f not in df.columns]
        if present:
            non_null_avg = int(np.mean([df[f].notna().sum() for f in present]))
            print(f"   {category}: {len(present)}/{len(features)} features ({non_null_avg} avg values)")
        else:
            print(f"   {category}: MISSING")
