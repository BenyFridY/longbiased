"""
ADD REGIME FEATURES - Features Sofisticadas de Regime de Mercado
=================================================================

Este script adiciona features sofisticadas de regime ao dataset usando
os modulos de src/features/regime/:

- Hurst DFA (Detrended Fluctuation Analysis) - mais robusto que R/S
- Ornstein-Uhlenbeck Parameters (theta, mu, sigma)
- Half-life de mean reversion
- CUSUM para deteccao de mudancas
- Structural Break Score
- Trend Strength e Aroon
- ADF/KPSS tests para estacionariedade

Features adicionadas (~20 novas):
- hurst_dfa_30d, hurst_dfa_60d
- variance_ratio_30d
- fractal_dimension_30d
- ou_theta_30d, ou_theta_60d
- ou_mu_30d, ou_sigma_30d
- half_life_30d, half_life_60d
- adf_stat_30d, adf_pvalue_30d
- kpss_stat_30d
- mr_score_30d (mean reversion score 0-100)
- cusum_pos, cusum_neg, cusum_score
- structural_break_score
- acceleration, velocity
- days_since_regime_change
- trend_strength_30d
- aroon_up_30d, aroon_down_30d
- linear_r2_30d

Uso:
    python scripts/add_regime_features.py

Ou importar funcoes individuais:
    from add_regime_features import add_all_regime_features
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import sys
import os

warnings.filterwarnings('ignore')

# =============================================================================
# PATH SETUP - Add src directory to path for imports
# =============================================================================
BASE_DIR = Path(__file__).parent.parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

# =============================================================================
# IMPORT REGIME FEATURE CALCULATORS
# =============================================================================
try:
    from features.regime.hurst_features import HurstFeatures
    from features.regime.mean_reversion_features import MeanReversionFeatures
    from features.regime.regime_change_features import RegimeChangeFeatures
    from features.regime.trend_features import TrendFeatures
    REGIME_MODULES_AVAILABLE = True
except ImportError as e:
    REGIME_MODULES_AVAILABLE = False
    print(f"WARNING: Could not import regime modules: {e}")
    print("Regime features will be skipped.")


# =============================================================================
# SIMPLIFIED FEATURE EXTRACTION
# =============================================================================
def add_hurst_features(df: pd.DataFrame, price_col: str = 'price_usd') -> pd.DataFrame:
    """
    Adiciona features de Hurst usando modulo sofisticado.

    Features:
    - hurst_dfa_30d, hurst_dfa_60d (DFA - mais robusto)
    - variance_ratio_30d
    - fractal_dimension_30d
    """
    if price_col not in df.columns:
        print(f"   WARNING: {price_col} not found. Skipping Hurst features.")
        return df

    print("   Adding Hurst features...")
    hurst = HurstFeatures()

    prices = df[price_col]

    # FIX 2.4: Add min_periods to all rolling windows to avoid excessive NaNs
    # min_periods = max(10, window//3) is a reasonable default

    # Hurst DFA - 30 dias
    print("      - hurst_dfa_30d")
    df['hurst_dfa_30d'] = prices.rolling(30, min_periods=10).apply(
        lambda x: hurst.calculate_hurst_dfa(pd.Series(x), max_window=15),
        raw=False
    )

    # Hurst DFA - 60 dias
    print("      - hurst_dfa_60d")
    df['hurst_dfa_60d'] = prices.rolling(60, min_periods=20).apply(
        lambda x: hurst.calculate_hurst_dfa(pd.Series(x), max_window=30),
        raw=False
    )

    # Variance Ratio - 30 dias
    print("      - variance_ratio_30d")
    df['variance_ratio_30d'] = prices.rolling(30, min_periods=10).apply(
        lambda x: hurst.calculate_variance_ratio(pd.Series(x), lag=8),
        raw=False
    )

    # Fractal Dimension - 30 dias
    print("      - fractal_dimension_30d")
    df['fractal_dimension_30d'] = prices.rolling(30, min_periods=10).apply(
        lambda x: hurst.calculate_fractal_dimension(pd.Series(x), window=len(x)),
        raw=False
    )

    return df


def add_mean_reversion_features(df: pd.DataFrame, price_col: str = 'price_usd') -> pd.DataFrame:
    """
    Adiciona features de Mean Reversion usando modulo sofisticado.

    Features:
    - ou_theta_30d, ou_theta_60d (velocidade de reversao)
    - ou_mu_30d, ou_sigma_30d
    - half_life_30d, half_life_60d
    - adf_stat_30d, adf_pvalue_30d
    - kpss_stat_30d
    - mr_score_30d (score composto 0-100)
    """
    if price_col not in df.columns:
        print(f"   WARNING: {price_col} not found. Skipping MR features.")
        return df

    print("   Adding Mean Reversion features...")
    mr = MeanReversionFeatures()

    prices = df[price_col]

    # FIX 2.4: Add min_periods to all rolling windows

    # OU Theta - 30 dias (velocidade de reversao)
    print("      - ou_theta_30d")
    df['ou_theta_30d'] = prices.rolling(30, min_periods=20).apply(
        lambda x: mr.calculate_ou_parameters(pd.Series(x))['theta'],
        raw=False
    )

    # OU Theta - 60 dias
    print("      - ou_theta_60d")
    df['ou_theta_60d'] = prices.rolling(60, min_periods=30).apply(
        lambda x: mr.calculate_ou_parameters(pd.Series(x))['theta'],
        raw=False
    )

    # OU Mu - 30 dias (nivel de equilibrio)
    print("      - ou_mu_30d")
    df['ou_mu_30d'] = prices.rolling(30, min_periods=20).apply(
        lambda x: mr.calculate_ou_parameters(pd.Series(x))['mu'],
        raw=False
    )

    # OU Sigma - 30 dias (volatilidade OU)
    print("      - ou_sigma_30d")
    df['ou_sigma_30d'] = prices.rolling(30, min_periods=20).apply(
        lambda x: mr.calculate_ou_parameters(pd.Series(x))['sigma'],
        raw=False
    )

    # Half-life - 30 dias
    print("      - half_life_30d")
    df['half_life_30d'] = prices.rolling(30, min_periods=20).apply(
        lambda x: mr.calculate_half_life(pd.Series(x)),
        raw=False
    )

    # Half-life - 60 dias
    print("      - half_life_60d")
    df['half_life_60d'] = prices.rolling(60, min_periods=30).apply(
        lambda x: mr.calculate_half_life(pd.Series(x)),
        raw=False
    )

    # ADF Statistic - 30 dias
    print("      - adf_stat_30d")
    df['adf_stat_30d'] = prices.rolling(50, min_periods=30).apply(
        lambda x: mr.calculate_adf_statistic(pd.Series(x)),
        raw=False
    )

    # KPSS Statistic - 30 dias
    print("      - kpss_stat_30d")
    df['kpss_stat_30d'] = prices.rolling(50, min_periods=30).apply(
        lambda x: mr.calculate_kpss_statistic(pd.Series(x)),
        raw=False
    )

    # Mean Reversion Score - 30 dias (composto)
    print("      - mr_score_30d")
    df['mr_score_30d'] = prices.rolling(50, min_periods=30).apply(
        lambda x: mr.calculate_mean_reversion_score(pd.Series(x), window=len(x)),
        raw=False
    )

    return df


def add_regime_change_features(df: pd.DataFrame, price_col: str = 'price_usd') -> pd.DataFrame:
    """
    Adiciona features de deteccao de mudanca de regime.

    Features:
    - cusum_pos, cusum_neg, cusum_score
    - structural_break_score
    - acceleration, velocity
    - days_since_regime_change
    """
    if price_col not in df.columns:
        print(f"   WARNING: {price_col} not found. Skipping Regime Change features.")
        return df

    print("   Adding Regime Change features...")
    rc = RegimeChangeFeatures()

    prices = df[price_col]
    returns = prices.pct_change()

    # CUSUM features - 30 dias
    print("      - cusum_pos, cusum_neg, cusum_score")

    def get_cusum_pos(x):
        result = rc.calculate_cusum(pd.Series(x))
        return result['cusum_pos']

    def get_cusum_neg(x):
        result = rc.calculate_cusum(pd.Series(x))
        return result['cusum_neg']

    # FIX 2.4: Add min_periods
    df['cusum_pos'] = returns.rolling(30, min_periods=10).apply(get_cusum_pos, raw=False)
    df['cusum_neg'] = returns.rolling(30, min_periods=10).apply(get_cusum_neg, raw=False)

    # CUSUM Score (combinado)
    df['cusum_score'] = df['cusum_pos'].abs() + df['cusum_neg'].abs()

    # Structural Break Score - 50 dias
    # FIX 2.4: Add min_periods
    print("      - structural_break_score")
    df['structural_break_score'] = prices.rolling(100, min_periods=50).apply(
        lambda x: rc.calculate_structural_break_score(pd.Series(x), window=50),
        raw=False
    )

    # Acceleration e Velocity - 20 dias
    print("      - velocity, acceleration")

    def get_velocity(x):
        result = rc.detect_acceleration(pd.Series(x), window=10)
        return result['velocity']

    def get_acceleration(x):
        result = rc.detect_acceleration(pd.Series(x), window=10)
        return result['acceleration']

    # FIX 2.4: Add min_periods
    df['velocity'] = prices.rolling(40, min_periods=20).apply(get_velocity, raw=False)
    df['acceleration'] = prices.rolling(40, min_periods=20).apply(get_acceleration, raw=False)

    # Days since significant change
    print("      - days_since_regime_change")
    threshold = returns.rolling(30).std() * 2
    significant_changes = returns.abs() > threshold

    days_counter = []
    counter = 0
    for change in significant_changes:
        if pd.isna(change):
            days_counter.append(np.nan)
        elif change:
            counter = 0
            days_counter.append(counter)
        else:
            counter += 1
            days_counter.append(counter)

    df['days_since_regime_change'] = days_counter

    return df


def add_trend_features(df: pd.DataFrame, price_col: str = 'price_usd') -> pd.DataFrame:
    """
    Adiciona features de tendencia.

    Features:
    - trend_strength_30d (0-100)
    - aroon_up_30d, aroon_down_30d
    - linear_r2_30d
    """
    if price_col not in df.columns:
        print(f"   WARNING: {price_col} not found. Skipping Trend features.")
        return df

    print("   Adding Trend features...")
    tf = TrendFeatures()

    prices = df[price_col]

    # Trend Strength - 30 dias
    # FIX 2.4: Add min_periods
    print("      - trend_strength_30d")
    df['trend_strength_30d'] = prices.rolling(30, min_periods=10).apply(
        lambda x: tf.calculate_trend_strength(pd.Series(x), window=len(x)),
        raw=False
    )

    # Aroon indicators - 30 dias
    print("      - aroon_up_30d, aroon_down_30d")
    aroon_df = tf.calculate_aroon(prices, period=30)
    df['aroon_up_30d'] = aroon_df['aroon_up'] if 'aroon_up' in aroon_df.columns else np.nan
    df['aroon_down_30d'] = aroon_df['aroon_down'] if 'aroon_down' in aroon_df.columns else np.nan

    # Linear R2 - 30 dias
    # FIX 2.4: Add min_periods
    print("      - linear_r2_30d")
    df['linear_r2_30d'] = prices.rolling(30, min_periods=10).apply(
        lambda x: tf.calculate_linear_trend(pd.Series(x))['r2'],
        raw=False
    )

    return df


# =============================================================================
# MAIN FUNCTION
# =============================================================================
def add_all_regime_features(df: pd.DataFrame, price_col: str = 'price_usd') -> pd.DataFrame:
    """
    Adiciona TODAS as features de regime ao DataFrame.

    Args:
        df: DataFrame com dados (deve ter coluna de preco)
        price_col: Nome da coluna de preco (default: 'price_usd')

    Returns:
        DataFrame com novas features adicionadas
    """
    if not REGIME_MODULES_AVAILABLE:
        print("WARNING: Regime modules not available. Skipping all regime features.")
        return df

    initial_cols = len(df.columns)

    print("\n" + "=" * 70)
    print("ADDING REGIME FEATURES (from src/features/regime/)")
    print("=" * 70)

    # 1. Hurst Features (4 features)
    df = add_hurst_features(df, price_col)

    # 2. Mean Reversion Features (10 features)
    df = add_mean_reversion_features(df, price_col)

    # 3. Regime Change Features (6 features)
    df = add_regime_change_features(df, price_col)

    # 4. Trend Features (4 features)
    df = add_trend_features(df, price_col)

    # Summary
    new_cols = len(df.columns) - initial_cols
    print(f"\n   Total new features added: {new_cols}")

    # Lista features adicionadas
    regime_features = [
        'hurst_dfa_30d', 'hurst_dfa_60d', 'variance_ratio_30d', 'fractal_dimension_30d',
        'ou_theta_30d', 'ou_theta_60d', 'ou_mu_30d', 'ou_sigma_30d',
        'half_life_30d', 'half_life_60d', 'adf_stat_30d', 'kpss_stat_30d', 'mr_score_30d',
        'cusum_pos', 'cusum_neg', 'cusum_score', 'structural_break_score',
        'velocity', 'acceleration', 'days_since_regime_change',
        'trend_strength_30d', 'aroon_up_30d', 'aroon_down_30d', 'linear_r2_30d'
    ]

    present = [f for f in regime_features if f in df.columns]
    missing = [f for f in regime_features if f not in df.columns]

    print(f"   Features present: {len(present)}/{len(regime_features)}")
    if missing:
        print(f"   Features missing: {missing}")

    return df


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("ADD REGIME FEATURES - Standalone Mode")
    print("=" * 70)

    # Load existing dataset
    OUTPUT_DIR = BASE_DIR / "outputs"
    INPUT_FILE = OUTPUT_DIR / "dataset_final.csv"

    if not INPUT_FILE.exists():
        print(f"ERROR: Dataset not found at {INPUT_FILE}")
        print("Run build_dataset.py first to create the base dataset.")
        sys.exit(1)

    print(f"\nLoading dataset from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded: {df.shape}")

    # Add regime features
    df = add_all_regime_features(df, price_col='price_usd')

    # Save
    OUTPUT_FILE = OUTPUT_DIR / "dataset_final_with_regime.csv"
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to: {OUTPUT_FILE}")
    print(f"Final shape: {df.shape}")

    # Validation
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    regime_features = [
        'hurst_dfa_30d', 'ou_theta_30d', 'cusum_score', 'trend_strength_30d'
    ]

    for feat in regime_features:
        if feat in df.columns:
            valid = df[feat].notna().sum()
            total = len(df)
            pct = valid / total * 100
            print(f"   {feat}: {valid}/{total} valid ({pct:.1f}%)")

    print("\nDone!")
