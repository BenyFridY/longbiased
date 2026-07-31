"""
Build the production feature set (32 model features + intermediates) from raw data.
Uses the ORIGINAL functions from build_dataset.py and add_regime_features.py.

Reads raw_data.csv → computes all features → saves dataset_production.csv

Usage:
    python scripts/production/build_features.py
"""
import sys, logging, os
import numpy as np, pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "data"))

from scripts.production.config import DATASET_PATH, FEATURES_37, FEATURES_ALL

RAW_CSV = Path(__file__).parent / "data" / "raw_data.csv"

# Import ORIGINAL functions from the existing codebase
from src.features.regime.hurst_features import HurstFeatures
from src.features.regime.mean_reversion_features import MeanReversionFeatures
from src.features.regime.regime_change_features import RegimeChangeFeatures
# NOTE: TrendFeatures intentionally NOT imported — production re-implements ADX
# (calc_adx) and Aroon-Down (calc_aroon_down) locally below. The src TrendFeatures
# class is used only by the legacy scripts/data/add_regime_features.py.


# ═══════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS (aligned with build_dataset.py)
# ═══════════════════════════════════════════════════════════════

def calc_adx(high, low, close, period=14):
    """ADX — same as build_dataset.py. Returns adx only (not tuple)."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    plus_dm = high.diff()
    minus_dm = (-low.diff())
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    return dx.ewm(alpha=1/period, adjust=False).mean()


def calc_macd_histogram(prices):
    """MACD histogram — aligned: adjust=False like build_dataset.py"""
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal


def calc_bb_position(prices, window=20, num_std=2):
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return (prices - lower) / (upper - lower + 1e-10)


def calc_obv_trend(prices, volume):
    """OBV trend — aligned with build_dataset.py: z-score of 20-day OBV change."""
    # Build OBV same as build_dataset.py (iterative, price comparison)
    obv = [0]
    for i in range(1, len(prices)):
        if prices.iloc[i] > prices.iloc[i-1]:
            obv.append(obv[-1] + volume.iloc[i])
        elif prices.iloc[i] < prices.iloc[i-1]:
            obv.append(obv[-1] - volume.iloc[i])
        else:
            obv.append(obv[-1])
    obv_series = pd.Series(obv, index=prices.index)
    obv_change = obv_series.diff(20)
    obv_change_std = obv_change.rolling(60).std()
    result = (obv_change / (obv_change_std + 1e-10)).clip(-5, 5)
    return result.fillna(0)


def calc_sortino_30d(log_return_1d, return_30d):
    """Sortino — aligned with build_dataset.py (vectorized, log returns)."""
    downside = log_return_1d.where(log_return_1d < 0, 0)
    downside_dev = np.sqrt((downside ** 2).rolling(30).mean()) * np.sqrt(365)
    annualized_return_30d = return_30d * (365 / 30)
    return np.where(downside_dev > 0, annualized_return_30d / downside_dev, 0)


def calc_aroon_down(prices, window=30):
    result = pd.Series(index=prices.index, dtype=float)
    for i in range(window, len(prices)):
        w = prices.iloc[i-window:i+1]
        days_since_low = window - w.values.argmin()
        result.iloc[i] = (window - days_since_low) / window * 100
    return result


def fractional_diff(series, d=0.3, thres=0.01):
    """Lopez de Prado fractional differentiation (fixed-window).

    Stationarizes a series while preserving memory, useful for features with
    strong trends (price, fed_balance_sheet). See AFML Ch. 5.
    """
    w = [1.0]
    k = 1
    while abs(w[-1]) > thres and k < len(series):
        w.append(-w[-1] * (d - k + 1) / k)
        k += 1
    w = np.array(w[::-1])
    arr = series.values
    out = np.full(len(arr), np.nan)
    W = len(w)
    for i in range(W - 1, len(arr)):
        window = arr[i - W + 1:i + 1]
        if np.any(np.isnan(window)):
            continue
        out[i] = (w * window).sum()
    return pd.Series(out, index=series.index)


# ═══════════════════════════════════════════════════════════════
# REGIME FEATURES — using ORIGINAL src/features/regime/ classes
# ═══════════════════════════════════════════════════════════════

def build_regime_features(df, prices):
    """Build regime features using the original class implementations.

    Returns are computed locally (simple pct_change for CUSUM) to match the
    training base in scripts/data/add_regime_features.py.
    """
    log.info("  Regime features (using original implementations)...")

    hf = HurstFeatures()
    mrf = MeanReversionFeatures()
    rc = RegimeChangeFeatures()

    # ── Hurst exponent (DFA, 60-day) — aligned with add_regime_features.py ──
    log.info("    hurst_60d (DFA)...")
    df['hurst_60d'] = prices.rolling(60, min_periods=20).apply(
        lambda x: hf.calculate_hurst_dfa(pd.Series(x), max_window=30), raw=False
    )

    # ── Fractal dimension (Higuchi, 30-day) — aligned with add_regime_features.py ──
    log.info("    fractal_dimension_30d (Higuchi)...")
    df['fractal_dimension_30d'] = prices.rolling(30, min_periods=10).apply(
        lambda x: hf.calculate_fractal_dimension(pd.Series(x), window=len(x)), raw=False
    )

    # ── Mean reversion features — scalars and dicts ──
    log.info("    half_life_60d, ou_theta_60d, kpss_stat_30d, mr_score_30d...")

    df['half_life_60d'] = prices.rolling(60, min_periods=30).apply(
        lambda x: mrf.calculate_half_life(pd.Series(x)), raw=False
    )
    df['ou_theta_60d'] = prices.rolling(60, min_periods=30).apply(
        lambda x: mrf.calculate_ou_parameters(pd.Series(x))['theta'], raw=False
    )
    # kpss_stat_30d — rolling(50, min_periods=30) like add_regime_features.py
    df['kpss_stat_30d'] = prices.rolling(50, min_periods=30).apply(
        lambda x: mrf.calculate_kpss_statistic(pd.Series(x)), raw=False
    )
    # mr_score_30d — rolling(50, min_periods=30) like add_regime_features.py
    df['mr_score_30d'] = prices.rolling(50, min_periods=30).apply(
        lambda x: mrf.calculate_mean_reversion_score(pd.Series(x), window=len(x)), raw=False
    )

    # ── CUSUM (rolling 30-day) — returns dict ──
    # TRAIN/SERVE FIX: use SIMPLE returns (prices.pct_change()) to match the
    # training base (scripts/data/add_regime_features.py:234 uses pct_change).
    # This previously used log_returns, creating (a) a train/serve skew on the
    # cusum_pos/cusum_neg features (which are top model features) and (b) a
    # discontinuity at the enhanced-base -> build_features seam (2026-03-03).
    log.info("    cusum_pos, cusum_neg...")
    cusum_returns = prices.pct_change()
    df['cusum_pos'] = cusum_returns.rolling(30, min_periods=10).apply(
        lambda x: rc.calculate_cusum(pd.Series(x))['cusum_pos'], raw=False
    )
    df['cusum_neg'] = cusum_returns.rolling(30, min_periods=10).apply(
        lambda x: rc.calculate_cusum(pd.Series(x))['cusum_neg'], raw=False
    )

    # ── Structural break score (100-day window, 50-day inner) — aligned with add_regime_features.py ──
    log.info("    structural_break_score...")
    df['structural_break_score'] = prices.rolling(100, min_periods=50).apply(
        lambda x: rc.calculate_structural_break_score(pd.Series(x), window=50), raw=False
    )

    # ── Velocity (nested 40-day window, 10-day inner) — returns dict ──
    log.info("    velocity...")
    df['velocity'] = prices.rolling(40, min_periods=20).apply(
        lambda x: rc.detect_acceleration(pd.Series(x), window=10)['velocity'], raw=False
    )

    # ── Trend strength: adx * hurst_30d — aligned with add_regime_features.py ──
    log.info("    trend_strength (adx * hurst_30d)...")
    hurst_30d = prices.rolling(30, min_periods=10).apply(
        lambda x: hf.calculate_hurst_dfa(pd.Series(x), max_window=15), raw=False
    )
    df['trend_strength'] = df['adx'] * hurst_30d.fillna(0.5)

    # ── vol_x_regime_duration — replicate build_dataset.py classify_regime + regime_v3 ──
    log.info("    vol_x_regime_duration...")
    vol_30d = df['volatility_30d']  # already annualized

    # Replicate classify_regime from build_dataset.py
    # Uses: vix (not available in prod), vol_30d, return_30d, adx, hurst_30d
    # Since we don't have vix, use vol_30d > 0.9 and return_30d < -0.20 for crisis
    return_30d = np.log(prices / prices.shift(30))  # log returns like build_dataset
    hurst_30d_vals = hurst_30d.fillna(0.5)  # from earlier calculation

    regime_raw = pd.Series(0, index=df.index)  # default: neutral
    for i in range(len(df)):
        v30 = vol_30d.iloc[i] if not pd.isna(vol_30d.iloc[i]) else 0.5
        r30 = return_30d.iloc[i] if not pd.isna(return_30d.iloc[i]) else 0.0
        a = df['adx'].iloc[i] if not pd.isna(df['adx'].iloc[i]) else 20.0
        h = hurst_30d_vals.iloc[i] if not pd.isna(hurst_30d_vals.iloc[i]) else 0.5

        # Crisis: extreme volatility OR rapid decline (skip VIX since not available)
        if (v30 > 0.9) or (r30 < -0.20):
            regime_raw.iloc[i] = 4
        elif a > 30:
            if r30 > 0.05:
                regime_raw.iloc[i] = 2
            elif r30 < -0.05:
                regime_raw.iloc[i] = -2
        elif a > 20:
            if r30 > 0.02:
                regime_raw.iloc[i] = 1
            elif r30 < -0.02:
                regime_raw.iloc[i] = -1
        elif (a < 20) and (h < 0.45):
            regime_raw.iloc[i] = -3

    # regime_v3 = smoothed (rolling median of 5, no center)
    regime_v3 = regime_raw.rolling(5, min_periods=1, center=False).median().round().astype(int)

    # Regime duration: consecutive days in same regime
    regime_change = (regime_v3 != regime_v3.shift(1))
    regime_group = regime_change.cumsum()
    regime_duration = regime_group.groupby(regime_group).cumcount() + 1
    df['vol_x_regime_duration'] = vol_30d * regime_duration

    return df


# ═══════════════════════════════════════════════════════════════
# MAIN: build the 32 model features (+ intermediates)
# ═══════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the production feature set (32 model features + intermediates)."""
    log.info(f"Building features from {len(df)} rows...")

    prices = df['price_usd']
    volume = df['volume_usd']

    # Forward fill sparse raw data (miners_revenue, open_interest, etc.)
    for col in ['miners_revenue_usd', 'open_interest', 'm2_supply', 'fed_balance_sheet']:
        if col in df.columns:
            df[col] = df[col].ffill()

    # ── Returns — use LOG RETURNS like build_dataset.py ──
    log.info("  Returns (log)...")
    df['log_return_1d'] = np.log(prices / prices.shift(1))
    simple_returns = prices.pct_change()  # still needed for some calcs

    # ── Volatility — annualized, using log returns, like build_dataset.py ──
    log.info("  Volatility (annualized)...")
    for period in [7, 30, 60]:
        df[f'volatility_{period}d'] = df['log_return_1d'].rolling(period).std() * np.sqrt(365)

    # ── Technical indicators ──
    log.info("  Technical indicators...")
    df['adx'] = calc_adx(df['price_high'], df['price_low'], prices)
    df['macd_histogram'] = calc_macd_histogram(prices)
    df['bb_position'] = calc_bb_position(prices)
    df['obv_trend'] = calc_obv_trend(prices, volume)
    df['volume_sma20_ratio'] = volume / volume.rolling(20).mean()
    return_30d = np.log(prices / prices.shift(30))  # log return like build_dataset
    df['sortino_30d'] = calc_sortino_30d(df['log_return_1d'], return_30d)
    df['price_percentile_1y'] = prices.rolling(365).rank(pct=True)
    df['aroon_down_30d'] = calc_aroon_down(prices)

    # ── Basis (futures - spot) ──
    log.info("  Basis...")
    if 'futures_close' in df.columns:
        df['basis_pct'] = (df['futures_close'] - prices) / (prices + 1e-10) * 100
        df['basis_ma7'] = df['basis_pct'].rolling(7).mean()
    else:
        df['basis_pct'] = 0.0
        df['basis_ma7'] = 0.0

    # ── yfinance derived — NO *100 (build_dataset.py uses plain pct_change) ──
    log.info("  yfinance derived...")
    if 'eth' in df.columns:
        df['eth_btc_ratio'] = df['eth'] / (prices + 1e-10)
        df['eth_pctchg_30d'] = df['eth'].pct_change(30)     # NO *100
    if 'copper' in df.columns:
        df['copper_return_30d'] = df['copper'].pct_change(30)  # NO *100
    if 'gold' in df.columns:
        gold_ret = df['gold'].pct_change()
        btc_ret = simple_returns
        df['btc_gold_corr_30d'] = btc_ret.rolling(30).corr(gold_ret)

    # ── FRED derived — aligned with build_dataset.py (V25 FIX: 252 business days) ──
    log.info("  FRED derived...")
    if 'm2_supply' in df.columns:
        # V25 FIX: pct_change(252) for real YoY on daily ffilled data
        df['m2_supply_pctchg_90d'] = df['m2_supply'].pct_change(90)
        df['m2_yoy_growth'] = (1 + df['m2_supply_pctchg_90d']) ** (365.0/90.0) - 1
    if 'fed_balance_sheet' in df.columns:
        # V25 NEW: fed_bs_yoy_change with real YoY (252 business days)
        df['fed_bs_yoy_change'] = df['fed_balance_sheet'].pct_change(252)

    # ── V25 NEW: Fractional differentiation (stationarize with memory) ──
    log.info("  Fractional differentiation (d=0.3)...")
    df['price_fracdiff_05'] = fractional_diff(np.log(prices), d=0.3)
    if 'fed_balance_sheet' in df.columns:
        fed_log = np.log(df['fed_balance_sheet'].replace(0, np.nan))
        df['fed_fracdiff_05'] = fractional_diff(fed_log, d=0.3)

    # ── On-chain derived ──
    log.info("  On-chain derived...")
    if 'nupl' in df.columns:
        df['nupl_ma30'] = df['nupl'].rolling(30).mean()
    if 'miners_revenue_usd' in df.columns:
        df['miners_revenue_ratio'] = df['miners_revenue_usd'] / df['miners_revenue_usd'].rolling(30).mean()
    if 'stablecoin_supply' in df.columns:
        sc = df['stablecoin_supply']
        # Aligned with add_extra_features.py: ma30 for mean, rolling(90) for std
        sc_ma30 = sc.rolling(30).mean()
        sc_std90 = sc.rolling(90, min_periods=30).std()
        df['stablecoin_zscore'] = ((sc - sc_ma30) / (sc_std90 + 1e-10)).clip(-5, 5)
        df['stablecoin_supply_change_30d'] = sc.pct_change(30)  # NO *100
    if 'hash_rate' in df.columns:
        df['hash_rate_pctchg_30d'] = df['hash_rate'].pct_change(30)  # NO *100

    # ── Regime features (original implementations) ──
    df = build_regime_features(df, prices)

    # ── V36/E1: new on-chain features (median-fill pre-history) ──
    log.info("  V36 on-chain features (median-fill)...")
    def _median_fill(series, first_n=30):
        """Fill NaN pre-history with median of first N available days."""
        s = pd.Series(series.values if hasattr(series, 'values') else series)
        first_valid = s.first_valid_index()
        if first_valid is None:
            return s.fillna(0).values
        available = s.loc[first_valid:].dropna().iloc[:first_n]
        fill_val = available.median() if len(available) > 0 else 0
        return s.fillna(fill_val).values
    for col in ['reserveRisk', 'puellMultiple', 'funding_rate_mean']:
        if col in df.columns:
            df[col] = _median_fill(df[col])
    # funding_rate_ma7 computed from funding_rate_mean (7d rolling mean)
    if 'funding_rate_mean' in df.columns:
        df['funding_rate_ma7'] = df['funding_rate_mean'].rolling(7, min_periods=1).mean()

    # ── Fill NaN ──
    for col in FEATURES_ALL:
        if col in df.columns:
            df[col] = df[col].ffill().fillna(0)

    # Verify
    missing = [f for f in FEATURES_ALL if f not in df.columns]
    if missing:
        log.warning(f"  MISSING features: {missing}")
        for f in missing:
            df[f] = 0.0

    present = [f for f in FEATURES_ALL if f in df.columns]
    log.info(f"  {len(present)}/{len(FEATURES_ALL)} features built")

    return df


def main():
    """Build features from raw data. NOTE: For production use bootstrap_from_original.py instead.
    This script is called internally by bootstrap and should NOT be run standalone
    unless you want to rebuild from scratch (which loses enhanced dataset quality)."""
    if not RAW_CSV.exists():
        log.error(f"Raw data not found: {RAW_CSV}")
        log.error("Run fetch_raw_data.py first")
        return

    # Safety: warn if overwriting a bootstrapped dataset
    if DATASET_PATH.exists():
        existing = pd.read_csv(DATASET_PATH, nrows=2)
        if existing['date'].iloc[0] == '2019-01-01':
            log.warning("!" * 60)
            log.warning("  WARNING: dataset_production.csv appears to be bootstrapped")
            log.warning("  (starts 2019-01-01 = enhanced base).")
            log.warning("  Running build_features.py directly will OVERWRITE it with")
            log.warning("  raw-only data (lower quality). Use bootstrap_from_original.py")
            log.warning("  instead for production. Saving to dataset_raw_built.csv.")
            log.warning("!" * 60)
            out_path = DATASET_PATH.parent / "dataset_raw_built.csv"
        else:
            out_path = DATASET_PATH
    else:
        out_path = DATASET_PATH

    df = pd.read_csv(RAW_CSV)
    log.info(f"Loaded raw data: {len(df)} rows, {len(df.columns)} columns")

    df = build_features(df)

    df.to_csv(out_path, index=False)
    log.info(f"Saved: {out_path} ({len(df)} rows, {len(df.columns)} columns)")

    log.info(f"\n{len(FEATURES_37)} model features status:")
    for f in FEATURES_37:
        if f in df.columns:
            valid = df[f].notna().sum()
            log.info(f"  ✓ {f:<30} {valid}/{len(df)} valid")
        else:
            log.info(f"  ✗ {f:<30} MISSING")


if __name__ == '__main__':
    main()
