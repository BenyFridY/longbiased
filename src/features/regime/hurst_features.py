"""
Hurst Exponent e Features Fractais para Detecção de Regime
Identifica se o mercado está em tendência (H > 0.5) ou mean reverting (H < 0.5)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class HurstFeatures:
    """
    Calcula Hurst Exponent e métricas relacionadas para identificar
    se o mercado está em regime de tendência ou mean reversion
    """
    
    def __init__(self):
        """Inicializa calculadora de features Hurst"""
        self.feature_names = []
        
    def calculate_hurst_rs(self, prices: pd.Series, max_lag: int = 100) -> float:
        """
        Calcula Hurst Exponent usando R/S Analysis (Rescaled Range) sobre LOG RETURNS.

        H > 0.5: Série persistente (trend following)
        H = 0.5: Random walk
        H < 0.5: Série anti-persistente (mean reverting)

        FIXED: Now uses log returns instead of raw prices.

        Args:
            prices: Série de preços
            max_lag: Lag máximo para análise

        Returns:
            Hurst exponent (clipped to 0-1 range)
        """
        if len(prices) < max_lag:
            return np.nan

        # CRITICAL FIX: Convert prices to LOG RETURNS first
        log_returns = np.log(prices / prices.shift(1)).dropna().values

        if len(log_returns) < 10:
            return np.nan

        lags = range(2, min(max_lag, len(log_returns)//2))
        rs_values_per_lag = []

        for lag in lags:
            # Divide série em blocos
            n_blocks = len(log_returns) // lag
            if n_blocks < 1:
                continue

            rs_values = []

            for i in range(n_blocks):
                block = log_returns[i*lag:(i+1)*lag]
                if len(block) < 2:
                    continue

                # Calcula desvios da média
                mean = np.mean(block)
                deviations = block - mean
                cumsum = np.cumsum(deviations)

                # Range
                R = np.max(cumsum) - np.min(cumsum)

                # Standard deviation
                S = np.std(block, ddof=1)

                if S > 0:
                    rs_values.append(R / S)

            if rs_values:
                rs_values_per_lag.append(np.mean(rs_values))

        if len(rs_values_per_lag) < 2:
            return np.nan

        # Regressão log-log
        log_lags = np.log(list(lags[:len(rs_values_per_lag)]))
        log_rs = np.log(rs_values_per_lag)

        # Remove valores inválidos
        mask = np.isfinite(log_lags) & np.isfinite(log_rs)
        if np.sum(mask) < 2:
            return np.nan

        log_lags = log_lags[mask]
        log_rs = np.array(log_rs)[mask]

        # Hurst é o slope da regressão
        poly = np.polyfit(log_lags, log_rs, 1)
        hurst = np.clip(poly[0], 0, 1)  # Clip to valid range

        return hurst
    
    def calculate_hurst_dfa(self, prices: pd.Series, min_window: int = 4, 
                            max_window: int = None) -> float:
        """
        Calcula Hurst usando Detrended Fluctuation Analysis (DFA)
        Mais robusto que R/S para séries não-estacionárias
        
        Args:
            prices: Série de preços
            min_window: Janela mínima
            max_window: Janela máxima
            
        Returns:
            Hurst exponent via DFA
        """
        if max_window is None:
            max_window = len(prices) // 4
            
        if len(prices) < max_window:
            return np.nan
            
        # Trabalha com log returns
        log_returns = np.log(prices / prices.shift(1)).dropna().values
        
        if len(log_returns) < max_window:
            return np.nan
            
        # Profile (cumulative sum)
        y = np.cumsum(log_returns - np.mean(log_returns))
        
        # Range de janelas
        windows = np.unique(np.logspace(np.log10(min_window), 
                                       np.log10(min(max_window, len(y)//2)), 
                                       50).astype(int))
        
        fluctuations = []
        
        for window in windows:
            if window < 4 or window > len(y)//2:
                continue
                
            # Divide em blocos
            n_blocks = len(y) // window
            if n_blocks < 1:
                continue
                
            f_values = []
            
            for i in range(n_blocks):
                block = y[i*window:(i+1)*window]
                if len(block) < 2:
                    continue
                    
                # Detrend usando polinômio linear
                x = np.arange(len(block))
                poly = np.polyfit(x, block, 1)
                fit = np.polyval(poly, x)
                
                # Fluctuation
                f = np.sqrt(np.mean((block - fit)**2))
                f_values.append(f)
            
            if f_values:
                fluctuations.append((window, np.mean(f_values)))
        
        if len(fluctuations) < 2:
            return np.nan

        # FIX 3.1: Filter fluctuations FIRST to ensure array length consistency
        # Both windows_log and fluct_log must have the same length
        valid_fluctuations = [(w, f) for w, f in fluctuations if f > 0]
        if len(valid_fluctuations) < 2:
            return np.nan

        # Regressão log-log - now arrays are guaranteed to have same length
        windows_log = np.log([f[0] for f in valid_fluctuations])
        fluct_log = np.log([f[1] for f in valid_fluctuations])

        # Hurst é o slope
        poly = np.polyfit(windows_log, fluct_log, 1)
        hurst_dfa = poly[0]
        
        return hurst_dfa
    
    def calculate_variance_ratio(self, prices: pd.Series, lag: int = 16) -> float:
        """
        Variance Ratio Test
        VR = 1: Random walk
        VR > 1: Trending (positive autocorrelation)
        VR < 1: Mean reverting (negative autocorrelation)
        
        Args:
            prices: Série de preços
            lag: Período para cálculo
            
        Returns:
            Variance ratio
        """
        if len(prices) < lag * 2:
            return np.nan
            
        returns = prices.pct_change().dropna()
        
        if len(returns) < lag * 2:
            return np.nan
            
        # Variância de retornos de 1 período
        var1 = returns.var()
        
        # Variância de retornos de k períodos
        returns_k = prices.pct_change(lag).dropna()
        vark = returns_k.var() / lag
        
        if var1 == 0:
            return np.nan
            
        vr = vark / var1
        
        return vr
    
    def calculate_fractal_dimension(self, prices: pd.Series, window: int = 30) -> float:
        """
        Dimensão Fractal usando método Higuchi (mais robusto que box-counting).

        FD ≈ 1.5: Random walk
        FD < 1.5: Trending (smoother)
        FD > 1.5: Mean reverting/choppy (rougher)

        FIXED: Now uses Higuchi's algorithm which is more suitable for time series.

        Args:
            prices: Série de preços
            window: Janela para cálculo

        Returns:
            Dimensão fractal (typically between 1.0 and 2.0)
        """
        if len(prices) < window:
            return np.nan

        # Work with log returns for stationarity
        log_returns = np.log(prices / prices.shift(1)).dropna().values

        if len(log_returns) < 10:
            return np.nan

        N = len(log_returns)

        # Higuchi's method
        k_max = min(8, N // 4)
        if k_max < 2:
            return np.nan

        lk = []
        for k in range(1, k_max + 1):
            Lmk = []
            for m in range(1, k + 1):
                # Create subsampled series
                idx = np.arange(m - 1, N, k)
                if len(idx) < 2:
                    continue

                x_m = log_returns[idx]

                # Calculate length
                length = 0
                for i in range(1, len(x_m)):
                    length += abs(x_m[i] - x_m[i-1])

                # Normalize
                # FIX 3.2: Correct Higuchi normalization formula
                # Lm = (length * (N - 1)) / (n_intervals * k^2)
                # Previous code had an extra 'k' factor
                n_intervals = (N - 1) // k
                if n_intervals > 0:
                    Lm = (length * (N - 1)) / (n_intervals * k * k)
                    Lmk.append(Lm)

            if Lmk:
                lk.append(np.mean(Lmk))

        if len(lk) < 2:
            return np.nan

        # Log-log regression
        log_k = np.log(np.arange(1, len(lk) + 1))
        log_lk = np.log(np.array(lk) + 1e-10)

        # Remove invalid values
        mask = np.isfinite(log_k) & np.isfinite(log_lk)
        if np.sum(mask) < 2:
            return np.nan

        # Fractal dimension is the negative slope
        poly = np.polyfit(log_k[mask], log_lk[mask], 1)
        fractal_dim = -poly[0]

        # Clip to reasonable range
        return np.clip(fractal_dim, 1.0, 2.0)
    
    def calculate_all_features(self, df: pd.DataFrame, price_col: str = 'price',
                             windows: List[int] = None) -> pd.DataFrame:
        """
        Calcula todas as features de Hurst/Fractal
        
        Args:
            df: DataFrame com dados
            price_col: Coluna de preços
            windows: Lista de janelas para cálculo
            
        Returns:
            DataFrame com features calculadas
        """
        if windows is None:
            windows = [20, 30, 50, 100, 200]
        
        result = df.copy()
        
        for window in windows:
            if len(df) < window:
                continue
                
            # Hurst R/S
            result[f'hurst_rs_{window}'] = df[price_col].rolling(window).apply(
                lambda x: self.calculate_hurst_rs(pd.Series(x), max_lag=window//2)
            )
            
            # Hurst DFA
            result[f'hurst_dfa_{window}'] = df[price_col].rolling(window).apply(
                lambda x: self.calculate_hurst_dfa(pd.Series(x), max_window=window//2)
            )
            
            # Variance Ratio
            for lag in [2, 4, 8, 16]:
                if lag < window:
                    result[f'variance_ratio_{lag}_{window}'] = df[price_col].rolling(window).apply(
                        lambda x: self.calculate_variance_ratio(pd.Series(x), lag=lag)
                    )
            
            # Fractal Dimension
            result[f'fractal_dim_{window}'] = df[price_col].rolling(window).apply(
                lambda x: self.calculate_fractal_dimension(pd.Series(x), window=len(x))
            )
            
            # Interpretação do Hurst
            result[f'hurst_regime_{window}'] = result[f'hurst_rs_{window}'].apply(
                lambda h: 'trending' if h > 0.55 else ('mean_reverting' if h < 0.45 else 'random_walk')
                if not pd.isna(h) else np.nan
            )
            
            # Força do regime
            result[f'regime_strength_{window}'] = result[f'hurst_rs_{window}'].apply(
                lambda h: abs(h - 0.5) * 2 if not pd.isna(h) else np.nan
            )
        
        return result
    
    def get_feature_interpretation(self) -> Dict[str, str]:
        """
        Retorna interpretação de cada feature
        
        Returns:
            Dict com interpretações
        """
        return {
            'hurst_rs': 'Hurst Exponent (R/S): >0.5 trending, <0.5 mean reverting',
            'hurst_dfa': 'Hurst Exponent (DFA): Mais robusto para séries não-estacionárias',
            'variance_ratio': 'Variance Ratio: >1 trending, <1 mean reverting',
            'fractal_dim': 'Dimensão Fractal: <1.5 trending, >1.5 choppy/mean reverting',
            'hurst_regime': 'Classificação do regime baseada no Hurst',
            'regime_strength': 'Força do regime (0-1): quanto maior, mais definido'
        }