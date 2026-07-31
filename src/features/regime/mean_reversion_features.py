"""
Mean Reversion Features
Identifica quando o mercado está em regime de reversão à média
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings('ignore')


class MeanReversionFeatures:
    """
    Calcula features relacionadas a mean reversion
    Útil para identificar quando sair de cripto e ir para renda fixa
    """
    
    def __init__(self):
        """Inicializa calculadora de mean reversion features"""
        self.feature_names = []
    
    def calculate_half_life(self, prices: pd.Series) -> float:
        """
        Calcula half-life de mean reversion usando Ornstein-Uhlenbeck
        Menor half-life = reversão mais rápida
        
        Args:
            prices: Série de preços
            
        Returns:
            Half-life em períodos (dias)
        """
        if len(prices) < 20:
            return np.nan
            
        # Log prices para linearizar
        log_prices = np.log(prices)
        
        # Regressão: y_t = a + b * y_{t-1} + epsilon
        lagged = log_prices.shift(1)
        
        # Remove NaN
        mask = ~(log_prices.isna() | lagged.isna())
        if mask.sum() < 10:
            return np.nan
            
        y = log_prices[mask].values
        x = lagged[mask].values
        
        # Adiciona intercepto
        X = np.column_stack([np.ones(len(x)), x])
        
        # OLS regression
        try:
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            b = coeffs[1]
            
            # Half-life = -log(2) / log(b)
            if b <= 0 or b >= 1:
                return np.nan
                
            half_life = -np.log(2) / np.log(b)
            
            # Limita valores extremos
            return min(max(half_life, 1), 252)
            
        except:
            return np.nan
    
    def calculate_ou_parameters(self, prices: pd.Series) -> Dict[str, float]:
        """
        Estima parâmetros do processo Ornstein-Uhlenbeck
        dX = theta * (mu - X) * dt + sigma * dW
        
        theta: velocidade de reversão (maior = reversão mais rápida)
        mu: nível de equilíbrio de longo prazo
        sigma: volatilidade
        
        Args:
            prices: Série de preços
            
        Returns:
            Dict com theta, mu, sigma
        """
        if len(prices) < 30:
            return {'theta': np.nan, 'mu': np.nan, 'sigma': np.nan}
            
        log_prices = np.log(prices)
        dt = 1  # Daily data
        
        # Diferenças
        dx = log_prices.diff().dropna()
        x = log_prices.iloc[:-1].values
        
        if len(dx) < 20:
            return {'theta': np.nan, 'mu': np.nan, 'sigma': np.nan}
        
        try:
            # Estima mu como média de longo prazo (in log-price space)
            mu = log_prices.mean()

            # Regressão para theta
            # FIX 3.3: Correct OU theta formula
            # dX = theta * (mu - X) * dt + sigma * dW
            # theta = -cov(X_{t-1}, dX) / var(X_{t-1}) / dt
            y = dx.values
            x_centered = x - mu

            # FIX 3.3: Ensure theta is positive for mean reversion
            # theta should be the rate of mean reversion (always positive)
            theta_estimate = -np.sum(x_centered * y) / (np.sum(x_centered ** 2) * dt)
            theta = abs(theta_estimate)  # Must be positive for mean reversion

            # FIX 3.4: Correct sigma calculation
            # sigma is the volatility of the residuals (daily)
            predicted = -theta * x_centered * dt
            residuals = y - predicted
            sigma = np.std(residuals)  # Already daily volatility, no need to divide by sqrt(dt)

            # FIX 3.5: Return mu in log-price space (more useful for OU analysis)
            # Converting to price level (np.exp(mu)) introduces bias due to Jensen's inequality
            return {
                'theta': max(0, min(theta, 10)),  # Limita valores
                'mu': mu,  # Keep in log-price space (no bias from exp transformation)
                'mu_price_level': np.exp(mu),  # Optional: price level (with bias warning)
                'sigma': sigma
            }

        except:
            return {'theta': np.nan, 'mu': np.nan, 'mu_price_level': np.nan, 'sigma': np.nan}
    
    def calculate_distance_from_ma(self, prices: pd.Series, 
                                  periods: List[int] = None) -> pd.DataFrame:
        """
        Calcula distância percentual dos preços em relação às médias móveis
        Valores extremos indicam potencial de reversão
        
        Args:
            prices: Série de preços
            periods: Períodos das médias móveis
            
        Returns:
            DataFrame com distâncias
        """
        if periods is None:
            periods = [5, 10, 20, 50, 100, 200]
        
        result = pd.DataFrame(index=prices.index)
        
        for period in periods:
            if len(prices) >= period:
                ma = prices.rolling(period).mean()
                result[f'distance_ma_{period}'] = ((prices - ma) / ma) * 100
                
                # Z-score (normalizado por desvio padrão)
                rolling_std = prices.rolling(period).std()
                result[f'zscore_ma_{period}'] = (prices - ma) / rolling_std
                
                # Percentil da distância
                result[f'distance_percentile_{period}'] = result[f'distance_ma_{period}'].rolling(
                    period * 2).rank(pct=True) * 100
        
        return result
    
    def calculate_adf_statistic(self, prices: pd.Series, regression: str = 'c',
                                  return_pvalue: bool = False) -> float:
        """
        Augmented Dickey-Fuller test statistic
        Valores mais negativos = mais estacionário (mean reverting)

        FIX 3.6: Use LOG PRICES for ADF test (raw prices are always non-stationary)
        FIX 3.7: Optionally return p-value for better interpretation

        Args:
            prices: Série de preços
            regression: 'c' (constant), 'ct' (constant + trend), 'ctt' (constant + trend + trend²)
            return_pvalue: If True, return p-value instead of statistic

        Returns:
            ADF test statistic (or p-value if return_pvalue=True)
        """
        if len(prices) < 30:
            return np.nan

        try:
            # FIX 3.6: Use log prices for stationarity test
            # Raw prices are almost always non-stationary (unit root)
            # Log prices may show mean reversion in some cases
            log_prices = np.log(prices)
            result = adfuller(log_prices, regression=regression, autolag='AIC')

            if return_pvalue:
                return result[1]  # p-value
            return result[0]  # Test statistic
        except:
            return np.nan
    
    def calculate_kpss_statistic(self, prices: pd.Series, regression: str = 'c',
                                   return_pvalue: bool = False) -> float:
        """
        KPSS test statistic
        Valores menores = mais estacionário

        FIX 3.7: Optionally return p-value for better interpretation

        Args:
            prices: Série de preços
            regression: 'c' (level) or 'ct' (trend)
            return_pvalue: If True, return p-value instead of statistic

        Returns:
            KPSS test statistic (or p-value if return_pvalue=True)
        """
        if len(prices) < 30:
            return np.nan

        try:
            # Use log prices for consistency with ADF
            log_prices = np.log(prices)
            result = kpss(log_prices, regression=regression, nlags='auto')

            if return_pvalue:
                return result[1]  # p-value
            return result[0]  # Test statistic
        except:
            return np.nan
    
    def calculate_price_oscillator(self, prices: pd.Series, 
                                  fast: int = 12, slow: int = 26) -> pd.Series:
        """
        Price Oscillator - diferença entre médias móveis
        Valores extremos indicam oversold/overbought
        
        Args:
            prices: Série de preços
            fast: Período MA rápida
            slow: Período MA lenta
            
        Returns:
            Price oscillator
        """
        if len(prices) < slow:
            return pd.Series(index=prices.index, dtype=float)
        
        ma_fast = prices.ewm(span=fast, adjust=False).mean()
        ma_slow = prices.ewm(span=slow, adjust=False).mean()
        
        oscillator = ((ma_fast - ma_slow) / ma_slow) * 100
        
        return oscillator
    
    def calculate_bollinger_position(self, prices: pd.Series, 
                                    period: int = 20, std_dev: float = 2) -> pd.DataFrame:
        """
        Posição relativa nas Bollinger Bands
        0 = lower band, 0.5 = média, 1 = upper band
        
        Args:
            prices: Série de preços
            period: Período para média e desvio
            std_dev: Número de desvios padrão
            
        Returns:
            DataFrame com posição e largura das bandas
        """
        result = pd.DataFrame(index=prices.index)
        
        if len(prices) < period:
            return result
        
        ma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        
        upper = ma + (std * std_dev)
        lower = ma - (std * std_dev)
        
        # Posição relativa (0 a 1)
        result[f'bb_position_{period}'] = (prices - lower) / (upper - lower)
        
        # Largura das bandas (volatilidade)
        result[f'bb_width_{period}'] = ((upper - lower) / ma) * 100
        
        # Distância do meio
        result[f'bb_distance_{period}'] = ((prices - ma) / std)
        
        return result
    
    def calculate_mean_reversion_score(self, prices: pd.Series, window: int = 50) -> float:
        """
        Score composto de mean reversion (0 a 100)
        Maior score = maior probabilidade de reversão
        
        Args:
            prices: Série de preços
            window: Janela para cálculo
            
        Returns:
            Mean reversion score
        """
        if len(prices) < window:
            return np.nan
        
        scores = []
        weights = []
        
        # Half-life (peso 30%)
        hl = self.calculate_half_life(prices)
        if not np.isnan(hl):
            # Normaliza: half-life curto = score alto
            hl_score = max(0, min(100, (1 - hl/window) * 100))
            scores.append(hl_score)
            weights.append(0.3)

        # FIX 3.8: Use ADF p-value instead of statistic for scoring
        # Lower p-value = more stationary = higher mean reversion score
        adf_pvalue = self.calculate_adf_statistic(prices, return_pvalue=True)
        if not np.isnan(adf_pvalue):
            # p-value < 0.05 = stationary = high score
            # p-value > 0.5 = non-stationary = low score
            adf_score = max(0, min(100, (1 - adf_pvalue) * 100))
            scores.append(adf_score)
            weights.append(0.25)
        
        # Distance from MA (peso 25%)
        ma = prices.rolling(window//2).mean().iloc[-1]
        if not np.isnan(ma):
            distance = abs((prices.iloc[-1] - ma) / ma)
            # Maior distância = maior chance de reversão
            dist_score = min(100, distance * 500)
            scores.append(dist_score)
            weights.append(0.25)
        
        # Bollinger position (peso 20%)
        bb_result = self.calculate_bollinger_position(prices, period=min(20, window//2))
        if not bb_result.empty and f'bb_position_{min(20, window//2)}' in bb_result.columns:
            bb_pos = bb_result[f'bb_position_{min(20, window//2)}'].iloc[-1]
            if not np.isnan(bb_pos):
                # Extremos = maior score
                bb_score = max(abs(bb_pos - 0.5) * 200, 0)
                scores.append(bb_score)
                weights.append(0.2)
        
        if not scores:
            return np.nan
        
        # Weighted average
        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        
        return weighted_score
    
    def calculate_all_features(self, df: pd.DataFrame, price_col: str = 'price',
                             windows: List[int] = None) -> pd.DataFrame:
        """
        Calcula todas as features de mean reversion
        
        Args:
            df: DataFrame com dados
            price_col: Coluna de preços
            windows: Lista de janelas para cálculo
            
        Returns:
            DataFrame com features calculadas
        """
        if windows is None:
            windows = [20, 30, 50, 100]
        
        result = df.copy()
        
        # Features que não dependem de janela
        distance_df = self.calculate_distance_from_ma(df[price_col])
        for col in distance_df.columns:
            result[col] = distance_df[col]
        
        # Price oscillator
        result['price_oscillator'] = self.calculate_price_oscillator(df[price_col])
        
        # Bollinger Bands
        for period in [20, 30]:
            bb_df = self.calculate_bollinger_position(df[price_col], period=period)
            for col in bb_df.columns:
                result[col] = bb_df[col]
        
        # Features por janela
        for window in windows:
            if len(df) < window:
                continue
            
            # Half-life
            result[f'half_life_{window}'] = df[price_col].rolling(window).apply(
                lambda x: self.calculate_half_life(pd.Series(x))
            )
            
            # OU parameters
            ou_params = df[price_col].rolling(window).apply(
                lambda x: self.calculate_ou_parameters(pd.Series(x))['theta']
            )
            result[f'ou_theta_{window}'] = ou_params
            
            # ADF statistic
            result[f'adf_stat_{window}'] = df[price_col].rolling(window).apply(
                lambda x: self.calculate_adf_statistic(pd.Series(x))
            )
            
            # KPSS statistic
            result[f'kpss_stat_{window}'] = df[price_col].rolling(window).apply(
                lambda x: self.calculate_kpss_statistic(pd.Series(x))
            )
            
            # Mean reversion score
            result[f'mean_reversion_score_{window}'] = df[price_col].rolling(window).apply(
                lambda x: self.calculate_mean_reversion_score(pd.Series(x), window=len(x))
            )
            
            # Classificação
            result[f'mr_regime_{window}'] = result[f'mean_reversion_score_{window}'].apply(
                lambda x: 'high_mr' if x > 70 else ('moderate_mr' if x > 40 else 'low_mr')
                if not pd.isna(x) else np.nan
            )
        
        return result
    
    def get_feature_interpretation(self) -> Dict[str, str]:
        """
        Retorna interpretação de cada feature
        
        Returns:
            Dict com interpretações
        """
        return {
            'half_life': 'Tempo para reverter 50% ao equilíbrio (dias). Menor = reversão mais rápida',
            'ou_theta': 'Velocidade de reversão Ornstein-Uhlenbeck. Maior = reversão mais rápida',
            'distance_ma': 'Distância % da média móvel. Extremos indicam oversold/overbought',
            'zscore_ma': 'Z-score da distância. |z| > 2 = extremo estatístico',
            'adf_stat': 'ADF statistic. Mais negativo = mais estacionário/mean reverting',
            'kpss_stat': 'KPSS statistic. Menor = mais estacionário',
            'price_oscillator': 'Oscilador de preço. Extremos indicam reversão iminente',
            'bb_position': 'Posição nas Bollinger Bands (0=lower, 1=upper)',
            'bb_width': 'Largura das Bollinger Bands (volatilidade)',
            'mean_reversion_score': 'Score composto 0-100. >70 = alta probabilidade de reversão'
        }