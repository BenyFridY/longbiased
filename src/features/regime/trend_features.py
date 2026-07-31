"""
Trend Detection Features
Identifica quando o mercado está em tendência forte (momento de estar em cripto)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class TrendFeatures:
    """
    Calcula features para identificação de tendências
    Útil para identificar quando entrar/aumentar exposição em cripto
    """
    
    def __init__(self):
        """Inicializa calculadora de trend features"""
        self.feature_names = []
    
    def calculate_linear_trend(self, prices: pd.Series) -> Dict[str, float]:
        """
        Calcula tendência linear e métricas relacionadas
        
        Args:
            prices: Série de preços
            
        Returns:
            Dict com slope, r2, angle, p-value
        """
        if len(prices) < 5:
            return {'slope': np.nan, 'r2': np.nan, 'angle': np.nan, 'p_value': np.nan}
        
        try:
            # Prepara dados
            y = prices.values
            x = np.arange(len(y))
            
            # Regressão linear
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Normaliza slope pelo preço médio (% por período)
            normalized_slope = (slope / np.mean(y)) * 100
            
            # Ângulo em graus
            angle = np.degrees(np.arctan(normalized_slope / 100))
            
            return {
                'slope': normalized_slope,
                'r2': r_value ** 2,
                'angle': angle,
                'p_value': p_value
            }
        except:
            return {'slope': np.nan, 'r2': np.nan, 'angle': np.nan, 'p_value': np.nan}
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                     period: int = 14) -> pd.Series:
        """
        Average Directional Index - força da tendência
        > 25: Tendência forte
        < 20: Sem tendência
        
        Args:
            high: Série de máximas
            low: Série de mínimas
            close: Série de fechamentos
            period: Período para cálculo
            
        Returns:
            ADX series
        """
        if len(high) < period * 2:
            return pd.Series(index=close.index, dtype=float)
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up = high - high.shift(1)
        down = low.shift(1) - low
        
        plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0), index=close.index)
        minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=close.index)
        
        # FIX 3.11: Use Wilder's smoothing (alpha = 1/period) instead of standard EMA (span)
        # Wilder's smoothing is the original formula for ADX
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)

        # ADX - also use Wilder's smoothing
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        
        return adx
    
    def calculate_aroon(self, prices: pd.Series, period: int = 25) -> pd.DataFrame:
        """
        Aroon Indicator - identifica início de tendências
        Aroon Up > 70 e Aroon Down < 30: Tendência de alta
        Aroon Down > 70 e Aroon Up < 30: Tendência de baixa

        FIX 3.9: Correct Aroon formula
        Aroon Up = ((period - days_since_high) / period) * 100
        Aroon Down = ((period - days_since_low) / period) * 100

        Args:
            prices: Série de preços
            period: Período para cálculo

        Returns:
            DataFrame com Aroon Up, Down e Oscillator
        """
        result = pd.DataFrame(index=prices.index)

        if len(prices) < period:
            return result

        # FIX 3.9: Correct Aroon calculation
        # days_since_high = len(window) - 1 - argmax(window)
        # Aroon Up = ((period - days_since_high) / period) * 100
        def aroon_up_calc(x):
            if len(x) < period + 1:
                return np.nan
            days_since_high = len(x) - 1 - np.argmax(x)
            return ((period - days_since_high) / period) * 100

        def aroon_down_calc(x):
            if len(x) < period + 1:
                return np.nan
            days_since_low = len(x) - 1 - np.argmin(x)
            return ((period - days_since_low) / period) * 100

        aroon_up = prices.rolling(period + 1).apply(aroon_up_calc, raw=True)
        aroon_down = prices.rolling(period + 1).apply(aroon_down_calc, raw=True)

        result['aroon_up'] = aroon_up
        result['aroon_down'] = aroon_down
        result['aroon_oscillator'] = aroon_up - aroon_down

        return result
    
    def calculate_trend_strength(self, prices: pd.Series, window: int = 20) -> float:
        """
        Calcula força da tendência combinando múltiplos indicadores
        0-100: quanto maior, mais forte a tendência
        
        Args:
            prices: Série de preços
            window: Janela para cálculo
            
        Returns:
            Trend strength score
        """
        if len(prices) < window:
            return np.nan
        
        scores = []
        weights = []
        
        # Linear trend R² (peso 40%)
        trend = self.calculate_linear_trend(prices)
        if not np.isnan(trend['r2']):
            # R² alto = tendência forte
            r2_score = trend['r2'] * 100
            scores.append(r2_score)
            weights.append(0.4)
        
        # Slope magnitude (peso 30%)
        if not np.isnan(trend['slope']):
            # Slope íngreme = tendência forte
            slope_score = min(100, abs(trend['slope']) * 10)
            scores.append(slope_score)
            weights.append(0.3)
        
        # Consecutive direction (peso 30%)
        returns = prices.pct_change()
        if len(returns) > 0:
            # Conta dias consecutivos na mesma direção
            same_direction = 0
            for i in range(1, min(window, len(returns))):
                if returns.iloc[-i] * returns.iloc[-(i+1)] > 0:
                    same_direction += 1
                else:
                    break
            
            consec_score = (same_direction / window) * 100
            scores.append(consec_score)
            weights.append(0.3)
        
        if not scores:
            return np.nan
        
        # Weighted average
        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        
        return weighted_score
    
    def calculate_price_position(self, prices: pd.Series, window: int = 100) -> float:
        """
        Posição do preço atual no range da janela (0-100)
        > 80: Perto da máxima (bullish)
        < 20: Perto da mínima (bearish)
        
        Args:
            prices: Série de preços
            window: Janela para cálculo
            
        Returns:
            Price position (0-100)
        """
        if len(prices) < window:
            return np.nan
        
        recent = prices.iloc[-window:]
        min_price = recent.min()
        max_price = recent.max()
        
        if max_price == min_price:
            return 50.0
        
        position = ((prices.iloc[-1] - min_price) / (max_price - min_price)) * 100
        
        return position
    
    def detect_higher_highs_lows(self, prices: pd.Series, window: int = 20) -> Dict[str, bool]:
        """
        Detecta padrão de Higher Highs/Higher Lows (tendência de alta)
        ou Lower Highs/Lower Lows (tendência de baixa)
        
        Args:
            prices: Série de preços
            window: Janela para análise
            
        Returns:
            Dict com padrões detectados
        """
        if len(prices) < window * 2:
            return {
                'higher_highs': False,
                'higher_lows': False,
                'lower_highs': False,
                'lower_lows': False
            }
        
        # Identifica picos e vales
        highs = []
        lows = []
        
        for i in range(1, len(prices) - 1):
            if i < window or i >= len(prices) - window:
                continue
                
            # Pico local
            if prices.iloc[i] > prices.iloc[i-1] and prices.iloc[i] > prices.iloc[i+1]:
                if prices.iloc[i] == prices.iloc[i-window:i+window+1].max():
                    highs.append((i, prices.iloc[i]))
            
            # Vale local
            if prices.iloc[i] < prices.iloc[i-1] and prices.iloc[i] < prices.iloc[i+1]:
                if prices.iloc[i] == prices.iloc[i-window:i+window+1].min():
                    lows.append((i, prices.iloc[i]))
        
        # Analisa padrões
        result = {
            'higher_highs': False,
            'higher_lows': False,
            'lower_highs': False,
            'lower_lows': False
        }
        
        if len(highs) >= 2:
            result['higher_highs'] = highs[-1][1] > highs[-2][1]
            result['lower_highs'] = highs[-1][1] < highs[-2][1]
        
        if len(lows) >= 2:
            result['higher_lows'] = lows[-1][1] > lows[-2][1]
            result['lower_lows'] = lows[-1][1] < lows[-2][1]
        
        return result
    
    def calculate_momentum_score(self, prices: pd.Series, 
                                periods: List[int] = None) -> float:
        """
        Score de momentum baseado em múltiplos períodos
        
        Args:
            prices: Série de preços
            periods: Lista de períodos para análise
            
        Returns:
            Momentum score (-100 a +100)
        """
        if periods is None:
            periods = [5, 10, 20, 50]
        
        scores = []
        
        for period in periods:
            if len(prices) >= period:
                ret = (prices.iloc[-1] / prices.iloc[-period] - 1) * 100
                scores.append(ret)
        
        if not scores:
            return np.nan
        
        # Média ponderada (períodos mais curtos têm mais peso)
        weights = [1/p for p in periods[:len(scores)]]
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        
        return weighted_score
    
    def calculate_trend_consistency(self, prices: pd.Series, window: int = 20) -> float:
        """
        Mede consistência da tendência (0-100)
        Alta consistência = movimento unidirecional
        
        Args:
            prices: Série de preços
            window: Janela para análise
            
        Returns:
            Consistency score
        """
        if len(prices) < window:
            return np.nan
        
        returns = prices.pct_change().dropna()
        if len(returns) < window:
            return np.nan
        
        recent_returns = returns.iloc[-window:]
        
        # Proporção de retornos positivos
        positive_ratio = (recent_returns > 0).sum() / len(recent_returns)
        
        # Desvio da proporção 50/50
        consistency = abs(positive_ratio - 0.5) * 200
        
        return consistency
    
    def calculate_all_features(self, df: pd.DataFrame, price_col: str = 'price',
                             high_col: str = None, low_col: str = None,
                             windows: List[int] = None) -> pd.DataFrame:
        """
        Calcula todas as features de tendência
        
        Args:
            df: DataFrame com dados
            price_col: Coluna de preços
            high_col: Coluna de máximas (opcional)
            low_col: Coluna de mínimas (opcional)
            windows: Lista de janelas para cálculo
            
        Returns:
            DataFrame com features calculadas
        """
        if windows is None:
            windows = [10, 20, 30, 50, 100]
        
        result = df.copy()
        
        # Se não temos high/low, usa o próprio preço
        if high_col is None or high_col not in df.columns:
            high = df[price_col]
        else:
            high = df[high_col]
            
        if low_col is None or low_col not in df.columns:
            low = df[price_col]
        else:
            low = df[low_col]
        
        # ADX (se temos high/low)
        if high_col and low_col:
            for period in [14, 20]:
                result[f'adx_{period}'] = self.calculate_adx(high, low, df[price_col], period)
        
        # Aroon
        for period in [14, 25]:
            aroon_df = self.calculate_aroon(df[price_col], period)
            for col in aroon_df.columns:
                result[f'{col}_{period}'] = aroon_df[col]
        
        # Features por janela
        for window in windows:
            if len(df) < window:
                continue
            
            # Linear trend
            trend_results = df[price_col].rolling(window).apply(
                lambda x: self.calculate_linear_trend(pd.Series(x))['slope']
            )
            result[f'trend_slope_{window}'] = trend_results
            
            r2_results = df[price_col].rolling(window).apply(
                lambda x: self.calculate_linear_trend(pd.Series(x))['r2']
            )
            result[f'trend_r2_{window}'] = r2_results
            
            # Trend strength
            result[f'trend_strength_{window}'] = df[price_col].rolling(window).apply(
                lambda x: self.calculate_trend_strength(pd.Series(x), window=len(x))
            )
            
            # Price position
            result[f'price_position_{window}'] = df[price_col].rolling(window).apply(
                lambda x: self.calculate_price_position(pd.Series(x), window=len(x))
            )
            
            # Momentum score
            result[f'momentum_score_{window}'] = df[price_col].rolling(window).apply(
                lambda x: self.calculate_momentum_score(pd.Series(x))
            )
            
            # Trend consistency
            result[f'trend_consistency_{window}'] = df[price_col].rolling(window).apply(
                lambda x: self.calculate_trend_consistency(pd.Series(x), window=len(x))
            )
            
            # Pattern detection
            patterns = df[price_col].rolling(window).apply(
                lambda x: 1 if self.detect_higher_highs_lows(pd.Series(x), window=len(x)//2)['higher_highs'] else 0
            )
            result[f'higher_highs_{window}'] = patterns
            
            # Classificação de tendência
            result[f'trend_regime_{window}'] = result.apply(
                lambda row: self._classify_trend(
                    row[f'trend_strength_{window}'],
                    row[f'momentum_score_{window}'],
                    row[f'trend_slope_{window}']
                ), axis=1
            )
        
        return result
    
    def _classify_trend(self, strength: float, momentum: float, slope: float) -> str:
        """
        Classifica o regime de tendência baseado nas métricas
        
        Args:
            strength: Força da tendência (0-100)
            momentum: Score de momentum (-100 a +100)
            slope: Inclinação da tendência
            
        Returns:
            Classificação do regime
        """
        if pd.isna(strength) or pd.isna(momentum) or pd.isna(slope):
            return np.nan
        
        if strength > 70:
            if momentum > 20:
                return 'strong_uptrend'
            elif momentum < -20:
                return 'strong_downtrend'
            else:
                return 'strong_sideways'
        elif strength > 40:
            if momentum > 10:
                return 'moderate_uptrend'
            elif momentum < -10:
                return 'moderate_downtrend'
            else:
                return 'moderate_sideways'
        else:
            return 'no_trend'
    
    def get_feature_interpretation(self) -> Dict[str, str]:
        """
        Retorna interpretação de cada feature
        
        Returns:
            Dict com interpretações
        """
        return {
            'trend_slope': 'Inclinação da tendência linear (% por período)',
            'trend_r2': 'R² da tendência (0-1). Maior = tendência mais definida',
            'adx': 'Average Directional Index. >25 = tendência forte',
            'aroon_up': 'Aroon Up. >70 = tendência de alta forte',
            'aroon_down': 'Aroon Down. >70 = tendência de baixa forte',
            'aroon_oscillator': 'Aroon Up - Down. >50 = bullish, <-50 = bearish',
            'trend_strength': 'Força da tendência (0-100). >70 = muito forte',
            'price_position': 'Posição no range (0-100). >80 = topo, <20 = fundo',
            'momentum_score': 'Score de momentum (-100 a +100). Positivo = bullish',
            'trend_consistency': 'Consistência da tendência (0-100). >70 = muito consistente',
            'higher_highs': 'Padrão de topos ascendentes detectado',
            'trend_regime': 'Classificação do regime de tendência atual'
        }