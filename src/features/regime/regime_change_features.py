"""
Regime Change Detection Features
Identifica mudanças estruturais no comportamento do mercado
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class RegimeChangeFeatures:
    """
    Detecta mudanças de regime e transições de mercado
    Crucial para timing de entrada/saída
    """
    
    def __init__(self):
        """Inicializa detector de mudança de regime"""
        self.feature_names = []
    
    def calculate_cusum(self, returns: pd.Series, threshold: float = None,
                         reset_on_detection: bool = True) -> Dict[str, float]:
        """
        CUSUM (Cumulative Sum) para detectar mudanças
        Detecta quando a média dos retornos muda significativamente

        FIX 3.10: Standard CUSUM implementation with optional reset after detection

        Args:
            returns: Série de retornos
            threshold: Limite para detecção (se None, usa 4*std for more robust detection)
            reset_on_detection: If True, reset CUSUM to 0 after detecting a change

        Returns:
            Dict com CUSUM positivo, negativo, sinal de mudança e número de detecções
        """
        if len(returns) < 10:
            return {'cusum_pos': np.nan, 'cusum_neg': np.nan, 'change_detected': 0, 'n_changes': 0}

        # Remove NaNs
        returns_clean = returns.dropna()
        if len(returns_clean) < 10:
            return {'cusum_pos': np.nan, 'cusum_neg': np.nan, 'change_detected': 0, 'n_changes': 0}

        # Parâmetros - target mean is 0 for returns under null hypothesis
        target_mean = 0
        std = returns_clean.std()

        if std == 0:
            return {'cusum_pos': 0, 'cusum_neg': 0, 'change_detected': 0, 'n_changes': 0}

        # FIX 3.10: Use 4*std as default threshold (more robust)
        if threshold is None:
            threshold = 4 * std

        # Slack parameter k (typically 0.5 * shift to detect)
        k = std / 2

        # CUSUM positivo (detects increase) e negativo (detects decrease)
        cusum_pos = 0
        cusum_neg = 0
        max_cusum_pos = 0
        max_cusum_neg = 0
        n_changes = 0

        for ret in returns_clean:
            # Standard CUSUM formulas
            cusum_pos = max(0, cusum_pos + (ret - target_mean - k))
            cusum_neg = max(0, cusum_neg + (-ret + target_mean - k))

            # Track maximum values
            max_cusum_pos = max(max_cusum_pos, cusum_pos)
            max_cusum_neg = max(max_cusum_neg, cusum_neg)

            # FIX 3.10: Reset after detection (standard CUSUM behavior)
            if reset_on_detection:
                if cusum_pos > threshold or cusum_neg > threshold:
                    n_changes += 1
                    cusum_pos = 0
                    cusum_neg = 0

        # Detecta mudança (based on current or max values)
        change_detected = 1 if (max_cusum_pos > threshold or max_cusum_neg > threshold) else 0

        return {
            'cusum_pos': max_cusum_pos,
            'cusum_neg': max_cusum_neg,
            'change_detected': change_detected,
            'n_changes': n_changes
        }
    
    def calculate_rolling_correlation_change(self, prices1: pd.Series, 
                                            prices2: pd.Series, 
                                            window: int = 30) -> float:
        """
        Detecta mudanças na correlação entre ativos
        Útil para identificar mudanças de regime de risco
        
        Args:
            prices1: Primeira série de preços
            prices2: Segunda série de preços
            window: Janela para correlação rolling
            
        Returns:
            Taxa de mudança da correlação
        """
        if len(prices1) < window * 2 or len(prices2) < window * 2:
            return np.nan
        
        # Retornos
        ret1 = prices1.pct_change().dropna()
        ret2 = prices2.pct_change().dropna()
        
        # Alinha séries
        aligned = pd.DataFrame({'ret1': ret1, 'ret2': ret2}).dropna()
        if len(aligned) < window * 2:
            return np.nan
        
        # Correlação rolling
        rolling_corr = aligned['ret1'].rolling(window).corr(aligned['ret2'])
        
        # Taxa de mudança
        if len(rolling_corr.dropna()) > window:
            recent_corr = rolling_corr.iloc[-window:].mean()
            previous_corr = rolling_corr.iloc[-window*2:-window].mean()
            
            if not np.isnan(recent_corr) and not np.isnan(previous_corr):
                change_rate = (recent_corr - previous_corr) / (abs(previous_corr) + 0.01)
                return change_rate
        
        return np.nan
    
    def calculate_volatility_regime_shift(self, returns: pd.Series, 
                                         short_window: int = 10, 
                                         long_window: int = 60) -> Dict[str, float]:
        """
        Detecta mudanças no regime de volatilidade
        
        Args:
            returns: Série de retornos
            short_window: Janela curta para vol
            long_window: Janela longa para vol
            
        Returns:
            Dict com ratio, percentil e classificação
        """
        if len(returns) < long_window:
            return {
                'vol_ratio': np.nan,
                'vol_percentile': np.nan,
                'vol_regime': 'unknown'
            }
        
        # Volatilidades
        vol_short = returns.rolling(short_window).std()
        vol_long = returns.rolling(long_window).std()
        
        # Ratio atual
        current_vol_short = vol_short.iloc[-1]
        current_vol_long = vol_long.iloc[-1]
        
        if pd.isna(current_vol_short) or pd.isna(current_vol_long) or current_vol_long == 0:
            return {
                'vol_ratio': np.nan,
                'vol_percentile': np.nan,
                'vol_regime': 'unknown'
            }
        
        vol_ratio = current_vol_short / current_vol_long
        
        # Percentil histórico
        historical_vol = returns.rolling(short_window).std().dropna()
        if len(historical_vol) > 0:
            vol_percentile = stats.percentileofscore(historical_vol, current_vol_short)
        else:
            vol_percentile = 50
        
        # Classificação
        if vol_percentile > 80:
            vol_regime = 'high_volatility'
        elif vol_percentile > 60:
            vol_regime = 'elevated_volatility'
        elif vol_percentile < 20:
            vol_regime = 'low_volatility'
        else:
            vol_regime = 'normal_volatility'
        
        return {
            'vol_ratio': vol_ratio,
            'vol_percentile': vol_percentile,
            'vol_regime': vol_regime
        }
    
    def calculate_structural_break_score(self, prices: pd.Series, window: int = 50) -> float:
        """
        Score composto para probabilidade de quebra estrutural
        Combina múltiplos indicadores
        
        Args:
            prices: Série de preços
            window: Janela para análise
            
        Returns:
            Score de 0-100 (maior = maior probabilidade de mudança)
        """
        if len(prices) < window * 2:
            return np.nan
        
        scores = []
        weights = []
        
        # 1. Mudança na média dos retornos
        returns = prices.pct_change().dropna()
        if len(returns) >= window * 2:
            recent_mean = returns.iloc[-window:].mean()
            previous_mean = returns.iloc[-window*2:-window].mean()
            overall_std = returns.std()
            
            if overall_std > 0:
                mean_change = abs(recent_mean - previous_mean) / overall_std
                mean_score = min(100, mean_change * 50)
                scores.append(mean_score)
                weights.append(0.3)
        
        # 2. Mudança na volatilidade
        if len(returns) >= window * 2:
            recent_vol = returns.iloc[-window:].std()
            previous_vol = returns.iloc[-window*2:-window].std()
            
            if previous_vol > 0:
                vol_change = abs(recent_vol - previous_vol) / previous_vol
                vol_score = min(100, vol_change * 100)
                scores.append(vol_score)
                weights.append(0.3)
        
        # 3. Mudança na autocorrelação
        if len(returns) >= window:
            recent_autocorr = returns.iloc[-window:].autocorr(lag=1)
            previous_autocorr = returns.iloc[-window*2:-window].autocorr(lag=1)
            
            if not np.isnan(recent_autocorr) and not np.isnan(previous_autocorr):
                autocorr_change = abs(recent_autocorr - previous_autocorr)
                autocorr_score = min(100, autocorr_change * 100)
                scores.append(autocorr_score)
                weights.append(0.2)
        
        # 4. Teste de Chow (simplificado)
        if len(prices) >= window * 2:
            # Divide dados em dois períodos
            first_half = prices.iloc[-window*2:-window]
            second_half = prices.iloc[-window:]
            
            # Testa se as distribuições são diferentes
            try:
                _, p_value = stats.ks_2samp(first_half, second_half)
                chow_score = (1 - p_value) * 100
                scores.append(chow_score)
                weights.append(0.2)
            except:
                pass
        
        if not scores:
            return np.nan
        
        # Weighted average
        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        
        return weighted_score
    
    def detect_acceleration(self, prices: pd.Series, window: int = 20) -> Dict[str, float]:
        """
        Detecta aceleração/desaceleração de preços
        Útil para identificar mudanças de momentum
        
        Args:
            prices: Série de preços
            window: Janela para cálculo
            
        Returns:
            Dict com velocidade, aceleração e classificação
        """
        if len(prices) < window * 2:
            return {
                'velocity': np.nan,
                'acceleration': np.nan,
                'momentum_regime': 'unknown'
            }
        
        # Primeira derivada (velocidade)
        returns = prices.pct_change()
        velocity = returns.rolling(window).mean()
        
        # Segunda derivada (aceleração)
        acceleration = velocity.diff()
        
        # Valores atuais
        current_velocity = velocity.iloc[-1] if not velocity.empty else np.nan
        current_acceleration = acceleration.iloc[-1] if not acceleration.empty else np.nan
        
        # Classificação
        if pd.isna(current_velocity) or pd.isna(current_acceleration):
            momentum_regime = 'unknown'
        elif current_velocity > 0 and current_acceleration > 0:
            momentum_regime = 'accelerating_up'
        elif current_velocity > 0 and current_acceleration < 0:
            momentum_regime = 'decelerating_up'
        elif current_velocity < 0 and current_acceleration < 0:
            momentum_regime = 'accelerating_down'
        elif current_velocity < 0 and current_acceleration > 0:
            momentum_regime = 'decelerating_down'
        else:
            momentum_regime = 'neutral'
        
        return {
            'velocity': current_velocity * 100 if not pd.isna(current_velocity) else np.nan,
            'acceleration': current_acceleration * 10000 if not pd.isna(current_acceleration) else np.nan,
            'momentum_regime': momentum_regime
        }
    
    def calculate_microstructure_noise_ratio(self, prices: pd.Series, 
                                            short_interval: int = 5,
                                            long_interval: int = 30) -> float:
        """
        Ratio de ruído de microestrutura
        Alto ratio = mais ruído, possível mudança de regime
        
        Args:
            prices: Série de preços
            short_interval: Intervalo curto
            long_interval: Intervalo longo
            
        Returns:
            Noise ratio
        """
        if len(prices) < long_interval:
            return np.nan
        
        # Variância de retornos em diferentes escalas
        ret_short = prices.pct_change(short_interval).var()
        ret_long = prices.pct_change(long_interval).var()
        
        if pd.isna(ret_short) or pd.isna(ret_long) or ret_long == 0:
            return np.nan
        
        # Ratio teórico sem ruído seria proporcional ao intervalo
        expected_ratio = short_interval / long_interval
        actual_ratio = ret_short / ret_long * long_interval / short_interval
        
        # Noise ratio
        noise_ratio = actual_ratio / expected_ratio if expected_ratio > 0 else np.nan
        
        return noise_ratio
    
    def calculate_all_features(self, df: pd.DataFrame, price_col: str = 'price',
                             volume_col: str = None, windows: List[int] = None) -> pd.DataFrame:
        """
        Calcula todas as features de mudança de regime
        
        Args:
            df: DataFrame com dados
            price_col: Coluna de preços
            volume_col: Coluna de volume (opcional)
            windows: Lista de janelas para cálculo
            
        Returns:
            DataFrame com features calculadas
        """
        if windows is None:
            windows = [20, 30, 50, 100]
        
        result = df.copy()
        
        # Calcula retornos
        returns = df[price_col].pct_change()
        
        # Features por janela
        for window in windows:
            if len(df) < window * 2:
                continue
            
            # CUSUM
            cusum_results = returns.rolling(window).apply(
                lambda x: self.calculate_cusum(pd.Series(x))['change_detected']
            )
            result[f'cusum_change_{window}'] = cusum_results
            
            # Volatility regime
            vol_regime = returns.rolling(window * 2).apply(
                lambda x: self.calculate_volatility_regime_shift(
                    pd.Series(x), 
                    short_window=window//3, 
                    long_window=window
                )['vol_ratio']
            )
            result[f'vol_regime_ratio_{window}'] = vol_regime
            
            # Structural break score
            result[f'structural_break_score_{window}'] = df[price_col].rolling(window * 2).apply(
                lambda x: self.calculate_structural_break_score(pd.Series(x), window=window)
            )
            
            # Acceleration
            accel = df[price_col].rolling(window * 2).apply(
                lambda x: self.detect_acceleration(pd.Series(x), window=window)['acceleration']
            )
            result[f'price_acceleration_{window}'] = accel
            
            # Microstructure noise
            result[f'noise_ratio_{window}'] = df[price_col].rolling(window).apply(
                lambda x: self.calculate_microstructure_noise_ratio(
                    pd.Series(x),
                    short_interval=max(1, window//6),
                    long_interval=window
                )
            )
            
            # Volume pattern change (se disponível)
            if volume_col and volume_col in df.columns:
                # Mudança no padrão de volume
                vol_mean = df[volume_col].rolling(window).mean()
                vol_std = df[volume_col].rolling(window).std()
                result[f'volume_zscore_{window}'] = (df[volume_col] - vol_mean) / vol_std
                
                # Mudança na relação preço-volume
                price_vol_corr = df[price_col].rolling(window).corr(df[volume_col])
                result[f'price_volume_corr_{window}'] = price_vol_corr
            
            # Classificação de probabilidade de mudança
            result[f'regime_change_probability_{window}'] = result[f'structural_break_score_{window}'].apply(
                lambda x: 'high' if x > 70 else ('medium' if x > 40 else 'low')
                if not pd.isna(x) else np.nan
            )
        
        # Features globais (não dependem de janela específica)
        
        # Dias desde última mudança significativa
        threshold_change = returns.rolling(30).std() * 2
        significant_changes = abs(returns) > threshold_change
        
        days_since_change = []
        counter = 0
        for change in significant_changes:
            if change:
                counter = 0
            else:
                counter += 1
            days_since_change.append(counter)
        
        result['days_since_significant_change'] = days_since_change
        
        # Regime de mercado atual (classificação final)
        if 'structural_break_score_50' in result.columns:
            result['market_regime_status'] = result.apply(
                lambda row: self._classify_regime_status(row), axis=1
            )
        
        return result
    
    def _classify_regime_status(self, row: pd.Series) -> str:
        """
        Classifica o status do regime atual
        
        Args:
            row: Linha do DataFrame com features
            
        Returns:
            Status do regime
        """
        # Procura por features relevantes
        break_score = row.get('structural_break_score_50', np.nan)
        vol_ratio = row.get('vol_regime_ratio_50', np.nan)
        cusum = row.get('cusum_change_50', np.nan)
        
        if pd.isna(break_score):
            return 'unknown'
        
        if break_score > 70:
            return 'regime_changing'
        elif break_score > 40:
            if not pd.isna(vol_ratio) and vol_ratio > 1.5:
                return 'volatility_expanding'
            elif not pd.isna(vol_ratio) and vol_ratio < 0.7:
                return 'volatility_contracting'
            else:
                return 'transition_period'
        else:
            return 'stable_regime'
    
    def get_feature_interpretation(self) -> Dict[str, str]:
        """
        Retorna interpretação de cada feature
        
        Returns:
            Dict com interpretações
        """
        return {
            'cusum_change': 'CUSUM detector. 1 = mudança detectada',
            'vol_regime_ratio': 'Ratio vol curto/longo prazo. >1.5 = expansão de vol',
            'structural_break_score': 'Score 0-100. >70 = alta prob de mudança estrutural',
            'price_acceleration': 'Aceleração do preço. Positivo = acelerando',
            'noise_ratio': 'Ratio de ruído. >2 = muito ruído/instabilidade',
            'volume_zscore': 'Z-score do volume. |z|>2 = volume anormal',
            'price_volume_corr': 'Correlação preço-volume. Mudanças = possível regime novo',
            'days_since_significant_change': 'Dias desde última mudança significativa',
            'regime_change_probability': 'Probabilidade de mudança de regime',
            'market_regime_status': 'Status atual do regime de mercado'
        }