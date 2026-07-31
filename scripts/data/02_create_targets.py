"""
02_create_targets.py
Cria novos targets: Triple Barrier e Volatility Regime.

Uso: python scripts/02_create_targets.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
ROOT = Path(__file__).parent.parent.parent
INPUT = ROOT / 'outputs/feature_selection/dataset_enhanced.csv'
OUTPUT = INPUT  # Sobrescreve

def triple_barrier_labels(prices, volatility, pt_mult=2.0, sl_mult=1.0, max_holding=10):
    """
    Triple Barrier Method (Lopez de Prado).

    Args:
        prices: Serie de precos
        volatility: Serie de volatilidade (para barreiras dinamicas)
        pt_mult: Multiplicador para profit-taking (default 2x vol)
        sl_mult: Multiplicador para stop-loss (default 1x vol)
        max_holding: Maximo de dias em posicao

    Returns:
        labels: 1 (profit), 0 (neutro), -1 (loss)
    """
    n = len(prices)
    labels = np.full(n, np.nan)

    for i in range(n - max_holding):
        entry_price = prices[i]
        vol = volatility[i]

        if pd.isna(vol) or vol <= 0 or pd.isna(entry_price) or entry_price <= 0:
            continue

        # Barreiras
        upper = entry_price * (1 + pt_mult * vol)
        lower = entry_price * (1 - sl_mult * vol)

        label = 0  # Default: neutro (max holding atingido)

        for j in range(1, max_holding + 1):
            if i + j >= n:
                break
            future_price = prices[i + j]

            if pd.isna(future_price):
                continue

            if future_price >= upper:
                label = 1  # Profit-taking
                break
            elif future_price <= lower:
                label = -1  # Stop-loss
                break

        labels[i] = label

    return labels

def create_triple_barrier_targets(df):
    """Cria targets Triple Barrier com diferentes parametros."""

    # Precisamos de preco. Se nao tiver, reconstruir do return
    if 'price_usd' in df.columns:
        prices = df['price_usd'].values
        print('Usando price_usd para Triple Barrier')
    elif 'PriceUSD' in df.columns:
        prices = df['PriceUSD'].values
        print('Usando PriceUSD para Triple Barrier')
    else:
        # Reconstruir preco a partir dos retornos
        print('Reconstruindo preco a partir de return_1d...')
        returns = df['return_1d'].fillna(0).values
        prices = 100 * np.cumprod(1 + returns)  # Preco inicial arbitrario

    # IMPORTANTE: volatility_30d esta ANUALIZADA!
    # Converter para volatilidade DIARIA: vol_diaria = vol_anual / sqrt(365)
    volatility_annual = df['volatility_30d'].values
    volatility = volatility_annual / np.sqrt(365)
    print(f'Volatilidade convertida: anual={np.nanmean(volatility_annual):.2%} -> diaria={np.nanmean(volatility):.2%}')

    # Configuracoes para testar
    configs = [
        {'pt_mult': 2.0, 'sl_mult': 1.0, 'max_holding': 10, 'name': 'tb_2_1_10'},
        {'pt_mult': 1.5, 'sl_mult': 1.0, 'max_holding': 5,  'name': 'tb_15_1_5'},
        {'pt_mult': 2.0, 'sl_mult': 1.5, 'max_holding': 10, 'name': 'tb_2_15_10'},
        {'pt_mult': 1.5, 'sl_mult': 0.75, 'max_holding': 7, 'name': 'tb_15_075_7'},
    ]

    print('\n=== TRIPLE BARRIER TARGETS ===')

    for cfg in configs:
        labels = triple_barrier_labels(
            prices, volatility,
            pt_mult=cfg['pt_mult'],
            sl_mult=cfg['sl_mult'],
            max_holding=cfg['max_holding']
        )

        col_name = f'target_{cfg["name"]}'
        df[col_name] = labels

        # Versao binaria (profit vs nao-profit)
        col_name_bin = f'target_{cfg["name"]}_bin'
        df[col_name_bin] = (labels == 1).astype(float)
        df.loc[pd.isna(labels), col_name_bin] = np.nan

        # Stats
        valid = ~pd.isna(labels)
        if valid.sum() > 0:
            dist = pd.Series(labels[valid]).value_counts(normalize=True)
            print(f'{col_name}: profit={dist.get(1, 0):.1%}, neutro={dist.get(0, 0):.1%}, loss={dist.get(-1, 0):.1%}')

    return df

def create_new_targets(df):
    """Cria 3 novos targets mais informativos para corrigir problemas do modelo.

    Problemas identificados:
    - target_vol_up_5d preve volatilidade mas nao garante retorno positivo
    - Kelly assume volatilidade = direcao (incorreto)

    Novos targets:
    1. target_vol_direction_5d: Volatilidade alta E direcao positiva
    2. target_bullish_regime_5d: Regime favoravel (Hurst alto + retorno positivo)
    3. target_risk_adjusted_5d: Retorno ajustado por volatilidade (Sharpe local)
    """
    print('\n=== NEW TARGETS (V2) ===')

    returns = df['return_1d'].fillna(0)

    # Componentes
    future_vol_5d = returns.rolling(5).std().shift(-5)
    current_vol_5d = returns.rolling(5).std()
    vol_up = future_vol_5d > current_vol_5d

    # Retorno futuro 5 dias
    if 'price_usd' in df.columns:
        future_return_5d = df['price_usd'].pct_change(5).shift(-5)
    else:
        future_return_5d = returns.rolling(5).sum().shift(-5)

    direction_up = future_return_5d > 0

    # Hurst para regime de tendencia
    if 'hurst_30d' in df.columns:
        hurst = df['hurst_30d']
    else:
        hurst = pd.Series(0.5, index=df.index)

    trending = hurst > 0.55

    # TARGET 1: Volatilidade + Direcao
    # Queremos alta vol APENAS quando a direcao e favoravel
    target_vol_dir = (vol_up & direction_up).astype(float)
    valid_mask = ~(pd.isna(future_vol_5d) | pd.isna(current_vol_5d) | pd.isna(future_return_5d))
    target_vol_dir[~valid_mask] = np.nan
    df['target_vol_direction_5d'] = target_vol_dir

    pct_pos = df.loc[valid_mask, 'target_vol_direction_5d'].mean()
    print(f'target_vol_direction_5d: {pct_pos:.1%} positivos (vol_up AND direction_up)')

    # TARGET 2: Regime Bullish (tendencia + direcao)
    # Hurst > 0.55 indica regime de tendencia
    target_bullish = (trending & direction_up).astype(float)
    target_bullish[~valid_mask] = np.nan
    df['target_bullish_regime_5d'] = target_bullish

    pct_pos = df.loc[valid_mask, 'target_bullish_regime_5d'].mean()
    print(f'target_bullish_regime_5d: {pct_pos:.1%} positivos (hurst>0.55 AND direction_up)')

    # TARGET 3: Risk-Adjusted (Sharpe local > 0)
    # Sharpe local = retorno / volatilidade
    sharpe_local = future_return_5d / (future_vol_5d + 1e-8)
    target_risk_adj = (sharpe_local > 0).astype(float)
    target_risk_adj[pd.isna(sharpe_local)] = np.nan
    df['target_risk_adjusted_5d'] = target_risk_adj

    pct_pos = df.loc[~pd.isna(sharpe_local), 'target_risk_adjusted_5d'].mean()
    print(f'target_risk_adjusted_5d: {pct_pos:.1%} positivos (sharpe_local > 0)')

    # TARGET 4 (bonus): Sharpe alto (> 0.5 anualizado equivalente)
    # Sharpe diario ~0.1 equivale a Sharpe anual ~1.5
    target_sharpe_high = (sharpe_local > 0.1).astype(float)
    target_sharpe_high[pd.isna(sharpe_local)] = np.nan
    df['target_sharpe_high_5d'] = target_sharpe_high

    pct_pos = df.loc[~pd.isna(sharpe_local), 'target_sharpe_high_5d'].mean()
    print(f'target_sharpe_high_5d: {pct_pos:.1%} positivos (sharpe_local > 0.1)')

    return df


def create_volatility_targets(df):
    """Cria targets de volatilidade.

    IMPORTANTE: NAO usar volatility_30d.shift(-5) diretamente!
    Isso causa look-ahead bias porque volatility_30d ja e uma rolling window.
    Devemos calcular a volatilidade REALIZADA nos proximos N dias.
    """

    print('\n=== VOLATILITY TARGETS ===')

    # Calcular volatilidade REALIZADA (nao a feature volatility_30d)
    # Usar retornos diarios para calcular vol realizada
    returns = df['return_1d'].fillna(0)

    # Volatilidade realizada nos proximos 5 dias
    future_vol_5d = returns.rolling(5).std().shift(-5)

    # Volatilidade realizada nos ultimos 5 dias (para comparacao justa)
    current_vol_5d = returns.rolling(5).std()

    # Target 1: Volatilidade vai aumentar em 5 dias?
    df['target_vol_up_5d'] = (future_vol_5d > current_vol_5d).astype(float)
    df.loc[pd.isna(future_vol_5d) | pd.isna(current_vol_5d), 'target_vol_up_5d'] = np.nan
    valid_mask = ~df['target_vol_up_5d'].isna()
    print(f'target_vol_up_5d: {df.loc[valid_mask, "target_vol_up_5d"].mean():.1%} positivos')

    # Target 2: Volatilidade futura vai estar acima da mediana historica?
    # Usar expanding para evitar look-ahead (mediana so de dados passados)
    vol_median_expanding = current_vol_5d.expanding().median()
    df['target_vol_high_5d'] = (future_vol_5d > vol_median_expanding).astype(float)
    df.loc[pd.isna(future_vol_5d), 'target_vol_high_5d'] = np.nan
    valid_mask = ~df['target_vol_high_5d'].isna()
    print(f'target_vol_high_5d: {df.loc[valid_mask, "target_vol_high_5d"].mean():.1%} positivos')

    # Target 3: Regime de volatilidade futura (tercis)
    # Usar quantiles de toda a serie (isso e ok pois e para definir os bins)
    vol_33 = current_vol_5d.quantile(0.33)
    vol_66 = current_vol_5d.quantile(0.66)

    def vol_regime(v):
        if pd.isna(v):
            return np.nan
        elif v < vol_33:
            return 0  # Low vol
        elif v < vol_66:
            return 1  # Medium vol
        else:
            return 2  # High vol

    df['target_vol_regime_5d'] = future_vol_5d.apply(vol_regime)
    print(f'target_vol_regime_5d: 3 classes')

    return df

def main():
    print('='*60)
    print('02_CREATE_TARGETS')
    print('='*60)

    # 1. Carregar
    df = pd.read_csv(INPUT)
    print(f'Input: {df.shape}')

    # 2. Triple Barrier
    df = create_triple_barrier_targets(df)

    # 3. Volatility (targets originais)
    df = create_volatility_targets(df)

    # 4. NOVOS TARGETS (V2) - mais informativos
    df = create_new_targets(df)

    # 5. Salvar
    df.to_csv(OUTPUT, index=False)

    # 6. Resumo
    targets = [c for c in df.columns if 'target' in c]
    print(f'\n=== RESULTADO ===')
    print(f'Dataset: {df.shape}')
    print(f'Total de targets: {len(targets)}')

    # Separar por categoria
    tb_targets = [t for t in targets if 'tb_' in t]
    vol_targets = [t for t in targets if 'vol_' in t and 'direction' not in t]
    new_targets = [t for t in targets if any(x in t for x in ['direction', 'bullish', 'risk_adj', 'sharpe'])]
    other_targets = [t for t in targets if t not in tb_targets + vol_targets + new_targets]

    print(f'\nTriple Barrier ({len(tb_targets)}): {tb_targets}')
    print(f'Volatility ({len(vol_targets)}): {vol_targets}')
    print(f'New V2 ({len(new_targets)}): {new_targets}')
    if other_targets:
        print(f'Other ({len(other_targets)}): {other_targets}')

    print(f'\nSalvo em: {OUTPUT}')

if __name__ == '__main__':
    main()
