"""
01_enhance_dataset.py
Adiciona features de derivativos e cria features de interacao.

Uso: python scripts/01_enhance_dataset.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
ROOT = Path(__file__).parent.parent.parent
SELECTED = ROOT / 'outputs/feature_selection/dataset_selected.csv'
FULL = ROOT / 'outputs/dataset_final.csv'
OUTPUT = ROOT / 'outputs/feature_selection/dataset_enhanced.csv'

def load_data():
    """Carrega datasets.

    Se dataset_selected.csv existir, usa ele como base e enriquece com
    dataset_final.csv.  Caso contrario, usa dataset_final.csv diretamente
    (todas as features ja estao nele apos build_dataset.py).
    """
    df_full = pd.read_csv(FULL)

    if SELECTED.exists():
        df_selected = pd.read_csv(SELECTED)
        print(f'Dataset selected: {df_selected.shape}')
        print(f'Dataset full: {df_full.shape}')
        assert len(df_selected) == len(df_full), "Datasets tem tamanhos diferentes!"
    else:
        print(f'dataset_selected.csv nao encontrado — usando dataset_final.csv direto')
        df_selected = df_full.copy()
        print(f'Dataset full: {df_full.shape}')

    return df_selected, df_full

def add_derivative_features(df_selected, df_full):
    """Adiciona features de derivativos do dataset completo."""

    # Features para adicionar
    derivative_features = [
        # PRECO (NECESSARIO para Triple Barrier no script 02!)
        'price_usd',

        # Funding (sentiment de mercado)
        'funding_rate',
        'funding_rate_ma7',
        'binance_funding_daily',

        # Basis (contango/backwardation)
        'basis_pct',
        'basis_annualized',
        'basis_zscore',
        'basis_ma7',

        # Open Interest (ja temos oi_change_7d)
        'oi_change_30d',

        # Taker (pressao de compra/venda)
        'spot_taker_buy_ratio',
        'spot_taker_sell_ratio',

        # Futures
        'futures_volume_usd',
        'futures_dominance',
    ]

    added = []
    skipped = []
    already_exists = []

    # Converter date para datetime para merge seguro
    df_selected['date'] = pd.to_datetime(df_selected['date'])
    df_full['date'] = pd.to_datetime(df_full['date'])

    for feat in derivative_features:
        if feat in df_selected.columns:
            already_exists.append(feat)
        elif feat in df_full.columns:
            # MERGE por data (NAO usar .values que pode desalinhar!)
            feat_df = df_full[['date', feat]].copy()
            df_selected = df_selected.merge(feat_df, on='date', how='left')
            added.append(feat)
        else:
            skipped.append(feat)

    print(f'\n=== FEATURES DE DERIVATIVOS ===')
    print(f'Adicionadas: {len(added)} - {added}')
    print(f'Ja existiam: {len(already_exists)} - {already_exists}')
    print(f'Nao encontradas: {len(skipped)} - {skipped}')

    return df_selected

def check_target_leakage(df, features, targets):
    """Verifica se alguma feature tem correlacao suspeita com targets."""
    print('\n=== TARGET LEAKAGE CHECK ===')
    warnings = []
    for target in targets:
        if target not in df.columns:
            continue
        for feat in features:
            if feat in df.columns:
                corr = df[feat].corr(df[target])
                if abs(corr) > 0.3:
                    warnings.append(f'{feat} x {target}: corr={corr:.3f}')
                    print(f'WARNING: {feat} tem corr={corr:.3f} com {target}')
    if not warnings:
        print('OK: Nenhuma correlacao suspeita (>0.3) encontrada')
    return warnings

def create_interaction_features(df):
    """Cria features de interacao."""

    interactions_created = []

    # 1. Funding x OI (sentiment combinado)
    if 'funding_rate' in df.columns and 'oi_change_7d' in df.columns:
        df['funding_x_oi'] = df['funding_rate'] * df['oi_change_7d']
        interactions_created.append('funding_x_oi')

    # 2. Funding - Basis spread (arbitragem)
    if 'funding_rate' in df.columns and 'basis_pct' in df.columns:
        df['funding_basis_spread'] = df['funding_rate'] - df['basis_pct']
        interactions_created.append('funding_basis_spread')

    # 3. MVRV x NUPL (on-chain combinado)
    if 'mvrv_zscore' in df.columns and 'nupl' in df.columns:
        df['mvrv_x_nupl'] = df['mvrv_zscore'] * df['nupl']
        interactions_created.append('mvrv_x_nupl')

    # 4. SOPR x NUPL
    if 'sopr' in df.columns and 'nupl' in df.columns:
        df['sopr_x_nupl'] = df['sopr'] * df['nupl']
        interactions_created.append('sopr_x_nupl')

    # 5. Volatility x Regime Duration
    if 'volatility_30d' in df.columns and 'regime_duration' in df.columns:
        df['vol_x_regime_duration'] = df['volatility_30d'] * df['regime_duration']
        interactions_created.append('vol_x_regime_duration')

    # 6. Hurst x Half-life (regime dynamics)
    # NOTA: half_life_30d NAO existe, usar half_life_60d
    if 'hurst_30d' in df.columns and 'half_life_60d' in df.columns:
        df['hurst_x_halflife'] = df['hurst_30d'] * df['half_life_60d']
        interactions_created.append('hurst_x_halflife')

    # 7. RSI x BB Position (tecnico combinado)
    if 'rsi_14d' in df.columns and 'bb_position' in df.columns:
        df['rsi_x_bb'] = (df['rsi_14d'] - 50) * df['bb_position']  # Centraliza RSI
        interactions_created.append('rsi_x_bb')

    # 8. Taker Pressure (buy - sell)
    if 'spot_taker_buy_ratio' in df.columns and 'spot_taker_sell_ratio' in df.columns:
        df['taker_pressure'] = df['spot_taker_buy_ratio'] - df['spot_taker_sell_ratio']
        interactions_created.append('taker_pressure')

    # 9. Fear & Greed x VIX (sentiment global)
    if 'fear_greed_ma7' in df.columns and 'vix_zscore' in df.columns:
        df['fg_x_vix'] = df['fear_greed_ma7'] * df['vix_zscore']
        interactions_created.append('fg_x_vix')

    # 10. Funding extremo (binario) - usa rolling para evitar data leakage
    if 'funding_rate' in df.columns:
        funding_std_rolling = df['funding_rate'].rolling(90, min_periods=30).std()
        df['funding_extreme'] = (df['funding_rate'].abs() > 2 * funding_std_rolling).astype(int)
        interactions_created.append('funding_extreme')

    print(f'\n=== FEATURES DE INTERACAO ===')
    print(f'Criadas: {len(interactions_created)} - {interactions_created}')

    return df

def main():
    print('='*60)
    print('01_ENHANCE_DATASET')
    print('='*60)

    # 1. Carregar dados
    df_selected, df_full = load_data()

    # 2. Adicionar features de derivativos
    df = add_derivative_features(df_selected.copy(), df_full)

    # 3. Criar features de interacao
    df = create_interaction_features(df)

    # 4. Salvar
    df.to_csv(OUTPUT, index=False)

    print(f'\n=== RESULTADO ===')
    print(f'Dataset enhanced: {df.shape}')
    print(f'Salvo em: {OUTPUT}')

    # 5. Resumo de colunas
    targets = [c for c in df.columns if 'target' in c]
    features = [c for c in df.columns if c not in targets and c != 'date']

    print(f'\nFeatures: {len(features)}')
    print(f'Targets: {len(targets)} - {targets}')

    # 6. Check de leakage
    check_target_leakage(df, features[:30], ['target_direction_1d', 'target_direction_5d'])

if __name__ == '__main__':
    main()
