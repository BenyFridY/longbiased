"""
Backfill the 2026-05-15 (Fri) signal that was missed during paper-trade week
+ fill forward retornos on the previous rebal row (2026-05-08).

Strict no-look-ahead: feature row for the 2026-05-15 signal is filtered to
dataset[date <= 2026-05-15]. Cached models (trained 2026-04-27) were live on
that date — the same ones that would have been used in real time.

Mirrors generate_signal.py exactly:
  - 32 features, regression (sizing) + classifier (confidence)
  - allocation = clip(pred * K[regime] * sigmoid(|P_cls - 0.5| * 15), 0, 1)
  - K_REGIME (H1) + risk controls (kill switch / acc derisk / PSI)
  - retorno_btc[T-1] = price[T] / price[T-1] - 1  (BTC log-return for the
    week ending at the new rebal)
  - retorno_strat[T-1] = prev_alloc * retorno_btc + (1 - prev_alloc) * cdi_acc
"""
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.production.config import (
    FEATURES_ALL, K_REGIME, ALLOC_MIN, ALLOC_MAX,
    SIGMOID_SCALE, EMERGENCY_THRESHOLD,
)
from scripts.production.risk_management import apply_risk_controls
from src.features.macro.cdi_rates import build_rf_daily

DATASET = ROOT / 'scripts' / 'production' / 'data' / 'dataset_production.csv'
CACHE = ROOT / 'scripts' / 'production' / 'data' / 'cached_models.pkl'
SIGNAL_LOG = ROOT / 'scripts' / 'production' / 'data' / 'signal_history.csv'
TARGET_DATE = pd.Timestamp('2026-05-15')
DEC = 4


def get_regime(price, sma50, sma200):
    if np.isnan(sma50) or np.isnan(sma200):
        return 'MILD'
    if price > sma50 and sma50 > sma200:
        return 'BULL'
    elif price > sma200:
        return 'MILD'
    return 'BEAR'


def main():
    # 1. Strict no-look-ahead filter for signal generation
    ds_full = pd.read_csv(DATASET, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    ds = ds_full[ds_full['date'] <= TARGET_DATE].reset_index(drop=True)
    print(f'Dataset filtered <= {TARGET_DATE.date()}: {len(ds)} rows')
    print('Last 3 rows:')
    print(ds[['date', 'price_usd']].tail(3).to_string(index=False))

    tgt = ds[ds['date'] == TARGET_DATE]
    if tgt.empty:
        raise SystemExit(f'{TARGET_DATE.date()} not in dataset (last: {ds["date"].iloc[-1].date()})')
    tgt_idx = tgt.index[0]
    dow = tgt.iloc[0]['date'].day_name()
    assert dow == 'Friday', f'Target {TARGET_DATE.date()} is {dow}, expected Friday'

    # 2. Features + model
    feats = tgt[FEATURES_ALL].iloc[0].values.astype(float)
    feats = np.nan_to_num(feats, nan=0.0)
    if (~np.isfinite(tgt[FEATURES_ALL].iloc[0].values)).any():
        miss = [f for f, v in zip(FEATURES_ALL, tgt[FEATURES_ALL].iloc[0].values) if pd.isna(v)]
        print(f'WARN: {len(miss)} missing features (zero-filled): {miss[:5]}{"..." if len(miss)>5 else ""}')

    with open(CACHE, 'rb') as f:
        cache = pickle.load(f)
    print(f'Cache: {cache.get("version")} trained {cache.get("trained_date")}, '
          f'{len(cache["reg_models"])} reg + {len(cache["cls_models"])} cls bags')

    X = feats.reshape(1, -1)
    pred = float(np.mean([m.predict(X)[0] for m in cache['reg_models']]))
    p_up = float(np.mean([m.predict_proba(X)[0, 1] for m in cache['cls_models']]))

    # 3. Confidence + regime (computed on prices <= TARGET_DATE)
    prices = ds['price_usd'].values
    sma50 = pd.Series(prices).rolling(50).mean().iloc[-1]
    sma200 = pd.Series(prices).rolling(200).mean().iloc[-1]
    price = float(prices[-1])
    regime = get_regime(price, sma50, sma200)
    K = K_REGIME[regime]

    confidence = abs(p_up - 0.5)
    confidence_factor = float(1.0 / (1.0 + np.exp(-confidence * SIGMOID_SCALE)))

    daily_ret = float(prices[-1] / prices[-2] - 1)
    is_emergency = abs(daily_ret) > EMERGENCY_THRESHOLD

    # 4. Allocation pre-risk
    raw_alloc = pred * K
    suggested_alloc = float(np.clip(raw_alloc * confidence_factor, ALLOC_MIN, ALLOC_MAX))

    # 5. Risk controls (kill switch / acc derisk / PSI) — uses history excluding target
    hist_for_risk = pd.read_csv(SIGNAL_LOG) if SIGNAL_LOG.exists() else None
    if hist_for_risk is not None:
        hist_for_risk = hist_for_risk[hist_for_risk['date'] != TARGET_DATE.strftime('%Y-%m-%d')]
    feature_cols = list(FEATURES_ALL)
    suggested_alloc_pre_risk = suggested_alloc
    allocation, risk_status = apply_risk_controls(
        suggested_alloc=suggested_alloc,
        signal_history=hist_for_risk,
        dataset=ds,
        feature_cols=feature_cols,
        verbose=False,
    )

    print()
    print('=' * 65)
    print(f'  REPLAYED SIGNAL — {TARGET_DATE.date()} ({dow})  [no look-ahead]')
    print('=' * 65)
    print(f'  BTC Price:      ${price:>12,.2f}')
    print(f'  Daily Return:   {daily_ret*100:>+11.2f}%')
    print(f'  Regime:         {regime}  (K_base={K})')
    print(f'  Prediction:     {pred*100:>+11.3f}% (3d)')
    print(f'  P(up):          {p_up*100:>11.1f}%   conf {confidence_factor:.4f}')
    print(f'  K effective:    {K * confidence_factor:>11.2f}')
    print(f'  Pre-risk alloc: {suggested_alloc_pre_risk*100:>+10.2f}%')
    print(f'  Final alloc:    {allocation*100:>+10.2f}% BTC  /  {(1-allocation)*100:.2f}% CDI')
    if abs(suggested_alloc_pre_risk - allocation) > 1e-6:
        print(f'  (risk controls adjusted)')
    if risk_status.get('rolling_acc') is not None:
        print(f'  Rolling acc 12w: {risk_status["rolling_acc"]*100:.1f}% (threshold 48%)')
    if risk_status.get('current_dd', 0) < -0.01:
        print(f'  Current DD:     {risk_status["current_dd"]*100:.2f}% (kill at -12%)')
    for w in risk_status.get('warnings', []):
        print(f'  WARN: {w}')
    print('=' * 65)

    # 6. Backfill forward retornos on previous row (2026-05-08)
    history = pd.read_csv(SIGNAL_LOG)
    history['date'] = history['date'].astype(str)
    if TARGET_DATE.strftime('%Y-%m-%d') in history['date'].values:
        print(f'Row {TARGET_DATE.date()} already exists in signal_history — replacing.')
        history = history[history['date'] != TARGET_DATE.strftime('%Y-%m-%d')].reset_index(drop=True)

    prev_row = history.iloc[-1]
    prev_date = pd.to_datetime(prev_row['date'])
    prev_price = float(prev_row['price_usd'])
    prev_alloc = float(prev_row['allocation'])

    retorno_btc = float(price / prev_price - 1)
    # CDI accumulation between prev_date (exclusive) and TARGET_DATE (inclusive)
    wk_dates = pd.date_range(prev_date, TARGET_DATE, freq='D')
    cdi_series = build_rf_daily(wk_dates)
    cdi_acc = float(np.prod(1 + cdi_series[1:]) - 1) if len(cdi_series) > 1 else 0.0
    retorno_strat = float(prev_alloc * retorno_btc + (1 - prev_alloc) * cdi_acc)

    print()
    print(f'  Backfilling forward retornos on {prev_row["date"]} (prev rebal):')
    print(f'    BTC:   {prev_price:,.2f}  ->  {price:,.2f}   ret = {retorno_btc*100:+.2f}%')
    print(f'    CDI acc ({len(cdi_series)-1}d):                          {cdi_acc*100:+.2f}%')
    print(f'    Strat ({prev_alloc*100:.2f}% alloc):                     {retorno_strat*100:+.2f}%')

    history.loc[history.index[-1], 'retorno_btc'] = round(retorno_btc, DEC)
    history.loc[history.index[-1], 'retorno_strat'] = round(retorno_strat, DEC)

    # 7. Append the new 2026-05-15 row (retornos NaN — filled on next rebal)
    new_row = {
        'date': TARGET_DATE.strftime('%Y-%m-%d'),
        'day': 'Fri',
        'price_usd': round(price, 2),
        'regime': regime,
        'previsao': round(pred, DEC),
        'p_up': round(p_up, DEC),
        'confidence_factor': round(confidence_factor, DEC),
        'allocation': round(allocation, DEC),
        'K_base': K,
        'K_effective': round(K * confidence_factor, 2),
        'is_emergency': is_emergency,
        'retorno_btc': None,
        'retorno_strat': None,
        'action': f'REBALANCE (Fri) [H1 BAGS=160 cut {cache.get("trained_date")}] [BACKFILLED {pd.Timestamp.today().date()}]',
    }
    history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
    history.to_csv(SIGNAL_LOG, index=False)
    print()
    print(f'signal_history.csv updated — {len(history)} rows total')
    print('Tail:')
    print(history[['date', 'price_usd', 'regime', 'allocation', 'retorno_btc', 'retorno_strat']].tail(3).to_string(index=False))


if __name__ == '__main__':
    main()
