"""
Generalized weekly-signal backfill — replaces the per-week backfill_signal_<date>.py scripts.

For each target Friday it replays generate_signal.py EXACTLY, strict no-look-ahead
(features/SMA filtered to date <= target, cached models that were live then), writes
a full-schema row to signal_history.csv, and fills the previous rebal row's forward
retornos (retorno_btc / retorno_strat).

Usage:
    python scripts/production/backfill_signal.py                 # AUTO: all missed Fridays
                                                                 # since the last logged signal,
                                                                 # up to the last date in the dataset
    python scripts/production/backfill_signal.py --date 2026-06-05   # one specific Friday
    python scripts/production/backfill_signal.py --date 2026-02-05 --emergency  # an emergency rebal

Notes:
  - Uses the CURRENT cached model (cached_models.pkl). The row's `action` records the
    cache trained_date so it's auditable. Requires cache trained_date < target (no look-ahead).
  - Reads the dataset as-is — run the fetch/bootstrap first so the target dates exist.
  - AUTO mode handles Fridays only (the weekly cadence). For a missed EMERGENCY day, use
    --date <day> --emergency.
"""
import sys
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.production.config import (
    FEATURES_ALL, K_REGIME, ALLOC_MIN, ALLOC_MAX, SIGMOID_SCALE, EMERGENCY_THRESHOLD,
)
from scripts.production.risk_management import apply_risk_controls
from src.features.macro.cdi_rates import build_rf_daily

DATASET = ROOT / 'scripts' / 'production' / 'data' / 'dataset_production.csv'
CACHE = ROOT / 'scripts' / 'production' / 'data' / 'cached_models.pkl'
SIGNAL_LOG = ROOT / 'scripts' / 'production' / 'data' / 'signal_history.csv'
DEC = 4


def get_regime(price, sma50, sma200):
    if np.isnan(sma50) or np.isnan(sma200):
        return 'MILD'
    if price > sma50 and sma50 > sma200:
        return 'BULL'
    elif price > sma200:
        return 'MILD'
    return 'BEAR'


def backfill_one(target_date: pd.Timestamp, ds_full: pd.DataFrame, cache: dict,
                 allow_non_friday: bool = False) -> bool:
    """Replay + append one rebal signal for target_date. Returns True if written."""
    ds = ds_full[ds_full['date'] <= target_date].reset_index(drop=True)
    tgt = ds[ds['date'] == target_date]
    if tgt.empty:
        print(f'  SKIP {target_date.date()}: not in dataset (last {ds_full["date"].max().date()})')
        return False
    dow = tgt.iloc[0]['date'].day_name()
    if dow != 'Friday' and not allow_non_friday:
        print(f'  SKIP {target_date.date()}: {dow} (not Friday; use --emergency to force)')
        return False

    trained = pd.to_datetime(cache.get('trained_date'))
    if pd.notna(trained) and trained >= target_date:
        print(f'  WARN {target_date.date()}: cache trained_date {trained.date()} >= target '
              f'-> potential look-ahead. Skipping (retrain at an earlier cutoff to backfill this date).')
        return False

    # Features + model (strict no-look-ahead: row from ds filtered <= target)
    raw = tgt[FEATURES_ALL].iloc[0].values
    feats = np.nan_to_num(raw.astype(float), nan=0.0)
    X = feats.reshape(1, -1)
    pred = float(np.mean([m.predict(X)[0] for m in cache['reg_models']]))
    p_up = float(np.mean([m.predict_proba(X)[0, 1] for m in cache['cls_models']]))

    prices = ds['price_usd'].values
    sma50 = pd.Series(prices).rolling(50).mean().iloc[-1]
    sma200 = pd.Series(prices).rolling(200).mean().iloc[-1]
    price = float(prices[-1])
    regime = get_regime(price, sma50, sma200)
    K = K_REGIME[regime]
    confidence_factor = float(1.0 / (1.0 + np.exp(-abs(p_up - 0.5) * SIGMOID_SCALE)))
    daily_ret = float(prices[-1] / prices[-2] - 1) if len(prices) > 1 else 0.0
    is_emergency = abs(daily_ret) > EMERGENCY_THRESHOLD
    suggested = float(np.clip(pred * K * confidence_factor, ALLOC_MIN, ALLOC_MAX))

    hist_for_risk = pd.read_csv(SIGNAL_LOG) if SIGNAL_LOG.exists() else None
    if hist_for_risk is not None:
        hist_for_risk = hist_for_risk[hist_for_risk['date'] != target_date.strftime('%Y-%m-%d')]
    allocation, risk_status = apply_risk_controls(
        suggested_alloc=suggested, signal_history=hist_for_risk,
        dataset=ds, feature_cols=list(FEATURES_ALL), verbose=False,
    )

    # Fill forward retornos on previous row, then append the new row
    history = pd.read_csv(SIGNAL_LOG)
    history['date'] = history['date'].astype(str)
    dstr = target_date.strftime('%Y-%m-%d')
    if dstr in history['date'].values:
        print(f'  {dstr} already in signal_history — replacing.')
        history = history[history['date'] != dstr].reset_index(drop=True)

    if len(history) > 0:
        prev = history.iloc[-1]
        prev_price = float(prev['price_usd']); prev_alloc = float(prev['allocation'])
        retorno_btc = float(price / prev_price - 1)
        wk = pd.date_range(pd.to_datetime(prev['date']), target_date, freq='D')
        cdi = build_rf_daily(wk)
        cdi_acc = float(np.prod(1 + cdi[1:]) - 1) if len(cdi) > 1 else 0.0
        retorno_strat = float(prev_alloc * retorno_btc + (1 - prev_alloc) * cdi_acc)
        history.loc[history.index[-1], 'retorno_btc'] = round(retorno_btc, DEC)
        history.loc[history.index[-1], 'retorno_strat'] = round(retorno_strat, DEC)

    action = ('EMERGENCY REBALANCE' if is_emergency else 'REBALANCE (Fri)') + \
             f' [H1 BAGS={len(cache["reg_models"])} cut {cache.get("trained_date")}] [BACKFILLED {_today()}]'
    new_row = {
        'date': dstr, 'day': tgt.iloc[0]['date'].strftime('%a'),
        'price_usd': round(price, 2), 'regime': regime,
        'previsao': round(pred, DEC), 'p_up': round(p_up, DEC),
        'confidence_factor': round(confidence_factor, DEC),
        'allocation': round(allocation, DEC), 'K_base': K,
        'K_effective': round(K * confidence_factor, 2), 'is_emergency': is_emergency,
        'retorno_btc': None, 'retorno_strat': None, 'action': action,
    }
    history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
    history.to_csv(SIGNAL_LOG, index=False)
    rk = ''
    if risk_status.get('derisked_by_acc'): rk += ' [acc-derisk x0.5]'
    if risk_status.get('kill_switch_active'): rk += ' [KILL SWITCH]'
    print(f'  + {dstr} {regime:<4} pred {pred*100:+.2f}% P(up) {p_up*100:.0f}% '
          f'-> alloc {allocation*100:.1f}% BTC{rk}')
    return True


def _today():
    return pd.Timestamp.now().strftime('%Y-%m-%d')


def main():
    ap = argparse.ArgumentParser(description='Backfill weekly BTC/CDI signal(s)')
    ap.add_argument('--date', default=None, help='specific target date YYYY-MM-DD')
    ap.add_argument('--emergency', action='store_true', help='allow a non-Friday (emergency) date')
    args = ap.parse_args()

    ds = pd.read_csv(DATASET, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    with open(CACHE, 'rb') as f:
        cache = pickle.load(f)
    print(f'Dataset to {ds["date"].max().date()} | cache trained {cache.get("trained_date")} '
          f'({len(cache["reg_models"])} reg + {len(cache["cls_models"])} cls bags)')

    if args.date:
        targets = [pd.Timestamp(args.date)]
    else:
        sig = pd.read_csv(SIGNAL_LOG, parse_dates=['date'])
        last = sig['date'].max()
        fri = ds[(ds['date'] > last) & (ds['date'].dt.dayofweek == 4)]['date']
        targets = sorted(fri.tolist())
        print(f'Last logged signal: {last.date()} | missed Fridays in data: '
              f'{[d.date().isoformat() for d in targets] or "none"}')

    if not targets:
        print('Nothing to backfill — signal_history is up to date.')
        return
    n = 0
    for t in targets:
        if backfill_one(t, ds, cache, allow_non_friday=args.emergency):
            n += 1
    print(f'\nDone: {n} signal row(s) written. signal_history.csv now has '
          f'{len(pd.read_csv(SIGNAL_LOG))} rows.')


if __name__ == '__main__':
    main()
