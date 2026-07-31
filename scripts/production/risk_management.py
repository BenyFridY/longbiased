"""
Risk management — drawdown kill-switch + feature drift monitor.

Addresses critical gaps identified in the 2026-04-22 overfit audit:
  1. No kill switch when strategy underperforms (DD > threshold)
  2. No feature drift monitoring (PSI on top features)
  3. No rolling accuracy check with automatic de-risking

Usage (from generate_signal.py, AFTER computing suggested_alloc):

    from scripts.production.risk_management import apply_risk_controls
    allocation, risk_status = apply_risk_controls(
        suggested_alloc=suggested_alloc,
        signal_history=history,
        dataset=df,
        feature_cols=feature_cols,
    )

Returns (adjusted_alloc, dict of warnings).
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# THRESHOLDS (conservative, overridable by caller)
# ─────────────────────────────────────────────────────────────

# Kill switch: when strategy drawdown from peak (signal_history) exceeds this,
# force allocation to the safe floor. -12% is chosen because walk-forward 4y
# max DD was -8%; breaching it signals regime differing from training.
KILL_SWITCH_DD = -0.12
KILL_SWITCH_FLOOR = 0.15  # what alloc to use when kill switch is active

# Soft de-risk: when rolling accuracy < threshold AND model was confident,
# halve K effective. The confidence gate avoids derisk in periods where the
# model was already uncertain (low confidence_factor) — those are not
# regime-shift signals, just noise.
#
# DISABLED 2026-06-09 (M2 adoption): 10-seed paired validation on the full
# 2022-2026 walkforward showed the rule costs +2.2pp CAGR and +0.08 Sortino_d
# with max DD UNCHANGED (it never reduced drawdown in 4.4y; it only cut good
# weeks). See outputs/results/multiseed_eval_2026_06_09.json and
# archive/experiments/variant_grid_2026_06_09.py. Accuracy/confidence are
# still computed and reported (informational), the x0.5 action is off.
ACC_DERISK_ENABLED = False
ROLLING_ACC_WINDOW = 12           # weeks
ROLLING_ACC_THRESHOLD = 0.48      # below random → reduce exposure
ROLLING_ACC_DERISK_MULT = 0.5     # halve allocation
ROLLING_CONF_THRESHOLD = 0.80     # only derisk if avg confidence_factor was high
                                   # (model was confidently wrong, real degradation)

# Feature drift (PSI — Population Stability Index).
# Thresholds calibrated to this dataset's actual distributions (wide-tailed,
# many zero-variance spans from median-fills), so standard textbook thresholds
# (0.10 warn / 0.25 derisk) would fire constantly. 1.0/3.0 captures real shifts.
PSI_WARN_THRESHOLD = 1.0    # > 1.0 = meaningful shift vs last training window
PSI_DERISK_THRESHOLD = 3.0  # > 3.0 = major shift → derisk alloc
PSI_TOP_FEATURES = [        # must match top features from model importance
    'cusum_pos', 'nupl_ma30', 'bb_position', 'eth_pctchg_30d',
    'm2_yoy_growth', 'reserveRisk', 'funding_rate_ma7', 'puellMultiple',
]


# ─────────────────────────────────────────────────────────────
# Kill switch — computes current drawdown from signal_history
# ─────────────────────────────────────────────────────────────

def compute_strat_drawdown(history: pd.DataFrame) -> float:
    """Compute current drawdown from peak using closed rebal returns.

    Uses retorno_strat column (forward returns per rebal). Ignores the most
    recent row since its retorno is still NaN (not closed yet).
    """
    if 'retorno_strat' not in history.columns or len(history) < 2:
        return 0.0
    closed = history[history['retorno_strat'].notna()].copy()
    if len(closed) < 2:
        return 0.0
    cum = (1 + closed['retorno_strat']).cumprod()
    peak = cum.cummax()
    current_dd = float(cum.iloc[-1] / peak.iloc[-1] - 1)
    return current_dd


# ─────────────────────────────────────────────────────────────
# Rolling accuracy — direction hit rate
# ─────────────────────────────────────────────────────────────

def compute_rolling_accuracy(history: pd.DataFrame, window: int = 12) -> Optional[float]:
    """Direction accuracy of `previsao` vs closed `retorno_btc` in last N."""
    if 'previsao' not in history.columns or 'retorno_btc' not in history.columns:
        return None
    closed = history[history['retorno_btc'].notna()].copy()
    if len(closed) < window:
        return None
    recent = closed.tail(window)
    correct = ((recent['previsao'] > 0) & (recent['retorno_btc'] > 0)) | \
              ((recent['previsao'] < 0) & (recent['retorno_btc'] < 0))
    return float(correct.mean())


def compute_rolling_confidence(history: pd.DataFrame, window: int = 12) -> Optional[float]:
    """Average classifier confidence_factor over last N closed rebals.

    Used by acc-derisk's confidence gate: only derisk when the model was
    high-confidence AND wrong, signalling real degradation (vs uncertain noise).
    """
    if 'confidence_factor' not in history.columns or 'retorno_btc' not in history.columns:
        return None
    closed = history[history['retorno_btc'].notna()].copy()
    if len(closed) < window:
        return None
    recent = closed.tail(window)
    return float(recent['confidence_factor'].mean())


# ─────────────────────────────────────────────────────────────
# PSI (Population Stability Index) — feature drift
# ─────────────────────────────────────────────────────────────

def compute_psi(baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI between baseline (training-era) and current distributions.

    PSI < 0.10  = stable
    PSI 0.10-0.25 = some drift
    PSI > 0.25  = significant drift
    """
    baseline = np.asarray(baseline)
    current = np.asarray(current)
    baseline = baseline[~np.isnan(baseline)]
    current = current[~np.isnan(current)]
    if len(baseline) < 50 or len(current) < 10:
        return 0.0
    # Quantile-based bins from baseline
    quantiles = np.linspace(0, 1, bins + 1)[1:-1]
    edges = np.unique(np.quantile(baseline, quantiles))
    if len(edges) == 0:
        return 0.0
    edges = np.concatenate([[-np.inf], edges, [np.inf]])
    b_hist, _ = np.histogram(baseline, edges)
    c_hist, _ = np.histogram(current, edges)
    b_pct = np.where(b_hist == 0, 1e-6, b_hist / max(b_hist.sum(), 1))
    c_pct = np.where(c_hist == 0, 1e-6, c_hist / max(c_hist.sum(), 1))
    psi = np.sum((c_pct - b_pct) * np.log(c_pct / b_pct))
    return float(psi)


def compute_feature_psi(dataset: pd.DataFrame, feature_cols: list,
                        baseline_months: int = 12,
                        baseline_offset_months: int = 2,
                        current_window_days: int = 60) -> dict:
    """PSI per feature comparing a recent rolling baseline vs the last N days.

    We use the 12 months ending `baseline_offset_months` ago as the baseline
    (roughly: data the most recent retrain saw) and the last 60 days as current.
    This avoids spurious PSI from median-filled pre-history (V36 features)
    and captures genuine regime drift relative to what the model just learned.
    """
    if 'date' not in dataset.columns:
        return {}
    ds = dataset.copy()
    ds['date'] = pd.to_datetime(ds['date'])
    last_date = ds['date'].max()
    base_end = last_date - pd.DateOffset(months=baseline_offset_months)
    base_start = base_end - pd.DateOffset(months=baseline_months)
    baseline = ds[(ds['date'] >= base_start) & (ds['date'] <= base_end)]
    current = ds.tail(current_window_days)
    psi_values = {}
    for feat in PSI_TOP_FEATURES:
        if feat in dataset.columns and feat in feature_cols:
            psi_values[feat] = compute_psi(baseline[feat].values, current[feat].values)
    return psi_values


# ─────────────────────────────────────────────────────────────
# Main entry — apply all controls
# ─────────────────────────────────────────────────────────────

def apply_risk_controls(suggested_alloc: float,
                        signal_history: Optional[pd.DataFrame],
                        dataset: pd.DataFrame,
                        feature_cols: list,
                        kill_switch_dd: float = KILL_SWITCH_DD,
                        acc_derisk_enabled: bool = ACC_DERISK_ENABLED,
                        verbose: bool = True) -> tuple[float, dict]:
    """Apply all risk controls to the suggested allocation.

    Returns:
        adjusted_alloc (float): final allocation after controls
        status (dict): diagnostic info with warnings/controls triggered
    """
    status = {
        'original_alloc': suggested_alloc,
        'current_dd': 0.0,
        'rolling_acc': None,
        'rolling_conf': None,
        'kill_switch_active': False,
        'derisked_by_acc': False,
        'derisked_by_psi': False,
        'psi_values': {},
        'warnings': [],
    }
    alloc = suggested_alloc

    # 1. Kill switch (highest priority)
    if signal_history is not None and len(signal_history) > 0:
        current_dd = compute_strat_drawdown(signal_history)
        status['current_dd'] = current_dd
        if current_dd <= kill_switch_dd:
            status['kill_switch_active'] = True
            status['warnings'].append(
                f'KILL SWITCH: DD {current_dd*100:.1f}% <= {kill_switch_dd*100:.1f}%. '
                f'Forcing alloc <= {KILL_SWITCH_FLOOR*100:.0f}%'
            )
            alloc = min(alloc, KILL_SWITCH_FLOOR)

    # 2. Rolling accuracy de-risk (CONFIDENCE-GATED)
    # Only derisk when both:
    #   (a) acc 12w < 48%  — modelo errando direção mais que coinflip
    #   (b) avg conf 12w > 0.80 — modelo estava CONFIANTE (não foi só ruído)
    # Gate (b) avoids false positives in low-confidence periods where the
    # model itself signaled uncertainty (sigmoid factor low). Backtest 4y
    # showed this halves false positives (31 -> 18) with same Sortino.
    if signal_history is not None and len(signal_history) >= ROLLING_ACC_WINDOW:
        acc = compute_rolling_accuracy(signal_history, ROLLING_ACC_WINDOW)
        conf = compute_rolling_confidence(signal_history, ROLLING_ACC_WINDOW)
        status['rolling_acc'] = acc
        status['rolling_conf'] = conf
        acc_low = acc is not None and acc < ROLLING_ACC_THRESHOLD
        conf_high = conf is not None and conf > ROLLING_CONF_THRESHOLD
        if acc_low and conf_high and acc_derisk_enabled:
            status['derisked_by_acc'] = True
            status['warnings'].append(
                f'ACCURACY DE-RISK: acc {acc*100:.1f}% < {ROLLING_ACC_THRESHOLD*100:.0f}% '
                f'AND avg conf {conf*100:.0f}% > {ROLLING_CONF_THRESHOLD*100:.0f}%. '
                f'Model confidently wrong — halving alloc ({ROLLING_ACC_DERISK_MULT}x).'
            )
            alloc = alloc * ROLLING_ACC_DERISK_MULT
        elif acc_low and conf_high and not acc_derisk_enabled:
            status['warnings'].append(
                f'ACC INFO: acc {acc*100:.1f}% < 48% AND avg conf {conf*100:.0f}% > 80% '
                f'— derisk rule conditions met but rule is DISABLED (M2, 2026-06-09). '
                f'Alloc NOT adjusted.'
            )
        elif acc_low and not conf_high:
            # Informational: acc dipped but conf was low → likely just noise
            status['warnings'].append(
                f'ACC INFO: acc {acc*100:.1f}% < 48% but avg conf {conf*100:.0f}% '
                f'<= {ROLLING_CONF_THRESHOLD*100:.0f}%. Model uncertain — NOT derisked.'
            )

    # 3. Feature PSI check — INFORMATIONAL ONLY (PSI is noisy for median-filled
    # and quantile-bounded features in this dataset; reliable gating comes from
    # drawdown + rolling accuracy, not PSI magnitude).
    try:
        psi_values = compute_feature_psi(dataset, feature_cols)
        status['psi_values'] = psi_values
        high_psi = [f for f, v in psi_values.items() if v > PSI_DERISK_THRESHOLD]
        if len(high_psi) >= 3:
            status['warnings'].append(
                f'PSI INFO: {len(high_psi)} features with PSI > {PSI_DERISK_THRESHOLD}: {high_psi}. '
                f'Investigate manually; alloc NOT auto-adjusted.'
            )
    except Exception as e:
        status['warnings'].append(f'PSI check failed: {e}')

    # Print if verbose and any control triggered
    if verbose and status['warnings']:
        print()
        print('!' * 65)
        print('  RISK MANAGEMENT CONTROLS ACTIVATED')
        print('!' * 65)
        for w in status['warnings']:
            print(f'  >> {w}')
        print(f'  Original alloc: {suggested_alloc*100:+.1f}%')
        print(f'  Adjusted alloc: {alloc*100:+.1f}%')
        print('!' * 65)

    return alloc, status


if __name__ == '__main__':
    # Smoke test: run against production signal_history.csv
    import sys
    ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(ROOT))

    sig_path = ROOT / 'scripts' / 'production' / 'data' / 'signal_history.csv'
    ds_path = ROOT / 'scripts' / 'production' / 'data' / 'dataset_production.csv'

    if not sig_path.exists() or not ds_path.exists():
        print('Data files missing — cannot run smoke test')
        sys.exit(1)

    history = pd.read_csv(sig_path, parse_dates=['date'])
    ds = pd.read_csv(ds_path, parse_dates=['date'])
    feature_cols = [f for f in PSI_TOP_FEATURES if f in ds.columns]

    print('=' * 65)
    print('SMOKE TEST: risk_management.apply_risk_controls')
    print('=' * 65)
    print(f'signal_history: {len(history)} rows')
    print(f'dataset: {len(ds)} rows')
    print(f'PSI features: {feature_cols}')
    print()

    # Simulate: suggested_alloc = 50% BTC
    alloc, status = apply_risk_controls(0.50, history, ds, feature_cols, verbose=True)
    print()
    print(f'Test results:')
    print(f'  suggested: 50.0%')
    print(f'  adjusted:  {alloc*100:.1f}%')
    print(f'  current DD: {status["current_dd"]*100:.2f}%')
    print(f'  rolling acc: {status["rolling_acc"]*100:.1f}%' if status['rolling_acc'] is not None else '  rolling acc: N/A')
    print(f'  kill switch: {status["kill_switch_active"]}')
    print(f'  PSI values:')
    for f, v in sorted(status['psi_values'].items(), key=lambda x: -x[1]):
        marker = '*' if v > PSI_DERISK_THRESHOLD else ('.' if v > PSI_WARN_THRESHOLD else ' ')
        print(f'    {marker} {f}: PSI={v:.3f}')
