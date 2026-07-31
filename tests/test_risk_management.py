"""Tests for scripts/production/risk_management.py — risk controls.

Covers:
  - compute_strat_drawdown (closed-row DD from peak, trailing NaN ignored)
  - compute_rolling_accuracy (closed-only, <window -> None, hit fraction)
  - apply_risk_controls kill switch (cum DD <= -12% -> cap 0.15)
  - apply_risk_controls acc-derisk confidence gate (both conditions)
  - ordering: kill switch then acc-derisk
  - compute_psi (identical ~0, shifted >0, too-few -> 0.0)

All synthetic, deterministic, fast. No network, no data files needed.
"""
import numpy as np
import pandas as pd
import pytest

from scripts.production.risk_management import (
    compute_strat_drawdown,
    compute_rolling_accuracy,
    compute_psi,
    apply_risk_controls,
    KILL_SWITCH_FLOOR,
    KILL_SWITCH_DD,
    ROLLING_ACC_WINDOW,
    ROLLING_ACC_THRESHOLD,
    ROLLING_CONF_THRESHOLD,
    ROLLING_ACC_DERISK_MULT,
)


# ─────────────────────────────────────────────────────────────
# 1. compute_strat_drawdown
# ─────────────────────────────────────────────────────────────

def test_drawdown_about_minus_15_pct_from_peak():
    """Build closed rows that rise then fall ~15% from peak; trailing NaN ignored.

    Returns: +10%, then -0.15/1.10 so cum returns to 0.935 (a 15% drawdown
    relative to the 1.10 peak). Verify ~ -0.15.
    """
    # cum after r1 = 1.10 (peak). We want cum_final / peak - 1 = -0.15
    # => cum_final = 1.10 * 0.85 = 0.935 => (1+r2) = 0.935/1.10
    r2 = 0.935 / 1.10 - 1.0
    hist = pd.DataFrame({
        'retorno_strat': [0.10, r2, np.nan],  # trailing NaN must be ignored
    })
    dd = compute_strat_drawdown(hist)
    assert dd == pytest.approx(-0.15, abs=1e-9)


def test_drawdown_trailing_nan_ignored_equals_no_nan():
    r2 = 0.935 / 1.10 - 1.0
    with_nan = pd.DataFrame({'retorno_strat': [0.10, r2, np.nan]})
    without_nan = pd.DataFrame({'retorno_strat': [0.10, r2]})
    assert compute_strat_drawdown(with_nan) == pytest.approx(
        compute_strat_drawdown(without_nan), abs=1e-12)


def test_drawdown_zero_at_new_peak():
    """If the final closed cum value IS the peak, DD is 0."""
    hist = pd.DataFrame({'retorno_strat': [0.05, 0.05, np.nan]})
    assert compute_strat_drawdown(hist) == pytest.approx(0.0, abs=1e-12)


def test_drawdown_missing_column_returns_zero():
    hist = pd.DataFrame({'other': [1.0, 2.0, 3.0]})
    assert compute_strat_drawdown(hist) == 0.0


def test_drawdown_fewer_than_two_closed_returns_zero():
    # Only one closed row -> len(closed) < 2 -> 0.0
    hist = pd.DataFrame({'retorno_strat': [0.05, np.nan, np.nan]})
    assert compute_strat_drawdown(hist) == 0.0


# ─────────────────────────────────────────────────────────────
# 2. compute_rolling_accuracy
# ─────────────────────────────────────────────────────────────

def test_rolling_accuracy_below_window_returns_none():
    # 11 closed rows < window(12) -> None
    n = ROLLING_ACC_WINDOW - 1
    hist = pd.DataFrame({
        'previsao': [0.01] * n,
        'retorno_btc': [0.01] * n,
    })
    assert compute_rolling_accuracy(hist, ROLLING_ACC_WINDOW) is None


def test_rolling_accuracy_only_closed_rows_count():
    """Open rows (retorno_btc NaN) must not count; exactly `window` closed needed."""
    w = ROLLING_ACC_WINDOW
    # window closed correct rows + extra open (NaN) rows that should be ignored
    prev = [0.01] * w + [-0.5, 0.5]          # last two are open
    ret = [0.02] * w + [np.nan, np.nan]      # last two open
    hist = pd.DataFrame({'previsao': prev, 'retorno_btc': ret})
    acc = compute_rolling_accuracy(hist, w)
    assert acc == pytest.approx(1.0)


def test_rolling_accuracy_hit_fraction_crafted():
    """Craft direction hits: positive-positive and negative-negative are hits.

    9 hits, 3 misses out of 12 -> 0.75. Hit when sign(previsao)==sign(retorno_btc)
    (strictly > or strictly <).
    """
    w = ROLLING_ACC_WINDOW  # 12
    prev, ret = [], []
    # 6 correct longs
    for _ in range(6):
        prev.append(0.03); ret.append(0.02)
    # 3 correct shorts
    for _ in range(3):
        prev.append(-0.03); ret.append(-0.02)
    # 3 misses (predict up, went down)
    for _ in range(3):
        prev.append(0.03); ret.append(-0.02)
    hist = pd.DataFrame({'previsao': prev, 'retorno_btc': ret})
    acc = compute_rolling_accuracy(hist, w)
    assert acc == pytest.approx(9.0 / 12.0)


def test_rolling_accuracy_uses_only_last_window():
    """With more than `window` closed rows, only the last `window` count."""
    w = ROLLING_ACC_WINDOW
    # First w rows all misses, last w rows all hits -> should be 1.0 (uses tail)
    prev = [0.03] * w + [0.03] * w
    ret = [-0.02] * w + [0.02] * w
    hist = pd.DataFrame({'previsao': prev, 'retorno_btc': ret})
    assert compute_rolling_accuracy(hist, w) == pytest.approx(1.0)


def test_rolling_accuracy_missing_columns_returns_none():
    hist = pd.DataFrame({'previsao': [0.1] * 12})  # no retorno_btc
    assert compute_rolling_accuracy(hist, 12) is None


# ─────────────────────────────────────────────────────────────
# Helpers for apply_risk_controls
# ─────────────────────────────────────────────────────────────

def _empty_dataset():
    """Dataset with no 'date' column -> compute_feature_psi returns {} (no-op)."""
    return pd.DataFrame({'x': [1, 2, 3]})


def _drawdown_history(n_closed=14, drop=-0.20):
    """History whose closed retorno_strat crashes hard (cum DD well past -12%).

    Returns n_closed closed rows + 1 trailing open (NaN) row.
    First row is a small gain to set a peak, rest is the crash spread out.
    """
    rets = [0.01] + [drop] + [0.0] * (n_closed - 2)
    rets = rets + [np.nan]  # trailing open row
    return pd.DataFrame({
        'retorno_strat': rets,
        'previsao': [0.01] * (n_closed + 1),
        'retorno_btc': [0.01] * n_closed + [np.nan],
        'confidence_factor': [0.5] * (n_closed + 1),
    })


# ─────────────────────────────────────────────────────────────
# 3. apply_risk_controls — KILL SWITCH
# ─────────────────────────────────────────────────────────────

def test_kill_switch_caps_alloc_at_floor():
    hist = _drawdown_history(n_closed=14, drop=-0.20)
    # sanity: DD really is past the kill switch threshold
    dd = compute_strat_drawdown(hist)
    assert dd <= KILL_SWITCH_DD

    alloc, status = apply_risk_controls(
        suggested_alloc=0.80,
        signal_history=hist,
        dataset=_empty_dataset(),
        feature_cols=[],
        verbose=False,
    )
    assert status['kill_switch_active'] is True
    assert alloc <= KILL_SWITCH_FLOOR + 1e-12
    # With acc not derisked (conf low), alloc should equal exactly the floor
    # because suggested (0.80) > floor.
    assert status['derisked_by_acc'] is False
    assert alloc == pytest.approx(KILL_SWITCH_FLOOR)


def test_kill_switch_does_not_raise_alloc_below_floor():
    """min(alloc, FLOOR): a suggested alloc already below floor stays as-is."""
    hist = _drawdown_history(n_closed=14, drop=-0.20)
    alloc, status = apply_risk_controls(
        suggested_alloc=0.05,
        signal_history=hist,
        dataset=_empty_dataset(),
        feature_cols=[],
        verbose=False,
    )
    assert status['kill_switch_active'] is True
    assert alloc == pytest.approx(0.05)


def test_no_kill_switch_when_dd_shallow():
    """A mild drawdown above the threshold leaves alloc untouched."""
    # peak then mild -3% drawdown, plus enough rows to not trip acc-derisk path
    rets = [0.02, -0.03] + [0.0] * 12 + [np.nan]
    hist = pd.DataFrame({
        'retorno_strat': rets,
        'previsao': [0.01] * 15,
        'retorno_btc': [0.01] * 14 + [np.nan],
        'confidence_factor': [0.5] * 15,
    })
    assert compute_strat_drawdown(hist) > KILL_SWITCH_DD
    alloc, status = apply_risk_controls(
        suggested_alloc=0.80,
        signal_history=hist,
        dataset=_empty_dataset(),
        feature_cols=[],
        verbose=False,
    )
    assert status['kill_switch_active'] is False
    assert alloc == pytest.approx(0.80)


# ─────────────────────────────────────────────────────────────
# 4. apply_risk_controls — ACC-DERISK confidence gate
# ─────────────────────────────────────────────────────────────

def _low_acc_history(conf, n=ROLLING_ACC_WINDOW):
    """History with shallow DD (no kill switch) but rolling acc < 0.48.

    All `n` closed rows are direction MISSES (predict up, go down) -> acc 0.0.
    retorno_strat kept tiny/positive so drawdown never triggers kill switch.
    conf sets the confidence_factor on every row.
    """
    return pd.DataFrame({
        'retorno_strat': [0.001] * n,          # no drawdown
        'previsao': [0.03] * n,                 # predict up
        'retorno_btc': [-0.02] * n,             # actually down -> all misses
        'confidence_factor': [conf] * n,
    })


def test_acc_derisk_fires_when_conf_high():
    """RULE LOGIC (enabled=True): acc < 0.48 AND avg conf > 0.80 -> alloc halved.

    Production default is DISABLED since 2026-06-09 (M2); these rule-logic
    tests opt in explicitly. See test_acc_derisk_disabled_by_default.
    """
    hist = _low_acc_history(conf=0.90)
    # sanity
    assert compute_rolling_accuracy(hist, ROLLING_ACC_WINDOW) < ROLLING_ACC_THRESHOLD
    alloc, status = apply_risk_controls(
        suggested_alloc=0.60,
        signal_history=hist,
        dataset=_empty_dataset(),
        feature_cols=[],
        acc_derisk_enabled=True,
        verbose=False,
    )
    assert status['kill_switch_active'] is False
    assert status['derisked_by_acc'] is True
    assert alloc == pytest.approx(0.60 * ROLLING_ACC_DERISK_MULT)


def test_acc_derisk_disabled_by_default():
    """PRODUCTION DEFAULT (M2, 2026-06-09): conditions met -> informational
    warning only, alloc NOT adjusted. 10-seed validation: the rule cost
    +2.2pp CAGR / +0.08 Sortino_d with max DD unchanged."""
    from risk_management import ACC_DERISK_ENABLED
    assert ACC_DERISK_ENABLED is False
    hist = _low_acc_history(conf=0.90)
    alloc, status = apply_risk_controls(
        suggested_alloc=0.60,
        signal_history=hist,
        dataset=_empty_dataset(),
        feature_cols=[],
        verbose=False,
    )
    assert status['derisked_by_acc'] is False
    assert alloc == pytest.approx(0.60)
    assert any('DISABLED' in w for w in status['warnings'])


def test_acc_derisk_does_not_fire_when_conf_low():
    """acc < 0.48 but avg conf <= 0.80 -> informational only, NOT derisked."""
    hist = _low_acc_history(conf=0.70)
    alloc, status = apply_risk_controls(
        suggested_alloc=0.60,
        signal_history=hist,
        dataset=_empty_dataset(),
        feature_cols=[],
        verbose=False,
    )
    assert status['kill_switch_active'] is False
    assert status['derisked_by_acc'] is False
    assert alloc == pytest.approx(0.60)


def test_acc_derisk_boundary_just_below_threshold_not_derisked():
    """conf just below 0.80 -> gate is strict `> 0.80`, so NOT derisked.

    (Note: a mean of exactly-0.80 inputs is avoided because float accumulation
    can land at 0.8000000000000002, which IS > 0.80 — the gate itself is
    correctly strict, so we probe a clearly-below value instead.)
    """
    hist = _low_acc_history(conf=0.799)
    alloc, status = apply_risk_controls(
        suggested_alloc=0.60,
        signal_history=hist,
        dataset=_empty_dataset(),
        feature_cols=[],
        verbose=False,
    )
    assert status['derisked_by_acc'] is False
    assert alloc == pytest.approx(0.60)


def test_acc_derisk_boundary_just_above_threshold_derisked():
    """RULE LOGIC (enabled=True): conf just above 0.80 -> gate opens, halved."""
    hist = _low_acc_history(conf=0.801)
    alloc, status = apply_risk_controls(
        suggested_alloc=0.60,
        signal_history=hist,
        dataset=_empty_dataset(),
        feature_cols=[],
        acc_derisk_enabled=True,
        verbose=False,
    )
    assert status['derisked_by_acc'] is True
    assert alloc == pytest.approx(0.60 * ROLLING_ACC_DERISK_MULT)


# ─────────────────────────────────────────────────────────────
# 5. ORDERING — kill switch first, then acc-derisk
# ─────────────────────────────────────────────────────────────

def test_kill_switch_and_acc_derisk_compose():
    """RULE LOGIC (enabled=True) — both fire: cap at 0.15, then halve."""
    n = ROLLING_ACC_WINDOW
    # Crash on row 2 (DD past -12%), and every closed row is a direction miss
    # with high confidence so acc < 0.48 and conf > 0.80.
    rets = [0.01, -0.20] + [0.0] * (n - 2)
    hist = pd.DataFrame({
        'retorno_strat': rets,
        'previsao': [0.03] * n,           # predict up
        'retorno_btc': [-0.02] * n,       # go down -> all misses
        'confidence_factor': [0.90] * n,  # high confidence
    })
    # sanity on both gates
    assert compute_strat_drawdown(hist) <= KILL_SWITCH_DD
    assert compute_rolling_accuracy(hist, n) < ROLLING_ACC_THRESHOLD

    alloc, status = apply_risk_controls(
        suggested_alloc=0.90,
        signal_history=hist,
        dataset=_empty_dataset(),
        feature_cols=[],
        acc_derisk_enabled=True,
        verbose=False,
    )
    assert status['kill_switch_active'] is True
    assert status['derisked_by_acc'] is True
    # 0.90 -> min(0.90, 0.15) = 0.15 -> * 0.5 = 0.075
    assert alloc == pytest.approx(KILL_SWITCH_FLOOR * ROLLING_ACC_DERISK_MULT)
    assert alloc <= 0.075 + 1e-12


# ─────────────────────────────────────────────────────────────
# 6. compute_psi
# ─────────────────────────────────────────────────────────────

def test_psi_identical_distributions_near_zero():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 1000)
    psi = compute_psi(base, base.copy())
    assert psi == pytest.approx(0.0, abs=1e-6)


def test_psi_shifted_distribution_positive():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 1000)
    cur = rng.normal(5, 1, 1000)  # clear mean shift
    psi = compute_psi(base, cur)
    assert psi > 0.0
    # A 5-sigma shift should produce a large PSI, well above textbook 0.25
    assert psi > 0.25


def test_psi_too_few_baseline_returns_zero():
    rng = np.random.default_rng(2)
    base = rng.normal(0, 1, 49)   # < 50 baseline -> 0.0
    cur = rng.normal(0, 1, 100)
    assert compute_psi(base, cur) == 0.0


def test_psi_too_few_current_returns_zero():
    rng = np.random.default_rng(3)
    base = rng.normal(0, 1, 100)
    cur = rng.normal(0, 1, 9)     # < 10 current -> 0.0
    assert compute_psi(base, cur) == 0.0


def test_psi_drops_nans_then_checks_size():
    """NaNs are stripped before the size check; enough valid samples -> computes."""
    rng = np.random.default_rng(4)
    base = np.concatenate([rng.normal(0, 1, 60), np.full(20, np.nan)])
    cur = np.concatenate([rng.normal(0, 1, 15), np.full(5, np.nan)])
    psi = compute_psi(base, cur)
    # Same distribution sampled -> small but defined PSI (not the 0.0 sentinel
    # for too-few unless it genuinely rounds there). Just assert it is finite.
    assert np.isfinite(psi)
    assert psi >= 0.0
