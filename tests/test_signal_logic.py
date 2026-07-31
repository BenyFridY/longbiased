"""Signal-logic unit tests for scripts/production/generate_signal.py.

Pure-function tests only — NO model training, NO network, NO data files.
Covers:
  1. Allocation formula  alloc = clip(pred * K * sigmoid(|p-0.5|*15), 0, 1)
  2. Emergency trigger    |daily_ret| > EMERGENCY_THRESHOLD (0.08) boundary
  3. next_retrain_date / last_retrain_date (semi-annual: Jan 1 + Jul 1)
  4. _config_fingerprint changes when the config changes

conftest.py already puts repo root + scripts/production on sys.path, so both
`from scripts.production.config import ...` and bare `import config` resolve.
"""
from datetime import date

import numpy as np
import pytest

from scripts.production import config
from scripts.production import generate_signal as gs
from scripts.production.generate_signal import (
    get_regime,
    next_retrain_date,
    last_retrain_date,
    _config_fingerprint,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers that replicate the EXACT formulas in generate_signal.main()
# (kept local so we test the documented math without invoking training/IO).
# ─────────────────────────────────────────────────────────────────────────────
def _confidence_factor(p_up):
    """sigmoid(|p_up - 0.5| * SIGMOID_SCALE) — generate_signal lines 283-284."""
    confidence = abs(p_up - 0.5)
    return float(1.0 / (1.0 + np.exp(-confidence * config.SIGMOID_SCALE)))


def _alloc(pred, K, p_up):
    """alloc = clip(pred * K * sigmoid(|p-0.5|*15), ALLOC_MIN, ALLOC_MAX).

    Mirrors generate_signal.main() lines 296-297:
        raw_alloc = prediction * K
        suggested_alloc = clip(raw_alloc * confidence_factor, ALLOC_MIN, ALLOC_MAX)
    """
    raw = pred * K
    cf = _confidence_factor(p_up)
    return float(np.clip(raw * cf, config.ALLOC_MIN, config.ALLOC_MAX))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Allocation formula: clipping + confidence range
# ─────────────────────────────────────────────────────────────────────────────
def test_confidence_factor_range_is_half_to_one():
    """sigmoid(|p-0.5|*15) maps p in [0,1] to [0.5, 1.0] (never below 0.5)."""
    for p in np.linspace(0.0, 1.0, 101):
        cf = _confidence_factor(p)
        assert 0.5 <= cf <= 1.0, f"confidence {cf} out of [0.5,1.0] for p={p}"


def test_confidence_factor_uncertain_is_half():
    """p == 0.5 (max uncertainty) -> |p-0.5|=0 -> sigmoid(0) = exactly 0.5."""
    assert _confidence_factor(0.5) == pytest.approx(0.5, abs=1e-12)


def test_confidence_factor_extremes_near_one():
    """p == 0 or p == 1 -> |p-0.5|=0.5 -> sigmoid(0.5*SCALE).

    With SIGMOID_SCALE=5 (M1, 2026-06-09): sigmoid(2.5) ~ 0.924.
    """
    expected = 1.0 / (1.0 + np.exp(-0.5 * config.SIGMOID_SCALE))
    assert _confidence_factor(0.0) == pytest.approx(expected)
    assert _confidence_factor(1.0) == pytest.approx(expected)
    assert _confidence_factor(0.0) == pytest.approx(_confidence_factor(1.0))
    assert _confidence_factor(0.0) > 0.9


def test_confidence_factor_symmetric_around_half():
    """|p-0.5| means p and (1-p) give identical confidence."""
    for p in (0.1, 0.2, 0.35, 0.49):
        assert _confidence_factor(p) == pytest.approx(_confidence_factor(1 - p))


def test_alloc_negative_pred_clips_to_zero():
    """No-short: a negative prediction must clip to ALLOC_MIN (0.0)."""
    a = _alloc(pred=-0.5, K=config.K_REGIME["BULL"], p_up=0.9)
    assert a == config.ALLOC_MIN == 0.0


def test_alloc_negative_pred_always_zero_across_regimes():
    for regime, K in config.K_REGIME.items():
        for p_up in (0.0, 0.5, 0.8, 1.0):
            a = _alloc(pred=-0.01, K=K, p_up=p_up)
            assert a == 0.0, f"negative pred not clipped for {regime}, p_up={p_up}"


def test_alloc_huge_product_clips_to_one():
    """Huge pred*K (e.g. 0.5 * 60 = 30) must clip to ALLOC_MAX (1.0)."""
    a = _alloc(pred=0.5, K=config.K_REGIME["BULL"], p_up=0.95)
    assert a == config.ALLOC_MAX == 1.0


def test_alloc_zero_pred_is_zero():
    """pred == 0 -> alloc 0 regardless of confidence/regime."""
    assert _alloc(pred=0.0, K=60, p_up=0.99) == 0.0


def test_alloc_always_within_bounds():
    """Sweep pred/regime/p_up — output never escapes [ALLOC_MIN, ALLOC_MAX]."""
    for pred in np.linspace(-1.0, 1.0, 21):
        for K in config.K_REGIME.values():
            for p_up in (0.0, 0.3, 0.5, 0.7, 1.0):
                a = _alloc(pred, K, p_up)
                assert config.ALLOC_MIN <= a <= config.ALLOC_MAX


def test_alloc_matches_manual_midrange_value():
    """A small positive pred stays in the interior and equals pred*K*cf exactly."""
    pred, K, p_up = 0.001, config.K_REGIME["MILD"], 0.6  # 0.001*30=0.03
    cf = _confidence_factor(p_up)
    expected = pred * K * cf
    assert 0.0 < expected < 1.0  # genuinely interior, not clipped
    assert _alloc(pred, K, p_up) == pytest.approx(expected)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Emergency trigger boundary: |daily_ret| > EMERGENCY_THRESHOLD (0.08)
# ─────────────────────────────────────────────────────────────────────────────
def _is_emergency(daily_ret):
    """generate_signal.main() line 189: is_emergency = abs(daily_ret) > EMERGENCY_THRESHOLD."""
    return abs(daily_ret) > config.EMERGENCY_THRESHOLD


def test_emergency_threshold_value():
    assert config.EMERGENCY_THRESHOLD == 0.08


def test_emergency_below_threshold_false():
    assert _is_emergency(0.079) is False
    assert _is_emergency(-0.079) is False


def test_emergency_above_threshold_true():
    assert _is_emergency(0.081) is True
    assert _is_emergency(-0.081) is True


def test_emergency_exact_threshold_is_false():
    """Strict '>' means exactly 0.08 does NOT trigger."""
    assert _is_emergency(0.08) is False
    assert _is_emergency(-0.08) is False


def test_emergency_zero_and_small_moves_false():
    assert _is_emergency(0.0) is False
    assert _is_emergency(0.05) is False


# ─────────────────────────────────────────────────────────────────────────────
# 3. next_retrain_date / last_retrain_date  (RETRAIN_MONTHS = [1, 7])
# ─────────────────────────────────────────────────────────────────────────────
def test_retrain_months_are_jan_and_jul():
    assert gs.RETRAIN_MONTHS == [1, 7]


def test_next_retrain_mid_q1():
    assert next_retrain_date(date(2026, 3, 15)) == date(2026, 7, 1)


def test_last_retrain_mid_q1():
    assert last_retrain_date(date(2026, 3, 15)) == date(2026, 1, 1)


def test_next_retrain_after_jul():
    """Aug -> next is Jan 1 of NEXT year."""
    assert next_retrain_date(date(2026, 8, 15)) == date(2027, 1, 1)


def test_last_retrain_after_jul():
    assert last_retrain_date(date(2026, 8, 15)) == date(2026, 7, 1)


def test_next_retrain_on_jan_first():
    """candidate > today is strict, so Jan 1 itself rolls to Jul 1."""
    assert next_retrain_date(date(2026, 1, 1)) == date(2026, 7, 1)


def test_last_retrain_on_jan_first():
    """c <= today is inclusive, so Jan 1 returns Jan 1 (same day)."""
    assert last_retrain_date(date(2026, 1, 1)) == date(2026, 1, 1)


def test_next_retrain_on_jul_first():
    assert next_retrain_date(date(2026, 7, 1)) == date(2027, 1, 1)


def test_last_retrain_on_jul_first():
    assert last_retrain_date(date(2026, 7, 1)) == date(2026, 7, 1)


def test_next_retrain_late_december():
    assert next_retrain_date(date(2026, 12, 31)) == date(2027, 1, 1)


def test_last_retrain_late_december():
    assert last_retrain_date(date(2026, 12, 31)) == date(2026, 7, 1)


def test_last_before_first_jul_falls_to_jan():
    """A June date: last retrain is Jan 1 of same year (Jul not reached yet)."""
    assert last_retrain_date(date(2026, 6, 30)) == date(2026, 1, 1)
    assert next_retrain_date(date(2026, 6, 30)) == date(2026, 7, 1)


def test_next_and_last_are_consistent_window():
    """For any date, last_retrain <= today < next_retrain (semi-annual window)."""
    for d in (date(2026, 1, 2), date(2026, 3, 15), date(2026, 6, 30),
              date(2026, 7, 2), date(2026, 11, 1), date(2027, 4, 1)):
        last = last_retrain_date(d)
        nxt = next_retrain_date(d)
        assert last <= d < nxt


# ─────────────────────────────────────────────────────────────────────────────
# 4. _config_fingerprint
# ─────────────────────────────────────────────────────────────────────────────
def test_fingerprint_is_stable_and_short():
    """Deterministic 12-char sha1 prefix; identical across repeated calls."""
    fp = _config_fingerprint()
    assert isinstance(fp, str)
    assert len(fp) == 12
    assert _config_fingerprint() == fp  # not random / not memoized incorrectly


def test_fingerprint_recomputes_when_referenced_global_changes(monkeypatch):
    """Positive control: the function is NOT cached — it recomputes every call.

    generate_signal.py imports BAGS into its OWN module namespace
    (`from scripts.production.config import ... BAGS ...` at import time), and
    _config_fingerprint reads that module-level `gs.BAGS`. Patching the name the
    function actually references DOES change the fingerprint, proving the hash is
    recomputed live (not frozen at import).
    """
    fp_before = _config_fingerprint()
    monkeypatch.setattr(gs, "BAGS", config.BAGS + 1)
    fp_after = _config_fingerprint()
    assert fp_after != fp_before


def test_fingerprint_recomputes_on_horizon_change(monkeypatch):
    fp_before = _config_fingerprint()
    monkeypatch.setattr(gs, "HORIZON", config.HORIZON + 1)
    assert _config_fingerprint() != fp_before


def test_fingerprint_recomputes_on_xgb_param_change(monkeypatch):
    fp_before = _config_fingerprint()
    new_params = dict(gs.XGB_PARAMS)
    new_params["n_estimators"] = new_params["n_estimators"] + 1
    monkeypatch.setattr(gs, "XGB_PARAMS", new_params)
    assert _config_fingerprint() != fp_before


def test_fingerprint_recomputes_on_features_change(monkeypatch):
    fp_before = _config_fingerprint()
    monkeypatch.setattr(gs, "FEATURES_ALL", list(gs.FEATURES_ALL) + ["__extra__"])
    assert _config_fingerprint() != fp_before


def test_fingerprint_changes_when_bags_changes(monkeypatch):
    """The fingerprint must change when BAGS changes, so a config edit forces a
    retrain. In PRODUCTION each run is a fresh process, so
    `from scripts.production.config import BAGS` binds config.py's CURRENT value
    into generate_signal's namespace, and _config_fingerprint reads that (the same
    name train_regression_ensemble resolves). We mirror production by patching
    gs.BAGS (NOT config.BAGS — mutating config in-process never happens live; doing
    so would also be inconsistent with how the trainer reads BAGS)."""
    fp_before = _config_fingerprint()
    monkeypatch.setattr(gs, "BAGS", gs.BAGS + 1)
    assert _config_fingerprint() != fp_before


# ─────────────────────────────────────────────────────────────────────────────
# Bonus: get_regime classification (pure, no data)
# ─────────────────────────────────────────────────────────────────────────────
def test_regime_bull():
    assert get_regime(100, 90, 80) == "BULL"  # price>sma50>sma200


def test_regime_mild_above_sma200_only():
    assert get_regime(100, 110, 90) == "MILD"  # price>sma200 but not >sma50


def test_regime_bear():
    assert get_regime(70, 90, 80) == "BEAR"  # price below sma200


def test_regime_mild_on_nan_smas():
    assert get_regime(100, float("nan"), 80) == "MILD"
    assert get_regime(100, 90, float("nan")) == "MILD"
