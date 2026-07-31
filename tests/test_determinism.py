"""Determinism tests for the production training/signal pipeline.

Targets:
  - scripts/production/generate_signal.py: _config_fingerprint, train_regression_ensemble
  - scripts/production/training.py: _train_one_xgb

These tests use tiny synthetic seeded data and a reduced bag count so the whole
module runs in well under 60s. No network, no data files required.

Key wiring fact (read from source, not assumed):
  train_regression_ensemble references BAGS / WORKERS / XGB_PARAMS that were
  imported INTO the generate_signal module namespace (`from ...config import ...`).
  So to shrink the bag count for the test we must monkeypatch
  `generate_signal.BAGS` (the name resolved at call time), NOT `config.BAGS`.
"""
import numpy as np
import pytest

import scripts.production.generate_signal as gs
import scripts.production.config as config
from scripts.production.training import _train_one_xgb


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _make_synth(n=80, d=32, seed=0):
    """Tiny seeded synthetic regression dataset (n rows x d cols)."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(float)
    # A learnable-but-noisy linear-ish target so trees actually split.
    y = (X[:, 0] * 0.5 - X[:, 1] * 0.3 + rng.randn(n) * 0.1).astype(float)
    return X, y


def _fast_params():
    """A copy of the production XGB params with n_estimators shrunk for speed.
    Keeps nthread=1 (the determinism-relevant knob) from the real config."""
    p = dict(config.XGB_PARAMS)
    p['n_estimators'] = 8
    return p


# --------------------------------------------------------------------------
# 1. _config_fingerprint stability
# --------------------------------------------------------------------------
def test_config_fingerprint_stable_across_calls():
    """Same config => identical fingerprint string on repeated calls."""
    fp1 = gs._config_fingerprint()
    fp2 = gs._config_fingerprint()
    assert fp1 == fp2
    assert isinstance(fp1, str)


def test_config_fingerprint_length_and_hex():
    """Fingerprint is the documented 12-char sha1 hex prefix."""
    fp = gs._config_fingerprint()
    assert len(fp) == 12
    int(fp, 16)  # raises ValueError if not hex


def test_config_fingerprint_changes_with_bags(monkeypatch):
    """Changing a config input (BAGS) changes the fingerprint.

    _config_fingerprint reads the module-level names BAGS/FEATURES_ALL/HORIZON/
    XGB_PARAMS bound in the generate_signal namespace, so patch those.
    """
    base = gs._config_fingerprint()
    monkeypatch.setattr(gs, 'BAGS', config.BAGS + 1)
    changed = gs._config_fingerprint()
    assert changed != base


def test_config_fingerprint_changes_with_features(monkeypatch):
    """Changing the feature list changes the fingerprint."""
    base = gs._config_fingerprint()
    monkeypatch.setattr(gs, 'FEATURES_ALL', list(config.FEATURES_ALL) + ['extra_feat'])
    assert gs._config_fingerprint() != base


# --------------------------------------------------------------------------
# 2. train_regression_ensemble determinism (same seed -> same mean prediction)
# --------------------------------------------------------------------------
def test_train_regression_ensemble_uses_nthread1():
    """Sanity: the params the ensemble feeds to _train_one_xgb pin nthread=1
    (single-thread is what makes XGBoost hist deterministic on CPU)."""
    assert config.XGB_PARAMS.get('nthread') == 1


def test_train_regression_ensemble_deterministic(monkeypatch):
    """Training the regression ensemble twice with the SAME seed produces an
    identical mean prediction on a fixed X (np.allclose, in fact exact)."""
    X, y = _make_synth(n=80, d=32, seed=1)
    X_fixed, _ = _make_synth(n=10, d=32, seed=99)

    # Shrink bags + estimators so this is fast. WORKERS stays as configured;
    # nthread=1 inside each model keeps results reproducible regardless of
    # the ThreadPoolExecutor worker count.
    monkeypatch.setattr(gs, 'BAGS', 4)
    monkeypatch.setattr(gs, 'XGB_PARAMS', _fast_params())

    m1 = gs.train_regression_ensemble(X, y, seed=242)
    m2 = gs.train_regression_ensemble(X, y, seed=242)

    assert len(m1) == 4 and len(m2) == 4

    p1 = np.mean([m.predict(X_fixed) for m in m1], axis=0)
    p2 = np.mean([m.predict(X_fixed) for m in m2], axis=0)

    assert np.allclose(p1, p2)
    # Single-thread + fixed seed should be bit-identical, not just close.
    assert np.array_equal(p1, p2)


def test_train_regression_ensemble_different_seed_differs(monkeypatch):
    """Different seed => different bag seeds => generally different predictions.
    (Guards against the ensemble ignoring the seed argument entirely.)"""
    X, y = _make_synth(n=80, d=32, seed=2)
    X_fixed, _ = _make_synth(n=10, d=32, seed=99)

    monkeypatch.setattr(gs, 'BAGS', 4)
    monkeypatch.setattr(gs, 'XGB_PARAMS', _fast_params())

    p_a = np.mean([m.predict(X_fixed) for m in gs.train_regression_ensemble(X, y, seed=242)], axis=0)
    p_b = np.mean([m.predict(X_fixed) for m in gs.train_regression_ensemble(X, y, seed=999)], axis=0)

    assert not np.allclose(p_a, p_b)


def test_train_regression_ensemble_seed_offset(monkeypatch):
    """The ensemble uses bag seeds seed + i*7; bag i's model must match a
    standalone _train_one_xgb trained with that exact seed (no bootstrap
    resampling in the regression head, per source)."""
    X, y = _make_synth(n=80, d=32, seed=3)
    X_fixed, _ = _make_synth(n=10, d=32, seed=99)

    params = _fast_params()
    monkeypatch.setattr(gs, 'BAGS', 3)
    monkeypatch.setattr(gs, 'XGB_PARAMS', params)

    models = gs.train_regression_ensemble(X, y, seed=242)
    for i, m in enumerate(models):
        ref = _train_one_xgb((242 + i * 7, X, y, params))
        assert np.array_equal(m.predict(X_fixed), ref.predict(X_fixed)), f"bag {i} mismatch"


# --------------------------------------------------------------------------
# 3. _train_one_xgb determinism
# --------------------------------------------------------------------------
def test_train_one_xgb_deterministic_same_seed():
    """_train_one_xgb with a fixed seed yields a model whose predict is
    bit-identical across two independent trainings."""
    X, y = _make_synth(n=80, d=32, seed=4)
    X_fixed, _ = _make_synth(n=10, d=32, seed=99)
    params = _fast_params()

    m1 = _train_one_xgb((123, X, y, params))
    m2 = _train_one_xgb((123, X, y, params))

    pr1 = m1.predict(X_fixed)
    pr2 = m2.predict(X_fixed)
    assert np.array_equal(pr1, pr2)


def test_train_one_xgb_predict_repeatable_same_model():
    """Calling predict twice on the same trained model returns identical output."""
    X, y = _make_synth(n=80, d=32, seed=5)
    X_fixed, _ = _make_synth(n=10, d=32, seed=99)
    m = _train_one_xgb((7, X, y, _fast_params()))
    assert np.array_equal(m.predict(X_fixed), m.predict(X_fixed))


def test_train_one_xgb_different_seed_can_differ():
    """Different seeds generally produce different models (seed is honored)."""
    X, y = _make_synth(n=80, d=32, seed=6)
    X_fixed, _ = _make_synth(n=10, d=32, seed=99)
    params = _fast_params()
    m_a = _train_one_xgb((1, X, y, params))
    m_b = _train_one_xgb((2, X, y, params))
    assert not np.array_equal(m_a.predict(X_fixed), m_b.predict(X_fixed))


def test_train_one_xgb_does_not_mutate_params():
    """_train_one_xgb copies params (dict(params_base)) and pops n_estimators
    off the COPY, so the caller's dict is left intact. The regression ensemble
    relies on this — it passes the same XGB_PARAMS dict to every bag."""
    X, y = _make_synth(n=40, d=32, seed=7)
    params = _fast_params()
    before = dict(params)
    _train_one_xgb((11, X, y, params))
    assert params == before
    assert 'n_estimators' in params  # not popped from the caller's dict


def test_train_one_xgb_honors_n_estimators():
    """The n_estimators passed in params is actually used by the booster."""
    X, y = _make_synth(n=60, d=32, seed=8)
    params = _fast_params()
    params['n_estimators'] = 5
    m = _train_one_xgb((3, X, y, params))
    booster = m.get_booster()
    # One tree per boosting round for a single-output regressor.
    assert booster.num_boosted_rounds() == 5
