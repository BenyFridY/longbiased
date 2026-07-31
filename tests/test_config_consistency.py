"""Config consistency tests for scripts/production/config.py.

These assert the canonical production constants documented in the model spec:
32 features (legacy name FEATURES_37), K_REGIME H1 (60/30/15), no-short bounds,
BAGS=160, HORIZON=3, SIGMOID_SCALE=15, EMERGENCY_THRESHOLD=0.08, Friday rebal,
and XGBoost determinism (nthread=1, reg:squarederror objective).

conftest.py puts both repo root and scripts/production on sys.path, so both
import styles resolve to the same module.
"""
import config


# ---------------------------------------------------------------------------
# FEATURES_37 / FEATURES_ALL
# ---------------------------------------------------------------------------
def test_features_37_has_32_entries():
    # Legacy name is FEATURES_37 but it actually holds 32 features.
    assert len(config.FEATURES_37) == 32


def test_features_37_no_duplicates():
    feats = config.FEATURES_37
    assert len(feats) == len(set(feats)), (
        "FEATURES_37 contains duplicate feature names: "
        f"{[f for f in set(feats) if feats.count(f) > 1]}"
    )


def test_features_37_all_strings_nonempty():
    for f in config.FEATURES_37:
        assert isinstance(f, str) and f, f"bad feature entry: {f!r}"


def test_features_all_is_features_37():
    # Must be the SAME object (alias), not just an equal copy.
    assert config.FEATURES_ALL is config.FEATURES_37


# ---------------------------------------------------------------------------
# K_REGIME (H1: 60/30/15)
# ---------------------------------------------------------------------------
def test_k_regime_keys():
    assert set(config.K_REGIME.keys()) == {"BULL", "MILD", "BEAR"}


def test_k_regime_values():
    assert config.K_REGIME["BULL"] == 60
    assert config.K_REGIME["MILD"] == 30
    assert config.K_REGIME["BEAR"] == 15


# ---------------------------------------------------------------------------
# Allocation bounds (no-short, fully invested cap)
# ---------------------------------------------------------------------------
def test_alloc_min_zero():
    assert config.ALLOC_MIN == 0.0


def test_alloc_max_one():
    assert config.ALLOC_MAX == 1.0


def test_alloc_bounds_ordered():
    assert config.ALLOC_MIN < config.ALLOC_MAX


# ---------------------------------------------------------------------------
# Scalar model params
# ---------------------------------------------------------------------------
def test_bags():
    assert config.BAGS == 160


def test_horizon():
    assert config.HORIZON == 3


def test_sigmoid_scale():
    # 15 -> 5 on 2026-06-09 (M1): +0.34 Sortino_d / -1.55pp DD_d, 10-seed.
    assert config.SIGMOID_SCALE == 5


def test_emergency_threshold():
    assert config.EMERGENCY_THRESHOLD == 0.08


def test_rebal_dow_friday():
    # Friday == weekday index 4; production rebalances weekly on Friday.
    assert config.REBAL_DOW == [4]


# ---------------------------------------------------------------------------
# XGBoost determinism + objective
# ---------------------------------------------------------------------------
def test_xgb_nthread_single():
    # Single-threaded for reproducibility (CPU non-determinism otherwise).
    assert config.XGB_PARAMS["nthread"] == 1


def test_xgb_objective_squarederror():
    assert config.XGB_PARAMS["objective"] == "reg:squarederror"
