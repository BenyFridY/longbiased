"""Train/serve consistency tests.

The production feature builder (scripts/production/build_features.py) must compute
features the SAME way as the frozen training base (scripts/data/add_regime_features.py),
otherwise live serving uses a different feature distribution than the model trained on.

Regression guard for the 2026-05-31 CUSUM fix: build_features must compute
cusum_pos/cusum_neg from SIMPLE returns (prices.pct_change()), matching the training
base — NOT log returns (the previous skew + seam discontinuity at 2026-03-03).
"""
import numpy as np
import pandas as pd

from scripts.production.build_features import build_regime_features
from src.features.regime.regime_change_features import RegimeChangeFeatures


def _cusum_series(returns: pd.Series) -> pd.Series:
    rc = RegimeChangeFeatures()
    return returns.rolling(30, min_periods=10).apply(
        lambda x: rc.calculate_cusum(pd.Series(x))["cusum_pos"], raw=False
    )


def test_build_features_cusum_uses_simple_returns():
    """build_regime_features cusum_pos must match a SIMPLE-return CUSUM (training-base
    convention), and must NOT match the log-return version."""
    rng = np.random.default_rng(7)
    n = 90
    prices = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.02, n)))
    df = pd.DataFrame({"price_usd": prices})
    # build_regime_features reads these two columns directly:
    df["adx"] = 20.0
    df["volatility_30d"] = 0.5

    out = build_regime_features(df.copy(), prices)

    exp_simple = _cusum_series(prices.pct_change())
    exp_log = _cusum_series(np.log(prices / prices.shift(1)))

    valid = out["cusum_pos"].notna() & exp_simple.notna()
    assert valid.sum() > 0, "no overlapping valid cusum rows"

    # (1) matches the SIMPLE-return convention (train/serve consistent)
    assert np.allclose(out.loc[valid, "cusum_pos"].values,
                       exp_simple[valid].values, atol=1e-9), \
        "build_features cusum_pos must equal the simple-return CUSUM (training-base convention)"

    # (2) sanity: simple != log here, so the test is meaningful (not vacuous)
    assert not np.allclose(exp_simple[valid].values, exp_log[valid].values, atol=1e-6), \
        "log and simple CUSUM should differ on this series (test would be vacuous otherwise)"
