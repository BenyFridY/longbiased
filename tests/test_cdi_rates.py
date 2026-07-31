"""Offline tests for src/features/macro/cdi_rates.py (source='copom' only).

All tests pass source='copom' so NO network is ever touched. They exercise the
COPOM-history fallback path in build_rf_daily / _build_from_copom.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.macro.cdi_rates import build_rf_daily, COPOM_HISTORY


# ---------------------------------------------------------------------------
# 1. Output length matches input length
# ---------------------------------------------------------------------------
def test_output_length_matches_input():
    dates = pd.date_range("2023-01-01", "2023-03-31", freq="D")
    rf = build_rf_daily(dates, source="copom")
    assert len(rf) == len(dates)
    assert isinstance(rf, np.ndarray)


def test_output_length_various_inputs():
    # numpy datetime64 array
    dates_np = pd.date_range("2023-06-01", "2023-06-30", freq="D").values
    rf = build_rf_daily(dates_np, source="copom")
    assert len(rf) == len(dates_np)

    # single-element index
    one = pd.DatetimeIndex(["2023-02-01"])  # a Wednesday (business day)
    rf_one = build_rf_daily(one, source="copom")
    assert len(rf_one) == 1


# ---------------------------------------------------------------------------
# 2. Weekends accrue 0
# ---------------------------------------------------------------------------
def test_weekends_accrue_zero():
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    rf = build_rf_daily(dates, source="copom")
    dow = pd.DatetimeIndex(dates).dayofweek
    weekend_mask = dow >= 5  # 5=Sat, 6=Sun
    assert np.all(rf[weekend_mask] == 0.0), "weekends must accrue exactly 0"


def test_business_days_are_positive():
    # Inside a single regime, every business day must accrue a positive rate.
    dates = pd.date_range("2023-02-01", "2023-02-28", freq="D")
    rf = build_rf_daily(dates, source="copom")
    dow = pd.DatetimeIndex(dates).dayofweek
    biz_mask = dow < 5
    assert np.all(rf[biz_mask] > 0.0), "business days must accrue a positive rate"


# ---------------------------------------------------------------------------
# 3. Business-day compounding inside a single Selic regime
# ---------------------------------------------------------------------------
def test_single_regime_compounding():
    # From 2022-08-04 (0.1375) until 2023-08-02 the COPOM rate is a flat 0.1375.
    # Pick a window fully inside that regime (Feb 2023) so every business day
    # should equal (1 + 0.1375)^(1/252) - 1.
    annual = 0.1375
    expected = (1 + annual) ** (1 / 252) - 1

    dates = pd.date_range("2023-02-06", "2023-02-24", freq="D")  # Mon..Fri span
    rf = build_rf_daily(dates, source="copom")

    # Sample a known business day: 2023-02-08 (a Wednesday)
    sample = pd.Timestamp("2023-02-08")
    idx = list(pd.DatetimeIndex(dates)).index(sample)
    assert sample.dayofweek < 5  # sanity: it is a weekday
    assert rf[idx] == pytest.approx(expected, rel=1e-12)

    # And ALL business days in this single-regime window share that value.
    dow = pd.DatetimeIndex(dates).dayofweek
    biz_mask = dow < 5
    assert np.allclose(rf[biz_mask], expected, rtol=1e-12)


def test_single_regime_different_rate():
    # 2025-06-19 onward the rate is 0.1500 (until 2026-03-19). Sample inside it.
    annual = 0.1500
    expected = (1 + annual) ** (1 / 252) - 1
    dates = pd.date_range("2025-07-01", "2025-07-31", freq="D")
    rf = build_rf_daily(dates, source="copom")
    sample = pd.Timestamp("2025-07-09")  # a Wednesday
    idx = list(pd.DatetimeIndex(dates)).index(sample)
    assert sample.dayofweek < 5
    assert rf[idx] == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# 4. Full-year 2023 cumulative CDI sane band + business-day count
# ---------------------------------------------------------------------------
def test_full_year_2023_cum_and_bizdays():
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    rf = build_rf_daily(dates, source="copom")

    cum = np.prod(1 + rf) - 1
    # 2023 Selic ran 13.75% -> 11.75%; realized CDI lands ~12-15%.
    assert 0.12 <= cum <= 0.15, f"2023 cum CDI out of band: {cum:.4f}"

    # Number of accruing (business) days should be a normal calendar's worth.
    biz_days = int((rf > 0).sum())
    assert 250 <= biz_days <= 262, f"unexpected business-day count: {biz_days}"


# ---------------------------------------------------------------------------
# 5. Determinism: identical inputs -> identical outputs
# ---------------------------------------------------------------------------
def test_determinism_same_dates():
    dates = pd.date_range("2024-01-01", "2024-06-30", freq="D")
    rf1 = build_rf_daily(dates, source="copom")
    rf2 = build_rf_daily(dates, source="copom")
    assert np.array_equal(rf1, rf2)


def test_determinism_independent_index_objects():
    # Build two separate (but equal) date indices to be sure no mutation/caching
    # leaks between calls.
    d1 = pd.date_range("2024-01-01", "2024-03-31", freq="D")
    d2 = pd.date_range("2024-01-01", "2024-03-31", freq="D")
    rf1 = build_rf_daily(d1, source="copom")
    rf2 = build_rf_daily(d2, source="copom")
    assert np.array_equal(rf1, rf2)


# ---------------------------------------------------------------------------
# Extra: dates before the first COPOM entry fall back to the earliest rate
# ---------------------------------------------------------------------------
def test_dates_before_history_use_first_rate():
    first_rate = COPOM_HISTORY[0][1]  # 0.0650
    expected = (1 + first_rate) ** (1 / 252) - 1
    # 2018-01-03 is a Wednesday, before the first COPOM date (2019-01-01).
    dates = pd.DatetimeIndex(["2018-01-03"])
    rf = build_rf_daily(dates, source="copom")
    assert rf[0] == pytest.approx(expected, rel=1e-12)
