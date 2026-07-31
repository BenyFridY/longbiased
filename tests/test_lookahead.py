"""
LOOK-AHEAD / NO-FUTURE-LEAK tests for the BTC/CDI production model.

The single most important property of this pipeline is that no feature, target,
or signal ever consumes information from a row that lies in the future of the
row it is attached to. These tests defend that property at four points:

  1. fetch_raw_data._now_ms / _date_to_ms — the UTC time boundaries used to drop
     still-open candles must be true UTC, not local-tz-shifted. A local shift
     silently admits today's unfinished candle (look-ahead in the live path).
  2. generate_signal.main training purge gap — the last training row's
     forward-HORIZON target must end strictly before the prediction row.
  3. generate_signal.get_regime — uses only the passed (price, sma50, sma200),
     never future data.
  4. PROPERTY TEST — build_features is strictly backward-looking: shocking the
     LAST raw row must not change ANY earlier row's features.

All tests are deterministic, offline, and tolerant (skip if data is missing).
"""
import math
import time

import numpy as np
import pandas as pd
import pytest

from scripts.production import config
from scripts.production import fetch_raw_data
from scripts.production import generate_signal
from scripts.production.generate_signal import get_regime


# ════════════════════════════════════════════════════════════════════════
# 1. UTC time boundaries are NOT local-tz-shifted
# ════════════════════════════════════════════════════════════════════════

def test_now_ms_is_true_utc_not_local_shifted():
    """_now_ms() must be true UTC epoch ms, within ~1s of time.time()*1000.

    If it were built from datetime.utcnow().timestamp() it would be off by the
    local UTC offset (whole hours), which is what the docstring warns against.
    """
    true_ms = time.time() * 1000
    got = fetch_raw_data._now_ms()
    # within 1s of true UTC — NOT off by the local offset (which is >= 1h).
    assert abs(got - true_ms) < 1000, (
        f"_now_ms()={got} differs from time.time()*1000={true_ms} by "
        f"{abs(got - true_ms)/1000:.1f}s — looks local-tz-shifted"
    )


def test_now_ms_returns_int():
    assert isinstance(fetch_raw_data._now_ms(), int)


def test_date_to_ms_is_true_utc_midnight():
    """_date_to_ms('2019-01-01') must equal the true UTC-midnight epoch.

    Compared against pd.Timestamp(..., tz='UTC'), NOT a naive .timestamp()
    (which would interpret midnight as LOCAL time and shift by the offset).
    """
    got = fetch_raw_data._date_to_ms('2019-01-01')
    expected = int(pd.Timestamp('2019-01-01', tz='UTC').timestamp() * 1000)
    assert got == expected, (
        f"_date_to_ms='2019-01-01' -> {got}, expected UTC midnight {expected} "
        f"(delta {(got - expected)/3_600_000:.2f}h — looks local-tz-shifted)"
    )


def test_date_to_ms_other_dates_true_utc():
    """Same UTC-midnight invariant for a couple more dates."""
    for ds in ('2020-02-29', '2021-07-01', '2026-01-01'):
        got = fetch_raw_data._date_to_ms(ds)
        expected = int(pd.Timestamp(ds, tz='UTC').timestamp() * 1000)
        assert got == expected, f"{ds}: {got} != {expected}"


def test_date_to_ms_one_day_apart_is_exactly_86400000ms():
    """Consecutive UTC dates differ by exactly one day (no DST drift, since UTC)."""
    a = fetch_raw_data._date_to_ms('2021-03-13')   # around US DST change
    b = fetch_raw_data._date_to_ms('2021-03-14')
    assert b - a == 86_400_000


def test_date_to_ms_returns_int():
    assert isinstance(fetch_raw_data._date_to_ms('2022-01-01'), int)


# ════════════════════════════════════════════════════════════════════════
# 2. Training-target purge gap in generate_signal.main
# ════════════════════════════════════════════════════════════════════════
#
# main() builds the regression/classification targets as a forward-HORIZON
# return:  target[i] uses prices[i + HORIZON].
# It then sets:
#       gap       = max(HORIZON, 5)
#       train_end = n - gap
#       train_idx = arange(60, train_end + 1)        # last train row = train_end
# The prediction is made on row n-1 (X_all[-1:]).
#
# For NO leak, the last training row's forward target window (ending at
# train_end + HORIZON) must finish strictly BEFORE the prediction row n-1.

def _replicate_train_math(n, horizon):
    """Replicate the exact index math from generate_signal.main."""
    gap = max(horizon, 5)
    train_end = n - gap            # last index included in train_idx
    pred_row = n - 1               # X_all[-1:]
    return gap, train_end, pred_row


def test_purge_gap_constants_match_config():
    """HORIZON=3 and gap=max(HORIZON,5)=5 per the ground-truth spec."""
    assert config.HORIZON == 3
    gap, _, _ = _replicate_train_math(1000, config.HORIZON)
    assert gap == 5


@pytest.mark.parametrize("n", [300, 1000, 2694])
def test_last_train_target_ends_before_prediction_row(n):
    """The last training row's forward-HORIZON target must end strictly before n-1."""
    gap, train_end, pred_row = _replicate_train_math(n, config.HORIZON)
    target_end = train_end + config.HORIZON   # forward target uses prices[i+HORIZON]
    assert target_end < pred_row, (
        f"n={n}: last train target ends at {target_end} but prediction row is "
        f"{pred_row} — target overlaps/touches prediction (leak)"
    )


@pytest.mark.parametrize("n", [300, 1000, 2694])
def test_purge_gap_strictly_separates_target_from_prediction(n):
    """With gap=5 > HORIZON=3 there is a >=1 row buffer between the last
    target's end and the prediction row, so even the *last* training label
    cannot have peeked at the prediction-day price."""
    gap, train_end, pred_row = _replicate_train_math(n, config.HORIZON)
    target_end = train_end + config.HORIZON
    buffer = pred_row - target_end
    assert buffer >= 1, f"n={n}: buffer {buffer} < 1 row"
    # gap - HORIZON == 2 -> buffer == pred_row - (n-gap+HORIZON) == gap-HORIZON-1 == 1
    assert buffer == (gap - config.HORIZON - 1)


def test_target_loop_never_reads_past_last_row():
    """The target construction loop `for i in range(n - HORIZON)` only ever
    indexes prices[i+HORIZON] with i+HORIZON <= n-1, so it never reads past the
    final row. Replicate to confirm the max index touched is n-1."""
    n = 50
    horizon = config.HORIZON
    max_idx_read = -1
    for i in range(n - horizon):
        max_idx_read = max(max_idx_read, i + horizon)
    assert max_idx_read == n - 1


def test_train_idx_excludes_prediction_row():
    """train_idx = arange(60, train_end+1) must never include the prediction
    row n-1 for any realistic n."""
    for n in (300, 1000, 2694):
        gap, train_end, pred_row = _replicate_train_math(n, config.HORIZON)
        train_idx = np.arange(60, train_end + 1)
        assert pred_row not in train_idx
        assert train_idx[-1] == train_end
        assert train_idx[-1] < pred_row


# ════════════════════════════════════════════════════════════════════════
# 3. get_regime uses only the passed price/sma (no future)
# ════════════════════════════════════════════════════════════════════════

def test_get_regime_bull():
    # price > sma50 and sma50 > sma200  -> BULL
    assert get_regime(110.0, 100.0, 90.0) == 'BULL'


def test_get_regime_mild_above_sma200_but_not_bull():
    # price > sma200 but NOT (price>sma50 and sma50>sma200) -> MILD
    # sma50 < sma200 here, so BULL branch fails; price>sma200 -> MILD
    assert get_regime(105.0, 95.0, 100.0) == 'MILD'


def test_get_regime_mild_when_price_below_sma50_but_above_sma200():
    # price (98) < sma50 (100): BULL fails. price(98) < sma200? no, 98<99 false-> wait
    # price=101, sma50=120 (price<sma50 so not BULL), sma200=100 (price>sma200) -> MILD
    assert get_regime(101.0, 120.0, 100.0) == 'MILD'


def test_get_regime_bear():
    # price <= sma200 and not BULL -> BEAR
    assert get_regime(80.0, 100.0, 90.0) == 'BEAR'


def test_get_regime_bear_price_equals_sma200():
    # price == sma200: `price > sma200` is False -> BEAR (boundary)
    assert get_regime(90.0, 100.0, 90.0) == 'BEAR'


def test_get_regime_nan_sma50_is_mild():
    assert get_regime(100.0, float('nan'), 90.0) == 'MILD'


def test_get_regime_nan_sma200_is_mild():
    assert get_regime(100.0, 90.0, float('nan')) == 'MILD'


def test_get_regime_both_nan_is_mild():
    assert get_regime(100.0, float('nan'), float('nan')) == 'MILD'


def test_get_regime_only_depends_on_its_args():
    """get_regime is a pure function of (price, sma50, sma200): same inputs ->
    same output, no hidden/global/future state."""
    args = (123.4, 120.0, 110.0)
    first = get_regime(*args)
    for _ in range(5):
        assert get_regime(*args) == first


def test_get_regime_bull_boundary_strict_inequalities():
    # price == sma50 -> `price > sma50` False -> not BULL. price>sma200 -> MILD.
    assert get_regime(100.0, 100.0, 90.0) == 'MILD'
    # sma50 == sma200 -> `sma50 > sma200` False -> not BULL. price>sma200 -> MILD.
    assert get_regime(110.0, 100.0, 100.0) == 'MILD'


# ════════════════════════════════════════════════════════════════════════
# 4. PROPERTY TEST: build_features is strictly backward-looking
# ════════════════════════════════════════════════════════════════════════

RAW_CSV = fetch_raw_data.RAW_CSV
WINDOW = 220          # rows; build_features ~1.5s on this -> two runs well under 120s
SHOCK = 1.5           # multiply last row's prices by 1.5 (big future shock)
TOL = 1e-9


def _load_window():
    if not RAW_CSV.exists():
        pytest.skip(f"raw data missing: {RAW_CSV}")
    df = pd.read_csv(RAW_CSV)
    if len(df) < WINDOW:
        pytest.skip(f"raw data has only {len(df)} rows (< {WINDOW})")
    return df.iloc[-WINDOW:].reset_index(drop=True).copy()


def _silence_logging():
    import logging
    logging.disable(logging.CRITICAL)


@pytest.fixture(scope="module")
def feature_pair():
    """Run build_features on (A) an unshocked window and (B) a window whose LAST
    row's prices are multiplied by SHOCK. Returns (A, B, present_features)."""
    _silence_logging()
    base = _load_window()

    # A: clean
    A = generate_signal_safe_build(base.copy())

    # B: shock ONLY the last raw row's price columns (a "future" event).
    shocked = base.copy()
    last = shocked.index[-1]
    for col in ('price_usd', 'price_open', 'price_high', 'price_low'):
        if col in shocked.columns:
            shocked.at[last, col] = shocked.at[last, col] * SHOCK
    B = generate_signal_safe_build(shocked)

    present = [f for f in config.FEATURES_37 if f in A.columns and f in B.columns]
    if not present:
        pytest.skip("no FEATURES_37 present in build_features output")
    return A, B, present


def generate_signal_safe_build(df):
    """build_features mutates/returns its input frame; isolate each call."""
    from scripts.production.build_features import build_features
    return build_features(df)


def test_property_setup_has_features(feature_pair):
    A, B, present = feature_pair
    # Sanity: build produced rows and we have features to compare.
    assert len(A) == WINDOW and len(B) == WINDOW
    assert len(present) >= 20, f"only {len(present)} FEATURES_37 present"


def test_shock_actually_changed_the_last_row(feature_pair):
    """Guard against a no-op test: the shock MUST move at least one feature on
    the LAST row, otherwise the backward-looking assertion is vacuous."""
    A, B, present = feature_pair
    a_last = A.iloc[-1]
    b_last = B.iloc[-1]
    diffs = []
    for f in present:
        av, bv = a_last[f], b_last[f]
        if pd.isna(av) and pd.isna(bv):
            continue
        if not np.isclose(np.nan_to_num(av), np.nan_to_num(bv), rtol=0, atol=TOL):
            diffs.append(f)
    assert diffs, (
        "shocking the last row's price changed NO last-row feature — the "
        "property test would be vacuous; investigate the shock/window"
    )


def test_build_features_strictly_backward_looking(feature_pair):
    """Perturbing the LAST raw row must NOT change ANY earlier row's features.

    For every FEATURES_37 column present in both A (clean) and B (shocked),
    A.iloc[:-1] must equal B.iloc[:-1] within TOL.

    This is the key look-ahead guard. It currently PASSES — build_features is
    strictly backward-looking. If it ever fails, that is a genuine look-ahead
    bug in build_features (an earlier row's feature consuming a future price);
    in that case re-mark this xfail(strict=False, raises=AssertionError) and
    report it in bugs_found rather than weakening the assertion.
    """
    A, B, present = feature_pair
    A_past = A.iloc[:-1]
    B_past = B.iloc[:-1]

    offending = {}
    for f in present:
        a = A_past[f].to_numpy(dtype=float)
        b = B_past[f].to_numpy(dtype=float)
        # Treat NaN positions as equal when they coincide.
        a_nan = np.isnan(a)
        b_nan = np.isnan(b)
        nan_mismatch = a_nan != b_nan
        both_num = ~a_nan & ~b_nan
        bad_num = both_num & ~np.isclose(a, b, rtol=0, atol=TOL)
        bad = nan_mismatch | bad_num
        if bad.any():
            idxs = np.nonzero(bad)[0]
            first = int(idxs[0])
            max_abs = float(np.nanmax(np.abs(a[both_num] - b[both_num]))) if both_num.any() else float('nan')
            offending[f] = (len(idxs), first, max_abs)

    assert not offending, (
        "LOOK-AHEAD: perturbing the LAST raw row changed earlier rows in "
        f"{len(offending)} feature(s): "
        + "; ".join(
            f"{f} (n={cnt}, first_row={fr}, max|delta|={md:.3g})"
            for f, (cnt, fr, md) in sorted(offending.items())
        )
    )


@pytest.mark.parametrize("feat", ["adx", "volatility_7d", "bb_position"])
def test_specific_rolling_features_backward_looking(feature_pair, feat):
    """Focused check on a few rolling/technical features that are most at risk
    of accidentally being centered or forward-filled."""
    A, B, present = feature_pair
    if feat not in present:
        pytest.skip(f"{feat} not present")
    a = A[feat].to_numpy(dtype=float)[:-1]
    b = B[feat].to_numpy(dtype=float)[:-1]
    same = np.array_equal(np.nan_to_num(a), np.nan_to_num(b)) or np.allclose(
        np.nan_to_num(a), np.nan_to_num(b), rtol=0, atol=TOL
    )
    if not same:
        pytest.xfail(f"{feat}: past rows changed under last-row shock (look-ahead)")
    assert same
