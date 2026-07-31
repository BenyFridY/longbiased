"""
CDI / Selic daily rates from Banco Central do Brasil (BCB).

Primary source: BCB SGS API (series 12 = CDI daily overnight rate)
Fallback:       Hardcoded COPOM decision history (no internet needed)

rf_daily is defined per date: business days accrue, weekends/holidays get 0
(CDI does not accrue). Business-day compounding: (1 + annual) ^ (1/252) - 1.

BCB API returns rates only for business days. We assign those to the
actual business day and set weekends/holidays to 0 (CDI does not accrue).

Usage:
    from src.features.macro.cdi_rates import build_rf_daily

    rf = build_rf_daily(dates)                  # auto: BCB API -> COPOM fallback
    rf = build_rf_daily(dates, source='copom')  # offline only
"""

import numpy as np
import pandas as pd
from typing import Union
import logging

logger = logging.getLogger(__name__)

# ============================================================
# COPOM SELIC TARGET RATE HISTORY (offline fallback)
# Source: https://www.bcb.gov.br/controleinflacao/historicotaxasjuros
# ============================================================
COPOM_HISTORY = [
    # (effective_date, annual_rate)
    ('2019-01-01', 0.0650),
    ('2019-08-01', 0.0600),
    ('2019-09-19', 0.0550),
    ('2019-10-31', 0.0500),
    ('2019-12-12', 0.0450),
    ('2020-02-06', 0.0425),
    ('2020-03-19', 0.0375),
    ('2020-05-07', 0.0300),
    ('2020-06-18', 0.0225),
    ('2020-08-06', 0.0200),
    ('2021-03-18', 0.0275),
    ('2021-05-06', 0.0350),
    ('2021-06-17', 0.0425),
    ('2021-08-05', 0.0525),
    ('2021-09-23', 0.0625),
    ('2021-10-28', 0.0775),
    ('2021-12-09', 0.0925),
    ('2022-02-03', 0.1075),
    ('2022-03-17', 0.1175),
    ('2022-05-05', 0.1275),
    ('2022-06-16', 0.1325),
    ('2022-08-04', 0.1375),
    ('2023-08-03', 0.1325),
    ('2023-09-21', 0.1275),
    ('2023-11-02', 0.1225),
    ('2023-12-14', 0.1175),
    ('2024-01-31', 0.1125),
    ('2024-03-20', 0.1075),
    ('2024-05-08', 0.1050),
    ('2024-09-18', 0.1075),
    ('2024-11-06', 0.1125),
    ('2024-12-11', 0.1225),
    ('2025-01-29', 0.1325),
    ('2025-03-19', 0.1425),
    ('2025-05-07', 0.1475),
    ('2025-06-19', 0.1500),
    ('2026-03-19', 0.1475),
]


def _fetch_bcb_cdi(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch daily CDI rates from BCB SGS API (series 12).
    Returns DataFrame with columns ['date', 'cdi_daily'] where cdi_daily is
    the actual overnight rate as a decimal (e.g. 0.000507 for ~13.25% annual).
    Only business-day rows are returned (no weekends/holidays).
    """
    import urllib.request
    import json

    url = (
        f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados"
        f"?formato=json&dataInicial={start_date}&dataFinal={end_date}"
    )

    logger.info(f"Fetching CDI from BCB: {url}")
    req = urllib.request.Request(url)
    req.add_header('Accept', 'application/json')

    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = json.loads(resp.read().decode('utf-8'))

    rows = []
    for entry in raw:
        dt = pd.to_datetime(entry['data'], dayfirst=True)
        rate = float(entry['valor']) / 100  # BCB returns % -> decimal
        rows.append({'date': dt, 'cdi_daily': rate})

    df = pd.DataFrame(rows)
    logger.info(f"  Got {len(df)} CDI business-day observations from BCB")
    return df


def _build_from_bcb(dates: pd.DatetimeIndex) -> np.ndarray:
    """Build daily RF array from BCB API. Weekends/holidays get rf=0."""
    start_str = dates.min().strftime('%d/%m/%Y')
    end_str = dates.max().strftime('%d/%m/%Y')

    cdi_df = _fetch_bcb_cdi(start_str, end_str)
    if cdi_df.empty:
        raise ValueError("BCB API returned empty data")

    cdi_df = cdi_df.set_index('date').sort_index()

    # Reindex to all calendar dates — non-business days stay NaN -> fill with 0
    full_idx = pd.DatetimeIndex(dates)
    cdi_reindexed = cdi_df['cdi_daily'].reindex(full_idx).fillna(0.0)

    return cdi_reindexed.values


def _build_from_copom(dates: pd.DatetimeIndex) -> np.ndarray:
    """Build daily RF array from COPOM target rate history (offline).
    Uses business-day compounding: (1 + annual)^(1/252) - 1.
    Weekends get rf=0 to match BCB behavior.
    """
    n = len(dates)
    ts_dates = pd.DatetimeIndex(dates)
    rf_arr = np.zeros(n)

    copom_dates = [pd.Timestamp(d) for d, _ in COPOM_HISTORY]
    copom_rates = [r for _, r in COPOM_HISTORY]

    for i in range(n):
        d = ts_dates[i]
        # Weekends: no accrual
        if d.dayofweek >= 5:
            rf_arr[i] = 0.0
            continue
        # Find most recent COPOM rate
        rate = copom_rates[0]
        for j in range(len(copom_dates)):
            if copom_dates[j] <= d:
                rate = copom_rates[j]
            else:
                break
        # CDI convention: annual rate over ~252 business days
        rf_arr[i] = (1 + rate) ** (1 / 252) - 1

    return rf_arr


def build_rf_daily(
    dates: Union[np.ndarray, pd.DatetimeIndex, pd.Series],
    source: str = 'auto',
) -> np.ndarray:
    """
    Build array of daily risk-free rates aligned with the given dates.

    Weekends/holidays = 0 (CDI does not accrue). Business days = actual rate.

    Args:
        dates: array-like of dates (same length as the dataset)
        source: 'auto' (try BCB API, fallback to COPOM),
                'bcb'  (BCB API only, raises on failure),
                'copom' (offline COPOM history only)

    Returns:
        numpy array of daily rf rates, same length as dates.
    """
    ts_dates = pd.DatetimeIndex(dates)

    if source == 'copom':
        logger.info("Using COPOM target rate history (offline)")
        return _build_from_copom(ts_dates)

    if source in ('auto', 'bcb'):
        try:
            arr = _build_from_bcb(ts_dates)
            logger.info("Using real CDI rates from BCB API")
            return arr
        except Exception as e:
            if source == 'bcb':
                raise
            logger.warning(f"BCB API failed ({e}), falling back to COPOM history")
            return _build_from_copom(ts_dates)

    raise ValueError(f"Unknown source: {source!r}. Use 'auto', 'bcb', or 'copom'.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    dates = pd.date_range('2022-01-01', '2026-01-31', freq='D')
    old_daily = (1 + 0.15) ** (1 / 365) - 1  # old flat 15% over 365

    print("Trying BCB API first, then COPOM fallback...\n")
    rf = build_rf_daily(dates, source='auto')

    print(f"{'Year':<8} {'Real CDI':>10} {'Old flat':>10} {'Diff':>8}")
    print("-" * 40)
    for year in range(2022, 2027):
        mask = dates.year == year
        if mask.sum() == 0:
            continue
        real_cum = np.prod(1 + rf[mask]) - 1
        old_cum = (1 + old_daily) ** mask.sum() - 1
        print(f"{year:<8} {real_cum*100:>9.2f}% {old_cum*100:>9.2f}% {(real_cum-old_cum)*100:>+7.2f}pp")

    real_total = np.prod(1 + rf) - 1
    old_total = (1 + old_daily) ** len(dates) - 1
    print("-" * 40)
    print(f"{'Total':<8} {real_total*100:>9.2f}% {old_total*100:>9.2f}% {(real_total-old_total)*100:>+7.2f}pp")

    # Sanity: check annual rates match expectation
    print(f"\nSanity check — 2023 had Selic 11.75-13.75%:")
    mask_23 = dates.year == 2023
    cum_23 = np.prod(1 + rf[mask_23]) - 1
    biz_days_23 = (rf[mask_23] > 0).sum()
    print(f"  CDI earned: {cum_23*100:.2f}% over {biz_days_23} business days")
