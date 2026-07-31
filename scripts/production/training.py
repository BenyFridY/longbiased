"""
Training helpers used by production (generate_signal.py, rebuild_signal_history.py).

These were originally in archive/old_pipelines/pipeline_v13.py (with
get_year_boundaries coming from pipeline_v09.py). Moved here so production is
self-contained and does not depend on files under archive/.
"""
import numpy as np
import pandas as pd
import xgboost as xgb


def _train_one_xgb(args):
    seed, Xt, yt, params_base = args
    p = dict(params_base)
    p['random_state'] = seed
    ne = p.pop('n_estimators', 200)
    m = xgb.XGBRegressor(n_estimators=ne, **p)
    m.fit(Xt, yt)
    return m


# NOTE: get_year_boundaries + _get_retrain_periods are used ONLY by the archive
# script archive/experiments/rebuild_signal_history.py, not by the live pipeline
# (generate_signal.py imports only _train_one_xgb). Kept here for that import;
# safe to delete if that archive script is retired. walkforward_backtest.py has
# its own retrain_cutoffs() and does not use these.
def get_year_boundaries(dates, n):
    yb = {}
    for i in range(n):
        yr = pd.Timestamp(dates[i]).year
        if yr not in yb:
            yb[yr] = {'start': i, 'end': i}
        yb[yr]['end'] = i
    return yb


def _get_retrain_periods(dates, n, retrain_freq='annual'):
    """Return list of (train_end_idx, test_start_idx, test_end_idx)."""
    yb = get_year_boundaries(dates, n)
    periods = []

    if retrain_freq == 'annual':
        for yr in range(2022, 2027):
            if yr not in yb:
                continue
            periods.append((yb[yr]['start'] - 1, yb[yr]['start'], yb[yr]['end']))

    elif retrain_freq == 'quarterly':
        dt_dates = pd.to_datetime(dates)
        for yr in range(2022, 2027):
            if yr not in yb:
                continue
            for q in range(1, 5):
                q_start_m = (q - 1) * 3 + 1
                q_end_m = q * 3
                q_mask = (dt_dates.year == yr) & (dt_dates.month >= q_start_m) & (dt_dates.month <= q_end_m)
                q_indices = np.where(q_mask)[0]
                if len(q_indices) == 0:
                    continue
                test_start = q_indices[0]
                test_end = q_indices[-1]
                train_end = test_start - 1
                if train_end >= 60:
                    periods.append((train_end, test_start, test_end))

    elif retrain_freq == 'semi':
        dt_dates = pd.to_datetime(dates)
        for yr in range(2022, 2027):
            if yr not in yb:
                continue
            for half in [1, 2]:
                if half == 1:
                    h_mask = (dt_dates.year == yr) & (dt_dates.month <= 6)
                else:
                    h_mask = (dt_dates.year == yr) & (dt_dates.month > 6)
                h_indices = np.where(h_mask)[0]
                if len(h_indices) == 0:
                    continue
                test_start = h_indices[0]
                test_end = h_indices[-1]
                train_end = test_start - 1
                if train_end >= 60:
                    periods.append((train_end, test_start, test_end))
    return periods
