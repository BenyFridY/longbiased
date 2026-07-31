"""
Bootstrap production dataset from original dataset_enhanced.csv.

Two modes:
  - Full:        if dataset_production.csv does NOT exist yet. Reads the
                 enhanced base (2019 → 2026-03-03), computes features from
                 raw for everything after that, concatenates.
  - Incremental: if dataset_production.csv already exists. Computes features
                 only for the new days (with a 400-day warmup so that rolling
                 windows like price_percentile_1y / m2_yoy_growth are valid),
                 then appends. ~10x faster than full on a typical daily run.

Usage:
    python scripts/production/bootstrap_from_original.py
"""
import sys, logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.production.config import DATASET_PATH, FEATURES_37
from scripts.production.build_features import build_features

ORIG_DATASET = ROOT / "outputs" / "feature_selection" / "dataset_enhanced.csv"
RAW_CSV = Path(__file__).parent / "data" / "raw_data.csv"

# Longest rolling window in build_features is 365 (price_percentile_1y) and
# 52-week for m2_yoy_growth. 400 days of warmup guarantees all rolling
# features converge to the same value they would have in a full recompute.
WARMUP_DAYS = 400


def _keep_cols(orig):
    cols = ['date', 'price_usd'] + [f for f in FEATURES_37 if f in orig.columns]
    return list(dict.fromkeys(cols))


# V36/E1 features added to FEATURES_37 but NOT present in dataset_enhanced.csv
# These are backfilled from raw_data.csv (processed through build_features.py)
V36_NEW_FEATURES = ['reserveRisk', 'puellMultiple', 'funding_rate_ma7']


def _backfill_new_features(orig_slim, full_feat):
    """Backfill V36/E1 new features into enhanced base.
       Always uses build_features output (not enhanced's stale values) to match V36 validation."""
    # Drop enhanced versions of V36 features if they exist (use build_features instead — same computation as V36 tests)
    to_drop = [f for f in V36_NEW_FEATURES if f in orig_slim.columns]
    if to_drop:
        log.info(f"Dropping enhanced versions of V36 features (will use build_features): {to_drop}")
        orig_slim = orig_slim.drop(columns=to_drop)
    available = [f for f in V36_NEW_FEATURES if f in full_feat.columns]
    if not available:
        return orig_slim
    log.info(f"Backfilling {len(available)} V36 features from build_features: {available}")
    merged = orig_slim.merge(full_feat[['date'] + available], on='date', how='left')
    return merged


def full_bootstrap():
    log.info("FULL bootstrap: enhanced base + build_features over all raw")
    orig = pd.read_csv(ORIG_DATASET)
    orig['date'] = pd.to_datetime(orig['date']).dt.strftime('%Y-%m-%d')
    last_orig_date = orig['date'].max()
    log.info(f"Original: {len(orig)} rows, to {last_orig_date}")

    keep = _keep_cols(orig)
    orig_slim = orig[keep].copy()

    raw = pd.read_csv(RAW_CSV)
    raw['date'] = pd.to_datetime(raw['date']).dt.strftime('%Y-%m-%d')
    log.info(f"Raw data: {len(raw)} rows, to {raw['date'].max()}")

    prod_full = build_features(raw.copy())
    prod_full['date'] = pd.to_datetime(prod_full['date']).dt.strftime('%Y-%m-%d')

    # V36/E1: backfill new on-chain features (reserveRisk, puellMultiple, funding_rate_ma7)
    # into enhanced base — they exist in raw (via build_features) but not in dataset_enhanced.csv
    orig_slim = _backfill_new_features(orig_slim, prod_full)
    keep_final = list(orig_slim.columns)

    new_days = prod_full[prod_full['date'] > last_orig_date].copy()
    log.info(f"New days to append: {len(new_days)}")

    if len(new_days) > 0:
        new_slim = new_days.reindex(columns=keep_final)
        combined = pd.concat([orig_slim, new_slim], ignore_index=True)
        combined = combined.drop_duplicates(subset='date', keep='first')
        combined = combined.sort_values('date').reset_index(drop=True)
    else:
        combined = orig_slim

    return combined, keep_final


def incremental_bootstrap():
    log.info(f"INCREMENTAL bootstrap (warmup={WARMUP_DAYS}d)")
    prod = pd.read_csv(DATASET_PATH)
    prod['date'] = pd.to_datetime(prod['date']).dt.strftime('%Y-%m-%d')
    last_prod_date = prod['date'].max()
    log.info(f"Existing dataset: {len(prod)} rows, to {last_prod_date}")

    raw = pd.read_csv(RAW_CSV)
    raw['date'] = pd.to_datetime(raw['date']).dt.strftime('%Y-%m-%d')
    last_raw_date = raw['date'].max()
    log.info(f"Raw data: {len(raw)} rows, to {last_raw_date}")

    if last_raw_date <= last_prod_date:
        log.info("Dataset is already up to date — nothing to do.")
        return prod, list(prod.columns)

    # Warmup window: WARMUP_DAYS before the first new day
    first_new_date = (pd.to_datetime(last_prod_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    warmup_start = (pd.to_datetime(first_new_date) - pd.Timedelta(days=WARMUP_DAYS)).strftime('%Y-%m-%d')

    slice_ = raw[raw['date'] >= warmup_start].copy().reset_index(drop=True)
    log.info(f"Building features on {len(slice_)} rows ({warmup_start} → {last_raw_date})")

    built = build_features(slice_)
    new_days = built[built['date'] > last_prod_date].copy()
    log.info(f"New days to append: {len(new_days)}")

    keep = list(prod.columns)
    new_slim = new_days.reindex(columns=keep)
    combined = pd.concat([prod, new_slim], ignore_index=True)
    combined = combined.drop_duplicates(subset='date', keep='first')
    combined = combined.sort_values('date').reset_index(drop=True)
    return combined, keep


def main():
    log.info("=" * 60)
    log.info("BOOTSTRAP")
    log.info("=" * 60)

    if not RAW_CSV.exists():
        log.error("No raw_data.csv found. Run fetch_raw_data.py first.")
        return

    # Decide mode: incremental if we already have a production dataset,
    # full otherwise.
    if DATASET_PATH.exists():
        try:
            combined, keep = incremental_bootstrap()
        except Exception as e:
            log.warning(f"Incremental failed ({e}) — falling back to full bootstrap")
            combined, keep = full_bootstrap()
    else:
        combined, keep = full_bootstrap()

    # Forward-fill any residual NaN in the transition zone
    for f in FEATURES_37:
        if f in combined.columns:
            combined[f] = combined[f].ffill().fillna(0)

    log.info(f"\nFinal dataset: {len(combined)} rows, {combined['date'].min()} to {combined['date'].max()}")
    missing = [f for f in FEATURES_37 if f not in combined.columns]
    if missing:
        log.warning(f"  MISSING: {missing}")
        for f in missing:
            combined[f] = 0.0
    else:
        log.info(f"  All {len(FEATURES_37)} features present")

    combined.to_csv(DATASET_PATH, index=False)
    log.info(f"Saved: {DATASET_PATH} ({len(combined)} rows)")


if __name__ == '__main__':
    main()
