"""
SHAP explainability for the production regressor (feature attribution).

Trains a representative XGBoost regressor with the production params/features on
the full dataset and computes SHAP values (TreeExplainer). Produces:
  - outputs/charts/fig_shap_summary.png  (beeswarm summary, top 20 features)
  - prints the top features by mean |SHAP|

This is a GLOBAL attribution on a single representative model; the production
ensemble averages 160 such regressors. Mitigates the opacity of the ensemble
(see docs/EXPLAINABILITY.md, docs/RISCOS_ETICOS.md, docs/OVERFIT_TESTS_2026-04-22.md).

Usage:
    python scripts/production/archive/experiments/shap_explainability.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts" / "production"))

from config import FEATURES_37 as FEATURES, XGB_PARAMS, HORIZON

ds = (
    pd.read_csv(ROOT / "scripts" / "production" / "data" / "dataset_production.csv",
                parse_dates=["date"])
    .sort_values("date")
    .reset_index(drop=True)
)

X = np.nan_to_num(ds[FEATURES].values.astype(float), nan=0.0)
prices = ds["price_usd"].values
n = len(ds)
y = np.zeros(n)
for i in range(n - HORIZON):
    y[i] = (prices[i + HORIZON] - prices[i]) / prices[i]

# Train on rows with a valid target and past the warmup window.
idx = np.arange(60, n - HORIZON)
idx = idx[~np.any(np.isnan(X[idx]), axis=1)]

params = {k: v for k, v in XGB_PARAMS.items()}
model = xgb.XGBRegressor(**params, random_state=242)
model.fit(X[idx], y[idx])

explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X[idx])

# Beeswarm summary
plt.figure()
shap.summary_plot(sv, ds[FEATURES].iloc[idx], show=False, max_display=20)
plt.title("SHAP feature attribution - 3d BTC return regressor", fontsize=11)
plt.tight_layout()
out = ROOT / "outputs" / "charts" / "fig_shap_summary.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
plt.close()

mean_abs = np.abs(sv).mean(axis=0)
order = np.argsort(mean_abs)[::-1]
print("TOP FEATURES by mean |SHAP|:")
for i in order[:15]:
    print(f"  {FEATURES[i]:30s} {mean_abs[i]:.6f}")
print(f"\nTrained on {len(idx)} samples, {len(FEATURES)} features.")
print(f"Saved: {out}")
