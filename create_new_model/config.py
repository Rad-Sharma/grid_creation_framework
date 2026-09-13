"""
config.py — Shared constants and model classes for the synthetic grid pipeline.

Imported by all pipeline scripts. Provides:
  - Region-specific settings (name, bounding box, country code)
  - NNRidge model class (must live here so joblib pickle can resolve it)
  - Pipeline thresholds and hyperparameters

To adapt this pipeline for a new region, update the region settings block below.
"""

import numpy as np
from scipy.optimize import minimize as _scipy_minimize

# ===========================================================================
# REGION SETTINGS — edit these for a new target region
# ===========================================================================

# Short lowercase name used in file names, e.g. matpower_{region}_buses.csv
region = "kenya"

# ISO 3-letter country code — used for NaturalEarth boundary clipping
COUNTRY_ISO3 = "KEN"

# Region bounding box: (min_lon, min_lat, max_lon, max_lat)
# Set to tightly bound the study area so that cross-border OSM features are excluded.
# Visit: https://github.com/sandstrom/country-bounding-boxes/blob/master/bounding-boxes.json
REGION_BBOX = (33.89, -4.68, 41.86, 5.51)   # (min_lon, min_lat, max_lon, max_lat)

# Country name as it appears in the NaturalEarth shapefile "name" column.
# Used by plot_grid_final.py to extract the country boundary polygon.
SHAPE_FILE_NAME = "Kenya"

# Sum of all domestic load bus Pd values (MW) from the target country's
# MATPOWER case. Exclude cross-border exchange buses.
# predict_bus_loads.py scales predictions to this total.
# Visit: https://ember-energy.org/data/electricity-data-explorer/
COUNTRY_DEMAND_MW = 2237

# Net national electricity export (MW) to be distributed equally across all
# external/cross-border buses. Set to 0 if the country is a net importer or
# if cross-border exchange is unknown.
NATIONAL_EXPORT_MW = 4.91

# --------------- Minimum transmission voltage ----------------
# Lines and substations with a KNOWN voltage below this threshold are removed.
# Eliminates sub-transmission and distribution assets that OSM tags as
# power=line / power=substation but are not part of the HV backbone.
#
# 110 kV is a common lower bound internationally. Change to 69000 for regions
# where 69 kV is considered transmission, or 132000 for UK-style networks.
MIN_TRANSMISSION_VOLTAGE_V = 115000   # Volts (110000 for 110 kV, 69000 for 69 kV, 132000 for UK-style networks)

# Manuscript-aligned load path. "legacy_ridge" explicitly selects the previous
# approximation. Paper mode requires measured industrial VIIRS data and a Huber
# artifact; it never silently substitutes Ridge, proxy light, or missing features.
LOAD_MODEL_METHOD = "paper_huber"
PAPER_HUBER_MODEL = "model_huber_paper.json"
VIIRS_RASTER_NAME = f"viirs_{region}.tif"
STEEL_FACILITIES_NAME = "steel_facilities_mw.geojson"
GOLD_FACILITIES_NAME = "gold_facilities.geojson"
PLANT_NAME_SEARCH_RADIUS_M = 1000


# ===========================================================================
# PREDICT BUS LOADS THRESHOLDS — edit these to tune load prediction
# ===========================================================================

# VIIRS NTL value (nW/cm²/sr) below which industrial features are treated as
# captive-powered and given zero grid credit.
NTL_CAPTIVE_THRESHOLD = 1.0   # nW/cm²/sr

# Buses within this radius (degrees, ~30 km at equator) are aggregated into
# one city-level training point to match the scale of NZ/GB training data.
CITY_CLUSTER_RADIUS_DEG = 0.27   # degrees (~30 km at equator)

# After city aggregation, clusters exceeding this load are capped to reduce
# the influence of megacities on the Ridge loss function.
MEGA_CITY_CAP_MW = 750.0   # MW


# ===========================================================================
# MODEL CLASS — must live here so joblib pickle can resolve it
# ===========================================================================

class NNRidge:
    """Ridge regression with non-negative coefficient constraint (coef >= 0).

    Physically motivated: every feature in RIDGE_FEATURES_RAW can only increase
    load. Enforcing non-negativity prevents collinearity-induced sign flips
    (e.g. road_density going negative when pop_density is also in the model).
    Implemented via L-BFGS-B with per-coefficient lower bounds of 0; the
    intercept is left unconstrained.

    Defined here (in config.py / the "constants" module) so that joblib pickle
    can resolve the class in both train_model.py and predict_bus_loads.py.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha      = alpha
        self.alpha_     = alpha
        self.coef_      = None
        self.intercept_ = None

    def fit(self, X, y):
        _, p   = X.shape
        x0     = np.zeros(p + 1)
        bounds = [(None, None)] + [(0.0, None)] * p   # intercept free; coefs >= 0

        def loss(params):
            b0, w = params[0], params[1:]
            resid = y - (X @ w + b0)
            return 0.5 * np.dot(resid, resid) + 0.5 * self.alpha * np.dot(w, w)

        def grad(params):
            b0, w = params[0], params[1:]
            resid = y - (X @ w + b0)
            return np.r_[-resid.sum(), -X.T @ resid + self.alpha * w]

        res = _scipy_minimize(loss, x0, jac=grad, bounds=bounds, method="L-BFGS-B",
                              options={"maxiter": 10_000, "ftol": 1e-14, "gtol": 1e-9})
        self.intercept_ = float(res.x[0])
        self.coef_      = res.x[1:].copy()
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_
