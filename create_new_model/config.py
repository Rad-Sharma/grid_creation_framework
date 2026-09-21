"""
config.py — Shared constants and model classes for the synthetic grid pipeline.

Imported by all pipeline scripts. Provides:
  - Region-specific settings (name, bounding box, country code)
  - NNRidge model class (must live here so joblib pickle can resolve it)
  - Pipeline thresholds and hyperparameters

To adapt this pipeline for a new region, update the region settings block below.
"""

import sys as _sys

import numpy as np
from scipy.optimize import minimize as _scipy_minimize

# Windows consoles default to cp1252, which cannot encode the arrows and
# box-drawing characters the pipeline prints; resolve_unmatched.py died with
# UnicodeEncodeError before printing a single prompt. Every pipeline script
# imports config, so widen the console encoding once here.
for _stream in (_sys.stdout, _sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

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

# Total national installed generating capacity (MW). OSM tags a capacity for
# some plants and not others; Sec. V-B-2 sends the rest to a manual lookup, and
# until someone does that a country with thin OSM coverage ends up with less
# generation than demand. Setting this distributes the shortfall over the
# untagged plants, weighted by the median capacity of the tagged plants burning
# the same fuel, exactly as Sec. III-C-6 scales predicted demand to a national
# total. The result is an ASSUMPTION: create_generators.py lists every plant it
# sizes and flags them in the output CSV. 0 disables it, leaving those plants
# out of service.
COUNTRY_CAPACITY_MW = 0

# --------------- Minimum transmission voltage ----------------
# Lines and substations with a KNOWN voltage below this threshold are removed.
# Eliminates sub-transmission and distribution assets that OSM tags as
# power=line / power=substation but are not part of the HV backbone.
#
# 110 kV is a common lower bound internationally. Change to 69000 for regions
# where 69 kV is considered transmission, or 132000 for UK-style networks.
MIN_TRANSMISSION_VOLTAGE_V = 115000   # Volts (110000 for 110 kV, 69000 for 69 kV, 132000 for UK-style networks)

# Fuel-price basis for the Table V generator cost curves.
#   "us"     - reproduces the published Table V coefficients (paper default)
#   "global" - rescales each curve by the Table IV global/US price ratio for its
#              fuel. The paper never states how to apply the global column,
#              but reference [48] builds $/h as heat_rate x P x fuel_price and
#              fits (a, b, c) by least squares, so all three scale exactly with
#              fuel price. Hydro is excluded (its curve was copied from CCGT
#              samples and it burns no fuel) and internal combustion is flagged
#              approximate (built from ERCOT bids, not heat rates); see
#              SCALING_BASIS in create_cost_curves.py. Results produced with
#              "global" still depart from the published Table V numbers.
FUEL_COST_BASIS = "us"

# Automate Grid Creation load path. "legacy_ridge" explicitly selects the previous
# approximation. Paper mode requires measured industrial VIIRS data and a Huber
# artifact; it never silently substitutes Ridge, proxy light, or missing features.
LOAD_MODEL_METHOD = "paper_huber"
PAPER_HUBER_MODEL = "model_huber_paper.json"
VIIRS_RASTER_NAME = f"viirs_{region}.tif"
STEEL_FACILITIES_NAME = "steel_facilities_mw.geojson"
GOLD_FACILITIES_NAME = "gold_facilities.geojson"
PLANT_NAME_SEARCH_RADIUS_M = 1000

# Two OSM substation records this close together are treated as one physical
# site when their names agree or one of them is unnamed (Sec. V-B-1's duplicate
# rule, applied to substations). Raise it only if the region's substations are
# mapped several times over; lowering it to 0 disables the merge.
SUBSTATION_MERGE_RADIUS_M = 500

# A national transmission grid is normally one synchronous system. When this is
# True the review offers to close any gap left between separate networks, and
# check_topology.py reports fragmentation as a failure. Set it False only for a
# region that genuinely runs as more than one network.
EXPECT_CONNECTED_GRID = True

# Branch angle-difference limit written into every line (MATPOWER angmin/angmax).
# Eq. (4) does not constrain angle differences and Eq. (5j) does not state a
# value, so this is a modelling default, not a measured one. A case that needs
# more than this is telling you a corridor is near its stability limit.
BRANCH_ANGLE_LIMIT_DEG = 60


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
# Automate Grid Creation Sec. III.B: a substation with a compound voltage tag (e.g.
# "330000;161000") is split into one bus per voltage level. Those buses are
# co-located in one physical switchyard, so the pipeline connects consecutive
# tiers with a transformer branch (1.6 % impedance on 1000 MVA, X/R = 80).
# Leaving this False splits the buses but omits the transformer, which strands
# every upper tier as its own island and does not reproduce the paper's model.
# Set False only for a labeled sensitivity experiment against reviewed equipment.
INFER_TRANSFORMERS_FROM_VOLTAGE = True
