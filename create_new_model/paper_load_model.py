"""Manuscript equations (1)-(2): six log features and Huber epsilon=1.1.

The manuscript does not supply the intercept or a fully specified training
preprocessing recipe. A provenance-bearing model artifact is required; the
published exponents are retained for comparison, not combined with an invented
intercept or the incompatible Ridge intercept.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["pop_density", "road_density", "gem_steel_mw_voronoi",
            "mining_ntl_product", "ind_heavy_chemical_km2", "fp_gold_ntl"]
PUBLISHED_COEFFICIENTS = [.510, .342, .075, .157, .372, .253]
FEATURE_CONTRACT = "paper_industrial_viirs_v1"


def feature_matrix(frame):
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing paper load predictors: {sorted(missing)}")
    if "feature_contract" not in frame or not frame.feature_contract.eq(FEATURE_CONTRACT).all():
        raise ValueError("Paper prediction requires measured industrial-polygon VIIRS features; rebuild features")
    values = frame[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Paper predictors must be finite and nonnegative; missing data is not zero")
    return np.log1p(values)


def load_model(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Supply the fitted Huber artifact or use train_paper_load_model.py "
            "with complete measured training features. The manuscript omits the intercept; "
            "the bundled Ridge model cannot reproduce it.")
    model = json.loads(path.read_text(encoding="utf-8"))
    if model.get("model_type") != "huber" or model.get("epsilon") != 1.1:
        raise ValueError("Expected the paper's Huber model with epsilon=1.1")
    if model.get("features") != FEATURES or model.get("feature_contract") != FEATURE_CONTRACT:
        raise ValueError("Huber artifact has incompatible feature ordering or preprocessing")
    parameters = [model.get("intercept")] + model.get("coefficients", [])
    if len(parameters) != 7 or not np.isfinite(np.asarray(parameters, float)).all():
        raise ValueError("Huber artifact requires a finite intercept and six coefficients")
    if not model.get("provenance"):
        raise ValueError("Huber artifact must describe training data and provenance")
    return model


def predict(frame, model):
    log_prediction = model["intercept"] + feature_matrix(frame) @ np.asarray(model["coefficients"])
    with np.errstate(over="raise", invalid="raise"):
        return np.maximum(np.expm1(log_prediction), 0.)


def scale_demands(weights, external, domestic_mw, export_mw):
    """Section III.C.6: scale all represented domestic substations, then exports."""
    weights = np.asarray(weights, float)
    external = np.asarray(external, bool)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Invalid predicted demand")
    if not np.isfinite([domestic_mw, export_mw]).all() or min(domestic_mw, export_mw) < 0:
        raise ValueError("Demand and export must be nonnegative MW")
    total = weights[~external].sum()
    if domestic_mw > 0 and total <= 0:
        raise ValueError("No positive domestic predictions to scale")
    out = np.zeros_like(weights)
    if total > 0:
        out[~external] = weights[~external] * domestic_mw / total
    if export_mw > 0:
        if not external.any():
            raise ValueError("Exports requested but no external substations detected")
        out[external] = export_mw / external.sum()
    return out


def run_prediction(features_path, bus_path, model_path, output_path,
                   domestic_mw, export_mw, detect_external, shape_name):
    model = load_model(model_path)
    features = pd.read_csv(features_path)
    buses = pd.read_csv(bus_path)
    if features.bus_idx.duplicated().any() or buses.bus_i.duplicated().any():
        raise ValueError("Duplicate bus IDs")
    if not features.bus_idx.isin(buses.bus_i).all():
        raise ValueError("Features reference stale bus IDs; rebuild features")
    # Paper features contain one receiving voltage tier per physical substation.
    expected = buses.sort_values(["base_kV", "bus_i"]).drop_duplicates(["latitude", "longitude"])
    if set(expected.bus_i) != set(features.bus_idx):
        raise ValueError("Feature coverage does not match current substations; rebuild features")
    lookup = buses.set_index("bus_i")
    for target, source in [("lat", "latitude"), ("lon", "longitude")]:
        if not np.allclose(features[target], features.bus_idx.map(lookup[source]), atol=1e-7):
            raise ValueError("Feature coordinates do not match current buses")
    raw = predict(features, model)
    external_names = detect_external(features, shape_name)
    external = features.bus_name.isin(external_names)
    mw = scale_demands(raw, external, domestic_mw, export_mw)
    prediction = pd.Series(mw, index=features.bus_idx)
    buses["Pd"] = buses.bus_i.map(prediction).fillna(0.)
    buses["Qd"] = buses.Pd * np.tan(np.arccos(.95))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    buses.to_csv(output_path, index=False)
    print(f"Paper Huber prediction: domestic={mw[~external].sum():.6f} MW, export={mw[external].sum():.6f} MW")
    return buses
