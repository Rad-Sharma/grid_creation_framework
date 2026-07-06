"""
predict_bus_loads.py — Apply the trained model to transmission buses in the target region.

Pipeline:
  1. Per-bus Ridge prediction (raw, unclustered).
  2. City-cluster rescaling (mirrors training aggregation; disaggregates back to buses).
  3. Bottom-up industrial addons (smelters, ports, etc. not in NZ/GB training data).
  4. Captive-power discount for buses with unregistered on-site generators.
  5. National scaling: auto-detect cross-border buses from NaturalEarth geometry;
     scale all domestic buses so Pd sums to COUNTRY_DEMAND_MW.
  6. Assign NATIONAL_EXPORT_MW equally across external/cross-border buses.
  7. Zero out load on all buses above the network's minimum voltage level
     (load is served only at the lowest voltage tier).

Output:
  data/bus_data/processed/{region}_substation_predictions_v4.csv
"""

import math
import warnings
import sys
import importlib.util
import numpy as np
import pandas as pd
import joblib
from pathlib import Path


# ---------------------------------------------------------------------------
# Import shared constants via importlib so joblib-pickled NNRidge can resolve
# ---------------------------------------------------------------------------
_const_path = Path(__file__).parent / "config.py"
_spec = importlib.util.spec_from_file_location("constants", _const_path)
_constants_mod = importlib.util.module_from_spec(_spec)
sys.modules["constants"] = _constants_mod
_spec.loader.exec_module(_constants_mod)


# Thresholds and NNRidge class are defined in config.py.


import config

region                  = config.region
NTL_CAPTIVE_THRESHOLD   = _constants_mod.NTL_CAPTIVE_THRESHOLD
CITY_CLUSTER_RADIUS_DEG = _constants_mod.CITY_CLUSTER_RADIUS_DEG
MEGA_CITY_CAP_MW        = _constants_mod.MEGA_CITY_CAP_MW
COUNTRY_DEMAND_MW       = _constants_mod.COUNTRY_DEMAND_MW
NATIONAL_EXPORT_MW      = _constants_mod.NATIONAL_EXPORT_MW
SHAPE_FILE_NAME         = config.SHAPE_FILE_NAME
NNRidge                 = _constants_mod.NNRidge

warnings.filterwarnings("ignore")

PROJ_ROOT = Path(__file__).parent.parent
FEAT_DIR  = PROJ_ROOT / "data" / "bus_data" / "features"
PROC_DIR  = PROJ_ROOT / "data" / "bus_data" / "processed"
MODEL_DIR = PROJ_ROOT / "data" / "bus_data" / "models"
WORK_DIR  = PROJ_ROOT / "data" / "working_files" / region

PROC_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Bottom-up industrial addon per-facility MW constants
# ---------------------------------------------------------------------------
# Used for facility types with no coefficient in the Ridge model because
# NZ+GB training data has no examples of these.
# ALUMINUM: Hall-Héroult electrolysis ~14.5 MWh/t. A 200 kt/yr smelter → ~200 MW.
# COPPER:   Flash smelting + electrorefining ~2.2 MWh/t. 500 kt/yr → ~100 MW.
# PORT:     Cranes + refrigeration + lighting + terminal ops. Major port ~80 MW avg.
ALUMINUM_MW_PER = 200.0
COPPER_MW_PER   = 100.0
PORT_MW_PER     =  80.0

# ---------------------------------------------------------------------------
# Captive-power discount thresholds
# ---------------------------------------------------------------------------
CAPTIVE_N_OSM_THRESHOLD  = 4
CAPTIVE_WRI_THRESHOLD_MW = 100.0
CAPTIVE_DISCOUNT         = 0.5

# ---------------------------------------------------------------------------
# Cross-border bus auto-detection
# ---------------------------------------------------------------------------

def detect_cross_border_buses(feat_df: pd.DataFrame,
                               shape_file_name: str) -> set:
    """Return a set of bus_name values whose coordinates fall outside the
    country boundary (NaturalEarth 110 m).  These buses carry international
    exchange flows and must not receive domestic load.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point

        try:
            import pyogrio
            _ne = (str(pyogrio.__file__).replace("__init__.py", "")
                   + "tests/fixtures/naturalearth_lowres/naturalearth_lowres.shp")
            world = gpd.read_file(_ne)
        except Exception:
            world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

        country = world[world["name"] == shape_file_name]
        if country.empty:
            print(f"  [cross-border] '{shape_file_name}' not found in NaturalEarth "
                  f"— cross-border detection disabled.")
            return set()

        country_geom = country.geometry.union_all()
        outside = set()
        for _, row in feat_df.iterrows():
            lat = row.get("lat")
            lon = row.get("lon")
            if pd.isna(lat) or pd.isna(lon):
                continue
            if not country_geom.contains(Point(float(lon), float(lat))):
                outside.add(row["bus_name"])

        print(f"  [cross-border] Detected {len(outside)} external bus(es) "
              f"outside '{shape_file_name}' boundary.")
        return outside

    except Exception as exc:
        print(f"  [cross-border] Detection failed ({exc}) — no buses excluded.")
        return set()


# ---------------------------------------------------------------------------
# City-level cluster prediction (mirrors train_model.py aggregation)
# ---------------------------------------------------------------------------

def cluster_predict_region(feat_df, features_r, model_r, imputer_r,
                            radius_deg=CITY_CLUSTER_RADIUS_DEG,
                            ntl_threshold=NTL_CAPTIVE_THRESHOLD,
                            generation_bus_keys=None):
    """Aggregate nearby buses into city clusters, predict cluster-level MW,
    then disaggregate back to individual buses proportionally.

    The Ridge model was trained on city-scale NZ/GB data where multiple
    distribution substations per city were aggregated. Applying it directly
    to individual transmission buses overpredicts clustered urban buses. This
    function groups buses with the same union-find algorithm used in training.
    """
    generation_bus_keys = generation_bus_keys or []

    gen_mask = feat_df["bus_name"].apply(
        lambda n: any(gk in n for gk in generation_bus_keys)
    )
    work_df = feat_df[~gen_mask].copy().reset_index(drop=True)
    gen_df  = feat_df[gen_mask].copy()

    lats = work_df["lat"].fillna(0).values
    lons = work_df["lon"].fillna(0).values
    n    = len(work_df)

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i + 1, n):
            if np.sqrt((lats[i]-lats[j])**2 + (lons[i]-lons[j])**2) < radius_deg:
                union(i, j)

    cluster_ids_arr = np.array([find(i) for i in range(n)])
    work_df["_cluster"] = cluster_ids_arr

    sum_cols = [
        "total_population", "ntl_sum", "ghsl_built_km2", "area_km2",
        "road_length_km", "ind_heavy_total_km2", "ind_heavy_chemical_km2",
        "ind_heavy_mining_km2", "gem_steel_mw_voronoi",
        "fp_gold_count", "fp_aluminum_count", "fp_copper_count", "fp_coal_count",
        "n_osm_generators_voronoi", "wri_capacity_mw_voronoi", "ne_port_soft",
    ]
    max_cols = ["ne_port_flag"]

    unique_clusters = sorted(work_df["_cluster"].unique())
    cluster_rows, cluster_order = [], []

    for cid in unique_clusters:
        grp = work_df[work_df["_cluster"] == cid]
        row = {}
        for col in sum_cols:
            row[col] = grp[col].fillna(0).sum() if col in grp.columns else 0.0
        for col in max_cols:
            row[col] = grp[col].fillna(0).max() if col in grp.columns else 0.0

        agg_area = max(grp["area_km2"].fillna(1).clip(lower=1).sum(), 1)
        row["pop_density"]  = row.get("total_population", 0) / agg_area
        row["road_density"] = row.get("road_length_km",   0) / agg_area
        row["ntl_mean"]     = row.get("ntl_sum",           0) / agg_area
        row["ghsl_pct"]     = row.get("ghsl_built_km2",    0) / agg_area * 100

        NZ_NTL_FLOOR = ntl_threshold + 0.735
        eff_ntl = max(row["ntl_mean"], NZ_NTL_FLOOR)
        ntl_guard = max(0.0, eff_ntl - ntl_threshold)
        row["mining_ntl_product"]     = row.get("ind_heavy_mining_km2",   0) * ntl_guard
        row["ind_heavy_chemical_km2"] = row.get("ind_heavy_chemical_km2", 0)
        row["fp_gold_ntl"] = row.get("fp_gold_count", 0) * ntl_guard

        cluster_rows.append(row)
        cluster_order.append(cid)

    cluster_df = pd.DataFrame(cluster_rows, index=cluster_order)

    X_cl = np.zeros((len(cluster_df), len(features_r)))
    for j, feat in enumerate(features_r):
        if feat in cluster_df.columns:
            X_cl[:, j] = cluster_df[feat].fillna(0).clip(lower=0).values

    X_cl_imp   = imputer_r.transform(X_cl)
    cl_mw      = np.expm1(model_r.predict(np.log1p(X_cl_imp))).clip(0)
    cluster_pred = dict(zip(cluster_order, cl_mw))

    sizes = work_df.groupby("_cluster").size()
    print(f"\n  [City clustering] {len(work_df)} load buses → {len(unique_clusters)} clusters  "
          f"({len(gen_df)} generation buses excluded)")
    print(f"  Sizes: 1-bus={(sizes==1).sum()}, 2-bus={(sizes==2).sum()}, "
          f"3+={(sizes>=3).sum()}")

    bus_raw  = work_df["predicted_mw_raw"].values.copy()
    rescaled = np.zeros(len(work_df))

    for cid in unique_clusters:
        mask = cluster_ids_arr == cid
        if mask.sum() == 1:
            rescaled[mask] = bus_raw[mask]
        else:
            cl_total = cluster_pred[cid]
            raw_sum  = bus_raw[mask].sum()
            if raw_sum > 0:
                rescaled[mask] = bus_raw[mask] * (cl_total / raw_sum)
            else:
                rescaled[mask] = cl_total / mask.sum()

    work_df["predicted_mw_raw"] = rescaled

    gen_df = gen_df.copy()
    max_cid = int(cluster_ids_arr.max()) if len(cluster_ids_arr) > 0 else 0
    gen_df["_cluster"] = range(max_cid + 1, max_cid + 1 + len(gen_df))

    merged = pd.concat([work_df, gen_df], ignore_index=True)
    orig_order = feat_df["bus_idx"].tolist()
    merged = merged.set_index("bus_idx").reindex(orig_order).reset_index()

    return merged["predicted_mw_raw"].values, merged["_cluster"].values


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"=== predict_bus_loads.py: {region} prediction ===\n")

    # Load model bundle
    bundle_path = MODEL_DIR / "model_bundle_v5.pkl"
    if not bundle_path.exists():
        raise FileNotFoundError(f"{bundle_path} not found. Run train_model.py first.")
    bundle = joblib.load(bundle_path)
    model_r    = bundle["model"]
    features_r = bundle["features"]
    imputer_r  = bundle["imputer"]
    print(f"Loaded model_bundle_v5.pkl")
    print(f"  Model type  : {bundle.get('model_type', 'unknown')}")
    print(f"  CV R²(MW)   : {bundle.get('cv_r2_mw', 0):.3f}")
    print(f"  CV Spearman : {bundle.get('cv_spearman', 0):.3f}")
    print(f"  Features    : {features_r}")

    # Load bus features
    feat_path = FEAT_DIR / f"{region}_features_with_industrial.csv"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"{feat_path} not found. Run build_region_features.py first.")
    feat_df = pd.read_csv(feat_path)
    print(f"\n{region} features: {len(feat_df)} buses × {feat_df.shape[1]} columns")

    # Pin lat/lon from matpower_{region}_buses.csv
    _csv_buses_path = WORK_DIR / f"matpower_{region}_buses.csv"
    if _csv_buses_path.exists():
        _csv_buses = pd.read_csv(_csv_buses_path)
        n_updated = 0
        for _, _cr in _csv_buses.iterrows():
            _bidx = int(_cr["bus_i"])
            _mask = feat_df["bus_idx"] == _bidx
            if _mask.any():
                feat_df.loc[_mask, "lat"] = float(_cr["latitude"])
                feat_df.loc[_mask, "lon"] = float(_cr["longitude"])
                n_updated += int(_mask.sum())
        print(f"  Pinned lat/lon for {n_updated}/{len(feat_df)} buses")
    else:
        print("  matpower buses CSV not found — using Voronoi centroid coordinates")

    # Feature engineering (must exactly match train_model.py)
    for col in ["ntl_sum", "ghsl_built_km2", "road_length_km",
                "ind_heavy_total_km2", "ind_heavy_chemical_km2"]:
        if col in feat_df.columns:
            feat_df[col] = feat_df[col].fillna(0)

    if "ne_port_flag" not in feat_df.columns:
        feat_df["ne_port_flag"] = 0.0

    area = feat_df["area_km2"].clip(lower=1)
    feat_df["ntl_per_km2"] = feat_df["ntl_sum"].fillna(0) / area
    feat_df["log_area_km2"] = np.log1p(feat_df.get("area_km2", 1))

    z = pd.Series(0.0, index=feat_df.index)
    ntl_g = feat_df["ntl_mean"].fillna(0).clip(lower=0) if "ntl_mean" in feat_df.columns else z

    NZ_NTL_FLOOR = NTL_CAPTIVE_THRESHOLD + 0.735
    ntl_g = ntl_g.clip(lower=NZ_NTL_FLOOR)
    ntl_guard_g = (ntl_g - NTL_CAPTIVE_THRESHOLD).clip(lower=0)

    mining_g  = feat_df["ind_heavy_mining_km2"].fillna(0).clip(lower=0)   if "ind_heavy_mining_km2"   in feat_df.columns else z
    chem_g    = feat_df["ind_heavy_chemical_km2"].fillna(0).clip(lower=0)  if "ind_heavy_chemical_km2"  in feat_df.columns else z
    ind_tot_g = feat_df["ind_heavy_total_km2"].fillna(0).clip(lower=0)     if "ind_heavy_total_km2"     in feat_df.columns else (mining_g + chem_g)
    gold_g    = feat_df["fp_gold_count"].fillna(0).clip(lower=0)           if "fp_gold_count"           in feat_df.columns else z

    feat_df["mining_ntl_product"]    = mining_g  * ntl_guard_g
    feat_df["ind_heavy_ntl_product"] = ind_tot_g * ntl_guard_g
    feat_df["fp_gold_ntl"]           = gold_g    * ntl_guard_g

    if "ghsl_built_km2" in feat_df.columns:
        feat_df["ghsl_pct"] = feat_df["ghsl_built_km2"].fillna(0) / area * 100

    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 1: Per-bus Ridge prediction (unclustered)")
    print("=" * 60)
    X_raw = np.zeros((len(feat_df), len(features_r)))
    for j, feat in enumerate(features_r):
        if feat in feat_df.columns:
            X_raw[:, j] = feat_df[feat].fillna(0).clip(lower=0).values

    X_imp    = imputer_r.transform(X_raw)
    X_log    = np.log1p(X_imp)
    log_pred = model_r.predict(X_log)
    bus_mw   = np.expm1(log_pred).clip(0)

    feat_df["predicted_mw_raw"] = bus_mw
    print(f"  Range: {bus_mw.min():.1f} – {bus_mw.max():.1f} MW  sum={bus_mw.sum():.0f} MW")

    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 2: City-cluster rescaling")
    print("=" * 60)
    cluster_rescaled, cluster_ids = cluster_predict_region(
        feat_df,
        features_r=features_r,
        model_r=model_r,
        imputer_r=imputer_r,
        radius_deg=CITY_CLUSTER_RADIUS_DEG,
        ntl_threshold=NTL_CAPTIVE_THRESHOLD,
        generation_bus_keys=[],
    )
    feat_df["_cluster"]         = cluster_ids
    feat_df["predicted_mw_raw"] = cluster_rescaled

    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 3: Bottom-up industrial addons")
    print("=" * 60)
    ntl_binary = (ntl_g >= NTL_CAPTIVE_THRESHOLD).astype(float)

    icmm_al  = feat_df.get("icmm_al_mw_est",   z).fillna(0).clip(lower=0)
    fp_al_mw = feat_df.get("fp_aluminum_count", z).fillna(0).clip(lower=0) * ALUMINUM_MW_PER * ntl_binary
    al_addon = pd.concat([icmm_al, fp_al_mw], axis=1).max(axis=1)

    icmm_cu  = feat_df.get("icmm_cu_mw_est",  z).fillna(0).clip(lower=0)
    fp_cu_mw = feat_df.get("fp_copper_count",  z).fillna(0).clip(lower=0) * COPPER_MW_PER * ntl_binary
    cu_addon = pd.concat([icmm_cu, fp_cu_mw], axis=1).max(axis=1)

    zn_addon     = feat_df.get("icmm_zn_mw_est",    z).fillna(0).clip(lower=0)
    ni_addon     = feat_df.get("icmm_ni_mw_est",    z).fillna(0).clip(lower=0)
    cement_addon = feat_df.get("gem_cement_mw_est", z).fillna(0).clip(lower=0)
    chem_addon   = feat_df.get("gem_chem_mw_est",   z).fillna(0).clip(lower=0)

    port_count = feat_df.get("ne_port_count_15km", z).fillna(0).clip(lower=0)
    port_addon = port_count * PORT_MW_PER * ntl_binary

    industrial_addon = (al_addon + cu_addon + zn_addon + ni_addon +
                        cement_addon + chem_addon + port_addon)
    feat_df["industrial_addon_mw"] = industrial_addon.values
    feat_df["predicted_mw_raw"]    = (feat_df["predicted_mw_raw"] + industrial_addon).values

    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 4: Captive-power discount")
    print("=" * 60)
    n_osm_col   = feat_df.get("n_osm_generators_voronoi", z).fillna(0)
    wri_cap_col = feat_df.get("wri_capacity_mw_voronoi",  z).fillna(0)
    captive_flag = (
        (n_osm_col >= CAPTIVE_N_OSM_THRESHOLD) &
        (wri_cap_col < CAPTIVE_WRI_THRESHOLD_MW)
    ).astype(float)
    n_captive = int(captive_flag.sum())
    if n_captive > 0:
        discount = 1.0 - captive_flag * CAPTIVE_DISCOUNT
        feat_df["predicted_mw_raw"] *= discount
        captive_buses = feat_df.loc[captive_flag > 0, "bus_name"].tolist()
        print(f"  Applied {CAPTIVE_DISCOUNT*100:.0f}% captive discount to "
              f"{n_captive} bus(es): {captive_buses}")
    else:
        print("  No captive-power buses detected")

    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 5: National scaling")
    print("=" * 60)
    # Detect cross-border buses automatically from their geographic coordinates.
    # External buses are excluded from national demand scaling and receive 0 load
    # (or NATIONAL_EXPORT_MW if scheduled exports are defined).
    cross_border_names = detect_cross_border_buses(feat_df, SHAPE_FILE_NAME)
    cb_mask = feat_df["bus_name"].isin(cross_border_names)

    domestic_sum = feat_df.loc[~cb_mask, "predicted_mw_raw"].sum()
    scale_f      = COUNTRY_DEMAND_MW / domestic_sum if domestic_sum > 0 else 1.0

    feat_df["predicted_mw"]          = feat_df["predicted_mw_raw"] * scale_f
    feat_df.loc[cb_mask, "predicted_mw"] = 0.0

    print(f"  Cross-border buses excluded : {cb_mask.sum()}")
    print(f"  Raw domestic sum            : {domestic_sum:.0f} MW")
    print(f"  Scale factor                : {scale_f:.4f}")
    print(f"  Domestic total after scaling: "
          f"{feat_df.loc[~cb_mask, 'predicted_mw'].sum():.0f} MW")

    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 6: Assign national export load to cross-border buses")
    print("=" * 60)
    # NATIONAL_EXPORT_MW (from config) is split equally among all cross-border buses.
    if NATIONAL_EXPORT_MW > 0 and cb_mask.sum() > 0:
        export_per_bus = NATIONAL_EXPORT_MW / cb_mask.sum()
        feat_df.loc[cb_mask, "predicted_mw"] = export_per_bus
        print(f"  {NATIONAL_EXPORT_MW:.0f} MW split among "
              f"{cb_mask.sum()} cross-border bus(es): {export_per_bus:.1f} MW each")
    else:
        print(f"  NATIONAL_EXPORT_MW = {NATIONAL_EXPORT_MW:.0f} MW  "
              f"(cross-border buses carry no export load)")

    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 7: Restrict load to lowest-voltage buses")
    print("=" * 60)
    # Load should only be assigned to the lowest voltage level in the network.
    # High-voltage transmission buses do not serve customer load directly.
    _bus_kv_map = (_csv_buses.set_index("bus_i")["base_kV"].to_dict()
                   if _csv_buses_path.exists()
                   else {})
    feat_df["_base_kV"] = feat_df["bus_idx"].map(_bus_kv_map)
    min_kv = feat_df["_base_kV"].dropna().min()
    if not pd.isna(min_kv):
        high_v_mask = feat_df["_base_kV"].fillna(float("inf")) > min_kv
        n_zeroed = int(high_v_mask.sum())
        feat_df.loc[high_v_mask, "predicted_mw"] = 0.0
        print(f"  Minimum voltage in network : {min_kv:.1f} kV")
        print(f"  Buses above minimum zeroed : {n_zeroed}  "
              f"({int((~high_v_mask).sum())} load buses remain)")
    else:
        print("  Bus voltage data unavailable — skipping voltage filter")

    # Build output matching matpower_{region}_buses.csv format with Pd/Qd filled in
    _PF      = 0.95
    _tan_phi = math.tan(math.acos(_PF))

    _csv_src = WORK_DIR / f"matpower_{region}_buses.csv"
    if not _csv_src.exists():
        raise FileNotFoundError(f"{_csv_src} not found.")

    out_df = pd.read_csv(_csv_src)
    pred_map = feat_df.set_index("bus_idx")["predicted_mw"].to_dict()
    out_df["Pd"] = out_df["bus_i"].map(pred_map).fillna(0.0).round(4)
    out_df["Qd"] = (out_df["Pd"] * _tan_phi).round(4)

    out_path = PROC_DIR / f"{region}_substation_predictions_v4.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}  "
          f"({len(out_df)} buses, Pd range: "
          f"{out_df['Pd'].min():.1f} – {out_df['Pd'].max():.1f} MW)")


if __name__ == "__main__":
    main()
