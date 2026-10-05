"""
build_region_features.py — Extract geospatial features for transmission buses.

Builds a Voronoi tessellation over the transmission bus coordinates from
matpower_{region}_buses.csv, then extracts the same feature set used for
NZ/GB model training: WorldPop population, OSM industrial polygons, road
density, amenity counts, and WRI power plant data.

Paper mode requires a measured VIIRS raster and normalized steel facilities;
it measures industrial footprints and writes {region}_paper_features.csv.
See README.md for required inputs and interpretation choices.

In legacy_ridge mode, NTL (VIIRS Nighttime Light) and GHSL (Built-up Area) are fetched
automatically from Google Earth Engine if the earthengine-api package is
installed and authenticated. Otherwise they are proxied from NZ GXP data.

Pipeline position: runs after resolve_unmatched.py.

Inputs (required):
  data/working_files/{region}/matpower_{region}_buses.csv
  data/bus_data/raw/worldpop_{region}.tif   — WorldPop 2020 raster (auto-downloaded)

Inputs (auto-downloaded if earthengine-api is available):
  data/bus_data/raw/gee_features_{region}_matpower.csv

Inputs (optional fallback for NTL proxy):
  data/bus_data/features/nz_transmission_features.csv

Legacy output:
  data/bus_data/features/{region}_features_with_industrial.csv

GEE setup (one-time, if not already done):
  pip install earthengine-api
  earthengine authenticate
"""

import json
import time
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
import requests
import urllib.request
from pathlib import Path
from scipy.spatial import Voronoi
from shapely.geometry import Point, Polygon, mapping, LineString
from shapely.ops import unary_union
from sklearn.linear_model import Ridge

import config

warnings.filterwarnings("ignore")

region       = config.region
REGION_BBOX  = config.REGION_BBOX
COUNTRY_ISO3 = config.COUNTRY_ISO3

PROJ_ROOT    = Path(__file__).parent.parent
RAW_DIR      = PROJ_ROOT / "data" / "bus_data" / "raw"
FEAT_DIR     = PROJ_ROOT / "data" / "bus_data" / "features"
WORK_DIR     = PROJ_ROOT / "data" / "working_files" / region
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
HEADERS      = {"User-Agent": "substation-load-model/1.0 research"}

FEAT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Region-specific known industrial sites (optional override table)
# ---------------------------------------------------------------------------
# Populate this list for your target region if OSM industrial polygon coverage
# is incomplete. Each entry: (lat, lon, bucket_type, label).
# The bucket_type must be one of: heavy_smelter, heavy_chemical, heavy_mining,
# heavy_port, heavy_other, light_industrial.
#
# Leave as an empty list to rely entirely on OSM and WRI data.
KNOWN_REGIONAL_INDUSTRIAL: list[tuple] = []

# ---------------------------------------------------------------------------
# Industrial classification keywords
# ---------------------------------------------------------------------------
PROCESS_KEYWORDS = {
    "heavy_smelter": [
        "smelter", "alumin", "aluminum", "aluminium", "steel mill",
        "iron works", "foundry", "forge", "metallurg", "blast furnace", "electric arc",
    ],
    "heavy_chemical": [
        "refinery", "petroleum refin", "oil refin", "chemical plant",
        "petrochemical", "lpg terminal", "lng terminal", "oil depot",
        "fertiliser plant", "fertilizer plant",
    ],
    "heavy_mining": [
        "gold mine", "coal mine", "open pit", "open cast",
        "iron ore mine", "bauxite mine", "manganese mine",
        "quarry works", "mining complex", "mineral processing",
    ],
    "heavy_port": [
        "port terminal", "container terminal", "bulk cargo",
        "harbour terminal", "loading terminal",
        "grain terminal", "coal terminal",
    ],
    "heavy_other": [
        "cement plant", "cement works", "clinker", "lime kiln",
        "power station", "thermal plant", "generating station",
    ],
}


def assign_industrial_bucket(tags):
    landuse    = tags.get("landuse", "")
    man_made   = tags.get("man_made", "")
    industrial = tags.get("industrial", "")
    product    = tags.get("product", "")
    resource   = tags.get("resource", "")
    if any(k in industrial for k in ["aluminium","aluminum","steel","metal","smelter"]):
        return "heavy_smelter"
    if "smelter" in man_made:
        return "heavy_smelter"
    if any(k in product for k in ["aluminium","aluminum","steel","iron"]):
        return "heavy_smelter"
    if any(k in resource for k in ["gold","bauxite","manganese","coal","iron_ore"]):
        return "heavy_mining"
    if any(k in product for k in ["gold","bauxite","manganese","coal"]):
        return "heavy_mining"
    if any(k in industrial for k in ["chemical","oil","petroleum","refinery","gas"]):
        return "heavy_chemical"
    if any(k in man_made for k in ["refinery","petroleum_well"]):
        return "heavy_chemical"
    if any(k in landuse for k in ["quarry","mine"]):
        return "heavy_mining"
    if man_made == "mineshaft":
        return "heavy_mining"
    if any(k in industrial for k in ["mining","quarry"]):
        return "heavy_mining"
    if landuse == "port" or tags.get("harbour") == "yes":
        return "heavy_port"
    if any(k in man_made for k in ["works","kiln","gasometer","chimney"]):
        return "heavy_other"
    if any(k in industrial for k in ["cement","power"]):
        return "heavy_other"
    return "light_industrial"


def name_based_bucket(tags):
    text = " ".join([
        tags.get("name",""), tags.get("operator",""),
        tags.get("description",""), tags.get("short_name",""),
    ]).lower()
    for bucket, keywords in PROCESS_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return bucket
    return "light_industrial"


def assign_bucket_with_fallback(tags):
    b = assign_industrial_bucket(tags)
    return name_based_bucket(tags) if b == "light_industrial" else b


# ---------------------------------------------------------------------------
# Voronoi tessellation helpers
# ---------------------------------------------------------------------------

def voronoi_finite_polygons(pts, bbox, clip_geom=None):
    """Build finite Voronoi polygons clipped to clip_geom (or bbox rectangle)."""
    min_x, min_y, max_x, max_y = bbox
    box_poly = Polygon([(min_x,min_y),(max_x,min_y),(max_x,max_y),(min_x,max_y)])
    final_clip = clip_geom if (clip_geom is not None and not clip_geom.is_empty) else box_poly
    pts = np.asarray(pts)
    n   = len(pts)
    if n < 3:
        return [final_clip] * n
    extras = np.vstack([
        np.column_stack([2*min_x - pts[:,0], pts[:,1]]),
        np.column_stack([2*max_x - pts[:,0], pts[:,1]]),
        np.column_stack([pts[:,0], 2*min_y - pts[:,1]]),
        np.column_stack([pts[:,0], 2*max_y - pts[:,1]]),
    ])
    vor = Voronoi(np.vstack([pts, extras]))
    result = []
    for i in range(n):
        reg = vor.regions[vor.point_region[i]]
        if not reg or -1 in reg:
            result.append(final_clip)
            continue
        verts = vor.vertices[reg]
        poly  = Polygon(verts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        poly = poly.intersection(final_clip)
        if poly.is_empty or not poly.is_valid:
            poly = Point(pts[i]).buffer(0.001)
        result.append(poly)
    return result


def get_land_polygon(iso_code, bbox):
    """Return country land polygon clipped to bbox (falls back to bbox rectangle)."""
    cache = RAW_DIR / f"ne_land_{iso_code}.geojson"
    min_lon, min_lat, max_lon, max_lat = bbox
    box_poly = Polygon([(min_lon,min_lat),(max_lon,min_lat),
                        (max_lon,max_lat),(min_lon,max_lat)])
    if not cache.exists():
        try:
            world = gpd.read_file(
                "https://naturalearth.s3.amazonaws.com/110m_cultural/"
                "ne_110m_admin_0_countries.zip"
            )
            country = world[world["ISO_A3"] == iso_code]
            if len(country) == 0:
                return box_poly
            try:
                geom = country.geometry.union_all()
            except AttributeError:
                geom = country.geometry.unary_union
            import json as _json
            with open(cache, "w") as f:
                _json.dump({"type": "Feature", "geometry": geom.__geo_interface__}, f)
        except Exception as exc:
            print(f"  [land polygon] {exc} — using bbox")
            return box_poly
    try:
        import json as _json
        from shapely.geometry import shape as _shape
        with open(cache) as f:
            data = _json.load(f)
        geom = _shape(data["geometry"])
        land = geom.intersection(box_poly)
        return land if not land.is_empty else box_poly
    except Exception as exc:
        print(f"  [land polygon] Load failed: {exc} — using bbox")
        return box_poly


def build_voronoi_gdf(df, lat_col, lon_col, id_col, bbox, clip_geom=None):
    pts   = list(zip(df[lon_col], df[lat_col]))
    polys = voronoi_finite_polygons(pts, bbox, clip_geom=clip_geom)
    records = []
    for idx, (_, row) in enumerate(df.iterrows()):
        poly     = polys[idx]
        lat_rad  = np.radians(row[lat_col])
        area_km2 = poly.area * (111.32**2) * abs(np.cos(lat_rad))
        records.append({
            id_col:     row[id_col],
            "lat":      row[lat_col],
            "lon":      row[lon_col],
            "area_km2": area_km2,
            "geometry": poly,
        })
    return gpd.GeoDataFrame(records, crs="EPSG:4326")


# ---------------------------------------------------------------------------
# Industrial feature extraction
# ---------------------------------------------------------------------------

def area_km2_approx(geom):
    lat = geom.centroid.y
    return geom.area * 111.32 * (111.32 * abs(np.cos(np.radians(lat))))


def parse_industrial_elements(elements):
    way_rows, point_rows = [], []
    for el in elements:
        tags   = el.get("tags", {})
        bucket = assign_bucket_with_fallback(tags)
        etype  = el.get("type")
        if etype == "way":
            geom = el.get("geometry", [])
            if len(geom) < 3:
                continue
            try:
                coords = [(pt["lon"], pt["lat"]) for pt in geom]
                poly   = Polygon(coords)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if not poly.is_empty:
                    way_rows.append({"geometry": poly, "bucket": bucket,
                                     "area_km2": area_km2_approx(poly), "tags": tags})
            except Exception:
                continue
        elif etype == "node":
            lat = el.get("lat")
            lon = el.get("lon")
            if lat is not None:
                point_rows.append({"lat": lat, "lon": lon, "bucket": bucket, "tags": tags})
    return way_rows, point_rows


def apply_wri_override(way_rows, wri_gdf, radius_km=1.5):
    """Override industrial polygon bucket for any polygon within radius_km of a WRI plant."""
    if wri_gdf is None or not way_rows:
        return way_rows
    centroids = gpd.GeoDataFrame(
        [{"row_i": i, "geometry": r["geometry"].centroid} for i, r in enumerate(way_rows)],
        crs="EPSG:4326"
    )
    buf_deg = radius_km / 111.32
    wri_buf = wri_gdf.copy()
    wri_buf["geometry"] = wri_gdf.geometry.buffer(buf_deg)
    joined = gpd.sjoin(centroids, wri_buf[["wri_type","geometry"]],
                       how="left", predicate="within")
    way_rows = [dict(r) for r in way_rows]
    n = 0
    for _, jrow in joined.iterrows():
        wt = jrow.get("wri_type")
        if pd.notna(wt):
            ri = int(jrow["row_i"])
            if way_rows[ri]["bucket"] != wt:
                way_rows[ri]["bucket"] = wt
                n += 1
    if n:
        print(f"  [WRI] Overrode {n} industrial polygon type(s)")
    return way_rows


def apply_known_industrial_override(way_rows, known_table, radius_km=3.0):
    """Override industrial polygon bucket to match a user-supplied facility table."""
    if not known_table or not way_rows:
        return way_rows
    known_df  = pd.DataFrame(known_table, columns=["lat","lon","known_type","label"])
    known_gdf = gpd.GeoDataFrame(
        known_df,
        geometry=gpd.points_from_xy(known_df["lon"], known_df["lat"]),
        crs="EPSG:4326"
    )
    centroids = gpd.GeoDataFrame(
        [{"row_i": i, "geometry": r["geometry"].centroid} for i, r in enumerate(way_rows)],
        crs="EPSG:4326"
    )
    buf_deg   = radius_km / 111.32
    known_buf = known_gdf.copy()
    known_buf["geometry"] = known_gdf.geometry.buffer(buf_deg)
    joined    = gpd.sjoin(centroids, known_buf[["known_type","label","geometry"]],
                          how="left", predicate="within")
    way_rows  = [dict(r) for r in way_rows]
    for _, jrow in joined.iterrows():
        kt = jrow.get("known_type")
        if pd.notna(kt):
            ri  = int(jrow["row_i"])
            old = way_rows[ri]["bucket"]
            if old != kt:
                way_rows[ri]["bucket"] = kt
                print(f"  [KNOWN OVERRIDE] {jrow.get('label','?')}: {old} -> {kt}")
    return way_rows


def compute_typed_industrial(voronoi_gdf, way_rows, point_rows, id_col):
    """Aggregate typed industrial polygon areas and point counts per Voronoi cell."""
    poc_ids = voronoi_gdf[id_col].values
    cols = {
        "ind_heavy_smelter_km2":  np.zeros(len(poc_ids)),
        "ind_heavy_chemical_km2": np.zeros(len(poc_ids)),
        "ind_heavy_mining_km2":   np.zeros(len(poc_ids)),
        "ind_heavy_port_km2":     np.zeros(len(poc_ids)),
        "ind_heavy_other_km2":    np.zeros(len(poc_ids)),
        "ind_light_km2":          np.zeros(len(poc_ids)),
        "ind_point_mines":        np.zeros(len(poc_ids), dtype=int),
        "ind_point_petroleum":    np.zeros(len(poc_ids), dtype=int),
        "ind_point_ports":        np.zeros(len(poc_ids), dtype=int),
    }

    if way_rows:
        poc_idx_map = {poc: i for i, poc in enumerate(poc_ids)}
        way_gdf = gpd.GeoDataFrame(
            [{"geometry": r["geometry"].centroid, "row_i": i}
             for i, r in enumerate(way_rows)],
            crs="EPSG:4326"
        )
        vor_lookup = voronoi_gdf[[id_col, "geometry"]].copy()
        joined = gpd.sjoin(way_gdf, vor_lookup, how="left", predicate="within")
        for _, jrow in joined.iterrows():
            poc = jrow.get(id_col)
            if poc not in poc_idx_map:
                continue
            ri  = int(jrow["row_i"])
            wr  = way_rows[ri]
            bkt = wr["bucket"]
            km2 = wr["area_km2"]
            idx = poc_idx_map[poc]
            col = f"ind_{bkt}_km2" if bkt != "light_industrial" else "ind_light_km2"
            if col in cols:
                cols[col][idx] += km2

    if point_rows:
        poc_idx_map = {poc: i for i, poc in enumerate(poc_ids)}
        pt_gdf = gpd.GeoDataFrame(
            [{"geometry": Point(r["lon"], r["lat"]), "bucket": r["bucket"], "row_i": i}
             for i, r in enumerate(point_rows)],
            crs="EPSG:4326"
        )
        jpt = gpd.sjoin(pt_gdf, voronoi_gdf[[id_col,"geometry"]],
                        how="left", predicate="within")
        for _, pr in jpt.iterrows():
            poc = pr.get(id_col)
            if poc not in poc_idx_map:
                continue
            idx = poc_idx_map[poc]
            bkt = pr["bucket"]
            if bkt == "heavy_mining":     cols["ind_point_mines"][idx] += 1
            elif bkt == "heavy_chemical": cols["ind_point_petroleum"][idx] += 1
            elif bkt == "heavy_port":     cols["ind_point_ports"][idx] += 1

    df_out = pd.DataFrame(cols)
    df_out[id_col] = poc_ids
    df_out["ind_heavy_total_km2"] = (
        df_out["ind_heavy_smelter_km2"]  +
        df_out["ind_heavy_chemical_km2"] +
        df_out["ind_heavy_mining_km2"]   +
        df_out["ind_heavy_port_km2"]     +
        df_out["ind_heavy_other_km2"]
    )
    df_out["ind_total_km2"] = df_out["ind_heavy_total_km2"] + df_out["ind_light_km2"]
    df_out["ntl_mean_inside_industrial"] = 0.0
    df_out["bare_soil_km2_inside_mining"] = 0.0
    return df_out


# ---------------------------------------------------------------------------
# Population zonal sum (rasterio-native)
# ---------------------------------------------------------------------------

def rasterio_zonal_sum(voronoi_gdf, raster_path, nodata=-99999):
    """Sum raster pixels within each Voronoi polygon."""
    results = []
    with rasterio.open(raster_path) as src:
        for _, row in voronoi_gdf.iterrows():
            try:
                out_image, _ = rio_mask(src, [mapping(row.geometry)], crop=True, nodata=nodata)
                arr = out_image[0]
                valid = arr[(arr != nodata) & ~np.isnan(arr) & (arr >= 0)]
                results.append(float(valid.sum()))
            except Exception:
                results.append(0.0)
    return results


# ---------------------------------------------------------------------------
# NTL/GHSL proxy from NZ transmission data
# ---------------------------------------------------------------------------

def proxy_ntl_ghsl_from_nz(bus_df):
    """Estimate ntl_sum and ghsl_built_km2 via Ridge regression trained on NZ GXP data.

    Used when no GEE cache is available. NZ GXPs have real VIIRS+GHSL values;
    the proxy is trained on features available for all countries (population,
    area, road density). The GEE cache overrides this when present.
    """
    nz_path = FEAT_DIR / "nz_transmission_features.csv"
    if not nz_path.exists():
        print("  [Proxy] nz_transmission_features.csv not found — NTL/GHSL remain 0")
        return bus_df

    nz = pd.read_csv(nz_path).dropna(subset=["ntl_sum","ghsl_built_km2"])
    nz = nz[(nz["ntl_sum"] > 0) & (nz["ghsl_built_km2"] > 0)]
    if len(nz) < 10:
        print(f"  [Proxy] NZ data too sparse ({len(nz)} rows)")
        return bus_df

    def make_X(df):
        pop  = df["total_population"].fillna(0).values
        area = df["area_km2"].fillna(1).values.clip(min=1)
        dens = pop / area
        road = (df["road_length_km"].fillna(0).values
                if "road_length_km" in df.columns else np.zeros(len(df)))
        comm = (df["num_commercial_nodes"].fillna(0).values
                if "num_commercial_nodes" in df.columns else np.zeros(len(df)))
        return np.column_stack([
            np.log1p(pop), np.log1p(area), np.log1p(dens),
            np.log1p(road), np.log1p(comm),
        ])

    X_nz   = make_X(nz)
    y_ntl  = np.log1p(nz["ntl_sum"].clip(lower=0.01).values)
    y_ghsl = np.log1p(nz["ghsl_built_km2"].clip(lower=0.01).values)
    m_ntl  = Ridge(alpha=1.0).fit(X_nz, y_ntl)
    m_ghsl = Ridge(alpha=1.0).fit(X_nz, y_ghsl)

    X_reg      = make_X(bus_df)
    ntl_proxy  = np.expm1(m_ntl.predict(X_reg)).clip(min=1.0)
    ghsl_proxy = np.expm1(m_ghsl.predict(X_reg)).clip(min=0.0)
    area_km2   = bus_df["area_km2"].fillna(1.0).values.clip(min=1.0)

    bus_df["ntl_sum"]        = ntl_proxy
    bus_df["ntl_mean"]       = ntl_proxy / area_km2 * 0.217
    bus_df["ntl_max"]        = ntl_proxy * 0.010
    bus_df["ntl_std"]        = bus_df["ntl_mean"] * 0.5
    bus_df["ghsl_built_km2"] = ghsl_proxy
    print(f"  [Proxy] ntl_sum: {ntl_proxy.min():.0f}-{ntl_proxy.max():.0f}")
    print(f"  [Proxy] ghsl_km2: {ghsl_proxy.min():.1f}-{ghsl_proxy.max():.1f}")
    return bus_df


# ---------------------------------------------------------------------------
# GEE satellite feature download
# ---------------------------------------------------------------------------

def download_gee_features(voronoi_gdf, bus_df, out_csv: Path) -> bool:
    """Download VIIRS NTL and GHSL built-up features per Voronoi cell via GEE.

    Uses:
      NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG  — 2020 annual mean NTL (500 m, 1 band avg_rad)
      JRC/GHSL/P2023A/GHS_BUILT_S/2020    — built-up surface in m² per pixel

    Returns True if the CSV was written successfully, False otherwise.

    GEE one-time setup:
      pip install earthengine-api
      earthengine authenticate
    """
    try:
        import ee  # type: ignore
    except ImportError:
        print("  [GEE] earthengine-api not installed.")
        print("        Run:  pip install earthengine-api")
        return False

    # Initialise (uses saved credentials; prompts for auth if first time)
    try:
        ee.Initialize()
    except Exception as init_err:
        print(f"  [GEE] Initialization failed: {init_err}")
        print("  [GEE] Run once in your terminal:  earthengine authenticate")
        print("        Then re-run this script.")
        return False

    print(f"  [GEE] Initialised. Uploading {len(voronoi_gdf)} Voronoi polygons …")

    # Build EE Features from Voronoi polygons
    ee_features = []
    for _, row in voronoi_gdf.iterrows():
        try:
            coords = [list(c) for c in row.geometry.exterior.coords]
            ee_features.append(
                ee.Feature(
                    ee.Geometry.Polygon([coords]),
                    {"poc_code": row["poc_code"]}
                )
            )
        except Exception:
            continue

    # VIIRS 2020 annual mean NTL
    ntl_img = (
        ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
        .filterDate("2020-01-01", "2020-12-31")
        .select("avg_rad")
        .mean()
        .rename("ntl")
    )
    # GHSL built-up surface 2020 (m² built-up per 10 m pixel)
    ghsl_img = (
        ee.Image("JRC/GHSL/P2023A/GHS_BUILT_S/2020")
        .select("built_surface")
        .rename("ghsl")
    )
    combined = ntl_img.addBands(ghsl_img)

    reducer = (
        ee.Reducer.sum()
          .combine(ee.Reducer.mean(),   sharedInputs=True)
          .combine(ee.Reducer.max(),    sharedInputs=True)
          .combine(ee.Reducer.stdDev(), sharedInputs=True)
    )

    # Process in batches of 100 to stay within EE compute limits
    BATCH = 100
    all_props: dict = {}
    n_total = len(ee_features)

    for start in range(0, n_total, BATCH):
        batch = ee_features[start:start + BATCH]
        try:
            result = combined.reduceRegions(
                collection=ee.FeatureCollection(batch),
                reducer=reducer,
                scale=500,          # 500 m matches VIIRS native resolution
                crs="EPSG:4326",
            ).getInfo()
        except Exception as exc:
            print(f"  [GEE] Batch {start}–{start+len(batch)} failed: {exc}")
            continue

        for feat in result.get("features", []):
            props = feat["properties"]
            poc   = props.get("poc_code")
            if poc is None:
                continue
            all_props[poc] = {
                # VIIRS radiance (nW/cm²/sr)
                "ntl_sum":        float(props.get("ntl_sum",    0) or 0),
                "ntl_mean":       float(props.get("ntl_mean",   0) or 0),
                "ntl_max":        float(props.get("ntl_max",    0) or 0),
                "ntl_std":        float(props.get("ntl_stdDev", 0) or 0),
                # GHSL: sum is total m² of built-up surface → convert to km²
                "ghsl_built_km2": float(props.get("ghsl_sum",  0) or 0) / 1e6,
                "built_up_pct":   float(props.get("ghsl_mean", 0) or 0) * 100,
            }

        done = min(start + BATCH, n_total)
        if done < n_total:
            print(f"  [GEE] {done}/{n_total} polygons processed …")

    # Build output DataFrame keyed by poc_code (always unique) + bus_name for reference
    poc_to_name = bus_df.set_index("poc_code")["bus_name"].to_dict()
    rows = []
    for poc in voronoi_gdf["poc_code"].tolist():
        r = all_props.get(poc, {})
        rows.append({
            "poc_code":       poc,                           # unique join key
            "name":           poc_to_name.get(poc, poc),    # human-readable (may repeat)
            "ntl_sum":        r.get("ntl_sum", 0.0),
            "ntl_mean":       r.get("ntl_mean", 0.0),
            "ntl_max":        r.get("ntl_max", 0.0),
            "ntl_std":        r.get("ntl_std", 0.0),
            "ghsl_built_km2": r.get("ghsl_built_km2", 0.0),
            "built_up_pct":   r.get("built_up_pct", 0.0),
            "cropland_pct":   0.0,    # optional — add ESA WorldCover if needed
            "tree_cover_pct": 0.0,    # optional — add ESA WorldCover if needed
            "era5_temp_c":    20.0,   # placeholder; not used in base Ridge model
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    n_nonzero = (df["ntl_sum"] > 0).sum()
    print(f"  [GEE] Saved → {out_csv.name}  "
          f"({n_nonzero}/{len(df)} buses with NTL > 0)")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    paper_mode = config.LOAD_MODEL_METHOD == "paper_huber"
    if paper_mode:
        from paper_features import read_steel, read_gold, industrial_features
        viirs_path = RAW_DIR / config.VIIRS_RASTER_NAME
        if not viirs_path.exists():
            raise FileNotFoundError(f"Paper mode requires measured VIIRS raster: {viirs_path}")
        steel_sites = read_steel(RAW_DIR / config.STEEL_FACILITIES_NAME)
        gold_path = RAW_DIR / config.GOLD_FACILITIES_NAME
        if not gold_path.exists():
            gold_path = RAW_DIR / "gem" / "fineprint_mine_production.zip"
        gold_sites = read_gold(gold_path)
    print(f"=== build_region_features.py: {region} feature extraction ===\n")

    # Load bus coordinates from working_files
    csv_path = WORK_DIR / f"matpower_{region}_buses.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run resolve_unmatched.py first.")

    raw_buses = pd.read_csv(csv_path)
    raw_buses["base_kV"] = pd.to_numeric(raw_buses["base_kV"], errors="coerce")
    if paper_mode:
        # One service area per physical substation. Assign its demand to the
        # lowest modeled voltage tier, preserving other tiers for transmission.
        raw_buses = raw_buses.sort_values(["base_kV", "bus_i"]).drop_duplicates(["latitude", "longitude"])
    # All buses in this CSV already passed the MIN_TRANSMISSION_VOLTAGE_V filter in
    # create_buses.py; include them all for load feature extraction.
    min_kv = config.MIN_TRANSMISSION_VOLTAGE_V / 1000.0

    bus_rows = []
    for _, row in raw_buses.iterrows():
        bus_rows.append({
            "bus_idx":  int(row["bus_i"]),
            "bus_name": str(row["name"]),
            "poc_code": f"BUS_{int(row['bus_i'])}",
            "lat":      float(row["latitude"]),
            "lon":      float(row["longitude"]),
        })
    bus_df = pd.DataFrame(bus_rows).sort_values("bus_idx").reset_index(drop=True)
    print(f"Loaded {len(bus_df)} buses (voltage ≥ {min_kv:.0f} kV) "
          f"from matpower_{region}_buses.csv")

    # Build Voronoi polygons clipped to country boundary
    print(f"\nBuilding Voronoi polygons (clipped to {COUNTRY_ISO3} boundary) ...")
    land_poly = get_land_polygon(COUNTRY_ISO3, REGION_BBOX)
    voronoi_gdf = build_voronoi_gdf(
        bus_df[["poc_code","lat","lon"]],
        lat_col="lat", lon_col="lon", id_col="poc_code",
        bbox=REGION_BBOX, clip_geom=land_poly
    )
    print(f"  Areas: {voronoi_gdf.area_km2.min():.0f}-{voronoi_gdf.area_km2.max():.0f} km² "
          f"(median {voronoi_gdf.area_km2.median():.0f} km²)")

    bus_df["area_km2"]     = bus_df["poc_code"].map(voronoi_gdf.set_index("poc_code")["area_km2"])
    bus_df["centroid_lat"] = bus_df["poc_code"].map(
        {r["poc_code"]: r["geometry"].centroid.y for _, r in voronoi_gdf.iterrows()})
    bus_df["centroid_lon"] = bus_df["poc_code"].map(
        {r["poc_code"]: r["geometry"].centroid.x for _, r in voronoi_gdf.iterrows()})

    # Initialise NTL/GHSL/land-cover columns to zero (overwritten by GEE cache below)
    for col in ["ntl_sum","ntl_mean","ntl_max","ntl_std","ghsl_built_km2",
                "built_up_pct","cropland_pct","tree_cover_pct"]:
        bus_df[col] = 0.0

    # WorldPop population extraction
    # WorldPop URLs use the 3-letter ISO code (lowercase) for both the path and filename.
    # e.g. URY → https://…/2020/URY/ury_ppp_2020_1km_Aggregated.tif
    iso3_lower = COUNTRY_ISO3.lower()
    pop_tif    = RAW_DIR / f"worldpop_{region}.tif"

    if not pop_tif.exists():
        wp_url = (
            "https://data.worldpop.org/GIS/Population/"
            f"Global_2000_2020_1km/2020/{COUNTRY_ISO3}/"
            f"{iso3_lower}_ppp_2020_1km_Aggregated.tif"
        )
        print(f"\nDownloading WorldPop raster for {COUNTRY_ISO3} ...")
        print(f"  URL: {wp_url}")
        try:
            urllib.request.urlretrieve(wp_url, str(pop_tif))
            print(f"  OK ({pop_tif.stat().st_size/1e6:.1f} MB)")
        except Exception as exc:
            print(f"  FAILED ({exc})")
            print(f"  Manually download from:")
            print(f"    {wp_url}")
            print(f"  Save as: {pop_tif}")
            print("  Population will be 0 until the file is present.")

    bus_df["total_population"] = 0.0
    if pop_tif.exists():
        try:
            pop_sums = rasterio_zonal_sum(voronoi_gdf, str(pop_tif))
            new_pop = pd.Series(pop_sums, index=voronoi_gdf["poc_code"].values)
            bus_df["total_population"] = bus_df["poc_code"].map(new_pop).fillna(0.0)
            print(f"\nWorldPop pop: {bus_df.total_population.min():.0f}-"
                  f"{bus_df.total_population.max():.0f}")
        except Exception as exc:
            print(f"  rasterio zonal sum failed ({exc}) — population = 0")

    # OSM industrial polygon query
    S = REGION_BBOX[1]; W = REGION_BBOX[0]
    N = REGION_BBOX[3]; E = REGION_BBOX[2]
    ind_cache = RAW_DIR / f"osm_{region}_industrial_typed.json"
    if ind_cache.exists():
        print(f"\n[Overpass] Cache: {ind_cache.name}")
        with open(ind_cache) as f:
            ind_data = json.load(f)
    else:
        print(f"\nQuerying {region} industrial polygons from Overpass ...")
        query = f"""[out:json][timeout:300];
(
  way["landuse"="industrial"]({S},{W},{N},{E});
  way["landuse"="port"]({S},{W},{N},{E});
  way["landuse"~"quarry|mine"]({S},{W},{N},{E});
  way["man_made"~"works|smelter|refinery|kiln|gasometer"]({S},{W},{N},{E});
  way["industrial"~"aluminium|aluminum|steel|metal|smelter|refinery|chemical|oil|petroleum|gas|cement|mining|quarry|power"]({S},{W},{N},{E});
  node["man_made"="mineshaft"]({S},{W},{N},{E});
  node["man_made"="petroleum_well"]({S},{W},{N},{E});
  node["harbour"="yes"]({S},{W},{N},{E});
);
out geom tags;
"""
        ind_data = {"elements": []}
        for attempt in range(3):
            try:
                r = requests.post(OVERPASS_URL, data={"data": query},
                                  headers=HEADERS, timeout=300)
                if r.status_code == 200:
                    ind_data = r.json()
                    with open(ind_cache, "w") as f:
                        json.dump(ind_data, f)
                    print(f"  OK ({len(ind_data.get('elements',[]))} elements)")
                    break
                time.sleep(15)
            except Exception as exc:
                print(f"  Error: {exc}")
                time.sleep(20)

    way_rows, point_rows = parse_industrial_elements(ind_data.get("elements", []))
    print(f"  {len(way_rows)} way polygons, {len(point_rows)} point features")

    # Load WRI Global Power Plant Database for industrial polygon type correction
    WRI_URL   = ("https://raw.githubusercontent.com/wri/global-power-plant-database"
                 "/master/output_database/global_power_plant_database.csv")
    WRI_CACHE = RAW_DIR / "global_power_plants.csv"
    wri_gdf = None
    if WRI_CACHE.exists() and WRI_CACHE.stat().st_size > 100_000:
        df_wri = pd.read_csv(WRI_CACHE, low_memory=False)
        df_wri = df_wri.dropna(subset=["latitude","longitude"])
        df_wri["capacity_mw"] = pd.to_numeric(df_wri["capacity_mw"], errors="coerce").fillna(0)
        fuel_map = {"Gas":"heavy_chemical","Oil":"heavy_chemical",
                    "Coal":"heavy_other","Biomass":"heavy_other"}
        df_wri["wri_type"] = df_wri["primary_fuel"].map(fuel_map)
        keep = ((df_wri["primary_fuel"].isin(["Gas","Oil"]) & (df_wri["capacity_mw"] > 50)) |
                (df_wri["primary_fuel"] == "Coal") |
                ((df_wri["primary_fuel"] == "Biomass") & (df_wri["capacity_mw"] > 50)))
        df_wri = df_wri[keep & df_wri["wri_type"].notna()]
        if len(df_wri) >= 10:
            wri_gdf = gpd.GeoDataFrame(df_wri,
                geometry=gpd.points_from_xy(df_wri["longitude"], df_wri["latitude"]),
                crs="EPSG:4326")
            print(f"  [WRI] {len(wri_gdf):,} heavy-industrial plants loaded")

    if wri_gdf is not None:
        way_rows = apply_wri_override(way_rows, wri_gdf)

    # Apply user-supplied known industrial overrides (see KNOWN_REGIONAL_INDUSTRIAL above)
    if KNOWN_REGIONAL_INDUSTRIAL:
        way_rows = apply_known_industrial_override(way_rows, KNOWN_REGIONAL_INDUSTRIAL)

    # Aggregate typed industrial features per Voronoi cell
    print("\nAggregating industrial features per Voronoi cell ...")
    ind_df = compute_typed_industrial(voronoi_gdf, way_rows, point_rows, "poc_code")
    bus_df = bus_df.merge(ind_df, on="poc_code", how="left")
    for col in ind_df.columns:
        if col != "poc_code":
            bus_df[col] = bus_df[col].fillna(0.0)

    # OSM road length (motorway + trunk)
    bus_df["road_length_km"] = 0.0
    road_cache = RAW_DIR / f"osm_{region}_roads.json"
    if not road_cache.exists():
        print(f"\nQuerying {region} roads from Overpass ...")
        road_query = f"""[out:json][timeout:300];
way["highway"~"^(motorway|trunk)$"]({S},{W},{N},{E});
out geom;
"""
        for attempt in range(3):
            try:
                r = requests.post(OVERPASS_URL, data={"data": road_query},
                                  headers=HEADERS, timeout=400)
                if r.status_code == 200:
                    road_data = r.json()
                    with open(road_cache, "w") as f:
                        json.dump(road_data, f)
                    break
                time.sleep(20)
            except Exception as exc:
                print(f"  Road query error: {exc}")
                time.sleep(20)
    if road_cache.exists():
        with open(road_cache) as f:
            road_data = json.load(f)
        road_rows_geo = []
        for el in road_data.get("elements", []):
            if el["type"] == "way" and "geometry" in el:
                coords = [(pt["lon"], pt["lat"]) for pt in el["geometry"]]
                if len(coords) >= 2:
                    road_rows_geo.append({"geometry": LineString(coords)})
        if road_rows_geo:
            roads_gdf = gpd.GeoDataFrame(road_rows_geo, crs="EPSG:4326")
            joined = gpd.sjoin(roads_gdf, voronoi_gdf[["poc_code","geometry"]],
                               how="left", predicate="intersects")
            def line_km(geom):
                coords = list(geom.coords)
                total = 0.0
                for i in range(len(coords)-1):
                    dlon = coords[i+1][0] - coords[i][0]
                    dlat = coords[i+1][1] - coords[i][1]
                    mid_lat = (coords[i][1] + coords[i+1][1]) / 2
                    dx = dlon * 111.32 * np.cos(np.radians(mid_lat))
                    dy = dlat * 111.32
                    total += np.sqrt(dx**2 + dy**2)
                return total
            joined["road_km"] = joined.geometry.apply(line_km)
            road_km_map = joined.groupby("poc_code")["road_km"].sum().to_dict()
            bus_df["road_length_km"] = bus_df["poc_code"].map(road_km_map).fillna(0.0)

    # OSM commercial/amenity node counts
    bus_df["num_commercial_nodes"] = 0
    comm_cache = RAW_DIR / f"osm_{region}_commercial.json"
    if not comm_cache.exists():
        print(f"\nQuerying {region} commercial nodes from Overpass ...")
        comm_query = f"""[out:json][timeout:300];
(
  node["shop"]({S},{W},{N},{E});
  node["amenity"~"^(marketplace|bank|pharmacy)$"]({S},{W},{N},{E});
  node["building"~"^(commercial|retail|supermarket|mall)$"]({S},{W},{N},{E});
);
out body;
"""
        for attempt in range(3):
            try:
                r = requests.post(OVERPASS_URL, data={"data": comm_query},
                                  headers=HEADERS, timeout=300)
                if r.status_code == 200:
                    comm_data = r.json()
                    with open(comm_cache, "w") as f:
                        json.dump(comm_data, f)
                    break
                time.sleep(15)
            except Exception:
                time.sleep(20)
    if comm_cache.exists():
        with open(comm_cache) as f:
            comm_data = json.load(f)
        nodes = [{"geometry": Point(el["lon"], el["lat"])}
                 for el in comm_data.get("elements", [])
                 if el.get("type") == "node" and el.get("lat") is not None]
        if nodes:
            nodes_gdf = gpd.GeoDataFrame(nodes, crs="EPSG:4326")
            jn = gpd.sjoin(nodes_gdf, voronoi_gdf[["poc_code","geometry"]],
                           how="left", predicate="within")
            cnt = jn.groupby("poc_code").size().to_dict()
            bus_df["num_commercial_nodes"] = bus_df["poc_code"].map(cnt).fillna(0).astype(int)

    if not paper_mode:
        # NTL/GHSL: try GEE first, fall back to NZ proxy
        gee_cache = RAW_DIR / f"gee_features_{region}_matpower.csv"

        if not gee_cache.exists():
            print("\n[GEE] Satellite feature cache not found — attempting download …")
            gee_ok = download_gee_features(voronoi_gdf, bus_df, gee_cache)
            if not gee_ok:
                print("  [GEE] Download skipped. Falling back to NZ proxy for NTL/GHSL.")
                print("  NTL/GHSL proxy accuracy is lower than real satellite values.")

        if gee_cache.exists():
            print(f"\n[GEE] Loading satellite features from {gee_cache.name} …")
            gee_df   = pd.read_csv(gee_cache)
            gee_cols = ["ntl_sum","ntl_mean","ntl_max","ntl_std",
                        "ghsl_built_km2","era5_temp_c","built_up_pct","cropland_pct","tree_cover_pct"]
            gee_cols_present = [c for c in gee_cols if c in gee_df.columns]

            if "poc_code" in gee_df.columns:
                # Preferred path: merge on poc_code (always unique)
                merged = bus_df.merge(
                    gee_df[["poc_code"] + gee_cols_present],
                    on="poc_code", how="left", suffixes=("", "_gee")
                )
                for col in gee_cols_present:
                    # pandas only appends the suffix when the column collides with
                    # one already in bus_df. A GEE column that bus_df does not have
                    # (era5_temp_c, for one) merges under its plain name, so look
                    # for the suffixed form first and fall back to the plain one.
                    source = col + "_gee" if col + "_gee" in merged.columns else col
                    values = merged[source].to_numpy(dtype=float, na_value=np.nan)
                    prior = (bus_df[col].to_numpy(dtype=float)
                             if col in bus_df.columns else 0.0)
                    bus_df[col] = np.where(np.isnan(values), prior, values)
            else:
                # Legacy fallback: name-based lookup; deduplicate first to avoid the
                # InvalidIndexError that occurs when two buses share the same name.
                if gee_df["name"].duplicated().any():
                    n_dup = int(gee_df["name"].duplicated().sum())
                    print(f"  [GEE] {n_dup} duplicate bus names — aggregating by mean.")
                    gee_df = gee_df.groupby("name", as_index=False)[gee_cols_present].mean(
                        numeric_only=True)
                gee_idx = gee_df.set_index("name")
                for col in gee_cols_present:
                    mapped = bus_df["bus_name"].map(gee_idx[col])
                    bus_df[col] = mapped.where(mapped.notna(), bus_df.get(col, 0.0))

            n_nonzero = (bus_df["ntl_sum"] > 0).sum()
            print(f"  Applied NTL/GHSL: {n_nonzero}/{len(bus_df)} buses with ntl_sum > 0")
        else:
            print("\nApplying NTL/GHSL proxy from NZ data (GEE not available) …")
            bus_df = proxy_ntl_ghsl_from_nz(bus_df)

    # Compute derived density features
    area_s = bus_df["area_km2"].clip(lower=1.0)
    bus_df["pop_density"]          = bus_df["total_population"] / area_s
    bus_df["road_density"]         = bus_df["road_length_km"] / area_s
    bus_df["ntl_per_km2"]          = bus_df["ntl_sum"].fillna(0) / area_s
    bus_df["ghsl_pct"]             = bus_df["ghsl_built_km2"].fillna(0) / area_s * 100
    bus_df["commercial_intensity"] = bus_df["num_commercial_nodes"] / area_s
    bus_df["log_area_km2"]         = np.log1p(bus_df["area_km2"])

    if paper_mode:
        guarded = industrial_features(voronoi_gdf, way_rows, steel_sites, gold_sites, viirs_path)
        replaced = [c for c in guarded if c != "poc_code" and c in bus_df]
        bus_df = bus_df.drop(columns=replaced).merge(guarded, on="poc_code", validate="one_to_one")
    suffix = "paper_features" if paper_mode else "features_with_industrial"
    out_path = FEAT_DIR / f"{region}_{suffix}.csv"
    bus_df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}  ({len(bus_df)} rows × {bus_df.shape[1]} cols)")
    print("\nNext step: python predict_bus_loads.py")


if __name__ == "__main__":
    main()
