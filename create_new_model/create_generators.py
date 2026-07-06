"""
create_generators.py

Processes OSM power plant data (GeoJSON from Overpass Turbo) to produce
a MATPOWER-compatible generator file for a given region.

Pipeline:
  1.  Load OSM plant GeoJSON
  2.  Filter factory/industrial false positives
  3.  Compute polygon centroids
  4.  Extract plant attributes (name, capacity, source, operator, status)
  5.  Normalize fields (parse capacity to MW, normalize fuel/source labels)
  6.  Deduplicate plants (same OSM object, same-name/same-location)
  7.  Assign nearest bus (from matpower_{region}_buses.csv)
  8.  Compute MATPOWER generator limits (Pmax, Qmax, Qmin)
  9.  Build MATPOWER generator table and save CSV

NOTE: Run create_buses_new.py before this script.  The bus CSV it
produces (working_files/matpower_{region}_buses.csv) must contain latitude/longitude
columns for the nearest-bus assignment in Step 7.

Author: Richard Asiamah
"""

import re
import math
import geopandas as gpd
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

from pathlib import Path
import config

region    = config.region
PROJ_ROOT = Path(__file__).parent.parent
WORK_DIR  = PROJ_ROOT / "data" / "working_files" / region
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Minimum capacity threshold (MW). Plants with a known capacity below
# this value are dropped after normalization. Plants with no capacity
# tag are kept (unknown ≠ zero).
MIN_CAPACITY_MW = 0.0


# ============================================================
# CONSTANTS
# ============================================================

# Reactive power scaling factors per technology.
# Qmax = qmax_factor * Pmax
# Qmin = qmin_factor * Pmax   (qmin_factor is negative)
REACTIVE_LIMIT_COEFFICIENTS = {
    "steam":       ( 0.466, -0.122),
    "gas":         ( 0.509, -0.111),
    "gas_turbine": ( 0.560, -0.064),
    "hydro":       ( 0.384, -0.049),
    "nuclear":     ( 0.368, -0.082),
    "wind":        ( 0.213, -0.144),
    "solar":       ( 0.329, -0.329),
}

# OSM fuel/source labels → generator technology category.
REACTIVE_SOURCE_MAP = {
    "biogas":      "gas",
    "biomass":     "steam",
    "coal":        "steam",
    "diesel":      "gas_turbine",
    "gas":         "gas",
    "gasoline":    "gas_turbine",
    "gas turbine": "gas_turbine",
    "geothermal":  "steam",
    "hydro":       "hydro",
    "nuclear":     "nuclear",
    "oil":         "gas_turbine",
    "solar":       "solar",
    "steam":       "steam",
    "waste":       "steam",
    "wind":        "wind",
}

# OSM tag names that may hold capacity/output information, in priority order.
CAPACITY_KEYS = [
    "plant:output:electricity",
    "generator:output:electricity",
    "output:electricity",
    "plant:capacity",
    "generator:capacity",
    "capacity",
]

# OSM tag names that may hold fuel/source, generation method, and status.
SOURCE_KEYS = ["plant:source", "generator:source", "source", "fuel"]
METHOD_KEYS = ["plant:method", "generator:method", "method"]
STATUS_KEYS = ["plant:status", "generator:status", "status"]

# Unit conversion to MW.
UNIT_TO_MW = {"gw": 1000.0, "mw": 1.0, "kw": 0.001, "w": 0.000001}

# OSM status values that indicate a plant is not yet in service.
INACTIVE_STATUS_WORDS = ("planned", "proposed", "construction", "abandoned", "disused")

# Keywords that suggest a false-positive power=plant tag on an industrial site.
FACTORY_KEYWORDS = {
    "brewery", "cement", "chemical", "factory", "industrial", "industry",
    "manufacturing", "mine", "mining", "mill", "processing", "refinery",
    "sawmill", "smelter", "steel", "works",
}
INDUSTRIAL_TAG_VALUES = {"industrial", "works", "factory"}

# Generic words removed before comparing plant names for duplicate detection.
DUPLICATE_GENERIC_WORDS = {
    "power", "plant", "station", "generating", "generation", "generator",
    "electric", "electricity", "energy", "company", "limited", "ltd",
    "inc", "llc", "the",
}

# Distance threshold (meters) for proximity-based duplicate detection.
DUPLICATE_RADIUS_METERS = 100.0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_text(value) -> str:
    """Trim, collapse whitespace, and strip non-breaking spaces."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()


def first_present(row: pd.Series, keys: list) -> str:
    """Return the first non-empty value from a list of column names."""
    for key in keys:
        val = normalize_text(row.get(key, ""))
        if val:
            return val
    return ""


def parse_capacity_mw(value) -> float | None:
    """Parse OSM capacity text such as '50 MW' or '2 x 20 MW' into MW."""
    text = normalize_text(value).lower()
    if not text:
        return None
    text = (text.replace("megawatt", "mw")
               .replace("kilowatt", "kw")
               .replace("gigawatt", "gw"))
    pattern = re.compile(
        r"(?:(?P<count>\d+(?:\.\d+)?)\s*(?:x|times|\*)\s*)?"
        r"(?P<value>\d[\d,]*(?:\.\d+)?)\s*(?P<unit>gw|mw|kw|w)\b"
    )
    matches = []
    for m in pattern.finditer(text):
        # Multiple-unit expressions like '2 x 20 MW' are summed to 40 MW.
        count = float(m.group("count") or 1)
        amount = float(m.group("value").replace(",", ""))
        unit = m.group("unit")
        matches.append(count * amount * UNIT_TO_MW[unit])
    if matches:
        return sum(matches)
    # Bare number with no unit is assumed to be MW.
    m = re.search(r"\d[\d,]*(?:\.\d+)?", text)
    return float(m.group(0).replace(",", "")) if m else None


def normalize_source(value: str) -> str:
    """Normalize OSM fuel/source labels to standard category names."""
    text = normalize_text(value).lower().replace("_", " ").replace("-", " ")
    replacements = {
        "natural gas": "gas", "fossil gas": "gas", "nat gas": "gas", "ng": "gas",
        "fuel oil": "oil", "hydroelectric": "hydro", "hydropower": "hydro", "water": "hydro",
        "photovoltaic": "solar", "pv": "solar",
    }
    parts = []
    for part in re.split(r"[;,/]+", text):
        clean = normalize_text(part)
        if clean:
            parts.append(replacements.get(clean, clean))
    # dict.fromkeys preserves order while removing repeated labels.
    return ";".join(dict.fromkeys(parts))


def primary_source(source: str) -> str:
    """Return the first label from a semicolon-separated source string."""
    s = normalize_source(source)
    return s.split(";")[0] if s else "unknown"


def reactive_category(source: str) -> str | None:
    """Map a source label to a reactive-limit technology category."""
    fuel = primary_source(source)
    return REACTIVE_SOURCE_MAP.get(fuel)


def status_to_matpower(value: str) -> int:
    """Convert OSM status text to MATPOWER status (1 = in service, 0 = inactive)."""
    text = normalize_text(value).lower()
    if any(word in text for word in INACTIVE_STATUS_WORDS):
        return 0
    return 1


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon coordinate pairs."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def has_power_evidence(row: pd.Series) -> bool:
    """Return True if the row has any capacity, fuel, or generation method tag."""
    empty = {"", "0", "none", "null", "unknown", "yes", "no", "n/a"}
    for key in CAPACITY_KEYS + SOURCE_KEYS + METHOD_KEYS:
        val = normalize_text(row.get(key, "")).lower()
        if val and val not in empty:
            return True
    return False


def is_factory_false_positive(row: pd.Series) -> bool:
    """Return True for industrial/factory sites tagged as power=plant without
    any evidence of actual electricity generation."""
    if normalize_text(row.get("power", "")).lower() != "plant":
        return False
    if has_power_evidence(row):
        return False
    haystack = " ".join(
        normalize_text(row.get(k, "")).lower()
        for k in ["name", "operator", "industrial", "landuse", "man_made", "building", "description"]
    )
    keyword_hit = any(kw in haystack for kw in FACTORY_KEYWORDS)
    tag_hit = any(
        normalize_text(row.get(k, "")).lower() in INDUSTRIAL_TAG_VALUES
        for k in ["landuse", "man_made", "building"]
    )
    return keyword_hit or tag_hit


def clean_for_dedup(name: str) -> str:
    """Reduce a plant name to comparable tokens for duplicate matching."""
    text = normalize_text(name).lower().strip("[]")
    text = re.sub(r"\bunnamed\b", "", text)
    text = re.sub(r"\b(node|way|relation)/?\d+\b", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [t for t in text.split() if t not in DUPLICATE_GENERIC_WORDS]
    return " ".join(tokens)


def names_match(a: str, b: str) -> bool:
    """Return True when two plant names are effectively the same after normalization."""
    left, right = clean_for_dedup(a), clean_for_dedup(b)
    if not left or not right:
        return False
    if left == right:
        return True
    if len(left) >= 7 and len(right) >= 7 and (left in right or right in left):
        return True
    lt, rt = set(left.split()), set(right.split())
    if len(lt | rt) < 2:
        return False
    return len(lt & rt) / len(lt | rt) >= 0.8


def completeness_score(row: pd.Series) -> tuple:
    """Score a plant record so the more informative duplicate is kept."""
    score = 0
    if not normalize_text(row.get("plant_name", "")).startswith("[unnamed"):
        score += 10
    if normalize_text(row.get("operator_out", "")):
        score += 3
    cap = row.get("capacity_mw")
    if cap is not None and not (isinstance(cap, float) and np.isnan(cap)):
        score += 3
    src = normalize_text(row.get("source", ""))
    if src and src != "unknown":
        score += 2
    cap_val = float(cap) if (cap is not None and not (isinstance(cap, float) and np.isnan(cap))) else 0.0
    return (score, cap_val, len(normalize_text(row.get("plant_name", ""))))


# ============================================================
# STEP 1 – LOAD DATA
# ============================================================
print("=" * 60)
print("STEP 1: Loading plant GeoJSON")
print("=" * 60)

gdf = gpd.read_file(PROJ_ROOT / "data" / "OSM_extracts" / f"export_{region}_plants.geojson")

# Keep only power=plant features (the GeoJSON may contain mixed feature types).
gdf = gdf[gdf["power"].str.lower().eq("plant")].reset_index(drop=True)

geom_counts = gdf.geometry.geom_type.value_counts().to_dict()
print(f"  Loaded {len(gdf)} power=plant features")
print(f"  Geometry types: {geom_counts}")


# ============================================================
# STEP 2 – FILTER FACTORY/INDUSTRIAL FALSE POSITIVES
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Filtering factory/industrial false positives")
print("=" * 60)

# Drop rows that look like factories or industrial compounds tagged as power=plant
# without any evidence of electricity generation (no capacity, source, or method).
# These arise when OSM contributors tag large industrial areas with power=plant
# in error. A row is flagged only when industrial keywords appear AND no power
# evidence is present, so genuine industrial generators (e.g. biomass at a factory)
# are not removed.

drop_fp = []
for idx, row in gdf.iterrows():
    if is_factory_false_positive(row):
        drop_fp.append(idx)
        print(f"  [DROP false-positive] idx={idx}  id={normalize_text(row.get('id', row.get('@id', '')))}  "
              f"name={normalize_text(row.get('name', ''))}")

gdf = gdf.drop(index=drop_fp).reset_index(drop=True)
print(f"  Dropped {len(drop_fp)} false positives.  {len(gdf)} plants remaining.")


# ============================================================
# STEP 3 – COMPUTE CENTROIDS
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Computing centroid coordinates")
print("=" * 60)

# For Polygon features the centroid is the geographic centre of the plant
# compound. For Point features (node-tagged plants) the point is used directly.
# These coordinates are used for bus assignment in Step 7.

centroids = gdf.geometry.centroid
gdf["longitude"] = centroids.x
gdf["latitude"]  = centroids.y

print(f"  Centroid computed for all {len(gdf)} features.")


# ============================================================
# STEP 4 – EXTRACT PLANT ATTRIBUTES
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Extracting plant attributes")
print("=" * 60)

# Name priority: 'name' → 'name:en' → 'official_name' → 'operator' → 'ref' → OSM-ID label
# Capacity: first non-empty value from CAPACITY_KEYS list
# Source/fuel, method, operator, status: first non-empty value from respective key lists

NAME_PRIORITY_KEYS = ["name", "name:en", "official_name", "operator", "ref"]


def extract_name(row: pd.Series) -> str:
    for key in NAME_PRIORITY_KEYS:
        val = normalize_text(row.get(key, ""))
        if val:
            return val
    osm_id = normalize_text(row.get("id", row.get("@id", "")))
    return f"[unnamed {osm_id}]" if osm_id else "[unnamed]"


gdf["plant_name"]   = gdf.apply(extract_name, axis=1)
gdf["capacity_raw"] = gdf.apply(lambda r: first_present(r, CAPACITY_KEYS), axis=1)
gdf["source_raw"]   = gdf.apply(lambda r: first_present(r, SOURCE_KEYS), axis=1)
gdf["operator_out"] = gdf.apply(lambda r: first_present(r, ["operator", "owner"]), axis=1)
gdf["status_raw"]   = gdf.apply(lambda r: first_present(r, STATUS_KEYS), axis=1)

n_named   = (~gdf["plant_name"].str.startswith("[unnamed")).sum()
n_unnamed = len(gdf) - n_named
print(f"  Named plants:   {n_named}")
print(f"  Unnamed plants: {n_unnamed}")


# ============================================================
# STEP 5 – NORMALIZE FIELDS
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Normalizing capacity and source fields")
print("=" * 60)

# capacity_mw: parse raw OSM text into a float (MW); None when absent or unparseable.
# source:      normalize fuel labels to standard category names; 'unknown' when absent.
# status:      convert OSM status text to MATPOWER 0/1.

gdf["capacity_mw"] = gdf["capacity_raw"].apply(parse_capacity_mw)
gdf["source"]      = gdf["source_raw"].apply(normalize_source)
gdf["source"]      = gdf["source"].replace("", "unknown")
gdf["status"]      = gdf["status_raw"].apply(status_to_matpower)

n_with_cap    = gdf["capacity_mw"].notna().sum()
n_without_cap = gdf["capacity_mw"].isna().sum()
n_inactive    = (gdf["status"] == 0).sum()

print(f"  Plants with capacity:    {n_with_cap}")
print(f"  Plants without capacity: {n_without_cap}  (will remain blank in output)")
print(f"  Inactive plants (status=0): {n_inactive}")

if MIN_CAPACITY_MW > 0:
    mask_below = gdf["capacity_mw"].notna() & (gdf["capacity_mw"] < MIN_CAPACITY_MW)
    n_below = int(mask_below.sum())
    gdf = gdf[~mask_below].reset_index(drop=True)
    print(f"  Dropped {n_below} plants below {MIN_CAPACITY_MW:g} MW.")


# ============================================================
# STEP 6 – DEDUPLICATE
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Deduplicating plant records")
print("=" * 60)

# Two rows are treated as duplicates when they share the same OSM id, or when
# they are within DUPLICATE_RADIUS_METERS of each other AND either:
#   (a) one is unnamed and the other is named (unnamed outline of named plant), or
#   (b) their names match after normalization.
# The row with the higher completeness score (name → operator → capacity → source)
# is kept; the lower-scoring row is dropped.

records = gdf.reset_index(drop=True).copy()
drop_dupes = set()

for i in range(len(records)):
    if i in drop_dupes:
        continue
    ri = records.iloc[i]
    id_i = normalize_text(ri.get("id", ri.get("@id", "")))

    for j in range(i + 1, len(records)):
        if j in drop_dupes:
            continue
        rj = records.iloc[j]
        id_j = normalize_text(rj.get("id", rj.get("@id", "")))

        same_id = bool(id_i and id_j and id_i == id_j)
        reason = ""

        if same_id:
            reason = f"same OSM id ({id_i})"
        else:
            lat_i, lon_i = ri["latitude"], ri["longitude"]
            lat_j, lon_j = rj["latitude"], rj["longitude"]
            if any(pd.isna(v) for v in [lat_i, lon_i, lat_j, lon_j]):
                continue
            dist_m = haversine_km(lat_i, lon_i, lat_j, lon_j) * 1000
            if dist_m > DUPLICATE_RADIUS_METERS:
                continue
            unnamed_i = ri["plant_name"].startswith("[unnamed")
            unnamed_j = rj["plant_name"].startswith("[unnamed")
            if unnamed_i != unnamed_j:
                reason = f"unnamed outline within {dist_m:.0f} m of named plant"
            elif names_match(ri["plant_name"], rj["plant_name"]):
                reason = f"same/similar name within {dist_m:.0f} m"
            else:
                continue

        si = completeness_score(ri)
        sj = completeness_score(rj)
        drop_idx = j if si >= sj else i
        keep_idx = i if drop_idx == j else j

        print(f"  [DROP duplicate] '{records.iloc[drop_idx]['plant_name']}' ({reason})"
              f" → keeping '{records.iloc[keep_idx]['plant_name']}'")
        drop_dupes.add(drop_idx)
        if drop_idx == i:
            break

gdf = records.drop(index=list(drop_dupes)).reset_index(drop=True)
print(f"  Dropped {len(drop_dupes)} duplicate records.  {len(gdf)} plants remaining.")


# ============================================================
# STEP 7 – ASSIGN NEAREST BUS
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Assigning nearest bus")
print("=" * 60)

# Load the bus CSV produced by create_buses_new.py and find the nearest bus
# for each plant by Haversine distance. No voltage-level filter is applied
# because power plants can connect at multiple voltage levels and OSM plant
# tags do not reliably carry voltage information.

buses     = pd.read_csv(WORK_DIR / f"matpower_{region}_buses.csv")
bus_lats  = buses["latitude"].values
bus_lons  = buses["longitude"].values
bus_names = buses["name"].values
bus_ids   = buses["bus_i"].values

assigned_bus_id   = []
assigned_bus_name = []
assigned_distance = []

for _, plant in gdf.iterrows():
    plat, plon = plant["latitude"], plant["longitude"]
    if pd.isna(plat) or pd.isna(plon):
        assigned_bus_id.append(np.nan)
        assigned_bus_name.append("")
        assigned_distance.append(np.nan)
        continue
    dists = [haversine_km(plat, plon, blat, blon)
             for blat, blon in zip(bus_lats, bus_lons)]
    idx = int(np.argmin(dists))
    assigned_bus_id.append(int(bus_ids[idx]))
    assigned_bus_name.append(str(bus_names[idx]))
    assigned_distance.append(round(dists[idx], 3))

gdf["bus_i"]            = assigned_bus_id
gdf["bus_name"]         = assigned_bus_name
gdf["bus_distance_km"]  = assigned_distance

valid_dists = [d for d in assigned_distance if not (isinstance(d, float) and np.isnan(d))]
print(f"  Assigned nearest bus for {len(gdf)} plants from {len(buses)} buses.")
if valid_dists:
    print(f"  Median assignment distance: {np.median(valid_dists):.1f} km")
    print(f"  Max    assignment distance: {max(valid_dists):.1f} km")


# ============================================================
# STEP 8 – COMPUTE GENERATOR LIMITS
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Computing MATPOWER generator limits")
print("=" * 60)

# Pmax = rated capacity (MW) from OSM; Pmin = 0 (conservative default, no OSM data).
# Qmax and Qmin are derived from Pmax by direct multiplication:
#   Qmax = qmax_factor * Pmax
#   Qmin = qmin_factor * Pmax   (qmin_factor is negative)
# Plants with no capacity leave all limits blank. Plants with capacity but an
# unrecognized fuel source get P limits only; Q limits are left blank for manual review.

pmax_vals, pmin_vals, qmax_vals, qmin_vals, genfuel_vals = [], [], [], [], []
n_no_capacity = 0
n_no_source   = 0

for _, row in gdf.iterrows():
    pmax = row["capacity_mw"]
    src  = row["source"]
    cat  = reactive_category(src)
    genfuel_vals.append(cat if cat else primary_source(src))

    if pd.isna(pmax):
        pmax_vals.append(np.nan)
        pmin_vals.append(np.nan)
        qmax_vals.append(np.nan)
        qmin_vals.append(np.nan)
        n_no_capacity += 1
        continue

    pmax = float(pmax)
    pmax_vals.append(pmax)
    pmin_vals.append(0.0)

    if cat is None:
        qmax_vals.append(np.nan)
        qmin_vals.append(np.nan)
        n_no_source += 1
    else:
        qmax_f, qmin_f = REACTIVE_LIMIT_COEFFICIENTS[cat]
        qmax_vals.append(round(qmax_f * pmax, 3))
        qmin_vals.append(round(qmin_f * pmax, 3))

gdf["Pmax"]    = pmax_vals
gdf["Pmin"]    = pmin_vals
gdf["Qmax"]    = qmax_vals
gdf["Qmin"]    = qmin_vals
gdf["genfuel"] = genfuel_vals

n_full = int(pd.Series(qmax_vals).notna().sum())
print(f"  Plants with full P and Q limits:                {n_full}")
print(f"  Plants missing capacity (no limits):            {n_no_capacity}")
print(f"  Plants with capacity but unknown source         {n_no_source}")
print(f"    (P limits set; Q limits blank — review fuel)")


# ============================================================
# STEP 9 – BUILD MATPOWER GENERATOR TABLE AND SAVE
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: Building MATPOWER generator table")
print("=" * 60)

# Standard MATPOWER mpc.gen columns:
#   bus  Pg  Qg  Qmax  Qmin  Vg  mBase  status  Pmax  Pmin
#
# Pg = Qg = 0.0  — initial dispatch placeholder; set by OPF/power flow.
# Vg = 1.0       — nominal per-unit voltage setpoint.
# mBase = 100    — system MVA base (matches MATPOWER convention).
#
# Additional reference columns appended for traceability:
#   generator_name, operator, source, genfuel, latitude, longitude,
#   osm_id, bus_name, bus_distance_km

osm_id_col = gdf.apply(
    lambda r: normalize_text(r.get("id", r.get("@id", ""))), axis=1
)

gen_out = pd.DataFrame({
    "bus":             gdf["bus_i"],
    "Pg":              0.0,
    "Qg":              0.0,
    "Qmax":            gdf["Qmax"],
    "Qmin":            gdf["Qmin"],
    "Vg":              1.0,
    "mBase":           100.0,
    "status":          gdf["status"],
    "Pmax":            gdf["Pmax"],
    "Pmin":            gdf["Pmin"],
    "generator_name":  gdf["plant_name"],
    "operator":        gdf["operator_out"].replace("", "unknown"),
    "source":          gdf["source"],
    "genfuel":         gdf["genfuel"],
    "latitude":        gdf["latitude"].round(6),
    "longitude":       gdf["longitude"].round(6),
    "osm_id":          osm_id_col,
    "bus_name":        gdf["bus_name"],
    "bus_distance_km": gdf["bus_distance_km"],
})

out_path = WORK_DIR / f"matpower_{region}_generators.csv"
gen_out.to_csv(out_path, index=False)
print(f"  Saved {len(gen_out)} generators → {out_path}")
print()
print(gen_out[["generator_name", "bus", "bus_name", "Pmax", "source", "genfuel"]].to_string(index=False))


# ============================================================
# POST-PROCESSING: Assign MATPOWER bus types (PQ / PV / slack)
# ============================================================
print("\n" + "=" * 60)
print("POST-PROCESSING: Assigning bus types (type 1/2/3)")
print("=" * 60)

# MATPOWER bus type convention:
#   1 = PQ  (load bus, no generator)
#   2 = PV  (generator bus, voltage-controlled)
#   3 = slack / reference bus (one bus per connected island)
#
# Rules:
#   - Buses with at least one generator → type 2 (PV)
#   - Buses with no generator           → type 1 (PQ)
#   - The bus connected to the generator with the highest Pmax → type 3 (slack)

buses_path = WORK_DIR / f"matpower_{region}_buses.csv"
buses_df   = pd.read_csv(buses_path)

# Default all buses to type 1
buses_df["type"] = 1

# Identify buses that have at least one generator
gen_valid = gen_out.dropna(subset=["bus"])
gen_bus_ids = set(gen_valid["bus"].astype(int).tolist())
buses_df.loc[buses_df["bus_i"].isin(gen_bus_ids), "type"] = 2

# Find the bus connected to the generator with the highest Pmax
if gen_valid["Pmax"].notna().any():
    max_pmax_row = gen_valid.loc[gen_valid["Pmax"].idxmax()]
    slack_bus    = int(max_pmax_row["bus"])
    buses_df.loc[buses_df["bus_i"] == slack_bus, "type"] = 3
    print(f"  Slack bus (type 3): bus {slack_bus} "
          f"({max_pmax_row.get('bus_name', '?')})  "
          f"Pmax={max_pmax_row['Pmax']:.1f} MW  "
          f"({max_pmax_row.get('generator_name', '?')})")
else:
    print("  Warning: no generators have a known Pmax; no type-3 bus assigned.")

n_type1 = (buses_df["type"] == 1).sum()
n_type2 = (buses_df["type"] == 2).sum()
n_type3 = (buses_df["type"] == 3).sum()
print(f"  type 1 (PQ):    {n_type1} buses")
print(f"  type 2 (PV):    {n_type2} buses")
print(f"  type 3 (slack): {n_type3} buses")

buses_df.to_csv(buses_path, index=False)
print(f"  Updated bus types → {buses_path}")

print("\nDone.")
