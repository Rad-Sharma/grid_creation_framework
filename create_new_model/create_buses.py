"""
create_buses.py

Processes OSM substation data (GeoJSON from Overpass Turbo) to produce
a MATPOWER-compatible bus file for a given region.

Key behaviours
--------------
* Substations with compound voltage tags (e.g. '330000;161000') are split
  into one bus per voltage level, each named {name}_{kV}kV.  A companion
  file transformer_pairs_{region}.csv records which bus pairs need a
  transformer branch added by create_synthetic_grid.py.
* Bus names are derived from OSM tags first (instant), then from an offline
  GeoNames reverse-geocoder (fast batch lookup), with Nominatim as a
  last-resort fallback.

Output
------
  data/working_files/{region}/matpower_{region}_buses.csv
  data/working_files/{region}/transformer_pairs_{region}.csv
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path

import config
from pipeline_integrity import TRANSFORMER_COLUMNS

region                     = config.region
MIN_TRANSMISSION_VOLTAGE_V = config.MIN_TRANSMISSION_VOLTAGE_V

PROJ_ROOT = Path(__file__).parent.parent
WORK_DIR  = PROJ_ROOT / "data" / "working_files" / region
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Transformer electrical parameters (100 MVA system base)
# Source: quoted specification — 1.6 % impedance on 1000 MVA base, X/R = 80
_Z_BASE  = 0.016 * (100.0 / 1000.0)   # 0.0016 p.u. on 100 MVA base
_R_XFMR  = round(_Z_BASE / 80.0, 7)   # ≈ 0.00002 p.u.
_X_XFMR  = round(_Z_BASE - _R_XFMR, 6)  # ≈ 0.0016 p.u.
_RATE_XFMR = 1000.0                    # MVA thermal rating


# ============================================================
# FAST REVERSE GEOCODER  (offline → fast Nominatim fallback)
# ============================================================

try:
    import reverse_geocoder as _rg

    def _batch_city_lookup(lons, lats):
        """Batch reverse-geocode using offline GeoNames database (very fast)."""
        results = _rg.search(list(zip(lats, lons)), mode=2)
        return [r.get("name", "") for r in results]

    print("[Geocoder] Using fast offline reverse_geocoder (GeoNames)")

except ImportError:
    from geopy.geocoders import Nominatim as _Nominatim
    _nom = _Nominatim(user_agent=f"{region}_grid_builder", timeout=10)

    def _batch_city_lookup(lons, lats):
        """Sequential Nominatim reverse geocoding (slow; install reverse_geocoder for speed)."""
        import time
        names = []
        for lon, lat in zip(lons, lats):
            try:
                loc = _nom.reverse((lat, lon), language="en")
                if loc:
                    addr = loc.raw.get("address", {})
                    for key in ("city", "town", "village", "suburb",
                                "county", "state_district", "state"):
                        if addr.get(key):
                            names.append(addr[key])
                            break
                    else:
                        names.append("")
                else:
                    names.append("")
            except Exception:
                names.append("")
            time.sleep(1.0)   # Nominatim rate limit: 1 request/s
        return names

    print("[Geocoder] reverse_geocoder not found — using Nominatim (slow).")
    print("  Tip: pip install reverse_geocoder to speed this step up significantly.")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parse_voltage_levels(voltage_val) -> list[int]:
    """Return a sorted (descending) list of integer Volt values from an OSM tag."""
    if voltage_val is None:
        return []
    if isinstance(voltage_val, float) and np.isnan(voltage_val):
        return []
    levels = []
    for part in str(voltage_val).split(";"):
        try:
            levels.append(int(float(part.strip())))
        except ValueError:
            pass
    return sorted(set(levels), reverse=True)


def geom_to_polygon_wkt(geom) -> str | None:
    """Return WKT string if the geometry is a Polygon, else None."""
    if geom is not None and geom.geom_type == "Polygon":
        return geom.wkt
    return None


# ============================================================
# STEP 1 – LOAD DATA
# ============================================================
print("=" * 60)
print("STEP 1: Loading substation GeoJSON")
print("=" * 60)

data_sub = gpd.read_file(
    PROJ_ROOT / "data" / "OSM_extracts" / f"export_{region}_buses.geojson"
)
print(f"  Loaded {len(data_sub)} substation features")
print(f"  Geometry types: {data_sub.geometry.geom_type.value_counts().to_dict()}")


# ============================================================
# STEP 2 – FILTER DISTRIBUTION / LOW-VOLTAGE SUBSTATIONS
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Filtering distribution and low-voltage substations")
print("=" * 60)

DISTRIBUTION_TYPES = {"distribution", "minor_distribution"}
drop_distribution = []

for idx, row in data_sub.iterrows():
    sub_type    = row.get("substation", None)
    voltage_val = row.get("voltage", None)

    if (sub_type is not None and
            not (isinstance(sub_type, float) and np.isnan(sub_type)) and
            str(sub_type).strip().lower() in DISTRIBUTION_TYPES):
        drop_distribution.append(idx)
        continue

    v_levels = parse_voltage_levels(voltage_val)
    if v_levels and max(v_levels) < MIN_TRANSMISSION_VOLTAGE_V:
        drop_distribution.append(idx)

data_sub = data_sub.drop(index=drop_distribution).reset_index(drop=True)
print(f"  Removed {len(drop_distribution)} distribution substations. "
      f"{len(data_sub)} remaining.")


# ============================================================
# STEP 3 – COMPUTE CENTROIDS & POLYGON WKT
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Computing centroids and extracting polygon WKT")
print("=" * 60)

centroids            = data_sub.geometry.centroid
data_sub["longitude"] = centroids.x
data_sub["latitude"]  = centroids.y
data_sub["polygon_wkt"] = data_sub.geometry.apply(geom_to_polygon_wkt)

print(f"  Polygon features: {data_sub['polygon_wkt'].notna().sum()}  "
      f"  Point features: {data_sub['polygon_wkt'].isna().sum()}")


# ============================================================
# STEP 4 – DERIVE SUBSTATION BASE NAMES  (fast geocoding)
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Deriving substation names")
print("=" * 60)

# Priority: OSM 'name' → OSM 'ref' → reverse-geocoded city name → OSM-ID fallback

base_names = []
needs_geocoding_idx = []
needs_geocoding_lons = []
needs_geocoding_lats = []

for idx, row in data_sub.iterrows():
    for field in ("name", "ref"):
        val = row.get(field, None)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            base_names.append(str(val).strip())
            break
    else:
        base_names.append(None)   # mark for geocoding
        needs_geocoding_idx.append(idx)
        needs_geocoding_lons.append(float(row["longitude"]))
        needs_geocoding_lats.append(float(row["latitude"]))

print(f"  OSM-named: {sum(n is not None for n in base_names)}  "
      f"  Need geocoding: {len(needs_geocoding_idx)}")

# Batch reverse-geocode the unnamed ones
if needs_geocoding_idx:
    print(f"  Batch-geocoding {len(needs_geocoding_idx)} substations …")
    city_names = _batch_city_lookup(needs_geocoding_lons, needs_geocoding_lats)
    geo_iter = iter(city_names)
    for i, base in enumerate(base_names):
        if base is None:
            city = next(geo_iter)
            if city:
                base_names[i] = f"{city} Substation"
            else:
                osm_id = str(data_sub.iloc[i].get(
                    "id", data_sub.iloc[i].get("@id", f"idx{i}")
                )).replace("way/", "W").replace("node/", "N")
                base_names[i] = f"Substation_{osm_id}"

# Number geocoded names that appear more than once so each substation is unique.
# OSM name/ref tag names are already specific and are left unchanged.
from collections import Counter as _Counter
name_counts = _Counter(base_names)
name_seen   = {}
for i, name in enumerate(base_names):
    if name_counts[name] > 1:
        seen = name_seen.get(name, 0) + 1
        name_seen[name] = seen
        base_names[i] = f"{name} {seen}"


# ============================================================
# STEP 5 – BUILD BUS ROWS  (split compound-voltage substations)
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Building bus table (splitting compound-voltage substations)")
print("=" * 60)

bus_rows        = []   # final list of dicts
transformer_pairs = []  # (bus_i_high, bus_i_low) for intra-substation transformers
next_bus_i      = 1

for feat_idx, row in data_sub.iterrows():
    voltage_val = row.get("voltage", None)
    v_levels    = parse_voltage_levels(voltage_val)
    base_name   = base_names[feat_idx]

    # Drop any voltage levels that fall below the transmission threshold.
    # e.g. "161000;69000" with MIN=110000 → only keep [161000]
    v_levels = [v for v in v_levels if v >= MIN_TRANSMISSION_VOLTAGE_V]

    if len(v_levels) > 1:
        # Split into one bus per transmission voltage level; highest first
        split_ids = []
        for v in v_levels:
            kv = v // 1000
            name = f"{base_name}_{kv}kV"
            bus_rows.append({
                "bus_i":       next_bus_i,
                "type":        1,
                "Pd":          0.0,
                "Qd":          0.0,
                "Gs":          0.0,
                "Bs":          0.0,
                "area":        1,
                "Vm":          1.0,
                "Va":          0.0,
                "base_kV":     float(v),
                "Vmax":        1.1,
                "Vmin":        0.9,
                "name":        name,
                "voltage_tag": str(v),
                "osm_id":      str(row.get("id", row.get("@id", ""))),
                "longitude":   float(row["longitude"]),
                "latitude":    float(row["latitude"]),
                "coordinates": f"({row['longitude']}, {row['latitude']})",
                "polygon_wkt": row["polygon_wkt"],
            })
            split_ids.append(next_bus_i)
            next_bus_i += 1

        # Add transformer pairs between consecutive voltage levels
        for k in range(len(split_ids) - 1):
            transformer_pairs.append((split_ids[k], split_ids[k + 1]))

    else:
        # Single voltage (or unknown)
        kv = (v_levels[0] // 1000) if v_levels else None
        name = f"{base_name}_{kv}kV" if kv else base_name
        bus_rows.append({
            "bus_i":       next_bus_i,
            "type":        1,
            "Pd":          0.0,
            "Qd":          0.0,
            "Gs":          0.0,
            "Bs":          0.0,
            "area":        1,
            "Vm":          1.0,
            "Va":          0.0,
            "base_kV":     float(v_levels[0]) if v_levels else np.nan,
            "Vmax":        1.1,
            "Vmin":        0.9,
            "name":        name,
            "voltage_tag": str(v_levels[0]) if v_levels else "",
            "osm_id":      str(row.get("id", row.get("@id", ""))),
            "longitude":   float(row["longitude"]),
            "latitude":    float(row["latitude"]),
            "coordinates": f"({row['longitude']}, {row['latitude']})",
            "polygon_wkt": row["polygon_wkt"],
        })
        next_bus_i += 1

buses_out = pd.DataFrame(bus_rows)
n_split = len(buses_out) - len(data_sub)
print(f"  OSM substations: {len(data_sub)}  →  buses: {len(buses_out)}  "
      f"(+{n_split} from compound-voltage splits)")
print(f"  Transformer pairs generated: {len(transformer_pairs)}")


# ============================================================
# STEP 6 – SAVE OUTPUTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Saving outputs")
print("=" * 60)

buses_path = WORK_DIR / f"matpower_{region}_buses.csv"
buses_out.to_csv(buses_path, index=False)
print(f"  Saved {len(buses_out)} buses → {buses_path}")

# Transformer pairs CSV (read by create_synthetic_grid.py)
xfmr_rows = []
for hi, lo in transformer_pairs:
    xfmr_rows.append({
        "fbus":   hi,
        "tbus":   lo,
        "r":      _R_XFMR,
        "x":      _X_XFMR,
        "b":      0.0,
        "rateA":  _RATE_XFMR,
        "rateB":  _RATE_XFMR,
        "rateC":  _RATE_XFMR,
        "ratio":  0,
        "angle":  0,
        "status": 1,
        "angmin": -60,
        "angmax":  60,
        "fbus_name": buses_out.loc[buses_out["bus_i"] == hi, "name"].iloc[0]
                     if hi in buses_out["bus_i"].values else "",
        "tbus_name": buses_out.loc[buses_out["bus_i"] == lo, "name"].iloc[0]
                     if lo in buses_out["bus_i"].values else "",
        "length_km":   0.0,
        "voltage_V":   "",
        "osm_id":      "transformer",
    })

xfmr_df = pd.DataFrame(xfmr_rows, columns=TRANSFORMER_COLUMNS)
xfmr_path = WORK_DIR / f"transformer_pairs_{region}.csv"
xfmr_df.to_csv(xfmr_path, index=False)
print(f"  Saved {len(xfmr_df)} transformer pairs → {xfmr_path}")

print()
print(buses_out[["bus_i", "name", "base_kV", "longitude", "latitude"]].to_string(index=False))
print("\nDone.")
