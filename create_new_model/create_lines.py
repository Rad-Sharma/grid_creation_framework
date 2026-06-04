"""
create_lines.py

Processes OSM transmission line data (GeoJSON from Overpass Turbo) to produce
a MATPOWER-compatible branch (line) file for a given region.

Pipeline:
  1.  Load OSM line GeoJSON and MATPOWER bus CSV
        (bus CSV must be produced first by create_buses_new.py)
  2.  Remove fully-foreign lines (bounding box); keep cross-border lines
  3.  Remove misread / junk rows (no attributes + very short geometry)
  4.  Remove sub-transmission / distribution lines (voltage < threshold)
  5.  Remove OSM busbars (all vertices inside a single substation polygon)
  6.  Deduplicate lines with the same approximate start+end (OSM digitising
        duplicates with slightly different intermediate coordinates)
  7.  Preliminary bus assignment → identify substation junction coordinates
  8.  Junction-aware merge of consecutive OSM segments
        (will not merge two segments through a substation junction)
  9.  Final bus assignment: proximity + voltage matching, distances,
        cross-border reverse geocoding, voltage inference from bus
  10. Output MATPOWER branch CSV (topology; R/X/B/ratings to be added later)
        – includes substation names, distances, and removes same-bus loops
  11. Output island buses CSV (buses with no lines connected)
  12. Plot topology for visual sanity check

NOTE: Run create_buses_new.py before this script.  The enriched bus CSV it
produces (matpower_{region}_buses.csv) must contain 'voltage_tag' and
'polygon_wkt' columns for Steps 5 and 8 to work correctly.

Author: Richard Asiamah
"""

import math
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
from shapely import wkt as shapely_wkt
import numpy as np
from geopy.distance import geodesic


# ============================================================
# CONFIGURATION  –  edit these to change region or thresholds
# ============================================================

from pathlib import Path
import config

region                     = config.region
MIN_TRANSMISSION_VOLTAGE_V = config.MIN_TRANSMISSION_VOLTAGE_V

PROJ_ROOT = Path(__file__).parent.parent
WORK_DIR  = PROJ_ROOT / "data" / "working_files" / region
WORK_DIR.mkdir(parents=True, exist_ok=True)

# --------------- Geographic bounding box ---------------------
# Lines whose ANY coordinate falls outside this box are dropped.
# Set these to the true extreme points of your study region so that
# cross-border stubs (partial lines into Ivory Coast, Togo, Burkina Faso)
# are removed before further processing.
#
# Format: (longitude, latitude)  using WGS-84 decimal degrees.
# The defaults below tightly bound region's territory.
#
# To DISABLE the bounding box filter (e.g. when studying a multi-country
# region), set APPLY_BBOX_FILTER = False.

APPLY_BBOX_FILTER = True

NORTHMOST_LAT = config.REGION_BBOX[3]  # northernmost latitude  (upper edge of bbox)
SOUTHMOST_LAT = config.REGION_BBOX[1]  # southernmost latitude  (lower edge of bbox)
WESTMOST_LON  = config.REGION_BBOX[0]  # westernmost longitude  (left  edge of bbox)
EASTMOST_LON  = config.REGION_BBOX[2]  # easternmost longitude  (right edge of bbox)

# --------------- Bus-assignment distance threshold -----------
# Geodesic distance (km) within which a line endpoint must fall to be
# assigned a bus.  Endpoints farther than this get bus = None (cross-border
# or no matching substation in the model).
#
# Physical rationale: OSM tower locations are typically within a few
# hundred metres of the real substation centroid; imagery offsets add
# further error.  5 km captures legitimate connections while avoiding
# false matches to distant substations 
# (Typical inter-substation spacing is typically 30–100 km on the 161/330 kV backbone).
#
# Tune upward (~10 km) if many endpoints go unmatched;
# tune downward (~2 km) if you see too many false matches.
BUS_ASSIGN_THRESHOLD_KM = 5.0



# --------------- Segment merge tolerances --------------------
# Two OSM segment endpoints are treated as "the same node" if they are
# within MERGE_TOLERANCE_DEG degrees of each other.
#
# We use a larger tolerance (0.005° ≈ 550 m) because different OSM
# contributors sometimes place the shared node of two adjacent ways at
# slightly different coordinates.  A naïve 0.001° tolerance misses many
# of these.
#
# Why this is now safe to increase:
#   The merge step runs AFTER a preliminary bus-assignment pass (Step 6)
#   that identifies which junction points are at substations.  The merge
#   algorithm refuses to merge two segments whose shared junction is a
#   known substation, so raising the tolerance can no longer accidentally
#   join AB + BC into AC when B is a substation.
#
# Tune higher (e.g. 0.01°) if chains of short OSM ways still go unmerged;
# tune lower if unrelated nearby segments are being merged by mistake.
MERGE_TOLERANCE_DEG = 0.01   # 0.01° ≈ 1.1 km

# --------------- Junction detection threshold ----------------
# Used ONLY in Step 7 (preliminary pass) to decide whether a line endpoint
# is truly at a substation.  This is kept intentionally tight so that
# mid-line OSM split points are never falsely treated as substation junctions.
#
# Two-tier check:
#   1. Endpoint falls INSIDE a substation polygon → definite junction
#   2. Endpoint within JUNCTION_DETECTION_THRESHOLD_KM of the CENTROID of
#      ANY substation (polygon or point-node) → junction.
#      For polygon substations the centroid is computed from the polygon
#      geometry; for point-only substations the OSM node coordinate is used.
#
# Do NOT increase this to match BUS_ASSIGN_THRESHOLD_KM.  With a 5 km
# radius almost every mid-line split point in southern Ghana would be
# flagged, preventing legitimate merges.
#
# 0.5 km keeps the check tight: a real line endpoint at a substation should
# be within a few hundred metres of the substation centroid.  Mid-line OSM
# split points are typically kilometres away from the nearest substation
# centroid, so they are correctly left un-flagged.
JUNCTION_DETECTION_THRESHOLD_KM = 0.5  # 0.5 km

# --------------- Trajectory extrapolation parameters --------
# Used as a secondary assignment pass when proximity-only matching fails.
# The line's approach bearing is projected forward and nearby buses are scored
# by their cross-track deviation from that projected ray.
#
# TRAJECTORY_N_TAIL: tail/head coordinates averaged for bearing estimation.
#   5 points span ~0.5–3 km of tower spacing, giving a stable bearing while
#   capturing local curvature near the endpoint.
#   [cf. Matisziw & Murray 2009, "Modeling network interdependencies"]
#
# TRAJECTORY_SEARCH_RADIUS_KM: candidate bus search radius.
#   10 km covers the typical GPS offset and right-of-way meander for HV lines
#   globally. Tune upward for sparse rural networks where substations may be
#   further from the line centerline in OSM.
#
# PROJECTION_KM: length of the forward bearing ray (150 km covers the longest
#   expected single HV span in most national grids; typically 50–200 km).
#
# CROSS_TRACK_THRESHOLD_KM: perpendicular tolerance around the projected ray.
#   5 km accounts for OSM GPS uncertainty (~1–5 km for infrastructure datasets),
#   right-of-way meander on long rural spans (~2 km), and substation node offset
#   from the line centreline.  Buses > 5 km cross-track are rejected.
TRAJECTORY_N_TAIL           = 5      # tail coordinates for bearing estimation
TRAJECTORY_SEARCH_RADIUS_KM = 10.0   # candidate search radius (km)
PROJECTION_KM               = 150.0  # projected ray length (km)
CROSS_TRACK_THRESHOLD_KM    = 5.0    # max cross-track deviation to auto-accept

# Attribute columns that must agree (when both are non-null) before two
# adjacent OSM segments are merged into one.
MERGE_MATCH_COLS = ["voltage", "operator", "cables", "circuits", "frequency"]

# Informational columns used to detect "junk" rows in Step 3.
INFO_COLS = ["cables", "voltage", "circuits", "frequency", "operator", "source"]

# --------------- Deduplication coordinate-count threshold ----
# Two lines that share the same approximate start and end are treated as OSM
# digitising duplicates only when EVERY line in the group has at most this
# many coordinate points.  Genuine parallel circuits on shared towers produce
# many intermediate tower coordinates; accidental duplicates are typically
# simple polylines with very few points.
#
# Default = 5.  Set higher if your data has genuinely short parallel segments
# that should still be deduplicated.
MAX_DEDUP_COORDS = 5


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def all_info_null(row: pd.Series) -> bool:
    """Return True when every informational column is null or None."""
    for col in INFO_COLS:
        val = row.get(col, None)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            return False
    return True


def coords_match(c1: tuple, c2: tuple,
                 tol: float = MERGE_TOLERANCE_DEG) -> bool:
    """True if two (lon, lat) tuples are within *tol* degrees of each other."""
    return abs(c1[0] - c2[0]) < tol and abs(c1[1] - c2[1]) < tol


def attrs_compatible(row_a: pd.Series, row_b: pd.Series) -> bool:
    """
    True if the two rows can be merged.
    Rule: for each checked column, values must be equal when BOTH are
    non-null.  If either side is null, there is no conflict → OK to merge.
    """
    for col in MERGE_MATCH_COLS:
        a = row_a.get(col, None)
        b = row_b.get(col, None)
        a_null = a is None or (isinstance(a, float) and np.isnan(a))
        b_null = b is None or (isinstance(b, float) and np.isnan(b))
        if not a_null and not b_null and str(a).strip() != str(b).strip():
            return False
    return True


def run_merge_pass(gdf: gpd.GeoDataFrame,
                   substation_junctions: set = None):
    """
    One pass over all segments looking for adjacent pairs that can be chained.

    OSM ways have no canonical direction, so all three orientations are tried:

      Case 1  A[-1] ≈ B[0]   →  A + B            (B in natural direction)
      Case 2  A[-1] ≈ B[-1]  →  A + rev(B)       (B digitised backwards)
      Case 3  A[0]  ≈ B[0]   →  rev(A) + B       (A digitised backwards)

    Case 4 (A[0] ≈ B[-1], i.e. B + A) is case 1 from B's perspective and is
    handled automatically when B is the outer segment — no extra logic needed.

    substation_junctions guards the merge junction: two segments sharing a
    known substation coordinate are NOT merged (they are separate physical
    lines that happen to meet at that substation).

    Returns (new_gdf, any_merges_happened).
    """
    positions = list(range(len(gdf)))
    start_lookup: dict = {}   # start grid-key → [positions]
    end_lookup:   dict = {}   # end   grid-key → [positions]

    for pos in positions:
        coords = list(gdf.iloc[pos].geometry.coords)
        sk = (round(coords[0][0]  / MERGE_TOLERANCE_DEG),
              round(coords[0][1]  / MERGE_TOLERANCE_DEG))
        ek = (round(coords[-1][0] / MERGE_TOLERANCE_DEG),
              round(coords[-1][1] / MERGE_TOLERANCE_DEG))
        start_lookup.setdefault(sk, []).append(pos)
        end_lookup.setdefault(ek, []).append(pos)

    to_drop_pos: set = set()
    new_rows:    list = []
    merged_any = False

    for pos_a in positions:
        if pos_a in to_drop_pos:
            continue
        row_a    = gdf.iloc[pos_a]
        a_coords = list(row_a.geometry.coords)
        a_start  = a_coords[0]
        a_end    = a_coords[-1]
        a_start_key = (round(a_start[0] / MERGE_TOLERANCE_DEG),
                       round(a_start[1] / MERGE_TOLERANCE_DEG))
        a_end_key   = (round(a_end[0]   / MERGE_TOLERANCE_DEG),
                       round(a_end[1]   / MERGE_TOLERANCE_DEG))

        # Build candidate list: (pos_b, reverse_a, reverse_b)
        candidates = []

        # Case 1: B[0] ≈ A[-1]  — standard forward chain
        if not (substation_junctions and a_end_key in substation_junctions):
            for p in start_lookup.get(a_end_key, []):
                candidates.append((p, False, False))

        # Case 2: B[-1] ≈ A[-1]  — reverse B before merging
        if not (substation_junctions and a_end_key in substation_junctions):
            for p in end_lookup.get(a_end_key, []):
                candidates.append((p, False, True))

        # Case 3: B[0] ≈ A[0]  — reverse A before merging
        if not (substation_junctions and a_start_key in substation_junctions):
            for p in start_lookup.get(a_start_key, []):
                candidates.append((p, True, False))

        for pos_b, reverse_a, reverse_b in candidates:
            if pos_b == pos_a or pos_b in to_drop_pos:
                continue
            row_b = gdf.iloc[pos_b]

            a_used = list(reversed(a_coords)) if reverse_a else a_coords
            b_used = (list(reversed(list(row_b.geometry.coords)))
                      if reverse_b else list(row_b.geometry.coords))

            # Hash-collision guard: verify the junction is actually close
            if not coords_match(a_used[-1], b_used[0]):
                continue
            if not attrs_compatible(row_a, row_b):
                continue

            # Stitch: drop the duplicated junction node
            merged_coords = a_used + b_used[1:]
            merged_geom   = LineString(merged_coords)

            merged_row = row_a.copy()
            merged_row["geometry"] = merged_geom
            for col in MERGE_MATCH_COLS:
                a_val = merged_row.get(col, None)
                b_val = row_b.get(col, None)
                a_null = a_val is None or (isinstance(a_val, float)
                                           and np.isnan(a_val))
                b_null = b_val is None or (isinstance(b_val, float)
                                           and np.isnan(b_val))
                if a_null and not b_null:
                    merged_row[col] = b_val

            merged_row["id"] = f"{row_a['id']}+{row_b['id']}"
            new_rows.append(merged_row)
            to_drop_pos.add(pos_a)
            to_drop_pos.add(pos_b)
            merged_any = True
            break   # each segment merges with at most one partner per pass

    if merged_any:
        keep_mask = [i not in to_drop_pos for i in range(len(gdf))]
        gdf = gdf.iloc[keep_mask].copy()
        new_gdf = gpd.GeoDataFrame(new_rows, crs=gdf.crs)
        gdf = pd.concat([gdf, new_gdf], ignore_index=True)

    return gdf, merged_any


def parse_voltage_tag(voltage_val) -> list[int]:
    """
    Parse an OSM voltage tag into a list of integer Volt values.

    OSM transformer nodes can carry semicolon-separated voltages, e.g.
    '161000;330000'.  Returns an empty list for missing / unparseable values.
    """
    if voltage_val is None:
        return []
    if isinstance(voltage_val, float) and np.isnan(voltage_val):
        return []
    parts = str(voltage_val).split(";")
    result = []
    for part in parts:
        try:
            result.append(int(float(part.strip())))
        except ValueError:
            pass
    return result


def find_nearest_bus(endpoint_lonlat: tuple,
                     line_voltage_tag,
                     all_bus_coords: list,
                     all_bus_voltages: np.ndarray,
                     threshold_km: float = BUS_ASSIGN_THRESHOLD_KM,
                     exclude_pos: int | None = None,
                     ) -> tuple:
    """
    Return (pos, dist_km) for the nearest compatible bus, or (None, None).

    pos      – 0-based positional index into all_bus_coords / all_bus_voltages
    dist_km  – geodesic distance in km to that bus (confidence metric)

    Compatibility rules
    -------------------
    1. Geodesic distance ≤ threshold_km.
    2. Voltage match: if BOTH the line voltage and the bus voltage are known,
       the two voltage lists must share at least one level.  This handles
       transformer buses (e.g. a bus tagged '161000;330000' matches a 161 kV
       or 330 kV line).  If either voltage is unknown, only proximity matters.

    Why None for unmatched endpoints?
    ----------------------------------
    If no qualifying bus is found within threshold_km the endpoint is either
    cross-border or belongs to a substation not yet in the dataset.  The
    caller handles these cases explicitly via reverse geocoding.
    """
    line_v_list = parse_voltage_tag(line_voltage_tag)

    best_pos      = None
    best_dist_km  = float("inf")

    for k, (bcoord, bv) in enumerate(zip(all_bus_coords, all_bus_voltages)):
        if k == exclude_pos:
            continue   # don't assign fbus and tbus to the same bus

        dist_km = geodesic(
            (endpoint_lonlat[1], endpoint_lonlat[0]),
            (bcoord[1],          bcoord[0])
        ).km

        if dist_km > threshold_km:
            continue

        bv_list      = parse_voltage_tag(bv)
        bv_known     = len(bv_list) > 0
        line_v_known = len(line_v_list) > 0

        if bv_known and line_v_known:
            if not set(line_v_list).intersection(bv_list):
                continue

        if dist_km < best_dist_km:
            best_dist_km = dist_km
            best_pos     = k

    if best_pos is None:
        return None, None
    return best_pos, round(best_dist_km, 4)


def is_outside_bbox(lonlat: tuple) -> bool:
    """True if the coordinate lies outside the study-region bounding box."""
    lon, lat = lonlat
    return not (SOUTHMOST_LAT <= lat <= NORTHMOST_LAT and
                WESTMOST_LON  <= lon <= EASTMOST_LON)


# ── Trajectory extrapolation helpers ──────────────────────────────────────

def _compute_bearing(lon1: float, lat1: float,
                     lon2: float, lat2: float) -> float:
    """Forward azimuth (0–360°) from (lon1,lat1) to (lon2,lat2)."""
    dlon = math.radians(lon2 - lon1)
    lr1, lr2 = math.radians(lat1), math.radians(lat2)
    x = math.sin(dlon) * math.cos(lr2)
    y = (math.cos(lr1) * math.sin(lr2)
         - math.sin(lr1) * math.cos(lr2) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _project_point(lon: float, lat: float,
                   bearing_deg: float, dist_km: float) -> tuple:
    """(lon, lat) of the point dist_km ahead along bearing_deg."""
    R = 6371.0
    d = dist_km / R
    b = math.radians(bearing_deg)
    la1, lo1 = math.radians(lat), math.radians(lon)
    la2 = math.asin(math.sin(la1) * math.cos(d)
                    + math.cos(la1) * math.sin(d) * math.cos(b))
    lo2 = lo1 + math.atan2(
        math.sin(b) * math.sin(d) * math.cos(la1),
        math.cos(d) - math.sin(la1) * math.sin(la2))
    return math.degrees(lo2), math.degrees(la2)


def _cross_track_km(endpoint: tuple, projected: tuple, bus: tuple) -> float:
    """Perpendicular distance (km) from bus to the ray endpoint→projected."""
    R = 6371.0
    la1, lo1 = math.radians(endpoint[1]),  math.radians(endpoint[0])
    la2, lo2 = math.radians(projected[1]), math.radians(projected[0])
    la3, lo3 = math.radians(bus[1]),       math.radians(bus[0])
    cos_d13 = max(-1.0, min(1.0,
        math.sin(la1) * math.sin(la3)
        + math.cos(la1) * math.cos(la3) * math.cos(lo3 - lo1)))
    d13 = math.acos(cos_d13)
    theta12 = math.atan2(
        math.sin(lo2 - lo1) * math.cos(la2),
        math.cos(la1) * math.sin(la2)
        - math.sin(la1) * math.cos(la2) * math.cos(lo2 - lo1))
    theta13 = math.atan2(
        math.sin(lo3 - lo1) * math.cos(la3),
        math.cos(la1) * math.sin(la3)
        - math.sin(la1) * math.cos(la3) * math.cos(lo3 - lo1))
    return abs(math.asin(
        math.sin(d13) * math.sin(theta13 - theta12))) * R


def _is_ahead(endpoint: tuple, projected: tuple, bus: tuple) -> bool:
    """True if bus lies in the general forward direction endpoint→projected."""
    return ((projected[0] - endpoint[0]) * (bus[0] - endpoint[0])
            + (projected[1] - endpoint[1]) * (bus[1] - endpoint[1])) > 0


def _trajectory_match(
    coords: list,
    which: str,
    line_voltage_tag,
    all_bus_coords: list,
    all_bus_voltages: np.ndarray,
    exclude_pos: int | None = None,
) -> tuple:
    """
    Secondary bus-assignment pass using trajectory extrapolation.

    Estimates the line's approach bearing from its tail/head coordinates,
    projects that bearing forward PROJECTION_KM, then returns the nearest
    voltage-compatible bus that is:
      • within TRAJECTORY_SEARCH_RADIUS_KM of the endpoint, AND
      • in the forward direction (ahead of the endpoint), AND
      • within CROSS_TRACK_THRESHOLD_KM of the projected ray.

    Returns (pos, dist_km, cross_track_km) or (None, None, None).
    """
    if len(coords) < 2:
        return None, None, None

    n = min(TRAJECTORY_N_TAIL, len(coords))
    if which == "fbus":
        tail    = coords[:n]
        ep      = tail[0]
        bearing = _compute_bearing(tail[1][0], tail[1][1],
                                   tail[0][0], tail[0][1])
    else:
        tail    = coords[-n:]
        ep      = tail[-1]
        bearing = _compute_bearing(tail[-2][0], tail[-2][1],
                                   tail[-1][0], tail[-1][1])

    proj        = _project_point(ep[0], ep[1], bearing, PROJECTION_KM)
    line_v_list = parse_voltage_tag(line_voltage_tag)

    best = None
    for k, (bcoord, bv) in enumerate(zip(all_bus_coords, all_bus_voltages)):
        if k == exclude_pos:
            continue   # don't assign fbus and tbus to the same bus
        dist_km = geodesic((ep[1], ep[0]), (bcoord[1], bcoord[0])).km
        if dist_km > TRAJECTORY_SEARCH_RADIUS_KM:
            continue
        bv_list = parse_voltage_tag(bv)
        if bv_list and line_v_list and not set(line_v_list).intersection(bv_list):
            continue
        bc = (float(bcoord[0]), float(bcoord[1]))
        if not _is_ahead(ep, proj, bc):
            continue
        ct = _cross_track_km(ep, proj, bc)
        if ct > CROSS_TRACK_THRESHOLD_KM:
            continue
        if best is None or ct < best[0]:
            best = (ct, dist_km, k)

    if best is None:
        return None, None, None
    ct, dist_km, pos = best
    return pos, round(dist_km, 4), round(ct, 4)


def _coord_key(lonlat: tuple) -> tuple:
    """Rounded coordinate key, consistent with the merge lookup grid."""
    return (round(lonlat[0] / MERGE_TOLERANCE_DEG),
            round(lonlat[1] / MERGE_TOLERANCE_DEG))


def deduplicate_lines(gdf: gpd.GeoDataFrame) -> tuple:
    """
    Remove lines that trace the same route (same approximate start+end).

    OSM sometimes has two contributors map the same physical span with slightly
    different intermediate coordinates.  These share the same start and end
    nodes but differ in their middle vertices.

    To avoid removing genuine parallel circuits (which share the same terminal
    substations but have many intermediate tower coordinates), deduplication
    is only applied when EVERY line in the group has at most MAX_DEDUP_COORDS
    coordinate points.  Long, detailed geometries are left untouched.

    For each qualifying group we keep the segment with the most coordinate
    points (most detailed geometry), fill any null attributes from the others,
    and concatenate their OSM IDs with '|' for traceability.

    Returns (deduplicated_gdf, n_removed).
    """
    endpoint_pairs: dict = {}
    for pos in range(len(gdf)):
        coords = list(gdf.iloc[pos].geometry.coords)
        sk = _coord_key(coords[0])
        ek = _coord_key(coords[-1])
        # Canonical key treats (A→B) and (B→A) as the same route
        canonical_key = (sk, ek) if sk <= ek else (ek, sk)
        endpoint_pairs.setdefault(canonical_key, []).append(pos)

    to_drop: set = set()
    new_rows: list = []
    n_deduped = 0

    for canonical_key, pos_list in endpoint_pairs.items():
        if len(pos_list) < 2:
            continue

        # Skip groups where any segment exceeds MAX_DEDUP_COORDS points:
        # those are likely genuine parallel circuits, not digitising duplicates.
        max_n = max(len(list(gdf.iloc[p].geometry.coords)) for p in pos_list)
        if max_n > MAX_DEDUP_COORDS:
            continue

        # Sort: most coordinate points first, then most non-null attributes
        def _sort_key(p: int) -> tuple:
            row = gdf.iloc[p]
            n_coords = len(list(row.geometry.coords))
            n_attrs = sum(
                1 for col in MERGE_MATCH_COLS
                if row.get(col) is not None
                and not (isinstance(row.get(col), float)
                         and np.isnan(row.get(col)))
            )
            return (-n_coords, -n_attrs)

        pos_list_sorted = sorted(pos_list, key=_sort_key)
        best_pos = pos_list_sorted[0]
        best_row = gdf.iloc[best_pos].copy()
        best_n = len(list(gdf.iloc[best_pos].geometry.coords))

        for other_pos in pos_list_sorted[1:]:
            other_row = gdf.iloc[other_pos]
            # Fill null attributes from duplicate
            for col in MERGE_MATCH_COLS:
                val = best_row.get(col, None)
                other_val = other_row.get(col, None)
                is_null = val is None or (isinstance(val, float)
                                          and np.isnan(val))
                other_not_null = (other_val is not None and
                                  not (isinstance(other_val, float)
                                       and np.isnan(other_val)))
                if is_null and other_not_null:
                    best_row[col] = other_val
            print(f"  [DEDUP] kept  id={gdf.iloc[best_pos]['id']} "
                  f"({best_n} pts)  "
                  f"dropped id={other_row['id']} "
                  f"({len(list(other_row.geometry.coords))} pts)")
            best_row["id"] = f"{best_row['id']}|{other_row['id']}"

        new_rows.append(best_row)
        for pos in pos_list:
            to_drop.add(pos)
        n_deduped += len(pos_list) - 1

    if to_drop:
        keep_mask = [i not in to_drop for i in range(len(gdf))]
        gdf_kept = gdf.iloc[keep_mask].copy()
        if new_rows:
            new_gdf = gpd.GeoDataFrame(new_rows, crs=gdf.crs)
            gdf = pd.concat([gdf_kept, new_gdf], ignore_index=True)
        else:
            gdf = gdf_kept.reset_index(drop=True)

    return gdf.reset_index(drop=True), n_deduped


# ============================================================
# STEP 1 – LOAD DATA
# ============================================================
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

data_lines = gpd.read_file(PROJ_ROOT / "data" / "OSM_extracts" / f"export_{region}_lines.geojson")
data_buses = pd.read_csv(WORK_DIR / f"matpower_{region}_buses.csv")

# Parse bus coordinates: stored as the string "(lon, lat)" in the CSV.
bus_coords = [np.array(eval(c)) for c in data_buses["coordinates"]]

# Use the compound voltage_tag column (produced by create_buses_new.py) for
# voltage matching.  This correctly handles transformer substations tagged
# '330000;161000': a 161 kV line will match such a bus even though its
# base_kV (primary voltage) is 330000.
# Fall back to base_kV if voltage_tag is not present (older bus CSV).
if "voltage_tag" in data_buses.columns:
    bus_voltages = data_buses["voltage_tag"].values
else:
    bus_voltages = data_buses["base_kV"].values

# Load substation names for annotating the MATPOWER branch output.
# Falls back gracefully if the column is absent.
bus_names = (data_buses["name"].values
             if "name" in data_buses.columns
             else np.array([None] * len(data_buses)))

# Load substation polygon WKT strings for the busbar filter (Step 5).
# Reconstruct Shapely Polygon objects; keep None for point-only features.
if "polygon_wkt" in data_buses.columns:
    bus_polygons = []
    for wkt_str in data_buses["polygon_wkt"]:
        if wkt_str and not (isinstance(wkt_str, float) and np.isnan(wkt_str)):
            bus_polygons.append(shapely_wkt.loads(wkt_str))
        else:
            bus_polygons.append(None)
else:
    bus_polygons = [None] * len(data_buses)

print(f"  Loaded {len(data_lines)} OSM line segments")
print(f"  Loaded {len(data_buses)} buses")
print(f"  Substation polygons available: "
      f"{sum(1 for p in bus_polygons if p is not None)}")


# ============================================================
# STEP 2 – GEOGRAPHIC BOUNDING BOX FILTER
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Geographic bounding box filter")
print("=" * 60)

if APPLY_BBOX_FILTER:
    print(f"  Bounding box: lon [{WESTMOST_LON}, {EASTMOST_LON}]  "
          f"lat [{SOUTHMOST_LAT}, {NORTHMOST_LAT}]")

    # Drop lines where EVERY coordinate is outside the bounding box.
    # These are completely foreign segments with no connection to the
    # study region at all (e.g. an OSM artefact from a neighbouring country).
    #
    # Lines where SOME coordinates are outside but at least ONE is inside
    # are kept as cross-border interconnects.  Their out-of-region endpoint
    # will receive a reverse-geocoded location label in Step 7 instead of a
    # bus number.
    drop_bbox = []
    for idx, row in data_lines.iterrows():
        coords = list(row.geometry.coords)
        all_outside = all(is_outside_bbox(c) for c in coords)
        if all_outside:
            drop_bbox.append(idx)

    data_lines = data_lines.drop(index=drop_bbox).reset_index(drop=True)
    n_crossborder = sum(
        1 for _, row in data_lines.iterrows()
        if any(is_outside_bbox(c) for c in row.geometry.coords)
    )
    print(f"  Removed {len(drop_bbox)} fully-foreign lines. "
          f"{len(data_lines)} lines remaining "
          f"({n_crossborder} cross-border lines retained).")
else:
    print("  Bounding box filter is DISABLED (APPLY_BBOX_FILTER = False).")


# ============================================================
# STEP 3 – REMOVE MISREAD / JUNK ROWS
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Removing junk/misread rows")
print("=" * 60)

# A row is junk if BOTH conditions hold:
#   (a) ALL informational columns are null / None
#   (b) The geometry has ≤ 2 coordinate points
#
# Rationale: a real transmission line should have at least ONE known
# attribute (voltage, operator, etc.) OR a polyline long enough to be
# unambiguous.  A 2-point segment with zero attributes is almost certainly
# a digitising error or OSM placeholder.

drop_junk = []
for idx, row in data_lines.iterrows():
    coords = list(row.geometry.coords)
    if all_info_null(row) and len(coords) <= 3:
        drop_junk.append(idx)
        print(f"  [DROP] idx={idx}  id={row['id']}  "
              f"({len(coords)} pts, all attributes null)")

data_lines = data_lines.drop(index=drop_junk).reset_index(drop=True)
print(f"  Removed {len(drop_junk)} junk rows. "
      f"{len(data_lines)} lines remaining.")


# ============================================================
# STEP 4 – VOLTAGE FILTER  (remove sub-transmission lines)
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Voltage filter")
print("=" * 60)

# Drop lines whose voltage tag is known and below MIN_TRANSMISSION_VOLTAGE_V.
# Lines with no voltage tag are retained because we cannot confirm they are
# sub-transmission; they may simply lack OSM attribution.

drop_voltage = []
for idx, row in data_lines.iterrows():
    v_levels = parse_voltage_tag(row.get("voltage", None))
    if len(v_levels) == 0:
        continue                        # unknown voltage → keep
    if max(v_levels) < MIN_TRANSMISSION_VOLTAGE_V:
        drop_voltage.append(idx)
        print(f"  [DROP] idx={idx}  id={row['id']}  "
              f"voltage={row.get('voltage')}  "
              f"(below {MIN_TRANSMISSION_VOLTAGE_V} V threshold)")

data_lines = data_lines.drop(index=drop_voltage).reset_index(drop=True)
print(f"  Removed {len(drop_voltage)} low-voltage lines. "
      f"{len(data_lines)} lines remaining.")


# ============================================================
# STEP 5 – BUSBAR FILTER  (remove lines inside substation polygons)
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Busbar / internal-substation line filter")
print("=" * 60)

# OSM sometimes maps substation busbars (internal connections between
# circuit-breakers and transformers) as power=line ways.  These are NOT
# transmission lines and must be removed.
#
# Detection rule: a line is an internal busbar if ALL of its vertices lie
# within the boundary of a SINGLE substation polygon.  We use covers()
# rather than contains() so that vertices exactly on the polygon edge
# (a common OSM digitising artefact) are still correctly caught.
#
# A line whose endpoints span two different polygons is a real
# inter-substation transmission line and is kept.

# Filter to non-None polygons only for speed
valid_polygons = [p for p in bus_polygons if p is not None]

def all_coords_in_polygon(coords: list, polygon) -> bool:
    """True if every coordinate of the line lies within or on the polygon."""
    return all(polygon.covers(Point(c)) for c in coords)

drop_busbar = []
for idx, row in data_lines.iterrows():
    coords = list(row.geometry.coords)
    for poly in valid_polygons:
        if all_coords_in_polygon(coords, poly):
            drop_busbar.append(idx)
            print(f"  [DROP busbar] idx={idx}  id={row['id']}  "
                  f"({len(coords)} pts entirely inside substation polygon)")
            break   # no need to check more polygons once matched

data_lines = data_lines.drop(index=drop_busbar).reset_index(drop=True)
print(f"  Removed {len(drop_busbar)} busbar/internal lines. "
      f"{len(data_lines)} lines remaining.")


# ============================================================
# STEP 6 – DEDUPLICATE LINES  (same start+end, different middle points)
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Deduplicating lines with same approximate start and end")
print("=" * 60)

# OSM contributors sometimes digitise the same physical transmission span
# independently, producing two ways with identical start/end coordinates
# but slightly different intermediate points.  These are not consecutive
# segments (end-to-start chain) so they escape the merge step — they need
# a separate deduplication pass based on canonical (start, end) keys.
#
# We keep the geometry with the most coordinate points (more detailed) and
# fill any missing attribute values from the duplicate(s).  All OSM IDs
# are joined with '|' for traceability.

data_lines, n_deduped = deduplicate_lines(data_lines)
print(f"  Removed {n_deduped} duplicate line(s). "
      f"{len(data_lines)} lines remaining.")


# ============================================================
# STEP 7 – PRELIMINARY BUS ASSIGNMENT  (identify substation junctions)
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Preliminary bus assignment – identifying substation junctions")
print("=" * 60)

# A line endpoint is a true substation junction if — and only if — it is
# physically inside a substation compound OR within JUNCTION_DETECTION_THRESHOLD_KM
# (0.5 km) of the centroid of any substation (polygon or point-node).
# We deliberately avoid the broad BUS_ASSIGN_THRESHOLD_KM here because using
# a 5 km radius would flag almost every mid-line OSM split point in southern
# Ghana as a "substation", blocking legitimate merges.
#
# Two-tier detection:
#   Tier 1 – Polygon containment (most reliable)
#       If the endpoint falls inside any substation polygon it is definitively
#       at that substation's physical location.
#   Tier 2 – Proximity to substation centroid (all substations)
#       For every substation — whether mapped as a polygon or a point-node —
#       if the endpoint is within JUNCTION_DETECTION_THRESHOLD_KM (0.5 km) of
#       the substation centroid it is treated as a junction.  The centroid of
#       a polygon substation is computed from the polygon geometry; for
#       point-only substations the OSM node coordinate is used directly.
#
# Mid-line split points that are merely near (but not inside) a substation
# are correctly left un-flagged, allowing the merge to proceed through them.

substation_junctions: set[tuple] = set()

# bus_coords already stores the centroid of every substation (computed by
# create_buses_new.py and saved as the 'coordinates' column of the bus CSV),
# so no further centroid computation is needed here.

for idx, row in data_lines.iterrows():
    coords = list(row.geometry.coords)

    for endpoint_ll in (coords[0], coords[-1]):
        if is_outside_bbox(endpoint_ll):
            continue   # cross-border endpoints never match a domestic bus

        pt = Point(endpoint_ll)
        is_junction = False

        # Tier 1: inside any substation polygon?
        for poly in valid_polygons:
            if poly.covers(pt):
                is_junction = True
                break

        # Tier 2: within JUNCTION_DETECTION_THRESHOLD_KM of ANY substation
        # centroid (polygon or point-node)?  bus_coords holds the pre-computed
        # centroid for every bus, so this check covers both types uniformly.
        if not is_junction:
            for bcoord in bus_coords:
                d_km = geodesic(
                    (endpoint_ll[1], endpoint_ll[0]),
                    (bcoord[1],      bcoord[0])
                ).km
                if d_km <= JUNCTION_DETECTION_THRESHOLD_KM:
                    is_junction = True
                    break

        if is_junction:
            substation_junctions.add(_coord_key(endpoint_ll))

print(f"  Identified {len(substation_junctions)} substation junction "
      f"coordinates (polygon containment OR ≤{JUNCTION_DETECTION_THRESHOLD_KM} km "
      f"from any substation centroid).")
print(f"  The merge step will not cross these junctions.")


# ============================================================
# STEP 8 – MERGE CONSECUTIVE OSM SEGMENTS  (junction-aware)
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Merging consecutive OSM line segments")
print("=" * 60)

# OSM sometimes splits one physical transmission line into multiple
# consecutive <way> elements.  Two segments A → B are merged when:
#   1. The last coordinate of A ≈ the first coordinate of B  (end-to-start,
#      within MERGE_TOLERANCE_DEG)
#   2. Their attribute values agree wherever both are known
#   3. The shared junction is NOT a known substation (from Step 6)
#
# Rule 3 prevents merging AB + BC into AC when B is a substation: those
# are two separate lines that happen to share a node at B, not one line.
#
# We repeat merge passes until no further merges are possible, so chains
# A → M₁ → M₂ → B (where M₁, M₂ are non-substation mid-points) are
# reduced to the single segment A → B.

pass_count = 0
while True:
    pass_count += 1
    data_lines, did_merge = run_merge_pass(data_lines, substation_junctions)
    if did_merge:
        print(f"  Pass {pass_count}: merged some segments "
              f"({len(data_lines)} lines remaining)")
    else:
        print(f"  Pass {pass_count}: no more merges possible → stopping")
        break

data_lines = data_lines.reset_index(drop=True)
print(f"  Final segment count after merging: {len(data_lines)}")

# Save the cleaned / merged line geometry for inspection
data_lines.to_csv(WORK_DIR / f"processed_{region}_lines.csv", index=False)
print(f"  Saved processed lines → data/working_files/processed_{region}_lines.csv")


# ============================================================
# STEP 9 – ASSIGN FROM / TO BUSES  (final, on merged segments)
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: Assigning from/to buses")
print("=" * 60)

# For each line:
#   • The FIRST coordinate is the "from" endpoint.
#   • The LAST  coordinate is the "to"   endpoint.
#
# For each endpoint we search for the nearest bus that satisfies:
#   1. Distance ≤ BUS_ASSIGN_THRESHOLD_KM
#   2. Voltage compatibility: the bus voltage_tag and line voltage must share
#      at least one level when both are known.  This handles transformer buses
#      (e.g. a bus tagged '330000;161000' matches a 161 kV or 330 kV line).
#
# If no qualifying bus exists within the threshold, bus = None.
# Expected for cross-border interconnects or substations not yet in dataset.

f_buses: list         = []   # MATPOWER bus_i numbers (1-indexed), or None
t_buses: list         = []
f_bus_names: list     = []   # substation name or None
t_bus_names: list     = []
f_bus_dist_km: list   = []   # km from line start to assigned fbus
t_bus_dist_km: list   = []   # km from line end   to assigned tbus
f_is_crossborder: list = []  # True when from-endpoint is outside the bbox
t_is_crossborder: list = []
line_lengths_km: list  = []

# bus_i → base_kV lookup for voltage inference
bus_i_to_basekv = {int(r["bus_i"]): r["base_kV"]
                   for _, r in data_buses.iterrows()}
inferred_voltages: list = []

n_proximity = 0
n_trajectory = 0

for idx, row in data_lines.iterrows():
    coords   = list(row.geometry.coords)
    start_ll = coords[0]
    end_ll   = coords[-1]
    line_v   = row.get("voltage", None)

    # ---- From-bus ----
    f_outside = is_outside_bbox(start_ll)
    if f_outside:
        f_bus, f_dist, f_name = None, None, None
    else:
        # Pass A: proximity + voltage matching within BUS_ASSIGN_THRESHOLD_KM
        f_pos, f_dist = find_nearest_bus(start_ll, line_v,
                                         bus_coords, bus_voltages)
        if f_pos is not None:
            f_bus  = int(data_buses.iloc[f_pos]["bus_i"])
            f_name = str(bus_names[f_pos])
            n_proximity += 1
        else:
            # Pass B: trajectory extrapolation
            f_pos, f_dist, f_ct = _trajectory_match(
                coords, "fbus", line_v, bus_coords, bus_voltages)
            if f_pos is not None:
                f_bus  = int(data_buses.iloc[f_pos]["bus_i"])
                f_name = str(bus_names[f_pos])
                n_trajectory += 1
                print(f"    [trajectory fbus] idx={idx} → bus {f_bus} "
                      f"({f_name})  ct={f_ct:.2f} km  dist={f_dist:.2f} km")
            else:
                f_bus, f_name = None, None

    # ---- To-bus ----
    t_outside = is_outside_bbox(end_ll)
    if t_outside:
        t_bus, t_dist, t_name = None, None, None
    else:
        # Pass f_pos so the to-bus is never the same bus as the from-bus
        t_pos, t_dist = find_nearest_bus(end_ll, line_v,
                                         bus_coords, bus_voltages,
                                         exclude_pos=f_pos)
        if t_pos is not None:
            t_bus  = int(data_buses.iloc[t_pos]["bus_i"])
            t_name = str(bus_names[t_pos])
            n_proximity += 1
        else:
            t_pos, t_dist, t_ct = _trajectory_match(
                coords, "tbus", line_v, bus_coords, bus_voltages,
                exclude_pos=f_pos)
            if t_pos is not None:
                t_bus  = int(data_buses.iloc[t_pos]["bus_i"])
                t_name = str(bus_names[t_pos])
                n_trajectory += 1
                print(f"    [trajectory tbus] idx={idx} → bus {t_bus} "
                      f"({t_name})  ct={t_ct:.2f} km  dist={t_dist:.2f} km")
            else:
                t_bus, t_name = None, None

    f_buses.append(f_bus);          t_buses.append(t_bus)
    f_bus_names.append(f_name);     t_bus_names.append(t_name)
    f_bus_dist_km.append(f_dist);   t_bus_dist_km.append(t_dist)
    f_is_crossborder.append(f_outside)
    t_is_crossborder.append(t_outside)

    # ---- Voltage inference ----
    v_levels = parse_voltage_tag(line_v)
    if v_levels:
        inferred_voltages.append(line_v)
    else:
        inferred_v = None
        for candidate_bus in (f_bus, t_bus):
            if candidate_bus is not None:
                bkv = bus_i_to_basekv.get(candidate_bus)
                if bkv is not None and not (isinstance(bkv, float)
                                            and np.isnan(bkv)):
                    inferred_v = int(bkv)
                    break
        inferred_voltages.append(inferred_v)

    # ---- Geodesic line length ----
    length_km = sum(
        geodesic((coords[i][1], coords[i][0]),
                 (coords[i+1][1], coords[i+1][0])).km
        for i in range(len(coords) - 1)
    )
    line_lengths_km.append(round(length_km, 4))

n_no_fbus    = sum(1 for b in f_buses if b is None)
n_no_tbus    = sum(1 for b in t_buses if b is None)
n_crossborder = sum(1 for a, b in zip(f_is_crossborder, t_is_crossborder)
                    if a or b)
n_inferred_v = sum(
    1 for orig, inf in zip(
        [row.get("voltage") for _, row in data_lines.iterrows()],
        inferred_voltages)
    if (not parse_voltage_tag(orig)) and inf is not None)
print(f"  Proximity matches  : {n_proximity}")
print(f"  Trajectory matches : {n_trajectory}")
print(f"  Missing fbus       : {n_no_fbus}  (unmatched or cross-border)")
print(f"  Missing tbus       : {n_no_tbus}  (unmatched or cross-border)")
print(f"  Cross-border lines : {n_crossborder}")
print(f"  Voltage inferred   : {n_inferred_v}")


# ============================================================
# STEP 10 – BUILD MATPOWER BRANCH OUTPUT  (topology only)
# ============================================================
print("\n" + "=" * 60)
print("STEP 10: Building MATPOWER branch output")
print("=" * 60)

# Standard MATPOWER mpc.branch columns:
#   fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status,
#   angmin, angmax
#
# r, x, b and ratings are left at 0 / placeholder.  They require
# conductor-type data (ACSR/AAAC gauge, temperature, etc.) that is not
# available directly from OSM.  They will be filled in a separate step
# once conductor specifications are sourced.
#
# Extra reference columns appended (strip before loading into MATPOWER):
#   fbus_name / tbus_name  – substation names, e.g. "Akosombo → Tafo"
#   length_km              – geodesic line length
#   voltage_V              – OSM voltage tag of the line
#   osm_id                 – original OSM way ID(s)

matpower_lines = pd.DataFrame({
    "fbus":              f_buses,
    "tbus":              t_buses,
    "r":                 0.0,    # resistance   (p.u.) – TBD
    "x":                 0.0,    # reactance    (p.u.) – TBD
    "b":                 0.0,    # susceptance  (p.u.) – TBD
    "rateA":             0.0,    # thermal MVA limit   – TBD
    "rateB":             0.0,
    "rateC":             0.0,
    "ratio":             0,      # 0 = transmission line, not transformer
    "angle":             0,      # phase-shift angle (degrees)
    "status":            1,      # 1 = in-service
    "angmin":           -60,
    "angmax":            60,
    # ----- extra reference columns (strip before loading into MATPOWER) -----
    "fbus_name":         f_bus_names,    # substation name or geocoded city
    "tbus_name":         t_bus_names,
    "fbus_dist_km":      f_bus_dist_km,  # distance from line start to fbus
    "tbus_dist_km":      t_bus_dist_km,  # distance from line end   to tbus
    "f_crossborder":     f_is_crossborder,  # True if from-end is outside bbox
    "t_crossborder":     t_is_crossborder,
    "length_km":         line_lengths_km,
    "voltage_V":         inferred_voltages,  # OSM tag, or inferred from bus
    "osm_id":            data_lines["id"].values,
})

# Remove self-loops that are also very short.
# A self-loop (fbus == tbus) with length < 1 km is almost certainly a
# leftover busbar or OSM digitising artefact that was not caught by the
# polygon filter (e.g. the polygon boundary was slightly too small).
# Self-loops that are ≥ 1 km long are kept for manual review because they
# may represent real parallel feeder arrangements within the substation.
self_loop_mask = (
    matpower_lines["fbus"].notna() &
    matpower_lines["tbus"].notna() &
    (matpower_lines["fbus"] == matpower_lines["tbus"]) &
    (matpower_lines["length_km"] < 1.0)
)
n_self_loops = self_loop_mask.sum()
if n_self_loops:
    print(f"  Removing {n_self_loops} self-loop(s) "
          f"(fbus == tbus and length < 1 km):")
    print(matpower_lines[self_loop_mask][
        ["fbus", "tbus", "fbus_name", "tbus_name", "length_km", "osm_id"]
    ].to_string(index=False))

matpower_lines = matpower_lines[~self_loop_mask].reset_index(drop=True)

# Sort by assignment confidence: lines whose endpoints are closest to a bus
# come first.  Cross-border or unmatched endpoints (None / NaN distance) sort
# to the end because they have the least certainty.
_f = matpower_lines["fbus_dist_km"].apply(
    lambda v: float("inf")
    if v is None or (isinstance(v, float) and np.isnan(v)) else float(v))
_t = matpower_lines["tbus_dist_km"].apply(
    lambda v: float("inf")
    if v is None or (isinstance(v, float) and np.isnan(v)) else float(v))
matpower_lines = (matpower_lines
                  .iloc[(_f + _t).argsort().values]
                  .reset_index(drop=True))

matpower_lines.to_csv(WORK_DIR / f"matpower_{region}_lines.csv", index=False)
print(f"  Saved {len(matpower_lines)} branches → data/working_files/matpower_{region}_lines.csv")


# ============================================================
# STEP 11 – ISLAND BUSES  (buses with no connected lines)
# ============================================================
print("\n" + "=" * 60)
print("STEP 11: Finding island buses")
print("=" * 60)

# A bus is 'islanded' if no branch in the final output uses it as a from-bus
# or to-bus.  Common causes:
#   • All lines near that substation were removed by the voltage or busbar
#     filters in Steps 4–5.
#   • The substation serves only distribution-level lines not captured in
#     the high-voltage OSM dataset.
#   • The substation is planned / under construction in OSM but not yet
#     connected to the transmission backbone.

connected_bus_nums: set[int] = set()
for fb, tb in zip(matpower_lines["fbus"], matpower_lines["tbus"]):
    if fb is not None and not (isinstance(fb, float) and np.isnan(fb)):
        connected_bus_nums.add(int(fb))
    if tb is not None and not (isinstance(tb, float) and np.isnan(tb)):
        connected_bus_nums.add(int(tb))

island_mask  = ~data_buses["bus_i"].isin(connected_bus_nums)
island_buses = data_buses[island_mask].copy()

# Unpack lon/lat from the coordinate string for human-readable output
island_buses["longitude"] = island_buses["coordinates"].apply(
    lambda c: eval(c)[0])
island_buses["latitude"]  = island_buses["coordinates"].apply(
    lambda c: eval(c)[1])

# Include name column if available
keep_cols = ["bus_i", "longitude", "latitude", "base_kV"]
rename_map = {"bus_i": "bus", "base_kV": "voltage_V"}
if "name" in island_buses.columns:
    keep_cols.append("name")

island_buses_out = (
    island_buses[keep_cols]
    .rename(columns=rename_map)
    .reset_index(drop=True)
)

island_buses_out.to_csv(WORK_DIR / f"island_buses_{region}.csv", index=False)
print(f"  Found {len(island_buses_out)} island buses "
      f"→ saved to data/working_files/island_buses_{region}.csv")
if len(island_buses_out) > 0:
    print(island_buses_out.to_string(index=False))

