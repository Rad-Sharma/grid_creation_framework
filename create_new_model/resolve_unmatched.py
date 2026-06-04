"""
resolve_unmatched.py

Interactive post-processing to assign buses to transmission-line endpoints
that the automated pipeline (create_lines.py) left unmatched.

Pass 1 – Folium map + interactive CLI (human-in-the-loop)
  • Opens an interactive HTML map showing all unmatched endpoints.
  • Each marker popup shows the exact coordinates and a Google Maps link.
  • Already-assigned lines are drawn in a muted background colour for context.
  • After each user decision the map is regenerated (refresh browser with F5).
  • Endpoints that share a map location are offset so their markers do not overlap.

  Commands at each prompt:
    <bus_i>       – assign that bus to this endpoint
    skip          – leave unmatched (stays on map; return to it later)
    delete        – mark the entire line for deletion
    new           – add a brand-new substation at the endpoint location
                    (prompts for name, primary kV, optional secondary kV)
    merge <N>     – merge this line segment with Line N from the unmatched list

After Pass 1 the script:
  • Deletes lines still unmatched after all decisions
  • Fills R, X, B, rateA using empirical power-law fits (V in kV, 100 MVA base)
  • Saves matpower_{region}_lines_resolved.csv and the updated buses CSV
  • Fills missing bus voltages from connected lines, then assigns the most
    common voltage to any buses that remain unassigned

Prerequisites
-------------
  pip install folium

Reads
-----
  data/working_files/{region}/matpower_{region}_lines.csv
  data/working_files/{region}/processed_{region}_lines.csv
  data/working_files/{region}/matpower_{region}_buses.csv

Writes
------
  data/working_files/{region}/matpower_{region}_lines_resolved.csv
  data/working_files/{region}/matpower_{region}_buses.csv   (updated in-place)
"""

import math
import os
import webbrowser
from collections import defaultdict

import folium
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from shapely import wkt as shapely_wkt


# ============================================================
# CONFIGURATION
# ============================================================

from pathlib import Path
import config

region = config.region

PROJ_ROOT = Path(__file__).parent.parent
WORK_DIR  = PROJ_ROOT / "data" / "working_files" / region
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Pass 1 – Folium / manual
FOLIUM_RADIUS_KM = 80.0
FOLIUM_OUTPUT    = str(WORK_DIR / f"unmatched_review_{region}.html")
OFFSET_DEG       = 0.005   # ~0.5 km angular offset for co-located markers

# R / X / B / rateA power-law fit coefficients  (V in kV, 100 MVA base)
# Formulae: value = a * V^b (per km for R, X, B; total MVA for rateA)
#
# Coefficients are derived from typical global overhead-line parameters:
#   - R, X sourced from IEC 60287 / Glover-Sarma-Overbye transmission-line tables
#     covering ACSR conductors at 69–765 kV common in international grids.
#   - B (shunt susceptance) from typical bundled conductor capacitance values.
#   - MVA thermal rating from IEEE Std 738 conductor ampacity at 75 °C / 25 °C ambient.
# These represent representative values for standard ACSR/AAAC conductors; adjust
# if conductor specifications for the target region are known.
_COEFF = {
    "MVA": (0.0184,       1.8998),
    "R":   (1115.415476, -2.9870),
    "X":   (80.978419,   -2.1092),
    "B":   (0.000049,     0.6896),
}


# ============================================================
# HELPERS
# ============================================================

def parse_voltage_tag(v) -> list:
    if v is None:
        return []
    if isinstance(v, float) and np.isnan(v):
        return []
    result = []
    for part in str(v).split(";"):
        try:
            result.append(int(float(part.strip())))
        except ValueError:
            pass
    return result


def _fmt_kv(v_raw) -> str:
    """Format a raw voltage tag (Volts) as a human-readable kV string."""
    levels = parse_voltage_tag(v_raw)
    if not levels:
        return "unknown"
    return " / ".join(f"{v // 1000} kV" for v in levels)


def voltage_compatible(line_v, bus_v) -> bool:
    lv = parse_voltage_tag(line_v)
    bv = parse_voltage_tag(bus_v)
    if not lv or not bv:
        return True
    return bool(set(lv) & set(bv))


def is_null(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _line_length_km(coords: list) -> float:
    return sum(
        geodesic((coords[i][1], coords[i][0]),
                 (coords[i+1][1], coords[i+1][0])).km
        for i in range(len(coords) - 1))


def nearby_candidates(
    endpoint: tuple,
    line_v,
    bus_df: pd.DataFrame,
    bus_coords: list,
    bus_voltages: list,
    radius_km: float,
) -> list:
    out = []
    for k, (bc, bv) in enumerate(zip(bus_coords, bus_voltages)):
        dist = geodesic((endpoint[1], endpoint[0]), (bc[1], bc[0])).km
        if dist > radius_km or not voltage_compatible(line_v, bv):
            continue
        out.append((k, dist,
                    int(bus_df.iloc[k]["bus_i"]),
                    str(bus_df.iloc[k].get("name", "")),
                    bv))
    out.sort(key=lambda c: c[1])
    return out


def _endpoint_from_coords(coords: list, which: str) -> tuple:
    return coords[0] if which == "fbus" else coords[-1]


def _compute_marker_offsets(positions: list) -> list:
    """
    For any group of endpoints at approximately the same location
    (within 0.01°), spread them in a circle of OFFSET_DEG radius so
    their map markers do not overlap.
    """
    groups: dict = defaultdict(list)
    for i, pos in enumerate(positions):
        key = (round(pos[0], 2), round(pos[1], 2))
        groups[key].append(i)

    offsets = [(0.0, 0.0)] * len(positions)
    for key, indices in groups.items():
        n = len(indices)
        if n == 1:
            continue
        for j, idx in enumerate(indices):
            angle = 2 * math.pi * j / n
            offsets[idx] = (OFFSET_DEG * math.cos(angle),
                            OFFSET_DEG * math.sin(angle))
    return offsets


# ============================================================
# FOLIUM MARKER STYLE HELPERS
# ============================================================

_SHAPE_STYLES = [
    ("circle",  "border-radius:50%"),
    ("square",  "border-radius:3px"),
    ("diamond", "clip-path:polygon(50% 0%,100% 50%,50% 100%,0% 50%)"),
]
_MARKER_COLORS = [
    ("#d62728", "red"),
    ("#2ca02c", "green"),
    ("#1f77b4", "blue"),
    ("#ff7f0e", "orange"),
    ("#9467bd", "purple"),
    ("#17becf", "teal"),
    ("#e377c2", "pink"),
    ("#8c564b", "brown"),
]


def _marker_combo(seq: int) -> tuple:
    color_hex, color_name = _MARKER_COLORS[seq % len(_MARKER_COLORS)]
    shape_name, shape_css = _SHAPE_STYLES[
        (seq // len(_MARKER_COLORS)) % len(_SHAPE_STYLES)]
    label = f"{color_name} {shape_name} {seq + 1}"
    html = (
        f'<div style="background:{color_hex};{shape_css};'
        f'border:2px solid white;width:28px;height:28px;'
        f'display:flex;align-items:center;justify-content:center;'
        f'box-shadow:1px 1px 3px rgba(0,0,0,0.6);">'
        f'<span style="font-size:12px;font-weight:bold;color:white;">'
        f'{seq + 1}</span></div>'
    )
    return html, color_hex, label


# ============================================================
# MAP BUILDER
# ============================================================

def _build_folium_map(
    unmatched_rows: list,
    handled_seqs: set,
    geom_lookup: dict,
    bus_df: pd.DataFrame,
    bus_coords: list,
    bus_voltages: list,
    lines_df: pd.DataFrame,
) -> str:
    """
    Build and save the Folium HTML map.

    unmatched_rows  – full list of (df_i, row, which, ep) tuples (original order)
    handled_seqs    – seq indices that are already done (deleted / assigned / merged)
                      Skipped endpoints are NOT in handled_seqs and stay visible.
    lines_df        – full lines DataFrame; already-assigned lines are drawn in
                      a muted background colour for network context.
    """
    remaining = [(seq, df_i, row, which, ep)
                 for seq, (df_i, row, which, ep) in enumerate(unmatched_rows)
                 if seq not in handled_seqs]

    if remaining:
        all_eps = [ep for _, _, _, _, ep in remaining]
        centre  = [float(np.mean([e[1] for e in all_eps])),
                   float(np.mean([e[0] for e in all_eps]))]
    else:
        # Fall back to the centre of the study-region bounding box
        centre = [
            (config.REGION_BBOX[1] + config.REGION_BBOX[3]) / 2,  # lat
            (config.REGION_BBOX[0] + config.REGION_BBOX[2]) / 2,  # lon
        ]

    m = folium.Map(location=centre, zoom_start=7, tiles="CartoDB positron")

    # ── Background: already-assigned lines (muted grey) ──────────────────
    for _, lrow in lines_df.iterrows():
        fb = lrow.get("fbus")
        tb = lrow.get("tbus")
        if is_null(fb) or is_null(tb):
            continue
        osm_id = str(lrow.get("osm_id", ""))
        coords = geom_lookup.get(osm_id, [])
        if not coords:
            continue
        folium.PolyLine(
            [(c[1], c[0]) for c in coords],
            color="#aaaaaa", weight=1.5, opacity=0.55,
            tooltip=f"Assigned: bus {int(fb)} → {int(tb)}",
        ).add_to(m)

    # ── All substation buses ──────────────────────────────────────────────
    seen_buses: set = set()
    for k, (bc, bv) in enumerate(zip(bus_coords, bus_voltages)):
        bus_i = int(bus_df.iloc[k]["bus_i"])
        if bus_i in seen_buses:
            continue
        seen_buses.add(bus_i)
        name  = bus_df.iloc[k].get("name", "")
        folium.CircleMarker(
            location=[bc[1], bc[0]],
            radius=5,
            color="#2c6fad",
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>bus {bus_i}</b><br>{name}<br>{_fmt_kv(bv)}",
                max_width=200),
            tooltip=f"bus {bus_i}",
        ).add_to(m)

    # ── Compute anti-overlap offsets for remaining markers ────────────────
    eps_list = [ep for _, _, _, _, ep in remaining]
    offsets  = _compute_marker_offsets(eps_list)

    # ── Unmatched endpoint markers ────────────────────────────────────────
    for i, (seq, _, row, which, ep) in enumerate(remaining):
        osm_id = str(row["osm_id"])
        coords = geom_lookup.get(osm_id, [])
        marker_html, color, marker_label = _marker_combo(seq)

        if coords:
            folium.PolyLine(
                [(c[1], c[0]) for c in coords],
                color=color, weight=3, opacity=0.9,
                tooltip=f"Line {seq + 1} ({which})",
            ).add_to(m)

        dx, dy     = offsets[i]
        ep_disp    = (ep[0] + dx, ep[1] + dy)
        ep_label   = "from-end" if which == "fbus" else "to-end"
        other      = "tbus" if which == "fbus" else "fbus"
        other_v    = row.get(other, "?")
        gmaps_url  = f"https://maps.google.com/?q={ep[1]:.6f},{ep[0]:.6f}"
        folium.Marker(
            location=[ep_disp[1], ep_disp[0]],
            icon=folium.DivIcon(
                html=marker_html,
                icon_size=(28, 28),
                icon_anchor=(14, 14),
            ),
            popup=folium.Popup(
                f"<b>Line {seq + 1}</b> ({marker_label}) – {ep_label}<br>"
                f"OSM id: {osm_id}<br>"
                f"Other end ({other}): {other_v}<br>"
                f"Voltage: {_fmt_kv(row.get('voltage_V'))}<br>"
                f"<b>Lat:</b> {ep[1]:.6f}°&nbsp;&nbsp;<b>Lon:</b> {ep[0]:.6f}°<br>"
                f"<a href='{gmaps_url}' target='_blank'>📍 Open in Google Maps</a>",
                max_width=300),
            tooltip=f"Line {seq + 1} – {marker_label} – {ep_label}",
        ).add_to(m)

    out_path = os.path.abspath(FOLIUM_OUTPUT)
    m.save(out_path)
    return out_path


# ============================================================
# PASS 1 – INTERACTIVE FOLIUM REVIEW
# ============================================================

def run_pass1(
    lines_df: pd.DataFrame,
    geom_lookup: dict,
    bus_df: pd.DataFrame,
    bus_coords: list,
    bus_voltages: list,
) -> pd.DataFrame:
    """
    Pass 1: Folium map + interactive CLI.

    For each unmatched (non-cross-border) endpoint the prompt accepts:
      <bus_i>     – assign this bus
      skip        – leave unmatched (stays on map; revisit later)
      delete      – mark the entire line for deletion
      new         – create a new substation at the endpoint coordinates
                    (prompts for name, primary kV, optional secondary kV)
      merge <N>   – merge this line with Line N from the unmatched list
    """
    df = lines_df.copy()

    # Build unmatched_rows: all endpoints that need assignment.
    # Cross-border endpoints (outside Ghana bbox) are included so the user
    # can decide whether to assign, skip, or delete them.
    unmatched_rows: list = []
    for i, row in df.iterrows():
        for which in ("fbus", "tbus"):
            if not is_null(row[which]):
                continue
            osm_id = str(row["osm_id"])
            coords = geom_lookup.get(osm_id, [])
            ep = _endpoint_from_coords(coords, which) if coords else (0.0, 0.0)
            unmatched_rows.append((i, row, which, ep))

    if not unmatched_rows:
        print("  No endpoints left for manual review.")
        return df

    print(f"\n  Building Folium map for {len(unmatched_rows)} endpoint(s)…")
    handled_seqs:  set = set()   # seq indices definitively resolved (not skipped)
    rows_to_delete: set = set()  # df row indices to delete at end
    consumed:       set = set()  # df row indices absorbed by a merge
    valid_bus_i_set = set(bus_df["bus_i"].astype(int).tolist())
    n_resolved = 0

    bus_df         = bus_df.copy()
    bus_coords_l   = list(bus_coords)
    bus_voltages_l = list(bus_voltages)

    def _refresh_map():
        path = _build_folium_map(
            unmatched_rows, handled_seqs, geom_lookup,
            bus_df, bus_coords_l, bus_voltages_l, df)
        print(f"  [Map updated → {path}  –  press F5 in browser to refresh]")

    _refresh_map()
    map_path = os.path.abspath(FOLIUM_OUTPUT)
    print(f"  Saved map → {map_path}")
    print("  Opening in browser – refer to it while answering prompts below.")
    webbrowser.open(f"file://{map_path}")

    def _print_candidates(cands):
        if cands:
            for k, (_, dist, bi, nm, bv) in enumerate(cands[:10], 1):
                print(f"    {k:2d}. bus {bi:<5d}  {nm:<35s}  "
                      f"{_fmt_kv(bv)}  {dist:.1f} km")
        else:
            print("    (none within radius)")

    for seq, (df_i, row, which, ep) in enumerate(unmatched_rows):
        # Skip if this row was already consumed by a merge or marked for deletion
        if df_i in consumed or df_i in rows_to_delete:
            handled_seqs.add(seq)
            continue

        osm_id  = str(row["osm_id"])
        other   = "tbus" if which == "fbus" else "fbus"
        line_v  = row.get("voltage_V", None)
        _, _, marker_label = _marker_combo(seq)

        which_w = which
        ep_w    = ep

        candidates = nearby_candidates(
            ep_w, line_v, bus_df, bus_coords_l, bus_voltages_l, FOLIUM_RADIUS_KM)

        print(f"\n{'─' * 60}")
        print(f"  Line {seq + 1}/{len(unmatched_rows)}  (map marker: {marker_label})")
        print(f"  OSM id  : {osm_id}")
        print(f"  Voltage : {_fmt_kv(line_v)}")
        print(f"  Length  : {row.get('length_km', '?')} km")
        print(f"  {other:4s}   : {df.at[df_i, other]} "
              f"({df.at[df_i, f'{other}_name']})")
        print(f"  Missing : {which_w}")
        print(f"  Nearest buses within {FOLIUM_RADIUS_KM} km:")
        _print_candidates(candidates)

        while True:
            raw = input(
                f"\n  Enter bus_i for {which_w}"
                f"  (skip / delete / new / merge <N>): "
            ).strip().lower()

            # ── skip ──────────────────────────────────────────────────────
            if raw == "skip":
                # Do NOT add to handled_seqs — the marker stays on the map.
                break

            # ── delete ────────────────────────────────────────────────────
            if raw == "delete":
                rows_to_delete.add(df_i)
                # Mark ALL seq entries for this df_i as handled so we don't
                # prompt for the partner endpoint (fbus/tbus) of this line.
                for s, (di, _, _, _) in enumerate(unmatched_rows):
                    if di == df_i:
                        handled_seqs.add(s)
                print("  Line marked for deletion.")
                _refresh_map()
                break

            # ── new substation ────────────────────────────────────────────
            if raw == "new":
                sub_name = input("  New substation name: ").strip()
                while True:
                    try:
                        sub_kv_1 = float(
                            input("  Primary voltage (kV, e.g. 161): ").strip())
                        break
                    except ValueError:
                        print("  Enter a numeric kV value (e.g. 161).")

                sec_raw = input(
                    "  Secondary voltage (kV) for transformer bus, "
                    "or press Enter to skip: ").strip()
                if sec_raw:
                    try:
                        sub_kv_2 = float(sec_raw)
                        voltage_tag_str = (f"{int(sub_kv_1 * 1000)}"
                                           f";{int(sub_kv_2 * 1000)}")
                        kv_display = f"{sub_kv_1:.0f} / {sub_kv_2:.0f} kV"
                    except ValueError:
                        sub_kv_2 = None
                        voltage_tag_str = str(int(sub_kv_1 * 1000))
                        kv_display = f"{sub_kv_1:.0f} kV"
                else:
                    sub_kv_2 = None
                    voltage_tag_str = str(int(sub_kv_1 * 1000))
                    kv_display = f"{sub_kv_1:.0f} kV"

                new_bus_i     = int(bus_df["bus_i"].max()) + 1
                new_coord_str = f"({ep_w[0]}, {ep_w[1]})"
                new_row       = {col: np.nan for col in bus_df.columns}
                new_row["bus_i"]       = new_bus_i
                new_row["coordinates"] = new_coord_str
                new_row["longitude"]   = ep_w[0]
                new_row["latitude"]    = ep_w[1]
                new_row["base_kV"]     = sub_kv_1
                new_row["voltage_tag"] = voltage_tag_str
                new_row["name"]        = sub_name
                bus_df = pd.concat(
                    [bus_df, pd.DataFrame([new_row])], ignore_index=True)
                bus_coords_l.append(np.array([ep_w[0], ep_w[1]]))
                bus_voltages_l.append(voltage_tag_str)
                valid_bus_i_set.add(new_bus_i)
                bus_df.to_csv(WORK_DIR / f"matpower_{region}_buses.csv", index=False)
                print(f"  Added bus {new_bus_i} ({sub_name}, {kv_display})"
                      f"  → saved matpower_{region}_buses.csv")

                df.at[df_i, which_w]                   = new_bus_i
                df.at[df_i, f"{which_w}_name"]         = sub_name
                df.at[df_i, f"{which_w}_dist_km"]      = 0.0
                df.at[df_i, f"{which_w}_resolve_pass"] = "manual_new"
                n_resolved += 1
                handled_seqs.add(seq)
                print(f"  Assigned {which_w} = {new_bus_i}")
                _refresh_map()
                break

            # ── merge with another line ───────────────────────────────────
            if raw.startswith("merge"):
                parts = raw.split()
                if len(parts) != 2:
                    print("  Usage: merge <line_number>")
                    continue
                try:
                    tgt_seq = int(parts[1]) - 1
                except ValueError:
                    print("  merge <N> – N must be an integer.")
                    continue
                if tgt_seq < 0 or tgt_seq >= len(unmatched_rows):
                    print(f"  Line number must be 1–{len(unmatched_rows)}.")
                    continue
                tgt_df_i, tgt_row, tgt_which, tgt_ep = unmatched_rows[tgt_seq]
                if tgt_df_i == df_i:
                    print("  Cannot merge a line with itself.")
                    continue
                if tgt_df_i in consumed or tgt_df_i in rows_to_delete:
                    print(f"  Line {tgt_seq + 1} is already consumed or deleted.")
                    continue

                tgt_osm_id = str(df.at[tgt_df_i, "osm_id"])
                tgt_coords = geom_lookup.get(tgt_osm_id, [])
                cur_coords = geom_lookup.get(str(df.at[df_i, "osm_id"]), [])

                print(f"\n  Merging with Line {tgt_seq + 1}:")
                print(f"    OSM id : {tgt_osm_id}")
                print(f"    fbus   : {df.at[tgt_df_i, 'fbus']}  "
                      f"({df.at[tgt_df_i, 'fbus_name']})")
                print(f"    tbus   : {df.at[tgt_df_i, 'tbus']}  "
                      f"({df.at[tgt_df_i, 'tbus_name']})")
                print(f"  Which end of Line {tgt_seq + 1} connects to "
                      f"Line {seq + 1}'s {which_w}?")
                print(f"    A) from-end  ({df.at[tgt_df_i, 'fbus_name']})")
                print(f"    B) to-end    ({df.at[tgt_df_i, 'tbus_name']})")

                while True:
                    ec = input("  Choice (A/B): ").strip().upper()
                    if ec in ("A", "B"):
                        break
                    print("  Enter A or B.")

                if which_w == "tbus":
                    if ec == "A":
                        merged_coords = cur_coords + tgt_coords[1:]
                        new_fbus, new_fbus_name = (df.at[df_i, "fbus"],
                                                    df.at[df_i, "fbus_name"])
                        new_tbus, new_tbus_name = (df.at[tgt_df_i, "tbus"],
                                                    df.at[tgt_df_i, "tbus_name"])
                    else:
                        merged_coords = cur_coords + list(reversed(tgt_coords))[1:]
                        new_fbus, new_fbus_name = (df.at[df_i, "fbus"],
                                                    df.at[df_i, "fbus_name"])
                        new_tbus, new_tbus_name = (df.at[tgt_df_i, "fbus"],
                                                    df.at[tgt_df_i, "fbus_name"])
                else:
                    if ec == "B":
                        merged_coords = tgt_coords + cur_coords[1:]
                        new_fbus, new_fbus_name = (df.at[tgt_df_i, "fbus"],
                                                    df.at[tgt_df_i, "fbus_name"])
                        new_tbus, new_tbus_name = (df.at[df_i, "tbus"],
                                                    df.at[df_i, "tbus_name"])
                    else:
                        merged_coords = list(reversed(tgt_coords)) + cur_coords[1:]
                        new_fbus, new_fbus_name = (df.at[tgt_df_i, "tbus"],
                                                    df.at[tgt_df_i, "tbus_name"])
                        new_tbus, new_tbus_name = (df.at[df_i, "tbus"],
                                                    df.at[df_i, "tbus_name"])

                new_length = _line_length_km(merged_coords)
                new_osm_id = f"{df.at[df_i, 'osm_id']}+{tgt_osm_id}"
                geom_lookup[new_osm_id] = merged_coords

                df.at[df_i, "fbus"]      = new_fbus
                df.at[df_i, "fbus_name"] = new_fbus_name
                df.at[df_i, "tbus"]      = new_tbus
                df.at[df_i, "tbus_name"] = new_tbus_name
                df.at[df_i, "length_km"] = round(new_length, 4)
                df.at[df_i, "osm_id"]    = new_osm_id
                df.at[df_i, f"{which_w}_resolve_pass"] = "manual_merge"

                consumed.add(tgt_df_i)
                rows_to_delete.add(tgt_df_i)
                # Mark all seq entries for the consumed line as handled
                for s, (di, _, _, _) in enumerate(unmatched_rows):
                    if di == tgt_df_i:
                        handled_seqs.add(s)
                handled_seqs.add(seq)

                print(f"  Merged: new length={new_length:.2f} km")
                print(f"  fbus={new_fbus} ({new_fbus_name})"
                      f"  tbus={new_tbus} ({new_tbus_name})")
                _refresh_map()

                still_missing_f = is_null(new_fbus)
                still_missing_t = is_null(new_tbus)
                if still_missing_f or still_missing_t:
                    which_w = "fbus" if still_missing_f else "tbus"
                    ep_w = _endpoint_from_coords(merged_coords, which_w)
                    candidates = nearby_candidates(
                        ep_w, line_v, bus_df, bus_coords_l, bus_voltages_l,
                        FOLIUM_RADIUS_KM)
                    print(f"  Still missing {which_w}. Nearest buses:")
                    _print_candidates(candidates)
                    continue
                else:
                    n_resolved += 1
                    break

            # ── assign existing bus ───────────────────────────────────────
            try:
                chosen = int(raw)
            except ValueError:
                print("  Please enter a bus_i integer,"
                      " skip, delete, new, or merge <N>.")
                continue
            if chosen not in valid_bus_i_set:
                print(f"  bus_i={chosen} not found. Valid range: "
                      f"{min(valid_bus_i_set)}–{max(valid_bus_i_set)}.")
                continue

            match_rows = bus_df[bus_df["bus_i"] == chosen]
            mi = match_rows.index[0]
            dist_km = geodesic(
                (ep_w[1], ep_w[0]),
                (bus_coords_l[mi][1], bus_coords_l[mi][0])
            ).km

            df.at[df_i, which_w]                   = chosen
            df.at[df_i, f"{which_w}_name"]         = str(
                bus_df.loc[mi].get("name", ""))
            df.at[df_i, f"{which_w}_dist_km"]      = round(dist_km, 4)
            df.at[df_i, f"{which_w}_resolve_pass"] = "manual"
            n_resolved += 1
            handled_seqs.add(seq)
            print(f"  Assigned {which_w} = {chosen}")
            _refresh_map()
            break

    if rows_to_delete:
        df = df.drop(index=list(rows_to_delete)).reset_index(drop=True)
        print(f"\n  Deleted {len(rows_to_delete)} line(s) (deletions + merges).")
    print(f"  Pass 1 resolved {n_resolved} endpoint(s).")
    return df, bus_df


# ============================================================
# MAIN
# ============================================================

print("=" * 60)
print("Loading data")
print("=" * 60)

lines_df = pd.read_csv(WORK_DIR / f"matpower_{region}_lines.csv")
bus_df   = pd.read_csv(WORK_DIR / f"matpower_{region}_buses.csv")
proc_df  = pd.read_csv(WORK_DIR / f"processed_{region}_lines.csv")

bus_coords   = [np.array(eval(c)) for c in bus_df["coordinates"]]
bus_voltages = list(bus_df["voltage_tag"].values
                    if "voltage_tag" in bus_df.columns
                    else bus_df["base_kV"].values)

geom_lookup: dict = {}
for _, prow in proc_df.iterrows():
    raw_geom = prow.get("geometry", None)
    if not raw_geom or (isinstance(raw_geom, float) and np.isnan(raw_geom)):
        continue
    try:
        geom   = shapely_wkt.loads(str(raw_geom))
        coords = list(geom.coords)
        geom_lookup[str(prow["id"])] = coords
    except Exception:
        pass

n_unmatched_f = int(lines_df["fbus"].isna().sum())
n_unmatched_t = int(lines_df["tbus"].isna().sum())
print(f"  {len(lines_df)} lines loaded")
print(f"  {n_unmatched_f} lines missing fbus,  {n_unmatched_t} missing tbus")

# ── Pass 1: Interactive review ───────────────────────────────
print("\n" + "=" * 60)
print("PASS 1 – Interactive manual review")
print("=" * 60)
lines_df, bus_df = run_pass1(
    lines_df, geom_lookup, bus_df, bus_coords, bus_voltages)

# ── Delete still-unmatched lines ────────────────────────────
print("\n" + "=" * 60)
print("Removing still-unmatched lines")
print("=" * 60)

unmatched_mask = lines_df["fbus"].isna() | lines_df["tbus"].isna()
n_drop = int(unmatched_mask.sum())
lines_df = lines_df[~unmatched_mask].reset_index(drop=True)
print(f"  Removed {n_drop} still-unmatched line(s).")

# ── Compute R, X, B, rateA ──────────────────────────────────
print("\n" + "=" * 60)
print("Computing R, X, B, rateA from power-law fits")
print("=" * 60)

def _apply_rxb(df, i, V_kv, L_km):
    """Write R/X/B/rateA for row i given voltage in kV and length in km."""
    a_mva, b_mva = _COEFF["MVA"]
    mva = a_mva * V_kv ** b_mva
    df.at[i, "r"]     = round(_COEFF["R"][0] * V_kv ** _COEFF["R"][1] * L_km, 6)
    df.at[i, "x"]     = round(_COEFF["X"][0] * V_kv ** _COEFF["X"][1] * L_km, 6)
    df.at[i, "b"]     = round(_COEFF["B"][0] * V_kv ** _COEFF["B"][1] * L_km, 6)
    df.at[i, "rateA"] = round(mva, 2)
    df.at[i, "rateB"] = round(mva, 2)
    df.at[i, "rateC"] = round(mva, 2)


# Bus voltage lookup (post-fill, so all buses should have a voltage now)
# Use pd.to_numeric to handle values that may have been read back as strings.
bus_df["base_kV"] = pd.to_numeric(bus_df["base_kV"], errors="coerce")
bus_kv_lookup = {int(r["bus_i"]): float(r["base_kV"])
                 for _, r in bus_df.iterrows()
                 if not is_null(r["base_kV"]) and float(r["base_kV"]) > 0}

n_computed = 0
n_inferred = 0

for i, row in lines_df.iterrows():
    L = float(row.get("length_km", 0) or 0)
    if L <= 0:
        continue

    # Pass 1: use the line's own voltage_V tag
    v_list = parse_voltage_tag(row.get("voltage_V", None))
    if v_list:
        _apply_rxb(lines_df, i, v_list[0] / 1000.0, L)
        n_computed += 1
        continue

    # Pass 2: infer from connected bus voltage (now filled for all buses)
    inferred_v = None
    for end in ("fbus", "tbus"):
        b = row.get(end)
        if is_null(b):
            continue
        bkv = bus_kv_lookup.get(int(b))
        if bkv and bkv > 0:
            inferred_v = bkv
            break

    if inferred_v:
        lines_df.at[i, "voltage_V"] = int(inferred_v)
        _apply_rxb(lines_df, i, inferred_v / 1000.0, L)
        n_inferred += 1
        n_computed += 1

# Pass 3: lines still without voltage → use the regional mode voltage
still_no_v = lines_df["r"].isna() | (lines_df["r"] == 0)
if still_no_v.any():
    # voltage_V may be stored as strings in the CSV; coerce to numeric first
    known_v = pd.to_numeric(lines_df["voltage_V"], errors="coerce").dropna()
    known_v = known_v[known_v > 0]
    if len(known_v) > 0:
        mode_v = float(known_v.mode()[0])
        n_mode = 0
        for i in lines_df[still_no_v].index:
            L = float(lines_df.at[i, "length_km"] or 0)
            if L <= 0:
                continue
            lines_df.at[i, "voltage_V"] = int(mode_v)
            _apply_rxb(lines_df, i, mode_v / 1000.0, L)
            n_mode += 1
            n_computed += 1
        if n_mode:
            kv = int(mode_v) // 1000
            print(f"  Assigned mode voltage {kv} kV to {n_mode} line(s) with no voltage tag.")

print(f"  R/X/B computed: {n_computed} line(s)  "
      f"({n_inferred} inferred from bus voltage).")

# ── Save ────────────────────────────────────────────────────
out_path = WORK_DIR / f"matpower_{region}_lines_resolved.csv"
lines_df.to_csv(out_path, index=False)
bus_df.to_csv(WORK_DIR / f"matpower_{region}_buses.csv", index=False)
print(f"\nSaved → {out_path}")
print(f"Saved → {WORK_DIR / f'matpower_{region}_buses.csv'}")

still_f = int(lines_df["fbus"].isna().sum())
still_t = int(lines_df["tbus"].isna().sum())
print(f"Remaining unmatched: {still_f} fbus,  {still_t} tbus")
print(f"Total lines in output: {len(lines_df)}")


# ── POST-PROCESSING 1: Fill missing bus voltages from connected lines ─────────
# After manual review some buses may still lack a base_kV value (voltage_tag
# was absent in OSM). We estimate their voltage from the lines connected to them
# — a bus inherits the voltage of any line that terminates at it.
print("\n" + "=" * 60)
print("POST-PROCESSING 1: Filling missing bus voltages from line connections")
print("=" * 60)

bus_df         = pd.read_csv(WORK_DIR / f"matpower_{region}_buses.csv")
lines_resolved = pd.read_csv(WORK_DIR / f"matpower_{region}_lines_resolved.csv")

# Step A: fill from connected lines
n_filled = 0
for _, lrow in lines_resolved.iterrows():
    v_raw  = lrow.get("voltage_V", None)
    v_list = parse_voltage_tag(v_raw)
    if not v_list:
        continue
    v_volts = max(v_list)
    for end in ("fbus", "tbus"):
        b = lrow.get(end)
        if is_null(b):
            continue
        bi   = int(b)
        mask = bus_df["bus_i"] == bi
        if not mask.any():
            continue
        idx      = bus_df[mask].index[0]
        existing = bus_df.at[idx, "base_kV"]
        if is_null(existing) or float(existing) == 0:
            bus_df.at[idx, "base_kV"] = float(v_volts)
            if is_null(bus_df.at[idx, "voltage_tag"]):
                bus_df.at[idx, "voltage_tag"] = str(v_volts)
            n_filled += 1

print(f"  Filled base_kV for {n_filled} bus(es) inferred from connected line voltages.")

# Step B: assign the most common voltage to any still-unassigned buses
still_missing = bus_df["base_kV"].isna() | (bus_df["base_kV"] == 0)
if still_missing.any():
    assigned = bus_df.loc[~still_missing, "base_kV"].dropna()
    assigned = assigned[assigned > 0]
    if len(assigned) > 0:
        mode_v = float(assigned.mode()[0])
        bus_df.loc[still_missing, "base_kV"]     = mode_v
        bus_df.loc[still_missing, "voltage_tag"] = str(int(mode_v))
        kv = int(mode_v) // 1000
        # Update bus names to include kV suffix if not already present
        for idx in bus_df[still_missing].index:
            name = str(bus_df.at[idx, "name"])
            if f"_{kv}kV" not in name:
                bus_df.at[idx, "name"] = f"{name}_{kv}kV"
        n_mode = int(still_missing.sum())
        print(f"  Assigned mode voltage {mode_v:.0f} V ({kv} kV) to {n_mode} remaining bus(es).")

bus_df.to_csv(WORK_DIR / f"matpower_{region}_buses.csv", index=False)
print(f"  Updated buses → {WORK_DIR / f'matpower_{region}_buses.csv'}")


# ── POST-PROCESSING 2: Remove island buses that remain unconnected ────────────
print("\n" + "=" * 60)
print("POST-PROCESSING 2: Removing remaining island buses")
print("=" * 60)

connected_set: set = set()
for fb, tb in zip(lines_resolved["fbus"], lines_resolved["tbus"]):
    if not is_null(fb):
        connected_set.add(int(fb))
    if not is_null(tb):
        connected_set.add(int(tb))

island_mask = ~bus_df["bus_i"].isin(connected_set)
n_islands   = int(island_mask.sum())
if n_islands:
    print(f"  Removing {n_islands} island bus(es):")
    print(bus_df[island_mask][["bus_i", "name", "base_kV"]].to_string(index=False))
    bus_df     = bus_df[~island_mask].reset_index(drop=True)
    old_to_new = {old: new for new, old in enumerate(bus_df["bus_i"], start=1)}
    bus_df["bus_i"] = range(1, len(bus_df) + 1)
    lines_resolved["fbus"] = lines_resolved["fbus"].apply(
        lambda b: old_to_new.get(int(b), np.nan) if not is_null(b) else np.nan)
    lines_resolved["tbus"] = lines_resolved["tbus"].apply(
        lambda b: old_to_new.get(int(b), np.nan) if not is_null(b) else np.nan)
    lines_resolved.to_csv(WORK_DIR / f"matpower_{region}_lines_resolved.csv", index=False)
    print(f"  Updated line bus numbers → {WORK_DIR / f'matpower_{region}_lines_resolved.csv'}")
else:
    print("  No island buses remaining.")

bus_df.to_csv(WORK_DIR / f"matpower_{region}_buses.csv", index=False)
print(f"  Final bus count: {len(bus_df)}")
