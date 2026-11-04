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
    tap           – the endpoint tees into another line: split that line at
                    the nearest point and join all three at a new junction bus
    merge <N>     – merge this line segment with Line N from the unmatched list

Automatic mode (--auto) answers every prompt itself, using the same commands a
reviewer would type and the rules in run_pass1's _auto_answer. Its journal
replays like a manual one, so a reviewer can audit or correct it afterwards.

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

import argparse
import json
import math
import os
import sys
import webbrowser
from collections import Counter, defaultdict
from datetime import datetime, timezone

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
from pipeline_integrity import (clean_bus_tables, describe_islands, island_report,
                                refresh_endpoint_names,
                                read_transformers, manual_bus,
                                split_multi_voltage_buses, retier_endpoints,
                                manual_transformer, validate_endpoints)

region = config.region

PROJ_ROOT = Path(__file__).parent.parent
WORK_DIR  = PROJ_ROOT / "data" / "working_files" / region
WORK_DIR.mkdir(parents=True, exist_ok=True)

import build_provenance

# Pass 1 – Folium / manual
FOLIUM_RADIUS_KM = 80.0
FOLIUM_OUTPUT    = str(WORK_DIR / f"unmatched_review_{region}.html")
OFFSET_DEG       = 0.005   # ~0.5 km angular offset for co-located markers
TAP_SEARCH_KM    = 1.0     # how far the tap command looks for the line being tapped

# Automatic mode (--auto). The Automate Grid Creation paper puts a reviewer at
# Sec. IV.B.3; these thresholds stand in for that reviewer, so a case built with
# them is not a reviewed model. Each one reuses a distance the paper already uses
# for the same judgement where it has one.
AUTO_JOIN_KM    = 0.5    # an endpoint this close to a bus or to another line's end
                         # is at it (d_junc, Sec. IV.B.2 Step 6)
AUTO_TAP_KM     = 0.1    # an endpoint this close to the inside of another line tees
                         # into it; OSM normally shares the node outright
AUTO_ASSIGN_KM  = 10.0   # proximity radius of Sec. IV.B.2 Step 8 (BUS_ASSIGN_THRESHOLD_KM)
AUTO_TIER_KM    = 2.0    # a substation at another voltage this close is taken to be
                         # the one the line serves, missing that voltage in OSM
AUTO_STUB_KM    = 2.0    # a dead-end line shorter than this is an OSM fragment;
                         # a longer one runs to a substation OSM has not mapped
AUTO_CONNECT_KM = 50.0   # longest voltage-compatible hop added to join two networks

# R / X / B / rateA power-law fit coefficients  (V in kV, 100 MVA base)
# Formulae: value = a * V^b (per km for R, X, B; total MVA for rateA)
#
# Coefficients are derived from typical global overhead-line parameters:
# Bergen, Arthur R. and Vittal, Vijay, “Power Systems
# Analysis.” January 2000.
# OPAL-RT Technologies, “Typical Electrical Parameters.” 2025.
#  A. B. Birchfield, E. Schweitzer, M. H. Athari, T. Xu, T. J.
# Overbye, A. Scaglione, and Z. Wang, “A Metric-Based
# Validation Process to Assess the Realism of Synthetic
# Power Grids,” Energies, vol. 10, no. 8, 2017.
# A. Birchfield, “Geographic coordinate validation and
# assignment using an edge-constrained layout,” Journal
# of Engineering and Applied Science, vol. 71, 05 2024.
# These represent representative values for standard ACSR/AAAC conductors; adjust
# if conductor specifications for the target region are known.
_COEFF = {
    "MVA": (0.018429,     1.8998),
    "R":   (1115.0,      -2.9870),
    "X":   (52.86,       -2.0342),
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


class ReviewLog:
    """Record every manual answer so a review session is auditable and repeatable.

    Sec. IV.B.3 is a human-in-the-loop stage, so the same OSM input previously
    produced a different model for every reviewer with no record of what was
    decided. Answers are captured in order; replaying them reproduces the session
    exactly, because the endpoint iteration order is fixed by the input tables.
    """

    def __init__(self, path=None, replay=False, mode="manual"):
        self.path = Path(path) if path else None
        self.entries = []
        self.pending = []
        self.queued = []     # (answer, rule, reason) decided by automatic mode
        self.mode = mode
        self.replay = replay
        if replay:
            if not (self.path and self.path.is_file()):
                raise FileNotFoundError(f"No review journal to replay at {self.path}")
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            # Numbered from 1, as a reviewer counts the answers in the file.
            self.pending = [dict(e, number=n)
                            for n, e in enumerate(saved.get("answers", []), 1)]
            # An automatic journal visits endpoints in stages, so it must be
            # replayed the same way for its answers to line up.
            self.mode = saved.get("mode", "manual")
            print(f"  Replaying {len(self.pending)} recorded answer(s) from {self.path}")

    def queue(self, answers, rule, reason=""):
        """Supply the answers automatic mode chose for the prompts that follow."""
        if self.queued:
            raise RuntimeError(f"Automatic answers left unused: {self.queued}")
        self.queued = [(answer, rule, reason) for answer in answers]

    def ask(self, prompt, context=None):
        rule = reason = None
        if self.queued:
            answer, rule, reason = self.queued.pop(0)
            # The rule is shown once, on the prompt that opens the decision.
            note = f"   [auto: {rule}{' - ' + reason if reason else ''}]" if context else ""
            print(f"{prompt}{answer}{note}")
        elif self.pending:
            recorded = self.pending.pop(0)
            self._check(recorded, prompt, context)
            answer, rule, reason = (recorded["answer"], recorded.get("rule"),
                                    recorded.get("reason"))
            print(f"{prompt}{answer}   [replayed]")
        elif self.mode == "automatic":
            raise RuntimeError(f"Automatic review has no answer for: {prompt.strip()}")
        else:
            answer = input(prompt)
        entry = dict(prompt=prompt.strip(), answer=answer, context=context)
        if rule:
            entry.update(rule=rule, reason=reason)
        self.entries.append(entry)
        return answer

    @staticmethod
    def _subject(prompt, context):
        """What a prompt asks about, comparable between a journal and a rerun.

        A prompt that opens a decision carries its endpoint, or the two buses a
        network join would connect. A follow-up (substation name, kV, merge end,
        voltage override) has no context of its own and is known by its question.
        """
        if not context:
            return ("question", prompt.strip())
        if "osm_id" in context:
            return ("endpoint", str(context["osm_id"]), str(context["endpoint"]))
        if "fbus" in context:
            return ("join", int(context["fbus"]), int(context["tbus"]))
        return ("context", json.dumps(context, sort_keys=True, default=str))

    @staticmethod
    def _describe(subject):
        kind, *detail = subject
        if kind == "endpoint":
            return f"{detail[1]} of {detail[0]}"
        if kind == "join":
            return f"joining bus {detail[0]} to bus {detail[1]}"
        if kind == "question":
            return f'the follow-up question "{detail[0]}"'
        return detail[0]

    def _recorded_for(self, recorded):
        return self._describe(self._subject(recorded.get("prompt", ""),
                                            recorded.get("context")))

    def _check(self, recorded, prompt, context):
        """Stop when a recorded answer belongs to a different prompt than this one.

        Replay hands out answers in order. A journal edit that changes how many
        prompts a decision takes, or which endpoints come up, moves every later
        answer onto the wrong prompt, where many are still valid commands - a
        substation's kV is a plausible bus number. Catching the first answer
        that moves keeps it from being applied.
        """
        if self._subject(recorded.get("prompt", ""), recorded.get("context")) \
                == self._subject(prompt, context):
            return
        raise RuntimeError(
            f"Journal answer {recorded['number']} ({recorded['answer']!r}) was recorded "
            f"for {self._recorded_for(recorded)}, but the review has reached "
            f"{self._describe(self._subject(prompt, context))}. The journal stops "
            "matching the review here, most likely because of an edit at or before "
            "this answer. Step 3 may already have changed the bus and transformer "
            "tables, so rebuild from Step 1 before replaying again.")

    def finish(self):
        """Fail a replay that ends with recorded answers it never used."""
        if not self.pending:
            return
        first = self.pending[0]
        raise RuntimeError(
            f"The review finished with {len(self.pending)} journal answer(s) unused, "
            f"from answer {first['number']} ({first['answer']!r}, recorded for "
            f"{self._recorded_for(first)}) on. The journal does not match this review, "
            "most likely because of an edit before that answer. The tables Step 3 "
            "wrote do not follow the journal; rebuild from Step 1 before replaying again.")

    def save(self, region=None):
        if self.path is None:
            return
        if self.pending:
            # A replay still under way. Writing now would cut the unused answers
            # out of the very journal being replayed.
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(dict(
            region=region, mode=self.mode,
            recorded_at=datetime.now(timezone.utc).isoformat(),
            answers=self.entries), indent=2), encoding="utf-8")
        print(f"  Review journal ({len(self.entries)} answer(s)) → {self.path}")


def normalise_kv(*values):
    """Shared kV of the given bus voltages, tolerating volts or kV in the column."""
    levels = []
    for value in values:
        for v in parse_voltage_tag(value):
            levels.append(v / 1000.0 if v > 1000 else v)
    shared = [v for v in levels if levels.count(v) == len(values)]
    return float(min(shared)) if shared else float(min(levels)) if levels else 0.0


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


def floor_lengths_at_bus_span(lines_df, bus_df):
    """Raise each line's length to the distance between the buses it connects.

    Equation (3) scales R, X and B by length, and until Step 3 finishes that
    length is the length of the OSM geometry which survived preprocessing. Once
    an endpoint is assigned to a substation the geometry does not reach - the
    automated passes allow it, and the manual review offers candidates up to
    80 km away - the two stop describing the same conductor. A conductor cannot
    be shorter than the straight line between the substations it joins, so that
    distance is its floor.

    Left alone this is not a rounding error. Colombia carried 560 of 979 lines
    below their own floor and one at 1/2500 of it: a 20 m impedance across a
    50 km span, which is a near short circuit between two distant buses. The
    resulting 1e-5 pu reactances leave the network ill-conditioned enough that
    PIPS, and Ipopt through PowerModels, both refuse even the DC problem.

    R, X and B are each linear in length, so they scale with it; the thermal
    limit of (3d) depends only on voltage and does not move. Returns
    (lines raised, worst ratio found, kilometres added).
    """
    located = {}
    for _, row in bus_df.iterrows():
        if not is_null(row.get("latitude")) and not is_null(row.get("longitude")):
            located[int(row["bus_i"])] = (float(row["latitude"]), float(row["longitude"]))
    raised, worst, added = 0, 1.0, 0.0
    for i, row in lines_df.iterrows():
        f, t = row.get("fbus"), row.get("tbus")
        if is_null(f) or is_null(t):
            continue
        here, there = located.get(int(f)), located.get(int(t))
        # Co-located buses are the two tiers of one switchyard, joined by a
        # transformer whose impedance is equipment, not distance.
        if here is None or there is None or here == there:
            continue
        span = geodesic(here, there).km
        recorded = float(row.get("length_km", 0) or 0)
        if span <= recorded + 1e-9:
            continue
        if recorded > 0:
            worst = min(worst, recorded / span)
            scale = span / recorded
            for column in ("r", "x", "b"):
                value = row.get(column)
                if not is_null(value):
                    lines_df.at[i, column] = round(float(value) * scale, 6)
        added += span - recorded
        lines_df.at[i, "length_km"] = round(span, 4)
        raised += 1
    return raised, worst, added


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


def _local_km(origin, points) -> np.ndarray:
    """(lon, lat) points as km east and north of origin.

    Equirectangular, like pipeline_integrity._separation_km: well inside 1 % over
    the tens of kilometres automatic mode compares, and vectorised, which the
    per-pair geodesic is not.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    scale = math.cos(math.radians(origin[1]))
    return np.column_stack(((pts[:, 0] - origin[0]) * 111.32 * scale,
                            (pts[:, 1] - origin[1]) * 110.57))


def find_tap(endpoint, line_v, lines, max_km):
    """Nearest point inside another line that this endpoint meets: a tee.

    lines yields (key, coords, voltage). A point within AUTO_JOIN_KM of that
    line's own ends is not a tee but two ends meeting, which merging or
    assignment handles, so only interior points count. Returns
    dict(key, segment, t, point, km) for coords[segment] + t * (next - this),
    or None.
    """
    best = None
    for key, coords, voltage in lines:
        if len(coords) < 2 or not voltage_compatible(line_v, voltage):
            continue
        xy = _local_km(endpoint, coords)
        start, step = xy[:-1], xy[1:] - xy[:-1]
        length2 = (step ** 2).sum(axis=1)
        t = np.divide(-(start * step).sum(axis=1), length2,
                      out=np.zeros(len(step)), where=length2 > 0).clip(0.0, 1.0)
        foot = start + t[:, None] * step
        dist = np.hypot(foot[:, 0], foot[:, 1])
        k = int(dist.argmin())
        if dist[k] > max_km or (best is not None and dist[k] >= best["km"]):
            continue
        if min(np.hypot(*(foot[k] - xy[0])), np.hypot(*(foot[k] - xy[-1]))) <= AUTO_JOIN_KM:
            continue
        a, b = np.asarray(coords[k], float), np.asarray(coords[k + 1], float)
        best = dict(key=key, segment=k, t=float(t[k]), km=float(dist[k]),
                    point=tuple(float(v) for v in a + t[k] * (b - a)))
    return best


def split_line(coords, segment, t, point):
    """The two halves of a polyline at point, which lies on coords[segment:segment + 2]."""
    coords = [tuple(c) for c in coords]
    if t <= 1e-9:
        return coords[:segment + 1], coords[segment:]
    if t >= 1 - 1e-9:
        return coords[:segment + 2], coords[segment + 1:]
    return coords[:segment + 1] + [point], [point] + coords[segment + 1:]


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
    log: "ReviewLog | None" = None,
) -> pd.DataFrame:
    """
    Pass 1: Folium map + interactive CLI.

    For each unmatched (non-cross-border) endpoint the prompt accepts:
      <bus_i>     – assign this bus
      skip        – leave unmatched (stays on map; revisit later)
      delete      – mark the entire line for deletion
      new         – create a new substation at the endpoint coordinates
                    (prompts for name, primary kV, optional secondary kV)
      tap         – split the line this endpoint tees into and join them
                    at a new junction bus
      merge <N>   – merge this line with Line N from the unmatched list

    When log.mode is "automatic" the prompts are answered by _auto_answer
    instead of a person, and no map is drawn.
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
        return df, bus_df

    log            = log if log is not None else ReviewLog()
    automatic      = log.mode == "automatic"
    deciding       = automatic and not log.replay

    handled_seqs:  set = set()   # seq indices definitively resolved (not skipped)
    rows_to_delete: set = set()  # df row indices to delete at end
    consumed:       set = set()  # df row indices absorbed by a merge
    valid_bus_i_set = set(bus_df["bus_i"].astype(int).tolist())
    n_resolved = 0

    bus_df         = bus_df.copy()
    bus_coords_l   = list(bus_coords)
    bus_voltages_l = list(bus_voltages)

    # Geometry of every live line, kept current through merges, taps and
    # deletions, with a bounding box each so a tap search skips distant lines.
    shapes: dict = {}
    boxes:  dict = {}
    box_index = {"keys": None, "array": None}

    def _reshape(i, coords):
        if coords is None or len(coords) < 2:
            shapes.pop(i, None)
            boxes.pop(i, None)
        else:
            pts = np.asarray(coords, dtype=float)
            shapes[i] = pts
            boxes[i] = (*pts.min(axis=0), *pts.max(axis=0))
        box_index["keys"] = None

    for i in df.index:
        _reshape(i, geom_lookup.get(str(df.at[i, "osm_id"]), []))

    def _near_lines(point, km, exclude):
        """(df index, coords, voltage) of live lines whose box is within km of point."""
        if box_index["keys"] is None:
            box_index["keys"] = list(boxes)
            box_index["array"] = np.array([boxes[k] for k in box_index["keys"]]).reshape(-1, 4)
        b = box_index["array"]
        pad_lat = km / 110.57
        pad_lon = km / (111.32 * max(math.cos(math.radians(point[1])), 0.01))
        hit = ((b[:, 0] - pad_lon <= point[0]) & (point[0] <= b[:, 2] + pad_lon)
               & (b[:, 1] - pad_lat <= point[1]) & (point[1] <= b[:, 3] + pad_lat))
        for j in np.flatnonzero(hit):
            k = box_index["keys"][j]
            if k != exclude and k in shapes:
                yield k, shapes[k], df.at[k, "voltage_V"]

    def _refresh_map():
        if automatic:
            return
        path = _build_folium_map(
            unmatched_rows, handled_seqs, geom_lookup,
            bus_df, bus_coords_l, bus_voltages_l, df)
        print(f"  [Map updated → {path}  –  press F5 in browser to refresh]")

    if automatic:
        print(f"\n  Automatic review of {len(unmatched_rows)} endpoint(s); no map is drawn.")
    else:
        print(f"\n  Building Folium map for {len(unmatched_rows)} endpoint(s)…")
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

    # ── Automatic mode ────────────────────────────────────────────────────
    known_kv = [parse_voltage_tag(v)[0] for v in df.get("voltage_V", pd.Series(dtype=object))
                if parse_voltage_tag(v)]
    fallback_kv = (Counter(known_kv).most_common(1)[0][0] if known_kv
                   else config.MIN_TRANSMISSION_VOLTAGE_V) / 1000.0
    compatible_cache: dict = {}

    def _kv_of(*voltages):
        """kV for a new bus: the first known level among voltages, else the regional mode."""
        for v in voltages:
            levels = parse_voltage_tag(v)
            if levels:
                return levels[0] / 1000.0
        return fallback_kv

    def _shared_kv(line_v, other_v):
        shared = [v for v in parse_voltage_tag(line_v) if v in parse_voltage_tag(other_v)]
        return shared[0] / 1000.0 if shared else _kv_of(line_v, other_v)

    def _compatible(line_v):
        """Per bus, whether the prompt's voltage guard would accept it for line_v."""
        known = compatible_cache.setdefault(str(line_v), [])
        volts = bus_df["base_kV"].tolist()
        known.extend(voltage_compatible(line_v, v) for v in volts[len(known):])
        return np.array(known, dtype=bool)

    def _meeting_ends(df_i, ep, line_v):
        """Unresolved ends of other lines within AUTO_JOIN_KM of ep, nearest first."""
        pad_lat = AUTO_JOIN_KM / 110.57
        pad_lon = AUTO_JOIN_KM / (111.32 * max(math.cos(math.radians(ep[1])), 0.01))
        found = {}
        for s, (di, _, w, _) in enumerate(unmatched_rows):
            if s in handled_seqs or di == df_i or di not in shapes:
                continue
            end = shapes[di][0] if w == "fbus" else shapes[di][-1]
            if abs(end[0] - ep[0]) > pad_lon or abs(end[1] - ep[1]) > pad_lat:
                continue
            if not is_null(df.at[di, w]) or not voltage_compatible(line_v, df.at[di, "voltage_V"]):
                continue
            d = float(np.hypot(*_local_km(ep, end)[0]))
            if d <= AUTO_JOIN_KM and (di not in found or d < found[di][0]):
                found[di] = (d, s, w)
        return [(s, di, w) for di, (d, s, w) in sorted(found.items(), key=lambda kv: kv[1][0])]

    def _new_bus(label, ep, kv):
        name = f"{label} {ep[1]:.4f} {ep[0]:.4f}"
        return ["new", name, f"{kv:g}", ""]

    def _auto_answer(df_i, which, ep, line_v, stage):
        """Answer the prompt for one endpoint: (answers, rule, reason).

        Each rule answers with a command a reviewer could have typed, so the
        journal replays like a manual one. Stage 1 acts only on what the OSM
        geometry itself shows - a substation, the rest of the line, a junction,
        a tee; stage 2 adds the judgements that keep the line; stage 3 decides
        what is left. Running them in that order means nothing is invented or
        deleted while a better-supported answer is still available elsewhere.
        """
        other = "tbus" if which == "fbus" else "fbus"
        other_bus = None if is_null(df.at[df_i, other]) else int(df.at[df_i, other])
        if df_i not in shapes:
            return ["skip"], "no geometry", "the line has no OSM geometry to judge"
        # 0. Both ends are open and at one point. Whatever the first attaches
        #    to, the second lands beside it, so the line can only become a
        #    self-loop. (A short line with one end attached is judged below; it
        #    may be a real tie to a neighbouring substation.)
        far_end = shapes[df_i][-1] if which == "fbus" else shapes[df_i][0]
        if other_bus is None and float(np.hypot(*_local_km(ep, far_end)[0])) <= AUTO_JOIN_KM:
            return ["delete"], "self-loop", "both open ends of the line are at one point"
        bus_ids = bus_df["bus_i"].astype(int).to_numpy()
        km = np.hypot(*_local_km(ep, np.asarray(bus_coords_l, dtype=float)).T)
        same = _compatible(line_v)
        other_end = bus_ids == (-1 if other_bus is None else other_bus)

        def nearest(mask):
            hits = np.flatnonzero(mask)
            return int(hits[km[hits].argmin()]) if len(hits) else None

        def described(k):
            return f"bus {bus_ids[k]} {bus_df.at[k, 'name']} ({_fmt_kv(bus_df.at[k, 'base_kV'])}, {km[k]:.2f} km)"

        # 1. The endpoint is at a substation of its own voltage.
        k = nearest(same & ~other_end & (km <= AUTO_JOIN_KM))
        if k is not None:
            return [str(bus_ids[k])], "at substation", described(k)

        # 2. It meets the open end of another line: one is the rest of this
        #    line, two or more make a junction. Two lines from the same bus
        #    meeting here would merge into a self-loop, so they are a junction too.
        meeting = _meeting_ends(df_i, ep, line_v)
        if len(meeting) == 1:
            tgt_seq, tgt_df_i, tgt_which = meeting[0]
            far = df.at[tgt_df_i, "tbus" if tgt_which == "fbus" else "fbus"]
            if other_bus is None or is_null(far) or int(far) != other_bus:
                return ([f"merge {tgt_seq + 1}", "A" if tgt_which == "fbus" else "B"],
                        "continues", f"into line {tgt_seq + 1} ({df.at[tgt_df_i, 'osm_id']})")
        if meeting:
            kv = _kv_of(line_v, *[df.at[m[1], "voltage_V"] for m in meeting])
            return (_new_bus("Junction", ep, kv), "junction",
                    f"{len(meeting) + 1} line ends meet here")

        # 3. It lies on another line, away from that line's ends: a tee.
        tee = find_tap(ep, line_v, _near_lines(ep, AUTO_TAP_KM, df_i), AUTO_TAP_KM)
        if tee is not None:
            return ["tap"], "tee", f"on {df.at[tee['key'], 'osm_id']} ({tee['km']:.2f} km)"

        # 4. A substation of its own voltage lies within the Step 2 radius.
        k = nearest(same & ~other_end & (km <= AUTO_ASSIGN_KM))
        if k is not None:
            return [str(bus_ids[k])], "nearest substation", described(k)

        if stage == 1:
            return ["skip"], "deferred", "revisited in stage 2"

        # 5. The line comes back to the substation it starts from: a self-loop,
        #    which Sec. IV.B.2 Step 10 would delete anyway. Ending farther out
        #    than a stub's length, it is a radial line, and rule 7 judges it.
        if other_end.any() and km[other_end].min() <= AUTO_STUB_KM:
            return ["delete"], "self-loop", (f"ends {km[other_end].min():.2f} km from "
                                             f"its own other end, bus {other_bus}")

        # 6. A substation at another voltage is close. The line is attached
        #    across the voltage guard; POST-PROCESSING 1b then gives the
        #    substation a tier at the line's voltage and a transformer.
        k = nearest(~same & ~other_end & (km <= AUTO_TIER_KM))
        if k is not None:
            return [str(bus_ids[k]), "y"], "other-voltage substation", described(k)

        if stage == 2:
            return ["skip"], "deferred", "revisited in stage 3"

        # 7. A dead end on a line long enough to be real runs to a substation
        #    OSM has not mapped.
        length = 0.0 if is_null(df.at[df_i, "length_km"]) else float(df.at[df_i, "length_km"])
        flag = df.loc[df_i].get("f_crossborder" if which == "fbus" else "t_crossborder")
        crossing = not is_null(flag) and bool(flag)
        if other_bus is not None and length >= AUTO_STUB_KM:
            return (_new_bus("Inferred substation", ep, _kv_of(line_v)), "inferred substation",
                    f"{'cross-border ' if crossing else ''}dead end of a {length:.1f} km line")

        # 8. Anything else is an OSM fragment.
        if other_bus is None:
            return ["delete"], "fragment", "no connection at either end"
        return ["delete"], "stub", f"{length:.2f} km dead end, shorter than {AUTO_STUB_KM:g} km"

    # Manual review visits each endpoint once. Automatic review makes three
    # passes, so that stage 1's rules settle every endpoint they can before
    # stage 2 and 3 add or delete anything.
    def _schedule():
        for stage in ((1, 2, 3) if automatic else (3,)):
            seq = 0
            while seq < len(unmatched_rows):   # taps and merges append endpoints
                yield seq, stage
                seq += 1

    for seq, stage in _schedule():
        df_i, row, which, ep = unmatched_rows[seq]
        if seq in handled_seqs:
            continue
        # Skip if this row was already consumed by a merge or marked for deletion
        if df_i in consumed or df_i in rows_to_delete:
            handled_seqs.add(seq)
            continue
        # A merge or tap since this endpoint was listed may already have settled it.
        if not is_null(df.at[df_i, which]):
            handled_seqs.add(seq)
            continue

        osm_id  = str(df.at[df_i, "osm_id"])
        other   = "tbus" if which == "fbus" else "fbus"
        line_v  = df.loc[df_i].get("voltage_V", None)
        _, _, marker_label = _marker_combo(seq)

        which_w = which
        ep_w    = (_endpoint_from_coords(geom_lookup[osm_id], which)
                   if geom_lookup.get(osm_id) else ep)

        if not automatic:
            candidates = nearby_candidates(
                ep_w, line_v, bus_df, bus_coords_l, bus_voltages_l, FOLIUM_RADIUS_KM)

            print(f"\n{'─' * 60}")
            print(f"  Line {seq + 1}/{len(unmatched_rows)}  (map marker: {marker_label})")
            print(f"  OSM id  : {osm_id}")
            print(f"  Voltage : {_fmt_kv(line_v)}")
            print(f"  Length  : {df.at[df_i, 'length_km'] if 'length_km' in df.columns else '?'} km")
            print(f"  {other:4s}   : {df.at[df_i, other]} "
                  f"({df.at[df_i, f'{other}_name']})")
            print(f"  Missing : {which_w}")
            print(f"  Nearest buses within {FOLIUM_RADIUS_KM} km:")
            _print_candidates(candidates)

        refused = 0
        while True:
            if deciding:
                refused += 1
                if refused > 3:
                    raise RuntimeError(
                        f"Automatic review could not settle line {seq + 1} ({osm_id}): "
                        "the prompt refused its answers.")
                log.queue(*_auto_answer(df_i, which_w, ep_w, line_v, stage))
            raw = log.ask(
                (f"  Line {seq + 1}/{len(unmatched_rows)} {which_w}: " if automatic else
                 f"\n  Enter bus_i for {which_w}"
                 f"  (skip / delete / new / tap / merge <N>): "),
                context=dict(osm_id=osm_id, endpoint=which_w,
                             lon=round(ep_w[0], 7), lat=round(ep_w[1], 7)),
            ).strip().lower()

            # ── skip ──────────────────────────────────────────────────────
            if raw == "skip":
                # Do NOT add to handled_seqs — the marker stays on the map.
                if automatic and which_w != which:
                    # A merge left this end open; list it so a later stage returns.
                    unmatched_rows.append((df_i, df.loc[df_i], which_w, ep_w))
                break

            # ── delete ────────────────────────────────────────────────────
            if raw == "delete":
                rows_to_delete.add(df_i)
                _reshape(df_i, None)
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
                sub_name = log.ask("  New substation name: ").strip()
                while True:
                    try:
                        sub_kv_1 = float(
                            log.ask("  Primary voltage (kV, e.g. 161): ").strip())
                        if not math.isfinite(sub_kv_1) or sub_kv_1 <= 0:
                            raise ValueError("Voltage must be positive")
                        break
                    except ValueError:
                        print("  Enter a positive numeric kV value (e.g. 161).")

                while True:
                    sec_raw = log.ask("  Secondary voltage (kV), or Enter for none: ").strip()
                    try:
                        sub_kv_2 = float(sec_raw) if sec_raw else None
                        if sub_kv_2 is not None and (not math.isfinite(sub_kv_2) or sub_kv_2 <= 0 or sub_kv_2 == sub_kv_1):
                            raise ValueError("Secondary voltage must be positive and distinct")
                        break
                    except ValueError:
                        print("  Enter a distinct positive voltage, or leave blank.")
                new_bus_i = int(bus_df["bus_i"].max()) + 1
                name = f"{sub_name}_{sub_kv_1:g}kV" if sub_kv_2 else sub_name
                new_row = manual_bus(new_bus_i, name, ep_w, sub_kv_1, bus_df.columns)
                new_rows = [new_row]
                if sub_kv_2 is not None:
                    secondary = manual_bus(new_bus_i + 1, f"{sub_name}_{sub_kv_2:g}kV",
                                           ep_w, sub_kv_2, bus_df.columns)
                    new_rows.append(secondary)
                    xfmr_path = WORK_DIR / f"transformer_pairs_{region}.csv"
                    transformers = read_transformers(xfmr_path)
                    transformers = pd.concat([transformers, pd.DataFrame([
                        manual_transformer(new_row, secondary)])], ignore_index=True)
                    transformers.to_csv(xfmr_path, index=False)
                bus_df = pd.concat([bus_df, pd.DataFrame(new_rows)], ignore_index=True)
                for added in new_rows:
                    bus_coords_l.append(np.array([ep_w[0], ep_w[1]]))
                    bus_voltages_l.append(added["voltage_tag"])
                    valid_bus_i_set.add(added["bus_i"])
                bus_df.to_csv(WORK_DIR / f"matpower_{region}_buses.csv", index=False)
                sub_name = new_row["name"]
                print(f"  Added {len(new_rows)} voltage-tier bus(es), assigned bus {new_bus_i}")

                df.at[df_i, which_w]                   = new_bus_i
                df.at[df_i, f"{which_w}_name"]         = sub_name
                df.at[df_i, f"{which_w}_dist_km"]      = 0.0
                df.at[df_i, f"{which_w}_resolve_pass"] = "manual_new"
                n_resolved += 1
                handled_seqs.add(seq)
                print(f"  Assigned {which_w} = {new_bus_i}")
                _refresh_map()
                break

            # ── tap into another line ─────────────────────────────────────
            if raw == "tap":
                tee = find_tap(ep_w, line_v, _near_lines(ep_w, TAP_SEARCH_KM, df_i),
                               TAP_SEARCH_KM)
                if tee is None:
                    print(f"  No voltage-compatible line passes within {TAP_SEARCH_KM:g} km "
                          "of this endpoint away from its own ends; nothing to tap.")
                    continue
                host = tee["key"]
                host_osm = str(df.at[host, "osm_id"])
                head, tail = split_line(geom_lookup[host_osm], tee["segment"],
                                        tee["t"], tee["point"])
                kv = _shared_kv(line_v, df.at[host, "voltage_V"])
                junction_i = int(bus_df["bus_i"].max()) + 1
                junction_name = f"Junction {tee['point'][1]:.4f} {tee['point'][0]:.4f}"
                junction = manual_bus(junction_i, junction_name, tee["point"], kv,
                                      bus_df.columns)
                bus_df = pd.concat([bus_df, pd.DataFrame([junction])], ignore_index=True)
                bus_coords_l.append(np.array(tee["point"]))
                bus_voltages_l.append(junction["voltage_tag"])
                valid_bus_i_set.add(junction_i)

                # The host becomes its head half, ending at the junction; a new
                # row carries the tail half from the junction to the host's to-end.
                tail_i = int(df.index.max()) + 1
                df.loc[tail_i] = df.loc[host]
                for i, part, end, flag in ((host, head, "tbus", "t_crossborder"),
                                           (tail_i, tail, "fbus", "f_crossborder")):
                    part_id = f"{host_osm}#{1 if i == host else 2}"
                    geom_lookup[part_id] = part
                    df.at[i, "osm_id"]             = part_id
                    df.at[i, "length_km"]          = round(_line_length_km(part), 4)
                    df.at[i, end]                  = junction_i
                    df.at[i, f"{end}_name"]        = junction_name
                    df.at[i, f"{end}_dist_km"]     = 0.0
                    df.at[i, f"{end}_resolve_pass"] = "tap_split"
                    if flag in df.columns:
                        df.at[i, flag] = False
                    _reshape(i, part)
                # The host's own open to-end, if it had one, is now the tail's.
                for s, (di, _, w, _) in enumerate(unmatched_rows):
                    if di == host and w == "tbus":
                        handled_seqs.add(s)
                if is_null(df.at[tail_i, "tbus"]):
                    unmatched_rows.append((tail_i, df.loc[tail_i], "tbus", tail[-1]))

                df.at[df_i, which_w]                   = junction_i
                df.at[df_i, f"{which_w}_name"]         = junction_name
                df.at[df_i, f"{which_w}_dist_km"]      = round(tee["km"], 4)
                df.at[df_i, f"{which_w}_resolve_pass"] = "manual_tap"
                n_resolved += 1
                handled_seqs.add(seq)
                print(f"  Split {host_osm} at a new {kv:g} kV junction, bus {junction_i}, "
                      f"and assigned {which_w} = {junction_i}")
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

                if not automatic:
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
                    ec = log.ask("  Choice (A/B): ").strip().upper()
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
                _reshape(df_i, merged_coords)
                _reshape(tgt_df_i, None)
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
                    refused = 0
                    if not automatic:
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
                      " skip, delete, new, tap, or merge <N>.")
                continue
            if chosen not in valid_bus_i_set:
                print(f"  bus_i={chosen} not found. Valid range: "
                      f"{min(valid_bus_i_set)}–{max(valid_bus_i_set)}.")
                continue

            # Guard 1 - self-loop. The automated passes pass exclude_pos so the
            # to-bus is never the from-bus; a manual pick had no such check, and
            # Sec. IV.B.2 Step 10 deletes the branch afterwards, losing the line.
            other_bus = df.at[df_i, other]
            if not is_null(other_bus) and int(other_bus) == chosen:
                print(f"  bus {chosen} is already the {other} of this line. That "
                      "would be a self-loop, which Step 10 removes - the line "
                      "would be lost. Pick a different bus, or 'delete' the line.")
                continue

            # Guard 2 - voltage. find_nearest_bus requires the bus voltage to
            # appear in the line's voltage list whenever both are known; a manual
            # pick meets the same bar or is recorded as a deliberate override.
            chosen_bus_v = bus_df.loc[bus_df["bus_i"] == chosen].iloc[0].get("base_kV")
            if not voltage_compatible(line_v, chosen_bus_v):
                print(f"  WARNING: line is {_fmt_kv(line_v)} but bus {chosen} is "
                      f"{_fmt_kv(chosen_bus_v)}. The automated passes reject this.")
                if log.ask("  Assign anyway? (y/N): ").strip().lower() != "y":
                    continue
                print("  Recorded as a deliberate voltage override.")

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
    if automatic:
        # One count per decision: the opening prompt of each carries its context.
        rules = Counter(e["rule"] for e in log.entries
                        if e.get("context") and e.get("rule") and e["rule"] != "deferred")
        print("  Automatic decisions:")
        for rule, count in rules.most_common():
            print(f"    {count:5d}  {rule}")
    return df, bus_df


# ============================================================
# MAIN
# ============================================================

def main(argv=None):
    parser = argparse.ArgumentParser(description="Endpoint review (Sec. IV.B.3)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--replay", action="store_true",
                      help="Re-apply the answers in review_decisions_{region}.json "
                           "instead of prompting, reproducing a saved session")
    mode.add_argument("--auto", action="store_true",
                      help="Answer every prompt by rule instead of asking. The "
                           "result is not a reviewed model; the journal records "
                           "the rule behind each answer")
    parser.add_argument("--journal", type=Path, default=None,
                        help="Where to read/write the review journal")
    args = parser.parse_args([] if argv is None else argv)
    # Verified here rather than at import so the module stays importable.
    build_provenance.verify(config, WORK_DIR, "resolve_unmatched")
    # Resolved here, not at import, so WORK_DIR stays overridable.
    journal = args.journal or (WORK_DIR / f"review_decisions_{region}.json")
    log = ReviewLog(journal, replay=args.replay,
                    mode="automatic" if args.auto else "manual")
    print("=" * 60)
    print("Loading data")
    print("=" * 60)

    lines_df = pd.read_csv(WORK_DIR / f"matpower_{region}_lines.csv")
    bus_df   = pd.read_csv(WORK_DIR / f"matpower_{region}_buses.csv")
    proc_df  = pd.read_csv(WORK_DIR / f"processed_{region}_lines.csv")
    validate_endpoints(bus_df, read_transformers(WORK_DIR / f"transformer_pairs_{region}.csv"))

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
    print("PASS 1 – " + ("Automatic review (not a reviewed model)"
                         if log.mode == "automatic" else "Interactive manual review"))
    print("=" * 60)
    lines_df, bus_df = run_pass1(
        lines_df, geom_lookup, bus_df, bus_coords, bus_voltages, log=log)

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


    raised, worst, added = floor_lengths_at_bus_span(lines_df, bus_df)
    if raised:
        print(f"  Raised {raised} line length(s) to the distance between their own "
              f"substations, adding {added:,.0f} km.")
        print(f"  The most understated was at {worst:.4f} of its floor; equation (3) "
              f"would have given it that fraction of its impedance.")

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
    log.save(region)

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
            # The rename just invalidated every endpoint name pointing at these
            # buses; leaving them stale fails validate_endpoints in Step 8.
            refresh_endpoint_names(bus_df, lines_resolved)

    bus_df.to_csv(WORK_DIR / f"matpower_{region}_buses.csv", index=False)
    print(f"  Updated buses → {WORK_DIR / f'matpower_{region}_buses.csv'}")


    # ── POST-PROCESSING 1b: Split substations carrying several line voltages ─────
    print("\n" + "=" * 60)
    print("POST-PROCESSING 1b: Splitting multi-voltage substations")
    print("=" * 60)

    xfmr_path = WORK_DIR / f"transformer_pairs_{region}.csv"
    transformers = read_transformers(xfmr_path)
    bus_df, lines_resolved, transformers, n_split = split_multi_voltage_buses(
        bus_df, lines_resolved, transformers)
    print(f"  Split {n_split} substation(s) into voltage tiers joined by transformers.")
    bus_df, lines_resolved, n_moved = retier_endpoints(bus_df, lines_resolved)
    print(f"  Moved {n_moved} endpoint(s) onto the matching tier of their substation.")
    refresh_endpoint_names(bus_df, lines_resolved, transformers)
    bus_df.to_csv(WORK_DIR / f"matpower_{region}_buses.csv", index=False)
    lines_resolved.to_csv(WORK_DIR / f"matpower_{region}_lines_resolved.csv", index=False)
    transformers.to_csv(xfmr_path, index=False)

    # ── POST-PROCESSING 2: Remove island buses that remain unconnected ────────────
    print("\n" + "=" * 60)
    print("POST-PROCESSING 2: Removing remaining island buses")
    print("=" * 60)

    xfmr_path = WORK_DIR / f"transformer_pairs_{region}.csv"
    transformers = read_transformers(xfmr_path)
    bus_df, lines_resolved, transformers = clean_bus_tables(bus_df, lines_resolved, transformers)
    lines_resolved.to_csv(WORK_DIR / f"matpower_{region}_lines_resolved.csv", index=False)
    transformers.to_csv(xfmr_path, index=False)
    bus_df.to_csv(WORK_DIR / f"matpower_{region}_buses.csv", index=False)
    print(f"  Final bus count: {len(bus_df)}")

    # ── POST-PROCESSING 3: Close gaps between separate networks ──────────────────
    # A national transmission grid is normally one synchronous system, so a model
    # in pieces usually means OSM never mapped the line that joins them, not that
    # the country is islanded. Sec. IV.B.3 puts a reviewer here for exactly this
    # kind of gap, but until now the review could only assign the endpoints of
    # lines OSM already had; it had no way to supply a line OSM was missing.
    if getattr(config, "EXPECT_CONNECTED_GRID", True):
        print("\n" + "=" * 60)
        print("POST-PROCESSING 3: Connecting separate networks")
        print("=" * 60)
        for _ in range(len(bus_df)):
            report = island_report(bus_df, lines_resolved, transformers)
            if report["components"] <= 1:
                break
            print(describe_islands(report, expect_connected=False))
            if log.mode == "automatic":
                # Unattended, only a short hop between buses of one voltage is
                # added; a longer gap or a voltage change is left for a reviewer.
                island = next((i for i in report["islands"]
                               if i["nearest_from"] is not None and i["voltage_compatible"]
                               and i["gap_km"] <= AUTO_CONNECT_KM), None)
                if island is None:
                    print("  No remaining gap is a voltage-compatible hop within "
                          f"{AUTO_CONNECT_KM:g} km; left for review.")
                    break
            else:
                island = report["islands"][0]
                if island["nearest_from"] is None:
                    break
            a, b = island["nearest_from"], island["nearest_to"]
            rows = bus_df.set_index("bus_i")
            kv = normalise_kv(rows.at[a, "base_kV"], rows.at[b, "base_kV"])
            note = "" if island["voltage_compatible"] else "  [VOLTAGES DIFFER]"
            if log.mode == "automatic" and not log.replay:
                log.queue(["y"], "join networks",
                          f"{island['buses']}-bus network, {island['gap_km']:.2f} km hop")
            answer = log.ask(
                f"\n  Add a {kv:g} kV line to join the {island['buses']}-bus network"
                f" — bus {a} ({rows.at[a, 'name']}) to bus {b} ({rows.at[b, 'name']}),"
                f" {island['gap_km']:.2f} km{note}?  (y / skip): ",
                context=dict(island_buses=island["buses"], fbus=a, tbus=b,
                             gap_km=island["gap_km"],
                             voltage_compatible=island["voltage_compatible"]))
            if str(answer).strip().lower() not in ("y", "yes"):
                print("  Left separate. The model will contain more than one network.")
                break
            new = {c: 0 for c in lines_resolved.columns}
            new.update(fbus=a, tbus=b, ratio=0, angle=0, status=1,
                       angmin=-config.BRANCH_ANGLE_LIMIT_DEG,
                       angmax=config.BRANCH_ANGLE_LIMIT_DEG,
                       fbus_name=rows.at[a, "name"], tbus_name=rows.at[b, "name"],
                       fbus_dist_km=0.0, tbus_dist_km=0.0,
                       f_crossborder=False, t_crossborder=False,
                       length_km=island["gap_km"], voltage_V=kv * 1000.0,
                       osm_id="review:connect")
            for column in ("fbus_resolve_pass", "tbus_resolve_pass"):
                if column in new:
                    new[column] = "connect"
            lines_resolved = pd.concat(
                [lines_resolved, pd.DataFrame([new])], ignore_index=True)
            _apply_rxb(lines_resolved, len(lines_resolved) - 1, kv, island["gap_km"])
            print(f"  Added line bus {a} — bus {b} ({island['gap_km']:.2f} km, {kv:g} kV).")
        final = island_report(bus_df, lines_resolved, transformers)
        verdict = describe_islands(final)
        print("\n  " + (verdict.replace("\n", "\n  ") if verdict
                        else "One connected network."))
        lines_resolved.to_csv(WORK_DIR / f"matpower_{region}_lines_resolved.csv", index=False)
        log.save(region)

    log.finish()


if __name__ == "__main__":
    main(sys.argv[1:])