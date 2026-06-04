"""
llm_assignment.py

LLM-based intermediate assignment pass between the automated pipeline
(create_lines_new.py) and the human review pass (resolve_unmatched.py).

Pipeline position:
  create_lines_new.py  →  llm_assignment.py  →  resolve_unmatched.py

For each still-unmatched non-cross-border endpoint the script:

  Step 1 – Trajectory extrapolation (optional, fast geometric pre-pass)
            Project the line's end bearing forward and auto-accept any bus
            whose cross-track distance to that ray is ≤ CROSS_TRACK_KM.
            Set RUN_TRAJECTORY_PREPASS = False to skip this.

  Step 2 – Claude Vision
            For each endpoint that is still unmatched after Step 1,
            render a map image and query Claude for one of four actions:

              assign  – line trajectory clearly targets a candidate bus
              delete  – line is a misread artefact (busbar stub, short noise, etc.)
              merge   – line fragment should be joined to another unmatched segment
              manual  – ambiguous; hand off to resolve_unmatched.py

            Auto-acceptance policy:
              assign  high confidence   → applied; resolve_pass = 'llm_high'
              assign  medium confidence → applied; resolve_pass = 'llm_medium'
              assign  low confidence    → NOT applied; marked 'manual'
              delete                    → flagged only (human confirms in Pass 4)
              merge                     → flagged with target info (human confirms)
              manual                    → passed through unchanged

Output columns added to the CSV:
  {f/t}bus_llm_action          assign | delete | merge | manual
  {f/t}bus_llm_confidence      high | medium | low
  {f/t}bus_llm_reasoning       one-sentence explanation from Claude
  {f/t}bus_llm_merge_target    "<osm_id>:<which>" when action=merge

Prerequisites
-------------
  pip install anthropic
  ANTHROPIC_API_KEY must be set in the environment.

Reads
-----
  matpower_{region}_lines.csv   – output of create_lines_new.py (Step 10)
  processed_{region}_lines.csv  – output of create_lines_new.py (Step 8)
  matpower_{region}_buses.csv

Writes
------
  matpower_{region}_lines_llm.csv   (resolve_unmatched.py auto-detects this file)
"""

import base64
import io
import json
import math
import os

import anthropic
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from shapely import wkt as shapely_wkt


# ============================================================
# CONFIGURATION
# ============================================================

region = "ghana"

# Step 1 – trajectory extrapolation (set False to skip)
RUN_TRAJECTORY_PREPASS   = False
TRAJECTORY_SEARCH_KM     = 50.0   # candidate search radius
TRAJECTORY_N_TAIL        = 5      # coordinates used for bearing
PROJECTION_KM            = 100.0  # bearing ray length
CROSS_TRACK_THRESHOLD_KM = 8.0    # max perpendicular distance to auto-accept

# Step 2 – Claude Vision
CLAUDE_MODEL     = "claude-opus-4-7"
SEARCH_RADIUS_KM = 60.0    # buses and unmatched endpoints shown in image
IMAGE_DPI        = 130
IMAGE_SIZE_IN    = (9, 9)

# ============================================================
# HELPERS  (bearing math, voltage parsing, candidate search)
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


def voltage_compatible(line_v, bus_v) -> bool:
    lv = parse_voltage_tag(line_v)
    bv = parse_voltage_tag(bus_v)
    if not lv or not bv:
        return True
    return bool(set(lv) & set(bv))


def is_null(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def compute_bearing(lon1, lat1, lon2, lat2) -> float:
    dlon = math.radians(lon2 - lon1)
    lr1, lr2 = math.radians(lat1), math.radians(lat2)
    x = math.sin(dlon) * math.cos(lr2)
    y = math.cos(lr1) * math.sin(lr2) - math.sin(lr1) * math.cos(lr2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def project_point(lon, lat, bearing_deg, distance_km) -> tuple:
    R = 6371.0
    d = distance_km / R
    b = math.radians(bearing_deg)
    la1, lo1 = math.radians(lat), math.radians(lon)
    la2 = math.asin(math.sin(la1) * math.cos(d)
                    + math.cos(la1) * math.sin(d) * math.cos(b))
    lo2 = lo1 + math.atan2(
        math.sin(b) * math.sin(d) * math.cos(la1),
        math.cos(d) - math.sin(la1) * math.sin(la2))
    return math.degrees(lo2), math.degrees(la2)


def cross_track_km(endpoint, projected, bus) -> float:
    R = 6371.0
    la1, lo1 = math.radians(endpoint[1]),  math.radians(endpoint[0])
    la2, lo2 = math.radians(projected[1]), math.radians(projected[0])
    la3, lo3 = math.radians(bus[1]),       math.radians(bus[0])
    cos_d13 = max(-1.0, min(1.0,
        math.sin(la1) * math.sin(la3)
        + math.cos(la1) * math.cos(la3) * math.cos(lo3 - lo1)))
    d13     = math.acos(cos_d13)
    theta12 = math.atan2(
        math.sin(lo2 - lo1) * math.cos(la2),
        math.cos(la1) * math.sin(la2)
        - math.sin(la1) * math.cos(la2) * math.cos(lo2 - lo1))
    theta13 = math.atan2(
        math.sin(lo3 - lo1) * math.cos(la3),
        math.cos(la1) * math.sin(la3)
        - math.sin(la1) * math.cos(la3) * math.cos(lo3 - lo1))
    return abs(math.asin(math.sin(d13) * math.sin(theta13 - theta12))) * R


def is_ahead(endpoint, projected, bus) -> bool:
    dx_p = projected[0] - endpoint[0]
    dy_p = projected[1] - endpoint[1]
    return (dx_p * (bus[0] - endpoint[0]) + dy_p * (bus[1] - endpoint[1])) > 0


def endpoint_and_bearing(coords, which) -> tuple:
    n = TRAJECTORY_N_TAIL
    if which == "tbus":
        tail = coords[-min(n, len(coords)):]
        ep = tail[-1]
        bearing = compute_bearing(tail[-2][0], tail[-2][1],
                                  tail[-1][0], tail[-1][1]) if len(tail) >= 2 else 0.0
    else:
        head = coords[:min(n, len(coords))]
        ep = head[0]
        bearing = compute_bearing(head[1][0], head[1][1],
                                  head[0][0], head[0][1]) if len(head) >= 2 else 0.0
    return ep, bearing


def nearby_buses(endpoint, line_v, bus_df, bus_coords, bus_voltages,
                 radius_km) -> list:
    """Return [(bus_pos, dist_km, bus_i, name, voltage_raw)]."""
    out = []
    for k, (bc, bv) in enumerate(zip(bus_coords, bus_voltages)):
        dist = geodesic((endpoint[1], endpoint[0]), (bc[1], bc[0])).km
        if dist <= radius_km and voltage_compatible(line_v, bv):
            out.append((k, dist, int(bus_df.iloc[k]["bus_i"]),
                        str(bus_df.iloc[k].get("name", "")), bv))
    out.sort(key=lambda c: c[1])
    return out


# ============================================================
# STEP 1 – TRAJECTORY EXTRAPOLATION
# ============================================================

def run_trajectory_prepass(df, geom_lookup, bus_df, bus_coords, bus_voltages):
    """
    Quick geometric pass: project each unmatched endpoint's bearing forward
    and auto-accept the bus with the smallest cross-track distance to that ray.
    Much faster and cheaper than an LLM call; narrows down what Claude needs to see.
    """
    n_resolved = 0
    for i, row in df.iterrows():
        for which in ("fbus", "tbus"):
            cross_flag = "f_crossborder" if which == "fbus" else "t_crossborder"
            if not is_null(row[which]):
                continue
            if str(row.get(cross_flag, False)).lower() == "true":
                continue

            osm_id = str(row["osm_id"])
            coords = geom_lookup.get(osm_id)
            if not coords or len(coords) < 2:
                continue

            ep, bearing = endpoint_and_bearing(coords, which)
            proj = project_point(ep[0], ep[1], bearing, PROJECTION_KM)
            candidates = nearby_buses(ep, row.get("voltage_V"), bus_df,
                                      bus_coords, bus_voltages, TRAJECTORY_SEARCH_KM)

            best = None
            for bus_pos, dist_km, bus_i_val, name, _ in candidates:
                bc = (bus_coords[bus_pos][0], bus_coords[bus_pos][1])
                if not is_ahead(ep, proj, bc):
                    continue
                ct = cross_track_km(ep, proj, bc)
                if ct > CROSS_TRACK_THRESHOLD_KM:
                    continue
                if best is None or ct < best[0]:
                    best = (ct, dist_km, bus_pos, bus_i_val, name)

            if best is None:
                continue
            ct, dist_km, bus_pos, bus_i_val, name = best
            df.at[i, which]                   = bus_i_val
            df.at[i, f"{which}_name"]         = name
            df.at[i, f"{which}_dist_km"]      = round(dist_km, 4)
            df.at[i, f"{which}_resolve_pass"] = "trajectory"
            n_resolved += 1
            print(f"  [traj {which}] row {i}  → bus {bus_i_val} ({name})"
                  f"  cross-track={ct:.2f} km")
    print(f"  Trajectory pre-pass resolved {n_resolved} endpoint(s).")
    return df


# ============================================================
# STEP 2 – CLAUDE VISION
# ============================================================

def _build_image(
    coords,
    endpoint,
    projected,
    bus_candidates,     # [(bus_pos, dist, bus_i, name, v), ...]  numbered 1..N
    nearby_unmatched,   # [(df_i, which, ep, osm_id, v), ...]  lettered A..Z
    bus_coords,
    which,
    osm_id,
    line_v,
) -> str:
    """
    Render a map image and return it as a base64-encoded PNG string.

    Blue numbered circles  = candidate substations (action: assign)
    Orange lettered diamonds = nearby unmatched endpoints (action: merge)
    Red dashed ray = projected bearing direction
    Red × = unmatched endpoint being reviewed
    """
    fig, ax = plt.subplots(figsize=IMAGE_SIZE_IN, dpi=IMAGE_DPI)

    # --- Line geometry ---
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    ax.plot(lons, lats, color="#c0392b", linewidth=2.2, zorder=3,
            label="Line segment")

    # --- Projected bearing ray ---
    ax.annotate("",
        xy=(projected[0], projected[1]),
        xytext=(endpoint[0], endpoint[1]),
        arrowprops=dict(arrowstyle="-|>", color="#c0392b",
                        lw=1.2, linestyle="dashed"),
        zorder=3)

    # --- Unmatched endpoint ---
    ax.plot(endpoint[0], endpoint[1], "rx", markersize=16,
            markeredgewidth=2.8, zorder=6, label="Unmatched endpoint (×)")

    # --- Candidate buses (blue numbered circles) ---
    for n, (bus_pos, dist_km, bus_i_val, name, bv) in enumerate(bus_candidates, 1):
        blon, blat = bus_coords[bus_pos]
        ax.plot(blon, blat, "o", color="#1f77b4", markersize=13,
                zorder=4, markeredgecolor="white", markeredgewidth=1.2)
        ax.text(blon, blat, str(n), fontsize=8, color="white",
                ha="center", va="center", fontweight="bold", zorder=5)
        v_str = (f"{parse_voltage_tag(bv)[0] // 1000} kV"
                 if parse_voltage_tag(bv) else "?")
        ax.annotate(f"  {name}\n  {v_str}",
                    xy=(blon, blat), fontsize=6.5, color="#1f4e79",
                    va="bottom", zorder=4)

    # --- Nearby unmatched endpoints (orange lettered diamonds) ---
    for idx, (_, _, u_ep, u_osm_id, u_v, *_) in enumerate(nearby_unmatched):
        letter = chr(65 + idx)   # A, B, C …
        u_v_str = (f"{parse_voltage_tag(u_v)[0] // 1000} kV"
                   if parse_voltage_tag(u_v) else "?")
        ax.plot(u_ep[0], u_ep[1], "D", color="#e67e22", markersize=11,
                zorder=4, markeredgecolor="white", markeredgewidth=1.0)
        ax.text(u_ep[0], u_ep[1], letter, fontsize=8, color="white",
                ha="center", va="center", fontweight="bold", zorder=5)
        ax.annotate(f"  {u_osm_id[:12]}\n  {u_v_str}",
                    xy=(u_ep[0], u_ep[1]), fontsize=6, color="#784212",
                    va="bottom", zorder=4)

    # --- Axis limits: encompass everything with 15 % padding ---
    all_lons = (lons + [projected[0]]
                + [bus_coords[c[0]][0] for c in bus_candidates]
                + [u[2][0] for u in nearby_unmatched])
    all_lats = (lats + [projected[1]]
                + [bus_coords[c[0]][1] for c in bus_candidates]
                + [u[2][1] for u in nearby_unmatched])
    lon_pad = (max(all_lons) - min(all_lons)) * 0.15 + 0.05
    lat_pad = (max(all_lats) - min(all_lats)) * 0.15 + 0.05
    ax.set_xlim(min(all_lons) - lon_pad, max(all_lons) + lon_pad)
    ax.set_ylim(min(all_lats) - lat_pad, max(all_lats) + lat_pad)

    ep_label = "from-end" if which == "fbus" else "to-end"
    v_list   = parse_voltage_tag(line_v)
    v_str    = f"{v_list[0] // 1000} kV" if v_list else "unknown voltage"
    ax.set_xlabel("Longitude");  ax.set_ylabel("Latitude")
    ax.set_title(
        f"Unmatched {ep_label}  |  OSM: {osm_id}  |  {v_str}\n"
        f"Blue ● = candidate buses (numbered)   "
        f"Orange ◆ = other unmatched endpoints (lettered)",
        fontsize=8.5)

    legend_patches = [
        mpatches.Patch(color="#c0392b", label="Line (solid) + projected ray (dashed)"),
        mpatches.Patch(color="#1f77b4", label="Candidate buses  (assign: use number)"),
        mpatches.Patch(color="#e67e22", label="Unmatched ends   (merge: use letter)"),
    ]
    ax.legend(handles=legend_patches, fontsize=7, loc="lower left")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.standard_b64encode(buf.getvalue()).decode()


# ---- Prompt ----------------------------------------------------------------

_PROMPT_TEMPLATE = """\
You are an expert power-systems GIS analyst reviewing a transmission network \
map for {region}.

═══ WHAT YOU SEE ═══
• Solid red line   – the transmission segment being examined
• Dashed red arrow – the segment's projected bearing if extended further
• Red ×            – the UNMATCHED endpoint (needs a substation assignment)
• Blue circles (numbered 1–{n_buses})  – candidate substations nearby
• Orange diamonds  (lettered A–{last_letter}) – other unmatched line endpoints nearby

═══ CANDIDATE SUBSTATIONS (blue circles) ═══
{candidate_table}

═══ NEARBY UNMATCHED ENDPOINTS (orange diamonds) ═══
{unmatched_table}

═══ YOUR TASK: choose exactly one of the four actions below ═══

  ASSIGN  (bus_candidate = number 1–{n_buses})
    When the dashed arrow clearly points toward a blue circle.
    The substation must be spatially consistent with the line direction.
    The NEAREST bus is NOT automatically the right one — check the trajectory.

  DELETE  (bus_candidate = null, merge_target = null)
    Only when the line is clearly NOT a real inter-substation transmission line:
      ✓ Very short stub with no clear direction (internal busbar / OSM noise)
      ✓ Projected trajectory does not approach ANY candidate substation
      ✓ Line appears entirely enclosed within a substation compound
    Do NOT delete because assignment is uncertain — use MANUAL for uncertainty.

  MERGE   (merge_target = letter A–{last_letter})
    When an orange diamond lies directly in the line's projected path,
    suggesting the two segments are parts of the same physical line.

  MANUAL  (bus_candidate = null, merge_target = null)
    When you are uncertain, the trajectory is ambiguous, or the correct
    substation is likely not in the candidate list.
    PREFER manual over a low-confidence assignment — a wrong assignment
    corrupts the power flow model and is worse than flagging for human review.

═══ CONFIDENCE ═══
  high   → you are certain
  medium → a reasonable call but not definitive
  low    → educated guess; only use with action=manual or action=delete

Respond with ONLY this JSON on a single line — no other text:
{{"action":"assign|delete|merge|manual","bus_candidate":<1..{n_buses} or null>,\
"merge_target":"<A..{last_letter} or null>","confidence":"high|medium|low",\
"reasoning":"<one or two sentences>"}}
"""


# Accumulates token usage across all Claude calls in this run.
_token_usage: dict = {"input": 0, "output": 0, "calls": 0}


def _query_claude(image_b64, bus_candidates, nearby_unmatched,
                  which, osm_id) -> dict:
    client = anthropic.Anthropic()

    n_buses     = len(bus_candidates)
    last_letter = chr(65 + len(nearby_unmatched) - 1) if nearby_unmatched else "–"

    cand_rows = "\n".join(
        f"  {n}. bus_i={bi:<5d}  {nm:<35s}  "
        f"{'  /  '.join(str(v // 1000) + ' kV' for v in parse_voltage_tag(bv)) or '?':12s}"
        f"  {dist:.1f} km"
        for n, (_, dist, bi, nm, bv) in enumerate(bus_candidates, 1)
    ) or "  (none within search radius)"

    unm_rows = "\n".join(
        f"  {chr(65 + j)}. osm_id={u_osm[:20]:<22s}  {dist:.1f} km  "
        f"{'  /  '.join(str(v // 1000) + ' kV' for v in parse_voltage_tag(u_v)) or '?'}"
        for j, (_, _, u_ep, u_osm, u_v, dist) in enumerate(nearby_unmatched)
    ) or "  (none within search radius)"

    prompt = _PROMPT_TEMPLATE.format(
        region=region,
        n_buses=n_buses,
        last_letter=last_letter,
        candidate_table=cand_rows,
        unmatched_table=unm_rows,
    )

    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )

    _token_usage["input"]  += message.usage.input_tokens
    _token_usage["output"] += message.usage.output_tokens
    _token_usage["calls"]  += 1
    print(f"[tokens] in={message.usage.input_tokens}  "
          f"out={message.usage.output_tokens}  "
          f"(total in={_token_usage['input']}  out={_token_usage['output']})")

    raw   = message.content[0].text.strip()
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start == -1 or end == 0:
        return {"action": "manual", "bus_candidate": None,
                "merge_target": None, "confidence": "low",
                "reasoning": f"Unparseable response: {raw[:200]}"}
    try:
        return json.loads(raw[start:end])
    except json.JSONDecodeError:
        return {"action": "manual", "bus_candidate": None,
                "merge_target": None, "confidence": "low",
                "reasoning": f"JSON error: {raw[:200]}"}


# ============================================================
# MAIN
# ============================================================

print("=" * 60)
print("Loading data")
print("=" * 60)

lines_df = pd.read_csv(f"matpower_{region}_lines.csv")
bus_df   = pd.read_csv(f"matpower_{region}_buses.csv")
proc_df  = pd.read_csv(f"processed_{region}_lines.csv")

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
        geom_lookup[str(prow["id"])] = list(shapely_wkt.loads(str(raw_geom)).coords)
    except Exception:
        pass

# Add LLM tracking columns
for col in ("fbus_llm_action", "fbus_llm_confidence",
            "fbus_llm_reasoning", "fbus_llm_merge_target",
            "tbus_llm_action", "tbus_llm_confidence",
            "tbus_llm_reasoning", "tbus_llm_merge_target"):
    lines_df[col] = None

if "fbus_resolve_pass" not in lines_df.columns:
    lines_df["fbus_resolve_pass"] = None
if "tbus_resolve_pass" not in lines_df.columns:
    lines_df["tbus_resolve_pass"] = None

n0_f = int(lines_df["fbus"].isna().sum())
n0_t = int(lines_df["tbus"].isna().sum())
print(f"  {len(lines_df)} lines  |  {n0_f} missing fbus  |  {n0_t} missing tbus")

# ── Step 1: Trajectory pre-pass ──────────────────────────────
if RUN_TRAJECTORY_PREPASS:
    print("\n" + "=" * 60)
    print("STEP 1 – Trajectory extrapolation pre-pass")
    print("=" * 60)
    lines_df = run_trajectory_prepass(
        lines_df, geom_lookup, bus_df, bus_coords, bus_voltages)

# ── Step 2: Claude Vision ─────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 – Claude Vision assignment")
print("=" * 60)

api_key = os.environ.get("ANTHROPIC_API_KEY", "")
if not api_key:
    print("  ANTHROPIC_API_KEY not set – skipping Claude Vision step.")
else:
    # Collect all still-unmatched endpoints and their coordinates
    # so we can show nearby unmatched segments as merge candidates.
    unmatched_eps: dict = {}   # (df_i, which) → (ep_lonlat, osm_id, line_v)
    for i, row in lines_df.iterrows():
        for which in ("fbus", "tbus"):
            cross_flag = "f_crossborder" if which == "fbus" else "t_crossborder"
            if not is_null(row[which]):
                continue
            if str(row.get(cross_flag, False)).lower() == "true":
                continue
            osm_id = str(row["osm_id"])
            coords = geom_lookup.get(osm_id, [])
            if not coords:
                continue
            ep = coords[0] if which == "fbus" else coords[-1]
            unmatched_eps[(i, which)] = (ep, osm_id, row.get("voltage_V", None))

    n_total = len(unmatched_eps)
    print(f"  {n_total} endpoint(s) to review with Claude")

    n_assigned  = 0
    n_deleted   = 0
    n_merged    = 0
    n_manual    = 0

    # Work through a snapshot of the keys (we may pop entries as they're handled)
    for seq, ((df_i, which), (ep, osm_id, line_v)) in enumerate(
            list(unmatched_eps.items()), 1):

        if is_null(lines_df.at[df_i, which]) is False:
            # Another endpoint of the same line may have been resolved
            # (or the dict was built before a merge update) — skip
            pass  # still review; the column may have changed

        coords = geom_lookup.get(osm_id, [])
        if not coords or len(coords) < 2:
            continue

        print(f"\n  [{seq}/{n_total}] osm_id={osm_id}  {which}  "
              f"voltage={line_v}", end="  ", flush=True)

        # --- Candidate buses ---
        bus_cands = nearby_buses(ep, line_v, bus_df, bus_coords, bus_voltages,
                                 SEARCH_RADIUS_KM)

        # --- Nearby other unmatched endpoints (for merge) ---
        nearby_unm = []
        for (other_i, other_which), (other_ep, other_osm, other_v) in \
                unmatched_eps.items():
            if (other_i, other_which) == (df_i, which):
                continue
            dist = geodesic((ep[1], ep[0]), (other_ep[1], other_ep[0])).km
            if dist <= SEARCH_RADIUS_KM:
                nearby_unm.append(
                    (other_i, other_which, other_ep, other_osm, other_v, dist))
        nearby_unm.sort(key=lambda x: x[5])

        # --- Build image ---
        ep_coord, bearing = endpoint_and_bearing(coords, which)
        proj = project_point(ep_coord[0], ep_coord[1], bearing, PROJECTION_KM)

        image_b64 = _build_image(
            coords, ep_coord, proj, bus_cands, nearby_unm,
            bus_coords, which, osm_id, line_v)

        # --- Query Claude ---
        result     = _query_claude(image_b64, bus_cands, nearby_unm, which, osm_id)
        action     = result.get("action", "manual")
        confidence = result.get("confidence", "low")
        reasoning  = result.get("reasoning", "")
        bus_num    = result.get("bus_candidate")
        merge_ltr  = result.get("merge_target")

        print(f"→ {action} ({confidence})")
        print(f"    {reasoning}")

        # Store LLM metadata
        lines_df.at[df_i, f"{which}_llm_action"]     = action
        lines_df.at[df_i, f"{which}_llm_confidence"] = confidence
        lines_df.at[df_i, f"{which}_llm_reasoning"]  = reasoning

        # --- Apply the result ---
        if action == "assign" and confidence in ("high", "medium"):
            if bus_num is not None and 1 <= int(bus_num) <= len(bus_cands):
                bus_pos, dist_km, bus_i_val, name, _ = bus_cands[int(bus_num) - 1]
                lines_df.at[df_i, which]               = bus_i_val
                lines_df.at[df_i, f"{which}_name"]     = name
                lines_df.at[df_i, f"{which}_dist_km"]  = round(dist_km, 4)
                lines_df.at[df_i, f"{which}_resolve_pass"] = f"llm_{confidence}"
                # Remove from unmatched pool so it doesn't appear in later images
                unmatched_eps.pop((df_i, which), None)
                n_assigned += 1
            else:
                lines_df.at[df_i, f"{which}_llm_action"] = "manual"
                n_manual += 1

        elif action == "delete":
            # Flag only — human confirms with 'delete' in resolve_unmatched.py
            n_deleted += 1

        elif action == "merge" and merge_ltr:
            merge_idx = ord(str(merge_ltr).upper()) - ord("A")
            if 0 <= merge_idx < len(nearby_unm):
                tgt_i, tgt_which, _, tgt_osm, _, _ = nearby_unm[merge_idx]
                lines_df.at[df_i, f"{which}_llm_merge_target"] = (
                    f"{tgt_osm}:{tgt_which}")
                n_merged += 1
            else:
                lines_df.at[df_i, f"{which}_llm_action"] = "manual"
                n_manual += 1

        else:
            n_manual += 1

    print(f"\n  Claude Vision summary:")
    print(f"    assigned       : {n_assigned}")
    print(f"    delete-flagged : {n_deleted}")
    print(f"    merge-flagged  : {n_merged}")
    print(f"    manual         : {n_manual}")
    print(f"\n  Token usage ({_token_usage['calls']} API call(s)):")
    print(f"    input  tokens : {_token_usage['input']:,}")
    print(f"    output tokens : {_token_usage['output']:,}")
    # Approximate cost — update rates if Anthropic pricing changes.
    # claude-opus-4-7: $15 / 1M input,  $75 / 1M output
    _cost = (_token_usage["input"] * 15 + _token_usage["output"] * 75) / 1_000_000
    print(f"    est. cost     : ${_cost:.4f}  (Opus 4 rates: $15/$75 per MTok)")

# ── Save ─────────────────────────────────────────────────────
out_path = f"matpower_{region}_lines_llm.csv"
lines_df.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}")
print("Run resolve_unmatched.py next — it will auto-detect this file.")

still_f = int(lines_df["fbus"].isna().sum())
still_t = int(lines_df["tbus"].isna().sum())
print(f"Remaining unmatched: {still_f} fbus,  {still_t} tbus  "
      f"(flagged lines still need human confirmation)")
