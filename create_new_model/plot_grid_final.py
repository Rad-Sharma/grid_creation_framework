"""
plot_grid_final.py

Plots the synthetic high-voltage transmission network on a country map.

Data sources
------------
* matpower_{region}_lines_resolved.csv — the final kept lines (post manual review).
  Only lines with valid fbus AND tbus are drawn; deleted lines are never shown.
* processed_{region}_lines.csv         — actual OSM geometry (WKT) for each line,
  matched to resolved lines by osm_id.  Lines whose geometry cannot be found
  (e.g. manually merged routes) fall back to a straight segment between buses.
* matpower_{region}_buses.csv          — bus coordinates and voltage tags.
  Buses with no connected line (islands) are not drawn.

Voltage class for unknown lines:  the most common (mode) voltage among all
resolved lines with a known voltage is assigned to the remaining ones.

Output
------
  {root}/grid_plots/grid_{region}.png
"""

import re
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator
from shapely import wkt as shapely_wkt
import numpy as np
from pathlib import Path

import config

REGION          = config.region
SHAPE_FILE_NAME = config.SHAPE_FILE_NAME

PROJ_ROOT  = Path(__file__).parent.parent
WORK_DIR   = PROJ_ROOT / "data" / "working_files" / REGION
PLOTS_DIR  = PROJ_ROOT / "grid_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RESOLVED_CSV  = WORK_DIR / f"matpower_{REGION}_lines_resolved.csv"
PROCESSED_CSV = WORK_DIR / f"processed_{REGION}_lines.csv"
BUSES_CSV     = WORK_DIR / f"matpower_{REGION}_buses.csv"
OUTPUT        = PLOTS_DIR / f"grid_{REGION}.png"


# ── Label cleaning ────────────────────────────────────────────────────────────

def clean_label(name: str) -> str:
    """Shorten a bus name for map display.

    'Salto Substation 2_150kV' → 'Salto 2'
    'Artigas_220kV'            → 'Artigas'
    """
    s = str(name)
    s = re.sub(r'[_\s]+\d+\s*kV$', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\s+[Ss]ubstation\b', '', s)
    return s.strip()


# ── Voltage class helpers ─────────────────────────────────────────────────────

_PALETTE = ["#C0392B","#E67E22","#2980B9","#27AE60","#8E44AD","#95A5A6"]


def _kv_class(voltage_volts: float) -> int:
    if np.isnan(voltage_volts) or voltage_volts <= 0:
        return 0
    kv = voltage_volts / 1000.0
    return int(round(kv / 10) * 10)


def dominant_kv(tag) -> int:
    if pd.isna(tag):
        return 0
    vals = []
    for part in str(tag).split(";"):
        try:
            vals.append(int(float(part.strip())))
        except ValueError:
            pass
    return _kv_class(float(max(vals))) if vals else 0


def line_kv(voltage_val) -> int:
    if pd.isna(voltage_val):
        return 0
    try:
        return _kv_class(float(voltage_val))
    except Exception:
        vals = []
        for part in str(voltage_val).split(";"):
            try:
                vals.append(float(part.strip()))
            except Exception:
                continue
        return _kv_class(float(max(vals))) if vals else 0


def build_voltage_styles(kv_classes: list) -> tuple[dict, dict]:
    non_zero = sorted([k for k in kv_classes if k > 0], reverse=True)
    classes  = non_zero + ([0] if 0 in kv_classes else [])
    line_styles: dict = {}
    bus_styles:  dict = {}
    markers = ["o","D","s","^","P","X"]
    for i, kv in enumerate(classes):
        color = _PALETTE[min(i, len(_PALETTE)-1)]
        lw    = max(0.5, 2.0 - 0.3*i)
        ls    = "--" if kv == 0 else "-"
        line_styles[kv] = dict(color=color, lw=lw, zorder=len(classes)-i+1,
                                label=f"{kv} kV", ls=ls)
        bus_styles[kv]  = dict(marker=markers[i % len(markers)], c=color,
                                s=max(20, 90-15*i), zorder=len(classes)-i+5,
                                linewidths=0.6, edgecolors="white")
    return line_styles, bus_styles


# ── Scale bar / north arrow ───────────────────────────────────────────────────

def add_scale_bar(ax, length_deg=1.0, location=(0.07, 0.06)):
    x0, y0 = ax.transData.inverted().transform(ax.transAxes.transform(location))
    x1 = x0 + length_deg
    for xs, ys in [([x0,x1],[y0,y0]),([x0,x0],[y0-.04,y0+.04]),([x1,x1],[y0-.04,y0+.04])]:
        ax.plot(xs, ys, color="black", lw=2.5, solid_capstyle="butt",
                transform=ax.transData, clip_on=False)
    ax.text((x0+x1)/2, y0-.14, f"≈ {int(length_deg*111)} km",
            ha="center", va="top", fontsize=7.5, transform=ax.transData)


def add_north_arrow(ax, location=(0.95, 0.10), size=0.045):
    x, y = location
    ax.annotate("N", xy=(x, y+size), xycoords="axes fraction",
                fontsize=9, fontweight="bold", ha="center", va="bottom")
    ax.annotate("", xy=(x, y+size*.9), xytext=(x, y),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))


# ── Load geographic base map ──────────────────────────────────────────────────

try:
    import pyogrio
    _ne_path = (str(pyogrio.__file__).replace("__init__.py","")
                + "tests/fixtures/naturalearth_lowres/naturalearth_lowres.shp")
    world = gpd.read_file(_ne_path)
except Exception:
    world = gpd.read_file(
        "https://naturalearth.s3.amazonaws.com/110m_cultural/"
        "ne_110m_admin_0_countries.zip"
    )

target = world[world["name"] == SHAPE_FILE_NAME]
if target.empty:
    raise ValueError(f"Country '{SHAPE_FILE_NAME}' not found. Check config.SHAPE_FILE_NAME.")
target_geom = target.geometry.iloc[0]
minx, miny, maxx, maxy = target_geom.bounds
pad    = 0.4
extent = (minx-pad, maxx+pad, miny-pad, maxy+pad)
neighbours = world[
    world.geometry.intersects(target_geom.envelope) & (world["name"] != SHAPE_FILE_NAME)
]


# ── Load resolved lines (only the kept ones) ──────────────────────────────────

if not RESOLVED_CSV.exists():
    raise FileNotFoundError(f"{RESOLVED_CSV} not found. Run resolve_unmatched.py first.")

resolved_df = pd.read_csv(RESOLVED_CSV)
resolved_df = resolved_df[resolved_df["fbus"].notna() & resolved_df["tbus"].notna()].copy()
resolved_df["fbus"]    = resolved_df["fbus"].astype(int)
resolved_df["tbus"]    = resolved_df["tbus"].astype(int)
resolved_df["osm_id"]  = resolved_df["osm_id"].astype(str)
resolved_df["voltage_V"] = pd.to_numeric(resolved_df["voltage_V"], errors="coerce")

# Assign mode voltage to lines with unknown/0 voltage_V
known_vs = resolved_df["voltage_V"].dropna()
known_vs = known_vs[known_vs > 0]
mode_voltage = float(known_vs.mode()[0]) if len(known_vs) > 0 else np.nan
resolved_df["voltage_V"] = resolved_df["voltage_V"].fillna(mode_voltage)
resolved_df.loc[resolved_df["voltage_V"] == 0, "voltage_V"] = mode_voltage

resolved_df["kv"] = resolved_df["voltage_V"].apply(line_kv)


# ── Build geometry lookup from processed lines ────────────────────────────────

geom_lookup: dict = {}   # osm_id str → Shapely geometry

if PROCESSED_CSV.exists():
    proc_df = pd.read_csv(PROCESSED_CSV)
    for _, prow in proc_df.iterrows():
        raw_geom = prow.get("geometry", None)
        if not raw_geom or (isinstance(raw_geom, float) and np.isnan(raw_geom)):
            continue
        try:
            geom_lookup[str(prow["id"])] = shapely_wkt.loads(str(raw_geom))
        except Exception:
            pass
else:
    print(f"  WARNING: {PROCESSED_CSV.name} not found — all lines will be drawn straight.")


# ── Load buses — filter to only those connected to a kept line ────────────────

buses_df = pd.read_csv(BUSES_CSV)
buses_df["kv"] = buses_df["voltage_tag"].apply(dominant_kv)
connected_ids  = set(resolved_df["fbus"]) | set(resolved_df["tbus"])
buses_df       = buses_df[buses_df["bus_i"].isin(connected_ids)].copy()
bus_xy         = buses_df.set_index("bus_i")[["longitude","latitude"]].to_dict("index")

buses_df["label"] = buses_df["name"].apply(
    lambda n: "" if (pd.isna(n) or str(n).startswith("Substation_")) else clean_label(n)
)


# ── Build dynamic voltage styles ──────────────────────────────────────────────

all_kv        = set(resolved_df["kv"].unique()) | set(buses_df["kv"].unique())
known_kv      = {k for k in all_kv if k > 0}
LINE_STYLES, BUS_STYLES = build_voltage_styles(sorted(known_kv | {0}, reverse=True))
kv_draw_order = sorted(LINE_STYLES.keys())


# ── Figure ────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":"serif","font.size":9,"axes.linewidth":0.8,
    "xtick.direction":"in","ytick.direction":"in",
    "xtick.major.size":4,"ytick.major.size":4,
})

fig, ax = plt.subplots(figsize=(7.5, 9.5), dpi=300)
fig.patch.set_facecolor("#D6EAF8")
ax.set_facecolor("#D6EAF8")

if not neighbours.empty:
    neighbours.plot(ax=ax, color="#E8E8E8", edgecolor="#AAAAAA", linewidth=0.5, zorder=0)
target.plot(ax=ax, color="#F5F0E8", edgecolor="black", linewidth=1.0, zorder=1)


# ── Draw lines using actual OSM geometry where available ─────────────────────

n_geom    = 0   # lines drawn with actual geometry
n_straight = 0  # fallback straight lines

for kv_class in kv_draw_order:
    style  = {k: v for k, v in LINE_STYLES[kv_class].items()
               if k not in ("label","zorder")}
    z      = LINE_STYLES[kv_class]["zorder"]
    subset = resolved_df[resolved_df["kv"] == kv_class]

    for _, row in subset.iterrows():
        osm_id = row["osm_id"]
        geom   = geom_lookup.get(osm_id)

        if geom is not None:
            # Actual OSM route
            try:
                xs, ys = geom.xy
                ax.plot(xs, ys, zorder=z, **style)
                n_geom += 1
                continue
            except Exception:
                pass

        # Fallback: straight line between bus centroids
        fb, tb = row["fbus"], row["tbus"]
        if fb in bus_xy and tb in bus_xy:
            x0, y0 = bus_xy[fb]["longitude"], bus_xy[fb]["latitude"]
            x1, y1 = bus_xy[tb]["longitude"], bus_xy[tb]["latitude"]
            ax.plot([x0,x1], [y0,y1], zorder=z, **style)
            n_straight += 1


print(f"Lines drawn: {n_geom} with OSM geometry, {n_straight} straight fallback")


# ── Draw buses ────────────────────────────────────────────────────────────────

for kv_class in kv_draw_order:
    subset = buses_df[buses_df["kv"] == kv_class]
    if subset.empty:
        continue
    ax.scatter(subset["longitude"], subset["latitude"], **BUS_STYLES[kv_class])


# ── Bus labels ────────────────────────────────────────────────────────────────

for _, row in buses_df[buses_df["label"] != ""].iterrows():
    ax.annotate(
        row["label"],
        xy=(row["longitude"], row["latitude"]),
        xytext=(3, 3), textcoords="offset points",
        fontsize=4.5, color="#1A1A1A",
        fontfamily="sans-serif", zorder=9, clip_on=True,
    )


# ── Extent and ticks ──────────────────────────────────────────────────────────

ax.set_xlim(extent[0], extent[1])
ax.set_ylim(extent[2], extent[3])
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.set_xlabel("Longitude (°E)", fontsize=9)
ax.set_ylabel("Latitude (°N)",  fontsize=9)
ax.set_title(f"{SHAPE_FILE_NAME} High-Voltage Transmission Network",
             fontsize=12, fontweight="bold", pad=8)


# ── Legend — line voltages only ───────────────────────────────────────────────

line_handles = [
    mlines.Line2D([], [], color=LINE_STYLES[kv]["color"],
                  lw=LINE_STYLES[kv]["lw"],
                  linestyle=LINE_STYLES[kv].get("ls", "-"),
                  label=f"{kv} kV")
    for kv in sorted(LINE_STYLES, reverse=True) if kv > 0
]
ax.legend(handles=line_handles, title="Line voltage",
          title_fontsize=8, fontsize=7.5, loc="upper left",
          framealpha=0.92, edgecolor="#CCCCCC",
          handlelength=2.4, borderpad=0.8)


# ── Scale bar & north arrow ───────────────────────────────────────────────────

add_scale_bar(ax, length_deg=1.5, location=(0.065, 0.075))
add_north_arrow(ax, location=(0.945, 0.09))

plt.tight_layout(pad=0.5)
plt.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {OUTPUT}")
plt.show()
