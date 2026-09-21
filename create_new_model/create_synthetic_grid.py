"""
create_synthetic_grid.py

Assembles all pipeline CSV outputs into a MATPOWER Version 2 .m case file.

Reads
-----
  data/bus_data/processed/{region}_substation_predictions_v4.csv  — buses with Pd/Qd
  data/working_files/{region}/matpower_{region}_buses.csv          — bus types (updated by
                                                                       create_generators.py)
  data/working_files/{region}/matpower_{region}_lines_resolved.csv — transmission branches
  data/working_files/{region}/transformer_pairs_{region}.csv        — intra-substation
                                                                       transformer branches
  data/working_files/{region}/matpower_{region}_generators.csv     — generators
  data/working_files/{region}/matpower_{region}_cost_curves.csv    — cost curves

Writes
------
  data/working_files/{region}/{region}_synthetic_grid.m   — MATPOWER case file

Citation
--------
  R. Asiamah, R. Sharma, C. Merkt, S. Talkington, J. Watson, and D. Molzahn,
  "A Framework for the Creation of Highly Realistic International Synthetic Grid Models,"
  to appear in IEEE Transactions on Energy Markets, Policy and Regulation, Jun. 2025.
"""

import math
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date

import config
from pipeline_integrity import read_transformers, validate_endpoints

region = config.region

PROJ_ROOT = Path(__file__).parent.parent
WORK_DIR  = PROJ_ROOT / "data" / "working_files" / region
PROC_DIR  = PROJ_ROOT / "data" / "bus_data" / "processed"
WORK_DIR.mkdir(parents=True, exist_ok=True)

BASE_MVA = 100.0


# ============================================================
# HELPER
# ============================================================

def _fmt(v, decimals=6) -> str:
    """Format a numeric value; NaN/None → '0.0'."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "0.0"
    return f"{round(float(v), decimals)}"


def _int(v, default=1) -> int:
    """Convert to int, returning default for NaN/None."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return default
    return int(float(v))


def _basekv_kv(v) -> float:
    """Convert base_kV (stored in Volts) to kV for MATPOWER."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return 0.0
    v = float(v)
    return v / 1000.0 if v > 1000 else v


# ============================================================
# LOAD DATA
# ============================================================
print("=" * 60)
print(f"Assembling MATPOWER case for: {region}")
print("=" * 60)

# Buses — predictions CSV has Pd/Qd; base CSV has the updated bus types
prediction_suffix = "paper" if config.LOAD_MODEL_METHOD == "paper_huber" else "v4"
bus_pred_path = PROC_DIR / f"{region}_substation_predictions_{prediction_suffix}.csv"
if config.LOAD_MODEL_METHOD == "paper_huber" and not bus_pred_path.exists():
    raise FileNotFoundError(f"{bus_pred_path} missing; run paper feature extraction and Huber prediction first")
bus_base_path = WORK_DIR / f"matpower_{region}_buses.csv"

if bus_pred_path.exists():
    buses = pd.read_csv(bus_pred_path)
    print(f"  Buses from predictions CSV ({len(buses)} rows)")
else:
    buses = pd.read_csv(bus_base_path)
    print(f"  Buses from base CSV ({len(buses)} rows) — Pd/Qd = 0")

# Merge bus types from the base CSV (updated by create_generators.py).
# The predictions CSV only has type=1 defaults; the base CSV has correct
# type=2 (PV) and type=3 (slack) assignments.
if bus_base_path.exists() and bus_pred_path.exists():
    base_types = pd.read_csv(bus_base_path)[["bus_i", "type"]]
    buses = buses.drop(columns=["type"], errors="ignore")
    buses = buses.merge(base_types, on="bus_i", how="left")
    buses["type"] = buses["type"].fillna(1).astype(int)
    print(f"  Merged bus types from base CSV  "
          f"(type 2: {(buses['type']==2).sum()}, type 3: {(buses['type']==3).sum()})")

# Transmission lines
lines_path = WORK_DIR / f"matpower_{region}_lines_resolved.csv"
if not lines_path.exists():
    raise FileNotFoundError(f"{lines_path} not found. Run resolve_unmatched.py first.")
lines = pd.read_csv(lines_path)
print(f"  Transmission lines: {len(lines)} rows")

# Transformer pairs (intra-substation connections from compound-voltage splits)
xfmr_path = WORK_DIR / f"transformer_pairs_{region}.csv"
if xfmr_path.exists():
    xfmrs = read_transformers(xfmr_path)
    print(f"  Transformer pairs: {len(xfmrs)} rows")
else:
    xfmrs = read_transformers(xfmr_path)
    print("  Transformer pairs CSV not found — no intra-substation transformers added")

validate_endpoints(buses, lines, check_names=False)
validate_endpoints(buses, xfmrs)

# Generators
gen_path = WORK_DIR / f"matpower_{region}_generators.csv"
if not gen_path.exists():
    raise FileNotFoundError(f"{gen_path} not found. Run create_generators.py first.")
gens = pd.read_csv(gen_path)
print(f"  Generators: {len(gens)} rows")

# Cost curves
cost_path = WORK_DIR / f"matpower_{region}_cost_curves.csv"
if not cost_path.exists():
    raise FileNotFoundError(f"{cost_path} not found. Run create_cost_curves.py first.")
costs = pd.read_csv(cost_path)
print(f"  Cost curves: {len(costs)} rows")
if len(costs) != len(gens):
    raise ValueError("Generator and cost tables have different row counts; regenerate costs")
for column in ("generator_name", "bus"):
    if not costs[column].equals(gens[column]):
        raise ValueError(f"Generator/cost {column} ordering differs; regenerate costs")
# Apply the same mask to dispatch and costs so MATPOWER rows stay aligned.
valid_gen = gens["bus"].notna()
costs = costs.loc[valid_gen].reset_index(drop=True)
gens = gens.loc[valid_gen].reset_index(drop=True)


# ============================================================
# MATPOWER BUS MATRIX
# ============================================================

buses["_basekv"] = buses["base_kV"].apply(_basekv_kv)
buses["Pd"]      = buses["Pd"].fillna(0.0)
buses["Qd"]      = buses["Qd"].fillna(0.0)
buses["type"]    = buses["type"].fillna(1).astype(int)

bus_rows = []
for _, r in buses.iterrows():
    bus_rows.append(
        f"\t{_int(r['bus_i'])}\t{_int(r['type'])}\t"
        f"{_fmt(r['Pd'],4)}\t{_fmt(r['Qd'],4)}\t"
        f"{_fmt(r.get('Gs',0.0),4)}\t{_fmt(r.get('Bs',0.0),4)}\t"
        f"{_int(r.get('area', 1))}\t"
        f"{_fmt(r.get('Vm',1.0),6)}\t{_fmt(r.get('Va',0.0),6)}\t"
        f"{_fmt(r['_basekv'],1)}\t"
        f"{_int(r.get('zone', 1))}\t"
        f"{_fmt(r.get('Vmax',1.1),2)}\t{_fmt(r.get('Vmin',0.9),2)}"
    )

print(f"\n  Bus matrix: {len(bus_rows)} rows")


# ============================================================
# MATPOWER GENERATOR MATRIX
# ============================================================

# A plant whose capacity OSM does not record has no limits (create_generators.py
# leaves them blank) and _fmt would write it as an in-service 0 MW unit. That is
# not a generator: it contributes nothing, it turns its bus into a phantom PV bus,
# and Pmin == Pmax == Qmin == Qmax == 0 is a degenerate box that stalls the
# interior point OPF. Sec. V-B-2 asks for a manual lookup of such plants, or for
# treating them as too small to matter, so until that happens they stay offline.
gen_rows = []
uncapacitated = []
for _, r in gens.iterrows():
    if pd.isna(r.get("bus")):
        continue
    pmax = r.get("Pmax")
    in_service = _int(r.get("status", 1))
    if pd.isna(pmax) or float(pmax) <= 0.0:
        if in_service:
            uncapacitated.append(str(r.get("generator_name", "?")).strip())
        in_service = 0
    gen_rows.append(
        f"\t{_int(r['bus'])}\t"
        f"{_fmt(r.get('Pg',0.0),4)}\t{_fmt(r.get('Qg',0.0),4)}\t"
        f"{_fmt(r.get('Qmax',0.0),4)}\t{_fmt(r.get('Qmin',0.0),4)}\t"
        f"{_fmt(r.get('Vg',1.0),4)}\t"
        f"{_fmt(r.get('mBase',BASE_MVA),1)}\t"
        f"{in_service}\t"
        f"{_fmt(r.get('Pmax',0.0),4)}\t{_fmt(r.get('Pmin',0.0),4)}"
    )

print(f"  Generator matrix: {len(gen_rows)} rows")
if uncapacitated:
    capacity = pd.to_numeric(gens["Pmax"], errors="coerce").fillna(0.).clip(lower=0.).sum()
    print(f"  WARNING: {len(uncapacitated)} of {len(gen_rows)} generators have no "
          f"capacity in OSM and were written out of service (status 0), leaving "
          f"{capacity:.0f} MW of dispatchable capacity against "
          f"{buses['Pd'].sum():.0f} MW of demand.")
    print("           Sec. V-B-2 asks for a manual lookup of these plants:")
    for name in uncapacitated[:15]:
        print(f"             - {name}")
    if len(uncapacitated) > 15:
        print(f"             ... and {len(uncapacitated) - 15} more")


# ============================================================
# MATPOWER BRANCH MATRIX  (transmission lines + transformers)
# ============================================================

def _branch_row(r) -> str | None:
    fb = r.get("fbus")
    tb = r.get("tbus")
    if pd.isna(fb) or pd.isna(tb):
        return None
    return (
        f"\t{_int(fb)}\t{_int(tb)}\t"
        f"{_fmt(r.get('r',0.0),8)}\t{_fmt(r.get('x',0.0),8)}\t{_fmt(r.get('b',0.0),8)}\t"
        f"{_fmt(r.get('rateA',0.0),4)}\t{_fmt(r.get('rateB',0.0),4)}\t{_fmt(r.get('rateC',0.0),4)}\t"
        f"0\t0\t1\t-60\t60"
    )

branch_rows = []
for _, r in lines.iterrows():
    row = _branch_row(r)
    if row:
        branch_rows.append(row)

# Append intra-substation transformer rows
n_xfmr_added = 0
for _, r in xfmrs.iterrows():
    row = _branch_row(r)
    if row:
        branch_rows.append(row)
        n_xfmr_added += 1

print(f"  Branch matrix: {len(branch_rows)} rows "
      f"({len(branch_rows) - n_xfmr_added} lines + {n_xfmr_added} transformers)")


# ============================================================
# MATPOWER GENCOST MATRIX
# ============================================================

cost_rows = []
for _, r in costs.iterrows():
    if pd.isna(r.get("c1")):
        cost_rows.append("\t2\t0.0\t0.0\t3\t0.0\t0.0\t0.0")
    else:
        cost_rows.append(
            f"\t{int(r.get('model',2))}\t0.0\t0.0\t{int(r.get('n',3))}\t"
            f"{_fmt(r.get('c2',0.0),8)}\t{_fmt(r.get('c1',0.0),6)}\t{_fmt(r.get('c0',0.0),4)}"
        )

print(f"  Gencost matrix: {len(cost_rows)} rows")


# ============================================================
# METADATA BLOCKS
# ============================================================

def _cell(value) -> str:
    """Quote a string for a MATLAB cell array.

    A real substation name can carry an apostrophe - Kenya has "Mang'u" - and
    written straight into a single-quoted cell it ends the string early, so the
    case stops parsing in MATLAB, Octave and PowerModels alike. MATLAB escapes a
    quote by doubling it. A semicolon would end the cell row, and a newline the
    row itself, so neither can survive inside a name either.
    """
    text = " ".join(str(value).split())
    return "'" + text.replace("'", "''").replace(";", ",") + "'"


bus_names = []
for _, r in buses.iterrows():
    name = str(r.get("name", "")).strip()
    if not name or name.lower() in ("nan", ""):
        kv = _basekv_kv(r.get("base_kV", 0.0))
        name = f"Bus_{int(r['bus_i'])}_{kv:.0f}kV"
    bus_names.append(f"\t{_cell(name)} %{int(r['bus_i'])}")

gen_valid  = gens[~gens["bus"].isna()]
gen_names  = [f"\t{_cell(r.get('generator_name', '?'))}" for _, r in gen_valid.iterrows()]
gen_fuels  = [f"\t{_cell(str(r.get('source', 'unknown')).split(';')[0])}"
              for _, r in gen_valid.iterrows()]

bus_coords = []
for _, r in buses.iterrows():
    lat = r.get("latitude", 0.0)
    lon = r.get("longitude", 0.0)
    if pd.isna(lat) or pd.isna(lon):
        lat, lon = 0.0, 0.0
    bus_coords.append(f"\t'{float(lat):.6f}, {float(lon):.6f}'")


# ============================================================
# WRITE MATPOWER .m FILE
# ============================================================

func_name    = f"{region}_synthetic_grid"
today        = date.today().strftime("%Y-%m-%d")

buses_block  = "\n".join(bus_rows)
gen_block    = "\n".join(gen_rows)
lines_block  = "\n".join(branch_rows)
cost_block   = "\n".join(cost_rows)
names_block  = "\n".join(bus_names)
genname_blk  = "\n".join(gen_names)
fuel_block   = "\n".join(gen_fuels)
coord_block  = "\n".join(bus_coords)

matpower_text = f"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%                                                                  %%%%%
%%%%       Synthetic Electric Grid for {config.SHAPE_FILE_NAME:<30}  %%%%%
%%%%                                                                  %%%%%
%%%%                          {today}                              %%%%%
%%%%                                                                  %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   DC and AC Optimal Power Flow data for the {config.SHAPE_FILE_NAME} electricity
%   transmission system. Please see CASEFORMAT for details on the case
%   file format.
%
%   This case represents the size and complexity of the {config.SHAPE_FILE_NAME}
%   high-voltage transmission network and can be used to validate mathematical
%   methods and for applied and policy research.
%
%   When publishing results using this data, please cite:
%   R. Asiamah, R. Sharma, C. Merkt, S. Talkington, J. Watson, and D. Molzahn,
%   "A Framework for the Creation of Highly Realistic International Synthetic
%   Grid Models," in preparation for submission to IEEE Transactions on Energy 
%   Markets, Policy and Regulation.
%
%   Contact: Richard Asiamah (asiamah@gatech.edu)
%   Grid Optimization and Algorithms Lab, 
%   Georgia Institute of Technology
%
%% MATPOWER Case Format : Version 2
function mpc = {func_name}
    mpc.version = '2';

    %%-----  Power Flow Data  -----%%
    %% system MVA base
    mpc.baseMVA = {BASE_MVA};

    %% bus data
    %    bus_i    type    Pd    Qd    Gs    Bs    area    Vm    Va    baseKV    zone    Vmax    Vmin
    mpc.bus = [
{buses_block}
    ];

    %% generator data
    %    bus    Pg    Qg    Qmax    Qmin    Vg    mBase    status    Pmax    Pmin
    mpc.gen = [
{gen_block}
    ];

    %% branch data
    %    fbus    tbus    r    x    b    rateA    rateB    rateC    ratio    angle    status    angmin    angmax
    mpc.branch = [
{lines_block}
    ];

    %%-----  OPF Data  -----%%
    %% cost data
    %    model    startup    shutdown    n    c(n-1)    ...    c0
    mpc.gencost = [
{cost_block}
    ];

    %% bus names
    mpc.bus_name = {{
{names_block}
    }};

    %% generator names
    mpc.gentype = {{
{genname_blk}
    }};

    %% generator fuel types
    mpc.fuel = {{
{fuel_block}
    }};

    %% bus coordinates  ('lat, lon')
    mpc.coordinates = {{
{coord_block}
    }};
"""

out_path = WORK_DIR / f"{region}_synthetic_grid.m"
out_path.write_text(matpower_text, encoding="utf-8")
print(f"\nSaved MATPOWER case → {out_path}")
print(f"  Buses: {len(bus_rows)}  |  Branches: {len(branch_rows)}  "
      f"({n_xfmr_added} transformers)  |  Generators: {len(gen_rows)}")
print("\nDone. Run ensure_feasibility.py to produce DC- and AC-feasible variants.")
