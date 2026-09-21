"""
ensure_feasibility.py

Attempts DC and AC feasibility relaxations; exports only validated solutions.

DC formulation  (eq:slack_dcopf)
---------------------------------
  min  Σ_G C_i(P_Gi) + M Σ_{ij∈L} (P_ij^sl)²
  s.t. Σ_G P_Gi = Σ_D P_Dj                    [global balance]
       P_Gi^min ≤ P_Gi ≤ P_Gi^max
       P_ij = baseMVA * (θ_i − θ_j − shift_ij) / (X_ij * tap_ij)
       −P_ij^max − P_ij^sl ≤ P_ij ≤ P_ij^max + P_ij^sl
       P_ij^sl ≥ 0
  Slack P^sl represents extra line capacity needed for feasibility.
  Qg = 0 in the DC case.

AC formulation  (eq:shunt_acopf)
----------------------------------
  min  Σ_g C_g(P_Gg) + M Σ_{i∈B} (B_i^sh)²
  s.t. θ_ref = 0
       AC real-power balance at each bus
       AC reactive-power balance at each bus (with shunt term B_i^sh · V_i²)
       P_g^min ≤ P_g ≤ P_g^max
       Q_g^min ≤ Q_g ≤ Q_g^max
       V_i^min ≤ V_i ≤ V_i^max
       |S_ij,fr| ≤ S_ij^max,  |S_ij,to| ≤ S_ij^max
       θ_ij^min ≤ θ_i − θ_j ≤ θ_ij^max
       B_i^sh,min ≤ B_i^sh ≤ B_i^sh,max

  The quadratic penalty follows manuscript equation (5a). It reduces shunt
  magnitudes but does not directly minimise the number of installed devices.

Outputs
-------
  matpower_files/{region}_dc_feasible.m
  matpower_files/{region}_ac_feasible.m

Both files use the MATPOWER Version 2 format. Generator Pg (and Qg for AC)
are set to the dispatch values obtained from the respective relaxation.
"""

import math
import re
import warnings
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
from power_network import solve_feasibility, check_solution

import config
from pipeline_integrity import (dispatchability_report,
                                describe_dispatchability)

warnings.filterwarnings("ignore")

region      = config.region
PROJ_ROOT   = Path(__file__).parent.parent
WORK_DIR    = PROJ_ROOT / "data" / "working_files" / region
MPOWER_DIR  = PROJ_ROOT / "matpower_files"
MPOWER_DIR.mkdir(parents=True, exist_ok=True)

M_PENALTY     = 1e4   # weight on slack/shunt penalty terms
MAX_BSH_PU    = 5.0   # p.u. maximum shunt susceptance per bus (±)
V_MIN, V_MAX  = 0.9, 1.1


# ============================================================
# MATPOWER .m PARSER
# ============================================================

def _parse_matrix(text: str, name: str) -> np.ndarray | None:
    """Extract a numeric matrix block from a MATPOWER .m file."""
    pat = rf"mpc\.{name}\s*=\s*\[(.*?)\];"
    m   = re.search(pat, text, re.DOTALL)
    if not m:
        return None
    rows = []
    body = re.sub(r"%[^\n]*", "", m.group(1))
    for line in body.replace(";", "\n").strip().splitlines():
        line = re.sub(r"%.*", "", line).strip()
        if not line:
            continue
        try:
            rows.append([float(x) for x in line.split()])
        except ValueError as exc:
            raise ValueError(f"Invalid numeric row in mpc.{name}: {line}") from exc
    return np.array(rows) if rows else None


def load_matpower(path: Path):
    """Parse a MATPOWER .m file and return dict with 'bus', 'gen', 'branch', 'gencost'."""
    text = path.read_text(encoding="utf-8")
    data = {}
    for key, width in {"bus": 13, "gen": 10, "branch": 13, "gencost": 7}.items():
        arr = _parse_matrix(text, key)
        data[key] = arr if arr is not None else np.zeros((0, width))

    # Also extract baseMVA
    m = re.search(r"mpc\.baseMVA\s*=\s*([\d.]+)", text)
    data["baseMVA"] = float(m.group(1)) if m else 100.0
    return data


def solve_dc_feasibility(data: dict) -> dict:
    """Return a DC case only after validating every bus balance and limit."""
    return solve_feasibility(data, ac=False, penalty=M_PENALTY)


def solve_ac_feasibility(data: dict) -> dict:
    """Return an AC case with validated dispatch and shunts in MATPOWER units."""
    return solve_feasibility(data, ac=True, penalty=M_PENALTY, max_bsh=MAX_BSH_PU)


# ============================================================
# WRITE MATPOWER .m FILE  (full descriptive format)
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


def _load_metadata():
    """Load bus/generator metadata from the working-files CSVs for annotation blocks."""
    buses_csv = WORK_DIR / f"matpower_{region}_buses.csv"
    gens_csv  = WORK_DIR / f"matpower_{region}_generators.csv"
    meta = {"bus_names": [], "bus_coords": [], "gen_names": [], "gen_fuels": []}
    if buses_csv.exists():
        bdf = pd.read_csv(buses_csv)
        for _, r in bdf.iterrows():
            name = str(r.get("name", "")).strip()
            kv   = r.get("base_kV", 0) or 0
            kv   = float(kv) / 1000.0 if float(kv) > 1000 else float(kv)
            label = name if name and name.lower() != "nan" else f"Bus_{int(r['bus_i'])}"
            meta["bus_names"].append(f"\t{_cell(f'{label}_{kv:.0f}kV')} %{int(r['bus_i'])}")
            lat = r.get("latitude", 0.0)
            lon = r.get("longitude", 0.0)
            lat = 0.0 if (lat is None or str(lat) == "nan") else float(lat)
            lon = 0.0 if (lon is None or str(lon) == "nan") else float(lon)
            meta["bus_coords"].append(f"\t'{lat:.6f}, {lon:.6f}'")
    if gens_csv.exists():
        gdf = pd.read_csv(gens_csv).dropna(subset=["bus"])
        for _, r in gdf.iterrows():
            meta["gen_names"].append(f"\t{_cell(r.get('generator_name', '?'))}")
            fuel = str(r.get("source", "unknown")).split(";")[0]
            meta["gen_fuels"].append(f"\t{_cell(fuel)}")
    return meta


def write_matpower(data: dict, out_path: Path, label: str):
    """Write a MATPOWER Version 2 .m file with the full descriptive header format."""
    check_solution(data, ac=label.lower().startswith("ac"))
    bus_arr    = data["bus"]
    gen_arr    = data["gen"]
    branch_arr = data["branch"]
    gencost    = data["gencost"]
    baseMVA    = data.get("baseMVA", 100.0)

    def _fmt(v, d=6):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "0.0"
        return f"{float(v):.15g}"

    bus_lines  = ["\t" + "\t".join(_fmt(v, 6) for v in r) for r in bus_arr]
    gen_lines  = ["\t" + "\t".join(_fmt(v, 4) for v in r) for r in gen_arr]
    br_lines   = ["\t" + "\t".join(_fmt(v, 8) for v in r) for r in branch_arr]
    cost_lines = ["\t" + "\t".join(_fmt(v, 8) for v in r) for r in gencost]

    meta      = _load_metadata()
    func_name = out_path.stem
    today     = date.today().strftime("%Y-%m-%d")

    header = (
        f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
        f"%%%%                                                                  %%%%%\n"
        f"%%%%   Synthetic Electric Grid for {config.SHAPE_FILE_NAME:<30} %%%%%\n"
        f"%%%%   {label:<62} %%%%%\n"
        f"%%%%   {today:<62} %%%%%\n"
        f"%%%%                                                                  %%%%%\n"
        f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
        f"%\n"
        f"%   DC and AC Optimal Power Flow data for the {config.SHAPE_FILE_NAME}\n"
        f"%   electricity transmission system ({label}).\n"
        f"%   Please see CASEFORMAT for details on the case file format.\n"
        f"%\n"
        f"%   When publishing results using this data, please cite:\n"
        f"%   R. Asiamah, R. Sharma, C. Merkt, S. Talkington, J. Watson, and D. Molzahn,\n"
        f"%   \"A Framework for the Creation of Highly Realistic International Synthetic\n"
        f"%   Grid Models,\" in preparation for submission to IEEE Transactions on Energy\n"
        f"%   Markets, Policy and Regulation, Jun. 2025.\n"
        f"%\n"
        f"%   Contact: Richard Asiamah (asiamah@gatech.edu)\n"
        f"%   Grid Optimization and Algorithms Lab\n"
        f"%   Georgia Institute of Technology\n"
    )

    names_block  = "\n".join(meta["bus_names"])  if meta["bus_names"]  else "\t% (none)"
    coords_block = "\n".join(meta["bus_coords"]) if meta["bus_coords"] else "\t% (none)"
    gname_block  = "\n".join(meta["gen_names"])  if meta["gen_names"]  else "\t% (none)"
    gfuel_block  = "\n".join(meta["gen_fuels"])  if meta["gen_fuels"]  else "\t% (none)"

    text = (
        header +
        f"%% MATPOWER Case Format : Version 2\n"
        f"function mpc = {func_name}\n"
        f"    mpc.version = '2';\n\n"
        f"    %%-----  Power Flow Data  -----%%\n"
        f"    %% system MVA base\n"
        f"    mpc.baseMVA = {baseMVA};\n\n"
        f"    %% bus data\n"
        f"    %    bus_i    type    Pd    Qd    Gs    Bs    area    Vm    Va    baseKV    zone    Vmax    Vmin\n"
        f"    mpc.bus = [\n" + "\n".join(bus_lines) + "\n    ];\n\n"
        f"    %% generator data\n"
        f"    %    bus    Pg    Qg    Qmax    Qmin    Vg    mBase    status    Pmax    Pmin\n"
        f"    mpc.gen = [\n" + "\n".join(gen_lines) + "\n    ];\n\n"
        f"    %% branch data\n"
        f"    %    fbus    tbus    r    x    b    rateA    rateB    rateC    ratio    angle    status    angmin    angmax\n"
        f"    mpc.branch = [\n" + "\n".join(br_lines) + "\n    ];\n\n"
        f"    %%-----  OPF Data  -----%%\n"
        f"    %% cost data\n"
        f"    %    model    startup    shutdown    n    c2    c1    c0\n"
        f"    mpc.gencost = [\n" + "\n".join(cost_lines) + "\n    ];\n\n"
        f"    %% bus names\n"
        f"    mpc.bus_name = {{\n{names_block}\n    }};\n\n"
        f"    %% generator names\n"
        f"    mpc.gentype = {{\n{gname_block}\n    }};\n\n"
        f"    %% generator fuel types\n"
        f"    mpc.fuel = {{\n{gfuel_block}\n    }};\n\n"
        f"    %% bus coordinates  ('lat, lon')\n"
        f"    mpc.coordinates = {{\n{coords_block}\n    }};\n"
    )

    out_path.write_text(text, encoding="utf-8")
    print(f"  Saved → {out_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    base_m = WORK_DIR / f"{region}_synthetic_grid.m"
    if not base_m.exists():
        raise FileNotFoundError(
            f"{base_m} not found. Run create_synthetic_grid.py first.")

    print("=" * 60)
    print(f"Loading: {base_m.name}")
    print("=" * 60)
    data = load_matpower(base_m)
    print(f"  Buses: {len(data['bus'])}  "
          f"Branches: {len(data['branch'])}  "
          f"Generators: {len(data['gen'])}")

    report = dispatchability_report(data['bus'], data['branch'], data['gen'])
    advice = describe_dispatchability(report)
    if advice:
        print()
        raise SystemExit(advice)

    # ── DC feasibility ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DC FEASIBILITY  (eq:slack_dcopf)")
    print("=" * 60)
    dc_data = solve_dc_feasibility(data)
    dc_path = MPOWER_DIR / f"{region}_dc_feasible.m"
    write_matpower(dc_data, dc_path, "DC_feasible")

    # ── AC feasibility ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("AC FEASIBILITY  (eq:shunt_acopf, quadratic shunt penalty)")
    print("=" * 60)
    ac_data = solve_ac_feasibility(dc_data)   # warm-start from DC solution
    ac_path = MPOWER_DIR / f"{region}_ac_feasible.m"
    write_matpower(ac_data, ac_path, "AC_feasible")

    print("\n" + "=" * 60)
    print("Done.")
    print(f"  DC-feasible model → {dc_path}")
    print(f"  AC-feasible model → {ac_path}")
    print("=" * 60)
