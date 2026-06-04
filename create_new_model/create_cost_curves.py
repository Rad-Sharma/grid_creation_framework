"""
generate_cost_curves.py

Reads matpower_{region}_generators.csv and writes a MATPOWER-format
polynomial (MODEL=2) generator cost curve CSV.

Cost coefficients are drawn from representative values tabulated by
Snodgrass et al. (2021), expressed as linear functions of Pmax:
  A(Pmax) [$/h]     – constant term  (c0 in MATPOWER notation)
  B(Pmax) [$/MWh]   – linear term    (c1)
  C(Pmax) [$/MWh²]  – quadratic term (c2)

The polynomial cost function is:  f(p) = C·p² + B·p + A

Output format follows the MATPOWER mpc.gencost convention for MODEL=2:
  model  startup  shutdown  n  c2  c1  c0
where model=2, startup=shutdown=0, n=3 (quadratic, three coefficients).

Fuel type is inferred from the OSM 'source' column (primary label
before any semicolons), with 'genfuel' used as a fallback.

Renewable and zero-cost units (solar, wind, biomass, nuclear,
geothermal) receive c2=c1=c0=0.

NOTE: Generators with no Pmax entry in the input CSV cannot have cost
coefficients computed and are written with blank cost fields.

Pipeline:
  1.  Load generator CSV
  2.  Extract primary source label
  3.  Map each generator to a cost curve category
  4.  Compute c2, c1, c0 coefficients from Pmax
  5.  Build MATPOWER gencost table and save CSV

Author: Richard Asiamah
"""

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

# ============================================================
# COST CURVE COEFFICIENT TABLE
# ============================================================

# Each entry maps a category key to three callables: A(p), B(p), C(p).
# These correspond to the columns of Table tab:gen_cost_curves in the paper.
# p is Pmax in MW.

def _coeffs(category: str, p: float) -> tuple[float, float, float]:
    """Return (A, B, C) for f(p) = C*p² + B*p + A given Pmax=p."""
    table = {
        # ── Steam units ────────────────────────────────────────
        "steam_coal":     (1.628*p + 7.44,       0.00284*p + 11.94,    5.48e-5*p  - 9.57e-5),
        "steam_oil":      (5.947*p + 115.2,      0.0210*p  + 66.52,    3.73e-4*p  - 5.85e-4),
        "steam_gas":      (2.570*p + 13.61,      0.00684*p + 18.86,    1.22e-4*p  - 7.34e-4),
        "steam_wood":     (4.118*p + 5.65,       0.0229*p  + 31.54,    1.59e-3*p  - 0.0118),
        "free_steam":     (0.0,                  0.0,                   0.0),         # nuclear/geothermal/MSW
        # ── Gas turbine units ──────────────────────────────────
        "gt_gas":         (7.913*p - 23.26,      0.0366*p  + 1.27,     9.45e-4*p  - 0.0184),
        "gt_petroleum":   (11.004*p + 195.9,     0.1786*p  + 17.43,    3.49e-3*p  - 0.0688),
        "gt_biomass":     (0.0,                  0.0,                   0.0),
        # ── Combined cycle units ───────────────────────────────
        "cc_gas":         (2.388*p + 182.2,      0.00217*p + 12.96,    5.96e-5*p  - 1.92e-4),
        "cc_petroleum":   (39.10*p + 10.0,       0.0353*p  + 51.29,    3.88e-3*p  - 0.0076),
        # ── Hydro units ────────────────────────────────────────
        "hydro":          (1.117*p - 0.34,       14.0,                  1.54e-5*p  - 8.0e-6),
        # ── Renewables / zero-cost ─────────────────────────────
        "renewable":      (0.0,                  0.0,                   0.0),         # wind/solar/battery
    }
    return table.get(category, table["cc_gas"])   # default: CC natural gas


# Human-readable label for each category (used in printed summary).
CATEGORY_LABELS = {
    "steam_coal":   "Steam / Coal",
    "steam_oil":    "Steam / Oil",
    "steam_gas":    "Steam / Gas",
    "steam_wood":   "Steam / Wood & Waste",
    "free_steam":   "Steam / Nuclear–Geothermal (zero cost)",
    "gt_gas":       "Gas Turbine / Natural Gas",
    "gt_petroleum": "Gas Turbine / Petroleum Liquids",
    "gt_biomass":   "Gas Turbine / Biomass (zero cost)",
    "cc_gas":       "Combined Cycle / Natural Gas",
    "cc_petroleum": "Combined Cycle / Petroleum Liquids",
    "hydro":        "Hydro / Conventional",
    "renewable":    "Renewable – Wind/Solar/Battery (zero cost)",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_primary_source(source_str) -> str:
    """Return the first fuel label from a possibly semicolon-separated OSM source string."""
    if not isinstance(source_str, str) or not source_str.strip():
        return "unknown"
    primary = source_str.split(";")[0].strip().lower()
    # Discard garbled entries (URLs accidentally placed in OSM source tag).
    if primary.startswith("http") or primary in {"", "yes", "no", "n/a", "null"}:
        return "unknown"
    return primary


SOURCE_TO_CATEGORY = {
    "coal":        "steam_coal",
    "oil":         "gt_petroleum",   # oil-fired plants are typically open-cycle gas turbines
    "gas":         "cc_gas",         # large gas plants → combined cycle
    "biogas":      "gt_biomass",
    "biomass":     "gt_biomass",
    "hydro":       "hydro",
    "solar":       "renewable",
    "wind":        "renewable",
    "nuclear":     "free_steam",
    "geothermal":  "free_steam",
    "waste":       "steam_wood",
    "diesel":      "gt_petroleum",
    "gasoline":    "gt_petroleum",
    "steam":       "steam_gas",      # generic steam tag → steam/gas
}

GENFUEL_TO_CATEGORY = {
    "gas_turbine": "gt_gas",
    "gas":         "cc_gas",
    "steam":       "steam_gas",
    "hydro":       "hydro",
    "solar":       "renewable",
    "wind":        "renewable",
    "nuclear":     "free_steam",
}


def map_to_category(primary_source: str, genfuel: str) -> str:
    """Resolve (primary_source, genfuel) → cost curve category key."""
    cat = SOURCE_TO_CATEGORY.get(primary_source)
    if cat:
        return cat
    # Source unrecognized — fall back to genfuel column.
    gf = str(genfuel).strip().lower() if isinstance(genfuel, str) else "unknown"
    # Guard against garbled genfuel strings (e.g. URLs).
    if gf.startswith("http") or gf in {"", "unknown", "n/a"}:
        return "cc_gas"   # conservative default: combined-cycle natural gas
    return GENFUEL_TO_CATEGORY.get(gf, "cc_gas")


def compute_polynomial_coeffs(pmax: float, category: str) -> tuple[float, float, float]:
    """Return (c2, c1, c0) — the MATPOWER polynomial coefficients for one generator.

    Maps directly to the MATPOWER gencost MODEL=2 columns [c2, c1, c0] where
    f(p) = c2*p² + c1*p + c0 with units $/MWh², $/MWh, $/h respectively.
    Coefficients are clamped to zero from below: negative values are artefacts
    of the linear regression fits for plants well outside the training range.
    """
    A, B, C = _coeffs(category, pmax)
    c2 = max(0.0, C)
    c1 = max(0.0, B)
    c0 = max(0.0, A)
    return round(c2, 6), round(c1, 6), round(c0, 4)


# ============================================================
# STEP 1 – LOAD GENERATOR CSV
# ============================================================
print("=" * 60)
print("STEP 1: Loading generator CSV")
print("=" * 60)

gen = pd.read_csv(WORK_DIR / f"matpower_{region}_generators.csv")

print(f"  Loaded {len(gen)} generator rows")
n_no_pmax = gen["Pmax"].isna().sum()
print(f"  Rows with no Pmax (cost curve cannot be computed): {n_no_pmax}")


# ============================================================
# STEP 2 – EXTRACT PRIMARY SOURCE LABEL
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Extracting primary fuel source")
print("=" * 60)

# The OSM 'source' tag sometimes contains multiple semicolon-separated
# fuels (e.g. 'gas;oil'). We use the first label as the primary fuel.
# Garbled entries such as URLs are treated as unknown.

gen["primary_source"] = gen["source"].apply(extract_primary_source)

source_counts = gen["primary_source"].value_counts()
print("  Primary source distribution:")
for src, count in source_counts.items():
    print(f"    {src:20s}: {count}")


# ============================================================
# STEP 3 – MAP TO COST CURVE CATEGORY
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Mapping generators to cost curve categories")
print("=" * 60)

gen["curve_type"] = gen.apply(
    lambda r: map_to_category(r["primary_source"], r["genfuel"]), axis=1
)

print(f"  {'Generator':<45} {'Primary Source':<18} {'Curve Type'}")
print(f"  {'-'*45} {'-'*18} {'-'*40}")
for _, row in gen.iterrows():
    label = CATEGORY_LABELS.get(row["curve_type"], row["curve_type"])
    print(f"  {row['generator_name'][:44]:<45} {row['primary_source']:<18} {label}")


# ============================================================
# STEP 4 – COMPUTE POLYNOMIAL COEFFICIENTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Computing polynomial cost coefficients")
print("=" * 60)

# For each generator with a known Pmax, evaluate A(Pmax), B(Pmax), C(Pmax)
# from the table to produce the three MATPOWER gencost coefficients:
#   c2 = C(Pmax)  [$/MWh²]  – quadratic term
#   c1 = B(Pmax)  [$/MWh]   – linear term (marginal cost)
#   c0 = A(Pmax)  [$/h]     – constant term (no-load cost)
# Coefficients are clamped at 0; regression fits can produce small negative
# values for plants below the original table's training range.
# Generators with no Pmax receive blank cost fields.

c2_vals, c1_vals, c0_vals = [], [], []
n_computed = 0
n_skipped  = 0

for _, row in gen.iterrows():
    pmax = row["Pmax"]

    if pd.isna(pmax):
        c2_vals.append(np.nan)
        c1_vals.append(np.nan)
        c0_vals.append(np.nan)
        n_skipped += 1
        continue

    c2, c1, c0 = compute_polynomial_coeffs(float(pmax), row["curve_type"])
    c2_vals.append(c2)
    c1_vals.append(c1)
    c0_vals.append(c0)
    n_computed += 1

gen["c2"] = c2_vals
gen["c1"] = c1_vals
gen["c0"] = c0_vals

print(f"  Coefficients computed: {n_computed}")
print(f"  Skipped (no Pmax):     {n_skipped}")


# ============================================================
# STEP 5 – BUILD MATPOWER GENCOST TABLE AND SAVE
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Building MATPOWER gencost table")
print("=" * 60)

# MATPOWER mpc.gencost convention (MODEL=2, polynomial, N=3 coefficients):
#   [model, startup, shutdown, n, c2, c1, c0]
# startup and shutdown costs are set to 0 (not available from OSM data).
# model = 2  (polynomial)
# n     = 3  (quadratic: c2·p² + c1·p + c0)
#
# Reference columns are kept for verification and are not part of the
# MATPOWER matrix; they are included so the CSV can be reviewed before
# the gencost matrix is assembled in MATLAB/Octave.

cost_out = pd.DataFrame({
    "generator_name": gen["generator_name"],
    "bus":            gen["bus"],
    "source":         gen["source"],
    "genfuel":        gen["genfuel"],
    "curve_type":     gen["curve_type"],
    "Pmax":           gen["Pmax"],
    "model":          2,
    "startup":        0.0,
    "shutdown":       0.0,
    "n":              3,
    "c2":             gen["c2"],
    "c1":             gen["c1"],
    "c0":             gen["c0"],
})

out_path = WORK_DIR / f"matpower_{region}_cost_curves.csv"
cost_out.to_csv(out_path, index=False)
print(f"  Saved {len(cost_out)} cost curve rows → {out_path}")
print()
print(cost_out[["generator_name", "Pmax", "curve_type", "c2", "c1", "c0"]].to_string(index=False))

print("\nDone.")
