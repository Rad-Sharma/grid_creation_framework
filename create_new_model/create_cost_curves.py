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
        "steam_landfill": (8.174*p - 15.0, 58.0, .0056*p - .0188),
        "steam_petcoke":  (2.242*p + 17.47, .0074*p + 17.80, 1.15e-4*p - .00111),
        "free_steam":     (0.0,                  0.0,                   0.0),         # nuclear/geothermal/MSW
        # ── Gas turbine units ──────────────────────────────────
        "gt_gas":         (7.913*p - 23.26,      0.0366*p  + 1.27,     9.45e-4*p  - 0.0184),
        "gt_petroleum":   (11.004*p + 195.9,     0.1786*p  + 17.43,    3.49e-3*p  - 0.0688),
        "gt_biomass":     (0.0,                  0.0,                   0.0),
        # ── Combined cycle units ───────────────────────────────
        "cc_gas":         (2.388*p + 182.2,      0.00217*p + 12.96,    5.96e-5*p  - 1.92e-4),
        "cc_petroleum":   (39.10*p + 10.0,       0.0353*p  + 51.29,    3.88e-3*p  - 0.0076),
        "cc_coal":        (992.0, 9.0, .01),
        "cc_landfill":    (27.40*p - 18.0, 37.0, .0090*p),
        "cc_petcoke":     (1207.0, 14.0, .018),
        "cc_wood":        (554.0, 22.0, .09),
        "ic_gas":         (.506*p - 1.12, .00556*p + 5.89, 6.67e-4*p - .00333),
        "ic_petroleum":   (1.607*p - 9.14, .00196*p + 24.96, 5.68e-4*p - .00136),
        "ic_landfill":    (53.0, 25.0, .3),
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
    "waste":       "free_steam",
    "wood":        "steam_wood",
    "landfill gas": "steam_landfill",
    "petroleum coke": "steam_petcoke",
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


def map_to_category(primary_source: str, genfuel: str, method: str = "") -> str:
    """Resolve (primary_source, genfuel) → cost curve category key."""
    method = str(method).lower().replace("_", " ").replace("-", " ")
    fuel = {"gas": "gas", "natural gas": "gas", "oil": "petroleum",
            "diesel": "petroleum", "gasoline": "petroleum", "coal": "coal",
            "landfill gas": "landfill", "biogas": "landfill",
            "petroleum coke": "petcoke", "wood": "wood", "waste wood": "wood"}.get(primary_source)
    prefix = None
    if "combined cycle" in method: prefix = "cc"
    elif "internal combustion" in method or "combustion engine" in method: prefix = "ic"
    elif "gas turbine" in method or "combustion turbine" in method: prefix = "gt"
    elif "steam" in method: prefix = "steam"
    valid = {"steam": {"coal", "oil", "gas", "wood", "landfill", "petcoke"},
             "cc": {"gas", "petroleum", "coal", "landfill", "petcoke", "wood"},
             "ic": {"gas", "petroleum", "landfill"}, "gt": {"gas", "petroleum"}}
    if prefix == "steam" and fuel == "petroleum": fuel = "oil"
    if prefix and fuel in valid[prefix]: return f"{prefix}_{fuel}"
    cat = SOURCE_TO_CATEGORY.get(primary_source)
    if cat:
        return cat
    # Source unrecognized — fall back to genfuel column.
    gf = str(genfuel).strip().lower() if isinstance(genfuel, str) else "unknown"
    # Guard against garbled genfuel strings (e.g. URLs).
    if gf.startswith("http") or gf in {"", "unknown", "n/a"}:
        return "cc_gas"   # conservative default: combined-cycle natural gas
    return GENFUEL_TO_CATEGORY.get(gf, "cc_gas")


# ============================================================
# TABLE IV – FUEL COSTS BY FUEL TYPE ($/MBTU)
# ============================================================
# The Automate Grid Creation paper reports a US and a global price for each
# fuel. The Table V
# curve coefficients are derived from [48], which prices fuel at the US EIA
# values, so "us" reproduces Table V exactly and is the default.
#
# The Automate Grid Creation paper prints the global column but never states
# how to apply it.
# Reference [48] (Babaeinejadsarookolaee, "Synthetic Power System, Generator Cost
# Curves", 2018) does, and it makes the scaling exact rather than a guess for
# most categories. There, $/h at 40/60/80/100 % of capacity is computed as
# heat_rate(P) x P x fuel_price, and (a, b, c) is the least-squares solution of
#
#     [a b c]^T = (A^T A)^-1 A^T Y,   A depending only on P, Y linear in price.
#
# Y scales linearly with fuel price and A does not, so (a, b, c) scales exactly
# linearly with it. Verified against that document's LMS100 sample: 26634 kW x
# 12689 BTU/kWh x $2.68/MBTU = $905.7297/h, its stated value to four decimals.
#
# Selecting "global" still departs from the published Table V numbers and must
# be reported as such.
FUEL_COSTS_MBTU = {
    # fuel            US     global
    "coal":         (1.78,   2.90),
    "natural_gas":  (2.68,   4.88),
    "residual_oil": (9.25,  10.50),
    "wood":         (3.72,   4.20),
    "petroleum_coke": (2.56,  3.10),
    "landfill_gas": (6.18,   5.50),
}

# Table V curve category -> Table IV fuel. A category with no entry is not
# rescaled; see SCALING_BASIS for why each one is treated the way it is.
CATEGORY_TO_FUEL = {
    "steam_coal": "coal",           "cc_coal": "coal",
    "steam_gas": "natural_gas",     "gt_gas": "natural_gas",
    "cc_gas": "natural_gas",        "ic_gas": "natural_gas",
    "steam_oil": "residual_oil",    "gt_petroleum": "residual_oil",
    "cc_petroleum": "residual_oil", "ic_petroleum": "residual_oil",
    "steam_wood": "wood",           "cc_wood": "wood",
    "steam_petcoke": "petroleum_coke", "cc_petcoke": "petroleum_coke",
    "steam_landfill": "landfill_gas",  "cc_landfill": "landfill_gas",
    "ic_landfill": "landfill_gas",
}


# How [48] produced each family, which decides how well fuel-price scaling is
# justified:
#   "heat_rate"  - $/h built as heat_rate x P x fuel_price, so scaling is exact.
#   "market_bid" - internal combustion had no physical heat-rate data, so ERCOT
#                  generator bid reports were used instead. A bid carries margin
#                  and O&M as well as fuel, so scaling is directionally right but
#                  not algebraically implied. These units are fuel-dominated
#                  gensets, so leaving them unscaled while gas scales 1.82x would
#                  distort the merit order more than scaling them does.
#   "borrowed"   - hydro has no fuel at all; [48] states its curves were copied
#                  from the CCGT samples because they look similar. Table V's
#                  hydro b = 14 $/MWh is therefore a gas heat rate priced at
#                  $2.68/MBTU, not a water or O&M value. Scaling it by a gas
#                  price ratio would propagate that placeholder into a fuel-price
#                  adjustment, so hydro is left alone. The borrowed curve is a
#                  limitation of the source data under either basis.
#   "zero"       - non-dispatchable in [48]; a, b, c are all zero.
SCALING_BASIS = {
    **{k: "heat_rate" for k in ("steam_coal", "steam_oil", "steam_gas", "steam_wood",
                                "steam_landfill", "steam_petcoke", "gt_gas",
                                "gt_petroleum", "cc_gas", "cc_petroleum", "cc_coal",
                                "cc_landfill", "cc_petcoke", "cc_wood")},
    **{k: "market_bid" for k in ("ic_gas", "ic_petroleum", "ic_landfill")},
    "hydro": "borrowed",
    **{k: "zero" for k in ("free_steam", "gt_biomass", "renewable")},
}


def fuel_cost_scale(category: str, basis: str = "us") -> float:
    """Multiplier on the Table V coefficients for the chosen fuel-price basis."""
    if basis not in ("us", "global"):
        raise ValueError("FUEL_COST_BASIS must be 'us' or 'global'")
    fuel = CATEGORY_TO_FUEL.get(category)
    if basis == "us" or fuel is None:
        return 1.0
    us, world = FUEL_COSTS_MBTU[fuel]
    return world / us


def compute_polynomial_coeffs(pmax: float, category: str,
                              basis: str | None = None) -> tuple[float, float, float]:
    """Return (c2, c1, c0) — the MATPOWER polynomial coefficients for one generator.

    Maps directly to the MATPOWER gencost MODEL=2 columns [c2, c1, c0] where
    f(p) = c2*p² + c1*p + c0 with units $/MWh², $/MWh, $/h respectively.
    Coefficients are clamped to zero from below: negative values are artefacts
    of the linear regression fits for plants well outside the training range.
    """
    if basis is None:
        basis = getattr(config, "FUEL_COST_BASIS", "us")
    A, B, C = _coeffs(category, pmax)
    scale = fuel_cost_scale(category, basis)
    c2 = max(0.0, C) * scale
    c1 = max(0.0, B) * scale
    c0 = max(0.0, A) * scale
    return round(c2, 6), round(c1, 6), round(c0, 4)


# ============================================================
# STEP 1 – LOAD GENERATOR CSV
# ============================================================
def main():
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
        lambda r: map_to_category(r["primary_source"], r["genfuel"], r.get("method", "")), axis=1
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


if __name__ == "__main__":
    main()
