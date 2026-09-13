# Synthetic Electric Grid Creation Framework

A Python pipeline for building highly realistic synthetic electric grid models
from openly available geospatial data (OpenStreetMap, WorldPop, WRI Global
Power Plant Database, NaturalEarth).

The framework produces MATPOWER Version 2 case files — including a base model,
a DC-feasible relaxation, and an AC-feasible relaxation — that accurately
represent the topology and scale of a target country's high-voltage
transmission network.

---

## Citation

When using or extending this framework, please cite:

> R. Asiamah, R. Sharma, C. Merkt, S. Talkington, J. Watson, and D. Molzahn,
> "A Framework for the Creation of Highly Realistic International Synthetic
> Grid Models," *IEEE Transactions on Energy Markets, Policy and Regulation*,
> Jun. 2025.

---

## Repository Structure

```
grid_creation_framework/
├── create_new_model/              # Pipeline scripts (run in order)
│   ├── config.py                  # Region settings — edit this first
│   ├── create_buses.py            # Step 1: substations from OSM
│   ├── create_lines.py            # Step 2: transmission lines from OSM
│   ├── resolve_unmatched.py       # Step 3: interactive review of unmatched endpoints
│   ├── build_region_features.py   # Step 4: Voronoi + WorldPop + OSM load features
│   ├── predict_bus_loads.py       # Step 5: predict Pd/Qd per bus
│   ├── create_generators.py       # Step 6: power plants from OSM; bus type assignment
│   ├── create_cost_curves.py      # Step 7: MATPOWER cost curve coefficients
│   ├── create_synthetic_grid.py   # Step 8: assemble MATPOWER .m file
│   ├── ensure_feasibility.py      # Step 9: DC and AC feasibility relaxation
│   └── plot_grid_final.py         # Step 10: map plot
└── data/
    ├── OSM_extracts/              # Place raw GeoJSON downloads here (all regions)
    ├── working_files/
    │   └── {region}/              # All intermediate and final outputs per region
    └── bus_data/
        ├── raw/                   # WorldPop rasters, WRI database, OSM caches
        ├── features/              # Feature tables from build_region_features.py
        ├── models/                # Trained Ridge model bundle (model_bundle_v5.pkl)
        └── processed/             # Bus load predictions
```

---

## Manuscript alignment

The default configuration now selects the manuscript Huber path. See
[PAPER_ALIGNMENT.md](PAPER_ALIGNMENT.md) for implemented equations, required
measured inputs, and unresolved calibration details. The published slopes alone
are insufficient to reconstruct the fitted model because its intercept is absent.
The old Ridge approximation remains available through the explicit
`LOAD_MODEL_METHOD = "legacy_ridge"` setting.

## Required Inputs

### 1. OpenStreetMap GeoJSON exports

Download three GeoJSON files from [Overpass Turbo](https://overpass-turbo.eu/)
using the queries described in the accompanying paper. Save them to:

| File | Contents |
|---|---|
| `data/OSM_extracts/export_{region}_buses.geojson` | Transmission substations (`power=substation`) |
| `data/OSM_extracts/export_{region}_lines.geojson` | Transmission lines (`power=line`) |
| `data/OSM_extracts/export_{region}_plants.geojson` | Power plants (`power=plant`) |

Replace `{region}` with the lowercase name set in `config.py` (e.g. `ghana`).

### 2. Load model and industrial features

Paper mode requires `data/bus_data/models/model_huber_paper.json`, measured VIIRS
radiance, and normalized steel/gold facility data described in
[PAPER_ALIGNMENT.md](PAPER_ALIGNMENT.md). These complete calibration inputs are
not bundled. Missing inputs stop the paper path with a diagnostic.

Legacy mode uses `data/bus_data/models/model_bundle_v5.pkl`, the earlier Ridge
model, imputer and feature list.

### 3. Region configuration (`config.py`)

Edit `create_new_model/config.py` to set:

```python
region            = "ghana"         # lowercase, used in all file names
COUNTRY_ISO3      = "GHA"           # ISO 3166-1 alpha-3 code
REGION_BBOX       = (-5.5, 4.0, 2.0, 13.0)  # (min_lon, min_lat, max_lon, max_lat)
SHAPE_FILE_NAME   = "Ghana"         # country name in NaturalEarth "name" column
COUNTRY_DEMAND_MW = 2451.0          # total domestic Pd (MW) from a reference case
NATIONAL_EXPORT_MW = 0.0            # net export (MW) split across cross-border buses
```

---

## Optional Speed Improvement

Step 1 (`create_buses.py`) performs reverse geocoding to name substations that
lack OSM name/ref tags. Install the offline geocoder for dramatically faster
performance:

```bash
pip install reverse_geocoder
```

Without it the script falls back to the Nominatim API (1 request/second).

---

## Pipeline Steps

All scripts are run from the `create_new_model/` directory.

### Step 1 — `create_buses.py`

**Input:** `data/OSM_extracts/export_{region}_buses.geojson`

**Outputs:**
- `data/working_files/{region}/matpower_{region}_buses.csv`
- `data/working_files/{region}/transformer_pairs_{region}.csv`

Reads OSM substation features and builds the MATPOWER bus table.

Key behaviours:
- **Compound-voltage substations** (e.g. voltage tag `330000;161000`) are split
  into one bus per voltage level, each named `{SubstationName}_{kV}kV`.
  A transformer branch connecting the two voltage levels is recorded in
  `transformer_pairs_{region}.csv` and included automatically in Step 8.
  Transformer parameters: 1.6 % impedance on 1000 MVA base (X/R = 80).
- Bus names are derived from OSM `name`/`ref` tags first (instant), then by
  batch offline reverse geocoding (`pip install reverse_geocoder`), falling
  back to Nominatim if the library is not installed.
- All buses default to `type = 1` (PQ); types 2 and 3 are assigned in Step 6.

---

### Step 2 — `create_lines.py`

**Input:** `data/OSM_extracts/export_{region}_lines.geojson`,
`data/working_files/{region}/matpower_{region}_buses.csv`

**Outputs:**
- `data/working_files/{region}/matpower_{region}_lines.csv`
- `data/working_files/{region}/processed_{region}_lines.csv`
- `data/working_files/{region}/island_buses_{region}.csv`

Processes OSM line ways: removes sub-transmission lines, filters busbar
segments inside substations, merges consecutive OSM ways into single physical
lines, and assigns each line endpoint to the nearest voltage-compatible bus.
Unassigned endpoints are flagged for manual review in Step 3.

---

### Step 3 — `resolve_unmatched.py`

**Input:** `data/working_files/{region}/matpower_{region}_lines.csv`,
`data/working_files/{region}/matpower_{region}_buses.csv`,
`data/working_files/{region}/processed_{region}_lines.csv`

**Outputs:**
- `data/working_files/{region}/matpower_{region}_lines_resolved.csv`
- `data/working_files/{region}/matpower_{region}_buses.csv` (updated)
- `data/working_files/{region}/unmatched_review_{region}.html` (map)

Opens an interactive Folium map in a browser. Each unmatched endpoint marker
popup shows the exact coordinates and a Google Maps link for easy lookup.

Available commands at each CLI prompt:

| Command | Action |
|---|---|
| `<bus_i>` | Assign this endpoint to an existing bus |
| `skip` | Leave for later in the session |
| `delete` | Delete the entire line |
| `new` | Create a new substation at the endpoint location |
| `merge <N>` | Merge this line with line N from the unmatched list |

After manual review the script:
1. Removes still-unmatched lines.
2. Computes R, X, B, rateA from empirical power-law fits.
3. **Fills missing bus voltages** from connected line voltages.
4. **Assigns the most common regional voltage** to any buses still unassigned.
5. **Removes unconnected buses**, considering both lines and transformers, and
   applies the same bus renumbering to both branch tables.

The `new` command stores voltages in volts internally. An optional secondary
voltage creates a separate bus and an explicit transformer connection.

---

### Step 4 — `build_region_features.py`

**Input:** `data/working_files/{region}/matpower_{region}_buses.csv`

**Output:** `data/bus_data/features/{region}_paper_features.csv` (paper mode),
or `{region}_features_with_industrial.csv` (legacy mode).

Builds a Voronoi tessellation over bus coordinates clipped to the country
boundary, then extracts:

- **WorldPop** population (auto-downloaded if not cached)
- **OSM industrial polygons** (Overpass API; auto-cached)
- **OSM road length** (motorway + trunk)
- **OSM commercial nodes** (shops, banks, marketplaces)
- **Industrial VIIRS features, steel MW and gold sites** in paper mode.
- **NTL / GHSL** via a GEE cache or NZ proxy in legacy mode.

To add known industrial sites with incomplete OSM coverage, populate
`KNOWN_REGIONAL_INDUSTRIAL` in this script with
`(lat, lon, bucket_type, label)` tuples.

---

### Step 5 — `predict_bus_loads.py`

**Input:** Bus feature CSV, model bundle, `data/working_files/{region}/matpower_{region}_buses.csv`

**Output:** `data/bus_data/processed/{region}_substation_predictions_paper.csv`
(paper mode), or `{region}_substation_predictions_v4.csv` (legacy mode).

Paper mode uses the six Huber features and polygon-level industrial guards, then
scales all represented domestic substations and assigns exports separately.

Predicts per-bus active load (Pd) and reactive load (Qd = Pd × tan(arccos(0.95))).
Pipeline: Legacy mode: Ridge prediction → city-cluster rescaling → industrial addons →
captive-power discount → generation bus pinning → lowest-voltage domestic bus selection → national scaling →
national export distribution across cross-border buses. Domestic demand is
preserved after voltage selection; export buses are exempt from that filter.

Cross-border buses are detected from the NaturalEarth country boundary.
Positive configured exports with no detected external buses cause an error
instead of silently omitting the export demand.

---

### Step 6 — `create_generators.py`

**Input:** `data/OSM_extracts/export_{region}_plants.geojson`,
`data/working_files/{region}/matpower_{region}_buses.csv`

**Output:** `data/working_files/{region}/matpower_{region}_generators.csv`
(also updates bus types in the buses CSV)

Reads OSM power plant data, normalises capacity and fuel type, deduplicates
plants, assigns each to its nearest bus, then assigns MATPOWER bus types:

| Type | Condition |
|---|---|
| 1 (PQ) | Bus has no generator |
| 2 (PV) | Bus has at least one generator |
| 3 (slack) | Bus connected to the generator with the highest Pmax |

---

### Step 7 — `create_cost_curves.py`

**Input:** `data/working_files/{region}/matpower_{region}_generators.csv`

**Output:** `data/working_files/{region}/matpower_{region}_cost_curves.csv`

Computes MATPOWER polynomial cost curves (MODEL=2, n=3) per generator.
Coefficients are parameterised by fuel type (Snodgrass et al. 2021).
Zero-cost fuels (solar, wind, nuclear, geothermal) receive c0 = c1 = c2 = 0.

---

### Step 8 — `create_synthetic_grid.py`

**Input:** All CSV outputs from Steps 1–7.

**Output:** `data/working_files/{region}/{region}_synthetic_grid.m`

Assembles the MATPOWER `.m` case file. Bus types are read from the buses CSV
(updated in Step 6) and merged with the Pd/Qd predictions CSV.
Intra-substation transformer branches (from `transformer_pairs_{region}.csv`)
are appended to the branch matrix automatically.

---

### Step 9 — `ensure_feasibility.py`

**Input:** `data/working_files/{region}/{region}_synthetic_grid.m`

**Outputs:**
- `matpower_files/{region}_dc_feasible.m`
- `matpower_files/{region}_ac_feasible.m`

Attempts two feasibility relaxations and exports a variant only when the
solver succeeds and the resulting case passes independent residual checks:

**DC relaxation** (`eq:slack_dcopf`):
Solves a DC OPF with slack variables on line thermal limits. The penalty term
M·Σ(P_sl²) minimises the magnitude of any needed limit expansion. Generator
Pg values are set to the optimal DC dispatch; Qg = 0.

**AC relaxation** (`eq:shunt_acopf`):
Solves an AC OPF with additional shunt susceptance variables at each bus.
The penalty term M·Σ(B_sh²) follows manuscript equation (5a). It reduces
shunt magnitudes but does not directly minimise the number of shunt devices.
Generator Pg and Qg are set to the optimal AC dispatch values.

Run directly:

```bash
python ensure_feasibility.py
```

---

### Step 10 — `plot_grid_final.py`

**Input:** `data/working_files/{region}/processed_{region}_lines.csv`,
`data/working_files/{region}/matpower_{region}_buses.csv`

**Output:** `data/working_files/{region}/grid_{region}.png`

Plots the finished transmission network on a country map. Voltage classes
are detected automatically from the data; lines are coloured by voltage level.
The legend shows line voltages only.

---

## Optional Inputs

| File | Purpose |
|---|---|
| `data/bus_data/raw/gee_features_{region}_matpower.csv` | Real VIIRS NTL and GHSL built-up area from Google Earth Engine, overriding the NZ proxy. |
| `data/bus_data/raw/global_power_plants.csv` | WRI Global Power Plant Database for industrial polygon correction in Step 4. |

---

## Dependencies

```bash
pip install geopandas pandas numpy scipy scikit-learn rasterio shapely \
            geopy folium matplotlib requests joblib reverse_geocoder
```

Python 3.10+ is required.

---

## Adapting to a New Region

1. Edit `config.py` with the region name, ISO code, bounding box,
   NaturalEarth country name, total demand, and net export.
2. Download the three Overpass GeoJSON files to `data/OSM_extracts/`.
3. Verify the region boundary and the detected cross-border buses.
4. Run Steps 1–10 in order.

## Correctness checks and regression tests

Use Python 3.10+ and an isolated environment. From the repository root on Windows:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
.\.venv\Scripts\python.exe -m pytest -q
```

The tests cover bus/transformer renumbering, transformer-only connections,
manual voltage tiers, empty transformer files, demand/export conservation,
nighttime-light gating, CSV-to-MATPOWER assembly, failed solver rejection,
and DC/AC solutions. Network admittances and solved AC balances are compared
against PYPOWER, including a standard nine-bus case. Test fixtures and outputs
are written to temporary directories.

The feasibility solver enforces balance at every bus, uses one reference angle
per connected component, and includes taps, phase shifts, charging, existing
bus shunts, thermal limits, and branch angle limits. It stores dispatch and
shunts in MATPOWER MW/MVAr units. Solver iteration limits are failures, not
convergence. Returned cases must pass a 0.0001 MVA balance/thermal tolerance;
voltage and angle bounds use a 1e-7 tolerance. Export retains sufficient numeric
precision for round-trip validation.

**Previously generated files are not repaired automatically.** Some checked-in
regional transformer tables contain stale IDs or name/ID mismatches. Rebuild
steps 1–8 in order from the OSM inputs and repeat any manual endpoint review
before running feasibility. Do not mix new tables with older predictions,
generator assignments, or cost curves. Unknown or stale transformer endpoints
now stop assembly instead of silently producing an incorrect network.

The SLSQP implementation still uses dense matrices and is intended for modest
networks. Passing these tests does not establish convergence or performance for
a statewide Texas case, or validate the geographic/load assumptions. A large
regional case may require a sparse OPF solver and further data calibration.
