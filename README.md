# Synthetic Electric Grid Creation Framework

A Python pipeline for building exploratory synthetic electric grid models
from openly available geospatial data (OpenStreetMap, WorldPop, WRI Global
Power Plant Database, NaturalEarth).

The framework produces MATPOWER Version 2 cases and validation reports.
Geographic coverage, inferred equipment, and operating feasibility are separate
questions; solver convergence alone does not establish real-grid accuracy.

**The numbered pipeline in `create_new_model/` is the Automate Grid Creation paper's method.**
Figures 2, 4 and 7 of the paper map onto `create_buses.py`, `create_lines.py` and
`create_generators.py` respectively, and its Kenya case reproduces Table VII
exactly. Follow [Pipeline Steps](#pipeline-steps) for a build intended to match
the paper.

The shared topology engine and the scenario/dispatch workflow in
[MODEL_QUALITY.md](MODEL_QUALITY.md) are a separate, stricter track: they preserve
station sections around conflicting connections, require reviewed transformer
evidence, and never expand equipment limits to obtain an AC solution. That track
is useful for equipment-evidence work but is **not** the published method and does
not reproduce Table VII.

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

Choose the load-model path before starting the numbered pipeline. **Neither path
runs out of the box without the inputs in its last column.**

| Setting in `create_new_model/config.py` | Purpose | Additional inputs |
|---|---|---|
| `LOAD_MODEL_METHOD = "paper_huber"` (default) | The published load equations; the exact fitted calibration is unavailable | Measured VIIRS, normalized steel/gold facilities, fitted Huber artifact |
| `LOAD_MODEL_METHOD = "legacy_ridge"` | Earlier exploratory approximation | Bundled `model_bundle_v5.pkl`; regional inputs and feature extraction still required |

Paper mode requires `data/bus_data/models/model_huber_paper.json`, measured VIIRS
radiance, and normalized steel/gold facility data. These complete calibration
inputs are not bundled, and the published slopes alone cannot reconstruct the
fitted model because its intercept is absent. Missing inputs stop the paper path
with a diagnostic.

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

Step 1 records these settings into `data/working_files/{region}/build_config.json`
together with a SHA-256 of the OSM input, and Steps 2 and 3 verify against it.
Editing `config.py` part-way through a region is refused with a list of what
changed, instead of silently mixing settings into one model. A region built
before that file existed adopts the current settings with a warning that they
were never checked against Step 1.

These seven settings are the user inputs of the Automate Grid Creation paper's Table VI. Two further
settings control Automate Grid Creation paper behaviour and should normally be left alone:

| Setting | Default | Effect |
|---|---|---|
| `INFER_TRANSFORMERS_FROM_VOLTAGE` | `True` | Sec. III.B. Joins the buses of a compound-voltage substation with a transformer. Setting this `False` splits the buses but omits the transformer, stranding every upper tier as an island — Kenya drops from 143 branches to 134 and no longer matches Table VII. |
| `FUEL_COST_BASIS` | `"us"` | Table IV. `"us"` reproduces the published Table V coefficients. `"global"` rescales each curve by that fuel's global/US price ratio; the Automate Grid Creation paper prints the global column but never states how to apply it, so results using it must be reported as departing from Table V. |

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

Install dependencies using the [setup instructions](#dependencies), edit
`create_new_model/config.py`, and run this availability check from the repository
root **before spending time on manual review**:

```bash
python create_new_model/check_inputs.py
```

It reports all missing local prerequisites together and exits with status 1 if
any are absent. It also checks `SHAPE_FILE_NAME`, `COUNTRY_ISO3` and
`REGION_BBOX` against the bundled NaturalEarth boundaries, because each of those
three inputs otherwise fails quietly: a name NaturalEarth does not have disables
cross-border detection, an ISO code it does not have clips every Voronoi cell to
the bounding rectangle, and a box that misses the country clips the network
away. A box smaller than the country is reported as a `NOTE` rather than an
error, since a study area may legitimately be smaller. It does not validate file
contents or guarantee a successful run.
The default paper path will report missing inputs in a fresh checkout; supply
them, or explicitly choose the legacy approximation if that matches your purpose.

Then run the numbered scripts from the `create_new_model/` directory:

```bash
cd create_new_model
python create_buses.py
python create_lines.py
python resolve_unmatched.py          # or: python resolve_unmatched.py --auto
```

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
| `skip` | Leave unmatched; the line is removed when review finishes |
| `delete` | Delete the entire line |
| `new` | Create a new substation at the endpoint location |
| `tap` | The endpoint tees into another line: split that line at the nearest point and join the three at a new junction bus |
| `merge <N>` | Merge this line with line N from the unmatched list |

After manual review the script:
1. Removes still-unmatched lines.
2. **Raises each line's length to the distance between the buses it connects.**
   Equation (3) scales R, X and B by length, and that length is the OSM geometry
   which survived preprocessing. An endpoint assigned to a substation the
   geometry does not reach leaves the two describing different conductors, and a
   conductor cannot be shorter than the straight line between its endpoints.
   Without this floor more than half the lines in most regions carried the
   impedance of a shorter conductor, the worst at 1/2500 of its own floor.
3. Computes R, X, B, rateA from empirical power-law fits.
4. **Fills missing bus voltages** from connected line voltages.
5. **Assigns the most common regional voltage** to any buses still unassigned.
6. **Removes unconnected buses**, considering both lines and transformers, and
   applies the same bus renumbering to both branch tables.

The `new` command stores voltages in volts internally. An optional secondary
voltage creates a separate bus and an explicit transformer connection.

This is the human-in-the-loop stage of Sec. IV.B.3; candidate buses are listed
within the Automate Grid Creation paper's 80 km radius.

Two rules the automated passes enforce now apply to manual picks as well: a bus
already used as the line's other endpoint is refused, because that would be a
self-loop and Step 10 deletes it, losing the line; and a bus whose voltage does
not match the line requires an explicit confirmation, recorded as a deliberate
override.

Every answer is written to `review_decisions_{region}.json`. That file is the
audit trail for the session, and it can be replayed to reproduce the same model
without prompting:

```bash
python resolve_unmatched.py --replay
```

Replay reruns the recorded answers in order. Because the endpoint iteration order
is fixed by the input tables, the same OSM extract plus the same journal yields
the same case. Commit the journal alongside the region so the build can be
reproduced by someone who was not present at the original review.

Each recorded answer is checked against the endpoint, network join or follow-up
question it was recorded for. If an answer would reach a different one, or the
review ends with answers unused, replay stops with an error that names the first
answer that no longer fits. That usually means the journal was edited at or
before that answer. The journal file is not rewritten until a replay has used
every answer, so the edited version is kept for fixing. Step 3 updates the bus
and transformer tables in place, so rebuild from Step 1 before replaying again.

**Review limitation:** the manual review prompts once per unmatched endpoint, has
no saved review session, and does not revisit `skip` answers. On a large region
that is thousands of prompts (the checked-in Australia lines leave 1,501
endpoints unmatched). Automatic mode, below, removes the prompts but not the
need for review. The shared topology workflow in
[MODEL_QUALITY.md](MODEL_QUALITY.md) exports review reports instead, but it
requires OSM JSON with node identities, produces different topology tables, and
is not a drop-in replacement for this step.

#### Automatic mode

```bash
python resolve_unmatched.py --auto
```

Answers every prompt by rule instead of asking. **This goes beyond the Automate
Grid Creation paper**, whose Sec. IV.B.3 puts a reviewer here and whose
conclusion names full automation as future work, so a case built this way is not
a reviewed model. It gives a first model of a large region without thousands of
prompts; the decisions it made still need review.

**How it answers.** Each rule answers with the command a reviewer could have
typed (a bus number, `merge`, `new`, `tap`, `delete` or `skip`), and that answer
goes through the same prompt as a manual one. The checks on manual picks still
apply: a self-loop is refused, and a bus at another voltage is accepted only with
the same explicit `y` a reviewer would give. No map is drawn and no person is
ever asked anything. If no rule supplies an answer, or the prompt refuses a
rule's answer three times, the run stops with an error rather than guessing or
waiting for input.

**Rules.** Endpoints are visited in three stages, so nothing is added or deleted
while a safer answer is still available elsewhere. For each endpoint the rules
are tried top to bottom and the first that applies answers. A rule stays active
in every stage after the one it starts in, and an endpoint no rule settles comes
up again in the next stage.

| From stage | Rule | Answer |
|---|---|---|
| 1 | Both ends of the line are open and within 0.5 km of each other | `delete`: it can only become a self-loop |
| 1 | A substation of the line's voltage within 0.5 km (d_junc, Sec. IV.B.2) | assign it |
| 1 | Exactly one other open line end within 0.5 km | `merge`: it is the rest of this line |
| 1 | Two or more other open ends within 0.5 km, or one whose merge would loop back to this line's own bus | `new` junction bus; the other ends then attach to it by the rule above |
| 1 | The end lies within 0.1 km of another line, away from that line's ends | `tap` |
| 1 | A substation of the line's voltage within 10 km (Step 2's `BUS_ASSIGN_THRESHOLD_KM`) | assign the nearest |
| 2 | The end is within 2 km of the line's own other substation | `delete`: a self-loop |
| 2 | A substation at another voltage within 2 km | assign it across the voltage guard; POST-PROCESSING 1b adds the tier and transformer |
| 3 | A dead end on a line of 2 km or more | `new` inferred substation at the endpoint, cross-border ends included |
| 3 | Anything else | `delete`: a short stub, or a fragment attached to nothing |

When a merge leaves the joined line with another open end, that end is decided
straight away, in the same stage.

**Joining separate networks.** After the endpoint pass, POST-PROCESSING 3 adds a
line between two separate networks only when the gap is a voltage-compatible hop
of at most 50 km. It repeats until the grid is one network or no gap qualifies.
Wider gaps, and gaps between different voltages, are left and reported, and
`check_topology.py` flags them. For a region that really is islanded, set
`EXPECT_CONNECTED_GRID = False` in `config.py`; this step is then skipped in both
modes.

**Finding what it changed.** The journal, `review_decisions_{region}.json`,
records `"mode": "automatic"` and, beside every answer, the endpoint it answers
(OSM id, which end, coordinates), the rule that chose it and the reason. The run
ends with a count of decisions per rule. In the output tables:

- buses it created are named `Junction` or `Inferred substation` followed by the
  coordinates;
- lines it added to join networks have `osm_id` `review:connect`;
- `fbus_resolve_pass` and `tbus_resolve_pass` record how each end was settled
  (`manual`, `manual_merge`, `manual_tap`, `manual_new`, `tap_split`, `connect`).

**Replaying and correcting.** `--replay` reruns an automatic journal in the same
three stages and reproduces the case exactly; `--auto` and `--replay` cannot be
combined. A correction made in the journal replays cleanly when the same prompts
accept it as accepted the answer it replaces: one bus number for another bus at
the line's voltage that is not the line's other end, for example. Some changes
move later answers onto the wrong prompts: changing how many answers a decision
takes (a `new` takes four, a bus number one), or which endpoints come up
(turning a `delete` into an assignment brings the line's other end up for
review). Replay stops at the first answer that moved, as described above, rather
than applying it.

The thresholds are named at the top of `resolve_unmatched.py` (`AUTO_*`).
Tested on fresh Step 2 output for Uruguay, Kenya, Taiwan and Colombia: every
case came out with no self-loops and no voltage-incompatible endpoints. Uruguay
and Taiwan came out as one network each; Kenya kept three gaps and Colombia
seven, all wider than 50 km or between different voltages.

---

### Step 4 — `build_region_features.py`

Before extraction, run `python check_inputs.py --stage 4`. This checks the bus
table and load-path prerequisites, including the model needed by Step 5, before
starting potentially lengthy downloads. Then run `python build_region_features.py`.

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

Two rules protect the features a region is built on:

- **Any source that cannot be read is named at the end of the step**, together
  with the predictor it leaves at zero for every bus. A failed Overpass call
  used to zero `road_density` in silence, which is a different load model
  rather than a slower run.
- **A cached satellite feature table is refused unless it describes the current
  buses.** Every column in `gee_features_{region}_matpower.csv` is a reduction
  over one bus's Voronoi cell, and the join key is the bus position, so a
  rebuilt region would take another substation's nighttime lights. Delete the
  file and re-run the step to fetch features for the current buses.

---

### Step 5 — `predict_bus_loads.py`

Run `python check_inputs.py --stage 5`, then `python predict_bus_loads.py`.
Feature filenames must match the configured load-model method; rerun Step 4
if you change methods.

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

Cross-border buses are the ones nearer another country's NaturalEarth polygon
than their own. Merely falling outside the boundary is not enough: the 110 m
coastline is simplified by kilometres, so coastal substations in Taiwan,
Australia and Puerto Rico plot outside countries that have no land border.
Buses outside the boundary but nearest their own country are reported and kept
domestic. Positive configured exports with no detected external buses cause an
error instead of silently omitting the export demand.

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

Computes MATPOWER polynomial cost curves (MODEL=2, n=3) per generator using
Table V of the Automate Grid Creation paper, parameterised by fuel type and Pmax
(Snodgrass et al. 2021).
Zero-cost fuels (solar, wind, nuclear, geothermal) receive c0 = c1 = c2 = 0.

Table IV's fuel prices are selected by `FUEL_COST_BASIS`. The default `"us"`
reproduces Table V exactly. `"global"` multiplies all three coefficients of a
generator's curve by that fuel's global/US price ratio, which reference [48]'s
heat-rate construction makes exact. Hydro and the zero-cost categories are never
rescaled; internal combustion is rescaled but flagged approximate because [48]
built it from ERCOT bids.

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
The penalty term M·Σ(B_sh²) follows equation (5a) of the Automate Grid Creation paper. It reduces
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

Python 3.10+ is required. From the repository root, create and activate a virtual
environment, then install the runtime requirements (including `matplotlib`):

```bash
python -m venv .venv
```

Activate on Windows PowerShell with `.\.venv\Scripts\Activate.ps1`, or on
macOS/Linux with `source .venv/bin/activate`, then run:

```bash
python -m pip install -r requirements.txt
```

`reverse_geocoder` is an optional speed improvement for Step 1. The test
requirements include the runtime requirements, so both installation paths use
the same dependency list.

## Troubleshooting Steps 4 and 5

Run `check_inputs.py --stage 4` or `--stage 5` from `create_new_model/` first.
Missing paper data or calibration must be supplied; installing more Python
packages cannot replace them. A new Huber refit is not the published fitted model.

If the availability check passes but a step fails, retain the full traceback,
the exact command, Python version, region, and `LOAD_MODEL_METHOD`. Step 4 also
uses downloaded boundaries, WorldPop, and Overpass data; the availability check
does not test those services. Step 5 validates model/features and demand/export
allocation. Positive exports without detected external buses are an explicit
error. These are possible failure categories, not a diagnosis without the error.

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

`tests/test_paper_methodology.py` pins the implementation to the Automate Grid Creation paper:
Table VII's Kenya counts, the Sec. III.B compound-voltage transformer, the
Table V coefficients, the Table IV price ratios, and the Sec. IV.B.1 Step 3
voltage-filter rules.

The remaining tests cover bus/transformer renumbering, transformer-only connections,
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

### Checking a built region

```bash
python create_new_model/check_topology.py --region kenya
```

Reports the three defect classes that the automated passes cannot create but the
manual review of Step 3 can: self-loops left after review, endpoints whose bus
and line voltages disagree, and `base_kV` values written in kV instead of volts.
It also prints the busiest bus and the next busiest, which exposes a review
session that dumped many endpoints onto one bus. Exit status is 1 when any
defect is found. **Every case currently checked into this repository fails it**,
including the four regions of Table VII, so treat those files as needing a
rebuild rather than as reference models.

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
