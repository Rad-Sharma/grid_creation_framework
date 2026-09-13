# Alignment with the supplied manuscript

Reference: **A Framework for the Creation of Highly Realistic International
Synthetic Electric Grid Models**, supplied as `Automate Synthetic Grid Creation.pdf`
(16 pages). Equation/table references below refer to that manuscript version.
The earlier `Rich-Machine/automate_grid_creation` repository was also inspected
at commit `4187354a`; it contains a Ghana prototype, not the final Huber model.

## Implemented methods

| Manuscript | Implementation |
|---|---|
| Eq. (1), Huber epsilon=1.1, six log(1+x) inputs | `paper_load_model.py` validates a Huber artifact and predicts with the specified feature ordering. `train_paper_load_model.py` fits a new Huber model from complete metered training rows. |
| Eq. (2), polygon-level VIIRS guard | `paper_features.py` measures each industrial footprint before aggregation. Missing raster coverage is an error, not zero load. No positive light floor or NZ proxy is used in paper mode. |
| Table I, steel and gold predictors | Normalized electrical-MW steel facilities and Fineprint gold-producing mine sites are assigned to service areas. Point facilities must have an enclosing industrial footprint or a supplied polygon. |
| Sec. III.C.6–7, demand and reactive load | Domestic predictions are scaled to total domestic MW, exports are assigned separately, and Qd uses lagging power factor 0.95. |
| Eq. (3a–d), line fits | R/km = 1115 V^-2.9870; X/km = 52.86 V^-2.0342; B/km = 0.000049 V^0.6896; rating = 0.018429 V^1.8998. V is kV, length is km, electrical parameters use the 100 MVA base. |
| Sec. V.B.1, unnamed generators | Cached Overpass queries find the closest named power plant, generator or substation; inferred names are bracketed and retain the source OSM ID. |
| Table III, reactive limits | Existing matching coefficients retained. |
| Table V, costs | Missing steam, combined-cycle and internal-combustion categories added. Available generation-method tags distinguish technology from fuel. |
| Eq. (4), DC relaxation | Squared nonnegative line-limit slack penalty, plus full nodal balance and validation. |
| Eq. (5a), AC relaxation | Squared added shunt penalty, replacing the earlier absolute-value penalty. Correct units and all-bus balance checks retained. |

## What is still required for numerical reproduction

**No exact manuscript Huber model has been reconstructed.** The manuscript gives
six rounded slopes but omits the fitted intercept, complete GB industrial-data
correction, and precise training aggregation recipe. The supplied Ridge model
is not substituted. The earlier repository does not supply the missing fit.

The bundled NZ/GB CSVs contain placeholders for industrial-polygon nighttime
light. They cannot be relabeled as fully measured paper features. A valid Huber
artifact must either come from the authors' final calibration or be a new refit
using regenerated features. New refits retain input SHA-256 hashes and are
explicitly labeled `new_refit_not_manuscript_artifact`.

The bundled GEM workbook reports nominal steel capacity in **thousand tonnes
per year**, not electrical MW. A normalized GEM-derived GeoJSON must contain
`facility_id`, `capacity_mw`, and geometry. `prepare_steel_sites.py` can convert
tonnage only when an explicit, documented MWh/tonne intensity is supplied:

```powershell
.\.venv\Scripts\python.exe create_new_model/prepare_steel_sites.py INPUT.xlsx OUTPUT.geojson --mwh-per-tonne YOUR_VALUE
```

That conversion produces an assumption-based annual-average electrical capacity;
it does not recover the paper's unspecified conversion or measured plant MW.
The helper rejects ambiguous operating-site capacities rather than silently
dropping facilities.

## Running the paper path

`config.LOAD_MODEL_METHOD = "paper_huber"` is the default. It requires:

1. `data/bus_data/raw/viirs_{region}.tif`: georeferenced **mean VIIRS radiance**
   in nW/cm²/sr, with valid nodata metadata, covering industrial footprints.
   The date/averaging period must be selected consistently with the demand case.
2. `data/bus_data/raw/steel_facilities_mw.geojson`: normalized GEM facilities
   with electrical MW and a documented source/conversion.
3. `data/bus_data/raw/gold_facilities.geojson`, or the bundled
   `gem/fineprint_mine_production.zip`. Fineprint records are source-snapshot
   observations, not a claim that every historical mine is currently operating.
4. `data/bus_data/models/model_huber_paper.json`: a fitted model with the
   feature contract, six coefficients, intercept, epsilon and provenance.

Run feature extraction and prediction as usual. Paper outputs have separate
names: `{region}_paper_features.csv` and
`{region}_substation_predictions_paper.csv`. Assembly requires the paper
prediction file and will not silently load older Ridge predictions or zero loads.

To fit a new model, first produce complete NZ/GB feature tables with metered
`avg_mw` and `feature_contract=paper_industrial_viirs_v1`, then run:

```powershell
.\.venv\Scripts\python.exe create_new_model/train_paper_load_model.py prepared_nz.csv prepared_gb.csv
```

The refitter uses epsilon=1.1 and explicitly records alpha=0 (the manuscript does
not specify regularization). It refuses missing, nonfinite or rank-deficient
training features. It does not invent the missing GB correction or claim the
new coefficients reproduce Table II.

Set `LOAD_MODEL_METHOD = "legacy_ridge"` only when deliberately choosing the
previous approximation. It retains the earlier load-model path and filenames;
it is not manuscript reproduction.

## Explicit interpretation choices

- Table I labels mining/gold as NTL products, but labels chemical area and steel
  capacity in km²/MW. The implementation uses continuous guard multiplication
  for mining/gold and a positive-guard gate for chemical/steel, preserving the
  latter units. This resolves an ambiguity in the prose about guarding "all"
  industrial features; the exact training transformation needs author confirmation.
- Co-located voltage-tier buses share one physical substation service area. Its
  demand goes to the lowest modeled tier. Paper mode does not discard all buses
  above the minimum voltage of the entire country.
- A facility is assigned once, using its representative point; exact boundary
  ties are deterministic. Its VIIRS mean is measured over the facility footprint.
- The manuscript does not specify a name-search radius. The default is 1000 m,
  configurable in `config.py`. Inferred names require review; capacity and fuel
  are never copied from the neighboring object.
- Table V takes precedence over the conflicting narrative about hydro being
  zero-cost. Missing generation-method tags still require a documented category
  assumption or manual correction.
- Eq. (5a)'s quadratic penalty reduces shunt magnitudes; it does not minimize
  device count. Feasibility is checked after solving; no universal convergence
  guarantee is asserted.

## Validation scope

Tests cover the published line/cost coefficients, quadratic shunt objective,
polygon guards (including dark and bright facilities within one service area),
missing-input rejection, Huber fitting/prediction, national scaling, cached name
inference, and the previous electrical/table integrity regressions. The local
CSV-to-AC-case and PYPOWER comparisons remain in the suite.

These checks establish implementation behavior. They do not establish geographic
accuracy, recover the manuscript's reported regional results, or validate a Texas
case. Existing generated cases need rebuilding after the input/calibration gaps
are resolved.
