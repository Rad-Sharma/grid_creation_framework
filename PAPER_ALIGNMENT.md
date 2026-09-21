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
| Eq. (4), DC relaxation | Squared nonnegative line-limit slack penalty, plus full nodal balance and validation. HiGHS settles feasibility exactly and warm-starts a trust-region solve of the stated QP. See "Solving equations (4) and (5)" below. |
| Eq. (5a), AC relaxation | A reactive source at each candidate bus stands in for `B_i^sh`, is squeezed towards zero, and is then converted into the bus shunt the equation installs. The exported case is re-solved with no source present, so what `check_solution` validates is what is exported. Deviates from the literal quadratic penalty; see below. |
| Table IV, fuel costs | `create_cost_curves.py` carries both the US and global $/MBTU columns. `FUEL_COST_BASIS = "us"` (default) reproduces Table V exactly; `"global"` rescales each curve by that fuel's global/US ratio. |
| Table VI, user inputs | All seven inputs are `config.py` settings. `MIN_TRANSMISSION_VOLTAGE` reaches the shared topology engine through `network(..., min_kv=)` instead of a hardcoded 69 kV floor. |
| Sec. III.B, compound voltages | A compound voltage tag splits into one bus per level and the consecutive tiers are joined by a transformer (1.6 % on 1000 MVA, X/R = 80). |
| Sec. IV.B.1 Step 3, voltage filter | A line is removed only when its voltage is *known* and sub-transmission. Untagged lines are retained; the shared engine assigns the regional modal voltage and records each one in the audit. |
| Sec. IV.B.2 Steps 6-9, bus assignment | `create_lines.py` implements d_junc = 0.5 km junction detection, three-orientation segment merging, proximity matching with voltage compatibility, and trajectory extrapolation with a cross-track acceptance test. |
| Table VII, reported models | Kenya reproduces exactly (92 buses, 75 generators, 143 branches); see "Verification against Table VII" below. |

## Equation (3c) does not scale with voltage the way per-unit susceptance must

`B per km = 4.9e-5 * V^0.6896` (p.u. on 100 MVA, V in kV) is implemented exactly
as published, but the exponent is not physically consistent, and on a
132 kV-dominated network it is the largest single source of reactive trouble.

Per-unit shunt susceptance is `b_pu = omega * C * Z_base` and `Z_base = V^2/100`,
so with the physical capacitance per km of an overhead line roughly constant
across voltage classes, **`b_pu` must scale close to `V^2`**. Equation (3b) for X
is consistent in exactly this sense - it implies about 0.43 ohm/km at every
voltage - but (3c) implies a capacitance that *falls* as `V^-1.31`:

| Voltage | Eq. (3c) b_pu/km | Implied C | Real overhead line | Implied surge impedance | Real |
|---|---|---|---|---|---|
| 132 kV | 1.42e-3 | **26.0 nF/km** | ~9-10 nF/km | **234 ohm** | ~390 ohm |
| 220 kV | 2.02e-3 | 14.9 nF/km | ~11-12 nF/km | 341 ohm | ~310 ohm |
| 400 kV | 3.05e-3 | **6.1 nF/km** | ~12-14 nF/km | **475 ohm** | ~275 ohm |

Real lines go the other way: surge impedance *falls* with voltage as conductors
are bundled. So (3c) is wrong in shape, not just in scale, and no single
multiplier repairs it.

Measured effect on the rebuilt Kenya main island (111 buses, 1645 MW): total line
charging is **1141.7 MVAr** against 540.8 MVAr of reactive demand, while the
generators can absorb only **-217.7 MVAr** between them. Equation (5) has to
install shunts at 108 of the 111 buses, netting **-142 MVAr**, i.e. most of the
reactive plant the relaxation adds exists to soak up charging that should not be
there.

Refitting (3c) alone is not enough. Anchoring B on (3b) through a realistic
surge impedance was tried and makes Kenya worse, because (3b)'s 0.43 ohm/km is
itself too high for a bundled 400 kV line, so the derived 400 kV charging comes
out ~3x the published value. **R, X and B need refitting together, per voltage
class, with B constrained to the `V^2` form.** That is a decision for the
authors, so the published coefficients are left in place and
`tests/test_paper_alignment.py::test_line_coefficients_match_equations_3a_to_3d`
still pins them.

## Solving equations (4) and (5)

Both relaxations were previously handed to `scipy.optimize.minimize(method="SLSQP")`
with finite-difference Jacobians. On the published 84-bus Ghana reference that
implementation ran for **18 minutes and then failed** with "Positive directional
derivative for linesearch". It is replaced as follows.

- **Eq. (4)** is a convex QP with linear constraints. HiGHS first answers
  feasibility exactly, which also distinguishes "needs more line capacity" from
  "no thermal slack can fix this" and warm-starts the QP; `trust-constr` then
  solves the stated objective on analytic derivatives. Ghana: **0.5 s**.
- **Eq. (5)** is non-convex, so it goes through PyPOWER's own AC OPF. PyPOWER can
  optimise a generator's Q but not a bus susceptance, so each candidate bus gets
  a zero-cost reactive source; the sources are then repeatedly offered a fraction
  of what they last drew, keeping the last schedule that still solved, and what
  survives is converted to `Bs = Q / V^2`. Ghana: **12-34 s**.
- Two deviations, both deliberate. (i) Section VI-B states the objective as "to
  minimize the number of individual shunt devices" and the quadratic penalty as
  the mechanism; the bound-squeezing above targets the stated objective directly,
  and on a case the generators can already support it installs **no shunt at
  all**. (ii) Equation (4) sizes its slack on DC flows, which omit losses and
  reactive flow, so the ratings it leaves behind can still be thermally
  infeasible for (5); `_relieve_thermal` applies the same idea of (4e) to `|S|`
  and reports the extra separately. Without it neither Ghana nor Kenya reaches an
  AC-feasible point.
- PIPS reports failure whenever it cannot reach its feasibility tolerance, even
  after the objective, gradient and complementarity conditions have all
  converged, and the floor it can reach is case-dependent. Demanding
  `PDIPM_FEASTOL=1e-8` therefore turned solved cases into failures; the solver now
  runs near its own default and `check_solution` is the verdict.

## Cross-check against the published Ghana case

`GhanaGrid.m` from `Haddoww/ORS-Ghana-Power` (84 buses, 160 branches, 30
generators, 2648 MW) is the reference for the companion Ghana paper. It was
re-examined with PowerModels.jl 0.21.5 and Ipopt 1.14.1, and the convex
relaxations matter here: an infeasible SOC or QC relaxation is a *certificate*
that no AC solution exists, where a bare Ipopt failure is only a failure to find
one.

**It cannot hold its buses inside [0.9, 1.1] pu.** The AC, SOC, QC and DC
formulations are all infeasible at its own limits. Lifting the thermal ratings
does not help, nor does lifting the branch angle limits, nor lifting the
generators' reactive limits. Widening the voltage band to [0.5, 1.5] solves it
immediately, at 12 949 $/h - so voltage is the binding constraint and nothing
else is.

That traces to its line charging, which is effectively zero: the implied surge
impedance is ~17 000 ohm, i.e. `b` was never updated from the Hawaii template
the file says it derives from ("Case File Saved from Hawaii Case Example",
"every section is unedited unless otherwise specified"). With no line charging,
no bus shunt anywhere (`Bs = 0` at all 84 buses) and long 161 kV lines, the
voltages sag below 0.9 and no amount of generator reactive output holds them up.

**This is what Sec. VI-B exists for.** A case in that state is exactly the case
equation (5) is meant to repair by placing shunts, so the finding is consistent
with the manuscript's own framing rather than a criticism of the Ghana work.

Separately, five of its 30 generators carry `status = 0` while also carrying a
non-zero `Pg` and `Qg`, which MATPOWER would have zeroed. Honouring that column
makes even the DC problem unrepairable - no dispatch exists for any amount of
thermal slack, because the +/-30 degree angle limits then cannot be met. Read
with all 30 units in service the DC problem needs only 2.07 MVA of extra
capacity at a worst angle of 18.8 degrees. So the status column looks stale too,
but it is a second, smaller issue: the voltage band is infeasible either way.

## What is still required for numerical reproduction

**No exact manuscript Huber model has been reconstructed**, and a search of this
repository, its full git history, the earlier `Rich-Machine/automate_grid_creation`
repository, and the surrounding workspace found no fitted Huber artifact. Table II
publishes six rounded slopes but omits the fitted intercept alpha_0.

The bundled `model_bundle_v5.pkl` is the closest surviving artifact and was clearly
produced by this training pipeline: it carries the six Table I features in the
manuscript's exact order, and its `training_n_raw` of 433 is precisely the 118 NZ
plus 315 GB substations of Sec. III.C.1. It is nonetheless **not** the Table II
model. It is an `NNRidge` with alpha = 1.0, not a Huber with epsilon = 1.1; it
aggregated to 103 clusters rather than the 168 shown in Fig. 3; and two of its six
coefficients collapsed to zero:

| Feature | `model_bundle_v5` | Table II |
|---|---|---|
| `pop_density` | 0.556 | 0.510 |
| `road_density` | 0.000 | 0.342 |
| `gem_steel_mw_voronoi` | 0.030 | 0.075 |
| `mining_ntl_product` | 0.284 | 0.157 |
| `ind_heavy_chemical_km2` | 0.332 | 0.372 |
| `fp_gold_ntl` | 0.000 | 0.253 |
| *intercept* | *1.8505* | *not published* |

Its 1.8505 intercept therefore cannot stand in for alpha_0.

**A refit is blocked by one specific missing input.** The bundled
`nz_transmission_features.csv` (118 rows) and `gb_transmission_features.csv`
(315 rows) have the correct shape and a real metered `avg_mw` target, but
`ntl_mean_inside_industrial` is **identically zero in all 433 rows**. Because the
Eq. (2) guard is `max(0, NTL - 1)`, a zero column zeroes the guard everywhere,
which in turn collapses `mining_ntl_product`, `fp_gold_ntl`,
`ind_heavy_chemical_km2` and `gem_steel_mw_voronoi` to zero. Only two of the six
predictors survive, so no refit on the bundled tables can approach Table II.

Closing the gap needs exactly one thing: VIIRS annual composite rasters covering
New Zealand and Great Britain (for example the Earth Observation Group VNL V2
product), measured over the industrial footprints by
`paper_features.industrial_features` to populate that column. The rest of the
refit path is already implemented. New refits retain input SHA-256 hashes and are
explicitly labeled `new_refit_not_manuscript_artifact`; they are not Table II.

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

## Reference [48] and the fuel-cost basis

Reference [48] is S. Babaeinejadsarookolaee, "Synthetic Power System, Generator
Cost Curves" (2018). It resolves three questions the manuscript leaves open.

**Table IV's US column is [48]'s fuel price table verbatim** - coal 1.78, natural
gas 2.68, residual oil 9.25, wood 3.72, petroleum coke 2.56, landfill gas 6.18.

**The global rescaling is exact, not an assumption.** [48] computes $/h at 40, 60,
80 and 100 % of capacity as `heat_rate(P) x P x fuel_price`, then solves

    [a b c]^T = (A^T A)^-1 A^T Y

where `A` depends only on `P` and `Y` is the $/h vector. `Y` is linear in fuel
price and `A` is not, so `(a, b, c)` scales exactly linearly with fuel price.
Replacing the US price with the global one therefore multiplies all three
coefficients by that fuel's ratio. Verified against [48]'s own LMS100 sample:
26634 kW x 12689 BTU/kWh x $2.68/MBTU = $905.7297/h, its stated value to four
decimals. This also settles that `a` carries **no** fixed O&M component - the
whole curve is fuel.

**Table V is [48]'s cluster ranges linearised two-point**, as Sec. V.D states.
For steam coal, [48] gives capacity 20-1425 MW and B 12-16 $/MWh, which yields
`b = 0.002847p + 11.943` against the published `0.00284p + 11.94`.

Two categories are not plain heat-rate curves, and `SCALING_BASIS` in
`create_cost_curves.py` records this:

| Basis | Categories | Consequence |
|---|---|---|
| `heat_rate` | steam, gas turbine, combined cycle | Scaling is exact. |
| `market_bid` | internal combustion | [48] had no heat-rate data and used ERCOT generator bid reports, which include margin and O&M. Scaling is directionally right but not algebraically implied. Left enabled because these are fuel-dominated gensets. |
| `borrowed` | hydro | [48] states hydro's curves were copied from the CCGT samples. Table V's hydro `b = 14 $/MWh` is a **gas heat rate priced at $2.68/MBTU**, not a water or O&M value. It is not rescaled, since that would turn a placeholder into a fuel-price adjustment for a generator that burns nothing. The borrowed curve is a limitation of the source data under either basis. |
| `zero` | nuclear, geothermal, MSW, biomass, wind, solar, battery | Non-dispatchable in [48]; a, b, c all zero. |

## Verification against Table VII

The checked-in cases under `data/working_files/` were compared with the model
sizes reported in Table VII:

| Country | Buses (built / paper) | Generators | Branches |
|---|---|---|---|
| Kenya | 92 / **92** | 75 / **75** | 143 / **143** |
| Puerto Rico | 170 / 172 | 45 / **45** | 352 / 337 |
| Uruguay | 155 / 160 | 35 / **35** | 232 / 201 |
| Taiwan | 287 / n/a | 447 / n/a | 598 / n/a | 

**Kenya's counts reproduce Table VII exactly**, and its 143 branches decompose as
134 resolved lines plus the 9 intra-substation transformers in
`transformer_pairs_kenya.csv`. That is why `INFER_TRANSFORMERS_FROM_VOLTAGE` must
stay at the manuscript default of `True`: with it disabled a rebuild yields 134
branches. `tests/test_paper_methodology.py` pins both numbers.

**Matching counts do not mean the case is sound.** Every checked-in case carries
post-review defects, reported by `create_new_model/check_topology.py`:

| Region | Self-loops | Voltage-incompatible endpoints | Busiest bus degree (next) |
|---|---|---|---|
| Kenya | 11 | 74 / 268 | bus 1: 55 (19) |
| Puerto Rico | 0 | 99 / 696 | bus 140: 26 (22) |
| Uruguay | 11 | 84 / 456 | bus 1: 62 (10) |
| Taiwan | 12 | 175 / 1066 | bus 1: 114 (18) |
| Colombia | 44 | 26 / 786 | bus 2: 205 (19) |
| Australia | 345 | 2 / 4946 | bus 717: 27 (19) |

Kenya, Uruguay, Taiwan and Colombia each have one bus acting as a sink, absorbing
several times the endpoints of any other bus. Kenya's bus 1 (Embakasi, 220 kV)
takes 43 lines of which only 15 are 220 kV, totalling 3685 km and including a
1059 km line. These are not physical.

The automated passes cannot produce this: `find_nearest_bus` returns `None` when
no bus qualifies, passes `exclude_pos` so the to-bus is never the from-bus, and
requires the bus voltage to appear in the line's voltage list whenever both are
known. The defects therefore enter at the manual assignment of Sec. IV.B.3, which
applies none of those rules, and nothing re-checked the tables afterwards.
`clean_bus_tables` now performs the Step 10 self-loop removal after review, and
`check_topology.py` reports all three defect classes for any built region.

Generator counts match for every country with a published value. Puerto Rico and
Uruguay differ in bus and branch counts. Their per-region `config.py` settings
were not recorded — `config.py` is global and overwritten for each region — so
the MIN_TRANSMISSION_VOLTAGE used for those builds is unknown and the difference
cannot be attributed without a rebuild. Taiwan has no published row in the
supplied manuscript draft.

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
- Table IV prints a global $/MBTU column beside the US one, but the prose never
  states how to apply it. Reference [48] settles the mechanics; see "Reference
  [48] and the fuel-cost basis" below. The default `FUEL_COST_BASIS = "us"`
  reproduces Table V exactly; `"global"` is an opt-in departure from the
  published numbers and results using it must say so.
- Sec. IV.B.1 Step 3 retains lines with no voltage tag, but Eq. (3) still needs a
  voltage. The shared engine assigns the regional modal voltage, matching the
  numbered pipeline's "most common regional voltage" rule, and lists every such
  way in the audit. If a region has no tagged line at all, the untagged lines are
  dropped rather than given an invented voltage.
- Eq. (5a)'s quadratic penalty reduces shunt magnitudes; it does not minimize
  device count. Sec. VI-B's prose states the objective *as* device count, so the
  implementation targets that instead of the literal penalty, and says so above.
  Feasibility is checked after solving; no universal convergence guarantee is
  asserted.
- Sec. V-B-2 says a plant whose capacity cannot be found "is likely insignificant
  to the overall grid". A generator with no OSM capacity is therefore written
  **out of service** rather than as an in-service 0 MW unit, and the assembly step
  prints the list so the manual lookup the section asks for can be done. Writing
  them in service both understated the dispatchable capacity and handed the OPF a
  generator pinned at `Pmin == Pmax == Qmin == Qmax == 0`, which stalls PIPS.
- **Sec. III-B has no substation deduplication step, and OSM needs one.** Sec.
  V-B-1 removes duplicate *plants* when they are the same OSM object, when an
  unnamed record sits beside a named one, or when names nearly match. OSM maps
  large substations the same way - Kenya has 32 pairs within 1 km, including
  "Kindaruma" 111 m from "Kindaruma" - and without the same rule each copy
  becomes a bus, the lines attach to whichever copy is nearest, and the other is
  left stranded. `merge_duplicate_substations` applies Sec. V-B-1's rule to
  substations, keeps the richer record and unions the voltage tags, so a site
  mapped once per voltage becomes one compound substation that Sec. III-B then
  splits into tiers. Two differently named substations are never merged.
  Radius: `SUBSTATION_MERGE_RADIUS_M`, default 500 m.
- **A substation's demand belongs on the lowest tier of *that substation*.** The
  rule was applied against the lowest voltage in the whole country, which put
  every megawatt of Kenyan demand on 132 kV and left its forty 220 kV
  substations with none - so all the power had to reach load through the
  transformers and the 132 kV lines. `allocate_loads` now groups by the OSM
  substation ID.
- **Sec. IV.B.3's review can now supply a line OSM never mapped.** The review
  could only assign the endpoints of lines already in the extract, so where OSM
  omits a corridor the model stayed in pieces with no way for the reviewer to
  say otherwise. With `EXPECT_CONNECTED_GRID` set, the review proposes the
  shortest voltage-compatible hop across each gap and records the answer in the
  journal like any other decision. A genuinely islanded region turns the flag
  off.
- **`BRANCH_ANGLE_LIMIT_DEG` (default 60) is a modelling default, not data.**
  Eq. (4) does not constrain angle differences and (5j) does not state a value.
  A case that needs more than the limit is reporting that a corridor is close to
  its steady-state stability limit, which is worth reading as a finding about
  the network rather than something to tune away.
- **`COUNTRY_CAPACITY_MW` is an assumption, and an optional one.** Sec. V-B-2
  sends plants OSM tags no capacity for to a manual lookup. Until that is done
  they carry no capacity, and where OSM coverage is thin the case ends up with
  less generation than demand, which no relaxation can repair - Kenya came out
  at a margin of 1.02 and had no DC dispatch at all. Setting this distributes
  the shortfall over the untagged plants, weighted by the median capacity of the
  tagged plants on the same fuel, in the same shape as Sec. III-C-6's national
  demand scaling. Every sized plant is printed at build time and flagged with
  `capacity_assumed` in the generator CSV. The default is 0, which leaves them
  out of service, so nothing is assumed unless it is asked for. Ghana needs no
  assumption at all (OSM tags 31 of 39 plants, margin 2.53).
- Eq. (4) omits branch angle-difference limits, but (5j) imposes them and
  `check_solution` enforces them, so a DC relaxation that ignored them could not
  be exported. They are included in (4) as well, and a case that cannot satisfy
  them is reported as unrepairable rather than silently relaxed.

## Validation scope

Tests cover the published line/cost coefficients, the shunt objective (no shunt
installed when the generators can cover the reactive demand, one installed when
they cannot), the relaxations leaving an already-feasible standard case almost
untouched (`case30`), uncapacitated generators being written out of service,
polygon guards (including dark and bright facilities within one service area),
missing-input rejection, Huber fitting/prediction, national scaling, cached name
inference, and the previous electrical/table integrity regressions. The local
CSV-to-AC-case and PYPOWER comparisons remain in the suite.

These checks establish implementation behavior. They do not establish geographic
accuracy, recover the manuscript's reported regional results, or validate a Texas
case. Existing generated cases need rebuilding after the input/calibration gaps
are resolved.
