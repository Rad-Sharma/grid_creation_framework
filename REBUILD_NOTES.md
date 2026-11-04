# Rebuild status and open threads

Working notes from the sessions that got the pipeline producing cases again:
what is *in progress*, what was fixed, and what to do next.

## Defects fixed in this pass

| # | Defect | Where | Effect before the fix |
|---|---|---|---|
| 1 | `matplotlib` declared but not installed | `requirements.txt` | Step 2 died immediately on every region |
| 2 | `UnicodeEncodeError` on a cp1252 console | `config.py` (widens stdout/stderr) | Step 3 crashed on Windows before the first prompt |
| 3 | Untagged multi-voltage substations never split | `pipeline_integrity.split_multi_voltage_buses` | 132 kV and 220 kV networks shorted together with no transformer, at 13 Kenyan buses |
| 4 | Line left on the wrong tier of a split substation | `pipeline_integrity.retier_endpoints` | Endpoint voltage contradicted its bus voltage |
| 5 | Self-loops survived the manual review | `pipeline_integrity.drop_self_loops` | 11 self-loops each in the checked-in Kenya and Uruguay cases |
| 6 | `KeyError: 'era5_temp_c_gee'` | `build_region_features.py` | Step 4 failed for every region; pandas only suffixes a column on collision |
| 7 | Opaque `Singular matrix C in LSQ subproblem` | `pipeline_integrity.dispatchability_report` | Step 9 failed without saying the case was fragmented |
| 8 | Per-region settings never recorded | `build_provenance.py` | `config.py` is global, so a finished case had no record of the settings that built it |
| 9 | Review sessions unrecorded and unrepeatable | `resolve_unmatched.ReviewLog`, `--replay` | Same OSM input gave a different model per reviewer, with no audit trail |
| 10 | Manual picks bypassed the automated rules | `resolve_unmatched` prompt guards | A pick could create a self-loop or a voltage mismatch that the automated passes would reject |
| 11 | Islands other than the slack's had no reference bus | `case_utils.island_case` | `IndexError: index 0 is out of bounds for axis 0 with size 0` from `pipsopf_solver`; Step 9 could not even attempt those islands |
| 12 | Every solved island rejected as `solver changed fixed inputs` | `constrained_dispatch.solve_island` | pypower read the 10-column `gen` matrix as MATPOWER **version 1** and overwrote `ANGMIN`/`ANGMAX` with +/-360, so the integrity guard fired on every success and the +/-60 deg limits were never enforced |
| 13 | Eq. (4)/(5) relaxation unusable | `power_network.solve_feasibility` | SLSQP with finite-difference Jacobians ran **18 minutes and failed** on the published 84-bus Ghana reference. Now HiGHS + `trust-constr` for (4) and PyPOWER's AC OPF for (5); Ghana solves in under a minute |
| 14 | Solver tolerance rejected converged answers | `power_network`, `constrained_dispatch` | `PDIPM_FEASTOL=1e-8` is below the floor PIPS can reach on these cases, so it reported failure after the objective, gradient and complementarity conditions had all converged |
| 15 | Eq. (5) imposed thermal limits sized on DC flows | `power_network._relieve_thermal` | AC apparent flows exceed DC active flows, so the ratings (4) left behind were still infeasible for (5); neither Ghana nor Kenya could reach an AC-feasible point |
| 16 | Plants with no OSM capacity written in service | `create_synthetic_grid.py` | 30 of Kenya's 75 generators became online 0 MW units, cutting dispatchable capacity to 1.02x demand and pinning `Pmin == Pmax == Qmin == Qmax == 0`, which stalls PIPS. Now offline, with the list printed for the Sec. V-B-2 lookup |
| 17 | Bus voltages compared in the wrong unit | `create_lines.py` | `base_kV` is stored in **Volts**, and the automatic matcher multiplied it by 1000 again, so no voltage-tagged bus could ever match a voltage-tagged line. Every named substation was starved of lines and the network fell apart. Kenya's unmatched endpoints: 87 -> 34 |
| 18 | Substations never deduplicated | `pipeline_integrity.merge_duplicate_substations` | OSM maps big sites twice ("Kindaruma" 111 m from "Kindaruma"). Each copy became a bus, the lines took whichever was nearest, and the other was stranded. Sec. V-B-1 states this rule for plants; it now applies to substations too |
| 19 | The review could not supply a missing line | `resolve_unmatched` POST-PROCESSING 3 | Step 3 could only assign endpoints of lines OSM already had. Where OSM has no line at all, the network stayed in pieces with no way for a reviewer to say so |
| 20 | All demand landed on one voltage level | `pipeline_integrity.allocate_loads` | The "load leaves at the lowest tier" rule was applied against the lowest voltage in the **country**, so Kenya's whole 2242 MW sat on 132 kV and the 220 kV substations carried nothing. Now per substation |
| 21 | Bus angles bounded at +/- pi | `power_network._solve_dc` | MATPOWER bounds angle *differences*, not bus angles; a radial chain at 60 deg a branch passes pi in three hops, so equation (4) reported feasible cases as infeasible |
| 22 | `reverse_geocoder` crashed the first run and misnamed every substation | `create_buses.py` | `mode=2` uses multiprocessing; because the step is a script, each worker re-imported `__main__` and re-ran the pipeline, spawning recursively on Windows. The failed lookup is why 87 of Kenya's 146 buses were called "El Tarter Substation N" - a village in Andorra. `mode=1` fixes both |
| 23 | NaturalEarth downloaded on every run, uncached and untimed | `predict_bus_loads._country_boundaries` | A slow network left Step 5 hanging indefinitely on the critical path |
| 24 | Exported cases could not run a power flow | `power_network._solve_ac` | The solved bus voltages were written to `bus.Vm` but each generator's `Vg` was left where it started. A power flow holds a PV bus at `Vg`, not at `Vm`, so `runpf` was handed a different operating point from the one just solved and diverged on a stressed network, even though the case passed `check_solution` and the AC OPF |
| 25 | Cases unparseable outside Python | `create_synthetic_grid.py`, `ensure_feasibility.py` | A real substation name can carry an apostrophe - Kenya has "Mang'u" - and written straight into a single-quoted MATLAB cell it ends the string early. MATLAB, Octave and PowerModels.jl all refused the case; only our own parser worked, because it ignores the name blocks. Quotes are now doubled MATLAB-style |
| 26 | One Overpass query per unnamed plant | `plant_names.region_named_power_objects` | Sec. V-B-1's name inference asked Overpass once per plant against a rate-limited endpoint. Taiwan has 337 unnamed plants at ~15 s each, heading for ~84 minutes of nothing but waiting. One bbox query returns the same 893 candidates and the ranking is local; the per-plant path stays as a fallback and now gives up after 5 consecutive failures instead of proving the point 300 more times |
| 27 | Bus renames left endpoint names stale | `pipeline_integrity.refresh_endpoint_names` | `resolve_unmatched` appends a `_{kV}` suffix to buses whose voltage it infers but never refreshed `fbus_name`/`tbus_name`, so Step 8 rejected the case with `Stale fbus name/ID mapping`. Killed Taiwan's assembly at 87 of 711 endpoints; the other four regions renamed too few buses with named endpoints to trip it |
| 28 | Dense network matrices | `power_network.admittance_matrices`, `dc_matrices` | Ybus, Yf, Yt and the DC incidence were dense, so Australia's 2401 buses meant 0.28 GB of matrices and an O(n^2 m) matmul inside equation (4), for two non-zeros per branch row. All sparse now |
| 29 | Satellite features keyed by bus position | `build_region_features.gee_cache_mismatch` | `gee_features_{region}_matpower.csv` joins on `BUS_{bus_i}`, but every column in it is a reduction over that bus's Voronoi cell. Rebuild a region and bus 7 is a different substation: Taiwan went from 287 buses to 392 and **only 5 of 392 cached rows still described the bus they landed on**, so the rest of the network was handed another site's nighttime lights and built-up area, silently. The cache is now checked against the current bus set and refused when it does not describe it |
| 30 | Coastal substations booked as cross-border | `predict_bus_loads.detect_cross_border_buses` | A bus outside the NaturalEarth **110 m** country polygon was declared an international exchange point and struck off the domestic total. That coastline is simplified by kilometres, so the test caught **31 of Taiwan's 287 buses, 49 of Australia's 2050 and 7 of Puerto Rico's 170 - countries with no land border at all** - along with 15 of Uruguay's and 5 of Kenya's. Their demand was redistributed over the remaining buses. A bus is now external only when another country's polygon is strictly closer than its own, which leaves Kenya 3, Uruguay 7, Ghana 2, Colombia 1 and the three island systems none |
| 31 | Feature sources failing to zero, in silence | `build_region_features` | A failed Overpass call left `road_density` or the industrial areas at zero for every bus with nothing said, and `get_land_polygon` re-downloaded NaturalEarth on every run and fell back to the bounding rectangle when the network was down, which makes each coastal Voronoi cell reach out to sea. Each of these is a different load model, not a slower run. The committed boundary file is now used first, and any source that could not be read is named at the end of the step with the predictor it zeroes |
| 32 | Equation (5) rejected its own feasible point | `power_network._solve_ac` | Two faults. The converted shunts **overwrote** any shunt the case already carried, though (5) defines `B_sh` as the susceptance installed *in addition to* what is there. And a shunt that reproduces exactly what the reactive source was drawing leaves voltages, angles and dispatch untouched, so the source-free case is feasible by construction; the step nonetheless raised `AC feasibility relaxation produced shunts the OPF cannot use` whenever PIPS declined to re-converge from that point, which sits on several active constraints. Re-solving is now about optimality, and the verified point is kept when it fails |
| 33 | Table VI settings unverified | `check_inputs.check_region_settings` | `SHAPE_FILE_NAME`, `COUNTRY_ISO3` and `REGION_BBOX` are three of the seven inputs a new region needs and each failed quietly: an unmatched name disables cross-border detection, an unmatched ISO code clips every Voronoi cell to the bounding rectangle, and a box that misses the country clips the network away. All three are now checked against the committed boundary file before the run starts |
| 34 | Lines shorter than the gap between their own buses | `resolve_unmatched.floor_lengths_at_bus_span` | Equation (3) scales R, X and B by length, and that length was the OSM geometry which survived preprocessing. Once an endpoint is assigned to a substation the geometry does not reach - the automated passes allow it and the review offers candidates up to 80 km away - the two stop describing the same conductor. **Every region was affected: 120 of Ghana's 219 lines, 226 of Puerto Rico's 383, 382 of Taiwan's 720 and 560 of Colombia's 979 were given the impedance of a conductor shorter than the straight line between the substations they join.** The worst sat at 1/2500 of its floor: a 20 m impedance across a 50 km span, a near short circuit between two distant buses. The resulting 1e-5 pu reactances left Colombia ill-conditioned enough that PIPS and Ipopt both refused even the DC problem it had already been shown to satisfy. A conductor cannot be shorter than the distance between its endpoints, so that distance is now its floor, and R, X and B scale with it |

## Checked-in cases are defective

Every case under `data/working_files/{region}/` fails
`python create_new_model/check_topology.py --region {region}`:

| Region | Self-loops | Voltage-incompatible | Busiest bus (next) |
|---|---|---|---|
| Kenya | 11 | 74 / 268 | **55** (19) |
| Puerto Rico | 0 | 99 / 696 | 26 (22) |
| Uruguay | 11 | 84 / 456 | **62** (10) |
| Taiwan | 12 | 175 / 1066 | **114** (18) |
| Colombia | 44 | 26 / 786 | **205** (19) |
| Australia | 345 | 2 / 4946 | 27 (19) |

Kenya, Uruguay, Taiwan and Colombia each have one bus acting as a sink. Kenya's
bus 1 (Embakasi, 220 kV) takes 43 lines of which only 15 are 220 kV, totalling
3685 km including a 1059 km line. **Kenya's counts match Table VII exactly while
containing these defects**, so a count match is not evidence of correctness.

The automated passes cannot produce any of this: `find_nearest_bus` returns
`None` when nothing qualifies, passes `exclude_pos` so the to-bus is never the
from-bus, and requires voltage compatibility when both values are known. The
defects enter at the Sec. IV.B.3 manual review, which had none of those rules.

## Rebuilt cases

`data/working_files/_rebuilt/{region}/` holds cases rebuilt from OSM with the
current pipeline. They are **not reviewed models** - Step 3 was answered
automatically; each folder's `REBUILD_INFO.json` records the policy used. They
are kept separate from the checked-in cases, which are untouched.

Two automatic review policies exist in the session scratchpad's `auto_review.py`:

- `skip` - every endpoint left unmatched, its line dropped. Invents nothing, but
  fragments the network.
- `assisted` - takes the nearest candidate the pipeline itself offers, the same
  voltage-compatible within-radius list the CLI prints. Reproduces the easy
  decisions a reviewer would make; still needs checking before anyone calls the
  result reviewed.

Rebuild quality against the checked-in equivalents:

| Region | Buses | Lines | Xfmr | Gen | Self-loops | Voltage-incompatible | Sink |
|---|---|---|---|---|---|---|---|
| Kenya (skip) | 107 | 83 | 23 | 75 | 0 | 0 / 166 | none |
| Kenya (assisted) | 146 | 138 | 29 | 75 | 0 | 0 / 276 | none |
| Puerto Rico (skip) | 211 | 347 | 41 | 45 | 0 | 5 / 702 | none |
| Uruguay (skip) | 158 | 163 | 18 | 35 | 0 | 0 / 326 | none |

Generator counts match Table VII exactly for all three (75 / 45 / 35).

## Four cases that run

All four were built from the OSM extracts already in the repository, using only
the Automate Grid Creation paper's steps. Step 3 was answered by `auto_review.py assisted`, so
none is a *reviewed* model.

| | Ghana | Uruguay | Puerto Rico | Kenya |
|---|---|---|---|---|
| Buses / branches / generators | 94 / 236 / 39 | 187 / 281 / 35 | 210 / 422 / 45 | 138 / 170 / 75 |
| Islands | **1** | **1** | **1** | **1** |
| Demand / capacity (margin) | 2100 / 5323 (2.53) | 1350 / 1828 (1.35) | 2300 / 5949 (2.59) | 2242 / 3200 (1.43) |
| Eq. (4) DC feasible | yes | yes | yes | yes |
| Eq. (5) AC feasible | yes | yes | yes | yes |
| **AC power flow** | **converges** | **converges** | **converges** | **converges** |
| Vmin / Vmax | 0.912 / 1.073 | 0.900 / 1.067 | 0.945 / 1.100 | 1.034 / 1.100 |
| Losses | 36.2 MW (1.7 %) | 50.6 MW (3.7 %) | 53.5 MW (2.3 %) | 159.3 MW (7.1 %) |
| `constrained_dispatch.py` | **validated** | **validated** | **validated** | **validated** |
| Assumptions beyond defaults | none | none | none | `COUNTRY_CAPACITY_MW` |

Every bus sits inside [0.9, 1.1] pu in all four, the branch angle limit is the
ordinary +/-60 degrees, and nothing in the relaxation was loosened. Saved under
`_rebuilt/ghana/`, `_rebuilt/uruguay_solved/`, `_rebuilt/puerto_rico_solved/`
and `_rebuilt/kenya_solved/`; the plain `_rebuilt/kenya/`, `_rebuilt/uruguay/`
and `_rebuilt/puerto_rico/` folders are the earlier, fragmented builds and are
superseded.

Independently cross-checked with PowerModels.jl 0.21.5 and Ipopt 1.14.1: every
case parses, and the AC, SOC-relaxation, QC-relaxation and DC formulations all
solve. Objectives agree with PyPOWER to five significant figures.

All four predate defect 34, so their lines still carry the impedance of
conductors shorter than the distance between the substations they join: 120 of
Ghana's 219 lines, 37 of Kenya's 149, 226 of Puerto Rico's 383 and 93 of
Uruguay's 255. `uruguay_corrected/` is the only one of the four rebuilt since.
The others still solve and still validate; they are simply less stiff than the
same network with correct lengths, so treat their transfer capability as
optimistic until they are rebuilt.

### The one assumption Kenya needs

Kenya reaches a solution only with `COUNTRY_CAPACITY_MW = 3200`, which sizes the
30 plants OSM tags no capacity for. Without it the case carries 2288 MW against
2242 MW of demand - a margin of 1.02 - and no dispatch exists, because every
plant has to run flat out regardless of where it sits. That is what was driving
the 84.9-degree corridor angles; they were a symptom, not the disease. Ipopt and
the SOC/QC relaxations all confirm the no-assumption case is genuinely
infeasible, so this is not a solver workaround.

Sensitivity, at the default +/-60 degree angle limit:

| Assumed national capacity | Margin | Extra line capacity eq. (4) needs | AC |
|---|---|---|---|
| 2288 MW (OSM only) | 1.02 | no DC dispatch exists | - |
| 2800 MW | 1.25 | +490 MVA | fails |
| 3000 MW | 1.34 | +293 MVA | solves |
| **3200 MW** | **1.43** | **+151 MVA** | **solves** |

3200 MW is Kenya's rough installed capacity; the split across plants is weighted
by the median capacity of the tagged plants on the same fuel, each sized plant
is printed at build time, and `matpower_kenya_generators.csv` flags them with
`capacity_assumed`. Sec. V-B-2's manual lookup remains the accurate route.
Ghana, Uruguay and Puerto Rico need no such assumption: OSM tags enough of their
plants to give margins of 2.53, 1.35 and 2.59 on their own.

## Rebuilt after defects 29-34

Equation (5) was re-run on all four solved cases with the repaired code. It
reproduces every one of them, so nothing here trades a working case for a fix:

| | Ghana | Kenya | Uruguay | Puerto Rico |
|---|---|---|---|---|
| Vmin / Vmax | 0.912 / 1.072 | 1.034 / 1.100 | 0.900 / 1.067 | 0.945 / 1.100 |
| Shunt devices above 1 MVAr | **0** | 108 | 51 | 193 |
| Largest shunt | under 0.1 MVAr | 136.6 MVAr | 16.8 MVAr | 450.9 MVAr |

Those shunt counts are worth keeping in view, because they are not explained by
line charging. Puerto Rico has the *least* charging of the four yet needs by far
the most reactive plant, and Ghana needs none at all. What drives them is the
local reactive balance at each bus, not any global quantity.

Uruguay was then rebuilt from OSM end to end, through all ten steps, to check
the fixes against a region that already worked:

| | uruguay_solved (before) | uruguay_corrected (after) |
|---|---|---|
| Buses / branches / generators | 187 / 281 / 35 | 187 / 281 / 35 |
| Buses denied domestic load as "cross-border" | **15** | **1** |
| Satellite features | a 166-bus cache merged onto 187 buses | cache refused, declared NZ proxy used |
| Lines shorter than their own bus span | **111 of 255** | **0** |
| Smallest reactance | 6.3e-5 pu | 1.41e-4 pu |
| Vmin / Vmax | 0.900 / 1.067 | 0.900 / 1.079 |
| Losses | 50.6 MW (3.7 %) | 60.4 MW (4.5 %) |
| `constrained_dispatch.py` | validated | **validated** |

The topology is identical, as it should be: none of the fixes touches Steps 1-3.
What changed is where the 1350 MW sits and how long the conductors carrying it
are. Losses rise because 934 km of conductor that the model had been pretending
was not there is now there; the old figure was the optimistic one.

Taiwan and Colombia were then built from scratch on the corrected pipeline. Both
come out as one clean connected network from OSM alone, with no assumption beyond
the seven Table VI inputs. Taiwan solves; Colombia does not:

| | Taiwan | Colombia |
|---|---|---|
| Buses / branches / generators | 392 / 807 / 443 | 488 / 1124 / 258 |
| Islands | 1 | 1 |
| Self-loops, voltage-incompatible endpoints | 0, 0 of 1440 | 0, 0 of 1958 |
| Busiest bus (checked-in case) | 16 (114) | 17 (205) |
| Demand / online capacity (margin) | 31 400 / 44 788 MW (1.43) | 9100 / 14 345 MW (1.58) |
| Lines raised to their own bus span | 421 of 720, +1525 km | 610 of 979, +8019 km |
| Eq. (4) DC feasible | yes | yes |
| Eq. (5) AC feasible | **yes** | **no** |
| AC power flow | **converges**, 3.5 % losses | - |
| Assumptions beyond defaults | **none** | **none** |

Both are saved: `_rebuilt/taiwan_solved/` with its three cases, and
`_rebuilt/colombia_dc_only/` with the raw and DC-feasible cases, so the
diagnosis in thread 6 can be picked up without rebuilding.

## Open threads

Threads 1, 3 and 6 from the pass before last are closed: Puerto Rico and Uruguay
have assisted builds that solve, fragmentation is fixed (defects 17-19), and
Puerto Rico's voltage mismatches went with the `create_lines.py` unit error.
Thread 5, Taiwan, is closed by defect 32; Colombia takes its place below.

1. **Kenya's assumed generator capacities should be replaced with real ones.**
   The case solves, but 912 MW of its 3200 MW is distributed by assumption over
   30 plants OSM gives no capacity for. Sec. V-B-2 asks for a manual lookup and
   that is still the right answer; the named ones are the place to start
   (`Marsabit`, `Standard Rolling Mills`, and the bracketed inferred names in
   `matpower_kenya_generators.csv` where `capacity_assumed` is true). Ghana,
   Uruguay and Puerto Rico need no assumption at all.
2. **Kenya's Nairobi corridor is under-built.** OSM maps one way per corridor
   where the 132 kV ring has several circuits, so eq. (4) adds 151 MVA of line
   capacity and the case runs at 7.1 % losses against 1.7-3.7 % for the other
   three. Correcting the circuit count would remove most of both.
3. **Paper mode is still blocked on data, not code, and the missing datum is
   now pinned down.** It needs `data/bus_data/raw/viirs_{region}.tif` and
   `steel_facilities_mw.geojson`; neither is present, so every case here was
   built with `LOAD_MODEL_METHOD = "legacy_ridge"`. The gold Fineprint archive
   *is* present and usable: `fineprint_mine_production.zip` yields 287 gold
   mines with unique IDs, offline. **Nothing here demonstrates the Automate Grid Creation
   paper's Eq. (1) load model**, only that the rest of the pipeline works around it.

   What a fresh audit of the cached data established, so nobody has to repeat it:

   - The **training targets are complete**. `nz_transmission_features.csv` holds
     118 rows and `gb_transmission_features.csv` 315, matching the paper's 118
     NZ Grid Exit Points and 315 GB Grid Supply Points, each with a measured
     `avg_mw` on every row (3986 MW and 22 806 MW in total).
   - **Four of the six predictors are present**, but two of those are nearly
     degenerate: `gem_steel_mw_voronoi` is non-zero on 6 of the 433 training
     rows and `fp_gold_ntl`'s source count on 3.
   - **The Eq. (2) guard has no input at any granularity.**
     `ntl_mean_inside_industrial` is the literal `0.0` assigned at
     `build_region_features.py:417`, on all 3420 rows across all nine feature
     tables, while roughly 2790 of them carry non-zero industrial area. Worse,
     **every nighttime-light column in the GB table is identically zero**, so 73 %
     of the training set has no radiance of any kind.
   - Consequently `mining_ntl_product` and `fp_gold_ntl` are **identically zero**
     over all 433 training rows, the pooled design matrix has **rank 5 where 7 is
     needed**, and `train_paper_load_model.py` stops on its own identifiability
     guard. Two of the six published coefficients are simply not estimable here.
   - Forcing a fit anyway, NZ-only with `ntl_max` standing in for the guard,
     reaches rank 7 and converges but returns **the wrong sign on three of six
     coefficients** (road density, steel and gold each reducing demand). That is
     a fit to noise, and it is the evidence that the gap is not cosmetic.
   - The `gee_features_{region}_matpower.csv` caches *do* hold genuine VIIRS
     (NOAA VIIRS DNB monthly, 2020 annual mean, 500 m), but reduced over Voronoi
     **cells**, not over industrial polygons, so they cannot satisfy Eq. (2) and
     paper mode correctly declines them.
   - `gem_gist.xlsx` gives steel capacity in **ttpa, not electrical MW**, and 142
     of its 990 operating plants have no tonnage at all - including the single
     New Zealand plant and the two largest British works. So the steel feature
     cannot be rebuilt from the bundled workbook even with an assumed intensity.

   The one blocker that matters is therefore **measured VIIRS radiance within
   industrial polygons, for NZ and GB as well as for each target region**.
   Everything else on the paper path is present or derivable.
4. **The Huber intercept matters less than it appears.** After national scaling,
   `alpha_0` above about 1 changes scaled demands by under 4 % and asymptotes;
   only negative values swing them (340 % at -4). The real blocker is the
   missing *features*, not `alpha_0`.
5. **Taiwan is closed. It was a false failure, and the case now runs.** Saved
   as `_rebuilt/taiwan_solved/`. The Automate Grid Creation paper leaves
   Taiwan's row of Table VII blank, so this is the first Taiwan case from these
   methods that anyone can run.

   | | Taiwan |
   |---|---|
   | Buses / branches / generators | 392 / 807 (87 transformers) / 443 |
   | Islands | **1** |
   | Self-loops, voltage-incompatible endpoints | **0**, **0 of 1440** |
   | Busiest bus (checked-in case: 114) | 16 |
   | Demand / online capacity (margin) | 31 400 / 44 788 MW (1.43) |
   | Eq. (4) DC feasible, eq. (5) AC feasible | yes, yes |
   | **AC power flow** | **converges** |
   | Vmin / Vmax | 1.076 / 1.100 |
   | Losses | 1110.6 MW (3.5 %) |
   | Max branch angle difference | 32.9 deg of the 60 allowed |
   | Shunts above 1 MVAr | 370 (296 capacitive, 74 reactive), largest 334.5 MVAr |
   | Assumptions beyond defaults | **none** |

   `AC feasibility relaxation produced shunts the OPF cannot use` was defect 32:
   the converted schedule reproduces exactly what the reactive source was
   drawing, so the source-free case is feasible at that point by construction,
   and PIPS declining to re-converge from it proved nothing. With that repaired
   the AC step takes its new fallback and the case solves. `COUNTRY_CAPACITY_MW`
   turned out not to be needed either: the 71 plants OSM tags carry 44 788 MW,
   and the 372 untagged ones stay out of service as Sec. V-B-2 intends.

   **One caveat.** `constrained_dispatch.py` does not validate this case, because
   PIPS will not re-converge on it at any tolerance in its ladder. That is the
   same solver difficulty, not a statement about the case: `runpf` converges,
   `check_solution` passes against the exact nonlinear model, and PowerModels.jl
   with Ipopt solves all four formulations of the exported case - DC 7.584e6,
   SOC 8.741e6, QC 8.741e6, AC 8.762e6, in the order a correct relaxation
   hierarchy requires. A case PyPOWER cannot re-solve is still worth having, but
   say so when reporting it.

6. **Colombia builds cleanly, and its AC infeasibility is now pinned to one
   constraint.** 488 buses, 1124 branches (145 transformers), 258 generators,
   one island, and `check_topology.py` reports it clean where the checked-in
   case has 44 self-loops and a bus of degree 205. Equation (4) needs only
   283 MVA of extra capacity on 5 branches. Equation (5) still ends with
   `found no solution even with a reactive source at every bus`.

   Relaxing one limit at a time, with equation (5)'s own reactive source of
   +/-405 MVAr at every one of the 488 buses already in place:

   | Relaxation | Result |
   |---|---|
   | none | fails |
   | thermal limits lifted | fails |
   | angle limits lifted | fails |
   | thermal and angle limits lifted | fails |
   | thermal lifted, voltages free to [0.5, 1.5] | **solves** at Vmin 1.461, Vmax 1.500 |
   | thermal and angle lifted, voltages [0.5, 1.5] | **solves**, unchanged |

   So the voltage band is the only binding constraint, and the solutions that
   exist sit at 1.46 to 1.50 pu with the sources absorbing 10 066 MVAr. The
   cause is a reactive surplus: Colombia's lines produce **5356 MVAr of
   charging against 1409 MVAr of generator absorption**, the widest gap of any
   region tried (Uruguay's charging is 1343 MVAr on 1350 MW of demand and it
   solves; Puerto Rico's is 309 MVAr).

   Two things were tried and did not fix it. Defect 34 was found here and
   repaired a large part of the case - 560 of 979 lines had an impedance below
   their physical floor, and after the fix Ipopt through PowerModels solves the
   DC problem it had previously called infeasible - but the AC problem stayed
   infeasible. A continuation that approaches the true voltage band from a
   wider one was written and removed again: it reaches the wide-band solution
   and then rides the 1.5 pu ceiling down, because nothing in the first solve
   penalises the shunt.

   That points at the likeliest real fix. Equation (5a)'s `M * sum(B_sh^2)`
   term is currently imposed only by the iterative squeeze *after* a first
   point is found, never as an objective term, so the first solve has no reason
   to prefer a small shunt and lands on the high-voltage branch. Putting the
   penalty in the objective needs reactive-power costs, and PyPOWER's support
   for them is broken: `opf_costfcn.py:145` slices `gencost[ng+1:2*ng]` where
   it should slice `gencost[ng:2*ng]`, so the Hessian is misaligned by a row.
   Worth checking first, though, is whether Colombia's 115 kV
   `MIN_TRANSMISSION_VOLTAGE` is right: it pulls a great deal of
   sub-transmission into the model, and that is where the charging comes from.

7. **Australia does not finish Step 2, and that is a scaling limit, not a
   fault.** Step 1 completes and gives 1698+ substations. `create_lines.py`
   then ran for **63 minutes without writing its first output** and was stopped.
   For comparison the same step takes about 10 minutes on Taiwan (392 buses,
   720 lines) and about 25 on Colombia (488 buses, 979 lines), so the cost is
   clearly superlinear in the bus count and Australia at 2401 buses is hours
   away, not minutes. Nothing suggests it would fail, only that it would take
   far longer than anyone would wait. The endpoint-assignment passes of
   Sec. IV.B.2 compare every unmatched endpoint against every bus; indexing
   the buses spatially is the obvious fix and would help every large region.
   Australia is also the one region here with no WorldPop raster cached, so
   Step 4 would need the network.

8. **No case here is a reviewed model.** Step 3 was answered by
   `auto_review.py assisted`, which takes the pipeline's own nearest
   voltage-compatible candidate and accepts every proposed island connection.
   Sec. IV.B.3 puts a human there. The journals are recorded and replayable, so
   a reviewer can start from these rather than from scratch.
9. **Rebuilding cannot reproduce the original reviews.** The checked-in regions
   have no review journal, so their decisions are unrecoverable; the first
   rebuild of each is a fresh review. From now on the journal makes it
   repeatable.

## Caveat on the Table VII regression test

`tests/test_paper_methodology.py::test_kenya_case_reproduces_table_vii` pins the
published 92 / 75 / 143. A correctly rebuilt Kenya will most likely **not** hit
those numbers, because the published case contains the defects above. When that
test fails after a genuine rebuild, that is a finding about the Automate Grid Creation paper, not a
regression - decide deliberately whether to update or retire the test.
