# Rebuild status and open threads

Working notes from the session that realigned the pipeline with the manuscript
and got it producing cases again. `PAPER_ALIGNMENT.md` holds the durable
methodology findings; this file holds what is *in progress* and what to do next.

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
the manuscript's steps. Step 3 was answered by `auto_review.py assisted`, so
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

## Open threads

Threads 1, 3 and 6 from the earlier pass are closed: Puerto Rico and Uruguay now
have assisted builds that solve, fragmentation is fixed (defects 17-19), and
Puerto Rico's voltage mismatches went with the `create_lines.py` unit error.

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
3. **Paper mode is still blocked on data, not code.** It needs
   `data/bus_data/raw/viirs_{region}.tif` and `steel_facilities_mw.geojson`;
   neither is present, so all four cases above were built with
   `LOAD_MODEL_METHOD = "legacy_ridge"`. The gold Fineprint archive *is*
   present. Legacy Ridge runs fully offline for Kenya, Puerto Rico, Uruguay and
   Ghana. **Nothing above demonstrates the manuscript's Eq. (1) load model**,
   only that the rest of the pipeline works around it.
4. **Eq. (3c)'s susceptance exponent is dimensionally wrong** and needs R, X and
   B refitted together per voltage class. Left as published; see
   `PAPER_ALIGNMENT.md`. It did not stop any of the four cases solving, but it
   is why Kenya's eq. (5) installs a shunt at nearly every bus.
5. **The Huber intercept matters less than it appears.** After national scaling,
   `alpha_0` above about 1 changes scaled demands by under 4 % and asymptotes;
   only negative values swing them (340 % at -4). The real blocker is the
   missing *features*, not `alpha_0`. See `PAPER_ALIGNMENT.md`.
6. **Taiwan builds and connects but equation (5) fails on it.** A full build
   gives **392 buses, 807 branches (87 transformers), 443 generators and one
   island**, and DC feasibility passes, but the AC step ends with
   `AC feasibility relaxation produced shunts the OPF cannot use` - the shunt
   schedule it settles on is one the source-free OPF will not re-solve. Taiwan
   is the largest case tried (2.8x Ghana's branch count) and the first where
   372 of 443 generators carry no OSM capacity, so `COUNTRY_CAPACITY_MW` is
   almost certainly needed as well. Worth chasing: it is the first failure of
   the repaired equation (5), so it says something the four working cases do
   not. Colombia and Australia remain untried.

7. **None of the four cases is a reviewed model.** Step 3 was answered by
   `auto_review.py assisted`, which takes the pipeline's own nearest
   voltage-compatible candidate and accepts every proposed island connection.
   Sec. IV.B.3 puts a human there. The journals are recorded and replayable, so
   a reviewer can start from these rather than from scratch.
8. **Rebuilding cannot reproduce the original reviews.** The checked-in regions
   have no review journal, so their decisions are unrecoverable; the first
   rebuild of each is a fresh review. From now on the journal makes it
   repeatable.

## Caveat on the Table VII regression test

`tests/test_paper_methodology.py::test_kenya_case_reproduces_table_vii` pins the
published 92 / 75 / 143. A correctly rebuilt Kenya will most likely **not** hit
those numbers, because the published case contains the defects above. When that
test fails after a genuine rebuild, that is a finding about the manuscript, not a
regression - decide deliberately whether to update or retire the test.
