# Building and checking better models

The current default favors explicit missing information over fabricated connectivity.
It cannot supply verified transformer inventories, industrial loads, switching
states, or hourly renewable profiles that are absent from the input data.

## Shared topology and equipment evidence

The shared engine in `create_new_model/grid_topology.py` works for a region or country through:

```powershell
.\.venv\Scripts\python.exe generate_region_topology.py --osm REGION.json --boundary REGION.geojson --metric-crs EPSG:32630 --evidence equipment.json --output data/region_topology
```

Choose a projected CRS appropriate to your region; EPSG:32630 above is only an
example. The input OSM JSON must preserve node identities (`out body geom`).
These outputs are topology tables, not a complete operating case.

The station model collects all terminals before merging. Stations near a
normally-open connection retain separate terminal sections; conflicting known
interconnections remain separate. Uncontested ordinary stations still use a
synthetic common-bus assumption, explicitly reported as such. This is not a
complete breaker-level reconstruction. Internal line geometry can still provide
connections; an omitted switch is not proof that no alternate path exists.

Multi-voltage stations no longer receive automatic 1,000-MVA transformers.
Unverified candidates are listed in `omitted_osm_assets.csv`. To retain the old
assumption for a labeled sensitivity experiment, use `--transformer-policy legacy`;
it still cannot bridge conflicting or unresolved station sections.

Supply `--evidence equipment.json` to define reviewed sections and connections:

```json
{
  "stations": {
    "way/123": {
      "source": "your switchyard drawing reference and date",
      "node_sections": {"1001": "A", "1002": "B"}
    }
  },
  "connections": [{
    "id": "your-transformer-id",
    "kind": "transformer",
    "source": "your equipment record reference and date",
    "from": {"station": "way/123", "kv": 345, "section": "A"},
    "to": {"station": "way/123", "kv": 138, "section": "B"},
    "r_pu": 0.001, "x_pu": 0.02, "rate_mva": 250, "closed": true
  }],
  "generator_connections": {
    "12345": {"station": "way/123", "kv": 138, "section": "B",
              "source": "your generator interconnection record"}
  }
}
```

All identifiers and electrical numbers above are illustrative, not measured grid data.
Connection impedances use a 100-MVA base; explicit switch connections also require
impedance and rating. Missing or ambiguous endpoints fail rather than silently
selecting a bus. Omitted `connections` means no reviewed connections supplied.

## Generator matching

`generator_matching.py` screens candidates within 5 km for voltage and known
interconnection compatibility. Containment in an OSM plant polygon receives
preference. A score margin below 0.25 km-equivalent is ambiguous and withheld.
Distance and voltage alone do not verify a connection. Accepted candidates retain
their confidence reason and alternatives; rejected candidates are logged.
Reviewed overrides resolve known connections without guessing transformers.
The radius and score margin are heuristics, not measured accuracy guarantees.

## Sector allocation and explicit operating intervals

The reusable `operating_scenario.allocate_sectors` function accepts county/sector totals and bus-specific weights.
Totals require `GEOID,sector,mw,source`; weights require `bus_i,sector,weight,source`.
Supported sectors are residential, commercial, industrial, and other. Supply
weights for the selected topology's bus IDs. Positive totals with no valid
weights fail; county/sector totals are preserved. Zero/absent supplied sector
totals mean no assigned demand for that sector. Population-based scenarios must remain explicitly labeled exploratory.

`operating_scenario.py` applies one interval of supplied sector demand, generator
availability, storage schedules, and interchange to any MATPOWER base case:

```powershell
.\.venv\Scripts\python.exe create_new_model/operating_scenario.py --case BASE.m --scenario hour.json --output data/hour_case
```

The JSON requires `source_case_sha256` matching the exact `.m` file, `source`, a
timezone-aware `timestamp`, positive `duration_hours`, `replace_all_loads: true`,
`sector_loads`, and `generator_availability`. Each sector load contains `bus`,
`sector`, `mw`, and `power_factor`. Unspecified bus loads become zero. Availability
records contain `row_1based` and `fraction` (0–1); every active generator must have
a record. This prevents an omitted renewable profile from silently implying full
availability. Real hourly inputs must be obtained separately.

Optional `storage` records require `bus,mw,initial_mwh,capacity_mwh,charge_limit_mw,
discharge_limit_mw,charge_efficiency,discharge_efficiency,qmin_mvar,qmax_mvar,source`.
Positive MW discharges; negative MW charges. Power, efficiency, and end-of-interval
energy bounds are checked. Storage is a supplied fixed schedule, not optimized
multi-hour operation. Optional `interchange` records require
`bus,mw,qmin_mvar,qmax_mvar,source`; positive MW imports. These are scheduled boundary
injections, not validated external-grid equivalents. Avoid double-counting units
already present in the base case.

## Constrained AC dispatch and validation

```powershell
.\.venv\Scripts\python.exe create_new_model/constrained_dispatch.py --case BASE.m --output data/ac_dispatch --timeout 120
```

Output folders must be empty. Each island is solved with PYPOWER's sparse AC OPF,
fixed demand, generator P/Q bounds, bus-specific voltage limits, thermal and angle
limits. No Q expansion, shunt additions, or load-scaling search is performed.
The dispatch workflow does not use load-scaling or expanded reactive limits.

Each successful solution is independently checked using sparse nodal power
balances and both ends of every branch. Only validated islands are exported.
`full_ac.m` is written only when the complete case passes. Timeouts and solver
failures are recorded in `dispatch_report.json`; they do not prove mathematical
infeasibility. `--timeout` applies per island; `--total-timeout` defaults to 600
seconds and explicitly labels remaining islands as untested if exhausted. A large incomplete case may fail
many islands even while the implementation passes its software tests.

## Model-quality reports and compatibility

`model_quality.json` separates unresolved topology, reviewed equipment, withheld
generation, geographic benchmarking, and operating feasibility. Coverage fractions
remain unknown until compared with an independent inventory. Fixed-dispatch diagnostics must be reported separately from AC-OPF results.

The historical numbered country CSV pipeline remains supported for compatibility.
It shares normally-open/mixed-circuit filtering and disables automatic voltage-tier
transformers by default, but has not been rewritten into a breaker-level model.
Use the new shared regional engine for station-section safeguards; do not assume
old CSV workflows inherit them merely because their source data is similar.
