"""Shared table invariants for the CSV pipeline. Internal bus voltages are volts."""
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


TRANSFORMER_COLUMNS = [
    "fbus", "tbus", "r", "x", "b", "rateA", "rateB", "rateC",
    "ratio", "angle", "status", "angmin", "angmax", "fbus_name",
    "tbus_name", "length_km", "voltage_V", "osm_id",
]


def read_transformers(path):
    """Accept legacy empty files as well as header-only transformer tables."""
    if not Path(path).exists():
        return pd.DataFrame(columns=TRANSFORMER_COLUMNS)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=TRANSFORMER_COLUMNS)


def refresh_endpoint_names(buses, *branch_tables):
    """Re-copy bus names onto the branch tables' endpoint name columns.

    The names are display metadata and the IDs are authoritative, so any step
    that renames a bus leaves them stale - and validate_endpoints then rejects
    the case as a stale mapping. Called after every rename rather than trusting
    each one to remember.
    """
    if "name" not in buses:
        return branch_tables
    lookup = buses.set_index("bus_i")["name"]
    for table in branch_tables:
        if table is None or table.empty:
            continue
        for end in ("fbus", "tbus"):
            column = end + "_name"
            if end in table and column in table:
                refreshed = table[end].map(lookup)
                table.loc[refreshed.notna(), column] = refreshed[refreshed.notna()]
    return branch_tables


def validate_endpoints(buses, branches, check_names=True):
    if buses["bus_i"].duplicated().any() or buses["bus_i"].isna().any():
        raise ValueError("Bus IDs must be present and unique")
    lookup = buses.set_index("bus_i")
    for end in ("fbus", "tbus"):
        if branches.empty:
            continue
        if not branches[end].isin(lookup.index).all():
            raise ValueError(f"Unknown {end} reference; regenerate upstream tables")
        name_col = end + "_name"
        if check_names and name_col in branches and "name" in lookup:
            known = branches[name_col].notna() & branches[name_col].ne("")
            actual = branches[end].map(lookup["name"])
            if (known & branches[name_col].ne(actual)).any():
                raise ValueError(f"Stale {end} name/ID mapping; regenerate upstream tables")


# Intra-substation transformer, matching create_buses.py: 1.6 % impedance on a
# 1000 MVA base with X/R = 80, expressed on the 100 MVA case base.
XFMR_X = 0.016 * (100.0 / 1000.0)
XFMR_R = round(XFMR_X / 80.0, 7)
XFMR_RATE = 1000.0


def _parse_volts(value):
    """Volts present in a voltage tag, tolerating 'a;b' and kV-scaled rows."""
    out = []
    for part in str(value or '').replace(',', ';').split(';'):
        try:
            v = float(part)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v) or v <= 0:
            continue
        out.append(v * 1000.0 if v < 1000.0 else v)
    return out


def _normalised_name(value):
    """Lower-case alphanumerics only, so punctuation and spacing cannot differ."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _separation_km(lon1, lat1, lon2, lat2):
    """Equirectangular separation; exact enough at the few-hundred-metre scale."""
    mean = np.deg2rad((lat1 + lat2) / 2.0)
    return float(np.hypot((lon1 - lon2) * 111.32 * np.cos(mean),
                          (lat1 - lat2) * 110.57))


def merge_duplicate_substations(frame, radius_km=0.5):
    """Collapse OSM records that describe one physical substation.

    Sec. V-B-1 states this rule for power plants - same OSM object, an unnamed
    record beside a named one, or near-identical names - but nothing applies it
    to substations, and OSM maps large sites more than once. The surviving copy
    keeps the richer identity and the union of the voltage tags, so a site mapped
    once per voltage level becomes one compound-voltage substation that Sec.
    III-B then splits into tiers joined by a transformer.

    Two differently named substations are never merged, however close they are.
    """
    if len(frame) < 2:
        return frame, 0
    lon = pd.to_numeric(frame["longitude"], errors="coerce").to_numpy(float)
    lat = pd.to_numeric(frame["latitude"], errors="coerce").to_numpy(float)
    names = [_normalised_name(v) for v in frame.get("name", pd.Series([None] * len(frame)))]
    ids = [str(frame.iloc[i].get("id") or frame.iloc[i].get("@id") or "")
           for i in range(len(frame))]

    parent = list(range(len(frame)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # Only pairs that are close enough to be the same site are even considered.
    order = np.argsort(lon)
    span = radius_km / 111.32 / max(np.cos(np.deg2rad(np.nanmean(lat))), 1e-6)
    for a in range(len(order)):
        i = order[a]
        if not (np.isfinite(lon[i]) and np.isfinite(lat[i])):
            continue
        for b in range(a + 1, len(order)):
            j = order[b]
            if lon[j] - lon[i] > span:
                break
            if not (np.isfinite(lon[j]) and np.isfinite(lat[j])):
                continue
            if _separation_km(lon[i], lat[i], lon[j], lat[j]) > radius_km:
                continue
            same = (ids[i] and ids[i] == ids[j]) or not names[i] or not names[j] \
                or names[i] == names[j]
            if same:
                parent[root(i)] = root(j)

    groups = defaultdict(list)
    for i in range(len(frame)):
        groups[root(i)].append(i)
    duplicates = {k: v for k, v in groups.items() if len(v) > 1}
    if not duplicates:
        return frame, 0

    voltages = frame["voltage"].tolist() if "voltage" in frame.columns else [None] * len(frame)
    keep, drop = [], []
    for members in groups.values():
        if len(members) == 1:
            keep.append(members[0])
            continue
        # Prefer a named record, then one that carries a voltage tag.
        chosen = max(members, key=lambda i: (bool(names[i]),
                                             bool(_parse_volts(voltages[i])), -i))
        levels = sorted({v for i in members for v in _parse_volts(voltages[i])})
        if levels:
            frame.iat[chosen, frame.columns.get_loc("voltage")] = \
                ";".join(str(int(v)) for v in levels)
        keep.append(chosen)
        drop.extend(i for i in members if i != chosen)

    merged = frame.drop(index=frame.index[drop]).reset_index(drop=True)
    return merged, len(drop)


def split_multi_voltage_buses(buses, lines, transformers):
    """Sec. III.B for substations OSM never gave a voltage tag.

    create_buses.py splits a substation whose OSM tag lists several voltages and
    joins the tiers with a transformer. A substation with no tag at all cannot be
    split there, so every line reaching it - 132 kV and 220 kV alike - lands on
    one bus. That shorts the two networks together with no transformer between
    them, which both misstates connectivity and leaves endpoints whose line
    voltage contradicts their bus voltage.

    Here each such bus becomes one bus per voltage present on its own lines, the
    line endpoints move to the tier that matches them, and consecutive tiers are
    joined by a transformer.
    """
    buses = buses.copy()
    lines = lines.copy()
    transformers = transformers.copy()
    if lines.empty:
        return buses, lines, transformers, 0

    attached = defaultdict(set)
    for row in lines.itertuples(index=False):
        volts = _parse_volts(getattr(row, 'voltage_V', None))
        if not volts:
            continue
        for end in ('fbus', 'tbus'):
            bus = getattr(row, end, None)
            if pd.notna(bus):
                attached[int(bus)].add(max(volts))

    multi = {bus: sorted(v) for bus, v in attached.items() if len(v) > 1}
    if not multi:
        return buses, lines, transformers, 0

    by_id = buses.set_index('bus_i')
    next_id = int(buses.bus_i.max()) + 1
    added_rows, new_xfmr = [], []
    tier_of = {}

    for bus, volts in sorted(multi.items()):
        if bus not in by_id.index:
            continue
        template = by_id.loc[bus]
        base_name = re.sub(r'_\d+kV$', '', str(template.get('name', f'bus {bus}')))
        # The original ID keeps the lowest tier so existing references stay valid.
        tier_of[(bus, volts[0])] = bus
        buses.loc[buses.bus_i == bus, 'base_kV'] = float(volts[0])
        if 'voltage_tag' in buses.columns:
            buses.loc[buses.bus_i == bus, 'voltage_tag'] = str(int(volts[0]))
        buses.loc[buses.bus_i == bus, 'name'] = f'{base_name}_{int(volts[0] // 1000)}kV'
        for volt in volts[1:]:
            row = template.to_dict()
            row['bus_i'] = next_id
            row['base_kV'] = float(volt)
            row['name'] = f'{base_name}_{int(volt // 1000)}kV'
            if 'voltage_tag' in row:
                row['voltage_tag'] = str(int(volt))
            # A split tier carries no load of its own; demand stays on the original.
            for zeroed in ('Pd', 'Qd'):
                if zeroed in row:
                    row[zeroed] = 0.0
            tier_of[(bus, volt)] = next_id
            added_rows.append(row)
            next_id += 1
        for low, high in zip(volts[:-1], volts[1:]):
            new_xfmr.append(dict(
                fbus=tier_of[(bus, low)], tbus=tier_of[(bus, high)],
                r=XFMR_R, x=XFMR_X, b=0.0, rateA=XFMR_RATE, rateB=XFMR_RATE,
                rateC=XFMR_RATE, ratio=1.0, angle=0.0, status=1,
                angmin=-360.0, angmax=360.0,
                fbus_name=f'{base_name}_{int(low // 1000)}kV',
                tbus_name=f'{base_name}_{int(high // 1000)}kV',
                length_km=0.0, voltage_V=f'{int(low)};{int(high)}',
                osm_id=str(template.get('osm_id', ''))))

    if added_rows:
        buses = pd.concat([buses, pd.DataFrame(added_rows)], ignore_index=True)

    # Move each line endpoint onto the tier matching its own voltage.
    names = buses.set_index('bus_i')['name'] if 'name' in buses.columns else None
    for index, row in lines.iterrows():
        volts = _parse_volts(row.get('voltage_V'))
        if not volts:
            continue
        want = max(volts)
        for end in ('fbus', 'tbus'):
            bus = row.get(end)
            if pd.isna(bus):
                continue
            target = tier_of.get((int(bus), want))
            if target is not None and target != int(bus):
                lines.at[index, end] = target
                if names is not None and f'{end}_name' in lines.columns:
                    lines.at[index, f'{end}_name'] = names.loc[target]

    if new_xfmr:
        frame = pd.DataFrame(new_xfmr)
        transformers = (frame if transformers.empty
                        else pd.concat([transformers, frame], ignore_index=True))
    return buses, lines, transformers, len(multi)


def _substation_key(row):
    """Group buses belonging to one physical switchyard."""
    osm = str(row.get('osm_id', '') or '').strip()
    if osm:
        return ('osm', osm)
    return ('xy', round(float(row['latitude']), 7), round(float(row['longitude']), 7))


def retier_endpoints(buses, lines):
    """Move an endpoint onto the tier of its own substation that matches the line.

    create_buses.py splits a substation whose OSM tag lists several voltages, so
    both tiers sit at identical coordinates. Sec. IV.B.2 matches on voltage, but
    only when both the line's and the bus's voltage are known at that moment; a
    voltage filled in later during Sec. IV.B.3 can leave a line hanging off the
    wrong busbar of the right switchyard. Once all voltages are settled, such an
    endpoint can be moved to the correct tier without inventing anything.
    """
    buses = buses.copy()
    lines = lines.copy()
    if lines.empty or buses.empty:
        return buses, lines, 0
    if not {'latitude', 'longitude'} <= set(buses.columns):
        return buses, lines, 0

    tiers = defaultdict(dict)
    for row in buses.to_dict('records'):
        volts = _parse_volts(row.get('base_kV'))
        if volts:
            tiers[_substation_key(row)].setdefault(volts[0], int(row['bus_i']))
    by_id = buses.set_index('bus_i')
    names = by_id['name'] if 'name' in buses.columns else None

    moved = 0
    for index, row in lines.iterrows():
        wanted = _parse_volts(row.get('voltage_V'))
        if not wanted:
            continue
        for end in ('fbus', 'tbus'):
            bus = row.get(end)
            if pd.isna(bus) or int(bus) not in by_id.index:
                continue
            current = _parse_volts(by_id.loc[int(bus), 'base_kV'])
            if current and any(abs(current[0] - v) < 1e-6 for v in wanted):
                continue
            siblings = tiers.get(_substation_key(by_id.loc[int(bus)].to_dict()), {})
            target = next((siblings[v] for v in wanted if v in siblings), None)
            if target is None or target == int(bus):
                continue
            lines.at[index, end] = target
            if names is not None and f'{end}_name' in lines.columns:
                lines.at[index, f'{end}_name'] = names.loc[target]
            moved += 1
    return buses, lines, moved


def _branch_pairs(*tables):
    for table in tables:
        if table is None or not len(table):
            continue
        for a, b in zip(table["fbus"], table["tbus"]):
            if pd.isna(a) or pd.isna(b):
                continue
            yield int(a), int(b)


def island_labels(buses, *branches):
    """Component label per bus row, counting transformers as connections."""
    ids = buses["bus_i"].astype(int).tolist()
    position = {b: i for i, b in enumerate(ids)}
    edges = [(position[a], position[b]) for a, b in _branch_pairs(*branches)
             if a in position and b in position]
    n = len(ids)
    if not edges:
        return np.arange(n)
    rows, cols = zip(*edges)
    graph = csr_matrix((np.ones(len(edges)), (rows, cols)), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def island_report(buses, *branches):
    """Describe every component and the shortest hop from each to the largest.

    A national transmission grid is normally one synchronous system, so more than
    one component means either the region really is islanded or the model is
    missing a connection. The proposed hop is the closest voltage-compatible pair
    of buses bridging the gap, which is what a reviewer needs in order to decide.
    """
    labels = island_labels(buses, *branches)
    ids = buses["bus_i"].astype(int).to_numpy()
    lon = pd.to_numeric(buses["longitude"], errors="coerce").to_numpy(float)
    lat = pd.to_numeric(buses["latitude"], errors="coerce").to_numpy(float)
    volts = np.array([_parse_volts(v)[0] if _parse_volts(v) else np.nan
                      for v in buses.get("base_kV", pd.Series([None] * len(buses)))])
    sizes = Counter(labels.tolist())
    main = max(sizes, key=lambda k: (sizes[k], -k))
    core = np.flatnonzero(labels == main)
    islands = []
    for label, size in sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0])):
        if label == main:
            continue
        members = np.flatnonzero(labels == label)
        best = None
        for i in members:
            for j in core:
                if not np.isfinite(lon[i]) or not np.isfinite(lon[j]):
                    continue
                compatible = (not np.isfinite(volts[i]) or not np.isfinite(volts[j])
                              or volts[i] == volts[j])
                gap = _separation_km(lon[i], lat[i], lon[j], lat[j])
                key = (not compatible, gap)
                if best is None or key < best[0]:
                    best = (key, int(ids[i]), int(ids[j]), gap, compatible)
        islands.append(dict(label=int(label), buses=int(size),
                            members=[int(v) for v in ids[members]],
                            nearest_from=None if best is None else best[1],
                            nearest_to=None if best is None else best[2],
                            gap_km=None if best is None else round(best[3], 3),
                            voltage_compatible=None if best is None else bool(best[4])))
    return dict(components=len(sizes), largest=int(sizes[main]),
                buses=len(ids), islands=islands)


def describe_islands(report, expect_connected=True):
    """Human-readable verdict, or an empty string when there is nothing to say."""
    if report["components"] <= 1:
        return ""
    lines = [f"{report['components']} separate networks: the largest holds "
             f"{report['largest']} of {report['buses']} buses."]
    for island in report["islands"]:
        gap = "no candidate" if island["gap_km"] is None else (
            f"{island['gap_km']:.2f} km to bus {island['nearest_to']}"
            + ("" if island["voltage_compatible"] else ", different voltage"))
        lines.append(f"  {island['buses']:>3} bus(es), closest approach from bus "
                     f"{island['nearest_from']}: {gap}")
    if expect_connected:
        lines.append("A national transmission grid is normally one synchronous "
                     "system. Either OSM is missing the lines that close these "
                     "gaps, in which case the Sec. IV.B.3 review should add them, "
                     "or the region really is islanded and EXPECT_CONNECTED_GRID "
                     "should be turned off in config.py.")
    return "\n".join(lines)


def dispatchability_report(bus, branch, gen):
    """Why a case cannot serve its load, in terms an operator would recognise.

    An OPF on a fragmented case fails inside the solver with a singular-matrix
    message that says nothing about the cause. Each island balances on its own,
    so load stranded in an island with no generation is infeasible no matter how
    the solver is configured, and that is worth saying plainly first.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    bus = np.asarray(bus, dtype=float)
    branch = np.asarray(branch, dtype=float)
    gen = np.asarray(gen, dtype=float)
    order = {int(b): i for i, b in enumerate(bus[:, 0])}
    n = len(bus)
    if len(branch):
        # Only in-service branches tie two buses together.
        live = branch[branch[:, 10] > 0] if branch.shape[1] > 10 else branch
        rows = [order[int(x)] for x in live[:, 0]]
        cols = [order[int(x)] for x in live[:, 1]]
    else:
        rows, cols = [], []
    graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    count, label = connected_components(graph, directed=False)

    demand = np.zeros(count)
    capacity = np.zeros(count)
    slack = np.zeros(count, dtype=int)
    buses = np.zeros(count, dtype=int)
    for i, group in enumerate(label):
        demand[group] += bus[i, 2]
        buses[group] += 1
        if bus[i, 1] == 3:
            slack[group] += 1
    for row in gen:
        if row[7] > 0:                      # in-service units only
            capacity[label[order[int(row[0])]]] += row[8]

    stranded = [i for i in range(count) if demand[i] > 1e-9 and capacity[i] <= 1e-9]
    short = [i for i in range(count)
             if demand[i] > 1e-9 and 0 < capacity[i] < demand[i]]
    return dict(islands=count, buses_in_largest=int(buses.max()) if count else 0,
                total_demand_mw=float(demand.sum()),
                total_capacity_mw=float(capacity.sum()),
                islands_without_generation=len(stranded),
                stranded_demand_mw=float(sum(demand[i] for i in stranded)),
                islands_short_of_capacity=len(short),
                islands_without_slack=int((slack == 0).sum()))


def describe_dispatchability(report):
    """Render dispatchability_report as advice, or None when nothing is wrong."""
    problems = []
    if report['islands'] > 1:
        problems.append(
            f"{report['islands']} electrical islands (largest holds "
            f"{report['buses_in_largest']} buses). Each balances separately.")
    if report['islands_without_generation']:
        problems.append(
            f"{report['islands_without_generation']} island(s) carry "
            f"{report['stranded_demand_mw']:.1f} MW with no generation; that load "
            "cannot be served at any cost.")
    if report['islands_short_of_capacity']:
        problems.append(
            f"{report['islands_short_of_capacity']} island(s) have less capacity "
            "than their own demand.")
    if report['total_capacity_mw'] < report['total_demand_mw']:
        problems.append(
            f"System capacity {report['total_capacity_mw']:.1f} MW is below demand "
            f"{report['total_demand_mw']:.1f} MW.")
    if not problems:
        return None
    return ("This case cannot be dispatched as built:\n  - "
            + "\n  - ".join(problems)
            + "\n  Fragmentation usually means endpoints were left unmatched in "
              "Step 3. Re-run the review and assign them instead of skipping.")


def drop_self_loops(branches):
    """Sec. IV.B.2 Step 10: a branch from a bus to itself is not a branch.

    create_lines.py only removes self-loops shorter than 1 km, deliberately
    keeping longer ones for the manual review of Sec. IV.B.3. Nothing re-checked
    them afterwards, so a review that assigned both endpoints of a line to one
    bus left a permanent self-loop in the case. This is that missing final pass.
    """
    if branches.empty or not {"fbus", "tbus"} <= set(branches.columns):
        return branches, 0
    loop = branches.fbus.notna() & branches.tbus.notna() & (branches.fbus == branches.tbus)
    return branches.loc[~loop].reset_index(drop=True), int(loop.sum())


def clean_bus_tables(buses, lines, transformers):
    """Drop self-loops, remove degree-zero buses, and remap all branches."""
    # Line display names may predate voltage inference; IDs are authoritative.
    validate_endpoints(buses, lines, check_names=False)
    validate_endpoints(buses, transformers)
    # Self-loops are removed before the connectivity sweep so that a bus left
    # attached only by its own self-loop is correctly retired as unconnected.
    lines, dropped_lines = drop_self_loops(lines)
    transformers, dropped_xf = drop_self_loops(transformers)
    if dropped_lines or dropped_xf:
        print(f"  Removed {dropped_lines} self-loop line(s) and {dropped_xf} "
              f"self-loop transformer(s) (Sec. IV.B.2 Step 10).")
    connected = set()
    for branches in (lines, transformers):
        for end in ("fbus", "tbus"):
            connected.update(branches[end].dropna())
    kept = buses.loc[buses.bus_i.isin(connected)].copy().reset_index(drop=True)
    mapping = dict(zip(kept.bus_i, range(1, len(kept) + 1)))
    kept["bus_i"] = kept.bus_i.map(mapping)
    outputs = []
    for branches in (lines, transformers):
        branches = branches.copy()
        for end in ("fbus", "tbus"):
            branches[end] = branches[end].map(mapping).astype(int)
        outputs.append(branches)
    return kept, *outputs


def allocate_loads(raw_mw, voltage_v, external, demand_mw, export_mw, site=None):
    """Scale eligible domestic buses, then assign exports without voltage filtering.

    A substation's demand leaves the grid at its lowest modelled voltage, so when
    Sec. III-B has split a site into tiers the load belongs on the lower one.
    Without ``site`` that rule can only be applied against the lowest voltage in
    the whole country, which hands every megawatt to one voltage level and leaves
    standalone higher-voltage substations carrying nothing - all the power then
    has to reach the load through the transformers and the lowest-voltage lines.
    """
    raw = pd.Series(raw_mw, copy=True).astype(float)
    voltage = pd.Series(voltage_v, index=raw.index).astype(float)
    external = pd.Series(external, index=raw.index).astype(bool)
    if not np.isfinite([demand_mw, export_mw]).all() or min(demand_mw, export_mw) < 0:
        raise ValueError("Demand and export must be finite nonnegative MW")
    if not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("Load weights must be finite and nonnegative")
    domestic = ~external
    known = domestic & voltage.gt(0)
    if site is None:
        eligible = known & voltage.eq(voltage.loc[known].min())
    else:
        key = pd.Series(site, index=raw.index).fillna("").astype(str)
        key = key.mask(key.eq(""), pd.Series(raw.index.astype(str), index=raw.index))
        lowest = voltage.where(known).groupby(key).transform("min")
        eligible = known & voltage.eq(lowest)
    total = raw.loc[eligible].sum()
    if demand_mw > 0 and total <= 0:
        raise ValueError("No positive load weights at eligible domestic buses")
    result = pd.Series(0.0, index=raw.index)
    if total > 0:
        result.loc[eligible] = raw.loc[eligible] * (demand_mw / total)
    if export_mw > 0:
        if not external.any():
            raise ValueError("Exports requested but no external buses were identified")
        result.loc[external] = export_mw / external.sum()
    return result


def manual_bus(bus_i, name, coordinates, voltage_kv, columns):
    """Create a valid bus row from interactive kV input."""
    if not np.isfinite(voltage_kv) or voltage_kv <= 0:
        raise ValueError("Voltage must be finite and positive")
    lon, lat = coordinates
    row = dict.fromkeys(columns, np.nan)
    row.update(bus_i=bus_i, name=name, coordinates=f"({lon}, {lat})",
               longitude=lon, latitude=lat, base_kV=voltage_kv * 1000,
               voltage_tag=str(int(voltage_kv * 1000)), type=1, Pd=0.0,
               Qd=0.0, Gs=0.0, Bs=0.0, area=1, zone=1, Vm=1.0,
               Va=0.0, Vmax=1.1, Vmin=0.9)
    return row


def manual_transformer(primary, secondary):
    """Use the same synthetic transformer parameters as create_buses.py."""
    return dict(fbus=primary["bus_i"], tbus=secondary["bus_i"],
                fbus_name=primary["name"], tbus_name=secondary["name"],
                r=.00002, x=.00158, b=0., rateA=1000., rateB=1000., rateC=1000.,
                ratio=0., angle=0., status=1, angmin=-60., angmax=60.,
                length_km=0., voltage_V="", osm_id="transformer")
