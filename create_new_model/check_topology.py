"""Post-review topology checks for a built region.

The automated passes of Sec. IV.B.2 cannot produce a self-loop or a
voltage-incompatible endpoint: find_nearest_bus returns None when nothing
qualifies, excludes the from-bus when choosing the to-bus, and requires the bus
voltage to appear in the line's voltage list whenever both are known.

The manual assignment of Sec. IV.B.3 has none of those guards, and nothing
re-checked the tables afterwards. This module supplies that missing check so a
review session cannot silently leave a sink bus behind.

Run against any built region:

    python create_new_model/check_topology.py --region kenya
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_integrity import read_transformers


def normalize_voltage(value):
    """Return volts, tolerating tables that mix kV and volts in one column.

    Some rows were written in kV by earlier versions of the manual-review
    'new substation' path, so a bare 132 and a 132000 mean the same thing.
    """
    try:
        v = float(str(value).split(";")[0])
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v) or v <= 0:
        return None
    return v * 1000.0 if v < 1000.0 else v


def line_voltages(value):
    out = []
    for part in str(value or "").split(";"):
        v = normalize_voltage(part)
        if v is not None:
            out.append(v)
    return out


def self_loops(lines):
    """Sec. IV.B.2 Step 10 removes self-loops; report any that survived."""
    if lines.empty:
        return lines
    return lines.loc[lines.fbus.notna() & lines.tbus.notna()
                     & (lines.fbus == lines.tbus)]


def mixed_voltage_units(buses):
    """Rows whose base_kV is recorded in kV while the table's norm is volts."""
    values = pd.to_numeric(buses.base_kV, errors="coerce")
    return buses.loc[values.notna() & (values > 0) & (values < 1000.0)]


def incompatible_endpoints(buses, lines):
    """Endpoints where both voltages are known and disagree (Step 8's rule)."""
    lookup = buses.set_index("bus_i").base_kV.map(normalize_voltage)
    rows = []
    for _, line in lines.iterrows():
        known = line_voltages(line.get("voltage_V"))
        if not known:
            continue
        for end in ("fbus", "tbus"):
            bus = line.get(end)
            if pd.isna(bus) or int(bus) not in lookup.index:
                continue
            bus_v = lookup.loc[int(bus)]
            if bus_v is None or pd.isna(bus_v):
                continue
            if not any(abs(bus_v - v) < 1e-6 for v in known):
                rows.append(dict(osm_id=line.get("osm_id"), end=end, bus=int(bus),
                                 bus_volts=bus_v, line_voltage=line.get("voltage_V"),
                                 length_km=line.get("length_km")))
    return pd.DataFrame(rows)


def degree_table(buses, lines, transformers):
    ends = [lines[["fbus", "tbus"]]]
    if not transformers.empty:
        ends.append(transformers[["fbus", "tbus"]])
    stacked = pd.concat(ends)
    degree = pd.concat([stacked.fbus, stacked.tbus]).dropna().astype(int).value_counts()
    return degree.reindex(buses.bus_i, fill_value=0).sort_values(ascending=False)


def report(region, root=None):
    root = Path(root or Path(__file__).resolve().parents[1])
    work = root / "data/working_files" / region
    buses = pd.read_csv(work / f"matpower_{region}_buses.csv")
    path = work / f"matpower_{region}_lines_resolved.csv"
    if not path.exists():
        path = work / f"matpower_{region}_lines.csv"
    lines = pd.read_csv(path)
    transformers = read_transformers(work / f"transformer_pairs_{region}.csv")

    loops = self_loops(lines)
    bad = incompatible_endpoints(buses, lines)
    units = mixed_voltage_units(buses)
    degree = degree_table(buses, lines, transformers)
    endpoints = 2 * len(lines)
    top = degree.index[0] if len(degree) else None
    return dict(region=region, buses=len(buses), lines=len(lines),
                transformers=len(transformers), self_loops=len(loops),
                incompatible_endpoints=len(bad), endpoints=endpoints,
                mixed_unit_buses=len(units),
                busiest_bus=int(top) if top is not None else None,
                busiest_degree=int(degree.iloc[0]) if len(degree) else 0,
                next_degree=int(degree.iloc[1]) if len(degree) > 1 else 0,
                detail=dict(loops=loops, incompatible=bad, units=units, degree=degree))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--region", required=True)
    p.add_argument("--root")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)
    r = report(args.region, args.root)
    print(f"Region {r['region']}: {r['buses']} buses, {r['lines']} lines, "
          f"{r['transformers']} transformers")
    print(f"  self-loops (Sec. IV.B.2 Step 10 removes these): {r['self_loops']}")
    print(f"  voltage-incompatible endpoints: {r['incompatible_endpoints']} of {r['endpoints']}")
    print(f"  buses with base_kV recorded in kV not volts: {r['mixed_unit_buses']}")
    print(f"  busiest bus: {r['busiest_bus']} with degree {r['busiest_degree']} "
          f"(next busiest {r['next_degree']})")
    if args.verbose and r["self_loops"]:
        print(r["detail"]["loops"][["fbus", "tbus", "voltage_V", "length_km", "osm_id"]]
              .to_string(index=False))
    problems = r["self_loops"] + r["incompatible_endpoints"] + r["mixed_unit_buses"]
    if problems:
        print("Rebuild Steps 1-3 and repeat the manual review; these tables are not "
              "a faithful model of the region.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
