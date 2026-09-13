"""Shared table invariants for the CSV pipeline. Internal bus voltages are volts."""
from pathlib import Path

import numpy as np
import pandas as pd


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


def clean_bus_tables(buses, lines, transformers):
    """Remove degree-zero buses and remap *all* branches in one operation."""
    # Line display names may predate voltage inference; IDs are authoritative.
    validate_endpoints(buses, lines, check_names=False)
    validate_endpoints(buses, transformers)
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


def allocate_loads(raw_mw, voltage_v, external, demand_mw, export_mw):
    """Scale eligible domestic buses, then assign exports without voltage filtering."""
    raw = pd.Series(raw_mw, copy=True).astype(float)
    voltage = pd.Series(voltage_v, index=raw.index).astype(float)
    external = pd.Series(external, index=raw.index).astype(bool)
    if not np.isfinite([demand_mw, export_mw]).all() or min(demand_mw, export_mw) < 0:
        raise ValueError("Demand and export must be finite nonnegative MW")
    if not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("Load weights must be finite and nonnegative")
    domestic = ~external
    min_v = voltage.loc[domestic & voltage.gt(0)].min()
    eligible = domestic & voltage.eq(min_v)
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
