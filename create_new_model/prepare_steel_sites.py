"""Convert GEM steel tonnage into explicitly documented electrical MW estimates.

The paper does not give this conversion. An energy intensity must therefore be
provided explicitly; the result is an assumption, not measured/registered MW.
Use externally established capacity_mw values instead when available.
"""
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def prepare(workbook, output, mwh_per_tonne):
    if not np.isfinite(mwh_per_tonne) or mwh_per_tonne <= 0:
        raise ValueError("A documented positive MWh/tonne intensity is required")
    table = pd.read_excel(workbook, sheet_name="Steel Plants")
    table = table.loc[table["Capacity operating status"].str.lower().eq("operating")].copy()
    coords = table.Coordinates.astype(str).str.extract(r"^\s*([-+\d.]+)\s*,\s*([-+\d.]+)\s*$")
    tonnage = pd.to_numeric(table["Nominal crude steel capacity (ttpa)"], errors="coerce")
    if coords.isna().any().any() or tonnage.isna().any():
        raise ValueError("Operating GEM sites have missing/ambiguous capacity or coordinates; curate them before conversion")
    out = gpd.GeoDataFrame(dict(facility_id=table["Plant ID"], name=table["Plant name (English)"],
                               capacity_mw=tonnage * 1000 * mwh_per_tonne / 8760,
                               capacity_basis="estimated_tonnes_to_mw", mwh_per_tonne=mwh_per_tonne,
                               source_file=Path(workbook).name),
                           geometry=gpd.points_from_xy(coords[1].astype(float), coords[0].astype(float)), crs=4326)
    out.to_file(output, driver="GeoJSON")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mwh-per-tonne", type=float, required=True)
    args = parser.parse_args()
    prepare(args.workbook, args.output, args.mwh_per_tonne)
