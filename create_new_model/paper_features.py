"""Measured industrial features for Automate Grid Creation Table I and equation (2).

Each industrial site's VIIRS mean is computed over its own footprint BEFORE
aggregation. Missing coverage is an error, not a dark site. Steel MW and chemical
area retain their Table I units and are zeroed below the threshold; mining area
and gold count are multiplied by max(0, mean_radiance - 1).
"""
from pathlib import Path
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_geom
from shapely.geometry import mapping

from paper_load_model import FEATURE_CONTRACT


def guard(mean_radiance):
    if not np.isfinite(mean_radiance):
        raise ValueError("Missing measured VIIRS radiance")
    return max(0., float(mean_radiance) - 1.)


def polygon_radiance(geometry, raster_path):
    if geometry.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError("An industrial polygon footprint is required for the VIIRS guard")
    with rasterio.open(raster_path) as source:
        if source.crs is None:
            raise ValueError("VIIRS raster needs a declared CRS")
        footprint = transform_geom("EPSG:4326", source.crs, mapping(geometry))
        try:
            values, _ = mask(source, [footprint], crop=True, filled=False)
        except ValueError as exc:
            raise ValueError("VIIRS raster does not cover an industrial footprint") from exc
        data = values[0].compressed()
        if not len(data) or not np.isfinite(data).all():
            raise ValueError("Industrial footprint has no valid VIIRS pixels")
        return float(np.maximum(data, 0).mean())


def read_steel(path):
    """Read normalized GEM sites with electrical capacity_mw, not steel tonnage."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing: supply GEM-derived sites with facility_id and capacity_mw. "
            "The bundled GEM workbook contains ttpa, not electrical MW; a documented conversion is required.")
    sites = gpd.read_file(path).to_crs(4326)
    if not {"facility_id", "capacity_mw"}.issubset(sites):
        raise ValueError("Steel data requires facility_id and electrical capacity_mw")
    values = pd.to_numeric(sites.capacity_mw, errors="raise")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Steel capacity must be finite nonnegative MW")
    if sites.facility_id.duplicated().any():
        raise ValueError("Duplicate steel facility IDs; resolve plant/unit aggregation first")
    sites["capacity_mw"] = values
    return sites


def read_gold(path):
    """Use a normalized GeoJSON or the bundled Fineprint facility/commodity archive."""
    path = Path(path)
    if path.suffix.lower() != ".zip":
        sites = gpd.read_file(path).to_crs(4326)
    else:
        with zipfile.ZipFile(path) as archive:
            commodities = pd.read_csv(archive.open("data/commodities.csv"))
        gold_ids = set(commodities.loc[commodities.commodity.eq("Me.Au") &
                                      commodities.value_tonnes.gt(0), "facility_id"])
        sites = gpd.read_file(f"/vsizip/{path.resolve().as_posix()}/data/facilities.gpkg").to_crs(4326)
        sites = sites.loc[sites.facility_id.isin(gold_ids)].copy()
        if "facility_type" in sites:
            sites = sites.loc[sites.facility_type.str.contains("mine", case=False, na=False)].copy()
        # Preserve source snapshot provenance; do not claim these records are current.
        sites["source_snapshot"] = path.name
    if "facility_id" not in sites or sites.facility_id.duplicated().any():
        raise ValueError("Gold sites require unique facility_id values")
    return sites


def _owner(geometry, service_areas, id_column):
    point = geometry.representative_point()
    candidates = service_areas.loc[service_areas.geometry.covers(point)]
    if candidates.empty:
        return None
    # Exact boundary ties have one deterministic owner, never duplicate capacity.
    return candidates.sort_values(id_column).iloc[0][id_column]


def _footprint(site, industrial_rows):
    geometry = site.geometry
    if geometry.geom_type in ("Polygon", "MultiPolygon"):
        return geometry
    candidates = [row["geometry"] for row in industrial_rows if row["geometry"].covers(geometry)]
    if not candidates:
        raise ValueError(f"No industrial footprint for facility {site.facility_id}; provide a polygon in the normalized source")
    return min(candidates, key=lambda p: p.area)


def industrial_features(service_areas, industrial_rows, steel, gold, raster_path, id_column="poc_code"):
    if service_areas[id_column].duplicated().any():
        raise ValueError("Duplicate service-area IDs")
    if not Path(raster_path).exists():
        raise FileNotFoundError(f"Measured VIIRS raster missing: {raster_path}; proxy NTL is not permitted in paper mode")
    columns = ["gem_steel_mw_voronoi", "mining_ntl_product", "ind_heavy_chemical_km2", "fp_gold_ntl"]
    out = pd.DataFrame(0., index=service_areas[id_column], columns=columns)
    measured = {}

    def radiance(geometry):
        key = geometry.wkb
        if key not in measured:
            measured[key] = polygon_radiance(geometry, raster_path)
        return measured[key]

    for row in industrial_rows:
        geometry = row["geometry"]
        bucket = row["bucket"]
        if bucket not in ("heavy_mining", "heavy_chemical"):
            continue
        owner = _owner(geometry, service_areas, id_column)
        if owner is None:
            continue
        g = guard(radiance(geometry))
        # Equal-area projection avoids treating degrees squared as area.
        area = gpd.GeoSeries([geometry], crs=4326).to_crs(6933).area.iloc[0] / 1e6
        if bucket == "heavy_mining":
            out.loc[owner, "mining_ntl_product"] += area * g
        elif g > 0:
            out.loc[owner, "ind_heavy_chemical_km2"] += area

    for sites, column in [(steel, "gem_steel_mw_voronoi"), (gold, "fp_gold_ntl")]:
        for _, site in sites.iterrows():
            owner = _owner(site.geometry, service_areas, id_column)
            if owner is None:
                continue
            g = guard(radiance(_footprint(site, industrial_rows)))
            if column == "fp_gold_ntl":
                out.loc[owner, column] += g
            elif g > 0:
                out.loc[owner, column] += float(site.capacity_mw)
    out["feature_contract"] = FEATURE_CONTRACT
    return out.rename_axis(id_column).reset_index()
