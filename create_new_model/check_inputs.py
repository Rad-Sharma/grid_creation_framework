"""Check local input availability before starting the numbered CSV pipeline."""
import argparse
from pathlib import Path

import config


def required_inputs(root, settings, stage="start"):
    """Return (purpose, alternative paths) pairs; do not fetch or modify data."""
    data = Path(root) / "data"
    raw = data / "bus_data" / "raw"
    models = data / "bus_data" / "models"
    region = settings.region
    method = settings.LOAD_MODEL_METHOD
    if method not in {"paper_huber", "legacy_ridge"}:
        raise ValueError("LOAD_MODEL_METHOD must be paper_huber or legacy_ridge")
    required = []
    if stage == "start":
        for kind in ("buses", "lines", "plants"):
            required.append((f"OSM {kind}", [data / "OSM_extracts" / f"export_{region}_{kind}.geojson"]))
    else:
        required.append(("bus table from Steps 1–3", [
            data / "working_files" / region / f"matpower_{region}_buses.csv"]))
    if method == "paper_huber":
        required.append(("fitted Huber artifact (see README.md)", [models / settings.PAPER_HUBER_MODEL]))
        if stage != "5":
            required.extend([
                ("measured VIIRS radiance", [raw / settings.VIIRS_RASTER_NAME]),
                ("normalized steel electrical MW", [raw / settings.STEEL_FACILITIES_NAME]),
                ("gold facilities", [raw / settings.GOLD_FACILITIES_NAME,
                                     raw / "gem" / "fineprint_mine_production.zip"]),
            ])
        feature_name = f"{region}_paper_features.csv"
    else:
        required.append(("legacy Ridge bundle", [models / "model_bundle_v5.pkl"]))
        feature_name = f"{region}_features_with_industrial.csv"
    if stage == "5":
        required.append(("features from Step 4", [data / "bus_data" / "features" / feature_name]))
    return required


def check_region_settings(root, settings):
    """Report Table VI settings that name something NaturalEarth does not have.

    SHAPE_FILE_NAME, COUNTRY_ISO3 and REGION_BBOX are three of the seven inputs a
    user supplies for a new region, and each one fails quietly today: an
    unmatched name disables cross-border detection, an unmatched ISO code demotes
    every Voronoi cell to the bounding rectangle, and a bbox that misses the
    country clips the network away. Checking them against the committed boundary
    file costs nothing and turns three silent wrong answers into three messages.
    """
    problems = []
    countries = Path(root) / "data" / "boundaries" / "ne_110m_admin_0_countries.zip"
    if not countries.is_file():
        return [("NOTE", f"cannot verify the region settings: {countries} is missing")]
    try:
        import geopandas as gpd
    except ImportError:
        return [("NOTE", "cannot verify the region settings: geopandas is not installed")]
    world = gpd.read_file(countries)
    names = set(world["NAME"].dropna())
    if settings.SHAPE_FILE_NAME not in names:
        import difflib
        close = difflib.get_close_matches(settings.SHAPE_FILE_NAME, sorted(names), n=3)
        hint = f"; closest NaturalEarth names: {', '.join(close)}" if close else ""
        problems.append(("SETTING",
                         f"SHAPE_FILE_NAME {settings.SHAPE_FILE_NAME!r} is not a NaturalEarth "
                         f"country name, so cross-border buses cannot be detected{hint}"))
    iso_columns = [c for c in ("ISO_A3", "ADM0_A3", "ISO_A3_EH") if c in world.columns]
    match = world[[any(world.loc[i, c] == settings.COUNTRY_ISO3 for c in iso_columns)
                   for i in world.index]]
    if match.empty:
        problems.append(("SETTING",
                         f"COUNTRY_ISO3 {settings.COUNTRY_ISO3!r} matches no NaturalEarth "
                         f"country, so Voronoi cells will be clipped to the bounding box "
                         f"instead of the coastline"))
    elif settings.SHAPE_FILE_NAME in names:
        named = world[world["NAME"] == settings.SHAPE_FILE_NAME]
        if not set(named.index) & set(match.index):
            problems.append(("SETTING",
                             f"COUNTRY_ISO3 {settings.COUNTRY_ISO3!r} and SHAPE_FILE_NAME "
                             f"{settings.SHAPE_FILE_NAME!r} name different countries "
                             f"({match.iloc[0]['NAME']!r} and {settings.SHAPE_FILE_NAME!r})"))
    box = tuple(settings.REGION_BBOX)
    min_lon, min_lat, max_lon, max_lat = box
    if not (min_lon < max_lon and min_lat < max_lat):
        problems.append(("SETTING",
                         f"REGION_BBOX {box} is not (min_lon, min_lat, max_lon, max_lat)"))
    elif not match.empty:
        from shapely.geometry import box as _box
        rectangle = _box(min_lon, min_lat, max_lon, max_lat)
        land = match.geometry.union_all()
        if not land.intersects(rectangle):
            problems.append(("SETTING",
                             f"REGION_BBOX {box} does not overlap {match.iloc[0]['NAME']!r} "
                             f"at all; every component would be clipped away"))
        else:
            # The box is a study area, not necessarily the whole country, and a
            # country with overseas territories is never fully inside one box.
            # So report the share left out rather than calling it an error.
            outside = 1 - land.intersection(rectangle).area / land.area
            if outside > 0.01:
                problems.append(("NOTE",
                                 f"REGION_BBOX leaves {outside:.0%} of "
                                 f"{match.iloc[0]['NAME']!r} outside the box; anything "
                                 f"there is clipped away, which is right only if the "
                                 f"study area is smaller than the country"))
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("start", "4", "5"), default="start",
                        help="Check before starting, before Step 4, or before Step 5")
    args = parser.parse_args(argv)
    print(f"Region: {config.region}; load model: {config.LOAD_MODEL_METHOD}")
    try:
        required = required_inputs(Path(__file__).resolve().parents[1], config, args.stage)
    except ValueError as exc:
        parser.error(str(exc))
    missing = 0
    for purpose, alternatives in required:
        found = next((path for path in alternatives if path.is_file()), None)
        if found is None:
            missing += 1
            print(f"MISSING {purpose}: " + " OR ".join(map(str, alternatives)))
        else:
            print(f"OK {purpose}: {found}")
    for severity, problem in check_region_settings(Path(__file__).resolve().parents[1], config):
        missing += severity == "SETTING"
        print(f"{severity} {problem}")
    print("Availability check only: contents, calibration, network services and solver success are not validated.")
    if missing:
        print(f"{missing} required input(s) missing. See README.md before proceeding.")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
