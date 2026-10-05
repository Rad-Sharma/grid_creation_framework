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
    print("Availability check only: contents, calibration, network services and solver success are not validated.")
    if missing:
        print(f"{missing} required input(s) missing. See README.md before proceeding.")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
