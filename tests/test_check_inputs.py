from types import SimpleNamespace

import pytest

from check_inputs import required_inputs


def settings(method):
    return SimpleNamespace(region="example", LOAD_MODEL_METHOD=method,
                           PAPER_HUBER_MODEL="custom.json", VIIRS_RASTER_NAME="custom.tif",
                           STEEL_FACILITIES_NAME="steel.geojson", GOLD_FACILITIES_NAME="gold.geojson")


def test_paper_start_checks_calibration_before_manual_review(tmp_path):
    required = required_inputs(tmp_path, settings("paper_huber"))
    paths = [p.name for _, alternatives in required for p in alternatives]
    assert "custom.json" in paths
    assert "custom.tif" in paths
    assert "export_example_plants.geojson" in paths
    assert "example_paper_features.csv" not in paths
    gold_options = next(paths for purpose, paths in required if purpose == "gold facilities")
    assert [p.name for p in gold_options] == ["gold.geojson", "fineprint_mine_production.zip"]


@pytest.mark.parametrize("method,feature,model", [
    ("paper_huber", "example_paper_features.csv", "custom.json"),
    ("legacy_ridge", "example_features_with_industrial.csv", "model_bundle_v5.pkl"),
])
def test_prediction_checks_only_consumed_inputs(tmp_path, method, feature, model):
    paths = [p.name for _, alternatives in required_inputs(tmp_path, settings(method), "5")
             for p in alternatives]
    assert set(paths) == {"matpower_example_buses.csv", feature, model}


def test_unknown_method_rejected(tmp_path):
    with pytest.raises(ValueError, match="LOAD_MODEL_METHOD"):
        required_inputs(tmp_path, settings("typo"))
