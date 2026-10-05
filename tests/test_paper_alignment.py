import json
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import requests
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, Point

import config
import create_cost_curves as costs
import paper_features as pf
import paper_load_model as lm
import power_network as pn
import resolve_unmatched as review
from plant_names import MAX_CONSECUTIVE_FAILURES, nearby_plant_name, reset_failure_count
from train_paper_load_model import train
from test_power_network import two_bus


def test_line_coefficients_match_equations_3a_to_3d():
    expected = dict(R=(1115., -2.9870), X=(52.86, -2.0342),
                    B=(4.9e-5, .6896), MVA=(.018429, 1.8998))
    assert review._COEFF == expected


def test_ac_relaxation_installs_no_shunt_it_does_not_need():
    """Section VI-B states the aim as the fewest individual shunt devices.

    The reactive source stands in for B_i^sh, so a case the generators can
    already support must come back with no shunt installed at all.
    """
    solved = pn.solve_feasibility(pn.solve_feasibility(two_bus(qload=60.)), ac=True)
    assert not (np.abs(solved['bus'][:, 5]) > 1e-6).any()
    assert solved['gen'][0, 2] == pytest.approx(65.68, abs=1.)


def test_ac_relaxation_installs_a_shunt_when_generation_cannot_cover_it():
    """Past the generators' reactive limit a shunt has to appear, and only there."""
    data = two_bus(qload=120.)
    solved = pn.solve_feasibility(pn.solve_feasibility(data), ac=True)
    assert (np.abs(solved['bus'][:, 5]) > 1e-6).any()
    assert solved['gen'][0, 2] <= data['gen'][0, 3] + 1e-4
    pn.check_solution(solved, ac=True)


@pytest.mark.parametrize('category, expected', [
    ('steam_landfill', (802.4, 58., .5412)),
    ('steam_petcoke', (241.67, 18.54, .01039)),
    ('cc_coal', (992., 9., .01)),
    ('cc_landfill', (2722., 37., .9)),
    ('cc_petcoke', (1207., 14., .018)),
    ('cc_wood', (554., 22., .09)),
    ('ic_gas', (49.48, 6.446, .06337)),
    ('ic_petroleum', (151.56, 25.156, .05544)),
    ('ic_landfill', (53., 25., .3)),
])
def test_added_table_v_cost_curves(category, expected):
    np.testing.assert_allclose(costs._coeffs(category, 100), expected)


@pytest.mark.parametrize('source, method, expected', [
    ('gas', 'combined_cycle', 'cc_gas'),
    ('gas', 'gas_turbine', 'gt_gas'),
    ('gas', 'internal_combustion', 'ic_gas'),
    ('oil', 'steam', 'steam_oil'),
    ('petroleum coke', 'combined cycle', 'cc_petcoke'),
    ('landfill gas', 'internal combustion', 'ic_landfill'),
])
def test_method_selects_cost_category(source, method, expected):
    assert costs.map_to_category(source, 'gas', method) == expected


@pytest.fixture
def viirs(tmp_path):
    path = tmp_path / 'viirs.tif'
    # Both light and dark sites belong to the same service area. A cell-wide
    # mean would incorrectly give the dark industrial sites credit.
    data = np.tile([0., 0., 3., 3.], (4, 1)).astype('float32')
    with rasterio.open(path, 'w', driver='GTiff', width=4, height=4, count=1,
                       dtype='float32', crs=4326, transform=from_origin(-2, 1, 1, .5), nodata=-9999) as f:
        f.write(data, 1)
    return path


def test_guard_applied_at_industrial_polygon_not_service_area(viirs):
    dark, bright = box(-2, -1, 0, 1), box(0, -1, 2, 1)
    areas = gpd.GeoDataFrame({'poc_code': ['A']}, geometry=[box(-2, -1, 2, 1)], crs=4326)
    industrial = [dict(geometry=g, bucket=b) for g in (dark, bright)
                  for b in ('heavy_mining', 'heavy_chemical')]
    steel = gpd.GeoDataFrame({'facility_id': ['S1', 'S2'], 'capacity_mw': [100., 50.]}, geometry=[dark, bright], crs=4326)
    gold = gpd.GeoDataFrame({'facility_id': ['G1', 'G2']}, geometry=[dark, bright], crs=4326)
    result = pf.industrial_features(areas, industrial, steel, gold, viirs).iloc[0]
    area = gpd.GeoSeries([bright], crs=4326).to_crs(6933).area.iloc[0] / 1e6
    assert result.gem_steel_mw_voronoi == 50
    assert result.fp_gold_ntl == 2
    assert result.mining_ntl_product == pytest.approx(area * 2)
    assert result.ind_heavy_chemical_km2 == pytest.approx(area)
    assert result.feature_contract == lm.FEATURE_CONTRACT


def test_guard_threshold_and_missing_coverage(viirs):
    assert pf.guard(.9) == 0
    assert pf.guard(1.) == 0
    assert pf.guard(2.5) == 1.5
    with pytest.raises(ValueError, match='Missing'):
        pf.guard(np.nan)
    with pytest.raises(ValueError, match='cover'):
        pf.polygon_radiance(box(10, 10, 11, 11), viirs)


def test_no_steel_tonnage_as_mw(tmp_path):
    path = tmp_path / 'steel.geojson'
    gpd.GeoDataFrame({'facility_id': ['S'], 'capacity_ttpa': [100]},
                      geometry=[Point(0, 0)], crs=4326).to_file(path, driver='GeoJSON')
    with pytest.raises(ValueError, match='electrical capacity_mw'):
        pf.read_steel(path)


def test_missing_huber_is_not_replaced_by_ridge(tmp_path):
    with pytest.raises(FileNotFoundError, match='intercept'):
        lm.load_model(tmp_path / 'absent.json')


def test_missing_features_not_silently_zeroed():
    with pytest.raises(ValueError, match='Missing paper'):
        lm.feature_matrix(pd.DataFrame({'pop_density': [10]}))
    frame = pd.DataFrame([dict.fromkeys(lm.FEATURES, 0.)])
    with pytest.raises(ValueError, match='measured'):
        lm.feature_matrix(frame)


def test_huber_refit_and_prediction_preserve_provenance(tmp_path):
    rng = np.random.default_rng(17)
    x = rng.uniform(0, 2, (150, 6))
    log_y = .7 + x @ np.asarray(lm.PUBLISHED_COEFFICIENTS) + rng.normal(0, .01, 150)
    log_y[:3] += 2  # robustness to outliers
    frame = pd.DataFrame(np.expm1(x), columns=lm.FEATURES)
    frame['feature_contract'] = lm.FEATURE_CONTRACT
    frame['avg_mw'] = np.expm1(log_y)
    path, output = tmp_path / 'training.csv', tmp_path / 'model.json'
    frame.to_csv(path, index=False)
    artifact = train([path], output)
    assert artifact['model_type'] == 'huber' and artifact['epsilon'] == 1.1
    assert artifact['provenance']['kind'] == 'new_refit_not_published_artifact'
    np.testing.assert_allclose(artifact['coefficients'], lm.PUBLISHED_COEFFICIENTS, atol=.03)
    loaded = lm.load_model(output)
    predicted = lm.predict(frame, loaded)
    assert np.isfinite(predicted).all() and (predicted >= 0).all()


def test_paper_scaling_keeps_all_substations():
    # Paper national scaling does not discard an entire higher-voltage class.
    result = lm.scale_demands([100., 300., 20.], [False, False, True], 1000., 50.)
    np.testing.assert_allclose(result, [250., 750., 50.])


def test_paper_prediction_files_and_voltage_tier_allocation(tmp_path):
    artifact = dict(model_type='huber', epsilon=1.1, features=lm.FEATURES,
                    feature_contract=lm.FEATURE_CONTRACT, intercept=.7,
                    coefficients=lm.PUBLISHED_COEFFICIENTS, provenance={'kind': 'synthetic_test_fixture'})
    model_path = tmp_path / 'model.json'
    model_path.write_text(json.dumps(artifact))
    buses = pd.DataFrame({'bus_i': [1, 2, 3], 'name': ['A-low', 'A-high', 'B'],
                          'base_kV': [138000, 345000, 345000],
                          'latitude': [30., 30., 31.], 'longitude': [-95., -95., -94.]})
    bus_path = tmp_path / 'buses.csv'
    buses.to_csv(bus_path, index=False)
    features = pd.DataFrame(np.ones((2, 6)), columns=lm.FEATURES)
    features['bus_idx'], features['bus_name'] = [1, 3], ['A-low', 'B']
    features['lat'], features['lon'] = [30., 31.], [-95., -94.]
    features['feature_contract'] = lm.FEATURE_CONTRACT
    feature_path = tmp_path / 'features.csv'
    features.to_csv(feature_path, index=False)
    result = lm.run_prediction(feature_path, bus_path, model_path, tmp_path / 'predictions.csv',
                                1000., 0., lambda *_: set(), 'Test')
    # Higher-voltage B still receives demand; co-located A-high does not duplicate A.
    np.testing.assert_allclose(result.Pd, [500., 0., 500.])
    np.testing.assert_allclose(result.Qd, result.Pd * np.tan(np.arccos(.95)))
    features = features.iloc[:1]
    features.to_csv(feature_path, index=False)
    with pytest.raises(ValueError, match='coverage'):
        lm.run_prediction(feature_path, bus_path, model_path, tmp_path / 'bad.csv',
                          1000., 0., lambda *_: set(), 'Test')
    assert not (tmp_path / 'bad.csv').exists()


def test_plant_name_inference_is_bracketed_nearest_and_cached(tmp_path):
    elements = [dict(type='node', id=1, lat=30, lon=-95.001, tags={'power': 'generator', 'name': 'Near'}),
                dict(type='way', id=2, center={'lat': 30, 'lon': -95.008}, tags={'power': 'plant', 'name': 'Far'})]
    calls = []
    class Session:
        def post(self, *args, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'elements': elements})
    assert nearby_plant_name(30, -95, tmp_path, session=Session()) == ('[Near]', 'node/1')
    assert nearby_plant_name(30, -95, tmp_path, session=Session()) == ('[Near]', 'node/1')
    assert len(calls) == 1

def test_untagged_plants_are_sized_to_the_national_capacity():
    """COUNTRY_CAPACITY_MW closes the Sec. V-B-2 gap the way Sec. III-C-6 scales demand.

    Weight by the median capacity of the tagged plants on the same fuel, then
    scale so the national total is matched. Nothing is assumed when the setting
    is left at 0.
    """
    tagged = pd.DataFrame(dict(genfuel=['hydro', 'hydro', 'solar'], Pmax=[100., 300., 20.]))
    untagged_fuels = ['hydro', 'solar', 'wind']
    by_fuel = tagged.groupby('genfuel')['Pmax'].median()
    fallback = float(tagged['Pmax'].median())
    weights = pd.Series(untagged_fuels).map(by_fuel).fillna(fallback).astype(float)
    shortfall = 1000. - tagged['Pmax'].sum()
    sized = weights * (shortfall / weights.sum())
    assert sized.sum() == pytest.approx(shortfall)
    # A fuel OSM tags well sets the scale; an untagged fuel falls back to the median.
    assert sized.iloc[0] > sized.iloc[1]
    assert sized.iloc[2] == pytest.approx(sized[weights == fallback].iloc[0])


def test_plant_name_lookup_gives_up_after_repeated_failures(tmp_path):
    """Overpass rate-limits, and a region can have hundreds of unnamed plants.

    Once it has clearly stopped answering, every remaining answer is None either
    way, so the run must not spend a request timeout per plant proving it.
    """
    class Refusing:
        calls = 0
        def post(self, *args, **kwargs):
            Refusing.calls += 1
            raise requests.RequestException("429 Too Many Requests")
    reset_failure_count()
    session = Refusing()
    with pytest.warns(UserWarning):
        for _ in range(200):
            assert nearby_plant_name(30, -95, tmp_path, session=session) == (None, None)
    assert Refusing.calls == MAX_CONSECUTIVE_FAILURES
    reset_failure_count()
