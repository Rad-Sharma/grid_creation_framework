from pathlib import Path
import runpy
import shutil

import numpy as np
import pandas as pd
import pytest

import config
import predict_bus_loads as loads
import resolve_unmatched as review
from pipeline_integrity import (TRANSFORMER_COLUMNS, allocate_loads, clean_bus_tables,
                                read_transformers, validate_endpoints, manual_bus)


def test_transformer_only_bus_survives_and_all_ids_are_remapped():
    buses = pd.DataFrame({'bus_i': [1, 2, 3, 4], 'name': ['orphan', 'HV', 'LV', 'remote']})
    lines = pd.DataFrame({'fbus': [2], 'tbus': [4], 'fbus_name': ['HV'], 'tbus_name': ['remote']})
    xf = pd.DataFrame({'fbus': [2], 'tbus': [3], 'fbus_name': ['HV'], 'tbus_name': ['LV']})
    buses, lines, xf = clean_bus_tables(buses, lines, xf)
    assert buses.name.tolist() == ['HV', 'LV', 'remote']
    assert lines[['fbus', 'tbus']].values.tolist() == [[1, 3]]
    assert xf[['fbus', 'tbus']].values.tolist() == [[1, 2]]
    validate_endpoints(buses, lines)
    validate_endpoints(buses, xf)


def test_stale_name_mapping_rejected():
    buses = pd.DataFrame({'bus_i': [1, 2], 'name': ['A', 'B']})
    xf = pd.DataFrame({'fbus': [1], 'tbus': [2], 'fbus_name': ['Old A']})
    with pytest.raises(ValueError, match='Stale'):
        validate_endpoints(buses, xf)


def test_missing_transformer_bus_rejected():
    # Reproduce stale IDs without depending on unrepaired regional artifacts.
    buses = pd.DataFrame({'bus_i': [1, 2], 'name': ['A', 'B']})
    xf = pd.DataFrame({'fbus': [1], 'tbus': [3]})
    with pytest.raises(ValueError, match='Unknown'):
        validate_endpoints(buses, xf)


def test_demand_and_export_conserved_after_voltage_selection():
    result = allocate_loads([600, 400, 100], [138000, 345000, 345000],
                            [False, False, True], 1000, 50)
    np.testing.assert_allclose(result, [1000, 0, 50])


@pytest.mark.parametrize('raw,voltage,external,demand,exports', [
    ([0], [138000], [False], 100, 0),
    ([100], [np.nan], [False], 100, 0),
    ([100], [138000], [False], 100, 5),
])
def test_load_allocation_rejects_missing_inputs(raw, voltage, external, demand, exports):
    with pytest.raises(ValueError):
        allocate_loads(raw, voltage, external, demand, exports)


def test_no_unmatched_returns_both_tables():
    lines = pd.DataFrame({'fbus': [1], 'tbus': [2]})
    buses = pd.DataFrame({'bus_i': [1, 2]})
    actual_lines, actual_buses = review.run_pass1(lines, {}, buses, [], [])
    pd.testing.assert_frame_equal(actual_lines, lines)
    pd.testing.assert_frame_equal(actual_buses, buses)


def test_manual_bus_uses_volts_and_valid_defaults():
    row = manual_bus(3, 'new', (-95, 30), 138, ['bus_i', 'base_kV', 'Vm'])
    assert row['base_kV'] == 138000
    assert row['voltage_tag'] == '138000'
    assert row['Vm'] == 1 and row['Vmin'] == .9 and row['type'] == 1


@pytest.mark.parametrize('kv', [0, -138, np.nan, np.inf])
def test_invalid_manual_voltage(kv):
    with pytest.raises(ValueError):
        manual_bus(3, 'new', (-95, 30), kv, [])


def test_empty_transformer_files(tmp_path):
    path = tmp_path / 'transformers.csv'
    assert read_transformers(path).empty
    path.write_text('\n')
    assert read_transformers(path).empty
    pd.DataFrame(columns=TRANSFORMER_COLUMNS).to_csv(path, index=False)
    assert list(pd.read_csv(path)) == TRANSFORMER_COLUMNS


def test_bus_script_writes_headers_without_transformers(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'region', 'test')
    script = tmp_path / 'create_new_model/create_buses.py'
    script.parent.mkdir()
    source = Path(__file__).resolve().parents[1] / 'create_new_model/create_buses.py'
    shutil.copyfile(source, script)
    inputs = tmp_path / 'data/OSM_extracts'
    inputs.mkdir(parents=True)
    (inputs / 'export_test_buses.geojson').write_text('''{"type":"FeatureCollection","features":[
      {"type":"Feature","properties":{"name":"Test","voltage":"138000"},
       "geometry":{"type":"Point","coordinates":[-95,30]}}]}''')
    runpy.run_path(str(script), run_name='__main__')
    path = tmp_path / 'data/working_files/test/transformer_pairs_test.csv'
    assert list(pd.read_csv(path)) == TRANSFORMER_COLUMNS


def test_dark_industrial_site_gets_no_addon(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOAD_MODEL_METHOD", "legacy_ridge")
    for attr in ('WORK_DIR', 'MODEL_DIR', 'FEAT_DIR', 'PROC_DIR'):
        monkeypatch.setattr(loads, attr, tmp_path)
    monkeypatch.setattr(loads, 'region', 'test')
    monkeypatch.setattr(loads, 'COUNTRY_DEMAND_MW', 1000.)
    monkeypatch.setattr(loads, 'NATIONAL_EXPORT_MW', 0.)
    monkeypatch.setattr(loads, 'detect_cross_border_buses', lambda *_: set())
    class ZeroModel:
        def predict(self, x):
            return np.zeros(len(x))
    class Identity:
        def transform(self, x):
            return x
    monkeypatch.setattr(loads.joblib, 'load', lambda _: dict(model=ZeroModel(), features=[], imputer=Identity()))
    (tmp_path / 'model_bundle_v5.pkl').touch()
    pd.DataFrame({'bus_i': [1, 2], 'latitude': [30, 32], 'longitude': [-95, -95],
                  'base_kV': [138000, 138000]}).to_csv(tmp_path / 'matpower_test_buses.csv', index=False)
    pd.DataFrame({'bus_idx': [1, 2], 'bus_name': ['dark', 'bright'],
                  'lat': [30, 32], 'lon': [-95, -95], 'ntl_mean': [0, 10],
                  'ntl_sum': [0, 10], 'area_km2': [1, 1],
                  'fp_aluminum_count': [1, 1]}).to_csv(tmp_path / 'test_features_with_industrial.csv', index=False)
    loads.main()
    result = pd.read_csv(tmp_path / 'test_substation_predictions_v4.csv')
    np.testing.assert_allclose(result.Pd, [0, 1000])


def test_review_main_keeps_transformer_and_can_run_without_prompts(tmp_path, monkeypatch):
    monkeypatch.setattr(review, 'WORK_DIR', tmp_path)
    monkeypatch.setattr(review, 'region', 'test')
    buses = pd.DataFrame([manual_bus(i, name, (-95 + i, 30), kv, [])
                          for i, name, kv in [(1, 'orphan', 138), (2, 'HV', 345),
                                               (3, 'LV', 138), (4, 'remote', 345)]])
    buses.to_csv(tmp_path / 'matpower_test_buses.csv', index=False)
    lines = pd.DataFrame({'fbus': [2], 'tbus': [4], 'osm_id': ['way/1'],
                          'voltage_V': [345000], 'length_km': [10.], 'r': [np.nan],
                          'fbus_name': ['HV'], 'tbus_name': ['remote']})
    lines.to_csv(tmp_path / 'matpower_test_lines.csv', index=False)
    pd.DataFrame({'id': ['way/1'], 'geometry': ['LINESTRING (-93 30, -91 30)']}).to_csv(
        tmp_path / 'processed_test_lines.csv', index=False)
    pd.DataFrame({'fbus': [2], 'tbus': [3], 'fbus_name': ['HV'], 'tbus_name': ['LV']}).to_csv(
        tmp_path / 'transformer_pairs_test.csv', index=False)
    review.main()
    buses = pd.read_csv(tmp_path / 'matpower_test_buses.csv')
    xf = read_transformers(tmp_path / 'transformer_pairs_test.csv')
    assert buses.name.tolist() == ['HV', 'LV', 'remote']
    assert xf[['fbus', 'tbus']].values.tolist() == [[1, 2]]
    assert pd.read_csv(tmp_path / 'matpower_test_lines_resolved.csv').r.iloc[0] > 0


def test_manual_review_creates_two_voltage_tiers_and_transformer(tmp_path, monkeypatch):
    monkeypatch.setattr(review, 'WORK_DIR', tmp_path)
    monkeypatch.setattr(review, 'region', 'test')
    monkeypatch.setattr(review, '_build_folium_map', lambda *args: 'test.html')
    monkeypatch.setattr(review.webbrowser, 'open', lambda *args: None)
    answers = iter(['new', 'Added', '138', '345'])
    monkeypatch.setattr('builtins.input', lambda *args: next(answers))
    buses = pd.DataFrame([manual_bus(1, 'Existing', (-95, 30), 138, [])])
    lines = pd.DataFrame({'fbus': [1], 'tbus': [np.nan], 'osm_id': ['way/1'],
                          'voltage_V': [138000], 'length_km': [10], 'fbus_name': ['Existing'],
                          'tbus_name': [None]})
    lines, buses = review.run_pass1(lines, {'way/1': [(-95, 30), (-94.9, 30)]},
                                    buses, [(-95, 30)], ['138000'])
    assert buses.base_kV.tolist() == [138000, 138000, 345000]
    assert lines.tbus.tolist() == [2]
    xf = read_transformers(tmp_path / 'transformer_pairs_test.csv')
    assert xf[['fbus', 'tbus']].values.tolist() == [[2, 3]]
    validate_endpoints(buses, xf)


def test_csv_assembly_and_solver_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOAD_MODEL_METHOD", "legacy_ridge")
    import ensure_feasibility as ef
    from power_network import check_solution
    monkeypatch.setattr(config, 'region', 'test')
    work = tmp_path / 'data/working_files/test'
    work.mkdir(parents=True)
    script = tmp_path / 'create_new_model/create_synthetic_grid.py'
    script.parent.mkdir()
    shutil.copyfile(Path(__file__).resolve().parents[1] / 'create_new_model/create_synthetic_grid.py', script)
    buses = pd.DataFrame([manual_bus(1, 'A', (-95, 30), 138, []),
                          manual_bus(2, 'B', (-94, 30), 138, [])])
    buses.loc[0, 'type'] = 3
    buses.loc[1, ['Pd', 'Qd']] = [20, 5]
    buses.to_csv(work / 'matpower_test_buses.csv', index=False)
    pd.DataFrame([dict(fbus=1, tbus=2, r=.01, x=.1, b=0., rateA=100,
                       rateB=100, rateC=100)]).to_csv(work / 'matpower_test_lines_resolved.csv', index=False)
    # Legacy headerless file must still assemble successfully.
    (work / 'transformer_pairs_test.csv').write_text('\n')
    pd.DataFrame([dict(bus=1, generator_name='G', source='gas', Pg=0., Qg=0., Qmax=50.,
                       Qmin=-50., Vg=1., mBase=100., status=1, Pmax=100., Pmin=0.)]).to_csv(
        work / 'matpower_test_generators.csv', index=False)
    pd.DataFrame([dict(bus=1, generator_name='G', model=2, n=3, c2=.01, c1=10., c0=0.)]).to_csv(
        work / 'matpower_test_cost_curves.csv', index=False)
    runpy.run_path(str(script), run_name='__main__')
    base = ef.load_matpower(work / 'test_synthetic_grid.m')
    dc = ef.solve_dc_feasibility(base)
    ac = ef.solve_ac_feasibility(dc)
    check_solution(ac, ac=True)
    assert ac['gen'][0, 1] > 20
    monkeypatch.setattr(ef, 'WORK_DIR', work)
    monkeypatch.setattr(ef, 'region', 'test')
    path = work / 'test_ac.m'
    ef.write_matpower(ac, path, 'AC_feasible')
    check_solution(ef.load_matpower(path), ac=True)
