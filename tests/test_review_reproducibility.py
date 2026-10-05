"""Reproducibility of the Sec. IV.B.3 manual review and of a region's settings."""
import json

import numpy as np
import pandas as pd
import pytest

import build_provenance
import config
import resolve_unmatched as review
from pipeline_integrity import manual_bus


class Settings:
    """Stand-in for the config module."""
    def __init__(self, **kw):
        defaults = dict(region='test', COUNTRY_ISO3='TST', REGION_BBOX=(0, 0, 1, 1),
                        SHAPE_FILE_NAME='Test', MIN_TRANSMISSION_VOLTAGE_V=115000,
                        COUNTRY_DEMAND_MW=100.0, NATIONAL_EXPORT_MW=0.0,
                        LOAD_MODEL_METHOD='paper_huber',
                        INFER_TRANSFORMERS_FROM_VOLTAGE=True, FUEL_COST_BASIS='us',
                        PLANT_NAME_SEARCH_RADIUS_M=1000)
        defaults.update(kw)
        self.__dict__.update(defaults)


# ------------------------------------------------------------ provenance

def test_settings_are_recorded_and_tuples_survive_json(tmp_path):
    build_provenance.record(Settings(), tmp_path, 'create_buses')
    saved = json.loads((tmp_path / 'build_config.json').read_text())
    assert saved['settings']['MIN_TRANSMISSION_VOLTAGE_V'] == 115000
    assert saved['settings']['REGION_BBOX'] == [0, 0, 1, 1]
    assert [s['step'] for s in saved['steps']] == ['create_buses']


def test_changing_a_setting_mid_build_is_refused(tmp_path):
    """The exact failure that leaves a region's real settings unknowable."""
    build_provenance.record(Settings(), tmp_path, 'create_buses')
    with pytest.raises(ValueError, match='MIN_TRANSMISSION_VOLTAGE_V'):
        build_provenance.verify(Settings(MIN_TRANSMISSION_VOLTAGE_V=69000), tmp_path)


def test_matching_settings_append_to_the_step_log(tmp_path):
    build_provenance.record(Settings(), tmp_path, 'create_buses')
    build_provenance.verify(Settings(), tmp_path, 'create_lines')
    saved = json.loads((tmp_path / 'build_config.json').read_text())
    assert [s['step'] for s in saved['steps']] == ['create_buses', 'create_lines']


def test_region_predating_provenance_is_adopted_with_a_warning(tmp_path, capsys):
    build_provenance.verify(Settings(), tmp_path, 'resolve_unmatched')
    assert 'WARNING' in capsys.readouterr().out
    assert (tmp_path / 'build_config.json').is_file()


def test_input_digests_are_recorded(tmp_path):
    osm = tmp_path / 'export_test_buses.geojson'
    osm.write_text('{}')
    build_provenance.record(Settings(), tmp_path, 'create_buses', inputs=[osm])
    saved = json.loads((tmp_path / 'build_config.json').read_text())
    assert len(saved['inputs']['export_test_buses.geojson']) == 64


# ---------------------------------------------------------- review guards

def _one_unmatched_line(tmp_path, monkeypatch, buses):
    monkeypatch.setattr(review, 'WORK_DIR', tmp_path)
    monkeypatch.setattr(review, 'region', 'test')
    monkeypatch.setattr(review, '_build_folium_map', lambda *a: 'test.html')
    monkeypatch.setattr(review.webbrowser, 'open', lambda *a: None)
    lines = pd.DataFrame({'fbus': [1], 'tbus': [np.nan], 'osm_id': ['way/1'],
                          'voltage_V': [138000], 'length_km': [10],
                          'fbus_name': ['A'], 'tbus_name': [None]})
    return lines, pd.DataFrame(buses)


def test_manual_pick_cannot_create_a_self_loop(tmp_path, monkeypatch, capsys):
    """The automated passes exclude the from-bus; the prompt now does too."""
    buses = [manual_bus(1, 'A', (-95, 30), 138, []),
             manual_bus(2, 'B', (-94.9, 30), 138, [])]
    lines, bus_df = _one_unmatched_line(tmp_path, monkeypatch, buses)
    log = review.ReviewLog()
    # First answer 1 (self-loop, rejected), then 2 (accepted).
    answers = iter(['1', '2'])
    monkeypatch.setattr('builtins.input', lambda *a: next(answers))
    lines, _ = review.run_pass1(lines, {'way/1': [(-95, 30), (-94.9, 30)]},
                                bus_df, [(-95, 30), (-94.9, 30)],
                                ['138000', '138000'], log=log)
    assert 'self-loop' in capsys.readouterr().out
    assert lines.tbus.tolist() == [2]


def test_voltage_mismatch_needs_confirmation_and_is_recorded(tmp_path, monkeypatch, capsys):
    buses = [manual_bus(1, 'A', (-95, 30), 138, []),
             manual_bus(2, 'B', (-94.9, 30), 345, [])]
    lines, bus_df = _one_unmatched_line(tmp_path, monkeypatch, buses)
    log = review.ReviewLog()
    # Pick the 345 kV bus for a 138 kV line, decline, then accept on retry.
    answers = iter(['2', 'n', '2', 'y'])
    monkeypatch.setattr('builtins.input', lambda *a: next(answers))
    lines, _ = review.run_pass1(lines, {'way/1': [(-95, 30), (-94.9, 30)]},
                                bus_df, [(-95, 30), (-94.9, 30)],
                                ['138000', '345000'], log=log)
    out = capsys.readouterr().out
    assert 'WARNING' in out and 'deliberate voltage override' in out
    assert lines.tbus.tolist() == [2]
    assert [e['answer'] for e in log.entries] == ['2', 'n', '2', 'y']


# ------------------------------------------------------------ journalling

def test_journal_records_endpoint_context(tmp_path, monkeypatch):
    buses = [manual_bus(1, 'A', (-95, 30), 138, []),
             manual_bus(2, 'B', (-94.9, 30), 138, [])]
    lines, bus_df = _one_unmatched_line(tmp_path, monkeypatch, buses)
    log = review.ReviewLog(tmp_path / 'journal.json')
    monkeypatch.setattr('builtins.input', lambda *a: '2')
    review.run_pass1(lines, {'way/1': [(-95, 30), (-94.9, 30)]}, bus_df,
                     [(-95, 30), (-94.9, 30)], ['138000', '138000'], log=log)
    log.save('test')
    saved = json.loads((tmp_path / 'journal.json').read_text())
    assert saved['region'] == 'test'
    context = saved['answers'][0]['context']
    assert context['osm_id'] == 'way/1' and context['endpoint'] == 'tbus'


def test_replaying_a_journal_reproduces_the_session_without_prompting(tmp_path, monkeypatch):
    """Same OSM input plus the journal must give the same model, hands-off."""
    buses = [manual_bus(1, 'A', (-95, 30), 138, []),
             manual_bus(2, 'B', (-94.9, 30), 138, [])]
    coords, volts = [(-95, 30), (-94.9, 30)], ['138000', '138000']
    geom = {'way/1': [(-95, 30), (-94.9, 30)]}

    lines, bus_df = _one_unmatched_line(tmp_path, monkeypatch, buses)
    log = review.ReviewLog(tmp_path / 'journal.json')
    monkeypatch.setattr('builtins.input', lambda *a: '2')
    first, _ = review.run_pass1(lines, geom, bus_df, coords, volts, log=log)
    log.save('test')

    lines2, bus_df2 = _one_unmatched_line(tmp_path, monkeypatch, buses)

    def refuse(*a):
        raise AssertionError('replay must not prompt')

    monkeypatch.setattr('builtins.input', refuse)
    replay = review.ReviewLog(tmp_path / 'journal.json', replay=True)
    second, _ = review.run_pass1(lines2, geom, bus_df2, coords, volts, log=replay)
    assert second.tbus.tolist() == first.tbus.tolist() == [2]


def test_replay_without_a_journal_is_an_explicit_error(tmp_path):
    with pytest.raises(FileNotFoundError, match='No review journal'):
        review.ReviewLog(tmp_path / 'missing.json', replay=True)
