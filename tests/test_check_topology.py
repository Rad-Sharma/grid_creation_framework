"""Post-review topology checks and the Sec. IV.B.2 Step 10 self-loop removal."""
import pandas as pd
import pytest

from check_topology import (incompatible_endpoints, mixed_voltage_units,
                            normalize_voltage, report, self_loops)
from pipeline_integrity import clean_bus_tables, drop_self_loops


def test_voltage_normalization_tolerates_mixed_units():
    """Earlier manual-review runs wrote base_kV in kV instead of volts."""
    assert normalize_voltage(132) == 132000.0
    assert normalize_voltage(132000) == 132000.0
    assert normalize_voltage("220000;132000") == 220000.0
    for bad in (None, "", "abc", 0, -5):
        assert normalize_voltage(bad) is None


def test_self_loops_are_removed_before_the_connectivity_sweep():
    """A bus held only by its own self-loop must not survive as connected."""
    buses = pd.DataFrame({'bus_i': [1, 2, 3], 'name': ['keep', 'other', 'loop_only']})
    lines = pd.DataFrame({'fbus': [1, 3], 'tbus': [2, 3],
                          'fbus_name': ['keep', 'loop_only'],
                          'tbus_name': ['other', 'loop_only']})
    xf = pd.DataFrame(columns=['fbus', 'tbus', 'fbus_name', 'tbus_name'])
    buses, lines, xf = clean_bus_tables(buses, lines, xf)
    assert buses.name.tolist() == ['keep', 'other']
    assert lines[['fbus', 'tbus']].values.tolist() == [[1, 2]]


def test_drop_self_loops_reports_its_count_and_leaves_real_branches():
    branches = pd.DataFrame({'fbus': [1, 2, 3], 'tbus': [1, 3, 4]})
    kept, dropped = drop_self_loops(branches)
    assert dropped == 1
    assert kept[['fbus', 'tbus']].values.tolist() == [[2, 3], [3, 4]]


def test_drop_self_loops_handles_empty_and_columnless_frames():
    for frame in (pd.DataFrame(), pd.DataFrame(columns=['fbus', 'tbus'])):
        kept, dropped = drop_self_loops(frame)
        assert dropped == 0 and kept.empty


def test_surviving_self_loop_is_detected():
    lines = pd.DataFrame({'fbus': [1, 2], 'tbus': [1, 3]})
    assert len(self_loops(lines)) == 1


def test_incompatible_endpoint_detected_and_multi_voltage_line_accepted():
    """Step 8 requires the bus voltage to appear in the line's voltage list."""
    buses = pd.DataFrame({'bus_i': [1, 2], 'base_kV': [220000.0, 132000.0]})
    lines = pd.DataFrame({
        'fbus': [1, 1], 'tbus': [2, 2],
        'voltage_V': ['220000;132000', '400000'],
        'osm_id': ['way/ok', 'way/bad'], 'length_km': [10.0, 10.0]})
    bad = incompatible_endpoints(buses, lines)
    # The multi-voltage line matches both buses; the 400 kV line matches neither.
    assert set(bad.osm_id) == {'way/bad'}
    assert len(bad) == 2


def test_incompatible_endpoints_ignores_unknown_voltages():
    buses = pd.DataFrame({'bus_i': [1, 2], 'base_kV': [220000.0, None]})
    lines = pd.DataFrame({'fbus': [1], 'tbus': [2], 'voltage_V': [''],
                          'osm_id': ['way/1'], 'length_km': [1.0]})
    assert incompatible_endpoints(buses, lines).empty


def test_mixed_unit_buses_flagged():
    buses = pd.DataFrame({'bus_i': [1, 2, 3], 'base_kV': [132.0, 132000.0, None]})
    assert mixed_voltage_units(buses).bus_i.tolist() == [1]


def test_kenya_report_still_records_the_known_defects(tmp_path):
    """The checked-in Kenya tables predate the Step 10 fix and must not pass."""
    try:
        r = report('kenya')
    except FileNotFoundError:
        pytest.skip('Kenya working files not present in this checkout')
    assert r['self_loops'] > 0
    assert r['busiest_degree'] > 2 * r['next_degree']
