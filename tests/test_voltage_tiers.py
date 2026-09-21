"""Multi-voltage substation splitting and endpoint re-tiering.

Both address the same defect: a line ending on a bus whose nominal voltage
contradicts the line's own. Sec. IV.B.2 matches on voltage, but only when both
values are known at that moment, and Sec. IV.B.3 fills missing voltages
afterwards. Without these passes a 132 kV and a 220 kV network end up shorted
together at one bus with no transformer between them.
"""
import numpy as np
import pandas as pd

from pipeline_integrity import (XFMR_R, XFMR_X, _parse_volts, retier_endpoints,
                                split_multi_voltage_buses)


def _bus(bus_i, name, kv, lat=30.0, lon=-95.0, osm_id=''):
    return dict(bus_i=bus_i, name=name, base_kV=float(kv), latitude=lat,
                longitude=lon, osm_id=osm_id, voltage_tag=str(int(kv)),
                Pd=0.0, Qd=0.0)


def _line(fbus, tbus, voltage, osm_id='way/1'):
    return dict(fbus=fbus, tbus=tbus, voltage_V=voltage, osm_id=osm_id,
                length_km=10.0, fbus_name='', tbus_name='')


EMPTY_XFMR = pd.DataFrame(columns=['fbus', 'tbus'])


def test_parse_volts_accepts_kv_and_volts_and_compound_tags():
    assert _parse_volts(132) == [132000.0]
    assert _parse_volts(132000) == [132000.0]
    assert _parse_volts('220000;132000') == [220000.0, 132000.0]
    for bad in (None, '', 'abc', 0, -1, float('nan')):
        assert _parse_volts(bad) == []


def test_untagged_substation_with_two_line_voltages_is_split():
    """The Mariakani case: 132 kV and 220 kV lines landing on one untagged bus."""
    buses = pd.DataFrame([_bus(1, 'Mariakani', 132000), _bus(2, 'A', 132000, lat=31.),
                          _bus(3, 'B', 220000, lat=32.)])
    lines = pd.DataFrame([_line(1, 2, '132000'), _line(1, 3, '220000', 'way/2')])
    buses, lines, xf, n = split_multi_voltage_buses(buses, lines, EMPTY_XFMR)
    assert n == 1
    # Bus 1 keeps the lowest tier; a new bus carries the 220 kV tier.
    low = buses.loc[buses.bus_i == 1].iloc[0]
    assert low.base_kV == 132000.0 and low['name'] == 'Mariakani_132kV'
    high = buses.loc[buses['name'] == 'Mariakani_220kV'].iloc[0]
    assert high.base_kV == 220000.0
    # The 220 kV line moved to the new tier; the 132 kV line did not move.
    assert lines.loc[lines.osm_id == 'way/2', 'fbus'].iloc[0] == high.bus_i
    assert lines.loc[lines.osm_id == 'way/1', 'fbus'].iloc[0] == 1
    # The tiers are joined by a transformer, not left shorted together.
    assert len(xf) == 1
    assert {xf.iloc[0].fbus, xf.iloc[0].tbus} == {1, high.bus_i}
    assert xf.iloc[0].x == XFMR_X and xf.iloc[0].r == XFMR_R


def test_three_voltage_substation_gets_two_cascaded_transformers():
    buses = pd.DataFrame([_bus(1, 'Mwavumbo', 132000), _bus(2, 'A', 132000, lat=31.),
                          _bus(3, 'B', 220000, lat=32.), _bus(4, 'C', 400000, lat=33.)])
    lines = pd.DataFrame([_line(1, 2, '132000'), _line(1, 3, '220000', 'way/2'),
                          _line(1, 4, '400000', 'way/3')])
    buses, lines, xf, n = split_multi_voltage_buses(buses, lines, EMPTY_XFMR)
    assert n == 1
    assert sorted(buses.loc[buses['name'].str.startswith('Mwavumbo'), 'base_kV']) == \
        [132000.0, 220000.0, 400000.0]
    assert len(xf) == 2  # 132-220 and 220-400, cascaded not star


def test_single_voltage_substation_is_untouched():
    buses = pd.DataFrame([_bus(1, 'Solo', 132000), _bus(2, 'A', 132000, lat=31.)])
    lines = pd.DataFrame([_line(1, 2, '132000')])
    out_b, out_l, xf, n = split_multi_voltage_buses(buses, lines, EMPTY_XFMR)
    assert n == 0 and len(out_b) == 2 and xf.empty
    assert out_l[['fbus', 'tbus']].values.tolist() == [[1, 2]]


def test_split_tier_carries_no_duplicate_load():
    """Demand stays on the original bus; a new tier must not duplicate it."""
    buses = pd.DataFrame([_bus(1, 'Load', 132000), _bus(2, 'A', 132000, lat=31.),
                          _bus(3, 'B', 220000, lat=32.)])
    buses.loc[buses.bus_i == 1, ['Pd', 'Qd']] = [50.0, 16.0]
    lines = pd.DataFrame([_line(1, 2, '132000'), _line(1, 3, '220000', 'way/2')])
    buses, _, _, _ = split_multi_voltage_buses(buses, lines, EMPTY_XFMR)
    assert buses.Pd.sum() == 50.0 and buses.Qd.sum() == 16.0


def test_endpoint_moves_to_the_matching_tier_of_its_own_substation():
    """The Isinya case: a 400 kV line landing on the 220 kV bus of a split pair."""
    buses = pd.DataFrame([_bus(1, 'Isinya_400kV', 400000, osm_id='way/9'),
                          _bus(2, 'Isinya_220kV', 220000, osm_id='way/9'),
                          _bus(3, 'Far', 400000, lat=33.)])
    lines = pd.DataFrame([_line(2, 3, '400000')])
    _, lines, moved = retier_endpoints(buses, lines)
    assert moved == 1
    assert lines.iloc[0].fbus == 1


def test_retier_leaves_an_endpoint_with_no_matching_tier_alone():
    buses = pd.DataFrame([_bus(1, 'Only_220kV', 220000, osm_id='way/9'),
                          _bus(2, 'Far', 400000, lat=33.)])
    lines = pd.DataFrame([_line(1, 2, '400000')])
    _, lines, moved = retier_endpoints(buses, lines)
    assert moved == 0 and lines.iloc[0].fbus == 1


def test_retier_groups_by_coordinates_when_osm_id_is_absent():
    buses = pd.DataFrame([_bus(1, 'X_400kV', 400000), _bus(2, 'X_220kV', 220000),
                          _bus(3, 'Far', 400000, lat=33.)])
    lines = pd.DataFrame([_line(2, 3, '400000')])
    _, lines, moved = retier_endpoints(buses, lines)
    assert moved == 1 and lines.iloc[0].fbus == 1


def test_retier_does_not_move_a_correctly_placed_endpoint():
    buses = pd.DataFrame([_bus(1, 'A_400kV', 400000, osm_id='way/9'),
                          _bus(2, 'A_220kV', 220000, osm_id='way/9'),
                          _bus(3, 'Far', 400000, lat=33.)])
    lines = pd.DataFrame([_line(1, 3, '400000')])
    _, lines, moved = retier_endpoints(buses, lines)
    assert moved == 0 and lines.iloc[0].fbus == 1


def test_empty_inputs_are_handled():
    empty = pd.DataFrame(columns=['fbus', 'tbus', 'voltage_V'])
    buses = pd.DataFrame([_bus(1, 'A', 132000)])
    assert split_multi_voltage_buses(buses, empty, EMPTY_XFMR)[3] == 0
    assert retier_endpoints(buses, empty)[2] == 0
