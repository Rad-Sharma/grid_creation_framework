"""A region anywhere in the world must fail loudly rather than quietly.

Each test here pins a failure that produced a plausible-looking case built on
the wrong numbers: a satellite cache describing different substations, a
coastal bus struck off the domestic total, a country name that matches nothing.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import ensure_feasibility as ef
import power_network as pn
from build_region_features import gee_cache_mismatch
from check_inputs import check_region_settings
from predict_bus_loads import detect_cross_border_buses
from test_power_network import independent_ac_check, two_bus

REPO = __import__("pathlib").Path(__file__).resolve().parents[1]


def cache(codes, names):
    return pd.DataFrame({"poc_code": codes, "name": names,
                         "ntl_mean": np.arange(len(codes), dtype=float)})


def buses(codes, names, lons=None, lats=None):
    frame = pd.DataFrame({"poc_code": codes, "bus_name": names})
    if lons is not None:
        frame["lon"], frame["lat"] = lons, lats
    return frame


# --------------------------------------------------------------------------
# Satellite feature cache
# --------------------------------------------------------------------------

def test_matching_satellite_cache_is_accepted():
    codes, names = ["BUS_1", "BUS_2"], ["Embakasi", "Juja Road"]
    assert gee_cache_mismatch(cache(codes, names), buses(codes, names)) is None


def test_cache_for_a_smaller_bus_set_is_refused():
    """A rebuild that adds buses must not reuse the old tessellation's features."""
    problem = gee_cache_mismatch(cache(["BUS_1"], ["Embakasi"]),
                                 buses(["BUS_1", "BUS_2"], ["Embakasi", "Juja Road"]))
    assert problem is not None
    assert "covers 1 bus(es)" in problem and "now has 2" in problem


def test_cache_describing_other_substations_is_refused():
    """Same count, same keys, different sites: the join key is only a position."""
    problem = gee_cache_mismatch(cache(["BUS_1", "BUS_2"], ["Embakasi", "Juja Road"]),
                                 buses(["BUS_1", "BUS_2"], ["Embakasi", "Rabai"]))
    assert problem is not None
    assert "BUS_2" in problem and "Rabai" in problem


def test_cache_without_a_name_column_is_refused():
    frame = cache(["BUS_1"], ["Embakasi"]).drop(columns=["name"])
    assert gee_cache_mismatch(frame, buses(["BUS_1"], ["Embakasi"])) is not None


# --------------------------------------------------------------------------
# Cross-border detection
# --------------------------------------------------------------------------

def test_coastal_buses_are_not_mistaken_for_cross_border():
    """The 110 m coastline is simplified by kilometres.

    Kilifi and Junju sit on the Kenyan coast, outside the polygon and inside the
    country. Tororo is in Uganda and Lemugur in Tanzania.
    """
    frame = buses(["BUS_%d" % i for i in range(1, 6)],
                  ["Kilifi", "Junju ward Substation 1", "Nairobi",
                   "Tororo Thermal Power Station", "Lemugur Substation"],
                  lons=[39.8467, 39.7829, 36.8219, 34.1179, 36.5771],
                  lats=[-3.6242, -3.7954, -1.2921, 0.6374, -3.4180])
    assert detect_cross_border_buses(frame, "Kenya") == {
        "Tororo Thermal Power Station", "Lemugur Substation"}


def test_an_island_country_has_no_cross_border_buses():
    """Puerto Rico has no land border, so nothing can be on the far side of one."""
    frame = pd.read_csv(REPO / "data" / "bus_data" / "features"
                        / "puerto_rico_features_with_industrial.csv")
    assert detect_cross_border_buses(frame, "Puerto Rico") == set()


def test_an_unknown_country_name_excludes_nothing():
    frame = buses(["BUS_1"], ["Somewhere"], lons=[0.0], lats=[0.0])
    assert detect_cross_border_buses(frame, "Not A Country") == set()


# --------------------------------------------------------------------------
# Table VI region settings
# --------------------------------------------------------------------------

def settings(name, iso, bbox):
    return SimpleNamespace(SHAPE_FILE_NAME=name, COUNTRY_ISO3=iso, REGION_BBOX=bbox)


@pytest.mark.parametrize("name,iso,bbox", [
    ("Kenya", "KEN", (33.89, -4.68, 41.86, 5.51)),
    ("Taiwan", "TWN", (119.3, 21.75, 122.11, 25.4)),
    ("Australia", "AUS", (112.9, -43.7, 153.7, -10.0)),
])
def test_settings_of_a_built_region_are_clean(name, iso, bbox):
    assert check_region_settings(REPO, settings(name, iso, bbox)) == []


def test_a_name_natural_earth_does_not_have_is_reported():
    problems = check_region_settings(REPO, settings("Ivory Coast", "CIV",
                                                    (-8.6, 4.3, -2.5, 10.8)))
    assert ("SETTING", ) == tuple({s for s, _ in problems})
    assert any("SHAPE_FILE_NAME" in m for _, m in problems)


def test_a_name_and_code_naming_different_countries_are_reported():
    problems = check_region_settings(REPO, settings("Kenya", "UGA",
                                                    (33.89, -4.68, 41.86, 5.51)))
    assert any(s == "SETTING" and "different countries" in m for s, m in problems)


def test_a_reversed_bounding_box_is_reported():
    problems = check_region_settings(REPO, settings("Kenya", "KEN",
                                                    (41.86, 5.51, 33.89, -4.68)))
    assert any(s == "SETTING" and "min_lon" in m for s, m in problems)


def test_a_box_that_misses_the_country_is_reported():
    problems = check_region_settings(REPO, settings("Kenya", "KEN",
                                                    (-10.0, 40.0, -5.0, 45.0)))
    assert any(s == "SETTING" and "does not overlap" in m for s, m in problems)


def test_overseas_territory_is_a_note_not_a_failure():
    """A study area smaller than the country is legitimate; say so, do not fail."""
    problems = check_region_settings(REPO, settings("France", "FRA",
                                                    (-5.2, 41.3, 9.6, 51.1)))
    assert problems and {s for s, _ in problems} == {"NOTE"}


# --------------------------------------------------------------------------
# Equation (5)
# --------------------------------------------------------------------------

def test_existing_bus_shunt_survives_equation_five():
    """(5) installs B_sh in addition to what the network already has."""
    data = two_bus()
    data["bus"][1, 5] = 5.
    result = ef.solve_ac_feasibility(ef.solve_dc_feasibility(data))
    assert result["bus"][1, 5] == pytest.approx(5., abs=1e-6)
    independent_ac_check(result)


def test_ac_keeps_the_verified_point_when_the_source_free_solve_fails(monkeypatch):
    """The converted schedule is feasible by construction, so refusing to
    re-optimise it is not evidence that the case is infeasible."""
    data = two_bus(qload=30.)
    data["gen"][0, 3:5] = 0
    dc = ef.solve_dc_feasibility(data)
    real_opf, width = pn._run_opf, len(dc["gen"])
    monkeypatch.setattr(pn, "_run_opf",
                        lambda case: None if len(case["gen"]) <= width else real_opf(case))
    result = ef.solve_ac_feasibility(dc)
    pn.check_solution(result, ac=True)
    independent_ac_check(result)
    assert abs(result["bus"][:, 5]).sum() > 20.


def test_equation_four_grows_only_the_branch_that_is_overloaded():
    """(4a) minimises the sum of squared slacks, so an unloaded branch gets none.

    Two generators feed one load down two branches of very different reactance,
    so almost all of the flow takes the low-reactance path and only that branch
    needs a higher rating.
    """
    data = two_bus(load=100., qload=0., rating=40.)
    data['bus'] = np.vstack([data['bus'],
                             [30, 1, 0, 0, 0, 0, 1, 1, 0, 138, 1, 1.1, .9]])
    data['branch'] = np.vstack([
        data['branch'],
        [10, 30, .01, .10, 0, 500., 0, 0, 0, 0, 1, -60, 60],
        [30, 20, .01, 10.0, 0, 500., 0, 0, 0, 0, 1, -60, 60]])
    before = data['branch'][:, 5].copy()
    result = ef.solve_dc_feasibility(data)
    grew = result['branch'][:, 5] - before
    assert grew[0] > 50., f'the loaded 40 MVA branch should carry most of 100 MW, got {grew[0]:.1f}'
    assert grew[1] == 0. and grew[2] == 0., f'unloaded branches grew by {grew[1:]}'
    pn.check_solution(result)


# --------------------------------------------------------------------------
# Equation (3) line length
# --------------------------------------------------------------------------

def line_tables():
    """Three lines: one honest, one far too short, one joining a co-located pair."""
    buses = pd.DataFrame({
        "bus_i": [1, 2, 3, 4],
        # 1 and 2 are about 111 km apart; 3 and 4 share a switchyard.
        "latitude":  [0.0, 1.0, 5.0, 5.0],
        "longitude": [0.0, 0.0, 5.0, 5.0],
    })
    lines = pd.DataFrame({
        "fbus": [1.0, 1.0, 3.0],
        "tbus": [2.0, 2.0, 4.0],
        "length_km": [200.0, 0.02, 0.0],
        "r": [2.0, 0.0002, 0.001],
        "x": [20.0, 0.002, 0.016],
        "b": [0.4, 0.00004, 0.0],
    })
    return lines, buses


def test_a_line_longer_than_its_span_is_left_alone():
    """Real conductors wander; only the impossible direction is corrected."""
    from resolve_unmatched import floor_lengths_at_bus_span
    lines, buses = line_tables()
    floor_lengths_at_bus_span(lines, buses)
    assert lines.loc[0, "length_km"] == 200.0
    assert lines.loc[0, "x"] == 20.0


def test_a_line_shorter_than_its_span_is_raised_and_rescaled():
    from resolve_unmatched import floor_lengths_at_bus_span
    lines, buses = line_tables()
    raised, worst, added = floor_lengths_at_bus_span(lines, buses)
    span = lines.loc[1, "length_km"]
    assert 110. < span < 112., f"expected about 111 km, got {span}"
    # R, X and B are linear in length under (3a)-(3c), so they scale with it.
    assert lines.loc[1, "x"] == pytest.approx(0.002 * span / 0.02, rel=1e-4)
    assert lines.loc[1, "r"] == pytest.approx(0.0002 * span / 0.02, rel=1e-4)
    assert raised == 1 and worst == pytest.approx(0.02 / span, rel=1e-4)
    assert added == pytest.approx(span - 0.02, rel=1e-4)


def test_a_transformer_between_co_located_buses_keeps_its_impedance():
    """Its impedance is equipment, not distance, and its buses are the same site."""
    from resolve_unmatched import floor_lengths_at_bus_span
    lines, buses = line_tables()
    floor_lengths_at_bus_span(lines, buses)
    assert lines.loc[2, "length_km"] == 0.0
    assert lines.loc[2, "x"] == 0.016
