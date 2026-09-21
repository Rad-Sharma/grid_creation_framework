"""Regressions pinning the implementation to the manuscript's methodology.

Reference: "A Framework for the Creation of Highly Realistic International
Synthetic Electric Grid Models". Section/equation/table numbers below refer to
that manuscript. These tests guard the places where later safeguard work has
previously drifted away from the published method.
"""
import re
import runpy
import shutil
from pathlib import Path

import pandas as pd
import pytest
from shapely.geometry import Polygon

import config
from create_cost_curves import (CATEGORY_TO_FUEL, FUEL_COSTS_MBTU, SCALING_BASIS,
                                _coeffs, compute_polynomial_coeffs, fuel_cost_scale)
from grid_topology import network

ROOT = Path(__file__).resolve().parents[1]


def _case_counts(path):
    text = path.read_text(encoding='utf-8', errors='replace')
    counts = {}
    for name in ('bus', 'gen', 'branch'):
        body = re.search(rf'mpc\.{name}\s*=\s*\[(.*?)\];', text, re.S).group(1)
        counts[name] = len([r for r in body.strip().splitlines()
                            if r.strip() and not r.strip().startswith('%')])
    return counts


# ---------------------------------------------------------------- Table VII

def test_kenya_case_reproduces_table_vii():
    """Table VII reports Kenya as 92 buses, 75 generators, 143 lines."""
    case = ROOT / 'data/working_files/kenya/kenya_synthetic_grid.m'
    if not case.exists():
        pytest.skip('Kenya case not present in this checkout')
    counts = _case_counts(case)
    assert (counts['bus'], counts['gen'], counts['branch']) == (92, 75, 143)


def test_kenya_branch_count_requires_the_voltage_tier_transformers():
    """143 branches = 134 resolved lines + 9 intra-substation transformers.

    Disabling INFER_TRANSFORMERS_FROM_VOLTAGE silently drops the 9 transformers
    and breaks Table VII reproduction, so the two files are pinned together.
    """
    work = ROOT / 'data/working_files/kenya'
    if not (work / 'transformer_pairs_kenya.csv').exists():
        pytest.skip('Kenya working files not present in this checkout')
    lines = len(pd.read_csv(work / 'matpower_kenya_lines_resolved.csv'))
    transformers = len(pd.read_csv(work / 'transformer_pairs_kenya.csv'))
    assert (lines, transformers) == (134, 9)
    assert lines + transformers == 143


# ------------------------------------------------- Sec. III.B substations

def test_compound_voltage_substation_is_split_and_transformed(tmp_path, monkeypatch):
    """Sec. III.B: a compound voltage tag yields one bus per level, joined.

    Splitting without the transformer strands the upper tier as an island.
    """
    monkeypatch.setattr(config, 'region', 'test')
    script = tmp_path / 'create_new_model/create_buses.py'
    script.parent.mkdir()
    shutil.copyfile(ROOT / 'create_new_model/create_buses.py', script)
    inputs = tmp_path / 'data/OSM_extracts'
    inputs.mkdir(parents=True)
    (inputs / 'export_test_buses.geojson').write_text(
        '{"type":"FeatureCollection","features":['
        '{"type":"Feature","properties":{"name":"Test","voltage":"330000;161000"},'
        '"geometry":{"type":"Point","coordinates":[-95,30]}}]}')
    runpy.run_path(str(script), run_name='__main__')
    work = tmp_path / 'data/working_files/test'
    buses = pd.read_csv(work / 'matpower_test_buses.csv')
    pairs = pd.read_csv(work / 'transformer_pairs_test.csv')
    assert sorted(buses.base_kV) == [161000.0, 330000.0]
    assert len(pairs) == 1
    assert {pairs.iloc[0].fbus, pairs.iloc[0].tbus} == set(buses.bus_i)


# ------------------------------------------------------- Tables IV and V

@pytest.mark.parametrize('category,pmax,expected', [
    ('steam_coal', 100., (1.628 * 100 + 7.44, 0.00284 * 100 + 11.94, 5.48e-5 * 100 - 9.57e-5)),
    ('gt_gas', 50., (7.913 * 50 - 23.26, 0.0366 * 50 + 1.27, 9.45e-4 * 50 - 0.0184)),
    ('cc_coal', 200., (992.0, 9.0, 0.01)),
    ('hydro', 80., (1.117 * 80 - 0.34, 14.0, 1.54e-5 * 80 - 8.0e-6)),
])
def test_table_v_coefficients_are_published_values(category, pmax, expected):
    assert _coeffs(category, pmax) == pytest.approx(expected)


def test_us_basis_is_the_default_and_leaves_table_v_unscaled():
    assert getattr(config, 'FUEL_COST_BASIS', 'us') == 'us'
    a, b, c = _coeffs('steam_coal', 100.)
    c2, c1, c0 = compute_polynomial_coeffs(100., 'steam_coal')
    assert (c0, c1, c2) == (round(a, 4), round(b, 6), round(c, 6))


@pytest.mark.parametrize('category,us,world', [
    ('steam_coal', 1.78, 2.90),
    ('cc_gas', 2.68, 4.88),
    ('gt_petroleum', 9.25, 10.50),
    ('steam_wood', 3.72, 4.20),
    ('cc_petcoke', 2.56, 3.10),
    ('ic_landfill', 6.18, 5.50),
])
def test_global_basis_uses_table_iv_price_ratio(category, us, world):
    assert fuel_cost_scale(category, 'global') == pytest.approx(world / us)


def test_fuels_absent_from_table_iv_are_never_rescaled():
    """[48] gave hydro a borrowed CCGT curve and zeroed the non-dispatchables."""
    for category in ('hydro', 'renewable', 'free_steam', 'gt_biomass'):
        assert fuel_cost_scale(category, 'global') == 1.0


def test_unknown_fuel_cost_basis_is_rejected():
    with pytest.raises(ValueError, match="us.*global"):
        fuel_cost_scale('steam_coal', 'imaginary')


# ------------------------------------------- Sec. IV.B.1 Step 3 voltages

def _way(number, nodes, points, **tags):
    return dict(type='way', id=number, nodes=nodes,
                tags={'power': 'line', **tags},
                geometry=[dict(lon=x, lat=y) for x, y in points])


def _two_line_region(second_tags):
    boundary = Polygon([(-101, 30), (-99, 30), (-99, 32), (-101, 32)])
    tagged = _way(1, [1, 2], [(-100.8, 31.0), (-100.2, 31.0)], voltage='138000')
    other = _way(2, [3, 4], [(-100.8, 31.5), (-100.2, 31.5)], **second_tags)
    return boundary, [tagged, other]


def test_untagged_line_is_retained_at_the_regional_modal_voltage():
    """Sec. IV.B.1 Step 3: an absent tag is missing attribution, not evidence."""
    boundary, elements = _two_line_region({})
    _, branches, audit = network(elements, boundary)
    assert {b['osm_id'] for b in branches} == {'way/1', 'way/2'}
    inferred = [b for b in branches if b['osm_id'] == 'way/2']
    assert all(b['voltage_kV'] == 138.0 for b in inferred)
    assert audit.reason.str.contains('no voltage tag').any()


def test_known_subtransmission_voltage_is_removed():
    """Sec. IV.B.1 Step 3: a known voltage below the threshold is dropped."""
    boundary, elements = _two_line_region({'voltage': '33000'})
    _, branches, audit = network(elements, boundary)
    assert {b['osm_id'] for b in branches} == {'way/1'}
    assert audit.reason.str.contains('below the 69 kV').any()


def test_region_minimum_voltage_is_honoured_not_hardcoded():
    """Table VI makes MIN_TRANSMISSION_VOLTAGE a user input, not a constant."""
    boundary, elements = _two_line_region({'voltage': '115000'})
    _, kept, _ = network(elements, boundary, min_kv=69.)
    assert {b['osm_id'] for b in kept} == {'way/1', 'way/2'}
    _, filtered, audit = network(elements, boundary, min_kv=138.)
    assert {b['osm_id'] for b in filtered} == {'way/1'}
    assert audit.reason.str.contains('below the 138 kV').any()


# ------------------------------------------- Reference [48] derivation

def test_cost_curves_are_fuel_price_times_heat_rate():
    """[48] builds $/h as heat_rate x P x fuel_price; its LMS100 gas sample.

    This identity is what makes the Table IV global rescaling exact rather than
    an assumption: the least-squares fit for (a, b, c) is linear in that $/h
    vector, so replacing the fuel price scales all three coefficients equally.
    """
    gas_price = FUEL_COSTS_MBTU["natural_gas"][0]
    assert gas_price == 2.68
    samples = [(26634, 12689, 905.7297), (42614.4, 10421, 1190.147),
               (63921.6, 8940, 1531.51), (85228.8, 8177, 1867.735),
               (106536, 7845, 2239.877)]
    for kw, btu_per_kwh, stated in samples:
        assert kw * btu_per_kwh / 1e6 * gas_price == pytest.approx(stated, abs=5e-4)


def test_table_v_follows_from_reference_48_ranges():
    """Sec. V.D linearises [48]'s per-cluster coefficient ranges two-point."""
    # [48] summary table: capacity 20-1425 MW, B 12-16 $/MWh for steam coal.
    slope = (16 - 12) / (1425 - 20)
    intercept = 12 - slope * 20
    _, b, _ = _coeffs('steam_coal', 0.)
    b_slope = (_coeffs('steam_coal', 1.)[1] - b)
    assert b_slope == pytest.approx(slope, rel=1e-2)
    assert b == pytest.approx(intercept, rel=1e-3)


def test_every_table_v_category_declares_a_scaling_basis():
    for category in CATEGORY_TO_FUEL:
        assert SCALING_BASIS[category] in ('heat_rate', 'market_bid')
    assert SCALING_BASIS['hydro'] == 'borrowed'


def test_hydro_is_not_rescaled_despite_being_a_borrowed_gas_curve():
    """[48] copied hydro's curve from CCGT samples, so its b=14 is a gas price.

    Scaling it by the gas ratio would turn that placeholder into a fuel-price
    adjustment for a generator that burns nothing.
    """
    assert fuel_cost_scale('hydro', 'global') == 1.0
    assert fuel_cost_scale('cc_gas', 'global') == pytest.approx(4.88 / 2.68)


def test_internal_combustion_scaling_is_flagged_as_market_derived():
    """[48] used ERCOT bids for IC units, not heat rates, so scaling is approximate."""
    for category in ('ic_gas', 'ic_petroleum', 'ic_landfill'):
        assert SCALING_BASIS[category] == 'market_bid'
        assert fuel_cost_scale(category, 'global') != 1.0
