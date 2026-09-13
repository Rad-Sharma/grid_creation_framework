from types import SimpleNamespace

import numpy as np
import pytest
from pypower.ext2int import ext2int
from pypower.makeYbus import makeYbus
from pypower.makeBdc import makeBdc
from pypower.case9 import case9

import power_network as pn
import ensure_feasibility as ef


def two_bus(base=100., load=50., qload=10., rating=200.):
    return dict(baseMVA=base,
                bus=np.array([[10, 3, 0, 0, 0, 0, 1, 1, 0, 138, 1, 1.1, .9],
                              [20, 1, load, qload, 0, 0, 1, 1, 0, 138, 1, 1.1, .9]], float),
                gen=np.array([[10, 0, 0, 100, -100, 1, base, 1, 200, 0]], float),
                branch=np.array([[10, 20, .01, .1, 0, rating, 0, 0, 0, 0, 1, -60, 60]], float),
                gencost=np.array([[2, 0, 0, 3, .01, 10, 0]], float))


def independent_ac_check(data):
    case = ext2int(dict(data, version='2'))
    ybus, yf, yt = makeYbus(case['baseMVA'], case['bus'], case['branch'])
    v = case['bus'][:, 7] * np.exp(1j * np.deg2rad(case['bus'][:, 8]))
    gen = np.zeros(len(v), complex)
    np.add.at(gen, case['gen'][:, 0].astype(int), case['gen'][:, 1] + 1j * case['gen'][:, 2])
    demand = case['bus'][:, 2] + 1j * case['bus'][:, 3]
    np.testing.assert_allclose(gen - demand, case['baseMVA'] * v * np.conj(ybus @ v), atol=1e-4)
    branch = case['branch']
    sf = case['baseMVA'] * v[branch[:, 0].astype(int)] * np.conj(yf @ v)
    st = case['baseMVA'] * v[branch[:, 1].astype(int)] * np.conj(yt @ v)
    limited = branch[:, 5] > 0
    assert (np.maximum(abs(sf), abs(st))[limited] <= branch[limited, 5] + 1e-4).all()


def test_no_flow_at_equal_voltage():
    data = two_bus()
    _, sf, st = pn.ac_flows(data, np.ones(2, complex))
    np.testing.assert_allclose(sf, 0, atol=1e-12)
    np.testing.assert_allclose(st, 0, atol=1e-12)


def test_admittance_matches_pypower_with_tap_shift_charging_and_shunts():
    data = two_bus()
    data['branch'][0, [4, 8, 9]] = [.02, 1.05, 12.]
    data['bus'][0, 4:6] = [2., 3.]
    expected = ext2int(dict(data, version='2'))
    reference = makeYbus(data['baseMVA'], expected['bus'], expected['branch'])
    for actual, ref in zip(pn.admittance_matrices(data), reference):
        np.testing.assert_allclose(actual, ref.toarray(), atol=1e-12)
    incidence, bf, shift = pn.dc_matrices(data)
    bbus, bref, pbus, pref = makeBdc(data['baseMVA'], expected['bus'], expected['branch'])
    np.testing.assert_allclose(bf, bref.toarray(), atol=1e-12)
    np.testing.assert_allclose(incidence.T @ bf, bbus.toarray(), atol=1e-12)
    np.testing.assert_allclose(shift, pref, atol=1e-12)
    np.testing.assert_allclose(incidence.T @ shift, pbus, atol=1e-12)


@pytest.mark.parametrize('base', [50., 100.])
def test_dc_balances_reference_bus_and_preserves_units(base):
    data = two_bus(base, rating=49.95)
    data['bus'][0, 2] = 20.
    result = ef.solve_dc_feasibility(data)
    assert result['gen'][0, 1] == pytest.approx(70., abs=1e-5)
    assert result['branch'][0, 5] >= 50. - 1e-5
    assert result['branch'][0, 6] == 0  # emergency limit remains unlimited
    assert result['bus'][1, 8] == pytest.approx(np.rad2deg(-50 / base * .1), abs=1e-5)
    assert data['gen'][0, 1] == 0  # solver does not mutate caller's input
    pn.check_solution(result)


def test_ac_requires_shunts_and_exports_them_in_mvar(tmp_path, monkeypatch):
    data = two_bus(qload=30.)
    data['gen'][0, 3:5] = 0  # no generator reactive support
    result = ef.solve_ac_feasibility(ef.solve_dc_feasibility(data))
    assert result['gen'][0, 1] > 50.  # losses supplied, including reference bus balance
    assert abs(result['bus'][:, 5]).sum() > 20.
    independent_ac_check(result)
    monkeypatch.setattr(ef, 'WORK_DIR', tmp_path)
    path = tmp_path / 'test_ac.m'
    ef.write_matpower(result, path, 'AC_feasible')
    reread = ef.load_matpower(path)
    independent_ac_check(reread)
    assert 'function mpc = test_ac' in path.read_text()


def test_reference_per_island():
    a = two_bus()
    b = two_bus(load=25.)
    b['bus'][:, 0] += 100
    b['gen'][:, 0] += 100
    b['branch'][:, :2] += 100
    for key in ('bus', 'gen', 'branch', 'gencost'):
        a[key] = np.vstack([a[key], b[key]])
    refs = pn.reference_buses(a)
    assert len(refs) == 2
    result = ef.solve_dc_feasibility(a)
    np.testing.assert_allclose(result['gen'][:, 1], [50, 25], atol=1e-4)
    pn.check_solution(result)


def test_insufficient_generation_is_rejected():
    data = two_bus()
    data['gen'][0, 8] = 10
    with pytest.raises((RuntimeError, ValueError)):
        ef.solve_dc_feasibility(data)


def test_invalid_bus_reference_is_rejected():
    data = two_bus()
    data['branch'][0, 1] = 999
    with pytest.raises(ValueError, match='unknown bus'):
        ef.solve_dc_feasibility(data)


def test_iteration_limit_is_failure(monkeypatch):
    def fail(fun, x0, **kwargs):
        return SimpleNamespace(x=x0, success=False, status=9, message='Iteration limit reached')
    monkeypatch.setattr(pn, 'minimize', fail)
    with pytest.raises(RuntimeError, match='status 9'):
        ef.solve_dc_feasibility(two_bus())


def test_false_success_is_rejected(monkeypatch):
    def fail(fun, x0, **kwargs):
        return SimpleNamespace(x=x0, success=True, status=0, message='Success')
    monkeypatch.setattr(pn, 'minimize', fail)
    with pytest.raises(RuntimeError, match='constraint-violating'):
        ef.solve_ac_feasibility(two_bus())


def test_infeasible_case_is_not_written(tmp_path):
    path = tmp_path / 'invalid.m'
    with pytest.raises(ValueError, match='balance'):
        ef.write_matpower(two_bus(), path, 'AC_feasible')
    assert not path.exists()


def test_standard_case9():
    data = case9()
    result = ef.solve_ac_feasibility(ef.solve_dc_feasibility(data))
    independent_ac_check(result)


def test_matpower_parser_handles_semicolons():
    matrix = ef._parse_matrix('mpc.gen = [1 2; 3 4; % comment\n 5 6;];', 'gen')
    np.testing.assert_array_equal(matrix, [[1, 2], [3, 4], [5, 6]])
