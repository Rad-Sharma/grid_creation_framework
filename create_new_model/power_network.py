"""MATPOWER network equations and checked DC/AC feasibility relaxations.

Network admittances and solver power variables use per unit. CSV/MATPOWER
dispatch, demand, ratings and bus shunts use MW/MVAr on the case baseMVA.
SLSQP is intended for modest cases; large regional cases need a sparse OPF solver.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

POWER_TOL_MVA = 1e-4
BOUND_TOL = 1e-7


def validate_case(data):
    base = data.get("baseMVA", 100.0)
    if not np.isfinite(base) or base <= 0:
        raise ValueError("baseMVA must be finite and positive")
    widths = {"bus": 13, "gen": 10, "branch": 13, "gencost": 7}
    for key, width in widths.items():
        a = data[key]
        if a.ndim != 2 or a.shape[1] < width or not np.isfinite(a).all():
            raise ValueError(f"Invalid {key} matrix: expected finite values and {width}+ columns")
    bus, gen, branch, cost = (data[k] for k in widths)
    ids = bus[:, 0]
    if not len(ids) or len(set(ids)) != len(ids) or (ids != ids.astype(int)).any():
        raise ValueError("Bus IDs must be unique integers")
    if not np.isin(gen[:, 0], ids).all() or not np.isin(branch[:, :2], ids).all():
        raise ValueError("Generator or branch references an unknown bus")
    if len(cost) != len(gen) or (cost[:, 0] != 2).any() or (cost[:, 3] != 3).any():
        raise ValueError("Expected one quadratic polynomial cost row per generator")
    if (gen[:, 9] > gen[:, 8]).any() or (gen[:, 4] > gen[:, 3]).any():
        raise ValueError("Generator minimum exceeds maximum")
    if (bus[:, 12] <= 0).any() or (bus[:, 12] > bus[:, 11]).any():
        raise ValueError("Invalid bus voltage bounds")
    if (branch[:, 5:8] < 0).any() or (branch[:, 8] < 0).any():
        raise ValueError("Negative branch rating or tap ratio")
    active = branch[:, 10] != 0
    if (np.abs(branch[active, 2] + 1j * branch[active, 3]) < 1e-12).any():
        raise ValueError("Zero-impedance active branch")
    lo, hi = angle_bounds(branch)
    if (lo > hi).any():
        raise ValueError("Invalid branch angle bounds")


def indices(data):
    lookup = {int(v): i for i, v in enumerate(data["bus"][:, 0])}
    branch = data["branch"]
    return (np.array([lookup[int(v)] for v in branch[:, 0]], dtype=int),
            np.array([lookup[int(v)] for v in branch[:, 1]], dtype=int),
            np.array([lookup[int(v)] for v in data["gen"][:, 0]], dtype=int))


def reference_buses(data):
    """One reference angle per connected component, including isolated buses."""
    f, t, _ = indices(data)
    active = data["branch"][:, 10] != 0
    n = len(data["bus"])
    graph = csr_matrix((np.ones(active.sum()), (f[active], t[active])), shape=(n, n))
    count, labels = connected_components(graph, directed=False)
    refs = []
    generating_ids = data["gen"][data["gen"][:, 7] > 0, 0]
    for label in range(count):
        members = np.flatnonzero(labels == label)
        generators = members[np.isin(data["bus"][members, 0], generating_ids)]
        candidates = generators if len(generators) else members
        preferred = candidates[data["bus"][candidates, 1] == 3]
        refs.append(preferred[0] if len(preferred) else candidates[0])
    return np.array(refs, dtype=int)


def angle_bounds(branch):
    # MATPOWER: both zero means unconstrained; +/-360 also disables that bound.
    lo, hi = np.deg2rad(branch[:, 11]), np.deg2rad(branch[:, 12])
    both_zero = (branch[:, 11] == 0) & (branch[:, 12] == 0)
    lo[both_zero | (branch[:, 11] <= -360)] = -np.inf
    hi[both_zero | (branch[:, 12] >= 360)] = np.inf
    return lo, hi


def angle_margins(data, theta):
    f, t, _ = indices(data)
    lo, hi = angle_bounds(data["branch"])
    active = data["branch"][:, 10] != 0
    delta = theta[f] - theta[t]
    lower = active & np.isfinite(lo)
    upper = active & np.isfinite(hi)
    return np.r_[delta[lower] - lo[lower], hi[upper] - delta[upper]]


def admittance_matrices(data):
    """Return Ybus, Yfrom, Yto, including taps, phase shifts and existing shunts."""
    f, t, _ = indices(data)
    branch = data["branch"]
    n, m = len(data["bus"]), len(branch)
    yf = np.zeros((m, n), complex)
    yt = np.zeros((m, n), complex)
    ybus = np.zeros((n, n), complex)
    for k, row in enumerate(branch):
        if row[10] == 0:
            continue
        y = 1 / complex(row[2], row[3])
        tap = (row[8] or 1.0) * np.exp(1j * np.deg2rad(row[9]))
        yff = (y + .5j * row[4]) / abs(tap)**2
        ytt = y + .5j * row[4]
        yft, ytf = -y / np.conj(tap), -y / tap
        yf[k, f[k]] += yff
        yf[k, t[k]] += yft
        yt[k, f[k]] += ytf
        yt[k, t[k]] += ytt
        ybus[f[k]] += yf[k]
        ybus[t[k]] += yt[k]
    shunts = (data["bus"][:, 4] + 1j * data["bus"][:, 5]) / data["baseMVA"]
    ybus += np.diag(shunts)
    return ybus, yf, yt


def ac_flows(data, voltage):
    ybus, yf, yt = admittance_matrices(data)
    f, t, _ = indices(data)
    base = data["baseMVA"]
    return (base * voltage * np.conj(ybus @ voltage),
            base * voltage[f] * np.conj(yf @ voltage),
            base * voltage[t] * np.conj(yt @ voltage))


def dc_matrices(data):
    f, t, _ = indices(data)
    branch = data["branch"]
    n, m = len(data["bus"]), len(branch)
    incidence = np.zeros((m, n))
    incidence[np.arange(m), f] += 1
    incidence[np.arange(m), t] -= 1
    active = branch[:, 10] != 0
    if (abs(branch[active, 3]) < 1e-12).any():
        raise ValueError("DC model requires nonzero active-branch reactance")
    b = np.zeros(m)
    b[active] = 1 / (branch[active, 3] * np.where(branch[active, 8] == 0, 1, branch[active, 8]))
    bf = b[:, None] * incidence
    shift = -b * np.deg2rad(branch[:, 9])
    return incidence, bf, shift


def generation(data, column):
    _, _, g = indices(data)
    return np.bincount(g, weights=data["gen"][:, column] * (data["gen"][:, 7] > 0),
                       minlength=len(data["bus"]))


def check_solution(data, ac=False):
    """Validate the actual returned/exportable case, including all bus balances."""
    validate_case(data)
    bus, gen, branch = (data[k] for k in ("bus", "gen", "branch"))
    theta = np.deg2rad(bus[:, 8])
    active_g = gen[:, 7] > 0
    if ((gen[active_g, 1] < gen[active_g, 9] - POWER_TOL_MVA).any()
            or (gen[active_g, 1] > gen[active_g, 8] + POWER_TOL_MVA).any()):
        raise ValueError("Generator active-power bound violated")
    if (np.abs(gen[~active_g, 1:3]) > POWER_TOL_MVA).any():
        raise ValueError("Offline generator has nonzero dispatch")
    if ac:
        if ((gen[active_g, 2] < gen[active_g, 4] - POWER_TOL_MVA).any()
                or (gen[active_g, 2] > gen[active_g, 3] + POWER_TOL_MVA).any()):
            raise ValueError("Generator reactive-power bound violated")
        if ((bus[:, 7] < bus[:, 12] - BOUND_TOL).any()
                or (bus[:, 7] > bus[:, 11] + BOUND_TOL).any()):
            raise ValueError("Voltage bound violated")
        s, sf, st = ac_flows(data, bus[:, 7] * np.exp(1j * theta))
        residual = generation(data, 1) + 1j * generation(data, 2) - bus[:, 2] - 1j * bus[:, 3] - s
        flow = np.maximum(abs(sf), abs(st))
    else:
        incidence, bf, shift = dc_matrices(data)
        flow = data["baseMVA"] * (bf @ theta + shift)
        residual = generation(data, 1) - bus[:, 2] - bus[:, 4] - incidence.T @ flow
        flow = abs(flow)
    if np.max(abs(residual), initial=0) > POWER_TOL_MVA:
        raise ValueError(f"Power-balance residual {np.max(abs(residual)):.6g} MVA exceeds tolerance")
    limited = (branch[:, 10] != 0) & (branch[:, 5] > 0)
    if (flow[limited] > branch[limited, 5] + POWER_TOL_MVA).any():
        raise ValueError("Branch thermal limit violated")
    if (angle_margins(data, theta) < -BOUND_TOL).any():
        raise ValueError("Branch angle limit violated")


def require_convergence(result, constraints, bounds):
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"OPF failed (status {result.status}): {result.message}")
    x = result.x
    for constraint in constraints:
        values = np.asarray(constraint["fun"](x))
        if not np.isfinite(values).all():
            raise RuntimeError("OPF returned nonfinite constraint values")
        error = abs(values) if constraint["type"] == "eq" else -values
        if np.max(error, initial=0) > 1e-7:
            raise RuntimeError("OPF returned a constraint-violating solution")
    lo, hi = np.asarray(bounds).T
    if (x < lo - BOUND_TOL).any() or (x > hi + BOUND_TOL).any():
        raise RuntimeError("OPF returned a bound-violating solution")


def solve_feasibility(data, ac=False, penalty=1e4, max_bsh=5.0):
    validate_case(data)
    data = dict(data)
    for key in ("bus", "gen", "branch", "gencost"):
        # MATPOWER fixtures can use integer arrays; never truncate solved dispatch.
        data[key] = np.array(data[key], dtype=float, copy=True)
    bus, gen, branch = (data[k] for k in ("bus", "gen", "branch"))
    base = data["baseMVA"]
    n, ng, nb = len(bus), len(gen), len(branch)
    f, t, g = indices(data)
    cg = np.zeros((n, ng))
    cg[g, np.arange(ng)] = 1
    active = gen[:, 7] > 0
    refs = reference_buses(data)
    bus[:, 1] = np.where(np.isin(np.arange(n), g[active]), 2, 1)
    bus[refs, 1] = 3
    theta_bounds = [(-np.pi, np.pi)] * n
    for ref in refs:
        theta_bounds[ref] = (0., 0.)
    pb = [(row[9] / base, row[8] / base) if on else (0., 0.) for row, on in zip(gen, active)]
    cost = data["gencost"][:, 4:7]

    def generation_cost(p):
        mw = p * base
        return np.sum((cost[:, 0] * mw**2 + cost[:, 1] * mw + cost[:, 2])[active])

    limited = (branch[:, 10] != 0) & (branch[:, 5] > 0)
    if ac:
        # Paper equation (5a): quadratic penalty on additional shunt susceptance.
        bounds = pb + [(row[4] / base, row[3] / base) if on else (0., 0.) for row, on in zip(gen, active)]
        bounds += list(zip(bus[:, 12], bus[:, 11])) + theta_bounds + [(-max_bsh, max_bsh)] * n
        ybus, yf, yt = admittance_matrices(data)

        def unpack(x):
            p, q = x[:ng], x[ng:2*ng]
            vm, th = x[2*ng:2*ng+n], x[2*ng+n:2*ng+2*n]
            sh = x[2*ng+2*n:2*ng+3*n]
            return p, q, vm, th, sh

        def balance(x):
            p, q, vm, th, sh = unpack(x)
            v = vm * np.exp(1j * th)
            s = v * np.conj(ybus @ v)
            mismatch = cg @ p + 1j * (cg @ q) - (bus[:, 2] + 1j * bus[:, 3]) / base - s + 1j * sh * vm**2
            return np.r_[mismatch.real, mismatch.imag]

        def margins(x):
            _, _, vm, th, sh = unpack(x)
            v = vm * np.exp(1j * th)
            sf, st = v[f] * np.conj(yf @ v), v[t] * np.conj(yt @ v)
            rate = branch[limited, 5] / base
            return np.r_[rate - abs(sf[limited]), rate - abs(st[limited]), angle_margins(data, th)]

        def objective(x):
            p, _, _, _, sh = unpack(x)
            return generation_cost(p) / penalty + np.dot(sh, sh)

        x0 = np.r_[gen[:, 1] / base, gen[:, 2] / base, bus[:, 7], np.deg2rad(bus[:, 8]), np.zeros(n)]
    else:
        incidence, bf, shift = dc_matrices(data)
        bounds = pb + theta_bounds + [(0., np.inf) if on else (0., 0.) for on in limited]

        def balance(x):
            return cg @ x[:ng] - (bus[:, 2] + bus[:, 4]) / base - incidence.T @ (bf @ x[ng:ng+n] + shift)

        def margins(x):
            th, slack = x[ng:ng+n], x[ng+n:]
            flow = (bf @ th + shift)[limited]
            rating = branch[limited, 5] / base + slack[limited]
            return np.r_[rating - flow, rating + flow, angle_margins(data, th)]

        def objective(x):
            return generation_cost(x[:ng]) / (penalty * base**2) + np.sum(x[ng+n:]**2)

        x0 = np.r_[gen[:, 1] / base, np.deg2rad(bus[:, 8]), np.zeros(nb)]

    lo, hi = np.asarray(bounds).T
    x0 = np.clip(x0, lo, hi)
    constraints = [{"type": "eq", "fun": balance}]
    if len(margins(x0)):
        constraints.append({"type": "ineq", "fun": margins})
    res = minimize(objective, x0, method="SLSQP", bounds=bounds,
                   constraints=constraints, options={"maxiter": 3000, "ftol": 1e-11})
    require_convergence(res, constraints, bounds)
    if ac:
        p, q, vm, th, sh = unpack(res.x)
        gen[:, 1], gen[:, 2] = p * base, q * base
        bus[:, 7], bus[:, 8] = vm, np.rad2deg(th)
        bus[:, 5] += sh * base  # MATPOWER Bs is MVAr at Vm=1, not per unit.
    else:
        gen[:, 1], gen[:, 2] = res.x[:ng] * base, 0.
        bus[:, 8] = np.rad2deg(res.x[ng:ng+n])
        expansion = np.maximum(res.x[ng+n:] * base, 0.)
        for col in (5, 6, 7):
            # Preserve zero (unlimited) emergency ratings.
            update = limited & (branch[:, col] > 0)
            branch[update, col] += expansion[update]
    check_solution(data, ac=ac)
    return data
