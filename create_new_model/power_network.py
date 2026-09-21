"""MATPOWER network equations and checked DC/AC feasibility relaxations.

Network admittances and solver power variables use per unit. CSV/MATPOWER
dispatch, demand, ratings and bus shunts use MW/MVAr on the case baseMVA.

Manuscript equation (4) is a convex QP and is solved as one: HiGHS settles
feasibility exactly and warm-starts a trust-region solve on analytic derivatives.
Equation (5) is non-convex, so it is solved through PyPOWER's own AC OPF with a
reactive source at each candidate bus, then converted to the shunts the equation
asks for and re-verified against the exact nonlinear model.
"""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp, minimize
from scipy.sparse import csr_matrix, hstack, vstack
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
    """Return Ybus, Yfrom, Yto, including taps, phase shifts and existing shunts.

    Sparse: a dense Ybus is n^2 entries, which at Australia's 2401 buses is
    already 0.09 GB for Ybus and 0.19 GB for Yf and Yt together, and the network
    only ever has a handful of entries per row.
    """
    f, t, _ = indices(data)
    branch = data["branch"]
    n, m = len(data["bus"]), len(branch)
    active = branch[:, 10] != 0
    rows = np.flatnonzero(active)
    y = 1 / (branch[rows, 2] + 1j * branch[rows, 3])
    tap = np.where(branch[rows, 8] == 0, 1.0, branch[rows, 8]) \
        * np.exp(1j * np.deg2rad(branch[rows, 9]))
    charging = .5j * branch[rows, 4]
    yff = (y + charging) / np.abs(tap)**2
    ytt = y + charging
    yft, ytf = -y / np.conj(tap), -y / tap
    shape = (m, n)
    yf = csr_matrix((np.r_[yff, yft], (np.r_[rows, rows], np.r_[f[rows], t[rows]])), shape=shape)
    yt = csr_matrix((np.r_[ytf, ytt], (np.r_[rows, rows], np.r_[f[rows], t[rows]])), shape=shape)
    shunts = (data["bus"][:, 4] + 1j * data["bus"][:, 5]) / data["baseMVA"]
    # Ybus accumulates each branch's from-row at its from-bus and its to-row at
    # its to-bus, which is what this incidence-style product does in one go.
    branch_side = csr_matrix((np.ones(2 * len(rows)),
                              (np.r_[f[rows], t[rows]], np.r_[rows, rows + m])),
                             shape=(n, 2 * m))
    ybus = branch_side @ vstack([yf, yt]).tocsr() \
        + csr_matrix((shunts, (np.arange(n), np.arange(n))), shape=(n, n))
    return ybus.tocsr(), yf, yt


def ac_flows(data, voltage):
    ybus, yf, yt = admittance_matrices(data)
    f, t, _ = indices(data)
    base = data["baseMVA"]
    return (base * voltage * np.conj(ybus @ voltage),
            base * voltage[f] * np.conj(yf @ voltage),
            base * voltage[t] * np.conj(yt @ voltage))


def dc_matrices(data):
    """Return the branch incidence, Bf and phase-shift injection, all sparse.

    Two non-zeros per branch row, so the dense form wasted m*n entries and made
    the incidence.T @ Bf product in equation (4) an O(n^2 m) dense matmul.
    """
    f, t, _ = indices(data)
    branch = data["branch"]
    n, m = len(data["bus"]), len(branch)
    span = np.arange(m)
    incidence = csr_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[span, span], np.r_[f, t])),
                           shape=(m, n))
    active = branch[:, 10] != 0
    if (abs(branch[active, 3]) < 1e-12).any():
        raise ValueError("DC model requires nonzero active-branch reactance")
    b = np.zeros(m)
    b[active] = 1 / (branch[active, 3] * np.where(branch[active, 8] == 0, 1, branch[active, 8]))
    bf = csr_matrix((b, (span, span)), shape=(m, m)) @ incidence
    shift = -b * np.deg2rad(branch[:, 9])
    return incidence, bf.tocsr(), shift


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


def _injection_matrix(data):
    """n x ng matrix placing each online generator's output at its bus."""
    _, _, g = indices(data)
    gen = data["gen"]
    return csr_matrix(((gen[:, 7] > 0).astype(float), (g, np.arange(len(gen)))),
                      shape=(len(data["bus"]), len(gen)))


def _assign_bus_types(data):
    """Mark generator buses PV and one reference bus per island REF."""
    bus, gen = data["bus"], data["gen"]
    _, _, g = indices(data)
    refs = reference_buses(data)
    bus[:, 1] = np.where(np.isin(np.arange(len(bus)), g[gen[:, 7] > 0]), 2, 1)
    bus[refs, 1] = 3
    return refs


def _dc_program(data):
    """Linear system behind manuscript equation (4): A, bounds and variable layout.

    Variables are [Pg (per unit), theta (rad), extra line capacity (per unit)].
    """
    bus, gen, branch = (data[k] for k in ("bus", "gen", "branch"))
    base = data["baseMVA"]
    n, ng, nb = len(bus), len(gen), len(branch)
    refs = _assign_bus_types(data)
    active = gen[:, 7] > 0
    incidence, bf, shift = dc_matrices(data)
    limited = (branch[:, 10] != 0) & (branch[:, 5] > 0)
    nl = int(limited.sum())

    # (4b) bus active-power balance; the DC model carries bus Gs as extra demand.
    rows = [hstack([_injection_matrix(data), -(incidence.T @ bf), csr_matrix((n, nb))])]
    demand = (bus[:, 2] + bus[:, 4]) / base + incidence.T @ shift
    lower, upper = [demand], [demand]

    # (4e) thermal limits, each relaxed by its own non-negative slack.
    rate = branch[limited, 5] / base
    picker = csr_matrix((np.ones(nl), (np.arange(nl), np.flatnonzero(limited))),
                        shape=(nl, nb))
    flow = bf[limited]
    for direction in (1., -1.):
        rows.append(hstack([csr_matrix((nl, ng)), direction * flow, -picker]))
        lower.append(np.full(nl, -np.inf))
        upper.append(rate - direction * shift[limited])

    # Branch angle limits are not in (4), but check_solution enforces them, so a
    # relaxation that ignored them could not be exported.
    lo, hi = angle_bounds(branch)
    spread = (branch[:, 10] != 0) & (np.isfinite(lo) | np.isfinite(hi))
    if spread.any():
        width = int(spread.sum())
        rows.append(hstack([csr_matrix((width, ng)), incidence[spread],
                            csr_matrix((width, nb))]))
        lower.append(np.where(np.isfinite(lo[spread]), lo[spread], -np.inf))
        upper.append(np.where(np.isfinite(hi[spread]), hi[spread], np.inf))

    # (4c) generation limits; offline units are pinned at zero. Bus angles are
    # unbounded, as in MATPOWER: only the differences of (5j) are limited, and a
    # long radial chain at 60 degrees a branch passes pi within a few hops.
    xmin = np.r_[np.where(active, gen[:, 9] / base, 0.), np.full(n, -np.inf), np.zeros(nb)]
    xmax = np.r_[np.where(active, gen[:, 8] / base, 0.), np.full(n, np.inf),
                 np.where(limited, np.inf, 0.)]   # (4f)
    xmin[ng + refs] = xmax[ng + refs] = 0.
    return (vstack(rows).tocsr(), np.concatenate(lower), np.concatenate(upper),
            xmin, xmax, limited)


def _diagnose_dc_infeasibility(data):
    """Name the constraint that makes equation (4) unsolvable, not just the fact."""
    bus, gen, branch = (data[k] for k in ("bus", "gen", "branch"))
    online = gen[:, 7] > 0
    labels = island_labels_from_case(data)
    short = []
    for label in np.unique(labels):
        members = labels == label
        _, _, g = indices(data)
        here = members[g] & online
        demand = float((bus[members, 2] + bus[members, 4]).sum())
        capacity = float(gen[here, 8].sum())
        if capacity < demand - POWER_TOL_MVA:
            short.append((int(members.sum()), demand, capacity))
    if short:
        worst = ", ".join(f"{n} bus(es) needing {d:.1f} MW with {c:.1f} MW of capacity"
                          for n, d, c in short[:3])
        return (f"{len(short)} part(s) of the network cannot meet their own demand: "
                f"{worst}. Add generation there, or connect them to the rest of the grid.")

    relaxed = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in data.items()}
    relaxed["branch"][:, 11], relaxed["branch"][:, 12] = -360., 360.
    matrix, lower, upper, xmin, xmax, _ = _dc_program(relaxed)
    probe = milp(c=np.r_[np.zeros(len(xmin) - len(branch)), np.ones(len(branch))],
                 constraints=LinearConstraint(matrix, lower, upper),
                 bounds=Bounds(xmin, xmax), integrality=np.zeros(len(xmin)))
    if probe.x is not None:
        n, ng = len(bus), len(gen)
        f, t, _ = indices(data)
        theta = probe.x[ng:ng + n]
        needed = float(np.abs(np.rad2deg(theta[f] - theta[t])).max())
        allowed = float(np.nanmax(np.abs(branch[:, 11:13])))
        return (f"The branch angle limits are what block it: the network needs "
                f"{needed:.1f} degrees across a branch but is limited to {allowed:.0f}. "
                f"Equation (4) does not constrain angle differences, and this limit is "
                f"a modelling default rather than a measured value, so consider raising "
                f"BRANCH_ANGLE_LIMIT_DEG in config.py. An angle this large also means "
                f"the corridor is near its steady-state stability limit, which usually "
                f"points at missing parallel circuits or too much load on one path.")
    return ("Generation limits or an islanded demand pocket are binding; the branch "
            "angle limits are not the only obstacle.")


def island_labels_from_case(data):
    """Component label per bus row of a MATPOWER case dict."""
    ids = {int(v): i for i, v in enumerate(data["bus"][:, 0])}
    branch = data["branch"]
    active = branch[:, 10] != 0
    f = np.array([ids[int(v)] for v in branch[active, 0]], dtype=int)
    t = np.array([ids[int(v)] for v in branch[active, 1]], dtype=int)
    n = len(ids)
    if not len(f):
        return np.arange(n)
    return connected_components(csr_matrix((np.ones(len(f)), (f, t)), shape=(n, n)),
                                directed=False)[1]


def _solve_dc(data, penalty):
    """Manuscript equation (4): least extra line capacity that makes DCOPF feasible.

    HiGHS answers feasibility exactly and warm-starts the convex QP that the
    equation actually states, which a trust-region method then solves.
    """
    bus, gen, branch = (data[k] for k in ("bus", "gen", "branch"))
    base = data["baseMVA"]
    n, ng, nb = len(bus), len(gen), len(branch)
    matrix, lower, upper, xmin, xmax, limited = _dc_program(data)
    nx = ng + n + nb
    pg, th, sl = slice(0, ng), slice(ng, ng + n), slice(ng + n, nx)
    constraint = LinearConstraint(matrix, lower, upper)
    bounds = Bounds(xmin, xmax)

    probe = milp(c=np.r_[np.zeros(ng + n), np.ones(nb)], constraints=constraint,
                 bounds=bounds, integrality=np.zeros(nx))
    if probe.x is None:
        raise RuntimeError(
            "No DC dispatch exists for this case even with unlimited extra line "
            "capacity, so equation (4) cannot repair it. "
            + _diagnose_dc_infeasibility(data))

    active = gen[:, 7] > 0
    cost = data["gencost"][:, 4:7]
    c2 = np.where(active, cost[:, 0], 0.) * base**2
    c1 = np.where(active, cost[:, 1], 0.) * base
    weight = penalty * base**2                    # slack is squared in MW, not per unit
    scale = max(1., float(np.abs(c1) @ np.abs(xmax[pg])))
    curvature = np.zeros(nx)
    curvature[pg], curvature[sl] = 2 * c2 / scale, 2 * weight / scale
    hessian = csr_matrix((curvature, (np.arange(nx), np.arange(nx))), shape=(nx, nx))

    def objective(x):
        p, s = x[pg], x[sl]
        return float((c2 @ (p * p) + c1 @ p + weight * (s @ s)) / scale)

    def gradient(x):
        out = np.zeros(nx)
        out[pg], out[sl] = (2 * c2 * x[pg] + c1) / scale, 2 * weight * x[sl] / scale
        return out

    result = minimize(objective, probe.x, jac=gradient, hess=lambda x: hessian,
                      method="trust-constr", bounds=bounds, constraints=[constraint],
                      options=dict(maxiter=3000, gtol=1e-10, xtol=1e-12))
    x = result.x if result.constr_violation <= 1e-8 else probe.x
    gen[:, 1], gen[:, 2] = x[pg] * base, 0.
    bus[:, 8] = np.rad2deg(x[th])
    expansion = np.maximum(x[sl] * base, 0.)
    for col in (5, 6, 7):
        # Preserve zero (unlimited) emergency ratings.
        update = limited & (branch[:, col] > 0)
        branch[update, col] += expansion[update]
    return data


# PyPOWER reads a gen matrix narrower than this as MATPOWER version 1 and then
# discards the branch angle limits; constrained_dispatch carries the same guard.
GEN_COLS_V2 = 21
# The reactive sources carry no active power. PIPS folds a variable whose bounds
# are exactly equal into its equality set, so pinning Pmin = Pmax = 0 converges,
# while leaving a hair of headroom instead makes a box too narrow to navigate.
UNLIMITED_MVA = 1e5
OPF_OPTIONS = dict(VERBOSE=0, OUT_ALL=0, OPF_ALG=560, PDIPM_MAX_IT=200)
# PIPS reports failure whenever it cannot reach its feasibility tolerance, even
# after the objective, gradient and complementarity conditions have all
# converged, and the floor it can reach depends on the case. Ask for the tightest
# first, because check_solution rejects anything above POWER_TOL_MVA, then settle
# for less rather than throwing away a converged answer.
FEASIBILITY_TOLERANCES = (1e-8, 1e-7, 1e-6)


def _run_opf(case):
    from pypower.api import ppoption, runopf
    case = dict(case)
    pad = GEN_COLS_V2 - case["gen"].shape[1]
    if pad > 0:
        case["gen"] = np.hstack([case["gen"], np.zeros((len(case["gen"]), pad))])
    for tolerance in FEASIBILITY_TOLERANCES:
        solved = runopf(dict(case), ppoption(OPF_VIOLATION=tolerance,
                                             PDIPM_FEASTOL=tolerance, **OPF_OPTIONS))
        if solved["success"]:
            return solved
    return None


def _reactive_opf(data, allowance):
    """AC OPF with a bounded reactive source at each bus whose allowance is nonzero.

    The source stands in for the shunt of equation (5): it injects Q rather than
    B*V^2, and _solve_ac converts one into the other once the schedule settles.
    """
    bus, gen = data["bus"], data["gen"]
    ng = len(gen)
    sources = np.flatnonzero(allowance > 0)
    # Match the input width; MATPOWER cases may carry multiplier columns already.
    extra = np.zeros((len(sources), gen.shape[1]))
    extra[:, 0] = bus[sources, 0]
    extra[:, 3], extra[:, 4] = allowance[sources], -allowance[sources]
    extra[:, 5], extra[:, 6], extra[:, 7] = 1., data["baseMVA"], 1.
    solved = _run_opf(dict(version="2", baseMVA=data["baseMVA"], bus=bus.copy(),
                           gen=np.vstack([gen, extra]), branch=data["branch"].copy(),
                           gencost=np.vstack([data["gencost"],
                                              np.tile([2, 0, 0, 3, 0., 0., 0.],
                                                      (len(sources), 1))])))
    if solved is None:
        return None
    reactive = np.zeros(len(bus))
    reactive[sources] = solved["gen"][ng:, 2]
    solved["gen"] = solved["gen"][:ng]
    return solved, reactive


def _apparent_flows(solved):
    branch = solved["branch"]
    return np.maximum(np.hypot(branch[:, 13], branch[:, 14]),
                      np.hypot(branch[:, 15], branch[:, 16]))


def _relieve_thermal(data, allowance, margin=1e-3, steps=8):
    """Raise ratings to the apparent flows the AC network needs.

    Equation (4) sizes its slack on DC flows, which omit losses and reactive
    flow, so the ratings it leaves behind can still be infeasible for equation
    (5). This applies the idea of (4e) to |S| instead of P. The ceiling comes
    from an unconstrained AC solve and is then walked back towards the original
    ratings, so the expansion reported is close to the least one that solves.
    """
    branch = data["branch"]
    limited = (branch[:, 10] != 0) & (branch[:, 5] > 0)
    if not limited.any():
        return False
    probe = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in data.items()}
    probe["branch"][limited, 5:8] = UNLIMITED_MVA
    attempt = _reactive_opf(probe, allowance)
    if attempt is None:
        return False
    original = branch[:, 5].copy()
    # Keep the limits strictly off the solution so the interior point method
    # does not start on top of an active constraint.
    ceiling = np.maximum(original, _apparent_flows(attempt[0]) * (1. + margin))
    if not (limited & (ceiling > original + POWER_TOL_MVA)).any():
        return False

    def solves(rate):
        trial = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in data.items()}
        trial["branch"][limited, 5] = rate[limited]
        return _reactive_opf(trial, allowance) is not None

    if not solves(ceiling):
        return False
    low, high = 0., 1.
    for _ in range(steps):
        middle = (low + high) / 2
        if solves(original + middle * (ceiling - original)):
            high = middle
        else:
            low = middle
    expanded = original + high * (ceiling - original)
    grew = limited & (expanded > original)
    for col in (5, 6, 7):
        # Preserve zero (unlimited) emergency ratings.
        update = grew & (branch[:, col] > 0)
        branch[update, col] = np.maximum(branch[update, col], expanded[update])
    return True


def _solve_ac(data, max_bsh, rounds=20, headroom=1e-3):
    """Manuscript equation (5): fewest shunt devices that make the ACOPF feasible.

    The shunt of (5d) is stood in for by a reactive source at each bus, because
    PyPOWER can optimise a generator's Q but not a bus susceptance. The sources
    cost nothing, so (5a)'s preference for the fewest devices is imposed by
    repeatedly offering each bus only what it drew last time and keeping the last
    schedule that still solved; buses that drew nothing lose their source. The
    surviving draws become fixed shunts and the case is re-solved without any
    source, so what check_solution validates is the case that gets exported.
    """
    bus, gen = data["bus"], data["gen"]
    base, n = data["baseMVA"], len(bus)
    _assign_bus_types(data)
    # A fixed shunt supplies Bs*V^2, so cap the source at the least it could draw
    # for a given Bs; otherwise converting back would need a Bs above the limit.
    ceiling = np.full(n, max_bsh * base * float(np.min(bus[:, 12]))**2)

    attempt = _reactive_opf(data, ceiling)
    if attempt is None and _relieve_thermal(data, ceiling):
        attempt = _reactive_opf(data, ceiling)
    if attempt is None:
        raise RuntimeError("AC feasibility relaxation found no solution even with a "
                           "reactive source at every bus and the thermal limits "
                           "raised to the flows the network needs")

    # Squeeze the sources towards zero. Each round offers a fraction of what was
    # drawn last time; a round that cannot be solved backs the fraction off
    # towards one instead of abandoning the search. Buses left drawing nothing
    # lose their source, which is what (5a) asks for.
    best, fraction = attempt, 0.5
    for _ in range(rounds):
        draw = np.abs(best[1])
        if not (draw > POWER_TOL_MVA).any():
            break
        trial = np.where(draw > POWER_TOL_MVA, draw * fraction, 0.)
        attempt = _reactive_opf(data, trial)
        if attempt is None:
            fraction = (1. + fraction) / 2.
            if fraction > 1. - 1e-3:
                break
        else:
            best = attempt

    solved, reactive = best
    keep = np.abs(reactive) > POWER_TOL_MVA
    # MATPOWER Bs is MVAr at 1.0 pu and a fixed shunt supplies Bs*V^2, so this
    # reproduces exactly the injection the source was making.
    bus[:, 5] = np.where(keep, reactive / solved["bus"][:, 7]**2, 0.)

    # That operating point already satisfies the source-free case, so hand it to
    # the solver as the starting point rather than making it search from flat.
    warm = dict(version="2", baseMVA=base, bus=bus.copy(), gen=gen.copy(),
                branch=data["branch"].copy(), gencost=data["gencost"].copy())
    warm["bus"][:, 7:9] = solved["bus"][:, 7:9]
    warm["gen"][:, 1:3] = solved["gen"][:, 1:3]
    final = _run_opf(warm)
    if final is None:
        raise RuntimeError("AC feasibility relaxation produced shunts the OPF cannot use")
    gen[:, 1:3] = final["gen"][:, 1:3]
    bus[:, 7:9] = final["bus"][:, 7:9]
    # A power flow holds each PV bus at its generator's Vg, not at the bus Vm, so
    # leaving Vg at its old value hands runpf a different operating point from
    # the one just solved and a stressed network will not converge to it.
    _, _, g = indices(data)
    gen[:, 5] = bus[g, 7]
    return data


def solve_feasibility(data, ac=False, penalty=1e4, max_bsh=5.0):
    validate_case(data)
    data = dict(data)
    for key in ("bus", "gen", "branch", "gencost"):
        # MATPOWER fixtures can use integer arrays; never truncate solved dispatch.
        data[key] = np.array(data[key], dtype=float, copy=True)
    data = _solve_ac(data, max_bsh) if ac else _solve_dc(data, penalty)
    check_solution(data, ac=ac)
    return data
