"""
ensure_feasibility.py

Produces DC-feasible and AC-feasible variants of the synthetic grid.

DC formulation  (eq:slack_dcopf)
---------------------------------
  min  Σ_G C_i(P_Gi) + M Σ_{ij∈L} (P_ij^sl)²
  s.t. Σ_G P_Gi = Σ_D P_Dj                    [global balance]
       P_Gi^min ≤ P_Gi ≤ P_Gi^max
       P_ij = (θ_i − θ_j) / X_ij
       −P_ij^max − P_ij^sl ≤ P_ij ≤ P_ij^max + P_ij^sl
       P_ij^sl ≥ 0
  Slack P^sl represents extra line capacity needed for feasibility.
  Qg = 0 in the DC case.

AC formulation  (eq:shunt_acopf)
----------------------------------
  min  Σ_g C_g(P_Gg) + M Σ_{i∈B} |B_i^sh|
  s.t. θ_ref = 0
       AC real-power balance at each bus
       AC reactive-power balance at each bus (with shunt term B_i^sh · V_i²)
       P_g^min ≤ P_g ≤ P_g^max
       Q_g^min ≤ Q_g ≤ Q_g^max
       V_i^min ≤ V_i ≤ V_i^max
       |S_ij,fr| ≤ S_ij^max,  |S_ij,to| ≤ S_ij^max
       θ_ij^min ≤ θ_i − θ_j ≤ θ_ij^max
       B_i^sh,min ≤ B_i^sh ≤ B_i^sh,max

  |B_i^sh| penalty (l1) promotes sparse shunt placement — fewer buses
  receive a shunt, each shunt being larger — vs. the l2 formulation that
  would spread small shunts across many buses.

Outputs
-------
  data/working_files/{region}/{region}_dc_feasible.m
  data/working_files/{region}/{region}_ac_feasible.m

Both files use the MATPOWER Version 2 format. Generator Pg (and Qg for AC)
are set to the dispatch values obtained from the respective relaxation.
"""

import math
import re
import warnings
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import lil_matrix

import config

warnings.filterwarnings("ignore")

region      = config.region
PROJ_ROOT   = Path(__file__).parent.parent
WORK_DIR    = PROJ_ROOT / "data" / "working_files" / region
MPOWER_DIR  = PROJ_ROOT / "matpower_files"
MPOWER_DIR.mkdir(parents=True, exist_ok=True)

M_PENALTY     = 1e4   # weight on slack/shunt penalty terms
MAX_BSH_PU    = 5.0   # p.u. maximum shunt susceptance per bus (±)
V_MIN, V_MAX  = 0.9, 1.1


# ============================================================
# MATPOWER .m PARSER
# ============================================================

def _parse_matrix(text: str, name: str) -> np.ndarray | None:
    """Extract a numeric matrix block from a MATPOWER .m file."""
    pat = rf"mpc\.{name}\s*=\s*\[(.*?)\];"
    m   = re.search(pat, text, re.DOTALL)
    if not m:
        return None
    rows = []
    for line in m.group(1).strip().splitlines():
        line = re.sub(r"%.*", "", line).strip()
        if not line:
            continue
        try:
            rows.append([float(x) for x in line.split()])
        except ValueError:
            pass
    return np.array(rows) if rows else None


def load_matpower(path: Path):
    """Parse a MATPOWER .m file and return dict with 'bus', 'gen', 'branch', 'gencost'."""
    text = path.read_text()
    data = {}
    for key in ("bus", "gen", "branch", "gencost"):
        arr = _parse_matrix(text, key)
        data[key] = arr if arr is not None else np.zeros((0, 1))

    # Also extract baseMVA
    m = re.search(r"mpc\.baseMVA\s*=\s*([\d.]+)", text)
    data["baseMVA"] = float(m.group(1)) if m else 100.0
    return data


# ============================================================
# NETWORK HELPERS
# ============================================================

def build_bus_index(bus_arr: np.ndarray) -> dict:
    """Map bus_i number → 0-based row index."""
    return {int(bus_arr[i, 0]): i for i in range(len(bus_arr))}


def build_b_matrix(bus_arr: np.ndarray, branch_arr: np.ndarray,
                   bus_idx: dict) -> np.ndarray:
    """Build DC susceptance matrix B (n×n)."""
    n = len(bus_arr)
    B = np.zeros((n, n))
    for row in branch_arr:
        if int(row[10]) == 0:   # status = 0 → out of service
            continue
        fi = bus_idx.get(int(row[0]))
        ti = bus_idx.get(int(row[1]))
        if fi is None or ti is None:
            continue
        x = row[3]
        if abs(x) < 1e-9:
            continue
        b = 1.0 / x
        B[fi, fi] += b
        B[ti, ti] += b
        B[fi, ti] -= b
        B[ti, fi] -= b
    return B


# ============================================================
# DC FEASIBILITY  (eq:slack_dcopf)
# ============================================================

def solve_dc_feasibility(data: dict) -> dict:
    """
    Solve the slack DC OPF and return updated data dict with:
      - Pg dispatch values
      - rateA expanded by slack (where needed)
      - P_sl_max recorded for logging
    """
    bus_arr    = data["bus"]
    gen_arr    = data["gen"].copy()
    branch_arr = data["branch"].copy()
    gencost    = data["gencost"]
    bus_idx    = build_bus_index(bus_arr)
    n          = len(bus_arr)
    nb         = len(branch_arr)
    ng         = len(gen_arr)

    total_load  = bus_arr[:, 2].sum()   # column 3 = Pd
    total_pmax  = gen_arr[:, 8].sum()   # column 9 = Pmax
    total_pmin  = gen_arr[:, 9].sum()   # column 10 = Pmin
    print(f"  Total Pd = {total_load:.1f} MW  |  ΣPmax = {total_pmax:.1f} MW  "
          f"|  ΣPmin = {total_pmin:.1f} MW")

    if total_pmax < total_load * 0.99:
        print(f"  WARNING: generation capacity ({total_pmax:.1f} MW) < load "
              f"({total_load:.1f} MW). DC OPF will need load shedding or infeasible.")

    # Build net injection vector: P_net[i] = ΣPg[i] - Pd[i]  (vars for Pg)
    # Decision variables: x = [Pg_0,…,Pg_{ng-1}, θ_0,…,θ_{n-1}, Psl_0,…,Psl_{nb-1}]
    #   total vars = ng + n + nb
    n_vars = ng + n + nb

    # Generator bus mapping: gen_bus_idx[g] = row index in bus_arr
    gen_bus_map = []
    for g in range(ng):
        bi = int(gen_arr[g, 0])
        gen_bus_map.append(bus_idx.get(bi, 0))

    B = build_b_matrix(bus_arr, branch_arr, bus_idx)

    # Reference bus = first type-3 bus, else first bus
    ref_candidates = np.where(bus_arr[:, 1] == 3)[0]
    ref_bus = int(ref_candidates[0]) if len(ref_candidates) else 0

    # Objective: Σ cost(Pg) + M*Σ(Psl²)
    def objective(x):
        Pg  = x[:ng]
        Psl = x[ng + n:]
        cost = 0.0
        for g in range(ng):
            pmax = gen_arr[g, 8]
            if pmax <= 0:
                continue
            if g < len(gencost):
                c2, c1, c0 = gencost[g, 4], gencost[g, 5], gencost[g, 6]
                cost += c2 * Pg[g]**2 + c1 * Pg[g] + c0
        cost += M_PENALTY * np.sum(Psl**2)
        return cost

    def gradient(x):
        Pg  = x[:ng]
        Psl = x[ng + n:]
        g_vec = np.zeros(n_vars)
        for g in range(ng):
            if g < len(gencost):
                c2, c1 = gencost[g, 4], gencost[g, 5]
                g_vec[g] = 2 * c2 * Pg[g] + c1
        g_vec[ng + n:] = 2 * M_PENALTY * Psl
        return g_vec

    # Bounds
    lb = np.zeros(n_vars)
    ub = np.full(n_vars, np.inf)
    # Pg
    for g in range(ng):
        lb[g] = max(0.0, gen_arr[g, 9])   # Pmin
        ub[g] = max(lb[g], gen_arr[g, 8]) # Pmax
    # Angles
    lb[ng:ng + n] = -math.pi
    ub[ng:ng + n] =  math.pi
    lb[ng + ref_bus] = ub[ng + ref_bus] = 0.0  # reference angle = 0
    # Slack (non-negative)
    lb[ng + n:] = 0.0

    # Constraints
    # 1. DC power balance — drop the reference bus row to avoid a singular
    #    Jacobian.  The ref bus angle is already pinned to 0 by its bounds,
    #    so removing that one equation leaves a full-rank (n-1) system.
    non_ref = [i for i in range(n) if i != ref_bus]

    def eq_power_balance(x):
        Pg    = x[:ng]
        th    = x[ng:ng + n]
        P_net = -bus_arr[:, 2].copy()   # start with −Pd
        for g, bi in enumerate(gen_bus_map):
            P_net[bi] += Pg[g]
        return (B @ th - P_net)[non_ref]   # n−1 equations

    # 2. Line flow limits: |P_ij| ≤ rateA + Psl
    def ineq_line_limits(x):
        th  = x[ng:ng + n]
        Psl = x[ng + n:]
        violations = []
        for k, row in enumerate(branch_arr):
            if int(row[10]) == 0:
                violations.extend([0.0, 0.0])
                continue
            fi  = bus_idx.get(int(row[0]), 0)
            ti  = bus_idx.get(int(row[1]), 0)
            xij = row[3]
            if abs(xij) < 1e-9:
                violations.extend([0.0, 0.0])
                continue
            Pij  = (th[fi] - th[ti]) / xij
            Pmax = row[5] if row[5] > 0 else 1e6   # 0 = unlimited
            violations.append(Pmax + Psl[k] - Pij)
            violations.append(Pmax + Psl[k] + Pij)
        return np.array(violations)

    constraints = [
        {"type": "eq",   "fun": eq_power_balance},
        {"type": "ineq", "fun": ineq_line_limits},
    ]

    # Initial point: uniform dispatch
    x0 = np.zeros(n_vars)
    if ng > 0:
        x0[:ng] = np.clip(total_load / ng, lb[:ng], ub[:ng])

    print("  Solving DC OPF with slack variables …")
    res = minimize(
        objective, x0,
        jac=gradient,
        method="SLSQP",
        bounds=list(zip(lb, ub)),
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-8, "disp": False},
    )

    if res.success or res.status in (0, 9):
        print(f"  DC OPF converged  (status={res.status})")
    else:
        print(f"  DC OPF did not converge cleanly (status={res.status}: {res.message})")
        print("  Using best-found solution.")

    Pg_opt  = res.x[:ng]
    Psl_opt = res.x[ng + n:]

    # Update generator Pg; set Qg = 0 (DC)
    gen_arr[:, 1] = np.clip(Pg_opt, gen_arr[:, 9], gen_arr[:, 8])  # Pg
    gen_arr[:, 2] = 0.0                                              # Qg

    # Expand rateA where slack is needed (>0.1 MVA)
    n_expanded = 0
    for k in range(min(nb, len(Psl_opt))):
        if Psl_opt[k] > 0.1:
            branch_arr[k, 5] += Psl_opt[k]   # rateA
            branch_arr[k, 6] += Psl_opt[k]   # rateB
            branch_arr[k, 7] += Psl_opt[k]   # rateC
            n_expanded += 1

    if n_expanded:
        print(f"  Expanded rateA for {n_expanded} line(s) (max slack = "
              f"{Psl_opt.max():.2f} MVA).")
    else:
        print("  No line limits needed expansion — system is DC-feasible.")

    result = dict(data)
    result["gen"]    = gen_arr
    result["branch"] = branch_arr
    return result


# ============================================================
# AC FEASIBILITY  (eq:shunt_acopf, l1 shunt penalty)
# ============================================================

def solve_ac_feasibility(data: dict) -> dict:
    """
    Solve the AC OPF with optimal shunt placement using l1 penalty on B_sh
    to minimise the number of shunts (sparsity-promoting).

    Decision variables: [Pg, Qg, V_mag, θ, B_sh]
      - Pg, Qg: generator real/reactive dispatch
      - V_mag, θ: bus voltage magnitude and angle
      - B_sh: additional shunt susceptance at each bus (p.u.)
    """
    bus_arr    = data["bus"].copy()
    gen_arr    = data["gen"].copy()
    branch_arr = data["branch"].copy()
    gencost    = data["gencost"]
    bus_idx    = build_bus_index(bus_arr)
    n          = len(bus_arr)
    nb         = len(branch_arr)
    ng         = len(gen_arr)

    gen_bus_map = []
    for g in range(ng):
        bi = int(gen_arr[g, 0])
        gen_bus_map.append(bus_idx.get(bi, 0))

    ref_candidates = np.where(bus_arr[:, 1] == 3)[0]
    ref_bus = int(ref_candidates[0]) if len(ref_candidates) else 0

    # Build Ybus admittance matrix (complex)
    Y = np.zeros((n, n), dtype=complex)
    for row in branch_arr:
        if int(row[10]) == 0:
            continue
        fi = bus_idx.get(int(row[0]))
        ti = bus_idx.get(int(row[1]))
        if fi is None or ti is None:
            continue
        r, x, b = row[2], row[3], row[4]
        if abs(r) < 1e-12 and abs(x) < 1e-12:
            continue
        y   = 1.0 / complex(r, x)
        bsh = 1j * b / 2.0
        Y[fi, fi] += y + bsh
        Y[ti, ti] += y + bsh
        Y[fi, ti] -= y
        Y[ti, fi] -= y

    G = Y.real
    B_net = Y.imag

    # Vars: [Pg(ng), Qg(ng), V(n), th(n), Bsh(n)]
    n_Pg  = ng;  off_Pg  = 0
    n_Qg  = ng;  off_Qg  = ng
    n_V   = n;   off_V   = 2 * ng
    n_th  = n;   off_th  = 2 * ng + n
    n_Bsh = n;   off_Bsh = 2 * ng + 2 * n
    n_vars = 2 * ng + 2 * n + n

    def objective(x):
        Pg  = x[off_Pg:off_Pg + n_Pg]
        Bsh = x[off_Bsh:off_Bsh + n_Bsh]
        cost = 0.0
        for g in range(ng):
            if gen_arr[g, 8] <= 0:
                continue
            if g < len(gencost):
                c2, c1, c0 = gencost[g, 4], gencost[g, 5], gencost[g, 6]
                cost += c2 * Pg[g]**2 + c1 * Pg[g] + c0
        cost += M_PENALTY * np.sum(np.abs(Bsh))   # l1 penalty (sparsity)
        return cost

    def gradient(x):
        Pg  = x[off_Pg:off_Pg + n_Pg]
        Bsh = x[off_Bsh:off_Bsh + n_Bsh]
        g   = np.zeros(n_vars)
        for gg in range(ng):
            if gg < len(gencost):
                c2, c1 = gencost[gg, 4], gencost[gg, 5]
                g[off_Pg + gg] = 2 * c2 * Pg[gg] + c1
        g[off_Bsh:off_Bsh + n_Bsh] = M_PENALTY * np.sign(Bsh)
        return g

    def _safe(v, default):
        """Return float(v) unless it is NaN/None, in which case return default."""
        if v is None:
            return default
        f = float(v)
        return default if math.isnan(f) or math.isinf(f) else f

    # Bounds — clamp and NaN-safe so lb ≤ ub is always satisfied
    lb = np.zeros(n_vars)
    ub = np.full(n_vars, np.inf)

    for gg in range(ng):
        pmin = _safe(gen_arr[gg, 9], 0.0)
        pmax = _safe(gen_arr[gg, 8], 0.0)
        qmin = _safe(gen_arr[gg, 4], -999.0)
        qmax = _safe(gen_arr[gg, 3],  999.0)
        lb[off_Pg + gg] = max(0.0, pmin)
        ub[off_Pg + gg] = max(lb[off_Pg + gg], pmax)
        # Ensure Qmin ≤ Qmax (swap if data has them inverted)
        lb[off_Qg + gg] = min(qmin, qmax)
        ub[off_Qg + gg] = max(qmin, qmax)

    for i in range(n):
        vmin_raw = bus_arr[i, 12] if bus_arr.shape[1] > 12 else V_MIN
        vmax_raw = bus_arr[i, 11] if bus_arr.shape[1] > 11 else V_MAX
        vmin = _safe(vmin_raw, V_MIN)
        vmax = _safe(vmax_raw, V_MAX)
        # Clamp to physical range and ensure lb ≤ ub
        lo = max(V_MIN, min(vmin, vmax))
        hi = min(V_MAX, max(vmin, vmax))
        lb[off_V + i] = lo if lo <= hi else V_MIN
        ub[off_V + i] = hi if lo <= hi else V_MAX

    lb[off_th:off_th + n_th] = -math.pi
    ub[off_th:off_th + n_th] =  math.pi
    lb[off_th + ref_bus] = ub[off_th + ref_bus] = 0.0
    lb[off_Bsh:off_Bsh + n_Bsh] = -MAX_BSH_PU
    ub[off_Bsh:off_Bsh + n_Bsh] =  MAX_BSH_PU

    non_ref_ac = [i for i in range(n) if i != ref_bus]

    def ac_power_balance(x):
        Pg  = x[off_Pg:off_Pg + n_Pg]
        Qg  = x[off_Qg:off_Qg + n_Qg]
        V   = x[off_V:off_V + n_V]
        th  = x[off_th:off_th + n_th]
        Bsh = x[off_Bsh:off_Bsh + n_Bsh]

        Pd = bus_arr[:, 2]
        Qd = bus_arr[:, 3]

        P_gen = np.zeros(n)
        Q_gen = np.zeros(n)
        for gg, bi in enumerate(gen_bus_map):
            P_gen[bi] += Pg[gg]
            Q_gen[bi] += Qg[gg]

        P_res = np.zeros(n)
        Q_res = np.zeros(n)
        for i in range(n):
            for j in range(n):
                dth    = th[i] - th[j]
                ViVj   = V[i] * V[j]
                P_res[i] += ViVj * (G[i, j] * math.cos(dth) + B_net[i, j] * math.sin(dth))
                Q_res[i] += ViVj * (G[i, j] * math.sin(dth) - B_net[i, j] * math.cos(dth))

        P_mis = (P_gen - Pd - P_res)[non_ref_ac]   # drop ref bus P equation
        Q_mis = (Q_gen - Qd + Bsh * V**2 - Q_res)[non_ref_ac]  # drop ref bus Q
        return np.concatenate([P_mis, Q_mis])

    def ineq_thermal(x):
        V  = x[off_V:off_V + n_V]
        th = x[off_th:off_th + n_th]
        viol = []
        for row in branch_arr:
            if int(row[10]) == 0:
                viol.extend([1.0, 1.0])
                continue
            fi = bus_idx.get(int(row[0]))
            ti = bus_idx.get(int(row[1]))
            if fi is None or ti is None:
                viol.extend([1.0, 1.0])
                continue
            r, x_br, b_br = row[2], row[3], row[4]
            smax = row[5] if row[5] > 0 else 1e6
            if abs(r) < 1e-12 and abs(x_br) < 1e-12:
                viol.extend([1.0, 1.0])
                continue
            y   = 1.0 / complex(r, x_br)
            dth = th[fi] - th[ti]
            Vi, Vj = V[fi], V[ti]
            S_fr = (Vi**2 * (y.real + b_br / 2)
                    - Vi * Vj * (y.real * math.cos(dth) + y.imag * math.sin(dth))
                    + 1j * (Vi**2 * (y.imag + b_br / 2) - Vi * Vj * (y.real * math.sin(dth)
                    - y.imag * math.cos(dth))))
            abs_fr = abs(S_fr)
            S_to = (Vj**2 * (y.real + b_br / 2)
                    - Vi * Vj * (y.real * math.cos(dth) - y.imag * math.sin(dth))
                    + 1j * (Vj**2 * (y.imag + b_br / 2) - Vi * Vj * (-y.real * math.sin(dth)
                    - y.imag * math.cos(dth))))
            abs_to = abs(S_to)
            viol.append(smax - abs_fr)
            viol.append(smax - abs_to)
        return np.array(viol)

    constraints = [
        {"type": "eq",   "fun": ac_power_balance},
        {"type": "ineq", "fun": ineq_thermal},
    ]

    # Warm start from DC solution
    x0 = np.zeros(n_vars)
    for gg in range(ng):
        x0[off_Pg + gg] = float(np.clip(gen_arr[gg, 1], lb[off_Pg + gg], ub[off_Pg + gg]))
        x0[off_Qg + gg] = 0.0
    x0[off_V:off_V + n_V] = np.clip(bus_arr[:, 7], V_MIN, V_MAX)   # Vm from bus data
    x0[off_th:off_th + n_th] = np.deg2rad(bus_arr[:, 8])            # Va from bus data

    print("\n  Solving AC OPF with l1 shunt placement …")
    res = minimize(
        objective, x0,
        jac=gradient,
        method="SLSQP",
        bounds=list(zip(lb, ub)),
        constraints=constraints,
        options={"maxiter": 3000, "ftol": 1e-7, "disp": False},
    )

    if res.success or res.status in (0, 9):
        print(f"  AC OPF converged  (status={res.status})")
    else:
        print(f"  AC OPF did not converge cleanly (status={res.status}: {res.message})")
        print("  Using best-found solution — check voltage profiles manually.")

    Pg_opt  = res.x[off_Pg:off_Pg + n_Pg]
    Qg_opt  = res.x[off_Qg:off_Qg + n_Qg]
    V_opt   = res.x[off_V:off_V + n_V]
    th_opt  = res.x[off_th:off_th + n_th]
    Bsh_opt = res.x[off_Bsh:off_Bsh + n_Bsh]

    # Update generator dispatch
    gen_arr[:, 1] = np.clip(Pg_opt, gen_arr[:, 9], gen_arr[:, 8])
    gen_arr[:, 2] = np.clip(Qg_opt, gen_arr[:, 4], gen_arr[:, 3])

    # Consolidate shunts: each bus gets a single Bs value (l1 already promotes sparsity)
    n_nonzero = 0
    for i in range(n):
        bsh = float(Bsh_opt[i])
        if abs(bsh) > 1e-4:
            bus_arr[i, 5] += bsh   # column 6 = Bs in MATPOWER (0-indexed: col 5)
            n_nonzero += 1

    if n_nonzero:
        print(f"  Added shunts at {n_nonzero} bus(es)  "
              f"(max |B_sh| = {np.abs(Bsh_opt).max():.4f} p.u.).")
    else:
        print("  No shunts needed — system is AC-feasible without reactive support.")

    # Update bus Vm and Va
    bus_arr[:, 7] = np.clip(V_opt, V_MIN, V_MAX)
    bus_arr[:, 8] = np.rad2deg(th_opt)

    result = dict(data)
    result["bus"]  = bus_arr
    result["gen"]  = gen_arr
    return result


# ============================================================
# WRITE MATPOWER .m FILE  (full descriptive format)
# ============================================================

def _load_metadata():
    """Load bus/generator metadata from the working-files CSVs for annotation blocks."""
    buses_csv = WORK_DIR / f"matpower_{region}_buses.csv"
    gens_csv  = WORK_DIR / f"matpower_{region}_generators.csv"
    meta = {"bus_names": [], "bus_coords": [], "gen_names": [], "gen_fuels": []}
    if buses_csv.exists():
        bdf = pd.read_csv(buses_csv)
        for _, r in bdf.iterrows():
            name = str(r.get("name", "")).strip()
            kv   = r.get("base_kV", 0) or 0
            kv   = float(kv) / 1000.0 if float(kv) > 1000 else float(kv)
            label = name if name and name.lower() != "nan" else f"Bus_{int(r['bus_i'])}"
            meta["bus_names"].append(f"\t'{label}_{kv:.0f}kV' %{int(r['bus_i'])}")
            lat = r.get("latitude", 0.0)
            lon = r.get("longitude", 0.0)
            lat = 0.0 if (lat is None or str(lat) == "nan") else float(lat)
            lon = 0.0 if (lon is None or str(lon) == "nan") else float(lon)
            meta["bus_coords"].append(f"\t'{lat:.6f}, {lon:.6f}'")
    if gens_csv.exists():
        gdf = pd.read_csv(gens_csv).dropna(subset=["bus"])
        for _, r in gdf.iterrows():
            meta["gen_names"].append(f"\t'{str(r.get('generator_name','?')).strip()}'")
            meta["gen_fuels"].append(
                f"\t'{str(r.get('source','unknown')).split(';')[0].strip()}'")
    return meta


def write_matpower(data: dict, out_path: Path, label: str):
    """Write a MATPOWER Version 2 .m file with the full descriptive header format."""
    bus_arr    = data["bus"]
    gen_arr    = data["gen"]
    branch_arr = data["branch"]
    gencost    = data["gencost"]
    baseMVA    = data.get("baseMVA", 100.0)

    def _fmt(v, d=6):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "0.0"
        return f"{round(float(v), d)}"

    bus_lines  = ["\t" + "\t".join(_fmt(v, 6) for v in r) for r in bus_arr]
    gen_lines  = ["\t" + "\t".join(_fmt(v, 4) for v in r) for r in gen_arr]
    br_lines   = ["\t" + "\t".join(_fmt(v, 8) for v in r) for r in branch_arr]
    cost_lines = ["\t" + "\t".join(_fmt(v, 8) for v in r) for r in gencost]

    meta      = _load_metadata()
    func_name = f"{config.SHAPE_FILE_NAME.replace(' ','_')}Grid_{label}"
    today     = date.today().strftime("%Y-%m-%d")

    header = (
        f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
        f"%%%%                                                                  %%%%%\n"
        f"%%%%   Synthetic Electric Grid for {config.SHAPE_FILE_NAME:<30} %%%%%\n"
        f"%%%%   {label:<62} %%%%%\n"
        f"%%%%   {today:<62} %%%%%\n"
        f"%%%%                                                                  %%%%%\n"
        f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
        f"%\n"
        f"%   DC and AC Optimal Power Flow data for the {config.SHAPE_FILE_NAME}\n"
        f"%   electricity transmission system ({label}).\n"
        f"%   Please see CASEFORMAT for details on the case file format.\n"
        f"%\n"
        f"%   When publishing results using this data, please cite:\n"
        f"%   R. Asiamah, R. Sharma, C. Merkt, S. Talkington, J. Watson, and D. Molzahn,\n"
        f"%   \"A Framework for the Creation of Highly Realistic International Synthetic\n"
        f"%   Grid Models,\" to appear in IEEE Transactions on Energy Markets, Policy\n"
        f"%   and Regulation, Jun. 2025.\n"
        f"%\n"
        f"%   Contact: Richard Asiamah (asiamah@gatech.edu)\n"
        f"%   Grid Optimization and Algorithms Lab, Georgia Institute of Technology\n"
        f"%\n"
    )

    names_block  = "\n".join(meta["bus_names"])  if meta["bus_names"]  else "\t% (none)"
    coords_block = "\n".join(meta["bus_coords"]) if meta["bus_coords"] else "\t% (none)"
    gname_block  = "\n".join(meta["gen_names"])  if meta["gen_names"]  else "\t% (none)"
    gfuel_block  = "\n".join(meta["gen_fuels"])  if meta["gen_fuels"]  else "\t% (none)"

    text = (
        header +
        f"%% MATPOWER Case Format : Version 2\n"
        f"function mpc = {func_name}\n"
        f"    mpc.version = '2';\n\n"
        f"    %%-----  Power Flow Data  -----%%\n"
        f"    %% system MVA base\n"
        f"    mpc.baseMVA = {baseMVA};\n\n"
        f"    %% bus data\n"
        f"    %    bus_i    type    Pd    Qd    Gs    Bs    area    Vm    Va    baseKV    zone    Vmax    Vmin\n"
        f"    mpc.bus = [\n" + "\n".join(bus_lines) + "\n    ];\n\n"
        f"    %% generator data\n"
        f"    %    bus    Pg    Qg    Qmax    Qmin    Vg    mBase    status    Pmax    Pmin\n"
        f"    mpc.gen = [\n" + "\n".join(gen_lines) + "\n    ];\n\n"
        f"    %% branch data\n"
        f"    %    fbus    tbus    r    x    b    rateA    rateB    rateC    ratio    angle    status    angmin    angmax\n"
        f"    mpc.branch = [\n" + "\n".join(br_lines) + "\n    ];\n\n"
        f"    %%-----  OPF Data  -----%%\n"
        f"    %% cost data\n"
        f"    %    model    startup    shutdown    n    c2    c1    c0\n"
        f"    mpc.gencost = [\n" + "\n".join(cost_lines) + "\n    ];\n\n"
        f"    %% bus names\n"
        f"    mpc.bus_name = {{\n{names_block}\n    }};\n\n"
        f"    %% generator names\n"
        f"    mpc.gentype = {{\n{gname_block}\n    }};\n\n"
        f"    %% generator fuel types\n"
        f"    mpc.fuel = {{\n{gfuel_block}\n    }};\n\n"
        f"    %% bus coordinates  ('lat, lon')\n"
        f"    mpc.coordinates = {{\n{coords_block}\n    }};\n"
    )

    out_path.write_text(text)
    print(f"  Saved → {out_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    base_m = WORK_DIR / f"{region}_synthetic_grid.m"
    if not base_m.exists():
        raise FileNotFoundError(
            f"{base_m} not found. Run create_synthetic_grid.py first.")

    print("=" * 60)
    print(f"Loading: {base_m.name}")
    print("=" * 60)
    data = load_matpower(base_m)
    print(f"  Buses: {len(data['bus'])}  "
          f"Branches: {len(data['branch'])}  "
          f"Generators: {len(data['gen'])}")

    # ── DC feasibility ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DC FEASIBILITY  (eq:slack_dcopf)")
    print("=" * 60)
    dc_data = solve_dc_feasibility(data)
    dc_path = MPOWER_DIR / f"{region}_dc_feasible.m"
    write_matpower(dc_data, dc_path, "DC_feasible")

    # ── AC feasibility ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("AC FEASIBILITY  (eq:shunt_acopf, l1 shunt penalty)")
    print("=" * 60)
    ac_data = solve_ac_feasibility(dc_data)   # warm-start from DC solution
    ac_path = MPOWER_DIR / f"{region}_ac_feasible.m"
    write_matpower(ac_data, ac_path, "AC_feasible")

    print("\n" + "=" * 60)
    print("Done.")
    print(f"  DC-feasible model → {dc_path}")
    print(f"  AC-feasible model → {ac_path}")
    print("=" * 60)
