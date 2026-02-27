from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import math
import numpy as np

from step1_io import NetworkData, PipeData


@dataclass(frozen=True)
class SolverResult:
    status: str
    demands: Dict[str, float]
    heads: Dict[str, float]
    flows: Dict[str, float]


@dataclass(frozen=True)
class DemandBounds:
    d_min: Dict[str, float]
    d_max: Dict[str, float]
    status: str
    diagnostics: Dict[str, Dict[str, float]]


@dataclass(frozen=True)
class SingleNodeResult:
    node_id: str
    objective: str
    demands: Dict[str, float]
    heads: Dict[str, float]
    flows: Dict[str, float]
    max_violation: float
    min_demand_viol: float
    success: bool


@dataclass(frozen=True)
class FeasibilityResult:
    flows: Dict[str, float]
    heads: Dict[str, float]
    max_violation: float
    min_demand_viol: float
    success: bool


def _require_scipy() -> None:
    try:
        import scipy  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'scipy' package is required for the nonlinear solver. "
            "Install it with: pip install scipy"
        ) from exc


def _build_index_maps(
    network: NetworkData,
    reservoir_nodes: Iterable[str],
) -> Tuple[List[str], List[str]]:
    junctions = [n for n in network.junctions.keys() if n not in reservoir_nodes]
    return junctions, list(network.pipes.keys())


def _head_value(node_id: str, heads: Dict[str, float], fixed_heads: Dict[str, float]) -> float:
    if node_id in fixed_heads:
        return fixed_heads[node_id]
    return heads[node_id]


def _constraints_residuals(
    x: np.ndarray,
    junctions: List[str],
    pipes: List[str],
    network: NetworkData,
    fixed_heads: Dict[str, float],
    pipe_resistances: Dict[str, float],
) -> np.ndarray:
    n_q = len(pipes)
    n_h = len(junctions)
    q = x[:n_q]
    h = {junctions[i]: x[n_q + i] for i in range(n_h)}

    residuals: List[float] = []

    # Headloss constraints per pipe
    for idx, pipe_id in enumerate(pipes):
        pipe = network.pipes[pipe_id]
        q_e = q[idx]
        h_u = _head_value(pipe.start_node, h, fixed_heads)
        h_v = _head_value(pipe.end_node, h, fixed_heads)
        r_e = pipe_resistances[pipe_id]
        residuals.append(h_u - h_v - r_e * abs(q_e) * q_e)

    # Fixed heads at measurement nodes that are junctions
    for node_id, head_val in fixed_heads.items():
        if node_id in junctions:
            residuals.append(h[node_id] - head_val)

    return np.array(residuals, dtype=float)


def solve_demand_bounds(
    network: NetworkData,
    sensor_heads: Dict[str, float],
    pipe_primary: Iterable[str],
    pipe_secondary: Iterable[str],
    c_secondary: Dict[str, float],
    C_primary: Dict[str, float],
    pipe_resistances: Dict[str, float],
    preferred_flow_sign: Optional[Dict[str, float]] = None,
    initial_guess: Optional[SolverResult] = None,
) -> DemandBounds:
    """
    Solve min/max demand at each junction with fixed heads at sensors.
    """
    _require_scipy()
    from scipy.optimize import Bounds, NonlinearConstraint, minimize

    reservoir_nodes = set(network.reservoirs.keys())
    fixed_heads = dict(sensor_heads)
    for res_id, res_node in network.reservoirs.items():
        if res_id not in fixed_heads:
            fixed_heads[res_id] = res_node.elevation_m

    junctions, pipes = _build_index_maps(network, reservoir_nodes)
    n_q = len(pipes)
    n_h = len(junctions)
    n_vars = n_q + n_h

    # Variable bounds
    lower = np.full(n_vars, -np.inf, dtype=float)
    upper = np.full(n_vars, np.inf, dtype=float)

    pipe_primary_set = set(pipe_primary)
    pipe_secondary_set = set(pipe_secondary)
    for i, pipe_id in enumerate(pipes):
        if pipe_id in pipe_primary_set:
            C_e = C_primary[pipe_id]
            sign = 1.0
            if preferred_flow_sign is not None and pipe_id in preferred_flow_sign:
                sign = 1.0 if preferred_flow_sign[pipe_id] >= 0 else -1.0
            if sign >= 0:
                lower[i] = C_e
            else:
                upper[i] = -C_e
        elif pipe_id in pipe_secondary_set:
            c_e = c_secondary[pipe_id]
            lower[i] = -c_e
            upper[i] = c_e

    bounds = Bounds(lower, upper)

    def constraints_fun(x: np.ndarray) -> np.ndarray:
        return _constraints_residuals(x, junctions, pipes, network, fixed_heads, pipe_resistances)

    n_constraints = n_q + sum(1 for n in fixed_heads if n in junctions)
    constraints = NonlinearConstraint(constraints_fun, lb=np.zeros(n_constraints), ub=np.zeros(n_constraints))

    def demand_from_q(q: np.ndarray) -> Dict[str, float]:
        inflow = {j: 0.0 for j in junctions}
        outflow = {j: 0.0 for j in junctions}
        for idx, pipe_id in enumerate(pipes):
            pipe = network.pipes[pipe_id]
            q_e = q[idx]
            if pipe.end_node in inflow:
                inflow[pipe.end_node] += q_e
            if pipe.start_node in outflow:
                outflow[pipe.start_node] += q_e
        # Negative demand denotes consumption
        return {j: outflow[j] - inflow[j] for j in junctions}

    def demand_ineq_fun(x: np.ndarray) -> np.ndarray:
        q = x[:n_q]
        d = demand_from_q(q)
        return np.array([d[j] for j in junctions], dtype=float)

    demand_ineq = NonlinearConstraint(demand_ineq_fun, lb=np.full(n_h, -np.inf), ub=np.zeros(n_h))

    if initial_guess is None:
        x0 = np.zeros(n_vars, dtype=float)
    else:
        q0 = np.array([initial_guess.flows.get(pid, 0.0) for pid in pipes], dtype=float)
        h0 = np.array([initial_guess.heads.get(j, 0.0) for j in junctions], dtype=float)
        x0 = np.concatenate([q0, h0])

    d_min: Dict[str, float] = {}
    d_max: Dict[str, float] = {}
    diagnostics: Dict[str, Dict[str, float]] = {}

    for target_idx, node_id in enumerate(junctions):
        def obj_min(x: np.ndarray) -> float:
            q = x[:n_q]
            d = demand_from_q(q)
            return d[junctions[target_idx]]

        def obj_max(x: np.ndarray) -> float:
            q = x[:n_q]
            d = demand_from_q(q)
            return -d[junctions[target_idx]]

        res_min = minimize(
            obj_min,
            x0,
            method="trust-constr",
            constraints=[constraints, demand_ineq],
            bounds=bounds,
            options={"maxiter": 2000, "verbose": 0},
        )
        res_max = minimize(
            obj_max,
            x0,
            method="trust-constr",
            constraints=[constraints, demand_ineq],
            bounds=bounds,
            options={"maxiter": 2000, "verbose": 0},
        )

        d_min[node_id] = float(demand_from_q(res_min.x[:n_q])[junctions[target_idx]])
        d_max[node_id] = float(demand_from_q(res_max.x[:n_q])[junctions[target_idx]])

        res_min_violation = float(np.max(np.abs(constraints_fun(res_min.x))))
        res_max_violation = float(np.max(np.abs(constraints_fun(res_max.x))))
        res_min_demand_viol = float(np.min(demand_ineq_fun(res_min.x)))
        res_max_demand_viol = float(np.min(demand_ineq_fun(res_max.x)))
        diagnostics[node_id] = {
            "min_success": float(res_min.success),
            "max_success": float(res_max.success),
            "min_violation": res_min_violation,
            "max_violation": res_max_violation,
            "min_demand_viol": res_min_demand_viol,
            "max_demand_viol": res_max_demand_viol,
        }

    status = "ok"
    if any(v["min_violation"] > 1e-4 or v["max_violation"] > 1e-4 for v in diagnostics.values()):
        status = "warn"
    return DemandBounds(d_min=d_min, d_max=d_max, status=status, diagnostics=diagnostics)


def solve_single_node_min(
    network: NetworkData,
    target_node: str,
    sensor_heads: Dict[str, float],
    pipe_primary: Iterable[str],
    pipe_secondary: Iterable[str],
    c_secondary: Dict[str, float],
    C_primary: Dict[str, float],
    pipe_resistances: Dict[str, float],
    preferred_flow_sign: Optional[Dict[str, float]] = None,
    initial_guess: Optional[SolverResult] = None,
) -> SingleNodeResult:
    _require_scipy()
    from scipy.optimize import Bounds, NonlinearConstraint, minimize

    reservoir_nodes = set(network.reservoirs.keys())
    fixed_heads = dict(sensor_heads)
    for res_id, res_node in network.reservoirs.items():
        if res_id not in fixed_heads:
            fixed_heads[res_id] = res_node.elevation_m

    junctions, pipes = _build_index_maps(network, reservoir_nodes)
    if target_node not in junctions:
        raise ValueError("target_node must be a junction node")

    n_q = len(pipes)
    n_h = len(junctions)
    n_vars = n_q + n_h

    lower = np.full(n_vars, -np.inf, dtype=float)
    upper = np.full(n_vars, np.inf, dtype=float)

    pipe_primary_set = set(pipe_primary)
    pipe_secondary_set = set(pipe_secondary)
    for i, pipe_id in enumerate(pipes):
        if pipe_id in pipe_primary_set:
            C_e = C_primary[pipe_id]
            sign = 1.0
            if preferred_flow_sign is not None and pipe_id in preferred_flow_sign:
                sign = 1.0 if preferred_flow_sign[pipe_id] >= 0 else -1.0
            if sign >= 0:
                lower[i] = C_e
            else:
                upper[i] = -C_e
        elif pipe_id in pipe_secondary_set:
            c_e = c_secondary[pipe_id]
            lower[i] = -c_e
            upper[i] = c_e

    bounds = Bounds(lower, upper)

    def constraints_fun(x: np.ndarray) -> np.ndarray:
        return _constraints_residuals(x, junctions, pipes, network, fixed_heads, pipe_resistances)

    n_constraints = n_q + sum(1 for n in fixed_heads if n in junctions)
    constraints = NonlinearConstraint(constraints_fun, lb=np.zeros(n_constraints), ub=np.zeros(n_constraints))

    def demand_from_q(q: np.ndarray) -> Dict[str, float]:
        inflow = {j: 0.0 for j in junctions}
        outflow = {j: 0.0 for j in junctions}
        for idx, pipe_id in enumerate(pipes):
            pipe = network.pipes[pipe_id]
            q_e = q[idx]
            if pipe.end_node in inflow:
                inflow[pipe.end_node] += q_e
            if pipe.start_node in outflow:
                outflow[pipe.start_node] += q_e
        # Negative demand denotes consumption
        return {j: outflow[j] - inflow[j] for j in junctions}

    def demand_ineq_fun(x: np.ndarray) -> np.ndarray:
        q = x[:n_q]
        d = demand_from_q(q)
        return np.array([d[j] for j in junctions], dtype=float)

    demand_ineq = NonlinearConstraint(demand_ineq_fun, lb=np.full(n_h, -np.inf), ub=np.zeros(n_h))

    if initial_guess is None:
        x0 = np.zeros(n_vars, dtype=float)
    else:
        q0 = np.array([initial_guess.flows.get(pid, 0.0) for pid in pipes], dtype=float)
        h0 = np.array([initial_guess.heads.get(j, 0.0) for j in junctions], dtype=float)
        x0 = np.concatenate([q0, h0])

    target_idx = junctions.index(target_node)

    def obj_min(x: np.ndarray) -> float:
        q = x[:n_q]
        d = demand_from_q(q)
        return d[junctions[target_idx]]

    res = minimize(
        obj_min,
        x0,
        method="trust-constr",
        constraints=[constraints, demand_ineq],
        bounds=bounds,
        options={"maxiter": 3000, "verbose": 0},
    )

    q_sol = {pipes[i]: float(res.x[i]) for i in range(n_q)}
    h_sol = {junctions[i]: float(res.x[n_q + i]) for i in range(n_h)}
    for node_id, head_val in fixed_heads.items():
        h_sol[node_id] = float(head_val)
    d_sol = demand_from_q(np.array([q_sol[p] for p in pipes], dtype=float))

    max_violation = float(np.max(np.abs(constraints_fun(res.x))))
    min_demand_viol = float(np.min(demand_ineq_fun(res.x)))

    return SingleNodeResult(
        node_id=target_node,
        objective="min",
        demands=d_sol,
        heads=h_sol,
        flows=q_sol,
        max_violation=max_violation,
        min_demand_viol=min_demand_viol,
        success=bool(res.success),
    )


def solve_feasibility(
    network: NetworkData,
    sensor_heads: Dict[str, float],
    pipe_primary: Iterable[str],
    pipe_secondary: Iterable[str],
    c_secondary: Dict[str, float],
    C_primary: Dict[str, float],
    pipe_resistances: Dict[str, float],
    preferred_flow_sign: Optional[Dict[str, float]] = None,
    initial_guess: Optional[SolverResult] = None,
) -> FeasibilityResult:
    _require_scipy()
    from scipy.optimize import Bounds, NonlinearConstraint, minimize

    reservoir_nodes = set(network.reservoirs.keys())
    fixed_heads = dict(sensor_heads)
    for res_id, res_node in network.reservoirs.items():
        if res_id not in fixed_heads:
            fixed_heads[res_id] = res_node.elevation_m

    junctions, pipes = _build_index_maps(network, reservoir_nodes)
    n_q = len(pipes)
    n_h = len(junctions)
    n_vars = n_q + n_h

    lower = np.full(n_vars, -np.inf, dtype=float)
    upper = np.full(n_vars, np.inf, dtype=float)

    pipe_primary_set = set(pipe_primary)
    pipe_secondary_set = set(pipe_secondary)
    for i, pipe_id in enumerate(pipes):
        if pipe_id in pipe_primary_set:
            C_e = C_primary[pipe_id]
            sign = 1.0
            if preferred_flow_sign is not None and pipe_id in preferred_flow_sign:
                sign = 1.0 if preferred_flow_sign[pipe_id] >= 0 else -1.0
            if sign >= 0:
                lower[i] = C_e
            else:
                upper[i] = -C_e
        elif pipe_id in pipe_secondary_set:
            c_e = c_secondary[pipe_id]
            lower[i] = -c_e
            upper[i] = c_e

    bounds = Bounds(lower, upper)

    def constraints_fun(x: np.ndarray) -> np.ndarray:
        return _constraints_residuals(x, junctions, pipes, network, fixed_heads, pipe_resistances)

    n_constraints = n_q + sum(1 for n in fixed_heads if n in junctions)
    constraints = NonlinearConstraint(constraints_fun, lb=np.zeros(n_constraints), ub=np.zeros(n_constraints))

    def demand_from_q(q: np.ndarray) -> Dict[str, float]:
        inflow = {j: 0.0 for j in junctions}
        outflow = {j: 0.0 for j in junctions}
        for idx, pipe_id in enumerate(pipes):
            pipe = network.pipes[pipe_id]
            q_e = q[idx]
            if pipe.end_node in inflow:
                inflow[pipe.end_node] += q_e
            if pipe.start_node in outflow:
                outflow[pipe.start_node] += q_e
        # Negative demand denotes consumption
        return {j: outflow[j] - inflow[j] for j in junctions}

    def demand_ineq_fun(x: np.ndarray) -> np.ndarray:
        q = x[:n_q]
        d = demand_from_q(q)
        return np.array([d[j] for j in junctions], dtype=float)

    demand_ineq = NonlinearConstraint(demand_ineq_fun, lb=np.full(n_h, -np.inf), ub=np.zeros(n_h))

    if initial_guess is None:
        x0 = np.zeros(n_vars, dtype=float)
    else:
        q0 = np.array([initial_guess.flows.get(pid, 0.0) for pid in pipes], dtype=float)
        h0 = np.array([initial_guess.heads.get(j, 0.0) for j in junctions], dtype=float)
        x0 = np.concatenate([q0, h0])

    def obj(x: np.ndarray) -> float:
        return float(np.sum(constraints_fun(x) ** 2))

    res = minimize(
        obj,
        x0,
        method="trust-constr",
        constraints=[constraints, demand_ineq],
        bounds=bounds,
        options={"maxiter": 4000, "verbose": 0},
    )

    q_sol = {pipes[i]: float(res.x[i]) for i in range(n_q)}
    h_sol = {junctions[i]: float(res.x[n_q + i]) for i in range(n_h)}
    for node_id, head_val in fixed_heads.items():
        h_sol[node_id] = float(head_val)

    max_violation = float(np.max(np.abs(constraints_fun(res.x))))
    min_demand_viol = float(np.min(demand_ineq_fun(res.x)))

    return FeasibilityResult(
        flows=q_sol,
        heads=h_sol,
        max_violation=max_violation,
        min_demand_viol=min_demand_viol,
        success=bool(res.success),
    )


def solve_inverse_problem(
    network: NetworkData,
    sensor_nodes: Iterable[str],
    sensor_heads: Dict[str, float],
    pipe_primary: Iterable[str],
    pipe_secondary: Iterable[str],
    c_secondary: Dict[str, float],
    C_primary: Dict[str, float],
) -> SolverResult:
    """
    Placeholder for the nonlinear optimization solver.
    """
    raise NotImplementedError("Nonlinear solver not implemented yet.")
