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


@dataclass(frozen=True)
class DemandDistanceResult:
    demands_a: Dict[str, float]
    demands_b: Dict[str, float]
    heads_a: Dict[str, float]
    heads_b: Dict[str, float]
    flows_a: Dict[str, float]
    flows_b: Dict[str, float]
    max_violation: float
    min_demand_viol: float
    success: bool
    objective: Optional[float] = None
    solver_status: Optional[str] = None
    best_bound: Optional[float] = None


def _compute_objective_value(
    demands_a: Dict[str, float],
    demands_b: Dict[str, float],
    norm_p: float,
) -> float:
    if not demands_a and not demands_b:
        return 0.0
    keys = sorted(set(demands_a.keys()) | set(demands_b.keys()))
    diffs = [abs(demands_a.get(k, 0.0) - demands_b.get(k, 0.0)) for k in keys]
    if not diffs:
        return 0.0
    if math.isinf(norm_p):
        return max(diffs)
    if norm_p <= 0:
        raise ValueError("norm_p must be positive.")
    return float(sum(d ** norm_p for d in diffs))


def _require_pyomo_ipopt() -> None:
    try:
        import pyomo.environ as pyo  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'pyomo' package is required for IPOPT. Install it with: pip install pyomo"
        ) from exc


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
    energy_head: Optional[float] = None,
    reservoir_node: Optional[str] = None,
    reservoir_outflow: Optional[float] = None,
    c_bounds: Optional[Dict[str, float]] = None,
    C_bounds: Optional[Dict[str, float]] = None,
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

    if c_bounds is not None and C_bounds is not None:
        for i, pipe_id in enumerate(pipes):
            lower[i] = c_bounds.get(pipe_id, -np.inf)
            upper[i] = C_bounds.get(pipe_id, np.inf)
    else:
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
        # Positive demand denotes consumption
        return {j: inflow[j] - outflow[j] for j in junctions}

    def demand_ineq_fun(x: np.ndarray) -> np.ndarray:
        q = x[:n_q]
        d = demand_from_q(q)
        return np.array([d[j] for j in junctions], dtype=float)

    demand_ineq = NonlinearConstraint(demand_ineq_fun, lb=np.zeros(n_h), ub=np.full(n_h, np.inf))

    energy_constraint = None
    if energy_head is not None:
        def energy_fun(x: np.ndarray) -> np.ndarray:
            q = x[:n_q]
            d = demand_from_q(q)
            total_demand = sum(d.values())
            energy = 0.0
            for idx, pipe_id in enumerate(pipes):
                q_e = q[idx]
                energy += pipe_resistances[pipe_id] * abs(q_e) ** 3
            return np.array([energy - energy_head * total_demand], dtype=float)

        energy_constraint = NonlinearConstraint(energy_fun, lb=-np.inf, ub=0.0)

    reservoir_constraint = None
    if reservoir_node is not None and reservoir_outflow is not None:
        def reservoir_fun(x: np.ndarray) -> np.ndarray:
            q = x[:n_q]
            net = 0.0
            for idx, pipe_id in enumerate(pipes):
                pipe = network.pipes[pipe_id]
                q_e = q[idx]
                if pipe.start_node == reservoir_node:
                    net += q_e
                if pipe.end_node == reservoir_node:
                    net -= q_e
            return np.array([net - reservoir_outflow], dtype=float)

        reservoir_constraint = NonlinearConstraint(reservoir_fun, lb=0.0, ub=0.0)

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

        constraint_list = [constraints, demand_ineq]
        if energy_constraint is not None:
            constraint_list.append(energy_constraint)
        if reservoir_constraint is not None:
            constraint_list.append(reservoir_constraint)

        res_min = minimize(
            obj_min,
            x0,
            method="trust-constr",
            constraints=constraint_list,
            bounds=bounds,
            options={"maxiter": 2000, "verbose": 0},
        )
        res_max = minimize(
            obj_max,
            x0,
            method="trust-constr",
            constraints=constraint_list,
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
    energy_head: Optional[float] = None,
    reservoir_node: Optional[str] = None,
    reservoir_outflow: Optional[float] = None,
    c_bounds: Optional[Dict[str, float]] = None,
    C_bounds: Optional[Dict[str, float]] = None,
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

    if c_bounds is not None and C_bounds is not None:
        for i, pipe_id in enumerate(pipes):
            lower[i] = c_bounds.get(pipe_id, -np.inf)
            upper[i] = C_bounds.get(pipe_id, np.inf)
    else:
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
        # Positive demand denotes consumption
        return {j: inflow[j] - outflow[j] for j in junctions}

    def demand_ineq_fun(x: np.ndarray) -> np.ndarray:
        q = x[:n_q]
        d = demand_from_q(q)
        return np.array([d[j] for j in junctions], dtype=float)

    demand_ineq = NonlinearConstraint(demand_ineq_fun, lb=np.zeros(n_h), ub=np.full(n_h, np.inf))

    energy_constraint = None
    if energy_head is not None:
        def energy_fun(x: np.ndarray) -> np.ndarray:
            q = x[:n_q]
            d = demand_from_q(q)
            total_demand = sum(d.values())
            energy = 0.0
            for idx, pipe_id in enumerate(pipes):
                q_e = q[idx]
                energy += pipe_resistances[pipe_id] * abs(q_e) ** 3
            return np.array([energy - energy_head * total_demand], dtype=float)

        energy_constraint = NonlinearConstraint(energy_fun, lb=-np.inf, ub=0.0)

    reservoir_constraint = None
    if reservoir_node is not None and reservoir_outflow is not None:
        def reservoir_fun(x: np.ndarray) -> np.ndarray:
            q = x[:n_q]
            net = 0.0
            for idx, pipe_id in enumerate(pipes):
                pipe = network.pipes[pipe_id]
                q_e = q[idx]
                if pipe.start_node == reservoir_node:
                    net += q_e
                if pipe.end_node == reservoir_node:
                    net -= q_e
            return np.array([net - reservoir_outflow], dtype=float)

        reservoir_constraint = NonlinearConstraint(reservoir_fun, lb=0.0, ub=0.0)

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

    constraint_list = [constraints, demand_ineq]
    if energy_constraint is not None:
        constraint_list.append(energy_constraint)
    if reservoir_constraint is not None:
        constraint_list.append(reservoir_constraint)

    res = minimize(
        obj_min,
        x0,
        method="trust-constr",
        constraints=constraint_list,
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
    energy_head: Optional[float] = None,
    reservoir_node: Optional[str] = None,
    reservoir_outflow: Optional[float] = None,
    c_bounds: Optional[Dict[str, float]] = None,
    C_bounds: Optional[Dict[str, float]] = None,
    initial_guess: Optional[SolverResult] = None,
    measurement_nodes: Optional[Iterable[str]] = None,
    measurement_heads_equal_only: bool = False,
) -> FeasibilityResult:
    _require_scipy()
    from scipy.optimize import Bounds, NonlinearConstraint, minimize

    reservoir_nodes = set(network.reservoirs.keys())
    fixed_heads = dict(sensor_heads)
    measurement_set = set(measurement_nodes or [])
    if measurement_heads_equal_only:
        fixed_heads = {k: v for k, v in fixed_heads.items() if k not in measurement_set}
    for res_id, res_node in network.reservoirs.items():
        if res_id not in fixed_heads:
            fixed_heads[res_id] = res_node.elevation_m

    junctions, pipes = _build_index_maps(network, reservoir_nodes)
    n_q = len(pipes)
    n_h = len(junctions)
    n_vars = n_q + n_h

    lower = np.full(n_vars, -np.inf, dtype=float)
    upper = np.full(n_vars, np.inf, dtype=float)

    if c_bounds is not None and C_bounds is not None:
        for i, pipe_id in enumerate(pipes):
            lower[i] = c_bounds.get(pipe_id, -np.inf)
            upper[i] = C_bounds.get(pipe_id, np.inf)
    else:
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
        # Positive demand denotes consumption
        return {j: inflow[j] - outflow[j] for j in junctions}

    def demand_ineq_fun(x: np.ndarray) -> np.ndarray:
        q = x[:n_q]
        d = demand_from_q(q)
        return np.array([d[j] for j in junctions], dtype=float)

    demand_ineq = NonlinearConstraint(demand_ineq_fun, lb=np.zeros(n_h), ub=np.full(n_h, np.inf))

    energy_constraint = None
    if energy_head is not None:
        def energy_fun(x: np.ndarray) -> np.ndarray:
            q = x[:n_q]
            d = demand_from_q(q)
            total_demand = sum(d.values())
            energy = 0.0
            for idx, pipe_id in enumerate(pipes):
                q_e = q[idx]
                energy += pipe_resistances[pipe_id] * abs(q_e) ** 3
            return np.array([energy - energy_head * total_demand], dtype=float)

        energy_constraint = NonlinearConstraint(energy_fun, lb=-np.inf, ub=0.0)

    reservoir_constraint = None
    if reservoir_node is not None and reservoir_outflow is not None:
        def reservoir_fun(x: np.ndarray) -> np.ndarray:
            q = x[:n_q]
            net = 0.0
            for idx, pipe_id in enumerate(pipes):
                pipe = network.pipes[pipe_id]
                q_e = q[idx]
                if pipe.start_node == reservoir_node:
                    net += q_e
                if pipe.end_node == reservoir_node:
                    net -= q_e
            return np.array([net - reservoir_outflow], dtype=float)

        reservoir_constraint = NonlinearConstraint(reservoir_fun, lb=0.0, ub=0.0)

    if initial_guess is None:
        x0 = np.zeros(n_vars, dtype=float)
    else:
        q0 = np.array([initial_guess.flows.get(pid, 0.0) for pid in pipes], dtype=float)
        h0 = np.array([initial_guess.heads.get(j, 0.0) for j in junctions], dtype=float)
        x0 = np.concatenate([q0, h0])

    def obj(x: np.ndarray) -> float:
        return float(np.sum(constraints_fun(x) ** 2))

    constraint_list = [constraints, demand_ineq]
    if energy_constraint is not None:
        constraint_list.append(energy_constraint)
    if reservoir_constraint is not None:
        constraint_list.append(reservoir_constraint)

    res = minimize(
        obj,
        x0,
        method="trust-constr",
        constraints=constraint_list,
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


def solve_max_demand_distance(
    network: NetworkData,
    sensor_heads: Dict[str, float],
    pipe_resistances: Dict[str, float],
    measurement_nodes: Optional[Iterable[str]] = None,
    measurement_heads_equal_only: bool = False,
    c_bounds: Optional[Dict[str, float]] = None,
    C_bounds: Optional[Dict[str, float]] = None,
    reservoir_node: Optional[str] = None,
    reservoir_outflow: Optional[float] = None,
    initial_guess: Optional[SolverResult] = None,
    norm_p: float = 2.0,
    demand_lb: float = 0.0,
    total_demand: Optional[float] = None,
) -> DemandDistanceResult:
    _require_scipy()
    from scipy.optimize import Bounds, NonlinearConstraint, minimize

    reservoir_nodes = set(network.reservoirs.keys())
    fixed_heads = dict(sensor_heads)
    measurement_set = set(measurement_nodes or [])
    if measurement_heads_equal_only:
        fixed_heads = {k: v for k, v in fixed_heads.items() if k not in measurement_set}
    for res_id, res_node in network.reservoirs.items():
        if res_id not in fixed_heads:
            fixed_heads[res_id] = res_node.elevation_m

    junctions, pipes = _build_index_maps(network, reservoir_nodes)
    n_q = len(pipes)
    n_h = len(junctions)

    # Variables: q_a, h_a, q_b, h_b
    n_vars = 2 * (n_q + n_h)
    lower = np.full(n_vars, -np.inf, dtype=float)
    upper = np.full(n_vars, np.inf, dtype=float)

    if c_bounds is not None and C_bounds is not None:
        for i, pipe_id in enumerate(pipes):
            lower[i] = c_bounds.get(pipe_id, -np.inf)
            upper[i] = C_bounds.get(pipe_id, np.inf)
            lower[n_q + n_h + i] = c_bounds.get(pipe_id, -np.inf)
            upper[n_q + n_h + i] = C_bounds.get(pipe_id, np.inf)

    bounds = Bounds(lower, upper)

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
        return {j: inflow[j] - outflow[j] for j in junctions}

    def constraints_fun(x: np.ndarray) -> np.ndarray:
        q_a = x[:n_q]
        h_a = {junctions[i]: x[n_q + i] for i in range(n_h)}
        q_b = x[n_q + n_h : 2 * n_q + n_h]
        h_b = {junctions[i]: x[2 * n_q + n_h + i] for i in range(n_h)}

        residuals: List[float] = []
        for idx, pipe_id in enumerate(pipes):
            pipe = network.pipes[pipe_id]
            r_e = pipe_resistances[pipe_id]
            q_e = q_a[idx]
            h_u = _head_value(pipe.start_node, h_a, fixed_heads)
            h_v = _head_value(pipe.end_node, h_a, fixed_heads)
            residuals.append(h_u - h_v - r_e * abs(q_e) * q_e)

        for idx, pipe_id in enumerate(pipes):
            pipe = network.pipes[pipe_id]
            r_e = pipe_resistances[pipe_id]
            q_e = q_b[idx]
            h_u = _head_value(pipe.start_node, h_b, fixed_heads)
            h_v = _head_value(pipe.end_node, h_b, fixed_heads)
            residuals.append(h_u - h_v - r_e * abs(q_e) * q_e)

        for node_id, head_val in fixed_heads.items():
            if node_id in junctions:
                residuals.append(h_a[node_id] - head_val)
                residuals.append(h_b[node_id] - head_val)

        if measurement_heads_equal_only:
            for node_id in measurement_set:
                if node_id in junctions and node_id not in fixed_heads:
                    residuals.append(h_a[node_id] - h_b[node_id])

        return np.array(residuals, dtype=float)

    n_constraints = 2 * n_q + 2 * sum(1 for n in fixed_heads if n in junctions)
    if measurement_heads_equal_only:
        n_constraints += sum(1 for n in measurement_set if n in junctions and n not in fixed_heads)
    constraints = NonlinearConstraint(constraints_fun, lb=np.zeros(n_constraints), ub=np.zeros(n_constraints))

    def demand_ineq_fun(x: np.ndarray) -> np.ndarray:
        q_a = x[:n_q]
        q_b = x[n_q + n_h : 2 * n_q + n_h]
        d_a = demand_from_q(q_a)
        d_b = demand_from_q(q_b)
        return np.array([d_a[j] for j in junctions] + [d_b[j] for j in junctions], dtype=float)

    demand_ineq = NonlinearConstraint(
        demand_ineq_fun,
        lb=np.full(2 * n_h, demand_lb),
        ub=np.full(2 * n_h, np.inf),
    )

    reservoir_constraint = None
    if reservoir_node is not None and reservoir_outflow is not None:
        def reservoir_fun(x: np.ndarray) -> np.ndarray:
            q_a = x[:n_q]
            q_b = x[n_q + n_h : 2 * n_q + n_h]
            net_a = 0.0
            net_b = 0.0
            for idx, pipe_id in enumerate(pipes):
                pipe = network.pipes[pipe_id]
                qa = q_a[idx]
                qb = q_b[idx]
                if pipe.start_node == reservoir_node:
                    net_a += qa
                    net_b += qb
                if pipe.end_node == reservoir_node:
                    net_a -= qa
                    net_b -= qb
            return np.array([net_a - reservoir_outflow, net_b - reservoir_outflow], dtype=float)

        reservoir_constraint = NonlinearConstraint(reservoir_fun, lb=np.zeros(2), ub=np.zeros(2))

    sign_constraint = None
    if initial_guess is not None:
        ref_pipe = None
        ref_val = 0.0
        for p in pipes:
            val = float(initial_guess.flows.get(p, 0.0))
            if abs(val) > abs(ref_val):
                ref_pipe = p
                ref_val = val
        if ref_pipe is not None and abs(ref_val) > 1e-8:
            ref_idx = pipes.index(ref_pipe)
            sign = 1.0 if ref_val >= 0.0 else -1.0

            def sign_fun(x: np.ndarray) -> np.ndarray:
                return np.array([sign * x[ref_idx]], dtype=float)

            sign_constraint = NonlinearConstraint(sign_fun, lb=np.zeros(1), ub=np.full(1, np.inf))

    if initial_guess is None:
        x0 = np.zeros(n_vars, dtype=float)
    else:
        q0 = np.array([initial_guess.flows.get(pid, 0.0) for pid in pipes], dtype=float)
        h0 = np.array([initial_guess.heads.get(j, 0.0) for j in junctions], dtype=float)
        # Break symmetry for the second solution to avoid zero-radius local optimum
        q1 = -q0
        h1 = h0 + 1e-2
        x0 = np.concatenate([q0, h0, q1, h1])

    def objective(x: np.ndarray) -> float:
        q_a = x[:n_q]
        q_b = x[n_q + n_h : 2 * n_q + n_h]
        d_a = demand_from_q(q_a)
        d_b = demand_from_q(q_b)
        diffs = np.array([d_a[j] - d_b[j] for j in junctions], dtype=float)
        if np.isinf(norm_p):
            return -float(np.max(np.abs(diffs)))
        if norm_p <= 0:
            raise ValueError("norm_p must be positive.")
        return -float(np.sum(np.abs(diffs) ** norm_p))

    constraint_list = [constraints, demand_ineq]
    if reservoir_constraint is not None:
        constraint_list.append(reservoir_constraint)
    if sign_constraint is not None:
        constraint_list.append(sign_constraint)
    if total_demand is not None:
        def total_demand_fun(x: np.ndarray) -> np.ndarray:
            q_a = x[:n_q]
            q_b = x[n_q + n_h : 2 * n_q + n_h]
            d_a = demand_from_q(q_a)
            d_b = demand_from_q(q_b)
            return np.array([
                sum(d_a.values()) - total_demand,
                sum(d_b.values()) - total_demand,
            ], dtype=float)

        total_constraint = NonlinearConstraint(total_demand_fun, lb=np.zeros(2), ub=np.zeros(2))
        constraint_list.append(total_constraint)

    res = minimize(
        objective,
        x0,
        method="trust-constr",
        constraints=constraint_list,
        bounds=bounds,
        options={"maxiter": 4000, "verbose": 0},
    )

    q_a = {pipes[i]: float(res.x[i]) for i in range(n_q)}
    h_a = {junctions[i]: float(res.x[n_q + i]) for i in range(n_h)}
    q_b = {pipes[i]: float(res.x[n_q + n_h + i]) for i in range(n_q)}
    h_b = {junctions[i]: float(res.x[2 * n_q + n_h + i]) for i in range(n_h)}
    for node_id, head_val in fixed_heads.items():
        h_a[node_id] = float(head_val)
        h_b[node_id] = float(head_val)

    d_a = demand_from_q(np.array([q_a[p] for p in pipes], dtype=float))
    d_b = demand_from_q(np.array([q_b[p] for p in pipes], dtype=float))

    max_violation = float(np.max(np.abs(constraints_fun(res.x))))
    min_demand_viol = float(np.min(demand_ineq_fun(res.x)))

    objective_val = -float(res.fun)
    status_msg = f"{res.status}: {res.message}"
    return DemandDistanceResult(
        demands_a=d_a,
        demands_b=d_b,
        heads_a=h_a,
        heads_b=h_b,
        flows_a=q_a,
        flows_b=q_b,
        max_violation=max_violation,
        min_demand_viol=min_demand_viol,
        success=bool(res.success),
        objective=objective_val,
        solver_status=status_msg,
    )


def solve_max_demand_distance_ipopt(
    network: NetworkData,
    sensor_heads: Dict[str, float],
    pipe_resistances: Dict[str, float],
    measurement_nodes: Optional[Iterable[str]] = None,
    measurement_heads_equal_only: bool = False,
    c_bounds: Optional[Dict[str, float]] = None,
    C_bounds: Optional[Dict[str, float]] = None,
    reservoir_node: Optional[str] = None,
    reservoir_outflow: Optional[float] = None,
    initial_guess: Optional[SolverResult] = None,
    norm_p: float = 2.0,
    demand_lb: float = 0.0,
    total_demand: Optional[float] = None,
    solver_name: str = "ipopt",
) -> DemandDistanceResult:
    _require_pyomo_ipopt()
    import pyomo.environ as pyo

    reservoir_nodes = set(network.reservoirs.keys())
    fixed_heads = dict(sensor_heads)
    measurement_set = set(measurement_nodes or [])
    if measurement_heads_equal_only:
        fixed_heads = {k: v for k, v in fixed_heads.items() if k not in measurement_set}
    for res_id, res_node in network.reservoirs.items():
        if res_id not in fixed_heads:
            fixed_heads[res_id] = res_node.elevation_m

    junctions, pipes = _build_index_maps(network, reservoir_nodes)

    m = pyo.ConcreteModel()
    m.J = pyo.Set(initialize=junctions)
    m.P = pyo.Set(initialize=pipes)

    m.qA = pyo.Var(m.P)
    m.hA = pyo.Var(m.J)
    m.qB = pyo.Var(m.P)
    m.hB = pyo.Var(m.J)

    if c_bounds is not None and C_bounds is not None:
        for p in pipes:
            m.qA[p].setlb(c_bounds.get(p, float("-inf")))
            m.qA[p].setub(C_bounds.get(p, float("inf")))
            m.qB[p].setlb(c_bounds.get(p, float("-inf")))
            m.qB[p].setub(C_bounds.get(p, float("inf")))

    def headloss_A_rule(mdl, p):
        pipe = network.pipes[p]
        h_u = fixed_heads.get(pipe.start_node, None)
        h_v = fixed_heads.get(pipe.end_node, None)
        if h_u is None:
            h_u = mdl.hA[pipe.start_node]
        if h_v is None:
            h_v = mdl.hA[pipe.end_node]
        r_e = pipe_resistances[p]
        return h_u - h_v == r_e * pyo.sqrt(mdl.qA[p] * mdl.qA[p]) * mdl.qA[p]

    def headloss_B_rule(mdl, p):
        pipe = network.pipes[p]
        h_u = fixed_heads.get(pipe.start_node, None)
        h_v = fixed_heads.get(pipe.end_node, None)
        if h_u is None:
            h_u = mdl.hB[pipe.start_node]
        if h_v is None:
            h_v = mdl.hB[pipe.end_node]
        r_e = pipe_resistances[p]
        return h_u - h_v == r_e * pyo.sqrt(mdl.qB[p] * mdl.qB[p]) * mdl.qB[p]

    m.headlossA = pyo.Constraint(m.P, rule=headloss_A_rule)
    m.headlossB = pyo.Constraint(m.P, rule=headloss_B_rule)

    if measurement_heads_equal_only:
        def meas_equal_rule(mdl, j):
            if j not in measurement_set or j in fixed_heads:
                return pyo.Constraint.Skip
            return mdl.hA[j] == mdl.hB[j]

        m.measurement_equal = pyo.Constraint(m.J, rule=meas_equal_rule)

    def demand_A(j):
        inflow = sum(m.qA[p] for p in pipes if network.pipes[p].end_node == j)
        outflow = sum(m.qA[p] for p in pipes if network.pipes[p].start_node == j)
        return inflow - outflow

    def demand_B(j):
        inflow = sum(m.qB[p] for p in pipes if network.pipes[p].end_node == j)
        outflow = sum(m.qB[p] for p in pipes if network.pipes[p].start_node == j)
        return inflow - outflow

    m.demA = pyo.Expression(m.J, rule=lambda mdl, j: demand_A(j))
    m.demB = pyo.Expression(m.J, rule=lambda mdl, j: demand_B(j))

    m.demA_nonneg = pyo.Constraint(m.J, rule=lambda mdl, j: mdl.demA[j] >= demand_lb)
    m.demB_nonneg = pyo.Constraint(m.J, rule=lambda mdl, j: mdl.demB[j] >= demand_lb)

    if total_demand is not None:
        m.demA_total = pyo.Constraint(expr=sum(m.demA[j] for j in junctions) == total_demand)
        m.demB_total = pyo.Constraint(expr=sum(m.demB[j] for j in junctions) == total_demand)

    if reservoir_node is not None and reservoir_outflow is not None:
        def res_out_A(mdl):
            net = sum(mdl.qA[p] for p in pipes if network.pipes[p].start_node == reservoir_node) \
                - sum(mdl.qA[p] for p in pipes if network.pipes[p].end_node == reservoir_node)
            return net == reservoir_outflow

        def res_out_B(mdl):
            net = sum(mdl.qB[p] for p in pipes if network.pipes[p].start_node == reservoir_node) \
                - sum(mdl.qB[p] for p in pipes if network.pipes[p].end_node == reservoir_node)
            return net == reservoir_outflow

        m.resA = pyo.Constraint(rule=res_out_A)
        m.resB = pyo.Constraint(rule=res_out_B)

    if initial_guess is not None:
        ref_pipe = None
        ref_val = 0.0
        for p in pipes:
            val = float(initial_guess.flows.get(p, 0.0))
            if abs(val) > abs(ref_val):
                ref_pipe = p
                ref_val = val
        if ref_pipe is not None and abs(ref_val) > 1e-8:
            sign = 1.0 if ref_val >= 0.0 else -1.0
            m.signA = pyo.Constraint(expr=sign * m.qA[ref_pipe] >= 0.0)

    if norm_p <= 0:
        raise ValueError("norm_p must be positive.")

    if norm_p == float("inf"):
        m.t = pyo.Var(bounds=(0.0, None))
        def inf_norm_rule(mdl, j):
            diff = mdl.demA[j] - mdl.demB[j]
            return mdl.t >= pyo.sqrt(diff * diff)

        m.inf_norm = pyo.Constraint(m.J, rule=inf_norm_rule)
        m.obj = pyo.Objective(expr=-m.t, sense=pyo.minimize)
    else:
        m.obj = pyo.Objective(
            expr=-sum(pyo.sqrt((m.demA[j] - m.demB[j]) ** 2) ** norm_p for j in junctions),
            sense=pyo.minimize,
        )

    if initial_guess is not None:
        for p in pipes:
            q_val = initial_guess.flows.get(p, 0.0)
            q_val_b = -q_val
            if c_bounds is not None and C_bounds is not None:
                lb = c_bounds.get(p, float("-inf"))
                ub = C_bounds.get(p, float("inf"))
                q_val = min(max(q_val, lb), ub)
                q_val_b = min(max(q_val_b, lb), ub)
            m.qA[p].value = q_val
            m.qB[p].value = q_val_b
        for j in junctions:
            m.hA[j].value = initial_guess.heads.get(j, 0.0)
            m.hB[j].value = initial_guess.heads.get(j, 0.0) + 1e-2

    solver = pyo.SolverFactory(solver_name)
    result = solver.solve(m, tee=False, load_solutions=False)
    term = getattr(result.solver, "termination_condition", None)
    ok_terms = {pyo.TerminationCondition.optimal, pyo.TerminationCondition.locallyOptimal}
    if term in ok_terms:
        m.solutions.load_from(result)

    q_a = {p: float(pyo.value(m.qA[p])) for p in pipes}
    q_b = {p: float(pyo.value(m.qB[p])) for p in pipes}
    h_a = {j: float(pyo.value(m.hA[j])) for j in junctions}
    h_b = {j: float(pyo.value(m.hB[j])) for j in junctions}
    for node_id, head_val in fixed_heads.items():
        h_a[node_id] = float(head_val)
        h_b[node_id] = float(head_val)

    d_a = {j: float(pyo.value(m.demA[j])) for j in junctions}
    d_b = {j: float(pyo.value(m.demB[j])) for j in junctions}

    success_flag = term in ok_terms

    objective_val = _compute_objective_value(d_a, d_b, norm_p)
    status_msg = str(term) if term is not None else None
    return DemandDistanceResult(
        demands_a=d_a,
        demands_b=d_b,
        heads_a=h_a,
        heads_b=h_b,
        flows_a=q_a,
        flows_b=q_b,
        max_violation=0.0,
        min_demand_viol=min(d_a.values()) if d_a else 0.0,
        success=bool(success_flag),
        objective=objective_val,
        solver_status=status_msg,
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
