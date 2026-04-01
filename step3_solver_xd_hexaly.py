from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import heapq
import math

from step1_io import NetworkData
from step3_solver import DemandDistanceResult, SolverResult, _build_index_maps, _compute_objective_value


def _require_hexaly() -> None:
    try:
        import hexaly.optimizer  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'hexaly' package is required for the Hexaly solver. "
            "Install it with: pip install hexaly -i https://pip.hexaly.com"
        ) from exc


def _estimate_flow_bounds(
    pipes: List[str],
    c_bounds: Optional[Dict[str, float]],
    C_bounds: Optional[Dict[str, float]],
    initial_guess: Optional[SolverResult],
) -> Dict[str, Tuple[float, float]]:
    if c_bounds is not None and C_bounds is not None:
        return {p: (c_bounds.get(p, -1e3), C_bounds.get(p, 1e3)) for p in pipes}

    scale = 1.0
    if initial_guess is not None:
        for p in pipes:
            scale = max(scale, abs(float(initial_guess.flows.get(p, 0.0))))
    bound = max(1.0, 10.0 * scale)
    return {p: (-bound, bound) for p in pipes}


def _phi(r_e: float, q: float, n_exp: float) -> float:
    return r_e * q * (abs(q) ** (n_exp - 1.0))


def _flow_to_head_bounds(
    pipe_resistances: Dict[str, float],
    flow_bounds: Dict[str, Tuple[float, float]],
    n_exp: float,
) -> Dict[str, Tuple[float, float]]:
    x_bounds: Dict[str, Tuple[float, float]] = {}
    for pipe_id, (lb, ub) in flow_bounds.items():
        r_e = pipe_resistances[pipe_id]
        x_lb = _phi(r_e, lb, n_exp)
        x_ub = _phi(r_e, ub, n_exp)
        x_bounds[pipe_id] = (min(x_lb, x_ub), max(x_lb, x_ub))
    return x_bounds


def _edge_coeff(
    pipe_start: str,
    pipe_end: str,
    u: str,
    v: str,
) -> int:
    if pipe_start == u and pipe_end == v:
        return 1
    if pipe_start == v and pipe_end == u:
        return -1
    raise ValueError("Edge orientation does not match pipe endpoints.")


def _build_cycle_matrix(
    network: NetworkData,
    pipes: List[str],
    mode: str = "graph",
) -> List[Dict[str, int]]:
    import networkx as nx

    edge_to_pipe: Dict[frozenset[str], str] = {}
    for pipe_id in pipes:
        pipe = network.pipes[pipe_id]
        key = frozenset((pipe.start_node, pipe.end_node))
        edge_to_pipe[key] = pipe_id

    G = nx.Graph()
    for pipe_id in pipes:
        pipe = network.pipes[pipe_id]
        G.add_edge(pipe.start_node, pipe.end_node)

    if mode == "planar":
        cycles = _build_planar_face_cycles(G, network)
    else:
        cycles = nx.cycle_basis(G)
    rows: List[Dict[str, int]] = []
    for cycle in cycles:
        row: Dict[str, int] = {}
        if len(cycle) < 2:
            continue
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            key = frozenset((u, v))
            pipe_id = edge_to_pipe.get(key)
            if pipe_id is None:
                continue
            pipe = network.pipes[pipe_id]
            coeff = _edge_coeff(pipe.start_node, pipe.end_node, u, v)
            row[pipe_id] = row.get(pipe_id, 0) + coeff
        if row:
            rows.append(row)
    return rows


def _build_planar_face_cycles(G, network: NetworkData) -> List[List[str]]:
    import networkx as nx

    coords = {
        node_id: node.coordinates
        for node_id, node in network.nodes.items()
        if node.coordinates is not None
    }
    if len(coords) < len(G.nodes):
        return nx.cycle_basis(G)

    is_planar, embedding = nx.check_planarity(G)
    if not is_planar:
        return nx.cycle_basis(G)

    try:
        faces = list(embedding.faces())
    except Exception:
        return nx.cycle_basis(G)

    if not faces:
        return nx.cycle_basis(G)

    def _face_area(face: List[str]) -> float:
        area = 0.0
        for i in range(len(face)):
            x1, y1 = coords[face[i]]
            x2, y2 = coords[face[(i + 1) % len(face)]]
            area += x1 * y2 - x2 * y1
        return 0.5 * area

    areas = [abs(_face_area(face)) for face in faces]
    outer_idx = max(range(len(faces)), key=lambda i: areas[i])

    return [face for i, face in enumerate(faces) if i != outer_idx]


def _build_path_matrix(
    network: NetworkData,
    pipes: List[str],
    pipe_resistances: Dict[str, float],
    reservoir_node: str,
    measurement_nodes: Iterable[str],
) -> List[Dict[str, int]]:
    import networkx as nx

    edge_to_pipe: Dict[frozenset[str], str] = {}
    for pipe_id in pipes:
        pipe = network.pipes[pipe_id]
        key = frozenset((pipe.start_node, pipe.end_node))
        edge_to_pipe[key] = pipe_id

    G = nx.Graph()
    for pipe_id in pipes:
        pipe = network.pipes[pipe_id]
        weight = pipe_resistances[pipe_id]
        G.add_edge(pipe.start_node, pipe.end_node, weight=weight)

    rows: List[Dict[str, int]] = []
    for meas in measurement_nodes:
        if meas not in G.nodes or meas == reservoir_node:
            continue
        try:
            path = nx.shortest_path(G, reservoir_node, meas, weight="weight")
        except nx.NetworkXNoPath:
            continue
        row: Dict[str, int] = {}
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            key = frozenset((u, v))
            pipe_id = edge_to_pipe.get(key)
            if pipe_id is None:
                continue
            pipe = network.pipes[pipe_id]
            coeff = _edge_coeff(pipe.start_node, pipe.end_node, u, v)
            row[pipe_id] = row.get(pipe_id, 0) + coeff
        if row:
            rows.append(row)
    return rows


def _compute_heads_from_x(
    network: NetworkData,
    pipes: List[str],
    x_vals: Dict[str, float],
    pipe_resistances: Dict[str, float],
    reservoir_node: Optional[str],
    reservoir_head: Optional[float],
) -> Dict[str, float]:
    import networkx as nx

    if reservoir_node is None:
        return {}

    edge_to_pipe: Dict[frozenset[str], str] = {}
    for pipe_id in pipes:
        pipe = network.pipes[pipe_id]
        edge_to_pipe[frozenset((pipe.start_node, pipe.end_node))] = pipe_id

    G = nx.Graph()
    for pipe_id in pipes:
        pipe = network.pipes[pipe_id]
        weight = pipe_resistances[pipe_id]
        G.add_edge(pipe.start_node, pipe.end_node, weight=weight)

    heads: Dict[str, float] = {}
    base_head = 0.0 if reservoir_head is None else float(reservoir_head)
    heads[reservoir_node] = base_head

    for node_id in G.nodes:
        if node_id == reservoir_node:
            continue
        try:
            path = nx.shortest_path(G, reservoir_node, node_id, weight="weight")
        except nx.NetworkXNoPath:
            continue
        h = base_head
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            pipe_id = edge_to_pipe.get(frozenset((u, v)))
            if pipe_id is None:
                continue
            pipe = network.pipes[pipe_id]
            x = x_vals.get(pipe_id, 0.0)
            if pipe.start_node == u and pipe.end_node == v:
                h = h - x
            else:
                h = h + x
        heads[node_id] = h
    return heads


def _sum_expr(model, exprs: List[object]):
    if not exprs:
        return 0.0
    return model.sum(*exprs)


def check_xd_feasibility_from_flows(
    network: NetworkData,
    pipe_resistances: Dict[str, float],
    flows: Dict[str, float],
    c_bounds: Optional[Dict[str, float]] = None,
    C_bounds: Optional[Dict[str, float]] = None,
    reservoir_node: Optional[str] = None,
    measurement_nodes: Optional[Iterable[str]] = None,
    demand_lb: float = 0.0,
    total_demand: Optional[float] = None,
    reservoir_outflow: Optional[float] = None,
    headloss_n: float = 2.0,
    cycle_basis_mode: str = "graph",
) -> Dict[str, float]:
    reservoir_nodes = set(network.reservoirs.keys())
    junctions, pipes = _build_index_maps(network, reservoir_nodes)

    x_vals = {p: _phi(pipe_resistances[p], flows.get(p, 0.0), headloss_n) for p in pipes}
    x_bounds = None
    if c_bounds is not None and C_bounds is not None:
        flow_bounds = {p: (c_bounds.get(p, -1e3), C_bounds.get(p, 1e3)) for p in pipes}
        x_bounds = _flow_to_head_bounds(pipe_resistances, flow_bounds, headloss_n)

    cycle_rows = _build_cycle_matrix(network, pipes, mode=cycle_basis_mode)
    path_rows: List[Dict[str, int]] = []
    if reservoir_node is not None and measurement_nodes:
        path_rows = _build_path_matrix(
            network,
            pipes,
            pipe_resistances,
            reservoir_node,
            measurement_nodes,
        )

    cycle_max = 0.0
    for row in cycle_rows:
        val = 0.0
        for p, coef in row.items():
            val += coef * x_vals.get(p, 0.0)
        cycle_max = max(cycle_max, abs(val))

    path_max = 0.0
    for row in path_rows:
        val = 0.0
        for p, coef in row.items():
            x = x_vals.get(p, 0.0)
            val += coef * (x - x)
        path_max = max(path_max, abs(val))

    demand_vals: Dict[str, float] = {j: 0.0 for j in junctions}
    for p in pipes:
        pipe = network.pipes[p]
        q = flows.get(p, 0.0)
        if pipe.end_node in demand_vals:
            demand_vals[pipe.end_node] += q
        if pipe.start_node in demand_vals:
            demand_vals[pipe.start_node] -= q

    min_demand_slack = 0.0
    if demand_vals:
        min_demand_slack = min(demand_vals.values()) - demand_lb

    total_demand_res = 0.0
    if total_demand is not None:
        total_demand_res = sum(demand_vals.values()) - total_demand

    reservoir_outflow_res = 0.0
    if reservoir_node is not None and reservoir_outflow is not None:
        net = 0.0
        for p in pipes:
            pipe = network.pipes[p]
            q = flows.get(p, 0.0)
            if pipe.start_node == reservoir_node:
                net += q
            if pipe.end_node == reservoir_node:
                net -= q
        reservoir_outflow_res = net - reservoir_outflow

    flow_bound_max = 0.0
    if c_bounds is not None and C_bounds is not None:
        for p in pipes:
            q = flows.get(p, 0.0)
            lb = c_bounds.get(p, -math.inf)
            ub = C_bounds.get(p, math.inf)
            if q < lb:
                flow_bound_max = max(flow_bound_max, lb - q)
            if q > ub:
                flow_bound_max = max(flow_bound_max, q - ub)

    x_bound_max = 0.0
    if x_bounds is not None:
        for p in pipes:
            x = x_vals.get(p, 0.0)
            lb, ub = x_bounds.get(p, (-math.inf, math.inf))
            if x < lb:
                x_bound_max = max(x_bound_max, lb - x)
            if x > ub:
                x_bound_max = max(x_bound_max, x - ub)

    return {
        "cycle_max_abs": float(cycle_max),
        "path_max_abs": float(path_max),
        "min_demand_slack": float(min_demand_slack),
        "total_demand_res": float(total_demand_res),
        "reservoir_outflow_res": float(reservoir_outflow_res),
        "flow_bound_max_viol": float(flow_bound_max),
        "x_bound_max_viol": float(x_bound_max),
    }


def check_xd_cycle_feasibility_from_bounds(
    network: NetworkData,
    pipe_resistances: Dict[str, float],
    c_bounds: Optional[Dict[str, float]] = None,
    C_bounds: Optional[Dict[str, float]] = None,
    initial_guess: Optional[SolverResult] = None,
    headloss_n: float = 2.0,
    cycle_basis_mode: str = "graph",
    max_examples: int = 5,
) -> Dict[str, object]:
    reservoir_nodes = set(network.reservoirs.keys())
    _, pipes = _build_index_maps(network, reservoir_nodes)

    flow_bounds = _estimate_flow_bounds(pipes, c_bounds, C_bounds, initial_guess)
    x_bounds = _flow_to_head_bounds(pipe_resistances, flow_bounds, headloss_n)
    cycle_rows = _build_cycle_matrix(network, pipes, mode=cycle_basis_mode)

    infeasible: List[Dict[str, object]] = []
    infeasible_count = 0
    max_violation = 0.0
    for idx, row in enumerate(cycle_rows, start=1):
        min_sum = 0.0
        max_sum = 0.0
        for p, coef in row.items():
            lb, ub = x_bounds[p]
            if coef >= 0:
                min_sum += coef * lb
                max_sum += coef * ub
            else:
                min_sum += coef * ub
                max_sum += coef * lb
        if 0.0 < min_sum or 0.0 > max_sum:
            violation = min_sum if min_sum > 0.0 else -max_sum
            max_violation = max(max_violation, float(violation))
            infeasible_count += 1
            if len(infeasible) < max_examples:
                infeasible.append(
                    {
                        "cycle_index": idx,
                        "min_sum": float(min_sum),
                        "max_sum": float(max_sum),
                        "size": len(row),
                    }
                )

    return {
        "cycle_count": len(cycle_rows),
        "infeasible_count": infeasible_count,
        "max_violation": float(max_violation),
        "examples": infeasible,
    }


def solve_max_demand_distance_xd_hexaly(
    network: NetworkData,
    sensor_heads: Dict[str, float],
    pipe_resistances: Dict[str, float],
    c_bounds: Optional[Dict[str, float]] = None,
    C_bounds: Optional[Dict[str, float]] = None,
    reservoir_node: Optional[str] = None,
    reservoir_head: Optional[float] = None,
    reservoir_outflow: Optional[float] = None,
    initial_guess: Optional[SolverResult] = None,
    norm_p: float = 2.0,
    demand_lb: float = 0.0,
    total_demand: Optional[float] = None,
    measurement_nodes: Optional[Iterable[str]] = None,
    measurement_heads_equal_only: bool = True,
    headloss_n: float = 2.0,
    cycle_basis_mode: str = "graph",
    license_path: Optional[str] = None,
    time_limit: Optional[int] = None,
    seed: Optional[int] = None,
    verbosity: Optional[int] = None,
) -> DemandDistanceResult:
    _require_hexaly()
    import hexaly.optimizer as hx

    if license_path:
        hx.version.license_path = license_path

    if not measurement_heads_equal_only:
        raise ValueError("xd solver currently supports equality-only measurement mode.")

    measurement_set = set(measurement_nodes or [])
    reservoir_nodes = set(network.reservoirs.keys())
    junctions, pipes = _build_index_maps(network, reservoir_nodes)

    flow_bounds = _estimate_flow_bounds(pipes, c_bounds, C_bounds, initial_guess)
    x_bounds = _flow_to_head_bounds(pipe_resistances, flow_bounds, headloss_n)

    cycle_rows = _build_cycle_matrix(network, pipes, mode=cycle_basis_mode)
    path_rows = []
    if reservoir_node is not None and measurement_set:
        path_rows = _build_path_matrix(
            network,
            pipes,
            pipe_resistances,
            reservoir_node,
            measurement_set,
        )

    with hx.HexalyOptimizer() as optimizer:
        m = optimizer.model

        qA = {p: m.float(flow_bounds[p][0], flow_bounds[p][1]) for p in pipes}
        qB = {p: m.float(flow_bounds[p][0], flow_bounds[p][1]) for p in pipes}
        xA = {p: m.float(x_bounds[p][0], x_bounds[p][1]) for p in pipes}
        xB = {p: m.float(x_bounds[p][0], x_bounds[p][1]) for p in pipes}

        for p in pipes:
            r_e = pipe_resistances[p]
            m.constraint(xA[p] == r_e * qA[p] * m.pow(m.abs(qA[p]), headloss_n - 1.0))
            m.constraint(xB[p] == r_e * qB[p] * m.pow(m.abs(qB[p]), headloss_n - 1.0))

        for row in cycle_rows:
            m.constraint(_sum_expr(m, [coef * xA[p] for p, coef in row.items()]) == 0.0)
            m.constraint(_sum_expr(m, [coef * xB[p] for p, coef in row.items()]) == 0.0)

        for row in path_rows:
            expr = _sum_expr(m, [coef * (xA[p] - xB[p]) for p, coef in row.items()])
            m.constraint(expr == 0.0)

        dA = {}
        dB = {}
        for j in junctions:
            inflow_a = []
            outflow_a = []
            inflow_b = []
            outflow_b = []
            for p in pipes:
                pipe = network.pipes[p]
                if pipe.end_node == j:
                    inflow_a.append(qA[p])
                    inflow_b.append(qB[p])
                if pipe.start_node == j:
                    outflow_a.append(qA[p])
                    outflow_b.append(qB[p])
            dA[j] = _sum_expr(m, inflow_a) - _sum_expr(m, outflow_a)
            dB[j] = _sum_expr(m, inflow_b) - _sum_expr(m, outflow_b)
            m.constraint(dA[j] >= demand_lb)
            m.constraint(dB[j] >= demand_lb)

        if total_demand is not None:
            m.constraint(_sum_expr(m, list(dA.values())) == float(total_demand))
            m.constraint(_sum_expr(m, list(dB.values())) == float(total_demand))

        if reservoir_node is not None and reservoir_outflow is not None:
            net_a = []
            net_b = []
            for p in pipes:
                pipe = network.pipes[p]
                if pipe.start_node == reservoir_node:
                    net_a.append(qA[p])
                    net_b.append(qB[p])
                if pipe.end_node == reservoir_node:
                    net_a.append(-qA[p])
                    net_b.append(-qB[p])
            m.constraint(_sum_expr(m, net_a) == float(reservoir_outflow))
            m.constraint(_sum_expr(m, net_b) == float(reservoir_outflow))

        if math.isinf(norm_p):
            t_ub = max(abs(flow_bounds[p][1]) for p in pipes) if pipes else 1.0
            t = m.float(0.0, 10.0 * t_ub)
            for j in junctions:
                m.constraint(t >= m.abs(dA[j] - dB[j]))
            obj_expr = t
            m.maximize(obj_expr)
        else:
            if norm_p <= 0:
                raise ValueError("norm_p must be positive.")
            terms = [m.pow(m.abs(dA[j] - dB[j]), float(norm_p)) for j in junctions]
            obj_expr = _sum_expr(m, terms)
            m.maximize(obj_expr)

        m.close()

        if time_limit is not None:
            optimizer.param.time_limit = int(time_limit)
        if seed is not None:
            optimizer.param.seed = int(seed)
        if verbosity is not None:
            optimizer.param.verbosity = int(verbosity)

        optimizer.solve()
        status = optimizer.solution.status
        success = status in {hx.HxSolutionStatus.FEASIBLE, hx.HxSolutionStatus.OPTIMAL}
        best_bound = None
        try:
            best_bound = getattr(optimizer.solution, "best_bound", None)
        except Exception:
            best_bound = None

        q_a = {p: float(qA[p].value) for p in pipes}
        q_b = {p: float(qB[p].value) for p in pipes}
        x_a = {p: float(xA[p].value) for p in pipes}
        x_b = {p: float(xB[p].value) for p in pipes}
        d_a = {j: float(dA[j].value) for j in junctions}
        d_b = {j: float(dB[j].value) for j in junctions}

    heads_a = _compute_heads_from_x(
        network,
        pipes,
        x_a,
        pipe_resistances,
        reservoir_node,
        reservoir_head,
    )
    heads_b = _compute_heads_from_x(
        network,
        pipes,
        x_b,
        pipe_resistances,
        reservoir_node,
        reservoir_head,
    )

    max_violation = 0.0
    min_demand_viol = 0.0
    if d_a or d_b:
        min_demand_viol = min(min(d_a.values()), min(d_b.values())) - demand_lb
    objective_val = _compute_objective_value(d_a, d_b, norm_p)
    status_msg = str(status)
    # best_bound retrieved inside the solver context

    return DemandDistanceResult(
        demands_a=d_a,
        demands_b=d_b,
        heads_a=heads_a,
        heads_b=heads_b,
        flows_a=q_a,
        flows_b=q_b,
        max_violation=float(max_violation),
        min_demand_viol=float(min_demand_viol),
        success=bool(success),
        objective=objective_val,
        solver_status=status_msg,
        best_bound=best_bound,
    )
