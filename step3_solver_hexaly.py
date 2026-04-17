from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import heapq
import math
import time

from step1_io import NetworkData
from step3_solver import (
    DemandDistanceResult,
    SolverResult,
    _build_index_maps,
    _head_value,
    _compute_objective_value,
)


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


def _estimate_head_bounds(
    junctions: List[str],
    network: NetworkData,
    fixed_heads: Dict[str, float],
    initial_guess: Optional[SolverResult],
    head_margin: float,
) -> Tuple[float, float]:
    candidates = [float(v) for v in fixed_heads.values()]
    for j in junctions:
        candidates.append(float(network.junctions[j].elevation_m))
        if initial_guess is not None and j in initial_guess.heads:
            candidates.append(float(initial_guess.heads[j]))
    if not candidates:
        return (-1000.0, 1000.0)
    margin = float(head_margin)
    return (min(candidates) - margin, max(candidates) + margin)


def _build_head_bounds_from_paths(
    junctions: List[str],
    network: NetworkData,
    fixed_heads: Dict[str, float],
    pipe_resistances: Dict[str, float],
    flow_bounds: Dict[str, Tuple[float, float]],
    default_bounds: Tuple[float, float],
    reservoir_nodes: Iterable[str],
) -> Dict[str, Tuple[float, float]]:
    bounds = {j: (default_bounds[0], default_bounds[1]) for j in junctions}
    reservoir_ids = [n for n in reservoir_nodes if n in fixed_heads]
    if not reservoir_ids:
        return bounds

    adjacency: Dict[str, List[Tuple[str, float]]] = {}
    for pipe_id, pipe in network.pipes.items():
        lb, ub = flow_bounds.get(pipe_id, (-math.inf, math.inf))
        qmax = max(abs(lb), abs(ub))
        if not math.isfinite(qmax):
            continue
        max_loss = pipe_resistances[pipe_id] * (qmax ** 2)
        adjacency.setdefault(pipe.start_node, []).append((pipe.end_node, max_loss))
        adjacency.setdefault(pipe.end_node, []).append((pipe.start_node, max_loss))

    def dijkstra(src: str) -> Dict[str, float]:
        dist: Dict[str, float] = {src: 0.0}
        heap = [(0.0, src)]
        while heap:
            d_u, u = heapq.heappop(heap)
            if d_u > dist.get(u, float("inf")):
                continue
            for v, w in adjacency.get(u, []):
                d_v = d_u + w
                if d_v < dist.get(v, float("inf")):
                    dist[v] = d_v
                    heapq.heappush(heap, (d_v, v))
        return dist

    for head_node in reservoir_ids:
        head_val = fixed_heads[head_node]
        dist = dijkstra(head_node)
        for j in junctions:
            if j not in dist:
                continue
            lb, ub = bounds[j]
            bounds[j] = (max(lb, head_val - dist[j]), min(ub, head_val + dist[j]))

    for j in junctions:
        if j in fixed_heads:
            lb, ub = bounds[j]
            head_val = fixed_heads[j]
            bounds[j] = (min(lb, head_val), max(ub, head_val))

    for j in junctions:
        lb, ub = bounds[j]
        if lb > ub:
            mid = 0.5 * (lb + ub)
            bounds[j] = (mid, mid)

    return bounds


def _sum_expr(model, exprs: List[object]):
    if not exprs:
        return 0.0
    return model.sum(*exprs)


def solve_max_demand_distance_hexaly(
    network: NetworkData,
    sensor_heads: Dict[str, float],
    pipe_resistances: Dict[str, float],
    c_bounds: Optional[Dict[str, float]] = None,
    C_bounds: Optional[Dict[str, float]] = None,
    reservoir_node: Optional[str] = None,
    reservoir_outflow: Optional[float] = None,
    initial_guess: Optional[SolverResult] = None,
    norm_p: float = 2.0,
    demand_lb: float = 0.0,
    total_demand: Optional[float] = None,
    measurement_nodes: Optional[Iterable[str]] = None,
    measurement_heads_equal_only: bool = False,
    reference_demands: Optional[Dict[str, float]] = None,
    restriction_mode: Optional[str] = None,
    radius_to_fixed: Optional[float] = None,
    deviation_alpha: Optional[float] = None,
    license_path: Optional[str] = None,
    time_limit: Optional[int] = None,
    seed: Optional[int] = None,
    verbosity: Optional[int] = None,
    stagnation_seconds: Optional[float] = None,
    stagnation_eps: float = 1e-6,
    head_margin: float = 50.0,
    use_path_head_bounds: bool = True,
) -> DemandDistanceResult:
    _require_hexaly()
    import hexaly.optimizer as hx

    if license_path:
        hx.version.license_path = license_path

    reservoir_nodes = set(network.reservoirs.keys())
    fixed_heads = dict(sensor_heads)
    measurement_set = set(measurement_nodes or [])
    if measurement_heads_equal_only:
        fixed_heads = {k: v for k, v in fixed_heads.items() if k not in measurement_set}
    for res_id, res_node in network.reservoirs.items():
        if res_id not in fixed_heads:
            fixed_heads[res_id] = res_node.elevation_m

    junctions, pipes = _build_index_maps(network, reservoir_nodes)

    flow_bounds = _estimate_flow_bounds(pipes, c_bounds, C_bounds, initial_guess)
    default_bounds = _estimate_head_bounds(junctions, network, fixed_heads, initial_guess, head_margin)
    if use_path_head_bounds:
        head_bounds = _build_head_bounds_from_paths(
            junctions,
            network,
            fixed_heads,
            pipe_resistances,
            flow_bounds,
            default_bounds,
            reservoir_nodes,
        )
    else:
        head_bounds = {j: default_bounds for j in junctions}

    with hx.HexalyOptimizer() as optimizer:
        m = optimizer.model

        qA = {p: m.float(flow_bounds[p][0], flow_bounds[p][1]) for p in pipes}
        qB = {p: m.float(flow_bounds[p][0], flow_bounds[p][1]) for p in pipes}
        hA = {j: m.float(head_bounds[j][0], head_bounds[j][1]) for j in junctions}
        hB = {j: m.float(head_bounds[j][0], head_bounds[j][1]) for j in junctions}

        for pipe_id in pipes:
            pipe = network.pipes[pipe_id]
            r_e = pipe_resistances[pipe_id]

            h_u = fixed_heads.get(pipe.start_node, None)
            h_v = fixed_heads.get(pipe.end_node, None)
            hu_a = h_u if h_u is not None else hA[pipe.start_node]
            hv_a = h_v if h_v is not None else hA[pipe.end_node]
            hu_b = h_u if h_u is not None else hB[pipe.start_node]
            hv_b = h_v if h_v is not None else hB[pipe.end_node]

            m.constraint(hu_a - hv_a == r_e * m.abs(qA[pipe_id]) * qA[pipe_id])
            m.constraint(hu_b - hv_b == r_e * m.abs(qB[pipe_id]) * qB[pipe_id])

        for node_id, head_val in fixed_heads.items():
            if node_id in junctions:
                m.constraint(hA[node_id] == float(head_val))
                m.constraint(hB[node_id] == float(head_val))

        if measurement_heads_equal_only:
            for node_id in measurement_set:
                if node_id in junctions and node_id not in fixed_heads:
                    m.constraint(hA[node_id] == hB[node_id])

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

        if restriction_mode is not None:
            if restriction_mode not in {"radius_to_fixed", "deviation_to_fixed"}:
                raise ValueError("restriction_mode must be None, 'radius_to_fixed', or 'deviation_to_fixed'.")
            if reference_demands is None:
                raise ValueError("reference_demands is required when restriction_mode is set.")

            if restriction_mode == "radius_to_fixed":
                if radius_to_fixed is None:
                    raise ValueError("radius_to_fixed must be set for 'radius_to_fixed' mode.")
                radius_val = float(radius_to_fixed)
                if math.isinf(norm_p):
                    for j in junctions:
                        ref = float(reference_demands.get(j, 0.0))
                        m.constraint(m.abs(dA[j] - ref) <= radius_val)
                        m.constraint(m.abs(dB[j] - ref) <= radius_val)
                else:
                    if norm_p <= 0:
                        raise ValueError("norm_p must be positive.")
                    power = float(norm_p)
                    terms_a = [m.pow(m.abs(dA[j] - float(reference_demands.get(j, 0.0))), power) for j in junctions]
                    terms_b = [m.pow(m.abs(dB[j] - float(reference_demands.get(j, 0.0))), power) for j in junctions]
                    m.constraint(_sum_expr(m, terms_a) <= radius_val ** power)
                    m.constraint(_sum_expr(m, terms_b) <= radius_val ** power)
            else:
                if deviation_alpha is None:
                    raise ValueError("deviation_alpha must be set for 'deviation_to_fixed' mode.")
                alpha = float(deviation_alpha)
                if alpha < 0.0:
                    raise ValueError("deviation_alpha must be non-negative.")
                for j in junctions:
                    ref = float(reference_demands.get(j, 0.0))
                    low = ref - alpha * ref
                    high = ref + alpha * ref
                    if low > high:
                        low, high = high, low
                    m.constraint(dA[j] >= low)
                    m.constraint(dA[j] <= high)
                    m.constraint(dB[j] >= low)
                    m.constraint(dB[j] <= high)

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
                m.constraint(sign * qA[ref_pipe] >= 0.0)

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

        if stagnation_seconds is not None and stagnation_seconds > 0:
            last_improve_time = time.monotonic()
            last_obj = float("-inf")

            def _on_tick(_opt, _cb_type):
                nonlocal last_improve_time, last_obj
                try:
                    current = float(obj_expr.value)
                except Exception:
                    current = last_obj
                if current > last_obj + stagnation_eps:
                    last_obj = current
                    last_improve_time = time.monotonic()
                elif time.monotonic() - last_improve_time >= stagnation_seconds:
                    _opt.stop()

            optimizer.add_callback(hx.HxCallbackType.TIME_TICKED, _on_tick)

        if time_limit is not None:
            optimizer.param.time_limit = int(time_limit)
        if seed is not None:
            optimizer.param.seed = int(seed)
        if verbosity is not None:
            optimizer.param.verbosity = int(verbosity)

        optimizer.solve()
        status = optimizer.solution.status
        success = status in {hx.HxSolutionStatus.FEASIBLE, hx.HxSolutionStatus.OPTIMAL}

        q_a = {p: float(qA[p].value) for p in pipes}
        q_b = {p: float(qB[p].value) for p in pipes}
        h_a = {j: float(hA[j].value) for j in junctions}
        h_b = {j: float(hB[j].value) for j in junctions}
        for node_id, head_val in fixed_heads.items():
            h_a[node_id] = float(head_val)
            h_b[node_id] = float(head_val)

        d_a = {j: float(dA[j].value) for j in junctions}
        d_b = {j: float(dB[j].value) for j in junctions}

    max_violation = 0.0
    for p in pipes:
        pipe = network.pipes[p]
        r_e = pipe_resistances[p]
        for q, h in ((q_a, h_a), (q_b, h_b)):
            q_e = q[p]
            h_u = _head_value(pipe.start_node, h, fixed_heads)
            h_v = _head_value(pipe.end_node, h, fixed_heads)
            max_violation = max(max_violation, abs(h_u - h_v - r_e * abs(q_e) * q_e))

    for node_id, head_val in fixed_heads.items():
        if node_id in h_a:
            max_violation = max(max_violation, abs(h_a[node_id] - head_val))
            max_violation = max(max_violation, abs(h_b[node_id] - head_val))

    if reservoir_node is not None and reservoir_outflow is not None:
        net_a = 0.0
        net_b = 0.0
        for p in pipes:
            pipe = network.pipes[p]
            if pipe.start_node == reservoir_node:
                net_a += q_a[p]
                net_b += q_b[p]
            if pipe.end_node == reservoir_node:
                net_a -= q_a[p]
                net_b -= q_b[p]
        max_violation = max(max_violation, abs(net_a - reservoir_outflow), abs(net_b - reservoir_outflow))

    if total_demand is not None:
        max_violation = max(max_violation, abs(sum(d_a.values()) - total_demand), abs(sum(d_b.values()) - total_demand))

    min_demand_viol = 0.0
    if d_a or d_b:
        min_demand_viol = min(min(d_a.values()), min(d_b.values())) - demand_lb
    objective_val = _compute_objective_value(d_a, d_b, norm_p)
    status_msg = str(status)
    best_bound = getattr(optimizer.solution, "best_bound", None)

    return DemandDistanceResult(
        demands_a=d_a,
        demands_b=d_b,
        heads_a=h_a,
        heads_b=h_b,
        flows_a=q_a,
        flows_b=q_b,
        max_violation=float(max_violation),
        min_demand_viol=float(min_demand_viol),
        success=bool(success),
        objective=objective_val,
        solver_status=status_msg,
        best_bound=best_bound,
    )
