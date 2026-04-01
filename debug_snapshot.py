from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, Tuple

from step1_io import load_inp_network, compute_pipe_resistances


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _latest_data_dir(base_dir: str) -> str:
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Missing data directory: {base_dir}")
    candidates = [os.path.join(base_dir, d) for d in os.listdir(base_dir)]
    candidates = [d for d in candidates if os.path.isdir(d)]
    if not candidates:
        raise FileNotFoundError(f"No runs found under: {base_dir}")
    candidates.sort(key=lambda d: os.path.getmtime(d))
    return candidates[-1]


def _compute_head_bounds_from_reservoir(
    network,
    pipe_resistances: Dict[str, float],
    reservoir_node: str,
    reservoir_head: float,
    c_bounds: Dict[str, float],
    C_bounds: Dict[str, float],
) -> Dict[str, Tuple[float, float]]:
    import heapq

    adjacency: Dict[str, list[Tuple[str, float]]] = {}
    for pipe_id, pipe in network.pipes.items():
        lb = c_bounds.get(pipe_id, float("-inf"))
        ub = C_bounds.get(pipe_id, float("inf"))
        qmax = max(abs(lb), abs(ub))
        if not math.isfinite(qmax):
            continue
        max_loss = pipe_resistances[pipe_id] * (qmax ** 2)
        adjacency.setdefault(pipe.start_node, []).append((pipe.end_node, max_loss))
        adjacency.setdefault(pipe.end_node, []).append((pipe.start_node, max_loss))

    dist: Dict[str, float] = {reservoir_node: 0.0}
    heap = [(0.0, reservoir_node)]
    while heap:
        d_u, u = heapq.heappop(heap)
        if d_u > dist.get(u, float("inf")):
            continue
        for v, w in adjacency.get(u, []):
            d_v = d_u + w
            if d_v < dist.get(v, float("inf")):
                dist[v] = d_v
                heapq.heappush(heap, (d_v, v))

    bounds: Dict[str, Tuple[float, float]] = {}
    for node_id in network.junctions.keys():
        if node_id not in dist:
            continue
        bounds[node_id] = (reservoir_head - dist[node_id], reservoir_head + dist[node_id])
    return bounds


def _demand_from_flows(network, flows: Dict[str, float]) -> Dict[str, float]:
    inflow = {j: 0.0 for j in network.junctions.keys()}
    outflow = {j: 0.0 for j in network.junctions.keys()}
    for pipe_id, pipe in network.pipes.items():
        q = flows.get(pipe_id, 0.0)
        if pipe.end_node in inflow:
            inflow[pipe.end_node] += q
        if pipe.start_node in outflow:
            outflow[pipe.start_node] += q
    return {j: inflow[j] - outflow[j] for j in inflow}


def _head_value(node_id: str, heads: Dict[str, float]) -> float:
    return float(heads.get(node_id, float("nan")))


def _at_bound(value: float, lb: float, ub: float, eps: float) -> str:
    if math.isfinite(lb) and abs(value - lb) <= eps:
        return "lb"
    if math.isfinite(ub) and abs(value - ub) <= eps:
        return "ub"
    return ""


def _build_report(
    output_dir: str,
    eps: float,
    use_solution_resistance: bool,
) -> dict:
    meta_path = os.path.join(output_dir, "metadata.json")
    dist_path = os.path.join(output_dir, "demand_distance.json")
    bounds_path = os.path.join(output_dir, "c_bounds.json")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing {meta_path}")
    if not os.path.exists(dist_path):
        raise FileNotFoundError(f"Missing {dist_path}")

    metadata = _read_json(meta_path)
    dist = _read_json(dist_path)

    wdn_name = metadata.get("WDN_NAME")
    if not wdn_name:
        raise ValueError("WDN_NAME missing in metadata.json")

    inp_path = os.path.join("wdn", f"{wdn_name}.inp")
    network = load_inp_network(inp_path)

    c_bounds = {}
    C_bounds = {}
    if os.path.exists(bounds_path):
        bounds = _read_json(bounds_path)
        c_bounds = {k: float(v) for k, v in bounds.get("c_bounds", {}).items()}
        C_bounds = {k: float(v) for k, v in bounds.get("C_bounds", {}).items()}

    resistances_raw = compute_pipe_resistances(network)
    pipe_resistances = {pid: vals["r_e"] for pid, vals in resistances_raw.items()}

    heads_a = {k: float(v) for k, v in dist.get("heads_a", {}).items()}
    heads_b = {k: float(v) for k, v in dist.get("heads_b", {}).items()}
    flows_a = {k: float(v) for k, v in dist.get("flows_a", {}).items()}
    flows_b = {k: float(v) for k, v in dist.get("flows_b", {}).items()}

    if use_solution_resistance:
        for pipe_id, pipe in network.pipes.items():
            q = flows_a.get(pipe_id, 0.0)
            if abs(q) < 1e-12:
                continue
            hu = _head_value(pipe.start_node, heads_a)
            hv = _head_value(pipe.end_node, heads_a)
            if math.isfinite(hu) and math.isfinite(hv):
                pipe_resistances[pipe_id] = (hu - hv) / (abs(q) * q)

    reservoir_nodes = list(network.reservoirs.keys())
    reservoir_node = reservoir_nodes[0] if reservoir_nodes else None
    reservoir_head = None
    if reservoir_node is not None:
        reservoir_head = heads_a.get(reservoir_node, network.reservoirs[reservoir_node].elevation_m)

    head_bounds = {}
    if reservoir_node is not None and reservoir_head is not None and c_bounds and C_bounds:
        head_bounds = _compute_head_bounds_from_reservoir(
            network=network,
            pipe_resistances=pipe_resistances,
            reservoir_node=reservoir_node,
            reservoir_head=float(reservoir_head),
            c_bounds=c_bounds,
            C_bounds=C_bounds,
        )

    d_a = _demand_from_flows(network, flows_a)
    d_b = _demand_from_flows(network, flows_b)

    demand_lb = float(metadata.get("DEMAND_LB", 0.0))
    total_demand_target = None
    if metadata.get("FIX_TOTAL_DEMAND"):
        total_demand_target = sum(d_a.values())

    junction_rows = {}
    head_at_bounds = {"a_lb": 0, "a_ub": 0, "b_lb": 0, "b_ub": 0}
    for j in network.junctions.keys():
        ha = heads_a.get(j, float("nan"))
        hb = heads_b.get(j, float("nan"))
        lb, ub = head_bounds.get(j, (float("-inf"), float("inf")))
        tag_a = _at_bound(ha, lb, ub, eps)
        tag_b = _at_bound(hb, lb, ub, eps)
        if tag_a == "lb":
            head_at_bounds["a_lb"] += 1
        if tag_a == "ub":
            head_at_bounds["a_ub"] += 1
        if tag_b == "lb":
            head_at_bounds["b_lb"] += 1
        if tag_b == "ub":
            head_at_bounds["b_ub"] += 1
        junction_rows[j] = {
            "head_a": ha,
            "head_b": hb,
            "head_lb": lb,
            "head_ub": ub,
            "head_a_tag": tag_a,
            "head_b_tag": tag_b,
            "demand_a": d_a.get(j, 0.0),
            "demand_b": d_b.get(j, 0.0),
            "demand_lb": demand_lb,
        }

    pipe_rows = {}
    flow_at_bounds = {"a_lb": 0, "a_ub": 0, "b_lb": 0, "b_ub": 0}
    for pipe_id, pipe in network.pipes.items():
        qa = flows_a.get(pipe_id, 0.0)
        qb = flows_b.get(pipe_id, 0.0)
        lb = float(c_bounds.get(pipe_id, float("-inf")))
        ub = float(C_bounds.get(pipe_id, float("inf")))
        tag_a = _at_bound(qa, lb, ub, eps)
        tag_b = _at_bound(qb, lb, ub, eps)
        if tag_a == "lb":
            flow_at_bounds["a_lb"] += 1
        if tag_a == "ub":
            flow_at_bounds["a_ub"] += 1
        if tag_b == "lb":
            flow_at_bounds["b_lb"] += 1
        if tag_b == "ub":
            flow_at_bounds["b_ub"] += 1

        r_e = pipe_resistances.get(pipe_id, 0.0)
        hu_a = _head_value(pipe.start_node, heads_a)
        hv_a = _head_value(pipe.end_node, heads_a)
        hu_b = _head_value(pipe.start_node, heads_b)
        hv_b = _head_value(pipe.end_node, heads_b)
        res_a = hu_a - hv_a - r_e * abs(qa) * qa
        res_b = hu_b - hv_b - r_e * abs(qb) * qb

        pipe_rows[pipe_id] = {
            "flow_a": qa,
            "flow_b": qb,
            "flow_lb": lb,
            "flow_ub": ub,
            "flow_a_tag": tag_a,
            "flow_b_tag": tag_b,
            "headloss_res_a": res_a,
            "headloss_res_b": res_b,
        }

    demand_viol_a = [d - demand_lb for d in d_a.values()]
    demand_viol_b = [d - demand_lb for d in d_b.values()]

    summary = {
        "junction_count": len(network.junctions),
        "pipe_count": len(network.pipes),
        "head_bounds_count": len(head_bounds),
        "head_at_bounds": head_at_bounds,
        "flow_at_bounds": flow_at_bounds,
        "min_demand_slack_a": min(demand_viol_a) if demand_viol_a else 0.0,
        "min_demand_slack_b": min(demand_viol_b) if demand_viol_b else 0.0,
        "total_demand_a": sum(d_a.values()),
        "total_demand_b": sum(d_b.values()),
    }

    if total_demand_target is not None:
        summary["total_demand_target"] = total_demand_target
        summary["total_demand_res_a"] = summary["total_demand_a"] - total_demand_target
        summary["total_demand_res_b"] = summary["total_demand_b"] - total_demand_target

    return {
        "output_dir": output_dir,
        "metadata": metadata,
        "summary": summary,
        "junctions": junction_rows,
        "pipes": pipe_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump debugging values for a run.")
    parser.add_argument("--output-dir", default=None, help="Run output directory under data/.")
    parser.add_argument("--eps", type=float, default=1e-6, help="Tolerance for bound checks.")
    parser.add_argument("--use-solution-resistance", action="store_true", help="Infer r_e from heads_a/flows_a.")
    parser.add_argument("--no-save", action="store_true", help="Skip saving debug_dump.json.")
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = _latest_data_dir("data")
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    report = _build_report(
        output_dir=output_dir,
        eps=float(args.eps),
        use_solution_resistance=bool(args.use_solution_resistance),
    )

    print("Debug summary:")
    for key, value in report["summary"].items():
        print(f"  {key}: {value}")

    if not args.no_save:
        out_path = os.path.join(output_dir, "debug_dump.json")
        _write_json(out_path, report)
        print(f"Saved debug report to: {out_path}")


if __name__ == "__main__":
    main()
