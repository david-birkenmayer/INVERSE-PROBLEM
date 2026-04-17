from typing import Dict, Iterable, Tuple
import json
import os
from datetime import datetime

### NETWORK AND MEASUREMENT SITES
WDN_NAME = "Alperovits"; MEASUREMENT_SITES = 2
# WDN_NAME = "Kadu"; MEASUREMENT_SITES = ["10", "14", "24"]
# WDN_NAME = "Anytown"; MEASUREMENT_SITES = 1 # ([], [50, 100]) # 2 # ([], [20, 140], [20,50,60,100,120,140])


### PIPE BOUNDS
PIPE_BOUNDS = True
PIPE_BOUND_SAFENESS = 0.85  # in [0, 1], higher means looser bounds
PIPE_BOUND_METHOD = "perturb"  # "montecarlo" or "perturb"
PIPE_BOUND_POLICY = "minmax"  # "minmax" or "quantile"
PIPE_BOUND_PERTURB_SAMPLES = 25
PIPE_BOUND_PERTURB_SIGMA = 0.10
PIPE_BOUND_PERTURB_SEED = 1
PIPE_BOUND_PERTURB_FIX_TOTAL = True
PIPE_BOUND_PERTURB_BASE = "base"  # "auto", "base", or "scenario"

SCENARIO_SOURCE = "base"  # "base" or "random"
HEADLOSS_MODEL = "hw"  # "auto", "dw", or "hw"

MULTI_STARTS = 1
MULTI_START_NOISE = 0.05
MULTI_START_NOISE_REL = 0.25
MULTI_START_SEED = None  # set int for reproducible multistarts

# SOLVER = "Hexaly"
SOLVER = "Hexaly_xd"

DEMAND_INTERVALS = False
SINGLE_NODE_SOLUTION = False
RADIUS = True
NORM = 2  # positive int/float, or float("inf") for infinity norm
DEMAND_LB = 1e-6  # enforce strictly positive junction demands
USE_RESERVOIR_OUTFLOW_CONSTRAINT = False  # fix reservoir demand via net outflow
FIX_TOTAL_DEMAND = True  # anchor sum of demands to scenario total
MEASUREMENT_HEADS_EQUAL_ONLY = True  # only enforce equal heads at measurements
MEASUREMENT_ALLOWLIST = None  # list of node ids or None for all junctions
MEASUREMENT_MAX_SUBSETS = 200
MEASUREMENT_SUBSET_SEED = 1
MEASUREMENT_SUBSET_MODE = "all"  # "exact" or "all" (sizes 1..p)

SHOW_INTERMEDIATE_PLOTS = True
SHOW_INTERMEDIATE_PLOTS_MULTIPLE = False

DEBUG_DUMP_ALWAYS = True
DEBUG_DUMP_EPS = 1e-6
CHECK_XD_BASE_FEASIBILITY = True
CHECK_XD_CYCLE_BOUNDS = True
XD_CYCLE_BASIS_MODE = "planar"  # "graph" (cycle_basis) or "planar" (faces)
SKIP_FEASIBILITY_SOLVE = True
FIXED_REFERENCE = False
DEMAND_RESTRICTION_MODE = None  # None, "radius_to_fixed", or "deviation_to_fixed"
RADIUS_TO_FIXED = None  # float radius for "radius_to_fixed" mode
DEVIATION_ALPHA = None  # float alpha for "deviation_to_fixed" mode

HEXALY_LICENSE_PATH = os.path.expanduser("~/opt/Hexaly_14_5/license.dat")
HEXALY_TIME_LIMIT = 30
HEXALY_STAGNATION_SECONDS = 0
HEXALY_STAGNATION_EPS = 1e-6
HEXALY_SEED = 0
HEXALY_VERBOSITY = 2
HEXALY_HEAD_MARGIN = 300.0
HEXALY_USE_PATH_HEAD_BOUNDS = True


from step1_io import load_inp_network, compute_pipe_resistances, compute_pipe_resistances_hw
from step2_estimation import (
	simulate_random_demand_scenarios,
	simulate_base_flows,
	simulate_base_scenario,
	simulate_single_random_scenario,
	simulate_perturbed_demand_scenarios,
)
from step3_solver import (
	DemandBounds,
	SolverResult,
	SingleNodeResult,
	FeasibilityResult,
	_build_index_maps,
	solve_demand_bounds,
	solve_feasibility,
	solve_single_node_min,
)
from step3_solver_hexaly import solve_max_demand_distance_hexaly
from step3_solver_xd_hexaly import solve_max_demand_distance_xd_hexaly
from step3_solver_xd_hexaly import check_xd_feasibility_from_flows, check_xd_cycle_feasibility_from_bounds
from step3_solver_xd_hexaly import _build_cycle_matrix, _build_path_matrix, _phi
import debug_snapshot


def _compute_pipe_bounds_all(
	flows_by_pipe: Dict[str, "np.ndarray"],
	safeness: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
	import numpy as np

	if not 0.0 <= safeness <= 1.0:
		raise ValueError("PIPE_BOUND_SAFENESS must be in [0, 1].")
	alpha = (1.0 - safeness) / 2.0
	low_q = alpha
	high_q = 1.0 - alpha

	c_bounds: Dict[str, float] = {}
	C_bounds: Dict[str, float] = {}
	for pipe_id, q in flows_by_pipe.items():
		low = float(np.quantile(q, low_q))
		high = float(np.quantile(q, high_q))
		if low > high:
			low, high = high, low
		c_bounds[pipe_id] = low
		C_bounds[pipe_id] = high
	return c_bounds, C_bounds


def _compute_pipe_bounds_from_samples(
	flows_by_pipe: Dict[str, "np.ndarray"],
	policy: str,
	safeness: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
	import numpy as np

	if policy not in {"minmax", "quantile"}:
		raise ValueError("PIPE_BOUND_POLICY must be 'minmax' or 'quantile'.")
	if policy == "quantile":
		return _compute_pipe_bounds_all(flows_by_pipe, safeness)

	c_bounds: Dict[str, float] = {}
	C_bounds: Dict[str, float] = {}
	for pipe_id, q in flows_by_pipe.items():
		if q.size == 0:
			low = 0.0
			high = 0.0
		else:
			low = float(np.min(q))
			high = float(np.max(q))
		if low > high:
			low, high = high, low
		c_bounds[pipe_id] = low
		C_bounds[pipe_id] = high
	return c_bounds, C_bounds


def _compute_head_bounds_from_fixed_heads(
	network,
	pipe_resistances: Dict[str, float],
	fixed_heads: Dict[str, float],
	c_bounds: Dict[str, float],
	C_bounds: Dict[str, float],
	head_margin: float,
) -> Dict[str, Tuple[float, float]]:
	import heapq
	import math

	junctions = list(network.junctions.keys())
	if not junctions:
		return {}

	default_candidates = [float(v) for v in fixed_heads.values()]
	for node_id in junctions:
		default_candidates.append(float(network.junctions[node_id].elevation_m))
	if not default_candidates:
		default_bounds = (-1000.0, 1000.0)
	else:
		default_bounds = (
			min(default_candidates) - float(head_margin),
			max(default_candidates) + float(head_margin),
		)

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

	bounds: Dict[str, Tuple[float, float]] = {}

	def _dijkstra(src: str) -> Dict[str, float]:
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

	for head_node, head_val in fixed_heads.items():
		dist = _dijkstra(head_node)
		for j in junctions:
			if j not in dist:
				continue
			lb = head_val - dist[j]
			ub = head_val + dist[j]
			if j in bounds:
				cur_lb, cur_ub = bounds[j]
				bounds[j] = (max(cur_lb, lb), min(cur_ub, ub))
			else:
				bounds[j] = (lb, ub)

	for node_id in junctions:
		if node_id not in bounds:
			bounds[node_id] = default_bounds

	for node_id, head_val in fixed_heads.items():
		if node_id in bounds:
			lb, ub = bounds[node_id]
			bounds[node_id] = (min(lb, head_val), max(ub, head_val))

	for node_id in list(bounds.keys()):
		lb, ub = bounds[node_id]
		if lb > ub:
			mid = 0.5 * (lb + ub)
			bounds[node_id] = (mid, mid)

	return bounds


def _get_planar_face_cycles(G, network) -> list[list[str]]:
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

	def _face_area(face: list[str]) -> float:
		area = 0.0
		for i in range(len(face)):
			x1, y1 = coords[face[i]]
			x2, y2 = coords[face[(i + 1) % len(face)]]
			area += x1 * y2 - x2 * y1
		return 0.5 * area

	areas = [abs(_face_area(face)) for face in faces]
	outer_idx = max(range(len(faces)), key=lambda i: areas[i])
	return [face for i, face in enumerate(faces) if i != outer_idx]


def _print_xd_paths_cycles(
	network,
	pipe_resistances: Dict[str, float],
	reservoir_node: str | None,
	measurement_nodes: Iterable[str],
	cycle_basis_mode: str = "graph",
) -> None:
	import networkx as nx

	G = nx.Graph()
	for pipe_id, pipe in network.pipes.items():
		weight = pipe_resistances.get(pipe_id, 1.0)
		G.add_edge(pipe.start_node, pipe.end_node, weight=weight)

	if cycle_basis_mode == "planar":
		cycles = _get_planar_face_cycles(G, network)
	else:
		cycles = nx.cycle_basis(G)
	print(f"XD cycles (count={len(cycles)}):")
	for idx, cycle in enumerate(cycles, start=1):
		print(f"  cycle {idx}: {cycle}")

	if reservoir_node is None:
		print("XD paths: reservoir node not found")
		return

	print("XD paths from reservoir to measurements:")
	for meas in measurement_nodes:
		if meas not in G.nodes:
			print(f"  {meas}: not in graph")
			continue
		if meas == reservoir_node:
			print(f"  {meas}: reservoir node")
			continue
		try:
			path = nx.shortest_path(G, reservoir_node, meas, weight="weight")
			print(f"  {meas}: {path}")
		except nx.NetworkXNoPath:
			print(f"  {meas}: no path")


def _check_xd_pair_feasibility(
	network,
	pipe_resistances: Dict[str, float],
	flows_a: Dict[str, float],
	flows_b: Dict[str, float],
	reservoir_node: str | None,
	measurement_nodes: Iterable[str],
	c_bounds: Dict[str, float] | None,
	C_bounds: Dict[str, float] | None,
	demand_lb: float,
	total_demand: float | None,
	reservoir_outflow: float | None,
	headloss_n: float,
	cycle_basis_mode: str,
) -> Dict[str, float]:
	reservoir_nodes = set(network.reservoirs.keys())
	junctions = [j for j in network.junctions.keys() if j not in reservoir_nodes]
	_, pipes = _build_index_maps(network, reservoir_nodes)

	def _demand_from_flows(flows: Dict[str, float]) -> Dict[str, float]:
		values = {j: 0.0 for j in junctions}
		for pipe_id, pipe in network.pipes.items():
			q = flows.get(pipe_id, 0.0)
			if pipe.end_node in values:
				values[pipe.end_node] += q
			if pipe.start_node in values:
				values[pipe.start_node] -= q
		return values

	def _reservoir_outflow_res(flows: Dict[str, float]) -> float:
		if reservoir_node is None or reservoir_outflow is None:
			return 0.0
		net = 0.0
		for pipe_id, pipe in network.pipes.items():
			q = flows.get(pipe_id, 0.0)
			if pipe.start_node == reservoir_node:
				net += q
			if pipe.end_node == reservoir_node:
				net -= q
		return net - reservoir_outflow

	def _flow_bound_max(flows: Dict[str, float]) -> float:
		if c_bounds is None or C_bounds is None:
			return 0.0
		max_viol = 0.0
		for p in pipes:
			q = flows.get(p, 0.0)
			lb = c_bounds.get(p, -float("inf"))
			ub = C_bounds.get(p, float("inf"))
			if q < lb:
				max_viol = max(max_viol, lb - q)
			if q > ub:
				max_viol = max(max_viol, q - ub)
		return max_viol

	x_a = {p: _phi(pipe_resistances[p], flows_a.get(p, 0.0), headloss_n) for p in pipes}
	x_b = {p: _phi(pipe_resistances[p], flows_b.get(p, 0.0), headloss_n) for p in pipes}

	cycle_rows = _build_cycle_matrix(network, pipes, mode=cycle_basis_mode)
	cycle_max_a = 0.0
	cycle_max_b = 0.0
	for row in cycle_rows:
		val_a = 0.0
		val_b = 0.0
		for p, coef in row.items():
			val_a += coef * x_a.get(p, 0.0)
			val_b += coef * x_b.get(p, 0.0)
		cycle_max_a = max(cycle_max_a, abs(val_a))
		cycle_max_b = max(cycle_max_b, abs(val_b))

	path_max = 0.0
	if reservoir_node is not None and measurement_nodes:
		path_rows = _build_path_matrix(
			network,
			pipes,
			pipe_resistances,
			reservoir_node,
			measurement_nodes,
		)
		for row in path_rows:
			val = 0.0
			for p, coef in row.items():
				val += coef * (x_a.get(p, 0.0) - x_b.get(p, 0.0))
			path_max = max(path_max, abs(val))

	d_a = _demand_from_flows(flows_a)
	d_b = _demand_from_flows(flows_b)
	min_demand_slack = 0.0
	if d_a or d_b:
		min_demand_slack = min(min(d_a.values()), min(d_b.values())) - demand_lb

	total_demand_res = 0.0
	if total_demand is not None:
		res_a = sum(d_a.values()) - total_demand
		res_b = sum(d_b.values()) - total_demand
		total_demand_res = max(abs(res_a), abs(res_b))

	reservoir_outflow_res = max(
		abs(_reservoir_outflow_res(flows_a)),
		abs(_reservoir_outflow_res(flows_b)),
	)

	flow_bound_max = max(_flow_bound_max(flows_a), _flow_bound_max(flows_b))

	return {
		"cycle_max_abs_a": float(cycle_max_a),
		"cycle_max_abs_b": float(cycle_max_b),
		"path_max_abs": float(path_max),
		"min_demand_slack": float(min_demand_slack),
		"total_demand_res": float(total_demand_res),
		"reservoir_outflow_res": float(reservoir_outflow_res),
		"flow_bound_max_viol": float(flow_bound_max),
	}


def _select_measurement_sets(network) -> list[list[str]]:
	import math
	import random
	from itertools import combinations

	def _sample_combinations(
		candidates: list[str],
		sizes: list[int],
		limit: int,
		seed: int,
	) -> list[list[str]]:
		rng = random.Random(seed)
		if limit <= 0:
			return []
		include_empty = 0 in sizes
		sizes = [k for k in sizes if k != 0]
		sample: list[list[str]] = []
		if include_empty:
			sample.append([])
			limit -= 1
			if limit <= 0:
				return sample

		total = 0
		for k in sizes:
			total += math.comb(len(candidates), k)
		if total <= limit:
			for k in sizes:
				for combo in combinations(candidates, k):
					sample.append(list(combo))
			rng.shuffle(sample)
			return sample

		idx = 0
		for k in sizes:
			for combo in combinations(candidates, k):
				idx += 1
				if idx <= limit:
					sample.append(list(combo))
				else:
					j = rng.randint(1, idx)
					if j <= limit:
						sample[j - 1] = list(combo)
		rng.shuffle(sample)
		return sample

	if isinstance(MEASUREMENT_SITES, int):
		p = int(MEASUREMENT_SITES)
		if p < 0:
			return []
		if MEASUREMENT_ALLOWLIST is None:
			candidates = sorted(network.junctions.keys())
		else:
			candidates = [str(x) for x in MEASUREMENT_ALLOWLIST]
			candidates = [c for c in candidates if c in network.junctions]
		if len(candidates) < p:
			return []
		limit = int(MEASUREMENT_MAX_SUBSETS)
		max_size = min(p, len(candidates))
		if MEASUREMENT_SUBSET_MODE == "all":
			sizes = list(range(0, max_size + 1))
		else:
			sizes = [p]
			if p == 0:
				return [[]]
		return _sample_combinations(candidates, sizes, limit, MEASUREMENT_SUBSET_SEED)

	if isinstance(MEASUREMENT_SITES, (list, tuple)):
		if len(MEASUREMENT_SITES) == 0:
			return []
		if all(isinstance(item, (list, tuple)) for item in MEASUREMENT_SITES):
			return [[str(x) for x in item] for item in MEASUREMENT_SITES]
		candidates = [str(x) for x in MEASUREMENT_SITES]
		return [candidates]

	return []


def _write_json(path: str, payload: Dict[str, object]) -> None:
	with open(path, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2, sort_keys=True)


def _build_parameters_snapshot(
	headloss_model_local: str | None = None,
	measurement_nodes: Iterable[str] | None = None,
	measurement_sets_count: int | None = None,
) -> Dict[str, object]:
	return {
		"WDN_NAME": WDN_NAME,
		"MEASUREMENT_SITES": MEASUREMENT_SITES,
		"MEASUREMENT_SUBSET_MODE": MEASUREMENT_SUBSET_MODE,
		"MEASUREMENT_ALLOWLIST": MEASUREMENT_ALLOWLIST,
		"MEASUREMENT_MAX_SUBSETS": MEASUREMENT_MAX_SUBSETS,
		"MEASUREMENT_SUBSET_SEED": MEASUREMENT_SUBSET_SEED,
		"MEASUREMENT_SET_COUNT": measurement_sets_count,
		"MEASUREMENT_NODES": list(measurement_nodes) if measurement_nodes is not None else None,
		"PIPE_BOUNDS": PIPE_BOUNDS,
		"PIPE_BOUND_SAFENESS": PIPE_BOUND_SAFENESS,
		"PIPE_BOUND_METHOD": PIPE_BOUND_METHOD,
		"PIPE_BOUND_POLICY": PIPE_BOUND_POLICY,
		"PIPE_BOUND_PERTURB_SAMPLES": PIPE_BOUND_PERTURB_SAMPLES,
		"PIPE_BOUND_PERTURB_SIGMA": PIPE_BOUND_PERTURB_SIGMA,
		"PIPE_BOUND_PERTURB_SEED": PIPE_BOUND_PERTURB_SEED,
		"PIPE_BOUND_PERTURB_FIX_TOTAL": PIPE_BOUND_PERTURB_FIX_TOTAL,
		"PIPE_BOUND_PERTURB_BASE": PIPE_BOUND_PERTURB_BASE,
		"SCENARIO_SOURCE": SCENARIO_SOURCE,
		"HEADLOSS_MODEL": HEADLOSS_MODEL,
		"HEADLOSS_MODEL_LOCAL": headloss_model_local,
		"SOLVER": SOLVER,
		"GLOBAL_SOLVER": None,
		"DEMAND_INTERVALS": DEMAND_INTERVALS,
		"SINGLE_NODE_SOLUTION": SINGLE_NODE_SOLUTION,
		"RADIUS": RADIUS,
		"NORM": NORM,
		"DEMAND_LB": DEMAND_LB,
		"USE_RESERVOIR_OUTFLOW_CONSTRAINT": USE_RESERVOIR_OUTFLOW_CONSTRAINT,
		"FIX_TOTAL_DEMAND": FIX_TOTAL_DEMAND,
		"MEASUREMENT_HEADS_EQUAL_ONLY": MEASUREMENT_HEADS_EQUAL_ONLY,
		"MULTI_STARTS": MULTI_STARTS,
		"MULTI_START_NOISE": MULTI_START_NOISE,
		"MULTI_START_NOISE_REL": MULTI_START_NOISE_REL,
		"MULTI_START_SEED": MULTI_START_SEED,
		"SHOW_INTERMEDIATE_PLOTS": SHOW_INTERMEDIATE_PLOTS,
		"SHOW_INTERMEDIATE_PLOTS_MULTIPLE": SHOW_INTERMEDIATE_PLOTS_MULTIPLE,
		"DEBUG_DUMP_ALWAYS": DEBUG_DUMP_ALWAYS,
		"DEBUG_DUMP_EPS": DEBUG_DUMP_EPS,
		"CHECK_XD_BASE_FEASIBILITY": CHECK_XD_BASE_FEASIBILITY,
		"CHECK_XD_CYCLE_BOUNDS": CHECK_XD_CYCLE_BOUNDS,
		"XD_CYCLE_BASIS_MODE": XD_CYCLE_BASIS_MODE,
		"SKIP_FEASIBILITY_SOLVE": SKIP_FEASIBILITY_SOLVE,
		"FIXED_REFERENCE": FIXED_REFERENCE,
		"DEMAND_RESTRICTION_MODE": DEMAND_RESTRICTION_MODE,
		"RADIUS_TO_FIXED": RADIUS_TO_FIXED,
		"DEVIATION_ALPHA": DEVIATION_ALPHA,
		"HEXALY_LICENSE_PATH": HEXALY_LICENSE_PATH,
		"HEXALY_TIME_LIMIT": HEXALY_TIME_LIMIT,
		"HEXALY_STAGNATION_SECONDS": HEXALY_STAGNATION_SECONDS,
		"HEXALY_STAGNATION_EPS": HEXALY_STAGNATION_EPS,
		"HEXALY_SEED": HEXALY_SEED,
		"HEXALY_VERBOSITY": HEXALY_VERBOSITY,
		"HEXALY_HEAD_MARGIN": HEXALY_HEAD_MARGIN,
		"HEXALY_USE_PATH_HEAD_BOUNDS": HEXALY_USE_PATH_HEAD_BOUNDS,
	}


def _compute_total_demand_from_flows(network, flows: Dict[str, float]) -> float:
	reservoir_nodes = set(network.reservoirs.keys())
	junctions = [j for j in network.junctions.keys() if j not in reservoir_nodes]
	if not junctions:
		return 0.0
	values: Dict[str, float] = {j: 0.0 for j in junctions}
	for pipe_id, pipe in network.pipes.items():
		q = flows.get(pipe_id, 0.0)
		if pipe.end_node in values:
			values[pipe.end_node] += q
		if pipe.start_node in values:
			values[pipe.start_node] -= q
	return float(sum(values.values()))


def _compute_demands_from_flows(network, flows: Dict[str, float]) -> Dict[str, float]:
	reservoir_nodes = set(network.reservoirs.keys())
	junctions = [j for j in network.junctions.keys() if j not in reservoir_nodes]
	values: Dict[str, float] = {j: 0.0 for j in junctions}
	for pipe_id, pipe in network.pipes.items():
		q = flows.get(pipe_id, 0.0)
		if pipe.end_node in values:
			values[pipe.end_node] += q
		if pipe.start_node in values:
			values[pipe.start_node] -= q
	return values


def plot_network(
	inp_path: str,
	base_flows: Dict[str, float],
	c_bounds: Dict[str, float],
	C_bounds: Dict[str, float],
	output_dir: str,
) -> None:
	import matplotlib.pyplot as plt
	import networkx as nx
	import wntr

	wn = wntr.network.WaterNetworkModel(inp_path)
	pos = {name: (node.coordinates[0], node.coordinates[1]) for name, node in wn.nodes()}

	G = nx.DiGraph()
	for name, node in wn.nodes():
		G.add_node(name)
	for pipe_name, pipe in wn.pipes():
		G.add_edge(pipe.start_node_name, pipe.end_node_name, key=pipe_name)

	fig, ax = plt.subplots(figsize=(9, 7))
	reservoir_nodes = [n for n in wn.reservoir_name_list if n in G.nodes]
	nx.draw_networkx_nodes(G, pos, node_size=350, node_color="#f2f2f2", edgecolors="#333333", ax=ax)
	nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
	if reservoir_nodes:
		nx.draw_networkx_nodes(G, pos, nodelist=reservoir_nodes, node_size=420, node_color="#90cdf4", node_shape="s", edgecolors="#1a365d", ax=ax)

	primary_edges = []
	secondary_edges = []
	edge_to_pipe = {}
	for pipe_name, pipe in wn.pipes():
		cl = c_bounds.get(pipe_name, float("-inf"))
		cu = C_bounds.get(pipe_name, float("inf"))
		edge = (pipe.start_node_name, pipe.end_node_name)
		edge_to_pipe[edge] = pipe_name
		if cl == float("-inf") or cu == float("inf"):
			secondary_edges.append(edge)
		elif cl * cu >= 0:
			primary_edges.append(edge)
		else:
			secondary_edges.append(edge)

	nx.draw_networkx_edges(
		G,
		pos,
		edgelist=secondary_edges,
		edge_color="#2b6cb0",
		arrows=False,
		width=2.0,
		ax=ax,
	)

	nx.draw_networkx_edges(
		G,
		pos,
		edgelist=primary_edges,
		edge_color="#000000",
		arrows=True,
		arrowstyle="-|>",
		arrowsize=12,
		width=2.2,
		ax=ax,
	)

	for (u, v), pipe_name in edge_to_pipe.items():
		x = (pos[u][0] + pos[v][0]) / 2.0
		y = (pos[u][1] + pos[v][1]) / 2.0
		q0 = base_flows.get(pipe_name, 0.0)
		cl = c_bounds.get(pipe_name, float("-inf"))
		cu = C_bounds.get(pipe_name, float("inf"))
		label = f"q0={q0:.4f}\n[{cl:.4f}, {cu:.4f}]"
		ax.text(x, y, label, fontsize=7, ha="center", va="center", color="#111111")

	ax.set_title("Pipe classes, base flows, and capacity bounds")
	ax.axis("off")
	plt.tight_layout()
	fig.savefig(os.path.join(output_dir, "pipe_classes.png"), dpi=200)
	plt.close(fig)


def plot_demand_bounds(
	inp_path: str,
	measurement_nodes: Iterable[str],
	actual_demands: Dict[str, float],
	bounds: DemandBounds,
	base_flows: Dict[str, float],
	c_bounds: Dict[str, float],
	C_bounds: Dict[str, float],
	sensor_heads: Dict[str, float],
	scenario_flows: Dict[str, float],
	base_demands: Dict[str, float],
	output_dir: str,
	show_plot: bool = True,
) -> None:
	import matplotlib.pyplot as plt
	import networkx as nx
	import wntr

	wn = wntr.network.WaterNetworkModel(inp_path)
	pos = {name: (node.coordinates[0], node.coordinates[1]) for name, node in wn.nodes()}

	G = nx.DiGraph()
	for name, node in wn.nodes():
		G.add_node(name)
	for pipe_name, pipe in wn.pipes():
		G.add_edge(pipe.start_node_name, pipe.end_node_name, key=pipe_name)

	fig, ax = plt.subplots(figsize=(9, 7))

	measurement_set = set(measurement_nodes)
	reservoir_nodes = [n for n in wn.reservoir_name_list if n in G.nodes]
	normal_nodes = [n for n in G.nodes if n not in measurement_set and n not in reservoir_nodes]
	meas_nodes = [n for n in G.nodes if n in measurement_set and n not in reservoir_nodes]
	nx.draw_networkx_nodes(G, pos, nodelist=normal_nodes, node_size=380, node_color="#f2f2f2", edgecolors="#333333", ax=ax)
	nx.draw_networkx_nodes(G, pos, nodelist=meas_nodes, node_size=420, node_color="#90cdf4", node_shape="h", edgecolors="#1a365d", ax=ax)
	nx.draw_networkx_edges(
		G,
		pos,
		edge_color="#a0aec0",
		width=1.6,
		arrows=True,
		arrowstyle="-|>",
		arrowsize=12,
		ax=ax,
	)
	if reservoir_nodes:
		nx.draw_networkx_nodes(G, pos, nodelist=reservoir_nodes, node_size=420, node_color="#90cdf4", node_shape="s", edgecolors="#1a365d", ax=ax)
	nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)

	for node_id in wn.junction_name_list:
		x, y = pos[node_id]
		actual = actual_demands.get(node_id, 0.0)
		dmin = bounds.d_min.get(node_id, float("nan"))
		dmax = bounds.d_max.get(node_id, float("nan"))
		d0 = base_demands.get(node_id, 0.0)
		label = f"d*={actual:.4f}\n[{dmin:.4f}, {dmax:.4f}]\nd0={d0:.4f}"
		ax.text(x, y - 180, label, fontsize=7, ha="center", va="top", color="#111111")

	for node_id in wn.reservoir_name_list:
		if node_id not in pos:
			continue
		x, y = pos[node_id]
		head = sensor_heads.get(node_id, float("nan"))
		ax.text(x, y - 180, f"h={head:.4f}", fontsize=7, ha="center", va="top", color="#111111")

	for pipe_name, pipe in wn.pipes():
		u = pipe.start_node_name
		v = pipe.end_node_name
		x = (pos[u][0] + pos[v][0]) / 2.0
		y = (pos[u][1] + pos[v][1]) / 2.0
		q0 = base_flows.get(pipe_name, 0.0)
		qs = scenario_flows.get(pipe_name, 0.0)
		cl = c_bounds.get(pipe_name, float("-inf"))
		cu = C_bounds.get(pipe_name, float("inf"))
		label = f"q0={q0:.4f}\nq*={qs:.4f}\n[{cl:.4f}, {cu:.4f}]"
		ax.text(x, y + 120, label, fontsize=7, ha="center", va="bottom", color="#111111")

	ax.set_title("Demand bounds vs actual demands")
	ax.axis("off")
	plt.tight_layout()
	fig.savefig(os.path.join(output_dir, "demand_bounds.png"), dpi=200)
	if show_plot:
		plt.show(block=True)
	plt.close(fig)


def plot_single_solution(
	inp_path: str,
	measurement_nodes: Iterable[str],
	solution: SingleNodeResult,
	c_bounds: Dict[str, float],
	C_bounds: Dict[str, float],
	output_dir: str,
	show_plot: bool = True,
) -> None:
	import matplotlib.pyplot as plt
	import networkx as nx
	import wntr

	wn = wntr.network.WaterNetworkModel(inp_path)
	pos = {name: (node.coordinates[0], node.coordinates[1]) for name, node in wn.nodes()}

	G = nx.DiGraph()
	for name, node in wn.nodes():
		G.add_node(name)
	for pipe_name, pipe in wn.pipes():
		q = solution.flows.get(pipe_name, 0.0)
		if q >= 0:
			G.add_edge(pipe.start_node_name, pipe.end_node_name, key=pipe_name)
		else:
			G.add_edge(pipe.end_node_name, pipe.start_node_name, key=pipe_name)

	fig, ax = plt.subplots(figsize=(9, 7))

	measurement_set = set(measurement_nodes)
	reservoir_nodes = [n for n in wn.reservoir_name_list if n in G.nodes]
	normal_nodes = [n for n in G.nodes if n not in measurement_set and n not in reservoir_nodes]
	meas_nodes = [n for n in G.nodes if n in measurement_set and n not in reservoir_nodes]

	nx.draw_networkx_nodes(G, pos, nodelist=normal_nodes, node_size=380, node_color="#f2f2f2", edgecolors="#333333", ax=ax)
	nx.draw_networkx_nodes(G, pos, nodelist=meas_nodes, node_size=420, node_color="#90cdf4", node_shape="h", edgecolors="#1a365d", ax=ax)
	if reservoir_nodes:
		nx.draw_networkx_nodes(G, pos, nodelist=reservoir_nodes, node_size=420, node_color="#90cdf4", node_shape="s", edgecolors="#1a365d", ax=ax)
	nx.draw_networkx_edges(
		G,
		pos,
		edge_color="#a0aec0",
		width=1.6,
		arrows=True,
		arrowstyle="-|>",
		arrowsize=12,
		ax=ax,
	)
	nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)

	for node_id in wn.junction_name_list:
		x, y = pos[node_id]
		d = solution.demands.get(node_id, 0.0)
		h = solution.heads.get(node_id, float("nan"))
		label = f"d={d:.4f}\nh={h:.4f}"
		ax.text(x, y - 180, label, fontsize=7, ha="center", va="top", color="#111111")

	for pipe_name, pipe in wn.pipes():
		q = solution.flows.get(pipe_name, 0.0)
		if q >= 0:
			u = pipe.start_node_name
			v = pipe.end_node_name
		else:
			u = pipe.end_node_name
			v = pipe.start_node_name
		x = (pos[u][0] + pos[v][0]) / 2.0
		y = (pos[u][1] + pos[v][1]) / 2.0
		q_abs = abs(q)
		cl = c_bounds.get(pipe_name, float("-inf"))
		cu = C_bounds.get(pipe_name, float("inf"))
		label = f"q={q_abs:.4f}\n[{cl:.4f}, {cu:.4f}]"
		ax.text(x, y + 120, label, fontsize=7, ha="center", va="bottom", color="#111111")

	ax.set_title(f"Single-node min solution for node {solution.node_id}")
	ax.axis("off")
	plt.tight_layout()
	fig.savefig(os.path.join(output_dir, "single_node_solution.png"), dpi=200)
	if show_plot:
		plt.show(block=True)
	plt.close(fig)


def plot_demand_distance(
	inp_path: str,
	measurement_nodes: Iterable[str],
	demands_a: Dict[str, float],
	demands_b: Dict[str, float],
	heads_a: Dict[str, float],
	heads_b: Dict[str, float],
	flows_a: Dict[str, float],
	flows_b: Dict[str, float],
	radius: float,
	norm_p: float,
	measurement_count: int,
	output_dir: str,
	c_bounds: Dict[str, float],
	C_bounds: Dict[str, float],
	head_bounds: Dict[str, Tuple[float, float]] | None = None,
	show_head_bounds_measurements: bool = False,
	show_plot: bool = True,
) -> None:
	import matplotlib.pyplot as plt
	from matplotlib.offsetbox import AnnotationBbox, TextArea, HPacker, VPacker
	import networkx as nx
	import wntr

	wn = wntr.network.WaterNetworkModel(inp_path)
	pos = {name: (node.coordinates[0], node.coordinates[1]) for name, node in wn.nodes()}

	G = nx.Graph()
	for name, node in wn.nodes():
		G.add_node(name)
	for pipe_name, pipe in wn.pipes():
		G.add_edge(pipe.start_node_name, pipe.end_node_name)

	fig, ax = plt.subplots(figsize=(9, 7))

	measurement_set = set(measurement_nodes)
	reservoir_nodes = [n for n in wn.reservoir_name_list if n in G.nodes]
	normal_nodes = [n for n in G.nodes if n not in measurement_set and n not in reservoir_nodes]
	meas_nodes = [n for n in G.nodes if n in measurement_set and n not in reservoir_nodes]

	nx.draw_networkx_nodes(G, pos, nodelist=normal_nodes, node_size=380, node_color="#f2f2f2", edgecolors="#333333", ax=ax)
	nx.draw_networkx_nodes(G, pos, nodelist=meas_nodes, node_size=420, node_color="#90cdf4", node_shape="h", edgecolors="#1a365d", ax=ax)
	if reservoir_nodes:
		nx.draw_networkx_nodes(G, pos, nodelist=reservoir_nodes, node_size=420, node_color="#90cdf4", node_shape="s", edgecolors="#1a365d", ax=ax)
	# Color edges by direction agreement between A and B
	edge_colors = []
	for pipe_name, pipe in wn.pipes():
		qa = flows_a.get(pipe_name, 0.0)
		qb = flows_b.get(pipe_name, 0.0)
		if qa != 0.0 and qb != 0.0 and (qa > 0) != (qb > 0):
			edge_colors.append("#2b6cb0")
		else:
			edge_colors.append("#000000")
	nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=1.6, ax=ax)
	nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)

	def _colored_inline(ax, x, y, segments, colors):
		areas = [TextArea(text, textprops={"color": color, "fontsize": 7})
			for text, color in zip(segments, colors)]
		box = HPacker(children=areas, align="center", pad=0, sep=2)
		ab = AnnotationBbox(
			box,
			(x, y),
			frameon=True,
			bboxprops={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.75},
		)
		ax.add_artist(ab)

	def _colored_two_lines(ax, x, y, segments_top, colors_top, segments_bottom, colors_bottom):
		areas_top = [TextArea(text, textprops={"color": color, "fontsize": 7})
			for text, color in zip(segments_top, colors_top)]
		areas_bottom = [TextArea(text, textprops={"color": color, "fontsize": 7})
			for text, color in zip(segments_bottom, colors_bottom)]
		line_top = HPacker(children=areas_top, align="center", pad=0, sep=2)
		line_bottom = HPacker(children=areas_bottom, align="center", pad=0, sep=2)
		box = VPacker(children=[line_top, line_bottom], align="center", pad=0, sep=1)
		ab = AnnotationBbox(
			box,
			(x, y),
			frameon=True,
			bboxprops={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.75},
		)
		ax.add_artist(ab)

	def _colored_three_lines(ax, x, y, segments_top, colors_top, segments_mid, colors_mid, segments_bottom, colors_bottom):
		areas_top = [TextArea(text, textprops={"color": color, "fontsize": 7})
			for text, color in zip(segments_top, colors_top)]
		areas_mid = [TextArea(text, textprops={"color": color, "fontsize": 7})
			for text, color in zip(segments_mid, colors_mid)]
		areas_bottom = [TextArea(text, textprops={"color": color, "fontsize": 7})
			for text, color in zip(segments_bottom, colors_bottom)]
		line_top = HPacker(children=areas_top, align="center", pad=0, sep=2)
		line_mid = HPacker(children=areas_mid, align="center", pad=0, sep=2)
		line_bottom = HPacker(children=areas_bottom, align="center", pad=0, sep=2)
		box = VPacker(children=[line_top, line_mid, line_bottom], align="center", pad=0, sep=1)
		ab = AnnotationBbox(
			box,
			(x, y),
			frameon=True,
			bboxprops={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.75},
		)
		ax.add_artist(ab)

	for node_id in wn.junction_name_list:
		x, y = pos[node_id]
		da = demands_a.get(node_id, 0.0)
		db = demands_b.get(node_id, 0.0)
		ha = heads_a.get(node_id, float("nan"))
		hb = heads_b.get(node_id, float("nan"))
		green = "#2f855a"
		segments_d = ["d = ", f"{da:.4f}", " | ", f"{db:.4f}"]
		colors_d = ["#111111", "#2b6cb0", "#111111", green]
		segments_h = ["h = ", f"{ha:.4f}", " | ", f"{hb:.4f}"]
		colors_h = ["#111111", "#2b6cb0", "#111111", green]
		show_bounds = head_bounds is not None and node_id in head_bounds
		if show_bounds and (node_id not in measurement_set or show_head_bounds_measurements):
			lb, ub = head_bounds[node_id]
			segments_b = ["h in [", f"{lb:.4f}", ", ", f"{ub:.4f}", "]"]
			colors_b = ["#111111"] * len(segments_b)
			_colored_three_lines(ax, x, y - 220, segments_d, colors_d, segments_h, colors_h, segments_b, colors_b)
		else:
			_colored_two_lines(ax, x, y - 210, segments_d, colors_d, segments_h, colors_h)

	for node_id in wn.reservoir_name_list:
		if node_id not in pos:
			continue
		x, y = pos[node_id]
		ha = heads_a.get(node_id, float("nan"))
		hb = heads_b.get(node_id, float("nan"))
		da = -sum(demands_a.values())
		db = -sum(demands_b.values())
		green = "#2f855a"
		segments_d = ["d = ", f"{da:.4f}", " | ", f"{db:.4f}"]
		colors_d = ["#111111", "#2b6cb0", "#111111", green]
		segments_h = ["h = ", f"{ha:.4f}", " | ", f"{hb:.4f}"]
		colors_h = ["#111111", "#2b6cb0", "#111111", green]
		_colored_two_lines(ax, x, y - 210, segments_d, colors_d, segments_h, colors_h)

	for pipe_name, pipe in wn.pipes():
		u = pipe.start_node_name
		v = pipe.end_node_name
		x = (pos[u][0] + pos[v][0]) / 2.0
		y = (pos[u][1] + pos[v][1]) / 2.0
		qa = flows_a.get(pipe_name, 0.0)
		qb = flows_b.get(pipe_name, 0.0)
		green = "#2f855a"
		segments_q = ["q = ", f"{qa:.4f}", " | ", f"{qb:.4f}"]
		colors_q = ["#111111", "#2b6cb0", "#111111", green]
		cl = c_bounds.get(pipe_name, float("-inf"))
		cu = C_bounds.get(pipe_name, float("inf"))
		if cl != float("-inf") or cu != float("inf"):
			segments_b = ["[", f"{cl:.4f}", ", ", f"{cu:.4f}", "]"]
			colors_b = ["#111111", "#111111", "#111111", "#111111", "#111111"]
			_colored_two_lines(ax, x, y + 150, segments_q, colors_q, segments_b, colors_b)
		else:
			_colored_inline(ax, x, y + 150, segments_q, colors_q)

	if norm_p == float("inf"):
		norm_label = "inf"
	else:
		norm_label = f"{norm_p:g}"
	ax.set_title(
		f"Max demand-distance solution ({norm_label}-norm, p={measurement_count}, radius={radius:.5f})"
	)
	ax.axis("off")
	plt.tight_layout()
	fig.savefig(os.path.join(output_dir, "demand_distance.png"), dpi=200)
	if show_plot:
		plt.show(block=True)
	plt.close(fig)


def _compute_norm(values: Iterable[float], norm_p: float) -> float:
	import math

	vals = [abs(v) for v in values]
	if not vals:
		return 0.0
	if math.isinf(norm_p):
		return max(vals)
	if norm_p <= 0:
		raise ValueError("NORM must be positive.")
	acc = 0.0
	for v in vals:
		acc += v ** norm_p
	return acc ** (1.0 / norm_p)


def main() -> None:
	inp_path = f"./wdn/{WDN_NAME}.inp"
	wdn_name = WDN_NAME
	network = load_inp_network(inp_path)
	print(f"Using network: {wdn_name} ({inp_path})")
	resistances = compute_pipe_resistances(network)
	print(f"Loaded {len(network.nodes)} nodes and {len(network.pipes)} pipes.")
	first_pipe = next(iter(resistances.values()), None)
	if first_pipe is not None:
		print("Example resistance:", first_pipe)

	base_flows = simulate_base_flows(inp_path=inp_path)
	if PIPE_BOUNDS:
		if PIPE_BOUND_METHOD == "perturb":
			base_demands_override = None
			if PIPE_BOUND_PERTURB_BASE in {"base", "auto"}:
				try:
					simulate_base_flows(inp_path=inp_path, simulator="auto")
				except Exception:
					if PIPE_BOUND_PERTURB_BASE == "base":
						raise
					base_demands_override, _, _ = simulate_single_random_scenario(
						inp_path=inp_path,
						demand_log_sigma=0.3,
						seed=2,
						simulator="auto",
					)
			elif PIPE_BOUND_PERTURB_BASE == "scenario":
				base_demands_override, _, _ = simulate_single_random_scenario(
					inp_path=inp_path,
					demand_log_sigma=0.3,
					seed=2,
					simulator="auto",
				)
			else:
				raise ValueError("PIPE_BOUND_PERTURB_BASE must be 'auto', 'base', or 'scenario'.")

			flows_by_pipe = simulate_perturbed_demand_scenarios(
				inp_path=inp_path,
				n_scenarios=PIPE_BOUND_PERTURB_SAMPLES,
				demand_sigma=PIPE_BOUND_PERTURB_SIGMA,
				base_demands=base_demands_override,
				enforce_total_demand=PIPE_BOUND_PERTURB_FIX_TOTAL,
				seed=PIPE_BOUND_PERTURB_SEED,
				simulator="auto",
			)
		else:
			flows_by_pipe = simulate_random_demand_scenarios(
				inp_path=inp_path,
				n_scenarios=200,
				demand_log_sigma=0.3,
				seed=1,
				simulator="auto",
			)

		c_bounds, C_bounds = _compute_pipe_bounds_from_samples(
			flows_by_pipe,
			PIPE_BOUND_POLICY,
			PIPE_BOUND_SAFENESS,
		)
	else:
		c_bounds = {}
		C_bounds = {}

	if SCENARIO_SOURCE == "base":
		scenario_demands_pos, scenario_heads, scenario_flows = simulate_base_scenario(
			inp_path=inp_path,
			simulator="auto",
		)
	elif SCENARIO_SOURCE == "random":
		scenario_demands_pos, scenario_heads, scenario_flows = simulate_single_random_scenario(
			inp_path=inp_path,
			demand_log_sigma=0.3,
			seed=2,
			simulator="auto",
		)
	else:
		raise ValueError("SCENARIO_SOURCE must be 'base' or 'random'.")
	scenario_demands = {k: v for k, v in scenario_demands_pos.items()}
	total_demand = float(sum(scenario_demands.values()))
	total_demand_from_flows = _compute_total_demand_from_flows(network, scenario_flows)

	measurement_sets = _select_measurement_sets(network)
	if not measurement_sets:
		print("WARNING: no measurement sets selected")
		return
	if isinstance(MEASUREMENT_SITES, int):
		mode = MEASUREMENT_SUBSET_MODE
		print(f"Measurement subsets: {len(measurement_sets)} (p={MEASUREMENT_SITES}, mode={mode})")
	show_intermediate_plots = SHOW_INTERMEDIATE_PLOTS
	if len(measurement_sets) > 1:
		show_intermediate_plots = SHOW_INTERMEDIATE_PLOTS_MULTIPLE

	master_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
	pipe_tag = "B" if PIPE_BOUNDS else "NB"
	summary_rows = []
	batch_dir = None
	if len(measurement_sets) > 1:
		batch_dir = os.path.join("data", f"{wdn_name}-{pipe_tag}-batch-{master_timestamp}")
		os.makedirs(batch_dir, exist_ok=True)
		_write_json(
			os.path.join(batch_dir, "parameters.json"),
			_build_parameters_snapshot(measurement_sets_count=len(measurement_sets)),
		)

	for set_idx, measurement_nodes in enumerate(measurement_sets, start=1):
		total_demand_solver = total_demand
		if SOLVER == "Hexaly_xd" and FIX_TOTAL_DEMAND:
			total_demand_solver = total_demand_from_flows
		reference_demands = dict(scenario_demands)
		reference_demands_from_flows = _compute_demands_from_flows(network, scenario_flows)
		restriction_mode = DEMAND_RESTRICTION_MODE
		if restriction_mode is not None:
			if restriction_mode not in {"radius_to_fixed", "deviation_to_fixed"}:
				raise ValueError("DEMAND_RESTRICTION_MODE must be None, 'radius_to_fixed', or 'deviation_to_fixed'.")
		reference_demands_restriction = reference_demands
		if SOLVER == "Hexaly_xd":
			reference_demands_restriction = reference_demands_from_flows
		reference_heads = dict(scenario_heads)
		reference_flows = dict(scenario_flows)
		measurement_tag = "_".join(measurement_nodes) if measurement_nodes else "none"
		if batch_dir:
			output_dir = os.path.join(batch_dir, measurement_tag)
		else:
			output_dir = os.path.join("data", f"{wdn_name}-{pipe_tag}-{measurement_tag}-{master_timestamp}")
		os.makedirs(output_dir, exist_ok=True)
		print(f"Saving results to: {output_dir}")
		model_opt = ""
		if HEADLOSS_MODEL == "auto":
			opt = network.options.get("headloss") or network.options.get("hydraulic")
			model_opt = str(opt).lower() if opt is not None else ""
			if "hazen" in model_opt or "hw" in model_opt:
				HEADLOSS_MODEL_LOCAL = "hw"
			else:
				HEADLOSS_MODEL_LOCAL = "dw"
		else:
			HEADLOSS_MODEL_LOCAL = HEADLOSS_MODEL
		model_note = model_opt if model_opt else "n/a"
		print(f"Headloss model: {HEADLOSS_MODEL_LOCAL} (options: {model_note})")

		_write_json(
			os.path.join(output_dir, "parameters.json"),
			_build_parameters_snapshot(
				headloss_model_local=HEADLOSS_MODEL_LOCAL,
				measurement_nodes=measurement_nodes,
				measurement_sets_count=len(measurement_sets),
			),
		)

		_write_json(
			os.path.join(output_dir, "metadata.json"),
			{
				"WDN_NAME": WDN_NAME,
				"MEASUREMENT_SITES": measurement_nodes,
				"PIPE_BOUNDS": PIPE_BOUNDS,
				"PIPE_BOUND_SAFENESS": PIPE_BOUND_SAFENESS,
				"PIPE_BOUND_METHOD": PIPE_BOUND_METHOD,
				"PIPE_BOUND_POLICY": PIPE_BOUND_POLICY,
				"PIPE_BOUND_PERTURB_SAMPLES": PIPE_BOUND_PERTURB_SAMPLES,
				"PIPE_BOUND_PERTURB_SIGMA": PIPE_BOUND_PERTURB_SIGMA,
				"PIPE_BOUND_PERTURB_SEED": PIPE_BOUND_PERTURB_SEED,
				"PIPE_BOUND_PERTURB_FIX_TOTAL": PIPE_BOUND_PERTURB_FIX_TOTAL,
				"PIPE_BOUND_PERTURB_BASE": PIPE_BOUND_PERTURB_BASE,
				"SCENARIO_SOURCE": SCENARIO_SOURCE,
				"HEADLOSS_MODEL": HEADLOSS_MODEL_LOCAL,
				"SOLVER": SOLVER,
				"GLOBAL_SOLVER": None,
				"DEMAND_INTERVALS": DEMAND_INTERVALS,
				"SINGLE_NODE_SOLUTION": SINGLE_NODE_SOLUTION,
				"RADIUS": RADIUS,
				"NORM": NORM,
				"DEMAND_LB": DEMAND_LB,
				"USE_RESERVOIR_OUTFLOW_CONSTRAINT": USE_RESERVOIR_OUTFLOW_CONSTRAINT,
				"FIX_TOTAL_DEMAND": FIX_TOTAL_DEMAND,
				"MEASUREMENT_HEADS_EQUAL_ONLY": MEASUREMENT_HEADS_EQUAL_ONLY,
				"MULTI_STARTS": MULTI_STARTS,
				"MULTI_START_NOISE": MULTI_START_NOISE,
				"MULTI_START_NOISE_REL": MULTI_START_NOISE_REL,
				"DEMAND_RESTRICTION_MODE": DEMAND_RESTRICTION_MODE,
				"RADIUS_TO_FIXED": RADIUS_TO_FIXED,
				"DEVIATION_ALPHA": DEVIATION_ALPHA,
				"timestamp": master_timestamp,
				"output_dir": output_dir,
			},
		)
		if PIPE_BOUNDS:
			_write_json(
				os.path.join(output_dir, "c_bounds.json"),
				{"c_bounds": c_bounds, "C_bounds": C_bounds},
			)

		plot_network(
			inp_path=inp_path,
			base_flows=base_flows,
			c_bounds=c_bounds,
			C_bounds=C_bounds,
			output_dir=output_dir,
		)

		sensor_heads = {node_id: scenario_heads[node_id] for node_id in measurement_nodes if node_id in scenario_heads}
		for res_id in network.reservoirs.keys():
			if res_id in scenario_heads:
				sensor_heads[res_id] = scenario_heads[res_id]

		initial_guess = SolverResult(
			status="init",
			demands=scenario_demands,
			heads=scenario_heads,
			flows=scenario_flows,
		)

		if HEADLOSS_MODEL_LOCAL == "hw":
			pipe_resistances = {pid: vals["r_e"] for pid, vals in compute_pipe_resistances_hw(network).items()}
			headloss_n = 1.852
		else:
			pipe_resistances = {pid: vals["r_e"] for pid, vals in resistances.items()}
			headloss_n = 2.0

		reservoir_node = next(iter(network.reservoirs.keys()), None)
		if SOLVER == "Hexaly_xd":
			_print_xd_paths_cycles(
				network,
				pipe_resistances,
				reservoir_node,
				measurement_nodes,
				cycle_basis_mode=XD_CYCLE_BASIS_MODE,
			)
		reservoir_head = None
		if reservoir_node is not None:
			reservoir_head = scenario_heads.get(reservoir_node, None)
		reservoir_outflow = 0.0
		if reservoir_node is not None:
			for pid, pipe in network.pipes.items():
				q = scenario_flows.get(pid, 0.0)
				if pipe.start_node == reservoir_node:
					reservoir_outflow += q
				if pipe.end_node == reservoir_node:
					reservoir_outflow -= q
		if not USE_RESERVOIR_OUTFLOW_CONSTRAINT:
			reservoir_outflow = None
		preferred_flow_sign = {pid: q for pid, q in scenario_flows.items()}

		if SOLVER == "Hexaly_xd" and CHECK_XD_BASE_FEASIBILITY:
			check = check_xd_feasibility_from_flows(
				network=network,
				pipe_resistances=pipe_resistances,
				flows=scenario_flows,
				c_bounds=c_bounds if PIPE_BOUNDS else None,
				C_bounds=C_bounds if PIPE_BOUNDS else None,
				reservoir_node=reservoir_node,
				measurement_nodes=measurement_nodes,
				demand_lb=DEMAND_LB,
				total_demand=total_demand_solver if FIX_TOTAL_DEMAND else None,
				reservoir_outflow=reservoir_outflow,
				headloss_n=headloss_n,
				cycle_basis_mode=XD_CYCLE_BASIS_MODE,
			)
			print("XD base scenario feasibility check:")
			for key, value in check.items():
				print(f"  {key}: {value:.3e}")

		if SOLVER == "Hexaly_xd" and CHECK_XD_CYCLE_BOUNDS and PIPE_BOUNDS:
			cycle_check = check_xd_cycle_feasibility_from_bounds(
				network=network,
				pipe_resistances=pipe_resistances,
				c_bounds=c_bounds if PIPE_BOUNDS else None,
				C_bounds=C_bounds if PIPE_BOUNDS else None,
				initial_guess=initial_guess,
				headloss_n=headloss_n,
				cycle_basis_mode=XD_CYCLE_BASIS_MODE,
				max_examples=5,
			)
			print("XD cycle feasibility from bounds:")
			print(
				"  cycles={}, infeasible={}, max_violation={:.3e}".format(
					cycle_check["cycle_count"],
					cycle_check["infeasible_count"],
					cycle_check["max_violation"],
				)
			)
			for row in cycle_check.get("examples", []):
				print(
					"  cycle #{cycle_index}: min_sum={min_sum:.3e}, max_sum={max_sum:.3e}, size={size}".format(
						**row
					)
				)

		if PIPE_BOUNDS:
			base_violation = check_xd_feasibility_from_flows(
				network=network,
				pipe_resistances=pipe_resistances,
				flows=scenario_flows,
				c_bounds=c_bounds,
				C_bounds=C_bounds,
				reservoir_node=reservoir_node,
				measurement_nodes=measurement_nodes,
				demand_lb=DEMAND_LB,
				total_demand=total_demand_solver if FIX_TOTAL_DEMAND else None,
				reservoir_outflow=reservoir_outflow,
				headloss_n=headloss_n,
				cycle_basis_mode=XD_CYCLE_BASIS_MODE,
			)
			print("Base scenario feasibility with current bounds:")
			for key, value in base_violation.items():
				print(f"  {key}: {value:.3e}")

		head_bounds_for_plot = None
		if PIPE_BOUNDS:
			fixed_heads = dict(sensor_heads)
			if MEASUREMENT_HEADS_EQUAL_ONLY:
				fixed_heads = {k: v for k, v in fixed_heads.items() if k not in measurement_nodes}
			for res_id, res_node in network.reservoirs.items():
				if res_id not in fixed_heads:
					fixed_heads[res_id] = res_node.elevation_m
			head_bounds_for_plot = _compute_head_bounds_from_fixed_heads(
				network=network,
				pipe_resistances=pipe_resistances,
				fixed_heads=fixed_heads,
				c_bounds=c_bounds,
				C_bounds=C_bounds,
				head_margin=HEXALY_HEAD_MARGIN,
			)
		if SKIP_FEASIBILITY_SOLVE:
			print("Skipping feasibility solve (using scenario flows/heads as initial guess).")
		else:
			feasible = solve_feasibility(
				network=network,
				sensor_heads=sensor_heads,
				measurement_nodes=measurement_nodes,
				measurement_heads_equal_only=MEASUREMENT_HEADS_EQUAL_ONLY,
				pipe_primary=set(),
				pipe_secondary=set(),
				c_secondary={},
				C_primary={},
				pipe_resistances=pipe_resistances,
				preferred_flow_sign=preferred_flow_sign,
				energy_head=reservoir_head,
				reservoir_node=reservoir_node,
				reservoir_outflow=reservoir_outflow,
				c_bounds=c_bounds if PIPE_BOUNDS else None,
				C_bounds=C_bounds if PIPE_BOUNDS else None,
				initial_guess=initial_guess,
			)
			print("Feasibility result:")
			print(
				f"success={feasible.success}, max_violation={feasible.max_violation:.3e}, min_demand_viol={feasible.min_demand_viol:.3e}",
				flush=True,
			)
			initial_guess = SolverResult(
				status="feasible",
				demands=scenario_demands,
				heads=feasible.heads,
				flows=feasible.flows,
			)

		if DEMAND_INTERVALS:
			bounds = solve_demand_bounds(
				network=network,
				sensor_heads=sensor_heads,
				pipe_primary=set(),
				pipe_secondary=set(),
				c_secondary={},
				C_primary={},
				pipe_resistances=pipe_resistances,
				preferred_flow_sign=preferred_flow_sign,
				energy_head=reservoir_head,
				reservoir_node=reservoir_node,
				reservoir_outflow=reservoir_outflow,
				c_bounds=c_bounds if PIPE_BOUNDS else None,
				C_bounds=C_bounds if PIPE_BOUNDS else None,
				initial_guess=initial_guess,
			)

			max_min_violation = max(v["min_violation"] for v in bounds.diagnostics.values())
			max_max_violation = max(v["max_violation"] for v in bounds.diagnostics.values())
			min_demand_viol = min(v["min_demand_viol"] for v in bounds.diagnostics.values())
			max_demand_viol = min(v["max_demand_viol"] for v in bounds.diagnostics.values())
			print(f"Solver status: {bounds.status}")
			print(f"Max constraint violation (min): {max_min_violation:.3e}")
			print(f"Max constraint violation (max): {max_max_violation:.3e}")
			print(f"Min demand constraint (min): {min_demand_viol:.3e}")
			print(f"Min demand constraint (max): {max_demand_viol:.3e}")

			base_demands = {node_id: node.base_demand for node_id, node in network.junctions.items()}

			plot_demand_bounds(
				inp_path=inp_path,
				measurement_nodes=measurement_nodes,
				actual_demands=scenario_demands,
				bounds=bounds,
				base_flows=base_flows,
				c_bounds=c_bounds,
				C_bounds=C_bounds,
				sensor_heads=sensor_heads,
				scenario_flows=scenario_flows,
				base_demands=base_demands,
				output_dir=output_dir,
				show_plot=show_intermediate_plots,
			)
			res_demand = -sum(scenario_demands.values())
			_write_json(
				os.path.join(output_dir, "demand_bounds.json"),
				{
					"demands_actual": {**scenario_demands, str(reservoir_node): res_demand} if reservoir_node else scenario_demands,
					"demands_min": bounds.d_min,
					"demands_max": bounds.d_max,
					"heads_fixed": sensor_heads,
					"flows_scenario": scenario_flows,
				},
			)

		if SINGLE_NODE_SOLUTION:
			single = solve_single_node_min(
				network=network,
				target_node="5",
				sensor_heads=sensor_heads,
				pipe_primary=set(),
				pipe_secondary=set(),
				c_secondary={},
				C_primary={},
				pipe_resistances=pipe_resistances,
				preferred_flow_sign=preferred_flow_sign,
				energy_head=reservoir_head,
				reservoir_node=reservoir_node,
				reservoir_outflow=reservoir_outflow,
				c_bounds=c_bounds if PIPE_BOUNDS else None,
				C_bounds=C_bounds if PIPE_BOUNDS else None,
				initial_guess=initial_guess,
			)

			print("Single-node min result:")
			print(f"node={single.node_id}, success={single.success}")
			print(f"max_violation={single.max_violation:.3e}, min_demand_viol={single.min_demand_viol:.3e}")
			print("Demands:")
			for k in sorted(single.demands.keys()):
				print(f"  {k}: {single.demands[k]:.6f}")
			print("Heads:")
			for k in sorted(single.heads.keys()):
				print(f"  {k}: {single.heads[k]:.6f}")
			print("Flows:")
			for k in sorted(single.flows.keys()):
				print(f"  {k}: {single.flows[k]:.6f}")

			plot_single_solution(
				inp_path=inp_path,
				measurement_nodes=measurement_nodes,
				solution=single,
				c_bounds=c_bounds,
				C_bounds=C_bounds,
				output_dir=output_dir,
				show_plot=show_intermediate_plots,
			)
			res_demand = -sum(single.demands.values())
			_write_json(
				os.path.join(output_dir, "single_node_solution.json"),
				{
					"demands": {**single.demands, str(reservoir_node): res_demand} if reservoir_node else single.demands,
					"heads": single.heads,
					"flows": single.flows,
					"max_violation": single.max_violation,
					"min_demand_viol": single.min_demand_viol,
				},
			)

		if RADIUS:
			import numpy as np

			solver_mode = SOLVER
			try:
				import hexaly.optimizer  # noqa: F401
			except Exception as exc:
				raise RuntimeError(f"Hexaly solver unavailable ({exc}).") from exc

			multi_starts = MULTI_STARTS
			multi_noise_abs = MULTI_START_NOISE
			multi_noise_rel = MULTI_START_NOISE_REL

			rng = np.random.default_rng(MULTI_START_SEED)
			runs = []
			best = None
			unique_radii = set()

			for run_idx in range(max(1, multi_starts)):
				q0 = {k: float(v) for k, v in initial_guess.flows.items()}
				h0 = {k: float(v) for k, v in initial_guess.heads.items()}
				if multi_starts > 1:
					for k in q0:
						scale = max(1.0, abs(q0[k]))
						q0[k] = q0[k] + (multi_noise_abs + multi_noise_rel * scale) * rng.standard_normal()
					for k in h0:
						scale = max(1.0, abs(h0[k]))
						h0[k] = h0[k] + (multi_noise_abs + multi_noise_rel * scale) * rng.standard_normal()

				run_guess = SolverResult(status="ms", demands=initial_guess.demands, heads=h0, flows=q0)
				hexaly_seed = None
				if HEXALY_SEED is not None:
					hexaly_seed = int(HEXALY_SEED) + int(run_idx)
				else:
					hexaly_seed = int(rng.integers(1, 2**31 - 1))
				if solver_mode == "Hexaly_xd":
					fixed_b = reference_demands_from_flows if FIXED_REFERENCE else None
					distance_res = solve_max_demand_distance_xd_hexaly(
						network=network,
						sensor_heads=sensor_heads,
						pipe_resistances=pipe_resistances,
						c_bounds=c_bounds if PIPE_BOUNDS else None,
						C_bounds=C_bounds if PIPE_BOUNDS else None,
						reservoir_node=reservoir_node,
						reservoir_head=reservoir_head,
						reservoir_outflow=reservoir_outflow,
						initial_guess=run_guess,
						norm_p=NORM,
						demand_lb=DEMAND_LB,
						total_demand=total_demand_solver if FIX_TOTAL_DEMAND else None,
						measurement_nodes=measurement_nodes,
						measurement_heads_equal_only=MEASUREMENT_HEADS_EQUAL_ONLY,
						headloss_n=headloss_n,
						cycle_basis_mode=XD_CYCLE_BASIS_MODE,
						fixed_demands_b=fixed_b,
						reference_demands=reference_demands_restriction,
						restriction_mode=restriction_mode,
						radius_to_fixed=RADIUS_TO_FIXED,
						deviation_alpha=DEVIATION_ALPHA,
						license_path=HEXALY_LICENSE_PATH,
						time_limit=HEXALY_TIME_LIMIT,
						seed=hexaly_seed,
						verbosity=HEXALY_VERBOSITY,
					)
				elif solver_mode == "Hexaly":
					distance_res = solve_max_demand_distance_hexaly(
						network=network,
						sensor_heads=sensor_heads,
						pipe_resistances=pipe_resistances,
						c_bounds=c_bounds if PIPE_BOUNDS else None,
						C_bounds=C_bounds if PIPE_BOUNDS else None,
						reservoir_node=reservoir_node,
						reservoir_outflow=reservoir_outflow,
						initial_guess=run_guess,
						norm_p=NORM,
						demand_lb=DEMAND_LB,
						total_demand=total_demand_solver if FIX_TOTAL_DEMAND else None,
						measurement_nodes=measurement_nodes,
						measurement_heads_equal_only=MEASUREMENT_HEADS_EQUAL_ONLY,
						reference_demands=reference_demands_restriction,
						restriction_mode=restriction_mode,
						radius_to_fixed=RADIUS_TO_FIXED,
						deviation_alpha=DEVIATION_ALPHA,
						license_path=HEXALY_LICENSE_PATH,
						time_limit=HEXALY_TIME_LIMIT,
						seed=hexaly_seed,
						verbosity=HEXALY_VERBOSITY,
						stagnation_seconds=HEXALY_STAGNATION_SECONDS,
						stagnation_eps=HEXALY_STAGNATION_EPS,
						head_margin=HEXALY_HEAD_MARGIN,
						use_path_head_bounds=HEXALY_USE_PATH_HEAD_BOUNDS,
					)
				else:
					raise ValueError(f"Unsupported SOLVER: {solver_mode}")
				diffs = []
				for node_id in network.junctions.keys():
					da = distance_res.demands_a.get(node_id, 0.0)
					db = distance_res.demands_b.get(node_id, 0.0)
					diffs.append(da - db)
				radius = _compute_norm(diffs, NORM)

				runs.append(
					{
						"run": run_idx,
						"radius": radius,
						"success": distance_res.success,
						"max_violation": distance_res.max_violation,
						"min_demand_viol": distance_res.min_demand_viol,
						"start_flows": q0,
						"start_heads": h0,
					}
				)
				unique_radii.add(round(radius, 8))

				if best is None or radius > best[0]:
					best = (radius, distance_res)

			best_radius, best_res = best
			print("Demand distance result:")
			print(f"best radius: {best_radius:.8f}")
			print(f"unique local optima (by radius): {len(unique_radii)}")


			plot_demand_distance(
				inp_path=inp_path,
				measurement_nodes=measurement_nodes,
				demands_a=best_res.demands_a,
				demands_b=best_res.demands_b,
				heads_a=best_res.heads_a,
				heads_b=best_res.heads_b,
				flows_a=best_res.flows_a,
				flows_b=best_res.flows_b,
				radius=best_radius,
				norm_p=NORM,
				measurement_count=len(measurement_nodes),
				output_dir=output_dir,
				c_bounds=c_bounds,
				C_bounds=C_bounds,
				head_bounds=head_bounds_for_plot,
				show_head_bounds_measurements=MEASUREMENT_HEADS_EQUAL_ONLY,
				show_plot=show_intermediate_plots,
			)
			res_demand_a = -sum(best_res.demands_a.values())
			res_demand_b = -sum(best_res.demands_b.values())
			_write_json(
				os.path.join(output_dir, "demand_distance.json"),
				{
					"demands_a": {**best_res.demands_a, str(reservoir_node): res_demand_a} if reservoir_node else best_res.demands_a,
					"demands_b": {**best_res.demands_b, str(reservoir_node): res_demand_b} if reservoir_node else best_res.demands_b,
					"heads_a": best_res.heads_a,
					"heads_b": best_res.heads_b,
					"flows_a": best_res.flows_a,
					"flows_b": best_res.flows_b,
					"radius": best_radius,
					"norm": NORM,
					"p": len(measurement_nodes),
					"max_violation": best_res.max_violation,
					"min_demand_viol": best_res.min_demand_viol,
					"objective": best_res.objective,
					"solver_status": best_res.solver_status,
					"best_bound": best_res.best_bound,
				},
			)
			_write_json(
				os.path.join(output_dir, "demand_distance_multistart.json"),
				{
					"runs": runs,
					"unique_local_optima": len(unique_radii),
					"best_radius": best_radius,
					"norm": NORM,
				},
			)
			if DEBUG_DUMP_ALWAYS:
				try:
					report = debug_snapshot._build_report(
						output_dir=output_dir,
						eps=DEBUG_DUMP_EPS,
						use_solution_resistance=False,
					)
					debug_snapshot._write_json(
						os.path.join(output_dir, "debug_dump.json"),
						report,
					)
				except Exception as exc:
					print(f"WARNING: debug dump failed ({exc})")

		summary_rows.append(
			{
				"measurement_sites": measurement_nodes,
				"measurement_count": len(measurement_nodes),
				"radius": best_radius if RADIUS else None,
				"success": best_res.success if RADIUS else None,
				"max_violation": best_res.max_violation if RADIUS else None,
				"min_demand_viol": best_res.min_demand_viol if RADIUS else None,
				"objective": best_res.objective if RADIUS else None,
				"solver_status": best_res.solver_status if RADIUS else None,
				"best_bound": best_res.best_bound if RADIUS else None,
				"output_dir": output_dir,
			}
		)

	write_summary = True
	if isinstance(MEASUREMENT_SITES, int):
		write_summary = True
	elif isinstance(MEASUREMENT_SITES, (list, tuple)):
		if len(MEASUREMENT_SITES) > 0 and all(isinstance(item, (list, tuple)) for item in MEASUREMENT_SITES):
			write_summary = True
		else:
			write_summary = False

	if summary_rows and write_summary:
		sizes = {row.get("measurement_count") for row in summary_rows}
		multi_sizes = len({s for s in sizes if s is not None}) > 1
		if not multi_sizes:
			for row in summary_rows:
				row.pop("measurement_count", None)

		def _summary_sort_key(row: Dict[str, object]) -> tuple[int, bool, float]:
			count = row.get("measurement_count")
			count_val = int(count) if isinstance(count, int) else 0
			val = row.get("radius")
			if val is None:
				return (count_val, True, float("inf"))
			try:
				return (count_val, False, float(val))
			except (TypeError, ValueError):
				return (count_val, True, float("inf"))

		summary_rows = sorted(summary_rows, key=_summary_sort_key)
		summary_root = batch_dir if batch_dir else "data"
		summary_path = os.path.join(summary_root, f"{wdn_name}-measurement-summary-{master_timestamp}.json")
		_write_json(summary_path, {"results": summary_rows})
		try:
			import csv
			csv_path = os.path.join(summary_root, f"{wdn_name}-measurement-summary-{master_timestamp}.csv")
			with open(csv_path, "w", encoding="utf-8", newline="") as f:
				writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
				writer.writeheader()
				writer.writerows(summary_rows)
			print(f"Wrote measurement summary to: {summary_path}")
			print(f"Wrote measurement summary to: {csv_path}")
		except Exception as exc:
			print(f"WARNING: failed to write CSV summary ({exc})")


if __name__ == "__main__":
	main()
