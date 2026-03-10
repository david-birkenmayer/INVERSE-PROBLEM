from typing import Dict, Iterable, Tuple
import json
import os
from datetime import datetime

# Easy-to-change configuration
WDN_NAME = "Alperovits"; MEASUREMENT_SITES = [1, 4, 3, 2]
# WDN_NAME = "Kadu"; MEASUREMENT_SITES = ["10", "14", "24"]
# WDN_NAME = "Anytown"; MEASUREMENT_SITES = ["20", "140"],

PIPE_BOUNDS = True
PIPE_BOUND_SAFENESS = 0.9  # in [0, 1], higher means looser bounds

MULTI_STARTS = 10
MULTI_START_NOISE = 0.05

# SOLVER = "ipopt"
SOLVER = "global"
# SOLVER = "scipy"
GLOBAL_SOLVER = "couenne"  # e.g., "couenne", "scip", "baron" (if installed)

DEMAND_INTERVALS = False
SINGLE_NODE_SOLUTION = False
RADIUS = True
NORM = 2  # positive int/float, or float("inf") for infinity norm
DEMAND_LB = 1e-6  # enforce strictly positive junction demands
USE_RESERVOIR_OUTFLOW_CONSTRAINT = True  # fix reservoir demand via net outflow
FIX_TOTAL_DEMAND = True  # anchor sum of demands to scenario total


from step1_io import load_inp_network, compute_pipe_resistances
from step2_estimation import (
	simulate_random_demand_scenarios,
	simulate_base_flows,
	simulate_single_random_scenario,
)
from step3_solver import (
	DemandBounds,
	SolverResult,
	SingleNodeResult,
	FeasibilityResult,
	solve_demand_bounds,
	solve_feasibility,
	solve_single_node_min,
	solve_max_demand_distance,
	solve_max_demand_distance_ipopt,
)


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


def _write_json(path: str, payload: Dict[str, object]) -> None:
	with open(path, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2, sort_keys=True)


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
	plt.show(block=True)
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
	plt.show(block=True)
	plt.close(fig)


def plot_single_solution(
	inp_path: str,
	measurement_nodes: Iterable[str],
	solution: SingleNodeResult,
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
		if qa == 0.0 or qb == 0.0:
			edge_colors.append("#2b6cb0")
		elif (qa > 0) == (qb > 0):
			edge_colors.append("#000000")
		else:
			edge_colors.append("#2b6cb0")
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
	resistances = compute_pipe_resistances(network)
	print(f"Loaded {len(network.nodes)} nodes and {len(network.pipes)} pipes.")
	first_pipe = next(iter(resistances.values()), None)
	if first_pipe is not None:
		print("Example resistance:", first_pipe)

	base_flows = simulate_base_flows(inp_path=inp_path)
	flows_by_pipe = simulate_random_demand_scenarios(
		inp_path=inp_path,
		n_scenarios=200,
		demand_log_sigma=0.3,
		seed=1,
		simulator="auto",
	)

	if PIPE_BOUNDS:
		c_bounds, C_bounds = _compute_pipe_bounds_all(flows_by_pipe, PIPE_BOUND_SAFENESS)
	else:
		c_bounds = {}
		C_bounds = {}

	scenario_demands_pos, scenario_heads, scenario_flows = simulate_single_random_scenario(
		inp_path=inp_path,
		demand_log_sigma=0.3,
		seed=2,
		simulator="auto",
	)
	scenario_demands = {k: v for k, v in scenario_demands_pos.items()}
	total_demand = float(sum(scenario_demands.values()))

	measurement_nodes = [str(x) for x in MEASUREMENT_SITES]
	measurement_tag = "_".join(measurement_nodes)
	timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
	pipe_tag = "B" if PIPE_BOUNDS else "NB"
	output_dir = os.path.join("data", f"{wdn_name}-{pipe_tag}-{measurement_tag}-{timestamp}")
	os.makedirs(output_dir, exist_ok=True)
	print(f"Saving results to: {output_dir}")
	_write_json(
		os.path.join(output_dir, "metadata.json"),
		{
			"WDN_NAME": WDN_NAME,
			"MEASUREMENT_SITES": measurement_nodes,
			"PIPE_BOUNDS": PIPE_BOUNDS,
			"PIPE_BOUND_SAFENESS": PIPE_BOUND_SAFENESS,
			"SOLVER": SOLVER,
			"GLOBAL_SOLVER": GLOBAL_SOLVER,
			"DEMAND_INTERVALS": DEMAND_INTERVALS,
			"SINGLE_NODE_SOLUTION": SINGLE_NODE_SOLUTION,
			"RADIUS": RADIUS,
			"NORM": NORM,
			"DEMAND_LB": DEMAND_LB,
			"USE_RESERVOIR_OUTFLOW_CONSTRAINT": USE_RESERVOIR_OUTFLOW_CONSTRAINT,
			"FIX_TOTAL_DEMAND": FIX_TOTAL_DEMAND,
			"MULTI_STARTS": MULTI_STARTS,
			"MULTI_START_NOISE": MULTI_START_NOISE,
			"timestamp": timestamp,
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

	pipe_resistances = {pid: vals["r_e"] for pid, vals in resistances.items()}
	for pid, pipe in network.pipes.items():
		q = scenario_flows.get(pid, 0.0)
		if abs(q) < 1e-8:
			continue
		hu = scenario_heads.get(pipe.start_node, None)
		hv = scenario_heads.get(pipe.end_node, None)
		if hu is None or hv is None:
			continue
		pipe_resistances[pid] = (hu - hv) / (abs(q) * q)

	reservoir_node = next(iter(network.reservoirs.keys()), None)
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

	reservoir_head = scenario_heads.get(reservoir_node, None) if reservoir_node is not None else None
	feasible = solve_feasibility(
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
	print("Feasibility result:")
	print(f"success={feasible.success}, max_violation={feasible.max_violation:.3e}, min_demand_viol={feasible.min_demand_viol:.3e}", flush=True)
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
		if SOLVER in {"ipopt", "global"}:
			try:
				import pyomo.environ as pyo
				if SOLVER == "ipopt":
					if not pyo.SolverFactory("ipopt").available(False):
						print("WARNING: IPOPT executable not found; falling back to SciPy.")
						solver_mode = "scipy"
				else:
					if not pyo.SolverFactory(GLOBAL_SOLVER).available(False):
						print(f"WARNING: global solver '{GLOBAL_SOLVER}' not found; falling back to SciPy.")
						solver_mode = "scipy"
			except Exception as exc:
				print(f"WARNING: solver unavailable ({exc}); falling back to SciPy.")
				solver_mode = "scipy"

		rng = np.random.default_rng(0)
		runs = []
		best = None
		unique_radii = set()

		for run_idx in range(max(1, MULTI_STARTS)):
			q0 = {k: float(v) for k, v in initial_guess.flows.items()}
			h0 = {k: float(v) for k, v in initial_guess.heads.items()}
			if MULTI_STARTS > 1:
				for k in q0:
					q0[k] = q0[k] + MULTI_START_NOISE * rng.standard_normal()
				for k in h0:
					h0[k] = h0[k] + MULTI_START_NOISE * rng.standard_normal()

			run_guess = SolverResult(status="ms", demands=initial_guess.demands, heads=h0, flows=q0)
			if solver_mode in {"ipopt", "global"}:
				distance_res = solve_max_demand_distance_ipopt(
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
					total_demand=total_demand if FIX_TOTAL_DEMAND else None,
					solver_name=GLOBAL_SOLVER if solver_mode == "global" else "ipopt",
				)
			else:
				distance_res = solve_max_demand_distance(
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
					total_demand=total_demand if FIX_TOTAL_DEMAND else None,
				)
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


if __name__ == "__main__":
	main()
