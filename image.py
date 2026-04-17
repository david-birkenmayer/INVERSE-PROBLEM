from types import SimpleNamespace
from typing import Dict, Iterable, Tuple
import json
import os

from step1_io import load_inp_network


DATA_DIR = "data/example"


def _read_json(path: str) -> Dict[str, object]:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def _load_optional_json(path: str) -> Dict[str, object] | None:
	if not os.path.exists(path):
		return None
	return _read_json(path)


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
		q0 = float(base_flows.get(pipe_name, 0.0))
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
	bounds: Dict[str, Dict[str, float]],
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

	d_min = bounds.get("demands_min", {})
	d_max = bounds.get("demands_max", {})

	for node_id in wn.junction_name_list:
		x, y = pos[node_id]
		actual = float(actual_demands.get(node_id, 0.0))
		dmin = float(d_min.get(node_id, float("nan")))
		dmax = float(d_max.get(node_id, float("nan")))
		d0 = float(base_demands.get(node_id, 0.0))
		label = f"d*={actual:.4f}\n[{dmin:.4f}, {dmax:.4f}]\nd0={d0:.4f}"
		ax.text(x, y - 180, label, fontsize=7, ha="center", va="top", color="#111111")

	for node_id in wn.reservoir_name_list:
		if node_id not in pos:
			continue
		x, y = pos[node_id]
		head = float(sensor_heads.get(node_id, float("nan")))
		ax.text(x, y - 180, f"h={head:.4f}", fontsize=7, ha="center", va="top", color="#111111")

	for pipe_name, pipe in wn.pipes():
		u = pipe.start_node_name
		v = pipe.end_node_name
		x = (pos[u][0] + pos[v][0]) / 2.0
		y = (pos[u][1] + pos[v][1]) / 2.0
		q0 = float(base_flows.get(pipe_name, 0.0))
		qs = float(scenario_flows.get(pipe_name, 0.0))
		cl = c_bounds.get(pipe_name, float("-inf"))
		cu = C_bounds.get(pipe_name, float("inf"))
		label = f"q0={q0:.4f}\nq*={qs:.4f}\n[{cl:.4f}, {cu:.4f}]"
		ax.text(x, y + 120, label, fontsize=7, ha="center", va="bottom", color="#111111")

	ax.set_title("Demand bounds vs actual demands")
	ax.axis("off")
	plt.tight_layout()
	fig.savefig(os.path.join(output_dir, "demand_bounds.png"), dpi=200)
	plt.close(fig)


def plot_single_solution(
	inp_path: str,
	measurement_nodes: Iterable[str],
	solution: SimpleNamespace,
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
		q = float(solution.flows.get(pipe_name, 0.0))
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
		d = float(solution.demands.get(node_id, 0.0))
		h = float(solution.heads.get(node_id, float("nan")))
		label = f"d={d:.4f}\nh={h:.4f}"
		ax.text(x, y - 180, label, fontsize=7, ha="center", va="top", color="#111111")

	for pipe_name, pipe in wn.pipes():
		q = float(solution.flows.get(pipe_name, 0.0))
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
		qa = float(flows_a.get(pipe_name, 0.0))
		qb = float(flows_b.get(pipe_name, 0.0))
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

	for node_id in wn.junction_name_list:
		x, y = pos[node_id]
		da = float(demands_a.get(node_id, 0.0))
		db = float(demands_b.get(node_id, 0.0))
		ha = float(heads_a.get(node_id, float("nan")))
		hb = float(heads_b.get(node_id, float("nan")))
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
		ha = float(heads_a.get(node_id, float("nan")))
		hb = float(heads_b.get(node_id, float("nan")))
		da = -sum(float(v) for v in demands_a.values())
		db = -sum(float(v) for v in demands_b.values())
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
		qa = float(flows_a.get(pipe_name, 0.0))
		qb = float(flows_b.get(pipe_name, 0.0))
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
	plt.close(fig)


def main() -> None:
	if not os.path.isdir(DATA_DIR):
		raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR}")

	params = _read_json(os.path.join(DATA_DIR, "parameters.json"))
	scenario_file = str(params.get("SCENARIO_FILE", ""))
	if not scenario_file:
		raise ValueError("parameters.json is missing SCENARIO_FILE.")
	if not os.path.exists(scenario_file):
		raise FileNotFoundError(f"Scenario file not found: {scenario_file}")

	scenario = _read_json(scenario_file)
	wdn_name = str(params.get("WDN_NAME") or scenario.get("wdn_name") or "")
	if not wdn_name:
		raise ValueError("WDN_NAME is missing from parameters.json and scenario file.")
	inp_path = f"./wdn/{wdn_name}.inp"

	measurement_nodes = params.get("MEASUREMENT_NODES") or []
	pipe_bounds_enabled = bool(params.get("PIPE_BOUNDS", False))

	c_bounds = {}
	C_bounds = {}
	bounds_payload = _load_optional_json(os.path.join(DATA_DIR, "c_bounds.json"))
	if bounds_payload:
		c_bounds = {k: float(v) for k, v in bounds_payload.get("c_bounds", {}).items()}
		C_bounds = {k: float(v) for k, v in bounds_payload.get("C_bounds", {}).items()}
	else:
		c_bounds = {k: float(v) for k, v in scenario.get("c_bounds", {}).items()}
		C_bounds = {k: float(v) for k, v in scenario.get("C_bounds", {}).items()}

	base_flows = {k: float(v) for k, v in scenario.get("base_flows", {}).items()}
	scenario_flows = {k: float(v) for k, v in scenario.get("scenario_flows", {}).items()}
	scenario_demands = {k: float(v) for k, v in scenario.get("scenario_demands", {}).items()}
	scenario_heads = {k: float(v) for k, v in scenario.get("scenario_heads", {}).items()}

	if pipe_bounds_enabled:
		plot_network(
			inp_path=inp_path,
			base_flows=base_flows,
			c_bounds=c_bounds,
			C_bounds=C_bounds,
			output_dir=DATA_DIR,
		)

	demand_distance = _load_optional_json(os.path.join(DATA_DIR, "demand_distance.json"))
	if demand_distance:
		plot_demand_distance(
			inp_path=inp_path,
			measurement_nodes=measurement_nodes,
			demands_a=demand_distance.get("demands_a", {}),
			demands_b=demand_distance.get("demands_b", {}),
			heads_a=demand_distance.get("heads_a", {}),
			heads_b=demand_distance.get("heads_b", {}),
			flows_a=demand_distance.get("flows_a", {}),
			flows_b=demand_distance.get("flows_b", {}),
			radius=float(demand_distance.get("radius", 0.0)),
			norm_p=float(demand_distance.get("norm", 2.0)),
			measurement_count=int(demand_distance.get("p", len(measurement_nodes))),
			output_dir=DATA_DIR,
			c_bounds=c_bounds,
			C_bounds=C_bounds,
		)

	demand_bounds = _load_optional_json(os.path.join(DATA_DIR, "demand_bounds.json"))
	if demand_bounds:
		network = load_inp_network(inp_path)
		base_demands = {node_id: node.base_demand for node_id, node in network.junctions.items()}
		sensor_heads = {k: float(v) for k, v in demand_bounds.get("heads_fixed", {}).items()}
		plot_demand_bounds(
			inp_path=inp_path,
			measurement_nodes=measurement_nodes,
			actual_demands=scenario_demands,
			bounds=demand_bounds,
			base_flows=base_flows,
			c_bounds=c_bounds,
			C_bounds=C_bounds,
			sensor_heads=sensor_heads,
			scenario_flows=scenario_flows,
			base_demands=base_demands,
			output_dir=DATA_DIR,
		)

	single_solution = _load_optional_json(os.path.join(DATA_DIR, "single_node_solution.json"))
	if single_solution:
		solution = SimpleNamespace(
			node_id=single_solution.get("node_id", "unknown"),
			demands=single_solution.get("demands", {}),
			heads=single_solution.get("heads", {}),
			flows=single_solution.get("flows", {}),
		)
		plot_single_solution(
			inp_path=inp_path,
			measurement_nodes=measurement_nodes,
			solution=solution,
			c_bounds=c_bounds,
			C_bounds=C_bounds,
			output_dir=DATA_DIR,
		)

	print(f"Plots generated in: {DATA_DIR}")


if __name__ == "__main__":
	main()
