from typing import Dict, Iterable, Tuple

from step1_io import load_alperovits_network, compute_pipe_resistances
from step2_estimation import (
	estimate_capacity_from_inp,
	simulate_base_flows,
	simulate_single_random_scenario,
)
from step3_solver import (
	DemandBounds,
	SolverResult,
	SingleNodeResult,
	solve_demand_bounds,
	solve_single_node_min,
)


def _classify_pipes_by_flow(
	base_flows: Dict[str, float],
	primary_fraction: float = 0.3,
) -> Tuple[Iterable[str], Iterable[str]]:
	if not 0.0 < primary_fraction < 1.0:
		raise ValueError("primary_fraction must be in (0, 1).")
	sorted_pipes = sorted(base_flows.items(), key=lambda kv: abs(kv[1]), reverse=True)
	cutoff = max(1, int(round(primary_fraction * len(sorted_pipes))))
	primary = {pipe_id for pipe_id, _ in sorted_pipes[:cutoff]}
	secondary = {pipe_id for pipe_id, _ in sorted_pipes[cutoff:]}
	return primary, secondary


def plot_network(
	inp_path: str,
	primary_pipes: Iterable[str],
	secondary_pipes: Iterable[str],
	base_flows: Dict[str, float],
	c_secondary: Dict[str, float],
	C_primary: Dict[str, float],
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
	nx.draw_networkx_nodes(G, pos, node_size=350, node_color="#f2f2f2", edgecolors="#333333", ax=ax)
	nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)

	primary_set = set(primary_pipes)
	secondary_set = set(secondary_pipes)

	primary_edges = []
	secondary_edges = []
	edge_to_pipe = {}
	for pipe_name, pipe in wn.pipes():
		edge = (pipe.start_node_name, pipe.end_node_name)
		edge_to_pipe[edge] = pipe_name
		if pipe_name in primary_set:
			primary_edges.append(edge)
		elif pipe_name in secondary_set:
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

	for (u, v), pipe_id in edge_to_pipe.items():
		x = (pos[u][0] + pos[v][0]) / 2.0
		y = (pos[u][1] + pos[v][1]) / 2.0
		q0 = base_flows.get(pipe_id, 0.0)
		if pipe_id in primary_set:
			Ce = C_primary.get(pipe_id, float("nan"))
			label = f"q0={q0:.2f}\nC={Ce:.2f}"
		else:
			ce = c_secondary.get(pipe_id, float("nan"))
			label = f"q0={q0:.2f}\nc={ce:.2f}"
		ax.text(x, y, label, fontsize=7, ha="center", va="center", color="#111111")

	ax.set_title("Pipe classes, base flows, and capacity bounds")
	ax.axis("off")
	plt.tight_layout()
	fig.savefig("pipe_classes.png", dpi=200)
	# plt.show(block=True)
	plt.close(fig)


def plot_demand_bounds(
	inp_path: str,
	measurement_nodes: Iterable[str],
	actual_demands: Dict[str, float],
	bounds: DemandBounds,
	base_flows: Dict[str, float],
	c_secondary: Dict[str, float],
	C_primary: Dict[str, float],
	sensor_heads: Dict[str, float],
	scenario_flows: Dict[str, float],
	base_demands: Dict[str, float],
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
	normal_nodes = [n for n in G.nodes if n not in measurement_set]
	meas_nodes = [n for n in G.nodes if n in measurement_set]

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
	nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)

	for node_id in wn.junction_name_list:
		x, y = pos[node_id]
		actual = actual_demands.get(node_id, 0.0)
		dmin = bounds.d_min.get(node_id, float("nan"))
		dmax = bounds.d_max.get(node_id, float("nan"))
		d0 = base_demands.get(node_id, 0.0)
		label = f"d*={actual:.2f}\n[{dmin:.2f}, {dmax:.2f}]\nd0={d0:.2f}"
		ax.text(x, y - 180, label, fontsize=7, ha="center", va="top", color="#111111")

	for node_id in wn.reservoir_name_list:
		if node_id not in pos:
			continue
		x, y = pos[node_id]
		head = sensor_heads.get(node_id, float("nan"))
		ax.text(x, y - 180, f"h={head:.2f}", fontsize=7, ha="center", va="top", color="#111111")

	for pipe_name, pipe in wn.pipes():
		u = pipe.start_node_name
		v = pipe.end_node_name
		x = (pos[u][0] + pos[v][0]) / 2.0
		y = (pos[u][1] + pos[v][1]) / 2.0
		q0 = base_flows.get(pipe_name, 0.0)
		qs = scenario_flows.get(pipe_name, 0.0)
		if pipe_name in C_primary:
			label = f"q0={q0:.2f}\nq*={qs:.2f}\nC={C_primary[pipe_name]:.2f}"
		else:
			label = f"q0={q0:.2f}\nq*={qs:.2f}\nc={c_secondary.get(pipe_name, float('nan')):.2f}"
		ax.text(x, y + 120, label, fontsize=7, ha="center", va="bottom", color="#111111")

	ax.set_title("Demand bounds vs actual demands")
	ax.axis("off")
	plt.tight_layout()
	fig.savefig("demand_bounds.png", dpi=200)
	# plt.show(block=True)
	plt.close(fig)


def plot_single_solution(
	inp_path: str,
	measurement_nodes: Iterable[str],
	solution: SingleNodeResult,
	c_secondary: Dict[str, float],
	C_primary: Dict[str, float],
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
	normal_nodes = [n for n in G.nodes if n not in measurement_set]
	meas_nodes = [n for n in G.nodes if n in measurement_set]

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
	nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)

	for node_id in wn.junction_name_list:
		x, y = pos[node_id]
		d = solution.demands.get(node_id, 0.0)
		h = solution.heads.get(node_id, float("nan"))
		label = f"d={d:.2f}\nh={h:.2f}"
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
		if pipe_name in C_primary:
			label = f"q={q_abs:.2f}\nC={C_primary[pipe_name]:.2f}"
		else:
			label = f"q={q_abs:.2f}\nc={c_secondary.get(pipe_name, float('nan')):.2f}"
		ax.text(x, y + 120, label, fontsize=7, ha="center", va="bottom", color="#111111")

	ax.set_title(f"Single-node min solution for node {solution.node_id}")
	ax.axis("off")
	plt.tight_layout()
	fig.savefig("single_node_solution.png", dpi=200)
	# plt.show(block=True)
	plt.close(fig)


def main() -> None:
	inp_path = "./wdn/Alperovits.inp"
	network = load_alperovits_network()
	resistances = compute_pipe_resistances(network)
	print(f"Loaded {len(network.nodes)} nodes and {len(network.pipes)} pipes.")
	first_pipe = next(iter(resistances.values()), None)
	if first_pipe is not None:
		print("Example resistance:", first_pipe)

	base_flows = simulate_base_flows(inp_path=inp_path)
	primary_pipes, secondary_pipes = _classify_pipes_by_flow(base_flows, primary_fraction=0.3)

	stats = estimate_capacity_from_inp(
		inp_path=inp_path,
		network=network,
		pipe_set_primary=primary_pipes,
		pipe_set_secondary=secondary_pipes,
		n_scenarios=200,
		demand_log_sigma=0.3,
		q_secondary_quantile=0.99,
		q_primary_quantile=0.10,
		seed=1,
		simulator="auto",
	)

	plot_network(
		inp_path=inp_path,
		primary_pipes=primary_pipes,
		secondary_pipes=secondary_pipes,
		base_flows=base_flows,
		c_secondary=stats.c_secondary,
		C_primary=stats.C_primary,
	)

	scenario_demands, scenario_heads, scenario_flows = simulate_single_random_scenario(
		inp_path=inp_path,
		demand_log_sigma=0.3,
		seed=2,
		simulator="auto",
	)

	measurement_nodes = ["1", "2"]
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
	bounds = solve_demand_bounds(
		network=network,
		sensor_heads=sensor_heads,
		pipe_primary=primary_pipes,
		pipe_secondary=secondary_pipes,
		c_secondary=stats.c_secondary,
		C_primary=stats.C_primary,
		pipe_resistances=pipe_resistances,
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
		c_secondary=stats.c_secondary,
		C_primary=stats.C_primary,
		sensor_heads=sensor_heads,
		scenario_flows=scenario_flows,
		base_demands=base_demands,
	)

	single = solve_single_node_min(
		network=network,
		target_node="5",
		sensor_heads=sensor_heads,
		pipe_primary=primary_pipes,
		pipe_secondary=secondary_pipes,
		c_secondary=stats.c_secondary,
		C_primary=stats.C_primary,
		pipe_resistances=pipe_resistances,
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
		c_secondary=stats.c_secondary,
		C_primary=stats.C_primary,
	)


if __name__ == "__main__":
	main()
