from typing import Dict, Iterable
import json
import os
from datetime import datetime
import argparse

### NETWORK AND MEASUREMENT SITES
WDN = "Alperovits"; MEASUREMENT_SITES = 2
# WDN = "Kadu"; MEASUREMENT_SITES = ["10", "14", "24"]
# WDN = "Anytown"; MEASUREMENT_SITES = 1 # ([], [50, 100]) # 2 # ([], [20, 140], [20,50,60,100,120,140])


SCENARIO = "Alperovits-base"
PIPE_BOUNDS = False
OUTPUT_DIR = None
HEADLOSS_MODEL = "hw"  # "auto", "dw", or "hw"

MULTI_STARTS = 1
MULTI_START_NOISE = 0.05
MULTI_START_NOISE_REL = 0.25
MULTI_START_SEED = None  # set int for reproducible multistarts

# SOLVER = "Hexaly"
SOLVER = "Hexaly_xd"

NORM = 2  # positive int/float, or float("inf") for infinity norm
DEMAND_LB = 1e-6  # enforce strictly positive junction demands
USE_RESERVOIR_OUTFLOW_CONSTRAINT = False  # fix reservoir demand via net outflow
FIX_TOTAL_DEMAND = True  # anchor sum of demands to scenario total
MEASUREMENT_HEADS_EQUAL_ONLY = True  # only enforce equal heads at measurements
MEASUREMENT_ALLOWLIST = None  # list of node ids or None for all junctions
MEASUREMENT_MAX_SUBSETS = 200
MEASUREMENT_SUBSET_SEED = 1
MEASUREMENT_SUBSET_MODE = "all"  # "exact" or "all" (sizes 1..p)

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
from step3_solver import (
	SolverResult,
	solve_feasibility,
)
from step3_solver_hexaly import solve_max_demand_distance_hexaly
from step3_solver_xd_hexaly import solve_max_demand_distance_xd_hexaly
from step3_solver_xd_hexaly import check_xd_feasibility_from_flows, check_xd_cycle_feasibility_from_bounds
import debug_snapshot




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


def _select_measurement_sets(network) -> list[list[str]]:
	import math
	import random
	import re
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

	if isinstance(MEASUREMENT_SITES, str):
		raw = MEASUREMENT_SITES.strip()
		if raw == "":
			return [[]]
		if MEASUREMENT_ALLOWLIST is None:
			candidates = sorted(network.junctions.keys())
		else:
			candidates = [str(x) for x in MEASUREMENT_ALLOWLIST]
			candidates = [c for c in candidates if c in network.junctions]
		limit = int(MEASUREMENT_MAX_SUBSETS)

		m_exact = re.fullmatch(r"#(\d+)", raw)
		if m_exact:
			k = int(m_exact.group(1))
			if k > len(candidates):
				return []
			if k == 0:
				return [[]]
			return _sample_combinations(candidates, [k], limit, MEASUREMENT_SUBSET_SEED)

		m_range = re.fullmatch(r"#(\d+)-#(\d+)", raw)
		if m_range:
			k_min = int(m_range.group(1))
			k_max = int(m_range.group(2))
			if k_min > k_max:
				return []
			if k_min > len(candidates):
				return []
			k_max = min(k_max, len(candidates))
			sizes = list(range(k_min, k_max + 1))
			return _sample_combinations(candidates, sizes, limit, MEASUREMENT_SUBSET_SEED)

		return []

	if isinstance(MEASUREMENT_SITES, (list, tuple)):
		if len(MEASUREMENT_SITES) == 0:
			return [[]]
		if all(isinstance(item, (list, tuple)) for item in MEASUREMENT_SITES):
			return [[str(x) for x in item] for item in MEASUREMENT_SITES]
		candidates = [str(x) for x in MEASUREMENT_SITES]
		return [candidates]

	return []


def _write_json(path: str, payload: Dict[str, object]) -> None:
	with open(path, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2, sort_keys=True)


def _read_json(path: str) -> Dict[str, object]:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def _apply_config(config: Dict[str, object]) -> None:
	global WDN, SCENARIO
	if "WDN" in config:
		WDN = str(config["WDN"])
	elif "WDN_NAME" in config:
		WDN = str(config["WDN_NAME"])

	if "SCENARIO" in config:
		SCENARIO = str(config["SCENARIO"])
	elif "SCENARIO_NAME" in config:
		SCENARIO = str(config["SCENARIO_NAME"])

	keys = {
		"MEASUREMENT_SITES",
		"MEASUREMENT_ALLOWLIST",
		"MEASUREMENT_MAX_SUBSETS",
		"MEASUREMENT_SUBSET_SEED",
		"MEASUREMENT_SUBSET_MODE",
		"HEADLOSS_MODEL",
		"SOLVER",
		"NORM",
		"DEMAND_LB",
		"USE_RESERVOIR_OUTFLOW_CONSTRAINT",
		"FIX_TOTAL_DEMAND",
		"MEASUREMENT_HEADS_EQUAL_ONLY",
		"MULTI_STARTS",
		"MULTI_START_NOISE",
		"MULTI_START_NOISE_REL",
		"MULTI_START_SEED",
		"DEBUG_DUMP_ALWAYS",
		"DEBUG_DUMP_EPS",
		"CHECK_XD_BASE_FEASIBILITY",
		"CHECK_XD_CYCLE_BOUNDS",
		"XD_CYCLE_BASIS_MODE",
		"SKIP_FEASIBILITY_SOLVE",
		"FIXED_REFERENCE",
		"DEMAND_RESTRICTION_MODE",
		"RADIUS_TO_FIXED",
		"DEVIATION_ALPHA",
		"HEXALY_LICENSE_PATH",
		"HEXALY_TIME_LIMIT",
		"HEXALY_STAGNATION_SECONDS",
		"HEXALY_STAGNATION_EPS",
		"HEXALY_SEED",
		"HEXALY_VERBOSITY",
		"HEXALY_HEAD_MARGIN",
		"HEXALY_USE_PATH_HEAD_BOUNDS",
		"OUTPUT_DIR",
	}
	for key in keys:
		if key in config:
			globals()[key] = config[key]


def _resolve_scenario_path(config: Dict[str, object] | None = None) -> str:
	if config and "SCENARIO_FILE" in config:
		return str(config["SCENARIO_FILE"])
	return os.path.join("scenario", WDN, f"{SCENARIO}.json")


def _build_parameters_snapshot(
	headloss_model_local: str | None = None,
	measurement_nodes: Iterable[str] | None = None,
	measurement_sets_count: int | None = None,
	scenario_path: str | None = None,
) -> Dict[str, object]:
	resolved_scenario_path = scenario_path if scenario_path is not None else _resolve_scenario_path()
	return {
		"WDN": WDN,
		"WDN_NAME": WDN,
		"SCENARIO": SCENARIO,
		"SCENARIO_FILE": resolved_scenario_path,
		"OUTPUT_DIR": OUTPUT_DIR,
		"MEASUREMENT_SITES": MEASUREMENT_SITES,
		"MEASUREMENT_SUBSET_MODE": MEASUREMENT_SUBSET_MODE,
		"MEASUREMENT_ALLOWLIST": MEASUREMENT_ALLOWLIST,
		"MEASUREMENT_MAX_SUBSETS": MEASUREMENT_MAX_SUBSETS,
		"MEASUREMENT_SUBSET_SEED": MEASUREMENT_SUBSET_SEED,
		"MEASUREMENT_SET_COUNT": measurement_sets_count,
		"MEASUREMENT_NODES": list(measurement_nodes) if measurement_nodes is not None else None,
		"PIPE_BOUNDS": PIPE_BOUNDS,
		"HEADLOSS_MODEL": HEADLOSS_MODEL,
		"HEADLOSS_MODEL_LOCAL": headloss_model_local,
		"SOLVER": SOLVER,
		"GLOBAL_SOLVER": None,
		"NORM": NORM,
		"DEMAND_LB": DEMAND_LB,
		"USE_RESERVOIR_OUTFLOW_CONSTRAINT": USE_RESERVOIR_OUTFLOW_CONSTRAINT,
		"FIX_TOTAL_DEMAND": FIX_TOTAL_DEMAND,
		"MEASUREMENT_HEADS_EQUAL_ONLY": MEASUREMENT_HEADS_EQUAL_ONLY,
		"MULTI_STARTS": MULTI_STARTS,
		"MULTI_START_NOISE": MULTI_START_NOISE,
		"MULTI_START_NOISE_REL": MULTI_START_NOISE_REL,
		"MULTI_START_SEED": MULTI_START_SEED,
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
	global PIPE_BOUNDS
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", help="Path to solver config JSON", default=None)
	args = parser.parse_args()
	config: Dict[str, object] = {}
	if args.config:
		config = _read_json(args.config)
		_apply_config(config)
	inp_path = f"./wdn/{WDN}.inp"
	wdn_name = WDN
	network = load_inp_network(inp_path)
	print(f"Using network: {wdn_name} ({inp_path})")
	resistances = compute_pipe_resistances(network)
	print(f"Loaded {len(network.nodes)} nodes and {len(network.pipes)} pipes.")
	first_pipe = next(iter(resistances.values()), None)
	if first_pipe is not None:
		print("Example resistance:", first_pipe)

	scenario_path = _resolve_scenario_path(config)
	scenario_payload = _read_json(scenario_path)
	scenario_wdn = str(scenario_payload.get("wdn") or scenario_payload.get("wdn_name") or "")
	if scenario_wdn and scenario_wdn != WDN:
		raise ValueError(f"Scenario WDN '{scenario_wdn}' does not match WDN '{WDN}'.")

	scenario_demands = {k: float(v) for k, v in scenario_payload["scenario_demands"].items()}
	scenario_heads = {k: float(v) for k, v in scenario_payload["scenario_heads"].items()}
	scenario_flows = {k: float(v) for k, v in scenario_payload["scenario_flows"].items()}

	c_bounds = {k: float(v) for k, v in scenario_payload.get("c_bounds", {}).items()}
	C_bounds = {k: float(v) for k, v in scenario_payload.get("C_bounds", {}).items()}
	PIPE_BOUNDS = bool(scenario_payload.get("pipe_bounds_enabled", bool(c_bounds or C_bounds)))
	total_demand = float(sum(scenario_demands.values()))
	total_demand_from_flows = _compute_total_demand_from_flows(network, scenario_flows)

	measurement_sets = _select_measurement_sets(network)
	if not measurement_sets:
		raise ValueError("No measurement sets selected. Provide MEASUREMENT_SITES as an integer or a non-empty list.")
	if isinstance(MEASUREMENT_SITES, int):
		mode = MEASUREMENT_SUBSET_MODE
		print(f"Measurement subsets: {len(measurement_sets)} (p={MEASUREMENT_SITES}, mode={mode})")

	master_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
	pipe_tag = "B" if PIPE_BOUNDS else "NB"
	summary_rows = []
	batch_dir = None
	if len(measurement_sets) > 1:
		if OUTPUT_DIR:
			batch_dir = OUTPUT_DIR
		else:
			batch_dir = os.path.join("data", wdn_name, f"{pipe_tag}-batch-{master_timestamp}")
		os.makedirs(batch_dir, exist_ok=True)
		_write_json(
			os.path.join(batch_dir, "parameters.json"),
			_build_parameters_snapshot(measurement_sets_count=len(measurement_sets), scenario_path=scenario_path),
		)

	print(f"PROGRESS_TOTAL: {len(measurement_sets)}", flush=True)
	for set_idx, measurement_nodes in enumerate(measurement_sets, start=1):
		print(f"PROGRESS: {set_idx}/{len(measurement_sets)}", flush=True)
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
		elif OUTPUT_DIR:
			output_dir = OUTPUT_DIR
		else:
			output_dir = os.path.join("data", wdn_name, f"{pipe_tag}-{measurement_tag}-{master_timestamp}")
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
				scenario_path=scenario_path,
			),
		)

		_write_json(
			os.path.join(output_dir, "metadata.json"),
			{
				"WDN": WDN,
				"WDN_NAME": WDN,
				"SCENARIO": SCENARIO,
				"SCENARIO_FILE": scenario_path,
				"OUTPUT_DIR": OUTPUT_DIR,
				"MEASUREMENT_SITES": measurement_nodes,
				"PIPE_BOUNDS": PIPE_BOUNDS,
				"HEADLOSS_MODEL": HEADLOSS_MODEL_LOCAL,
				"SOLVER": SOLVER,
				"GLOBAL_SOLVER": None,
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
				"radius": best_radius,
				"success": best_res.success,
				"max_violation": best_res.max_violation,
				"min_demand_viol": best_res.min_demand_viol,
				"objective": best_res.objective,
				"solver_status": best_res.solver_status,
				"best_bound": best_res.best_bound,
				"output_dir": output_dir,
			}
		)

	if summary_rows:
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
			print(f"GUI_SUMMARY_CSV: {csv_path}")
		except Exception as exc:
			print(f"WARNING: failed to write CSV summary ({exc})")


if __name__ == "__main__":
	main()
