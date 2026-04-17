from typing import Dict, Tuple
import json
import os
from datetime import datetime

from step1_io import load_inp_network
from step2_estimation import (
	simulate_random_demand_scenarios,
	simulate_base_flows,
	simulate_base_scenario,
	simulate_single_random_scenario,
	simulate_perturbed_demand_scenarios,
)


### NETWORK AND SCENARIO
WDN_NAME = "Alperovits"
SCENARIO_NAME = "Alperovits-base"
SCENARIO_SOURCE = "base"  # "base" or "random"


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


def _write_json(path: str, payload: Dict[str, object]) -> None:
	with open(path, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2, sort_keys=True)


def main() -> None:
	inp_path = f"./wdn/{WDN_NAME}.inp"
	network = load_inp_network(inp_path)
	print(f"Using network: {WDN_NAME} ({inp_path})")
	print(f"Loaded {len(network.nodes)} nodes and {len(network.pipes)} pipes.")

	base_flows = simulate_base_flows(inp_path=inp_path)

	c_bounds: Dict[str, float] = {}
	C_bounds: Dict[str, float] = {}
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

	scenario_dir = "scenario"
	os.makedirs(scenario_dir, exist_ok=True)
	timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
	output_path = os.path.join(scenario_dir, f"{SCENARIO_NAME}.json")

	payload = {
		"wdn_name": WDN_NAME,
		"inp_path": inp_path,
		"scenario_name": SCENARIO_NAME,
		"scenario_source": SCENARIO_SOURCE,
		"pipe_bounds_enabled": PIPE_BOUNDS,
		"pipe_bounds_params": {
			"PIPE_BOUND_SAFENESS": PIPE_BOUND_SAFENESS,
			"PIPE_BOUND_METHOD": PIPE_BOUND_METHOD,
			"PIPE_BOUND_POLICY": PIPE_BOUND_POLICY,
			"PIPE_BOUND_PERTURB_SAMPLES": PIPE_BOUND_PERTURB_SAMPLES,
			"PIPE_BOUND_PERTURB_SIGMA": PIPE_BOUND_PERTURB_SIGMA,
			"PIPE_BOUND_PERTURB_SEED": PIPE_BOUND_PERTURB_SEED,
			"PIPE_BOUND_PERTURB_FIX_TOTAL": PIPE_BOUND_PERTURB_FIX_TOTAL,
			"PIPE_BOUND_PERTURB_BASE": PIPE_BOUND_PERTURB_BASE,
		},
		"timestamp": timestamp,
		"scenario_demands": scenario_demands,
		"scenario_heads": scenario_heads,
		"scenario_flows": scenario_flows,
		"base_flows": base_flows,
		"c_bounds": c_bounds,
		"C_bounds": C_bounds,
	}

	_write_json(output_path, payload)
	print(f"Wrote scenario: {output_path}")


if __name__ == "__main__":
	main()
