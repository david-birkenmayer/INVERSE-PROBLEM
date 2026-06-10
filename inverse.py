from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from itertools import combinations
from typing import Dict, Iterable, List, Tuple

import numpy as np

from gui.cache import compute_hash, load_index, save_index
from step1_io import compute_pipe_resistances, compute_pipe_resistances_hw, load_inp_network
from step2_estimation import simulate_base_scenario
from step3_solver import SolverResult
from step3_solver import solve_single_pipe_bounds
from step3_solver_hexaly import solve_head_center_in_class_hexaly, solve_max_demand_distance_hexaly
from step3_solver_xd_hexaly import solve_max_demand_distance_xd_hexaly


WDN = "Alperovits"
MEASUREMENT_SITES: object = 2
MODE = "W_d"  # W_d, W_d_M, C_d, W_h, W_h_M, C_h_fixed, B
METHOD = "xd"  # xd, classical (used for W_d/C_d)
NORM = 2.0
DEMAND_LB = 1e-6
MULTI_STARTS = 1
MULTI_START_NOISE = 0.05
MULTI_START_NOISE_REL = 0.25
MULTI_START_SEED = None
DYNAMIC_MULTISTART = False
DMS_CONSISTENCY = 3
DMS_DEVIATION = 0.95
DMS_RADIUS = float("inf")
DMS_MAX_STARTS = 10
DMS_DISCARD_UNCLEAR = True
LINEARIZATION_LOOKUP = False
LINEARIZATION_ENABLED = False
LINEARIZATION_EPS_SCALE = 1e-3
LINEARIZED_PIPES = None
MEASUREMENT_HEADS_EQUAL_ONLY = True
HEXALY_LICENSE_PATH = os.path.expanduser("~/opt/Hexaly_14_5/license.dat")
HEXALY_TIME_LIMIT = 30
HEXALY_SEED = 0
HEXALY_VERBOSITY = 2
HEADLOSS_MODEL = "hw"  # auto, hw, dw
OUTPUT_DIR = None
MATCH_TOTAL_DEMAND = True
MATCH_RESERVOIR_OUTFLOW_BETWEEN_PAIRS = False
MEASUREMENT_SOURCE = "from_w_d"
MEASUREMENT_DATA = None
SOLVER_HASH = None
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TOTAL_DEMAND_KEY = "-1"
_RAW_CONFIG: Dict[str, object] = {}


def _read_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, payload: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _linearization_scale_from_heads(network, base_heads: Dict[str, float]) -> float:
    deltas: List[float] = []
    for pipe in network.pipes.values():
        h_u = float(base_heads.get(pipe.start_node, 0.0))
        h_v = float(base_heads.get(pipe.end_node, 0.0))
        deltas.append(abs(h_u - h_v))
    if not deltas:
        return 1.0
    return float(np.median(deltas))


def _linearization_certificate(
    network,
    pipe_resistances: Dict[str, float],
    base_flows: Dict[str, float],
    base_heads: Dict[str, float],
    sensor_heads: Dict[str, float],
    measurement_nodes: List[str],
    measurement_heads_equal_only_local: bool,
    reservoir_node: str | None,
    reservoir_outflow: float | None,
    demand_lb: float,
    demand_lb_per_node: Dict[str, float],
    total_demand: float | None,
    total_demand_upper: float | None,
    initial_guess: SolverResult,
    epsilon_h_scale: float,
) -> Dict[str, object]:
    median_delta_h = _linearization_scale_from_heads(network, base_heads)
    epsilon_h = float(epsilon_h_scale) * float(median_delta_h)
    linearized: Dict[str, float] = {}
    bounds: Dict[str, Dict[str, float]] = {}
    total_pipes = len(network.pipes)
    checked = 0

    for pipe_id in network.pipes:
        q0 = float(base_flows.get(pipe_id, 0.0))
        if q0 == 0.0:
            checked += 1
            print(
                f"LINEARIZATION_PROGRESS: {checked}/{total_pipes} certified={len(linearized)}",
                flush=True,
            )
            continue
        res = solve_single_pipe_bounds(
            network=network,
            target_pipe=pipe_id,
            sensor_heads=sensor_heads,
            pipe_primary=set(),
            pipe_secondary=set(),
            c_secondary={},
            C_primary={},
            pipe_resistances=pipe_resistances,
            preferred_flow_sign=base_flows,
            reservoir_node=reservoir_node,
            reservoir_outflow=reservoir_outflow,
            total_demand=total_demand,
            total_demand_upper=total_demand_upper,
            demand_lb_per_node=demand_lb_per_node,
            demand_lb=demand_lb,
            initial_guess=initial_guess,
            measurement_nodes=measurement_nodes,
            measurement_heads_equal_only=measurement_heads_equal_only_local,
        )
        delta_e = math.sqrt(max(epsilon_h, 0.0) / max(float(pipe_resistances[pipe_id]), 1e-12))
        q_min = float(res.q_min)
        q_max = float(res.q_max)
        lower = q0 - delta_e
        upper = q0 + delta_e
        same_sign = (q0 > 0.0 and q_min > 0.0 and q_max > 0.0) or (q0 < 0.0 and q_min < 0.0 and q_max < 0.0)
        linearizable = same_sign and q_min >= lower and q_max <= upper
        bounds[pipe_id] = {
            "q0": q0,
            "q_min": q_min,
            "q_max": q_max,
            "delta_e": delta_e,
            "epsilon_h": epsilon_h,
            "linearizable": bool(linearizable),
            "min_success": bool(res.min_success),
            "max_success": bool(res.max_success),
            "min_violation": float(res.min_violation),
            "max_violation": float(res.max_violation),
            "min_demand_viol": float(res.min_demand_viol),
            "max_demand_viol": float(res.max_demand_viol),
        }
        if linearizable:
            linearized[pipe_id] = q0
        checked += 1
        print(
            f"LINEARIZATION_PROGRESS: {checked}/{total_pipes} certified={len(linearized)}",
            flush=True,
        )

    return {
        "epsilon_h": epsilon_h,
        "median_delta_h": median_delta_h,
        "linearized_pipes": linearized,
        "pipe_bounds": bounds,
    }


def _apply_config(config: Dict[str, object]) -> None:
    global WDN, MEASUREMENT_SITES, MODE, METHOD, _RAW_CONFIG

    _RAW_CONFIG = dict(config)

    if "WDN" in config:
        WDN = str(config["WDN"])
    elif "WDN_NAME" in config:
        WDN = str(config["WDN_NAME"])

    for key in {
        "MEASUREMENT_SITES",
        "MODE",
        "METHOD",
        "NORM",
        "DEMAND_LB",
        "MULTI_STARTS",
        "MULTI_START_NOISE",
        "MULTI_START_NOISE_REL",
        "MULTI_START_SEED",
        "DYNAMIC_MULTISTART",
        "DMS_CONSISTENCY",
        "DMS_DEVIATION",
        "DMS_RADIUS",
        "DMS_MAX_STARTS",
        "DMS_DISCARD_UNCLEAR",
        "LINEARIZATION_LOOKUP",
        "LINEARIZATION_ENABLED",
        "LINEARIZATION_EPS_SCALE",
        "LINEARIZED_PIPES",
        "MEASUREMENT_HEADS_EQUAL_ONLY",
        "HEXALY_LICENSE_PATH",
        "HEXALY_TIME_LIMIT",
        "HEXALY_SEED",
        "HEXALY_VERBOSITY",
        "HEADLOSS_MODEL",
        "OUTPUT_DIR",
        "SOLVER_HASH",
        "MATCH_TOTAL_DEMAND",
        "MATCH_RESERVOIR_OUTFLOW_BETWEEN_PAIRS",
        "MEASUREMENT_SOURCE",
        "MEASUREMENT_DATA",
    }:
        if key in config:
            globals()[key] = config[key]


def _normalize_measurement_dict(data: object) -> Dict[str, float]:
    if not isinstance(data, dict):
        raise ValueError("Measurement data must be a dictionary.")
    normalized: Dict[str, float] = {}
    for key, value in data.items():
        normalized[str(key)] = float(value)
    return normalized


def _build_measurement_dict(
    measurement_nodes: Iterable[str],
    heads: Dict[str, float],
    total_demand: float,
    reservoir_node: str | None,
    reservoir_head: float | None,
) -> Dict[str, float]:
    payload = {str(node): float(heads[node]) for node in measurement_nodes if node in heads}
    payload[TOTAL_DEMAND_KEY] = float(total_demand)
    if reservoir_node is not None and reservoir_head is not None:
        payload[str(reservoir_node)] = float(reservoir_head)
    return payload


def _measurement_heads_for_sites(measurement_data: Dict[str, float], measurement_nodes: Iterable[str]) -> Dict[str, float]:
    return {
        str(node): float(measurement_data[str(node)])
        for node in measurement_nodes
        if str(node) in measurement_data
    }


def _measurement_total_demand(measurement_data: Dict[str, float], fallback: float) -> float:
    return float(measurement_data.get(TOTAL_DEMAND_KEY, fallback))


def _measurement_reservoir_head(
    measurement_data: Dict[str, float],
    reservoir_node: str | None,
    fallback: float | None,
) -> float | None:
    if reservoir_node is not None and str(reservoir_node) in measurement_data:
        return float(measurement_data[str(reservoir_node)])
    return fallback


def _solver_index_path(wdn: str) -> str:
    return os.path.join(ROOT_DIR, "data", wdn, "cache_index.json")


def _legacy_index_path() -> str:
    return os.path.join(ROOT_DIR, "data", "cache_index.json")


def _sanitize_hash_payload(payload: Dict[str, object]) -> Dict[str, object]:
    return {
        str(k): v
        for k, v in payload.items()
        if k not in {"OUTPUT_DIR", "SOLVER_HASH", "_index", "_index_path", "DMS_RADIUS", "LINEARIZATION_EPS_SCALE"}
    }


def _load_index_with_legacy(path: str) -> Dict[str, str]:
    index = load_index(path, ROOT_DIR)
    if index:
        return index
    return load_index(_legacy_index_path(), ROOT_DIR)


def _resolve_cached_output_dir(hash_key: str, wdn: str) -> str | None:
    index = _load_index_with_legacy(_solver_index_path(wdn))
    cached_dir = index.get(hash_key)
    if not cached_dir:
        return None
    resolved = cached_dir if os.path.isabs(cached_dir) else os.path.join(ROOT_DIR, cached_dir)
    if os.path.isdir(resolved):
        return resolved
    return None


def _write_gui_hash(output_dir: str, solver_hash: str) -> None:
    try:
        with open(os.path.join(output_dir, "_gui_hash.txt"), "w", encoding="utf-8") as f:
            f.write(solver_hash)
    except OSError:
        pass


def _register_output_dir(hash_key: str | None, output_dir: str | None, wdn: str) -> None:
    if not hash_key or not output_dir:
        return
    resolved = output_dir if os.path.isabs(output_dir) else os.path.join(ROOT_DIR, str(output_dir))
    if not os.path.isdir(resolved):
        return
    index_path = _solver_index_path(wdn)
    index = _load_index_with_legacy(index_path)
    rel_output = output_dir if not os.path.isabs(str(output_dir)) else os.path.relpath(str(output_dir), ROOT_DIR)
    index[str(hash_key)] = rel_output
    save_index(index_path, index, ROOT_DIR)
    _write_gui_hash(resolved, str(hash_key))


def _run_prerequisite_w_mode(wdn: str, payload: Dict[str, object], hash_key: str) -> str:
    output_dir = os.path.join("data", wdn, hash_key[:8])
    config_payload = dict(payload)
    config_payload["OUTPUT_DIR"] = output_dir
    config_payload["SOLVER_HASH"] = hash_key
    cache_dir = os.path.join(ROOT_DIR, ".gui_cache")
    os.makedirs(cache_dir, exist_ok=True)
    config_path = os.path.join(cache_dir, f"solver-{hash_key}.json")
    _write_json(config_path, config_payload)
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT_DIR, "inverse.py"), "--config", config_path],
        capture_output=True,
        text=True,
        cwd=ROOT_DIR,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Automatic prerequisite W_d run failed: {proc.stderr.strip() or proc.stdout.strip()}")
    _register_output_dir(hash_key, output_dir, wdn)
    resolved = os.path.join(ROOT_DIR, output_dir)
    if not os.path.isdir(resolved):
        raise RuntimeError("Automatic prerequisite W_d run completed without output directory.")
    return resolved


def _resolve_measurement_from_w_mode(
    wdn: str,
    mode: str,
    source_mode: str,
    measurement_nodes: List[str],
    base_measurement: Dict[str, float],
) -> Dict[str, float]:
    if mode == source_mode:
        return dict(base_measurement)

    if not _RAW_CONFIG:
        raise ValueError("from_w_* measurement source requires a config-backed solver run.")

    w_payload = _sanitize_hash_payload(dict(_RAW_CONFIG))
    w_payload["MODE"] = source_mode
    w_payload["MEASUREMENT_SITES"] = list(measurement_nodes)
    w_payload["MEASUREMENT_SOURCE"] = "base"
    w_payload["MEASUREMENT_DATA"] = None
    w_hash = compute_hash(w_payload)
    output_dir = _resolve_cached_output_dir(w_hash, wdn)
    if output_dir is None:
        output_dir = _run_prerequisite_w_mode(wdn, w_payload, w_hash)

    candidates_path = os.path.join(output_dir, "measurement_candidates.json")
    if not os.path.isfile(candidates_path):
        raise ValueError(f"Cached {source_mode} result has no measurement_candidates.json.")
    payload = _read_json(candidates_path)
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"Cached {source_mode} result has no measurement candidates.")
    first = candidates[0]
    if not isinstance(first, dict):
        raise ValueError(f"Cached {source_mode} result has invalid measurement candidate format.")
    return _normalize_measurement_dict(first.get("data", {}))


def _resolve_center_state_from_w_mode(
    wdn: str,
    mode: str,
    source_mode: str,
    measurement_nodes: List[str],
) -> Dict[str, Dict[str, float]] | None:
    if mode == source_mode:
        return None
    if not _RAW_CONFIG:
        return None

    w_payload = _sanitize_hash_payload(dict(_RAW_CONFIG))
    w_payload["MODE"] = source_mode
    w_payload["MEASUREMENT_SITES"] = list(measurement_nodes)
    w_payload["MEASUREMENT_SOURCE"] = "base"
    w_payload["MEASUREMENT_DATA"] = None
    w_hash = compute_hash(w_payload)
    output_dir = _resolve_cached_output_dir(w_hash, wdn)
    if output_dir is None:
        output_dir = _run_prerequisite_w_mode(wdn, w_payload, w_hash)

    candidates_path = os.path.join(output_dir, "measurement_candidates.json")
    if not os.path.isfile(candidates_path):
        return None
    payload = _read_json(candidates_path)
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    first = candidates[0]
    if not isinstance(first, dict):
        return None
    state = first.get("state")
    if not isinstance(state, dict):
        return None
    demands = {str(k): float(v) for k, v in dict(state.get("demands", {})).items()}
    heads = {str(k): float(v) for k, v in dict(state.get("heads", {})).items()}
    flows = {str(k): float(v) for k, v in dict(state.get("flows", {})).items()}
    return {"demands": demands, "heads": heads, "flows": flows}


def _resolve_center_state(
    wdn: str,
    mode: str,
    measurement_nodes: List[str],
    base_demands: Dict[str, float],
    base_heads: Dict[str, float],
    base_flows: Dict[str, float],
) -> Dict[str, Dict[str, float]] | None:
    source = str(MEASUREMENT_SOURCE or "from_w_d").strip().lower()
    if source == "base":
        return {
            "demands": {str(k): float(v) for k, v in base_demands.items()},
            "heads": {str(k): float(v) for k, v in base_heads.items()},
            "flows": {str(k): float(v) for k, v in base_flows.items()},
        }
    if source == "from_w_d":
        return _resolve_center_state_from_w_mode(wdn, mode, "W_d", measurement_nodes)
    if source == "from_w_h":
        return _resolve_center_state_from_w_mode(wdn, mode, "W_h", measurement_nodes)
    return None


def _resolve_measurement_data(
    wdn: str,
    mode: str,
    measurement_nodes: List[str],
    base_heads: Dict[str, float],
    total_demand: float,
    reservoir_node: str | None,
    reservoir_head: float | None,
) -> Dict[str, float]:
    base_measurement = _build_measurement_dict(measurement_nodes, base_heads, total_demand, reservoir_node, reservoir_head)
    source = str(MEASUREMENT_SOURCE or "from_w_d").strip().lower()
    if source == "base":
        return base_measurement
    if source == "custom":
        return _normalize_measurement_dict(MEASUREMENT_DATA)
    if source == "from_w_d":
        return _resolve_measurement_from_w_mode(wdn, mode, "W_d", measurement_nodes, base_measurement)
    if source == "from_w_h":
        return _resolve_measurement_from_w_mode(wdn, mode, "W_h", measurement_nodes, base_measurement)
    raise ValueError(f"Unsupported MEASUREMENT_SOURCE: {MEASUREMENT_SOURCE}")


def _compute_norm(values: Iterable[float], norm_p: float) -> float:
    vals = [abs(float(v)) for v in values]
    if not vals:
        return 0.0
    if math.isinf(norm_p):
        return max(vals)
    if norm_p <= 0:
        raise ValueError("NORM must be positive.")
    return float(sum(v ** norm_p for v in vals) ** (1.0 / norm_p))


def _compute_demands_from_flows(network, flows: Dict[str, float]) -> Dict[str, float]:
    reservoir_nodes = set(network.reservoirs.keys())
    junctions = [j for j in network.junctions.keys() if j not in reservoir_nodes]
    values: Dict[str, float] = {j: 0.0 for j in junctions}
    for pipe_id, pipe in network.pipes.items():
        q = float(flows.get(pipe_id, 0.0))
        if pipe.end_node in values:
            values[pipe.end_node] += q
        if pipe.start_node in values:
            values[pipe.start_node] -= q
    return values


def _select_measurement_sets(network) -> List[List[str]]:
    if isinstance(MEASUREMENT_SITES, int):
        p = int(MEASUREMENT_SITES)
        candidates = sorted(network.junctions.keys())
        if p < 0 or p > len(candidates):
            return []
        return [list(x) for x in combinations(candidates, p)] if p > 0 else [[]]

    if isinstance(MEASUREMENT_SITES, str):
        text = MEASUREMENT_SITES.strip()
        if not text or text == "#0":
            return [[]]
        m_range = None
        if text.startswith("#"):
            m_range = text.split("-#", 1) if "-#" in text else None
        if m_range and len(m_range) == 2 and m_range[0].startswith("#"):
            try:
                a = int(m_range[0][1:])
                b = int(m_range[1])
            except ValueError:
                return []
            candidates = sorted(network.junctions.keys())
            if a < 0 or b < 0 or a > b or a > len(candidates):
                return []
            b = min(b, len(candidates))
            sets: List[List[str]] = []
            for p in range(a, b + 1):
                if p == 0:
                    sets.append([])
                else:
                    sets.extend([list(x) for x in combinations(candidates, p)])
            return sets
        if text.startswith("#"):
            try:
                p = int(text[1:])
            except ValueError:
                return []
            candidates = sorted(network.junctions.keys())
            if p < 0 or p > len(candidates):
                return []
            return [list(x) for x in combinations(candidates, p)] if p > 0 else [[]]
        return [[tok.strip() for tok in text.split(",") if tok.strip()]]

    if isinstance(MEASUREMENT_SITES, (list, tuple)):
        if len(MEASUREMENT_SITES) == 0:
            return [[]]
        if all(isinstance(item, (list, tuple)) for item in MEASUREMENT_SITES):
            return [[str(x) for x in item] for item in MEASUREMENT_SITES]
        return [[str(x) for x in MEASUREMENT_SITES]]

    return [[]]


def _mode_value(
    mode: str,
    norm_p: float,
    result,
    junctions: List[str],
    all_nodes: List[str],
    reference_demands: Dict[str, float],
    reference_heads: Dict[str, float],
) -> float:
    if mode == "W_d":
        return _compute_norm(
            (result.demands_a.get(j, 0.0) - result.demands_b.get(j, 0.0) for j in junctions),
            norm_p,
        )
    if mode == "C_d":
        return _compute_norm(
            (result.demands_a.get(j, 0.0) - reference_demands.get(j, 0.0) for j in junctions),
            norm_p,
        )
    if mode == "W_h":
        return _compute_norm(
            (result.heads_a.get(n, 0.0) - result.heads_b.get(n, 0.0) for n in all_nodes),
            norm_p,
        )
    if mode in {"C_h", "C_h_fixed"}:
        return _compute_norm(
            (result.heads_a.get(n, 0.0) - reference_heads.get(n, 0.0) for n in all_nodes),
            norm_p,
        )
    if mode == "B":
        return float(result.objective or 0.0)
    raise ValueError(f"Unsupported MODE: {mode}")


def _center_mode_value(
    mode: str,
    norm_p: float,
    result,
    junctions: List[str],
    all_nodes: List[str],
    reference_demands: Dict[str, float],
    reference_heads: Dict[str, float],
) -> float:
    if mode == "C_d":
        d_a = _compute_norm((result.demands_a.get(j, 0.0) - reference_demands.get(j, 0.0) for j in junctions), norm_p)
        d_b = _compute_norm((result.demands_b.get(j, 0.0) - reference_demands.get(j, 0.0) for j in junctions), norm_p)
        return max(d_a, d_b)
    if mode in {"C_h", "C_h_fixed"}:
        h_a = _compute_norm((result.heads_a.get(n, 0.0) - reference_heads.get(n, 0.0) for n in all_nodes), norm_p)
        h_b = _compute_norm((result.heads_b.get(n, 0.0) - reference_heads.get(n, 0.0) for n in all_nodes), norm_p)
        return max(h_a, h_b)
    return _mode_value(mode, norm_p, result, junctions, all_nodes, reference_demands, reference_heads)


def _build_parameters_snapshot(
    measurement_nodes: Iterable[str],
    measurement_sets_count: int,
    headloss_model_local: str,
    measurement_data: Dict[str, float] | None,
) -> Dict[str, object]:
    extra_demand = _RAW_CONFIG.get("EXTRA_DEMAND") if isinstance(_RAW_CONFIG, dict) else None
    demand_model = _RAW_CONFIG.get("DEMAND_MODEL") if isinstance(_RAW_CONFIG, dict) else None
    return {
        "WDN": WDN,
        "WDN_NAME": WDN,
        "MEASUREMENT_SITES": MEASUREMENT_SITES,
        "MEASUREMENT_NODES": list(measurement_nodes),
        "MEASUREMENT_SET_COUNT": measurement_sets_count,
        "MEASUREMENT_SOURCE": MEASUREMENT_SOURCE,
        "MEASUREMENT_DATA": measurement_data,
        "MODE": MODE,
        "METHOD": METHOD,
        "NORM": NORM,
        "DEMAND_LB": DEMAND_LB,
        "MULTI_STARTS": MULTI_STARTS,
        "MULTI_START_NOISE": MULTI_START_NOISE,
        "MULTI_START_NOISE_REL": MULTI_START_NOISE_REL,
        "MULTI_START_SEED": MULTI_START_SEED,
        "DYNAMIC_MULTISTART": DYNAMIC_MULTISTART,
        "DMS_CONSISTENCY": DMS_CONSISTENCY,
        "DMS_DEVIATION": DMS_DEVIATION,
        "DMS_RADIUS": DMS_RADIUS,
        "DMS_MAX_STARTS": DMS_MAX_STARTS,
        "DMS_DISCARD_UNCLEAR": DMS_DISCARD_UNCLEAR,
        "LINEARIZATION_LOOKUP": LINEARIZATION_LOOKUP,
        "LINEARIZATION_ENABLED": LINEARIZATION_ENABLED,
        "LINEARIZATION_EPS_SCALE": LINEARIZATION_EPS_SCALE,
        "LINEARIZED_PIPES": LINEARIZED_PIPES,
        "MEASUREMENT_HEADS_EQUAL_ONLY": MEASUREMENT_HEADS_EQUAL_ONLY,
        "HEADLOSS_MODEL": HEADLOSS_MODEL,
        "HEADLOSS_MODEL_LOCAL": headloss_model_local,
        "HEXALY_LICENSE_PATH": HEXALY_LICENSE_PATH,
        "HEXALY_TIME_LIMIT": HEXALY_TIME_LIMIT,
        "HEXALY_SEED": HEXALY_SEED,
        "HEXALY_VERBOSITY": HEXALY_VERBOSITY,
        "OUTPUT_DIR": OUTPUT_DIR,
        "MATCH_RESERVOIR_OUTFLOW_BETWEEN_PAIRS": MATCH_RESERVOIR_OUTFLOW_BETWEEN_PAIRS,
        "EXTRA_DEMAND": extra_demand,
        "DEMAND_MODEL": demand_model,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to solver config JSON", default=None)
    args = parser.parse_args()

    if args.config:
        _apply_config(_read_json(args.config))

    mode_raw = str(MODE)
    mode = mode_raw
    is_posteriori_mode = False
    method = str(METHOD).lower()
    if mode in {"W_d(M)", "W_d_M"}:
        mode = "W_d"
        is_posteriori_mode = True
    elif mode in {"W_h(M)", "W_h_M"}:
        mode = "W_h"
        is_posteriori_mode = True
    elif mode in {"H_h", "C_h"}:
        mode = "C_h_fixed"
    if mode not in {"W_d", "C_d", "W_h", "C_h_fixed", "B"}:
        raise ValueError("MODE must be one of: W_d, W_d_M, C_d, W_h, W_h_M, C_h_fixed, B.")
    if method not in {"xd", "classical"}:
        raise ValueError("METHOD must be 'xd' or 'classical'.")

    inp_path = f"./wdn/{WDN}.inp"
    network = load_inp_network(inp_path)
    print(f"Using network: {WDN} ({inp_path})")

    base_demands, base_heads, base_flows = simulate_base_scenario(inp_path=inp_path, simulator="auto")
    reference_demands = _compute_demands_from_flows(network, base_flows)

    if HEADLOSS_MODEL == "auto":
        opt = network.options.get("headloss") or network.options.get("hydraulic")
        model_opt = str(opt).lower() if opt is not None else ""
        headloss_model_local = "hw" if ("hazen" in model_opt or "hw" in model_opt) else "dw"
    else:
        headloss_model_local = str(HEADLOSS_MODEL).lower()

    if headloss_model_local == "hw":
        pipe_res = {pid: vals["r_e"] for pid, vals in compute_pipe_resistances_hw(network).items()}
        headloss_n = 1.852
    else:
        pipe_res = {pid: vals["r_e"] for pid, vals in compute_pipe_resistances(network).items()}
        headloss_n = 2.0

    reservoir_node = next(iter(network.reservoirs.keys()), None)
    reservoir_head = base_heads.get(reservoir_node) if reservoir_node else None

    total_demand = float(sum(reference_demands.values()))
    sensor_sets = _select_measurement_sets(network)
    if not sensor_sets:
        raise ValueError("No measurement sets selected.")

    master_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    batch_dir = None
    if len(sensor_sets) > 1:
        batch_dir = OUTPUT_DIR or os.path.join("data", WDN, f"batch-{master_timestamp}")
        os.makedirs(batch_dir, exist_ok=True)
        _write_json(
            os.path.join(batch_dir, "parameters.json"),
            _build_parameters_snapshot([], len(sensor_sets), headloss_model_local, None),
        )

    print(f"PROGRESS_TOTAL: {len(sensor_sets)}", flush=True)

    rng = np.random.default_rng(MULTI_START_SEED)
    junctions = list(network.junctions.keys())
    all_nodes = list(network.nodes.keys())

    for set_idx, measurement_nodes in enumerate(sensor_sets, start=1):
        print(f"PROGRESS: {set_idx}/{len(sensor_sets)}", flush=True)

        measurement_tag = "_".join(measurement_nodes) if measurement_nodes else "none"
        if batch_dir:
            output_dir = os.path.join(batch_dir, measurement_tag)
        elif OUTPUT_DIR:
            output_dir = str(OUTPUT_DIR)
        else:
            output_dir = os.path.join("data", WDN, f"{measurement_tag}-{master_timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        _write_json(
            os.path.join(output_dir, "parameters.json"),
            _build_parameters_snapshot(measurement_nodes, len(sensor_sets), headloss_model_local, None),
        )

        if not is_posteriori_mode and mode in {"W_d", "W_h", "B"}:
            # A-priori W-modes must not condition on a fixed measurement instance.
            measurement_data = _build_measurement_dict(
                measurement_nodes,
                base_heads,
                total_demand,
                reservoir_node,
                reservoir_head,
            )
        else:
            measurement_data = _resolve_measurement_data(
                WDN,
                mode,
                list(measurement_nodes),
                base_heads,
                total_demand,
                reservoir_node,
                reservoir_head,
            )
        sensor_heads = _measurement_heads_for_sites(measurement_data, measurement_nodes)
        reservoir_head_local = _measurement_reservoir_head(measurement_data, reservoir_node, reservoir_head)
        total_demand_local = _measurement_total_demand(measurement_data, total_demand)
        if reservoir_node and reservoir_head_local is not None:
            sensor_heads[reservoir_node] = float(reservoir_head_local)

        measurement_heads_equal_only_local = bool(MEASUREMENT_HEADS_EQUAL_ONLY)
        if is_posteriori_mode and mode in {"W_d", "W_h"}:
            measurement_heads_equal_only_local = False
        elif mode in {"W_d", "W_h", "B"}:
            # A-priori W-modes must stay in equivalence-class form, not absolute head fitting.
            measurement_heads_equal_only_local = True

        _write_json(
            os.path.join(output_dir, "parameters.json"),
            _build_parameters_snapshot(measurement_nodes, len(sensor_sets), headloss_model_local, measurement_data),
        )

        center_state = _resolve_center_state(
            WDN,
            mode,
            list(measurement_nodes),
            reference_demands,
            base_heads,
            base_flows,
        )
        if not is_posteriori_mode and mode in {"W_d", "W_h", "B"}:
            center_state = None
        reference_demands_local = (
            center_state.get("demands", reference_demands) if center_state is not None else reference_demands
        )
        reference_heads_local = (
            center_state.get("heads", base_heads) if center_state is not None else base_heads
        )

        demand_lb_per_node = {
            j: max(float(DEMAND_LB), float(reference_demands_local.get(j, DEMAND_LB)))
            for j in reference_demands_local
        }

        # With per-node Dirichlet lower bounds (d_j >= base_j for all j),
        # enforcing sum(d_j) == base_total simultaneously forces d_j == base_j
        # exactly, leaving no room for demand variation.  Only pass a total-demand
        # equality constraint when the measured value genuinely exceeds the base
        # total (i.e. the measurement captured extra demand above the base).
        base_total = float(sum(reference_demands_local.values()))
        solver_total_demand: "float | None" = None
        solver_total_demand_upper: "float | None" = None
        match_total_between_pairs_local = bool(MATCH_RESERVOIR_OUTFLOW_BETWEEN_PAIRS)
        extra_budget = None
        try:
            extra_budget = float(_RAW_CONFIG.get("EXTRA_DEMAND")) if isinstance(_RAW_CONFIG, dict) and _RAW_CONFIG.get("EXTRA_DEMAND") is not None else None
        except (TypeError, ValueError):
            extra_budget = None

        # Posteriori modes represent a fixed measurement instance; honor explicit measured total demand.
        if is_posteriori_mode and TOTAL_DEMAND_KEY in measurement_data:
            solver_total_demand = float(total_demand_local) if total_demand_local is not None else None
        # A-priori W-modes should not condition on a concrete measurement M.
        # Use only the known extra-demand budget around base demand.
        elif mode in {"W_d", "W_h", "B"}:
            if extra_budget is not None and extra_budget >= 0.0:
                solver_total_demand_upper = base_total + extra_budget
            else:
                solver_total_demand_upper = None
            # Measure-equivalence classes require paired scenarios to have equal total demand.
            match_total_between_pairs_local = True
        else:
            enforce_total_demand = bool(MATCH_TOTAL_DEMAND)
            if enforce_total_demand:
                solver_total_demand = float(total_demand_local) if total_demand_local is not None else None
            else:
                solver_total_demand = (
                    total_demand_local if total_demand_local is not None and total_demand_local > base_total + 1e-6
                    else None
                )

        base_guess = SolverResult(
            status="base",
            demands=dict(reference_demands),
            heads=dict(base_heads),
            flows=dict(base_flows),
        )

        linearization_payload = None
        linearized_pipes: Dict[str, float] = {}
        if mode == "W_d" and (LINEARIZATION_LOOKUP or LINEARIZATION_ENABLED):
            selected_linearized_pipes: Dict[str, float] = {}
            if LINEARIZATION_ENABLED and isinstance(LINEARIZED_PIPES, dict):
                selected_linearized_pipes = {
                    str(pid): float(qref)
                    for pid, qref in LINEARIZED_PIPES.items()
                }
            if selected_linearized_pipes and not LINEARIZATION_LOOKUP:
                linearized_pipes = dict(selected_linearized_pipes)
                linearization_payload = {
                    "linearized_pipes": dict(selected_linearized_pipes),
                    "pipe_bounds": {},
                }

            linearization_path = os.path.join(output_dir, "linearization.json") if OUTPUT_DIR else ""
            if linearization_payload is None and linearization_path and os.path.isfile(linearization_path):
                try:
                    linearization_payload = _read_json(linearization_path)
                except Exception:
                    linearization_payload = None
            if linearization_payload is None and LINEARIZATION_ENABLED and isinstance(_RAW_CONFIG, dict):
                lookup_config = dict(_RAW_CONFIG)
                lookup_config["LINEARIZATION_LOOKUP"] = True
                lookup_config["LINEARIZATION_ENABLED"] = False
                lookup_hash = compute_hash(_sanitize_hash_payload(lookup_config))
                lookup_output_dir = os.path.join("data", WDN, lookup_hash)
                lookup_linearization_path = os.path.join(lookup_output_dir, "linearization.json")
                if os.path.isfile(lookup_linearization_path):
                    try:
                        linearization_payload = _read_json(lookup_linearization_path)
                    except Exception:
                        linearization_payload = None
            if linearization_payload is None:
                linearization_payload = _linearization_certificate(
                    network=network,
                    pipe_resistances=pipe_res,
                    base_flows=base_flows,
                    base_heads=base_heads,
                    sensor_heads=sensor_heads,
                    measurement_nodes=list(measurement_nodes),
                    measurement_heads_equal_only_local=measurement_heads_equal_only_local,
                    reservoir_node=reservoir_node,
                    reservoir_outflow=None,
                    demand_lb=float(DEMAND_LB),
                    demand_lb_per_node=demand_lb_per_node,
                    total_demand=solver_total_demand,
                    total_demand_upper=solver_total_demand_upper,
                    initial_guess=base_guess,
                    epsilon_h_scale=float(LINEARIZATION_EPS_SCALE),
                )
            default_linearized_pipes = {
                str(pid): float(qref)
                for pid, qref in linearization_payload.get("linearized_pipes", {}).items()
            }
            if not linearized_pipes:
                linearized_pipes = selected_linearized_pipes if selected_linearized_pipes else default_linearized_pipes
            _write_json(
                os.path.join(output_dir, "linearization.json"),
                {
                    "WDN": WDN,
                    "mode": mode,
                    "measurement_nodes": list(measurement_nodes),
                    "selected_linearized_pipes": linearized_pipes,
                    **linearization_payload,
                },
            )
            if LINEARIZATION_LOOKUP and not LINEARIZATION_ENABLED:
                print(
                    f"Linearization lookup complete: {len(linearized_pipes)} / {len(network.pipes)} pipes certified.",
                    flush=True,
                )
                continue

        minimize_mode = mode in {"C_d", "C_h", "C_h_fixed"}
        best_value = float("inf") if minimize_mode else -float("inf")
        best_result = None
        best_center_result = None
        runs = []
        dms_enabled = bool(DYNAMIC_MULTISTART)
        dms_consistency = max(1, int(DMS_CONSISTENCY))
        dms_max_starts_cfg = max(1, int(DMS_MAX_STARTS))
        dms_deviation = float(DMS_DEVIATION)
        if dms_deviation < 0.0:
            dms_deviation = 0.0
        if dms_deviation > 1.0:
            dms_deviation = 1.0
        try:
            dms_reference_radius = float(DMS_RADIUS)
        except (TypeError, ValueError):
            dms_reference_radius = float("inf")
        dms_lower_bound = -float("inf")
        dms_upper_bound = float("inf")
        dms_certificate = "none"
        dms_finished = False
        valid_mode_values: List[float] = []
        run_idx = 0
        dms_seed_base = int(MULTI_START_SEED) if MULTI_START_SEED is not None else int(HEXALY_SEED)
        dms_max_starts = max(max(1, int(MULTI_STARTS)), dms_consistency)
        if dms_enabled:
            dms_max_starts = max(dms_max_starts, dms_max_starts_cfg)

        while True:
            q0 = {k: float(v) for k, v in base_guess.flows.items()}
            h0 = {k: float(v) for k, v in base_guess.heads.items()}

            multi_run_target = dms_max_starts if dms_enabled else max(1, int(MULTI_STARTS))
            if multi_run_target > 1:
                if dms_enabled:
                    run_rng = np.random.default_rng(dms_seed_base + 1009 * (run_idx + 1))
                    profile = run_idx % 4
                    profile_scale = 1.0 + 0.35 * float(run_idx // 4)
                    flow_abs = float(MULTI_START_NOISE) * profile_scale
                    flow_rel = float(MULTI_START_NOISE_REL) * profile_scale
                    head_abs = float(MULTI_START_NOISE) * profile_scale
                    head_rel = float(MULTI_START_NOISE_REL) * profile_scale

                    if profile == 0:
                        perturb_q = True
                        perturb_h = True
                    elif profile == 1:
                        perturb_q = True
                        perturb_h = False
                        flow_abs *= 2.0
                        flow_rel *= 2.0
                    elif profile == 2:
                        perturb_q = False
                        perturb_h = True
                        head_abs *= 2.0
                        head_rel *= 2.0
                    else:
                        perturb_q = True
                        perturb_h = True
                        flow_abs *= 1.5
                        flow_rel *= 1.5
                        head_abs *= 1.5
                        head_rel *= 1.5

                    if perturb_q:
                        for k in q0:
                            scale = max(1.0, abs(q0[k]))
                            q0[k] += (flow_abs + flow_rel * scale) * run_rng.standard_normal()
                    if perturb_h:
                        for k in h0:
                            scale = max(1.0, abs(h0[k]))
                            h0[k] += (head_abs + head_rel * scale) * run_rng.standard_normal()
                else:
                    for k in q0:
                        scale = max(1.0, abs(q0[k]))
                        q0[k] += (float(MULTI_START_NOISE) + float(MULTI_START_NOISE_REL) * scale) * rng.standard_normal()
                    for k in h0:
                        scale = max(1.0, abs(h0[k]))
                        h0[k] += (float(MULTI_START_NOISE) + float(MULTI_START_NOISE_REL) * scale) * rng.standard_normal()

            run_guess = SolverResult(status="ms", demands=base_guess.demands, heads=h0, flows=q0)
            hexaly_seed = int(HEXALY_SEED) + run_idx

            solve_with_xd = method == "xd" and mode in {"W_d", "C_d"}
            if mode in {"W_h", "C_h", "C_h_fixed"}:
                solve_with_xd = False
            if not measurement_heads_equal_only_local:
                solve_with_xd = False
            if LINEARIZATION_ENABLED and mode == "W_d":
                solve_with_xd = False

            if solve_with_xd:
                fixed_b = reference_demands if mode == "C_d" else None
                result = solve_max_demand_distance_xd_hexaly(
                    network=network,
                    sensor_heads=sensor_heads,
                    pipe_resistances=pipe_res,
                    c_bounds=None,
                    C_bounds=None,
                    reservoir_node=reservoir_node,
                    reservoir_head=reservoir_head_local,
                    reservoir_outflow=None,
                    initial_guess=run_guess,
                    norm_p=float(NORM),
                    demand_lb=float(DEMAND_LB),
                    demand_lb_per_node=demand_lb_per_node,
                    total_demand=solver_total_demand,
                    total_demand_upper=solver_total_demand_upper,
                    measurement_nodes=measurement_nodes,
                    measurement_heads_equal_only=measurement_heads_equal_only_local,
                    match_reservoir_outflow_between_pairs=match_total_between_pairs_local,
                    headloss_n=headloss_n,
                    cycle_basis_mode="planar",
                    fixed_demands_b=fixed_b,
                    reference_demands=reference_demands_local,
                    restriction_mode=None,
                    radius_to_fixed=None,
                    deviation_alpha=None,
                    license_path=str(HEXALY_LICENSE_PATH),
                    time_limit=int(HEXALY_TIME_LIMIT),
                    seed=hexaly_seed,
                    verbosity=int(HEXALY_VERBOSITY),
                )
            else:
                result = solve_max_demand_distance_hexaly(
                    network=network,
                    sensor_heads=sensor_heads,
                    pipe_resistances=pipe_res,
                    c_bounds=None,
                    C_bounds=None,
                    reservoir_node=reservoir_node,
                    reservoir_outflow=None,
                    initial_guess=run_guess,
                    norm_p=float(NORM),
                    demand_lb=float(DEMAND_LB),
                    demand_lb_per_node=demand_lb_per_node,
                    total_demand=solver_total_demand,
                    total_demand_upper=solver_total_demand_upper,
                    measurement_nodes=measurement_nodes,
                    measurement_heads_equal_only=measurement_heads_equal_only_local,
                    linearized_pipes=linearized_pipes if (LINEARIZATION_ENABLED and mode == "W_d") else None,
                    objective_mode="bregman_energy" if mode == "B" else "demand_distance",
                    match_reservoir_outflow_between_pairs=match_total_between_pairs_local,
                    reference_demands=reference_demands_local,
                    restriction_mode=None,
                    radius_to_fixed=None,
                    deviation_alpha=None,
                    license_path=str(HEXALY_LICENSE_PATH),
                    time_limit=int(HEXALY_TIME_LIMIT),
                    seed=hexaly_seed,
                    verbosity=int(HEXALY_VERBOSITY),
                )

            center_result = None
            if mode == "C_h_fixed":
                center_result = solve_head_center_in_class_hexaly(
                    network=network,
                    sensor_heads=sensor_heads,
                    pipe_resistances=pipe_res,
                    reference_heads_a=result.heads_a,
                    reference_heads_b=result.heads_b,
                    c_bounds=None,
                    C_bounds=None,
                    reservoir_node=reservoir_node,
                    reservoir_outflow=None,
                    initial_guess=run_guess,
                    norm_p=float(NORM),
                    demand_lb=float(DEMAND_LB),
                    demand_lb_per_node=demand_lb_per_node,
                    total_demand=solver_total_demand,
                    license_path=str(HEXALY_LICENSE_PATH),
                    time_limit=int(HEXALY_TIME_LIMIT),
                    seed=hexaly_seed,
                    verbosity=int(HEXALY_VERBOSITY),
                )
                mode_value = float(center_result.objective) if center_result.objective is not None else float("inf")
            else:
                mode_value = _center_mode_value(
                    mode=mode,
                    norm_p=float(NORM),
                    result=result,
                    junctions=junctions,
                    all_nodes=all_nodes,
                    reference_demands=reference_demands_local,
                    reference_heads=reference_heads_local,
                )
            runs.append(
                {
                    "run": run_idx,
                    "mode_value": mode_value,
                    "success": bool(result.success),
                    "max_violation": float(result.max_violation),
                    "min_demand_viol": float(result.min_demand_viol),
                    "center_success": bool(center_result.success) if center_result is not None else None,
                }
            )
            run_valid = bool(result.success) and float(result.max_violation) <= 1e-5 and float(result.min_demand_viol) >= -1e-5
            if run_valid:
                valid_mode_values.append(float(mode_value))
            if (
                best_result is None
                or (not minimize_mode and mode_value > best_value)
                or (minimize_mode and mode_value < best_value)
            ):
                best_value = mode_value
                best_result = result
                best_center_result = center_result

            if dms_enabled and run_valid:
                # For minimization in local search: if one run is already >= current best radius,
                # this configuration is certified as no-improvement.
                if float(mode_value) >= dms_reference_radius:
                    dms_lower_bound = float(mode_value)
                    dms_upper_bound = -float("inf")
                    dms_certificate = "no-improvement"
                    dms_finished = True
                else:
                    sorted_vals = sorted(valid_mode_values, reverse=True)
                    if len(sorted_vals) >= dms_consistency and sorted_vals[dms_consistency - 1] >= dms_deviation * sorted_vals[0]:
                        dms_lower_bound = float(sorted_vals[0])
                        dms_upper_bound = float(sorted_vals[0])
                        dms_certificate = "improvement"
                        dms_finished = True

            run_idx += 1
            if dms_enabled:
                if dms_finished:
                    break
                if run_idx >= dms_max_starts:
                    if valid_mode_values:
                        dms_lower_bound = float(max(valid_mode_values))
                    else:
                        dms_lower_bound = -float("inf")
                    dms_upper_bound = float("inf")
                    dms_certificate = "inconclusive"
                    break
            else:
                if run_idx >= max(1, int(MULTI_STARTS)):
                    break

        if best_result is None:
            raise RuntimeError("No solver result produced.")

        demand_w = _mode_value("W_d", float(NORM), best_result, junctions, all_nodes, reference_demands_local, reference_heads_local)
        demand_c = _center_mode_value("C_d", float(NORM), best_result, junctions, all_nodes, reference_demands_local, reference_heads_local)
        head_w = _mode_value("W_h", float(NORM), best_result, junctions, all_nodes, reference_demands_local, reference_heads_local)
        head_c = _center_mode_value("C_h", float(NORM), best_result, junctions, all_nodes, reference_demands_local, reference_heads_local)
        b_score = _mode_value("B", float(NORM), best_result, junctions, all_nodes, reference_demands_local, reference_heads_local)

        out_demands_a = dict(best_result.demands_a)
        out_demands_b = dict(best_result.demands_b)
        out_heads_a = dict(best_result.heads_a)
        out_heads_b = dict(best_result.heads_b)
        out_flows_a = dict(best_result.flows_a)
        out_flows_b = dict(best_result.flows_b)
        center_side = None
        center_symbol = None
        if mode == "C_h_fixed" and best_center_result is not None and best_center_result.heads_a:
            center_heads = dict(best_center_result.heads_a)
            center_demands = dict(best_center_result.demands_a)
            center_flows = dict(best_center_result.flows_a)
            dist_a = _compute_norm((best_result.heads_a.get(n, 0.0) - center_heads.get(n, 0.0) for n in all_nodes), float(NORM))
            dist_b = _compute_norm((best_result.heads_b.get(n, 0.0) - center_heads.get(n, 0.0) for n in all_nodes), float(NORM))
            choose_a = dist_a >= dist_b
            if choose_a:
                out_demands_a = dict(best_result.demands_a)
                out_heads_a = dict(best_result.heads_a)
                out_flows_a = dict(best_result.flows_a)
            else:
                out_demands_a = dict(best_result.demands_b)
                out_heads_a = dict(best_result.heads_b)
                out_flows_a = dict(best_result.flows_b)
            out_demands_b = center_demands
            out_heads_b = center_heads
            out_flows_b = center_flows
            center_side = "green"
            center_symbol = "g"
            best_value = float(best_center_result.objective or 0.0)
            head_c = best_value
        elif mode in {"C_h", "C_h_fixed"} and center_state is not None and center_state.get("heads"):
            ref_heads = center_state.get("heads", {})
            dist_a = _compute_norm((best_result.heads_a.get(n, 0.0) - ref_heads.get(n, 0.0) for n in all_nodes), float(NORM))
            dist_b = _compute_norm((best_result.heads_b.get(n, 0.0) - ref_heads.get(n, 0.0) for n in all_nodes), float(NORM))
            choose_a = dist_a >= dist_b
            if choose_a:
                out_demands_a = dict(best_result.demands_a)
                out_heads_a = dict(best_result.heads_a)
                out_flows_a = dict(best_result.flows_a)
            else:
                out_demands_a = dict(best_result.demands_b)
                out_heads_a = dict(best_result.heads_b)
                out_flows_a = dict(best_result.flows_b)
            out_demands_b = dict(center_state.get("demands", out_demands_b))
            out_heads_b = dict(center_state.get("heads", out_heads_b))
            out_flows_b = dict(center_state.get("flows", out_flows_b))
            center_side = "green"
            center_symbol = "g"
            best_value = head_c

        if mode == "C_d" and center_state is not None and center_state.get("demands"):
            ref_demands = center_state.get("demands", {})
            dist_a = _compute_norm((best_result.demands_a.get(j, 0.0) - ref_demands.get(j, 0.0) for j in junctions), float(NORM))
            dist_b = _compute_norm((best_result.demands_b.get(j, 0.0) - ref_demands.get(j, 0.0) for j in junctions), float(NORM))
            choose_a = dist_a >= dist_b
            if choose_a:
                out_demands_a = dict(best_result.demands_a)
                out_heads_a = dict(best_result.heads_a)
                out_flows_a = dict(best_result.flows_a)
            else:
                out_demands_a = dict(best_result.demands_b)
                out_heads_a = dict(best_result.heads_b)
                out_flows_a = dict(best_result.flows_b)
            out_demands_b = dict(center_state.get("demands", out_demands_b))
            out_heads_b = dict(center_state.get("heads", out_heads_b))
            out_flows_b = dict(center_state.get("flows", out_flows_b))
            center_side = "green"
            center_symbol = "c"
            best_value = demand_c

        _write_json(
            os.path.join(output_dir, "demand_distance.json"),
            {
                "demands_a": out_demands_a,
                "demands_b": out_demands_b,
                "heads_a": out_heads_a,
                "heads_b": out_heads_b,
                "flows_a": out_flows_a,
                "flows_b": out_flows_b,
                "radius": best_value,
                "mode": mode_raw,
                "mode_effective": mode,
                "objective_mode": "bregman_energy" if mode == "B" else "demand_distance",
                "method": method,
                "W_d": demand_w,
                "C_d": demand_c,
                "W_h": head_w,
                "C_h": head_c,
                "B": b_score,
                "norm": float(NORM),
                "p": len(measurement_nodes),
                "success": bool(best_result.success),
                "max_violation": float(best_result.max_violation),
                "min_demand_viol": float(best_result.min_demand_viol),
                "objective": best_result.objective,
                "solver_status": best_result.solver_status,
                "best_bound": best_result.best_bound,
                "runs": runs,
                "dms_enabled": bool(dms_enabled),
                "dms_consistency": int(dms_consistency),
                "dms_deviation": float(dms_deviation),
                "dms_max_starts": int(dms_max_starts),
                "dms_discard_unclear": bool(DMS_DISCARD_UNCLEAR),
                "dms_reference_radius": float(dms_reference_radius),
                "dms_certificate": dms_certificate,
                "dms_lower_bound": float(dms_lower_bound),
                "dms_upper_bound": float(dms_upper_bound),
                "measurement_source": MEASUREMENT_SOURCE,
                "measurement_data": measurement_data,
                "linearization_enabled": bool(LINEARIZATION_ENABLED),
                "linearized_pipes": linearized_pipes,
                "reference_demands": reference_demands_local,
                "center_side": center_side,
                "center_symbol": center_symbol,
            },
        )

        measurement_candidates = []
        if mode in {"W_d", "W_h", "B"}:
            measurement_candidates = [
                {
                    "label": "a",
                    "data": _build_measurement_dict(measurement_nodes, best_result.heads_a, total_demand_local, reservoir_node, reservoir_head_local),
                    "state": {
                        "demands": best_result.demands_a,
                        "heads": best_result.heads_a,
                        "flows": best_result.flows_a,
                    },
                },
                {
                    "label": "b",
                    "data": _build_measurement_dict(measurement_nodes, best_result.heads_b, total_demand_local, reservoir_node, reservoir_head_local),
                    "state": {
                        "demands": best_result.demands_b,
                        "heads": best_result.heads_b,
                        "flows": best_result.flows_b,
                    },
                },
            ]
        _write_json(
            os.path.join(output_dir, "measurement_candidates.json"),
            {
                "mode": mode,
                "measurement_source": MEASUREMENT_SOURCE,
                "sites": list(measurement_nodes),
                "used_measurement": measurement_data,
                "candidates": measurement_candidates,
            },
        )
        _write_json(
            os.path.join(output_dir, "demand_distance_multistart.json"),
            {
                "runs": runs,
                "best_mode_value": best_value,
                "mode": mode,
                "method": method,
            },
        )

    _register_output_dir(SOLVER_HASH, OUTPUT_DIR, WDN)


if __name__ == "__main__":
    main()
