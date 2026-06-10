#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
	 sys.path.insert(0, ROOT_DIR)

from gui.cache import compute_hash, load_index, save_index
from gui.state import SolverParams
from step1_io import load_inp_network


CACHE_DIR = ".gui_cache"
LEGACY_DATA_INDEX = os.path.join("data", "cache_index.json")
DEFAULT_WDNS = ["Baghmalek", "Anytown", "Hanoi", "Modena"]
WDN_ALIASES = {
	"baghmalek": "Baghmalek",
	"anytown": "Anytown",
	"anytwon": "Anytown",
	"hanoi": "Hanoi",
	"modena": "Modena",
}


def _write_json(path: str, payload: Dict[str, object]) -> None:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w", encoding="utf-8") as handle:
		json.dump(payload, handle, indent=2, sort_keys=True)


def _read_json(path: str) -> Dict[str, object]:
	with open(path, "r", encoding="utf-8") as handle:
		return json.load(handle)


def _data_index_path(wdn: str) -> str:
	return os.path.join("data", wdn, "cache_index.json")


def _load_index_with_legacy(path: str, legacy_path: str) -> Dict[str, str]:
	index = load_index(path, ROOT_DIR)
	if index:
		return index
	return load_index(legacy_path, ROOT_DIR)


def _write_gui_hash(output_dir: str, solver_hash: str) -> None:
	try:
		hash_file = os.path.join(output_dir, "_gui_hash.txt")
		if not os.path.isfile(hash_file):
			with open(hash_file, "w", encoding="utf-8") as handle:
				handle.write(solver_hash)
	except OSError:
		pass


def _solver_cache_hash_payload(payload: Dict[str, object]) -> Dict[str, object]:
	return {
		str(key): value
		for key, value in payload.items()
		if isinstance(key, str)
		and key.isupper()
		and key not in {"OUTPUT_DIR", "SOLVER_HASH", "_index", "_index_path", "DMS_RADIUS"}
	}


def _float_or_default(value: object, default: float) -> float:
	try:
		if isinstance(value, str):
			text = value.strip().lower()
			if text in {"", "inf", "+inf", "infinity", "+infinity"}:
				return float("inf")
			if text in {"-inf", "-infinity"}:
				return -float("inf")
		return float(value)
	except (TypeError, ValueError):
		return default


def _dms_cache_conclusive(dd: Dict[str, object], reference_radius: float) -> tuple[bool, float, float, str]:
	lower = _float_or_default(dd.get("dms_lower_bound"), -float("inf"))
	upper = _float_or_default(dd.get("dms_upper_bound"), float("inf"))
	cert = str(dd.get("dms_certificate", ""))
	conclusive = (reference_radius < lower) or (reference_radius > upper)
	return conclusive, lower, upper, cert


def _read_wdn_config(wdn: str) -> Dict[str, object]:
	path = os.path.join(ROOT_DIR, "wdn", f"{wdn}.json")
	if not os.path.isfile(path):
		raise FileNotFoundError(f"Missing WDN config: {path}")
	return _read_json(path)


def _default_measurement_nodes(wdn: str) -> List[str]:
	config = _read_wdn_config(wdn)
	nodes = [str(node).strip() for node in config.get("measurement_nodes", []) if str(node).strip()]
	if not nodes:
		raise ValueError(f"No default measurement_nodes configured in wdn/{wdn}.json")
	return list(dict.fromkeys(nodes))


def _default_extra_demand(wdn: str, fallback: float) -> float:
	config = _read_wdn_config(wdn)
	try:
		return float(config.get("extra_demand", fallback))
	except (TypeError, ValueError):
		return fallback


def _normalize_wdn_name(name: str) -> str:
	key = str(name).strip().lower()
	if key in WDN_ALIASES:
		return WDN_ALIASES[key]
	raise ValueError(f"Unsupported WDN '{name}'. Expected one of: {', '.join(DEFAULT_WDNS)}")


def _build_payload_template(wdn: str) -> Dict[str, object]:
	defaults = SolverParams()
	extra_demand = _default_extra_demand(wdn, defaults.extra_demand)
	payload = asdict(defaults)
	payload.update(
		{
			"WDN": wdn,
			"MODE": defaults.mode,
			"MEASUREMENT_SITES": [],
			"MEASUREMENT_SOURCE": "base",
			"MEASUREMENT_DATA": None,
			"METHOD": defaults.method,
			"NORM": defaults.norm,
			"DEMAND_LB": defaults.demand_lb,
			"EXTRA_DEMAND": extra_demand,
			"MEASUREMENT_HEADS_EQUAL_ONLY": defaults.measurement_heads_equal_only,
			"MATCH_TOTAL_DEMAND": defaults.match_total_demand,
			"MATCH_RESERVOIR_OUTFLOW_BETWEEN_PAIRS": defaults.match_total_demand,
			"DYNAMIC_MULTISTART": True,
			"DMS_CONSISTENCY": defaults.dms_consistency,
			"DMS_DEVIATION": defaults.dms_deviation,
			"DMS_RADIUS": float("inf"),
			"DMS_MAX_STARTS": defaults.dms_max_starts,
			"DMS_DISCARD_UNCLEAR": defaults.dms_discard_unclear,
			"MULTI_STARTS": defaults.multi_starts,
			"MULTI_START_NOISE": defaults.multi_start_noise,
			"MULTI_START_NOISE_REL": defaults.multi_start_noise_rel,
			"HEXALY_TIME_LIMIT": defaults.hexaly_time_limit,
		}
	)
	return payload


def _build_junction_adjacency(wdn: str) -> Dict[str, List[str]]:
	network = load_inp_network(os.path.join(ROOT_DIR, "wdn", f"{wdn}.inp"))
	junction_ids = set(str(node_id) for node_id in network.junctions.keys())
	adjacency: Dict[str, List[str]] = {node_id: [] for node_id in junction_ids}
	for pipe in network.pipes.values():
		start_node = str(pipe.start_node)
		end_node = str(pipe.end_node)
		if start_node in junction_ids and end_node in junction_ids:
			adjacency[start_node].append(end_node)
			adjacency[end_node].append(start_node)
	return adjacency


class HeadlessLocalSearch:
	def __init__(
		self,
		wdn: str,
		start_nodes: List[str],
		adjacency: Dict[str, List[str]],
		payload_template: Dict[str, object],
		index_path: str,
		existing_index: Dict[str, str],
		*,
		verbose: bool = True,
	) -> None:
		self._wdn = wdn
		self._start_nodes = list(start_nodes)
		self._adjacency = adjacency
		self._payload_template = dict(payload_template)
		self._index_path = index_path
		self._index: Dict[str, str] = dict(existing_index)
		self._rows: List[Dict[str, str]] = []
		self._lookup: Dict[frozenset[str], float] = {}
		self._cancelled = False
		self._last_eval_unclear = False
		self._last_eval_starts = 0
		self._last_eval_certificate = ""
		self._last_eval_radius = float("inf")
		self._evaluation_count = 0
		self._improvement_count = 0
		self._discarded_count = 0
		self._logs: List[str] = []
		self._verbose = verbose

	def _log(self, message: str) -> None:
		line = f"[{self._wdn}] {message}"
		self._logs.append(line)
		if self._verbose:
			print(line, flush=True)

	def run(self) -> Dict[str, object]:
		current: frozenset[str] = frozenset(self._start_nodes)
		dms_enabled = bool(self._payload_template.get("DYNAMIC_MULTISTART", False))
		discard_unclear = bool(self._payload_template.get("DMS_DISCARD_UNCLEAR", True))
		final_current: frozenset[str] = current
		final_radius = float("inf")

		while not self._cancelled:
			if current not in self._lookup:
				radius = self._evaluate(current, float("inf") if dms_enabled else None)
				if radius is None:
					if dms_enabled and self._last_eval_unclear:
						if discard_unclear:
							self._log(f"> results inconclusive after {self._last_eval_starts} starts")
							self._discarded_count += 1
						else:
							self._log(
								f"Unclear configuration {sorted(current)} encountered; stopping because discard unclear is disabled."
							)
					else:
						self._log(f"Solver failed for {sorted(current)}, stopping.")
					break
				self._lookup[current] = radius

			current_radius = self._lookup[current]
			final_current = current
			final_radius = current_radius

			candidates = self._generate_neighbors(current)
			best: frozenset[str] | None = None
			best_radius = current_radius
			abort_search = False
			for candidate in candidates:
				if self._cancelled:
					break
				if candidate not in self._lookup:
					radius = self._evaluate(candidate, current_radius if dms_enabled else None)
					if radius is None:
						if dms_enabled and self._last_eval_unclear:
							if discard_unclear:
								self._log(f"> results inconclusive after {self._last_eval_starts} starts")
								self._discarded_count += 1
								continue
							self._log(
								f"Unclear configuration {sorted(candidate)} encountered; stopping because discard unclear is disabled."
							)
							abort_search = True
							break
						continue
					self._lookup[candidate] = radius

				radius_candidate = self._lookup[candidate]
				if radius_candidate < best_radius:
					best_radius = radius_candidate
					best = candidate

			if abort_search:
				break

			if best is None or self._cancelled:
				break

			self._improvement_count += 1
			current = best
			final_current = current
			final_radius = best_radius

		self._log("Done.")
		if not self._cancelled:
			self._log(f"> Local optimum found after {self._evaluation_count} evaluations")
		else:
			self._log(f"> Search cancelled after {self._evaluation_count} evaluations")
		self._log(f"> {self._improvement_count} improvements")
		self._log(f"> {self._discarded_count} discarded")
		if final_current in self._lookup:
			final_radius = self._lookup[final_current]
		self._log(f"> Optimum is {sorted(final_current)} with radius {final_radius:.6f}")

		return {
			"wdn": self._wdn,
			"start_nodes": sorted(self._start_nodes),
			"optimum_nodes": sorted(final_current),
			"optimum_radius": final_radius,
			"evaluations": self._evaluation_count,
			"improvements": self._improvement_count,
			"discarded": self._discarded_count,
			"rows": self._rows,
			"logs": self._logs,
		}

	def _generate_neighbors(self, current: frozenset[str]) -> List[frozenset[str]]:
		seen: set[frozenset[str]] = set()
		candidates: List[frozenset[str]] = []
		for node in current:
			for neighbor in self._adjacency.get(str(node), []):
				if neighbor not in current:
					candidate = frozenset((current - {node}) | {neighbor})
					if candidate not in seen:
						seen.add(candidate)
						candidates.append(candidate)
		return candidates

	def _evaluate(self, config: frozenset[str], dms_reference_radius: float | None = None) -> float | None:
		self._last_eval_unclear = False
		self._last_eval_starts = 0
		self._last_eval_certificate = ""
		self._last_eval_radius = float("inf")
		self._evaluation_count += 1
		nodes = sorted(config)
		self._log(f"Evaluating {nodes}...")

		payload = dict(self._payload_template)
		payload["MEASUREMENT_SITES"] = nodes
		expected_dms = bool(payload.get("DYNAMIC_MULTISTART", False))
		if expected_dms and dms_reference_radius is not None:
			payload["DMS_RADIUS"] = float(dms_reference_radius)
		expected_mode = str(payload.get("MODE", ""))
		expected_method = str(payload.get("METHOD", ""))
		expected_source = str(payload.get("MEASUREMENT_SOURCE", ""))
		solver_hash = compute_hash(_solver_cache_hash_payload(payload))

		cached_dir = self._index.get(solver_hash)
		if cached_dir:
			resolved_dir = cached_dir if os.path.isabs(cached_dir) else os.path.join(ROOT_DIR, cached_dir)
			if os.path.isdir(resolved_dir):
				radius = self._read_radius(
					resolved_dir,
					nodes,
					expected_mode,
					expected_method,
					expected_source,
					expected_dms,
					dms_reference_radius,
					cached=True,
				)
				if radius is not None:
					self._emit_dms_eval_summary(expected_dms, dms_reference_radius, radius)
					return radius
				if not expected_dms:
					self._index.pop(solver_hash, None)
					save_index(self._index_path, self._index, ROOT_DIR)

		output_dir = os.path.join("data", self._wdn, solver_hash)
		payload["OUTPUT_DIR"] = output_dir
		payload["SOLVER_HASH"] = solver_hash
		os.makedirs(CACHE_DIR, exist_ok=True)
		config_path = os.path.join(CACHE_DIR, f"solver-{solver_hash}.json")
		_write_json(config_path, payload)

		proc = subprocess.run(
			[sys.executable, os.path.join(ROOT_DIR, "inverse.py"), "--config", config_path],
			capture_output=True,
			text=True,
			cwd=ROOT_DIR,
		)
		resolved_output = output_dir if os.path.isabs(output_dir) else os.path.join(ROOT_DIR, output_dir)
		if proc.returncode != 0 or not os.path.isdir(resolved_output):
			error_text = (proc.stderr or "").strip()
			stdout_text = (proc.stdout or "").strip()
			detail = error_text if error_text else stdout_text
			if len(detail) > 1200:
				detail = detail[-1200:]
			self._log(f"Solver failed for {nodes}: {detail or 'no output captured'}")
			return None

		radius = self._read_radius(
			resolved_output,
			nodes,
			expected_mode,
			expected_method,
			expected_source,
			expected_dms,
			dms_reference_radius,
			cached=False,
		)
		if radius is None:
			return None

		self._emit_dms_eval_summary(expected_dms, dms_reference_radius, radius)
		self._index[solver_hash] = output_dir
		save_index(self._index_path, self._index, ROOT_DIR)
		_write_gui_hash(resolved_output, solver_hash)
		return radius

	def _emit_dms_eval_summary(self, expected_dms: bool, dms_reference_radius: float | None, radius: float) -> None:
		if not expected_dms:
			return
		starts = max(1, int(self._last_eval_starts))
		cert = self._last_eval_certificate
		if cert == "improvement":
			self._log(f"> Improvement found after {starts} starts")
			if dms_reference_radius is not None and math.isfinite(float(dms_reference_radius)):
				self._log(f"> Radius: {float(dms_reference_radius):.6f} -> {float(radius):.6f}")
		elif cert == "no-improvement":
			self._log(f"> Improvement ruled out after {starts} starts")
		elif cert == "inconclusive":
			self._log(f"> results inconclusive after {starts} starts")

	def _read_radius(
		self,
		output_dir: str,
		nodes: List[str],
		expected_mode: str,
		expected_method: str,
		expected_source: str,
		expected_dms: bool,
		dms_reference_radius: float | None,
		*,
		cached: bool,
	) -> float | None:
		expected_nodes = sorted(str(node) for node in nodes)
		candidate_paths: List[str] = []
		direct_path = os.path.join(output_dir, "demand_distance.json")
		if os.path.isfile(direct_path):
			candidate_paths.append(direct_path)
		for root, _dirs, files in os.walk(output_dir):
			if "demand_distance.json" in files:
				path = os.path.join(root, "demand_distance.json")
				if path not in candidate_paths:
					candidate_paths.append(path)
		if not candidate_paths:
			return None

		dd = None
		for dd_path in candidate_paths:
			try:
				dd_candidate = _read_json(dd_path)
			except Exception:
				continue
			params_path = os.path.join(os.path.dirname(dd_path), "parameters.json")
			params: Dict[str, object] = {}
			if os.path.isfile(params_path):
				try:
					params = _read_json(params_path)
				except Exception:
					params = {}
			nodes_params = params.get("MEASUREMENT_NODES")
			if isinstance(nodes_params, list) and sorted(str(x) for x in nodes_params) != expected_nodes:
				continue
			source_candidate = str(params.get("MEASUREMENT_SOURCE", ""))
			if expected_source and source_candidate and source_candidate != expected_source:
				continue
			mode_candidate = str(dd_candidate.get("mode", ""))
			method_candidate = str(dd_candidate.get("method", ""))
			if expected_mode and mode_candidate and mode_candidate != expected_mode:
				continue
			if expected_method and method_candidate and method_candidate != expected_method:
				continue
			dd = dd_candidate
			break
		if dd is None:
			return None

		success = bool(dd.get("success", False))
		max_violation = float(dd.get("max_violation", float("inf")))
		min_demand_viol = float(dd.get("min_demand_viol", -float("inf")))
		solver_status = str(dd.get("solver_status", ""))
		valid = success and max_violation <= 1e-5 and min_demand_viol >= -1e-5
		if not valid:
			self._log(
				f"Ignoring {'cached ' if cached else ''}result for {nodes}: "
				f"success={success}, max_violation={max_violation:.3e}, "
				f"min_demand_viol={min_demand_viol:.3e}, status={solver_status}"
			)
			return None

		if expected_dms and dms_reference_radius is not None:
			conclusive, _lower, _upper, cert = _dms_cache_conclusive(dd, float(dms_reference_radius))
			runs_raw = dd.get("runs")
			runs_count = len(runs_raw) if isinstance(runs_raw, list) else 0
			self._last_eval_starts = runs_count
			self._last_eval_certificate = cert
			if not conclusive:
				self._last_eval_unclear = True
				return None

		radius = float(dd.get("radius", float("inf")))
		self._last_eval_radius = radius
		row: Dict[str, str] = {
			"measurement_sites": str(nodes),
			"radius": f"{radius:.6f}",
			"cached": str(cached),
			"success": str(success),
			"max_violation": f"{max_violation:.6e}",
			"min_demand_viol": f"{min_demand_viol:.6e}",
			"solver_status": solver_status,
			"dms_certificate": str(dd.get("dms_certificate", "")),
			"dms_lower_bound": str(dd.get("dms_lower_bound", "")),
			"dms_upper_bound": str(dd.get("dms_upper_bound", "")),
			"output_dir": output_dir,
		}
		self._rows.append(row)
		return radius


def _default_summary_path() -> str:
	timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
	return os.path.join(ROOT_DIR, "data", f"local-search-defaults-dms-{timestamp}.json")


def main() -> int:
	parser = argparse.ArgumentParser(
		description=(
			"Run headless local search with GUI-default solver settings and dynamic multistart enabled. "
			"Starting nodes come from wdn/<name>.json measurement_nodes."
		)
	)
	parser.add_argument(
		"wdns",
		nargs="*",
		help="Networks to run. Defaults to Baghmalek, Anytown, Hanoi, Modena.",
	)
	parser.add_argument(
		"--summary-json",
		default=_default_summary_path(),
		help="Path to write a JSON summary of all runs.",
	)
	parser.add_argument(
		"--quiet",
		action="store_true",
		help="Suppress live log printing and only write the summary JSON.",
	)
	args = parser.parse_args()

	selected = args.wdns or DEFAULT_WDNS
	wdns = [_normalize_wdn_name(name) for name in selected]

	summary: List[Dict[str, object]] = []
	for wdn in wdns:
		start_nodes = _default_measurement_nodes(wdn)
		adjacency = _build_junction_adjacency(wdn)
		index_path = _data_index_path(wdn)
		existing_index = _load_index_with_legacy(index_path, LEGACY_DATA_INDEX)
		payload_template = _build_payload_template(wdn)
		search = HeadlessLocalSearch(
			wdn,
			start_nodes,
			adjacency,
			payload_template,
			index_path,
			existing_index,
			verbose=not args.quiet,
		)
		summary.append(search.run())

	_write_json(args.summary_json, {"runs": summary})
	if not args.quiet:
		print(f"Summary written to {args.summary_json}", flush=True)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())