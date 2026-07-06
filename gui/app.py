import json
import os
import re
import hashlib
import math
from datetime import datetime
import runpy
import traceback
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict
from typing import Dict, List, Optional

from PyQt6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backend_bases import MouseButton

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
	sys.path.insert(0, ROOT_DIR)

from gui.cache import compute_hash, load_index, save_index
from gui.state import SolverParams
from step1_io import load_inp_network


CACHE_DIR = ".gui_cache"
LEGACY_DATA_INDEX = os.path.join("data", "cache_index.json")


def _write_json(path: str, payload: Dict[str, object]) -> None:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2, sort_keys=True)


def _read_json(path: str) -> Dict[str, object]:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def _data_index_path(wdn: str) -> str:
	return os.path.join("data", wdn, "cache_index.json")


def _write_gui_hash(output_dir: str, solver_hash: str) -> None:
	"""Write the GUI hash into the result directory so the index can be rebuilt later."""
	try:
		hash_file = os.path.join(output_dir, "_gui_hash.txt")
		if not os.path.isfile(hash_file):
			with open(hash_file, "w", encoding="utf-8") as f:
				f.write(solver_hash)
	except OSError:
		pass


def _sync_wdn_index(wdn: str) -> Dict[str, str]:
	"""Scan data/<wdn>/ for _gui_hash.txt files, backfill missing ones, and rebuild the index.

	Returns the (possibly updated) index dict.
	"""
	index_path = _data_index_path(wdn)
	index = load_index(index_path, ROOT_DIR)

	data_dir = os.path.join(ROOT_DIR, "data", wdn)
	if not os.path.isdir(data_dir):
		return index

	changed = False

	# Backfill: write _gui_hash.txt for entries already in the index
	for hash_key, rel_path in list(index.items()):
		resolved = rel_path if os.path.isabs(rel_path) else os.path.join(ROOT_DIR, rel_path)
		if os.path.isdir(resolved):
			hash_file = os.path.join(resolved, "_gui_hash.txt")
			if not os.path.isfile(hash_file):
				try:
					with open(hash_file, "w", encoding="utf-8") as f:
						f.write(hash_key)
				except OSError:
					pass

	# Forward-fill: scan subdirs for _gui_hash.txt and add to index if missing
	try:
		entries = list(os.scandir(data_dir))
	except OSError:
		entries = []
	for entry in entries:
		if not entry.is_dir():
			continue
		hash_file = os.path.join(entry.path, "_gui_hash.txt")
		if not os.path.isfile(hash_file):
			continue
		try:
			hash_key = open(hash_file, encoding="utf-8").read().strip()
		except OSError:
			continue
		if not hash_key:
			continue
		expected_rel = os.path.join("data", wdn, entry.name)
		if index.get(hash_key) != expected_rel:
			index[hash_key] = expected_rel
			changed = True

	if changed:
		save_index(index_path, index, ROOT_DIR)

	return index


def _build_demand_distance_plot_fn(
	data_dir: str,
	temp_dir: str,
	index: int,
	name_prefix: str = "",
	show_only_changed_demands: bool = False,
) -> "tuple[str, str, float, int] | None":
	"""Module-level helper so PlotWorker can call it from a background thread."""
	from image import plot_demand_distance

	params_path = os.path.join(data_dir, "parameters.json")
	dd_path = os.path.join(data_dir, "demand_distance.json")
	if not (os.path.isfile(params_path) and os.path.isfile(dd_path)):
		return None

	params = _read_json(params_path)
	demand_distance = _read_json(dd_path)
	radius = float(demand_distance.get("radius", 0.0))
	wdn = str(params.get("WDN") or params.get("WDN_NAME") or "")
	if not wdn:
		return None
	inp_path = os.path.join("wdn", f"{wdn}.inp")

	bounds_path = os.path.join(data_dir, "c_bounds.json")
	if os.path.isfile(bounds_path):
		bounds_payload = _read_json(bounds_path)
		c_bounds = {k: float(v) for k, v in bounds_payload.get("c_bounds", {}).items()}
		C_bounds = {k: float(v) for k, v in bounds_payload.get("C_bounds", {}).items()}
	else:
		c_bounds = {}
		C_bounds = {}

	measurement_nodes = [str(x) for x in params.get("MEASUREMENT_NODES") or []]
	measurement_data = params.get("MEASUREMENT_DATA") if isinstance(params.get("MEASUREMENT_DATA"), dict) else {}
	measurement_total_demand = None
	if isinstance(measurement_data, dict) and "-1" in measurement_data:
		try:
			measurement_total_demand = float(measurement_data.get("-1"))
		except (TypeError, ValueError):
			measurement_total_demand = None
	solver_reference_demands = None
	if isinstance(demand_distance.get("reference_demands"), dict):
		try:
			solver_reference_demands = {
				str(k): float(v)
				for k, v in demand_distance.get("reference_demands", {}).items()
			}
		except (TypeError, ValueError):
			solver_reference_demands = None
	configured_extra_demand = None
	if "EXTRA_DEMAND" in params:
		try:
			configured_extra_demand = float(params.get("EXTRA_DEMAND"))
		except (TypeError, ValueError):
			configured_extra_demand = None
	plot_demand_distance(
		inp_path=inp_path,
		measurement_nodes=measurement_nodes,
		demands_a=demand_distance.get("demands_a", {}),
		demands_b=demand_distance.get("demands_b", {}),
		heads_a=demand_distance.get("heads_a", {}),
		heads_b=demand_distance.get("heads_b", {}),
		flows_a=demand_distance.get("flows_a", {}),
		flows_b=demand_distance.get("flows_b", {}),
		radius=radius,
		norm_p=float(demand_distance.get("norm", 2.0)),
		mode=str(demand_distance.get("mode", "W_d")),
		center_side=demand_distance.get("center_side"),
		center_symbol=demand_distance.get("center_symbol"),
		metrics={
			"W_d": float(demand_distance.get("W_d", float("nan"))),
			"C_d": float(demand_distance.get("C_d", float("nan"))),
			"W_h": float(demand_distance.get("W_h", float("nan"))),
			"C_h": float(demand_distance.get("C_h", float("nan"))),
		},
		measurement_count=int(demand_distance.get("p", len(measurement_nodes))),
		output_dir=temp_dir,
		c_bounds=c_bounds,
		C_bounds=C_bounds,
		measurement_total_demand=measurement_total_demand,
		solver_reference_demands=solver_reference_demands,
		configured_extra_demand=configured_extra_demand,
		show_only_changed_demands=show_only_changed_demands,
	)

	src_png = os.path.join(temp_dir, "demand_distance.png")
	if not os.path.isfile(src_png):
		return None
	dst_png = os.path.join(temp_dir, f"demand_distance_{index:03d}.png")
	if os.path.exists(dst_png):
		os.remove(dst_png)
	os.replace(src_png, dst_png)
	name = os.path.basename(data_dir)
	if not measurement_nodes:
		name = f"{name} (no sites)"
	if name_prefix:
		name = f"[{name_prefix}] {name}"
	return name, dst_png, radius, len(measurement_nodes)


def _load_index_with_legacy(path: str, legacy_path: str) -> Dict[str, str]:
	index = load_index(path, ROOT_DIR)
	if index:
		return index
	return load_index(legacy_path, ROOT_DIR)


def _solver_cache_hash_payload(payload: Dict[str, object]) -> Dict[str, object]:
	"""Keep only canonical solver config keys for cache hashing.

	GUI state contains lowercase helper fields (from SolverParams) that must not
	affect cache identity.
	"""
	hash_payload = {
		str(k): v
		for k, v in payload.items()
		if isinstance(k, str)
		and k.isupper()
		and k not in {"OUTPUT_DIR", "SOLVER_HASH", "_index", "_index_path", "DMS_RADIUS", "LINEARIZATION_EPS_SCALE"}
	}
	if str(hash_payload.get("MODE", "")) == "B":
		hash_payload.pop("NORM", None)
	return hash_payload


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


class SolverWorker(QtCore.QThread):
	"""Runs the solver subprocess off the GUI thread, streaming progress lines."""

	progress_updated = QtCore.pyqtSignal(int, int)   # current, total
	linearization_updated = QtCore.pyqtSignal(int, int, int)  # checked, total, certified
	finished_with_code = QtCore.pyqtSignal(int, str, str)  # returncode, stdout, stderr

	def __init__(self, cmd: List[str], parent: "QtWidgets.QWidget | None" = None) -> None:
		super().__init__(parent)
		self._cmd = cmd

	def run(self) -> None:
		proc = subprocess.Popen(
			self._cmd,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True,
		)
		stderr_lines: List[str] = []

		def _read_stderr() -> None:
			assert proc.stderr is not None
			for line in proc.stderr:
				stderr_lines.append(line)
			proc.stderr.close()

		stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
		stderr_thread.start()

		stdout_lines: List[str] = []
		assert proc.stdout is not None
		for raw_line in proc.stdout:
			line = raw_line.rstrip("\n")
			stdout_lines.append(line)
			m = re.match(r"^PROGRESS:\s*(\d+)/(\d+)", line)
			if m:
				self.progress_updated.emit(int(m.group(1)), int(m.group(2)))
			lm = re.match(r"^LINEARIZATION_PROGRESS:\s*(\d+)/(\d+)\s+certified=(\d+)", line)
			if lm:
				self.linearization_updated.emit(int(lm.group(1)), int(lm.group(2)), int(lm.group(3)))

		proc.stdout.close()
		proc.wait()
		stderr_thread.join(timeout=5.0)

		self.finished_with_code.emit(
			proc.returncode,
			"\n".join(stdout_lines),
			"".join(stderr_lines),
		)


class PlotWorker(QtCore.QThread):
	"""Builds demand-distance PNGs off the GUI thread."""

	plot_progress = QtCore.pyqtSignal(int, int)  # done_so_far, total
	plots_ready = QtCore.pyqtSignal(list, str)   # items: List[tuple[str,str,float,int]], temp_dir

	def __init__(
		self,
		plot_dir_labels: "List[tuple[str, str]]",
		temp_dir: str,
		show_only_changed_demands: bool,
		parent: "QtWidgets.QWidget | None" = None,
	) -> None:
		super().__init__(parent)
		self._plot_dir_labels = plot_dir_labels
		self._temp_dir = temp_dir
		self._show_only_changed_demands = show_only_changed_demands

	def run(self) -> None:
		items: List[tuple[str, str, float, int]] = []
		total = len(self._plot_dir_labels)
		for idx, (run_dir, label) in enumerate(self._plot_dir_labels, start=1):
			built = _build_demand_distance_plot_fn(
				run_dir,
				self._temp_dir,
				idx,
				label,
				self._show_only_changed_demands,
			)
			if built is not None:
				items.append(built)
			self.plot_progress.emit(idx, total)
		items.sort(key=lambda item: item[2])
		self.plots_ready.emit(items, self._temp_dir)


class LocalSearchWorker(QtCore.QThread):
	"""Runs node-swap local search off the GUI thread.

	Starting from *start_nodes* it tries all 1-swap neighbours (remove one
	node, add a pipe-adjacent node not already in the set) and greedily moves
	to the configuration with the greatest radius reduction.  Already-tried
	configurations are re-used from the lookup dict without re-running the solver.
	"""

	status_updated = QtCore.pyqtSignal(str)
	row_added = QtCore.pyqtSignal(dict)
	finished_signal = QtCore.pyqtSignal(list)
	highlight_updated = QtCore.pyqtSignal(list, object, object)

	def __init__(
		self,
		start_nodes: List[str],
		adjacency: Dict[str, List[str]],
		payload_template: Dict[str, object],
		wdn: str,
		index_path: str,
		existing_index: Dict[str, str],
		parent: "QtWidgets.QWidget | None" = None,
	) -> None:
		super().__init__(parent)
		self._start_nodes = start_nodes
		self._adjacency = adjacency
		self._payload_template = payload_template
		self._wdn = wdn
		self._index_path = index_path
		self._index: Dict[str, str] = dict(existing_index)
		self._rows: List[Dict[str, str]] = []
		self._lookup: Dict[frozenset, float] = {}
		self._cancelled = False
		self._last_eval_unclear = False
		self._last_eval_starts = 0
		self._last_eval_certificate = ""
		self._last_eval_radius = float("inf")
		self._evaluation_count = 0
		self._improvement_count = 0
		self._discarded_count = 0

	def cancel(self) -> None:
		self._cancelled = True

	def run(self) -> None:
		current: frozenset = frozenset(self._start_nodes)
		dms_enabled = bool(self._payload_template.get("DYNAMIC_MULTISTART", False))
		discard_unclear = bool(self._payload_template.get("DMS_DISCARD_UNCLEAR", True))
		final_current: frozenset = current
		final_radius = float("inf")
		while not self._cancelled:
			if current not in self._lookup:
				radius = self._evaluate(current, float("inf") if dms_enabled else None)
				if radius is None:
					if dms_enabled and self._last_eval_unclear:
						if discard_unclear:
							self.status_updated.emit(f"> results inconclusive after {self._last_eval_starts} starts")
							self._discarded_count += 1
						else:
							self.status_updated.emit(
								f"Unclear configuration {sorted(current)} encountered; stopping because discard unclear is disabled."
							)
					else:
						self.status_updated.emit(f"Solver failed for {sorted(current)}, stopping.")
					break
				self._lookup[current] = radius
			current_radius = self._lookup[current]
			final_current = current
			final_radius = current_radius
			self.highlight_updated.emit(sorted(current), None, None)

			candidates = self._generate_neighbors(current)
			best: "frozenset | None" = None
			best_radius = current_radius
			abort_search = False
			for candidate in candidates:
				if self._cancelled:
					break
				removed = sorted(current - candidate)
				added = sorted(candidate - current)
				swap_out = removed[0] if len(removed) == 1 else None
				swap_in = added[0] if len(added) == 1 else None
				self.highlight_updated.emit(sorted(current), swap_out, swap_in)
				if candidate not in self._lookup:
					radius = self._evaluate(candidate, current_radius if dms_enabled else None)
					if radius is None:
						if dms_enabled and self._last_eval_unclear:
							if discard_unclear:
								self.status_updated.emit(f"> results inconclusive after {self._last_eval_starts} starts")
								self._discarded_count += 1
								continue
							self.status_updated.emit(
								f"Unclear configuration {sorted(candidate)} encountered; stopping because discard unclear is disabled."
							)
							abort_search = True
							break
						continue
					self._lookup[candidate] = radius
				r = self._lookup[candidate]
				if r < best_radius:
					best_radius = r
					best = candidate

			if abort_search:
				break

			self.highlight_updated.emit(sorted(current), None, None)

			if best is None or self._cancelled:
				break
			self._improvement_count += 1
			current = best
			final_current = current
			final_radius = best_radius
			self.highlight_updated.emit(sorted(current), None, None)

		self.status_updated.emit("Done.")
		if not self._cancelled:
			self.status_updated.emit(f"> Local optimum found after {self._evaluation_count} evaluations")
		else:
			self.status_updated.emit(f"> Search cancelled after {self._evaluation_count} evaluations")
		self.status_updated.emit(f"> {self._improvement_count} improvements")
		self.status_updated.emit(f"> {self._discarded_count} discarded")
		if final_current in self._lookup:
			self.status_updated.emit(
				f"> Optimum is {sorted(final_current)} with radius {self._lookup[final_current]:.6f}"
			)
		else:
			self.status_updated.emit(
				f"> Optimum is {sorted(final_current)} with radius {final_radius:.6f}"
			)
		self.finished_signal.emit(self._rows)

	def _generate_neighbors(self, current: frozenset) -> "List[frozenset]":
		seen: "set[frozenset]" = set()
		candidates: "List[frozenset]" = []
		for node in current:
			for neighbor in self._adjacency.get(str(node), []):
				if neighbor not in current:
					candidate = frozenset((current - {node}) | {neighbor})
					if candidate not in seen:
						seen.add(candidate)
						candidates.append(candidate)
		return candidates

	def _evaluate(self, config: frozenset, dms_reference_radius: float | None = None) -> "float | None":
		self._last_eval_unclear = False
		self._last_eval_starts = 0
		self._last_eval_certificate = ""
		self._last_eval_radius = float("inf")
		self._evaluation_count += 1
		nodes = sorted(config)
		self.status_updated.emit(f"Evaluating {nodes}...")
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
				# Drop stale/invalid cache entries so future runs recompute cleanly.
				# Keep DMS entries: they can be valid but inconclusive for the current r.
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
			[sys.executable, "inverse.py", "--config", config_path],
			capture_output=True,
			text=True,
		)
		resolvedout = output_dir if os.path.isabs(output_dir) else os.path.join(ROOT_DIR, output_dir)
		if proc.returncode != 0 or not os.path.isdir(resolvedout):
			err = (proc.stderr or "").strip()
			out = (proc.stdout or "").strip()
			detail = err if err else out
			if len(detail) > 1200:
				detail = detail[-1200:]
			self.status_updated.emit(
				f"Solver failed for {nodes}: {detail or 'no output captured'}"
			)
			return None
		radius = self._read_radius(
			resolvedout,
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
		_write_gui_hash(resolvedout, solver_hash)
		return radius

	def _emit_dms_eval_summary(self, expected_dms: bool, dms_reference_radius: float | None, radius: float) -> None:
		if not expected_dms:
			return
		starts = max(1, int(self._last_eval_starts))
		cert = self._last_eval_certificate
		if cert == "improvement":
			self.status_updated.emit(f"> Improvement found after {starts} starts")
			if dms_reference_radius is not None and math.isfinite(float(dms_reference_radius)):
				self.status_updated.emit(f"> Radius: {float(dms_reference_radius):.6f} -> {float(radius):.6f}")
		elif cert == "no-improvement":
			self.status_updated.emit(f"> Improvement ruled out after {starts} starts")
		elif cert == "inconclusive":
			self.status_updated.emit(f"> results inconclusive after {starts} starts")

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
	) -> "float | None":
		expected_nodes = sorted(str(n) for n in nodes)
		candidate_paths: List[str] = []
		direct = os.path.join(output_dir, "demand_distance.json")
		if os.path.isfile(direct):
			candidate_paths.append(direct)
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
			if isinstance(nodes_params, list):
				if sorted(str(x) for x in nodes_params) != expected_nodes:
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
			self.status_updated.emit(
				f"Ignoring {'cached ' if cached else ''}result for {nodes}: "
				f"success={success}, max_violation={max_violation:.3e}, "
				f"min_demand_viol={min_demand_viol:.3e}, status={solver_status}"
			)
			return None
		if expected_dms and dms_reference_radius is not None:
			conclusive, lower, upper, cert = _dms_cache_conclusive(dd, float(dms_reference_radius))
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
		self.row_added.emit(row)
		return radius



class SolverModelWidget(QtWidgets.QGroupBox):
	"""All model-specific solver parameters in one reusable group box."""

	def __init__(self, title: str, defaults: "SolverParams", parent: "QtWidgets.QWidget | None" = None) -> None:
		super().__init__(title, parent)
		self._rows: Dict[str, tuple[QtWidgets.QLabel, QtWidgets.QWidget]] = {}
		self._solver_rows: Dict[str, tuple[QtWidgets.QLabel, QtWidgets.QWidget]] = {}
		self._form: QtWidgets.QFormLayout | None = None
		self._build_ui(defaults)

	def _dot_locale(self) -> QtCore.QLocale:
		locale = QtCore.QLocale.c()
		locale.setNumberOptions(QtCore.QLocale.NumberOption.RejectGroupSeparator)
		return locale

	def _new_double_spin(self, minimum: float, maximum: float, value: float, decimals: int = 6, step: float = 0.01) -> QtWidgets.QDoubleSpinBox:
		spin = QtWidgets.QDoubleSpinBox()
		spin.setLocale(self._dot_locale())
		spin.setDecimals(decimals)
		spin.setRange(minimum, maximum)
		spin.setSingleStep(step)
		spin.setValue(value)
		return spin

	def _add_row(self, form: QtWidgets.QFormLayout, key: str, label: str, widget: QtWidgets.QWidget) -> None:
		row_label = QtWidgets.QLabel(label)
		form.addRow(row_label, widget)
		self._rows[key] = (row_label, widget)

	def _set_row_visible(self, key: str, visible: bool) -> None:
		row = self._rows.get(key)
		if not row:
			return
		lbl, w = row
		if self._form is not None:
			self._form.setRowVisible(lbl, visible)
		else:
			lbl.setVisible(visible)
			w.setVisible(visible)

	def _add_solver_row(self, form: QtWidgets.QFormLayout, key: str, label: str, widget: QtWidgets.QWidget) -> None:
		row_label = QtWidgets.QLabel(label)
		form.addRow(row_label, widget)
		self._solver_rows[key] = (row_label, widget)

	def _set_solver_row_visible(self, key: str, visible: bool) -> None:
		row = self._solver_rows.get(key)
		if not row:
			return
		lbl, w = row
		lbl.setVisible(visible)
		w.setVisible(visible)

	def _wdn_extra_demand_default(self) -> float | None:
		win = self.window()
		wdn_input = getattr(win, "wdn_input", None)
		wdn = str(wdn_input.currentText()).strip() if wdn_input is not None else ""
		if not wdn:
			return None
		json_path = os.path.join(ROOT_DIR, "wdn", f"{wdn}.json")
		if not os.path.isfile(json_path):
			return None
		try:
			with open(json_path, encoding="utf-8") as f:
				cfg = json.load(f)
			if "extra_demand" not in cfg:
				return None
			return float(cfg.get("extra_demand"))
		except (OSError, TypeError, ValueError, json.JSONDecodeError):
			return None

	def _update_extra_demand_state(self) -> None:
		use_default = bool(self.extra_demand_use_default.isChecked()) if hasattr(self, "extra_demand_use_default") else False
		self.extra_demand.setEnabled(not use_default)
		if use_default:
			default_value = self._wdn_extra_demand_default()
			if default_value is not None:
				self.extra_demand.blockSignals(True)
				self.extra_demand.setValue(default_value)
				self.extra_demand.blockSignals(False)

	def refresh_wdn_dependent_defaults(self) -> None:
		if hasattr(self, "extra_demand_use_default") and self.extra_demand_use_default.isChecked():
			default_value = self._wdn_extra_demand_default()
			if default_value is not None:
				self.extra_demand.blockSignals(True)
				self.extra_demand.setValue(default_value)
				self.extra_demand.blockSignals(False)

	def _build_ui(self, defaults: "SolverParams") -> None:
		root = QtWidgets.QVBoxLayout(self)

		general_group = QtWidgets.QGroupBox("Parameters")
		general_form = QtWidgets.QFormLayout(general_group)
		self._form = general_form

		self.norm_value = self._new_double_spin(0.1, 100.0, defaults.norm, decimals=4, step=0.1)
		self._add_row(general_form, "norm", "Norm", self.norm_value)

		self.measurement_heads_equal = QtWidgets.QCheckBox()
		self.measurement_heads_equal.setChecked(defaults.measurement_heads_equal_only)
		self._add_row(general_form, "heads_equal", "Heads Equal at Sensors", self.measurement_heads_equal)

		self.match_total_demand = QtWidgets.QCheckBox()
		self.match_total_demand.setChecked(bool(getattr(defaults, "match_total_demand", True)))
		self._add_row(general_form, "match_total_demand", "Match Total Demand", self.match_total_demand)
		root.addWidget(general_group)

		solver_group = QtWidgets.QGroupBox("Solver Specification")
		solver_form = QtWidgets.QFormLayout(solver_group)

		self.method = QtWidgets.QComboBox()
		self.method.addItem("head loss (x)", "xd")
		self.method.addItem("head (h)", "classical")
		self.method.setCurrentIndex(max(0, self.method.findData(defaults.method)))
		self._add_solver_row(solver_form, "method", "Method", self.method)

		self.demand_lb = self._new_double_spin(0.0, 1e6, defaults.demand_lb, decimals=8, step=1e-6)
		self._add_solver_row(solver_form, "demand_lb", "Demand LB", self.demand_lb)

		self.extra_demand_use_default = QtWidgets.QCheckBox()
		self.extra_demand_use_default.setChecked(bool(getattr(defaults, "extra_demand_use_default", True)))
		self.extra_demand_use_default.toggled.connect(lambda *_: self._update_extra_demand_state())
		self._add_solver_row(solver_form, "extra_demand_use_default", "Use Default Extra Demand", self.extra_demand_use_default)

		extra_demand_default = self._wdn_extra_demand_default()
		extra_demand_initial = float(getattr(defaults, "extra_demand", 1.2))
		if self.extra_demand_use_default.isChecked() and extra_demand_default is not None:
			extra_demand_initial = extra_demand_default
		self.extra_demand = self._new_double_spin(0.0, 1e6, extra_demand_initial, decimals=6, step=0.1)
		self._add_solver_row(solver_form, "extra_demand", "Extra Demand", self.extra_demand)

		self.dynamic_multistart = QtWidgets.QCheckBox()
		self.dynamic_multistart.setChecked(bool(getattr(defaults, "dynamic_multistart", False)))
		self.dynamic_multistart.stateChanged.connect(lambda *_: self._update_visibility())
		self._add_solver_row(solver_form, "dynamic_multistart", "Dynamic Multistart", self.dynamic_multistart)

		self.multi_starts = QtWidgets.QSpinBox()
		self.multi_starts.setRange(1, 100)
		self.multi_starts.setValue(defaults.multi_starts)
		self._add_solver_row(solver_form, "multi_starts", "Multi Starts", self.multi_starts)

		self.dms_consistency = QtWidgets.QSpinBox()
		self.dms_consistency.setRange(1, 100)
		self.dms_consistency.setValue(int(getattr(defaults, "dms_consistency", 3)))
		self._add_solver_row(solver_form, "dms_consistency", "Consistency", self.dms_consistency)

		self.dms_deviation = self._new_double_spin(0.0, 1.0, float(getattr(defaults, "dms_deviation", 0.95)), decimals=4, step=0.01)
		self._add_solver_row(solver_form, "dms_deviation", "Deviation", self.dms_deviation)

		self.dms_radius = QtWidgets.QLineEdit()
		dms_radius_default = float(getattr(defaults, "dms_radius", float("inf")))
		self.dms_radius.setText("inf" if math.isinf(dms_radius_default) else f"{dms_radius_default:g}")
		self._add_solver_row(solver_form, "dms_radius", "Radius r", self.dms_radius)

		self.dms_max_starts = QtWidgets.QSpinBox()
		self.dms_max_starts.setRange(1, 1000)
		self.dms_max_starts.setValue(int(getattr(defaults, "dms_max_starts", 10)))
		self._add_solver_row(solver_form, "dms_max_starts", "DMS Max Starts", self.dms_max_starts)

		self.dms_discard_unclear = QtWidgets.QCheckBox()
		self.dms_discard_unclear.setChecked(bool(getattr(defaults, "dms_discard_unclear", True)))
		self._add_solver_row(solver_form, "dms_discard_unclear", "Discard Unclear Configs", self.dms_discard_unclear)

		self.multi_noise = self._new_double_spin(0.0, 100.0, defaults.multi_start_noise, decimals=4, step=0.01)
		self._add_solver_row(solver_form, "multi_noise", "Noise Abs", self.multi_noise)

		self.multi_noise_rel = self._new_double_spin(0.0, 100.0, defaults.multi_start_noise_rel, decimals=4, step=0.01)
		self._add_solver_row(solver_form, "multi_noise_rel", "Noise Rel", self.multi_noise_rel)

		self.hexaly_time_limit = QtWidgets.QSpinBox()
		self.hexaly_time_limit.setRange(1, 36000)
		self.hexaly_time_limit.setValue(defaults.hexaly_time_limit)
		self._add_solver_row(solver_form, "hexaly_time_limit", "Time Limit", self.hexaly_time_limit)
		root.addWidget(solver_group)

		self._mode = defaults.mode
		self._solver_group = solver_group
		self._update_extra_demand_state()
		self._update_visibility()

	def set_mode(self, mode: str) -> None:
		self._mode = str(mode)
		self._update_visibility()

	def _update_visibility(self) -> None:
		mode = str(getattr(self, "_mode", "W_d"))
		show_method = mode in {"W_d", "W_d_M", "C_d"}
		dms_enabled = self.dynamic_multistart.isChecked() if hasattr(self, "dynamic_multistart") else False
		self._solver_group.setVisible(True)
		self._set_row_visible("norm", mode != "B")
		self._set_row_visible("match_total_demand", mode in {"W_d", "W_d_M", "C_d", "B"})
		self._set_solver_row_visible("method", show_method)
		self._set_solver_row_visible("extra_demand_use_default", mode in {"W_d", "W_d_M", "C_d", "B"})
		self._set_solver_row_visible("extra_demand", mode in {"W_d", "W_d_M", "C_d", "B"})
		self._set_solver_row_visible("multi_starts", not dms_enabled)
		self._set_solver_row_visible("dms_consistency", dms_enabled)
		self._set_solver_row_visible("dms_deviation", dms_enabled)
		self._set_solver_row_visible("dms_radius", dms_enabled)
		self._set_solver_row_visible("dms_max_starts", dms_enabled)
		self._set_solver_row_visible("dms_discard_unclear", dms_enabled)

	def _parse_dms_radius(self) -> float:
		return _float_or_default(self.dms_radius.text(), float("inf"))

	def get_payload(self) -> Dict[str, object]:
		self._update_extra_demand_state()
		return {
			"METHOD": str(self.method.currentData()),
			"NORM": self.norm_value.value(),
			"DEMAND_LB": self.demand_lb.value(),
			"EXTRA_DEMAND": self.extra_demand.value(),
			"MEASUREMENT_HEADS_EQUAL_ONLY": self.measurement_heads_equal.isChecked(),
			"MATCH_TOTAL_DEMAND": self.match_total_demand.isChecked(),
			"MATCH_RESERVOIR_OUTFLOW_BETWEEN_PAIRS": self.match_total_demand.isChecked(),
			"DYNAMIC_MULTISTART": self.dynamic_multistart.isChecked(),
			"DMS_CONSISTENCY": self.dms_consistency.value(),
			"DMS_DEVIATION": self.dms_deviation.value(),
			"DMS_RADIUS": self._parse_dms_radius(),
			"DMS_MAX_STARTS": self.dms_max_starts.value(),
			"DMS_DISCARD_UNCLEAR": self.dms_discard_unclear.isChecked(),
			"MULTI_STARTS": self.multi_starts.value(),
			"MULTI_START_NOISE": self.multi_noise.value(),
			"MULTI_START_NOISE_REL": self.multi_noise_rel.value(),
			"HEXALY_TIME_LIMIT": self.hexaly_time_limit.value(),
		}


class NetworkPlot(QtWidgets.QWidget):
	measurement_changed = QtCore.pyqtSignal(list)
	node_right_clicked = QtCore.pyqtSignal(str)
	reservoir_right_clicked = QtCore.pyqtSignal(str)

	def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
		super().__init__(parent)
		self.figure = Figure(figsize=(6, 5))
		self.canvas = FigureCanvas(self.figure)
		self.ax = self.figure.add_subplot(111)
		self._node_ids: List[str] = []
		self._node_pos: Dict[str, tuple[float, float]] = {}
		self._measurement_set: set[str] = set()
		self._reservoir_set: set[str] = set()
		self._ls_current_set: set[str] = set()
		self._ls_swap_out: Optional[str] = None
		self._ls_swap_in: Optional[str] = None
		self._show_sensors_mode = False
		self._network = None
		self._pipe_classes: Optional[Dict[str, Dict[str, float]]] = None
		self._linearized_pipes: Dict[str, float] = {}
		self._linearization_scale_needed: Dict[str, float] = {}
		self._display_demands: Dict[str, float] = {}
		self._display_base_demands: Dict[str, float] = {}
		self._allow_measurement_edit = True
		self._highlight_node: Optional[str] = None
		self._node_mae: Dict[str, float] = {}
		self._mae_max: float = 1.0
		self._elimination_node: Optional[str] = None
		self._node_deltas: Dict[str, float] = {}
		self._init_ui()

	def _init_ui(self) -> None:
		layout = QtWidgets.QVBoxLayout(self)
		layout.addWidget(self.canvas)
		self.canvas.mpl_connect("button_press_event", self._on_press)

	def load_network(self, wdn_name: str) -> None:
		self.ax.clear()
		self._node_ids = []
		self._node_pos = {}
		self._measurement_set = set()
		self._reservoir_set = set()
		self._ls_current_set = set()
		self._ls_swap_out = None
		self._ls_swap_in = None
		self._pipe_classes = None
		self._linearized_pipes = {}
		self._linearization_scale_needed = {}
		self._display_demands = {}
		self._display_base_demands = {}
		self._highlight_node = None
		self._node_mae = {}
		self._mae_max = 1.0
		self._elimination_node = None
		self._node_deltas = {}

		inp_path = f"./wdn/{wdn_name}.inp"
		network = load_inp_network(inp_path)
		self._network = network
		coords = {node_id: node.coordinates for node_id, node in network.nodes.items() if node.coordinates is not None}
		if len(coords) != len(network.nodes):
			import networkx as nx
			G = nx.Graph()
			for node_id in network.nodes:
				G.add_node(node_id)
			for pipe in network.pipes.values():
				G.add_edge(pipe.start_node, pipe.end_node)
			coords = nx.spring_layout(G, seed=1)

		self._node_ids = list(coords.keys())
		self._node_pos = {k: (float(v[0]), float(v[1])) for k, v in coords.items()}
		self._reservoir_set = set(network.reservoirs.keys())
		self._redraw(wdn_name)

	def set_pipe_classes(
		self,
		base_flows: Dict[str, float],
		c_bounds: Dict[str, float],
		C_bounds: Dict[str, float],
	) -> None:
		self._pipe_classes = {
			"base_flows": base_flows,
			"c_bounds": c_bounds,
			"C_bounds": C_bounds,
		}
		self._redraw()

	def set_linearized_pipes(self, linearized_pipes: Dict[str, float]) -> None:
		self._linearized_pipes = {str(pipe_id): float(q0) for pipe_id, q0 in linearized_pipes.items()}
		self._redraw()

	def set_linearization_scale_needed(self, required_scale: Dict[str, float]) -> None:
		self._linearization_scale_needed = {str(pipe_id): float(val) for pipe_id, val in required_scale.items()}
		self._redraw()

	def set_measurements(self, nodes: List[str]) -> None:
		self._measurement_set = set(nodes)
		self._redraw()

	def set_show_sensors_mode(self, enabled: bool) -> None:
		self._show_sensors_mode = bool(enabled)
		self._redraw()

	def set_measurement_editable(self, editable: bool) -> None:
		self._allow_measurement_edit = bool(editable)

	def set_demands_overlay(self, base_demands: Dict[str, float], current_demands: Dict[str, float]) -> None:
		self._display_base_demands = {str(k): float(v) for k, v in base_demands.items()}
		self._display_demands = {str(k): float(v) for k, v in current_demands.items()}
		self._redraw()

	def clear_demands_overlay(self) -> None:
		self._display_base_demands = {}
		self._display_demands = {}
		self._redraw()

	def set_node_mae(self, mae: Dict[str, float], max_value: float) -> None:
		self._node_mae = {str(k): float(v) for k, v in mae.items()}
		self._mae_max = max(float(max_value), 1e-9)
		self._redraw()

	def clear_node_mae(self) -> None:
		self._node_mae = {}
		self._mae_max = 1.0
		self._redraw()

	def set_elimination_node(self, node_id: Optional[str]) -> None:
		self._elimination_node = str(node_id) if node_id else None
		self._redraw()

	def set_node_deltas(self, deltas: Dict[str, float]) -> None:
		self._node_deltas = {str(k): float(v) for k, v in deltas.items()}
		self._redraw()

	def _delta_to_color(self, delta: float) -> str:
		max_delta = max(self._node_deltas.values()) if self._node_deltas else 0.0
		if max_delta <= 1e-12:
			return "#e0f2fe"
		t = min(1.0, max(0.0, delta / max_delta))
		r = int(round(224 + t * (124 - 224)))
		g = int(round(242 + t * (58 - 242)))
		b = int(round(254 + t * (237 - 254)))
		return f"#{r:02x}{g:02x}{b:02x}"

	def _mae_to_color(self, mae: float) -> str:
		t = min(1.0, max(0.0, mae / self._mae_max))
		r = int(round(39 + t * 192))   # #27ae60 → #e74c3c
		g = int(round(174 - t * 98))
		b = int(round(96 - t * 36))
		return f"#{r:02x}{g:02x}{b:02x}"

	def set_highlight_node(self, node_id: Optional[str]) -> None:
		self._highlight_node = str(node_id) if node_id else None
		self._redraw()

	def set_local_search_highlight(self, current_nodes: List[str], swap_out: Optional[str] = None, swap_in: Optional[str] = None) -> None:
		self._ls_current_set = set(str(n) for n in current_nodes)
		self._ls_swap_out = str(swap_out) if swap_out else None
		self._ls_swap_in = str(swap_in) if swap_in else None
		self._redraw()

	def clear_local_search_highlight(self) -> None:
		self._ls_current_set = set()
		self._ls_swap_out = None
		self._ls_swap_in = None
		self._redraw()

	def get_junction_nodes(self) -> List[str]:
		if self._network is None:
			return []
		return sorted(str(node_id) for node_id in self._network.junctions.keys())

	def get_reservoir_adjacent_junctions(self) -> List[str]:
		"""Returns junction IDs directly connected to any reservoir via a pipe.

		These are the only valid elimination nodes: dg/dh_v is nonzero only for
		junctions that share a pipe with the reservoir.
		"""
		if self._network is None:
			return []
		junction_ids = set(str(jid) for jid in self._network.junctions.keys())
		reservoir_ids = set(str(rid) for rid in self._network.reservoirs.keys())
		neighbors: List[str] = []
		seen: set = set()
		for pipe in self._network.pipes.values():
			s, t = str(pipe.start_node), str(pipe.end_node)
			for res_end, junc_end in [(s, t), (t, s)]:
				if res_end in reservoir_ids and junc_end in junction_ids and junc_end not in seen:
					neighbors.append(junc_end)
					seen.add(junc_end)
		return neighbors

	def get_pipe_adjacency(self) -> Dict[str, List[str]]:
		"""Returns junction→[neighboring junctions] connected by a pipe."""
		if self._network is None:
			return {}
		junction_ids = set(str(jid) for jid in self._network.junctions.keys())
		adj: Dict[str, List[str]] = {jid: [] for jid in junction_ids}
		for pipe in self._network.pipes.values():
			s = str(pipe.start_node)
			e = str(pipe.end_node)
			if s in junction_ids and e in junction_ids:
				adj[s].append(e)
				adj[e].append(s)
		return adj

	def _redraw(self, title_override: Optional[str] = None) -> None:
		self.ax.clear()
		if self._network is None:
			self.canvas.draw_idle()
			return

		for pipe in self._network.pipes.values():
			start = self._node_pos.get(pipe.start_node)
			end = self._node_pos.get(pipe.end_node)
			if start is None or end is None:
				continue
			edge_color = "#8b5cf6" if pipe.pipe_id in self._linearized_pipes else ("#cbd5e0" if self._show_sensors_mode else "#a0aec0")
			edge_alpha = 0.35 if self._show_sensors_mode else 1.0
			line_width = 2.4 if pipe.pipe_id in self._linearized_pipes else 1.0
			self.ax.plot([start[0], end[0]], [start[1], end[1]], color=edge_color, linewidth=line_width, alpha=edge_alpha, zorder=1)
			if self._linearization_scale_needed:
				scale_req = self._linearization_scale_needed.get(str(pipe.pipe_id))
				if scale_req is not None:
					x = (start[0] + end[0]) / 2.0
					y = (start[1] + end[1]) / 2.0
					if math.isinf(float(scale_req)):
						label = "eps_scale*=inf"
					else:
						label = f"eps_scale*={float(scale_req):.3g}"
					self.ax.text(x, y, label, fontsize=6, ha="center", va="center", color="#4a1d96", zorder=4)

		junctions = [n for n in self._node_ids if n not in self._reservoir_set]
		measurements = [n for n in junctions if n in self._measurement_set]
		others = [n for n in junctions if n not in self._measurement_set]
		reservoirs = [n for n in self._node_ids if n in self._reservoir_set]

		def _scatter(nodes: List[str], marker: str, color: str, edge: str, size: float = 110.0, alpha: float = 1.0) -> None:
			if not nodes:
				return
			xs = [self._node_pos[n][0] for n in nodes]
			ys = [self._node_pos[n][1] for n in nodes]
			self.ax.scatter(xs, ys, s=size, marker=marker, c=color, edgecolors=edge, alpha=alpha, zorder=2)

		use_mae = bool(self._node_mae)
		use_delta = bool(self._node_deltas) and not use_mae
		elim = self._elimination_node

		def _node_fill(node_id: str) -> str:
			if use_mae:
				return self._mae_to_color(self._node_mae.get(node_id, 0.0))
			if use_delta:
				return self._delta_to_color(self._node_deltas.get(node_id, 0.0))
			return "#f2f2f2"

		def _scatter_per_node(nodes: List[str], marker: str, edge: str, size: float, lw: float = 1.0, z: float = 2.0, alpha: float = 1.0) -> None:
			if not nodes:
				return
			xs = [self._node_pos[n][0] for n in nodes]
			ys = [self._node_pos[n][1] for n in nodes]
			colors = [_node_fill(n) for n in nodes]
			self.ax.scatter(xs, ys, s=size, marker=marker, c=colors, edgecolors=edge, linewidths=lw, alpha=alpha, zorder=z)

		if self._show_sensors_mode:
			_scatter(others, "o", "#e2e8f0", "#94a3b8", size=70.0, alpha=0.45)
			_scatter(measurements, "h", "#ffdd57", "#8a5a00", size=190.0, alpha=1.0)
			_scatter(reservoirs, "s", "#90cdf4", "#1a365d", size=135.0, alpha=0.95)
		else:
			junc_set = set(junctions)
			elim_grp = [elim] if elim and elim in junc_set else []
			meas_grp = [n for n in junctions if n in self._measurement_set and n != elim]
			other_grp = [n for n in junctions if n not in self._measurement_set and n != elim]
			_scatter_per_node(other_grp, "o", "#555555", 90.0, lw=0.5, z=2.0)
			_scatter_per_node(meas_grp, "h", "#333333", 190.0, lw=1.5, z=2.2)
			_scatter_per_node(elim_grp, "^", "#7c3aed", 200.0, lw=2.0, z=2.5)
			_scatter(reservoirs, "s", "#a0aec0", "#4a5568", size=110.0)

		if self._highlight_node and self._highlight_node in self._node_pos and self._highlight_node not in self._reservoir_set and not self._node_mae and self._highlight_node != elim:
			xh, yh = self._node_pos[self._highlight_node]
			self.ax.scatter([xh], [yh], s=250.0, marker="o", c="#48bb78", edgecolors="#22543d", linewidths=1.6, zorder=2.8)

		# Local-search overlay:
		# - current best nodes: blue hexagons
		# - swap-out node: yellow hexagon
		# - swap-in node: red hexagon
		if self._ls_current_set or self._ls_swap_out or self._ls_swap_in:
			valid_junctions = set(junctions)
			swap_out = self._ls_swap_out if self._ls_swap_out in valid_junctions else None
			swap_in = self._ls_swap_in if self._ls_swap_in in valid_junctions else None
			current_nodes = [
				n for n in self._ls_current_set
				if n in valid_junctions and n != swap_out and n != swap_in
			]
			_scatter(sorted(current_nodes), "h", "#4f9dff", "#0b3d91", size=220.0, alpha=0.95)
			if swap_out is not None:
				_scatter([swap_out], "h", "#ffd966", "#8a5a00", size=260.0, alpha=1.0)
			if swap_in is not None:
				_scatter([swap_in], "h", "#ff6b6b", "#7f1d1d", size=260.0, alpha=1.0)

		xs_all = [p[0] for p in self._node_pos.values()]
		ys_all = [p[1] for p in self._node_pos.values()]
		y_offset = 0.03 * max((max(ys_all) - min(ys_all)) if ys_all else 1.0, (max(xs_all) - min(xs_all)) if xs_all else 1.0, 1.0)

		for node_id in self._node_ids:
			pos = self._node_pos.get(node_id)
			if pos is None:
				continue
			if self._show_sensors_mode and node_id not in self._measurement_set and node_id not in self._reservoir_set:
				continue
			label = str(node_id)
			if self._show_sensors_mode and node_id in self._measurement_set:
				label = f"{label} (S)"
			self.ax.text(pos[0], pos[1], label, fontsize=7, ha="center", va="center", color="#111111", zorder=3)
			text_lines: List[str] = []
			if node_id in self._display_demands:
				base_val = float(self._display_base_demands.get(node_id, 0.0))
				cur_val = float(self._display_demands.get(node_id, 0.0))
				delta_val = cur_val - base_val
				text_lines.append(f"d={cur_val:.4f}  Δ={delta_val:.4f}")
			if node_id in self._node_mae:
				text_lines.append(f"MAE={self._node_mae[node_id]:.4f}")
			if text_lines:
				self.ax.text(pos[0], pos[1] - y_offset, "\n".join(text_lines), fontsize=6.5, ha="center", va="top", color="#1f2937", zorder=3)

		if self._pipe_classes:
			base_flows = self._pipe_classes.get("base_flows", {})
			c_bounds = self._pipe_classes.get("c_bounds", {})
			C_bounds = self._pipe_classes.get("C_bounds", {})
			for pipe in self._network.pipes.values():
				start = self._node_pos.get(pipe.start_node)
				end = self._node_pos.get(pipe.end_node)
				if start is None or end is None:
					continue
				x = (start[0] + end[0]) / 2.0
				y = (start[1] + end[1]) / 2.0
				q0 = float(base_flows.get(pipe.pipe_id, 0.0))
				cl = float(c_bounds.get(pipe.pipe_id, float("-inf")))
				cu = float(C_bounds.get(pipe.pipe_id, float("inf")))
				label = f"q0={q0:.4f}\n[{cl:.4f}, {cu:.4f}]"
				self.ax.text(x, y, label, fontsize=6, ha="center", va="center", color="#111111", zorder=4)

		wdn_name = title_override if title_override else "Network"
		if self._show_sensors_mode:
			wdn_name = f"{wdn_name} | sensors: {len(measurements)}"
		if self._node_mae:
			wdn_name = f"{wdn_name} | MAE  green=0 … red={self._mae_max:.4f}"
		elif self._node_deltas:
			max_d = max(self._node_deltas.values()) if self._node_deltas else 0.0
			wdn_name = f"{wdn_name} | Δ  light-blue=0 … dark-purple={max_d:.4f}"
		self.ax.set_title(wdn_name)
		self.ax.axis("off")
		self.canvas.draw_idle()

	def _click_threshold_sq(self) -> float:
		xs = [p[0] for p in self._node_pos.values()]
		ys = [p[1] for p in self._node_pos.values()]
		if not xs or not ys:
			return 0.0
		t = 0.02 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
		return t * t

	def _find_closest_junction(self, x: float, y: float) -> tuple[Optional[str], Optional[float]]:
		min_dist: Optional[float] = None
		closest: Optional[str] = None
		for node_id in self._node_ids:
			if node_id in self._reservoir_set:
				continue
			pos = self._node_pos.get(node_id)
			if pos is None:
				continue
			d = (pos[0] - x) ** 2 + (pos[1] - y) ** 2
			if min_dist is None or d < min_dist:
				min_dist = d
				closest = node_id
		return closest, min_dist

	def _find_closest_any(self, x: float, y: float) -> tuple[Optional[str], Optional[float]]:
		min_dist: Optional[float] = None
		closest: Optional[str] = None
		for node_id in self._node_ids:
			pos = self._node_pos.get(node_id)
			if pos is None:
				continue
			d = (pos[0] - x) ** 2 + (pos[1] - y) ** 2
			if min_dist is None or d < min_dist:
				min_dist = d
				closest = node_id
		return closest, min_dist

	def _on_press(self, event) -> None:
		if self._network is None or event.inaxes != self.ax:
			return
		if event.xdata is None or event.ydata is None:
			return
		x, y = float(event.xdata), float(event.ydata)
		thresh_sq = self._click_threshold_sq()

		if event.button == MouseButton.RIGHT:
			closest, dist = self._find_closest_any(x, y)
			if closest is None or dist is None or dist > thresh_sq:
				return
			if closest in self._reservoir_set:
				self.reservoir_right_clicked.emit(str(closest))
			else:
				self.node_right_clicked.emit(str(closest))
			return

		if event.button == MouseButton.LEFT:
			closest, dist = self._find_closest_junction(x, y)
			if closest is None or dist is None or dist > thresh_sq:
				return
			if not self._allow_measurement_edit:
				return
			if closest == self._elimination_node:
				return
			if closest in self._measurement_set:
				self._measurement_set.remove(closest)
			else:
				self._measurement_set.add(closest)
			self._redraw()
			self.measurement_changed.emit(sorted(self._measurement_set))


class ScenarioViewerDialog(QtWidgets.QDialog):
	"""Shows individual MCMC samples with per-node absolute error coloured green→red."""

	def __init__(
		self,
		samples_d,
		junc_ids: List[str],
		scenario_demands: Dict[str, float],
		burn_in: int,
		network,
		node_pos: Dict[str, tuple],
		reservoir_set: set,
		measurement_set: set,
		elimination_node: Optional[str],
		log_targets=None,
		parent=None,
	) -> None:
		super().__init__(parent)
		self.setWindowTitle("Simulated Scenarios Viewer")
		self.resize(780, 620)
		import numpy as _np
		self._samples_d = samples_d
		self._junc_ids = junc_ids
		self._scenario_arr = _np.array([float(scenario_demands.get(str(j), 0.0)) for j in junc_ids])
		self._burn_in = burn_in
		self._network = network
		self._node_pos = node_pos
		self._reservoir_set = reservoir_set
		self._measurement_set = measurement_set
		self._elimination_node = elimination_node
		self._log_targets = log_targets
		self._current_idx = 0
		n_samples = len(samples_d)
		all_err = _np.abs(samples_d - self._scenario_arr[None, :])
		self._max_error = max(float(_np.max(all_err)), 1e-9)

		layout = QtWidgets.QVBoxLayout(self)
		self._idx_label = QtWidgets.QLabel()
		layout.addWidget(self._idx_label)

		nav = QtWidgets.QHBoxLayout()
		self._prev_btn = QtWidgets.QPushButton("◀  Prev")
		self._next_btn = QtWidgets.QPushButton("Next  ▶")
		self._skip_btn = QtWidgets.QPushButton("Skip burn-in")
		nav.addWidget(self._prev_btn)
		nav.addWidget(self._next_btn)
		nav.addWidget(self._skip_btn)
		layout.addLayout(nav)

		self._figure = Figure(figsize=(6, 5))
		self._canvas = FigureCanvas(self._figure)
		self._ax = self._figure.add_subplot(111)
		layout.addWidget(self._canvas)

		self._prev_btn.clicked.connect(self._go_prev)
		self._next_btn.clicked.connect(self._go_next)
		self._skip_btn.clicked.connect(self._skip_burn_in)

		self._draw_current()

	def _error_to_color(self, err: float) -> str:
		t = min(1.0, max(0.0, err / self._max_error))
		r = int(round(39 + t * 192))
		g = int(round(174 - t * 98))
		b = int(round(96 - t * 36))
		return f"#{r:02x}{g:02x}{b:02x}"

	def _go_prev(self) -> None:
		if self._current_idx > 0:
			self._current_idx -= 1
			self._draw_current()

	def _go_next(self) -> None:
		if self._current_idx < len(self._samples_d) - 1:
			self._current_idx += 1
			self._draw_current()

	def _skip_burn_in(self) -> None:
		target = min(self._burn_in, len(self._samples_d) - 1)
		if target != self._current_idx:
			self._current_idx = target
			self._draw_current()

	def _draw_current(self) -> None:
		import numpy as _np
		self._ax.clear()
		idx = self._current_idx
		n = len(self._samples_d)

		d_sample = self._samples_d[idx]
		delta_arr = d_sample - self._scenario_arr
		errors = _np.abs(delta_arr)
		err_dict = {str(self._junc_ids[i]): float(errors[i]) for i in range(len(self._junc_ids))}
		d_dict = {str(self._junc_ids[i]): float(d_sample[i]) for i in range(len(self._junc_ids))}
		delta_dict = {str(self._junc_ids[i]): float(delta_arr[i]) for i in range(len(self._junc_ids))}

		for pipe in self._network.pipes.values():
			s = self._node_pos.get(pipe.start_node)
			e = self._node_pos.get(pipe.end_node)
			if s and e:
				self._ax.plot([s[0], e[0]], [s[1], e[1]], color="#a0aec0", linewidth=1.0, zorder=1)

		all_nodes = list(self._node_pos.keys())
		junctions = [nd for nd in all_nodes if nd not in self._reservoir_set]
		reservoirs = [nd for nd in all_nodes if nd in self._reservoir_set]
		elim = self._elimination_node
		junc_set = set(junctions)
		elim_grp = [elim] if elim and elim in junc_set else []
		meas_grp = [nd for nd in junctions if nd in self._measurement_set and nd != elim]
		other_grp = [nd for nd in junctions if nd not in self._measurement_set and nd != elim]

		def _sc(nodes, marker, edge, size, lw=1.0, z=2):
			if not nodes:
				return
			xs = [self._node_pos[nd][0] for nd in nodes]
			ys = [self._node_pos[nd][1] for nd in nodes]
			colors = [self._error_to_color(err_dict.get(nd, 0.0)) for nd in nodes]
			self._ax.scatter(xs, ys, s=size, marker=marker, c=colors, edgecolors=edge, linewidths=lw, zorder=z)

		_sc(other_grp, "o", "#555555", 90.0, lw=0.5)
		_sc(meas_grp, "h", "#333333", 190.0, lw=1.5, z=2.2)
		_sc(elim_grp, "^", "#7c3aed", 200.0, lw=2.0, z=2.5)
		if reservoirs:
			rxs = [self._node_pos[nd][0] for nd in reservoirs]
			rys = [self._node_pos[nd][1] for nd in reservoirs]
			self._ax.scatter(rxs, rys, s=110.0, marker="s", c="#a0aec0", edgecolors="#4a5568", zorder=2)

		all_pos = list(self._node_pos.values())
		if all_pos:
			y_off = 0.03 * max(max(p[1] for p in all_pos) - min(p[1] for p in all_pos),
								max(p[0] for p in all_pos) - min(p[0] for p in all_pos), 1.0)
		else:
			y_off = 0.05
		for nd in junctions:
			pos = self._node_pos.get(nd)
			if pos:
				self._ax.text(pos[0], pos[1], str(nd), fontsize=7, ha="center", va="center", color="#111111", zorder=3)
				d_val = d_dict.get(nd, 0.0)
				delta_val = delta_dict.get(nd, 0.0)
				e_val = err_dict.get(nd, 0.0)
				self._ax.text(pos[0], pos[1] - y_off,
					f"d={d_val:.4f}  Δ={delta_val:+.4f}\ne={e_val:.4f}",
					fontsize=6, ha="center", va="top", color="#1f2937", zorder=3)

		lp_tag = ""
		if self._log_targets is not None and idx < len(self._log_targets):
			lp_tag = f"  |  log-pdf={self._log_targets[idx]:.4f}"
		burn_tag = " [burn-in]" if idx < self._burn_in else ""
		self._ax.set_title(f"Sample {idx + 1}/{n}{burn_tag}{lp_tag}  |  error  green=0 … red={self._max_error:.4f}")
		self._ax.axis("off")
		self._idx_label.setText(f"Sample {idx + 1} / {n}{burn_tag}")
		self._prev_btn.setEnabled(idx > 0)
		self._next_btn.setEnabled(idx < n - 1)
		self._skip_btn.setEnabled(idx < self._burn_in and self._burn_in < n)
		self._canvas.draw_idle()


class DemandDistanceViewerDialog(QtWidgets.QDialog):
	def __init__(
		self,
		items: List[tuple[str, str, float, int]],
		temp_dir: str,
		parent: "QtWidgets.QWidget | None" = None,
	) -> None:
		super().__init__(parent)
		self._temp_dir = temp_dir
		# Group items by measurement count; within each group items are already sorted by radius.
		groups_dict: Dict[int, List[tuple[str, str, float, int]]] = {}
		for item in items:
			count = item[3]
			if count not in groups_dict:
				groups_dict[count] = []
			groups_dict[count].append(item)
		self._sorted_counts: List[int] = sorted(groups_dict.keys())
		self._groups: List[List[tuple[str, str, float, int]]] = [
			groups_dict[c] for c in self._sorted_counts
		]
		self._group_idx = 0
		self._item_idx = 0
		self._build_ui()
		self._refresh_view()

	def _build_ui(self) -> None:
		self.setWindowTitle("Demand Distance Plots")
		self.resize(1100, 800)
		layout = QtWidgets.QVBoxLayout(self)

		self.caption = QtWidgets.QLabel("")
		layout.addWidget(self.caption)

		self.image_label = QtWidgets.QLabel()
		self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
		self.image_label.setMinimumSize(900, 650)
		layout.addWidget(self.image_label, 1)

		buttons = QtWidgets.QHBoxLayout()
		self.up_button = QtWidgets.QPushButton("▲ fewer sensors")
		self.up_button.clicked.connect(self._go_fewer)
		buttons.addWidget(self.up_button)
		self.prev_button = QtWidgets.QPushButton("< Prev")
		self.prev_button.clicked.connect(self._prev)
		buttons.addWidget(self.prev_button)
		self.next_button = QtWidgets.QPushButton("Next >")
		self.next_button.clicked.connect(self._next)
		buttons.addWidget(self.next_button)
		self.down_button = QtWidgets.QPushButton("▼ more sensors")
		self.down_button.clicked.connect(self._go_more)
		buttons.addWidget(self.down_button)
		buttons.addStretch(1)
		close_button = QtWidgets.QPushButton("Close")
		close_button.clicked.connect(self.close)
		buttons.addWidget(close_button)
		layout.addLayout(buttons)

	def _current_group(self) -> List[tuple[str, str, float, int]]:
		return self._groups[self._group_idx]

	def _refresh_view(self) -> None:
		group = self._current_group()
		name, path, radius, count = group[self._item_idx]
		n_groups = len(self._sorted_counts)
		sensor_word = "sensor" if count == 1 else "sensors"
		self.caption.setText(
			f"{count} {sensor_word}  |  config {self._item_idx + 1}/{len(group)}"
			f"  |  group {self._group_idx + 1}/{n_groups}  |  radius={radius:.6f}  |  {name}"
		)
		pixmap = QtGui.QPixmap(path)
		if pixmap.isNull():
			self.image_label.setText(f"Failed to load image: {path}")
		else:
			scaled = pixmap.scaled(
				self.image_label.size(),
				QtCore.Qt.AspectRatioMode.KeepAspectRatio,
				QtCore.Qt.TransformationMode.SmoothTransformation,
			)
			self.image_label.setPixmap(scaled)
		self.prev_button.setEnabled(self._item_idx > 0)
		self.next_button.setEnabled(self._item_idx < len(group) - 1)
		self.up_button.setEnabled(self._group_idx > 0)
		self.down_button.setEnabled(self._group_idx < len(self._groups) - 1)

	def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
		super().resizeEvent(event)
		self._refresh_view()

	def _prev(self) -> None:
		if self._item_idx > 0:
			self._item_idx -= 1
			self._refresh_view()

	def _next(self) -> None:
		if self._item_idx < len(self._current_group()) - 1:
			self._item_idx += 1
			self._refresh_view()

	def _go_fewer(self) -> None:
		if self._group_idx > 0:
			self._group_idx -= 1
			self._item_idx = min(self._item_idx, len(self._current_group()) - 1)
			self._refresh_view()

	def _go_more(self) -> None:
		if self._group_idx < len(self._groups) - 1:
			self._group_idx += 1
			self._item_idx = min(self._item_idx, len(self._current_group()) - 1)
			self._refresh_view()

	def closeEvent(self, event: QtGui.QCloseEvent) -> None:
		shutil.rmtree(self._temp_dir, ignore_errors=True)
		super().closeEvent(event)


class ResultsTableDialog(QtWidgets.QDialog):
	"""Shows the solver summary CSV in a table with toggleable extra columns."""

	_DEFAULT_COLS = ["measurement_sites", "radius"]
	_EXTRA_COLS = ["success", "max_violation", "min_demand_viol", "objective", "solver_status", "best_bound"]

	def __init__(
		self,
		rows: "List[Dict[str, str]]",
		comparison: bool = False,
		parent: "QtWidgets.QWidget | None" = None,
	) -> None:
		super().__init__(parent)
		self.setWindowTitle("Solver Results")
		self.setMinimumSize(500, 300)
		self._rows = rows
		self._comparison = comparison
		self._view_rows: List[Dict[str, str]] = []
		self._build_ui()
		self._populate()

	def _build_ui(self) -> None:
		root = QtWidgets.QVBoxLayout(self)

		# toggle checkboxes
		toggle_bar = QtWidgets.QHBoxLayout()
		toggle_bar.addWidget(QtWidgets.QLabel("Show:"))
		self._toggles: Dict[str, QtWidgets.QCheckBox] = {}
		for col in self._EXTRA_COLS:
			cb = QtWidgets.QCheckBox(col.replace("_", " "))
			cb.setChecked(False)
			cb.stateChanged.connect(self._refresh_columns)
			toggle_bar.addWidget(cb)
			self._toggles[col] = cb
		toggle_bar.addStretch()
		root.addLayout(toggle_bar)

		if self._comparison:
			sort_bar = QtWidgets.QHBoxLayout()
			sort_bar.addWidget(QtWidgets.QLabel("Order:"))
			self.sort_radius_a_btn = QtWidgets.QPushButton("Radius A ↑")
			self.sort_radius_b_btn = QtWidgets.QPushButton("Radius B ↑")
			self.sort_radius_a_btn.clicked.connect(lambda: self._sort_by_radius("A"))
			self.sort_radius_b_btn.clicked.connect(lambda: self._sort_by_radius("B"))
			sort_bar.addWidget(self.sort_radius_a_btn)
			sort_bar.addWidget(self.sort_radius_b_btn)
			sort_bar.addStretch()
			root.addLayout(sort_bar)

		self._table = QtWidgets.QTableWidget()
		self._table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
		self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
		self._table.horizontalHeader().setStretchLastSection(True)
		root.addWidget(self._table)

	def _visible_cols(self) -> "List[str]":
		base_cols = list(self._DEFAULT_COLS)
		for c in self._EXTRA_COLS:
			if self._toggles[c].isChecked():
				base_cols.append(c)
		if not self._comparison:
			return base_cols
		cols: List[str] = ["measurement_sites"]
		for col in base_cols:
			if col == "measurement_sites":
				continue
			cols.append(f"{col} A")
			cols.append(f"{col} B")
		return cols

	def _pivot_rows_for_comparison(self) -> "List[Dict[str, str]]":
		grouped: Dict[str, Dict[str, Dict[str, str]]] = {}
		for row in self._rows:
			meas = str(row.get("measurement_sites", ""))
			model = str(row.get("model", "")).strip()
			if model not in {"A", "B"}:
				continue
			if meas not in grouped:
				grouped[meas] = {}
			grouped[meas][model] = row

		out: List[Dict[str, str]] = []
		for meas in sorted(grouped.keys()):
			entry: Dict[str, str] = {"measurement_sites": meas}
			for col in self._DEFAULT_COLS + self._EXTRA_COLS:
				if col == "measurement_sites":
					continue
				entry[f"{col} A"] = str(grouped[meas].get("A", {}).get(col, ""))
				entry[f"{col} B"] = str(grouped[meas].get("B", {}).get(col, ""))
			out.append(entry)
		return out

	def _as_float_or_inf(self, value: str) -> float:
		try:
			return float(value)
		except Exception:
			return float("inf")

	def _sort_by_radius(self, model: str) -> None:
		if not self._comparison:
			return
		key_col = "radius A" if model == "A" else "radius B"
		self._view_rows.sort(key=lambda row: self._as_float_or_inf(str(row.get(key_col, ""))))
		self._populate(use_existing_view=True)

	def _populate(self, use_existing_view: bool = False) -> None:
		if not use_existing_view:
			if self._comparison:
				self._view_rows = self._pivot_rows_for_comparison()
			else:
				self._view_rows = list(self._rows)
		cols = self._visible_cols()
		headers = ["sites" if col == "measurement_sites" else col for col in cols]
		self._table.setColumnCount(len(cols))
		self._table.setHorizontalHeaderLabels(headers)
		self._table.setRowCount(len(self._view_rows))
		for r, row in enumerate(self._view_rows):
			for c, col in enumerate(cols):
				val = row.get(col, "")
				item = QtWidgets.QTableWidgetItem(str(val))
				item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
				self._table.setItem(r, c, item)
		self._table.resizeColumnsToContents()

	def _refresh_columns(self) -> None:
		self._populate()


# ---------------------------------------------------------------------------
# GNN worker threads
# ---------------------------------------------------------------------------

class GNNDatasetWorker(QtCore.QThread):
	"""Runs remote.run_dataset() off the GUI thread with cancellation support."""

	log_line = QtCore.pyqtSignal(str)
	finished = QtCore.pyqtSignal(bool, str)  # success, artifact_dir

	def __init__(
		self,
		wdn_name: str,
		measurement_nodes: List[str],
		extra_demand: float,
		num_simulations: int,
		demand_model: str,
		node_label_threshold: float,
		timeout: int = 36000,
		parent: "QtWidgets.QWidget | None" = None,
	) -> None:
		super().__init__(parent)
		self._kwargs = dict(
			wdn_name=wdn_name,
			measurement_nodes=measurement_nodes,
			extra_demand=extra_demand,
			num_simulations=num_simulations,
			demand_model=demand_model,
			node_label_threshold=node_label_threshold,
			timeout=timeout,
			log_fn=self._emit_log,
		)
		self._artifact_dir: str = ""

	def _emit_log(self, line: str) -> None:
		self.log_line.emit(line)

	def cancel(self) -> None:
		"""Request cancellation of the running notebook."""
		from old.remote import get_current_process
		proc = get_current_process()
		if proc and proc.poll() is None:
			self._emit_log("Stopping dataset generation...")
			proc.terminate()

	def run(self) -> None:
		try:
			from old.remote import run_dataset
			import shutil
			artifact_dir, was_cancelled = run_dataset(**self._kwargs)
			self._artifact_dir = artifact_dir
			if was_cancelled:
				shutil.rmtree(artifact_dir, ignore_errors=True)
				self._emit_log("Dataset generation was cancelled. Artifacts deleted.")
				self.finished.emit(False, "")
			else:
				self.finished.emit(True, artifact_dir)
		except Exception as exc:
			self._emit_log(f"ERROR: {exc}")
			if self._artifact_dir:
				import shutil
				shutil.rmtree(self._artifact_dir, ignore_errors=True)
			self.finished.emit(False, "")


class GNNTestSetWorker(QtCore.QThread):
	"""Runs remote.run_test_set() off the GUI thread with cancellation support."""

	log_line = QtCore.pyqtSignal(str)
	finished = QtCore.pyqtSignal(bool, str)  # success, artifact_dir

	def __init__(
		self,
		wdn_name: str,
		extra_demand: float,
		num_simulations: int,
		demand_model: str,
		seed: int,
		timeout: int = 36000,
		parent: "QtWidgets.QWidget | None" = None,
	) -> None:
		super().__init__(parent)
		self._kwargs = dict(
			wdn_name=wdn_name,
			extra_demand=extra_demand,
			num_simulations=num_simulations,
			demand_model=demand_model,
			seed=seed,
			timeout=timeout,
			log_fn=self._emit_log,
		)
		self._artifact_dir: str = ""

	def _emit_log(self, line: str) -> None:
		self.log_line.emit(line)

	def cancel(self) -> None:
		"""Request cancellation of the running notebook."""
		from old.remote import get_current_process
		proc = get_current_process()
		if proc and proc.poll() is None:
			self._emit_log("Stopping shared test-set generation...")
			proc.terminate()

	def run(self) -> None:
		try:
			from old.remote import run_test_set
			import shutil
			artifact_dir, was_cancelled = run_test_set(**self._kwargs)
			self._artifact_dir = artifact_dir
			if was_cancelled:
				shutil.rmtree(artifact_dir, ignore_errors=True)
				self._emit_log("Shared test-set generation was cancelled. Artifacts deleted.")
				self.finished.emit(False, "")
			else:
				self.finished.emit(True, artifact_dir)
		except Exception as exc:
			self._emit_log(f"ERROR: {exc}")
			if self._artifact_dir:
				import shutil
				shutil.rmtree(self._artifact_dir, ignore_errors=True)
			self.finished.emit(False, "")


class GNNModelWorker(QtCore.QThread):
	"""Runs remote.run_model() off the GUI thread with cancellation support."""

	log_line = QtCore.pyqtSignal(str)
	finished = QtCore.pyqtSignal(bool, str)  # success, artifact_dir

	def __init__(
		self,
		wdn_name: str,
		d_hash: str,
		dataset_dir: str,
		epochs: int,
		lr: float,
		batch_size: int,
		hidden_dim: int,
		num_layers: int,
		seed: int,
		timeout: int = 36000,
		parent: "QtWidgets.QWidget | None" = None,
	) -> None:
		super().__init__(parent)
		self._kwargs = dict(
			wdn_name=wdn_name,
			d_hash=d_hash,
			dataset_dir=dataset_dir,
			epochs=epochs,
			lr=lr,
			batch_size=batch_size,
			hidden_dim=hidden_dim,
			num_layers=num_layers,
			seed=seed,
			timeout=timeout,
			log_fn=self._emit_log,
		)
		self._artifact_dir: str = ""

	def _emit_log(self, line: str) -> None:
		self.log_line.emit(line)

	def cancel(self) -> None:
		"""Request cancellation of the running notebook."""
		from old.remote import get_current_process
		proc = get_current_process()
		if proc and proc.poll() is None:
			self._emit_log("Stopping model training...")
			proc.terminate()

	def run(self) -> None:
		try:
			from old.remote import run_model
			import shutil
			artifact_dir, was_cancelled = run_model(**self._kwargs)
			self._artifact_dir = artifact_dir
			if was_cancelled:
				shutil.rmtree(artifact_dir, ignore_errors=True)
				self._emit_log("Model training was cancelled. Artifacts deleted.")
				self.finished.emit(False, "")
			else:
				self.finished.emit(True, artifact_dir)
		except Exception as exc:
			self._emit_log(f"ERROR: {exc}")
			if self._artifact_dir:
				import shutil
				shutil.rmtree(self._artifact_dir, ignore_errors=True)
			self.finished.emit(False, "")


class GNNCompareWorker(QtCore.QThread):
	"""Runs remote.run_comparison() off the GUI thread with cancellation support."""

	log_line = QtCore.pyqtSignal(str)
	finished = QtCore.pyqtSignal(bool, str)  # success, artifact_dir

	def __init__(
		self,
		wdn_name: str,
		model_a_hash: str,
		model_a_dir: str,
		dataset_a_dir: str,
		model_b_hash: str,
		model_b_dir: str,
		dataset_b_dir: str,
		test_set_hash: str | None = None,
		demand_reconstruction: str = "algebraic",
		timeout: int = 36000,
		parent: "QtWidgets.QWidget | None" = None,
	) -> None:
		super().__init__(parent)
		self._kwargs = dict(
			wdn_name=wdn_name,
			model_a_hash=model_a_hash,
			model_a_dir=model_a_dir,
			dataset_a_dir=dataset_a_dir,
			model_b_hash=model_b_hash,
			model_b_dir=model_b_dir,
			dataset_b_dir=dataset_b_dir,
			test_set_hash=test_set_hash,
			demand_reconstruction=demand_reconstruction,
			timeout=timeout,
			log_fn=self._emit_log,
		)
		self._artifact_dir: str = ""

	def _emit_log(self, line: str) -> None:
		self.log_line.emit(line)

	def cancel(self) -> None:
		"""Request cancellation of the running comparison."""
		self._emit_log("Stopping comparison...")
		# Comparisons don't use papermill, so they can't be interrupted as cleanly.
		# Just signal that we want to stop.

	def run(self) -> None:
		try:
			from old.remote import run_comparison
			import shutil
			artifact_dir, was_cancelled = run_comparison(**self._kwargs)
			self._artifact_dir = artifact_dir
			if was_cancelled:
				shutil.rmtree(artifact_dir, ignore_errors=True)
				self._emit_log("Comparison was cancelled. Artifacts deleted.")
				self.finished.emit(False, "")
			else:
				self.finished.emit(True, artifact_dir)
		except Exception as exc:
			self._emit_log(f"ERROR: {exc}")
			if self._artifact_dir:
				import shutil
				shutil.rmtree(self._artifact_dir, ignore_errors=True)
			self.finished.emit(False, "")


class MainWindow(QtWidgets.QMainWindow):
	def __init__(self) -> None:
		super().__init__()
		self.setWindowTitle("Inverse Problem GUI")
		self.solver_params = SolverParams()
		self._post_updating_measurement_text = False
		self._post_base_demands: Dict[str, float] = {}
		self._post_current_demands: Dict[str, float] = {}
		self._post_active_scenario_name: str = ""
		self._post_active_scenario_path: str = ""
		self._post_loaded_heads: Dict[str, float] = {}
		self._post_loaded_flows: Dict[str, float] = {}
		self._post_dirty: bool = False
		self._post_last_saved_name: str = ""
		self._post_auto_elim_node: Optional[str] = None
		self._post_mh_result = None
		self._post_mh_burn_in: int = 0
		self._post_mh_junc_ids: List[str] = []
		self._measurement_value: object = []
		self._measurement_valid = True
		self._measurement_data_valid = True
		self._updating_measurement_text = False
		self._last_output_title = ""
		self._last_output_text = ""
		self._solver_worker: SolverWorker | None = None
		self._plot_worker: PlotWorker | None = None
		self._pending_runs: List[tuple[str, Dict[str, object]]] = []
		self._completed_runs: List[tuple] = []
		self._active_run_label: str = ""
		self._gnn_last_log_time: float = 0.0
		self._gnn_watchdog_warned: bool = False
		self._gnn_watchdog_timer = QtCore.QTimer(self)
		self._gnn_watchdog_timer.setInterval(5000)
		self._gnn_watchdog_timer.timeout.connect(self._gnn_watchdog_tick)
		self._init_ui()
		self._ls_worker: LocalSearchWorker | None = None
		self._load_network()

	def _dot_locale(self) -> QtCore.QLocale:
		locale = QtCore.QLocale.c()
		locale.setNumberOptions(QtCore.QLocale.NumberOption.RejectGroupSeparator)
		return locale

	def _new_double_spin(
		self,
		minimum: float,
		maximum: float,
		value: float,
		decimals: int = 6,
		step: float = 0.01,
	) -> QtWidgets.QDoubleSpinBox:
		spin = QtWidgets.QDoubleSpinBox()
		spin.setLocale(self._dot_locale())
		spin.setDecimals(decimals)
		spin.setRange(minimum, maximum)
		spin.setSingleStep(step)
		spin.setValue(value)
		return spin

	def _init_ui(self) -> None:
		central = QtWidgets.QWidget()
		self.setCentralWidget(central)
		layout = QtWidgets.QVBoxLayout(central)

		global_layout = QtWidgets.QHBoxLayout()
		layout.addLayout(global_layout)
		global_layout.addWidget(QtWidgets.QLabel("WDN Name:"))
		self.wdn_input = QtWidgets.QComboBox()
		self.wdn_input.setEditable(True)
		self.wdn_input.addItems(self._get_wdn_names())
		self.wdn_input.setCurrentText(self.solver_params.wdn)
		self.wdn_input.currentTextChanged.connect(self._wdn_changed)
		global_layout.addWidget(self.wdn_input)
		self.reload_button = QtWidgets.QPushButton("Load Network")
		self.reload_button.clicked.connect(self._load_network)
		global_layout.addWidget(self.reload_button)
		global_layout.addStretch(1)

		splitter = QtWidgets.QSplitter()
		layout.addWidget(splitter)

		left = QtWidgets.QWidget()
		left_layout = QtWidgets.QVBoxLayout(left)
		self.tabs = QtWidgets.QTabWidget()
		left_layout.addWidget(self.tabs)
		self._build_solver_tab()
		self._build_posteriori_tab()

		splitter.addWidget(left)
		self._build_local_search_tab()
		self._build_gnn_tab()
		self.tabs.currentChanged.connect(self._on_tab_changed)
		self.plot = NetworkPlot()
		self.plot.measurement_changed.connect(self._measurement_updated)
		self.plot.node_right_clicked.connect(self._posteriori_node_right_clicked)
		self.plot.reservoir_right_clicked.connect(self._posteriori_reservoir_right_clicked)
		splitter.addWidget(self.plot)
		self._apply_show_sensors_mode(self.show_sensors_mode.isChecked())
		splitter.setStretchFactor(1, 1)
		self._measurement_text_changed(self.measurement_list.text())

		self.status_bar = self.statusBar()
		self.output_button = QtWidgets.QPushButton("Show Output")
		self.output_button.clicked.connect(self._show_last_output)
		self.status_bar.addPermanentWidget(self.output_button)

	def _build_solver_tab(self) -> None:
		# Outer scrollable container
		scroll = QtWidgets.QScrollArea()
		scroll.setWidgetResizable(True)
		container = QtWidgets.QWidget()
		scroll.setWidget(container)
		outer = QtWidgets.QVBoxLayout(container)
		outer.setContentsMargins(4, 4, 4, 4)

		linear_group = QtWidgets.QGroupBox("Linearization")
		linear_form = QtWidgets.QFormLayout(linear_group)
		self.linearization_check = QtWidgets.QCheckBox()
		self.linearization_check.setChecked(False)
		self.linearization_check.toggled.connect(lambda *_: self._update_linearization_controls())
		linear_form.addRow("Enable", self.linearization_check)
		self.linearization_eps_label = QtWidgets.QLabel("epsilon_h scale")
		self.linearization_eps = self._new_double_spin(0.0, 1e6, 1e-3, decimals=6, step=1e-4)
		self.linearization_eps.valueChanged.connect(lambda *_: self._on_linearization_eps_changed())
		linear_form.addRow(self.linearization_eps_label, self.linearization_eps)
		self.linearization_button = QtWidgets.QPushButton("Look for Linearization")
		self.linearization_button.clicked.connect(self._look_for_linearization)
		linear_form.addRow(self.linearization_button)
		self.linearization_status = QtWidgets.QLabel("Linearization disabled.")
		self.linearization_status.setWordWrap(True)
		linear_form.addRow("Status", self.linearization_status)
		outer.addWidget(linear_group)

		# Shared fields
		shared_group = QtWidgets.QGroupBox("Shared")
		shared_form = QtWidgets.QFormLayout(shared_group)
		self.mode_input = QtWidgets.QComboBox()
		self.mode_input.addItem("W_d(M)", "W_d_M")
		self.mode_input.addItem("W_d", "W_d")
		self.mode_input.addItem("W_h(M)", "W_h_M")
		self.mode_input.addItem("B (Bregman)", "B")
		self.mode_input.addItem("C_d", "C_d")
		self.mode_input.addItem("W_h", "W_h")
		self.mode_input.addItem("C_h - fixed", "C_h_fixed")
		mode_default = self.solver_params.mode
		if mode_default in {"C_h", "H_h"}:
			mode_default = "C_h_fixed"
		self.mode_input.setCurrentIndex(max(0, self.mode_input.findData(mode_default)))
		self.mode_input.currentIndexChanged.connect(self._mode_changed)
		shared_form.addRow("Mode", self.mode_input)
		self.measurement_list = QtWidgets.QLineEdit("")
		self.measurement_list.setPlaceholderText("blank/#0, #a, #a-#b, or comma-separated site ids")
		self.measurement_list.textChanged.connect(self._measurement_text_changed)
		shared_form.addRow("Sites", self.measurement_list)
		self.measurement_source = QtWidgets.QComboBox()
		self.measurement_source.addItem("from W_d", "from_w_d")
		self.measurement_source.addItem("base", "base")
		self.measurement_source.addItem("custom input", "custom")
		self.measurement_source.setCurrentIndex(max(0, self.measurement_source.findData(self.solver_params.measurement_source)))
		self.measurement_source.currentIndexChanged.connect(self._measurement_source_changed)
		shared_form.addRow("Measurement", self.measurement_source)
		self.measurement_data_input = QtWidgets.QPlainTextEdit()
		self.measurement_data_input.setPlaceholderText('{\n  "12": 53.1,\n  "17": 49.8,\n  "-1": 0.031,\n  "R1": 78.4\n}')
		self.measurement_data_input.setPlainText(self.solver_params.measurement_data)
		self.measurement_data_input.textChanged.connect(self._measurement_data_text_changed)
		self.measurement_data_input.setMaximumHeight(120)
		shared_form.addRow("Custom data", self.measurement_data_input)
		self.show_sensors_mode = QtWidgets.QCheckBox("Show sensors")
		self.show_sensors_mode.setChecked(False)
		self.show_sensors_mode.toggled.connect(self._apply_show_sensors_mode)
		shared_form.addRow("Visualization", self.show_sensors_mode)
		self._rebuild_measurement_source_options(preserve_current=False)
		self._update_measurement_source_visibility()
		outer.addWidget(shared_group)

		# Options row
		opts_layout = QtWidgets.QHBoxLayout()
		self.generate_plots_check = QtWidgets.QCheckBox("Generate Plots")
		self.generate_plots_check.setChecked(True)
		opts_layout.addWidget(self.generate_plots_check)
		self.show_only_demand_deltas_check = QtWidgets.QCheckBox("Only show demand deltas")
		self.show_only_demand_deltas_check.setChecked(True)
		opts_layout.addWidget(self.show_only_demand_deltas_check)
		self.comparison_mode_check = QtWidgets.QCheckBox("Comparison Mode")
		self.comparison_mode_check.setChecked(False)
		opts_layout.addWidget(self.comparison_mode_check)
		opts_layout.addStretch()
		outer.addLayout(opts_layout)

		# Model widgets
		self.model_a = SolverModelWidget("Model A", self.solver_params, container)
		outer.addWidget(self.model_a)
		from gui.state import SolverParams as _SP
		self.model_b = SolverModelWidget("Model B", _SP(), container)
		self.model_b.setVisible(False)
		outer.addWidget(self.model_b)
		self.model_a.set_mode(self._current_mode())
		self.model_b.set_mode(self._current_mode())
		self.comparison_mode_check.stateChanged.connect(
			lambda: self.model_b.setVisible(self.comparison_mode_check.isChecked())
		)

		# Run controls
		self.solve_button = QtWidgets.QPushButton("Run Solver")
		self.solve_button.clicked.connect(self._run_solver)
		outer.addWidget(self.solve_button)

		progress_row = QtWidgets.QHBoxLayout()
		self.progress_label = QtWidgets.QLabel("")
		self.progress_label.setVisible(False)
		progress_row.addWidget(self.progress_label)
		self.progress_bar = QtWidgets.QProgressBar()
		self.progress_bar.setRange(0, 1)
		self.progress_bar.setValue(0)
		self.progress_bar.setVisible(False)
		progress_row.addWidget(self.progress_bar)
		outer.addLayout(progress_row)
		outer.addStretch()

		self.tabs.addTab(scroll, "Solver")
		self._linearization_ready = False
		self._linearization_lookup_done = False
		self._linearized_pipe_ids: Dict[str, float] = {}
		self._linearized_pipe_ids_base: Dict[str, float] = {}
		self._linearization_scale_required: Dict[str, float] = {}
		self._linearization_lookup_active = False
		self._linearization_auto_solve_pending = False
		self._update_linearization_controls()

	def _build_posteriori_tab(self) -> None:
		scroll = QtWidgets.QScrollArea()
		scroll.setWidgetResizable(True)
		container = QtWidgets.QWidget()
		scroll.setWidget(container)
		outer = QtWidgets.QVBoxLayout(container)
		outer.setContentsMargins(4, 4, 4, 4)

		# --- Sampling method selector (sits above the scenario choice) ---------------------
		method_group = QtWidgets.QGroupBox("Sampling Method")
		method_form = QtWidgets.QFormLayout(method_group)
		self.post_method_combo = QtWidgets.QComboBox()
		# (label, method, proposal) — data carried on each item.
		self._post_methods = [
			("M2 · Demand-space (Dirichlet, soft sensor) — ensemble", "demand", "ensemble"),
			("M1 · Pressure-space (Dirichlet, hard sensor) — ensemble", "pressure", "ensemble"),
			("M1 · Pressure-space — random-walk", "pressure", "rwm"),
			("M2 · Demand-space — random-walk", "demand", "rwm"),
		]
		for label, meth, prop in self._post_methods:
			self.post_method_combo.addItem(label, userData=(meth, prop))
		self.post_method_combo.setToolTip(
			"Demand-space (M2): demands primary, pressures forward-solved, sensor imposed softly "
			"by measurement noise ε — well-conditioned on low-flow networks.\n"
			"Pressure-space (M1): reduced pressure coordinates with demand reconstruction (original)."
		)
		self.post_method_combo.currentIndexChanged.connect(self._post_method_changed)
		method_form.addRow("Method", self.post_method_combo)

		self.post_sensor_eps = self._new_double_spin(1e-4, 10.0, 0.05, decimals=4, step=0.01)
		self.post_sensor_eps.setToolTip("Sensor measurement-noise σ (m of head), used by the demand method's soft likelihood. Smaller → tighter posterior (→ exact as ε→0).")
		method_form.addRow("Sensor noise ε (M2)", self.post_sensor_eps)
		outer.addWidget(method_group)

		scenario_group = QtWidgets.QGroupBox("Scenario Choice")
		scenario_form = QtWidgets.QFormLayout(scenario_group)

		scenario_row = QtWidgets.QHBoxLayout()
		self.post_scenario_combo = QtWidgets.QComboBox()
		self.post_scenario_combo.currentIndexChanged.connect(self._posteriori_scenario_changed)
		scenario_row.addWidget(self.post_scenario_combo)
		self.post_refresh_scenarios_btn = QtWidgets.QPushButton("Refresh")
		self.post_refresh_scenarios_btn.clicked.connect(lambda: self._posteriori_refresh_scenario_list(select_name=self.post_scenario_combo.currentText().strip()))
		scenario_row.addWidget(self.post_refresh_scenarios_btn)
		scenario_form.addRow("Scenario", scenario_row)

		self.post_scenario_editable = QtWidgets.QCheckBox()
		self.post_scenario_editable.setChecked(True)
		self.post_scenario_editable.toggled.connect(self._posteriori_editability_changed)
		scenario_form.addRow("Scenario may be changed", self.post_scenario_editable)

		self.post_save_name = QtWidgets.QLineEdit()
		self.post_save_name.setPlaceholderText("custom scenario name")
		scenario_form.addRow("Save as", self.post_save_name)

		self.post_default_name_btn = QtWidgets.QPushButton("Use Default Name")
		self.post_default_name_btn.clicked.connect(self._posteriori_set_default_name)
		scenario_form.addRow("", self.post_default_name_btn)

		self.post_loaded_label = QtWidgets.QLabel("")
		self.post_loaded_label.setWordWrap(True)
		scenario_form.addRow("State", self.post_loaded_label)

		outer.addWidget(scenario_group)

		params_group = QtWidgets.QGroupBox("Scenario Parameters")
		params_form = QtWidgets.QFormLayout(params_group)

		self.post_measurement_sites = QtWidgets.QLineEdit("")
		self.post_measurement_sites.setPlaceholderText("comma-separated node ids")
		self.post_measurement_sites.textChanged.connect(self._posteriori_measurement_sites_changed)
		params_form.addRow("Measurement sites", self.post_measurement_sites)

		extra_row = QtWidgets.QHBoxLayout()
		self.post_extra_demand = self._new_double_spin(0.0, 1e6, 0.0, decimals=6, step=0.01)
		self.post_extra_demand.valueChanged.connect(self._posteriori_extra_demand_changed)
		extra_row.addWidget(self.post_extra_demand)
		self.post_extra_default_btn = QtWidgets.QPushButton("Use Default")
		self.post_extra_default_btn.clicked.connect(self._posteriori_apply_default_extra_demand)
		extra_row.addWidget(self.post_extra_default_btn)
		params_form.addRow("Extra demand", extra_row)

		outer.addWidget(params_group)

		mh_group = QtWidgets.QGroupBox("M.H. Parameters")
		mh_form = QtWidgets.QFormLayout(mh_group)

		self.post_elimination_node_label = QtWidgets.QLabel("(auto)")
		self.post_elimination_node_label.setStyleSheet("color: #6b7280; font-style: italic;")
		mh_form.addRow("Eliminated node", self.post_elimination_node_label)

		self.post_num_samples = QtWidgets.QSpinBox()
		self.post_num_samples.setRange(10, 200000)
		self.post_num_samples.setValue(300)
		mh_form.addRow("Sample size", self.post_num_samples)

		self.post_burn_in = QtWidgets.QSpinBox()
		self.post_burn_in.setRange(0, 200000)
		self.post_burn_in.setValue(100)
		mh_form.addRow("Burn-in", self.post_burn_in)

		self.post_num_chains = QtWidgets.QSpinBox()
		self.post_num_chains.setRange(1, 64)
		self.post_num_chains.setValue(4)
		self.post_num_chains.setToolTip("Independent chains from dispersed starts; enables the R-hat convergence check (need ≥2).")
		mh_form.addRow("Chains (R-hat)", self.post_num_chains)

		self.post_proposal_std = self._new_double_spin(1e-4, 10.0, 0.05, decimals=4, step=0.01)
		mh_form.addRow("Proposal std", self.post_proposal_std)

		self.post_use_gram = QtWidgets.QCheckBox()
		self.post_use_gram.setChecked(True)
		mh_form.addRow("Robust Jacobian (Gram)", self.post_use_gram)

		self.post_penalty_a = self._new_double_spin(0.0, 100000.0, 1000.0, decimals=4, step=1.0)
		mh_form.addRow("Punish negativity (a)", self.post_penalty_a)

		outer.addWidget(mh_group)

		self.post_run_button = QtWidgets.QPushButton("Run Posteriori")
		self.post_run_button.clicked.connect(self._posteriori_run_clicked)
		outer.addWidget(self.post_run_button)

		self._post_view_scenarios_btn = QtWidgets.QPushButton("View Simulated Scenarios")
		self._post_view_scenarios_btn.setEnabled(False)
		self._post_view_scenarios_btn.clicked.connect(self._posteriori_view_scenarios_clicked)
		outer.addWidget(self._post_view_scenarios_btn)

		self.post_status = QtWidgets.QLabel("")
		self.post_status.setWordWrap(True)
		outer.addWidget(self.post_status)
		outer.addStretch()

		self._post_tab_index = self.tabs.addTab(scroll, "posteriori")

	def _set_linearization_status(self, text: str) -> None:
		if hasattr(self, "linearization_status"):
			self.linearization_status.setText(str(text))

	def _update_linearization_controls(self) -> None:
		enabled = bool(self.linearization_check.isChecked()) if hasattr(self, "linearization_check") else False
		if hasattr(self, "linearization_eps"):
			self.linearization_eps.setVisible(enabled)
			self.linearization_eps.setEnabled(enabled and self._linearization_lookup_done)
		if hasattr(self, "linearization_eps_label"):
			self.linearization_eps_label.setVisible(enabled)
		if hasattr(self, "linearization_button"):
			self.linearization_button.setVisible(enabled)
		if not enabled:
			self._linearization_ready = False
			self._linearization_lookup_done = False
			self._linearized_pipe_ids = {}
			self._linearized_pipe_ids_base = {}
			self._linearization_scale_required = {}
			if hasattr(self, "plot"):
				self.plot.set_linearized_pipes({})
				self.plot.set_linearization_scale_needed({})
			self._set_linearization_status("Linearization disabled.")
		else:
			if self._linearization_lookup_done:
				self._set_linearization_status(
					f"Lookup done: {len(self._linearized_pipe_ids)} pipes linearizable at eps scale {self.linearization_eps.value():.3g}."
				)
			else:
				self._set_linearization_status("Not certified yet. Run 'Look for Linearization'.")
		if hasattr(self, "solve_button"):
			self.solve_button.setEnabled((not enabled) or self._linearization_lookup_done)

	def _linearization_payload_from_widget(self, model_widget: "SolverModelWidget", lookup_only: bool) -> Dict[str, object]:
		payload = self._solver_payload_from_widget(model_widget)
		payload["LINEARIZATION_LOOKUP"] = bool(lookup_only)
		payload["LINEARIZATION_ENABLED"] = bool(self.linearization_check.isChecked() and not lookup_only)
		payload["LINEARIZATION_EPS_SCALE"] = float(self.linearization_eps.value()) if hasattr(self, "linearization_eps") else 1e-3
		if not lookup_only and self.linearization_check.isChecked() and self._linearization_lookup_done:
			payload["LINEARIZED_PIPES"] = dict(self._linearized_pipe_ids)
		payload["_linearization_lookup"] = bool(lookup_only)
		return payload

	def _on_linearization_eps_changed(self) -> None:
		if not getattr(self, "_linearization_lookup_done", False):
			return
		self._recompute_linearized_pipes_from_scale()
		self._set_linearization_status(
			f"Lookup done: {len(self._linearized_pipe_ids)} pipes linearizable at eps scale {self.linearization_eps.value():.3g}."
		)

	def _recompute_linearized_pipes_from_scale(self) -> None:
		if not self._linearization_scale_required:
			self._linearized_pipe_ids = {}
			if hasattr(self, "plot"):
				self.plot.set_linearized_pipes({})
			return
		eps_scale = float(self.linearization_eps.value()) if hasattr(self, "linearization_eps") else 0.0
		selected: Dict[str, float] = {}
		for pipe_id, need in self._linearization_scale_required.items():
			if math.isinf(float(need)):
				continue
			if float(need) <= eps_scale and pipe_id in self._linearized_pipe_ids_base:
				selected[str(pipe_id)] = float(self._linearized_pipe_ids_base[pipe_id])
		self._linearized_pipe_ids = selected
		self._linearization_ready = bool(self._linearization_lookup_done)
		if hasattr(self, "plot"):
			self.plot.set_linearized_pipes(self._linearized_pipe_ids)

	def _launch_solver_runs(self, runs: List[tuple[str, Dict[str, object]]]) -> None:
		self._pending_runs = []
		self._completed_runs = []
		for label, payload in runs:
			self._pending_runs.append((label, dict(payload)))
		self.solve_button.setEnabled(False)
		self.progress_bar.setRange(0, 1)
		self.progress_bar.setValue(0)
		self.progress_bar.setVisible(True)
		self.progress_label.setVisible(True)
		self._start_next_solver_run()

	def _look_for_linearization(self) -> None:
		if self._solver_worker is not None and self._solver_worker.isRunning():
			return
		if self._current_mode() != "W_d":
			QtWidgets.QMessageBox.information(self, "Linearization", "Linearization lookup is currently implemented for W_d only.")
			return
		if not self.linearization_check.isChecked():
			return
		self._linearization_lookup_active = True
		self._linearization_lookup_done = False
		self._set_linearization_status("Running certification lookup...")
		payload = self._linearization_payload_from_widget(self.model_a, lookup_only=True)
		self._launch_solver_runs([("L", payload)])

	def _build_local_search_tab(self) -> None:
		scroll = QtWidgets.QScrollArea()
		scroll.setWidgetResizable(True)
		container = QtWidgets.QWidget()
		scroll.setWidget(container)
		outer = QtWidgets.QVBoxLayout(container)
		outer.setContentsMargins(4, 4, 4, 4)

		info = QtWidgets.QLabel(
			"Enter explicit node IDs (comma-separated). "
			"The algorithm tries all 1-swap moves and greedily improves until no "
			"swap reduces the radius. Uses Model A solver settings."
		)
		info.setWordWrap(True)
		outer.addWidget(info)

		form = QtWidgets.QFormLayout()
		self.ls_use_default_nodes = QtWidgets.QCheckBox("Use default")
		self.ls_use_default_nodes.setChecked(False)
		self.ls_use_default_nodes.toggled.connect(self._ls_update_starting_nodes_state)
		form.addRow("Use Default", self.ls_use_default_nodes)
		self.ls_nodes_input = QtWidgets.QLineEdit()
		self.ls_nodes_input.setPlaceholderText("e.g. 3, 7, 12")
		form.addRow("Starting Nodes", self.ls_nodes_input)
		outer.addLayout(form)
		self._ls_update_starting_nodes_state()

		btn_row = QtWidgets.QHBoxLayout()
		self.ls_run_button = QtWidgets.QPushButton("Run Local Search")
		self.ls_run_button.clicked.connect(self._run_local_search)
		btn_row.addWidget(self.ls_run_button)
		self.ls_cancel_button = QtWidgets.QPushButton("Cancel")
		self.ls_cancel_button.setEnabled(False)
		self.ls_cancel_button.clicked.connect(self._cancel_local_search)
		btn_row.addWidget(self.ls_cancel_button)
		btn_row.addStretch()
		outer.addLayout(btn_row)

		self.ls_status_label = QtWidgets.QLabel("")
		outer.addWidget(self.ls_status_label)

		self.ls_log = QtWidgets.QPlainTextEdit()
		self.ls_log.setReadOnly(True)
		self.ls_log.setMaximumBlockCount(500)
		outer.addWidget(self.ls_log)
		outer.addStretch()

		self._ls_tab_index = self.tabs.addTab(scroll, "Local Search")

	def _build_gnn_tab(self) -> None:
		"""GNN tab with Run and Compare sub-pages."""
		outer = QtWidgets.QWidget()
		outer_layout = QtWidgets.QVBoxLayout(outer)
		outer_layout.setContentsMargins(4, 4, 4, 4)
		gnn_tabs = QtWidgets.QTabWidget()
		outer_layout.addWidget(gnn_tabs)

		# ── Run sub-page ─────────────────────────────────────────────────────
		run_scroll = QtWidgets.QScrollArea()
		run_scroll.setWidgetResizable(True)
		run_widget = QtWidgets.QWidget()
		run_scroll.setWidget(run_widget)
		run_layout = QtWidgets.QVBoxLayout(run_widget)
		run_layout.setContentsMargins(4, 4, 4, 4)

		# Dataset group
		ds_group = QtWidgets.QGroupBox("Dataset")
		ds_form = QtWidgets.QFormLayout(ds_group)

		default_nodes = self._gnn_default_nodes()
		self.gnn_default_nodes_label = QtWidgets.QLabel(", ".join(default_nodes) if default_nodes else "(none)")
		ds_form.addRow("Default nodes:", self.gnn_default_nodes_label)

		self.gnn_use_default_nodes = QtWidgets.QCheckBox("Use default placement")
		self.gnn_use_default_nodes.setChecked(False)
		self.gnn_use_default_nodes.toggled.connect(self._gnn_refresh_dataset_status)
		ds_form.addRow("Default placement:", self.gnn_use_default_nodes)

		self.gnn_nodes_input = QtWidgets.QLineEdit("")
		self.gnn_nodes_input.setPlaceholderText("comma-separated node ids")
		self.gnn_nodes_input.textChanged.connect(self._gnn_refresh_dataset_status)
		ds_form.addRow("Nodes to use:", self.gnn_nodes_input)

		self.gnn_nodes_info_label = QtWidgets.QLabel("")
		self.gnn_nodes_info_label.setWordWrap(True)
		ds_form.addRow("Selection info:", self.gnn_nodes_info_label)

		self.gnn_extra_demand = self._new_double_spin(0.0, 1e6, 1.2, decimals=4, step=0.1)
		self.gnn_extra_demand.valueChanged.connect(self._gnn_refresh_dataset_status)
		ds_form.addRow("Extra demand:", self.gnn_extra_demand)

		self.gnn_num_sims = QtWidgets.QSpinBox()
		self.gnn_num_sims.setRange(1, 1000000)
		self.gnn_num_sims.setValue(5000)
		self.gnn_num_sims.valueChanged.connect(self._gnn_refresh_dataset_status)
		ds_form.addRow("Num simulations:", self.gnn_num_sims)

		self.gnn_demand_model = QtWidgets.QComboBox()
		self.gnn_demand_model.addItems(["uniform", "dirichlet", "perturb"])
		self.gnn_demand_model.setCurrentText("dirichlet")
		self.gnn_demand_model.currentTextChanged.connect(self._gnn_refresh_dataset_status)
		ds_form.addRow("Demand model:", self.gnn_demand_model)

		ds_hash_row = QtWidgets.QHBoxLayout()
		self.gnn_ds_hash_label = QtWidgets.QLabel("—")
		ds_hash_row.addWidget(self.gnn_ds_hash_label)
		self.gnn_ds_status_label = QtWidgets.QLabel("○ Missing")
		ds_hash_row.addWidget(self.gnn_ds_status_label)
		ds_hash_row.addStretch()
		ds_form.addRow("Dataset hash:", ds_hash_row)

		self.gnn_generate_btn = QtWidgets.QPushButton("Generate Dataset")
		self.gnn_generate_btn.clicked.connect(self._gnn_generate_dataset)
		ds_form.addRow(self.gnn_generate_btn)
		run_layout.addWidget(ds_group)

		# Model group
		mdl_group = QtWidgets.QGroupBox("Model")
		mdl_form = QtWidgets.QFormLayout(mdl_group)

		self.gnn_epochs = QtWidgets.QSpinBox()
		self.gnn_epochs.setRange(1, 10000)
		self.gnn_epochs.setValue(50)
		self.gnn_epochs.valueChanged.connect(self._gnn_refresh_model_status)
		mdl_form.addRow("Epochs:", self.gnn_epochs)

		self.gnn_lr = self._new_double_spin(1e-6, 1.0, 0.001, decimals=6, step=0.0001)
		self.gnn_lr.valueChanged.connect(self._gnn_refresh_model_status)
		mdl_form.addRow("Learning rate:", self.gnn_lr)

		self.gnn_batch_size = QtWidgets.QSpinBox()
		self.gnn_batch_size.setRange(1, 4096)
		self.gnn_batch_size.setValue(32)
		self.gnn_batch_size.valueChanged.connect(self._gnn_refresh_model_status)
		mdl_form.addRow("Batch size:", self.gnn_batch_size)

		self.gnn_hidden_dim = QtWidgets.QSpinBox()
		self.gnn_hidden_dim.setRange(1, 4096)
		self.gnn_hidden_dim.setValue(64)
		self.gnn_hidden_dim.valueChanged.connect(self._gnn_refresh_model_status)
		mdl_form.addRow("Hidden dim:", self.gnn_hidden_dim)

		self.gnn_num_layers = QtWidgets.QSpinBox()
		self.gnn_num_layers.setRange(1, 32)
		self.gnn_num_layers.setValue(3)
		self.gnn_num_layers.valueChanged.connect(self._gnn_refresh_model_status)
		mdl_form.addRow("Num layers:", self.gnn_num_layers)

		self.gnn_seed = QtWidgets.QSpinBox()
		self.gnn_seed.setRange(0, 2**31 - 1)
		self.gnn_seed.setValue(42)
		self.gnn_seed.valueChanged.connect(self._gnn_refresh_model_status)
		mdl_form.addRow("Seed:", self.gnn_seed)

		mdl_hash_row = QtWidgets.QHBoxLayout()
		self.gnn_mdl_hash_label = QtWidgets.QLabel("—")
		mdl_hash_row.addWidget(self.gnn_mdl_hash_label)
		self.gnn_mdl_status_label = QtWidgets.QLabel("○ Missing")
		mdl_hash_row.addWidget(self.gnn_mdl_status_label)
		mdl_hash_row.addStretch()
		mdl_form.addRow("Model hash:", mdl_hash_row)

		self.gnn_train_btn = QtWidgets.QPushButton("Train Model")
		self.gnn_train_btn.setEnabled(False)
		self.gnn_train_btn.clicked.connect(self._gnn_train_model)
		mdl_form.addRow(self.gnn_train_btn)
		run_layout.addWidget(mdl_group)

		# Control group (Stop button)
		ctrl_group = QtWidgets.QGroupBox("Control")
		ctrl_layout = QtWidgets.QHBoxLayout(ctrl_group)
		self.gnn_stop_btn = QtWidgets.QPushButton("Stop")
		self.gnn_stop_btn.setStyleSheet("background-color: #ff6b6b; color: white; font-weight: bold;")
		self.gnn_stop_btn.setEnabled(False)
		self.gnn_stop_btn.clicked.connect(self._gnn_stop_worker)
		ctrl_layout.addWidget(self.gnn_stop_btn)
		self.gnn_progress_label = QtWidgets.QLabel("Idle")
		ctrl_layout.addWidget(self.gnn_progress_label)
		self.gnn_progress_bar = QtWidgets.QProgressBar()
		self.gnn_progress_bar.setMinimumWidth(260)
		self.gnn_progress_bar.setRange(0, 100)
		self.gnn_progress_bar.setValue(0)
		self.gnn_progress_bar.setFormat("Idle")
		ctrl_layout.addWidget(self.gnn_progress_bar, 1)
		ctrl_layout.addStretch()
		run_layout.addWidget(ctrl_group)

		# Log
		log_group = QtWidgets.QGroupBox("Log")
		log_layout = QtWidgets.QVBoxLayout(log_group)
		self.gnn_log = QtWidgets.QPlainTextEdit()
		self.gnn_log.setReadOnly(True)
		self.gnn_log.setMaximumBlockCount(2000)
		self.gnn_log.setMinimumHeight(120)
		log_layout.addWidget(self.gnn_log)
		run_layout.addWidget(log_group)
		run_layout.addStretch()

		gnn_tabs.addTab(run_scroll, "Run")

		# ── Compare sub-page ─────────────────────────────────────────────────
		cmp_scroll = QtWidgets.QScrollArea()
		cmp_scroll.setWidgetResizable(True)
		cmp_widget = QtWidgets.QWidget()
		cmp_scroll.setWidget(cmp_widget)
		cmp_layout = QtWidgets.QVBoxLayout(cmp_widget)
		cmp_layout.setContentsMargins(4, 4, 4, 4)

		# Model selectors
		sel_group = QtWidgets.QGroupBox("Models")
		sel_form = QtWidgets.QFormLayout(sel_group)
		self.gnn_model_a_combo = QtWidgets.QComboBox()
		sel_form.addRow("Model A:", self.gnn_model_a_combo)
		self.gnn_model_b_combo = QtWidgets.QComboBox()
		sel_form.addRow("Model B:", self.gnn_model_b_combo)
		refresh_btn = QtWidgets.QPushButton("Refresh model list")
		refresh_btn.clicked.connect(self._gnn_populate_model_combos)
		sel_form.addRow(refresh_btn)
		cmp_layout.addWidget(sel_group)

		# Shared test set
		test_group = QtWidgets.QGroupBox("Shared Test Set")
		test_form = QtWidgets.QFormLayout(test_group)
		self.gnn_test_sims = QtWidgets.QSpinBox()
		self.gnn_test_sims.setRange(1, 100000)
		self.gnn_test_sims.setValue(1000)
		self.gnn_test_sims.valueChanged.connect(self._gnn_refresh_test_status)
		test_form.addRow("Test simulations:", self.gnn_test_sims)
		self.gnn_test_seed = QtWidgets.QSpinBox()
		self.gnn_test_seed.setRange(0, 2**31 - 1)
		self.gnn_test_seed.setValue(9999)
		self.gnn_test_seed.valueChanged.connect(self._gnn_refresh_test_status)
		test_form.addRow("Test seed:", self.gnn_test_seed)
		self.gnn_test_demand_model = QtWidgets.QComboBox()
		self.gnn_test_demand_model.addItems(["uniform", "dirichlet", "perturb"])
		self.gnn_test_demand_model.currentTextChanged.connect(self._gnn_refresh_test_status)
		test_form.addRow("Demand model:", self.gnn_test_demand_model)
		test_hash_row = QtWidgets.QHBoxLayout()
		self.gnn_test_hash_label = QtWidgets.QLabel("—")
		test_hash_row.addWidget(self.gnn_test_hash_label)
		self.gnn_test_status_label = QtWidgets.QLabel("○ Missing")
		test_hash_row.addWidget(self.gnn_test_status_label)
		test_hash_row.addStretch()
		test_form.addRow("Test set hash:", test_hash_row)
		self.gnn_generate_test_btn = QtWidgets.QPushButton("Generate shared test set")
		self.gnn_generate_test_btn.clicked.connect(self._gnn_generate_test_set)
		test_form.addRow(self.gnn_generate_test_btn)
		cmp_layout.addWidget(test_group)

		# Comparison controls
		comp_ctrl_group = QtWidgets.QGroupBox("Comparison")
		comp_ctrl_form = QtWidgets.QFormLayout(comp_ctrl_group)
		self.gnn_reconstruction = QtWidgets.QComboBox()
		self.gnn_reconstruction.addItems(["algebraic", "wntr"])
		comp_ctrl_form.addRow("Reconstruction:", self.gnn_reconstruction)
		cmp_hash_row = QtWidgets.QHBoxLayout()
		self.gnn_cmp_hash_label = QtWidgets.QLabel("—")
		cmp_hash_row.addWidget(self.gnn_cmp_hash_label)
		self.gnn_cmp_status_label = QtWidgets.QLabel("○ Missing")
		cmp_hash_row.addWidget(self.gnn_cmp_status_label)
		cmp_hash_row.addStretch()
		comp_ctrl_form.addRow("Comparison hash:", cmp_hash_row)
		self.gnn_run_comparison_btn = QtWidgets.QPushButton("Run Comparison")
		self.gnn_run_comparison_btn.setEnabled(False)
		self.gnn_run_comparison_btn.clicked.connect(self._gnn_run_comparison)
		comp_ctrl_form.addRow(self.gnn_run_comparison_btn)
		cmp_layout.addWidget(comp_ctrl_group)

		# Results table
		results_group = QtWidgets.QGroupBox("Results")
		results_layout = QtWidgets.QVBoxLayout(results_group)
		self.gnn_results_table = QtWidgets.QTableWidget()
		self.gnn_results_table.setColumnCount(3)
		self.gnn_results_table.setHorizontalHeaderLabels(["Metric", "Model A", "Model B"])
		self.gnn_results_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
		self.gnn_results_table.horizontalHeader().setStretchLastSection(True)
		results_layout.addWidget(self.gnn_results_table)
		self.gnn_export_btn = QtWidgets.QPushButton("Export results to JSON")
		self.gnn_export_btn.setEnabled(False)
		self.gnn_export_btn.clicked.connect(self._gnn_export_results)
		results_layout.addWidget(self.gnn_export_btn)
		self.gnn_example_btn = QtWidgets.QPushButton("Show pressure-demand example")
		self.gnn_example_btn.setEnabled(False)
		self.gnn_example_btn.clicked.connect(self._gnn_show_pressure_demand_example)
		results_layout.addWidget(self.gnn_example_btn)
		cmp_layout.addWidget(results_group)
		cmp_layout.addStretch()

		gnn_tabs.addTab(cmp_scroll, "Compare")

		self._gnn_tab_index = self.tabs.addTab(outer, "GNN")
		self._gnn_dataset_hash: str = ""
		self._gnn_model_hash: str = ""
		self._gnn_dataset_dir: str = ""
		self._gnn_worker: "GNNDatasetWorker | GNNModelWorker | GNNCompareWorker | None" = None
		self._gnn_results: dict = {}
		self._gnn_active_task: str = ""
		self._gnn_refresh_dataset_status()

	def _gnn_start_progress(self, task_name: str) -> None:
		self._gnn_active_task = task_name
		self.gnn_progress_label.setText(task_name)
		self.gnn_progress_bar.setRange(0, 0)  # busy / indeterminate
		self.gnn_progress_bar.setFormat(f"{task_name}...")
		self._gnn_last_log_time = time.monotonic()
		self._gnn_watchdog_warned = False
		self._gnn_watchdog_timer.start()

	def _gnn_finish_progress(self, success: bool) -> None:
		self._gnn_watchdog_timer.stop()
		if self._gnn_active_task:
			status = "done" if success else "failed/cancelled"
			self.gnn_progress_label.setText(f"{self._gnn_active_task}: {status}")
		else:
			self.gnn_progress_label.setText("Idle")
		self.gnn_progress_bar.setRange(0, 100)
		self.gnn_progress_bar.setValue(100 if success else 0)
		self.gnn_progress_bar.setFormat("Done" if success else "Stopped")
		self._gnn_active_task = ""
		self._gnn_watchdog_warned = False

	def _gnn_watchdog_tick(self) -> None:
		"""Warn if a worker is running but no log/progress has appeared for a while."""
		if self._gnn_worker is None or not self._gnn_worker.isRunning():
			self._gnn_watchdog_warned = False
			return
		if self._gnn_active_task != "Training model":
			return
		inactive_for = time.monotonic() - self._gnn_last_log_time
		if inactive_for < 120.0 or self._gnn_watchdog_warned:
			return
		mins = int(inactive_for // 60)
		self._gnn_watchdog_warned = True
		self.gnn_log.appendPlainText(
			f"WARNING: no training log output for ~{mins} min. "
			"Training may be very slow or stalled; use Stop to cancel safely."
		)
		self.gnn_progress_label.setText(f"{self._gnn_active_task}: no output for ~{mins} min")

	def _gnn_on_worker_log(self, line: str) -> None:
		self.gnn_log.appendPlainText(line)
		self._gnn_last_log_time = time.monotonic()
		self._gnn_watchdog_warned = False
		# If worker logs include x/y information, switch to determinate mode.
		m = re.search(r"(?:Progress:\s*|Epoch\s+)(\d+)\s*/\s*(\d+)", line)
		if not m:
			m = re.search(r"\b(\d+)\s*/\s*(\d+)\b", line)
		if m:
			current = int(m.group(1))
			total = int(m.group(2))
		else:
			# gnn_model prints: "Epoch 000, Train Loss: ..., Val Loss: ..."
			epoch_only = re.search(r"\bEpoch\s+(\d+)\b", line)
			if not epoch_only or self._gnn_active_task != "Training model":
				return
			current = int(epoch_only.group(1)) + 1
			total = int(self.gnn_epochs.value())
		if total <= 0:
			return
		if self.gnn_progress_bar.minimum() == 0 and self.gnn_progress_bar.maximum() == 0:
			self.gnn_progress_bar.setRange(0, total)
		elif self.gnn_progress_bar.maximum() != total:
			self.gnn_progress_bar.setRange(0, total)
		self.gnn_progress_bar.setValue(max(0, min(current, total)))
		task = self._gnn_active_task or "Running"
		self.gnn_progress_bar.setFormat(f"{task}: {current}/{total}")

	def _gnn_default_nodes(self) -> List[str]:
		wdn = self.wdn_input.currentText().strip()
		if not wdn:
			return []
		json_path = os.path.join(ROOT_DIR, "wdn", f"{wdn}.json")
		if not os.path.isfile(json_path):
			return []
		try:
			with open(json_path, encoding="utf-8") as f:
				cfg = json.load(f)
			return [str(n) for n in cfg.get("measurement_nodes", [])]
		except (OSError, json.JSONDecodeError):
			return []

	def _gnn_current_nodes(self) -> List[str]:
		default_nodes = sorted(self._gnn_default_nodes())
		if self.gnn_use_default_nodes.isChecked():
			return default_nodes
		text = self.gnn_nodes_input.text()
		tokens = [t.strip() for t in text.split(",") if t.strip()]
		return sorted(dict.fromkeys(tokens))

	def _gnn_update_nodes_info(self) -> None:
		default_nodes = sorted(self._gnn_default_nodes())
		current_nodes = self._gnn_current_nodes()
		default_count = len(default_nodes)
		current_count = len(current_nodes)

		self.gnn_nodes_input.setEnabled(not self.gnn_use_default_nodes.isChecked())

		if self.gnn_use_default_nodes.isChecked():
			self.gnn_nodes_info_label.setText(
				f"Using the default placement with {default_count} node(s): "
				+ (", ".join(default_nodes) if default_nodes else "(none)")
			)
			return

		if current_count == 0:
			self.gnn_nodes_info_label.setText(
				f"Manual placement selected. Enter node ids above. The default placement uses {default_count} node(s)."
			)
			return

		delta = current_count - default_count
		if delta < 0:
			comparison = f"{abs(delta)} fewer than"
		elif delta > 0:
			comparison = f"{delta} more than"
		else:
			comparison = "the same number as"

		self.gnn_nodes_info_label.setText(
			f"Manual placement uses {current_count} node(s), {comparison} the default placement ({default_count})."
		)

	def _gnn_nodes_display_text(self, nodes: List[str], default_nodes: List[str]) -> str:
		n = sorted(str(x) for x in nodes)
		d = sorted(str(x) for x in default_nodes)
		if n == d and n:
			return "standard"
		if not n:
			return "none"
		return "{" + ",".join(n) + "}"

	def _gnn_refresh_dataset_status(self) -> None:
		from old.gnn_cache import dataset_inputs, dataset_hash, find_dataset
		self._gnn_update_nodes_info()
		wdn = self.wdn_input.currentText().strip()
		current_nodes = self._gnn_current_nodes()
		if not current_nodes:
			self._gnn_dataset_hash = ""
			self._gnn_dataset_dir = ""
			self.gnn_ds_hash_label.setText("—")
			self.gnn_ds_status_label.setText("○ Missing")
			self.gnn_generate_btn.setEnabled(False)
			self._gnn_refresh_model_status()
			return
		inp = dataset_inputs(
			wdn=wdn,
			measurement_nodes=current_nodes,
			extra_demand=self.gnn_extra_demand.value(),
			num_simulations=self.gnn_num_sims.value(),
			demand_model=self.gnn_demand_model.currentText(),
		)
		h = dataset_hash(inp)
		self._gnn_dataset_hash = h
		exists = find_dataset(wdn, h) is not None
		self.gnn_ds_hash_label.setText(h[:8] + "...")
		self.gnn_ds_status_label.setText("● Exists" if exists else "○ Missing")
		self.gnn_generate_btn.setEnabled(not exists)
		if exists:
			self._gnn_dataset_dir = find_dataset(wdn, h) or ""
		self._gnn_refresh_model_status()

	def _gnn_refresh_model_status(self) -> None:
		from old.gnn_cache import model_inputs, model_hash, find_model, find_dataset
		if self._gnn_worker is not None and self._gnn_worker.isRunning():
			self.gnn_train_btn.setEnabled(False)
			return
		wdn = self.wdn_input.currentText().strip()
		if not self._gnn_dataset_hash:
			self.gnn_mdl_hash_label.setText("—")
			self.gnn_mdl_status_label.setText("○ Missing")
			self.gnn_train_btn.setEnabled(False)
			return
		inp = model_inputs(
			dataset_hash=self._gnn_dataset_hash,
			epochs=self.gnn_epochs.value(),
			lr=self.gnn_lr.value(),
			batch_size=self.gnn_batch_size.value(),
			hidden_dim=self.gnn_hidden_dim.value(),
			num_layers=self.gnn_num_layers.value(),
			seed=self.gnn_seed.value(),
		)
		h = model_hash(inp)
		self._gnn_model_hash = h
		exists = find_model(wdn, h) is not None
		ds_ready = bool(self._gnn_dataset_dir) or (find_dataset(wdn, self._gnn_dataset_hash) is not None)
		self.gnn_mdl_hash_label.setText(h[:8] + "...")
		self.gnn_mdl_status_label.setText("● Exists" if exists else "○ Missing")
		self.gnn_train_btn.setEnabled(ds_ready and not exists)

	def _gnn_refresh_test_status(self) -> None:
		from old.gnn_cache import test_set_inputs, test_set_hash, find_test_set
		wdn = self.wdn_input.currentText().strip()
		inp = test_set_inputs(
			wdn=wdn,
			extra_demand=self.gnn_extra_demand.value(),
			num_simulations=self.gnn_test_sims.value(),
			demand_model=self.gnn_test_demand_model.currentText(),
			seed=self.gnn_test_seed.value(),
		)
		h = test_set_hash(inp)
		exists = find_test_set(wdn, h) is not None
		self.gnn_test_hash_label.setText(h[:8] + "...")
		self.gnn_test_status_label.setText("● Exists" if exists else "○ Missing")
		self.gnn_generate_test_btn.setEnabled(not exists)

	def _gnn_generate_test_set(self) -> None:
		if self._gnn_worker is not None and self._gnn_worker.isRunning():
			return
		wdn = self.wdn_input.currentText().strip()
		self.gnn_log.appendPlainText(f"Starting shared test-set generation for {wdn}...")
		self._gnn_start_progress("Generating shared test set")
		self.gnn_generate_test_btn.setEnabled(False)
		self.gnn_stop_btn.setEnabled(True)
		worker = GNNTestSetWorker(
			wdn_name=wdn,
			extra_demand=self.gnn_extra_demand.value(),
			num_simulations=self.gnn_test_sims.value(),
			demand_model=self.gnn_test_demand_model.currentText(),
			seed=self.gnn_test_seed.value(),
			parent=self,
		)
		worker.log_line.connect(self._gnn_on_worker_log)
		worker.finished.connect(self._gnn_on_test_set_done)
		self._gnn_worker = worker
		worker.start()

	def _gnn_on_test_set_done(self, success: bool, artifact_dir: str) -> None:
		self.gnn_stop_btn.setEnabled(False)
		self._gnn_finish_progress(success)
		if success:
			self.gnn_log.appendPlainText(f"Shared test set ready: {artifact_dir}")
		else:
			self.gnn_log.appendPlainText("Shared test-set generation failed or was cancelled.")
		self._gnn_refresh_test_status()

	def _gnn_generate_dataset(self) -> None:
		if self._gnn_worker is not None and self._gnn_worker.isRunning():
			return
		wdn = self.wdn_input.currentText().strip()
		self.gnn_log.appendPlainText(f"Starting dataset generation for {wdn}...")
		self._gnn_start_progress("Generating dataset")
		self.gnn_generate_btn.setEnabled(False)
		self.gnn_stop_btn.setEnabled(True)
		worker = GNNDatasetWorker(
			wdn_name=wdn,
			measurement_nodes=self._gnn_current_nodes(),
			extra_demand=self.gnn_extra_demand.value(),
			num_simulations=self.gnn_num_sims.value(),
			demand_model=self.gnn_demand_model.currentText(),
			node_label_threshold=0.0,
			parent=self,
		)
		worker.log_line.connect(self._gnn_on_worker_log)
		worker.finished.connect(self._gnn_on_dataset_done)
		self._gnn_worker = worker
		worker.start()

	def _gnn_on_dataset_done(self, success: bool, artifact_dir: str) -> None:
		self.gnn_stop_btn.setEnabled(False)
		self._gnn_finish_progress(success)
		if success:
			self.gnn_log.appendPlainText(f"Dataset ready: {artifact_dir}")
			self._gnn_dataset_dir = artifact_dir
		else:
			self.gnn_log.appendPlainText("Dataset generation failed or was cancelled.")
		self._gnn_refresh_dataset_status()

	def _gnn_train_model(self) -> None:
		if self._gnn_worker is not None and self._gnn_worker.isRunning():
			return
		wdn = self.wdn_input.currentText().strip()
		if not self._gnn_dataset_dir:
			from old.gnn_cache import find_dataset
			self._gnn_dataset_dir = find_dataset(wdn, self._gnn_dataset_hash) or ""
		if not self._gnn_dataset_dir:
			self.gnn_log.appendPlainText("Dataset not found — generate it first.")
			return
		self.gnn_log.appendPlainText(f"Starting model training for {wdn}...")
		self._gnn_start_progress("Training model")
		self.gnn_train_btn.setEnabled(False)
		self.gnn_stop_btn.setEnabled(True)
		worker = GNNModelWorker(
			wdn_name=wdn,
			d_hash=self._gnn_dataset_hash,
			dataset_dir=self._gnn_dataset_dir,
			epochs=self.gnn_epochs.value(),
			lr=self.gnn_lr.value(),
			batch_size=self.gnn_batch_size.value(),
			hidden_dim=self.gnn_hidden_dim.value(),
			num_layers=self.gnn_num_layers.value(),
			seed=self.gnn_seed.value(),
			parent=self,
		)
		worker.log_line.connect(self._gnn_on_worker_log)
		worker.finished.connect(self._gnn_on_model_done)
		self._gnn_worker = worker
		worker.start()

	def _gnn_on_model_done(self, success: bool, artifact_dir: str) -> None:
		self.gnn_stop_btn.setEnabled(False)
		self._gnn_finish_progress(success)
		if success:
			self.gnn_log.appendPlainText(f"Model ready: {artifact_dir}")
		else:
			self.gnn_log.appendPlainText("Model training failed or was cancelled.")
		self._gnn_refresh_model_status()
		self._gnn_populate_model_combos()

	def _gnn_populate_model_combos(self) -> None:
		from old.gnn_cache import list_models, find_dataset
		from pathlib import Path
		wdn = self.wdn_input.currentText().strip()
		models = list_models(wdn)
		default_nodes = self._gnn_default_nodes()
		for combo in [self.gnn_model_a_combo, self.gnn_model_b_combo]:
			combo.clear()
			for m in models:
				inputs = m.get("inputs", {})
				ds_hash = inputs.get("dataset_hash", "?")[:8]
				node_text = "unknown"
				ds_full_hash = inputs.get("dataset_hash")
				if ds_full_hash:
					ds_dir = find_dataset(wdn, ds_full_hash)
					if ds_dir:
						manifest_path = Path(ds_dir) / "manifest.json"
						if manifest_path.exists():
							try:
								with open(manifest_path, encoding="utf-8") as f:
									ds_manifest = json.load(f)
								nodes = ds_manifest.get("inputs", {}).get("measurement_nodes", [])
								node_text = self._gnn_nodes_display_text(nodes, default_nodes)
							except (OSError, json.JSONDecodeError):
								pass
				combo.addItem(
					f"{m['hash'][:8]}... (ds:{ds_hash}, nodes:{node_text})",
					userData=m["hash"],
				)
		has_two = self.gnn_model_a_combo.count() >= 1 and self.gnn_model_b_combo.count() >= 1
		self.gnn_run_comparison_btn.setEnabled(has_two)

	def _gnn_stop_worker(self) -> None:
		"""Cancel the currently running worker."""
		if self._gnn_worker and self._gnn_worker.isRunning():
			self.gnn_log.appendPlainText("Requesting cancellation...")
			self._gnn_worker.cancel()
			self.gnn_stop_btn.setEnabled(False)
			self.gnn_progress_label.setText("Cancelling...")
			self.gnn_progress_bar.setFormat("Cancelling...")

	def _gnn_run_comparison(self) -> None:
		if self._gnn_worker is not None and self._gnn_worker.isRunning():
			return
		if self.gnn_model_a_combo.count() < 1 or self.gnn_model_b_combo.count() < 1:
			QtWidgets.QMessageBox.warning(self, "Missing Models", "Select both Model A and Model B.")
			return

		wdn = self.wdn_input.currentText().strip()
		model_a_hash = self.gnn_model_a_combo.currentData()
		model_b_hash = self.gnn_model_b_combo.currentData()

		if not model_a_hash or not model_b_hash:
			QtWidgets.QMessageBox.warning(self, "Invalid Selection", "Could not get model hashes.")
			return

		from old.gnn_cache import find_model, find_dataset
		from old.gnn_cache import test_set_inputs, test_set_hash, find_test_set

		model_a_dir = find_model(wdn, model_a_hash)
		model_b_dir = find_model(wdn, model_b_hash)
		if not model_a_dir or not model_b_dir:
			QtWidgets.QMessageBox.warning(self, "Model Not Found", "Could not locate one or both models.")
			return

		# Get dataset dirs from model manifests
		def _get_dataset_dir(model_h):
			from pathlib import Path
			manifest = Path(find_model(wdn, model_h)) / "manifest.json"
			if manifest.exists():
				try:
					with open(manifest, encoding="utf-8") as f:
						m = json.load(f)
					ds_h = m.get("inputs", {}).get("dataset_hash")
					if ds_h:
						return find_dataset(wdn, ds_h)
				except (OSError, json.JSONDecodeError):
					pass
			return None

		dataset_a_dir = _get_dataset_dir(model_a_hash)
		dataset_b_dir = _get_dataset_dir(model_b_hash)
		if not dataset_a_dir or not dataset_b_dir:
			QtWidgets.QMessageBox.warning(self, "Dataset Not Found", "Could not locate dataset for one or both models.")
			return

		test_set_hash_value = None
		test_inp = test_set_inputs(
			wdn=wdn,
			extra_demand=self.gnn_extra_demand.value(),
			num_simulations=self.gnn_test_sims.value(),
			demand_model=self.gnn_test_demand_model.currentText(),
			seed=self.gnn_test_seed.value(),
		)
		test_set_hash_value = test_set_hash(test_inp)
		if find_test_set(wdn, test_set_hash_value):
			self.gnn_log.appendPlainText(f"Using shared test set: {test_set_hash_value[:8]}")

		self.gnn_log.appendPlainText(f"Starting comparison: {model_a_hash[:8]} vs {model_b_hash[:8]}")
		self._gnn_start_progress("Running comparison")
		self.gnn_run_comparison_btn.setEnabled(False)
		self.gnn_stop_btn.setEnabled(True)

		worker = GNNCompareWorker(
			wdn_name=wdn,
			model_a_hash=model_a_hash,
			model_a_dir=model_a_dir,
			dataset_a_dir=dataset_a_dir,
			model_b_hash=model_b_hash,
			model_b_dir=model_b_dir,
			dataset_b_dir=dataset_b_dir,
			test_set_hash=test_set_hash_value,
			demand_reconstruction=self.gnn_reconstruction.currentText(),
			parent=self,
		)
		worker.log_line.connect(self._gnn_on_worker_log)
		worker.finished.connect(self._gnn_on_comparison_done)
		self._gnn_worker = worker
		worker.start()

	def _gnn_on_comparison_done(self, success: bool, artifact_dir: str) -> None:
		self.gnn_stop_btn.setEnabled(False)
		self._gnn_finish_progress(success)
		if success:
			self.gnn_log.appendPlainText(f"Comparison complete: {artifact_dir}")
			self._gnn_load_results(artifact_dir)
		else:
			self.gnn_log.appendPlainText("Comparison failed or was cancelled.")
		self.gnn_run_comparison_btn.setEnabled(True)

	def _gnn_load_results(self, artifact_dir: str) -> None:
		"""Load comparison results from results.json and populate the table."""
		from pathlib import Path
		results_path = Path(artifact_dir) / "results.json"
		if not results_path.exists():
			self.gnn_log.appendPlainText(f"Results file not found: {results_path}")
			return
		try:
			with open(results_path, encoding="utf-8") as f:
				results = json.load(f)
			self._gnn_results = results
			self._gnn_populate_results_table(results)
		except (OSError, json.JSONDecodeError) as e:
			self.gnn_log.appendPlainText(f"Error loading results: {e}")

	def _gnn_populate_results_table(self, results: dict) -> None:
		"""Populate the results QTableWidget from comparison results."""
		self.gnn_results_table.setRowCount(0)
		metrics_a = results.get("metrics_a", {})
		metrics_b = results.get("metrics_b", {})
		label_a = results.get("label_a", "Model A")
		label_b = results.get("label_b", "Model B")

		# Update column headers
		self.gnn_results_table.setHorizontalHeaderLabels(["Metric", label_a, label_b])

		# Common metrics to display
		metric_keys = [
			"r2", "mse", "mae", "rmse", "nmae",
			"mape", "max_pe", "accuracy_10pct", "accuracy_5pct",
		]

		row = 0
		for key in metric_keys:
			val_a = metrics_a.get(key)
			val_b = metrics_b.get(key)
			if val_a is None and val_b is None:
				continue

			self.gnn_results_table.insertRow(row)
			self.gnn_results_table.setItem(row, 0, QtWidgets.QTableWidgetItem(key))
			self.gnn_results_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{val_a:.4f}" if val_a is not None else "—"))
			self.gnn_results_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{val_b:.4f}" if val_b is not None else "—"))
			row += 1

		# Demand reconstruction metrics
		demand_metrics_a = results.get("demand_metrics_a") or {}
		demand_metrics_b = results.get("demand_metrics_b") or {}
		demand_keys = ["mae", "rmse", "r2", "mape"]
		for key in demand_keys:
			val_a = demand_metrics_a.get(key)
			val_b = demand_metrics_b.get(key)
			if val_a is None and val_b is None:
				continue
			self.gnn_results_table.insertRow(row)
			self.gnn_results_table.setItem(row, 0, QtWidgets.QTableWidgetItem(f"demand:{key}"))
			self.gnn_results_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{val_a:.4f}" if val_a is not None else "—"))
			self.gnn_results_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{val_b:.4f}" if val_b is not None else "—"))
			row += 1

		self.gnn_export_btn.setEnabled(bool(self._gnn_results))
		self.gnn_example_btn.setEnabled(bool(self._gnn_results))

	def _gnn_show_pressure_demand_example(self) -> None:
		if not self._gnn_results:
			QtWidgets.QMessageBox.information(self, "No Comparison Results", "Run a comparison first.")
			return

		try:
			import pickle
			from pathlib import Path
			import numpy as np
			import torch
			from old.compare_pressure import GCN, _denorm, _headloss_n_from_inp, _reconstruct_demands, _run_inference
			from old.gnn_cache import find_dataset, find_model
		except Exception as exc:
			QtWidgets.QMessageBox.critical(self, "Unavailable", f"Cannot load comparison helpers: {exc}")
			return

		wdn = str(self._gnn_results.get("wdn_name") or self.wdn_input.currentText().strip())
		model_a_hash = str(self._gnn_results.get("model_a_hash") or self.gnn_model_a_combo.currentData() or "")
		model_b_hash = str(self._gnn_results.get("model_b_hash") or self.gnn_model_b_combo.currentData() or "")
		if not wdn or not model_a_hash or not model_b_hash:
			QtWidgets.QMessageBox.warning(self, "Missing Context", "Comparison metadata is incomplete. Run a fresh comparison first.")
			return

		def _resolve_saved_dir(path_value: str) -> str:
			"""Resolve saved artifact paths from results.json across machines."""
			p = str(path_value or "").strip()
			if not p:
				return ""
			if os.path.isdir(p):
				return p
			if not os.path.isabs(p):
				candidate = os.path.join(ROOT_DIR, p)
				if os.path.isdir(candidate):
					return candidate
			# Backward compatibility: old results stored absolute paths from another machine.
			marker = f"{os.sep}old{os.sep}data{os.sep}"
			if marker in p:
				suffix = p.split(marker, 1)[1]
				candidate = os.path.join(ROOT_DIR, "old", "data", suffix)
				if os.path.isdir(candidate):
					return candidate
			return ""

		def _dataset_dir(model_hash: str, fallback_key: str) -> str:
			dir_path = _resolve_saved_dir(str(self._gnn_results.get(fallback_key) or ""))
			if dir_path:
				return dir_path
			model_dir = _resolve_saved_dir(str(self._gnn_results.get(fallback_key.replace("dataset", "model")) or ""))
			if not model_dir and model_hash:
				model_dir = find_model(wdn, model_hash) or ""
			if not model_dir:
				return ""
			manifest = os.path.join(model_dir, "manifest.json")
			if not os.path.isfile(manifest):
				return ""
			try:
				with open(manifest, encoding="utf-8") as f:
					manifest_data = json.load(f)
				dataset_hash = manifest_data.get("inputs", {}).get("dataset_hash")
				if dataset_hash:
					return find_dataset(wdn, dataset_hash) or ""
			except (OSError, json.JSONDecodeError):
				return ""
			return ""

		model_a_dir = _resolve_saved_dir(str(self._gnn_results.get("model_a_dir") or "")) or str(find_model(wdn, model_a_hash) or "")
		model_b_dir = _resolve_saved_dir(str(self._gnn_results.get("model_b_dir") or "")) or str(find_model(wdn, model_b_hash) or "")
		dataset_a_dir = _dataset_dir(model_a_hash, "dataset_a_dir")
		dataset_b_dir = _dataset_dir(model_b_hash, "dataset_b_dir")
		if not model_a_dir or not model_b_dir or not dataset_a_dir or not dataset_b_dir:
			QtWidgets.QMessageBox.warning(self, "Missing Artifacts", "Could not resolve model/dataset paths for the example plot.")
			return

		def _load_model(model_dir: str, input_dim: int):
			model = GCN(dim_in=input_dim, dim_h=256, dim_out=1)
			state = torch.load(os.path.join(model_dir, "best_model.pt"), map_location="cpu")
			model.load_state_dict(state)
			model.eval()
			return model

		def _load_sample_payload(dataset_dir: str, model_dir: str):
			data_dir = os.path.join(dataset_dir, "data_generator") if os.path.isdir(os.path.join(dataset_dir, "data_generator")) else dataset_dir
			stats_path = os.path.join(data_dir, "dataset_stats.json")
			artifacts_path = os.path.join(data_dir, "evaluation_artifacts.json")
			test_path = os.path.join(data_dir, "test_dataset.pt")
			graph_path = os.path.join(data_dir, "graph_with_measurements.pickle")
			with open(stats_path, encoding="utf-8") as f:
				stats = json.load(f)
			with open(artifacts_path, encoding="utf-8") as f:
				artifacts = json.load(f)
			with open(graph_path, "rb") as f:
				graph = pickle.load(f)
			dataset = torch.load(test_path, weights_only=False)
			return data_dir, stats, artifacts, graph, dataset

		try:
			_, stats_a, artifacts_a, graph_a, dataset_a = _load_sample_payload(dataset_a_dir, model_a_dir)
			_, stats_b, artifacts_b, graph_b, dataset_b = _load_sample_payload(dataset_b_dir, model_b_dir)
		except Exception as exc:
			QtWidgets.QMessageBox.critical(self, "Load Failed", f"Could not load comparison artifacts: {exc}")
			return

		if not dataset_a or not dataset_b:
			QtWidgets.QMessageBox.warning(self, "No Data", "Comparison datasets are empty.")
			return

		def _prepare_example(preds_all, actuals_all, sample_idx: int, stats: dict, artifacts: dict, graph, dataset):
			def _clean_node_name(node_name: str) -> str:
				s = str(node_name)
				return s[5:] if s.startswith("meas_") else s

			def _node_sort_key(node_name: str):
				s = str(node_name)
				if s.isdigit():
					return (0, int(s))
				return (1, s)

			pred_phys = _denorm(preds_all[sample_idx: sample_idx + 1], stats)[0]
			actual_phys = _denorm(actuals_all[sample_idx: sample_idx + 1], stats)[0]
			node_mapping = artifacts.get("node_mapping", {})
			measurement_nodes = set()
			for n in artifacts.get("measurement_nodes", []) or []:
				measurement_nodes.add(_clean_node_name(n))
			for n in stats.get("measurement_nodes", []) or []:
				measurement_nodes.add(_clean_node_name(n))

			reservoir_nodes = set()
			for n in artifacts.get("reservoir_nodes", []) or []:
				reservoir_nodes.add(_clean_node_name(n))
			if stats.get("reservoir_node") is not None:
				reservoir_nodes.add(_clean_node_name(stats.get("reservoir_node")))

			display_nodes = []
			for i in range(len(actual_phys)):
				node_name = str(node_mapping.get(str(i), i))
				if node_name.startswith("meas_"):
					continue
				base = _clean_node_name(node_name)
				if base in reservoir_nodes:
					continue
				label = base
				if base in measurement_nodes:
					label = f"{label} (M)"
				if base in reservoir_nodes:
					label = f"{label}R"
				display_nodes.append((i, base, label))

			display_nodes.sort(key=lambda t: _node_sort_key(t[1]))
			display_indices = np.array([t[0] for t in display_nodes], dtype=int)
			p_labels = [t[2] for t in display_nodes]

			# Clamp known boundary pressures (measurement + reservoir) to ground truth.
			for idx, node_name in node_mapping.items():
				i = int(idx)
				if i >= len(pred_phys) or i >= len(actual_phys):
					continue
				base = _clean_node_name(node_name)
				if base in measurement_nodes or base in reservoir_nodes:
					pred_phys[i] = actual_phys[i]

			inp_path = os.path.join(ROOT_DIR, "wdn", f"{wdn}.inp")
			headloss_n = _headloss_n_from_inp(inp_path)
			from step1_io import load_inp_network
			network = load_inp_network(inp_path)
			elev_by_node = {str(nid): float(node.elevation_m) for nid, node in network.nodes.items()}
			
			# Apply reservoir head override to actual/predicted pressures for visualization
			# (matching what _reconstruct_demands does internally)
			if stats and stats.get("reservoir_head") is not None:
				res_node_name = str(stats.get("reservoir_node", ""))
				if res_node_name:
					node_mapping = artifacts.get("node_mapping", {})
					# Find the index of the reservoir node
					node_idx_res = None
					for idx_str, node_name in node_mapping.items():
						if str(node_name) == res_node_name:
							node_idx_res = int(idx_str)
							break
					if node_idx_res is not None:
						# Load network to get elevation
						node_elev = 0.0
						if res_node_name in network.nodes:
							node_elev = float(network.nodes[res_node_name].elevation_m)
						head_res = float(stats["reservoir_head"])
						actual_phys[node_idx_res] = head_res - node_elev
						pred_phys[node_idx_res] = head_res - node_elev

			actual_head = np.array(actual_phys, copy=True)
			pred_head = np.array(pred_phys, copy=True)
			for idx_str, node_name in node_mapping.items():
				i = int(idx_str)
				if i >= len(actual_head) or i >= len(pred_head):
					continue
				elev = elev_by_node.get(_clean_node_name(node_name), 0.0)
				actual_head[i] += elev
				pred_head[i] += elev

			res_pressure_text = "n/a"
			if stats and stats.get("reservoir_head") is not None:
				res_node = str(stats.get("reservoir_node", ""))
				if res_node:
					res_elev = elev_by_node.get(res_node, 0.0)
					res_pressure = float(stats["reservoir_head"]) - float(res_elev)
					res_pressure_text = f"{res_pressure:.2f} m"
			
			d_pred, d_actual, jn_list = _reconstruct_demands(
				pred_phys[np.newaxis, :], [dataset[sample_idx]], artifacts, graph, headloss_n, inp_path, stats=stats
			)
			if d_pred is None or d_actual is None:
				raise RuntimeError("Demand reconstruction failed for the example plot.")
			d_pred = d_pred[0]
			d_actual = d_actual[0]
			worst_idx = int(np.argmax(np.abs(d_pred - d_actual)))
			demand_nodes = [(i, str(jn_list[i])) for i in range(len(jn_list))]
			demand_nodes.sort(key=lambda t: _node_sort_key(t[1]))
			demand_order = np.array([t[0] for t in demand_nodes], dtype=int)
			demand_labels = [t[1] for t in demand_nodes]
			alpha_values = np.linspace(0.0, 1.0, 21)
			sweep = []
			for alpha in alpha_values:
				mixed = actual_phys + alpha * (pred_phys - actual_phys)
				mixed_pred, _, _ = _reconstruct_demands(
					mixed[np.newaxis, :], [dataset[sample_idx]], artifacts, graph, headloss_n, inp_path, stats=stats
				)
				sweep.append(float(mixed_pred[0, worst_idx]))
			return {
				"pred_phys": pred_phys,
				"actual_phys": actual_phys,
				"pred_head": pred_head,
				"actual_head": actual_head,
				"reservoir_pressure_text": res_pressure_text,
				"display_indices": display_indices,
				"p_labels": p_labels,
				"d_pred": d_pred,
				"d_actual": d_actual,
				"jn_list": jn_list,
				"demand_order": demand_order,
				"demand_labels": demand_labels,
				"worst_idx": worst_idx,
				"alpha_values": alpha_values,
				"sweep": np.array(sweep),
			}

		device = torch.device("cpu")
		try:
			model_a = _load_model(model_a_dir, dataset_a[0].x.shape[1])
			model_b = _load_model(model_b_dir, dataset_b[0].x.shape[1])
			preds_a_all, actuals_a_all, _ = _run_inference(model_a, dataset_a, device)
			preds_b_all, actuals_b_all, _ = _run_inference(model_b, dataset_b, device)
		except Exception as exc:
			QtWidgets.QMessageBox.critical(self, "Inference Failed", f"Could not prepare sample viewer: {exc}")
			return

		def _clean_node_name(node_name: str) -> str:
			s = str(node_name)
			return s[5:] if s.startswith("meas_") else s

		def _node_sort_key(node_name: str):
			s = str(node_name)
			if s.isdigit():
				return (0, int(s))
			return (1, s)

		def _physical_nodes_from_mapping(artifacts: dict) -> list[str]:
			node_mapping = artifacts.get("node_mapping", {})
			nodes = []
			seen = set()
			for idx_str, node_name in sorted(node_mapping.items(), key=lambda kv: int(kv[0])):
				clean = _clean_node_name(node_name)
				if str(node_name).startswith("meas_"):
					continue
				if clean not in seen:
					seen.add(clean)
					nodes.append(clean)
			return nodes

		def _index_by_physical_node(artifacts: dict) -> dict:
			node_mapping = artifacts.get("node_mapping", {})
			idx_by_node = {}
			for idx_str, node_name in node_mapping.items():
				if str(node_name).startswith("meas_"):
					continue
				clean = _clean_node_name(node_name)
				if clean not in idx_by_node:
					idx_by_node[clean] = int(idx_str)
			return idx_by_node

		def _scenario_ids(actuals_all: np.ndarray, stats: dict, artifacts: dict, dataset: list, nodes: list[str]) -> list[str]:
			actual_phys_all = _denorm(actuals_all, stats)
			idx_by_node = _index_by_physical_node(artifacts)
			valid_nodes = [n for n in nodes if n in idx_by_node]
			scenario_ids = []
			for i, row in enumerate(actual_phys_all):
				payload = {
					"nodes": valid_nodes,
					"heads": [round(float(row[idx_by_node[n]]), 8) for n in valid_nodes],
				}
				d_field = getattr(dataset[i], "d", None)
				if d_field is not None:
					d_vals = d_field.detach().cpu().numpy().reshape(-1)
					jn_list = [str(x) for x in list(getattr(dataset[i], "junction_names", []) or [])]
					# Canonicalize demand by node name (not raw tensor order) so IDs remain
					# stable across datasets with different internal ordering.
					d_map = {jn: round(float(d_vals[k]), 8) for k, jn in enumerate(jn_list) if k < len(d_vals)}
					d_nodes_sorted = sorted(d_map.keys(), key=_node_sort_key)
					payload["d_extra_by_node"] = [(jn, d_map[jn]) for jn in d_nodes_sorted]
				raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
				scenario_ids.append(hashlib.sha256(raw.encode("utf-8")).hexdigest())
			return scenario_ids

		shared_count = min(len(dataset_a), len(dataset_b))
		if shared_count <= 0:
			QtWidgets.QMessageBox.warning(self, "No Data", "Comparison datasets are empty.")
			return

		nodes_a = _physical_nodes_from_mapping(artifacts_a)
		nodes_b = _physical_nodes_from_mapping(artifacts_b)
		common_nodes = sorted(list(set(nodes_a) & set(nodes_b)), key=_node_sort_key)

		default_pairs = [(i, i) for i in range(shared_count)]
		strict_pairs = []
		if common_nodes:
			from collections import defaultdict, deque
			ids_a = _scenario_ids(actuals_a_all, stats_a, artifacts_a, dataset_a, common_nodes)
			ids_b = _scenario_ids(actuals_b_all, stats_b, artifacts_b, dataset_b, common_nodes)
			by_id_b = defaultdict(deque)
			for j, sid in enumerate(ids_b):
				by_id_b[sid].append(j)
			for i, sid in enumerate(ids_a):
				if by_id_b[sid]:
					strict_pairs.append((i, by_id_b[sid].popleft()))

		figure = Figure(figsize=(14, 10), constrained_layout=True)

		current_pairs = default_pairs
		current_view_idx = 0

		def _render_sample(view_idx: int) -> None:
			a_sample_idx, b_sample_idx = current_pairs[view_idx]
			try:
				example_a = _prepare_example(preds_a_all, actuals_a_all, a_sample_idx, stats_a, artifacts_a, graph_a, dataset_a)
				example_b = _prepare_example(preds_b_all, actuals_b_all, b_sample_idx, stats_b, artifacts_b, graph_b, dataset_b)
			except Exception as exc:
				QtWidgets.QMessageBox.critical(self, "Example Failed", f"Could not build sample pair {view_idx}: {exc}")
				return

			demand_min = min(
				float(np.min(example_a["d_actual"])),
				float(np.min(example_a["d_pred"])),
				float(np.min(example_b["d_actual"])),
				float(np.min(example_b["d_pred"])),
			)
			demand_max = max(
				float(np.max(example_a["d_actual"])),
				float(np.max(example_a["d_pred"])),
				float(np.max(example_b["d_actual"])),
				float(np.max(example_b["d_pred"])),
			)
			demand_span = demand_max - demand_min
			demand_pad = 0.05 * demand_span if demand_span > 0 else 0.05

			head_a = np.concatenate(
				[example_a["actual_head"][example_a["display_indices"]], example_a["pred_head"][example_a["display_indices"]]]
			)
			head_b = np.concatenate(
				[example_b["actual_head"][example_b["display_indices"]], example_b["pred_head"][example_b["display_indices"]]]
			)
			head_min = float(min(np.min(head_a), np.min(head_b)))
			head_max = float(max(np.max(head_a), np.max(head_b)))
			head_span = head_max - head_min
			# Deliberately expand head limits so bars look less tall while preserving values.
			head_pad = 0.25 * head_span if head_span > 0 else 1.0

			figure.clear()
			axes = figure.subplots(3, 2)
			models = [("Model A", example_a, model_a_hash), ("Model B", example_b, model_b_hash)]
			for col, (title, ex, model_hash) in enumerate(models):
				labels = ex["jn_list"]
				p_labels = ex["p_labels"]
				demand_order = ex["demand_order"]
				demand_labels = ex["demand_labels"]
				pressure_ax = axes[0, col]
				demand_ax = axes[1, col]
				sens_ax = axes[2, col]
				display_idx = ex["display_indices"]
				actual_plot = ex["actual_head"][display_idx]
				pred_plot = ex["pred_head"][display_idx]
				d_actual_plot = ex["d_actual"][demand_order]
				d_pred_plot = ex["d_pred"][demand_order]
				p_idx = np.arange(len(actual_plot))
				d_idx = np.arange(len(d_actual_plot))
				width = 0.35

				pressure_ax.bar(p_idx - width / 2, actual_plot, width=width, label="Actual", color="#4C78A8")
				pressure_ax.bar(p_idx + width / 2, pred_plot, width=width, label="Predicted", color="#F58518")
				y_offset = 0.03 * (head_max - head_min + 2.0 * head_pad)
				for x, a_val, p_val in zip(p_idx, actual_plot, pred_plot):
					diff = float(a_val - p_val)
					y_text = max(float(a_val), float(p_val)) + y_offset
					pressure_ax.text(x, y_text, f"{diff:+.2f}", ha="center", va="bottom", fontsize=7, rotation=90)
				pressure_ax.set_title(f"{title}: heads (node labels)")
				pressure_ax.set_xticks(p_idx)
				pressure_ax.set_xticklabels(p_labels, fontsize=8)
				pressure_ax.set_ylabel("head [m]")
				pressure_ax.set_ylim(head_min - head_pad, head_max + head_pad)
				pressure_ax.text(
					0.99,
					0.98,
					f"Reservoir pressure: {ex['reservoir_pressure_text']}",
					transform=pressure_ax.transAxes,
					ha="right",
					va="top",
					fontsize=8,
					bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="#BBBBBB"),
				)
				pressure_ax.grid(True, axis="y", alpha=0.25)
				if col == 0:
					pressure_ax.legend(fontsize=8)

				demand_ax.bar(d_idx - width / 2, d_actual_plot, width=width, label="Actual", color="#54A24B")
				demand_ax.bar(d_idx + width / 2, d_pred_plot, width=width, label="Reconstructed", color="#E45756")
				worst_label = demand_labels[int(np.argmax(np.abs(d_pred_plot - d_actual_plot)))]
				demand_ax.set_title(f"{title}: demand (worst junction {worst_label})")
				demand_ax.set_xticks(d_idx)
				demand_ax.set_xticklabels(demand_labels, fontsize=8)
				demand_ax.set_ylabel("m³/s")
				demand_ax.set_ylim(demand_min - demand_pad, demand_max + demand_pad)
				demand_ax.grid(True, axis="y", alpha=0.25)
				if col == 0:
					demand_ax.legend(fontsize=8)

				sens_ax.plot(ex["alpha_values"], ex["sweep"], color="#7F3C8D", lw=2)
				sens_ax.scatter([0, 1], [ex["d_actual"][ex["worst_idx"]], ex["d_pred"][ex["worst_idx"]]], color=["#54A24B", "#E45756"], zorder=3)
				sens_ax.set_title(f"{title}: sensitivity to pressure interpolation")
				sens_ax.set_xlabel("alpha = actual -> predicted")
				sens_ax.set_ylabel(f"Demand at junction {labels[ex['worst_idx']]}")
				sens_ax.grid(True, alpha=0.25)
				metrics_text = (
					f"head MAE={float(np.mean(np.abs(pred_plot - actual_plot))):.2f} m\n"
					f"demand MAE={float(np.mean(np.abs(ex['d_pred'] - ex['d_actual']))):.3f} m³/s"
				)
				sens_ax.text(0.02, 0.98, metrics_text, transform=sens_ax.transAxes, va="top", ha="left", fontsize=8,
					bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#BBBBBB"))

			mode_text = "strict fair" if strict_mode_check.isChecked() else "same index"
			figure.suptitle(
				f"Pressure-demand example: {mode_text} | view {view_idx} (A:{a_sample_idx}, B:{b_sample_idx})",
				fontsize=13,
			)
			view_label.setText(f"Scenario {view_idx + 1} / {len(current_pairs)}")
			canvas.draw_idle()
		dialog = QtWidgets.QDialog(self)
		dialog.setWindowTitle("Pressure-Demand Example")
		dialog.resize(1400, 1000)
		layout = QtWidgets.QVBoxLayout(dialog)
		caption = QtWidgets.QLabel(
			"Use Left/Right to browse scenarios. Head labels above bars are (actual - predicted) in meters. "
			"Site and reservoir nodes are clamped to ground truth, and meas_* pseudo nodes are ignored in pressure plots."
		)
		caption.setWordWrap(True)
		layout.addWidget(caption)
		canvas = FigureCanvas(figure)
		layout.addWidget(canvas, 1)

		nav_row = QtWidgets.QHBoxLayout()
		prev_button = QtWidgets.QPushButton("Left")
		next_button = QtWidgets.QPushButton("Right")
		view_label = QtWidgets.QLabel("")
		strict_mode_check = QtWidgets.QCheckBox("Strict fair mode (match identical ground-truth scenarios)")
		strict_mode_check.setChecked(bool(strict_pairs))
		strict_mode_check.setEnabled(bool(strict_pairs))
		strict_mode_status = QtWidgets.QLabel("")
		nav_row.addWidget(prev_button)
		nav_row.addWidget(next_button)
		nav_row.addWidget(view_label)
		nav_row.addStretch(1)
		nav_row.addWidget(strict_mode_check)
		nav_row.addWidget(strict_mode_status)
		layout.addLayout(nav_row)

		if not strict_pairs:
			strict_mode_check.setToolTip("No exact cross-dataset ground-truth matches found; falling back to same-index mode.")
			strict_mode_status.setText("Unavailable")
			strict_mode_status.setStyleSheet("color: #B85042;")
		else:
			strict_mode_status.setText(f"{len(strict_pairs)} matched scenarios")
			strict_mode_status.setStyleSheet("color: #3C7A3C;")

		def _set_mode() -> None:
			nonlocal current_pairs, current_view_idx
			current_pairs = strict_pairs if (strict_mode_check.isChecked() and strict_pairs) else default_pairs
			current_view_idx = 0
			_render_sample(current_view_idx)

		def _step(delta: int) -> None:
			nonlocal current_view_idx
			if not current_pairs:
				return
			current_view_idx = (current_view_idx + delta) % len(current_pairs)
			_render_sample(current_view_idx)

		strict_mode_check.toggled.connect(lambda _: _set_mode())
		prev_button.clicked.connect(lambda: _step(-1))
		next_button.clicked.connect(lambda: _step(1))
		_set_mode()

		buttons = QtWidgets.QHBoxLayout()
		buttons.addStretch(1)
		close_button = QtWidgets.QPushButton("Close")
		close_button.clicked.connect(dialog.close)
		buttons.addWidget(close_button)
		layout.addLayout(buttons)
		dialog.exec()

	def _gnn_export_results(self) -> None:
		if not self._gnn_results:
			return
		path, _ = QtWidgets.QFileDialog.getSaveFileName(
			self, "Export Results", "gnn_comparison.json", "JSON Files (*.json);;All Files (*)"
		)
		if path:
			with open(path, "w", encoding="utf-8") as f:
				json.dump(self._gnn_results, f, indent=2)

	def _on_tab_changed(self, index: int) -> None:
		on_ls = index == self._ls_tab_index
		self.comparison_mode_check.setEnabled(not on_ls)
		if on_ls:
			self.comparison_mode_check.setChecked(False)
		on_post = index == getattr(self, "_post_tab_index", -1)
		if on_post:
			self._posteriori_apply_plot_state()
		else:
			self.plot.set_measurement_editable(True)
			self.plot.clear_demands_overlay()

	def _parse_ls_nodes(self, text: str) -> "tuple[bool, List[str]]":
		"""Parse comma-separated explicit node IDs; reject #N shortcuts."""
		junction_set = set(self.plot.get_junction_nodes())
		tokens = [t.strip() for t in text.split(",") if t.strip()]
		if not tokens:
			return False, []
		nodes: List[str] = []
		seen: set[str] = set()
		for tok in tokens:
			if tok.startswith("#"):
				return False, []
			if tok not in junction_set:
				return False, []
			if tok not in seen:
				seen.add(tok)
				nodes.append(tok)
		return True, nodes

	def _ls_default_nodes(self) -> List[str]:
		wdn = self.wdn_input.currentText().strip()
		if not wdn:
			return []
		json_path = os.path.join(ROOT_DIR, "wdn", f"{wdn}.json")
		if not os.path.isfile(json_path):
			return []
		try:
			with open(json_path, encoding="utf-8") as f:
				cfg = json.load(f)
			nodes = [str(n).strip() for n in cfg.get("measurement_nodes", []) if str(n).strip()]
			return list(dict.fromkeys(nodes))
		except (OSError, json.JSONDecodeError):
			return []

	def _ls_update_starting_nodes_state(self) -> None:
		if not hasattr(self, "ls_nodes_input"):
			return
		use_default = bool(getattr(self, "ls_use_default_nodes", None) and self.ls_use_default_nodes.isChecked())
		if use_default:
			nodes = self._ls_default_nodes()
			self.ls_nodes_input.setText(", ".join(nodes))
			self.ls_nodes_input.setEnabled(False)
			self.ls_nodes_input.setPlaceholderText("Using default measurement_nodes from wdn/<name>.json")
		else:
			self.ls_nodes_input.setEnabled(True)
			self.ls_nodes_input.setPlaceholderText("e.g. 3, 7, 12")

	def _run_local_search(self) -> None:
		if self._ls_worker is not None and self._ls_worker.isRunning():
			return
		if not self._measurement_data_valid:
			QtWidgets.QMessageBox.warning(
				self,
				"Invalid Measurement Data",
				"Custom measurement data must be a JSON dictionary mapping node ids to numbers, with reserved keys like -1 for total demand.",
			)
			self.ls_status_label.setText("Local search not started: invalid measurement data.")
			return
		if bool(hasattr(self, "ls_use_default_nodes") and self.ls_use_default_nodes.isChecked()):
			nodes = self._ls_default_nodes()
			valid = len(nodes) > 0
		else:
			valid, nodes = self._parse_ls_nodes(self.ls_nodes_input.text())
		if not valid or len(nodes) < 1:
			QtWidgets.QMessageBox.warning(
				self,
				"Invalid Nodes",
				"Please provide at least one valid starting node.\n"
				"If 'Use default' is enabled, ensure wdn/<name>.json contains measurement_nodes.",
			)
			return
		wdn = self.wdn_input.currentText().strip() or self.solver_params.wdn
		adjacency = self.plot.get_pipe_adjacency()
		index_path = _data_index_path(wdn)
		existing_index = _load_index_with_legacy(index_path, LEGACY_DATA_INDEX)
		# Build payload template (no MEASUREMENT_SITES — worker sets it per config)
		payload_template = self._solver_payload_from_widget(self.model_a)
		payload_template["MODE"] = self._current_mode()
		payload_template.pop("MEASUREMENT_SITES", None)
		if bool(payload_template.get("DYNAMIC_MULTISTART", False)):
			# Local search ignores solver-tab radius and starts with r=+inf.
			payload_template["DMS_RADIUS"] = float("inf")

		self.ls_log.clear()
		self.ls_status_label.setText("Running...")
		self.ls_run_button.setEnabled(False)
		self.ls_cancel_button.setEnabled(True)

		self._ls_worker = LocalSearchWorker(
			nodes, adjacency, payload_template, wdn, index_path, existing_index, self
		)
		self._ls_worker.status_updated.connect(self._on_ls_status)
		self._ls_worker.row_added.connect(self._on_ls_row)
		self._ls_worker.highlight_updated.connect(self._on_ls_highlight)
		self._ls_worker.finished_signal.connect(self._on_ls_finished)
		self.plot.set_local_search_highlight(nodes, None, None)
		self._ls_worker.start()

	def _cancel_local_search(self) -> None:
		if self._ls_worker is not None:
			self._ls_worker.cancel()
			self.ls_status_label.setText("Cancelling...")

	def _on_ls_status(self, msg: str) -> None:
		self.ls_log.appendPlainText(msg)
		self.ls_status_label.setText(msg[:120])

	def _on_ls_row(self, row: dict) -> None:
		# Nothing to do live — ResultsTableDialog shown at the end.
		pass

	def _on_ls_highlight(self, current_nodes: list, swap_out: object, swap_in: object) -> None:
		out_node = str(swap_out) if isinstance(swap_out, str) else None
		in_node = str(swap_in) if isinstance(swap_in, str) else None
		self.plot.set_local_search_highlight([str(n) for n in current_nodes], out_node, in_node)

	def _on_ls_finished(self, rows: list) -> None:
		self._ls_worker = None
		self.ls_run_button.setEnabled(True)
		self.ls_cancel_button.setEnabled(False)
		self.plot.clear_local_search_highlight()
		self.ls_status_label.setText("Done.")
		if rows:
			dlg = ResultsTableDialog(rows, comparison=False, parent=self)
			dlg.show()

	def _posteriori_scenario_dir(self, wdn: str) -> str:
		return os.path.join(ROOT_DIR, "scenario", wdn)

	def _posteriori_cache_path(self, wdn: str) -> str:
		return os.path.join(self._posteriori_scenario_dir(wdn), "cache_index.json")

	def _posteriori_wdn_defaults(self, wdn: str) -> tuple[List[str], float]:
		json_path = os.path.join(ROOT_DIR, "wdn", f"{wdn}.json")
		if not os.path.isfile(json_path):
			return [], 0.0
		try:
			with open(json_path, encoding="utf-8") as f:
				cfg = json.load(f)
		except (OSError, json.JSONDecodeError):
			return [], 0.0
		nodes = [str(n).strip() for n in cfg.get("measurement_nodes", []) if str(n).strip()]
		extra = _float_or_default(cfg.get("extra_demand"), 0.0)
		return list(dict.fromkeys(nodes)), max(0.0, float(extra))

	def _posteriori_load_cache(self, wdn: str) -> Dict[str, Dict[str, str]]:
		path = self._posteriori_cache_path(wdn)
		if not os.path.isfile(path):
			return {}
		try:
			with open(path, encoding="utf-8") as f:
				payload = json.load(f)
		except (OSError, json.JSONDecodeError):
			return {}
		entries = payload.get("entries", {}) if isinstance(payload, dict) else {}
		if not isinstance(entries, dict):
			return {}
		out: Dict[str, Dict[str, str]] = {}
		for name, info in entries.items():
			if not isinstance(info, dict):
				continue
			file_name = str(info.get("file", "")).strip()
			hash_value = str(info.get("hash", "")).strip()
			if not file_name:
				continue
			out[str(name)] = {"file": file_name, "hash": hash_value}
		return out

	def _posteriori_save_cache(self, wdn: str, entries: Dict[str, Dict[str, str]]) -> None:
		cache_path = self._posteriori_cache_path(wdn)
		os.makedirs(os.path.dirname(cache_path), exist_ok=True)
		payload = {
			"wdn": wdn,
			"updated_at": datetime.now().strftime("%Y%m%d-%H%M%S"),
			"entries": entries,
		}
		with open(cache_path, "w", encoding="utf-8") as f:
			json.dump(payload, f, indent=2, sort_keys=True)

	def _posteriori_hash_for_payload(self, payload: Dict[str, object]) -> str:
		hash_payload = {
			"wdn": payload.get("wdn"),
			"scenario_name": payload.get("scenario_name"),
			"measurement_nodes": payload.get("measurement_nodes", []),
			"extra_demand": payload.get("extra_demand", 0.0),
			"scenario_demands": payload.get("scenario_demands", {}),
		}
		return compute_hash(hash_payload)

	def _posteriori_slugify_name(self, text: str) -> str:
		cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", text.strip())
		cleaned = cleaned.strip("-_")
		return cleaned or f"custom-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

	def _posteriori_set_default_name(self) -> None:
		self.post_save_name.setText(f"custom-{datetime.now().strftime('%Y%m%d-%H%M%S')}")

	def _posteriori_mark_dirty(self) -> None:
		self._post_dirty = True
		self._post_active_scenario_name = ""
		self.post_loaded_label.setText("Unsaved edits")

	def _posteriori_base_payload(self, wdn: str) -> Dict[str, object]:
		network = load_inp_network(os.path.join(ROOT_DIR, "wdn", f"{wdn}.inp"))
		base_demands = {str(jid): float(node.base_demand) for jid, node in network.junctions.items()}
		measurement_nodes, extra_demand = self._posteriori_wdn_defaults(wdn)
		payload: Dict[str, object] = {
			"wdn": wdn,
			"wdn_name": wdn,
			"inp_path": f"./wdn/{wdn}.inp",
			"scenario_name": "base",
			"scenario_source": "generated_base",
			"timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
			"measurement_nodes": measurement_nodes,
			"extra_demand": float(extra_demand),
			"base_demands": base_demands,
			"scenario_demands": dict(base_demands),
			"scenario_heads": {},
			"scenario_flows": {},
		}
		payload["scenario_hash"] = self._posteriori_hash_for_payload(payload)
		return payload

	def _posteriori_ensure_base_scenario(self, wdn: str) -> None:
		scenario_dir = self._posteriori_scenario_dir(wdn)
		os.makedirs(scenario_dir, exist_ok=True)
		base_path = os.path.join(scenario_dir, "base.json")
		if not os.path.isfile(base_path):
			_write_json(base_path, self._posteriori_base_payload(wdn))

	def _posteriori_rebuild_cache(self, wdn: str) -> Dict[str, Dict[str, str]]:
		scenario_dir = self._posteriori_scenario_dir(wdn)
		os.makedirs(scenario_dir, exist_ok=True)
		entries: Dict[str, Dict[str, str]] = {}
		for name in sorted(os.listdir(scenario_dir)):
			if not name.lower().endswith(".json"):
				continue
			if name == "cache_index.json":
				continue
			path = os.path.join(scenario_dir, name)
			try:
				payload = _read_json(path)
			except Exception:
				continue
			scenario_name = str(payload.get("scenario_name") or os.path.splitext(name)[0])
			hash_value = str(payload.get("scenario_hash") or self._posteriori_hash_for_payload(payload))
			if not payload.get("scenario_hash"):
				payload["scenario_hash"] = hash_value
				_write_json(path, payload)
			entries[scenario_name] = {"file": name, "hash": hash_value}
		self._posteriori_save_cache(wdn, entries)
		return entries

	def _posteriori_refresh_scenario_list(self, select_name: str | None = None) -> None:
		wdn = self.wdn_input.currentText().strip()
		if not wdn:
			return
		self._posteriori_ensure_base_scenario(wdn)
		entries = self._posteriori_rebuild_cache(wdn)
		names = ["base"] + sorted([n for n in entries.keys() if n != "base"])
		self.post_scenario_combo.blockSignals(True)
		self.post_scenario_combo.clear()
		self.post_scenario_combo.addItems(names)
		target = select_name if select_name in names else "base"
		self.post_scenario_combo.setCurrentIndex(max(0, self.post_scenario_combo.findText(target)))
		self.post_scenario_combo.blockSignals(False)
		self._posteriori_scenario_changed()

	def _posteriori_scenario_changed(self) -> None:
		wdn = self.wdn_input.currentText().strip()
		if not wdn:
			return
		selected = self.post_scenario_combo.currentText().strip() or "base"
		entries = self._posteriori_load_cache(wdn)
		entry = entries.get(selected, {"file": f"{selected}.json"})
		scenario_path = os.path.join(self._posteriori_scenario_dir(wdn), str(entry.get("file", f"{selected}.json")))
		if not os.path.isfile(scenario_path):
			if selected == "base":
				self._posteriori_ensure_base_scenario(wdn)
			else:
				return
			if not os.path.isfile(scenario_path):
				return

		payload = _read_json(scenario_path)
		network = load_inp_network(os.path.join(ROOT_DIR, "wdn", f"{wdn}.inp"))
		base_demands = {str(jid): float(node.base_demand) for jid, node in network.junctions.items()}
		scenario_demands = {
			str(jid): float(payload.get("scenario_demands", {}).get(str(jid), base_demands.get(str(jid), 0.0)))
			for jid in base_demands.keys()
		}
		measurement_nodes, default_extra = self._posteriori_wdn_defaults(wdn)
		loaded_nodes = [str(n).strip() for n in payload.get("measurement_nodes", measurement_nodes) if str(n).strip()]
		extra = _float_or_default(payload.get("extra_demand"), default_extra)

		self._post_active_scenario_name = selected
		self._post_active_scenario_path = scenario_path
		self._post_base_demands = base_demands
		self._post_current_demands = scenario_demands
		self._post_loaded_heads = {str(k): float(v) for k, v in payload.get("scenario_heads", {}).items() if isinstance(v, (int, float))}
		self._post_loaded_flows = {str(k): float(v) for k, v in payload.get("scenario_flows", {}).items() if isinstance(v, (int, float))}
		self._post_dirty = False
		self._post_last_saved_name = selected

		self._post_updating_measurement_text = True
		self.post_measurement_sites.setText(", ".join(loaded_nodes))
		self._post_updating_measurement_text = False
		self.post_extra_demand.blockSignals(True)
		self.post_extra_demand.setValue(max(0.0, float(extra)))
		self.post_extra_demand.blockSignals(False)
		self.post_scenario_editable.blockSignals(True)
		self.post_scenario_editable.setChecked(True)
		self.post_scenario_editable.blockSignals(False)
		self._posteriori_set_default_name()
		self._posteriori_refresh_elimination_node(loaded_nodes)
		self._posteriori_apply_plot_state()
		self.post_loaded_label.setText(f"Loaded: {selected}")
		self.post_status.setText(f"Loaded scenario '{selected}'.")

	def _posteriori_parse_nodes(self, text: str) -> List[str]:
		junction_set = set(self.plot.get_junction_nodes())
		if not junction_set:
			return []
		tokens = [tok.strip() for tok in re.split(r"[\s,;]+", text) if tok.strip()]
		out: List[str] = []
		seen: set[str] = set()
		for tok in tokens:
			if tok not in junction_set:
				continue
			if tok in seen:
				continue
			seen.add(tok)
			out.append(tok)
		return out

	def _posteriori_measurement_sites_changed(self, text: str) -> None:
		if self._post_updating_measurement_text:
			return
		nodes = self._posteriori_parse_nodes(text)
		self._posteriori_mark_dirty()
		self._posteriori_refresh_elimination_node(nodes)
		self.plot.set_measurements(nodes)

	def _posteriori_refresh_elimination_node(self, measurement_nodes: List[str]) -> None:
		meas_set = set(measurement_nodes)
		res_adj = [n for n in self.plot.get_reservoir_adjacent_junctions() if n not in meas_set]
		if not res_adj:
			res_adj = [n for n in self.plot.get_junction_nodes() if n not in meas_set]
		if res_adj:
			# Prefer highest-degree node among candidates (mirrors _choose_central_node).
			adj = self.plot.get_pipe_adjacency()
			self._post_auto_elim_node = max(res_adj, key=lambda n: (len(adj.get(n, [])), -int(n) if str(n).isdigit() else 0))
		else:
			self._post_auto_elim_node = None
		label = self._post_auto_elim_node or "(none)"
		self.post_elimination_node_label.setText(label)
		self.plot.set_elimination_node(self._post_auto_elim_node)

	def _posteriori_apply_default_extra_demand(self) -> None:
		wdn = self.wdn_input.currentText().strip()
		if not wdn:
			return
		_default_nodes, default_extra = self._posteriori_wdn_defaults(wdn)
		self.post_extra_demand.setValue(default_extra)

	def _posteriori_extra_demand_changed(self, _value: float) -> None:
		self._posteriori_mark_dirty()

	def _posteriori_editability_changed(self, editable: bool) -> None:
		self.post_measurement_sites.setEnabled(bool(editable))
		self.post_extra_demand.setEnabled(bool(editable))
		self.post_extra_default_btn.setEnabled(bool(editable))
		self.plot.set_measurement_editable(bool(editable) and self.tabs.currentIndex() == self._post_tab_index)

	def _posteriori_apply_plot_state(self) -> None:
		self.plot.clear_node_mae()
		if not self._post_base_demands:
			self.plot.clear_demands_overlay()
			self.plot.set_highlight_node(None)
			self.plot.set_elimination_node(None)
			self.plot.set_node_deltas({})
			return
		nodes = self._posteriori_parse_nodes(self.post_measurement_sites.text())
		self.plot.set_measurements(nodes)
		self.plot.set_demands_overlay(self._post_base_demands, self._post_current_demands)
		self.plot.set_highlight_node(None)
		self.plot.set_elimination_node(self._post_auto_elim_node)
		self._update_post_plot_deltas()
		self.plot.set_measurement_editable(self.post_scenario_editable.isChecked() and self.tabs.currentIndex() == self._post_tab_index)

	def _post_method_changed(self) -> None:
		data = self.post_method_combo.currentData() or ("pressure", "ensemble")
		method = data[0]
		# Sensor-noise ε only applies to the demand method (soft sensor likelihood).
		self.post_sensor_eps.setEnabled(method == "demand")

	def _posteriori_node_right_clicked(self, node_id: str) -> None:
		if self.tabs.currentIndex() != getattr(self, "_post_tab_index", -1):
			return
		if not self.post_scenario_editable.isChecked():
			return
		if node_id not in self._post_base_demands:
			return

		base_val = float(self._post_base_demands.get(node_id, 0.0))
		current_val = float(self._post_current_demands.get(node_id, base_val))
		current_delta = max(0.0, current_val - base_val)
		extra_budget = float(self.post_extra_demand.value())
		total_delta = sum(max(0.0, float(self._post_current_demands.get(j, 0.0) - self._post_base_demands.get(j, 0.0))) for j in self._post_base_demands.keys())
		available_for_node = max(0.0, extra_budget - (total_delta - current_delta))

		dialog = QtWidgets.QDialog(self)
		dialog.setWindowTitle(f"Set Δd for node {node_id}")
		layout = QtWidgets.QVBoxLayout(dialog)
		info = QtWidgets.QLabel(f"Allowed range: 0.0 to {available_for_node:.6f}")
		layout.addWidget(info)
		value_label = QtWidgets.QLabel("")
		layout.addWidget(value_label)
		slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
		scale = 10000
		slider.setRange(0, max(0, int(round(available_for_node * scale))))
		slider.setValue(max(0, int(round(current_delta * scale))))
		layout.addWidget(slider)

		def _update_label() -> None:
			delta = slider.value() / scale
			value_label.setText(f"Δd = {delta:.6f}   d = {base_val + delta:.6f}")

		slider.valueChanged.connect(_update_label)
		_update_label()

		btns = QtWidgets.QDialogButtonBox(
			QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
		)
		btns.accepted.connect(dialog.accept)
		btns.rejected.connect(dialog.reject)
		layout.addWidget(btns)

		if dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
			return

		new_delta = slider.value() / scale
		self._post_current_demands[node_id] = base_val + new_delta
		self._posteriori_mark_dirty()
		self._posteriori_apply_plot_state()

	def _update_post_plot_deltas(self) -> None:
		if not self._post_base_demands or not self._post_current_demands:
			self.plot.set_node_deltas({})
			return
		deltas = {
			j: max(0.0, self._post_current_demands.get(j, 0.0) - self._post_base_demands.get(j, 0.0))
			for j in self._post_base_demands
		}
		self.plot.set_node_deltas(deltas)

	def _posteriori_reservoir_right_clicked(self, node_id: str) -> None:
		if self.tabs.currentIndex() != getattr(self, "_post_tab_index", -1):
			return
		if not self.post_scenario_editable.isChecked():
			return
		if not self._post_base_demands:
			return

		extra_budget = float(self.post_extra_demand.value())
		total_delta = sum(
			max(0.0, float(self._post_current_demands.get(j, 0.0) - self._post_base_demands.get(j, 0.0)))
			for j in self._post_base_demands
		)
		remaining = max(0.0, extra_budget - total_delta)
		min_node_delta = min(
			max(0.0, float(self._post_current_demands.get(j, 0.0) - self._post_base_demands.get(j, 0.0)))
			for j in self._post_base_demands
		) if self._post_base_demands else 0.0
		max_increase = remaining
		max_decrease = -min_node_delta

		dialog = QtWidgets.QDialog(self)
		dialog.setWindowTitle("Adjust All Δ (uniform shift)")
		layout = QtWidgets.QVBoxLayout(dialog)
		scale = 100000
		range_min = int(round(max_decrease * scale))
		range_max = int(round(max_increase * scale))
		if range_min >= range_max:
			QtWidgets.QMessageBox.information(self, "No room", "No budget remaining and all nodes are at minimum.")
			return
		info = QtWidgets.QLabel(f"Allowed shift: {max_decrease:.6f} to +{max_increase:.6f}")
		layout.addWidget(info)
		value_label = QtWidgets.QLabel("")
		layout.addWidget(value_label)
		slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
		slider.setRange(range_min, range_max)
		slider.setValue(0)
		layout.addWidget(slider)

		def _update_label() -> None:
			shift = slider.value() / scale
			value_label.setText(f"Shift = {shift:+.6f}")

		slider.valueChanged.connect(_update_label)
		_update_label()

		btns = QtWidgets.QDialogButtonBox(
			QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
		)
		btns.accepted.connect(dialog.accept)
		btns.rejected.connect(dialog.reject)
		layout.addWidget(btns)

		if dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
			return

		shift = slider.value() / scale
		if abs(shift) < 1e-9:
			return
		for j, base_val in self._post_base_demands.items():
			cur = float(self._post_current_demands.get(j, base_val))
			cur_delta = max(0.0, cur - base_val)
			new_delta = max(0.0, cur_delta + shift)
			self._post_current_demands[j] = base_val + new_delta
		self._posteriori_mark_dirty()
		self._posteriori_apply_plot_state()

	def _posteriori_view_scenarios_clicked(self) -> None:
		result = self._post_mh_result
		junc_ids = self._post_mh_junc_ids
		if result is None or not junc_ids:
			QtWidgets.QMessageBox.information(self, "No data", "Run the sampler first.")
			return
		try:
			import numpy as _np
			wdn = self.wdn_input.currentText().strip()
			inp_abs = os.path.join(ROOT_DIR, "wdn", f"{wdn}.inp")
			net = load_inp_network(inp_abs)
			coords = {nid: node.coordinates for nid, node in net.nodes.items() if node.coordinates is not None}
			if len(coords) != len(net.nodes):
				import networkx as _nx
				G = _nx.Graph()
				for nid in net.nodes:
					G.add_node(nid)
				for pipe in net.pipes.values():
					G.add_edge(pipe.start_node, pipe.end_node)
				coords = _nx.spring_layout(G, seed=1)
			node_pos = {k: (float(v[0]), float(v[1])) for k, v in coords.items()}
			measurement_nodes = set(self._posteriori_parse_nodes(self.post_measurement_sites.text()))
			elim = self._post_auto_elim_node
			dlg = ScenarioViewerDialog(
				samples_d=result.samples_d,
				junc_ids=junc_ids,
				scenario_demands=self._post_current_demands,
				burn_in=self._post_mh_burn_in,
				network=net,
				node_pos=node_pos,
				reservoir_set=set(net.reservoirs.keys()),
				measurement_set=measurement_nodes,
				elimination_node=elim,
				log_targets=getattr(result, "log_targets", None),
				parent=self,
			)
			dlg.show()
		except Exception as exc:
			QtWidgets.QMessageBox.warning(self, "Error opening viewer", str(exc))

	def _posteriori_simulate_scenario(self, wdn: str, demands: Dict[str, float]) -> tuple[bool, Dict[str, float], Dict[str, float], str]:
		try:
			import wntr
		except ImportError:
			return False, {}, {}, "wntr is not available. Install wntr to simulate scenario heads."

		inp_path = os.path.join(ROOT_DIR, "wdn", f"{wdn}.inp")
		try:
			wn = wntr.network.WaterNetworkModel(inp_path)
			# Flatten demand patterns to 1.0: the demands dict already contains the
			# intended values in m³/s; we do not want EPANET to apply extra multipliers
			# from the .inp file (e.g. a 0.5 diurnal-pattern factor).
			for _, pat in wn.patterns():
				pat.multipliers = [1.0] * len(pat.multipliers)
			for jid, val in demands.items():
				if jid in wn.junction_name_list:
					node = wn.get_node(jid)
					new_val = float(val)
					if not math.isfinite(new_val) or new_val < 0.0:
						return False, {}, {}, f"Invalid demand for junction {jid}: {new_val}"
					if hasattr(node, "demand_timeseries_list") and len(node.demand_timeseries_list) > 0:
						node.demand_timeseries_list[0].base_value = new_val
					else:
						# Fallback for unusual junction objects.
						try:
							node.add_demand(new_val, pattern_name=None)
						except Exception:
							try:
								node.base_demand = new_val
							except Exception:
								return False, {}, {}, f"Could not set demand for junction {jid}"

			# Prefer EPANET; if it fails with input parsing errors, fall back to WNTR simulator for diagnostics/continuity.
			try:
				sim = wntr.sim.EpanetSimulator(wn)
				res = sim.run_sim()
			except Exception as epanet_exc:
				try:
					sim = wntr.sim.WNTRSimulator(wn)
					res = sim.run_sim()
				except Exception as wntr_exc:
					detail = (
						f"EPANET: {type(epanet_exc).__name__}: {epanet_exc}\n"
						f"WNTR fallback: {type(wntr_exc).__name__}: {wntr_exc}"
					)
					return False, {}, {}, detail
			head_series = res.node["head"].iloc[0]
			flow_series = res.link["flowrate"].iloc[0]
			heads = {str(k): float(v) for k, v in head_series.to_dict().items() if isinstance(v, (int, float)) and math.isfinite(float(v))}
			flows = {str(k): float(v) for k, v in flow_series.to_dict().items() if isinstance(v, (int, float)) and math.isfinite(float(v))}
			return True, heads, flows, ""
		except Exception as exc:
			return False, {}, {}, f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=2)}"

	def _posteriori_run_clicked(self) -> None:
		if not self.post_scenario_editable.isChecked():
			QtWidgets.QMessageBox.warning(self, "Run blocked", "Run cannot start while 'Scenario may be changed' is disabled.")
			return

		wdn = self.wdn_input.currentText().strip()
		if not wdn:
			return
		if not self._post_base_demands:
			QtWidgets.QMessageBox.warning(self, "No scenario", "No scenario is loaded.")
			return

		extra_budget = float(self.post_extra_demand.value())
		total_delta = sum(max(0.0, float(self._post_current_demands.get(j, 0.0) - self._post_base_demands.get(j, 0.0))) for j in self._post_base_demands.keys())
		if total_delta > extra_budget + 1e-9:
			QtWidgets.QMessageBox.warning(
				self,
				"Extra demand exceeded",
				f"Total Δd={total_delta:.6f} exceeds extra_demand={extra_budget:.6f}.",
			)
			return

		measurement_nodes = self._posteriori_parse_nodes(self.post_measurement_sites.text())
		scenario_name = self._posteriori_slugify_name(self.post_save_name.text())
		ok, heads, flows, err = self._posteriori_simulate_scenario(wdn, self._post_current_demands)
		if not ok:
			fake_cmd = ["posteriori", "simulate", wdn]
			fake_proc = subprocess.CompletedProcess(fake_cmd, 1, stdout="", stderr=str(err))
			self._store_output("Posteriori Simulation Output", fake_cmd, fake_proc)
			QtWidgets.QMessageBox.warning(self, "Scenario simulation failed", f"Could not simulate scenario:\n{err}")
			return

		scenario_payload: Dict[str, object] = {
			"wdn": wdn,
			"wdn_name": wdn,
			"inp_path": f"./wdn/{wdn}.inp",
			"scenario_name": scenario_name,
			"scenario_source": self._post_active_scenario_name or "gui",
			"timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
			"measurement_nodes": measurement_nodes,
			"extra_demand": extra_budget,
			"base_demands": self._post_base_demands,
			"scenario_demands": self._post_current_demands,
			"scenario_heads": heads,
			"scenario_flows": flows,
		}
		scenario_payload["scenario_hash"] = self._posteriori_hash_for_payload(scenario_payload)

		scenario_dir = self._posteriori_scenario_dir(wdn)
		os.makedirs(scenario_dir, exist_ok=True)
		if self._post_dirty:
			scenario_path = os.path.join(scenario_dir, f"{scenario_name}.json")
			_write_json(scenario_path, scenario_payload)
			self._posteriori_refresh_scenario_list(select_name=scenario_name)
			self._post_last_saved_name = scenario_name
			self.post_loaded_label.setText(f"Loaded: {scenario_name}")
		else:
			scenario_name = self._post_last_saved_name or (self.post_scenario_combo.currentText().strip() or "base")

		missing_measurements = [n for n in measurement_nodes if n not in heads]
		if missing_measurements:
			QtWidgets.QMessageBox.warning(
				self,
				"Missing measurement heads",
				"Scenario simulation did not provide heads for selected measurement nodes: " + ", ".join(missing_measurements),
			)
			return

		try:
			ns = runpy.run_path(os.path.join(ROOT_DIR, "mh_posteriori-scenario-gen.py"))
			cfg_cls = ns["MHPosteriorConfig"]
			run_sampler = ns["sample_posterior_scenarios"]
			inp_abs = os.path.join(ROOT_DIR, "wdn", f"{wdn}.inp")
			measurement_heads = {n: float(heads[n]) for n in measurement_nodes}
			total_demand_value = float(sum(float(v) for v in self._post_current_demands.values()))
			predictor_heads = {str(k): float(v) for k, v in heads.items()}
			method, proposal = self.post_method_combo.currentData() or ("pressure", "ensemble")
			# Demand method (M2) needs walkers started inside the feasible region -> small
			# init dispersion; the pressure method tolerates a wider spread.
			ens_disp = 0.3 if method == "demand" else 0.02
			cfg = cfg_cls(
				burn_in=int(self.post_burn_in.value()),
				num_samples=int(self.post_num_samples.value()),
				proposal_std=float(self.post_proposal_std.value()),
				use_square_reduced_jacobian=not bool(self.post_use_gram.isChecked()),
				demand_penalty_a=float(self.post_penalty_a.value()),
				num_chains=int(self.post_num_chains.value()),
				method=method,
				proposal=proposal,
				sensor_noise_eps=float(self.post_sensor_eps.value()),
				ensemble_init_dispersion=ens_disp,
			)
			result = run_sampler(
				inp_path=inp_abs,
				measurement_heads=measurement_heads,
				measured_total_demand=total_demand_value,
				predictor_heads=predictor_heads,
				elimination_node=self._post_auto_elim_node,
				config=cfg,
			)
		except Exception as exc:
			fake_cmd = ["posteriori", "mh", wdn]
			fake_proc = subprocess.CompletedProcess(fake_cmd, 1, stdout="", stderr=str(exc))
			self._store_output("Posteriori MH Output", fake_cmd, fake_proc)
			QtWidgets.QMessageBox.warning(self, "Posteriori run failed", str(exc))
			return

		self._post_mh_result = result
		self._post_mh_burn_in = int(self.post_burn_in.value())
		try:
			import numpy as _np_mae
			_net_mae = load_inp_network(inp_abs)
			_junc_ids = list(_net_mae.junctions.keys())
			self._post_mh_junc_ids = _junc_ids
			if result.samples_d.shape[0] > 0 and result.samples_d.shape[1] == len(_junc_ids):
				_ref = _np_mae.array([self._post_current_demands.get(str(j), 0.0) for j in _junc_ids])
				_mae_arr = _np_mae.mean(_np_mae.abs(result.samples_d - _ref[None, :]), axis=0)
				_mae_dict = {str(_junc_ids[i]): float(_mae_arr[i]) for i in range(len(_junc_ids))}
				self.plot.set_node_mae(_mae_dict, max(extra_budget / 2.0, 1e-9))
		except Exception:
			pass
		if hasattr(self, "_post_view_scenarios_btn"):
			self._post_view_scenarios_btn.setEnabled(True)

		mh_out_path = os.path.join(scenario_dir, f"{scenario_name}_mh_result.json")
		mh_payload = {
			"scenario": scenario_name,
			"measurement_nodes": measurement_nodes,
			"num_samples": int(result.samples_h.shape[0]),
			"acceptance_rate": float(result.acceptance_rate),
			"infeasible_rate": float(result.infeasible_rate),
			"punished_rate": float(result.punished_rate),
			"proposal_std_final": float(result.proposal_std_final),
			"min_ess": float(result.min_ess),
			"median_ess": float(result.median_ess),
			"elapsed_seconds": float(result.elapsed_seconds),
			"min_ess_per_sec": float(result.min_ess_per_sec),
			"median_ess_per_sec": float(result.median_ess_per_sec),
			"num_chains": int(result.num_chains),
			"max_rhat": float(result.max_rhat),
			"diagnostics": {str(k): float(v) for k, v in result.diagnostics.items()},
		}
		_write_json(mh_out_path, mh_payload)
		self._post_dirty = False
		_rhat = float(result.max_rhat)
		_rhat_txt = "n/a" if not math.isfinite(_rhat) else f"{_rhat:.3f}"
		_rhat_flag = " ⚠converge" if (math.isfinite(_rhat) and _rhat > 1.01) else ""
		self.post_status.setText(
			f"Run complete. acceptance={float(result.acceptance_rate):.3f}, "
			f"min_ess/s={float(result.min_ess_per_sec):.2f}, "
			f"median_ess={float(result.median_ess):.2f}, "
			f"R-hat={_rhat_txt}{_rhat_flag}, "
			f"output={os.path.basename(mh_out_path)}"
		)
		self.status_bar.showMessage("Posteriori run completed.")

	def _wdn_changed(self) -> None:
		self.solver_params.wdn = self.wdn_input.currentText().strip()
		if hasattr(self, "model_a"):
			self.model_a.refresh_wdn_dependent_defaults()
		if hasattr(self, "model_b"):
			self.model_b.refresh_wdn_dependent_defaults()
		if hasattr(self, "ls_use_default_nodes"):
			self._ls_update_starting_nodes_state()
		if not self._updating_measurement_text:
			self._measurement_text_changed(self.measurement_list.text())
		if hasattr(self, "gnn_default_nodes_label"):
			default_nodes = self._gnn_default_nodes()
			self.gnn_default_nodes_label.setText(", ".join(default_nodes) if default_nodes else "(none)")
			self._gnn_refresh_dataset_status()
			self._gnn_populate_model_combos()
		if hasattr(self, "post_scenario_combo"):
			self._posteriori_refresh_scenario_list(select_name="base")

	def _load_network(self) -> None:
		wdn = self.wdn_input.currentText().strip()
		if not wdn:
			return
		self.solver_params.wdn = wdn
		try:
			self.plot.load_network(wdn)
			self._posteriori_ensure_base_scenario(wdn)
			self._posteriori_refresh_scenario_list(select_name="base")
			if not self._updating_measurement_text:
				self._measurement_text_changed(self.measurement_list.text())
			self.status_bar.showMessage(f"Loaded network {wdn}")
		except Exception as exc:
			self.status_bar.showMessage(f"Failed to load network: {exc}")

	def _measurement_updated(self, nodes: List[str]) -> None:
		if self.tabs.currentIndex() == getattr(self, "_post_tab_index", -1):
			self._post_updating_measurement_text = True
			self.post_measurement_sites.setText(", ".join(nodes))
			self._post_updating_measurement_text = False
			self._post_dirty = True
			self._posteriori_refresh_elimination_node(nodes)
			return
		self._updating_measurement_text = True
		self.measurement_list.setText(", ".join(nodes))
		self._updating_measurement_text = False
		self._measurement_text_changed(self.measurement_list.text())

	def _apply_show_sensors_mode(self, enabled: bool) -> None:
		if hasattr(self, "plot"):
			self.plot.set_show_sensors_mode(bool(enabled))

	def _set_measurement_input_validity(self, valid: bool) -> None:
		if valid:
			self.measurement_list.setStyleSheet("")
		else:
			self.measurement_list.setStyleSheet("QLineEdit { border: 2px solid #c53030; background: #fff5f5; }")

	def _set_measurement_data_input_validity(self, valid: bool) -> None:
		if valid:
			self.measurement_data_input.setStyleSheet("")
		else:
			self.measurement_data_input.setStyleSheet("QPlainTextEdit { border: 2px solid #c53030; background: #fff5f5; }")

	def _measurement_source_value(self) -> str:
		return str(self.measurement_source.currentData() or "from_w_d")

	def _current_mode(self) -> str:
		return str(self.mode_input.currentData() or self.mode_input.currentText())

	def _measurement_source_options_for_mode(self, mode: str) -> List[tuple[str, str]]:
		if mode == "W_d_M":
			return [
				("custom input", "custom"),
				("from W_d", "from_w_d"),
				("base", "base"),
			]
		if mode == "W_h_M":
			return [
				("custom input", "custom"),
				("from W_h", "from_w_h"),
				("base", "base"),
			]
		if mode == "C_h_fixed":
			return [
				("from W_h", "from_w_h"),
			]
		if mode == "C_h":
			return [
				("from W_h", "from_w_h"),
				("from W_d", "from_w_d"),
				("base", "base"),
				("custom input", "custom"),
			]
		if mode == "C_d":
			return [
				("from W_d", "from_w_d"),
				("from W_h", "from_w_h"),
				("base", "base"),
				("custom input", "custom"),
			]
		return [
			("from W_d", "from_w_d"),
			("from W_h", "from_w_h"),
			("base", "base"),
			("custom input", "custom"),
		]

	def _default_measurement_source_for_mode(self, mode: str) -> str:
		if mode == "W_d_M":
			return "custom"
		if mode == "W_h_M":
			return "custom"
		if mode in {"C_h", "C_h_fixed"}:
			return "from_w_h"
		if mode == "C_d":
			return "from_w_d"
		return "from_w_d"

	def _rebuild_measurement_source_options(self, preserve_current: bool = True) -> None:
		mode = self._current_mode()
		current = self._measurement_source_value() if preserve_current else ""
		options = self._measurement_source_options_for_mode(mode)
		valid_values = [value for _label, value in options]
		if current not in valid_values:
			current = self._default_measurement_source_for_mode(mode)
		elif current in {"from_w_d", "from_w_h"}:
			current = self._default_measurement_source_for_mode(mode)

		self.measurement_source.blockSignals(True)
		self.measurement_source.clear()
		for label, value in options:
			self.measurement_source.addItem(label, value)
			self.measurement_source.setCurrentIndex(max(0, self.measurement_source.findData(current)))
		self.measurement_source.blockSignals(False)
		self.solver_params.measurement_source = self._measurement_source_value()

	def _mode_changed(self, *_args) -> None:
		mode = self._current_mode()
		self.solver_params.mode = str(mode)
		self.model_a.set_mode(mode)
		self.model_b.set_mode(mode)
		self._rebuild_measurement_source_options(preserve_current=True)
		self._update_measurement_source_visibility()
		self._measurement_data_text_changed()

	def _update_measurement_source_visibility(self) -> None:
		show_measurement = self._current_mode() in {"W_d_M", "W_h_M", "C_d", "C_h", "C_h_fixed"}
		measurement_label = self.measurement_source.parentWidget().layout().labelForField(self.measurement_source)
		if measurement_label is not None:
			measurement_label.setVisible(show_measurement)
		self.measurement_source.setVisible(show_measurement)

		is_custom = show_measurement and self._measurement_source_value() == "custom"
		self.measurement_data_input.setVisible(is_custom)
		label = self.measurement_data_input.parentWidget().layout().labelForField(self.measurement_data_input)
		if label is not None:
			label.setVisible(is_custom)

	def _measurement_source_changed(self) -> None:
		self.solver_params.measurement_source = self._measurement_source_value()
		self._update_measurement_source_visibility()
		self._measurement_data_text_changed()

	def _parse_measurement_data_input(self, text: str) -> tuple[bool, Dict[str, float] | None]:
		payload = text.strip()
		if self._current_mode() not in {"W_d", "W_h", "B", "W_d_M", "W_h_M", "C_d", "C_h", "C_h_fixed"} or self._measurement_source_value() != "custom":
			return True, None
		if not payload:
			return False, None
		try:
			data = json.loads(payload)
		except json.JSONDecodeError:
			return False, None
		if not isinstance(data, dict):
			return False, None
		parsed: Dict[str, float] = {}
		for key, value in data.items():
			try:
				parsed[str(key)] = float(value)
			except (TypeError, ValueError):
				return False, None
		return True, parsed

	def _measurement_data_text_changed(self) -> None:
		text = self.measurement_data_input.toPlainText()
		self.solver_params.measurement_data = text
		valid, _parsed = self._parse_measurement_data_input(text)
		self._measurement_data_valid = valid
		self._set_measurement_data_input_validity(valid)

	def _parse_measurement_input(self, text: str) -> tuple[bool, object, List[str]]:
		value = text.strip()
		junction_nodes: List[str] = []
		if hasattr(self, "plot"):
			junction_nodes = self.plot.get_junction_nodes()
		junction_set = set(junction_nodes)
		n_nodes = len(junction_nodes)

		if value == "":
			return True, [], []

		m_range = re.fullmatch(r"#(\d+)-#(\d+)", value)
		if m_range:
			a = int(m_range.group(1))
			b = int(m_range.group(2))
			if a > b:
				return False, None, []
			if a > n_nodes or b > n_nodes:
				return False, None, []
			return True, f"#{a}-#{b}", []

		m_exact = re.fullmatch(r"#(\d+)", value)
		if m_exact:
			a = int(m_exact.group(1))
			if a > n_nodes:
				return False, None, []
			return True, f"#{a}", []

		tokens = [tok for tok in re.split(r"[\s,;]+", value) if tok]
		if not tokens:
			return True, [], []
		ordered: List[str] = []
		seen: set[str] = set()
		for tok in tokens:
			if tok not in junction_set:
				return False, None, []
			if tok in seen:
				continue
			seen.add(tok)
			ordered.append(tok)
		return True, ordered, ordered

	def _measurement_text_changed(self, text: str) -> None:
		valid, parsed_value, graph_nodes = self._parse_measurement_input(text)
		self._measurement_valid = valid
		self._set_measurement_input_validity(valid)
		if not valid:
			return
		self._measurement_value = parsed_value
		if isinstance(parsed_value, list):
			self.solver_params.measurement_sites = [str(x) for x in parsed_value]
		else:
			self.solver_params.measurement_sites = []
		if hasattr(self, "plot"):
			self.plot.set_measurements(graph_nodes)

	def _get_wdn_names(self) -> List[str]:
		wdn_dir = os.path.join(ROOT_DIR, "wdn")
		if not os.path.isdir(wdn_dir):
			return []
		names = []
		for entry in os.listdir(wdn_dir):
			if entry.lower().endswith(".inp"):
				names.append(os.path.splitext(entry)[0])
		return sorted(names)

	def _solver_payload_from_widget(self, model_widget: "SolverModelWidget") -> Dict[str, object]:
		payload = asdict(self.solver_params)
		wdn = self.wdn_input.currentText().strip() or self.solver_params.wdn
		mode = self._current_mode()
		measurement_data = None
		if self._measurement_source_value() == "custom":
			_valid, parsed = self._parse_measurement_data_input(self.measurement_data_input.toPlainText())
			measurement_data = parsed
		payload.update(
			{
				"WDN": wdn,
				"MODE": mode,
				"MEASUREMENT_SITES": self._measurement_value,
				"MEASUREMENT_SOURCE": self._measurement_source_value(),
				"MEASUREMENT_DATA": measurement_data,
			}
		)
		payload.update(model_widget.get_payload())
		# Posteriori modes are measurement-fitting modes; force absolute measurement matching.
		if mode in {"W_d_M", "W_h_M"}:
			payload["MEASUREMENT_HEADS_EQUAL_ONLY"] = False
		# A-priori W-modes must not depend on a fixed measurement instance.
		if mode in {"W_d", "W_h", "B"}:
			payload["MEASUREMENT_SOURCE"] = "base"
			payload["MEASUREMENT_DATA"] = None
		# xd backend currently supports equality-only measurement mode, so route W_d(M) to classical.
		if mode == "W_d_M" and str(payload.get("METHOD", "")).lower() == "xd":
			payload["METHOD"] = "classical"
		if mode == "B":
			payload.pop("NORM", None)
		return payload

	def _expand_measurement_sets(self) -> List[List[str]]:
		from itertools import combinations as _combinations

		value = self._measurement_value
		if isinstance(value, list):
			return [[str(x) for x in value]]

		if isinstance(value, str):
			text = value.strip()
			junction_nodes = self.plot.get_junction_nodes() if hasattr(self, "plot") else []
			if text in {"", "#0"}:
				return [[]]
			m_exact = re.fullmatch(r"#(\d+)", text)
			if m_exact:
				k = int(m_exact.group(1))
				if k < 0 or k > len(junction_nodes):
					return []
				if k == 0:
					return [[]]
				return [list(c) for c in _combinations(junction_nodes, k)]

			m_range = re.fullmatch(r"#(\d+)-#(\d+)", text)
			if m_range:
				a = int(m_range.group(1))
				b = int(m_range.group(2))
				if a < 0 or b < 0 or a > b:
					return []
				if a > len(junction_nodes):
					return []
				b = min(b, len(junction_nodes))
				sets: List[List[str]] = []
				for k in range(a, b + 1):
					if k == 0:
						sets.append([])
					else:
						sets.extend(list(c) for c in _combinations(junction_nodes, k))
				return sets

		return [[]]

	def _run_solver(self) -> None:
		if self._solver_worker is not None and self._solver_worker.isRunning():
			return  # already running
		self._linearization_lookup_active = False

		if not self._measurement_valid:
			QtWidgets.QMessageBox.warning(
				self,
				"Invalid Sites",
				"Sites input is invalid. Valid examples:\n"
				"- blank or #0\n"
				"- #2\n"
				"- #1-#3\n"
				"- 1, 5, 8",
			)
			self.status_bar.showMessage("Solver not started: invalid site input.")
			return

		if not self._measurement_data_valid:
			QtWidgets.QMessageBox.warning(
				self,
				"Invalid Measurement Data",
				"Custom measurement data must be a JSON dictionary mapping node ids to numbers, with reserved keys like -1 for total demand.",
			)
			self.status_bar.showMessage("Solver not started: invalid measurement data input.")
			return

		wdn = self.wdn_input.currentText().strip() or self.solver_params.wdn
		_sync_wdn_index(wdn)
		if (
			isinstance(self._measurement_value, str)
			and self._measurement_value.startswith("#")
			and self._measurement_source_value() == "custom"
		):
			QtWidgets.QMessageBox.warning(
				self,
				"Custom Measurement Not Supported for #k",
				"Custom measurement data defines one fixed measurement instance.\n"
				"Combinatorial site specs like #k or #a-#b represent many different site sets.\n"
				"Please select explicit sites (e.g. 1,5,8) when using custom measurement data.",
			)
			self.status_bar.showMessage("Solver not started: custom measurement with combinatorial site spec.")
			return

		if self.linearization_check.isChecked() and not self._linearization_lookup_done:
			self._linearization_auto_solve_pending = True
			self._set_linearization_status("Running certification lookup before solve...")
			self._look_for_linearization()
			return
		self._linearization_auto_solve_pending = False

		runs: List[tuple[str, Dict[str, object]]] = []
		base_a = self._linearization_payload_from_widget(self.model_a, lookup_only=False) if self.linearization_check.isChecked() else self._solver_payload_from_widget(self.model_a)
		runs.append(("A", dict(base_a)))
		if self.comparison_mode_check.isChecked():
			base_b = self._solver_payload_from_widget(self.model_b)
			runs.append(("B", dict(base_b)))
		self._launch_solver_runs(runs)

	def _start_next_solver_run(self) -> None:
		if not self._pending_runs:
			self._finalize_all_runs()
			return

		label, payload = self._pending_runs[0]
		self._active_run_label = label
		total_runs = len(self._pending_runs) + len(self._completed_runs)
		run_num = len(self._completed_runs) + 1
		self.progress_label.setText(f"Run {run_num}/{total_runs} ({label})  0 / ?")

		wdn = str(payload.get("WDN", "wdn"))
		solver_hash = compute_hash(_solver_cache_hash_payload(payload))
		index_path = _data_index_path(wdn)
		index = _load_index_with_legacy(index_path, LEGACY_DATA_INDEX)
		cached_dir = index.get(solver_hash)
		resolved_cached = (cached_dir if os.path.isabs(cached_dir) else os.path.join(ROOT_DIR, cached_dir)) if cached_dir else None
		if resolved_cached and os.path.isdir(resolved_cached):
			use_cached = True
			if bool(payload.get("DYNAMIC_MULTISTART", False)):
				dd_path = os.path.join(resolved_cached, "demand_distance.json")
				if os.path.isfile(dd_path):
					try:
						dd = _read_json(dd_path)
					except Exception:
						dd = {}
					radius_ref = _float_or_default(payload.get("DMS_RADIUS"), float("inf"))
					conclusive, lower, upper, cert = _dms_cache_conclusive(dd, radius_ref)
					if not conclusive:
						use_cached = False
						self.status_bar.showMessage(
							f"Run {label}: cached DMS inconclusive for r={radius_ref:.6f} in [{lower:.6f}, {upper:.6f}] ({cert or 'n/a'}), recomputing..."
						)
			if use_cached:
				self._apply_linearization_result(resolved_cached, bool(payload.get("_linearization_lookup", False)))
				self.status_bar.showMessage(f"Run {label} cached: {resolved_cached}")
				self._completed_runs.append((label, 0, "", "", resolved_cached))
				self._pending_runs.pop(0)
				self._start_next_solver_run()
				return

		output_dir = os.path.join("data", wdn, solver_hash)
		payload = dict(payload)
		payload["OUTPUT_DIR"] = output_dir
		payload["SOLVER_HASH"] = solver_hash

		os.makedirs(CACHE_DIR, exist_ok=True)
		config_path = os.path.join(CACHE_DIR, f"solver-{solver_hash}.json")
		_write_json(config_path, payload)

		# store index info for the finished callback
		payload["_index_path"] = index_path
		payload["_index"] = index
		self._pending_runs[0] = (label, payload)

		cmd = [sys.executable, "inverse.py", "--config", config_path]
		self._solver_worker = SolverWorker(cmd, self)
		self._solver_worker.progress_updated.connect(self._on_solver_progress)
		self._solver_worker.linearization_updated.connect(self._on_linearization_progress)
		self._solver_worker.finished_with_code.connect(self._on_solver_finished)
		self._solver_worker.start()
		self.status_bar.showMessage(f"Running solver (run {run_num}/{total_runs}, model {label})...")

	def _on_solver_progress(self, current: int, total: int) -> None:
		total_runs = len(self._pending_runs) + len(self._completed_runs)
		run_num = len(self._completed_runs) + 1
		self.progress_bar.setRange(0, total)
		self.progress_bar.setValue(current)
		self.progress_label.setText(f"Run {run_num}/{total_runs} ({self._active_run_label})  {current} / {total}")
		self.status_bar.showMessage(
			f"Solver run {run_num}/{total_runs} ({self._active_run_label}): configuration {current} of {total}..."
		)

	def _on_linearization_progress(self, checked: int, total: int, certified: int) -> None:
		if not hasattr(self, "linearization_check") or not self.linearization_check.isChecked():
			return
		self._set_linearization_status(
			f"Scanning pipes: {checked}/{total} checked, {certified} linearizable."
		)

	def _on_solver_finished(self, returncode: int, stdout: str, stderr: str) -> None:
		self._solver_worker = None

		label, payload = self._pending_runs[0]
		output_dir = str(payload.get("OUTPUT_DIR", ""))
		index_path = str(payload.get("_index_path", ""))
		index = payload.get("_index", {})

		fake_cmd = [sys.executable, "inverse.py"]
		completed = subprocess.CompletedProcess(fake_cmd, returncode, stdout=stdout, stderr=stderr)
		self._store_output(f"Solver Output ({label})", fake_cmd, completed)

		resolvedout = output_dir if os.path.isabs(output_dir) else os.path.join(ROOT_DIR, output_dir)
		if returncode == 0 and os.path.isdir(resolvedout):
			if isinstance(index, dict):
				solver_hash = str(payload.get("SOLVER_HASH", ""))
				index[solver_hash] = output_dir
				save_index(index_path, index, ROOT_DIR)
			solver_hash = str(payload.get("SOLVER_HASH", ""))
			_write_gui_hash(resolvedout, solver_hash)
			self._apply_linearization_result(resolvedout, bool(payload.get("_linearization_lookup", False)))
			self.status_bar.showMessage(f"Run {label} done: {output_dir}")
		else:
			output_dir = ""
			self.status_bar.showMessage(f"Run {label} failed. See Show Output.")

		self._completed_runs.append((label, returncode, stdout, stderr, output_dir))
		self._pending_runs.pop(0)
		self._start_next_solver_run()

	def _apply_linearization_result(self, output_dir: str, lookup_run: bool) -> None:
		lin_path = os.path.join(output_dir, "linearization.json")
		if not os.path.isfile(lin_path):
			if lookup_run:
				self._linearization_ready = False
				self._linearization_lookup_done = False
				self._linearized_pipe_ids = {}
				self._linearized_pipe_ids_base = {}
				self._linearization_scale_required = {}
				if hasattr(self, "plot"):
					self.plot.set_linearized_pipes({})
					self.plot.set_linearization_scale_needed({})
				self.solve_button.setEnabled(False)
				self._set_linearization_status("Lookup finished, but no linearization artifact was found.")
			return
		try:
			payload = _read_json(lin_path)
		except Exception:
			if lookup_run:
				self._set_linearization_status("Lookup finished, but linearization artifact could not be read.")
			return
		linearized = payload.get("linearized_pipes", {})
		if not isinstance(linearized, dict):
			linearized = {}
		self._linearized_pipe_ids_base = {str(pid): float(q0) for pid, q0 in linearized.items()}

		scale_required: Dict[str, float] = {}
		pipe_bounds = payload.get("pipe_bounds", {})
		epsilon_h = _float_or_default(payload.get("epsilon_h"), 0.0)
		median_delta_h = _float_or_default(payload.get("median_delta_h"), 1.0)
		scale_now = epsilon_h / max(median_delta_h, 1e-12)
		if isinstance(pipe_bounds, dict):
			for pipe_id, raw in pipe_bounds.items():
				if not isinstance(raw, dict):
					continue
				q0 = _float_or_default(raw.get("q0"), 0.0)
				self._linearized_pipe_ids_base[str(pipe_id)] = q0
				q_min = _float_or_default(raw.get("q_min"), 0.0)
				q_max = _float_or_default(raw.get("q_max"), 0.0)
				delta_now = abs(_float_or_default(raw.get("delta_e"), 0.0))
				delta_req = max(q_max - q0, q0 - q_min, 0.0)
				sign_change = (q0 > 0.0 and (q_min <= 0.0 or q_max <= 0.0)) or (q0 < 0.0 and (q_min >= 0.0 or q_max >= 0.0))
				if sign_change:
					scale_required[str(pipe_id)] = float("inf")
					continue
				if delta_req <= 0.0:
					scale_required[str(pipe_id)] = 0.0
					continue
				if delta_now <= 1e-12:
					scale_required[str(pipe_id)] = float("inf")
					continue
				scale_required[str(pipe_id)] = float(scale_now) * float((delta_req / delta_now) ** 2)
		self._linearization_scale_required = scale_required

		if hasattr(self, "plot"):
			self.plot.set_linearization_scale_needed(self._linearization_scale_required)
		self._linearization_lookup_done = True
		self._recompute_linearized_pipes_from_scale()
		self._update_linearization_controls()
		if lookup_run:
			self.solve_button.setEnabled(self._linearization_lookup_done)
			if self._linearization_lookup_done:
				self._set_linearization_status(
					f"Lookup done: {len(self._linearized_pipe_ids)} linearizable at eps scale {self.linearization_eps.value():.3g}, {len(self._linearization_scale_required)} pipes scored."
				)
				self.status_bar.showMessage(
					f"Linearization lookup complete: {len(self._linearized_pipe_ids)} linearizable pipes."
				)
			else:
				self._set_linearization_status("Certification complete: 0 pipes certified.")
				self.status_bar.showMessage("No pipes were certified for linearization.")

	def _finalize_all_runs(self) -> None:
		if getattr(self, "_linearization_lookup_active", False):
			self.solve_button.setEnabled(self._linearization_lookup_done)
			self._linearization_lookup_active = False
			if self._linearization_auto_solve_pending and self._linearization_lookup_done:
				QtCore.QTimer.singleShot(0, self._run_solver)
				self._linearization_auto_solve_pending = False
		else:
			self.solve_button.setEnabled(True)
		self.progress_bar.setVisible(False)
		self.progress_label.setVisible(False)

		successful = [(label, out_dir) for label, rc, _, _, out_dir in self._completed_runs if rc == 0 and out_dir]
		if not successful:
			return

		# Try to show CSV results table
		rows = self._collect_result_rows()
		if rows:
			comparison = len(successful) > 1
			dlg = ResultsTableDialog(rows, comparison=comparison, parent=self)
			dlg.show()

		# Show demand-distance plots unless user opted out
		if self.generate_plots_check.isChecked():
			dirs_with_labels: List[tuple[str, str]] = []
			for label, out_dir in successful:
				for d in self._collect_demand_distance_dirs(out_dir):
					dirs_with_labels.append((d, label))
			self._show_demand_distance_plots(dirs_with_labels)

	def _collect_result_rows(self) -> "List[Dict[str, str]]":
		rows: List[Dict[str, str]] = []
		for label, rc, _stdout, _stderr, out_dir in self._completed_runs:
			if rc != 0 or not out_dir:
				continue
			for data_dir in self._collect_demand_distance_dirs(str(out_dir)):
				dd_path = os.path.join(data_dir, "demand_distance.json")
				params_path = os.path.join(data_dir, "parameters.json")
				try:
					with open(dd_path, encoding="utf-8") as fh:
						dd = json.load(fh)
					params: Dict[str, object] = {}
					if os.path.isfile(params_path):
						with open(params_path, encoding="utf-8") as fh:
							params = json.load(fh)
					meas_nodes = params.get("MEASUREMENT_NODES", [])
					row: Dict[str, str] = {
						"measurement_sites": str(meas_nodes),
						"measurement_count": str(len(meas_nodes) if isinstance(meas_nodes, list) else 0),
						"radius": str(dd.get("radius", "")),
						"mode": str(dd.get("mode", "")),
						"method": str(dd.get("method", "")),
						"W_d": str(dd.get("W_d", "")),
						"C_d": str(dd.get("C_d", "")),
						"W_h": str(dd.get("W_h", "")),
						"C_h": str(dd.get("C_h", "")),
						"success": str(dd.get("success", "")),
						"max_violation": str(dd.get("max_violation", "")),
						"min_demand_viol": str(dd.get("min_demand_viol", "")),
						"objective": str(dd.get("objective", "")),
						"solver_status": str(dd.get("solver_status", "")),
						"best_bound": str(dd.get("best_bound", "")),
						"output_dir": data_dir,
						"model": label,
					}
					rows.append(row)
				except Exception:
					pass
		return rows

	def _collect_demand_distance_dirs(self, root_dir: str) -> List[str]:
		dirs: List[str] = []
		if os.path.isfile(os.path.join(root_dir, "demand_distance.json")):
			dirs.append(root_dir)
		for current, _, files in os.walk(root_dir):
			if current == root_dir:
				continue
			if "demand_distance.json" in files:
				dirs.append(current)
		return sorted(dirs)

	def _build_demand_distance_plot(self, data_dir: str, temp_dir: str, index: int, name_prefix: str = "") -> tuple[str, str, float, int] | None:
		return _build_demand_distance_plot_fn(
			data_dir,
			temp_dir,
			index,
			name_prefix,
			bool(self.show_only_demand_deltas_check.isChecked()),
		)

	def _show_demand_distance_plots(self, dirs_with_labels: "List[tuple[str, str]]") -> None:
		if not dirs_with_labels:
			return
		os.makedirs(CACHE_DIR, exist_ok=True)
		temp_dir = tempfile.mkdtemp(prefix="demand_distance_", dir=CACHE_DIR)

		self.progress_bar.setRange(0, len(dirs_with_labels))
		self.progress_bar.setValue(0)
		self.progress_label.setText(f"Generating plots (0 / {len(dirs_with_labels)})...")
		self.progress_bar.setVisible(True)
		self.progress_label.setVisible(True)

		self._plot_worker = PlotWorker(
			dirs_with_labels,
			temp_dir,
			bool(self.show_only_demand_deltas_check.isChecked()),
			self,
		)
		self._plot_worker.plots_ready.connect(
			lambda items, td: self._on_plots_ready(items, td)
		)
		self._plot_worker.plot_progress.connect(self._on_plot_progress)
		self._plot_worker.start()

	def _on_plot_progress(self, done: int, total: int) -> None:
		self.progress_bar.setRange(0, total)
		self.progress_bar.setValue(done)
		self.progress_label.setText(f"Generating plots ({done} / {total})...")

	def _on_plots_ready(self, items: List[tuple[str, str, float, int]], temp_dir: str) -> None:
		self.progress_bar.setVisible(False)
		self.progress_label.setVisible(False)
		self._plot_worker = None
		if not items:
			shutil.rmtree(temp_dir, ignore_errors=True)
			return
		dialog = DemandDistanceViewerDialog(items, temp_dir, self)
		dialog.exec()

	def _store_output(self, title: str, cmd: List[str], proc: subprocess.CompletedProcess) -> None:
		cmd_text = " ".join(cmd)
		stdout_text = proc.stdout or ""
		stderr_text = proc.stderr or ""
		combined = (
			f"Command:\n{cmd_text}\n\n"
			f"Exit Code: {proc.returncode}\n\n"
			"STDOUT:\n"
			f"{stdout_text}\n\n"
			"STDERR:\n"
			f"{stderr_text}"
		)
		self._last_output_title = title
		self._last_output_text = combined

	def _show_last_output(self) -> None:
		if not self._last_output_text:
			self.status_bar.showMessage("No command output captured yet.")
			return
		dialog = QtWidgets.QDialog(self)
		dialog.setWindowTitle(self._last_output_title)
		dialog.resize(900, 600)
		layout = QtWidgets.QVBoxLayout(dialog)
		text_edit = QtWidgets.QPlainTextEdit()
		text_edit.setReadOnly(True)
		text_edit.setPlainText(self._last_output_text)
		layout.addWidget(text_edit)

		buttons = QtWidgets.QHBoxLayout()
		buttons.addStretch(1)
		save_button = QtWidgets.QPushButton("Save as .txt")
		buttons.addWidget(save_button)
		close_button = QtWidgets.QPushButton("Close")
		buttons.addWidget(close_button)
		layout.addLayout(buttons)

		def _save_output() -> None:
			path, _ = QtWidgets.QFileDialog.getSaveFileName(
				dialog,
				"Save Output",
				"output.txt",
				"Text Files (*.txt);;All Files (*)",
			)
			if not path:
				return
			with open(path, "w", encoding="utf-8") as f:
				f.write(self._last_output_text)

		save_button.clicked.connect(_save_output)
		close_button.clicked.connect(dialog.close)
		dialog.exec()


def main() -> None:
	app = QtWidgets.QApplication(sys.argv)
	locale = QtCore.QLocale.c()
	locale.setNumberOptions(QtCore.QLocale.NumberOption.RejectGroupSeparator)
	QtCore.QLocale.setDefault(locale)
	window = MainWindow()
	window.resize(1200, 800)
	window.show()
	sys.exit(app.exec())


if __name__ == "__main__":
	main()
