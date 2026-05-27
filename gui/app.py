import json
import os
import re
import hashlib
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


def _build_demand_distance_plot_fn(data_dir: str, temp_dir: str, index: int, name_prefix: str = "") -> "tuple[str, str, float, int] | None":
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
		configured_extra_demand=configured_extra_demand,
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


class SolverWorker(QtCore.QThread):
	"""Runs the solver subprocess off the GUI thread, streaming progress lines."""

	progress_updated = QtCore.pyqtSignal(int, int)   # current, total
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
		parent: "QtWidgets.QWidget | None" = None,
	) -> None:
		super().__init__(parent)
		self._plot_dir_labels = plot_dir_labels
		self._temp_dir = temp_dir

	def run(self) -> None:
		items: List[tuple[str, str, float, int]] = []
		total = len(self._plot_dir_labels)
		for idx, (run_dir, label) in enumerate(self._plot_dir_labels, start=1):
			built = _build_demand_distance_plot_fn(run_dir, self._temp_dir, idx, label)
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

	def cancel(self) -> None:
		self._cancelled = True

	def run(self) -> None:
		current: frozenset = frozenset(self._start_nodes)
		while not self._cancelled:
			if current not in self._lookup:
				radius = self._evaluate(current)
				if radius is None:
					self.status_updated.emit(f"Solver failed for {sorted(current)}, stopping.")
					break
				self._lookup[current] = radius
			current_radius = self._lookup[current]
			self.status_updated.emit(f"Current: {sorted(current)} -> radius={current_radius:.6f}")

			candidates = self._generate_neighbors(current)
			best: "frozenset | None" = None
			best_radius = current_radius
			for candidate in candidates:
				if self._cancelled:
					break
				if candidate not in self._lookup:
					radius = self._evaluate(candidate)
					if radius is None:
						continue
					self._lookup[candidate] = radius
				r = self._lookup[candidate]
				if r < best_radius:
					best_radius = r
					best = candidate

			if best is None or self._cancelled:
				if not self._cancelled:
					self.status_updated.emit("Local optimum reached - search complete.")
				break
			self.status_updated.emit(
				f"Improving: {sorted(current)} -> {sorted(best)}  "
				f"(radius {current_radius:.6f} -> {best_radius:.6f})"
			)
			current = best
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

	def _evaluate(self, config: frozenset) -> "float | None":
		nodes = sorted(config)
		payload = dict(self._payload_template)
		payload["MEASUREMENT_SITES"] = nodes
		solver_hash = compute_hash(payload)
		cached_dir = self._index.get(solver_hash)
		if cached_dir:
			resolved_dir = cached_dir if os.path.isabs(cached_dir) else os.path.join(ROOT_DIR, cached_dir)
			if os.path.isdir(resolved_dir):
				return self._read_radius(resolved_dir, nodes, cached=True)
		output_dir = os.path.join("data", self._wdn, solver_hash[:8])
		payload["OUTPUT_DIR"] = output_dir
		payload["SOLVER_HASH"] = solver_hash
		os.makedirs(CACHE_DIR, exist_ok=True)
		config_path = os.path.join(CACHE_DIR, f"solver-{solver_hash}.json")
		_write_json(config_path, payload)
		self.status_updated.emit(f"Evaluating {nodes}...")
		proc = subprocess.run(
			[sys.executable, "inverse.py", "--config", config_path],
			capture_output=True,
			text=True,
		)
		resolvedout = output_dir if os.path.isabs(output_dir) else os.path.join(ROOT_DIR, output_dir)
		if proc.returncode != 0 or not os.path.isdir(resolvedout):
			self.status_updated.emit(f"Solver failed for {nodes}: {proc.stderr.strip()[:200]}")
			return None
		self._index[solver_hash] = output_dir
		save_index(self._index_path, self._index, ROOT_DIR)
		_write_gui_hash(resolvedout, solver_hash)
		return self._read_radius(resolvedout, nodes, cached=False)
	def _read_radius(self, output_dir: str, nodes: List[str], *, cached: bool) -> "float | None":
		dd_path = os.path.join(output_dir, "demand_distance.json")
		if not os.path.isfile(dd_path):
			for root, _dirs, files in os.walk(output_dir):
				if "demand_distance.json" in files:
					dd_path = os.path.join(root, "demand_distance.json")
					break
		if not os.path.isfile(dd_path):
			return None
		dd = _read_json(dd_path)
		radius = float(dd.get("radius", float("inf")))
		row: Dict[str, str] = {
			"measurement_sites": str(nodes),
			"radius": f"{radius:.6f}",
			"cached": str(cached),
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
		self.match_total_demand.setChecked(defaults.match_reservoir_outflow_between_pairs)
		self._add_row(general_form, "match_total_demand", "Match Total Demand", self.match_total_demand)
		root.addWidget(general_group)

		solver_group = QtWidgets.QGroupBox("Solver Specification")
		solver_form = QtWidgets.QFormLayout(solver_group)

		self.method = QtWidgets.QComboBox()
		self.method.addItem("head loss (x)", "xd")
		self.method.addItem("head (h)", "classical")
		self.method.setCurrentIndex(max(0, self.method.findData(defaults.method)))
		solver_form.addRow("Method", self.method)

		self.demand_lb = self._new_double_spin(0.0, 1e6, defaults.demand_lb, decimals=8, step=1e-6)
		solver_form.addRow("Demand LB", self.demand_lb)

		self.multi_starts = QtWidgets.QSpinBox()
		self.multi_starts.setRange(1, 100)
		self.multi_starts.setValue(defaults.multi_starts)
		solver_form.addRow("Multi Starts", self.multi_starts)

		self.multi_noise = self._new_double_spin(0.0, 100.0, defaults.multi_start_noise, decimals=4, step=0.01)
		solver_form.addRow("Noise Abs", self.multi_noise)

		self.multi_noise_rel = self._new_double_spin(0.0, 100.0, defaults.multi_start_noise_rel, decimals=4, step=0.01)
		solver_form.addRow("Noise Rel", self.multi_noise_rel)

		self.hexaly_time_limit = QtWidgets.QSpinBox()
		self.hexaly_time_limit.setRange(1, 36000)
		self.hexaly_time_limit.setValue(defaults.hexaly_time_limit)
		solver_form.addRow("Time Limit", self.hexaly_time_limit)
		root.addWidget(solver_group)

		self._mode = defaults.mode
		self._solver_group = solver_group
		self._update_visibility()

	def set_mode(self, mode: str) -> None:
		self._mode = str(mode)
		self._update_visibility()

	def _update_visibility(self) -> None:
		mode = str(getattr(self, "_mode", "W_d"))
		show_method = mode in {"W_d", "C_d"}
		self._solver_group.setVisible(True)
		for row in range(self._solver_group.layout().rowCount()):
			pass
		self._set_row_visible("match_total_demand", mode in {"W_d", "C_d"})
		self.method.setVisible(show_method)
		method_label = self._solver_group.layout().labelForField(self.method)
		if method_label is not None:
			method_label.setVisible(show_method)

	def get_payload(self) -> Dict[str, object]:
		return {
			"METHOD": str(self.method.currentData()),
			"NORM": self.norm_value.value(),
			"DEMAND_LB": self.demand_lb.value(),
			"MEASUREMENT_HEADS_EQUAL_ONLY": self.measurement_heads_equal.isChecked(),
			"MATCH_RESERVOIR_OUTFLOW_BETWEEN_PAIRS": self.match_total_demand.isChecked(),
			"MULTI_STARTS": self.multi_starts.value(),
			"MULTI_START_NOISE": self.multi_noise.value(),
			"MULTI_START_NOISE_REL": self.multi_noise_rel.value(),
			"HEXALY_TIME_LIMIT": self.hexaly_time_limit.value(),
		}


class NetworkPlot(QtWidgets.QWidget):
	measurement_changed = QtCore.pyqtSignal(list)

	def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
		super().__init__(parent)
		self.figure = Figure(figsize=(6, 5))
		self.canvas = FigureCanvas(self.figure)
		self.ax = self.figure.add_subplot(111)
		self._node_ids: List[str] = []
		self._node_pos: Dict[str, tuple[float, float]] = {}
		self._measurement_set: set[str] = set()
		self._reservoir_set: set[str] = set()
		self._show_sensors_mode = False
		self._network = None
		self._pipe_classes: Optional[Dict[str, Dict[str, float]]] = None
		self._init_ui()

	def _init_ui(self) -> None:
		layout = QtWidgets.QVBoxLayout(self)
		layout.addWidget(self.canvas)
		self.canvas.mpl_connect("button_press_event", self._on_click)

	def load_network(self, wdn_name: str) -> None:
		self.ax.clear()
		self._node_ids = []
		self._node_pos = {}
		self._measurement_set = set()
		self._reservoir_set = set()
		self._pipe_classes = None

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

	def set_measurements(self, nodes: List[str]) -> None:
		self._measurement_set = set(nodes)
		self._redraw()

	def set_show_sensors_mode(self, enabled: bool) -> None:
		self._show_sensors_mode = bool(enabled)
		self._redraw()

	def get_junction_nodes(self) -> List[str]:
		if self._network is None:
			return []
		return sorted(str(node_id) for node_id in self._network.junctions.keys())

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
			edge_color = "#cbd5e0" if self._show_sensors_mode else "#a0aec0"
			edge_alpha = 0.35 if self._show_sensors_mode else 1.0
			self.ax.plot([start[0], end[0]], [start[1], end[1]], color=edge_color, linewidth=1.0, alpha=edge_alpha, zorder=1)

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

		if self._show_sensors_mode:
			_scatter(others, "o", "#e2e8f0", "#94a3b8", size=70.0, alpha=0.45)
			_scatter(measurements, "h", "#ffdd57", "#8a5a00", size=190.0, alpha=1.0)
			_scatter(reservoirs, "s", "#90cdf4", "#1a365d", size=135.0, alpha=0.95)
		else:
			_scatter(others, "o", "#f2f2f2", "#333333")
			_scatter(measurements, "h", "#90cdf4", "#1a365d")
			_scatter(reservoirs, "s", "#90cdf4", "#1a365d")

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
		self.ax.set_title(wdn_name)
		self.ax.axis("off")
		self.canvas.draw_idle()

	def _on_click(self, event) -> None:
		if self._network is None or event.inaxes != self.ax:
			return
		if event.xdata is None or event.ydata is None:
			return

		x, y = float(event.xdata), float(event.ydata)
		min_dist = None
		closest = None
		for node_id in self._node_ids:
			if node_id in self._reservoir_set:
				continue
			pos = self._node_pos.get(node_id)
			if pos is None:
				continue
			dx = pos[0] - x
			dy = pos[1] - y
			dist = dx * dx + dy * dy
			if min_dist is None or dist < min_dist:
				min_dist = dist
				closest = node_id

		if closest is None:
			return
		xs = [p[0] for p in self._node_pos.values()]
		ys = [p[1] for p in self._node_pos.values()]
		if not xs or not ys:
			return
		range_x = max(xs) - min(xs)
		range_y = max(ys) - min(ys)
		threshold = 0.02 * max(range_x, range_y, 1.0)
		if min_dist is None or min_dist > threshold * threshold:
			return

		if closest in self._measurement_set:
			self._measurement_set.remove(closest)
		else:
			self._measurement_set.add(closest)
		self._redraw()
		self.measurement_changed.emit(sorted(self._measurement_set))


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

		splitter.addWidget(left)
		self._build_local_search_tab()
		self._build_gnn_tab()
		self.tabs.currentChanged.connect(self._on_tab_changed)
		self.plot = NetworkPlot()
		self.plot.measurement_changed.connect(self._measurement_updated)
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

		# Shared fields
		shared_group = QtWidgets.QGroupBox("Shared")
		shared_form = QtWidgets.QFormLayout(shared_group)
		self.mode_input = QtWidgets.QComboBox()
		self.mode_input.addItem("W_d", "W_d")
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
		self.ls_nodes_input = QtWidgets.QLineEdit()
		self.ls_nodes_input.setPlaceholderText("e.g. 3, 7, 12")
		form.addRow("Starting Nodes", self.ls_nodes_input)
		outer.addLayout(form)

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

	def _run_local_search(self) -> None:
		if self._ls_worker is not None and self._ls_worker.isRunning():
			return
		valid, nodes = self._parse_ls_nodes(self.ls_nodes_input.text())
		if not valid or len(nodes) < 1:
			QtWidgets.QMessageBox.warning(
				self,
				"Invalid Nodes",
				"Please enter at least one explicit junction node ID, comma-separated.\n"
				"Shortcuts like #3 are not supported here.",
			)
			return
		wdn = self.wdn_input.currentText().strip() or self.solver_params.wdn
		adjacency = self.plot.get_pipe_adjacency()
		index_path = _data_index_path(wdn)
		existing_index = _load_index_with_legacy(index_path, LEGACY_DATA_INDEX)
		# Build payload template (no MEASUREMENT_SITES — worker sets it per config)
		payload_template = self._solver_payload_from_widget(self.model_a)
		payload_template.pop("MEASUREMENT_SITES", None)

		self.ls_log.clear()
		self.ls_status_label.setText("Running...")
		self.ls_run_button.setEnabled(False)
		self.ls_cancel_button.setEnabled(True)

		self._ls_worker = LocalSearchWorker(
			nodes, adjacency, payload_template, wdn, index_path, existing_index, self
		)
		self._ls_worker.status_updated.connect(self._on_ls_status)
		self._ls_worker.row_added.connect(self._on_ls_row)
		self._ls_worker.finished_signal.connect(self._on_ls_finished)
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

	def _on_ls_finished(self, rows: list) -> None:
		self._ls_worker = None
		self.ls_run_button.setEnabled(True)
		self.ls_cancel_button.setEnabled(False)
		self.ls_status_label.setText(f"Done. {len(rows)} configurations evaluated.")
		self.ls_log.appendPlainText(f"--- Finished: {len(rows)} configurations evaluated ---")
		if rows:
			dlg = ResultsTableDialog(rows, comparison=False, parent=self)
			dlg.show()

	def _wdn_changed(self) -> None:
		self.solver_params.wdn = self.wdn_input.currentText().strip()
		if not self._updating_measurement_text:
			self._measurement_text_changed(self.measurement_list.text())
		if hasattr(self, "gnn_default_nodes_label"):
			default_nodes = self._gnn_default_nodes()
			self.gnn_default_nodes_label.setText(", ".join(default_nodes) if default_nodes else "(none)")
			self._gnn_refresh_dataset_status()
			self._gnn_populate_model_combos()

	def _load_network(self) -> None:
		wdn = self.wdn_input.currentText().strip()
		if not wdn:
			return
		self.solver_params.wdn = wdn
		try:
			self.plot.load_network(wdn)
			if not self._updating_measurement_text:
				self._measurement_text_changed(self.measurement_list.text())
			self.status_bar.showMessage(f"Loaded network {wdn}")
		except Exception as exc:
			self.status_bar.showMessage(f"Failed to load network: {exc}")

	def _measurement_updated(self, nodes: List[str]) -> None:
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
		show_measurement = self._current_mode() in {"C_d", "C_h", "C_h_fixed"}
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
		if self._current_mode() not in {"C_d", "C_h", "C_h_fixed"} or self._measurement_source_value() != "custom":
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
			if a >= b:
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
		measurement_data = None
		if self._measurement_source_value() == "custom":
			_valid, parsed = self._parse_measurement_data_input(self.measurement_data_input.toPlainText())
			measurement_data = parsed
		payload.update(
			{
				"WDN": wdn,
				"MODE": self._current_mode(),
				"MEASUREMENT_SITES": self._measurement_value,
				"MEASUREMENT_SOURCE": self._measurement_source_value(),
				"MEASUREMENT_DATA": measurement_data,
			}
		)
		payload.update(model_widget.get_payload())
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
		measurement_sets = self._expand_measurement_sets()
		if not measurement_sets:
			QtWidgets.QMessageBox.warning(
				self,
				"Invalid Sites",
				"No valid site sets could be expanded from the site input.",
			)
			self.status_bar.showMessage("Solver not started: no valid site sets.")
			return

		base_a = self._solver_payload_from_widget(self.model_a)
		self._pending_runs = []
		self._completed_runs = []
		for nodes in measurement_sets:
			payload_a = dict(base_a)
			payload_a["MEASUREMENT_SITES"] = nodes
			self._pending_runs.append(("A", payload_a))
		if self.comparison_mode_check.isChecked():
			base_b = self._solver_payload_from_widget(self.model_b)
			for nodes in measurement_sets:
				payload_b = dict(base_b)
				payload_b["MEASUREMENT_SITES"] = nodes
				self._pending_runs.append(("B", payload_b))

		self.solve_button.setEnabled(False)
		self.progress_bar.setRange(0, 1)
		self.progress_bar.setValue(0)
		self.progress_bar.setVisible(True)
		self.progress_label.setVisible(True)
		self._start_next_solver_run()

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
		solver_hash = compute_hash(payload)
		index_path = _data_index_path(wdn)
		index = _load_index_with_legacy(index_path, LEGACY_DATA_INDEX)
		cached_dir = index.get(solver_hash)
		resolved_cached = (cached_dir if os.path.isabs(cached_dir) else os.path.join(ROOT_DIR, cached_dir)) if cached_dir else None
		if resolved_cached and os.path.isdir(resolved_cached):
			self.status_bar.showMessage(f"Run {label} cached: {resolved_cached}")
			self._completed_runs.append((label, 0, "", "", resolved_cached))
			self._pending_runs.pop(0)
			self._start_next_solver_run()
			return

		output_dir = os.path.join("data", wdn, solver_hash[:8])
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
			self.status_bar.showMessage(f"Run {label} done: {output_dir}")
		else:
			output_dir = ""
			self.status_bar.showMessage(f"Run {label} failed. See Show Output.")

		self._completed_runs.append((label, returncode, stdout, stderr, output_dir))
		self._pending_runs.pop(0)
		self._start_next_solver_run()

	def _finalize_all_runs(self) -> None:
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
		return _build_demand_distance_plot_fn(data_dir, temp_dir, index, name_prefix)

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

		self._plot_worker = PlotWorker(dirs_with_labels, temp_dir, self)
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
