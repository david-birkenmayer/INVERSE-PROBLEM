import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import asdict
from typing import Dict, List, Optional

from PyQt6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
	sys.path.insert(0, ROOT_DIR)

from gui.cache import compute_hash, load_index, save_index
from gui.state import ScenarioParams, SolverParams
from step1_io import load_inp_network


CACHE_DIR = ".gui_cache"
LEGACY_SCENARIO_INDEX = os.path.join("scenario", "cache_index.json")
LEGACY_DATA_INDEX = os.path.join("data", "cache_index.json")


def _write_json(path: str, payload: Dict[str, object]) -> None:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2, sort_keys=True)


def _read_json(path: str) -> Dict[str, object]:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def _scenario_index_path(wdn: str) -> str:
	return os.path.join("scenario", wdn, "cache_index.json")


def _data_index_path(wdn: str) -> str:
	return os.path.join("data", wdn, "cache_index.json")


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
	scenario_file = str(params.get("SCENARIO_FILE", ""))
	if not scenario_file or not os.path.isfile(scenario_file):
		return None
	scenario = _read_json(scenario_file)

	wdn = str(params.get("WDN") or params.get("WDN_NAME") or scenario.get("wdn") or scenario.get("wdn_name") or "")
	if not wdn:
		return None
	inp_path = os.path.join("wdn", f"{wdn}.inp")

	bounds_path = os.path.join(data_dir, "c_bounds.json")
	if os.path.isfile(bounds_path):
		bounds_payload = _read_json(bounds_path)
		c_bounds = {k: float(v) for k, v in bounds_payload.get("c_bounds", {}).items()}
		C_bounds = {k: float(v) for k, v in bounds_payload.get("C_bounds", {}).items()}
	else:
		c_bounds = {k: float(v) for k, v in scenario.get("c_bounds", {}).items()}
		C_bounds = {k: float(v) for k, v in scenario.get("C_bounds", {}).items()}

	measurement_nodes = [str(x) for x in params.get("MEASUREMENT_NODES") or []]
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
		measurement_count=int(demand_distance.get("p", len(measurement_nodes))),
		output_dir=temp_dir,
		c_bounds=c_bounds,
		C_bounds=C_bounds,
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
		name = f"{name} (no measurements)"
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
		form = QtWidgets.QFormLayout(self)
		self._form = form

		self.solver_mode = QtWidgets.QComboBox()
		self.solver_mode.addItems(["Hexaly_xd", "Hexaly"])
		self.solver_mode.setCurrentText(defaults.solver)
		self.solver_mode.currentTextChanged.connect(self._update_visibility)
		self._add_row(form, "solver", "Solver", self.solver_mode)

		self.norm_value = self._new_double_spin(0.1, 100.0, defaults.norm, decimals=4, step=0.1)
		self._add_row(form, "norm", "Norm", self.norm_value)

		self.demand_lb = self._new_double_spin(0.0, 1e6, defaults.demand_lb, decimals=8, step=1e-6)
		self._add_row(form, "demand_lb", "Demand LB", self.demand_lb)

		self.fix_total_demand = QtWidgets.QCheckBox()
		self.fix_total_demand.setChecked(defaults.fix_total_demand)
		self._add_row(form, "fix_total", "Fix Total Demand", self.fix_total_demand)

		self.measurement_heads_equal = QtWidgets.QCheckBox()
		self.measurement_heads_equal.setChecked(defaults.measurement_heads_equal_only)
		self._add_row(form, "heads_equal", "Heads Equal at Sensors", self.measurement_heads_equal)

		self.multi_starts = QtWidgets.QSpinBox()
		self.multi_starts.setRange(1, 100)
		self.multi_starts.setValue(defaults.multi_starts)
		self._add_row(form, "multi_starts", "Multi Starts", self.multi_starts)

		self.multi_noise = self._new_double_spin(0.0, 100.0, defaults.multi_start_noise, decimals=4, step=0.01)
		self._add_row(form, "multi_noise", "Noise Abs", self.multi_noise)

		self.multi_noise_rel = self._new_double_spin(0.0, 100.0, defaults.multi_start_noise_rel, decimals=4, step=0.01)
		self._add_row(form, "multi_noise_rel", "Noise Rel", self.multi_noise_rel)

		self.restriction_mode = QtWidgets.QComboBox()
		self.restriction_mode.addItems(["none", "radius_to_fixed", "deviation_to_fixed"])
		self.restriction_mode.setCurrentText(defaults.demand_restriction_mode or "none")
		self.restriction_mode.currentTextChanged.connect(self._update_visibility)
		self._add_row(form, "restriction_mode", "Restriction Mode", self.restriction_mode)

		radius_value = defaults.radius_to_fixed if defaults.radius_to_fixed is not None else 0.0
		self.radius_to_fixed = self._new_double_spin(0.0, 1e9, float(radius_value), decimals=6, step=0.1)
		self._add_row(form, "radius_to_fixed", "Radius to Fixed", self.radius_to_fixed)

		alpha_value = defaults.deviation_alpha if defaults.deviation_alpha is not None else 0.0
		self.deviation_alpha = self._new_double_spin(0.0, 1e9, float(alpha_value), decimals=6, step=0.1)
		self._add_row(form, "deviation_alpha", "Deviation Alpha", self.deviation_alpha)

		self.check_xd_base = QtWidgets.QCheckBox()
		self.check_xd_base.setChecked(defaults.check_xd_base_feasibility)
		self._add_row(form, "check_xd_base", "Check XD Base Feasibility", self.check_xd_base)

		self.check_xd_cycle = QtWidgets.QCheckBox()
		self.check_xd_cycle.setChecked(defaults.check_xd_cycle_bounds)
		self._add_row(form, "check_xd_cycle", "Check XD Cycle Bounds", self.check_xd_cycle)

		self.xd_cycle_basis = QtWidgets.QComboBox()
		self.xd_cycle_basis.addItems(["planar", "graph"])
		self.xd_cycle_basis.setCurrentText(defaults.xd_cycle_basis_mode)
		self._add_row(form, "xd_cycle_basis", "XD Cycle Basis", self.xd_cycle_basis)

		self.skip_feasibility = QtWidgets.QCheckBox()
		self.skip_feasibility.setChecked(defaults.skip_feasibility_solve)
		self._add_row(form, "skip_feasibility", "Skip Feasibility Solve", self.skip_feasibility)

		self.fixed_reference = QtWidgets.QCheckBox()
		self.fixed_reference.setChecked(defaults.fixed_reference)
		self._add_row(form, "fixed_reference", "Fixed Reference (b)", self.fixed_reference)

		self.hexaly_time_limit = QtWidgets.QSpinBox()
		self.hexaly_time_limit.setRange(1, 36000)
		self.hexaly_time_limit.setValue(defaults.hexaly_time_limit)
		self._add_row(form, "hexaly_time", "Time Limit", self.hexaly_time_limit)

		self._update_visibility()

	def _update_visibility(self) -> None:
		solver = self.solver_mode.currentText()
		restriction = self.restriction_mode.currentText()
		for key in ["check_xd_base", "check_xd_cycle", "xd_cycle_basis", "skip_feasibility", "fixed_reference"]:
			self._set_row_visible(key, solver == "Hexaly_xd")
		self._set_row_visible("radius_to_fixed", restriction == "radius_to_fixed")
		self._set_row_visible("deviation_alpha", restriction == "deviation_to_fixed")

	def get_payload(self) -> Dict[str, object]:
		restriction = self.restriction_mode.currentText()
		return {
			"SOLVER": self.solver_mode.currentText(),
			"NORM": self.norm_value.value(),
			"DEMAND_LB": self.demand_lb.value(),
			"FIX_TOTAL_DEMAND": self.fix_total_demand.isChecked(),
			"MEASUREMENT_HEADS_EQUAL_ONLY": self.measurement_heads_equal.isChecked(),
			"MULTI_STARTS": self.multi_starts.value(),
			"MULTI_START_NOISE": self.multi_noise.value(),
			"MULTI_START_NOISE_REL": self.multi_noise_rel.value(),
			"DEMAND_RESTRICTION_MODE": None if restriction == "none" else restriction,
			"RADIUS_TO_FIXED": self.radius_to_fixed.value() if restriction == "radius_to_fixed" else None,
			"DEVIATION_ALPHA": self.deviation_alpha.value() if restriction == "deviation_to_fixed" else None,
			"CHECK_XD_BASE_FEASIBILITY": self.check_xd_base.isChecked(),
			"CHECK_XD_CYCLE_BOUNDS": self.check_xd_cycle.isChecked(),
			"XD_CYCLE_BASIS_MODE": self.xd_cycle_basis.currentText(),
			"SKIP_FEASIBILITY_SOLVE": self.skip_feasibility.isChecked(),
			"FIXED_REFERENCE": self.fixed_reference.isChecked(),
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
			self.ax.plot([start[0], end[0]], [start[1], end[1]], color="#a0aec0", linewidth=1.0, zorder=1)

		junctions = [n for n in self._node_ids if n not in self._reservoir_set]
		measurements = [n for n in junctions if n in self._measurement_set]
		others = [n for n in junctions if n not in self._measurement_set]
		reservoirs = [n for n in self._node_ids if n in self._reservoir_set]

		def _scatter(nodes: List[str], marker: str, color: str, edge: str) -> None:
			if not nodes:
				return
			xs = [self._node_pos[n][0] for n in nodes]
			ys = [self._node_pos[n][1] for n in nodes]
			self.ax.scatter(xs, ys, s=110, marker=marker, c=color, edgecolors=edge, zorder=2)

		_scatter(others, "o", "#f2f2f2", "#333333")
		_scatter(measurements, "h", "#90cdf4", "#1a365d")
		_scatter(reservoirs, "s", "#90cdf4", "#1a365d")

		for node_id in self._node_ids:
			pos = self._node_pos.get(node_id)
			if pos is None:
				continue
			self.ax.text(pos[0], pos[1], str(node_id), fontsize=7, ha="center", va="center", color="#111111", zorder=3)

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
	"""Shows the measurement-summary CSV in a table with toggleable extra columns."""

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
		self._table.setColumnCount(len(cols))
		self._table.setHorizontalHeaderLabels(cols)
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


class MainWindow(QtWidgets.QMainWindow):
	def __init__(self) -> None:
		super().__init__()
		self.setWindowTitle("Inverse Problem GUI")
		self.scenario_params = ScenarioParams()
		self.solver_params = SolverParams()
		self._scenario_rows: Dict[str, tuple[QtWidgets.QLabel, QtWidgets.QWidget]] = {}
		self._scenario_form: QtWidgets.QFormLayout | None = None
		self._measurement_value: object = []
		self._measurement_valid = True
		self._updating_measurement_text = False
		self._last_output_title = ""
		self._last_output_text = ""
		self._solver_worker: SolverWorker | None = None
		self._plot_worker: PlotWorker | None = None
		self._pending_runs: List[tuple[str, Dict[str, object]]] = []
		self._completed_runs: List[tuple] = []
		self._active_run_label: str = ""
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
		self.wdn_input.setCurrentText(self.scenario_params.wdn)
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
		self._build_scenario_tab()
		self._build_solver_tab()

		splitter.addWidget(left)
		self._build_local_search_tab()
		self.tabs.currentChanged.connect(self._on_tab_changed)
		self.plot = NetworkPlot()
		self.plot.measurement_changed.connect(self._measurement_updated)
		splitter.addWidget(self.plot)
		splitter.setStretchFactor(1, 1)
		self._measurement_text_changed(self.measurement_list.text())

		self.status_bar = self.statusBar()
		self.output_button = QtWidgets.QPushButton("Show Output")
		self.output_button.clicked.connect(self._show_last_output)
		self.status_bar.addPermanentWidget(self.output_button)

	def _build_scenario_tab(self) -> None:
		widget = QtWidgets.QWidget()
		form = QtWidgets.QFormLayout(widget)
		self._scenario_form = form
		self.scenario_name = QtWidgets.QLineEdit(self.scenario_params.scenario_name)
		self._add_form_row(form, "scenario_name", "Scenario Name", self.scenario_name)
		self.scenario_source = QtWidgets.QComboBox()
		self.scenario_source.addItems(["base", "random"])
		self.scenario_source.setCurrentText(self.scenario_params.scenario_source)
		self._add_form_row(form, "scenario_source", "Scenario Source", self.scenario_source)

		self.pipe_bounds = QtWidgets.QCheckBox()
		self.pipe_bounds.setChecked(self.scenario_params.pipe_bounds)
		self.pipe_bounds.stateChanged.connect(self._update_pipe_bounds_visibility)
		self._add_form_row(form, "pipe_bounds", "Pipe Bounds", self.pipe_bounds)

		self.pipe_bound_safeness = self._new_double_spin(0.0, 1.0, self.scenario_params.pipe_bound_safeness, decimals=4, step=0.05)
		self._add_form_row(form, "pipe_bound_safeness", "Bound Safeness", self.pipe_bound_safeness)

		self.pipe_bound_method = QtWidgets.QComboBox()
		self.pipe_bound_method.addItems(["perturb", "montecarlo"])
		self.pipe_bound_method.setCurrentText(self.scenario_params.pipe_bound_method)
		self.pipe_bound_method.currentTextChanged.connect(self._update_pipe_bounds_visibility)
		self._add_form_row(form, "pipe_bound_method", "Bound Method", self.pipe_bound_method)

		self.pipe_bound_policy = QtWidgets.QComboBox()
		self.pipe_bound_policy.addItems(["minmax", "quantile"])
		self.pipe_bound_policy.setCurrentText(self.scenario_params.pipe_bound_policy)
		self.pipe_bound_policy.currentTextChanged.connect(self._update_pipe_bounds_visibility)
		self._add_form_row(form, "pipe_bound_policy", "Bound Policy", self.pipe_bound_policy)

		self.pipe_bound_samples = QtWidgets.QSpinBox()
		self.pipe_bound_samples.setRange(1, 10000)
		self.pipe_bound_samples.setValue(self.scenario_params.pipe_bound_perturb_samples)
		self._add_form_row(form, "pipe_bound_samples", "Perturb Samples", self.pipe_bound_samples)

		self.pipe_bound_sigma = self._new_double_spin(0.0, 10.0, self.scenario_params.pipe_bound_perturb_sigma, decimals=4, step=0.01)
		self._add_form_row(form, "pipe_bound_sigma", "Perturb Sigma", self.pipe_bound_sigma)

		self.pipe_bound_seed = QtWidgets.QSpinBox()
		self.pipe_bound_seed.setRange(0, 2**31 - 1)
		self.pipe_bound_seed.setValue(self.scenario_params.pipe_bound_perturb_seed)
		self._add_form_row(form, "pipe_bound_seed", "Perturb Seed", self.pipe_bound_seed)

		self.pipe_bound_fix_total = QtWidgets.QCheckBox()
		self.pipe_bound_fix_total.setChecked(self.scenario_params.pipe_bound_perturb_fix_total)
		self._add_form_row(form, "pipe_bound_fix_total", "Fix Total Demand", self.pipe_bound_fix_total)

		self.pipe_bound_base = QtWidgets.QComboBox()
		self.pipe_bound_base.addItems(["base", "auto", "scenario"])
		self.pipe_bound_base.setCurrentText(self.scenario_params.pipe_bound_perturb_base)
		self._add_form_row(form, "pipe_bound_base", "Perturb Base", self.pipe_bound_base)

		self.generate_button = QtWidgets.QPushButton("Generate Scenario")
		self.generate_button.clicked.connect(self._generate_scenario)
		form.addRow(self.generate_button)

		self.tabs.addTab(widget, "Scenario")
		self._update_pipe_bounds_visibility()
		self._freeze_form_label_width()

	def _add_form_row(
		self,
		form: QtWidgets.QFormLayout,
		key: str,
		label: str,
		widget: QtWidgets.QWidget,
	) -> None:
		row_label = QtWidgets.QLabel(label)
		form.addRow(row_label, widget)
		self._scenario_rows[key] = (row_label, widget)

	def _freeze_form_label_width(self) -> None:
		if self._scenario_form is None:
			return
		max_width = 0
		for label, _ in self._scenario_rows.values():
			max_width = max(max_width, label.sizeHint().width())
		for label, _ in self._scenario_rows.values():
			label.setMinimumWidth(max_width)

	def _set_row_visible(self, key: str, visible: bool) -> None:
		row = self._scenario_rows.get(key)
		if not row:
			return
		label, widget = row
		if self._scenario_form is not None:
			self._scenario_form.setRowVisible(label, visible)
		else:
			label.setVisible(visible)
			widget.setVisible(visible)

	def _update_pipe_bounds_visibility(self) -> None:
		pipe_bounds_on = self.pipe_bounds.isChecked()
		policy = self.pipe_bound_policy.currentText()
		method = self.pipe_bound_method.currentText()

		for key in [
			"pipe_bound_method",
			"pipe_bound_policy",
			"pipe_bound_safeness",
			"pipe_bound_samples",
			"pipe_bound_sigma",
			"pipe_bound_seed",
			"pipe_bound_fix_total",
			"pipe_bound_base",
		]:
			self._set_row_visible(key, pipe_bounds_on)

		if not pipe_bounds_on:
			return

		self._set_row_visible("pipe_bound_safeness", policy == "quantile")

		perturb_only = method == "perturb"
		for key in [
			"pipe_bound_samples",
			"pipe_bound_sigma",
			"pipe_bound_seed",
			"pipe_bound_fix_total",
			"pipe_bound_base",
		]:
			self._set_row_visible(key, perturb_only)

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
		self.solver_scenario_name = QtWidgets.QLineEdit(self.solver_params.scenario_name)
		shared_form.addRow("Scenario Name", self.solver_scenario_name)
		self.measurement_list = QtWidgets.QLineEdit("")
		self.measurement_list.setPlaceholderText("blank/#0, #a, #a-#b, or comma-separated node ids")
		self.measurement_list.textChanged.connect(self._measurement_text_changed)
		shared_form.addRow("Measurements", self.measurement_list)
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
		self.ls_scenario_name = QtWidgets.QLineEdit(self.solver_params.scenario_name)
		form.addRow("Scenario Name", self.ls_scenario_name)
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
		scenario_name = self.ls_scenario_name.text().strip() or self.solver_params.scenario_name
		scenario_path = os.path.join("scenario", wdn, f"{scenario_name}.json")
		if not os.path.exists(scenario_path):
			QtWidgets.QMessageBox.warning(
				self,
				"Scenario Missing",
				f"Scenario file not found:\n{scenario_path}\n\nGenerate it in the Scenario tab first.",
			)
			return
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
		self.scenario_params.wdn = self.wdn_input.currentText().strip()
		self.solver_params.wdn = self.scenario_params.wdn
		if not self._updating_measurement_text:
			self._measurement_text_changed(self.measurement_list.text())

	def _load_network(self) -> None:
		wdn = self.wdn_input.currentText().strip()
		if not wdn:
			return
		self.scenario_params.wdn = wdn
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

	def _set_measurement_input_validity(self, valid: bool) -> None:
		if valid:
			self.measurement_list.setStyleSheet("")
		else:
			self.measurement_list.setStyleSheet("QLineEdit { border: 2px solid #c53030; background: #fff5f5; }")

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

	def _scenario_payload(self) -> Dict[str, object]:
		payload = asdict(self.scenario_params)
		wdn = self.wdn_input.currentText().strip() or self.scenario_params.wdn
		scenario_name = self.scenario_name.text().strip() or self.scenario_params.scenario_name
		payload.update(
			{
				"WDN": wdn,
				"SCENARIO_NAME": scenario_name,
				"SCENARIO_SOURCE": self.scenario_source.currentText(),
				"PIPE_BOUNDS": self.pipe_bounds.isChecked(),
				"PIPE_BOUND_SAFENESS": self.pipe_bound_safeness.value(),
				"PIPE_BOUND_METHOD": self.pipe_bound_method.currentText(),
				"PIPE_BOUND_POLICY": self.pipe_bound_policy.currentText(),
				"PIPE_BOUND_PERTURB_SAMPLES": self.pipe_bound_samples.value(),
				"PIPE_BOUND_PERTURB_SIGMA": self.pipe_bound_sigma.value(),
				"PIPE_BOUND_PERTURB_SEED": self.pipe_bound_seed.value(),
				"PIPE_BOUND_PERTURB_FIX_TOTAL": self.pipe_bound_fix_total.isChecked(),
				"PIPE_BOUND_PERTURB_BASE": self.pipe_bound_base.currentText(),
			}
		)
		return payload

	def _solver_payload_from_widget(self, model_widget: "SolverModelWidget") -> Dict[str, object]:
		payload = asdict(self.solver_params)
		wdn = self.wdn_input.currentText().strip() or self.solver_params.wdn
		scenario_name = self.solver_scenario_name.text().strip() or self.solver_params.scenario_name
		payload.update(
			{
				"WDN": wdn,
				"SCENARIO": scenario_name,
				"SCENARIO_FILE": os.path.join("scenario", wdn, f"{scenario_name}.json"),
				"MEASUREMENT_SITES": self._measurement_value,
			}
		)
		payload.update(model_widget.get_payload())
		return payload

	def _generate_scenario(self) -> None:
		payload = self._scenario_payload()
		wdn = str(payload.get("WDN", "wdn"))
		scenario_hash = compute_hash(payload)
		index_path = _scenario_index_path(wdn)
		index = _load_index_with_legacy(index_path, LEGACY_SCENARIO_INDEX)
		cached_path = index.get(scenario_hash)
		resolved_scenario = (cached_path if os.path.isabs(cached_path) else os.path.join(ROOT_DIR, cached_path)) if cached_path else None
		if resolved_scenario and os.path.exists(resolved_scenario):
			self.status_bar.showMessage(f"Scenario cached: {cached_path}")
			self.solver_scenario_name.setText(str(payload.get("SCENARIO_NAME", "")))
			return

		scenario_name = payload.get("SCENARIO_NAME", "scenario")
		output_path = os.path.join("scenario", wdn, f"{scenario_name}.json")
		payload["SCENARIO_HASH"] = scenario_hash

		os.makedirs(CACHE_DIR, exist_ok=True)
		config_path = os.path.join(CACHE_DIR, f"scenario-{scenario_hash}.json")
		_write_json(config_path, payload)

		cmd = [sys.executable, "scenario.py", "--config", config_path, "--output", output_path]
		self.status_bar.showMessage("Generating scenario...")
		proc = subprocess.run(cmd, capture_output=True, text=True)
		self._store_output("Scenario Output", cmd, proc)
		if proc.returncode != 0:
			self.status_bar.showMessage(f"Scenario failed: {proc.stderr.strip()}")
			return

		index[scenario_hash] = output_path
		save_index(index_path, index, ROOT_DIR)
		self.status_bar.showMessage(f"Scenario ready: {output_path}")
		self.solver_scenario_name.setText(str(scenario_name))
		try:
			scenario_payload = _read_json(output_path)
			base_flows = {k: float(v) for k, v in scenario_payload.get("base_flows", {}).items()}
			c_bounds = {k: float(v) for k, v in scenario_payload.get("c_bounds", {}).items()}
			C_bounds = {k: float(v) for k, v in scenario_payload.get("C_bounds", {}).items()}
			self.plot.set_pipe_classes(base_flows, c_bounds, C_bounds)
		except Exception as exc:
			self.status_bar.showMessage(f"Scenario ready (plot skipped): {exc}")

	def _run_solver(self) -> None:
		if self._solver_worker is not None and self._solver_worker.isRunning():
			return  # already running

		if not self._measurement_valid:
			QtWidgets.QMessageBox.warning(
				self,
				"Invalid Measurements",
				"Measurements input is invalid. Valid examples:\n"
				"- blank or #0\n"
				"- #2\n"
				"- #1-#3\n"
				"- 1, 5, 8",
			)
			self.status_bar.showMessage("Solver not started: invalid measurement input.")
			return

		payload_a = self._solver_payload_from_widget(self.model_a)
		scenario_path = str(payload_a.get("SCENARIO_FILE", ""))
		if not scenario_path or not os.path.exists(scenario_path):
			QtWidgets.QMessageBox.warning(
				self,
				"Scenario Missing",
				f"Scenario file does not exist for selected WDN/scenario:\n{scenario_path}\n\n"
				"Generate the scenario first, or choose a valid scenario name.",
			)
			self.status_bar.showMessage("Solver not started: scenario file is missing.")
			return

		self._pending_runs = [("A", payload_a)]
		self._completed_runs = []
		if self.comparison_mode_check.isChecked():
			payload_b = self._solver_payload_from_widget(self.model_b)
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

	def _parse_summary_csv_path(self, stdout: str) -> "str | None":
		for line in stdout.splitlines():
			if line.startswith("GUI_SUMMARY_CSV:"):
				return line[len("GUI_SUMMARY_CSV:"):].strip()
		return None

	def _find_summary_csv_in_output(self, out_dir: str) -> "str | None":
		"""Fallback lookup for cached runs where stdout has no CSV marker."""
		if not out_dir or not os.path.isdir(out_dir):
			return None
		matches: List[str] = []
		for current, _dirs, files in os.walk(out_dir):
			for name in files:
				if "measurement-summary" in name and name.lower().endswith(".csv"):
					matches.append(os.path.join(current, name))
		if not matches:
			return None
		matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
		return matches[0]

	def _collect_result_rows(self) -> "List[Dict[str, str]]":
		import csv as _csv
		rows: List[Dict[str, str]] = []
		for label, rc, stdout, _stderr, out_dir in self._completed_runs:
			if rc != 0:
				continue
			csv_path = self._parse_summary_csv_path(stdout)
			if (not csv_path) or (not os.path.isfile(csv_path)):
				csv_path = self._find_summary_csv_in_output(str(out_dir))
			if csv_path and os.path.isfile(csv_path):
				with open(csv_path, newline="", encoding="utf-8") as fh:
					reader = _csv.DictReader(fh)
					for row in reader:
						row["model"] = label
						rows.append(dict(row))
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
