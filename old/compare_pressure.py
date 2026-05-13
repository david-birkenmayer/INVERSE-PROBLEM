"""
Pressure-only comparison of two GNN models on a shared test split.

Both models are evaluated on the test data from dataset_dir_a.
Generates:
  - r2_scatter_a.png / r2_scatter_b.png   (per-model R² scatter)
  - r2_scatter_overlay.png                (side-by-side comparison)
  - aed_a.png / aed_b.png                 (advanced error distribution, if viz available)
  - results.json                          (global metrics for both models)
"""

from __future__ import annotations

import json
import hashlib
import pickle
import importlib.util
import sys
from pathlib import Path
from typing import Optional

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

_HERE = Path(__file__).resolve().parent        # old/
ROOT_DIR = _HERE.parent                        # project root

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ---------------------------------------------------------------------------
# GCN model — must match gnn_model.ipynb / evaluation.ipynb exactly
# ---------------------------------------------------------------------------

class GCN(torch.nn.Module):
    def __init__(self, dim_in: int, dim_h: int = 256, dim_out: int = 1):
        super().__init__()
        self.dim_hidden = dim_h * 4
        self.batch_norm1 = torch.nn.BatchNorm1d(self.dim_hidden)
        self.batch_norm2 = torch.nn.BatchNorm1d(self.dim_hidden)
        self.batch_norm3 = torch.nn.BatchNorm1d(self.dim_hidden)
        self.gcn1 = GCNConv(dim_in, self.dim_hidden, improved=False, cached=False)
        self.gcn2 = GCNConv(self.dim_hidden, self.dim_hidden, improved=False, cached=False)
        self.gcn3 = GCNConv(self.dim_hidden, self.dim_hidden, improved=False, cached=False)
        self.linear1 = torch.nn.Linear(self.dim_hidden, dim_h)
        self.linear2 = torch.nn.Linear(dim_h, dim_out)
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x, edge_index, edge_attr=None):
        h = self.gcn1(x, edge_index, edge_attr)
        h = self.batch_norm1(h)
        h = torch.relu(h)
        h = self.dropout(h)
        h2 = self.gcn2(h, edge_index, edge_attr)
        h2 = self.batch_norm2(h2)
        h2 = torch.relu(h2)
        h2 = self.dropout(h2)
        h2 = h2 + h
        h3 = self.gcn3(h2, edge_index, edge_attr)
        h3 = self.batch_norm3(h3)
        h3 = torch.relu(h3)
        h3 = self.dropout(h3)
        h3 = h3 + h2
        h = self.linear1(h3)
        h = torch.relu(h)
        h = self.dropout(h)
        h = self.linear2(h)
        return h


# ---------------------------------------------------------------------------
# Locate the shared algorithms/ directory
# ---------------------------------------------------------------------------

def _find_algorithms_root() -> Optional[Path]:
    candidates = [
        ROOT_DIR / "algorithms",
        ROOT_DIR.parent / "SMARTWINE" / "algorithms",
    ]
    for c in candidates:
        if (c / "8_evaluate-model" / "metrics.py").exists():
            return c
    return None


def _load_metrics_module(algorithms_root: Path):
    metrics_path = algorithms_root / "8_evaluate-model" / "metrics.py"
    spec = importlib.util.spec_from_file_location("_cmp_metrics", metrics_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.calculate_node_metrics, mod.calculate_global_metrics, mod.create_scatter_plot


def _load_viz_module(algorithms_root: Path):
    """Load create_advanced_error_distribution_viz, patching the REGISTRY to avoid registration errors."""
    try:
        import algorithms.registry as registry  # type: ignore
        def _noop(_name):
            def decorator(func):
                return func
            return decorator
        registry.REGISTRY.register = _noop
    except ImportError:
        pass

    viz_path = algorithms_root / "9_visualize-evaluation" / "visualize_evaluation.py"
    spec = importlib.util.spec_from_file_location("_cmp_viz", viz_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.create_advanced_error_distribution_viz


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _run_inference(model: GCN, dataset, device):
    """Returns (predictions, actuals, masks) as np.ndarray of shape (N, num_nodes)."""
    predictions, actuals, masks = [], [], []
    model.eval()
    with torch.no_grad():
        for data in dataset:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr)
            predictions.append(out.squeeze().cpu().numpy())
            actuals.append(data.y.squeeze().cpu().numpy())
            masks.append(data.mask.cpu().numpy())
    return np.stack(predictions), np.stack(actuals), np.stack(masks)


# ---------------------------------------------------------------------------
# Node index helpers (mirrors evaluation.ipynb)
# ---------------------------------------------------------------------------

def _indices_from_mapping(node_mapping: dict, target_nodes: list) -> list:
    if not node_mapping or not target_nodes:
        return []
    target_strs = {str(n) for n in target_nodes}
    indices = []
    for idx_str, node in node_mapping.items():
        if str(node) in target_strs:
            try:
                indices.append(int(idx_str))
            except ValueError:
                continue
    return indices


def _indices_from_list(node_list: list, target_nodes: list) -> list:
    if not node_list or not target_nodes:
        return []
    target_strs = {str(n) for n in target_nodes}
    return [idx for idx, node in enumerate(node_list) if str(node) in target_strs]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_wdn_config(wdn_name: str) -> dict:
    path = ROOT_DIR / "wdn" / f"{wdn_name}.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _prep_aed_graph(G: nx.Graph, virtual_nodes: list) -> nx.Graph:
    G_aed = G.copy()
    if virtual_nodes:
        G_aed.remove_nodes_from([n for n in virtual_nodes if n in G_aed])
    if isinstance(G_aed, nx.MultiDiGraph):
        G_aed = nx.DiGraph(G_aed)
    elif isinstance(G_aed, nx.MultiGraph):
        G_aed = nx.Graph(G_aed)
    return G_aed


def _denorm(arr: np.ndarray, stats: Optional[dict]) -> np.ndarray:
    """Denormalize a pressure array using dataset_stats keys.

    Accepts either ``{"pressure_range": {"min": …, "max": …}}`` (legacy) or
    the flat ``{"min_p": …, "max_p": …}`` format used by data_generator.
    """
    if not stats:
        return arr
    # flat format: min_p / max_p
    if "min_p" in stats and "max_p" in stats:
        pr_min = float(stats["min_p"])
        pr_max = float(stats["max_p"])
        return arr * (pr_max - pr_min) + pr_min
    # legacy nested format
    pr = stats.get("pressure_range", {})
    if pr and "min" in pr and "max" in pr:
        pr_min = float(pr["min"])
        pr_max = float(pr["max"])
        return arr * (pr_max - pr_min) + pr_min
    return arr


# ---------------------------------------------------------------------------
# Demand reconstruction from predicted pressures
# ---------------------------------------------------------------------------

def _headloss_n_from_inp(inp_path: str) -> float:
    """Return headloss exponent: 1.852 (Hazen-Williams) or 2.0 (Darcy-Weisbach)."""
    try:
        import ast
        from step1_io import load_inp_network
        network = load_inp_network(inp_path)
        raw = network.options.get("hydraulic", "")
        opts = ast.literal_eval(raw) if isinstance(raw, str) else raw
        model = str(opts.get("headloss", "")).upper()
        return 1.852 if "H" in model else 2.0
    except Exception:
        return 1.852


def _reconstruct_demands(
    preds_phys: np.ndarray,
    dataset: list,
    artifacts: dict,
    G: "nx.Graph",
    headloss_n: float,
    inp_path: str,
    stats: Optional[dict] = None,
) -> "tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[list]]":
    """Algebraic demand reconstruction via inverse headloss + nodal mass balance.

    Parameters
    ----------
    preds_phys : (N_samples, N_nodes) denormalized pressure array [m]
    dataset    : list of PyG Data objects (from test_dataset.pt)
    artifacts  : evaluation_artifacts.json dict for this model
    G          : graph_with_measurements NetworkX graph
    headloss_n : exponent n (1.852 for H-W, 2.0 for D-W)
    inp_path   : path to the .inp file
    stats      : dataset_stats.json dict; used to fix the reservoir head which
                 is a boundary condition and cannot be recovered from the
                 normalised y tensor (its true head lies outside [min_p, max_p]).

    Returns
    -------
    d_pred     : (N_samples, N_junctions)  reconstructed demands
    d_actual   : (N_samples, N_junctions)  ground-truth demands from .d field
                 PLUS base_demands (total = base + extra)
    jn_list    : ordered list of junction names
    All three are None when setup fails (missing inp, missing d field, …).
    """
    try:
        from step1_io import load_inp_network, compute_pipe_resistances_hw, compute_pipe_resistances
    except ImportError:
        return None, None, None

    try:
        network = load_inp_network(inp_path)
    except Exception:
        return None, None, None

    # Pipe resistances
    try:
        if abs(headloss_n - 1.852) < 1e-3:
            r_map = {pid: v["r_e"] for pid, v in compute_pipe_resistances_hw(network).items()}
        else:
            r_map = {pid: v["r_e"] for pid, v in compute_pipe_resistances(network).items()}
    except Exception:
        return None, None, None

    # Junction order from stored 'd' tensor
    sample0 = dataset[0]
    jn_stored = list(getattr(sample0, "junction_names", []) or [])
    if not jn_stored:
        return None, None, None

    node_mapping: dict = artifacts.get("node_mapping", {})
    node_list: list = artifacts.get("node_list", list(G.nodes()))
    if node_mapping:
        idx_to_node = {int(k): str(v) for k, v in node_mapping.items()}
    else:
        idx_to_node = {i: str(n) for i, n in enumerate(node_list)}

    # Elevation per node (for pressure → head conversion)
    elev: dict = {str(nid): node.elevation_m for nid, node in network.nodes.items()}

    # Reservoir head: fixed boundary condition, not recoverable from normalised y.
    # Use the value stored in dataset_stats.json.
    reservoir_heads: dict = {}
    if stats:
        res_node = str(stats.get("reservoir_node", ""))
        res_head = stats.get("reservoir_head")
        if res_node and res_head is not None:
            reservoir_heads[res_node] = float(res_head)

    def _clean_node_name(node_name: str) -> str:
        s = str(node_name)
        return s[5:] if s.startswith("meas_") else s

    # Measurement nodes are treated as known-pressure boundaries for reconstruction.
    measurement_nodes: set[str] = set()
    for n in artifacts.get("measurement_nodes", []) or []:
        measurement_nodes.add(_clean_node_name(n))
    if stats:
        for n in stats.get("measurement_nodes", []) or []:
            measurement_nodes.add(_clean_node_name(n))

    measurement_indices: list[int] = []
    for idx, node_name in idx_to_node.items():
        if _clean_node_name(node_name) in measurement_nodes:
            measurement_indices.append(int(idx))

    p_min = p_max = None
    if stats:
        p_min = stats.get("min_p")
        p_max = stats.get("max_p")
    denorm_available = p_min is not None and p_max is not None

    # Base demands per junction from network (for total demand = base + extra)
    base_demands: dict = {}
    try:
        import wntr
        wn = wntr.network.WaterNetworkModel(inp_path)
        for jn_name in jn_stored:
            node = wn.get_node(str(jn_name))
            if node is not None:
                base_demands[str(jn_name)] = float(node.base_demand)
    except Exception:
        # If unable to load base demands, treat as zero
        pass

    jn_set = set(jn_stored)
    jn_idx = {j: i for i, j in enumerate(jn_stored)}
    N_junc = len(jn_stored)
    N_samp = preds_phys.shape[0]

    d_pred_all = np.zeros((N_samp, N_junc), dtype=float)
    d_actual_all = np.zeros((N_samp, N_junc), dtype=float)

    for s_idx, (pressure_row, data) in enumerate(zip(preds_phys, dataset)):
        pressure_eff = np.array(pressure_row, dtype=float, copy=True)

        # Override measurement-node pressures with sample ground truth.
        if measurement_indices and denorm_available:
            y_gt = getattr(data, "y", None)
            if y_gt is not None:
                y_gt_phys = y_gt.detach().cpu().numpy().reshape(-1)
                y_gt_phys = y_gt_phys * (float(p_max) - float(p_min)) + float(p_min)
                for m_idx in measurement_indices:
                    if m_idx < len(pressure_eff) and m_idx < len(y_gt_phys):
                        pressure_eff[m_idx] = float(y_gt_phys[m_idx])

        # pressure → head  (junction nodes only; reservoir gets fixed head below)
        heads: dict = {}
        for idx, node_name in idx_to_node.items():
            if idx < len(pressure_eff):
                p = float(pressure_eff[idx])
                e = elev.get(str(node_name), 0.0)
                heads[str(node_name)] = p + e
        # Override reservoir(s) with fixed boundary head
        heads.update(reservoir_heads)

        # inverse headloss → pipe flows, then nodal mass balance
        d_vec = np.zeros(N_junc, dtype=float)
        for pipe_id, pipe in network.pipes.items():
            r = r_map.get(pipe_id)
            if r is None or r <= 0:
                continue
            hu = heads.get(str(pipe.start_node), 0.0)
            hv = heads.get(str(pipe.end_node), 0.0)
            dh = hu - hv
            q = math.copysign(abs(dh / r) ** (1.0 / headloss_n), dh)
            # d = B q  →  outgoing flow is demand (positive), incoming is supply
            if str(pipe.start_node) in jn_set:
                d_vec[jn_idx[str(pipe.start_node)]] -= q  # flow leaving node
            if str(pipe.end_node) in jn_set:
                d_vec[jn_idx[str(pipe.end_node)]] += q    # flow entering node
        d_pred_all[s_idx] = d_vec

        # Ground-truth demand: stored d (extra) + base_demand = total
        d_gt = getattr(data, "d", None)
        if d_gt is not None:
            d_extra = d_gt.cpu().numpy()
            d_total = np.array([d_extra[i] + base_demands.get(str(jn_stored[i]), 0.0)
                               for i in range(len(jn_stored))])
            d_actual_all[s_idx] = d_total

    return d_pred_all, d_actual_all, jn_stored


def _demand_global_metrics(d_pred: np.ndarray, d_actual: np.ndarray) -> dict:
    """Return MAE, RMSE, MAPE and R² over all samples and junctions."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    p = d_pred.flatten()
    a = d_actual.flatten()
    if len(p) == 0:
        return {}
    mse  = float(mean_squared_error(a, p))
    mae  = float(mean_absolute_error(a, p))
    r2   = float(r2_score(a, p))
    nz   = np.abs(a) > 1e-9
    mape = float(np.mean(np.abs((a[nz] - p[nz]) / a[nz])) * 100) if np.any(nz) else 0.0
    return {"mae": mae, "rmse": float(np.sqrt(mse)), "r2": r2, "mape": mape}


def _create_overlay_scatter(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    actuals_a: np.ndarray,
    actuals_b: np.ndarray,
    node_mask_a,
    node_mask_b,
    exclude_indices_a: list,
    exclude_indices_b: list,
    label_a: str,
    label_b: str,
    output_path: str,
) -> Optional[str]:
    """Side-by-side R² scatter — each model evaluated on its own test data."""
    from sklearn.metrics import r2_score

    def _filter(preds, actuals, node_mask, exclude_indices):
        masks_array = node_mask.numpy() if hasattr(node_mask, "numpy") else np.array(node_mask)
        exclude_set = set(exclude_indices)
        valid_mask = ~masks_array
        for idx in exclude_set:
            if idx < len(valid_mask):
                valid_mask[idx] = False
        return preds[:, valid_mask].flatten(), actuals[:, valid_mask].flatten()

    pred_a_flat, actual_a_flat = _filter(preds_a, actuals_a, node_mask_a, exclude_indices_a)
    pred_b_flat, actual_b_flat = _filter(preds_b, actuals_b, node_mask_b, exclude_indices_b)

    if len(actual_a_flat) == 0 or len(actual_b_flat) == 0:
        return None

    r2_a = r2_score(actual_a_flat, pred_a_flat)
    r2_b = r2_score(actual_b_flat, pred_b_flat)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, pred_flat, actual_flat, label, r2 in [
        (axes[0], pred_a_flat, actual_a_flat, label_a, r2_a),
        (axes[1], pred_b_flat, actual_b_flat, label_b, r2_b),
    ]:
        ax.scatter(actual_flat, pred_flat, alpha=0.4, s=8)
        lo = min(float(actual_flat.min()), float(pred_flat.min()))
        hi = max(float(actual_flat.max()), float(pred_flat.max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=2, label="Perfect")
        ax.set_xlabel("Actual Pressure")
        ax.set_ylabel("Predicted Pressure")
        ax.set_title(f"{label}\nR² = {r2:.4f}\n(own test set)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Symmetric comparison — each model evaluated on its own test data", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compare_pressure(
    wdn_name: str,
    dataset_dir_a: str,
    model_dir_a: str,
    dataset_dir_b: str,
    model_dir_b: str,
    output_dir: str,
    label_a: str = "Model A",
    label_b: str = "Model B",
    ignore_measurements: bool = True,
    algorithms_root: Optional[str] = None,
    model_a_hash: Optional[str] = None,
    model_b_hash: Optional[str] = None,
    log_fn=None,
) -> dict:
    """Compare two GNN models, each evaluated on their own test set.

    Each model is evaluated on the test data from its own dataset (which uses
    that model's measurement placement). Metrics are computed on physical
    (non-reservoir, non-measurement) nodes in denormalized pressure space.
    This allows a fair comparison of different measurement placements on the
    same water distribution network.

    Parameters
    ----------
    wdn_name : str
        Water distribution network name (used to look up scale params).
    dataset_dir_a : str
        Dataset artifact directory for model A. Must contain a ``data_generator/``
        sub-directory with ``test_dataset.pt``, ``graph_with_measurements.pickle``,
        ``dataset_stats.json``, and ``evaluation_artifacts.json``.
    model_dir_a / model_dir_b : str
        Model artifact directories. ``best_model.pt`` is searched first directly
        inside this dir, then inside ``<dataset_dir>/gnn_model/`` as a fallback.
    dataset_dir_b : str
        Dataset artifact directory for model B. Model B is evaluated on its own
        test set from this directory.
    output_dir : str
        Where to write PNG plots and ``results.json``.
    label_a / label_b : str
        Human-readable names shown on plots and in the results dict.
    ignore_measurements : bool
        Exclude measurement nodes from the metric computation (default True).
    algorithms_root : str, optional
        Explicit path to the ``algorithms/`` directory. Auto-detected if omitted.
    log_fn : callable, optional
        Called with log messages. Falls back to ``print``.

    Returns
    -------
    dict
        ``{"label_a", "label_b", "num_test_samples_a", "num_test_samples_b",
        "metrics_a", "metrics_b", "plots"}``
    """

    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)
        else:
            print(msg)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Algorithms root ────────────────────────────────────────────────────
    alg_root = Path(algorithms_root) if algorithms_root else _find_algorithms_root()
    if alg_root is None:
        raise RuntimeError(
            "Cannot locate algorithms/ directory. "
            "Expected at project root or ../SMARTWINE/algorithms/."
        )
    _log(f"Algorithms root: {alg_root}")

    # ── Load metrics / viz ─────────────────────────────────────────────────
    calculate_node_metrics, calculate_global_metrics, create_scatter_plot = (
        _load_metrics_module(alg_root)
    )
    try:
        create_advanced_error_distribution_viz = _load_viz_module(alg_root)
        has_viz = True
    except Exception as exc:
        _log(f"AED viz unavailable ({exc}); skipping.")
        has_viz = False

    # ── Load dataset artifacts (both A and B independently) ───────────────
    def _load_dataset_artifacts(ds_dir: str, label: str):
        data_dir = Path(ds_dir) / "data_generator"
        _log(f"Loading graph / artifacts for {label} from {data_dir}")

        with (data_dir / "graph_with_measurements.pickle").open("rb") as fh:
            G = pickle.load(fh)

        with (data_dir / "evaluation_artifacts.json").open("r", encoding="utf-8") as fh:
            artifacts = json.load(fh)

        dataset_stats: dict = {}
        stats_path = data_dir / "dataset_stats.json"
        if stats_path.exists():
            with stats_path.open("r", encoding="utf-8") as fh:
                dataset_stats = json.load(fh)

        test_data = torch.load(data_dir / "test_dataset.pt", weights_only=False)
        input_dim: int = test_data[0].x.shape[1]
        _log(f"  {label}: {len(test_data)} test samples, input_dim={input_dim}")

        return G, artifacts, dataset_stats, test_data, input_dim

    G_a, artifacts_a, stats_a, test_data_a, input_dim_a = _load_dataset_artifacts(
        dataset_dir_a, label_a
    )
    G_b, artifacts_b, stats_b, test_data_b, input_dim_b = _load_dataset_artifacts(
        dataset_dir_b, label_b
    )

    # ── Load models (each with its own input dimension) ────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(model_dir: str, ds_dir: str, input_dim: int, name: str) -> GCN:
        candidates = [
            Path(model_dir) / "best_model.pt",
            Path(ds_dir) / "gnn_model" / "best_model.pt",
        ]
        for path in candidates:
            if path.exists():
                _log(f"Loading {name} from {path}")
                m = GCN(dim_in=input_dim, dim_h=256, dim_out=1).to(device)
                m.load_state_dict(torch.load(str(path), map_location=device))
                m.eval()
                return m
        raise FileNotFoundError(
            f"Cannot find best_model.pt for {name}. "
            f"Searched: {[str(c) for c in candidates]}"
        )

    model_a = _load_model(model_dir_a, dataset_dir_a, input_dim_a, label_a)
    model_b = _load_model(model_dir_b, dataset_dir_b, input_dim_b, label_b)

    # ── Inference — each model on its own test set ─────────────────────────
    _log(f"Running inference for {label_a} on its own test set …")
    preds_a, actuals_a, masks_a = _run_inference(model_a, test_data_a, device)
    _log(f"Running inference for {label_b} on its own test set …")
    preds_b, actuals_b, masks_b = _run_inference(model_b, test_data_b, device)

    node_mask_a = masks_a[0]
    node_mask_b = masks_b[0]

    # ── Exclude indices — computed independently for each model ────────────
    def _get_exclude_indices(artifacts, G):
        measurement_nodes = artifacts.get("measurement_nodes", [])
        reservoir_nodes = artifacts.get("reservoir_nodes", [])
        node_mapping = artifacts.get("node_mapping", {})
        node_list = artifacts.get("node_list", list(G.nodes()))

        meas_clean = [n for n in measurement_nodes if not str(n).startswith("meas_")]
        reservoir_indices = _indices_from_mapping(node_mapping, reservoir_nodes)
        measurement_indices = _indices_from_mapping(node_mapping, meas_clean)
        if not reservoir_indices:
            reservoir_indices = _indices_from_list(node_list, reservoir_nodes)
        if not measurement_indices:
            measurement_indices = _indices_from_list(node_list, meas_clean)

        exclude_r2 = list(dict.fromkeys(
            reservoir_indices + (measurement_indices if ignore_measurements else [])
        ))
        exclude_aed = list(dict.fromkeys(reservoir_indices))
        return exclude_r2, exclude_aed, meas_clean

    exclude_r2_a, exclude_aed_a, meas_clean_a = _get_exclude_indices(artifacts_a, G_a)
    exclude_r2_b, exclude_aed_b, meas_clean_b = _get_exclude_indices(artifacts_b, G_b)

    # ── De-normalise — each model with its own pressure range ─────────────
    preds_a_sc = _denorm(preds_a, stats_a)
    actuals_a_sc = _denorm(actuals_a, stats_a)
    preds_b_sc = _denorm(preds_b, stats_b)
    actuals_b_sc = _denorm(actuals_b, stats_b)

    # ── Shared evaluation set (strict matching first, index fallback) ─────
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
            if str(node_name).startswith("meas_"):
                continue
            clean = _clean_node_name(node_name)
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

    nodes_a = _physical_nodes_from_mapping(artifacts_a)
    nodes_b = _physical_nodes_from_mapping(artifacts_b)
    common_nodes = sorted(list(set(nodes_a) & set(nodes_b)), key=_node_sort_key)

    strict_pairs = []
    if common_nodes:
        from collections import defaultdict, deque

        ids_a = _scenario_ids(actuals_a, stats_a, artifacts_a, test_data_a, common_nodes)
        ids_b = _scenario_ids(actuals_b, stats_b, artifacts_b, test_data_b, common_nodes)
        by_id_b = defaultdict(deque)
        for j, sid in enumerate(ids_b):
            by_id_b[sid].append(j)
        for i, sid in enumerate(ids_a):
            if by_id_b[sid]:
                strict_pairs.append((i, by_id_b[sid].popleft()))

    if strict_pairs:
        eval_mode = "strict_shared"
        eval_idx_a = np.array([p[0] for p in strict_pairs], dtype=int)
        eval_idx_b = np.array([p[1] for p in strict_pairs], dtype=int)
        _log(f"Shared evaluation mode: strict ({len(strict_pairs)} matched scenarios)")
    else:
        eval_mode = "shared_index"
        n_shared = min(len(preds_a), len(preds_b))
        eval_idx_a = np.arange(n_shared, dtype=int)
        eval_idx_b = np.arange(n_shared, dtype=int)
        _log(f"Shared evaluation mode: index fallback ({n_shared} paired samples)")

    preds_a = preds_a[eval_idx_a]
    actuals_a = actuals_a[eval_idx_a]
    preds_b = preds_b[eval_idx_b]
    actuals_b = actuals_b[eval_idx_b]

    preds_a_sc = preds_a_sc[eval_idx_a]
    actuals_a_sc = actuals_a_sc[eval_idx_a]
    preds_b_sc = preds_b_sc[eval_idx_b]
    actuals_b_sc = actuals_b_sc[eval_idx_b]

    test_data_a_eval = [test_data_a[int(i)] for i in eval_idx_a]
    test_data_b_eval = [test_data_b[int(i)] for i in eval_idx_b]

    # ── Global metrics ─────────────────────────────────────────────────────
    _log("Computing global metrics …")
    metrics_a = calculate_global_metrics(preds_a, actuals_a, node_mask_a, exclude_r2_a)
    metrics_b = calculate_global_metrics(preds_b, actuals_b, node_mask_b, exclude_r2_b)

    # ── Node metrics (for AED) ─────────────────────────────────────────────
    node_metrics_a = calculate_node_metrics(preds_a, actuals_a, node_mask_a, exclude_aed_a)
    node_metrics_b = calculate_node_metrics(preds_b, actuals_b, node_mask_b, exclude_aed_b)

    # ── R² scatter plots ───────────────────────────────────────────────────
    _log("Generating R² scatter plots …")
    scatter_a_path = str(output_path / "r2_scatter_a.png")
    scatter_b_path = str(output_path / "r2_scatter_b.png")
    create_scatter_plot(preds_a_sc, actuals_a_sc, node_mask_a, exclude_r2_a, scatter_a_path)
    create_scatter_plot(preds_b_sc, actuals_b_sc, node_mask_b, exclude_r2_b, scatter_b_path)

    overlay_path = _create_overlay_scatter(
        preds_a_sc, preds_b_sc,
        actuals_a_sc, actuals_b_sc,
        node_mask_a, node_mask_b,
        exclude_r2_a, exclude_r2_b,
        label_a, label_b,
        str(output_path / "r2_scatter_overlay.png"),
    )
    _log(f"Scatter plots saved to {output_path}")

    # ── AED visualisations — each model with its own graph ─────────────────
    aed_a_path: Optional[str] = None
    aed_b_path: Optional[str] = None

    if has_viz:
        _log("Generating AED visualisations …")
        wdn_config = _load_wdn_config(wdn_name)
        aed_kwargs_base = dict(
            scale=wdn_config.get("scale", 1.0),
            node_scale=wdn_config.get("node_scale", 1.0),
            font_scale=wdn_config.get("font_scale", 1.0),
            node_label_threshold=wdn_config.get("node_label_threshold", 0.01),
            non_special_node_scale=wdn_config.get("non_special_node_scale", 1.0),
        )

        for G, artifacts, node_metrics, meas_clean, exclude_aed, path_key, label in [
            (G_a, artifacts_a, node_metrics_a, meas_clean_a, exclude_aed_a, "aed_a.png", label_a),
            (G_b, artifacts_b, node_metrics_b, meas_clean_b, exclude_aed_b, "aed_b.png", label_b),
        ]:
            node_mapping = artifacts.get("node_mapping", {})
            secondary_pipes = artifacts.get("secondary_pipes", [])
            aed_kwargs = {**aed_kwargs_base, "node_mapping": node_mapping}
            virtual_nodes = [n for n in G.nodes() if str(n).startswith("meas_")]
            G_aed = _prep_aed_graph(G, virtual_nodes)
            try:
                img = create_advanced_error_distribution_viz(
                    G_aed, node_metrics,
                    secondary_pipes, meas_clean,
                    **aed_kwargs,
                )
                if img is not None:
                    save_path = str(output_path / path_key)
                    img.save(save_path)
                    _log(f"Saved AED ({label}): {save_path}")
                    if path_key == "aed_a.png":
                        aed_a_path = save_path
                    else:
                        aed_b_path = save_path
            except Exception as exc:
                _log(f"Warning: AED ({label}) failed: {exc}")

    # ── Demand reconstruction (algebraic, inverse headloss) ───────────────
    _log("Running algebraic demand reconstruction …")
    inp_path = str(Path(alg_root).parent / "wdn" / f"{wdn_name}.inp")
    headloss_n = _headloss_n_from_inp(inp_path)
    _log(f"Headloss exponent n={headloss_n:.4f} (from {inp_path})")

    d_pred_a, d_actual_a, _jn_a = _reconstruct_demands(
        preds_a_sc, test_data_a_eval, artifacts_a, G_a, headloss_n, inp_path, stats=stats_a
    )
    d_pred_b, d_actual_b, _jn_b = _reconstruct_demands(
        preds_b_sc, test_data_b_eval, artifacts_b, G_b, headloss_n, inp_path, stats=stats_b
    )

    demand_metrics_a = _demand_global_metrics(d_pred_a, d_actual_a) if d_pred_a is not None else {}
    demand_metrics_b = _demand_global_metrics(d_pred_b, d_actual_b) if d_pred_b is not None else {}
    if demand_metrics_a:
        _log(f"Demand metrics A: {demand_metrics_a}")
    if demand_metrics_b:
        _log(f"Demand metrics B: {demand_metrics_b}")

    # ── Save results.json ──────────────────────────────────────────────────
    def _portable_path(path_str: str) -> str:
        """Prefer project-relative paths so artifacts can be moved across devices."""
        try:
            p = Path(path_str)
            if not p.is_absolute():
                return str(p)
            return str(p.relative_to(ROOT_DIR))
        except Exception:
            return path_str

    results = {
        "wdn_name": wdn_name,
        "label_a": label_a,
        "label_b": label_b,
        "model_a_hash": model_a_hash,
        "model_b_hash": model_b_hash,
        "model_a_dir": _portable_path(model_dir_a),
        "model_b_dir": _portable_path(model_dir_b),
        "dataset_a_dir": _portable_path(dataset_dir_a),
        "dataset_b_dir": _portable_path(dataset_dir_b),
        "num_test_samples_a": len(test_data_a),
        "num_test_samples_b": len(test_data_b),
        "evaluation_mode": eval_mode,
        "num_eval_samples": int(len(eval_idx_a)),
        "num_strict_pairs": int(len(strict_pairs)),
        "metrics_a": metrics_a,
        "metrics_b": metrics_b,
        "demand_metrics_a": demand_metrics_a,
        "demand_metrics_b": demand_metrics_b,
        "plots": {
            "scatter_a": _portable_path(scatter_a_path),
            "scatter_b": _portable_path(scatter_b_path),
            "scatter_overlay": _portable_path(overlay_path) if overlay_path else None,
            "aed_a": _portable_path(aed_a_path) if aed_a_path else None,
            "aed_b": _portable_path(aed_b_path) if aed_b_path else None,
        },
    }
    results_path = output_path / "results.json"
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    _log(f"Results saved to {results_path}")

    return results
