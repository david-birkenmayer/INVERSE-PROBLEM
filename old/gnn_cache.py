"""
Artifact cache for the GNN pipeline.

Directory layout (all paths relative to project root):
  old/data/<WDN>/datasets/<dataset_hash>/   — generated datasets
  old/data/<WDN>/models/<model_hash>/       — trained models
  old/data/<WDN>/shared_test_sets/<hash>/   — shared evaluation test sets
  old/data/<WDN>/comparisons/<hash>/        — comparison results

Each artifact dir contains:
  manifest.json    — inputs used to create the artifact + creation timestamp
  _gnn_hash.txt    — the artifact hash (mirrors _gui_hash.txt in solver cache)
  <artifact files>

Index files (one per WDN per type):
  old/data/<WDN>/datasets/index.json
  old/data/<WDN>/models/index.json
  old/data/<WDN>/shared_test_sets/index.json
  old/data/<WDN>/comparisons/index.json
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Locate project root and import shared cache utilities
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent          # old/
ROOT_DIR = str(_HERE.parent)                     # project root
import sys
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from gui.cache import compute_hash, load_index, save_index


# ---------------------------------------------------------------------------
# Artifact type names
# ---------------------------------------------------------------------------
_DATASETS = "datasets"
_MODELS = "models"
_TEST_SETS = "shared_test_sets"
_COMPARISONS = "comparisons"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _artifact_base(wdn: str, artifact_type: str) -> str:
    return os.path.join("old", "data", wdn, artifact_type)


def _index_path(wdn: str, artifact_type: str) -> str:
    return os.path.join(_artifact_base(wdn, artifact_type), "index.json")


def _artifact_dir(wdn: str, artifact_type: str, h: str) -> str:
    return os.path.join(_artifact_base(wdn, artifact_type), h)


def _write_manifest(artifact_dir: str, artifact_type: str, h: str, inputs: dict) -> None:
    os.makedirs(artifact_dir, exist_ok=True)
    manifest = {
        "artifact_type": artifact_type,
        "hash": h,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
    }
    manifest_path = os.path.join(artifact_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    hash_file = os.path.join(artifact_dir, "_gnn_hash.txt")
    if not os.path.isfile(hash_file):
        with open(hash_file, "w", encoding="utf-8") as f:
            f.write(h)


def _find(wdn: str, artifact_type: str, h: str) -> Optional[str]:
    """Return the artifact directory if it exists and is registered, else None."""
    idx_path = _index_path(wdn, artifact_type)
    index = load_index(idx_path, ROOT_DIR)
    rel = index.get(h)
    if rel is None:
        return None
    resolved = rel if os.path.isabs(rel) else os.path.join(ROOT_DIR, rel)
    return resolved if os.path.isdir(resolved) else None


def _register(wdn: str, artifact_type: str, h: str, artifact_dir: str) -> None:
    idx_path = _index_path(wdn, artifact_type)
    index = load_index(idx_path, ROOT_DIR)
    rel = os.path.relpath(artifact_dir if os.path.isabs(artifact_dir) else
                          os.path.join(ROOT_DIR, artifact_dir), ROOT_DIR)
    index[h] = rel
    save_index(idx_path, index, ROOT_DIR)
    # Also scan for orphaned dirs (mirrors _sync_wdn_index pattern)
    _sync_type_index(wdn, artifact_type)


def _sync_type_index(wdn: str, artifact_type: str) -> Dict[str, str]:
    """Scan for _gnn_hash.txt files and backfill the index."""
    idx_path = _index_path(wdn, artifact_type)
    index = load_index(idx_path, ROOT_DIR)
    base = os.path.join(ROOT_DIR, _artifact_base(wdn, artifact_type))
    if not os.path.isdir(base):
        return index
    changed = False
    try:
        entries = list(os.scandir(base))
    except OSError:
        return index
    for entry in entries:
        if not entry.is_dir():
            continue
        hash_file = os.path.join(entry.path, "_gnn_hash.txt")
        if not os.path.isfile(hash_file):
            continue
        try:
            h = open(hash_file, encoding="utf-8").read().strip()
        except OSError:
            continue
        if not h:
            continue
        expected_rel = os.path.join("old", "data", wdn, artifact_type, entry.name)
        if index.get(h) != expected_rel:
            index[h] = expected_rel
            changed = True
    if changed:
        save_index(idx_path, index, ROOT_DIR)
    return index


# ---------------------------------------------------------------------------
# Hash-input builders
# ---------------------------------------------------------------------------

def dataset_inputs(
    wdn: str,
    measurement_nodes: list,
    extra_demand: float,
    num_simulations: int,
    demand_model: str = "uniform",
    node_label_threshold: float = 0.0,
    seed: int | None = None,
    code_version: str = "data_generator_v1",
) -> dict:
    inputs = {
        "wdn": wdn,
        "measurement_nodes": sorted(str(n) for n in measurement_nodes),
        "extra_demand": float(extra_demand),
        "num_simulations": int(num_simulations),
        "demand_model": demand_model,
        "node_label_threshold": float(node_label_threshold),
        "code_version": code_version,
    }
    # Optional explicit seed for deterministic shared-evaluation datasets.
    if seed is not None:
        inputs["seed"] = int(seed)
    return inputs


def model_inputs(
    dataset_hash: str,
    epochs: int = 200,
    lr: float = 0.001,
    batch_size: int = 32,
    hidden_dim: int = 64,
    num_layers: int = 3,
    seed: int = 42,
    code_version: str = "gnn_model_v1",
) -> dict:
    return {
        "dataset_hash": dataset_hash,
        "epochs": int(epochs),
        "lr": float(lr),
        "batch_size": int(batch_size),
        "hidden_dim": int(hidden_dim),
        "num_layers": int(num_layers),
        "seed": int(seed),
        "code_version": code_version,
    }


def test_set_inputs(
    wdn: str,
    extra_demand: float,
    num_simulations: int,
    demand_model: str = "uniform",
    seed: int = 9999,
    code_version: str = "data_generator_v1",
) -> dict:
    # NOTE: measurement_nodes intentionally excluded — the shared test set has
    # no measurement nodes injected; both models observe their own subset.
    return {
        "wdn": wdn,
        "extra_demand": float(extra_demand),
        "num_simulations": int(num_simulations),
        "demand_model": demand_model,
        "seed": int(seed),
        "code_version": code_version,
    }


def comparison_inputs(
    model_a_hash: str,
    model_b_hash: str,
    test_hash: str,
    demand_reconstruction: str = "algebraic",
    comparison_mode: str = "symmetric",
) -> dict:
    # Sort the model hashes so A/B order doesn't affect the comparison hash.
    a, b = sorted([model_a_hash, model_b_hash])
    return {
        "model_a_hash": a,
        "model_b_hash": b,
        "test_hash": test_hash,
        "demand_reconstruction": demand_reconstruction,
        "comparison_mode": comparison_mode,
    }


# ---------------------------------------------------------------------------
# Public API: hash computation
# ---------------------------------------------------------------------------

def dataset_hash(inputs: dict) -> str:
    return compute_hash(inputs)


def model_hash(inputs: dict) -> str:
    return compute_hash(inputs)


def test_set_hash(inputs: dict) -> str:
    return compute_hash(inputs)


def comparison_hash(inputs: dict) -> str:
    return compute_hash(inputs)


# ---------------------------------------------------------------------------
# Public API: find / register
# ---------------------------------------------------------------------------

def find_dataset(wdn: str, h: str) -> Optional[str]:
    return _find(wdn, _DATASETS, h)


def find_model(wdn: str, h: str) -> Optional[str]:
    return _find(wdn, _MODELS, h)


def find_test_set(wdn: str, h: str) -> Optional[str]:
    return _find(wdn, _TEST_SETS, h)


def find_comparison(wdn: str, h: str) -> Optional[str]:
    return _find(wdn, _COMPARISONS, h)


def register_dataset(wdn: str, h: str, inputs: dict, artifact_dir: Optional[str] = None) -> str:
    d = artifact_dir or os.path.join(ROOT_DIR, _artifact_dir(wdn, _DATASETS, h))
    _write_manifest(d, "dataset", h, inputs)
    _register(wdn, _DATASETS, h, d)
    return d


def register_model(wdn: str, h: str, inputs: dict, artifact_dir: Optional[str] = None) -> str:
    d = artifact_dir or os.path.join(ROOT_DIR, _artifact_dir(wdn, _MODELS, h))
    _write_manifest(d, "model", h, inputs)
    _register(wdn, _MODELS, h, d)
    return d


def register_test_set(wdn: str, h: str, inputs: dict, artifact_dir: Optional[str] = None) -> str:
    d = artifact_dir or os.path.join(ROOT_DIR, _artifact_dir(wdn, _TEST_SETS, h))
    _write_manifest(d, "test_set", h, inputs)
    _register(wdn, _TEST_SETS, h, d)
    return d


def register_comparison(wdn: str, h: str, inputs: dict, artifact_dir: Optional[str] = None) -> str:
    d = artifact_dir or os.path.join(ROOT_DIR, _artifact_dir(wdn, _COMPARISONS, h))
    _write_manifest(d, "comparison", h, inputs)
    _register(wdn, _COMPARISONS, h, d)
    return d


# ---------------------------------------------------------------------------
# Public API: list available artifacts for a WDN
# ---------------------------------------------------------------------------

def list_models(wdn: str) -> list[dict]:
    """Return a list of {hash, artifact_dir, inputs} dicts for all registered models."""
    index = _sync_type_index(wdn, _MODELS)
    result = []
    for h, rel in index.items():
        resolved = rel if os.path.isabs(rel) else os.path.join(ROOT_DIR, rel)
        manifest_path = os.path.join(resolved, "manifest.json")
        inputs = {}
        if os.path.isfile(manifest_path):
            try:
                manifest = json.loads(open(manifest_path, encoding="utf-8").read())
                inputs = manifest.get("inputs", {})
            except (OSError, json.JSONDecodeError):
                pass
        result.append({"hash": h, "artifact_dir": resolved, "inputs": inputs})
    return result


def list_test_sets(wdn: str) -> list[dict]:
    index = _sync_type_index(wdn, _TEST_SETS)
    result = []
    for h, rel in index.items():
        resolved = rel if os.path.isabs(rel) else os.path.join(ROOT_DIR, rel)
        manifest_path = os.path.join(resolved, "manifest.json")
        inputs = {}
        if os.path.isfile(manifest_path):
            try:
                manifest = json.loads(open(manifest_path, encoding="utf-8").read())
                inputs = manifest.get("inputs", {})
            except (OSError, json.JSONDecodeError):
                pass
        result.append({"hash": h, "artifact_dir": resolved, "inputs": inputs})
    return result
