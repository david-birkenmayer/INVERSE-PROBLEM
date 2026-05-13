"""
Run old notebooks with injected parameters.

Hash-based artifact layout (all relative to project root):
  old/data/<WDN>/datasets/<dataset_hash>/  — dataset artifacts
  old/data/<WDN>/models/<model_hash>/      — trained model artifacts

Each run is cached via gnn_cache.py; re-running with the same inputs is a no-op
unless force=True.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

CWD = Path(__file__).resolve().parents[1]
OLD_DIR = CWD / "old"
DATA_NOTEBOOK = OLD_DIR / "data_generator.ipynb"
GNN_NOTEBOOK = OLD_DIR / "gnn_model.ipynb"
EVAL_NOTEBOOK = OLD_DIR / "evaluation.ipynb"

# Ensure project root is importable
if str(CWD) not in sys.path:
    sys.path.insert(0, str(CWD))

from old.gnn_cache import (  # noqa: E402
    dataset_inputs,
    dataset_hash,
    find_dataset,
    register_dataset,
    model_inputs,
    model_hash,
    find_model,
    register_model,
    test_set_inputs,
    test_set_hash,
    find_test_set,
    register_test_set,
    comparison_inputs,
    comparison_hash,
    find_comparison,
    register_comparison,
)


def _replace_assignment(cell_source, var_name: str, value_expr: str) -> bool:
    updated = False
    for i, line in enumerate(cell_source):
        if line.startswith(f"{var_name} ="):
            cell_source[i] = f"{var_name} = {value_expr}\n"
            updated = True
    return updated


def _inject_or_replace(nb: Dict[str, Any], replacements: Dict[str, str]) -> Dict[str, Any]:
    found = {k: False for k in replacements}

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        if not isinstance(source, list):
            continue
        for var_name, value_expr in replacements.items():
            if _replace_assignment(source, var_name, value_expr):
                found[var_name] = True

    if not all(found.values()):
        # Prepend a new parameters cell
        params_lines = ["# Injected parameters\n"]
        for var_name, value_expr in replacements.items():
            params_lines.append(f"{var_name} = {value_expr}\n")
        nb.setdefault("cells", []).insert(
            0,
            {
                "cell_type": "code",
                "metadata": {"language": "python", "tags": ["parameters"]},
                "source": params_lines,
                "outputs": [],
                "execution_count": None,
            },
        )

    return nb


def _write_notebook(nb: Dict[str, Any], out_path: Path) -> None:
    out_path.write_text(json.dumps(nb, indent=2), encoding="utf-8")


# Global process tracking for cancellation support
_current_process: Optional[subprocess.Popen] = None
_process_lock = __import__("threading").Lock()


def _set_current_process(proc: Optional[subprocess.Popen]) -> None:
    global _current_process
    with _process_lock:
        _current_process = proc


def get_current_process() -> Optional[subprocess.Popen]:
    """Retrieve the currently running papermill process for external cancellation."""
    with _process_lock:
        return _current_process


def _execute_notebook(nb_path: Path, output_path: Path, timeout: int) -> subprocess.Popen:
    """Execute notebook and return the process object for potential termination."""
    cmd = [
        sys.executable,
        "-m",
        "papermill",
        str(nb_path),
        str(output_path),
        "--request-save-on-cell-execute",
        "--cwd",
        str(CWD),
        "-k",
        "python3",
        "--no-progress-bar",
        "--execution-timeout",
        str(timeout),
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _set_current_process(proc)
        _, stderr_bytes = proc.communicate(timeout=timeout)
        _set_current_process(None)
        if proc.returncode < 0:
            # Killed by signal (e.g. SIGTERM from cancel) — caller handles cancellation
            return proc
        if proc.returncode != 0:
            stderr_tail = (stderr_bytes or b"").decode("utf-8", errors="replace").strip()
            lines = stderr_tail.splitlines()
            relevant = "\n".join(lines[-20:]) if lines else "(no stderr output)"
            raise RuntimeError(f"Notebook execution failed (exit {proc.returncode}):\n{relevant}")
        return proc
    except subprocess.TimeoutExpired:
        _set_current_process(None)
        proc.kill()
        raise RuntimeError(f"Notebook execution exceeded {timeout}s timeout")


def _load_wdn_params(wdn_name: str) -> Dict[str, Any]:
    json_path = CWD / "wdn" / f"{wdn_name}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing config: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# New hash-based entry points (used by the GUI)
# ---------------------------------------------------------------------------

def run_dataset(
    wdn_name: str,
    measurement_nodes: List[str],
    extra_demand: float,
    num_simulations: int,
    demand_model: str = "dirichlet",
    node_label_threshold: float = 0.0,
    timeout: int = 36000,
    force: bool = False,
    log_fn=None,
) -> tuple[str, bool]:
    """Generate (or reuse) a dataset artifact. Returns (artifact_dir, was_cancelled).

    If an artifact with the same hash already exists and force=False, returns
    immediately without running the notebook.
    """
    inp = dataset_inputs(
        wdn=wdn_name,
        measurement_nodes=measurement_nodes,
        extra_demand=extra_demand,
        num_simulations=num_simulations,
        demand_model=demand_model,
        node_label_threshold=node_label_threshold,
    )
    h = dataset_hash(inp)

    if not force:
        existing = find_dataset(wdn_name, h)
        if existing:
            if log_fn:
                log_fn(f"Dataset cache hit: {h[:8]}  →  {existing}")
            return existing, False

    inp_path = str(CWD / "wdn" / f"{wdn_name}.inp")
    artifact_dir = str(CWD / "old" / "data" / wdn_name / "datasets" / h)
    Path(artifact_dir).mkdir(parents=True, exist_ok=True)

    data_nb = json.loads(DATA_NOTEBOOK.read_text(encoding="utf-8"))
    replacements = {
        "WDN_NAME": repr(wdn_name),
        "EXTRA_DEMAND": repr(float(extra_demand)),
        "MEASUREMENT_NODES": repr(measurement_nodes),
        "INP_DIR": repr(inp_path),
        "BASE_DIR": repr(artifact_dir),
        "NUM_SIMULATIONS": repr(int(num_simulations)),
        "SEED": repr(int(h[:8], 16)),
    }
    data_nb = _inject_or_replace(data_nb, replacements)

    nb_out = Path(artifact_dir) / f"data_generator_{wdn_name}.ran.ipynb"
    nb_tmp = Path(artifact_dir) / f"data_generator_{wdn_name}.tmp.ipynb"
    _write_notebook(data_nb, nb_tmp)

    if log_fn:
        log_fn(f"Running data_generator notebook → {artifact_dir}")
    
    try:
        proc = _execute_notebook(nb_tmp, nb_out, timeout)
        if proc.returncode < 0:
            if log_fn:
                log_fn("Dataset generation cancelled.")
            return artifact_dir, True
        register_dataset(wdn_name, h, inp, artifact_dir)
        return artifact_dir, False
    except Exception:
        import shutil
        shutil.rmtree(artifact_dir, ignore_errors=True)
        raise


def run_test_set(
    wdn_name: str,
    extra_demand: float,
    num_simulations: int,
    demand_model: str = "uniform",
    seed: int = 9999,
    timeout: int = 36000,
    force: bool = False,
    log_fn=None,
) -> tuple[str, bool]:
    """Generate (or reuse) a shared test-set artifact. Returns (artifact_dir, was_cancelled)."""
    inp = test_set_inputs(
        wdn=wdn_name,
        extra_demand=extra_demand,
        num_simulations=num_simulations,
        demand_model=demand_model,
        seed=seed,
    )
    h = test_set_hash(inp)

    if not force:
        existing = find_test_set(wdn_name, h)
        if existing:
            if log_fn:
                log_fn(f"Test set cache hit: {h[:8]}  →  {existing}")
            return existing, False

    inp_path = str(CWD / "wdn" / f"{wdn_name}.inp")
    artifact_dir = str(CWD / "old" / "data" / wdn_name / "shared_test_sets" / h)
    Path(artifact_dir).mkdir(parents=True, exist_ok=True)

    data_nb = json.loads(DATA_NOTEBOOK.read_text(encoding="utf-8"))
    replacements = {
        "WDN_NAME": repr(wdn_name),
        "EXTRA_DEMAND": repr(float(extra_demand)),
        "MEASUREMENT_NODES": repr([]),
        "INP_DIR": repr(inp_path),
        "BASE_DIR": repr(artifact_dir),
        "NUM_SIMULATIONS": repr(int(num_simulations)),
        "SEED": repr(int(seed)),
    }
    data_nb = _inject_or_replace(data_nb, replacements)

    nb_out = Path(artifact_dir) / f"data_generator_{wdn_name}.ran.ipynb"
    nb_tmp = Path(artifact_dir) / f"data_generator_{wdn_name}.tmp.ipynb"
    _write_notebook(data_nb, nb_tmp)

    if log_fn:
        log_fn(f"Running shared test-set notebook → {artifact_dir}")

    try:
        proc = _execute_notebook(nb_tmp, nb_out, timeout)
        if proc.returncode < 0:
            if log_fn:
                log_fn("Test-set generation cancelled.")
            return artifact_dir, True
        register_test_set(wdn_name, h, inp, artifact_dir)
        return artifact_dir, False
    except Exception:
        import shutil
        shutil.rmtree(artifact_dir, ignore_errors=True)
        raise


def run_shared_eval_dataset(
    wdn_name: str,
    measurement_nodes: List[str],
    extra_demand: float,
    num_simulations: int,
    demand_model: str = "uniform",
    seed: int = 9999,
    node_label_threshold: float = 0.0,
    timeout: int = 36000,
    force: bool = False,
    log_fn=None,
) -> tuple[str, bool]:
    """Generate (or reuse) a dataset with model-specific measurements and shared scenarios.

    This is used for fair model comparison: both models see the same scenario bank
    (controlled by ``seed``), but each keeps its own measurement placement.
    """
    inp = dataset_inputs(
        wdn=wdn_name,
        measurement_nodes=measurement_nodes,
        extra_demand=extra_demand,
        num_simulations=num_simulations,
        demand_model=demand_model,
        node_label_threshold=node_label_threshold,
        seed=seed,
        code_version="data_generator_shared_eval_v1",
    )
    h = dataset_hash(inp)

    if not force:
        existing = find_dataset(wdn_name, h)
        if existing:
            if log_fn:
                log_fn(f"Shared-eval dataset cache hit: {h[:8]}  →  {existing}")
            return existing, False

    inp_path = str(CWD / "wdn" / f"{wdn_name}.inp")
    artifact_dir = str(CWD / "old" / "data" / wdn_name / "datasets" / h)
    Path(artifact_dir).mkdir(parents=True, exist_ok=True)

    data_nb = json.loads(DATA_NOTEBOOK.read_text(encoding="utf-8"))
    replacements = {
        "WDN_NAME": repr(wdn_name),
        "EXTRA_DEMAND": repr(float(extra_demand)),
        "MEASUREMENT_NODES": repr(measurement_nodes),
        "INP_DIR": repr(inp_path),
        "BASE_DIR": repr(artifact_dir),
        "NUM_SIMULATIONS": repr(int(num_simulations)),
        "SEED": repr(int(seed)),
    }
    data_nb = _inject_or_replace(data_nb, replacements)

    nb_out = Path(artifact_dir) / f"data_generator_{wdn_name}.ran.ipynb"
    nb_tmp = Path(artifact_dir) / f"data_generator_{wdn_name}.tmp.ipynb"
    _write_notebook(data_nb, nb_tmp)

    if log_fn:
        log_fn(f"Running shared-eval data_generator notebook → {artifact_dir}")

    try:
        proc = _execute_notebook(nb_tmp, nb_out, timeout)
        if proc.returncode < 0:
            if log_fn:
                log_fn("Shared-eval dataset generation cancelled.")
            return artifact_dir, True
        register_dataset(wdn_name, h, inp, artifact_dir)
        return artifact_dir, False
    except Exception:
        import shutil
        shutil.rmtree(artifact_dir, ignore_errors=True)
        raise


def run_model(
    wdn_name: str,
    d_hash: str,
    dataset_dir: str,
    epochs: int = 50,
    lr: float = 0.001,
    batch_size: int = 32,
    hidden_dim: int = 64,
    num_layers: int = 3,
    seed: int = 42,
    timeout: int = 36000,
    force: bool = False,
    log_fn=None,
) -> tuple[str, bool]:
    """Train (or reuse) a model artifact. Returns (artifact_dir, was_cancelled)."""
    inp = model_inputs(
        dataset_hash=d_hash,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        seed=seed,
    )
    h = model_hash(inp)

    if not force:
        existing = find_model(wdn_name, h)
        if existing:
            if log_fn:
                log_fn(f"Model cache hit: {h[:8]}  →  {existing}")
            return existing, False

    artifact_dir = str(CWD / "old" / "data" / wdn_name / "models" / h)
    Path(artifact_dir).mkdir(parents=True, exist_ok=True)

    gnn_nb = json.loads(GNN_NOTEBOOK.read_text(encoding="utf-8"))
    replacements = {
        "WDN_NAME": repr(wdn_name),
        "BASE_DIR": repr(dataset_dir),
        "MODEL_DIR": repr(artifact_dir),
        "EPOCHS": repr(int(epochs)),
        "LR": repr(float(lr)),
        "BATCH_SIZE": repr(int(batch_size)),
        "HIDDEN_DIM": repr(int(hidden_dim)),
        "NUM_LAYERS": repr(int(num_layers)),
        "SEED": repr(int(seed)),
    }
    gnn_nb = _inject_or_replace(gnn_nb, replacements)

    nb_out = Path(artifact_dir) / f"gnn_model_{wdn_name}.ran.ipynb"
    nb_tmp = Path(artifact_dir) / f"gnn_model_{wdn_name}.tmp.ipynb"
    _write_notebook(gnn_nb, nb_tmp)

    if log_fn:
        log_fn(f"Running gnn_model notebook → {artifact_dir}")
    
    try:
        proc = _execute_notebook(nb_tmp, nb_out, timeout)
        if proc.returncode < 0:
            if log_fn:
                log_fn("Model training cancelled.")
            return artifact_dir, True
        register_model(wdn_name, h, inp, artifact_dir)
        return artifact_dir, False
    except Exception:
        import shutil
        shutil.rmtree(artifact_dir, ignore_errors=True)
        raise


def run_comparison(
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
    force: bool = False,
    log_fn=None,
) -> tuple[str, bool]:
    """Compare two models, each evaluated on its own test set (symmetric comparison).

    Returns (comparison_artifact_dir, was_cancelled).
    """
    comparison_mode = "symmetric"
    if test_set_hash:
        # Different comparison mode key to avoid cache hits from older logic
        # where both models were evaluated on a dataset with MEASUREMENT_NODES=[].
        comparison_mode = "shared_testset_per_model_measurements_v1"

    inp = comparison_inputs(
        model_a_hash=model_a_hash,
        model_b_hash=model_b_hash,
        test_hash=test_set_hash,
        demand_reconstruction=demand_reconstruction,
        comparison_mode=comparison_mode,
    )
    h = comparison_hash(inp)

    if not force:
        existing = find_comparison(wdn_name, h)
        if existing:
            if log_fn:
                log_fn(f"Comparison cache hit: {h[:8]}  →  {existing}")
            return existing, False

    artifact_dir = str(CWD / "old" / "data" / wdn_name / "comparisons" / h)
    Path(artifact_dir).mkdir(parents=True, exist_ok=True)

    # Import compare_pressure module and run
    from old.compare_pressure import compare_pressure

    if test_set_hash:
        shared_test_dir = find_test_set(wdn_name, test_set_hash)
        if not shared_test_dir:
            raise FileNotFoundError(f"Shared test set not found: {test_set_hash}")

        manifest_path = Path(shared_test_dir) / "manifest.json"
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        shared_inputs = manifest.get("inputs", {})

        def _read_measurement_nodes(ds_dir: str) -> List[str]:
            art_path = Path(ds_dir) / "data_generator" / "evaluation_artifacts.json"
            with art_path.open("r", encoding="utf-8") as fh:
                art = json.load(fh)
            return [str(n) for n in (art.get("measurement_nodes") or [])]

        meas_a = _read_measurement_nodes(dataset_a_dir)
        meas_b = _read_measurement_nodes(dataset_b_dir)

        if log_fn:
            log_fn(
                "Shared test set selected: generating model-specific evaluation datasets "
                "with shared scenario seed."
            )

        dataset_a_dir, cancelled_a = run_shared_eval_dataset(
            wdn_name=wdn_name,
            measurement_nodes=meas_a,
            extra_demand=float(shared_inputs.get("extra_demand", 0.0)),
            num_simulations=int(shared_inputs.get("num_simulations", 1000)),
            demand_model=str(shared_inputs.get("demand_model", "uniform")),
            seed=int(shared_inputs.get("seed", 9999)),
            timeout=timeout,
            force=force,
            log_fn=log_fn,
        )
        if cancelled_a:
            return artifact_dir, True

        dataset_b_dir, cancelled_b = run_shared_eval_dataset(
            wdn_name=wdn_name,
            measurement_nodes=meas_b,
            extra_demand=float(shared_inputs.get("extra_demand", 0.0)),
            num_simulations=int(shared_inputs.get("num_simulations", 1000)),
            demand_model=str(shared_inputs.get("demand_model", "uniform")),
            seed=int(shared_inputs.get("seed", 9999)),
            timeout=timeout,
            force=force,
            log_fn=log_fn,
        )
        if cancelled_b:
            return artifact_dir, True

    if log_fn:
        log_fn(f"Running pressure comparison → {artifact_dir}")

    try:
        results = compare_pressure(
            wdn_name=wdn_name,
            dataset_dir_a=dataset_a_dir,
            model_dir_a=model_a_dir,
            dataset_dir_b=dataset_b_dir,
            model_dir_b=model_b_dir,
            output_dir=artifact_dir,
            label_a=f"Model A ({model_a_hash[:8]})",
            label_b=f"Model B ({model_b_hash[:8]})",
            ignore_measurements=True,
            model_a_hash=model_a_hash,
            model_b_hash=model_b_hash,
            log_fn=log_fn,
        )
        register_comparison(wdn_name, h, inp, artifact_dir)
        return artifact_dir, False
    except Exception:
        import shutil
        shutil.rmtree(artifact_dir, ignore_errors=True)
        raise


# ---------------------------------------------------------------------------
# Legacy batch runner (kept for backward compatibility)
# ---------------------------------------------------------------------------

def run(wdn_name: str, timeout: int, phases: Dict[str, bool], ignore_measurements: bool) -> None:
    data_nb = json.loads(DATA_NOTEBOOK.read_text(encoding="utf-8"))
    gnn_nb = json.loads(GNN_NOTEBOOK.read_text(encoding="utf-8"))
    eval_nb = json.loads(EVAL_NOTEBOOK.read_text(encoding="utf-8"))
    params = _load_wdn_params(wdn_name)

    base_dir = params.get("base_dir")
    if not base_dir:
        base_dir = str(CWD / "old" / "data" / wdn_name)

    inp_dir = params.get("inp_dir")
    if not inp_dir:
        inp_dir = str(CWD / "wdn" / f"{wdn_name}.inp")

    base_dir_path = Path(base_dir)
    base_dir_path.mkdir(parents=True, exist_ok=True)
    source_params_path = base_dir_path / "parameters.json"
    source_params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")

    data_replacements = {
        "WDN_NAME": repr(params.get("wdn_name", wdn_name)),
        "EXTRA_DEMAND": repr(float(params.get("extra_demand", 0.0))),
        "MEASUREMENT_NODES": repr(params.get("measurement_nodes", [])),
        "INP_DIR": repr(inp_dir),
        "BASE_DIR": repr(base_dir),
    }

    if phases.get("data_generation", False):
        print(f"\n=== Running data_generation for {wdn_name} ===")
        data_nb = _inject_or_replace(data_nb, data_replacements)
        output_dir = base_dir_path / "data_generator"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        injected_params_path = output_dir / "parameters.json"
        injected_params_path.write_text(json.dumps(data_replacements, indent=2), encoding="utf-8")
        data_out = output_dir / f"data_generator_{wdn_name}.ran.ipynb"
        data_tmp = output_dir / f"data_generator_{wdn_name}.tmp.ipynb"
        _write_notebook(data_nb, data_tmp)
        _execute_notebook(data_tmp, data_out, timeout)

    gnn_replacements = {
        "WDN_NAME": repr(params.get("wdn_name", wdn_name)),
        "BASE_DIR": repr(base_dir),
    }

    if phases.get("gnn_model", False):
        print(f"\n=== Running gnn_model for {wdn_name} ===")
        gnn_nb = _inject_or_replace(gnn_nb, gnn_replacements)
        gnn_output_dir = base_dir_path / "gnn_model"
        if gnn_output_dir.exists():
            shutil.rmtree(gnn_output_dir)
        gnn_output_dir.mkdir(parents=True, exist_ok=True)
        gnn_params_path = gnn_output_dir / "parameters.json"
        gnn_params_path.write_text(json.dumps(gnn_replacements, indent=2), encoding="utf-8")
        gnn_out = gnn_output_dir / f"gnn_model_{wdn_name}.ran.ipynb"
        gnn_tmp = gnn_output_dir / f"gnn_model_{wdn_name}.tmp.ipynb"
        _write_notebook(gnn_nb, gnn_tmp)
        _execute_notebook(gnn_tmp, gnn_out, timeout)

    if phases.get("evaluation", False):
        print(f"\n=== Running evaluation for {wdn_name} ===")
        eval_replacements = {
            "WDN_NAME": repr(params.get("wdn_name", wdn_name)),
            "BASE_DIR": repr(base_dir),
            "IGNORE_MEASUREMENTS": repr(bool(ignore_measurements)),
        }
        eval_nb = _inject_or_replace(eval_nb, eval_replacements)
        eval_output_dir = base_dir_path / "evaluation"
        if eval_output_dir.exists():
            shutil.rmtree(eval_output_dir)
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        eval_params_path = eval_output_dir / "parameters.json"
        eval_params_path.write_text(json.dumps(eval_replacements, indent=2), encoding="utf-8")
        eval_out = eval_output_dir / f"evaluation_{wdn_name}.ran.ipynb"
        eval_tmp = eval_output_dir / f"evaluation_{wdn_name}.tmp.ipynb"
        _write_notebook(eval_nb, eval_tmp)
        _execute_notebook(eval_tmp, eval_out, timeout)


if __name__ == "__main__":
    WDN_NAMES = [
        "Anytown",
        "BAK",
        "Hanoi",
        "Baghmalek",
        "Kadu",
        "Modena",
        "ZhiJiang",
    ] 
    PHASES = {
        "data_generation": True,
        "gnn_model": True,
        "evaluation": True,
    }
    
    IGNORE_MEASUREMENTS = True
    TIMEOUT = 3600*10  # 10 hours

    for wdn_name in WDN_NAMES:
        run(wdn_name, TIMEOUT, PHASES, IGNORE_MEASUREMENTS)
