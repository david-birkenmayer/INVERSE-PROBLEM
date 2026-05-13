# Integration Plan: GNN Pipeline ↔ Inverse Solver

**Goal:** Train two GNNs with different sensor placements — the default placement
from `wdn/*.json` and the inverse-solver-optimal placement — then compare their
quality in demand space.

---

## Status

| Item | State |
|---|---|
| Cache portability (`gui/cache.py`, `gui/app.py`) | ✅ Done |
| Hexaly-free cached runs confirmed | ✅ Done |
| Dirichlet demand model (`step2_estimation.py`) | ✅ Done |
| `scenario.py` dirichlet option | ✅ Done |
| GUI dirichlet controls | ✅ Done |
| Empirical validation (radius not much worse under Dirichlet) | ✅ Confirmed |
| Store demands in GNN dataset | ✅ Done |
| Artifact manifests and cache keys | ✅ Done |
| GUI page for GNN dataset/model runs | ✅ Done |
| Flexible measurement placement input | ✅ Done |
| GUI-initiated model comparison | ✅ Done (Compare sub-page wired) |

---

## Revised Approach (simplified)

Train the GNN multiple times with the same dataset format and architecture, varying only
the set of measurement nodes chosen by the user in the GUI.

Two placements matter initially:

| Variant label | Measurement nodes source | Intended use |
|---|---|---|
| `default` | `wdn/<WDN>.json → measurement_nodes` | baseline model |
| `candidate` | chosen interactively in GUI | inverse-optimal or any custom placement |

The important change is that the placement stays **flexible**. It should not be baked into
`wdn/*.json` beyond the existing default placement.

Evaluate all trained models on the **same held-out test set** in demand space.

### Alperovits example (1 sensor)

- Default: `["1"]` (current JSON)
- Current inverse-optimal candidate under Dirichlet (hash `895be2d9`): **`["5"]`**, radius = 0.167
- A future GUI run should be able to use `["5"]` without changing the JSON config

---

## Step 1 — Store demands in the GNN dataset

**File:** `old/data_generator.ipynb`

### 1a. `generate_training_data` — capture demand_distribution

Currently discards it:
```python
wn_mod, _ = add_uniform_extra_demand(wn_mod, extra_demand)
...
return pressures, False, set()
```

Change to:
```python
wn_mod, demand_distribution = add_uniform_extra_demand(wn_mod, extra_demand)
...
return pressures, demand_distribution, False, set()
```

### 1b. `generate_gnn_dataset` — unpack and store demands

Currently:
```python
simulated_pressures, _, _ = result
```

Change to:
```python
simulated_pressures, demand_distribution, _, _ = result
```

Add to `data_dict`:
```python
junction_names = [n for n in G.nodes()
                  if not str(n).startswith('meas_')
                  and G.nodes[n].get('node_type','').lower() != 'reservoir']
d_values = [demand_distribution.get(str(n), 0.0) for n in junction_names]
data_dict['d'] = torch.tensor(d_values, dtype=torch.float)
data_dict['junction_names'] = junction_names
```

### 1c. `save_dataset_splits` — pass `d` through `Data`

In `create_pyg_data`:
```python
return Data(
    x=..., edge_index=..., edge_attr=...,
    y=..., edge_type=..., mask=...,
    d=simulation_data['d'],   # ← add
)
```

---

## Step 2 — Artifact schema and cache keys

**Purpose:** Avoid recomputation when the same placement and generation settings are used again.

Uses `compute_hash()` from `gui/cache.py` (SHA-256 of sorted JSON).

### 2a. Directory layout

```text
old/data/<WDN>/
  datasets/<dataset_hash>/
    manifest.json
    train_dataset.pt
    val_dataset.pt
    test_dataset.pt
    dataset_stats.json          ← {min_p, max_p}
    graph_with_measurements.pickle
    _gnn_hash.txt               ← contains dataset_hash (mirrors _gui_hash.txt pattern)

  models/<model_hash>/
    manifest.json
    best_model.pt
    training_stats.json         ← {train_loss[], val_loss[], best_epoch}
    _gnn_hash.txt

  shared_test_sets/<test_hash>/
    manifest.json
    test_dataset.pt             ← same PyG Data objects (with d field)
    junction_names.json
    _gnn_hash.txt

  comparisons/<comparison_hash>/
    manifest.json
    results.json                ← metrics per model
    comparison_plot.png
    _gnn_hash.txt
```

### 2b. Hash inputs per artifact type

**dataset_hash** (`dict`, all keys sorted):
```json
{
  "wdn":               "Alperovits",
  "measurement_nodes": ["1"],           // sorted list
  "extra_demand":      1.2,
  "num_simulations":   5000,
  "demand_model":      "uniform",       // "uniform" | "dirichlet"
  "node_label_threshold": 0.0,
  "code_version":      "data_generator_v1"  // bump when notebook logic changes
}
```

**model_hash** (`dict`):
```json
{
  "dataset_hash": "<hex>",
  "epochs":       200,
  "lr":           0.001,
  "batch_size":   32,
  "hidden_dim":   64,
  "num_layers":   3,
  "seed":         42,
  "code_version": "gnn_model_v1"
}
```

**test_hash** (shared test set):
```json
{
  "wdn":               "Alperovits",
  "extra_demand":      1.2,
  "num_simulations":   1000,            // test-set size
  "demand_model":      "uniform",
  "seed":              9999,            // fixed, separate from training seed
  "code_version":      "data_generator_v1"
}
```
Note: measurement_nodes are **not** part of the test hash. The shared test set has no
measurement nodes injected — only ground-truth demands and pressures. Each model
observes its own subset at inference time.

**comparison_hash**:
```json
{
  "model_a_hash": "<hex>",
  "model_b_hash": "<hex>",
  "test_hash":    "<hex>",
  "demand_reconstruction": "algebraic"  // "algebraic" | "wntr"
}
```

### 2c. manifest.json fields (common across all types)

```json
{
  "artifact_type":  "dataset",          // "dataset" | "model" | "test_set" | "comparison"
  "hash":           "<hex>",
  "created_at":     "2026-05-05T12:00:00",
  "inputs":         { ... }             // the exact hash-input dict from 2b
}
```

### 2d. Cache index

Mirror the solver index pattern: one JSON file per WDN per artifact type.

```text
old/data/<WDN>/datasets/index.json    → { "<hash>": "old/data/<WDN>/datasets/<hash>" }
old/data/<WDN>/models/index.json
old/data/<WDN>/shared_test_sets/index.json
old/data/<WDN>/comparisons/index.json
```

New helper: `old/gnn_cache.py`

```python
# old/gnn_cache.py
from gui.cache import compute_hash, load_index, save_index
ROOT_DIR = ...

def dataset_hash(params: dict) -> str: ...
def model_hash(params: dict) -> str: ...
def test_set_hash(params: dict) -> str: ...
def comparison_hash(params: dict) -> str: ...

def find_dataset(wdn: str, h: str) -> str | None: ...   # returns dir or None
def find_model(wdn: str, h: str) -> str | None: ...
def find_test_set(wdn: str, h: str) -> str | None: ...
def find_comparison(wdn: str, h: str) -> str | None: ...

def register_dataset(wdn: str, h: str, artifact_dir: str) -> None: ...
def register_model(wdn: str, h: str, artifact_dir: str) -> None: ...
# etc.
```

---

## Step 3 — GUI page for individual GNN runs

### 3a. New tab: "GNN" (added after "Scenario" tab)

Two sub-pages, controlled by a `QTabWidget` inside the tab:

**Sub-page A: "Run"**

```
┌─ GNN Run ────────────────────────────────────────────┐
│ WDN:            [Alperovits ▾]                        │
│                                                       │
│ ── Dataset ─────────────────────────────────────────  │
│ Default nodes:  1                      (read-only)    │
│ Nodes to use:   [1                    ] (editable)    │
│ Extra demand:   [1.2  ]                               │
│ Num sims:       [5000 ]                               │
│ Demand model:   [uniform ▾]                           │
│ Dataset hash:   a3f9...  [● Exists / ○ Missing]       │
│ [Generate Dataset]    [progress bar]                  │
│                                                       │
│ ── Model ───────────────────────────────────────────  │
│ Epochs:         [200  ]                               │
│ LR:             [0.001]                               │
│ Batch size:     [32   ]                               │
│ Hidden dim:     [64   ]                               │
│ Num layers:     [3    ]                               │
│ Seed:           [42   ]                               │
│ Model hash:     b2c1...  [● Exists / ○ Missing]       │
│ [Train Model]         [progress bar]                  │
│                                                       │
│ ── Log ─────────────────────────────────────────────  │
│ [scrollable text area]                                │
└──────────────────────────────────────────────────────┘
```

**Sub-page B: "Compare"**

```
┌─ GNN Compare ────────────────────────────────────────┐
│ WDN:            [Alperovits ▾]                        │
│                                                       │
│ Model A:        [b2c1... (nodes: 1, 5000 sims) ▾]     │
│ Model B:        [d8fa... (nodes: 5, 5000 sims) ▾]     │
│                                                       │
│ ── Shared test set ────────────────────────────────── │
│ Test sims:      [1000  ]                              │
│ Test seed:      [9999  ]                              │
│ Demand model:   [uniform ▾]                           │
│ Test set hash:  e7b3...  [● Exists / ○ Missing]       │
│ [Generate Test Set]                                   │
│                                                       │
│ Reconstruction: [algebraic ▾]                         │
│ Comparison hash: f1a2...  [● Exists / ○ Missing]      │
│ [Run Comparison]                                      │
│                                                       │
│ ── Results ────────────────────────────────────────── │
│ ┌──────────────────┬─────────┬─────────┐              │
│ │ Metric           │ Model A │ Model B │              │
│ │ Demand L2 (mean) │         │         │              │
│ │ Demand L2 (std)  │         │         │              │
│ │ Demand R²        │         │         │              │
│ │ Pressure R²      │         │         │              │
│ └──────────────────┴─────────┴─────────┘              │
│ [Export results to JSON]                              │
└──────────────────────────────────────────────────────┘
```

### 3b. Worker classes (mirroring `SolverWorker`)

```python
class GNNDatasetWorker(QtCore.QThread):
    log_line = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(bool, str)   # success, artifact_dir
    # Runs remote.run_dataset_only(wdn, params) in subprocess

class GNNModelWorker(QtCore.QThread):
    log_line = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(bool, str)
    # Runs remote.run_model_only(wdn, dataset_hash, params) in subprocess

class GNNCompareWorker(QtCore.QThread):
    log_line = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(bool, dict)   # success, results_dict
    # Runs remote.run_comparison(wdn, model_a_hash, model_b_hash, test_hash, method)
```

### 3c. Hash display and status check

On any input change, recompute hash and check the index file:
```python
def _refresh_dataset_status(self):
    params = self._current_dataset_params()
    h = dataset_hash(params)
    self.dataset_hash_label.setText(h[:8] + "...")
    exists = find_dataset(self.wdn_selector.currentText(), h) is not None
    self.dataset_status_label.setText("● Exists" if exists else "○ Missing")
    self.generate_dataset_btn.setEnabled(not exists)
    # enable Train Model only if dataset exists
    self.train_model_btn.setEnabled(exists and not model_exists)
```

### 3d. Model dropdown population

The "Compare" sub-page populates model dropdowns by scanning the index file:
```python
def _populate_model_dropdowns(self, wdn: str):
    index = load_index(f"old/data/{wdn}/models/index.json", ROOT_DIR)
    items = []
    for h, artifact_dir in index.items():
        manifest = _read_json(os.path.join(ROOT_DIR, artifact_dir, "manifest.json"))
        inputs = manifest.get("inputs", {})
        nodes = inputs.get("measurement_nodes", [])
        nsims = inputs.get("num_simulations", "?")
        label = f"{h[:8]}... (nodes: {','.join(nodes)}, {nsims} sims)"
        items.append((label, h))
    for combo in [self.model_a_combo, self.model_b_combo]:
        combo.clear()
        for label, h in items:
            combo.addItem(label, userData=h)
```

---

## Step 4 — Shared test set and GUI-initiated comparison

The shared test set is generated independently of any model. It uses a fixed seed (9999
by default) to ensure reproducibility across model comparisons. The test set hash does
**not** include `measurement_nodes` — both models receive the same ground-truth pressures
and each model observes only its own measurement subset at inference.

Comparison flow:
1. user selects Model A and Model B from dropdowns (populated from index)
2. user configures test set (num sims, seed, demand model)
3. GUI shows test-set hash; if exists, reuse; else "Generate Test Set"
4. user clicks "Run Comparison"
5. `GNNCompareWorker` calls `remote.run_comparison(...)` which runs `comparison.ipynb`
   (or inline Python) and writes `results.json` + `comparison_plot.png`
6. results table populated from `results.json`
7. user can export to JSON with "Export results"

---

## Step 5 — Demand-space evaluation

**Entry point:** GUI action backed by a Python module or notebook runner

### How to go from GNN predictions to demands

The GNN predicts `ŷ` (normalized pressures). Denormalize:

```
ĥ_i = ŷ_i * (max_p - min_p) + min_p
```

Then recover demands from the predicted hydraulic state.

Two candidate methods:
1. **Algebraic flow-continuity route**
   - compute pipe flows from predicted head differences and pipe resistance
   - compute junction demands via flow conservation, exactly like `_compute_demands_from_flows` in `inverse.py`
2. **WNTR-backed route**
   - use predicted heads as the target hydraulic state
   - solve for flows/demands with WNTR or a small reconstruction routine

Method 1 is cheaper. Method 2 is more defensible if head prediction noise makes the algebraic route unstable.

### Metrics (per test sample, aggregated over shared test set)

```
demand_L2 = ||d̂ - d||₂      (d from stored data.d, d̂ from predicted heads)
demand_R2                     (across all junctions)
pressure_R2                   (sanity-check metric already used by the GNN pipeline)
```

Report per comparison:

| Metric | Model A | Model B |
|---|---|---|
| Demand L2 (mean ± std) | ? | ? |
| Demand R² | ? | ? |
| Pressure R² | ? | ? |

---

## GNN input format (resolved)

Feature vector per node — 3 values for steady-state networks:

```
x[i] = [(base_pressure - min_p) / (max_p - min_p),
         additional_value,
         node_type]
```

where `additional_value` is:
- **measurement node** (`meas_<id>`): normalized observed pressure (injected at inference)
- **regular junction**: distance-weighted average of nearby measurement pressures
- **isolated node**: mean of base pressure vector (fallback)

`node_type`: 0=junction, 1=measurement, 2=reservoir, 3=other  
`min_p`, `max_p` saved in `old/data/<WDN>/data_generator/dataset_stats.json`  
Graph saved as `old/data/<WDN>/data_generator/graph_with_measurements.pickle`

---

## Demand model alignment (completed)

Both training and scenario generation now use the Dirichlet model:
```
d_j = d_j_base + α_j · extra_demand,   α ~ Dirichlet(1,...,1)
```

Available via `SCENARIO_SOURCE="dirichlet"` and `PIPE_BOUND_METHOD="dirichlet"` in
`scenario.py`. GUI exposes these controls.

---

## Open Questions

- **Demand inversion on looped networks**: the algebraic approach (flow continuity)
  is exact if predicted heads are exact; with GNN error in heads, how much demand
  error accumulates? Worth checking on Alperovits.
- **Comparison metric**: demand L2 is most directly comparable to inverse radius;
  pressure R² is what `evaluation.ipynb` already reports. Report both.
- **Shared test set**: both variants are trained on independently generated datasets
  (same demand distribution, different random seeds implied by independent runs).
  For a truly fair comparison, both should be evaluated on the *same* held-out
  demand scenarios. This requires generating one shared test set and evaluating
  both models on it, rather than using each model's own test split.
