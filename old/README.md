# Old Pipeline (SMARTWINE)

This folder contains the notebook-based pipeline used to generate datasets, train a GNN, and produce evaluation visualizations.

## Prerequisites
- Python 3.11
- Packages: numpy, wntr (1.2.0), torch, torch_geometric, matplotlib, pillow, papermill

## Pipeline Overview
The pipeline is three phases executed in order:
1. data_generation: Generates datasets and artifacts in old/data/<wdn>/data_generator
2. gnn_model: Trains the model and writes old/data/<wdn>/gnn_model/best_model.pt
3. evaluation: Produces evaluation plots in old/data/<wdn>/evaluation

The runner is old/remote.py, which injects parameters into notebooks and executes them with papermill.

## Effective Usage
1. Configure per-network parameters in wdn/<WDN>.json.
2. Open old/remote.py and set:
   - WDN_NAMES to the networks you want
   - PHASES to enable/disable data_generation, gnn_model, evaluation
   - IGNORE_MEASUREMENTS if you want to exclude measurement nodes from R2 plots
   - TIMEOUT as needed
3. Run:
   python old/remote.py

Outputs are stored under old/data/<wdn> per phase.

## Recommended Parameters (Best Results)
Use the per-network values in wdn/*.json. These files capture the tuned settings used for stable training and clean evaluation plots:
- extra_demand: Network-specific; use the value in the JSON
- measurement_nodes: Use the list in the JSON
- num_simulations: 5000 is the default across networks and works well
- node_label_threshold: 0.0 in current configs for dense labeling

Optional visualization parameters supported by evaluation:
- scale, node_scale, font_scale, non_special_node_scale (if present in wdn/*.json)

## Notes
- data_generation must run before gnn_model and evaluation (it writes the datasets).
- If you only want evaluation plots, ensure best_model.pt exists in old/data/<wdn>/gnn_model.
- For post-processing evaluation PNGs, see old/compactify.py.
