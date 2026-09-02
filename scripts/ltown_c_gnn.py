#!/usr/bin/env python3
"""GNN demand/leak predictor for L-TOWN_C, scored against the exact enumerator.

Three subcommands:  dataset -> train -> eval

Why this differs from the existing old/ pipeline
------------------------------------------------
The pipeline in old/ trains the GNN to predict *pressures* and then recovers demands
algebraically (`_reconstruct_demands`: q_e = sign(dh)*(|dh|/r_e)^(1/n), then mass balance).
That inversion is catastrophically ill-conditioned on Area C, whose median pipe head loss is
2e-5 m.  Measured on this network, reconstructing demands from true heads perturbed by
Gaussian noise:

    head noise   1e-4 m  ->  demand error  1.1x the mean demand
    head noise   1e-3 m  ->                7.1x
    head noise   1e-2 m  ->               33.6x   (= the sensor reading resolution)

So this script predicts **demands directly** and never inverts pressures.  (This also answers
the last open question in PLAN.md, "with GNN error in heads, how much demand error
accumulates?", for the low-flow case: far too much to be usable.)

Prior consistency
-----------------
Training scenarios come from the *same* single-leak prior the enumerator assumes
(`step2_estimation.simulate_single_leak_scenarios`): one node v uniform, lambda ~ U[0, Delta],
d = d0 + lambda*e_v.  Scoring a Dirichlet-trained model against a single-leak posterior would
be meaningless -- the model would never have seen a leak.

Target
------
y_j = (d_j - d0_j) / Delta, the *excess* demand share at junction j -- zero everywhere except
the leak node.  Predicting the excess rather than the absolute demand puts the leak signal at
O(1) instead of buried under a ~5e-5 m3/s baseline.

Features (per node, 4 values)
-----------------------------
    [ normalised base head,
      normalised observed head (sensors: measured & quantised; others: distance-weighted
                                interpolation of the sensor values),
      residual (observed - base) / residual_scale,      <- the actual leak signal
      node type (0 junction, 1 sensor, 2 reservoir) ]

The explicit residual channel matters: absolute heads are ~101.9 m while the leak signal is
~1e-2 m, so a model given only absolute heads has to recover a 1e-4 relative difference.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

INP = os.path.join(ROOT_DIR, "wdn", "L-TOWN_C.inp")
ART = os.path.join(ROOT_DIR, "data", "L-TOWN_C", "gnn")
QUANTUM = 0.01
XMAX = 50.0
OFFICIAL_SENSORS = ["n1", "n4", "n31"]
DEFAULT_DELTA = 18.0 / 3600.0     # BattLeDIM medium burst (10% of the 180 m3/h system inflow)


# --------------------------------------------------------------------------- helpers
def base_state(inp_path):
	import wntr
	wn = wntr.network.WaterNetworkModel(inp_path)
	wn.options.time.duration = 0
	wn.options.hydraulic.accuracy = 1e-8
	wn.options.hydraulic.trials = 200
	res = wntr.sim.EpanetSimulator(wn).run_sim()
	h = res.node["head"].iloc[0]
	node_names = list(wn.node_name_list)
	return wn, node_names, np.array([float(h[n]) for n in node_names], dtype=float)


def graph_arrays(inp_path, node_names):
	"""edge_index (2,E) both directions, plus the shortest-path distance matrix over junctions."""
	import wntr, networkx as nx
	wn = wntr.network.WaterNetworkModel(inp_path)
	idx = {n: i for i, n in enumerate(node_names)}
	src, dst = [], []
	G = nx.Graph()
	for p in wn.pipe_name_list:
		l = wn.get_link(p)
		a, b = l.start_node_name, l.end_node_name
		src += [idx[a], idx[b]]
		dst += [idx[b], idx[a]]
		G.add_edge(a, b, weight=float(l.length))
	return np.array([src, dst], dtype=np.int64), G


def sensor_interpolation_weights(G, node_names, sensors):
	"""W[i, s] = normalised 1/(1+d) weight of sensor s at node i (graph shortest path)."""
	import networkx as nx
	sp = {s: nx.single_source_dijkstra_path_length(G, s, weight="weight") for s in sensors}
	W = np.zeros((len(node_names), len(sensors)), dtype=float)
	for i, n in enumerate(node_names):
		for k, s in enumerate(sensors):
			d = sp[s].get(n, np.inf)
			W[i, k] = 1.0 / (1.0 + d) if np.isfinite(d) else 0.0
		tot = W[i].sum()
		W[i] = W[i] / tot if tot > 0 else 1.0 / len(sensors)
	return W


def build_features(heads, base_heads, sensor_cols, W, hmin, hmax, res_scale):
	"""(n_scenarios, n_nodes, 4) feature tensor.  `heads` are the true scenario heads; only
	the sensor columns are ever read (quantised), everything else is interpolated."""
	n_s, n_n = heads.shape
	obs = np.round(heads[:, sensor_cols] / QUANTUM) * QUANTUM        # (n_s, |Y|)
	interp = obs @ W.T                                               # (n_s, n_n)
	interp[:, sensor_cols] = obs                                     # sensors keep their own value
	span = max(hmax - hmin, 1e-12)
	base_n = np.broadcast_to((base_heads - hmin) / span, (n_s, n_n))
	obs_n = (interp - hmin) / span
	resid = (interp - base_heads[None, :]) / res_scale
	ntype = np.zeros(n_n, dtype=float)
	ntype[sensor_cols] = 1.0
	X = np.stack([base_n, obs_n, resid, np.broadcast_to(ntype, (n_s, n_n))], axis=2)
	return X.astype(np.float32)


def make_gcn(dim_in, dim_h=32):
	import torch
	from torch_geometric.nn import GCNConv

	class GCN(torch.nn.Module):
		"""Same residual-GCN architecture as old/compare_pressure.py, retargeted to demands."""
		def __init__(self, dim_in, dim_h, dim_out=1):
			super().__init__()
			H = dim_h * 4
			self.gcn1, self.gcn2, self.gcn3 = GCNConv(dim_in, H), GCNConv(H, H), GCNConv(H, H)
			self.bn1, self.bn2, self.bn3 = (torch.nn.BatchNorm1d(H) for _ in range(3))
			self.lin1, self.lin2 = torch.nn.Linear(H, dim_h), torch.nn.Linear(dim_h, dim_out)
			self.drop = torch.nn.Dropout(p=0.1)

		def forward(self, x, edge_index):
			h = self.drop(torch.relu(self.bn1(self.gcn1(x, edge_index))))
			h2 = self.drop(torch.relu(self.bn2(self.gcn2(h, edge_index)))) + h
			h3 = self.drop(torch.relu(self.bn3(self.gcn3(h2, edge_index)))) + h2
			return self.lin2(torch.relu(self.lin1(h3)))

	return GCN(dim_in, dim_h)


# --------------------------------------------------------------------------- dataset
def cmd_dataset(args):
	from step2_estimation import simulate_single_leak_scenarios
	os.makedirs(ART, exist_ok=True)
	print(f"Generating {args.n} single-leak scenarios (Delta = {args.delta * 3600:.2f} m3/h) ...")
	out = simulate_single_leak_scenarios(
		INP, args.n, args.delta, min_fraction=args.min_fraction, max_fraction=1.0, seed=args.seed)
	path = os.path.join(ART, "dataset.npz")
	np.savez_compressed(
		path,
		demands=out["demands"], heads=out["heads"],
		leak_index=out["leak_index"], leak_size=out["leak_size"],
		base_demands=out["base_demands"],
		junction_names=np.array(out["junction_names"]),
		node_names=np.array(out["node_names"]),
		delta=np.array([args.delta]),
	)
	print(f"wrote {path}  ({out['demands'].shape[0]} scenarios, "
		  f"{out['demands'].shape[1]} junctions, {out['heads'].shape[1]} nodes)")
	lam = out["leak_size"]
	print(f"leak sizes: {lam.min() * 3600:.2f}-{lam.max() * 3600:.2f} m3/h; "
		  f"{len(set(out['leak_index'].tolist()))} distinct leak nodes covered")
	return 0


# --------------------------------------------------------------------------- train
def cmd_train(args):
	import torch
	ds = np.load(os.path.join(ART, "dataset.npz"), allow_pickle=True)
	junc = [str(x) for x in ds["junction_names"]]
	node_names = [str(x) for x in ds["node_names"]]
	demands, heads = ds["demands"], ds["heads"]
	d0, delta = ds["base_demands"], float(ds["delta"][0])
	sensors = args.sensors
	scols = [node_names.index(s) for s in sensors]
	jcols = [node_names.index(j) for j in junc]

	_, _, base_heads = base_state(INP)
	edge_index, G = graph_arrays(INP, node_names)
	W = sensor_interpolation_weights(G, node_names, sensors)

	hmin, hmax = float(heads.min()), float(heads.max())
	res_scale = float(np.abs(heads - base_heads[None, :]).std()) or 1.0
	X = build_features(heads, base_heads, scols, W, hmin, hmax, res_scale)
	Y = ((demands - d0[None, :]) / delta).astype(np.float32)        # excess share, junctions only

	n = X.shape[0]
	rng = np.random.default_rng(args.seed)
	perm = rng.permutation(n)
	n_te = max(1, int(0.15 * n)); n_va = max(1, int(0.15 * n))
	te, va, tr = perm[:n_te], perm[n_te:n_te + n_va], perm[n_te + n_va:]
	print(f"split: train {len(tr)}, val {len(va)}, test {len(te)}")
	baseline = float(np.mean(Y[va] ** 2))
	print(f"all-zeros baseline val MSE = {baseline:.6f} (target is one-hot-ish: a STRONG trivial baseline)")

	dev = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
	# CRITICAL on CPU: this graph is tiny (93 nodes, 218 edges), so per-op tensors are far too
	# small to parallelise and OpenMP overhead dominates.  Measured on this machine, one
	# GCNConv(128->128) over a 32-scenario batch takes 193 ms on 8 threads and 5.5 ms on 1 --
	# a 35x slowdown (253 s/epoch vs ~7 s/epoch) purely from thread thrashing.
	if dev.type == "cpu":
		torch.set_num_threads(args.threads)
	print(f"device: {dev}  (torch threads {torch.get_num_threads()})")
	ei = torch.tensor(edge_index, device=dev)
	Xt = torch.tensor(X, device=dev)
	Yt = torch.tensor(Y, device=dev)
	jc = torch.tensor(jcols, dtype=torch.long, device=dev)

	model = make_gcn(X.shape[2], args.dim_h).to(dev)
	print(f"model width {args.dim_h} (hidden {args.dim_h * 4}), "
		  f"{sum(q.numel() for q in model.parameters())} params")
	opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
	sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=15)

	# The graph is identical across scenarios, so a batch is one big block-diagonal graph:
	# replicate edge_index with a node offset per scenario and run a single forward pass.
	# (Looping one scenario at a time is ~batch-size times slower for no benefit.)
	n_nodes = X.shape[1]
	_ei_cache: dict = {}

	def batched_ei(b):
		if b not in _ei_cache:
			_ei_cache[b] = torch.cat([ei + k * n_nodes for k in range(b)], dim=1)
		return _ei_cache[b]

	def run_batch(idxs, train):
		model.train(train)
		tot, cnt = 0.0, 0
		for k in range(0, len(idxs), args.batch):
			sel = idxs[k:k + args.batch]
			b = len(sel)
			if b < 2 and train:
				continue                        # BatchNorm needs >1 sample
			xb = Xt[sel].reshape(b * n_nodes, -1)
			pred = model(xb, batched_ei(b)).squeeze(-1).reshape(b, n_nodes)[:, jc]
			loss = torch.nn.functional.mse_loss(pred, Yt[sel])
			if train:
				opt.zero_grad()
				loss.backward()
				opt.step()
			tot += float(loss) * b; cnt += b
		return tot / max(cnt, 1)

	best, bad, best_state = np.inf, 0, None
	for ep in range(1, args.epochs + 1):
		trl = run_batch(rng.permutation(tr), True)
		with torch.no_grad():
			val = run_batch(va, False)
		sched.step(val)
		if val < best - 1e-9:
			best, bad = val, 0
			best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
		else:
			bad += 1
		if ep % 5 == 0 or ep == 1:
			print(f"  epoch {ep:4d}  train {trl:.6f}  val {val:.6f}  best {best:.6f}")
		if bad >= args.patience:
			print(f"  early stop at epoch {ep} (no val improvement for {args.patience})")
			break

	model.load_state_dict(best_state)
	path = os.path.join(ART, "model.pt")
	torch.save({"state_dict": best_state, "sensors": sensors, "hmin": hmin, "hmax": hmax,
				"res_scale": res_scale, "dim_in": X.shape[2], "dim_h": args.dim_h, "test_idx": te,
				"delta": delta, "val_mse": best}, path)
	print(f"wrote {path}  (best val MSE {best:.6f})")
	return 0


# --------------------------------------------------------------------------- eval
def _load_enumerator():
	"""Import scripts/ltown_c_enumerate.py for its exact-posterior helpers."""
	import importlib.util
	path = os.path.join(ROOT_DIR, "scripts", "ltown_c_enumerate.py")
	spec = importlib.util.spec_from_file_location("ltown_c_enumerate", path)
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod


def cmd_eval(args):
	"""Score the GNN against the exact single-leak posterior on the same test scenarios.

	The enumerator is Bayes-optimal under this prior, so its localization error is a FLOOR
	the GNN cannot beat; the gap between them is what 'how much does the GNN lose' means.
	"""
	import torch
	enum = _load_enumerator()

	ds = np.load(os.path.join(ART, "dataset.npz"), allow_pickle=True)
	junc = [str(x) for x in ds["junction_names"]]
	node_names = [str(x) for x in ds["node_names"]]
	demands, heads = ds["demands"], ds["heads"]
	leak_index, leak_size = ds["leak_index"], ds["leak_size"]
	d0, delta = ds["base_demands"], float(ds["delta"][0])

	ck = torch.load(os.path.join(ART, "model.pt"), weights_only=False)
	sensors = ck["sensors"]
	scols = [node_names.index(s) for s in sensors]
	jcols = [node_names.index(j) for j in junc]
	sjcols = [junc.index(s) for s in sensors]

	_, _, base_heads = base_state(INP)
	edge_index, G = graph_arrays(INP, node_names)
	W = sensor_interpolation_weights(G, node_names, sensors)
	Gd = enum.graph_distances(INP, junc)

	test = ck["test_idx"]
	if args.max_test and len(test) > args.max_test:
		test = test[:args.max_test]
	print(f"evaluating on {len(test)} test scenarios (sensors {sensors}, Delta {delta * 3600:.2f} m3/h)")

	X = build_features(heads[test], base_heads, scols, W, ck["hmin"], ck["hmax"], ck["res_scale"])
	dev = torch.device("cpu")
	model = make_gcn(ck["dim_in"], ck.get("dim_h", 32)).to(dev)
	model.load_state_dict(ck["state_dict"]); model.eval()
	ei = torch.tensor(edge_index, device=dev)
	jc = torch.tensor(jcols, dtype=torch.long, device=dev)

	with torch.no_grad():
		Yhat = np.stack([model(torch.tensor(X[i], device=dev), ei).squeeze(-1)[jc].numpy()
						 for i in range(len(test))])
	d_hat = d0[None, :] + Yhat * delta                       # predicted demands
	d_act = demands[test]

	gnn_err, enum_err, enum_exp, gnn_rank, enum_rank = [], [], [], [], []
	for k, i in enumerate(test):
		t = int(leak_index[i])
		lam = float(leak_size[i])                            # exactly recoverable from D
		# --- GNN: leak = junction with the largest predicted excess
		v_gnn = int(np.argmax(Yhat[k]))
		gnn_err.append(Gd[v_gnn, t])
		gnn_rank.append(int((Yhat[k] > Yhat[k][t]).sum()) + 1)
		# --- enumerator: exact posterior at this lambda
		H = enum.candidate_heads(INP, junc, d0, lam)
		Hs = H[:, [junc.index(s) for s in sensors]]
		obs = np.round(heads[i][scols] / QUANTUM) * QUANTUM
		P = enum.posterior(Hs, obs, args.eps)
		enum_err.append(Gd[int(P.argmax()), t])
		enum_exp.append(float(P @ Gd[:, t]))
		enum_rank.append(int((P > P[t]).sum()) + 1)
		if (k + 1) % 25 == 0:
			print(f"  {k + 1}/{len(test)}", flush=True)

	gnn_err, enum_err = np.array(gnn_err), np.array(enum_err)
	enum_exp = np.array(enum_exp)

	l2 = np.linalg.norm(d_hat - d_act, axis=1)
	ss_res = float(np.sum((d_hat - d_act) ** 2))
	ss_tot = float(np.sum((d_act - d_act.mean()) ** 2))
	print("\n=== demand prediction ===")
	print(f"L2 error         median {np.median(l2) * 3600:.4f} m3/h   mean {l2.mean() * 3600:.4f} m3/h")
	print(f"R^2 (demands)    {1.0 - ss_res / ss_tot:.4f}")
	err_leak = np.abs(d_hat[np.arange(len(test)), leak_index[test]] - d_act[np.arange(len(test)), leak_index[test]])
	print(f"|error| at the true leak node: median {np.median(err_leak) * 3600:.4f} m3/h "
		  f"(leak sizes {leak_size[test].min() * 3600:.2f}-{leak_size[test].max() * 3600:.2f})")

	print("\n=== leak localization (metres along the pipes) ===")
	print(f"{'':<26}{'GNN':>12}{'enumerator (exact)':>22}")
	print(f"{'MAP error, median':<26}{np.median(gnn_err):>12.1f}{np.median(enum_err):>22.1f}")
	print(f"{'MAP error, mean':<26}{gnn_err.mean():>12.1f}{enum_err.mean():>22.1f}")
	print(f"{'within 50 m':<26}{100 * np.mean(gnn_err <= XMAX):>11.0f}%{100 * np.mean(enum_err <= XMAX):>21.0f}%")
	print(f"{'exact node':<26}{100 * np.mean(gnn_err == 0):>11.0f}%{100 * np.mean(enum_err == 0):>21.0f}%")
	print(f"{'rank of true node, median':<26}{np.median(gnn_rank):>12.0f}{np.median(enum_rank):>22.0f}")
	print(f"\nenumerator expected (posterior-weighted) error: median {np.median(enum_exp):.1f} m")
	print(f"gap GNN - enumerator (MAP, mean): {gnn_err.mean() - enum_err.mean():+.1f} m")
	print("\nthe enumerator is Bayes-optimal under this prior, so its row is a floor;")
	print("a negative gap on a finite sample is noise, not a better-than-optimal predictor.")

	if args.out:
		with open(args.out, "w") as fh:
			json.dump({"sensors": sensors, "eps": args.eps, "delta": delta,
				"n_test": int(len(test)),
				"gnn_map_error_m": gnn_err.tolist(), "enum_map_error_m": enum_err.tolist(),
				"enum_expected_error_m": enum_exp.tolist(),
				"demand_l2": l2.tolist(), "demand_r2": 1.0 - ss_res / ss_tot}, fh, indent=2)
		print(f"\nwrote {args.out}")
	return 0


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	sub = ap.add_subparsers(dest="cmd", required=True)

	p = sub.add_parser("dataset", help="generate single-leak scenarios")
	p.add_argument("-n", type=int, default=6000)
	p.add_argument("--delta", type=float, default=DEFAULT_DELTA)
	p.add_argument("--min-fraction", type=float, default=0.05,
		help="lower end of lambda/Delta; 0.05 avoids a mass of invisible sub-resolution leaks")
	p.add_argument("--seed", type=int, default=1)
	p.set_defaults(func=cmd_dataset)

	p = sub.add_parser("train", help="train the demand-predicting GNN")
	p.add_argument("--sensors", nargs="*", default=OFFICIAL_SENSORS)
	p.add_argument("--epochs", type=int, default=200)
	p.add_argument("--batch", type=int, default=32)
	p.add_argument("--lr", type=float, default=1e-3)
	p.add_argument("--patience", type=int, default=40)
	p.add_argument("--seed", type=int, default=1)
	p.add_argument("--cpu", action="store_true")
	p.add_argument("--dim-h", type=int, default=32, dest="dim_h")
	p.add_argument("--threads", type=int, default=1,
		help="CPU torch threads; 1 is ~35x faster here (see comment in cmd_train)")
	p.set_defaults(func=cmd_train)

	p = sub.add_parser("eval", help="score the GNN against the exact enumerator")
	p.add_argument("--eps", type=float, default=0.015)
	p.add_argument("--max-test", type=int, default=200,
		help="the enumerator costs 92 forward solves per scenario, so subsample")
	p.add_argument("--out", default=None)
	p.set_defaults(func=cmd_eval)

	args = ap.parse_args()
	return args.func(args)


if __name__ == "__main__":
	raise SystemExit(main())
