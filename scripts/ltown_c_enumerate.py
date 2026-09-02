#!/usr/bin/env python3
"""Exact single-leak posterior for L-TOWN_C (BattLeDIM Area C) — no MCMC.

The model
---------
Prior:  pick a leak node v uniformly over the 92 junctions, and a magnitude
        lambda ~ U[0, Delta];  demands are  d = d0 + lambda * e_v.
Data:   an observation z = (D^z, h^z_Y): the total demand (BattLeDIM measures it — there
        are flow sensors at the DMA entrances) and the pressures at the sensor set Y,
        rounded to 2 decimal places as the challenge specifies.

Because the leak changes exactly one coordinate,  1'd = D0 + lambda, so the observed total
demand determines lambda EXACTLY:  lambda* = D^z - D0.  The prior's support — a union of 92
line segments radiating from d0 — collapses to 92 isolated points, and Bayes reduces to a
finite sum:

    P(v | z)  proportional to  K_eps( h_Y(d0 + lambda* e_v) - h^z_Y ),     v = 1..92

which is 92 forward solves.  No Metropolis, no Jacobian, no burn-in, no R-hat: this is the
exact posterior, and it is therefore the ground truth against which the MCMC samplers
(M1/M2/M5/M3) should be scored — strictly better than the sampled ABC oracle, which carries
its own Monte-Carlo error.

This is the "no demand noise" case: d0 is trusted exactly.  BattLeDIM's nominal model
actually randomizes base demands +-10%, which would make lambda uncertain and add 92 nuisance
dimensions — that is the case where MCMC becomes necessary (see scripts/ltown_c_posterior.py).

Metrics
-------
Localization is scored the way BattLeDIM scores it: by DISTANCE, not by demand error.  The
competition used a maximum detection distance of x_max = 50 m.  We report the shortest-path
(pipe-length) distance from the posterior to the true leak:

    expected error  =  sum_v P(v|z) * d_G(v, v_true)      [the graph-Wasserstein barycentric
    MAP error       =  d_G(argmax_v P(v|z), v_true)        distance to the truth]

Usage
-----
  python3 scripts/ltown_c_enumerate.py                       # all 92 leaks, official sensors
  python3 scripts/ltown_c_enumerate.py --leak-node n370      # one scenario, verbose
  python3 scripts/ltown_c_enumerate.py --eps-sweep           # how eps changes localization
  python3 scripts/ltown_c_enumerate.py --sensors n1 n4 n31 n360 --leak-size 0.0005
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
QUANTUM = 0.01          # BattLeDIM: readings rounded to 2 decimal places
XMAX = 50.0             # BattLeDIM: maximum detection distance, metres
OFFICIAL_SENSORS = ["n1", "n4", "n31"]   # from SMARTWINE l-town_edt.inp ";AMR & PRESSURE SENSOR"


def build(inp_path):
	"""Model + graph.  EPANET accuracy is tightened: Area C's head differences are ~1e-2 m
	and the .inp ships with Accuracy 0.01, which is far too loose to resolve them."""
	import wntr
	wn = wntr.network.WaterNetworkModel(inp_path)
	wn.options.time.duration = 0
	wn.options.hydraulic.accuracy = 1e-8
	wn.options.hydraulic.trials = 200
	return wn


def solve_heads(inp_path, junc_ids, demands):
	import wntr
	wn = build(inp_path)
	for j, d in zip(junc_ids, demands):
		wn.get_node(j).demand_timeseries_list[0].base_value = float(d)
	res = wntr.sim.EpanetSimulator(wn).run_sim()
	h = res.node["head"].iloc[0]
	return np.array([float(h[j]) for j in junc_ids], dtype=float)


def graph_distances(inp_path, junc_ids):
	import networkx as nx
	wn = build(inp_path)
	G = nx.Graph()
	for p in wn.pipe_name_list:
		l = wn.get_link(p)
		G.add_edge(l.start_node_name, l.end_node_name, weight=float(l.length))
	sp = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
	n = len(junc_ids)
	D = np.zeros((n, n), dtype=float)
	for i, a in enumerate(junc_ids):
		for j, b in enumerate(junc_ids):
			D[i, j] = sp[a][b]
	return D


def candidate_heads(inp_path, junc_ids, d0, lam):
	"""H[v] = full head vector for a leak of size lam at junction v.  92 forward solves.

	This is the whole computation: every scenario in the sweep reuses these rows, because a
	'true' leak at v produces exactly the observation H[v] restricted to the sensors."""
	H = np.empty((len(junc_ids), len(junc_ids)), dtype=float)
	for i in range(len(junc_ids)):
		d = d0.copy()
		d[i] += lam
		H[i] = solve_heads(inp_path, junc_ids, d)
		if (i + 1) % 20 == 0:
			print(f"  forward solve {i + 1}/{len(junc_ids)}", flush=True)
	return H


def posterior(H_sens, obs, eps):
	"""P(v | z) for a Gaussian sensor kernel of width eps (log-sum-exp stabilised)."""
	r = H_sens - obs[None, :]
	logk = -0.5 * np.sum(r * r, axis=1) / (eps * eps)
	logk -= logk.max()
	w = np.exp(logk)
	return w / w.sum()


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--sensors", nargs="*", default=OFFICIAL_SENSORS)
	ap.add_argument("--leak-node", default=None, help="report one scenario in detail")
	ap.add_argument("--leak-size", type=float, default=None,
		help="lambda in m3/s (default 1.8 m3/h = BattLeDIM's smallest background leak)")
	ap.add_argument("--eps", type=float, default=0.015, help="sensor kernel width, m")
	ap.add_argument("--eps-sweep", action="store_true", help="sweep eps and report localization")
	ap.add_argument("--out", default=None)
	args = ap.parse_args()

	from step1_io import load_inp_network
	net = load_inp_network(INP)
	junc_ids = list(net.junctions.keys())
	idx = {j: i for i, j in enumerate(junc_ids)}
	d0 = np.array([net.junctions[j].base_demand for j in junc_ids], dtype=float)
	D0 = float(d0.sum())
	lam = args.leak_size if args.leak_size is not None else 1.8 / 3600.0

	bad = [s for s in args.sensors if s not in idx]
	if bad:
		print(f"ERROR: unknown sensor(s) {bad}", file=sys.stderr)
		return 2
	scols = [idx[s] for s in args.sensors]

	print(f"L-TOWN_C: {len(junc_ids)} junctions, D0 = {D0 * 3600:.2f} m3/h")
	print(f"sensors  = {args.sensors}")
	print(f"lambda   = {lam * 3600:.3f} m3/h ({lam / D0 * 100:.1f}% of Area-C inflow)")
	print(f"NOTE: total demand is observed, so lambda is recovered exactly; only v is unknown.\n")

	print("Enumerating the 92 candidate leaks (this is the entire computation):")
	H = candidate_heads(INP, junc_ids, d0, lam)
	Hs = H[:, scols]                                  # (92 candidates, |Y|)
	Gd = graph_distances(INP, junc_ids)

	def evaluate(eps):
		"""Localization performance over every possible true leak node."""
		exp_err, map_err, p_true, rank_true, entropy = [], [], [], [], []
		for t in range(len(junc_ids)):
			obs = np.round(Hs[t] / QUANTUM) * QUANTUM      # BattLeDIM quantization
			P = posterior(Hs, obs, eps)
			exp_err.append(float(P @ Gd[:, t]))
			map_err.append(float(Gd[int(P.argmax()), t]))
			p_true.append(float(P[t]))
			rank_true.append(int((P > P[t]).sum()) + 1)
			entropy.append(float(-np.sum(P * np.log2(np.maximum(P, 1e-300)))))
		return (np.array(exp_err), np.array(map_err), np.array(p_true),
				np.array(rank_true), np.array(entropy))

	if args.eps_sweep:
		print(f"\n{'eps (m)':>9} {'exp err (m)':>12} {'MAP err (m)':>12} {'<=50m':>7} "
			  f"{'P(true)':>9} {'rank':>6} {'eff.cands':>10}")
		for eps in [0.001, 0.005, 0.01, 0.015, 0.03, 0.05, 0.1]:
			e, m, p, r, ent = evaluate(eps)
			print(f"{eps:9.3f} {np.median(e):12.1f} {np.median(m):12.1f} "
				  f"{100 * np.mean(m <= XMAX):6.0f}% {np.median(p):9.3f} {np.median(r):6.0f} "
				  f"{np.median(2 ** ent):10.1f}")
		print(f"\n(medians over all 92 possible leak positions; 'eff.cands' = 2^entropy, the")
		print(f" effective number of indistinguishable candidates; random guessing would give ~92)")
		return 0

	if args.leak_node:
		if args.leak_node not in idx:
			print(f"ERROR: unknown leak node {args.leak_node}", file=sys.stderr)
			return 2
		t = idx[args.leak_node]
		obs = np.round(Hs[t] / QUANTUM) * QUANTUM
		P = posterior(Hs, obs, args.eps)
		print(f"\n--- scenario: true leak at {args.leak_node} ---")
		print(f"observed heads (rounded): " + ", ".join(f"{s}={v:.2f}" for s, v in zip(args.sensors, obs)))
		ent = float(-np.sum(P * np.log2(np.maximum(P, 1e-300))))
		print(f"\nposterior entropy {ent:.2f} bits  ->  {2 ** ent:.1f} effective candidates (of 92)")
		print(f"P(true node) = {P[t]:.4f},  rank {int((P > P[t]).sum()) + 1}")
		print(f"expected localization error {float(P @ Gd[:, t]):.1f} m,  "
			  f"MAP error {float(Gd[int(P.argmax()), t]):.1f} m")
		print(f"\ntop-10 candidates:")
		print(f"  {'node':>6} {'P(v|z)':>9} {'dist to truth':>14}")
		for v in np.argsort(-P)[:10]:
			mark = "  <-- TRUE" if v == t else ""
			print(f"  {junc_ids[v]:>6} {P[v]:9.4f} {Gd[v, t]:13.1f} m{mark}")
		return 0

	e, m, p, r, ent = evaluate(args.eps)
	print(f"\n--- localization over all 92 possible leak positions (eps = {args.eps} m) ---")
	print(f"expected error   median {np.median(e):7.1f} m   mean {e.mean():7.1f} m   max {e.max():7.1f} m")
	print(f"MAP error        median {np.median(m):7.1f} m   mean {m.mean():7.1f} m   max {m.max():7.1f} m")
	print(f"MAP within {XMAX:.0f} m (BattLeDIM x_max):  {100 * np.mean(m <= XMAX):.0f}% of leaks")
	print(f"exactly right node:                {100 * np.mean(m == 0):.0f}% of leaks")
	print(f"P(true node)     median {np.median(p):.3f}      rank of true node: median {np.median(r):.0f}")
	print(f"posterior entropy median {np.median(ent):.2f} bits -> {np.median(2 ** ent):.1f} effective candidates (of 92)")
	print(f"\nfor scale: network diameter 1142 m; guessing uniformly gives ~{float(Gd.mean()):.0f} m expected error")

	if args.out:
		with open(args.out, "w") as fh:
			json.dump({"sensors": args.sensors, "eps": args.eps, "lambda": lam,
				"junctions": junc_ids, "expected_error_m": e.tolist(),
				"map_error_m": m.tolist(), "p_true": p.tolist(),
				"rank_true": r.tolist(), "entropy_bits": ent.tolist()}, fh, indent=2)
		print(f"\nwrote {args.out}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
