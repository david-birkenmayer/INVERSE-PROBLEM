#!/usr/bin/env python3
"""
ABC (Approximate Bayesian Computation) reference posterior for validating the MCMC
posterior sampler in `mh_posteriori-scenario-gen.py`, on any small/medium WDN.

The MCMC samples pi(d | M) — demand scenarios consistent with a measurement
M = (sensor pressures, total demand) — in reduced *pressure* coordinates, with a
Newton-eliminated node and an exact Gram-determinant change-of-variables factor.

This computes the SAME posterior independently, in *demand* space, using only the forward
simulator (EPANET):
  1. draw shares  alpha ~ Dirichlet(1,...,1)
  2. scenario     d = d0 + Delta * alpha         (=> d >= d0 and sum d = D for free)
  3. simulate      h = EPANET(d)                 (reservoir head fixed, as physics demands)
  4. weight        w = exp(-||h[S]-M_S||^2 / 2 eps^2)
The weighted d-samples approximate pi(d | M).  It shares none of the MCMC machinery, so
agreement validates that machinery end to end.  The oracle is cached (it depends only on
network+sensors+Delta+seed+N, not on the sampler) and reused across sampler variants.

Examples:
    python3 scripts/abc_reference.py                              # Alperovits, sensor 1
    python3 scripts/abc_reference.py --wdn Kadu --sensors 14 --delta 0.15
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys
import warnings
from typing import Dict, List

import numpy as np

warnings.filterwarnings("ignore")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
	sys.path.insert(0, ROOT_DIR)

from step1_io import load_inp_network  # noqa: E402


def _epanet_solver(inp_path: str, junc_ids: List[str]):
	"""Return a function d_vector -> heads_dict using a reused WNTR/EPANET model."""
	import wntr

	wn = wntr.network.WaterNetworkModel(inp_path)

	def solve(d: np.ndarray) -> Dict[str, float]:
		for i, j in enumerate(junc_ids):
			wn.get_node(j).demand_timeseries_list[0].base_value = float(d[i])
		results = wntr.sim.EpanetSimulator(wn).run_sim()
		head_df = results.node["head"]
		t0 = head_df.index[0]
		return {n: float(head_df.loc[t0, n]) for n in head_df.columns}, True

	return solve


def _internal_solver(ns, inp_path: str, junc_ids: List[str], d0: np.ndarray, delta: float, init_heads):
	"""Return d_vector -> heads_dict using the sampler's OWN forward hydraulics.

	Using the same convex Newton solve as the M2 sampler (rather than EPANET) makes the ABC
	oracle share M2's hydraulics, so any M2-vs-oracle gap is purely the sampler, not a
	solver-consistency artifact.
	"""
	Sampler = ns["PosteriorScenarioSampler"]
	Cfg = ns["MHPosteriorConfig"]
	s = Sampler(
		inp_path=inp_path,
		measurement_heads={junc_ids[0]: float(init_heads[junc_ids[0]])},
		measured_total_demand=float(d0.sum() + delta),
		predictor_heads={k: float(v) for k, v in init_heads.items()},
		config=Cfg(method="demand"),
	)
	order = [s.junction_idx[j] for j in junc_ids]  # map junc_ids -> sampler junction order
	inv = np.argsort(order)

	def solve(d: np.ndarray):
		d_sampler = np.asarray(d, dtype=float)[inv]  # reorder into sampler junction order
		ok, h = s._forward_solve(d_sampler, s._base_head_vec.copy())
		return {nid: float(h[s.node_idx[nid]]) for nid in s.node_ids}, bool(ok)

	return solve


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--wdn", default="Alperovits")
	ap.add_argument("--sensors", nargs="+", default=["1"], help="measurement node ids")
	ap.add_argument("--delta", type=float, default=0.6, help="total extra demand D - D0")
	ap.add_argument("--seed", type=int, default=7, help="seed for the true scenario")
	ap.add_argument("--n-abc", type=int, default=25000)
	ap.add_argument("--abc-quantile", type=float, default=0.02)
	ap.add_argument("--burn-in", type=int, default=6000)
	ap.add_argument("--samples", type=int, default=6000)
	ap.add_argument("--rwm-chains", type=int, default=4)
	ap.add_argument("--rwm-proposal-std", type=float, default=0.5)
	ap.add_argument("--ens-walkers", type=int, default=0, help="0 = auto (2*dim+2)")
	ap.add_argument("--ens-disp", type=float, default=0.02)
	ap.add_argument("--m2-eps", type=float, default=0.0, help="M2 soft-sensor width; 0 = match ABC tolerance")
	ap.add_argument("--no-rwm", action="store_true", help="skip the random-walk variant")
	ap.add_argument("--hydraulics", choices=["epanet", "internal"], default="epanet",
		help="forward model for the ABC oracle: EPANET (independent) or the sampler's own solve "
			 "(same hydraulics as M2, isolates sampler correctness from solver consistency)")
	args = ap.parse_args()

	inp = os.path.join(ROOT_DIR, "wdn", f"{args.wdn}.inp")
	cache = os.path.join(ROOT_DIR, "scripts",
		f"abc_cache_{args.wdn}_{'-'.join(args.sensors)}_D{args.delta}_s{args.seed}_N{args.n_abc}_{args.hydraulics}.npz")

	ns = runpy.run_path(os.path.join(ROOT_DIR, "mh_posteriori-scenario-gen.py"))

	net = load_inp_network(inp)
	junc_ids = list(net.junctions.keys())
	d0 = np.array([net.junctions[j].base_demand for j in junc_ids], dtype=float)
	D = float(d0.sum()) + args.delta

	# Base heads (one EPANET call) seed the internal forward solve and fix the reservoir head.
	base_heads, _ = _epanet_solver(inp, junc_ids)(d0)
	if args.hydraulics == "internal":
		solve = _internal_solver(ns, inp, junc_ids, d0, args.delta, base_heads)
	else:
		solve = _epanet_solver(inp, junc_ids)

	# ---- ABC oracle (cached; independent of the MCMC sampler)
	if os.path.exists(cache):
		print(f"Loading cached ABC oracle from {os.path.basename(cache)}")
		z = np.load(cache)
		d_samples, resid, d_true = z["d_samples"], z["resid"], z["d_true"]
	else:
		rng = np.random.default_rng(args.seed)
		d_true = d0 + args.delta * rng.dirichlet(np.ones(len(junc_ids)))
		h_td, _ = solve(d_true)
		M_S = np.array([h_td[s] for s in args.sensors], dtype=float)
		print(f"Running ABC with {args.n_abc} draws ({args.hydraulics} hydraulics, no cache) ...")
		abc_rng = np.random.default_rng(20260703)
		d_samples = np.empty((args.n_abc, len(junc_ids)), dtype=float)
		resid = np.full(args.n_abc, np.inf, dtype=float)
		for i in range(args.n_abc):
			d = d0 + args.delta * abc_rng.dirichlet(np.ones(len(junc_ids)))
			h, ok = solve(d)
			d_samples[i] = d
			if ok:
				resid[i] = float(np.linalg.norm(np.array([h[s] for s in args.sensors]) - M_S))
			if (i + 1) % 5000 == 0:
				print(f"  {i + 1}/{args.n_abc}")
		np.savez_compressed(cache, d_samples=d_samples, resid=resid, d_true=d_true)
		print(f"Cached ABC oracle to {os.path.basename(cache)}")

	h_true, _ = solve(d_true)
	M_S = np.array([h_true[s] for s in args.sensors], dtype=float)

	eps = max(float(np.quantile(resid, args.abc_quantile)), 1e-9)
	w = np.exp(-0.5 * (resid / eps) ** 2)
	w_sum = float(w.sum())
	ess_abc = (w_sum ** 2) / float(np.sum(w * w)) if w_sum > 0 else 0.0
	abc_mean = (w[:, None] * d_samples).sum(axis=0) / w_sum
	abc_var = (w[:, None] * (d_samples - abc_mean[None, :]) ** 2).sum(axis=0) / w_sum
	abc_std = np.sqrt(np.maximum(abc_var, 0.0))
	print(f"\n{args.wdn}: {len(junc_ids)} junctions, sensors={args.sensors}, Delta={args.delta}, D={D:.4f}")
	print(f"ABC oracle: eps={eps:.4g} (q={args.abc_quantile}), effective sample size={ess_abc:.0f}")

	# ---- MCMC samplers on the same measurement (all hard floor => target = ABC oracle)
	run = ns["sample_posterior_scenarios"]
	Cfg = ns["MHPosteriorConfig"]
	meas = {s: h_true[s] for s in args.sensors}
	col = [junc_ids.index(j) for j in junc_ids]  # identity; samples_d is in junction order

	# Match M2's soft-sensor width to the ABC tolerance so the two are directly comparable
	# (both -> exact posterior as eps -> 0).
	m2_eps = args.m2_eps if args.m2_eps > 0 else eps

	variants = {}
	if not args.no_rwm:
		variants["M1 pressure/RWM"] = Cfg(
			method="pressure", proposal="rwm", burn_in=args.burn_in, num_samples=args.samples,
			proposal_std=args.rwm_proposal_std, demand_penalty_a=0.0,
			num_chains=args.rwm_chains, chain_init_dispersion=0.2)
	variants["M1 pressure/ensemble"] = Cfg(
		method="pressure", proposal="ensemble", burn_in=args.burn_in, num_samples=args.samples,
		demand_penalty_a=0.0, ensemble_walkers=args.ens_walkers, ensemble_init_dispersion=args.ens_disp)
	variants["M2 demand/ensemble"] = Cfg(
		method="demand", proposal="ensemble", burn_in=args.burn_in, num_samples=args.samples,
		sensor_noise_eps=m2_eps, ensemble_walkers=args.ens_walkers, ensemble_init_dispersion=0.3)
	variants["M5 demand_exact/ensemble"] = Cfg(
		method="demand_exact", proposal="ensemble", burn_in=args.burn_in, num_samples=args.samples,
		ensemble_walkers=args.ens_walkers, ensemble_init_dispersion=0.01)

	print(f"Dimension (free z): first build to report ...")
	results = {}
	for name, cfg in variants.items():
		print(f"Running MCMC: {name} ...")
		res = run(inp_path=inp, measurement_heads=meas, measured_total_demand=D,
				  predictor_heads=h_true, config=cfg)
		d_s = res.samples_d[:, col]
		gap = np.abs(d_s.mean(axis=0) - abc_mean) / np.maximum(abc_std, 1e-9)
		results[name] = (res, d_s, gap)
		print(f"  n={res.num_chains} acc={res.acceptance_rate:.3f} min_ess={res.min_ess:.0f} "
			  f"med_ess={res.median_ess:.0f} max_rhat={res.max_rhat:.3f} t={res.elapsed_seconds:.0f}s"
			  f"  |  gap vs ABC: max={gap.max():.2f}sigma mean={gap.mean():.2f}sigma")

	print("\nMean gap vs ABC (in ABC-sigma; < ~0.3 = agrees with the oracle):")
	for nm in results:
		g = results[nm][2]
		print(f"  {nm:>20}: max={g.max():.2f}  mean={g.mean():.2f}")

	# ---- overlaid posterior histograms
	try:
		import matplotlib
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt

		names = list(results.keys())
		colors = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
		n = len(junc_ids)
		cols = min(6, n)
		rows = int(np.ceil(n / cols))
		fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.4 * rows))
		axes = np.atleast_1d(axes).ravel()
		for k, j in enumerate(junc_ids):
			ax = axes[k]
			allv = [d_samples[:, k]] + [results[nm][1][:, k] for nm in names]
			bins = np.linspace(min(v.min() for v in allv), max(v.max() for v in allv), 35)
			ax.hist(d_samples[:, k], bins=bins, weights=w, density=True, alpha=0.5,
					color="#1f77b4", label="ABC (oracle)")
			for c, nm in enumerate(names):
				ax.hist(results[nm][1][:, k], bins=bins, density=True, histtype="step",
						color=colors[c % len(colors)], lw=1.6, label=nm)
			ax.axvline(d_true[k], color="k", ls="--", lw=0.8)
			ax.set_title(f"node {j}", fontsize=8)
			ax.tick_params(labelsize=6)
			if k == 0:
				ax.legend(fontsize=6)
		for k in range(n, len(axes)):
			axes[k].axis("off")
		fig.suptitle(f"{args.wdn}: ABC oracle vs MCMC proposals (hard floor), sensors={args.sensors}")
		fig.tight_layout()
		out_png = os.path.join(ROOT_DIR, "scripts", f"abc_reference_{args.wdn}.png")
		fig.savefig(out_png, dpi=110)
		print(f"\nSaved overlay plot to {out_png}")
	except Exception as exc:
		print(f"\n[plot skipped] {exc}")


if __name__ == "__main__":
	main()
