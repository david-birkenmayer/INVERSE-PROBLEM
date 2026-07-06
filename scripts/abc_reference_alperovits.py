#!/usr/bin/env python3
"""
ABC (Approximate Bayesian Computation) reference posterior for a small WDN, used to
validate the MCMC posterior sampler in `mh_posteriori-scenario-gen.py`.

The MCMC samples the posterior over demand scenarios consistent with a measurement
M = (sensor pressures, total demand) in reduced *pressure* coordinates, with a
Newton-eliminated node and an exact change-of-variables (Gram-determinant) factor.

This script computes the SAME posterior the dumb, independent way — entirely in
*demand* space, using only the forward simulator (EPANET) — so it shares none of the
MCMC's machinery (no reduced coordinates, no Newton elimination, no Jacobian). If the
two posteriors agree, that machinery is validated end to end; if they disagree, the
MCMC is either not converged or targeting the wrong law.

Method (rejection/importance ABC):
  1. draw demand shares  alpha ~ Dirichlet(1,...,1)
  2. form a scenario     d = d0 + Delta * alpha        (=> d >= d0 and sum d = D for free)
  3. forward-simulate     h = EPANET(d)                (reservoir head FIXED, as physics demands)
  4. weight by sensor match  w = exp(-||h[S] - M_S||^2 / (2 eps^2))
The weighted d-samples approximate pi(d | M).  Only the sensor-match condition needs
enforcing; the demand floor and total-demand constraints are satisfied by construction.

Usage:
    /home/birkenma/venv_3-11/bin/python3 scripts/abc_reference_alperovits.py
"""
from __future__ import annotations

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
from step2_estimation import simulate_base_scenario  # noqa: E402

# --------------------------------------------------------------------------- config
WDN = "Alperovits"
SENSORS = ["1"]          # measurement (sensor) nodes
DELTA = 0.6              # total extra demand D - D0 (fixed by the total-demand measurement)
TRUE_SEED = 7            # seed for the ground-truth scenario's Dirichlet shares
N_ABC = 25000            # number of ABC prior draws
ABC_KEEP_QUANTILE = 0.02  # tolerance eps set to this quantile of the sensor residuals
MCMC_BURN_IN = 6000
MCMC_SAMPLES = 6000
MCMC_CHAINS = 4          # random-walk chains (for its R-hat)
MCMC_PROPOSAL_STD = 0.5
ENS_WALKERS = 20         # ensemble walkers
ENS_INIT_DISP = 0.02     # small, so walkers start inside the feasible sliver
INP = os.path.join(ROOT_DIR, "wdn", f"{WDN}.inp")
CACHE = os.path.join(ROOT_DIR, "scripts",
	f"abc_cache_{WDN}_{'-'.join(SENSORS)}_D{DELTA}_s{TRUE_SEED}_N{N_ABC}.npz")


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
		return {n: float(head_df.loc[t0, n]) for n in head_df.columns}

	return solve


def main() -> None:
	net = load_inp_network(INP)
	junc_ids = list(net.junctions.keys())
	d0 = np.array([net.junctions[j].base_demand for j in junc_ids], dtype=float)
	D0 = float(d0.sum())
	D = D0 + DELTA

	# ---- ABC oracle: cached (it depends only on network+sensors+Delta+seed+N, NOT on the
	#      MCMC sampler), so it is computed once and reused across sampler variants.
	if os.path.exists(CACHE):
		print(f"Loading cached ABC oracle from {os.path.basename(CACHE)}")
		z = np.load(CACHE)
		d_samples, resid, d_true = z["d_samples"], z["resid"], z["d_true"]
	else:
		solve = _epanet_solver(INP, junc_ids)
		rng = np.random.default_rng(TRUE_SEED)
		d_true = d0 + DELTA * rng.dirichlet(np.ones(len(junc_ids)))
		M_S = np.array([solve(d_true)[s] for s in SENSORS], dtype=float)
		print(f"Running ABC with {N_ABC} draws (no cache) ...")
		abc_rng = np.random.default_rng(20260703)
		d_samples = np.empty((N_ABC, len(junc_ids)), dtype=float)
		resid = np.empty(N_ABC, dtype=float)
		for i in range(N_ABC):
			d = d0 + DELTA * abc_rng.dirichlet(np.ones(len(junc_ids)))
			h = solve(d)
			d_samples[i] = d
			resid[i] = float(np.linalg.norm(np.array([h[s] for s in SENSORS]) - M_S))
			if (i + 1) % 5000 == 0:
				print(f"  {i + 1}/{N_ABC}")
		np.savez_compressed(CACHE, d_samples=d_samples, resid=resid, d_true=d_true)
		print(f"Cached ABC oracle to {os.path.basename(CACHE)}")

	# The measurement M is defined by the true scenario; recover its heads with one solve.
	solve = _epanet_solver(INP, junc_ids)
	h_true = solve(d_true)
	M_S = np.array([h_true[s] for s in SENSORS], dtype=float)
	print(f"Network {WDN}: {len(junc_ids)} junctions, sensors={SENSORS}, Delta={DELTA}, D={D:.4f}")
	print(f"Measurement sensor heads M_S = {dict(zip(SENSORS, np.round(M_S, 4)))}\n")

	# ABC weighting: soft tolerance eps at a low quantile of the sensor residuals.
	eps = max(float(np.quantile(resid, ABC_KEEP_QUANTILE)), 1e-9)
	w = np.exp(-0.5 * (resid / eps) ** 2)
	w_sum = float(w.sum())
	ess_abc = (w_sum ** 2) / float(np.sum(w * w)) if w_sum > 0 else 0.0
	abc_mean = (w[:, None] * d_samples).sum(axis=0) / w_sum
	abc_var = (w[:, None] * (d_samples - abc_mean[None, :]) ** 2).sum(axis=0) / w_sum
	abc_std = np.sqrt(np.maximum(abc_var, 0.0))
	print(f"ABC oracle: eps={eps:.4g} (q={ABC_KEEP_QUANTILE}), effective sample size={ess_abc:.0f}\n")

	# ---- MCMC samplers on the same measurement (reservoir fixed inside the sampler)
	ns = runpy.run_path(os.path.join(ROOT_DIR, "mh_posteriori-scenario-gen.py"))
	run = ns["sample_posterior_scenarios"]
	Cfg = ns["MHPosteriorConfig"]
	meas = {s: h_true[s] for s in SENSORS}
	junc_order = list(load_inp_network(INP).junctions.keys())
	col = [junc_order.index(j) for j in junc_ids]

	def run_mcmc(cfg):
		res = run(inp_path=INP, measurement_heads=meas, measured_total_demand=D,
				  predictor_heads=h_true, config=cfg)  # start at true scenario -> feasible
		return res, res.samples_d[:, col]

	# All variants use the HARD floor (penalty=0) so they target exactly the ABC oracle.
	variants = {
		"RWM (random-walk)": Cfg(proposal="rwm", burn_in=MCMC_BURN_IN, num_samples=MCMC_SAMPLES,
								 proposal_std=MCMC_PROPOSAL_STD, demand_penalty_a=0.0,
								 num_chains=MCMC_CHAINS, chain_init_dispersion=0.2),
		"Ensemble (stretch)": Cfg(proposal="ensemble", burn_in=MCMC_BURN_IN, num_samples=MCMC_SAMPLES,
								  demand_penalty_a=0.0, ensemble_walkers=ENS_WALKERS,
								  ensemble_init_dispersion=ENS_INIT_DISP),
	}
	results = {}
	for name, cfg in variants.items():
		print(f"Running MCMC: {name} ...")
		res, d_s = run_mcmc(cfg)
		gap = np.abs(d_s.mean(axis=0) - abc_mean) / np.maximum(abc_std, 1e-9)
		results[name] = (res, d_s, gap)
		print(f"  acc={res.acceptance_rate:.3f} min_ess={res.min_ess:.0f} med_ess={res.median_ess:.0f} "
			  f"max_rhat={res.max_rhat:.3f}  |  mean-gap vs ABC: max={gap.max():.2f}sigma mean={gap.mean():.2f}sigma")
	print()

	# ---- comparison table: posterior mean +/- std per junction
	names = list(results.keys())
	header = f"{'node':>5} | {'true':>8} | {'ABC (oracle)':>20}"
	for nm in names:
		header += f" | {nm:>20}"
	print("Per-junction posterior demand  (mean +/- std):")
	print(header)
	print("-" * len(header))
	for k, j in enumerate(junc_ids):
		row = f"{j:>5} | {d_true[k]:8.4f} | {abc_mean[k]:9.4f} +/- {abc_std[k]:6.4f}"
		for nm in names:
			d_s = results[nm][1]
			row += f" | {d_s[:, k].mean():9.4f} +/- {d_s[:, k].std():6.4f}"
		print(row)
	print("\nMean gap vs ABC (in ABC-sigma; < ~0.3 = agrees with the oracle):")
	for nm in names:
		g = results[nm][2]
		print(f"  {nm:>20}: max={g.max():.2f}  mean={g.mean():.2f}")

	# ---- overlaid posterior histograms
	try:
		import matplotlib
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt

		colors = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
		n = len(junc_ids)
		cols = 3
		rows = int(np.ceil(n / cols))
		fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
		axes = np.atleast_1d(axes).ravel()
		for k, j in enumerate(junc_ids):
			ax = axes[k]
			allmin = [d_samples[:, k].min()] + [results[nm][1][:, k].min() for nm in names]
			allmax = [d_samples[:, k].max()] + [results[nm][1][:, k].max() for nm in names]
			bins = np.linspace(min(allmin), max(allmax), 40)
			ax.hist(d_samples[:, k], bins=bins, weights=w, density=True, alpha=0.5,
					color="#1f77b4", label="ABC (oracle)")
			for c, nm in enumerate(names):
				ax.hist(results[nm][1][:, k], bins=bins, density=True, histtype="step",
						color=colors[c % len(colors)], lw=1.8, label=nm)
			ax.axvline(d_true[k], color="k", ls="--", lw=1.0)
			ax.axvline(d0[k], color="gray", ls=":", lw=1.0)
			ax.set_title(f"node {j}")
			if k == 0:
				ax.legend(fontsize=7)
		for k in range(n, len(axes)):
			axes[k].axis("off")
		fig.suptitle(f"{WDN}: ABC oracle vs MCMC proposals (hard floor)  (dashed=true, dotted=base)")
		fig.tight_layout()
		out_png = os.path.join(ROOT_DIR, "scripts", "abc_reference_alperovits.png")
		fig.savefig(out_png, dpi=110)
		print(f"\nSaved overlay plot to {out_png}")
	except Exception as exc:
		print(f"\n[plot skipped] {exc}")


if __name__ == "__main__":
	main()
