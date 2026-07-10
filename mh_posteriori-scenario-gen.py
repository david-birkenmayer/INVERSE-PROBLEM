from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import ast
import math
import time

import numpy as np

from step1_io import (
	load_inp_network,
	compute_pipe_resistances,
	compute_pipe_resistances_hw,
)


@dataclass
class MHPosteriorConfig:
	burn_in: int = 2000
	num_samples: int = 4000
	thin: int = 1
	proposal_std: float = 0.15
	adapt_proposal: bool = True
	adapt_interval: int = 50
	adapt_target_accept: float = 0.234
	adapt_gain: float = 0.8
	adapt_until_fraction: float = 0.8
	max_newton_iter: int = 12
	newton_tol: float = 1e-9
	newton_min_abs_derivative: float = 1e-12
	newton_max_step: float = 10.0
	demand_lb_tolerance: float = 1e-9
	simplex_eps: float = 1e-14
	dh_smoothing_eps: float = 1e-9
	use_rank1_det_lemma: bool = True
	use_square_reduced_jacobian: bool = True
	rng_seed: Optional[int] = 42
	# Convergence diagnostics: run several independent chains from dispersed starts so a
	# split-R-hat (Gelman-Rubin) statistic can be computed.  num_chains=1 keeps the original
	# single-chain behaviour.  chain_init_dispersion scales the Gaussian spread of the extra
	# chain starts around the predictor init (in the same units as the reduced pressures z).
	num_chains: int = 1
	chain_init_dispersion: float = 1.0
	# Proposal mechanism: "rwm" = random-walk Metropolis (isotropic Gaussian, the default),
	# "ensemble" = affine-invariant ensemble sampler (Goodman-Weare stretch move).  The
	# ensemble is gradient-free (reuses the same target evaluation) and adapts to thin,
	# linearly-correlated feasible regions where random-walk stalls.
	proposal: str = "rwm"
	ensemble_walkers: int = 0          # 0 => auto = max(2*dim + 2, 8), rounded up to even
	ensemble_stretch_a: float = 2.0    # stretch scale a > 1 (larger = bolder proposals)
	ensemble_init_dispersion: float = 0.3  # Gaussian spread of walker starts around predictor init
	# Sampling coordinate / method:
	#   "pressure" (M1) = reduced pressure coordinates, demands reconstructed, hard sensor +
	#                     total-demand elimination + Gram-Jacobian (the original construction).
	#   "demand"   (M2) = demand-share coordinates (softmax of a free vector on the simplex),
	#                     pressures forward-solved, Dirichlet prior evaluated natively, sensors
	#                     imposed softly by a Gaussian likelihood of width sensor_noise_eps.
	#                     Well-conditioned on low-flow networks; total demand + floor are exact
	#                     by construction; ε -> 0 reproduces the exact (ABC) posterior.
	method: str = "pressure"
	# Demand model / reference floor (applies to the demand methods M2, M5):
	#   "base" = demands are base + Dirichlet split of the *extra* demand D-D0, so d >= d_base
	#            (the thesis "base + extra" model).  On low-flow networks with small extra this
	#            squeezes demands against the floor and makes the feasible region thin.
	#   "zero" = demands are a Dirichlet split of the *total* demand D, so d >= 0 only (the
	#            physical constraint).  Feasible region is the full simplex -> far easier to
	#            sample; the base demands enter only as the Dirichlet concentration if desired.
	demand_reference: str = "base"
	# Gaussian-prior demand methods:
	#   method="gaussian"     (M3) = Gaussian prior N(d_base, diag((prior_sigma*scale)^2)) on the
	#                                raw demands, hard total (Sigma d = D via a linear slack),
	#                                soft sensor likelihood, forward-solved pressures.  Sampled
	#                                by MCMC (the full non-linearised posterior).
	#   method="gaussian_map" (MAP) = same model, but returns the posterior mode via Gauss-Newton
	#                                (scipy least_squares) plus a Laplace-Gaussian sample cloud.
	# prior_sigma is the relative prior std: sigma_j = prior_sigma * max(d_base_j, mean demand).
	prior_sigma: float = 0.5
	sensor_noise_eps: float = 0.05     # sensor measurement-noise std (m of head), M2 soft likelihood
	forward_max_iter: int = 40
	forward_tol: float = 1e-10
	forward_max_step: float = 50.0
	# Soft demand-floor penalty (model 2a).  When > 0, demands below d_base are
	# allowed but penalised by  -(a / target_extra) * Σ_j max(0, d_base_j - d_j)².
	# Set to 0.0 to keep the original hard floor (backward-compatible default).
	demand_penalty_a: float = 0.0


@dataclass
class MHSamplingResult:
	samples_z: np.ndarray
	samples_h: np.ndarray
	samples_d: np.ndarray
	log_targets: np.ndarray
	acceptance_rate: float
	infeasible_rate: float
	punished_rate: float
	proposal_std_final: float
	ess_per_dimension: np.ndarray
	min_ess: float
	median_ess: float
	elapsed_seconds: float
	min_ess_per_sec: float
	median_ess_per_sec: float
	rhat_per_dimension: np.ndarray
	max_rhat: float
	num_chains: int
	# Mean agreement: spread of the per-chain DEMAND means as a fraction of the pooled
	# posterior std, per junction.  Small (<~0.2) => the posterior mean (the demand estimate)
	# is trustworthy even when R-hat is elevated, because the chains agree on where the mass
	# is centred; large => chains disagree on location and the mean itself is unreliable.
	mean_disagreement_per_dim: np.ndarray
	max_mean_disagreement: float
	diagnostics: Dict[str, float]


@dataclass
class _StateEval:
	feasible: bool
	z: np.ndarray
	hv: float
	h: np.ndarray
	d: np.ndarray
	log_target: float
	log_dirichlet: float
	log_jacobian: float
	g_residual: float
	# Generic (method-agnostic) state vector and warm-start payload, filled by _eval so the
	# proposal loops work for both the pressure ("z", warm=hv) and demand ("y", warm=h) methods.
	x: Optional[np.ndarray] = None
	warm: object = None


class PosteriorScenarioSampler:
	"""Metropolis-Hastings posterior sampler in reduced pressure coordinates.

	Notes
	-----
	- The state is the reduced pressure vector z over free, unobserved nodes.
	- One elimination node v is recovered via Newton from the total-demand equation.
	- The target is evaluated in log form: Dirichlet term + Jacobian-volume term.
	- Determinant speedup: optional matrix-determinant-lemma path for rank-1 updates.
	"""

	def __init__(
		self,
		inp_path: str,
		measurement_heads: Dict[str, float],
		measured_total_demand: float,
		predictor_heads: Dict[str, float],
		elimination_node: Optional[str] = None,
		drop_demand_row_node: Optional[str] = None,
		dirichlet_alpha: Optional[Dict[str, float]] = None,
		config: Optional[MHPosteriorConfig] = None,
	) -> None:
		self.inp_path = inp_path
		self.network = load_inp_network(inp_path)
		self.cfg = config or MHPosteriorConfig()
		self.rng = np.random.default_rng(self.cfg.rng_seed)

		self.headloss_n = self._infer_headloss_exponent(self.network.options)
		self.resistances = self._compute_pipe_resistances()

		self.node_ids: List[str] = list(self.network.nodes.keys())
		self.node_idx: Dict[str, int] = {nid: i for i, nid in enumerate(self.node_ids)}
		self.n_nodes = len(self.node_ids)

		self.junction_ids: List[str] = list(self.network.junctions.keys())
		self.junction_idx: Dict[str, int] = {jid: i for i, jid in enumerate(self.junction_ids)}
		self.n_junctions = len(self.junction_ids)

		self.base_demands = np.array(
			[float(self.network.junctions[j].base_demand) for j in self.junction_ids],
			dtype=float,
		)
		self.base_total = float(self.base_demands.sum())
		self.measured_total_demand = float(measured_total_demand)
		self.target_extra = float(self.measured_total_demand - self.base_total)

		# Demand model: d = demand_offset + demand_scale * alpha, with alpha ~ Dirichlet on the
		# simplex (sum alpha = 1).  "base" reproduces the thesis d >= d_base model; "zero" uses
		# the physical d >= 0 with demands a split of the total D (fat feasible region).
		if str(self.cfg.demand_reference).lower() == "zero":
			self.demand_offset = np.zeros(self.n_junctions, dtype=float)
			self.demand_scale = float(self.measured_total_demand)
			self.demand_floor = np.zeros(self.n_junctions, dtype=float)
		else:
			self.demand_offset = self.base_demands.copy()
			self.demand_scale = float(self.target_extra)
			self.demand_floor = self.base_demands.copy()

		self.fixed_heads: Dict[str, float] = {str(k): float(v) for k, v in measurement_heads.items()}
		self.predictor_heads: Dict[str, float] = {
			str(k): float(v) for k, v in predictor_heads.items()
		}

		# Sensor set = the true measurement nodes (before reservoirs are pinned below).  The
		# demand method (M2) uses these as a soft Gaussian likelihood rather than fixing them.
		self.sensor_ids: List[str] = [str(k) for k in measurement_heads.keys()]
		self.sensor_node_idxs = np.array(
			[self.node_idx[s] for s in self.sensor_ids if s in self.node_idx], dtype=int
		)
		self.sensor_targets = np.array(
			[float(measurement_heads[s]) for s in self.sensor_ids if s in self.node_idx], dtype=float
		)

		# Reservoir heads are known boundary conditions, not latent coordinates.  Pinning
		# them (rather than sampling them) is essential: a free reservoir head is a spurious,
		# weakly-identified dimension that destroys chain mixing.  Use the instance's known
		# head (predictor/measurement) when available, else the .inp base head.
		for rid in self.network.reservoirs:
			rid = str(rid)
			if rid not in self.fixed_heads:
				res_head = self.predictor_heads.get(rid)
				if res_head is None:
					res_head = float(self.network.reservoirs[rid].elevation_m)
				self.fixed_heads[rid] = float(res_head)

		self.unobserved_nodes = [nid for nid in self.node_ids if nid not in self.fixed_heads]
		if not self.unobserved_nodes:
			raise ValueError("All node pressures are fixed; no latent state for MH.")

		if elimination_node is None:
			elimination_node = self._choose_elimination_node()
		if elimination_node not in self.unobserved_nodes:
			raise ValueError("elimination_node must be an unobserved node.")
		self.elimination_node = elimination_node
		self.elim_idx = self.node_idx[self.elimination_node]

		self.free_nodes = [nid for nid in self.unobserved_nodes if nid != self.elimination_node]
		self.free_idxs = np.array([self.node_idx[nid] for nid in self.free_nodes], dtype=int)
		self.dim = len(self.free_nodes)
		if self.dim <= 0:
			raise ValueError("No free dimensions remain after elimination. Need at least 1 free node.")

		if drop_demand_row_node is None:
			if self.elimination_node in self.junction_idx:
				drop_demand_row_node = self.elimination_node
			else:
				drop_demand_row_node = self.junction_ids[0]
		if drop_demand_row_node not in self.junction_idx:
			raise ValueError("drop_demand_row_node must be a junction node.")
		self.drop_demand_row_node = drop_demand_row_node
		self.drop_demand_row_idx = self.junction_idx[self.drop_demand_row_node]

		# Dirichlet concentration parameters for the demand-share simplex.
		if dirichlet_alpha is None:
			self.alpha = np.ones(self.n_junctions, dtype=float)
		else:
			self.alpha = np.array(
				[float(dirichlet_alpha.get(jid, 1.0)) for jid in self.junction_ids],
				dtype=float,
			)
			if np.any(self.alpha <= 0.0):
				raise ValueError("All Dirichlet alpha parameters must be positive.")
		self.log_dirichlet_norm = float(math.lgamma(float(self.alpha.sum())) - np.sum([math.lgamma(float(a)) for a in self.alpha]))

		self.pipe_data = self._build_pipe_arrays()
		self.initial_z, self.initial_hv = self._initial_state_from_predictor()

		# --- Demand-space (M2) setup -------------------------------------------------------
		# State is a free vector y in R^(n_junctions-1); shares alpha = softmax([y, 0]);
		# demands d = base + target_extra * alpha (floor and total exact by construction);
		# pressures are forward-solved and sensors imposed by a Gaussian likelihood.
		self.junction_node_idxs = np.array(
			[self.node_idx[j] for j in self.junction_ids], dtype=int
		)
		# Full-node head vector holding the fixed (reservoir) heads; free junction heads are
		# overwritten by the forward solve.
		self._base_head_vec = np.array(
			[float(self.predictor_heads.get(nid, self.fixed_heads.get(nid, 0.0))) for nid in self.node_ids],
			dtype=float,
		)
		for nid, val in self.fixed_heads.items():
			if nid in self.node_idx:
				self._base_head_vec[self.node_idx[nid]] = float(val)

		method = str(self.cfg.method).lower()
		if method == "demand":
			self.dim = self.n_junctions - 1
			if self.dim <= 0:
				raise ValueError("Demand method needs at least 2 junctions.")
			self.initial_x = np.zeros(self.dim, dtype=float)  # uniform shares
			self.initial_warm = self._base_head_vec.copy()
		elif method == "demand_exact":
			self._setup_demand_exact()
		elif method in ("gaussian", "gaussian_map"):
			self._setup_gaussian()
		else:
			self.initial_x = self.initial_z
			self.initial_warm = self.initial_hv

	def _setup_gaussian(self) -> None:
		"""Gaussian-prior demand model (M3 / MAP).

		Raw demands d in R^n with a Gaussian prior N(mu, Sigma), mu = base demands and
		Sigma = diag(sigma_j^2), sigma_j = prior_sigma * max(d_base_j, mean demand).  The total
		demand is imposed hard by a linear slack: the free coordinates are the demands at all
		junctions except one 'slack' junction, whose demand closes Sigma d = D.  Pressures are
		forward-solved and the sensor pressures enter through a Gaussian likelihood; d >= 0 is a
		soft (rarely-binding) physical constraint.
		"""
		mean_scale = float(self.base_total) / float(self.n_junctions) if self.n_junctions else 1.0
		self.gauss_mean = self.base_demands.copy()
		sig = float(self.cfg.prior_sigma) * np.maximum(self.base_demands, mean_scale)
		self.gauss_sigma = np.maximum(sig, 1e-9)
		# Slack junction = largest base demand (least likely to be driven negative).
		self.gauss_slack_jrow = int(np.argmax(self.base_demands))
		self.gauss_free_jrows = np.array(
			[j for j in range(self.n_junctions) if j != self.gauss_slack_jrow], dtype=int
		)
		self.dim = self.n_junctions - 1
		if self.dim <= 0:
			raise ValueError("Gaussian method needs at least 2 junctions.")
		# Initial demands: base scaled to the measured total (positive, sums to D).
		if self.base_total > 0:
			d0 = self.base_demands * (self.measured_total_demand / self.base_total)
		else:
			d0 = np.full(self.n_junctions, self.measured_total_demand / self.n_junctions)
		self.initial_x = d0[self.gauss_free_jrows].copy()
		self.initial_warm = self._base_head_vec.copy()

	def _gauss_full_demand(self, x: np.ndarray) -> np.ndarray:
		"""Map free demands x -> full demand vector, closing Sigma d = D at the slack junction."""
		d = np.empty(self.n_junctions, dtype=float)
		d[self.gauss_free_jrows] = x
		d[self.gauss_slack_jrow] = self.measured_total_demand - float(np.sum(x))
		return d

	def _reservoir_adjacent_junctions(self) -> List[str]:
		"""Junction ids that share a pipe with a reservoir (needed to close total demand)."""
		res = set(str(r) for r in self.network.reservoirs.keys())
		out: List[str] = []
		seen: set = set()
		for pipe in self.network.pipes.values():
			s, t = str(pipe.start_node), str(pipe.end_node)
			for a, b in [(s, t), (t, s)]:
				if a in res and b in self.junction_idx and b not in seen:
					out.append(b)
					seen.add(b)
		return out

	def _setup_demand_exact(self) -> None:
		"""Exact (ε=0) demand-space elimination.

		Free coordinates are the demands at the free junctions; the demands at the sensor
		junctions and one 'slack' junction are recovered by a mixed-boundary hydraulic solve
		that imposes the sensor *pressures* exactly (h_S = M_S) and closes the total demand.
		Because sensors are eliminated (not softened), adding sensors *reduces* the free
		dimension — the opposite of the soft (ε>0) method.
		"""
		self.exact_sensor_juncs = [s for s in self.sensor_ids if s in self.junction_idx]
		if not self.exact_sensor_juncs:
			raise ValueError("demand_exact needs at least one sensor on a junction.")

		res_adj = [j for j in self._reservoir_adjacent_junctions() if j not in set(self.exact_sensor_juncs)]
		# Case (a): a reservoir-adjacent junction is free -> use it as the slack node that
		#           closes the total-demand constraint.
		# Case (b): every reservoir-adjacent junction is a sensor -> the fixed sensor pressure
		#           already pins the reservoir outflow, so total demand is implied; no slack.
		if res_adj:
			self.exact_total_implied = False
			self.exact_slack_junc = self._choose_central_node(res_adj) if len(res_adj) > 1 else res_adj[0]
			dependent = set(self.exact_sensor_juncs) | {self.exact_slack_junc}
		else:
			self.exact_total_implied = True
			self.exact_slack_junc = None
			dependent = set(self.exact_sensor_juncs)

		self.exact_dep_juncs = [j for j in self.junction_ids if j in dependent]
		self.exact_free_juncs = [j for j in self.junction_ids if j not in dependent]
		self.exact_free_jrows = np.array([self.junction_idx[j] for j in self.exact_free_juncs], dtype=int)
		self.exact_dep_jrows = np.array([self.junction_idx[j] for j in self.exact_dep_juncs], dtype=int)
		# Non-sensor junction pressures are the unknowns of the mixed-BC solve.
		self.exact_unknown_juncs = [j for j in self.junction_ids if j not in set(self.exact_sensor_juncs)]
		self.exact_unknown_ncols = np.array([self.node_idx[j] for j in self.exact_unknown_juncs], dtype=int)
		self.exact_sensor_ncols = np.array([self.node_idx[s] for s in self.exact_sensor_juncs], dtype=int)
		self.exact_sensor_targets = np.array([float(self.fixed_heads.get(s, self.predictor_heads.get(s, 0.0)))
											  for s in self.exact_sensor_juncs], dtype=float)

		self.dim = len(self.exact_free_juncs)
		if self.dim <= 0:
			raise ValueError("demand_exact: no free demands remain (too many sensors).")

		# Initial free demands: uniform Dirichlet share over all junctions (interior/feasible).
		d0_full = self.demand_offset + self.demand_scale / float(self.n_junctions)
		self.initial_x = d0_full[self.exact_free_jrows].copy()
		# Seed the mixed-BC solve with sensor pressures fixed at their targets.
		h0 = self._base_head_vec.copy()
		h0[self.exact_sensor_ncols] = self.exact_sensor_targets
		self.initial_warm = h0

	@staticmethod
	def _infer_headloss_exponent(options: Dict[str, str]) -> float:
		try:
			raw = options.get("hydraulic", "")
			opts = ast.literal_eval(raw) if isinstance(raw, str) else raw
			model = str(opts.get("headloss", "")).upper()
			return 1.852 if "H" in model else 2.0
		except Exception:
			return 1.852

	def _compute_pipe_resistances(self) -> Dict[str, float]:
		if abs(self.headloss_n - 1.852) < 1e-3:
			rs = compute_pipe_resistances_hw(self.network)
		else:
			rs = compute_pipe_resistances(self.network)
		return {pid: float(payload["r_e"]) for pid, payload in rs.items()}

	def _build_pipe_arrays(self) -> Dict[str, np.ndarray]:
		pids = list(self.network.pipes.keys())
		start = np.array([self.node_idx[self.network.pipes[pid].start_node] for pid in pids], dtype=int)
		end = np.array([self.node_idx[self.network.pipes[pid].end_node] for pid in pids], dtype=int)
		resist = np.array([self.resistances.get(pid, np.nan) for pid in pids], dtype=float)
		if np.any(~np.isfinite(resist)) or np.any(resist <= 0.0):
			raise ValueError("Invalid pipe resistance detected while building sampler.")
		return {
			"pipe_ids": np.array(pids, dtype=object),
			"start": start,
			"end": end,
			"resist": resist,
		}

	def _choose_central_node(self, candidates: List[str]) -> str:
		# Lightweight graph-centrality proxy: highest undirected degree among candidates.
		degree = {nid: 0 for nid in self.node_ids}
		for pipe in self.network.pipes.values():
			degree[pipe.start_node] = degree.get(pipe.start_node, 0) + 1
			degree[pipe.end_node] = degree.get(pipe.end_node, 0) + 1
		return max(candidates, key=lambda n: (degree.get(n, 0), -self.node_idx[n]))

	def _choose_elimination_node(self) -> str:
		"""Choose which node is eliminated via Newton, ensuring dg_dhv != 0.

		dg_dhv = sum of the Jacobian column for the elimination node. For junction j,
		this sum is non-zero only if j is directly connected to a reservoir (each such
		pipe contributes -se to the column sum; all junction-to-junction pipes cancel).
		For a reservoir node r, the column sum equals the sensitivity of its outflow pipe(s).

		When all reservoir neighbors are sensors (fixed heads), no junction has dg_dhv != 0.
		In that case the reservoir itself is used as the elimination node so Newton can
		recover the reservoir head from the total-demand constraint.
		"""
		# Build the set of junctions that neighbour any non-fixed reservoir.
		free_reservoir_neighbors: List[str] = []
		for pipe in self.network.pipes.values():
			s, t = pipe.start_node, pipe.end_node
			for res_id, nbr_id in [(s, t), (t, s)]:
				if res_id in self.network.reservoirs and res_id not in self.fixed_heads:
					if nbr_id not in self.fixed_heads and nbr_id in self.junction_idx:
						free_reservoir_neighbors.append(nbr_id)

		# Prefer: a free junction directly connected to a reservoir (strong dg_dhv).
		candidates = [n for n in free_reservoir_neighbors if n in self.unobserved_nodes]
		if candidates:
			return self._choose_central_node(candidates)

		# Fallback A: use the reservoir itself as the elimination node.
		# dg_dhv = sum(J[:, res_col]) = se of the reservoir's pipe(s), which is non-zero.
		for rid in self.network.reservoirs:
			rid_str = str(rid)
			if rid_str not in self.fixed_heads and rid_str in self.unobserved_nodes:
				return rid_str

		# Fallback B: standard highest-degree junction (may have dg_dhv = 0 for some topologies).
		return self._choose_central_node(self.unobserved_nodes)

	def _initial_state_from_predictor(self) -> Tuple[np.ndarray, float]:
		# Fall back to fixed-head average if predictor does not include a node.
		if self.fixed_heads:
			avg_head = float(sum(self.fixed_heads.values()) / len(self.fixed_heads))
		else:
			avg_head = 0.0
		z0 = np.array(
			[float(self.predictor_heads.get(nid, avg_head)) for nid in self.free_nodes],
			dtype=float,
		)
		hv0 = float(self.predictor_heads.get(self.elimination_node, avg_head))
		return z0, hv0

	def _assemble_heads(self, z: np.ndarray, hv: float) -> np.ndarray:
		h = np.zeros(self.n_nodes, dtype=float)
		for nid, value in self.fixed_heads.items():
			if nid in self.node_idx:
				h[self.node_idx[nid]] = float(value)
		h[self.free_idxs] = z
		h[self.elim_idx] = float(hv)
		return h

	def _flows_and_sensitivity(self, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		start = self.pipe_data["start"]
		end = self.pipe_data["end"]
		resist = self.pipe_data["resist"]
		dh = h[start] - h[end]

		inv_n = 1.0 / float(self.headloss_n)
		abs_dh = np.abs(dh)
		safe_abs_dh = np.maximum(abs_dh, float(self.cfg.dh_smoothing_eps))
		base = np.power(safe_abs_dh / resist, inv_n)
		q = np.sign(dh) * base

		# dq / d(dh): monotone odd-power inverse with smoothing near zero.
		sens = inv_n * np.power(1.0 / resist, inv_n) * np.power(safe_abs_dh, inv_n - 1.0)
		return q, sens

	def _demands_and_jacobian(self, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		q, sens = self._flows_and_sensitivity(h)
		d = np.zeros(self.n_junctions, dtype=float)
		J = np.zeros((self.n_junctions, self.n_nodes), dtype=float)

		start = self.pipe_data["start"]
		end = self.pipe_data["end"]
		for e in range(len(start)):
			s_idx = int(start[e])
			t_idx = int(end[e])
			s_name = self.node_ids[s_idx]
			t_name = self.node_ids[t_idx]
			qe = float(q[e])
			se = float(sens[e])

			if s_name in self.junction_idx:
				r = self.junction_idx[s_name]
				d[r] -= qe
				J[r, s_idx] -= se
				J[r, t_idx] += se
			if t_name in self.junction_idx:
				r = self.junction_idx[t_name]
				d[r] += qe
				J[r, s_idx] += se
				J[r, t_idx] -= se
		return d, J

	def _g_eval(self, z: np.ndarray, hv: float) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
		h = self._assemble_heads(z, hv)
		d, J = self._demands_and_jacobian(h)
		g = float(np.sum(d) - self.measured_total_demand)
		dg_dhv = float(np.sum(J[:, self.elim_idx]))
		dg_dz = np.sum(J[:, self.free_idxs], axis=0)
		return g, dg_dhv, dg_dz, d, J

	def _recover_hv_newton(self, z: np.ndarray, hv0: float) -> Tuple[bool, float, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
		hv = float(hv0)
		g = np.nan
		dg_dhv = np.nan
		dg_dz = np.zeros(self.dim, dtype=float)
		d = np.zeros(self.n_junctions, dtype=float)
		J = np.zeros((self.n_junctions, self.n_nodes), dtype=float)

		for _ in range(self.cfg.max_newton_iter):
			g, dg_dhv, dg_dz, d, J = self._g_eval(z, hv)
			if abs(g) <= self.cfg.newton_tol:
				return True, hv, d, J, dg_dz, g, self._assemble_heads(z, hv)
			if abs(dg_dhv) < self.cfg.newton_min_abs_derivative:
				return False, hv, d, J, dg_dz, g, self._assemble_heads(z, hv)
			step = -g / dg_dhv
			step = float(np.clip(step, -self.cfg.newton_max_step, self.cfg.newton_max_step))
			hv += step

		g, dg_dhv, dg_dz, d, J = self._g_eval(z, hv)
		ok = abs(g) <= max(self.cfg.newton_tol * 10.0, 1e-8)
		return ok, hv, d, J, dg_dz, g, self._assemble_heads(z, hv)

	def _dirichlet_logpdf(self, d: np.ndarray) -> float:
		extra = d - self.base_demands
		soft_mode = self.cfg.demand_penalty_a > 0.0

		if np.any(extra < -self.cfg.demand_lb_tolerance):
			if not soft_mode:
				return -np.inf
			# In the degenerate case (target_extra ≈ 0) no Dirichlet applies: return flat 0
			# so the contribution is the same for all states, consistent with the above-base path.
			_extra_eps = self.cfg.simplex_eps * max(1.0, abs(self.base_total))
			if self.target_extra <= _extra_eps:
				return 0.0
			# Non-degenerate soft mode: return the normalising constant so the Dirichlet
			# contribution is continuous at the boundary.  For alpha=1 (uniform) the density
			# equals log_norm everywhere on the interior simplex, so this is exact.
			return float(self.log_dirichlet_norm)

		total_extra = float(np.sum(extra))
		# Degenerate case: target_extra is zero or negligibly small relative to base total.
		# A plain absolute threshold (simplex_eps = 1e-14) misses cases where Python sum()
		# and numpy .sum() give different float results (off by ~1 ulp ≈ 1e-16 for values
		# around 1), leaving target_extra slightly positive and breaking the open-simplex
		# share check below for junctions whose demand equals their base demand exactly.
		if self.target_extra <= self.cfg.simplex_eps * max(1.0, abs(self.base_total)):
			return 0.0 if abs(total_extra) <= 1e-8 else -np.inf
		if abs(total_extra - self.target_extra) > 1e-6:
			return -np.inf

		share = extra / self.target_extra
		if np.any(share <= self.cfg.simplex_eps):
			if not soft_mode:
				return -np.inf
			# Soft mode: clamp near-zero shares and renormalise so log() is well-defined.
			# For the default α=1 the (α-1)·log(share) terms are all zero, so this is exact.
			share = np.maximum(share, self.cfg.simplex_eps)
			share = share / float(np.sum(share))
		if abs(float(np.sum(share)) - 1.0) > 1e-6:
			return -np.inf

		return float(self.log_dirichlet_norm + np.sum((self.alpha - 1.0) * np.log(share)))

	def _log_abs_det_square_reduced(self, J: np.ndarray, dg_dz: np.ndarray, dg_dhv: float) -> float:
		if abs(dg_dhv) < self.cfg.newton_min_abs_derivative:
			return -np.inf

		ratio = dg_dz / dg_dhv
		J_F = J[:, self.free_idxs]
		J_v = J[:, self.elim_idx]

		mask = np.ones(self.n_junctions, dtype=bool)
		mask[self.drop_demand_row_idx] = False
		A = J_F[mask, :]
		u = J_v[mask]

		if A.shape[0] != A.shape[1]:
			# Rectangular case (dim < n_junctions - 1): use Gram determinant of M.
			# Note: the full J_red (without dropping a row) has a zero Gram determinant
			# because the total-demand constraint makes its rows linearly dependent.
			# Dropping the elimination row breaks that dependence, so M^T M is valid.
			M = A - np.outer(u, ratio)
			G = M.T @ M
			sign_G, logabs_G = np.linalg.slogdet(G)
			if sign_G <= 0:
				return -np.inf
			return float(0.5 * logabs_G)

		# Fast path: determinant lemma on rank-1 correction.
		if self.cfg.use_rank1_det_lemma:
			sign_A, logabs_A = np.linalg.slogdet(A)
			if sign_A == 0:
				return -np.inf
			try:
				x = np.linalg.solve(A, u)
			except np.linalg.LinAlgError:
				return -np.inf
			corr = float(1.0 - np.dot(ratio, x))
			if abs(corr) <= 1e-18:
				return -np.inf
			return float(logabs_A + math.log(abs(corr)))

		M = A - np.outer(u, ratio)
		sign, logabs = np.linalg.slogdet(M)
		if sign == 0:
			return -np.inf
		return float(logabs)

	def _log_jacobian_volume(self, J: np.ndarray, dg_dz: np.ndarray, dg_dhv: float) -> float:
		if self.cfg.use_square_reduced_jacobian:
			return self._log_abs_det_square_reduced(J, dg_dz, dg_dhv)

		if abs(dg_dhv) < self.cfg.newton_min_abs_derivative:
			return -np.inf
		J_F = J[:, self.free_idxs]
		J_v = J[:, self.elim_idx]
		J_red = J_F - np.outer(J_v, dg_dz / dg_dhv)
		G = J_red.T @ J_red
		sign, logabs = np.linalg.slogdet(G)
		if sign <= 0:
			return -np.inf
		return float(0.5 * logabs)

	def _evaluate_state(self, z: np.ndarray, hv_guess: float) -> _StateEval:
		ok, hv, d, J, dg_dz, g, h = self._recover_hv_newton(z, hv_guess)
		if not ok:
			return _StateEval(False, z, hv, h, d, -np.inf, -np.inf, -np.inf, g)

		soft_mode = self.cfg.demand_penalty_a > 0.0
		if not soft_mode and np.any(d < (self.base_demands - self.cfg.demand_lb_tolerance)):
			return _StateEval(False, z, hv, h, d, -np.inf, -np.inf, -np.inf, g)

		log_dir = self._dirichlet_logpdf(d)
		if not np.isfinite(log_dir):
			return _StateEval(False, z, hv, h, d, -np.inf, log_dir, -np.inf, g)

		dg_dhv = float(np.sum(J[:, self.elim_idx]))
		log_jac = self._log_jacobian_volume(J, dg_dz, dg_dhv)
		if not np.isfinite(log_jac):
			return _StateEval(False, z, hv, h, d, -np.inf, log_dir, log_jac, g)

		log_penalty = 0.0
		_extra_eps = self.cfg.simplex_eps * max(1.0, abs(self.base_total))
		if soft_mode and self.target_extra > _extra_eps:
			shortfalls = np.maximum(0.0, self.base_demands - d)
			log_penalty = -(self.cfg.demand_penalty_a / self.target_extra) * float(np.sum(shortfalls ** 2))

		log_target = float(log_dir + log_jac + log_penalty)
		return _StateEval(True, z, hv, h, d, log_target, log_dir, log_jac, g)

	# ---- Demand-space method (M2) --------------------------------------------------------
	def _shares_from_y(self, y: np.ndarray) -> np.ndarray:
		"""Softmax of [y, 0] -> shares alpha on the interior simplex (bijective, unconstrained)."""
		ext = np.concatenate([y, [0.0]])
		ext = ext - np.max(ext)
		e = np.exp(ext)
		return e / np.sum(e)

	def _forward_solve(self, d_target: np.ndarray, h_init: np.ndarray) -> Tuple[bool, np.ndarray]:
		"""Solve the hydraulics for pressures given demands (reservoir heads fixed).

		Newton on the junction heads: find h s.t. F(h) = d_target, where F is the nodal-demand
		map.  This is the well-conditioned forward (convex) direction — unlike reconstructing
		demands from pressures.  Warm-started from h_init.
		"""
		h = h_init.copy()
		jidx = self.junction_node_idxs
		for _ in range(int(self.cfg.forward_max_iter)):
			d_cur, J = self._demands_and_jacobian(h)
			resid = d_cur - d_target
			if float(np.max(np.abs(resid))) <= self.cfg.forward_tol:
				return True, h
			Jjj = J[:, jidx]
			try:
				step = np.linalg.solve(Jjj, -resid)
			except np.linalg.LinAlgError:
				return False, h
			if not np.all(np.isfinite(step)):
				return False, h
			step = np.clip(step, -self.cfg.forward_max_step, self.cfg.forward_max_step)
			h[jidx] += step
		d_cur, _ = self._demands_and_jacobian(h)
		ok = float(np.max(np.abs(d_cur - d_target))) <= max(self.cfg.forward_tol * 100.0, 1e-8)
		return ok, h

	def _evaluate_demand(self, y: np.ndarray, h_warm: np.ndarray) -> _StateEval:
		alpha = self._shares_from_y(y)
		d = self.demand_offset + self.demand_scale * alpha
		h_init = h_warm if (h_warm is not None and np.all(np.isfinite(h_warm))) else self._base_head_vec
		ok, h = self._forward_solve(d, h_init)
		if not ok:
			st = _StateEval(False, y, 0.0, h, d, -np.inf, -np.inf, -np.inf, 0.0)
			st.x = y
			st.warm = h
			return st

		# Prior (Dirichlet on shares) + reparametrisation Jacobian of the softmax transform.
		# For alpha ~ Dirichlet(a), the induced density on y is proportional to prod alpha_j^{a_j},
		# so the combined log term is sum_j a_j * log(alpha_j).
		log_prior = float(np.sum(self.alpha * np.log(np.maximum(alpha, 1e-300))))
		# Soft sensor likelihood (Gaussian measurement noise of width sensor_noise_eps).
		if self.sensor_node_idxs.size:
			resid = h[self.sensor_node_idxs] - self.sensor_targets
			eps = max(float(self.cfg.sensor_noise_eps), 1e-9)
			log_lik = -0.5 * float(np.sum(resid ** 2)) / (eps * eps)
		else:
			log_lik = 0.0

		log_target = log_prior + log_lik
		st = _StateEval(True, y, 0.0, h, d, log_target, log_prior, log_lik, 0.0)
		st.x = y
		st.warm = h
		return st

	# ---- Gaussian-prior demand method (M3) ----------------------------------------------
	def _gauss_log_prior(self, d: np.ndarray) -> float:
		z = (d - self.gauss_mean) / self.gauss_sigma
		return -0.5 * float(np.dot(z, z))

	def _evaluate_gaussian(self, x: np.ndarray, h_warm: np.ndarray) -> _StateEval:
		d = self._gauss_full_demand(x)
		# Soft physical constraint d >= 0 (rarely binds for a base-centred Gaussian).
		if np.any(d < -self.cfg.demand_lb_tolerance):
			st = _StateEval(False, x, 0.0, h_warm if h_warm is not None else self._base_head_vec,
							d, -np.inf, -np.inf, -np.inf, 0.0)
			st.x = x; st.warm = st.h
			return st
		h_init = h_warm if (h_warm is not None and np.all(np.isfinite(h_warm))) else self._base_head_vec
		ok, h = self._forward_solve(d, h_init)
		if not ok:
			st = _StateEval(False, x, 0.0, h, d, -np.inf, -np.inf, -np.inf, 0.0)
			st.x = x; st.warm = h
			return st
		log_prior = self._gauss_log_prior(d)
		if self.sensor_node_idxs.size:
			resid = h[self.sensor_node_idxs] - self.sensor_targets
			eps = max(float(self.cfg.sensor_noise_eps), 1e-9)
			log_lik = -0.5 * float(np.sum(resid ** 2)) / (eps * eps)
		else:
			log_lik = 0.0
		log_target = float(log_prior + log_lik)
		st = _StateEval(True, x, 0.0, h, d, log_target, log_prior, log_lik, 0.0)
		st.x = x; st.warm = h
		return st

	# ---- Exact demand-space elimination (M5, ε=0) ---------------------------------------
	def _mixed_bc_solve(self, theta: np.ndarray, h_init: np.ndarray) -> Tuple[bool, np.ndarray]:
		"""Mixed-boundary hydraulic solve.

		Fix the sensor pressures at their targets (h_S = M_S); solve for the non-sensor
		junction pressures such that the free-junction demands equal `theta` and (unless the
		total is implied by a reservoir-adjacent sensor) the total demand equals D.
		Returns (converged, full head vector).
		"""
		h = h_init.copy()
		h[self.exact_sensor_ncols] = self.exact_sensor_targets  # keep sensors pinned
		ucols = self.exact_unknown_ncols
		for _ in range(int(self.cfg.forward_max_iter)):
			d_cur, J = self._demands_and_jacobian(h)
			r_free = d_cur[self.exact_free_jrows] - theta
			if self.exact_total_implied:
				resid = r_free
				Jsys = J[np.ix_(self.exact_free_jrows, ucols)]
			else:
				r_tot = float(np.sum(d_cur)) - self.measured_total_demand
				resid = np.concatenate([r_free, [r_tot]])
				A = J[np.ix_(self.exact_free_jrows, ucols)]
				b = np.sum(J[:, ucols], axis=0)[None, :]
				Jsys = np.concatenate([A, b], axis=0)
			if float(np.max(np.abs(resid))) <= self.cfg.forward_tol:
				return True, h
			try:
				step = np.linalg.solve(Jsys, -resid)
			except np.linalg.LinAlgError:
				return False, h
			if not np.all(np.isfinite(step)):
				return False, h
			step = np.clip(step, -self.cfg.forward_max_step, self.cfg.forward_max_step)
			h[ucols] += step
		d_cur, _ = self._demands_and_jacobian(h)
		r_free = d_cur[self.exact_free_jrows] - theta
		ok = float(np.max(np.abs(r_free))) <= max(self.cfg.forward_tol * 100.0, 1e-8)
		return ok, h

	def _exact_dep_from_theta(self, theta: np.ndarray, h_warm: np.ndarray):
		ok, h = self._mixed_bc_solve(theta, h_warm)
		if not ok:
			return False, h, None
		d = self._demands_and_jacobian(h)[0]
		return True, h, d

	def _evaluate_demand_exact(self, theta: np.ndarray, h_warm: np.ndarray) -> _StateEval:
		hw = h_warm if (h_warm is not None and np.all(np.isfinite(h_warm))) else self.initial_warm
		ok, h, d = self._exact_dep_from_theta(theta, hw)
		if not ok or d is None:
			st = _StateEval(False, theta, 0.0, h, d if d is not None else np.zeros(self.n_junctions),
							-np.inf, -np.inf, -np.inf, 0.0)
			st.x = theta; st.warm = h
			return st

		# Feasibility: every demand at or above its floor (d >= d_base for "base", d >= 0 for "zero").
		if np.any(d < self.demand_floor - self.cfg.demand_lb_tolerance):
			st = _StateEval(False, theta, 0.0, h, d, -np.inf, -np.inf, -np.inf, 0.0)
			st.x = theta; st.warm = h
			return st

		# Dirichlet prior on the reconstructed shares alpha = (d - offset) / scale.
		if self.demand_scale <= self.cfg.simplex_eps * max(1.0, abs(self.measured_total_demand)):
			log_dir = 0.0
		else:
			alpha = (d - self.demand_offset) / self.demand_scale
			log_dir = float(self.log_dirichlet_norm + np.sum((self.alpha - 1.0) * np.log(np.maximum(alpha, 1e-300))))

		# Change-of-variables (Gram) factor for the map free-demands theta -> full demand
		# vector on the constraint manifold: d_full = [ theta (free) ; d_dep(theta) ], so the
		# Gram matrix is G = I + (d_dep/d_theta)^T (d_dep/d_theta).
		#
		# Analytic d_dep/d_theta via the implicit function theorem on the mixed-BC solve.
		# Unknowns are the non-sensor junction pressures h_u.  Residuals:
		#   free-demand rows:  F_free(h) - theta = 0    (Jacobian A = J[free, u])
		#   total row (case a): sum_j F_j(h) - D = 0    (Jacobian b = colsum J[:, u])
		# Differentiating at the solution:  M (dh_u/dtheta) = [I_m ; 0],  M = [A ; b].
		# Then d_dep/d_theta = J[dep, u] @ (dh_u/dtheta).
		m = self.dim
		_, J = self._demands_and_jacobian(h)
		ucols = self.exact_unknown_ncols
		A = J[np.ix_(self.exact_free_jrows, ucols)]
		C = J[np.ix_(self.exact_dep_jrows, ucols)]
		try:
			if self.exact_total_implied:
				dh_dtheta = np.linalg.solve(A, np.eye(m))
			else:
				b = np.sum(J[:, ucols], axis=0)[None, :]
				M = np.concatenate([A, b], axis=0)              # (m+1) x (m+1)
				rhs = np.zeros((m + 1, m), dtype=float); rhs[:m, :] = np.eye(m)
				dh_dtheta = np.linalg.solve(M, rhs)              # (m+1) x m
			ddep = C @ dh_dtheta                                  # (|dep|) x m
		except np.linalg.LinAlgError:
			st = _StateEval(False, theta, 0.0, h, d, -np.inf, log_dir, -np.inf, 0.0)
			st.x = theta; st.warm = h
			return st
		G = np.eye(m) + ddep.T @ ddep
		sign, logabs = np.linalg.slogdet(G)
		log_jac = 0.5 * float(logabs) if sign > 0 else -np.inf
		if not np.isfinite(log_jac):
			st = _StateEval(False, theta, 0.0, h, d, -np.inf, log_dir, log_jac, 0.0)
			st.x = theta; st.warm = h
			return st

		log_target = float(log_dir + log_jac)
		st = _StateEval(True, theta, 0.0, h, d, log_target, log_dir, log_jac, 0.0)
		st.x = theta; st.warm = h
		return st

	def _eval(self, x: np.ndarray, warm: object) -> _StateEval:
		"""Method-agnostic state evaluation used by the proposal loops."""
		method = str(self.cfg.method).lower()
		if method == "demand":
			return self._evaluate_demand(x, warm)
		if method == "demand_exact":
			return self._evaluate_demand_exact(x, warm)
		if method in ("gaussian", "gaussian_map"):
			return self._evaluate_gaussian(x, warm)
		st = self._evaluate_state(x, float(warm) if warm is not None else float(self.initial_hv))
		st.x = st.z
		st.warm = st.hv
		return st

	def _run_chain(self, x_init: np.ndarray, warm_init: object, rng: np.random.Generator,
				   progress_cb=None) -> Dict[str, object]:
		"""Run a single Metropolis-Hastings chain and return its samples and counters."""
		total_iters = int(self.cfg.burn_in + self.cfg.num_samples * self.cfg.thin)
		adapt_until = int(self.cfg.burn_in * self.cfg.adapt_until_fraction)
		report_every = max(1, total_iters // 100)

		current = self._eval(np.asarray(x_init, dtype=float).copy(), warm_init)
		# An infeasible initial state (log_target = -inf) is fine: any feasible proposal
		# from a -inf state has log_alpha = min(0, finite - (-inf)) = 0, so it is always
		# accepted, and the chain reaches the posterior region immediately.

		samples_z: List[np.ndarray] = []
		samples_h: List[np.ndarray] = []
		samples_d: List[np.ndarray] = []
		log_targets: List[float] = []

		proposal_std = float(self.cfg.proposal_std)
		accepted = 0
		infeasible = 0
		block_accepted = 0

		for it in range(total_iters):
			x_prop = current.x + rng.normal(0.0, proposal_std, size=self.dim)
			prop = self._eval(x_prop, current.warm)  # warm-start from current state

			if not prop.feasible:
				infeasible += 1
				accept = False
			else:
				log_alpha = min(0.0, prop.log_target - current.log_target)
				accept = bool(math.log(rng.random()) < log_alpha)

			if accept:
				current = prop
				accepted += 1
				block_accepted += 1

			# Adaptive random-walk scale during burn-in.
			if (
				self.cfg.adapt_proposal
				and it < adapt_until
				and (it + 1) % self.cfg.adapt_interval == 0
			):
				block_rate = block_accepted / float(self.cfg.adapt_interval)
				# Robbins-Monro style log-scale update.
				t = max(1.0, (it + 1) / float(self.cfg.adapt_interval))
				gamma = self.cfg.adapt_gain / math.sqrt(t)
				proposal_std = float(proposal_std * math.exp(gamma * (block_rate - self.cfg.adapt_target_accept)))
				proposal_std = float(np.clip(proposal_std, 1e-6, 50.0))
				block_accepted = 0

			if it >= self.cfg.burn_in and ((it - self.cfg.burn_in) % self.cfg.thin == 0):
				samples_z.append(current.x.copy())
				samples_h.append(current.h.copy())
				samples_d.append(current.d.copy())
				log_targets.append(float(current.log_target))

			if progress_cb is not None and (it + 1) % report_every == 0:
				progress_cb((it + 1) / float(total_iters))

		return {
			"samples_z": np.asarray(samples_z, dtype=float),
			"samples_h": np.asarray(samples_h, dtype=float),
			"samples_d": np.asarray(samples_d, dtype=float),
			"log_targets": np.asarray(log_targets, dtype=float),
			"accepted": accepted,
			"infeasible": infeasible,
			"proposal_std_final": proposal_std,
			"total_iters": total_iters,
		}

	def _run_ensemble(self, rng: np.random.Generator, progress_cb=None) -> List[Dict[str, object]]:
		"""Affine-invariant ensemble sampler (Goodman-Weare stretch move).

		Runs an ensemble of K walkers.  A walker k is moved toward/away from a randomly
		chosen partner walker j:  z' = z_j + Z (z_k - z_j),  with the stretch factor Z drawn
		from g(Z) proportional to 1/sqrt(Z) on [1/a, a].  Proposals built from walker
		differences automatically align with the ensemble's shape, so a thin, tilted feasible
		region is explored as easily as a round one (affine invariance).  No gradients are
		needed; the same self._evaluate_state target and feasibility filter are reused.

		Returns one per-walker result dict (same schema as _run_chain) so the ensemble slots
		straight into the shared post-processing / R-hat path.  Each walker's trajectory is
		treated as a chain for the split-R-hat convergence check.
		"""
		total_iters = int(self.cfg.burn_in + self.cfg.num_samples * self.cfg.thin)
		report_every = max(1, total_iters // 100)
		dim = self.dim
		a = float(self.cfg.ensemble_stretch_a)
		if a <= 1.0:
			a = 2.0

		k_req = int(self.cfg.ensemble_walkers) if self.cfg.ensemble_walkers > 0 else max(2 * dim + 2, 8)
		K = k_req + (k_req % 2)  # even, so the red-black split is balanced

		# Initialise all walkers over-dispersed around the predictor init (walker 0 exactly at
		# it).  A spread ensemble is required for the difference vectors to span the target.
		walkers: List[_StateEval] = []
		for k in range(K):
			if k == 0:
				x0 = np.asarray(self.initial_x, dtype=float).copy()
			else:
				x0 = np.asarray(self.initial_x, dtype=float) + self.cfg.ensemble_init_dispersion * rng.normal(0.0, 1.0, size=dim)
			walkers.append(self._eval(x0, self.initial_warm))

		samples_z: List[List[np.ndarray]] = [[] for _ in range(K)]
		samples_h: List[List[np.ndarray]] = [[] for _ in range(K)]
		samples_d: List[List[np.ndarray]] = [[] for _ in range(K)]
		log_targets: List[List[float]] = [[] for _ in range(K)]
		accepted = np.zeros(K, dtype=int)
		infeasible = np.zeros(K, dtype=int)

		half = K // 2
		idx = np.arange(K)
		splits = [(idx[:half], idx[half:]), (idx[half:], idx[:half])]

		for it in range(total_iters):
			# Red-black update: each half is moved using partners drawn from the other half,
			# which preserves detailed balance for the ensemble move.
			for active, complement in splits:
				for k in active:
					j = int(complement[rng.integers(len(complement))])
					xk = walkers[k].x
					xj = walkers[j].x
					u = rng.random()
					z_stretch = ((a - 1.0) * u + 1.0) ** 2 / a
					x_prop = xj + z_stretch * (xk - xj)
					prop = self._eval(x_prop, walkers[k].warm)
					if not prop.feasible:
						infeasible[k] += 1
						continue
					# Stretch-move acceptance carries the Z^(dim-1) volume factor.
					log_alpha = (dim - 1) * math.log(z_stretch) + prop.log_target - walkers[k].log_target
					if math.log(rng.random()) < log_alpha:
						walkers[k] = prop
						accepted[k] += 1

			if it >= self.cfg.burn_in and ((it - self.cfg.burn_in) % self.cfg.thin == 0):
				for k in range(K):
					samples_z[k].append(walkers[k].x.copy())
					samples_h[k].append(walkers[k].h.copy())
					samples_d[k].append(walkers[k].d.copy())
					log_targets[k].append(float(walkers[k].log_target))

			if progress_cb is not None and (it + 1) % report_every == 0:
				progress_cb((it + 1) / float(total_iters))

		results: List[Dict[str, object]] = []
		for k in range(K):
			results.append({
				"samples_z": np.asarray(samples_z[k], dtype=float),
				"samples_h": np.asarray(samples_h[k], dtype=float),
				"samples_d": np.asarray(samples_d[k], dtype=float),
				"log_targets": np.asarray(log_targets[k], dtype=float),
				"accepted": int(accepted[k]),
				"infeasible": int(infeasible[k]),
				"proposal_std_final": float("nan"),  # not applicable to the ensemble move
				"total_iters": total_iters,          # one proposal per walker per iteration
			})
		return results

	def _run_map(self, progress_callback=None) -> MHSamplingResult:
		"""Gaussian-prior MAP via Gauss-Newton + a Laplace-Gaussian sample cloud.

		Minimises  ½‖(h_S(d)-M_S)/ε‖² + ½‖(d-μ)/σ‖²  over the free demands (total closed by the
		slack), then approximates the posterior near the mode by N(x_MAP, (JᵀJ)⁻¹) using the
		residual Jacobian, and draws `num_samples` from it so the result plugs into the same
		display path as the samplers.
		"""
		from scipy import optimize
		start_time = time.perf_counter()
		eps = max(float(self.cfg.sensor_noise_eps), 1e-9)
		sig_free = self.gauss_sigma  # per-junction prior std (full vector)
		big = 1e3

		def residuals(x):
			d = self._gauss_full_demand(x)
			ok, h = self._forward_solve(d, self._base_head_vec.copy())
			sensor_res = (h[self.sensor_node_idxs] - self.sensor_targets) / eps if self.sensor_node_idxs.size else np.zeros(0)
			prior_res = (d - self.gauss_mean) / sig_free
			# Soft d>=0 at the slack node (free demands are bounded >=0 below).
			slack_pen = np.array([big * max(0.0, -d[self.gauss_slack_jrow])])
			r = np.concatenate([sensor_res, prior_res, slack_pen])
			if not ok:
				r = r + big  # penalise non-converged solves
			return r

		x0 = np.asarray(self.initial_x, dtype=float)
		res = optimize.least_squares(residuals, x0, bounds=(0.0, np.inf), method="trf", xtol=1e-12, ftol=1e-12)
		if progress_callback is not None:
			progress_callback(0.7)
		x_map = res.x
		d_map = self._gauss_full_demand(x_map)
		_, h_map = self._forward_solve(d_map, self._base_head_vec.copy())

		# Laplace covariance in free coordinates: (JᵀJ)⁻¹ from the residual Jacobian at the mode.
		J = np.asarray(res.jac, dtype=float)
		H = J.T @ J
		try:
			cov = np.linalg.inv(H + 1e-9 * np.eye(H.shape[0]))
			cov = 0.5 * (cov + cov.T)
			L = np.linalg.cholesky(cov + 1e-12 * np.eye(cov.shape[0]))
		except np.linalg.LinAlgError:
			L = np.diag(np.sqrt(np.maximum(np.diag(np.linalg.pinv(H)), 0.0)))

		rng = np.random.default_rng(self.cfg.rng_seed)
		n = max(1, int(self.cfg.num_samples))
		samples_d = np.empty((n, self.n_junctions), dtype=float)
		samples_h = np.empty((n, self.n_nodes), dtype=float)
		for i in range(n):
			xs = x_map + L @ rng.normal(size=x_map.size)
			d = self._gauss_full_demand(xs)
			samples_d[i] = d
			ok, h = self._forward_solve(d, h_map)
			samples_h[i] = h
		if progress_callback is not None:
			progress_callback(1.0)

		elapsed = float(time.perf_counter() - start_time)
		diagnostics = {
			"method": "gaussian_map",
			"map_cost": float(res.cost),
			"map_optimality": float(res.optimality),
			"map_iterations": float(res.nfev),
			"elapsed_seconds": elapsed,
			"measured_total_demand": self.measured_total_demand,
		}
		# MAP samples are i.i.d. from the Laplace Gaussian -> convergence diagnostics are trivial.
		return MHSamplingResult(
			samples_z=samples_d[:, self.gauss_free_jrows], samples_h=samples_h, samples_d=samples_d,
			log_targets=np.zeros(n), acceptance_rate=1.0, infeasible_rate=0.0, punished_rate=0.0,
			proposal_std_final=float("nan"), ess_per_dimension=np.full(self.dim, float(n)),
			min_ess=float(n), median_ess=float(n), elapsed_seconds=elapsed,
			min_ess_per_sec=float(n) / elapsed if elapsed > 0 else 0.0,
			median_ess_per_sec=float(n) / elapsed if elapsed > 0 else 0.0,
			rhat_per_dimension=np.ones(self.dim), max_rhat=1.0, num_chains=1,
			mean_disagreement_per_dim=np.zeros(self.n_junctions), max_mean_disagreement=0.0,
			diagnostics=diagnostics,
		)

	def sample(self, progress_callback=None) -> MHSamplingResult:
		if str(self.cfg.method).lower() == "gaussian_map":
			return self._run_map(progress_callback)
		start_time = time.perf_counter()
		if str(self.cfg.proposal).lower() == "ensemble":
			# One ensemble of walkers; each walker trajectory becomes a "chain" for R-hat.
			ens_rng = np.random.default_rng(self.cfg.rng_seed)
			chain_results = self._run_ensemble(ens_rng, progress_cb=progress_callback)
		else:
			num_chains = max(1, int(self.cfg.num_chains))
			# Chain 0 starts at the predictor init; extra chains start over-dispersed around it
			# so a split-R-hat can detect chains that fail to reach a common distribution.
			x_base = np.asarray(self.initial_x, dtype=float)
			inits: List[Tuple[np.ndarray, object]] = [(x_base.copy(), self.initial_warm)]
			for c in range(1, num_chains):
				disp_rng = np.random.default_rng(
					None if self.cfg.rng_seed is None else self.cfg.rng_seed + 1000 + c
				)
				x0 = x_base + self.cfg.chain_init_dispersion * disp_rng.normal(0.0, 1.0, size=self.dim)
				inits.append((x0, self.initial_warm))

			chain_results = []
			for c in range(num_chains):
				chain_rng = np.random.default_rng(
					None if self.cfg.rng_seed is None else self.cfg.rng_seed + c
				)
				x0, warm0 = inits[c]
				# Compose overall progress across the sequential chains.
				cb = None
				if progress_callback is not None:
					cb = (lambda f, _c=c: progress_callback((_c + f) / num_chains))
				chain_results.append(self._run_chain(x0, warm0, chain_rng, progress_cb=cb))
		elapsed_seconds = float(time.perf_counter() - start_time)
		num_chains = len(chain_results)

		# Split-R-hat is computed from the per-chain z samples before concatenation.
		per_chain_z = [np.asarray(r["samples_z"], dtype=float) for r in chain_results]
		rhat_vec = self._split_rhat_per_dim(per_chain_z)
		max_rhat = float(np.max(rhat_vec)) if rhat_vec.size else float("nan")

		# Concatenate all chains for the returned posterior sample set.
		samples_z_arr = np.concatenate(per_chain_z, axis=0) if per_chain_z else np.zeros((0, self.dim))
		samples_h_arr = np.concatenate([r["samples_h"] for r in chain_results], axis=0)
		samples_d_arr = np.concatenate([r["samples_d"] for r in chain_results], axis=0)
		log_targets_arr = np.concatenate([r["log_targets"] for r in chain_results], axis=0)

		# Mean agreement on the DEMANDS: spread of per-chain demand means / pooled posterior std.
		# Unlike R-hat (which also reacts to variance mismatch and tails), this isolates whether
		# the chains agree on the posterior *mean* — the quantity used as the demand estimate.
		per_chain_d_means = np.array(
			[r["samples_d"].mean(axis=0) for r in chain_results if r["samples_d"].size]
		)
		if per_chain_d_means.shape[0] >= 2 and samples_d_arr.size:
			pooled_std = samples_d_arr.std(axis=0)
			mean_disagree_vec = per_chain_d_means.std(axis=0) / np.maximum(pooled_std, 1e-12)
			max_mean_disagreement = float(np.max(mean_disagree_vec))
		else:
			mean_disagree_vec = np.full(self.n_junctions, np.nan, dtype=float)
			max_mean_disagreement = float("nan")

		# Total ESS is the sum of per-chain ESS (independent chains contribute additively).
		ess_vec = np.sum([self._effective_sample_size_per_dim(z) for z in per_chain_z], axis=0)
		min_ess = float(np.min(ess_vec)) if getattr(ess_vec, "size", 0) else 0.0
		med_ess = float(np.median(ess_vec)) if getattr(ess_vec, "size", 0) else 0.0
		min_ess_per_sec = min_ess / elapsed_seconds if elapsed_seconds > 0 else 0.0
		med_ess_per_sec = med_ess / elapsed_seconds if elapsed_seconds > 0 else 0.0

		total_iters_all = sum(int(r["total_iters"]) for r in chain_results)
		accepted_all = sum(int(r["accepted"]) for r in chain_results)
		infeasible_all = sum(int(r["infeasible"]) for r in chain_results)
		acceptance_rate = accepted_all / float(total_iters_all) if total_iters_all else 0.0
		infeasible_rate = infeasible_all / float(total_iters_all) if total_iters_all else 0.0
		proposal_std_final = float(chain_results[0]["proposal_std_final"])

		if samples_d_arr.size > 0:
			_punished_mask = np.any(
				samples_d_arr < self.base_demands[None, :] - self.cfg.demand_lb_tolerance,
				axis=1,
			)
			punished_rate = float(_punished_mask.mean())
		else:
			punished_rate = 0.0

		diagnostics = {
			"acceptance_rate": acceptance_rate,
			"infeasible_rate": infeasible_rate,
			"punished_rate": punished_rate,
			"proposal_std_final": proposal_std_final,
			"mean_log_target": float(np.mean(log_targets_arr)) if log_targets_arr.size else float("nan"),
			"std_log_target": float(np.std(log_targets_arr)) if log_targets_arr.size else float("nan"),
			"target_extra": self.target_extra,
			"base_total": self.base_total,
			"measured_total_demand": self.measured_total_demand,
			"elapsed_seconds": elapsed_seconds,
			"min_ess_per_sec": min_ess_per_sec,
			"median_ess_per_sec": med_ess_per_sec,
			"num_chains": float(num_chains),
			"max_rhat": max_rhat,
			"max_mean_disagreement": max_mean_disagreement,
		}

		return MHSamplingResult(
			samples_z=samples_z_arr,
			samples_h=samples_h_arr,
			samples_d=samples_d_arr,
			log_targets=log_targets_arr,
			acceptance_rate=acceptance_rate,
			infeasible_rate=infeasible_rate,
			punished_rate=punished_rate,
			proposal_std_final=proposal_std_final,
			ess_per_dimension=ess_vec,
			min_ess=min_ess,
			median_ess=med_ess,
			elapsed_seconds=elapsed_seconds,
			min_ess_per_sec=min_ess_per_sec,
			median_ess_per_sec=med_ess_per_sec,
			rhat_per_dimension=rhat_vec,
			max_rhat=max_rhat,
			num_chains=num_chains,
			mean_disagreement_per_dim=mean_disagree_vec,
			max_mean_disagreement=max_mean_disagreement,
			diagnostics=diagnostics,
		)

	@staticmethod
	def _effective_sample_size_per_dim(samples: np.ndarray) -> np.ndarray:
		if samples.ndim != 2 or samples.shape[0] < 4:
			return np.zeros(samples.shape[1] if samples.ndim == 2 else 0, dtype=float)

		n, d = samples.shape
		ess = np.zeros(d, dtype=float)
		for j in range(d):
			x = samples[:, j].astype(float)
			x -= np.mean(x)
			var = float(np.dot(x, x) / n)
			if var <= 0.0:
				ess[j] = 0.0
				continue

			# Initial positive sequence estimator.
			rho_sum = 0.0
			max_lag = min(n - 1, 1000)
			lag = 1
			while lag + 1 <= max_lag:
				ac1 = float(np.dot(x[:-lag], x[lag:]) / (n - lag) / var)
				lag2 = lag + 1
				ac2 = float(np.dot(x[:-lag2], x[lag2:]) / (n - lag2) / var)
				pair = ac1 + ac2
				if pair <= 0.0:
					break
				rho_sum += pair
				lag += 2

			tau = max(1.0, 1.0 + 2.0 * rho_sum)
			ess[j] = n / tau
		return ess

	@staticmethod
	def _split_rhat_per_dim(per_chain: List[np.ndarray]) -> np.ndarray:
		"""Split-R-hat (Gelman-Rubin) per coordinate across chains.

		Each chain is split in half to expose within-chain non-stationarity, giving
		2*num_chains sequences.  R-hat compares between-chain variance B to within-chain
		variance W: values near 1 indicate the chains have mixed to a common distribution,
		while R-hat > ~1.01 flags non-convergence.  Returns all-NaN if fewer than 2 chains
		or too few samples to split.
		"""
		chains = [np.asarray(c, dtype=float) for c in per_chain if getattr(c, "ndim", 0) == 2 and c.shape[0] >= 4]
		if len(chains) < 2:
			dim = per_chain[0].shape[1] if per_chain and per_chain[0].ndim == 2 else 0
			return np.full(dim, np.nan, dtype=float)

		# Trim all chains to a common even length, then split each into two halves.
		n = min(c.shape[0] for c in chains)
		n -= n % 2
		if n < 4:
			return np.full(chains[0].shape[1], np.nan, dtype=float)
		half = n // 2
		splits = []
		for c in chains:
			splits.append(c[:half])
			splits.append(c[half:n])

		seqs = np.stack(splits, axis=0)          # (M sequences, half, dim)
		m, npd = seqs.shape[0], seqs.shape[1]     # M = 2*num_chains, npd = half length
		chain_means = seqs.mean(axis=1)           # (M, dim)
		grand_mean = chain_means.mean(axis=0)     # (dim,)

		# Between- and within-sequence variance.
		b = npd / (m - 1.0) * np.sum((chain_means - grand_mean[None, :]) ** 2, axis=0)
		chain_vars = seqs.var(axis=1, ddof=1)     # (M, dim)
		w = chain_vars.mean(axis=0)               # (dim,)

		var_plus = (npd - 1.0) / npd * w + b / npd
		with np.errstate(divide="ignore", invalid="ignore"):
			rhat = np.sqrt(var_plus / w)
		# Degenerate (constant) coordinates have w = 0 → treat as perfectly mixed.
		rhat = np.where(w > 0.0, rhat, 1.0)
		return rhat


def sample_posterior_scenarios(
	inp_path: str,
	measurement_heads: Dict[str, float],
	measured_total_demand: float,
	predictor_heads: Dict[str, float],
	elimination_node: Optional[str] = None,
	drop_demand_row_node: Optional[str] = None,
	dirichlet_alpha: Optional[Dict[str, float]] = None,
	config: Optional[MHPosteriorConfig] = None,
	progress_callback=None,
) -> MHSamplingResult:
	sampler = PosteriorScenarioSampler(
		inp_path=inp_path,
		measurement_heads=measurement_heads,
		measured_total_demand=measured_total_demand,
		predictor_heads=predictor_heads,
		elimination_node=elimination_node,
		drop_demand_row_node=drop_demand_row_node,
		dirichlet_alpha=dirichlet_alpha,
		config=config,
	)
	return sampler.sample(progress_callback=progress_callback)
