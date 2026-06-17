from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import ast
import math

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

		self.fixed_heads: Dict[str, float] = {str(k): float(v) for k, v in measurement_heads.items()}
		self.predictor_heads: Dict[str, float] = {
			str(k): float(v) for k, v in predictor_heads.items()
		}

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

	def sample(self) -> MHSamplingResult:
		total_iters = int(self.cfg.burn_in + self.cfg.num_samples * self.cfg.thin)
		adapt_until = int(self.cfg.burn_in * self.cfg.adapt_until_fraction)

		z = self.initial_z.copy()
		hv = float(self.initial_hv)
		current = self._evaluate_state(z, hv)
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
			z_prop = current.z + self.rng.normal(0.0, proposal_std, size=self.dim)
			prop = self._evaluate_state(z_prop, current.hv)  # warm-start hv from current state

			if not prop.feasible:
				infeasible += 1
				accept = False
			else:
				log_alpha = min(0.0, prop.log_target - current.log_target)
				accept = bool(math.log(self.rng.random()) < log_alpha)

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
				samples_z.append(current.z.copy())
				samples_h.append(current.h.copy())
				samples_d.append(current.d.copy())
				log_targets.append(float(current.log_target))

		samples_z_arr = np.asarray(samples_z, dtype=float)
		samples_h_arr = np.asarray(samples_h, dtype=float)
		samples_d_arr = np.asarray(samples_d, dtype=float)
		log_targets_arr = np.asarray(log_targets, dtype=float)

		ess_vec = self._effective_sample_size_per_dim(samples_z_arr)
		min_ess = float(np.min(ess_vec)) if ess_vec.size else 0.0
		med_ess = float(np.median(ess_vec)) if ess_vec.size else 0.0

		acceptance_rate = accepted / float(total_iters)
		infeasible_rate = infeasible / float(total_iters)

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
			"proposal_std_final": proposal_std,
			"mean_log_target": float(np.mean(log_targets_arr)) if log_targets_arr.size else float("nan"),
			"std_log_target": float(np.std(log_targets_arr)) if log_targets_arr.size else float("nan"),
			"target_extra": self.target_extra,
			"base_total": self.base_total,
			"measured_total_demand": self.measured_total_demand,
		}

		return MHSamplingResult(
			samples_z=samples_z_arr,
			samples_h=samples_h_arr,
			samples_d=samples_d_arr,
			log_targets=log_targets_arr,
			acceptance_rate=acceptance_rate,
			infeasible_rate=infeasible_rate,
			punished_rate=punished_rate,
			proposal_std_final=proposal_std,
			ess_per_dimension=ess_vec,
			min_ess=min_ess,
			median_ess=med_ess,
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


def sample_posterior_scenarios(
	inp_path: str,
	measurement_heads: Dict[str, float],
	measured_total_demand: float,
	predictor_heads: Dict[str, float],
	elimination_node: Optional[str] = None,
	drop_demand_row_node: Optional[str] = None,
	dirichlet_alpha: Optional[Dict[str, float]] = None,
	config: Optional[MHPosteriorConfig] = None,
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
	return sampler.sample()
