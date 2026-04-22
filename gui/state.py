from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ScenarioParams:
	wdn: str = "Alperovits"
	scenario_name: str = "Alperovits-base"
	scenario_source: str = "base"
	pipe_bounds: bool = True
	pipe_bound_safeness: float = 0.85
	pipe_bound_method: str = "perturb"
	pipe_bound_policy: str = "minmax"
	pipe_bound_perturb_samples: int = 25
	pipe_bound_perturb_sigma: float = 0.10
	pipe_bound_perturb_seed: int = 1
	pipe_bound_perturb_fix_total: bool = True
	pipe_bound_perturb_base: str = "base"


@dataclass
class SolverParams:
	wdn: str = "Alperovits"
	scenario_name: str = "Alperovits-base"
	solver: str = "Hexaly_xd"
	norm: float = 2.0
	demand_lb: float = 1e-6
	use_reservoir_outflow_constraint: bool = False
	fix_total_demand: bool = True
	measurement_heads_equal_only: bool = True
	measurement_sites: List[str] = field(default_factory=list)
	measurement_max_subsets: int = 200
	measurement_subset_seed: int = 1
	measurement_subset_mode: str = "all"
	multi_starts: int = 1
	multi_start_noise: float = 0.05
	multi_start_noise_rel: float = 0.25
	multi_start_seed: Optional[int] = None
	check_xd_base_feasibility: bool = True
	check_xd_cycle_bounds: bool = True
	xd_cycle_basis_mode: str = "planar"
	skip_feasibility_solve: bool = True
	fixed_reference: bool = False
	demand_restriction_mode: Optional[str] = None
	radius_to_fixed: Optional[float] = None
	deviation_alpha: Optional[float] = None
	hexaly_license_path: str = "~/opt/Hexaly_14_5/license.dat"
	hexaly_time_limit: int = 30
	hexaly_stagnation_seconds: int = 0
	hexaly_stagnation_eps: float = 1e-6
	hexaly_seed: int = 0
	hexaly_verbosity: int = 2
	hexaly_head_margin: float = 300.0
	hexaly_use_path_head_bounds: bool = True
	output_dir: Optional[str] = None
