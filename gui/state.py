from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SolverParams:
	wdn: str = "Alperovits"
	mode: str = "W_d"
	method: str = "xd"
	norm: float = 2.0
	demand_lb: float = 1e-6
	extra_demand: float = 1.2
	extra_demand_use_default: bool = True
	measurement_heads_equal_only: bool = True
	measurement_sites: List[str] = field(default_factory=list)
	measurement_source: str = "from_w_d"
	measurement_data: str = ""
	multi_starts: int = 1
	multi_start_noise: float = 0.05
	multi_start_noise_rel: float = 0.25
	multi_start_seed: Optional[int] = None
	dynamic_multistart: bool = False
	dms_consistency: int = 3
	dms_deviation: float = 0.95
	dms_radius: float = float("inf")
	dms_max_starts: int = 10
	dms_discard_unclear: bool = True
	hexaly_license_path: str = "~/opt/Hexaly_14_5/license.dat"
	hexaly_time_limit: int = 30
	hexaly_seed: int = 0
	hexaly_verbosity: int = 2
	match_total_demand: bool = True
	match_reservoir_outflow_between_pairs: bool = True
	output_dir: Optional[str] = None
