from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple
import numpy as np

from step1_io import NetworkData


@dataclass(frozen=True)
class ScenarioStats:
    flows_by_pipe: Dict[str, np.ndarray]
    c_secondary: Dict[str, float]
    C_primary: Dict[str, float]


def _require_wntr() -> None:
    try:
        import wntr  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'wntr' package is required for scenario simulation. "
            "Install it with: pip install wntr"
        ) from exc


def simulate_random_demand_scenarios(
    inp_path: str,
    n_scenarios: int,
    demand_log_sigma: float = 0.3,
    seed: Optional[int] = 1,
    simulator: str = "auto",
) -> Dict[str, np.ndarray]:
    """
    Simulate random demand scenarios and return per-pipe flow arrays.

    Each scenario draws independent lognormal multipliers for junction demands:
        m = exp(N(0, demand_log_sigma^2))

    Returns: {pipe_id: np.ndarray(shape=(n_scenarios,))}
    """
    _require_wntr()
    import wntr

    rng = np.random.default_rng(seed)
    wn = wntr.network.WaterNetworkModel(inp_path)

    base_demands: Dict[str, float] = {}
    for name, node in wn.junctions():
        if node.demand_timeseries_list is None or len(node.demand_timeseries_list) == 0:
            base_demands[name] = 0.0
        else:
            base_demands[name] = float(node.demand_timeseries_list[0].base_value or 0.0)

    flows_by_pipe: Dict[str, np.ndarray] = {
        pipe_name: np.zeros(n_scenarios, dtype=float) for pipe_name, _ in wn.pipes()
    }

    if simulator == "auto":
        try:
            sim = wntr.sim.EpanetSimulator(wn)
        except Exception:
            sim = wntr.sim.WNTRSimulator(wn)
    elif simulator == "epanet":
        sim = wntr.sim.EpanetSimulator(wn)
    elif simulator == "wntr":
        sim = wntr.sim.WNTRSimulator(wn)
    else:
        raise ValueError("simulator must be 'auto', 'epanet', or 'wntr'.")

    for i in range(n_scenarios):
        for junc_name, node in wn.junctions():
            base = base_demands[junc_name]
            if node.demand_timeseries_list is None or len(node.demand_timeseries_list) == 0:
                continue
            if base <= 0:
                node.demand_timeseries_list[0].base_value = 0.0
                continue
            multiplier = float(np.exp(rng.normal(0.0, demand_log_sigma)))
            node.demand_timeseries_list[0].base_value = base * multiplier

        try:
            results = sim.run_sim()
        except Exception:
            sim = wntr.sim.WNTRSimulator(wn)
            results = sim.run_sim()
        flow_df = results.link["flowrate"]
        t0 = flow_df.index[0]
        for pipe_name in flows_by_pipe:
            flows_by_pipe[pipe_name][i] = float(flow_df.loc[t0, pipe_name])

    for junc_name, node in wn.junctions():
        if node.demand_timeseries_list is None or len(node.demand_timeseries_list) == 0:
            continue
        node.demand_timeseries_list[0].base_value = base_demands[junc_name]

    return flows_by_pipe


def simulate_perturbed_demand_scenarios(
    inp_path: str,
    n_scenarios: int,
    demand_sigma: float = 0.1,
    base_demands: Optional[Dict[str, float]] = None,
    enforce_total_demand: bool = False,
    total_demand: Optional[float] = None,
    include_base: bool = True,
    seed: Optional[int] = 1,
    simulator: str = "auto",
) -> Dict[str, np.ndarray]:
    """
    Simulate demand scenarios by perturbing a fixed base demand vector.

    Each perturbed demand is: d = max(0, d* (1 + sigma * N(0,1))).
    Optionally rescales each draw to keep total demand fixed.

    Returns: {pipe_id: np.ndarray(shape=(n_scenarios + include_base,))}
    """
    _require_wntr()
    import wntr

    rng = np.random.default_rng(seed)
    wn = wntr.network.WaterNetworkModel(inp_path)

    base_vals: Dict[str, float] = {}
    for name, node in wn.junctions():
        if node.demand_timeseries_list is None or len(node.demand_timeseries_list) == 0:
            base_vals[name] = 0.0
        else:
            base_vals[name] = float(node.demand_timeseries_list[0].base_value or 0.0)

    if base_demands is not None:
        for name in base_vals:
            if name in base_demands:
                base_vals[name] = float(base_demands[name])

    target_total = total_demand
    if target_total is None:
        target_total = float(sum(base_vals.values()))

    total_samples = n_scenarios + (1 if include_base else 0)
    flows_by_pipe: Dict[str, np.ndarray] = {
        pipe_name: np.zeros(total_samples, dtype=float) for pipe_name, _ in wn.pipes()
    }

    if simulator == "auto":
        try:
            sim = wntr.sim.EpanetSimulator(wn)
        except Exception:
            sim = wntr.sim.WNTRSimulator(wn)
    elif simulator == "epanet":
        sim = wntr.sim.EpanetSimulator(wn)
    elif simulator == "wntr":
        sim = wntr.sim.WNTRSimulator(wn)
    else:
        raise ValueError("simulator must be 'auto', 'epanet', or 'wntr'.")

    for i in range(total_samples):
        if include_base and i == 0:
            demands = dict(base_vals)
        else:
            demands = {}
            for junc_name, base in base_vals.items():
                if base <= 0:
                    demands[junc_name] = 0.0
                else:
                    pert = base * (1.0 + demand_sigma * float(rng.standard_normal()))
                    demands[junc_name] = max(0.0, pert)

            if enforce_total_demand and target_total is not None and target_total > 0:
                current_total = sum(demands.values())
                if current_total > 0:
                    scale = target_total / current_total
                    for name in demands:
                        demands[name] *= scale

        for junc_name, node in wn.junctions():
            if node.demand_timeseries_list is None or len(node.demand_timeseries_list) == 0:
                continue
            node.demand_timeseries_list[0].base_value = demands.get(junc_name, 0.0)

        try:
            results = sim.run_sim()
        except Exception:
            sim = wntr.sim.WNTRSimulator(wn)
            results = sim.run_sim()

        flow_df = results.link["flowrate"]
        t0 = flow_df.index[0]
        for pipe_name in flows_by_pipe:
            flows_by_pipe[pipe_name][i] = float(flow_df.loc[t0, pipe_name])

    for junc_name, node in wn.junctions():
        if node.demand_timeseries_list is None or len(node.demand_timeseries_list) == 0:
            continue
        node.demand_timeseries_list[0].base_value = base_vals[junc_name]

    return flows_by_pipe


def simulate_base_flows(
    inp_path: str,
    simulator: str = "auto",
) -> Dict[str, float]:
    """
    Run a single steady-state simulation with base demands.

    Returns: {pipe_id: flowrate}
    """
    _require_wntr()
    import wntr

    wn = wntr.network.WaterNetworkModel(inp_path)
    if simulator == "auto":
        try:
            sim = wntr.sim.EpanetSimulator(wn)
        except Exception:
            sim = wntr.sim.WNTRSimulator(wn)
    elif simulator == "epanet":
        sim = wntr.sim.EpanetSimulator(wn)
    elif simulator == "wntr":
        sim = wntr.sim.WNTRSimulator(wn)
    else:
        raise ValueError("simulator must be 'auto', 'epanet', or 'wntr'.")

    try:
        results = sim.run_sim()
    except Exception:
        sim = wntr.sim.WNTRSimulator(wn)
        results = sim.run_sim()
    flow_df = results.link["flowrate"]
    t0 = flow_df.index[0]
    return {pipe_name: float(flow_df.loc[t0, pipe_name]) for pipe_name in flow_df.columns}


def simulate_base_scenario(
    inp_path: str,
    simulator: str = "auto",
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Simulate one steady-state scenario using base demands from the .inp.

    Returns: (demands, heads, flows)
    """
    _require_wntr()
    import wntr

    wn = wntr.network.WaterNetworkModel(inp_path)

    demands: Dict[str, float] = {}
    for name, node in wn.junctions():
        if node.demand_timeseries_list is None or len(node.demand_timeseries_list) == 0:
            demands[name] = 0.0
            continue
        demands[name] = float(node.demand_timeseries_list[0].base_value or 0.0)

    if simulator == "auto":
        try:
            sim = wntr.sim.EpanetSimulator(wn)
        except Exception:
            sim = wntr.sim.WNTRSimulator(wn)
    elif simulator == "epanet":
        sim = wntr.sim.EpanetSimulator(wn)
    elif simulator == "wntr":
        sim = wntr.sim.WNTRSimulator(wn)
    else:
        raise ValueError("simulator must be 'auto', 'epanet', or 'wntr'.")

    try:
        results = sim.run_sim()
    except Exception:
        sim = wntr.sim.WNTRSimulator(wn)
        results = sim.run_sim()

    flow_df = results.link["flowrate"]
    head_df = results.node["head"]
    t0 = flow_df.index[0]

    flows = {pipe_name: float(flow_df.loc[t0, pipe_name]) for pipe_name in flow_df.columns}
    heads = {node_name: float(head_df.loc[t0, node_name]) for node_name in head_df.columns}
    return demands, heads, flows


def simulate_single_random_scenario(
    inp_path: str,
    demand_log_sigma: float = 0.3,
    seed: Optional[int] = 1,
    simulator: str = "auto",
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Simulate one random demand scenario and return (demands, heads, flows).
    """
    _require_wntr()
    import wntr

    rng = np.random.default_rng(seed)
    wn = wntr.network.WaterNetworkModel(inp_path)

    demands: Dict[str, float] = {}
    for name, node in wn.junctions():
        if node.demand_timeseries_list is None or len(node.demand_timeseries_list) == 0:
            demands[name] = 0.0
            continue
        base = float(node.demand_timeseries_list[0].base_value or 0.0)
        if base <= 0:
            node.demand_timeseries_list[0].base_value = 0.0
            demands[name] = 0.0
            continue
        multiplier = float(np.exp(rng.normal(0.0, demand_log_sigma)))
        node.demand_timeseries_list[0].base_value = base * multiplier
        demands[name] = float(node.demand_timeseries_list[0].base_value)

    if simulator == "auto":
        try:
            sim = wntr.sim.EpanetSimulator(wn)
        except Exception:
            sim = wntr.sim.WNTRSimulator(wn)
    elif simulator == "epanet":
        sim = wntr.sim.EpanetSimulator(wn)
    elif simulator == "wntr":
        sim = wntr.sim.WNTRSimulator(wn)
    else:
        raise ValueError("simulator must be 'auto', 'epanet', or 'wntr'.")

    try:
        results = sim.run_sim()
    except Exception:
        sim = wntr.sim.WNTRSimulator(wn)
        results = sim.run_sim()
    flow_df = results.link["flowrate"]
    head_df = results.node["head"]
    t0 = flow_df.index[0]

    flows = {pipe_name: float(flow_df.loc[t0, pipe_name]) for pipe_name in flow_df.columns}
    heads = {node_name: float(head_df.loc[t0, node_name]) for node_name in head_df.columns}
    return demands, heads, flows


def simulate_dirichlet_demand_scenarios(
    inp_path: str,
    n_scenarios: int,
    extra_demand: float,
    min_deviation: float = 1.0,
    max_deviation: float = 1.0,
    seed: Optional[int] = 1,
    simulator: str = "auto",
) -> Dict[str, np.ndarray]:
    """
    Simulate demand scenarios using the Dirichlet extra-demand model from the GNN pipeline.

    Each scenario draws alpha ~ Dirichlet(1, ..., 1) and sets:
        d_j = d_j_base + alpha_j * (deviation_factor * extra_demand)
    where deviation_factor ~ Uniform(min_deviation, max_deviation).

    Total demand per scenario lies in:
        [sum(d_j_base) + min_deviation * extra_demand,
         sum(d_j_base) + max_deviation * extra_demand]
    (for min_deviation <= max_deviation).

    Returns: {pipe_id: np.ndarray(shape=(n_scenarios,))}
    """
    _require_wntr()
    import wntr

    if min_deviation > max_deviation:
        raise ValueError("min_deviation must be <= max_deviation.")

    rng = np.random.default_rng(seed)
    wn = wntr.network.WaterNetworkModel(inp_path)

    junction_names = [name for name, _ in wn.junctions()]
    n_junctions = len(junction_names)

    base_demands: Dict[str, float] = {}
    for name, node in wn.junctions():
        if node.demand_timeseries_list is None or len(node.demand_timeseries_list) == 0:
            base_demands[name] = 0.0
        else:
            base_demands[name] = float(node.demand_timeseries_list[0].base_value or 0.0)

    flows_by_pipe: Dict[str, np.ndarray] = {
        pipe_name: np.zeros(n_scenarios, dtype=float) for pipe_name, _ in wn.pipes()
    }

    if simulator == "auto":
        try:
            sim = wntr.sim.EpanetSimulator(wn)
        except Exception:
            sim = wntr.sim.WNTRSimulator(wn)
    elif simulator == "epanet":
        sim = wntr.sim.EpanetSimulator(wn)
    elif simulator == "wntr":
        sim = wntr.sim.WNTRSimulator(wn)
    else:
        raise ValueError("simulator must be 'auto', 'epanet', or 'wntr'.")

    alpha = np.ones(n_junctions)
    for i in range(n_scenarios):
        deviation_factor = float(rng.uniform(min_deviation, max_deviation))
        total_extra = deviation_factor * float(extra_demand)
        shares = rng.dirichlet(alpha)
        for j, junc_name in enumerate(junction_names):
            node = wn.get_node(junc_name)
            if node.demand_timeseries_list is None or len(node.demand_timeseries_list) == 0:
                continue
            node.demand_timeseries_list[0].base_value = (
                base_demands[junc_name] + float(shares[j]) * total_extra
            )

        try:
            results = sim.run_sim()
        except Exception:
            sim = wntr.sim.WNTRSimulator(wn)
            results = sim.run_sim()

        flow_df = results.link["flowrate"]
        t0 = flow_df.index[0]
        for pipe_name in flows_by_pipe:
            flows_by_pipe[pipe_name][i] = float(flow_df.loc[t0, pipe_name])

    for junc_name, node in wn.junctions():
        if node.demand_timeseries_list is None or len(node.demand_timeseries_list) == 0:
            continue
        node.demand_timeseries_list[0].base_value = base_demands[junc_name]

    return flows_by_pipe


def simulate_single_dirichlet_scenario(
    inp_path: str,
    extra_demand: float,
    min_deviation: float = 1.0,
    max_deviation: float = 1.0,
    seed: Optional[int] = 1,
    simulator: str = "auto",
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Simulate one scenario using the Dirichlet extra-demand model.

    Draws alpha ~ Dirichlet(1, ..., 1) and sets:
        d_j = d_j_base + alpha_j * (deviation_factor * extra_demand)
    where deviation_factor ~ Uniform(min_deviation, max_deviation).

    Returns: (demands, heads, flows)
    """
    _require_wntr()
    import wntr

    if min_deviation > max_deviation:
        raise ValueError("min_deviation must be <= max_deviation.")

    rng = np.random.default_rng(seed)
    wn = wntr.network.WaterNetworkModel(inp_path)

    junction_names = [name for name, _ in wn.junctions()]
    n_junctions = len(junction_names)

    base_demands: Dict[str, float] = {}
    for name, node in wn.junctions():
        if node.demand_timeseries_list is None or len(node.demand_timeseries_list) == 0:
            base_demands[name] = 0.0
        else:
            base_demands[name] = float(node.demand_timeseries_list[0].base_value or 0.0)

    alpha = np.ones(n_junctions)
    deviation_factor = float(rng.uniform(min_deviation, max_deviation))
    total_extra = deviation_factor * float(extra_demand)
    shares = rng.dirichlet(alpha)

    demands: Dict[str, float] = {}
    for j, junc_name in enumerate(junction_names):
        node = wn.get_node(junc_name)
        if node.demand_timeseries_list is None or len(node.demand_timeseries_list) == 0:
            demands[junc_name] = 0.0
            continue
        new_demand = base_demands[junc_name] + float(shares[j]) * total_extra
        node.demand_timeseries_list[0].base_value = new_demand
        demands[junc_name] = new_demand

    if simulator == "auto":
        try:
            sim = wntr.sim.EpanetSimulator(wn)
        except Exception:
            sim = wntr.sim.WNTRSimulator(wn)
    elif simulator == "epanet":
        sim = wntr.sim.EpanetSimulator(wn)
    elif simulator == "wntr":
        sim = wntr.sim.WNTRSimulator(wn)
    else:
        raise ValueError("simulator must be 'auto', 'epanet', or 'wntr'.")

    try:
        results = sim.run_sim()
    except Exception:
        sim = wntr.sim.WNTRSimulator(wn)
        results = sim.run_sim()

    flow_df = results.link["flowrate"]
    head_df = results.node["head"]
    t0 = flow_df.index[0]

    flows = {pipe_name: float(flow_df.loc[t0, pipe_name]) for pipe_name in flow_df.columns}
    heads = {node_name: float(head_df.loc[t0, node_name]) for node_name in head_df.columns}
    return demands, heads, flows


def estimate_capacity_bounds(
    network: NetworkData,
    pipe_set_primary: Iterable[str],
    pipe_set_secondary: Iterable[str],
    simulated_flows_by_pipe: Dict[str, np.ndarray],
    q_secondary_quantile: float = 0.95,
    q_primary_quantile: float = 0.50,
) -> ScenarioStats:
    """
    Estimate c_e and C_e from simulated flow scenarios.

    For E2: c_e = quantile(|q_e|, q_secondary_quantile)
    For E1: C_e = quantile(q_e, q_primary_quantile)

    This assumes a consistent flow orientation in the simulation outputs.
    """
    c_secondary: Dict[str, float] = {}
    C_primary: Dict[str, float] = {}

    for pipe_id in pipe_set_secondary:
        q = simulated_flows_by_pipe[pipe_id]
        c_secondary[pipe_id] = float(np.quantile(np.abs(q), q_secondary_quantile))

    for pipe_id in pipe_set_primary:
        q = simulated_flows_by_pipe[pipe_id]
        C_primary[pipe_id] = float(np.quantile(q, q_primary_quantile))

    return ScenarioStats(
        flows_by_pipe=simulated_flows_by_pipe,
        c_secondary=c_secondary,
        C_primary=C_primary,
    )


def estimate_capacity_from_inp(
    inp_path: str,
    network: NetworkData,
    pipe_set_primary: Iterable[str],
    pipe_set_secondary: Iterable[str],
    n_scenarios: int = 200,
    demand_log_sigma: float = 0.3,
    q_secondary_quantile: float = 0.95,
    q_primary_quantile: float = 0.50,
    seed: Optional[int] = 1,
    simulator: str = "auto",
) -> ScenarioStats:
    flows_by_pipe = simulate_random_demand_scenarios(
        inp_path=inp_path,
        n_scenarios=n_scenarios,
        demand_log_sigma=demand_log_sigma,
        seed=seed,
        simulator=simulator,
    )
    return estimate_capacity_bounds(
        network=network,
        pipe_set_primary=pipe_set_primary,
        pipe_set_secondary=pipe_set_secondary,
        simulated_flows_by_pipe=flows_by_pipe,
        q_secondary_quantile=q_secondary_quantile,
        q_primary_quantile=q_primary_quantile,
    )


def simulate_single_leak_scenarios(
    inp_path: str,
    n_scenarios: int,
    extra_demand: float,
    leak_nodes: Optional[Iterable[str]] = None,
    min_fraction: float = 0.0,
    max_fraction: float = 1.0,
    seed: Optional[int] = 1,
    simulator: str = "epanet",
    epanet_accuracy: float = 1e-8,
) -> Dict[str, object]:
    """
    Simulate the BattLeDIM single-leak prior.

    Each scenario draws one leak node v uniformly from `leak_nodes` (default: all junctions)
    and a magnitude lambda ~ Uniform(min_fraction, max_fraction) * extra_demand, then sets

        d = d_base + lambda * e_v

    so exactly one junction deviates from its base demand.  This is the prior used by
    `scripts/ltown_c_enumerate.py`; training a predictor on any *other* prior (e.g. the
    Dirichlet extra-demand model) and then scoring it against a single-leak posterior is
    incoherent, because the predictor never sees a leak.

    Unlike the sibling `simulate_*_scenarios` functions -- which return only pipe flows,
    all the pipe-bound solvers need -- this returns demands, heads and the leak labels as
    well, because a demand-predicting GNN needs all three.

    Note on solver accuracy: EPANET's default convergence `Accuracy` (0.01 in several of the
    shipped .inp files) is far too loose for low-flow networks such as L-TOWN_C, whose whole
    head spread is ~0.09 m.  `epanet_accuracy` tightens it; pass None to leave the file's
    own setting alone.

    Returns a dict with keys:
        "flows"          {pipe_id: np.ndarray(n_scenarios,)}   (sibling-compatible)
        "demands"        np.ndarray(n_scenarios, n_junctions)
        "heads"          np.ndarray(n_scenarios, n_nodes)
        "leak_index"     np.ndarray(n_scenarios,) int  -- index into junction_names
        "leak_size"      np.ndarray(n_scenarios,) float -- lambda, m3/s
        "junction_names" list[str]
        "node_names"     list[str]
        "base_demands"   np.ndarray(n_junctions,)
    """
    _require_wntr()
    import wntr

    if min_fraction > max_fraction:
        raise ValueError("min_fraction must be <= max_fraction.")

    rng = np.random.default_rng(seed)
    wn = wntr.network.WaterNetworkModel(inp_path)
    wn.options.time.duration = 0
    if epanet_accuracy is not None:
        wn.options.hydraulic.accuracy = float(epanet_accuracy)
        wn.options.hydraulic.trials = 200

    junction_names = [name for name, _ in wn.junctions()]
    node_names = list(wn.node_name_list)
    n_junctions = len(junction_names)

    base_demands = np.zeros(n_junctions, dtype=float)
    for j, name in enumerate(junction_names):
        node = wn.get_node(name)
        if node.demand_timeseries_list is not None and len(node.demand_timeseries_list) > 0:
            base_demands[j] = float(node.demand_timeseries_list[0].base_value or 0.0)

    if leak_nodes is None:
        candidates = list(range(n_junctions))
    else:
        wanted = [str(v) for v in leak_nodes]
        missing = [v for v in wanted if v not in junction_names]
        if missing:
            raise ValueError(f"leak_nodes not in the network: {missing}")
        candidates = [junction_names.index(v) for v in wanted]

    pipe_names = [p for p, _ in wn.pipes()]
    flows_by_pipe = {p: np.zeros(n_scenarios, dtype=float) for p in pipe_names}
    demands = np.zeros((n_scenarios, n_junctions), dtype=float)
    heads = np.zeros((n_scenarios, len(node_names)), dtype=float)
    leak_index = np.zeros(n_scenarios, dtype=int)
    leak_size = np.zeros(n_scenarios, dtype=float)

    if simulator not in ("epanet", "wntr", "auto"):
        raise ValueError("simulator must be 'auto', 'epanet', or 'wntr'.")

    for i in range(n_scenarios):
        v = int(rng.choice(candidates))
        lam = float(rng.uniform(min_fraction, max_fraction)) * float(extra_demand)
        d = base_demands.copy()
        d[v] += lam

        for j, name in enumerate(junction_names):
            node = wn.get_node(name)
            if node.demand_timeseries_list is None or len(node.demand_timeseries_list) == 0:
                continue
            node.demand_timeseries_list[0].base_value = float(d[j])

        sim = wntr.sim.WNTRSimulator(wn) if simulator == "wntr" else wntr.sim.EpanetSimulator(wn)
        results = sim.run_sim()

        flow_df = results.link["flowrate"]
        head_df = results.node["head"]
        t0 = flow_df.index[0]
        for p in pipe_names:
            flows_by_pipe[p][i] = float(flow_df.loc[t0, p])
        heads[i] = np.array([float(head_df.loc[t0, n]) for n in node_names], dtype=float)
        demands[i] = d
        leak_index[i] = v
        leak_size[i] = lam

    # restore the model's base demands
    for j, name in enumerate(junction_names):
        node = wn.get_node(name)
        if node.demand_timeseries_list is None or len(node.demand_timeseries_list) == 0:
            continue
        node.demand_timeseries_list[0].base_value = float(base_demands[j])

    return {
        "flows": flows_by_pipe,
        "demands": demands,
        "heads": heads,
        "leak_index": leak_index,
        "leak_size": leak_size,
        "junction_names": junction_names,
        "node_names": node_names,
        "base_demands": base_demands,
    }
