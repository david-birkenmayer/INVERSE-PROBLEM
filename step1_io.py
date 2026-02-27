from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math


@dataclass(frozen=True)
class NodeData:
    node_id: str
    elevation_m: float
    base_demand: float
    node_type: str  # junction | reservoir | tank


@dataclass(frozen=True)
class PipeData:
    pipe_id: str
    start_node: str
    end_node: str
    length_m: float
    diameter_m: float
    roughness: float
    minor_loss: float
    status: str


@dataclass(frozen=True)
class NetworkData:
    nodes: Dict[str, NodeData]
    pipes: Dict[str, PipeData]
    reservoirs: Dict[str, NodeData]
    junctions: Dict[str, NodeData]
    tanks: Dict[str, NodeData]
    options: Dict[str, str]


def _require_wntr() -> None:
    try:
        import wntr  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'wntr' package is required to load EPANET .inp files. "
            "Install it with: pip install wntr"
        ) from exc


def load_inp_network(file_path: str) -> NetworkData:
    """
    Load an EPANET .inp file using WNTR and extract node/pipe data.

    Returns values in SI units as provided by WNTR (length: m, diameter: m).
    """
    _require_wntr()
    import wntr

    wn = wntr.network.WaterNetworkModel(file_path)

    nodes: Dict[str, NodeData] = {}
    reservoirs: Dict[str, NodeData] = {}
    junctions: Dict[str, NodeData] = {}
    tanks: Dict[str, NodeData] = {}

    for name, node in wn.nodes():
        if node.node_type == "Junction":
            demand = float(node.base_demand) if node.base_demand is not None else 0.0
            n = NodeData(name, float(node.elevation), demand, "junction")
            junctions[name] = n
            nodes[name] = n
        elif node.node_type == "Reservoir":
            head_value = node.head if node.head is not None else getattr(node, "elevation", None)
            head_value = 0.0 if head_value is None else float(head_value)
            n = NodeData(name, head_value, 0.0, "reservoir")
            reservoirs[name] = n
            nodes[name] = n
        elif node.node_type == "Tank":
            n = NodeData(name, float(node.elevation), 0.0, "tank")
            tanks[name] = n
            nodes[name] = n

    pipes: Dict[str, PipeData] = {}
    for name, link in wn.links():
        if link.link_type != "Pipe":
            continue
        pipes[name] = PipeData(
            pipe_id=name,
            start_node=link.start_node_name,
            end_node=link.end_node_name,
            length_m=float(link.length),
            diameter_m=float(link.diameter),
            roughness=float(link.roughness),
            minor_loss=float(link.minor_loss),
            status=str(link.status),
        )

    options = {k: str(v) for k, v in wn.options.to_dict().items()}
    return NetworkData(
        nodes=nodes,
        pipes=pipes,
        reservoirs=reservoirs,
        junctions=junctions,
        tanks=tanks,
        options=options,
    )


def darcy_weisbach_resistance(
    length_m: float,
    diameter_m: float,
    roughness_m: float,
    q_nominal_m3s: Optional[float] = None,
    v_nominal_ms: float = 1.0,
    nu_m2s: float = 1.0e-6,
    g_ms2: float = 9.80665,
) -> Tuple[float, float, float]:
    """
    Compute r_e for Darcy–Weisbach: h = r_e * |q| q.

    Returns (r_e, f, Re).
    """
    if diameter_m <= 0 or length_m <= 0:
        raise ValueError("Pipe length and diameter must be positive.")

    area = math.pi * (diameter_m ** 2) / 4.0
    if q_nominal_m3s is None:
        v = max(v_nominal_ms, 1.0e-3)
    else:
        v = max(abs(q_nominal_m3s) / area, 1.0e-6)

    Re = max(v * diameter_m / nu_m2s, 1.0)
    rel_rough = max(roughness_m / diameter_m, 0.0)
    f = 0.25 / (math.log10(rel_rough / 3.7 + 5.74 / (Re ** 0.9))) ** 2
    r_e = 8.0 * f * length_m / (math.pi ** 2 * g_ms2 * diameter_m ** 5)
    return r_e, f, Re


def compute_pipe_resistances(
    network: NetworkData,
    q_nominal_by_pipe: Optional[Dict[str, float]] = None,
    v_nominal_ms: float = 1.0,
    roughness_unit_factor: float = 1.0e-3,
) -> Dict[str, Dict[str, float]]:
    """
    Compute Darcy–Weisbach resistance r_e for each pipe.

    roughness_unit_factor converts the .inp roughness to meters.
    For Darcy–Weisbach EPANET expects roughness in mm, so 1e-3 is typical.
    """
    results: Dict[str, Dict[str, float]] = {}
    for pipe_id, pipe in network.pipes.items():
        q_nom = None if q_nominal_by_pipe is None else q_nominal_by_pipe.get(pipe_id)
        roughness_m = pipe.roughness * roughness_unit_factor
        r_e, f, Re = darcy_weisbach_resistance(
            length_m=pipe.length_m,
            diameter_m=pipe.diameter_m,
            roughness_m=roughness_m,
            q_nominal_m3s=q_nom,
            v_nominal_ms=v_nominal_ms,
        )
        results[pipe_id] = {
            "r_e": r_e,
            "f": f,
            "Re": Re,
        }
    return results


def load_alperovits_network() -> NetworkData:
    """Convenience loader for the Alperovits test case in ./wdn."""
    return load_inp_network("./wdn/Alperovits.inp")
