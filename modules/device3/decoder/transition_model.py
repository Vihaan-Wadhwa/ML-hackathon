"""HMM Transition Probability Model for PathSnap Device 3.

Task 2: Step 2 - Road Network Distance, GPS Displacement, Effective Beta,
and Transition Probability Calculations.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple, Union
import networkx as nx
import numpy as np
import pandas as pd

try:
    from modules.device3.graph.graph_builder import EARTH_RADIUS_METERS, haversine_distance
except (ImportError, ModuleNotFoundError):
    try:
        from graph.graph_builder import EARTH_RADIUS_METERS, haversine_distance
    except (ImportError, ModuleNotFoundError):
        from ..graph.graph_builder import EARTH_RADIUS_METERS, haversine_distance


class ArcProjection(NamedTuple):
    """Result of projecting a GPS coordinate onto a directed road arc."""
    projected_longitude: float
    projected_latitude: float
    fraction_along_arc: float
    perpendicular_distance_m: float


def _extract_gps(gps_point: Any) -> Tuple[float, float, float]:
    """Extract (longitude, latitude, timestamp) from dict, Series, object, or tuple/list."""
    if isinstance(gps_point, (tuple, list)):
        if len(gps_point) < 3:
            raise ValueError("GPS tuple/list must contain at least 3 elements: (lon, lat, timestamp).")
        lon, lat, ts = gps_point[0], gps_point[1], gps_point[2]
    elif isinstance(gps_point, dict):
        lon = gps_point.get("longitude", gps_point.get("lon"))
        lat = gps_point.get("latitude", gps_point.get("lat"))
        ts = gps_point.get("timestamp", gps_point.get("time", gps_point.get("t")))
        if lon is None or lat is None or ts is None:
            raise ValueError(f"GPS dict missing required keys (longitude, latitude, timestamp): {gps_point}")
    elif hasattr(gps_point, "longitude") and hasattr(gps_point, "latitude") and hasattr(gps_point, "timestamp"):
        lon, lat, ts = gps_point.longitude, gps_point.latitude, gps_point.timestamp
    elif hasattr(gps_point, "__getitem__"):
        try:
            lon = gps_point["longitude"] if "longitude" in gps_point else gps_point["lon"]
            lat = gps_point["latitude"] if "latitude" in gps_point else gps_point["lat"]
            ts = gps_point["timestamp"] if "timestamp" in gps_point else gps_point["time"]
        except (KeyError, TypeError) as e:
            raise ValueError(f"Unable to extract GPS fields from {gps_point}: {e}") from e
    else:
        raise TypeError(f"Unsupported GPS point format: {type(gps_point)}")

    lon, lat, ts = float(lon), float(lat), float(ts)
    if not (math.isfinite(lon) and math.isfinite(lat) and math.isfinite(ts)):
        raise ValueError(f"GPS coordinates and timestamp must be finite. Got lon={lon}, lat={lat}, ts={ts}")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"GPS longitude {lon} out of bounds [-180.0, 180.0].")
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"GPS latitude {lat} out of bounds [-90.0, 90.0].")

    return lon, lat, ts


def _extract_coords(pt: Any) -> Tuple[float, float]:
    """Extract (longitude, latitude) from tuple, list, dict, or object."""
    if isinstance(pt, (tuple, list)):
        if len(pt) < 2:
            raise ValueError("Coordinate tuple/list must contain at least 2 elements (lon, lat).")
        return float(pt[0]), float(pt[1])
    elif isinstance(pt, dict):
        lon = pt.get("longitude", pt.get("lon"))
        lat = pt.get("latitude", pt.get("lat"))
        if lon is None or lat is None:
            raise ValueError(f"Coordinate dict missing longitude/latitude: {pt}")
        return float(lon), float(lat)
    elif hasattr(pt, "longitude") and hasattr(pt, "latitude"):
        return float(pt.longitude), float(pt.latitude)
    elif hasattr(pt, "__getitem__"):
        lon = pt["longitude"] if "longitude" in pt else pt["lon"]
        lat = pt["latitude"] if "latitude" in pt else pt["lat"]
        return float(lon), float(lat)
    raise TypeError(f"Cannot extract coordinates from: {type(pt)}")


def _build_arc_lookup_if_needed(
    road_graph: nx.MultiDiGraph,
    arc_lookup: Optional[Dict[int, Tuple[int, int]]] = None,
) -> Dict[int, Tuple[int, int]]:
    """Retrieve or build the mapping of arc_id -> (u, v) on the MultiDiGraph."""
    if arc_lookup is not None:
        return arc_lookup
    if "arc_lookup" in road_graph.graph:
        return road_graph.graph["arc_lookup"]

    lookup: Dict[int, Tuple[int, int]] = {}
    for u, v, k, data in road_graph.edges(keys=True, data=True):
        aid = data.get("arc_id", k)
        lookup[aid] = (u, v)
    road_graph.graph["arc_lookup"] = lookup
    return lookup


def _get_shortest_path_distance(
    road_graph: nx.MultiDiGraph,
    source: int,
    target: int,
    cache: Optional[Dict[Tuple[int, int], float]] = None,
) -> float:
    """Query directed shortest path routing distance on road_graph with weight 'length_m'."""
    if source == target:
        return 0.0

    if cache is not None and (source, target) in cache:
        return cache[(source, target)]

    try:
        dist = nx.shortest_path_length(
            road_graph,
            source=source,
            target=target,
            weight="length_m",
        )
        dist_f = float(dist)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        dist_f = float("inf")

    if cache is not None:
        cache[(source, target)] = dist_f

    return dist_f


def _extract_arc_ids(cands: Any) -> List[int]:
    """Extract integer arc IDs from a list, array, DataFrame, or Series."""
    if isinstance(cands, pd.DataFrame):
        if "arc_id" not in cands.columns:
            raise ValueError(f"Candidates DataFrame must contain 'arc_id' column, got: {list(cands.columns)}")
        return [int(a) for a in cands["arc_id"].values]
    elif isinstance(cands, pd.Series):
        return [int(a) for a in cands.values]
    elif isinstance(cands, (list, tuple, np.ndarray)):
        return [int(a) for a in cands]
    raise TypeError(f"Unsupported candidate container type: {type(cands)}")


def gps_displacement_m(previous_gps: Any, current_gps: Any) -> Tuple[float, float]:
    """Calculate the Haversine distance and time difference between consecutive GPS observations.

    Parameters
    ----------
    previous_gps : Any
        Previous GPS point with longitude, latitude, and timestamp.
    current_gps : Any
        Current GPS point with longitude, latitude, and timestamp.

    Returns
    -------
    displacement_m : float
        Great-circle displacement in meters.
    delta_t_s : float
        Time difference in seconds (t_current - t_previous).

    Raises
    ------
    ValueError
        If coordinates are out of bounds, non-finite, or timestamps are not strictly increasing.
    """
    lon1, lat1, t1 = _extract_gps(previous_gps)
    lon2, lat2, t2 = _extract_gps(current_gps)

    delta_t = t2 - t1
    if delta_t <= 0.0:
        raise ValueError(
            f"GPS timestamps must be strictly increasing: previous t={t1}, "
            f"current t={t2} (delta_t={delta_t}s)."
        )

    d_gps = haversine_distance(lon1, lat1, lon2, lat2)
    return float(d_gps), float(delta_t)


def project_point_onto_arc(
    lon: Union[float, int, np.number],
    lat: Union[float, int, np.number],
    arc_start: Any,
    arc_end: Any,
) -> ArcProjection:
    """Project a GPS point onto a directed road segment using local equirectangular metric projection.

    Methodology
    -----------
    For short road segments (typically < 1-2 km), spherical curvature effects across the segment
    are negligible. We project the GPS point and segment endpoints into a local Cartesian plane
    centered at the segment's mean latitude:
        lat_m = (lat_start + lat_end) / 2
        m_lon = (pi / 180) * R * cos(lat_m)
        m_lat = (pi / 180) * R
    The scalar projection fraction t is calculated via vector dot product:
        t = (P - A) . (B - A) / |B - A|^2
    The fraction is clipped to q in [0.0, 1.0]. The projected geographic point is interpolated,
    and perpendicular distance is calculated via the Haversine formula.

    Zero-length arcs (where start and end nodes share identical coordinates) are handled safely
    by setting q = 0.0 and measuring direct distance to the node.

    Parameters
    ----------
    lon : float
        Longitude of the GPS observation in decimal degrees.
    lat : float
        Latitude of the GPS observation in decimal degrees.
    arc_start : Any
        Coordinates of the start node (lon, lat).
    arc_end : Any
        Coordinates of the end node (lon, lat).

    Returns
    -------
    ArcProjection
        NamedTuple with:
        - projected_longitude: float
        - projected_latitude: float
        - fraction_along_arc: float (clipped to [0.0, 1.0])
        - perpendicular_distance_m: float
    """
    p_lon, p_lat = float(lon), float(lat)
    lon1, lat1 = _extract_coords(arc_start)
    lon2, lat2 = _extract_coords(arc_end)

    for coord, name in [(p_lon, "lon"), (lon1, "start_lon"), (lon2, "end_lon")]:
        if not math.isfinite(coord) or not (-180.0 <= coord <= 180.0):
            raise ValueError(f"Invalid longitude for {name}: {coord}")
    for coord, name in [(p_lat, "lat"), (lat1, "start_lat"), (lat2, "end_lat")]:
        if not math.isfinite(coord) or not (-90.0 <= coord <= 90.0):
            raise ValueError(f"Invalid latitude for {name}: {coord}")

    # Mean latitude in radians
    lat_m_rad = math.radians((lat1 + lat2) / 2.0)
    m_per_deg_lat = (math.pi / 180.0) * EARTH_RADIUS_METERS
    m_per_deg_lon = (math.pi / 180.0) * EARTH_RADIUS_METERS * math.cos(lat_m_rad)

    # Vector from arc_start to point P in meters
    x_p = (p_lon - lon1) * m_per_deg_lon
    y_p = (p_lat - lat1) * m_per_deg_lat

    # Vector of arc segment from arc_start to arc_end in meters
    v_x = (lon2 - lon1) * m_per_deg_lon
    v_y = (lat2 - lat1) * m_per_deg_lat

    seg_len_sq = v_x * v_x + v_y * v_y

    if seg_len_sq < 1e-12:
        # Zero-length arc: coincident start and end nodes
        q = 0.0
        proj_lon = lon1
        proj_lat = lat1
        perp_dist = haversine_distance(p_lon, p_lat, proj_lon, proj_lat)
    else:
        # Scalar projection fraction
        t = (x_p * v_x + y_p * v_y) / seg_len_sq
        q = max(0.0, min(1.0, float(t)))
        proj_lon = lon1 + q * (lon2 - lon1)
        proj_lat = lat1 + q * (lat2 - lat1)
        perp_dist = haversine_distance(p_lon, p_lat, proj_lon, proj_lat)

    return ArcProjection(
        projected_longitude=float(proj_lon),
        projected_latitude=float(proj_lat),
        fraction_along_arc=float(q),
        perpendicular_distance_m=float(perp_dist),
    )


def network_transition_distance(
    previous_arc_id: int,
    current_arc_id: int,
    previous_gps: Any,
    current_gps: Any,
    road_graph: nx.MultiDiGraph,
    shortest_path_cache: Optional[Dict[Tuple[int, int], float]] = None,
    arc_lookup: Optional[Dict[int, Tuple[int, int]]] = None,
) -> float:
    """Calculate directed road network routing distance between two candidate arc projections.

    Formula
    -------
    Let previous candidate arc be u1 -> v1 with length L1 and projection fraction q1.
    Let current candidate arc be u2 -> v2 with length L2 and projection fraction q2.

    For different arcs (previous_arc_id != current_arc_id):
        d_net = (1 - q1) * L1 + shortest_path_length(v1, u2, weight="length_m") + q2 * L2

    For the same arc (previous_arc_id == current_arc_id):
        If q2 >= q1 (forward progress):
            d_net = (q2 - q1) * L1
        If q2 < q1 (reverse movement):
            Checked against directed network connectivity: if a directed return loop
            v1 ->* u1 exists, d_net = (1 - q1) * L1 + shortest_path(v1, u1) + q2 * L1.
            If no directed path exists, returns float('inf').

    Parameters
    ----------
    previous_arc_id : int
        Arc ID of the candidate road segment at observation t-1.
    current_arc_id : int
        Arc ID of the candidate road segment at observation t.
    previous_gps : Any
        GPS observation at t-1 (lon, lat, timestamp).
    current_gps : Any
        GPS observation at t (lon, lat, timestamp).
    road_graph : nx.MultiDiGraph
        Directed road network multigraph.
    shortest_path_cache : Optional[Dict[Tuple[int, int], float]], default=None
        Dictionary cache mapping (source_node, target_node) to shortest path distance in meters.
    arc_lookup : Optional[Dict[int, Tuple[int, int]]], default=None
        Lookup dictionary mapping arc_id -> (u, v).

    Returns
    -------
    float
        Directed network routing distance in meters, or float('inf') if disconnected.

    Raises
    ------
    KeyError
        If previous_arc_id or current_arc_id is not found in the road graph.
    """
    lookup = _build_arc_lookup_if_needed(road_graph, arc_lookup)

    if previous_arc_id not in lookup:
        raise KeyError(f"previous_arc_id {previous_arc_id} not found in road graph.")
    if current_arc_id not in lookup:
        raise KeyError(f"current_arc_id {current_arc_id} not found in road graph.")

    u1, v1 = lookup[previous_arc_id]
    u2, v2 = lookup[current_arc_id]

    if previous_arc_id in road_graph[u1][v1]:
        edge_data1 = road_graph[u1][v1][previous_arc_id]
    else:
        edge_data1 = None
        for _, d in road_graph[u1][v1].items():
            if d.get("arc_id") == previous_arc_id:
                edge_data1 = d
                break
        if edge_data1 is None:
            edge_data1 = list(road_graph[u1][v1].values())[0]

    if current_arc_id in road_graph[u2][v2]:
        edge_data2 = road_graph[u2][v2][current_arc_id]
    else:
        edge_data2 = None
        for _, d in road_graph[u2][v2].items():
            if d.get("arc_id") == current_arc_id:
                edge_data2 = d
                break
        if edge_data2 is None:
            edge_data2 = list(road_graph[u2][v2].values())[0]

    L1 = float(edge_data1.get("length_m", edge_data1.get("length", 0.0)))
    L2 = float(edge_data2.get("length_m", edge_data2.get("length", 0.0)))

    coord_u1 = (road_graph.nodes[u1]["longitude"], road_graph.nodes[u1]["latitude"])
    coord_v1 = (road_graph.nodes[v1]["longitude"], road_graph.nodes[v1]["latitude"])
    coord_u2 = (road_graph.nodes[u2]["longitude"], road_graph.nodes[u2]["latitude"])
    coord_v2 = (road_graph.nodes[v2]["longitude"], road_graph.nodes[v2]["latitude"])

    lon1, lat1, _ = _extract_gps(previous_gps)
    lon2, lat2, _ = _extract_gps(current_gps)

    proj1 = project_point_onto_arc(lon1, lat1, coord_u1, coord_v1)
    proj2 = project_point_onto_arc(lon2, lat2, coord_u2, coord_v2)

    q1 = proj1.fraction_along_arc
    q2 = proj2.fraction_along_arc

    if previous_arc_id == current_arc_id:
        if q2 >= q1:
            # Forward motion along same directed arc
            d_net = (q2 - q1) * L1
        else:
            # Reverse motion on directed arc: requires network return loop
            d_loop = _get_shortest_path_distance(road_graph, v1, u1, shortest_path_cache)
            if math.isinf(d_loop):
                return float("inf")
            d_net = (1.0 - q1) * L1 + d_loop + q2 * L1
    else:
        # Different arcs
        if v1 == u2:
            # Directly adjacent consecutive arcs
            d_between = 0.0
        else:
            d_between = _get_shortest_path_distance(road_graph, v1, u2, shortest_path_cache)

        if math.isinf(d_between):
            return float("inf")

        d_net = (1.0 - q1) * L1 + d_between + q2 * L2

    return float(max(0.0, d_net))


def calculate_effective_beta(
    beta: float,
    delta_t: float,
    dropout_threshold_s: float = 5.0,
    reference_dt_s: float = 1.0,
) -> float:
    """Calculate effective transition beta scaling for GPS dropouts.

    When delta_t > dropout_threshold_s (default 5.0s), transition uncertainty increases
    as the vehicle could have traversed complex routes during the outage. The scale
    parameter beta is scaled proportionally:
        effective_beta = beta * (delta_t / reference_dt_s)

    For normal intervals (delta_t <= dropout_threshold_s), the unscaled beta is used.

    Parameters
    ----------
    beta : float
        Nominal transition scale parameter. Must be positive.
    delta_t : float
        Time difference between consecutive GPS observations in seconds. Must be positive.
    dropout_threshold_s : float, default=5.0
        Threshold in seconds above which dropout scaling is applied.
    reference_dt_s : float, default=1.0
        Reference time delta in seconds for linear scaling.

    Returns
    -------
    float
        Effective beta value in meters.
    """
    if beta <= 0.0 or not math.isfinite(beta):
        raise ValueError(f"beta must be a positive finite float, got: {beta}")
    if delta_t <= 0.0 or not math.isfinite(delta_t):
        raise ValueError(f"delta_t must be a positive finite float, got: {delta_t}")
    if dropout_threshold_s < 0.0 or not math.isfinite(dropout_threshold_s):
        raise ValueError(f"dropout_threshold_s must be non-negative and finite, got: {dropout_threshold_s}")

    if delta_t > dropout_threshold_s:
        return float(beta * (delta_t / reference_dt_s))
    return float(beta)


def transition_probability(
    d_net: float,
    d_gps: float,
    beta: float = 5.0,
    epsilon: float = 1e-6,
) -> float:
    """Calculate the HMM transition probability density using exponential decay.

    Formula
    -------
        P = (1 / beta) * exp(-|d_net - d_gps| / beta)

    Notes
    -----
    This value represents a continuous transition probability density (Newson & Krumm, 2009).
    It is not a normalized categorical distribution across all destination candidates;
    normalization occurs downstream if required.

    Parameters
    ----------
    d_net : float
        Directed road network routing distance in meters, or float('inf') if disconnected.
    d_gps : float
        Great-circle displacement between consecutive GPS points in meters.
    beta : float, default=5.0
        Transition probability scale parameter in meters. Must be positive.
    epsilon : float, default=1e-6
        Floor probability returned for disconnected or highly improbable transitions.

    Returns
    -------
    float
        Transition probability density, floored at epsilon.

    Raises
    ------
    ValueError
        If beta <= 0, epsilon <= 0, or d_gps < 0 or non-finite.
    """
    if beta <= 0.0 or not math.isfinite(beta):
        raise ValueError(f"beta must be a positive finite float, got: {beta}")
    if epsilon <= 0.0 or not math.isfinite(epsilon):
        raise ValueError(f"epsilon must be a positive finite float, got: {epsilon}")
    if d_gps < 0.0 or not math.isfinite(d_gps):
        raise ValueError(f"d_gps must be a non-negative finite float, got: {d_gps}")

    # Handle disconnected transition
    if math.isinf(d_net) or d_net < 0.0 or not math.isfinite(d_net):
        return float(epsilon)

    diff = abs(d_net - d_gps)
    exponent = -diff / beta

    # Numerical underflow protection
    if exponent < -700.0:
        return float(epsilon)

    p = (1.0 / beta) * math.exp(exponent)
    return float(max(epsilon, p))


def log_transition_probability(
    d_net: float,
    d_gps: float,
    beta: float = 5.0,
    epsilon: float = 1e-6,
) -> float:
    """Calculate the log-space HMM transition probability score for Viterbi decoding.

    Formula
    -------
        ln(P) = -ln(beta) - (|d_net - d_gps| / beta)

    Parameters
    ----------
    d_net : float
        Directed road network routing distance in meters, or float('inf') if disconnected.
    d_gps : float
        Great-circle displacement between consecutive GPS points in meters.
    beta : float, default=5.0
        Transition probability scale parameter. Must be positive.
    epsilon : float, default=1e-6
        Floor probability for disconnected transitions.

    Returns
    -------
    float
        Log-transition score, floored at ln(epsilon).
    """
    min_log = math.log(epsilon)

    if beta <= 0.0 or not math.isfinite(beta):
        raise ValueError(f"beta must be a positive finite float, got: {beta}")
    if epsilon <= 0.0 or not math.isfinite(epsilon):
        raise ValueError(f"epsilon must be a positive finite float, got: {epsilon}")
    if d_gps < 0.0 or not math.isfinite(d_gps):
        raise ValueError(f"d_gps must be a non-negative finite float, got: {d_gps}")

    if math.isinf(d_net) or d_net < 0.0 or not math.isfinite(d_net):
        return float(min_log)

    diff = abs(d_net - d_gps)
    log_val = -math.log(beta) - (diff / beta)
    return float(max(min_log, log_val))


def build_transition_matrix(
    previous_candidates: Union[List[int], np.ndarray, pd.DataFrame, pd.Series],
    current_candidates: Union[List[int], np.ndarray, pd.DataFrame, pd.Series],
    previous_gps: Any,
    current_gps: Any,
    road_graph: nx.MultiDiGraph,
    beta: float = 5.0,
    shortest_path_cache: Optional[Dict[Tuple[int, int], float]] = None,
    arc_lookup: Optional[Dict[int, Tuple[int, int]]] = None,
    dropout_threshold_s: float = 5.0,
    epsilon: float = 1e-6,
) -> Dict[str, Any]:
    """Build the HMM transition probability matrix between candidate road segments.

    Calculates directed routing distances and transition probability densities
    for all N x M candidate pairs between consecutive GPS observations.

    Parameters
    ----------
    previous_candidates : Union[List[int], np.ndarray, pd.DataFrame, pd.Series]
        Candidate arc IDs or candidate DataFrame at observation t-1 (length N).
    current_candidates : Union[List[int], np.ndarray, pd.DataFrame, pd.Series]
        Candidate arc IDs or candidate DataFrame at observation t (length M).
    previous_gps : Any
        GPS observation at t-1 (lon, lat, timestamp).
    current_gps : Any
        GPS observation at t (lon, lat, timestamp).
    road_graph : nx.MultiDiGraph
        Directed road network MultiDiGraph.
    beta : float, default=5.0
        Nominal transition scale parameter.
    shortest_path_cache : Optional[Dict[Tuple[int, int], float]], default=None
        Reusable cache for shortest path queries across observations.
    arc_lookup : Optional[Dict[int, Tuple[int, int]]], default=None
        Lookup dictionary mapping arc_id -> (u, v).
    dropout_threshold_s : float, default=5.0
        Threshold in seconds above which beta scales with delta_t.
    epsilon : float, default=1e-6
        Floor probability for disconnected transitions.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - transition_matrix: np.ndarray of shape (N, M)
        - log_transition_matrix: np.ndarray of shape (N, M)
        - network_distances: np.ndarray of shape (N, M)
        - previous_arcs: List[int] of length N (row indices)
        - current_arcs: List[int] of length M (column indices)
        - d_gps: float (GPS displacement in meters)
        - delta_t: float (observation time delta in seconds)
        - effective_beta: float (dropout-adjusted beta)
        - disconnected_transitions: List[Tuple[int, int]] of unreachable arc pairs
        - num_disconnected: int
        - shortest_path_cache_size: int
    """
    prev_arcs = _extract_arc_ids(previous_candidates)
    curr_arcs = _extract_arc_ids(current_candidates)

    d_gps, delta_t = gps_displacement_m(previous_gps, current_gps)
    eff_beta = calculate_effective_beta(beta, delta_t, dropout_threshold_s)

    N = len(prev_arcs)
    M = len(curr_arcs)

    trans_matrix = np.zeros((N, M), dtype=np.float64)
    log_trans_matrix = np.zeros((N, M), dtype=np.float64)
    net_distances = np.zeros((N, M), dtype=np.float64)
    disconnected_list: List[Tuple[int, int]] = []

    if shortest_path_cache is None:
        shortest_path_cache = {}

    lookup = _build_arc_lookup_if_needed(road_graph, arc_lookup)

    for i, a1 in enumerate(prev_arcs):
        for j, a2 in enumerate(curr_arcs):
            d_net = network_transition_distance(
                previous_arc_id=a1,
                current_arc_id=a2,
                previous_gps=previous_gps,
                current_gps=current_gps,
                road_graph=road_graph,
                shortest_path_cache=shortest_path_cache,
                arc_lookup=lookup,
            )
            net_distances[i, j] = d_net
            if math.isinf(d_net):
                disconnected_list.append((a1, a2))

            p = transition_probability(d_net, d_gps, beta=eff_beta, epsilon=epsilon)
            log_p = log_transition_probability(d_net, d_gps, beta=eff_beta, epsilon=epsilon)

            trans_matrix[i, j] = p
            log_trans_matrix[i, j] = log_p

    return {
        "transition_matrix": trans_matrix,
        "log_transition_matrix": log_trans_matrix,
        "network_distances": net_distances,
        "previous_arcs": prev_arcs,
        "current_arcs": curr_arcs,
        "d_gps": d_gps,
        "delta_t": delta_t,
        "effective_beta": eff_beta,
        "disconnected_transitions": disconnected_list,
        "num_disconnected": len(disconnected_list),
        "shortest_path_cache_size": len(shortest_path_cache),
    }


def normalize_transition_matrix(transition_matrix: np.ndarray) -> np.ndarray:
    """Normalize row-wise to convert transition densities into categorical probability distributions.

    Notes
    -----
    By default, PathSnap HMM formulations use transition probability densities.
    This optional function normalizes each row i such that sum_j P(r_t^(j) | r_{t-1}^(i)) = 1.0.

    Parameters
    ----------
    transition_matrix : np.ndarray
        2D array of transition probabilities of shape (N, M).

    Returns
    -------
    np.ndarray
        Row-normalized 2D array of shape (N, M).
    """
    if transition_matrix.ndim != 2:
        raise ValueError(f"Expected 2D transition matrix, got shape {transition_matrix.shape}")

    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums_safe = np.where(row_sums > 0, row_sums, 1.0)
    return transition_matrix / row_sums_safe
