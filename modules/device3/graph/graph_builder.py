"""Road Topology and Graph Construction for PathSnap Device 3.

Task 1: Steps 2, 3 & 4 - Data Loading, Validation, Haversine Distance, and MultiDiGraph Construction.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union
import networkx as nx
import numpy as np
import pandas as pd

EARTH_RADIUS_METERS: float = 6_371_000.0


class RoadDataValidationError(ValueError):
    """Custom exception raised when road network data violates schema or integrity constraints."""
    pass


def haversine_distance(
    lon1: Union[float, int, np.number],
    lat1: Union[float, int, np.number],
    lon2: Union[float, int, np.number],
    lat2: Union[float, int, np.number],
) -> float:
    """Calculate the great-circle geographic distance between two points in meters.

    Uses the spherical Haversine formula with Earth's mean radius of 6,371,000 meters.

    Parameters
    ----------
    lon1 : Union[float, int, np.number]
        Longitude of the first point in decimal degrees [-180.0, 180.0].
    lat1 : Union[float, int, np.number]
        Latitude of the first point in decimal degrees [-90.0, 90.0].
    lon2 : Union[float, int, np.number]
        Longitude of the second point in decimal degrees [-180.0, 180.0].
    lat2 : Union[float, int, np.number]
        Latitude of the second point in decimal degrees [-90.0, 90.0].

    Returns
    -------
    float
        Great-circle distance in meters.

    Raises
    ------
    ValueError
        If any coordinate is NaN, Inf, non-numeric, or outside valid geographic bounds.
    """
    try:
        f_lon1 = float(lon1)
        f_lat1 = float(lat1)
        f_lon2 = float(lon2)
        f_lat2 = float(lat2)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Coordinates must be numeric: {e}") from e

    # Check finite
    if not (
        math.isfinite(f_lon1)
        and math.isfinite(f_lat1)
        and math.isfinite(f_lon2)
        and math.isfinite(f_lat2)
    ):
        raise ValueError("Coordinates must be finite (not NaN or Inf).")

    # Geographical range check
    if not (-180.0 <= f_lon1 <= 180.0):
        raise ValueError(f"lon1 out of bounds [-180, 180]: {f_lon1}")
    if not (-180.0 <= f_lon2 <= 180.0):
        raise ValueError(f"lon2 out of bounds [-180, 180]: {f_lon2}")
    if not (-90.0 <= f_lat1 <= 90.0):
        raise ValueError(f"lat1 out of bounds [-90, 90]: {f_lat1}")
    if not (-90.0 <= f_lat2 <= 90.0):
        raise ValueError(f"lat2 out of bounds [-90, 90]: {f_lat2}")

    # Convert decimal degrees to radians
    phi1 = math.radians(f_lat1)
    phi2 = math.radians(f_lat2)
    delta_phi = math.radians(f_lat2 - f_lat1)
    delta_lambda = math.radians(f_lon2 - f_lon1)

    # Haversine formula
    sin_dphi_2 = math.sin(delta_phi / 2.0)
    sin_dlambda_2 = math.sin(delta_lambda / 2.0)

    a = sin_dphi_2 * sin_dphi_2 + math.cos(phi1) * math.cos(phi2) * sin_dlambda_2 * sin_dlambda_2

    # Clamp intermediate value to [0.0, 1.0] to safeguard against numerical inaccuracy
    a = max(0.0, min(1.0, a))

    c = 2.0 * math.asin(math.sqrt(a))
    distance_m = EARTH_RADIUS_METERS * c

    return float(distance_m)


def load_nodes(nodes_path: Union[str, Path]) -> pd.DataFrame:
    """Load and validate the road network .nodes file.

    Parameters
    ----------
    nodes_path : Union[str, Path]
        Path to the .nodes file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['node_id', 'longitude', 'latitude'].

    Raises
    ------
    FileNotFoundError
        If nodes_path does not exist.
    ValueError
        If the file has invalid column counts, missing values, non-finite values,
        or coordinates out of geographical bounds.
    """
    path = Path(nodes_path)
    if not path.is_file():
        raise FileNotFoundError(f"Nodes file not found: {path}")

    try:
        df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    except pd.errors.ParserError as e:
        raise ValueError(f"Malformed nodes file at {path}: {e}") from e

    if df.empty:
        raise ValueError(f"Nodes file is empty: {path}")

    if df.shape[1] != 2:
        raise ValueError(
            f"Malformed nodes file at {path}: expected exactly 2 columns (longitude, latitude), "
            f"found {df.shape[1]} columns."
        )

    # Check for missing values
    if df.isna().any().any():
        raise ValueError(f"Nodes file at {path} contains missing or unparseable values.")

    # Convert coordinates to numeric and float64
    lon = pd.to_numeric(df[0], errors="coerce")
    lat = pd.to_numeric(df[1], errors="coerce")

    if lon.isna().any() or lat.isna().any():
        raise ValueError(f"Nodes file at {path} contains non-numeric coordinates.")

    if not np.all(np.isfinite(lon.values)) or not np.all(np.isfinite(lat.values)):
        raise ValueError(f"Nodes file at {path} contains non-finite coordinates (NaN or Inf).")

    # Geographical range check
    invalid_lon = (lon < -180.0) | (lon > 180.0)
    if invalid_lon.any():
        first_bad = lon[invalid_lon].iloc[0]
        raise ValueError(
            f"Invalid longitude in {path}: {first_bad}. Longitude must be between -180.0 and 180.0 degrees."
        )

    invalid_lat = (lat < -90.0) | (lat > 90.0)
    if invalid_lat.any():
        first_bad = lat[invalid_lat].iloc[0]
        raise ValueError(
            f"Invalid latitude in {path}: {first_bad}. Latitude must be between -90.0 and 90.0 degrees."
        )

    node_ids = np.arange(len(df), dtype=np.int64)
    nodes_df = pd.DataFrame({
        "node_id": node_ids,
        "longitude": lon.astype(np.float64),
        "latitude": lat.astype(np.float64),
    })

    return nodes_df


def load_arcs(arcs_path: Union[str, Path]) -> pd.DataFrame:
    """Load and validate the road network .arcs file.

    Parameters
    ----------
    arcs_path : Union[str, Path]
        Path to the .arcs file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['arc_id', 'from_node_id', 'to_node_id'].

    Raises
    ------
    FileNotFoundError
        If arcs_path does not exist.
    ValueError
        If the file has invalid column counts, missing values, non-integer values,
        or negative IDs.
    """
    path = Path(arcs_path)
    if not path.is_file():
        raise FileNotFoundError(f"Arcs file not found: {path}")

    try:
        df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    except pd.errors.ParserError as e:
        raise ValueError(f"Malformed arcs file at {path}: {e}") from e

    if df.empty:
        raise ValueError(f"Arcs file is empty: {path}")

    if df.shape[1] != 2:
        raise ValueError(
            f"Malformed arcs file at {path}: expected exactly 2 columns (from_node_id, to_node_id), "
            f"found {df.shape[1]} columns."
        )

    if df.isna().any().any():
        raise ValueError(f"Arcs file at {path} contains missing or unparseable values.")

    from_node = pd.to_numeric(df[0], errors="coerce")
    to_node = pd.to_numeric(df[1], errors="coerce")

    if from_node.isna().any() or to_node.isna().any():
        raise ValueError(f"Arcs file at {path} contains non-numeric node IDs.")

    # Check integers
    if ((from_node % 1 != 0) | (to_node % 1 != 0)).any():
        raise ValueError(f"Arcs file at {path} contains non-integer node IDs.")

    # Check non-negative
    if (from_node < 0).any() or (to_node < 0).any():
        raise ValueError(f"Arcs file at {path} contains negative node IDs.")

    arc_ids = np.arange(len(df), dtype=np.int64)
    arcs_df = pd.DataFrame({
        "arc_id": arc_ids,
        "from_node_id": from_node.astype(np.int64),
        "to_node_id": to_node.astype(np.int64),
    })

    return arcs_df


def validate_road_data(
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    strict: bool = False,
) -> Dict[str, Any]:
    """Validate relational integrity and topological properties of nodes and arcs.

    Parameters
    ----------
    nodes_df : pd.DataFrame
        DataFrame of loaded nodes containing ['node_id', 'longitude', 'latitude'].
    arcs_df : pd.DataFrame
        DataFrame of loaded arcs containing ['arc_id', 'from_node_id', 'to_node_id'].
    strict : bool, default=False
        If True, raises RoadDataValidationError upon detecting fatal schema or
        referential integrity violations.

    Returns
    -------
    Dict[str, Any]
        Dictionary reporting validation results and topology characteristics.

    Raises
    ------
    RoadDataValidationError
        If strict=True and any fatal errors are identified.
    """
    errors: List[str] = []

    # 1. Duplicate IDs
    dup_node_ids = int(nodes_df["node_id"].duplicated().sum())
    if dup_node_ids > 0:
        errors.append(f"Found {dup_node_ids} duplicate node IDs in nodes_df.")

    dup_arc_ids = int(arcs_df["arc_id"].duplicated().sum())
    if dup_arc_ids > 0:
        errors.append(f"Found {dup_arc_ids} duplicate arc IDs in arcs_df.")

    # 2. Coordinates validity
    lon = nodes_df["longitude"]
    lat = nodes_df["latitude"]
    invalid_coords_mask = (
        lon.isna()
        | lat.isna()
        | ~np.isfinite(lon)
        | ~np.isfinite(lat)
        | (lon < -180.0)
        | (lon > 180.0)
        | (lat < -90.0)
        | (lat > 90.0)
    )
    invalid_coords_count = int(invalid_coords_mask.sum())
    if invalid_coords_count > 0:
        errors.append(f"Found {invalid_coords_count} nodes with invalid coordinates.")

    # 3. Negative IDs
    neg_nodes = int((nodes_df["node_id"] < 0).sum())
    neg_from = int((arcs_df["from_node_id"] < 0).sum())
    neg_to = int((arcs_df["to_node_id"] < 0).sum())
    total_neg_ids = neg_nodes + neg_from + neg_to
    if total_neg_ids > 0:
        errors.append(f"Found negative node IDs (nodes: {neg_nodes}, from: {neg_from}, to: {neg_to}).")

    # 4. Missing node references in arcs
    valid_node_ids: Set[int] = set(nodes_df["node_id"])
    missing_from_nodes: List[int] = sorted(
        list(set(arcs_df["from_node_id"]) - valid_node_ids)
    )
    missing_to_nodes: List[int] = sorted(
        list(set(arcs_df["to_node_id"]) - valid_node_ids)
    )

    if missing_from_nodes:
        errors.append(
            f"Found {len(missing_from_nodes)} from_node_id references to non-existent nodes: {missing_from_nodes[:5]}..."
        )
    if missing_to_nodes:
        errors.append(
            f"Found {len(missing_to_nodes)} to_node_id references to non-existent nodes: {missing_to_nodes[:5]}..."
        )

    # 5. Self-loops (from_node == to_node)
    self_loop_mask = arcs_df["from_node_id"] == arcs_df["to_node_id"]
    self_loops_count = int(self_loop_mask.sum())
    self_loop_arc_ids: List[int] = arcs_df.loc[self_loop_mask, "arc_id"].tolist()

    # 6. Duplicate directed endpoint pairs (parallel edges)
    pair_counts = arcs_df.groupby(["from_node_id", "to_node_id"]).size()
    dup_pair_groups = pair_counts[pair_counts > 1]
    num_duplicate_groups = len(dup_pair_groups)
    num_extra_duplicate_arcs = int((dup_pair_groups - 1).sum())

    # 7. Reverse-direction connections
    directed_pairs: Set[Tuple[int, int]] = set(
        zip(arcs_df["from_node_id"], arcs_df["to_node_id"])
    )
    reverse_pairs = {
        (u, v) for (u, v) in directed_pairs if (v, u) in directed_pairs and u != v
    }
    num_reverse_direction_connections = len(reverse_pairs) // 2

    # 8. Isolated nodes (degree 0)
    active_nodes: Set[int] = set(arcs_df["from_node_id"]).union(set(arcs_df["to_node_id"]))
    isolated_nodes: List[int] = sorted(list(valid_node_ids - active_nodes))
    num_isolated_nodes = len(isolated_nodes)

    is_valid = len(errors) == 0

    report: Dict[str, Any] = {
        "is_valid": is_valid,
        "errors": errors,
        "node_count": len(nodes_df),
        "arc_count": len(arcs_df),
        "duplicate_node_ids_count": dup_node_ids,
        "duplicate_arc_ids_count": dup_arc_ids,
        "invalid_coordinates_count": invalid_coords_count,
        "negative_node_ids_count": total_neg_ids,
        "missing_from_node_references": missing_from_nodes,
        "missing_to_node_references": missing_to_nodes,
        "self_loops_count": self_loops_count,
        "self_loop_arc_ids": self_loop_arc_ids,
        "duplicate_endpoint_groups_count": num_duplicate_groups,
        "extra_duplicate_arcs_count": num_extra_duplicate_arcs,
        "reverse_direction_connections_count": num_reverse_direction_connections,
        "isolated_nodes_count": num_isolated_nodes,
        "isolated_node_ids": isolated_nodes[:10],
    }

    if strict and not is_valid:
        raise RoadDataValidationError(
            f"Road network validation failed with {len(errors)} error(s):\n" + "\n".join(f"- {e}" for e in errors)
        )

    return report


def get_graph_statistics(road_graph: nx.MultiDiGraph) -> Dict[str, Any]:
    """Calculate comprehensive topological and metric statistics for a road network MultiDiGraph.

    Parameters
    ----------
    road_graph : nx.MultiDiGraph
        The directed road network graph.

    Returns
    -------
    Dict[str, Any]
        Dictionary of graph statistics including component counts, parallel arc counts,
        and length distributions.
    """
    total_nodes = int(road_graph.number_of_nodes())
    total_directed_arcs = int(road_graph.number_of_edges())

    # Isolated nodes (in-degree == 0 and out-degree == 0)
    isolated_nodes = int(nx.number_of_isolates(road_graph))

    # Connected components
    # In directed graphs, weakly connected components represent reachability in the underlying undirected graph.
    weakly_connected = int(nx.number_weakly_connected_components(road_graph))
    strongly_connected = int(nx.number_strongly_connected_components(road_graph))

    # Self loops
    self_loops = int(nx.number_of_selfloops(road_graph))

    # Parallel edge statistics
    # In MultiDiGraph, road_graph[u][v] contains a dict of {key: edge_data}
    num_dup_groups = 0
    num_extra_parallel_arcs = 0
    for u in road_graph:
        for v in road_graph[u]:
            edge_count = len(road_graph[u][v])
            if edge_count > 1:
                num_dup_groups += 1
                num_extra_parallel_arcs += (edge_count - 1)

    # Arc lengths
    lengths: List[float] = [
        float(data["length_m"])
        for _, _, _, data in road_graph.edges(keys=True, data=True)
    ]

    if lengths:
        total_length_m = float(sum(lengths))
        min_length_m = float(min(lengths))
        max_length_m = float(max(lengths))
        avg_length_m = float(total_length_m / len(lengths))
    else:
        total_length_m = 0.0
        min_length_m = 0.0
        max_length_m = 0.0
        avg_length_m = 0.0

    return {
        "total_nodes": total_nodes,
        "total_directed_arcs": total_directed_arcs,
        "isolated_nodes": isolated_nodes,
        "weakly_connected_components": weakly_connected,
        "strongly_connected_components": strongly_connected,
        "duplicate_endpoint_groups": num_dup_groups,
        "extra_parallel_arcs": num_extra_parallel_arcs,
        "self_loops": self_loops,
        "total_directed_length_m": total_length_m,
        "min_arc_length_m": min_length_m,
        "max_arc_length_m": max_length_m,
        "avg_arc_length_m": avg_length_m,
    }


def build_road_graph(
    nodes_path: Union[str, Path],
    arcs_path: Union[str, Path],
) -> Tuple[nx.MultiDiGraph, Dict[int, Tuple[int, int]], Dict[str, Any]]:
    """Construct a directed NetworkX MultiDiGraph representing the road network.

    Parameters
    ----------
    nodes_path : Union[str, Path]
        Path to the .nodes file.
    arcs_path : Union[str, Path]
        Path to the .arcs file.

    Returns
    -------
    road_graph : nx.MultiDiGraph
        Directed multigraph where nodes have attributes (node_id, longitude, latitude)
        and directed edges have attributes (arc_id, length_m, from_node_id, to_node_id)
        with arc_id as the edge key.
    arc_lookup : Dict[int, Tuple[int, int]]
        Mapping from arc_id to (from_node_id, to_node_id).
    graph_statistics : Dict[str, Any]
        Topological and geometric summary statistics of the constructed graph.

    Raises
    ------
    FileNotFoundError
        If nodes_path or arcs_path does not exist.
    RoadDataValidationError
        If road data fails strict validation (e.g. missing node references, invalid coordinates).
    """
    # 1. Load data
    nodes_df = load_nodes(nodes_path)
    arcs_df = load_arcs(arcs_path)

    # 2. Strict validation: aborts before graph creation if integrity violations exist
    validate_road_data(nodes_df, arcs_df, strict=True)

    # 3. Create MultiDiGraph
    road_graph = nx.MultiDiGraph()

    # Pre-extract arrays for maximum construction efficiency
    node_ids = nodes_df["node_id"].values
    lons = nodes_df["longitude"].values
    lats = nodes_df["latitude"].values

    # Batch add all valid nodes
    nodes_to_add = [
        (
            int(node_ids[i]),
            {
                "node_id": int(node_ids[i]),
                "longitude": float(lons[i]),
                "latitude": float(lats[i]),
            },
        )
        for i in range(len(node_ids))
    ]
    road_graph.add_nodes_from(nodes_to_add)

    # Pre-build coordinate lookup for fast Haversine distance calculation
    # Using 1D array indexing since node_id is sequential 0..N-1
    arc_ids = arcs_df["arc_id"].values
    from_nodes = arcs_df["from_node_id"].values
    to_nodes = arcs_df["to_node_id"].values

    # Build arc lookup: arc_id -> (from_node_id, to_node_id)
    arc_lookup: Dict[int, Tuple[int, int]] = {}
    edges_to_add = []

    for i in range(len(arc_ids)):
        arc_id = int(arc_ids[i])
        u = int(from_nodes[i])
        v = int(to_nodes[i])

        arc_lookup[arc_id] = (u, v)

        # Retrieve geographic coordinates
        lon1 = lons[u]
        lat1 = lats[u]
        lon2 = lons[v]
        lat2 = lats[v]

        dist_m = haversine_distance(lon1, lat1, lon2, lat2)

        edge_data = {
            "arc_id": arc_id,
            "length_m": dist_m,
            "from_node_id": u,
            "to_node_id": v,
        }
        # MultiDiGraph accepts 4-tuple: (u, v, key, data_dict)
        edges_to_add.append((u, v, arc_id, edge_data))

    road_graph.add_edges_from(edges_to_add)

    # Verify arc lookup integrity
    if len(arc_lookup) != len(arcs_df):
        raise ValueError(
            f"Arc lookup size ({len(arc_lookup)}) does not match arcs count ({len(arcs_df)})."
        )

    # 4. Calculate graph statistics
    graph_statistics = get_graph_statistics(road_graph)

    return road_graph, arc_lookup, graph_statistics
