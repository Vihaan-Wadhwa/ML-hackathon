"""Route Coordinate Snapping and Projection for PathSnap Device 3.

Task 2: Step 4 - Snapping Viterbi-reconstructed arc sequences to exact
geographic coordinates on the road network.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import networkx as nx
import numpy as np
import pandas as pd

try:
    from modules.device3.decoder.transition_model import (
        ArcProjection,
        _build_arc_lookup_if_needed,
        project_point_onto_arc,
    )
except (ImportError, ModuleNotFoundError):
    try:
        from decoder.transition_model import (
            ArcProjection,
            _build_arc_lookup_if_needed,
            project_point_onto_arc,
        )
    except (ImportError, ModuleNotFoundError):
        from .transition_model import (
            ArcProjection,
            _build_arc_lookup_if_needed,
            project_point_onto_arc,
        )


RECONSTRUCTED_OUTPUT_COLUMNS = [
    "trip_id",
    "obs_idx",
    "timestamp",
    "latitude",
    "longitude",
    "arc_id",
]


def route_to_gps_coordinates(
    predicted_arcs: Union[List[int], np.ndarray, pd.Series],
    road_graph: nx.MultiDiGraph,
    track_df: pd.DataFrame,
    trip_id: str,
    arc_lookup: Optional[Dict[int, Tuple[int, int]]] = None,
) -> pd.DataFrame:
    """Project GPS observations onto their Viterbi-predicted road segments.

    For every observation in track_df:
    1. Retrieves its predicted arc_id from predicted_arcs.
    2. Identifies the directed edge in road_graph and extracts start/end coordinates.
    3. Projects the GPS point (lon, lat) perpendicularly onto the segment.
    4. Clamps the projection fraction q to [0.0, 1.0].
    5. Emits the snapped geographic coordinate on the road centerline.
    6. Preserves the original observation timestamp and zero-based obs_idx.

    Parameters
    ----------
    predicted_arcs : Union[List[int], np.ndarray, pd.Series]
        Sequence of predicted arc IDs, one per GPS observation in track_df.
    road_graph : nx.MultiDiGraph
        Directed road network graph containing node coordinates and edge keys.
    track_df : pd.DataFrame
        GPS trajectory DataFrame containing ['obs_idx', 'longitude', 'latitude', 'timestamp'].
    trip_id : str
        String identifier for the trip.
    arc_lookup : Optional[Dict[int, Tuple[int, int]]], default=None
        Lookup dictionary mapping arc_id -> (u, v).

    Returns
    -------
    pd.DataFrame
        Reconstructed route DataFrame with columns:
        - trip_id: string
        - obs_idx: int64
        - timestamp: float64 (preserves original track timestamp in seconds)
        - latitude: float64 (reconstructed latitude on road segment)
        - longitude: float64 (reconstructed longitude on road segment)
        - arc_id: int64 (matched road arc ID)

    Raises
    ------
    ValueError
        If inputs are empty, lengths mismatch, or required track columns are missing.
    KeyError
        If any predicted arc_id does not exist in the road graph.
    """
    if road_graph is None or road_graph.number_of_nodes() == 0 or road_graph.number_of_edges() == 0:
        raise ValueError("road_graph must be a non-empty NetworkX MultiDiGraph.")

    if track_df is None or track_df.empty:
        raise ValueError("track_df cannot be empty.")

    req_cols = ["longitude", "latitude", "timestamp"]
    missing = [c for c in req_cols if c not in track_df.columns]
    if missing:
        raise ValueError(f"track_df missing required coordinate columns: {missing}")

    # Ensure sorted by obs_idx
    if "obs_idx" in track_df.columns:
        sorted_track = track_df.sort_values("obs_idx").reset_index(drop=True)
    else:
        sorted_track = track_df.reset_index(drop=True)
        sorted_track["obs_idx"] = np.arange(len(sorted_track), dtype=np.int64)

    arc_list = [int(a) for a in predicted_arcs]
    T = len(sorted_track)
    if len(arc_list) != T:
        raise ValueError(
            f"Length mismatch: predicted_arcs has {len(arc_list)} items, "
            f"but track_df has {T} observations."
        )

    lookup = _build_arc_lookup_if_needed(road_graph, arc_lookup)

    out_trip_id: List[str] = []
    out_obs_idx: List[int] = []
    out_timestamp: List[float] = []
    out_latitude: List[float] = []
    out_longitude: List[float] = []
    out_arc_id: List[int] = []

    for i in range(T):
        row = sorted_track.iloc[i]
        obs_idx = int(row["obs_idx"])
        ts = float(row["timestamp"])
        gps_lon = float(row["longitude"])
        gps_lat = float(row["latitude"])
        aid = arc_list[i]

        if aid not in lookup:
            raise KeyError(
                f"Predicted arc ID {aid} at observation {obs_idx} does not exist in the road graph."
            )

        u, v = lookup[aid]
        u_node = road_graph.nodes[u]
        v_node = road_graph.nodes[v]

        u_coord = (float(u_node["longitude"]), float(u_node["latitude"]))
        v_coord = (float(v_node["longitude"]), float(v_node["latitude"]))

        # Project GPS point onto directed road segment
        proj: ArcProjection = project_point_onto_arc(gps_lon, gps_lat, u_coord, v_coord)

        out_trip_id.append(str(trip_id))
        out_obs_idx.append(obs_idx)
        out_timestamp.append(ts)
        out_latitude.append(proj.projected_latitude)
        out_longitude.append(proj.projected_longitude)
        out_arc_id.append(aid)

    result_df = pd.DataFrame({
        "trip_id": pd.Series(out_trip_id, dtype="string"),
        "obs_idx": np.array(out_obs_idx, dtype=np.int64),
        "timestamp": np.array(out_timestamp, dtype=np.float64),
        "latitude": np.array(out_latitude, dtype=np.float64),
        "longitude": np.array(out_longitude, dtype=np.float64),
        "arc_id": np.array(out_arc_id, dtype=np.int64),
    })

    return result_df
