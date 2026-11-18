"""Complete End-to-End Route Reconstruction Pipeline for PathSnap Device 3.

Task 2: Step 4 - Integration wrapper connecting candidate loading,
emission normalization, Viterbi decoding, and GPS coordinate snapping.
"""

from datetime import datetime
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import networkx as nx
import numpy as np
import pandas as pd

try:
    from modules.device3.decoder.candidate_io import (
        RECONSTRUCTED_OUTPUT_COLUMNS,
        normalize_emission_probabilities,
        validate_candidates_against_graph,
    )
    from modules.device3.decoder.route_projection import route_to_gps_coordinates
    from modules.device3.decoder.transition_model import _build_arc_lookup_if_needed
    from modules.device3.decoder.viterbi import reconstruct_route
except (ImportError, ModuleNotFoundError):
    try:
        from decoder.candidate_io import (
            RECONSTRUCTED_OUTPUT_COLUMNS,
            normalize_emission_probabilities,
            validate_candidates_against_graph,
        )
        from decoder.route_projection import route_to_gps_coordinates
        from decoder.transition_model import _build_arc_lookup_if_needed
        from decoder.viterbi import reconstruct_route
    except (ImportError, ModuleNotFoundError):
        from .candidate_io import (
            RECONSTRUCTED_OUTPUT_COLUMNS,
            normalize_emission_probabilities,
            validate_candidates_against_graph,
        )
        from .route_projection import route_to_gps_coordinates
        from .transition_model import _build_arc_lookup_if_needed
        from .viterbi import reconstruct_route


def reconstruct_trip(
    candidates_df: pd.DataFrame,
    road_graph: nx.MultiDiGraph,
    track_df: pd.DataFrame,
    beta: float = 5.0,
    *,
    return_diagnostics: bool = False,
    arc_lookup: Optional[Dict[int, Tuple[int, int]]] = None,
    shortest_path_cache: Optional[Dict[Tuple[int, int], float]] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """Execute complete end-to-end distributed route reconstruction for a single vehicle trip.

    Pipeline Stages
    ----------------
    1. Schema & Single-Trip Validation: Supports either 'prediction_probability' or 'match_probability'.
       Rejects mixed trip IDs and empty tables.
    2. Emission Normalization: Applies numerical probability floor and normalizes candidate emissions
       per observation group into valid categorical distributions.
    3. Road Graph Integrity: Validates that candidate arc IDs belong to the specific trip's road network.
    4. Log-Space Viterbi Trellis: Finds the maximum-likelihood sequence of directed road arcs with
       LogSumExp transition normalization and automatic nearest-road fallback for missing observations.
    5. Geometric Coordinate Snapping: Projects original GPS points onto matched road segments to produce
       precise centerline coordinates while strictly preserving original timestamps and observation order.

    Parameters
    ----------
    candidates_df : pd.DataFrame
        Candidate road predictions containing ['trip_id', 'obs_idx', 'arc_id']
        and 'prediction_probability' (Device 2) or 'match_probability'.
    road_graph : nx.MultiDiGraph
        Directed road network MultiDiGraph for this specific trip.
    track_df : pd.DataFrame
        GPS trajectory DataFrame containing ['obs_idx', 'longitude', 'latitude', 'timestamp'].
    beta : float, default=5.0
        Transition probability scale parameter in meters.
    return_diagnostics : bool, default=False
        If True, returns a tuple (reconstructed_df, diagnostics_dict).
    arc_lookup : Optional[Dict[int, Tuple[int, int]]], default=None
        Lookup dictionary mapping arc_id -> (u, v).
    shortest_path_cache : Optional[Dict[Tuple[int, int], float]], default=None
        Reusable cache for shortest path routing queries.

    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]
        Reconstructed DataFrame with columns:
        - trip_id: string
        - obs_idx: int64
        - timestamp: float64 (preserves original track timestamp in seconds)
        - latitude: float64 (reconstructed latitude on road segment)
        - longitude: float64 (reconstructed longitude on road segment)
        - arc_id: int64 (Viterbi-matched road arc ID)
        If return_diagnostics=True, also returns a diagnostics dictionary.

    Raises
    ------
    ValueError
        If inputs are empty, mixed trips are detected, or candidate arc IDs are invalid.
    """
    t_start = time.perf_counter()

    if road_graph is None or road_graph.number_of_nodes() == 0 or road_graph.number_of_edges() == 0:
        raise ValueError("road_graph must be a non-empty NetworkX MultiDiGraph.")

    if candidates_df is None or candidates_df.empty:
        raise ValueError("candidates_df cannot be empty.")

    if track_df is None or track_df.empty:
        raise ValueError("track_df cannot be empty.")

    # 1. Single Trip ID Enforcement
    if "trip_id" in candidates_df.columns:
        unique_trips = candidates_df["trip_id"].unique()
        if len(unique_trips) > 1:
            raise ValueError(
                f"Mixed trip IDs found in candidates_df: {list(unique_trips)}. "
                f"Exactly one trip must be processed per call."
            )
        trip_id = str(unique_trips[0])
    elif "trip_id" in track_df.columns:
        trip_id = str(track_df["trip_id"].iloc[0])
    else:
        trip_id = "trip_unknown"

    if "trip_id" in track_df.columns:
        track_trips = track_df["trip_id"].unique()
        if len(track_trips) > 1 or (trip_id != "trip_unknown" and str(track_trips[0]) != trip_id):
            raise ValueError(
                f"trip_id mismatch between candidates_df ({trip_id}) and track_df ({list(track_trips)})."
            )

    lookup = _build_arc_lookup_if_needed(road_graph, arc_lookup)

    # 2. Normalize emission probabilities
    clean_candidates = candidates_df.copy()
    if "emission_probability" not in clean_candidates.columns:
        clean_candidates = normalize_emission_probabilities(clean_candidates)

    # 3. Candidate validation against road graph
    validate_candidates_against_graph(
        clean_candidates,
        road_graph=road_graph,
        arc_lookup=lookup,
        track_df=track_df,
        strict=True,
    )

    # 4. Log-Space Viterbi Route Decoding
    t0_viterbi = time.perf_counter()
    predicted_arcs, viterbi_diag = reconstruct_route(
        candidates_with_probs=clean_candidates,
        road_graph=road_graph,
        beta=beta,
        track_df=track_df,
        return_diagnostics=True,
        arc_lookup=lookup,
        shortest_path_cache=shortest_path_cache,
    )
    t1_viterbi = time.perf_counter()
    viterbi_time_ms = (t1_viterbi - t0_viterbi) * 1000.0

    # 5. GPS Coordinate Snapping
    t0_proj = time.perf_counter()
    result_df = route_to_gps_coordinates(
        predicted_arcs=predicted_arcs,
        road_graph=road_graph,
        track_df=track_df,
        trip_id=trip_id,
        arc_lookup=lookup,
    )
    t1_proj = time.perf_counter()
    proj_time_ms = (t1_proj - t0_proj) * 1000.0

    t_end = time.perf_counter()
    total_time_ms = (t_end - t_start) * 1000.0

    if not return_diagnostics:
        return result_df

    diagnostics: Dict[str, Any] = {
        "trip_id": trip_id,
        "total_gps_observations": len(track_df),
        "reconstructed_observations_count": len(result_df),
        "fallback_observations_count": viterbi_diag.get("fallback_observations_count", 0),
        "fallback_obs_indices": viterbi_diag.get("fallback_obs_indices", []),
        "decoded_arcs_count": len(predicted_arcs),
        "viterbi_execution_time_ms": viterbi_time_ms,
        "projection_execution_time_ms": proj_time_ms,
        "total_reconstruction_time_ms": total_time_ms,
        "final_sequence_log_score": viterbi_diag.get("final_sequence_log_score", 0.0),
        "disconnected_transitions_count": viterbi_diag.get("disconnected_transitions_count", 0),
    }

    return result_df, diagnostics


def convert_timestamps_to_datetime(
    reconstructed_df: pd.DataFrame,
    start_datetime: Union[str, pd.Timestamp, datetime],
    timezone: Optional[str] = "UTC",
) -> pd.DataFrame:
    """Optionally convert relative elapsed seconds into localized datetime timestamps.

    Parameters
    ----------
    reconstructed_df : pd.DataFrame
        Reconstructed DataFrame containing 'timestamp' column in elapsed seconds.
    start_datetime : Union[str, pd.Timestamp, datetime]
        Trip start time anchor.
    timezone : Optional[str], default='UTC'
        Timezone string (e.g., 'UTC', 'Europe/Berlin').

    Returns
    -------
    pd.DataFrame
        Copy of DataFrame with an added 'datetime' column.
    """
    df = reconstructed_df.copy()
    base_ts = pd.to_datetime(start_datetime)
    if timezone:
        if base_ts.tzinfo is None:
            base_ts = base_ts.tz_localize(timezone)
        else:
            base_ts = base_ts.tz_convert(timezone)

    df["datetime"] = base_ts + pd.to_timedelta(df["timestamp"], unit="s")
    return df
