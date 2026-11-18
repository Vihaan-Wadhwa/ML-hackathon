"""Log-Space Viterbi Trellis Decoder for PathSnap Device 3.

Task 2: Step 3 - Viterbi Decoding, Fallback Nearest-Arc Search,
Log-Space Categorical Transition Normalization, and Diagnostics.
"""

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import networkx as nx
import numpy as np
import pandas as pd

try:
    from modules.device3.decoder.candidate_io import normalize_emission_probabilities
    from modules.device3.decoder.transition_model import (
        ArcProjection,
        _build_arc_lookup_if_needed,
        _extract_coords,
        _extract_gps,
        build_transition_matrix,
        calculate_effective_beta,
        project_point_onto_arc,
    )
except (ImportError, ModuleNotFoundError):
    try:
        from decoder.candidate_io import normalize_emission_probabilities
        from decoder.transition_model import (
            ArcProjection,
            _build_arc_lookup_if_needed,
            _extract_coords,
            _extract_gps,
            build_transition_matrix,
            calculate_effective_beta,
            project_point_onto_arc,
        )
    except (ImportError, ModuleNotFoundError):
        from .candidate_io import normalize_emission_probabilities
        from .transition_model import (
            ArcProjection,
            _build_arc_lookup_if_needed,
            _extract_coords,
            _extract_gps,
            build_transition_matrix,
            calculate_effective_beta,
            project_point_onto_arc,
        )


def log_normalize_transition_matrix(
    log_density_matrix: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Normalize row-wise in log-space while preserving the disconnected transition floor.

    For candidate transitions:
    - Connected transitions are normalized row-wise via LogSumExp so that they form
      a valid categorical probability distribution over viable forward states.
    - Disconnected transitions (at or below ln(epsilon)) preserve the documented
      disconnected floor ln(epsilon) from transition_model.py, preventing impossible
      transitions from artificially competing with continuous road paths.

    Parameters
    ----------
    log_density_matrix : np.ndarray
        2D array of raw log transition densities of shape (N, M).
    epsilon : float, default=1e-6
        Floor probability for disconnected transitions.

    Returns
    -------
    np.ndarray
        Log-normalized 2D array of shape (N, M).
    """
    if log_density_matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {log_density_matrix.shape}")

    N, M = log_density_matrix.shape
    if M == 0 or N == 0:
        return log_density_matrix.copy()

    log_floor = math.log(epsilon)
    # A transition at or below log_floor represents a disconnected or floored transition
    is_disc = log_density_matrix <= (log_floor + 1e-4)

    norm_log = np.full_like(log_density_matrix, log_floor)

    for i in range(N):
        conn_mask = ~is_disc[i]
        if np.any(conn_mask):
            row_conn = log_density_matrix[i, conn_mask]
            m = np.max(row_conn)
            lse = m + np.log(np.sum(np.exp(row_conn - m)))
            norm_log[i, conn_mask] = row_conn - lse
        else:
            # All candidates disconnected: keep disconnected floor
            norm_log[i, :] = log_floor

    return norm_log


def _get_or_build_edge_spatial_index(
    road_graph: nx.MultiDiGraph,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int, int]]]:
    """Retrieve or construct a vectorized edge bounding-box spatial index for the road graph.

    Stores segment bounding boxes [min_lon, max_lon, min_lat, max_lat] to provide a
    rigorous mathematical lower bound on point-to-segment distance, preventing long road
    segments from being erroneously excluded during nearest-road queries.
    """
    if "_edge_spatial_index" in road_graph.graph:
        return road_graph.graph["_edge_spatial_index"]

    edge_list: List[Tuple[int, int, int]] = []
    min_lon_list: List[float] = []
    max_lon_list: List[float] = []
    min_lat_list: List[float] = []
    max_lat_list: List[float] = []
    arc_id_list: List[int] = []

    for u, v, k, data in road_graph.edges(keys=True, data=True):
        aid = int(data.get("arc_id", k))
        u_data = road_graph.nodes[u]
        v_data = road_graph.nodes[v]
        lon_u, lat_u = float(u_data["longitude"]), float(u_data["latitude"])
        lon_v, lat_v = float(v_data["longitude"]), float(v_data["latitude"])

        edge_list.append((u, v, aid))
        min_lon_list.append(min(lon_u, lon_v))
        max_lon_list.append(max(lon_u, lon_v))
        min_lat_list.append(min(lat_u, lat_v))
        max_lat_list.append(max(lat_u, lat_v))
        arc_id_list.append(aid)

    min_lons = np.array(min_lon_list, dtype=np.float64)
    max_lons = np.array(max_lon_list, dtype=np.float64)
    min_lats = np.array(min_lat_list, dtype=np.float64)
    max_lats = np.array(max_lat_list, dtype=np.float64)
    arc_ids = np.array(arc_id_list, dtype=np.int64)

    index_data = (min_lons, max_lons, min_lats, max_lats, arc_ids, edge_list)
    road_graph.graph["_edge_spatial_index"] = index_data
    return index_data


def find_nearest_candidate_arc(
    gps_observation: Any,
    road_graph: nx.MultiDiGraph,
    arc_lookup: Optional[Dict[int, Tuple[int, int]]] = None,
    candidate_pool_size: int = 100,
) -> Tuple[int, float]:
    """Find the geographically nearest road arc in road_graph using perpendicular projection.

    Vectorized Bounding-Box Acceleration
    -----------------------------------
    Pre-filters candidate arcs by computing the true lower-bound distance to each edge's
    axis-aligned bounding box. This guarantees that long road segments passing close to
    the GPS point are never excluded. Exact perpendicular projection is then performed
    on the top candidate pool.

    Fallback Probability Policy
    ---------------------------
    When an observation lacks candidates from Device 2/4, this helper selects the nearest
    physical road segment. The single fallback candidate receives categorical emission
    P = 1.0 (indicating it is the sole available candidate state for that observation,
    NOT high model classification confidence). It is recorded in fallback diagnostics.

    Parameters
    ----------
    gps_observation : Any
        GPS observation with longitude and latitude.
    road_graph : nx.MultiDiGraph
        Directed road network graph.
    arc_lookup : Optional[Dict[int, Tuple[int, int]]], default=None
        Lookup dictionary mapping arc_id -> (u, v).
    candidate_pool_size : int, default=100
        Number of nearest bounding-box candidate arcs evaluated with exact perpendicular distance.

    Returns
    -------
    nearest_arc_id : int
        Arc ID of the nearest road segment.
    min_distance_m : float
        Perpendicular distance in meters from the GPS point to the nearest arc.

    Raises
    ------
    ValueError
        If road_graph contains no edges.
    """
    if road_graph.number_of_edges() == 0:
        raise ValueError("Cannot find nearest arc in an empty road graph.")

    p_lon, p_lat = _extract_coords(gps_observation)

    min_lons, max_lons, min_lats, max_lats, arc_ids, edge_list = _get_or_build_edge_spatial_index(road_graph)
    total_edges = len(arc_ids)

    # Compute bounding-box lower bound distances in degrees
    if total_edges <= candidate_pool_size:
        eval_indices = range(total_edges)
    else:
        dx = np.maximum(0.0, np.maximum(min_lons - p_lon, p_lon - max_lons))
        dy = np.maximum(0.0, np.maximum(min_lats - p_lat, p_lat - max_lats))
        box_dist_sq = dx * dx + dy * dy
        eval_indices = np.argpartition(box_dist_sq, min(candidate_pool_size, total_edges - 1))[:candidate_pool_size]

    best_arc: Optional[int] = None
    min_dist = float("inf")

    for idx in eval_indices:
        u, v, aid = edge_list[idx]
        u_coord = (road_graph.nodes[u]["longitude"], road_graph.nodes[u]["latitude"])
        v_coord = (road_graph.nodes[v]["longitude"], road_graph.nodes[v]["latitude"])

        proj = project_point_onto_arc(p_lon, p_lat, u_coord, v_coord)
        # Deterministic tie-breaking on lowest arc_id
        if (proj.perpendicular_distance_m < min_dist) or (
            proj.perpendicular_distance_m == min_dist and (best_arc is None or aid < best_arc)
        ):
            min_dist = proj.perpendicular_distance_m
            best_arc = aid

    if best_arc is None:
        raise RuntimeError("Failed to identify nearest road arc.")

    return best_arc, float(min_dist)


def reconstruct_route(
    candidates_with_probs: pd.DataFrame,
    road_graph: nx.MultiDiGraph,
    beta: float = 5.0,
    *,
    track_df: Optional[pd.DataFrame] = None,
    return_diagnostics: bool = False,
    arc_lookup: Optional[Dict[int, Tuple[int, int]]] = None,
    shortest_path_cache: Optional[Dict[Tuple[int, int], float]] = None,
    dropout_threshold_s: float = 5.0,
    epsilon: float = 1e-12,
) -> Union[List[int], Tuple[List[int], Dict[str, Any]]]:
    """Reconstruct optimal road route sequence from GPS track and candidate predictions via Log-Space Viterbi.

    Algorithm Workflow
    ------------------
    1. Validation: Verifies single trip ID, valid finite probabilities, and presence of GPS coordinates.
    2. Missing Observation Fallback: For any GPS observation lacking candidates in the Parquet input,
       geometrically queries the nearest road arc via perpendicular distance with documented P=1.0.
    3. Trellis Initialization: Uniform initial prior over first observation candidates in log-space.
    4. Trellis Recurrence: For observations t = 1 ... T-1, evaluates transition matrix with effective
       dropout-adjusted beta, normalizes transition rows into categorical probabilities via LogSumExp,
       and records maximum path log-scores with deterministic backpointers.
    5. Backtracking: Identifies the globally optimal final candidate state and follows backpointers
       in reverse order, recovering exactly one predicted arc ID per valid GPS observation.

    Parameters
    ----------
    candidates_with_probs : pd.DataFrame
        Candidate predictions containing ['trip_id', 'obs_idx', 'arc_id'] and
        'emission_probability' (or 'match_probability' / 'prediction_probability').
    road_graph : nx.MultiDiGraph
        Directed road network MultiDiGraph for this specific trip.
    beta : float, default=5.0
        Transition probability scale parameter in meters.
    track_df : Optional[pd.DataFrame], default=None
        GPS trajectory DataFrame containing ['obs_idx', 'longitude', 'latitude', 'timestamp'].
        If None, coordinates and timestamps must be present in candidates_with_probs.
    return_diagnostics : bool, default=False
        If True, returns a tuple (predicted_arcs, diagnostics_dict).
    arc_lookup : Optional[Dict[int, Tuple[int, int]]], default=None
        Lookup dictionary mapping arc_id -> (u, v).
    shortest_path_cache : Optional[Dict[Tuple[int, int], float]], default=None
        Reusable cache for shortest path queries across observations.
    dropout_threshold_s : float, default=5.0
        Elapsed seconds above which transition beta scales linearly with delta_t.
    epsilon : float, default=1e-12
        Floor probability to prevent log(0).

    Returns
    -------
    Union[List[int], Tuple[List[int], Dict[str, Any]]]
        List of predicted arc IDs corresponding to each observation.
        If return_diagnostics=True, returns (predicted_arcs, diagnostics_dict).

    Raises
    ------
    ValueError
        If inputs are empty, mixed trips are detected, coordinates are unavailable,
        or invalid candidate arc IDs are present in the candidate data.
    """
    t_start = time.perf_counter()

    if road_graph is None or road_graph.number_of_nodes() == 0 or road_graph.number_of_edges() == 0:
        raise ValueError("road_graph must be a non-empty NetworkX MultiDiGraph.")

    if candidates_with_probs is None or candidates_with_probs.empty:
        raise ValueError("candidates_with_probs DataFrame cannot be empty.")

    lookup = _build_arc_lookup_if_needed(road_graph, arc_lookup)
    valid_graph_arc_ids: Set[int] = set(lookup.keys())

    # 1. Single Trip ID Enforcement
    if "trip_id" in candidates_with_probs.columns:
        unique_trips = candidates_with_probs["trip_id"].unique()
        if len(unique_trips) > 1:
            raise ValueError(
                f"Mixed trip IDs found in candidates: {list(unique_trips)}. "
                f"Exactly one trip must be processed per call."
            )
        trip_id = str(unique_trips[0])
    elif track_df is not None and "trip_id" in track_df.columns:
        trip_id = str(track_df["trip_id"].iloc[0])
    else:
        trip_id = "trip_default"

    # 2. Extract or Resolve Observation GPS Track
    if track_df is not None:
        if track_df.empty:
            raise ValueError("Provided track_df is empty.")
        if "trip_id" in track_df.columns:
            track_trips = track_df["trip_id"].unique()
            if len(track_trips) > 1 or (trip_id != "trip_default" and str(track_trips[0]) != trip_id):
                raise ValueError(f"track_df trip ID mismatch: expected {trip_id}, found {list(track_trips)}")

        req_track_cols = ["longitude", "latitude", "timestamp"]
        missing_track = [c for c in req_track_cols if c not in track_df.columns]
        if missing_track:
            raise ValueError(f"track_df missing required coordinate columns: {missing_track}")

        # Ensure sorted by obs_idx or row order
        if "obs_idx" in track_df.columns:
            sorted_track = track_df.sort_values("obs_idx").reset_index(drop=True)
            obs_indices = sorted_track["obs_idx"].astype(np.int64).tolist()
        else:
            sorted_track = track_df.reset_index(drop=True)
            obs_indices = list(range(len(sorted_track)))

        gps_by_obs: Dict[int, Tuple[float, float, float]] = {
            obs_idx: (
                float(sorted_track.loc[i, "longitude"]),
                float(sorted_track.loc[i, "latitude"]),
                float(sorted_track.loc[i, "timestamp"]),
            )
            for i, obs_idx in enumerate(obs_indices)
        }
    else:
        # Check if GPS coordinates and timestamps are embedded in candidates_with_probs
        gps_cols = ["obs_idx", "longitude", "latitude", "timestamp"]
        missing_gps = [c for c in gps_cols if c not in candidates_with_probs.columns]
        if missing_gps:
            raise ValueError(
                f"GPS coordinates and timestamps must be provided via track_df or included "
                f"directly in candidates_with_probs. Missing columns: {missing_gps}"
            )

        unique_obs = candidates_with_probs.drop_duplicates(subset=["obs_idx"]).sort_values("obs_idx")
        obs_indices = unique_obs["obs_idx"].astype(np.int64).tolist()
        gps_by_obs = {
            int(row["obs_idx"]): (
                float(row["longitude"]),
                float(row["latitude"]),
                float(row["timestamp"]),
            )
            for _, row in unique_obs.iterrows()
        }

    total_T = len(obs_indices)
    if total_T == 0:
        raise ValueError("No GPS observations available to reconstruct.")

    # 3. Ensure Emission Probabilities are Normalized
    df_cands = candidates_with_probs.copy()
    if "emission_probability" not in df_cands.columns:
        df_cands = normalize_emission_probabilities(df_cands)

    # Validate that arc_ids in candidate dataframe exist in road_graph
    invalid_candidate_arcs = set(df_cands["arc_id"].unique()) - valid_graph_arc_ids
    if invalid_candidate_arcs:
        sample_invalid = sorted(list(invalid_candidate_arcs))[:5]
        raise ValueError(
            f"Candidate predictions contain {len(invalid_candidate_arcs)} arc ID(s) "
            f"that do not exist in the road graph: {sample_invalid}"
        )

    # 4. Group Candidates and Handle Missing Observations (Fallback Nearest Arc)
    grouped = df_cands.groupby("obs_idx")
    candidates_per_obs: List[List[int]] = []
    emissions_per_obs: List[List[float]] = []
    fallback_obs_indices: List[int] = []

    for t, obs_idx in enumerate(obs_indices):
        if obs_idx in grouped.groups:
            group = grouped.get_group(obs_idx)
            arcs = group["arc_id"].astype(np.int64).tolist()
            emissions = group["emission_probability"].astype(np.float64).tolist()

            # Check emission validity
            if any(not math.isfinite(e) or e < 0.0 for e in emissions):
                raise ValueError(f"Invalid emission probability detected at observation {obs_idx}: {emissions}")

            candidates_per_obs.append(arcs)
            emissions_per_obs.append(emissions)
        else:
            # Observation is missing from candidate Parquet: fallback to nearest road arc
            gps_pt = gps_by_obs[obs_idx]
            nearest_arc, _ = find_nearest_candidate_arc(gps_pt, road_graph, lookup)

            candidates_per_obs.append([nearest_arc])
            # Fallback policy: single candidate receives full categorical probability 1.0
            emissions_per_obs.append([1.0])
            fallback_obs_indices.append(obs_idx)

    # 5. Viterbi Initialization (t = 0)
    K_0 = len(candidates_per_obs[0])
    init_prior = 1.0 / K_0
    log_init_prior = math.log(max(init_prior, epsilon))

    V_current = np.array(
        [
            log_init_prior + math.log(max(emissions_per_obs[0][j], epsilon))
            for j in range(K_0)
        ],
        dtype=np.float64,
    )

    backpointers: List[Optional[np.ndarray]] = [None] * total_T
    total_disconnected = 0
    if shortest_path_cache is None:
        shortest_path_cache = {}

    # 6. Viterbi Forward Recurrence (t = 1 ... T-1)
    for t in range(1, total_T):
        prev_obs = obs_indices[t - 1]
        curr_obs = obs_indices[t]
        prev_gps = gps_by_obs[prev_obs]
        curr_gps = gps_by_obs[curr_obs]

        prev_arcs = candidates_per_obs[t - 1]
        curr_arcs = candidates_per_obs[t]
        curr_emissions = emissions_per_obs[t]
        K_curr = len(curr_arcs)

        # Build raw transition matrix
        trans_res = build_transition_matrix(
            previous_candidates=prev_arcs,
            current_candidates=curr_arcs,
            previous_gps=prev_gps,
            current_gps=curr_gps,
            road_graph=road_graph,
            beta=beta,
            shortest_path_cache=shortest_path_cache,
            arc_lookup=lookup,
            dropout_threshold_s=dropout_threshold_s,
        )
        total_disconnected += trans_res["num_disconnected"]

        # Row-wise log-normalization: log_P(j | i) = log_density(i,j) - logsumexp_j(...)
        log_trans = log_normalize_transition_matrix(trans_res["log_transition_matrix"])

        # Trellis step
        # V_current shape (K_prev,), log_trans shape (K_prev, K_curr)
        # Broadcasting V_current[:, None] + log_trans yields shape (K_prev, K_curr)
        candidate_transitions = V_current[:, None] + log_trans

        # Deterministic max and argmax across previous candidates (axis 0)
        best_prev_scores = np.max(candidate_transitions, axis=0)
        best_prev_indices = np.argmax(candidate_transitions, axis=0)

        log_emissions = np.array(
            [math.log(max(e, epsilon)) for e in curr_emissions], dtype=np.float64
        )
        V_current = best_prev_scores + log_emissions
        backpointers[t] = best_prev_indices

    # 7. Backtracking Optimal State Sequence
    best_final_idx = int(np.argmax(V_current))
    final_log_score = float(V_current[best_final_idx])

    state_indices = [0] * total_T
    state_indices[total_T - 1] = best_final_idx

    for t in range(total_T - 2, -1, -1):
        bp = backpointers[t + 1]
        assert bp is not None
        state_indices[t] = int(bp[state_indices[t + 1]])

    predicted_arcs: List[int] = [
        candidates_per_obs[t][state_indices[t]] for t in range(total_T)
    ]

    t_end = time.perf_counter()
    execution_time_ms = (t_end - t_start) * 1000.0

    if not return_diagnostics:
        return predicted_arcs

    diagnostics: Dict[str, Any] = {
        "trip_id": trip_id,
        "total_gps_observations": total_T,
        "normal_observations_count": total_T - len(fallback_obs_indices),
        "fallback_observations_count": len(fallback_obs_indices),
        "fallback_obs_indices": fallback_obs_indices,
        "candidate_counts_per_obs": [len(c) for c in candidates_per_obs],
        "total_candidate_states": sum(len(c) for c in candidates_per_obs),
        "disconnected_transitions_count": total_disconnected,
        "final_sequence_log_score": final_log_score,
        "execution_time_ms": execution_time_ms,
        "predicted_arcs": predicted_arcs,
        "observation_indices": obs_indices,
    }

    return predicted_arcs, diagnostics
