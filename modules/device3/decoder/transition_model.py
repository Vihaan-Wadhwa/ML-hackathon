"""Directed-road transition probabilities for PathSnap Device 3."""

import math
from functools import lru_cache
from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

try:
    from ..graph.graph_builder import haversine_distance
except ImportError:
    from graph.graph_builder import haversine_distance


def _get_arc_lookup(road_graph, arc_lookup=None):
    """Return arc_id -> (from_node, to_node)."""
    if arc_lookup is not None:
        return arc_lookup

    result = {}
    for u, v, key, data in road_graph.edges(keys=True, data=True):
        arc_id = data.get("arc_id", key)
        result[int(arc_id)] = (int(u), int(v))
    return result


def _get_arc_lengths(road_graph):
    """Return arc_id -> length in metres."""
    lengths = {}
    for _, _, key, data in road_graph.edges(keys=True, data=True):
        arc_id = int(data.get("arc_id", key))
        length = float(data["length_m"])

        if not math.isfinite(length) or length < 0:
            raise ValueError(f"Invalid length for arc {arc_id}: {length}")

        lengths[arc_id] = length

    return lengths


def _optional_distance(candidate, column, default):
    """Read an optional projection distance from a candidate row."""
    value = candidate.get(column, default)

    if pd.isna(value):
        return default

    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid {column}: {value}")

    return value


def transition_distance(
    previous,
    current,
    road_graph,
    arc_lookup=None,
    arc_lengths=None,
    shortest_path_cache=None,
):
    """Estimate directed network distance between two candidate positions.

    Projection-aware formula for different arcs:

        previous remaining distance
        + shortest directed path between arc endpoints
        + current distance from its arc start

    When projection fields are absent, the previous arc's full length and
    zero current prefix are used as an explicit approximation.

    For candidates on the same arc, projection distances are used when
    available. A backward movement on the same directed arc is impossible.
    """
    lookup = _get_arc_lookup(road_graph, arc_lookup)
    lengths = arc_lengths if arc_lengths is not None else _get_arc_lengths(road_graph)

    previous_arc = int(previous["arc_id"])
    current_arc = int(current["arc_id"])

    if previous_arc not in lookup or current_arc not in lookup:
        raise ValueError("Candidate references an arc missing from the road graph.")

    if previous_arc not in lengths or current_arc not in lengths:
        raise ValueError("Candidate references an arc without a valid length.")

    previous_length = lengths[previous_arc]
    current_length = lengths[current_arc]

    if previous_arc == current_arc:
        previous_position = _optional_distance(
            previous, "dist_to_start_m", None
        )
        current_position = _optional_distance(
            current, "dist_to_start_m", None
        )

        if previous_position is None or current_position is None:
            return 0.0

        if previous_position > previous_length or current_position > current_length:
            raise ValueError("Candidate projection exceeds its arc length.")

        distance = current_position - previous_position

        if distance < -1e-9:
            return math.inf

        return max(0.0, distance)

    previous_end_distance = _optional_distance(
        previous, "dist_to_end_m", previous_length
    )
    current_start_distance = _optional_distance(
        current, "dist_to_start_m", 0.0
    )

    if previous_end_distance > previous_length:
        raise ValueError("Previous candidate projection exceeds its arc length.")

    if current_start_distance > current_length:
        raise ValueError("Current candidate projection exceeds its arc length.")

    _, previous_end_node = lookup[previous_arc]
    current_start_node, _ = lookup[current_arc]

    if shortest_path_cache is None:
        shortest_path_cache = {}

    node_pair = (previous_end_node, current_start_node)

    if node_pair not in shortest_path_cache:
        try:
            middle_distance = nx.shortest_path_length(
                road_graph,
                source=previous_end_node,
                target=current_start_node,
                weight="length_m",
                method="dijkstra",
            )
            shortest_path_cache[node_pair] = float(middle_distance)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            shortest_path_cache[node_pair] = math.inf

    middle_distance = shortest_path_cache[node_pair]

    if not math.isfinite(middle_distance):
        return math.inf

    return (
        previous_end_distance
        + middle_distance
        + current_start_distance
    )


def transition_log_probability(
    previous,
    current,
    road_graph,
    gps_distance_m,
    beta=20.0,
    arc_lookup=None,
    arc_lengths=None,
    shortest_path_cache=None,
):
    """Calculate a log transition likelihood using a Laplace distance model.

    log(P) = -abs(network_distance - gps_distance) / beta

    beta controls tolerance in metres. Larger values make the model
    less sensitive to distance differences.
    """
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be a positive finite number.")

    if not math.isfinite(gps_distance_m) or gps_distance_m < 0:
        raise ValueError("gps_distance_m must be finite and non-negative.")

    network_distance = transition_distance(
        previous,
        current,
        road_graph,
        arc_lookup=arc_lookup,
        arc_lengths=arc_lengths,
        shortest_path_cache=shortest_path_cache,
    )

    if not math.isfinite(network_distance):
        return -math.inf

    return -abs(network_distance - gps_distance_m) / beta


def build_transition_matrix(
    previous_candidates,
    current_candidates,
    road_graph,
    gps_distance_m,
    beta=20.0,
    arc_lookup=None,
):
    """Build a matrix of log transition probabilities.

    Rows correspond to previous candidates and columns to current candidates.
    """
    if previous_candidates.empty or current_candidates.empty:
        raise ValueError("Candidate groups must not be empty.")

    lookup = _get_arc_lookup(road_graph, arc_lookup)
    lengths = _get_arc_lengths(road_graph)
    cache = {}

    matrix = np.full(
        (len(previous_candidates), len(current_candidates)),
        -np.inf,
        dtype=np.float64,
    )

    for i, (_, previous) in enumerate(previous_candidates.iterrows()):
        for j, (_, current) in enumerate(current_candidates.iterrows()):
            matrix[i, j] = transition_log_probability(
                previous,
                current,
                road_graph,
                gps_distance_m,
                beta=beta,
                arc_lookup=lookup,
                arc_lengths=lengths,
                shortest_path_cache=cache,
            )

    return matrix


def viterbi_decode(
    candidates_df,
    road_graph,
    arc_lookup=None,
    track_df=None,
    beta=20.0,
):
    """Decode the most likely directed road-arc sequence for each trip.

    Required candidate columns:
        trip_id, obs_idx, arc_id, emission_probability

    Returns one selected candidate per observation, including its
    cumulative Viterbi log score.
    """
    required = {
        "trip_id",
        "obs_idx",
        "arc_id",
        "emission_probability",
    }
    missing = required - set(candidates_df.columns)

    if missing:
        raise ValueError(f"Missing candidate columns: {sorted(missing)}")

    if candidates_df.empty:
        raise ValueError("Cannot decode an empty candidate DataFrame.")

    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be a positive finite number.")

    if candidates_df[list(required)].isna().any().any():
        raise ValueError("Candidate decoding columns contain null values.")

    emissions = pd.to_numeric(
        candidates_df["emission_probability"], errors="coerce"
    )

    if (
        emissions.isna().any()
        or not np.all(np.isfinite(emissions.to_numpy()))
        or (emissions <= 0).any()
        or (emissions > 1).any()
    ):
        raise ValueError(
            "emission_probability values must be finite and in (0, 1]."
        )

    lookup = _get_arc_lookup(road_graph, arc_lookup)
    lengths = _get_arc_lengths(road_graph)

    track_lookup = {}
    if track_df is not None:
        required_track = {"obs_idx", "longitude", "latitude"}
        missing_track = required_track - set(track_df.columns)
        if missing_track:
            raise ValueError(
                f"Missing track columns: {sorted(missing_track)}"
            )

        track_lookup = track_df.set_index("obs_idx").to_dict("index")

    decoded_trips = []

    for trip_id, trip_candidates in candidates_df.groupby(
        "trip_id", sort=False
    ):
        groups = [
            group.sort_values("arc_id").reset_index(drop=True)
            for _, group in trip_candidates.groupby("obs_idx", sort=True)
        ]

        if not groups:
            continue

        # Validate candidate arc references before decoding.
        for group in groups:
            invalid = set(group["arc_id"].astype(int)) - set(lookup)
            if invalid:
                raise ValueError(
                    f"Trip {trip_id} contains unknown arc IDs: "
                    f"{sorted(invalid)[:5]}"
                )

        scores = np.log(
            groups[0]["emission_probability"].to_numpy(dtype=np.float64)
        )

        backpointers = []

        for index in range(1, len(groups)):
            previous_group = groups[index - 1]
            current_group = groups[index]

            previous_obs = int(previous_group["obs_idx"].iloc[0])
            current_obs = int(current_group["obs_idx"].iloc[0])

            gps_distance = 0.0

            if track_lookup:
                if previous_obs not in track_lookup or current_obs not in track_lookup:
                    raise ValueError(
                        f"Track is missing observation {previous_obs} "
                        f"or {current_obs} for trip {trip_id}."
                    )

                previous_fix = track_lookup[previous_obs]
                current_fix = track_lookup[current_obs]

                gps_distance = haversine_distance(
                    previous_fix["longitude"],
                    previous_fix["latitude"],
                    current_fix["longitude"],
                    current_fix["latitude"],
                )

            transitions = build_transition_matrix(
                previous_group,
                current_group,
                road_graph,
                gps_distance,
                beta=beta,
                arc_lookup=lookup,
            )

            current_emissions = np.log(
                current_group["emission_probability"].to_numpy(
                    dtype=np.float64
                )
            )

            combined = scores[:, None] + transitions
            best_previous = np.argmax(combined, axis=0)
            best_scores = combined[
                best_previous, np.arange(len(current_group))
            ]

            new_scores = best_scores + current_emissions

            if not np.isfinite(new_scores).any():
                raise ValueError(
                    f"No valid directed route transition for trip {trip_id} "
                    f"between observations {previous_obs} and {current_obs}."
                )

            backpointers.append(best_previous)
            scores = new_scores

        # Backtrack from the highest-scoring final candidate.
        selected_indices = [int(np.argmax(scores))]

        for pointers in reversed(backpointers):
            selected_indices.append(
                int(pointers[selected_indices[-1]])
            )

        selected_indices.reverse()

        for group, selected_index in zip(groups, selected_indices):
            selected = group.iloc[selected_index].copy()
            selected["viterbi_log_probability"] = float(
                scores[selected_indices[-1]]
            )
            decoded_trips.append(selected)

    if not decoded_trips:
        return pd.DataFrame()

    return pd.DataFrame(decoded_trips).reset_index(drop=True)


__all__ = [
    "transition_distance",
    "transition_log_probability",
    "build_transition_matrix",
    "viterbi_decode",
]
