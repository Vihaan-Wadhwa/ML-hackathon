"""Unit tests for the Log-Space Viterbi Decoder.

Task 2: Step 3 of PathSnap Device 3.
Covers all 20 required scenarios:
1. Single GPS observation
2. Single candidate per observation
3. Multiple candidates per observation
4. Correct backpointer reconstruction
5. Connected roads preferred when emission evidence supports them
6. Strong emissions competing with disconnected roads
7. Disconnected transitions with probability floor
8. Variable candidate counts
9. GPS gaps greater than five seconds
10. Missing candidate observations
11. Nearest-road fallback
12. Invalid arc IDs
13. Invalid emission probabilities
14. Empty candidate input
15. Missing GPS coordinates
16. One trip per decoding call
17. Stable log-space calculation
18. Deterministic tie handling
19. Original arc IDs preserved
20. Output length matches the number of valid GPS observations
"""

import math
from pathlib import Path
from typing import Dict, Tuple
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from decoder.viterbi import (
    find_nearest_candidate_arc,
    log_normalize_transition_matrix,
    reconstruct_route,
)
from graph.graph_builder import build_road_graph


@pytest.fixture
def test_network() -> Tuple[nx.MultiDiGraph, Dict[int, Tuple[int, int]]]:
    """Fixture providing a connected path and a disconnected road for testing.

    Graph Topology:
    - Main Path: Node 0 -> Node 1 (Arc 10, L=100m) -> Node 2 (Arc 20, L=100m) -> Node 3 (Arc 30, L=100m)
    - Parallel Arc: Node 0 -> Node 1 (Arc 15, L=100m)
    - Disconnected Arc: Node 90 -> Node 91 (Arc 99, L=100m)
    """
    G = nx.MultiDiGraph()
    G.add_node(0, node_id=0, longitude=8.580, latitude=49.850)
    G.add_node(1, node_id=1, longitude=8.581, latitude=49.850)
    G.add_node(2, node_id=2, longitude=8.582, latitude=49.850)
    G.add_node(3, node_id=3, longitude=8.583, latitude=49.850)

    G.add_edge(0, 1, key=10, arc_id=10, length_m=100.0, from_node_id=0, to_node_id=1)
    G.add_edge(0, 1, key=15, arc_id=15, length_m=100.0, from_node_id=0, to_node_id=1)
    G.add_edge(1, 2, key=20, arc_id=20, length_m=100.0, from_node_id=1, to_node_id=2)
    G.add_edge(2, 3, key=30, arc_id=30, length_m=100.0, from_node_id=2, to_node_id=3)

    # Disconnected road
    G.add_node(90, node_id=90, longitude=8.600, latitude=49.870)
    G.add_node(91, node_id=91, longitude=8.601, latitude=49.870)
    G.add_edge(90, 91, key=99, arc_id=99, length_m=100.0, from_node_id=90, to_node_id=91)

    arc_lookup = {10: (0, 1), 15: (0, 1), 20: (1, 2), 30: (2, 3), 99: (90, 91)}
    G.graph["arc_lookup"] = arc_lookup
    return G, arc_lookup


# ==============================================================================
# 1. Single GPS observation
# ==============================================================================
def test_single_gps_observation(test_network):
    G, lookup = test_network
    track = pd.DataFrame({
        "obs_idx": [0],
        "longitude": [8.5805],
        "latitude": [49.8500],
        "timestamp": [1.0],
    })
    cands = pd.DataFrame({
        "trip_id": ["t1", "t1"],
        "obs_idx": [0, 0],
        "arc_id": [10, 15],
        "match_probability": [0.8, 0.2],
    })

    route = reconstruct_route(cands, G, track_df=track, arc_lookup=lookup)
    assert route == [10]


# ==============================================================================
# 2. Single candidate per observation
# ==============================================================================
def test_single_candidate_per_observation(test_network):
    G, lookup = test_network
    track = pd.DataFrame({
        "obs_idx": [0, 1, 2],
        "longitude": [8.5805, 8.5815, 8.5825],
        "latitude": [49.8500, 49.8500, 49.8500],
        "timestamp": [1.0, 2.0, 3.0],
    })
    cands = pd.DataFrame({
        "trip_id": ["t1", "t1", "t1"],
        "obs_idx": [0, 1, 2],
        "arc_id": [10, 20, 30],
        "match_probability": [1.0, 1.0, 1.0],
    })

    route = reconstruct_route(cands, G, track_df=track, arc_lookup=lookup)
    assert route == [10, 20, 30]


# ==============================================================================
# 3. Multiple candidates per observation
# ==============================================================================
def test_multiple_candidates_per_observation(test_network):
    G, lookup = test_network
    track = pd.DataFrame({
        "obs_idx": [0, 1],
        "longitude": [8.5805, 8.5815],
        "latitude": [49.8500, 49.8500],
        "timestamp": [1.0, 2.0],
    })
    # Obs 0 has Arc 10 and Arc 15; Obs 1 has Arc 20 and Arc 99
    cands = pd.DataFrame({
        "trip_id": ["t1"] * 4,
        "obs_idx": [0, 0, 1, 1],
        "arc_id": [10, 15, 20, 99],
        "match_probability": [0.6, 0.4, 0.7, 0.3],
    })

    route = reconstruct_route(cands, G, track_df=track, arc_lookup=lookup)
    assert route == [10, 20]


# ==============================================================================
# 4. Correct backpointer reconstruction
# ==============================================================================
def test_correct_backpointer_reconstruction(test_network):
    G, lookup = test_network
    track = pd.DataFrame({
        "obs_idx": [0, 1, 2],
        "longitude": [8.5805, 8.5815, 8.5825],
        "latitude": [49.8500, 49.8500, 49.8500],
        "timestamp": [10.0, 11.0, 12.0],
    })
    cands = pd.DataFrame({
        "trip_id": ["t1"] * 6,
        "obs_idx": [0, 0, 1, 1, 2, 2],
        "arc_id": [10, 99, 20, 99, 30, 99],
        "match_probability": [0.55, 0.45, 0.55, 0.45, 0.55, 0.45],
    })

    route, diag = reconstruct_route(
        cands, G, track_df=track, return_diagnostics=True, arc_lookup=lookup
    )
    assert route == [10, 20, 30]
    assert diag["predicted_arcs"] == [10, 20, 30]
    assert diag["final_sequence_log_score"] > -20.0


# ==============================================================================
# 5. Connected roads preferred when emission evidence supports them
# ==============================================================================
def test_connected_roads_preferred(test_network):
    G, lookup = test_network
    track = pd.DataFrame({
        "obs_idx": [0, 1],
        "longitude": [8.5805, 8.5815],
        "latitude": [49.8500, 49.8500],
        "timestamp": [1.0, 2.0],
    })
    # Both candidates have equal emission (0.5 vs 0.5)
    cands = pd.DataFrame({
        "trip_id": ["t1"] * 4,
        "obs_idx": [0, 0, 1, 1],
        "arc_id": [10, 99, 20, 99],
        "match_probability": [0.5, 0.5, 0.5, 0.5],
    })

    route = reconstruct_route(cands, G, track_df=track, arc_lookup=lookup)
    # 10 -> 20 is physically connected; 99 is disconnected
    assert route == [10, 20]


# ==============================================================================
# 6. Strong emissions competing with disconnected roads
# ==============================================================================
def test_strong_emissions_competing_with_disconnected_roads(test_network):
    G, lookup = test_network
    track = pd.DataFrame({
        "obs_idx": [0, 1],
        "longitude": [8.5805, 8.5815],
        "latitude": [49.8500, 49.8500],
        "timestamp": [1.0, 2.0],
    })
    # Disconnected Arc 99 has higher classifier score (0.65 vs 0.35)
    # Independent ArgMax would pick [99, 99].
    # But Viterbi transition penalty heavily penalizes disconnected transitions!
    cands = pd.DataFrame({
        "trip_id": ["t1"] * 4,
        "obs_idx": [0, 0, 1, 1],
        "arc_id": [10, 99, 20, 99],
        "match_probability": [0.40, 0.60, 0.40, 0.60],
    })

    route = reconstruct_route(cands, G, beta=5.0, track_df=track, arc_lookup=lookup)
    # Viterbi selects the continuous connected path 10 -> 20
    assert route == [10, 20]


# ==============================================================================
# 7. Disconnected transitions with probability floor
# ==============================================================================
def test_disconnected_transitions_with_probability_floor(test_network):
    G, lookup = test_network
    track = pd.DataFrame({
        "obs_idx": [0, 1],
        "longitude": [8.5805, 8.6005],
        "latitude": [49.8500, 49.8700],
        "timestamp": [1.0, 2.0],
    })
    # Only disconnected transitions are available
    cands = pd.DataFrame({
        "trip_id": ["t1", "t1"],
        "obs_idx": [0, 1],
        "arc_id": [10, 99],
        "match_probability": [1.0, 1.0],
    })

    route, diag = reconstruct_route(
        cands, G, track_df=track, return_diagnostics=True, arc_lookup=lookup
    )
    assert route == [10, 99]
    assert diag["disconnected_transitions_count"] >= 1


# ==============================================================================
# 8. Variable candidate counts
# ==============================================================================
def test_variable_candidate_counts(test_network):
    G, lookup = test_network
    track = pd.DataFrame({
        "obs_idx": [0, 1, 2],
        "longitude": [8.5805, 8.5815, 8.5825],
        "latitude": [49.8500, 49.8500, 49.8500],
        "timestamp": [1.0, 2.0, 3.0],
    })
    # Obs 0 has 2 cands, Obs 1 has 1 cand, Obs 2 has 3 cands
    cands = pd.DataFrame({
        "trip_id": ["t1"] * 6,
        "obs_idx": [0, 0, 1, 2, 2, 2],
        "arc_id": [10, 15, 20, 30, 10, 99],
        "match_probability": [0.6, 0.4, 1.0, 0.7, 0.2, 0.1],
    })

    route, diag = reconstruct_route(
        cands, G, track_df=track, return_diagnostics=True, arc_lookup=lookup
    )
    assert route == [10, 20, 30]
    assert diag["candidate_counts_per_obs"] == [2, 1, 3]


# ==============================================================================
# 9. GPS gaps greater than five seconds
# ==============================================================================
def test_gps_gaps_greater_than_five_seconds(test_network):
    G, lookup = test_network
    track = pd.DataFrame({
        "obs_idx": [0, 1],
        "longitude": [8.5805, 8.5815],
        "latitude": [49.8500, 49.8500],
        "timestamp": [1.0, 15.0],  # 14-second GPS outage
    })
    cands = pd.DataFrame({
        "trip_id": ["t1"] * 4,
        "obs_idx": [0, 0, 1, 1],
        "arc_id": [10, 15, 20, 99],
        "match_probability": [0.7, 0.3, 0.8, 0.2],
    })

    route = reconstruct_route(cands, G, track_df=track, arc_lookup=lookup)
    assert route == [10, 20]


# ==============================================================================
# 10. Missing candidate observations & Output Length
# ==============================================================================
def test_missing_candidate_observations(test_network):
    G, lookup = test_network
    # Track has 3 observations (0, 1, 2)
    track = pd.DataFrame({
        "obs_idx": [0, 1, 2],
        "longitude": [8.5805, 8.5815, 8.5825],
        "latitude": [49.8500, 49.8500, 49.8500],
        "timestamp": [1.0, 2.0, 3.0],
    })
    # Candidates table ONLY has observations 0 and 2 (Obs 1 is missing!)
    cands = pd.DataFrame({
        "trip_id": ["t1"] * 2,
        "obs_idx": [0, 2],
        "arc_id": [10, 30],
        "match_probability": [1.0, 1.0],
    })

    route, diag = reconstruct_route(
        cands, G, track_df=track, return_diagnostics=True, arc_lookup=lookup
    )
    assert len(route) == 3
    assert diag["fallback_observations_count"] == 1
    assert diag["fallback_obs_indices"] == [1]
    # Obs 1 fell back to the nearest road arc, which is Arc 20!
    assert route[1] == 20


# ==============================================================================
# 11. Nearest-road fallback
# ==============================================================================
def test_nearest_candidate_arc_helper(test_network):
    G, lookup = test_network
    # Point near Arc 20 (Node 1 -> Node 2, around lon 8.5815)
    gps = (8.5815, 49.8502)
    nearest_arc, dist_m = find_nearest_candidate_arc(gps, G, lookup)
    assert nearest_arc == 20
    assert dist_m > 0.0 and dist_m < 50.0


# ==============================================================================
# 12. Invalid arc IDs rejection
# ==============================================================================
def test_invalid_arc_ids_rejection(test_network):
    G, lookup = test_network
    track = pd.DataFrame({
        "obs_idx": [0],
        "longitude": [8.5805],
        "latitude": [49.8500],
        "timestamp": [1.0],
    })
    cands = pd.DataFrame({
        "trip_id": ["t1"],
        "obs_idx": [0],
        "arc_id": [99999],  # Non-existent arc ID
        "match_probability": [1.0],
    })

    with pytest.raises(ValueError, match="do not exist in the road graph"):
        reconstruct_route(cands, G, track_df=track, arc_lookup=lookup)


# ==============================================================================
# 13. Invalid emission probabilities rejection
# ==============================================================================
def test_invalid_emission_probabilities_rejection(test_network):
    G, lookup = test_network
    track = pd.DataFrame({
        "obs_idx": [0],
        "longitude": [8.5805],
        "latitude": [49.8500],
        "timestamp": [1.0],
    })
    cands = pd.DataFrame({
        "trip_id": ["t1"],
        "obs_idx": [0],
        "arc_id": [10],
        "emission_probability": [-0.5],  # Negative emission
    })

    with pytest.raises(ValueError, match="Invalid emission probability"):
        reconstruct_route(cands, G, track_df=track, arc_lookup=lookup)


# ==============================================================================
# 14. Empty candidate input rejection
# ==============================================================================
def test_empty_candidate_input_rejection(test_network):
    G, lookup = test_network
    track = pd.DataFrame({
        "obs_idx": [0],
        "longitude": [8.5805],
        "latitude": [49.8500],
        "timestamp": [1.0],
    })
    empty_cands = pd.DataFrame(columns=["trip_id", "obs_idx", "arc_id", "match_probability"])

    with pytest.raises(ValueError, match="cannot be empty"):
        reconstruct_route(empty_cands, G, track_df=track, arc_lookup=lookup)


# ==============================================================================
# 15. Missing GPS coordinates rejection
# ==============================================================================
def test_missing_gps_coordinates_rejection(test_network):
    G, lookup = test_network
    cands_no_gps = pd.DataFrame({
        "trip_id": ["t1"],
        "obs_idx": [0],
        "arc_id": [10],
        "match_probability": [1.0],
    })

    with pytest.raises(ValueError, match="GPS coordinates and timestamps must be provided"):
        reconstruct_route(cands_no_gps, G, track_df=None, arc_lookup=lookup)


# ==============================================================================
# 16. One trip per decoding call (Reject mixed trips)
# ==============================================================================
def test_reject_mixed_trip_ids(test_network):
    G, lookup = test_network
    track = pd.DataFrame({
        "obs_idx": [0, 1],
        "longitude": [8.5805, 8.5815],
        "latitude": [49.8500, 49.8500],
        "timestamp": [1.0, 2.0],
    })
    mixed_cands = pd.DataFrame({
        "trip_id": ["trip_01", "trip_02"],  # Mixed trip IDs
        "obs_idx": [0, 1],
        "arc_id": [10, 20],
        "match_probability": [1.0, 1.0],
    })

    with pytest.raises(ValueError, match="Mixed trip IDs"):
        reconstruct_route(mixed_cands, G, track_df=track, arc_lookup=lookup)


# ==============================================================================
# 17. Stable log-space calculation
# ==============================================================================
def test_stable_log_space_calculation():
    """Test that log_normalize_transition_matrix produces finite numbers."""
    extreme_log = np.array([
        [-1000.0, -1000.0, -1000.0],
        [-0.01, -500.0, -13.8],
    ])
    norm = log_normalize_transition_matrix(extreme_log)
    assert np.all(np.isfinite(norm))
    probs = np.exp(norm)
    # Row 1 has viable candidate (-0.01) which normalizes to ~1.0
    assert probs[1, 0] == pytest.approx(1.0, rel=1e-3)
    # Row 0 has all disconnected entries which stay floored at ln(1e-6)
    assert np.all(norm[0] <= math.log(1e-6) + 1e-5)


# ==============================================================================
# 18. Deterministic tie handling
# ==============================================================================
def test_deterministic_tie_handling(test_network):
    G, lookup = test_network
    track = pd.DataFrame({
        "obs_idx": [0],
        "longitude": [8.5805],
        "latitude": [49.8500],
        "timestamp": [1.0],
    })
    # Both arcs have identical probabilities
    cands = pd.DataFrame({
        "trip_id": ["t1", "t1"],
        "obs_idx": [0, 0],
        "arc_id": [10, 15],
        "match_probability": [0.5, 0.5],
    })

    # Repeating call 3 times should give identical result
    res1 = reconstruct_route(cands, G, track_df=track, arc_lookup=lookup)
    res2 = reconstruct_route(cands, G, track_df=track, arc_lookup=lookup)
    res3 = reconstruct_route(cands, G, track_df=track, arc_lookup=lookup)
    assert res1 == res2 == res3 == [10]


# ==============================================================================
# 19. Original arc IDs preserved
# ==============================================================================
def test_original_arc_ids_preserved(test_network):
    G, lookup = test_network
    track = pd.DataFrame({
        "obs_idx": [0, 1],
        "longitude": [8.5805, 8.5815],
        "latitude": [49.8500, 49.8500],
        "timestamp": [1.0, 2.0],
    })
    cands = pd.DataFrame({
        "trip_id": ["t1", "t1"],
        "obs_idx": [0, 1],
        "arc_id": [15, 20],
        "match_probability": [1.0, 1.0],
    })

    route = reconstruct_route(cands, G, track_df=track, arc_lookup=lookup)
    assert route == [15, 20]
    assert all(isinstance(a, int) for a in route)


# ==============================================================================
# 20. Output length matches the number of valid GPS observations
# ==============================================================================
def test_output_length_matches_gps_observations(test_network):
    G, lookup = test_network
    N_obs = 7
    track = pd.DataFrame({
        "obs_idx": list(range(N_obs)),
        "longitude": [8.5805 + i * 0.0003 for i in range(N_obs)],
        "latitude": [49.8500] * N_obs,
        "timestamp": [float(i + 1) for i in range(N_obs)],
    })
    cands = pd.DataFrame({
        "trip_id": ["t1"] * N_obs,
        "obs_idx": list(range(N_obs)),
        "arc_id": [10] * N_obs,
        "match_probability": [1.0] * N_obs,
    })

    route = reconstruct_route(cands, G, track_df=track, arc_lookup=lookup)
    assert len(route) == N_obs
