"""Unit tests for the HMM Transition Probability Model.

Task 2: Step 2 of PathSnap Device 3.
Tests GPS displacement, arc projection, directed network distance,
transition probabilities, GPS dropout scaling, and transition matrix construction.
"""

import math
from pathlib import Path
from typing import Dict, Tuple
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from decoder.transition_model import (
    ArcProjection,
    build_transition_matrix,
    calculate_effective_beta,
    gps_displacement_m,
    log_transition_probability,
    network_transition_distance,
    normalize_transition_matrix,
    project_point_onto_arc,
    transition_probability,
)
from graph.graph_builder import EARTH_RADIUS_METERS, build_road_graph, haversine_distance


@pytest.fixture
def linear_graph() -> Tuple[nx.MultiDiGraph, Dict[int, Tuple[int, int]]]:
    """Fixture providing a 3-node linear road graph: 0 -> 1 -> 2."""
    G = nx.MultiDiGraph()
    # At latitude 0.0 for simple calculation:
    # 0.001 deg lon ~ 111.195 meters
    G.add_node(0, node_id=0, longitude=0.0, latitude=0.0)
    G.add_node(1, node_id=1, longitude=0.001, latitude=0.0)
    G.add_node(2, node_id=2, longitude=0.002, latitude=0.0)

    # Arc 0: 0 -> 1 (100.0 m)
    G.add_edge(0, 1, key=0, arc_id=0, length_m=100.0, from_node_id=0, to_node_id=1)
    # Arc 1: 1 -> 2 (100.0 m)
    G.add_edge(1, 2, key=1, arc_id=1, length_m=100.0, from_node_id=1, to_node_id=2)

    arc_lookup = {0: (0, 1), 1: (1, 2)}
    G.graph["arc_lookup"] = arc_lookup
    return G, arc_lookup


@pytest.fixture
def branching_and_parallel_graph() -> Tuple[nx.MultiDiGraph, Dict[int, Tuple[int, int]]]:
    """Fixture with branching paths, parallel edges, and disconnected components."""
    G = nx.MultiDiGraph()
    # Component A:
    # Node 0 -> Node 1 has parallel arcs 10 (100m) and 11 (250m)
    # Node 1 -> Node 2: Arc 20 (150m)
    # Node 2 -> Node 0: Arc 30 (300m) - creates return loop
    G.add_node(0, node_id=0, longitude=8.580, latitude=49.850)
    G.add_node(1, node_id=1, longitude=8.581, latitude=49.850)
    G.add_node(2, node_id=2, longitude=8.582, latitude=49.850)

    G.add_edge(0, 1, key=10, arc_id=10, length_m=100.0, from_node_id=0, to_node_id=1)
    G.add_edge(0, 1, key=11, arc_id=11, length_m=250.0, from_node_id=0, to_node_id=1)
    G.add_edge(1, 2, key=20, arc_id=20, length_m=150.0, from_node_id=1, to_node_id=2)
    G.add_edge(2, 0, key=30, arc_id=30, length_m=300.0, from_node_id=2, to_node_id=0)

    # Component B (disconnected from Component A):
    G.add_node(90, node_id=90, longitude=8.600, latitude=49.870)
    G.add_node(91, node_id=91, longitude=8.601, latitude=49.870)
    G.add_edge(90, 91, key=99, arc_id=99, length_m=80.0, from_node_id=90, to_node_id=91)

    # Zero-length arc (coincident node coordinates):
    G.add_node(50, node_id=50, longitude=8.590000, latitude=49.860000)
    G.add_node(51, node_id=51, longitude=8.590000, latitude=49.860000)
    G.add_edge(50, 51, key=55, arc_id=55, length_m=0.0, from_node_id=50, to_node_id=51)

    arc_lookup = {
        10: (0, 1),
        11: (0, 1),
        20: (1, 2),
        30: (2, 0),
        99: (90, 91),
        55: (50, 51),
    }
    G.graph["arc_lookup"] = arc_lookup
    return G, arc_lookup


# ==============================================================================
# 1. GPS Displacement Calculation
# ==============================================================================

def test_gps_displacement_calculation():
    """Test standard displacement between two consecutive GPS observations."""
    p1 = {"longitude": 8.5810, "latitude": 49.8500, "timestamp": 10.0}
    p2 = {"longitude": 8.5815, "latitude": 49.8505, "timestamp": 12.0}

    dist_m, delta_t = gps_displacement_m(p1, p2)
    expected_dist = haversine_distance(8.5810, 49.8500, 8.5815, 49.8505)

    assert dist_m == pytest.approx(expected_dist, rel=1e-5)
    assert delta_t == pytest.approx(2.0, abs=1e-6)


def test_gps_displacement_tuples_and_series():
    """Test that gps_displacement_m accepts tuples, lists, and pandas Series."""
    t1 = (8.5810, 49.8500, 1.0)
    t2 = (8.5820, 49.8500, 3.5)
    d, dt = gps_displacement_m(t1, t2)
    assert d > 0.0
    assert dt == 2.5

    s1 = pd.Series({"longitude": 8.5810, "latitude": 49.8500, "timestamp": 100.0})
    s2 = pd.Series({"longitude": 8.5810, "latitude": 49.8510, "timestamp": 101.0})
    d_s, dt_s = gps_displacement_m(s1, s2)
    assert d_s > 0.0
    assert dt_s == 1.0


def test_gps_displacement_identical_coordinates():
    """Test identical GPS coordinates produce 0.0m displacement with positive delta_t."""
    p1 = {"longitude": 8.5810, "latitude": 49.8500, "timestamp": 10.0}
    p2 = {"longitude": 8.5810, "latitude": 49.8500, "timestamp": 11.0}

    dist_m, delta_t = gps_displacement_m(p1, p2)
    assert dist_m == 0.0
    assert delta_t == 1.0


def test_gps_displacement_known_geographic_distance():
    """Test GPS displacement matches known 1-degree offset along equator (~111,195 m)."""
    p1 = (0.0, 0.0, 10.0)
    p2 = (1.0, 0.0, 20.0)

    dist_m, delta_t = gps_displacement_m(p1, p2)
    expected = (math.pi / 180.0) * EARTH_RADIUS_METERS
    assert dist_m == pytest.approx(expected, rel=1e-4)
    assert delta_t == 10.0


def test_gps_displacement_non_increasing_timestamps():
    """Test that non-increasing or negative timestamps raise ValueError."""
    p1 = (8.58, 49.85, 10.0)
    p2_equal = (8.59, 49.85, 10.0)
    p2_backward = (8.59, 49.85, 9.0)

    with pytest.raises(ValueError, match="strictly increasing"):
        gps_displacement_m(p1, p2_equal)

    with pytest.raises(ValueError, match="strictly increasing"):
        gps_displacement_m(p1, p2_backward)


def test_gps_displacement_invalid_coordinates():
    """Test that invalid, NaN, or out-of-bounds coordinates raise ValueError."""
    with pytest.raises(ValueError):
        gps_displacement_m((200.0, 49.0, 1.0), (8.0, 49.0, 2.0))

    with pytest.raises(ValueError):
        gps_displacement_m((8.0, 49.0, 1.0), (8.0, 95.0, 2.0))

    with pytest.raises(ValueError):
        gps_displacement_m((np.nan, 49.0, 1.0), (8.0, 49.0, 2.0))


# ==============================================================================
# 2. Point Projection onto Arc
# ==============================================================================

def test_project_point_onto_arc_midpoint():
    """Test projecting a point equidistant above the segment midpoint."""
    arc_start = (0.0, 0.0)
    arc_end = (0.01, 0.0)  # ~1111.95 meters along equator
    p_lon, p_lat = 0.005, 0.001  # Point above midpoint

    proj = project_point_onto_arc(p_lon, p_lat, arc_start, arc_end)
    assert isinstance(proj, ArcProjection)
    assert proj.fraction_along_arc == pytest.approx(0.5, abs=1e-4)
    assert proj.projected_longitude == pytest.approx(0.005, abs=1e-5)
    assert proj.projected_latitude == pytest.approx(0.0, abs=1e-5)
    # Perpendicular distance is ~0.001 deg lat = ~111.2 m
    assert proj.perpendicular_distance_m == pytest.approx(111.195, rel=1e-3)


def test_project_point_clipping_endpoints():
    """Test that points extending beyond endpoints are clipped to [0.0, 1.0]."""
    arc_start = (0.0, 0.0)
    arc_end = (0.01, 0.0)

    # Point before start
    proj_before = project_point_onto_arc(-0.005, 0.001, arc_start, arc_end)
    assert proj_before.fraction_along_arc == 0.0
    assert proj_before.projected_longitude == 0.0
    assert proj_before.projected_latitude == 0.0

    # Point after end
    proj_after = project_point_onto_arc(0.015, 0.001, arc_start, arc_end)
    assert proj_after.fraction_along_arc == 1.0
    assert proj_after.projected_longitude == 0.01
    assert proj_after.projected_latitude == 0.0


def test_project_point_zero_length_arc():
    """Test safe projection onto a zero-length arc where start and end coincide."""
    arc_start = (8.580000, 49.850000)
    arc_end = (8.580000, 49.850000)

    proj = project_point_onto_arc(8.580500, 49.850500, arc_start, arc_end)
    assert proj.fraction_along_arc == 0.0
    assert proj.projected_longitude == pytest.approx(8.580000)
    assert proj.projected_latitude == pytest.approx(49.850000)
    assert proj.perpendicular_distance_m > 0.0


# ==============================================================================
# 3. Directed Network Transition Distance
# ==============================================================================

def test_same_arc_forward_movement(linear_graph):
    """Test movement along the same directed arc when q2 >= q1."""
    G, lookup = linear_graph
    # Arc 0 is 0 -> 1 with length 100.0m
    # 0 is at (0, 0), 1 is at (0.001, 0)
    # Obs 1 at q1 = 0.2: (0.0002, 0.0)
    # Obs 2 at q2 = 0.7: (0.0007, 0.0)
    gps1 = (0.0002, 0.0, 10.0)
    gps2 = (0.0007, 0.0, 11.0)

    d_net = network_transition_distance(
        previous_arc_id=0,
        current_arc_id=0,
        previous_gps=gps1,
        current_gps=gps2,
        road_graph=G,
        arc_lookup=lookup,
    )
    # Expected: (0.7 - 0.2) * 100.0 = 50.0 m
    assert d_net == pytest.approx(50.0, abs=1.0)


def test_adjacent_directed_arcs(linear_graph):
    """Test transition between adjacent consecutive directed arcs (v1 == u2)."""
    G, lookup = linear_graph
    # Arc 0 (0 -> 1, L=100m) and Arc 1 (1 -> 2, L=100m)
    # Obs 1 at q1 = 0.8 on Arc 0: (0.0008, 0.0)
    # Obs 2 at q2 = 0.3 on Arc 1: (0.0013, 0.0)
    gps1 = (0.0008, 0.0, 10.0)
    gps2 = (0.0013, 0.0, 11.0)

    d_net = network_transition_distance(
        previous_arc_id=0,
        current_arc_id=1,
        previous_gps=gps1,
        current_gps=gps2,
        road_graph=G,
        arc_lookup=lookup,
    )
    # Expected: (1 - 0.8)*100 + 0 + 0.3*100 = 20 + 30 = 50.0 m
    assert d_net == pytest.approx(50.0, abs=1.0)


def test_valid_multi_edge_path(branching_and_parallel_graph):
    """Test transition requiring a multi-edge path across intermediate nodes."""
    G, lookup = branching_and_parallel_graph
    # Arc 10 (0 -> 1, L=100m), Arc 20 (1 -> 2, L=150m), Arc 30 (2 -> 0, L=300m)
    # Transition from Arc 10 to Arc 30:
    # v1 is Node 1, u2 is Node 2.
    # Shortest path from Node 1 to Node 2 is Arc 20 (length 150m).
    gps1 = (8.5805, 49.850, 1.0)  # q1 ~ 0.5 on Arc 10
    gps2 = (8.5810, 49.850, 2.0)  # q2 on Arc 30 (from Node 2 to Node 0)

    d_net = network_transition_distance(
        previous_arc_id=10,
        current_arc_id=30,
        previous_gps=gps1,
        current_gps=gps2,
        road_graph=G,
        arc_lookup=lookup,
    )
    assert math.isfinite(d_net)
    assert d_net > 150.0  # Must include the 150m intermediate arc


def test_disconnected_directed_roads(branching_and_parallel_graph):
    """Test transition between disconnected components reports infinity."""
    G, lookup = branching_and_parallel_graph
    # Arc 10 in Component A (0 -> 1), Arc 99 in Component B (90 -> 91)
    gps1 = (8.5805, 49.850, 1.0)
    gps2 = (8.6005, 49.870, 2.0)

    d_net = network_transition_distance(
        previous_arc_id=10,
        current_arc_id=99,
        previous_gps=gps1,
        current_gps=gps2,
        road_graph=G,
        arc_lookup=lookup,
    )
    assert math.isinf(d_net)


def test_reverse_direction_invalid_movement(linear_graph):
    """Test that reverse motion on a directed one-way road without return loop yields infinity."""
    G, lookup = linear_graph
    # Arc 0 is one-way 0 -> 1. No loop from 1 back to 0.
    # Obs 1 at q1 = 0.8: (0.0008, 0.0)
    # Obs 2 at q2 = 0.2: (0.0002, 0.0)  (backward movement)
    gps1 = (0.0008, 0.0, 10.0)
    gps2 = (0.0002, 0.0, 11.0)

    d_net = network_transition_distance(
        previous_arc_id=0,
        current_arc_id=0,
        previous_gps=gps1,
        current_gps=gps2,
        road_graph=G,
        arc_lookup=lookup,
    )
    assert math.isinf(d_net)


def test_parallel_arc_handling(branching_and_parallel_graph):
    """Test that parallel edges (0 -> 1) preserve their distinct lengths."""
    G, lookup = branching_and_parallel_graph
    # Arc 10 has length 100m, Arc 11 has length 250m between identical nodes 0 -> 1
    gps1 = (8.5805, 49.850, 1.0)
    gps2 = (8.5808, 49.850, 2.0)

    d_net_10 = network_transition_distance(10, 10, gps1, gps2, G, arc_lookup=lookup)
    d_net_11 = network_transition_distance(11, 11, gps1, gps2, G, arc_lookup=lookup)

    assert d_net_10 > 0.0
    assert d_net_11 > 0.0
    # Length of Arc 11 is 2.5x length of Arc 10, so proportional net distance must be larger
    assert d_net_11 > d_net_10
    assert (d_net_11 / d_net_10) == pytest.approx(2.5, rel=1e-2)


def test_zero_length_edge_handling(branching_and_parallel_graph):
    """Test handling of a zero-length arc without crashing or division by zero."""
    G, lookup = branching_and_parallel_graph
    # Arc 55 has length 0.0m
    gps1 = (8.5900, 49.8600, 1.0)
    gps2 = (8.5900, 49.8600, 2.0)

    d_net = network_transition_distance(55, 55, gps1, gps2, G, arc_lookup=lookup)
    assert d_net == 0.0


# ==============================================================================
# 4. Transition Probability & Log Probability
# ==============================================================================

def test_transition_probability_exact_match():
    """Test that d_net == d_gps produces P = 1 / beta."""
    beta = 5.0
    p = transition_probability(d_net=50.0, d_gps=50.0, beta=beta)
    assert p == pytest.approx(1.0 / beta, rel=1e-6)

    log_p = log_transition_probability(d_net=50.0, d_gps=50.0, beta=beta)
    assert log_p == pytest.approx(-math.log(beta), rel=1e-6)


def test_transition_probability_exponential_decay():
    """Test exponential decay as |d_net - d_gps| increases."""
    beta = 10.0
    d_gps = 100.0

    p_close = transition_probability(d_net=105.0, d_gps=d_gps, beta=beta)
    p_far = transition_probability(d_net=150.0, d_gps=d_gps, beta=beta)

    assert p_close > p_far
    expected_ratio = math.exp(-5.0 / beta) / math.exp(-50.0 / beta)
    assert (p_close / p_far) == pytest.approx(expected_ratio, rel=1e-4)


def test_transition_probability_disconnected_floor():
    """Test that infinite d_net returns epsilon floor."""
    eps = 1e-6
    p = transition_probability(d_net=float("inf"), d_gps=20.0, beta=5.0, epsilon=eps)
    assert p == eps

    log_p = log_transition_probability(d_net=float("inf"), d_gps=20.0, beta=5.0, epsilon=eps)
    assert log_p == math.log(eps)


def test_invalid_beta_raises():
    """Test that non-positive beta raises ValueError."""
    with pytest.raises(ValueError, match="beta must be a positive finite float"):
        transition_probability(d_net=10.0, d_gps=10.0, beta=0.0)

    with pytest.raises(ValueError, match="beta must be a positive finite float"):
        transition_probability(d_net=10.0, d_gps=10.0, beta=-2.5)

    with pytest.raises(ValueError, match="beta must be a positive finite float"):
        log_transition_probability(d_net=10.0, d_gps=10.0, beta=-1.0)


def test_invalid_distances_raise():
    """Test that negative d_gps raises ValueError."""
    with pytest.raises(ValueError, match="d_gps must be a non-negative finite float"):
        transition_probability(d_net=10.0, d_gps=-5.0, beta=5.0)

    with pytest.raises(ValueError, match="d_gps must be a non-negative finite float"):
        log_transition_probability(d_net=10.0, d_gps=-5.0, beta=5.0)


# ==============================================================================
# 5. GPS Dropout Handling
# ==============================================================================

def test_gps_dropout_scaling():
    """Test scaling of beta when delta_t exceeds 5.0 seconds."""
    nominal_beta = 5.0

    # Normal interval (<= 5s): unscaled
    assert calculate_effective_beta(nominal_beta, delta_t=1.0) == 5.0
    assert calculate_effective_beta(nominal_beta, delta_t=5.0) == 5.0

    # Dropout (> 5s): scales proportionally
    assert calculate_effective_beta(nominal_beta, delta_t=6.0) == 30.0  # 5 * 6
    assert calculate_effective_beta(nominal_beta, delta_t=10.0) == 50.0  # 5 * 10


# ==============================================================================
# 6. Build Transition Matrix & Caching
# ==============================================================================

def test_build_transition_matrix_dimensions_and_keys(linear_graph):
    """Test that build_transition_matrix produces correct shape, keys, and values."""
    G, lookup = linear_graph
    prev_cands = [0, 1]
    curr_cands = [0, 1]

    gps1 = (0.0002, 0.0, 10.0)
    gps2 = (0.0008, 0.0, 12.0)

    result = build_transition_matrix(
        previous_candidates=prev_cands,
        current_candidates=curr_cands,
        previous_gps=gps1,
        current_gps=gps2,
        road_graph=G,
        beta=5.0,
        arc_lookup=lookup,
    )

    assert result["transition_matrix"].shape == (2, 2)
    assert result["log_transition_matrix"].shape == (2, 2)
    assert result["network_distances"].shape == (2, 2)
    assert result["previous_arcs"] == [0, 1]
    assert result["current_arcs"] == [0, 1]
    assert result["delta_t"] == 2.0
    assert result["effective_beta"] == 5.0
    assert isinstance(result["disconnected_transitions"], list)


def test_build_transition_matrix_with_dataframe_candidates(linear_graph):
    """Test passing pandas DataFrames containing 'arc_id' column."""
    G, lookup = linear_graph
    df_prev = pd.DataFrame({"trip_id": ["t1"], "obs_idx": [0], "arc_id": [0]})
    df_curr = pd.DataFrame({"trip_id": ["t1"], "obs_idx": [1], "arc_id": [1]})

    gps1 = (0.0002, 0.0, 1.0)
    gps2 = (0.0012, 0.0, 2.0)

    result = build_transition_matrix(df_prev, df_curr, gps1, gps2, G, arc_lookup=lookup)
    assert result["transition_matrix"].shape == (1, 1)
    assert result["transition_matrix"][0, 0] > 0.0


def test_shortest_path_caching(branching_and_parallel_graph):
    """Test that shortest_path_cache is populated and reutilized across queries."""
    G, lookup = branching_and_parallel_graph
    cache: Dict[Tuple[int, int], float] = {}

    gps1 = (8.5805, 49.850, 1.0)
    gps2 = (8.5815, 49.850, 2.0)

    # Arc 10 and 11 end at Node 1 (v1 = 1).
    # Arc 30 starts at Node 2 (u2 = 2). Non-adjacent: requires shortest path (1, 2).
    # Arc 99 starts at Node 90 (u2 = 90). Disconnected: requires shortest path (1, 90).
    res1 = build_transition_matrix(
        [10, 11], [30, 99], gps1, gps2, G, shortest_path_cache=cache, arc_lookup=lookup
    )
    initial_cache_size = len(cache)
    assert initial_cache_size > 0
    assert (1, 2) in cache  # Endpoint v1=1 to u2=2 was cached
    assert (1, 90) in cache  # Endpoint v1=1 to u2=90 was cached (as inf)

    # Second run reuses cache without adding duplicate keys
    res2 = build_transition_matrix(
        [10, 11], [30, 99], gps1, gps2, G, shortest_path_cache=cache, arc_lookup=lookup
    )
    assert len(cache) == initial_cache_size
    np.testing.assert_allclose(res1["transition_matrix"], res2["transition_matrix"])


def test_normalize_transition_matrix_utility():
    """Test that normalize_transition_matrix normalizes each row to 1.0."""
    raw = np.array([
        [0.2, 0.3],
        [0.05, 0.05],
    ])
    norm = normalize_transition_matrix(raw)
    assert norm.shape == (2, 2)
    assert np.allclose(norm.sum(axis=1), [1.0, 1.0])


# ==============================================================================
# 7. Integration with Real Synthetic Dataset on Disk
# ==============================================================================

def test_transition_model_with_disk_synthetic_dataset():
    """Test transition model on the synthetic test dataset from data/synthetic/."""
    synth_dir = Path(__file__).resolve().parent.parent / "data" / "synthetic"
    nodes_path = synth_dir / "synthetic_sample.nodes"
    arcs_path = synth_dir / "synthetic_sample.arcs"

    if not nodes_path.is_file() or not arcs_path.is_file():
        pytest.skip("Synthetic dataset files not found on disk.")

    road_graph, arc_lookup, _ = build_road_graph(nodes_path, arcs_path)

    # Arcs: 0 (0 -> 1), 1 (1 -> 2), 5 (2 -> 4)
    # Consecutive observations along path 0 -> 1 -> 2
    gps_prev = {"longitude": 8.5815, "latitude": 49.8580, "timestamp": 10.0}
    gps_curr = {"longitude": 8.5825, "latitude": 49.8590, "timestamp": 12.0}

    result = build_transition_matrix(
        previous_candidates=[0, 4],  # Arc 0 and parallel Arc 4
        current_candidates=[1, 2],   # Arc 1 (1->2) and branching Arc 2 (1->3)
        previous_gps=gps_prev,
        current_gps=gps_curr,
        road_graph=road_graph,
        arc_lookup=arc_lookup,
        beta=5.0,
    )

    assert result["transition_matrix"].shape == (2, 2)
    assert np.all(result["transition_matrix"] >= 1e-6)
    assert result["num_disconnected"] == 0
