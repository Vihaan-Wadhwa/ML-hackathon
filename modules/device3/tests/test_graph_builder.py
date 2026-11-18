"""Unit tests for graph_builder: loading, validation, Haversine, and MultiDiGraph construction.

Steps 2, 3 & 4 of PathSnap Device 3 Task 1.
"""

import math
from pathlib import Path
from typing import Tuple
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from graph.graph_builder import (
    EARTH_RADIUS_METERS,
    RoadDataValidationError,
    build_road_graph,
    get_graph_statistics,
    haversine_distance,
    load_nodes,
    load_arcs,
    validate_road_data,
)


@pytest.fixture
def sample_nodes_file(tmp_path: Path) -> Path:
    """Fixture providing a valid 5-node temporary .nodes file."""
    content = (
        "8.581679 49.857880\n"
        "8.584637 49.858295\n"
        "8.580173 49.858567\n"
        "8.584392 49.859119\n"
        "8.584935 49.857396\n"
    )
    file_path = tmp_path / "valid.nodes"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def sample_arcs_file(tmp_path: Path) -> Path:
    """Fixture providing a valid temporary .arcs file with parallel edges and bidirectional connections."""
    content = (
        "0 1\n"
        "1 0\n"
        "0 2\n"
        "1 2\n"
        "0 1\n"  # Duplicate directed pair (parallel edge)
        "2 3\n"
        "3 4\n"
    )
    file_path = tmp_path / "valid.arcs"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def synthetic_network(tmp_path: Path) -> Tuple[Path, Path]:
    """Fixture providing a deterministic 6-node, 7-arc synthetic road network."""
    nodes_content = (
        "8.580000 49.850000\n"  # Node 0
        "8.581000 49.850000\n"  # Node 1
        "8.582000 49.850000\n"  # Node 2
        "8.581000 49.851000\n"  # Node 3 (branch)
        "8.583000 49.850000\n"  # Node 4
        "8.584000 49.850000\n"  # Node 5 (isolated)
    )
    arcs_content = (
        "0 1\n"  # Arc 0: 0 -> 1
        "1 2\n"  # Arc 1: 1 -> 2
        "1 3\n"  # Arc 2: 1 -> 3 (branching)
        "1 0\n"  # Arc 3: 1 -> 0 (reverse direction)
        "0 1\n"  # Arc 4: 0 -> 1 (parallel to Arc 0)
        "2 4\n"  # Arc 5: 2 -> 4
        "4 4\n"  # Arc 6: 4 -> 4 (self-loop)
    )
    nodes_file = tmp_path / "synthetic.nodes"
    arcs_file = tmp_path / "synthetic.arcs"
    nodes_file.write_text(nodes_content, encoding="utf-8")
    arcs_file.write_text(arcs_content, encoding="utf-8")
    return nodes_file, arcs_file


# --- Step 2 Tests: Data Loading and Validation ---

def test_correct_node_loading(sample_nodes_file: Path):
    df = load_nodes(sample_nodes_file)
    assert len(df) == 5
    assert list(df.columns) == ["node_id", "longitude", "latitude"]
    assert np.isclose(df.loc[0, "longitude"], 8.581679)
    assert np.isclose(df.loc[0, "latitude"], 49.857880)


def test_correct_arc_loading(sample_arcs_file: Path):
    df = load_arcs(sample_arcs_file)
    assert len(df) == 7
    assert list(df.columns) == ["arc_id", "from_node_id", "to_node_id"]
    assert df.loc[0, "from_node_id"] == 0
    assert df.loc[0, "to_node_id"] == 1


def test_zero_based_node_ids(sample_nodes_file: Path):
    df = load_nodes(sample_nodes_file)
    expected_ids = list(range(5))
    assert df["node_id"].tolist() == expected_ids
    assert df["node_id"].iloc[0] == 0


def test_zero_based_arc_ids(sample_arcs_file: Path):
    df = load_arcs(sample_arcs_file)
    expected_ids = list(range(7))
    assert df["arc_id"].tolist() == expected_ids
    assert df["arc_id"].iloc[0] == 0


def test_correct_data_types(sample_nodes_file: Path, sample_arcs_file: Path):
    nodes_df = load_nodes(sample_nodes_file)
    arcs_df = load_arcs(sample_arcs_file)

    assert nodes_df["node_id"].dtype == np.int64
    assert nodes_df["longitude"].dtype == np.float64
    assert nodes_df["latitude"].dtype == np.float64

    assert arcs_df["arc_id"].dtype == np.int64
    assert arcs_df["from_node_id"].dtype == np.int64
    assert arcs_df["to_node_id"].dtype == np.int64


def test_invalid_coordinates(tmp_path: Path):
    bad_lon_file = tmp_path / "bad_lon.nodes"
    bad_lon_file.write_text("185.0 45.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid longitude"):
        load_nodes(bad_lon_file)

    bad_lat_file = tmp_path / "bad_lat.nodes"
    bad_lat_file.write_text("10.0 -95.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid latitude"):
        load_nodes(bad_lat_file)


def test_missing_values(tmp_path: Path):
    missing_val_nodes = tmp_path / "missing.nodes"
    missing_val_nodes.write_text("8.581679 NaN\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or unparseable"):
        load_nodes(missing_val_nodes)

    missing_val_arcs = tmp_path / "missing.arcs"
    missing_val_arcs.write_text("0 None\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or unparseable"):
        load_arcs(missing_val_arcs)


def test_malformed_rows(tmp_path: Path):
    malformed_nodes = tmp_path / "malformed.nodes"
    malformed_nodes.write_text("8.581679\n8.584637 49.858295\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed nodes file"):
        load_nodes(malformed_nodes)

    malformed_arcs = tmp_path / "malformed.arcs"
    malformed_arcs.write_text("0 1 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed arcs file"):
        load_arcs(malformed_arcs)


def test_invalid_node_references(sample_nodes_file: Path, tmp_path: Path):
    nodes_df = load_nodes(sample_nodes_file)
    bad_arcs_file = tmp_path / "invalid_ref.arcs"
    bad_arcs_file.write_text("0 1\n1 999\n", encoding="utf-8")
    arcs_df = load_arcs(bad_arcs_file)

    report = validate_road_data(nodes_df, arcs_df, strict=False)
    assert not report["is_valid"]
    assert 999 in report["missing_to_node_references"]

    with pytest.raises(RoadDataValidationError, match="Road network validation failed"):
        validate_road_data(nodes_df, arcs_df, strict=True)


def test_duplicate_endpoint_pairs_preserved(sample_arcs_file: Path):
    arcs_df = load_arcs(sample_arcs_file)
    assert len(arcs_df) == 7
    assert arcs_df.loc[0, "from_node_id"] == 0 and arcs_df.loc[0, "to_node_id"] == 1
    assert arcs_df.loc[4, "from_node_id"] == 0 and arcs_df.loc[4, "to_node_id"] == 1
    assert arcs_df.loc[0, "arc_id"] == 0
    assert arcs_df.loc[4, "arc_id"] == 4


def test_self_loop_detection(sample_nodes_file: Path, tmp_path: Path):
    nodes_df = load_nodes(sample_nodes_file)
    self_loop_arcs = tmp_path / "self_loop.arcs"
    self_loop_arcs.write_text("0 0\n0 1\n2 2\n", encoding="utf-8")
    arcs_df = load_arcs(self_loop_arcs)

    report = validate_road_data(nodes_df, arcs_df, strict=False)
    assert report["self_loops_count"] == 2
    assert report["self_loop_arc_ids"] == [0, 2]


# --- Step 3 Tests: Haversine Distance ---

def test_haversine_identical_coordinates_zero_distance():
    dist = haversine_distance(8.581679, 49.857880, 8.581679, 49.857880)
    assert dist == pytest.approx(0.0, abs=1e-7)


def test_haversine_known_one_degree_equator():
    dist = haversine_distance(0.0, 0.0, 1.0, 0.0)
    expected_dist = EARTH_RADIUS_METERS * (math.pi / 180.0)
    assert dist == pytest.approx(expected_dist, rel=1e-5)
    assert abs(dist - 111195.0) < 1.0


def test_haversine_symmetry():
    p1 = (8.581679, 49.857880)
    p2 = (8.584637, 49.858295)
    d1 = haversine_distance(p1[0], p1[1], p2[0], p2[1])
    d2 = haversine_distance(p2[0], p2[1], p1[0], p1[1])
    assert d1 == pytest.approx(d2, abs=1e-7)


def test_haversine_small_distance():
    dist = haversine_distance(0.0, 0.0, 0.0, 0.00001)
    assert dist == pytest.approx(1.11195, rel=1e-3)
    assert dist > 0.0


def test_haversine_invalid_longitude():
    with pytest.raises(ValueError, match="lon1 out of bounds"):
        haversine_distance(185.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="lon2 out of bounds"):
        haversine_distance(0.0, 0.0, -181.0, 0.0)


def test_haversine_invalid_latitude():
    with pytest.raises(ValueError, match="lat1 out of bounds"):
        haversine_distance(0.0, 95.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="lat2 out of bounds"):
        haversine_distance(0.0, 0.0, 0.0, -90.5)


def test_haversine_nan_coordinates():
    with pytest.raises(ValueError, match="finite"):
        haversine_distance(float("nan"), 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        haversine_distance(0.0, 0.0, 0.0, np.nan)


def test_haversine_infinite_coordinates():
    with pytest.raises(ValueError, match="finite"):
        haversine_distance(float("inf"), 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        haversine_distance(0.0, 0.0, 0.0, -float("inf"))


# --- Step 4 Tests: NetworkX MultiDiGraph Construction & Statistics ---

def test_returned_graph_is_multidigraph(synthetic_network: Tuple[Path, Path]):
    nodes_file, arcs_file = synthetic_network
    road_graph, arc_lookup, stats = build_road_graph(nodes_file, arcs_file)
    assert isinstance(road_graph, nx.MultiDiGraph)
    assert road_graph.is_directed()
    assert road_graph.is_multigraph()


def test_all_nodes_included(synthetic_network: Tuple[Path, Path]):
    nodes_file, arcs_file = synthetic_network
    road_graph, _, stats = build_road_graph(nodes_file, arcs_file)
    assert road_graph.number_of_nodes() == 6
    assert set(road_graph.nodes()) == {0, 1, 2, 3, 4, 5}


def test_all_arcs_included(synthetic_network: Tuple[Path, Path]):
    nodes_file, arcs_file = synthetic_network
    road_graph, _, stats = build_road_graph(nodes_file, arcs_file)
    assert road_graph.number_of_edges() == 7
    assert stats["total_directed_arcs"] == 7


def test_arc_ids_preserved_as_keys(synthetic_network: Tuple[Path, Path]):
    nodes_file, arcs_file = synthetic_network
    road_graph, _, _ = build_road_graph(nodes_file, arcs_file)
    assert 0 in road_graph[0][1]
    assert 4 in road_graph[0][1]
    assert road_graph[0][1][0]["arc_id"] == 0
    assert road_graph[0][1][4]["arc_id"] == 4


def test_edge_lengths_non_negative_and_finite(synthetic_network: Tuple[Path, Path]):
    """Edge lengths must be non-negative and finite; zero-length edges are valid for identical coordinates."""
    nodes_file, arcs_file = synthetic_network
    road_graph, _, _ = build_road_graph(nodes_file, arcs_file)
    for u, v, k, data in road_graph.edges(keys=True, data=True):
        length = data["length_m"]
        assert isinstance(length, float)
        assert math.isfinite(length)
        assert length >= 0.0
        if u == v:
            assert length == pytest.approx(0.0, abs=1e-7)


def test_zero_length_edge_handling(tmp_path: Path):
    """Verify that edges connecting distinct nodes with identical coordinates produce length 0.0m cleanly."""
    zero_nodes = tmp_path / "zero_edge.nodes"
    zero_arcs = tmp_path / "zero_edge.arcs"
    # Distinct nodes 0 and 1, but identical coordinates
    zero_nodes.write_text(
        "8.616805 49.994635\n"
        "8.616805 49.994635\n",
        encoding="utf-8"
    )
    zero_arcs.write_text("0 1\n", encoding="utf-8")

    G, lookup, stats = build_road_graph(zero_nodes, zero_arcs)
    assert G.number_of_nodes() == 2
    assert G.number_of_edges() == 1
    edge_data = G[0][1][0]
    assert edge_data["arc_id"] == 0
    assert edge_data["length_m"] == pytest.approx(0.0, abs=1e-7)
    assert stats["min_arc_length_m"] == pytest.approx(0.0, abs=1e-7)
    assert stats["self_loops"] == 0  # Distinct nodes: NOT a self loop


def test_node_coordinates_stored_correctly(synthetic_network: Tuple[Path, Path]):
    nodes_file, arcs_file = synthetic_network
    road_graph, _, _ = build_road_graph(nodes_file, arcs_file)
    assert road_graph.nodes[0]["longitude"] == pytest.approx(8.580000)
    assert road_graph.nodes[0]["latitude"] == pytest.approx(49.850000)
    assert road_graph.nodes[0]["node_id"] == 0
    assert isinstance(road_graph.nodes[0]["node_id"], int)


def test_parallel_edges_preserved(synthetic_network: Tuple[Path, Path]):
    nodes_file, arcs_file = synthetic_network
    road_graph, _, _ = build_road_graph(nodes_file, arcs_file)
    edges_0_1 = road_graph[0][1]
    assert len(edges_0_1) == 2
    assert 0 in edges_0_1 and 4 in edges_0_1
    assert edges_0_1[0]["arc_id"] != edges_0_1[4]["arc_id"]


def test_reverse_direction_edges_remain_separate(synthetic_network: Tuple[Path, Path]):
    nodes_file, arcs_file = synthetic_network
    road_graph, _, _ = build_road_graph(nodes_file, arcs_file)
    assert road_graph.has_edge(0, 1)
    assert road_graph.has_edge(1, 0)
    assert 3 in road_graph[1][0]
    assert 3 not in road_graph[0][1]


def test_arc_lookup_correct(synthetic_network: Tuple[Path, Path]):
    nodes_file, arcs_file = synthetic_network
    _, arc_lookup, _ = build_road_graph(nodes_file, arcs_file)
    assert len(arc_lookup) == 7
    assert arc_lookup[0] == (0, 1)
    assert arc_lookup[1] == (1, 2)
    assert arc_lookup[2] == (1, 3)
    assert arc_lookup[3] == (1, 0)
    assert arc_lookup[4] == (0, 1)
    assert arc_lookup[5] == (2, 4)
    assert arc_lookup[6] == (4, 4)


def test_isolated_node_counting(synthetic_network: Tuple[Path, Path]):
    nodes_file, arcs_file = synthetic_network
    _, _, stats = build_road_graph(nodes_file, arcs_file)
    assert stats["isolated_nodes"] == 1


def test_weakly_connected_component_counting(synthetic_network: Tuple[Path, Path]):
    nodes_file, arcs_file = synthetic_network
    _, _, stats = build_road_graph(nodes_file, arcs_file)
    assert stats["weakly_connected_components"] == 2


def test_strongly_connected_component_counting(synthetic_network: Tuple[Path, Path]):
    nodes_file, arcs_file = synthetic_network
    _, _, stats = build_road_graph(nodes_file, arcs_file)
    assert stats["strongly_connected_components"] == 5


def test_duplicate_groups_vs_extra_parallel_edges(synthetic_network: Tuple[Path, Path]):
    nodes_file, arcs_file = synthetic_network
    _, _, stats = build_road_graph(nodes_file, arcs_file)
    assert stats["duplicate_endpoint_groups"] == 1
    assert stats["extra_parallel_arcs"] == 1


def test_graph_statistics_match_synthetic_graph(synthetic_network: Tuple[Path, Path]):
    nodes_file, arcs_file = synthetic_network
    road_graph, _, stats = build_road_graph(nodes_file, arcs_file)
    assert stats["total_nodes"] == 6
    assert stats["total_directed_arcs"] == 7
    assert stats["isolated_nodes"] == 1
    assert stats["weakly_connected_components"] == 2
    assert stats["strongly_connected_components"] == 5
    assert stats["duplicate_endpoint_groups"] == 1
    assert stats["extra_parallel_arcs"] == 1
    assert stats["self_loops"] == 1

    d_0_1 = haversine_distance(8.580, 49.850, 8.581, 49.850)
    d_1_2 = haversine_distance(8.581, 49.850, 8.582, 49.850)
    d_1_3 = haversine_distance(8.581, 49.850, 8.581, 49.851)
    d_1_0 = haversine_distance(8.581, 49.850, 8.580, 49.850)
    d_parallel = haversine_distance(8.580, 49.850, 8.581, 49.850)
    d_2_4 = haversine_distance(8.582, 49.850, 8.583, 49.850)
    d_4_4 = 0.0

    expected_total = d_0_1 + d_1_2 + d_1_3 + d_1_0 + d_parallel + d_2_4 + d_4_4
    assert stats["total_directed_length_m"] == pytest.approx(expected_total, rel=1e-5)
    assert stats["min_arc_length_m"] == pytest.approx(0.0, abs=1e-7)
    assert stats["max_arc_length_m"] == pytest.approx(max(d_0_1, d_1_2, d_1_3, d_1_0, d_parallel, d_2_4), rel=1e-5)
    assert stats["avg_arc_length_m"] == pytest.approx(expected_total / 7, rel=1e-5)


def test_standalone_synthetic_dataset_disk_files():
    """Verify loading and graph construction directly against data/synthetic/ files on disk."""
    base_dir = Path(__file__).resolve().parent.parent / "data" / "synthetic"
    nodes_file = base_dir / "synthetic_sample.nodes"
    arcs_file = base_dir / "synthetic_sample.arcs"

    assert nodes_file.is_file(), f"Missing synthetic nodes at {nodes_file}"
    assert arcs_file.is_file(), f"Missing synthetic arcs at {arcs_file}"

    G, lookup, stats = build_road_graph(nodes_file, arcs_file)
    assert G.number_of_nodes() == 6
    assert G.number_of_edges() == 7
    assert stats["isolated_nodes"] == 1
    assert stats["weakly_connected_components"] == 2
    assert len(lookup) == 7
