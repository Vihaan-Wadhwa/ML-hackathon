"""Unit tests for road_network_plot visualization module.

Step 5 of PathSnap Device 3 Task 1.
"""

from pathlib import Path
import networkx as nx
import pytest

from graph.graph_builder import build_road_graph
from visualization.road_network_plot import visualize_road_graph


@pytest.fixture
def sample_graph(tmp_path: Path) -> nx.MultiDiGraph:
    """Fixture providing a small 4-node MultiDiGraph with geographic coordinates."""
    nodes_file = tmp_path / "plot_sample.nodes"
    arcs_file = tmp_path / "plot_sample.arcs"

    nodes_file.write_text(
        "8.580000 49.850000\n"
        "8.582000 49.850000\n"
        "8.582000 49.852000\n"
        "8.580000 49.852000\n",
        encoding="utf-8",
    )
    arcs_file.write_text(
        "0 1\n"
        "1 2\n"
        "2 3\n"
        "3 0\n"
        "0 1\n",  # Parallel edge
        encoding="utf-8",
    )

    G, _, _ = build_road_graph(nodes_file, arcs_file)
    return G


# 1. Visualization function executes successfully
def test_visualization_executes_successfully(sample_graph: nx.MultiDiGraph, tmp_path: Path):
    out_file = tmp_path / "output.png"
    result_path = visualize_road_graph(sample_graph, out_file, title="Test Graph")
    assert result_path == out_file


# 2. Output PNG file is created
def test_output_file_created(sample_graph: nx.MultiDiGraph, tmp_path: Path):
    out_file = tmp_path / "graph.png"
    visualize_road_graph(sample_graph, out_file)
    assert out_file.is_file()


# 3. Output file is non-empty
def test_output_file_non_empty(sample_graph: nx.MultiDiGraph, tmp_path: Path):
    out_file = tmp_path / "non_empty.png"
    visualize_road_graph(sample_graph, out_file)
    assert out_file.stat().st_size > 1000  # PNG image should be several kilobytes


# 4. Actual node coordinates are used
def test_actual_node_coordinates_used(sample_graph: nx.MultiDiGraph, tmp_path: Path):
    # Verify coordinates match graph node attributes
    for n, data in sample_graph.nodes(data=True):
        assert "longitude" in data and "latitude" in data
        assert isinstance(data["longitude"], float)
        assert isinstance(data["latitude"], float)

    # Calling with bbox filtering around actual node coordinates works
    bbox = (8.579, 49.849, 8.583, 49.853)
    out_file = tmp_path / "bbox_graph.png"
    visualize_road_graph(sample_graph, out_file, bbox=bbox)
    assert out_file.is_file()
    assert out_file.stat().st_size > 1000


# 5. Empty graphs are handled appropriately
def test_empty_graph_raises_error(tmp_path: Path):
    empty_g = nx.MultiDiGraph()
    out_file = tmp_path / "empty.png"
    with pytest.raises(ValueError, match="Cannot visualize an empty road graph"):
        visualize_road_graph(empty_g, out_file)


# 6. Output directories are created when missing
def test_output_directories_created_when_missing(sample_graph: nx.MultiDiGraph, tmp_path: Path):
    deep_path = tmp_path / "nested" / "subfolder" / "deep_plot.png"
    assert not deep_path.parent.exists()
    visualize_road_graph(sample_graph, deep_path)
    assert deep_path.is_file()
    assert deep_path.stat().st_size > 0
