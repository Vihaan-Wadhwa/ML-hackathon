"""Visualization package for PathSnap Device 3."""

try:
    from .road_network_plot import visualize_road_graph
except (ImportError, ValueError):
    from visualization.road_network_plot import visualize_road_graph

__all__ = [
    "visualize_road_graph",
]
