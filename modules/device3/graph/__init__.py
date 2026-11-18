"""Graph package for PathSnap Device 3: Road Topology and Directed Graph Construction."""

try:
    from .graph_builder import (
        EARTH_RADIUS_METERS,
        RoadDataValidationError,
        build_road_graph,
        get_graph_statistics,
        haversine_distance,
        load_nodes,
        load_arcs,
        validate_road_data,
    )
except (ImportError, ValueError):
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

__all__ = [
    "EARTH_RADIUS_METERS",
    "RoadDataValidationError",
    "build_road_graph",
    "get_graph_statistics",
    "haversine_distance",
    "load_nodes",
    "load_arcs",
    "validate_road_data",
]
