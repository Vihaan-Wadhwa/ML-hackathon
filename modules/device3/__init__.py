"""PathSnap Device 3: Road Topology, Graph Construction & Candidate I/O."""

try:
    from modules.device3.graph.graph_builder import (
        EARTH_RADIUS_METERS,
        RoadDataValidationError,
        build_road_graph,
        get_graph_statistics,
        haversine_distance,
        load_arcs,
        load_nodes,
        validate_road_data,
    )
    from modules.device3.decoder.candidate_io import (
        load_candidate_predictions,
        load_track,
        normalize_emission_probabilities,
        validate_candidates_against_graph,
    )
    from modules.device3.visualization.road_network_plot import visualize_road_graph
except ModuleNotFoundError:
    from graph.graph_builder import (
        EARTH_RADIUS_METERS,
        RoadDataValidationError,
        build_road_graph,
        get_graph_statistics,
        haversine_distance,
        load_arcs,
        load_nodes,
        validate_road_data,
    )
    from decoder.candidate_io import (
        load_candidate_predictions,
        load_track,
        normalize_emission_probabilities,
        validate_candidates_against_graph,
    )
    from visualization.road_network_plot import visualize_road_graph

__all__ = [
    "EARTH_RADIUS_METERS",
    "RoadDataValidationError",
    "build_road_graph",
    "get_graph_statistics",
    "haversine_distance",
    "load_arcs",
    "load_candidate_predictions",
    "load_nodes",
    "load_track",
    "normalize_emission_probabilities",
    "validate_candidates_against_graph",
    "validate_road_data",
    "visualize_road_graph",
]
