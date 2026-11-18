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
        RECONSTRUCTED_OUTPUT_COLUMNS,
        load_candidate_predictions,
        load_track,
        normalize_emission_probabilities,
        reconstruct_trip,
        validate_candidates_against_graph,
    )
    from modules.device3.decoder.reconstruction import convert_timestamps_to_datetime
    from modules.device3.decoder.route_projection import route_to_gps_coordinates
    from modules.device3.decoder.transition_model import (
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
    from modules.device3.decoder.viterbi import (
        find_nearest_candidate_arc,
        log_normalize_transition_matrix,
        reconstruct_route,
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
        RECONSTRUCTED_OUTPUT_COLUMNS,
        load_candidate_predictions,
        load_track,
        normalize_emission_probabilities,
        reconstruct_trip,
        validate_candidates_against_graph,
    )
    from decoder.reconstruction import convert_timestamps_to_datetime
    from decoder.route_projection import route_to_gps_coordinates
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
    from decoder.viterbi import (
        find_nearest_candidate_arc,
        log_normalize_transition_matrix,
        reconstruct_route,
    )
    from visualization.road_network_plot import visualize_road_graph

__all__ = [
    "ArcProjection",
    "EARTH_RADIUS_METERS",
    "RECONSTRUCTED_OUTPUT_COLUMNS",
    "RoadDataValidationError",
    "build_road_graph",
    "build_transition_matrix",
    "calculate_effective_beta",
    "convert_timestamps_to_datetime",
    "find_nearest_candidate_arc",
    "get_graph_statistics",
    "gps_displacement_m",
    "haversine_distance",
    "load_arcs",
    "load_candidate_predictions",
    "load_nodes",
    "load_track",
    "log_normalize_transition_matrix",
    "log_transition_probability",
    "network_transition_distance",
    "normalize_emission_probabilities",
    "normalize_transition_matrix",
    "project_point_onto_arc",
    "reconstruct_route",
    "reconstruct_trip",
    "route_to_gps_coordinates",
    "transition_probability",
    "validate_candidates_against_graph",
    "validate_road_data",
    "visualize_road_graph",
]
