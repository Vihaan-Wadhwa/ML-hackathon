"""Decoder package for PathSnap Device 3: Candidate I/O and Emission Integration."""

try:
    from .candidate_io import (
        RECONSTRUCTED_OUTPUT_COLUMNS,
        load_candidate_predictions,
        load_track,
        normalize_emission_probabilities,
        reconstruct_trip,
        validate_candidates_against_graph,
    )
    from .reconstruction import (
        convert_timestamps_to_datetime,
        reconstruct_trip as reconstruct_trip_pipeline,
    )
    from .route_projection import route_to_gps_coordinates
    from .transition_model import (
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
    from .viterbi import (
        find_nearest_candidate_arc,
        log_normalize_transition_matrix,
        reconstruct_route,
    )
except (ImportError, ValueError):
    from decoder.candidate_io import (
        RECONSTRUCTED_OUTPUT_COLUMNS,
        load_candidate_predictions,
        load_track,
        normalize_emission_probabilities,
        reconstruct_trip,
        validate_candidates_against_graph,
    )
    from decoder.reconstruction import (
        convert_timestamps_to_datetime,
        reconstruct_trip as reconstruct_trip_pipeline,
    )
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

__all__ = [
    "ArcProjection",
    "RECONSTRUCTED_OUTPUT_COLUMNS",
    "build_transition_matrix",
    "calculate_effective_beta",
    "convert_timestamps_to_datetime",
    "find_nearest_candidate_arc",
    "gps_displacement_m",
    "load_candidate_predictions",
    "load_track",
    "log_normalize_transition_matrix",
    "log_transition_probability",
    "network_transition_distance",
    "normalize_emission_probabilities",
    "normalize_transition_matrix",
    "project_point_onto_arc",
    "reconstruct_route",
    "reconstruct_trip",
    "reconstruct_trip_pipeline",
    "route_to_gps_coordinates",
    "transition_probability",
    "validate_candidates_against_graph",
]
