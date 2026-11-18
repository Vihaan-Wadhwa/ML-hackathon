"""Decoder package for PathSnap Device 3."""

from .candidate_io import (
    load_candidate_predictions,
    load_track,
    normalize_emission_probabilities,
    validate_candidates_against_graph,
)

from .transition_model import (
    transition_distance,
    transition_log_probability,
    build_transition_matrix,
    viterbi_decode,
)

__all__ = [
    "load_candidate_predictions",
    "load_track",
    "normalize_emission_probabilities",
    "validate_candidates_against_graph",
    "transition_distance",
    "transition_log_probability",
    "build_transition_matrix",
    "viterbi_decode",
]
