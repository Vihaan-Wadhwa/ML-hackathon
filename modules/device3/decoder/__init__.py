"""Decoder package for PathSnap Device 3: Candidate I/O and Emission Integration."""

try:
    from .candidate_io import (
        load_candidate_predictions,
        load_track,
        normalize_emission_probabilities,
        validate_candidates_against_graph,
    )
except (ImportError, ValueError):
    from decoder.candidate_io import (
        load_candidate_predictions,
        load_track,
        normalize_emission_probabilities,
        validate_candidates_against_graph,
    )

__all__ = [
    "load_candidate_predictions",
    "load_track",
    "normalize_emission_probabilities",
    "validate_candidates_against_graph",
]
