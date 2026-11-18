"""Unit tests for the complete end-to-end route reconstruction pipeline.

Task 2: Step 4 - reconstruct_trip integration validation across probability schemas,
missing observation fallback, Parquet inputs, diagnostics, and error cases.
"""

import math
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
import pytest

try:
    from modules.device3.decoder.candidate_io import (
        RECONSTRUCTED_OUTPUT_COLUMNS,
        load_candidate_predictions,
        reconstruct_trip as candidate_io_reconstruct_trip,
    )
    from modules.device3.decoder.reconstruction import (
        convert_timestamps_to_datetime,
        reconstruct_trip,
    )
except (ImportError, ModuleNotFoundError):
    from decoder.candidate_io import (
        RECONSTRUCTED_OUTPUT_COLUMNS,
        load_candidate_predictions,
        reconstruct_trip as candidate_io_reconstruct_trip,
    )
    from decoder.reconstruction import (
        convert_timestamps_to_datetime,
        reconstruct_trip,
    )


@pytest.fixture
def linear_two_edge_graph() -> nx.MultiDiGraph:
    """Fixture with two connected road segments: 0 -> 1 -> 2."""
    g = nx.MultiDiGraph()
    # 0 at (0.0, 0.0), 1 at (0.001, 0.0), 2 at (0.002, 0.0)
    g.add_node(0, longitude=0.0, latitude=0.0)
    g.add_node(1, longitude=0.001, latitude=0.0)
    g.add_node(2, longitude=0.002, latitude=0.0)

    # Arc 1: 0 -> 1 (length ~111m)
    g.add_edge(0, 1, key=1, arc_id=1, length_m=111.3)
    # Arc 2: 1 -> 2 (length ~111m)
    g.add_edge(1, 2, key=2, arc_id=2, length_m=111.3)

    return g


@pytest.fixture
def linear_track_df() -> pd.DataFrame:
    """Fixture with two sequential GPS points matching the linear graph."""
    return pd.DataFrame({
        "trip_id": ["trip_demo", "trip_demo"],
        "obs_idx": [0, 1],
        "longitude": [0.0005, 0.0015],
        "latitude": [0.00002, 0.00001],
        "timestamp": [10.0, 20.0],
    })


class TestEndToEndReconstruction:
    """Tests for reconstruct_trip pipeline."""

    def test_reconstruction_with_prediction_probability(
        self, linear_two_edge_graph: nx.MultiDiGraph, linear_track_df: pd.DataFrame
    ):
        """Pipeline successfully processes Device 2/4 'prediction_probability' schema."""
        candidates_df = pd.DataFrame({
            "trip_id": ["trip_demo", "trip_demo"],
            "obs_idx": [0, 1],
            "arc_id": [1, 2],
            "prediction_probability": [0.95, 0.92],
        })

        res = reconstruct_trip(candidates_df, linear_two_edge_graph, linear_track_df, beta=5.0)

        assert isinstance(res, pd.DataFrame)
        assert len(res) == 2
        assert list(res.columns) == RECONSTRUCTED_OUTPUT_COLUMNS
        assert list(res["arc_id"]) == [1, 2]
        assert list(res["obs_idx"]) == [0, 1]
        assert list(res["timestamp"]) == [10.0, 20.0]
        assert (res["trip_id"] == "trip_demo").all()

        # Check coordinate snapping
        assert pytest.approx(res.iloc[0]["latitude"], abs=1e-5) == 0.0
        assert pytest.approx(res.iloc[0]["longitude"], abs=1e-5) == 0.0005
        assert pytest.approx(res.iloc[1]["latitude"], abs=1e-5) == 0.0
        assert pytest.approx(res.iloc[1]["longitude"], abs=1e-5) == 0.0015

    def test_reconstruction_with_match_probability(
        self, linear_two_edge_graph: nx.MultiDiGraph, linear_track_df: pd.DataFrame
    ):
        """Pipeline successfully processes 'match_probability' schema."""
        candidates_df = pd.DataFrame({
            "trip_id": ["trip_demo", "trip_demo"],
            "obs_idx": [0, 1],
            "arc_id": [1, 2],
            "match_probability": [0.88, 0.85],
        })

        res = reconstruct_trip(candidates_df, linear_two_edge_graph, linear_track_df, beta=5.0)

        assert len(res) == 2
        assert list(res["arc_id"]) == [1, 2]

    def test_parquet_input_integration(
        self, tmp_path: Path, linear_two_edge_graph: nx.MultiDiGraph, linear_track_df: pd.DataFrame
    ):
        """Pipeline functions smoothly on candidates loaded from Parquet."""
        parquet_path = tmp_path / "test_candidates.parquet"
        raw_df = pd.DataFrame({
            "trip_id": ["trip_demo", "trip_demo", "trip_demo"],
            "obs_idx": [0, 0, 1],
            "arc_id": [1, 2, 2],
            "prediction_probability": [0.85, 0.15, 0.90],
        })
        raw_df.to_parquet(parquet_path)

        loaded_candidates = load_candidate_predictions(parquet_path)
        res = reconstruct_trip(loaded_candidates, linear_two_edge_graph, linear_track_df)

        assert len(res) == 2
        assert list(res["arc_id"]) == [1, 2]

    def test_missing_observation_candidate_fallback(
        self, linear_two_edge_graph: nx.MultiDiGraph
    ):
        """When an observation has 0 candidate rows, it automatically falls back to nearest road."""
        track_df = pd.DataFrame({
            "trip_id": ["trip_fb", "trip_fb", "trip_fb"],
            "obs_idx": [0, 1, 2],
            "longitude": [0.0005, 0.0015, 0.0018],
            "latitude": [0.0, 0.0, 0.0],
            "timestamp": [10.0, 15.0, 20.0],
        })
        # Observation 1 is missing entirely from candidates_df!
        candidates_df = pd.DataFrame({
            "trip_id": ["trip_fb", "trip_fb"],
            "obs_idx": [0, 2],
            "arc_id": [1, 2],
            "prediction_probability": [0.95, 0.95],
        })

        res, diag = reconstruct_trip(
            candidates_df, linear_two_edge_graph, track_df, return_diagnostics=True
        )

        assert len(res) == 3
        assert list(res["obs_idx"]) == [0, 1, 2]
        assert diag["fallback_observations_count"] == 1
        assert diag["fallback_obs_indices"] == [1]
        # Snapped coordinate for obs 1 should be on the road network
        assert pytest.approx(res.iloc[1]["latitude"], abs=1e-5) == 0.0
        assert pytest.approx(res.iloc[1]["longitude"], abs=1e-5) == 0.0015

    def test_disconnected_transition_resilience(self):
        """Pipeline handles disconnected graph components without errors or NaNs."""
        g = nx.MultiDiGraph()
        # Disconnected edges: 0 -> 1 and 10 -> 11
        g.add_node(0, longitude=0.0, latitude=0.0)
        g.add_node(1, longitude=0.001, latitude=0.0)
        g.add_node(10, longitude=0.05, latitude=0.05)
        g.add_node(11, longitude=0.051, latitude=0.05)
        g.add_edge(0, 1, key=0, arc_id=1, length=111.0)
        g.add_edge(10, 11, key=0, arc_id=2, length=111.0)

        track_df = pd.DataFrame({
            "trip_id": ["trip_disc", "trip_disc"],
            "obs_idx": [0, 1],
            "longitude": [0.0005, 0.0505],
            "latitude": [0.0, 0.05],
            "timestamp": [1.0, 2.0],
        })
        candidates_df = pd.DataFrame({
            "trip_id": ["trip_disc", "trip_disc"],
            "obs_idx": [0, 1],
            "arc_id": [1, 2],
            "prediction_probability": [0.9, 0.9],
        })

        res = reconstruct_trip(candidates_df, g, track_df)

        assert len(res) == 2
        assert not res.isna().any().any()
        assert list(res["arc_id"]) == [1, 2]

    def test_mixed_trip_id_rejection(
        self, linear_two_edge_graph: nx.MultiDiGraph, linear_track_df: pd.DataFrame
    ):
        """Mixed trip IDs in candidates_df raises ValueError."""
        candidates_df = pd.DataFrame({
            "trip_id": ["trip_A", "trip_B"],
            "obs_idx": [0, 1],
            "arc_id": [1, 2],
            "prediction_probability": [0.9, 0.9],
        })
        with pytest.raises(ValueError, match="Mixed trip IDs"):
            reconstruct_trip(candidates_df, linear_two_edge_graph, linear_track_df)

    def test_trip_id_mismatch_with_track_df(
        self, linear_two_edge_graph: nx.MultiDiGraph, linear_track_df: pd.DataFrame
    ):
        """Trip ID mismatch between candidates and track raises ValueError."""
        candidates_df = pd.DataFrame({
            "trip_id": ["trip_other", "trip_other"],
            "obs_idx": [0, 1],
            "arc_id": [1, 2],
            "prediction_probability": [0.9, 0.9],
        })
        with pytest.raises(ValueError, match="trip_id mismatch"):
            reconstruct_trip(candidates_df, linear_two_edge_graph, linear_track_df)

    def test_empty_inputs_raise_errors(
        self, linear_two_edge_graph: nx.MultiDiGraph, linear_track_df: pd.DataFrame
    ):
        """Empty candidates or track or graph raise ValueError."""
        empty_df = pd.DataFrame()
        cand = pd.DataFrame({
            "trip_id": ["trip_demo"],
            "obs_idx": [0],
            "arc_id": [1],
            "prediction_probability": [0.9],
        })

        with pytest.raises(ValueError, match="candidates_df cannot be empty"):
            reconstruct_trip(empty_df, linear_two_edge_graph, linear_track_df)

        with pytest.raises(ValueError, match="track_df cannot be empty"):
            reconstruct_trip(cand, linear_two_edge_graph, empty_df)

        with pytest.raises(ValueError, match="road_graph must be a non-empty"):
            reconstruct_trip(cand, nx.MultiDiGraph(), linear_track_df)

    def test_candidate_io_delegation(
        self, linear_two_edge_graph: nx.MultiDiGraph, linear_track_df: pd.DataFrame
    ):
        """reconstruct_trip called from candidate_io delegates correctly and matches result."""
        candidates_df = pd.DataFrame({
            "trip_id": ["trip_demo", "trip_demo"],
            "obs_idx": [0, 1],
            "arc_id": [1, 2],
            "prediction_probability": [0.95, 0.92],
        })

        res_direct = reconstruct_trip(candidates_df, linear_two_edge_graph, linear_track_df)
        res_delegated = candidate_io_reconstruct_trip(candidates_df, linear_two_edge_graph, linear_track_df)

        pd.testing.assert_frame_equal(res_direct, res_delegated)

    def test_diagnostics_telemetry(
        self, linear_two_edge_graph: nx.MultiDiGraph, linear_track_df: pd.DataFrame
    ):
        """Diagnostics dictionary returns complete telemetry."""
        candidates_df = pd.DataFrame({
            "trip_id": ["trip_demo", "trip_demo"],
            "obs_idx": [0, 1],
            "arc_id": [1, 2],
            "prediction_probability": [0.95, 0.92],
        })

        res, diag = reconstruct_trip(
            candidates_df, linear_two_edge_graph, linear_track_df, return_diagnostics=True
        )

        assert isinstance(diag, dict)
        assert diag["trip_id"] == "trip_demo"
        assert diag["total_gps_observations"] == 2
        assert diag["reconstructed_observations_count"] == 2
        assert diag["fallback_observations_count"] == 0
        assert diag["decoded_arcs_count"] == 2
        assert diag["viterbi_execution_time_ms"] > 0
        assert diag["projection_execution_time_ms"] > 0
        assert diag["total_reconstruction_time_ms"] > 0
        assert math.isfinite(diag["final_sequence_log_score"])

    def test_convert_timestamps_to_datetime(self):
        """convert_timestamps_to_datetime adds a properly localized datetime column."""
        reconstructed = pd.DataFrame({
            "trip_id": ["trip_dt", "trip_dt"],
            "obs_idx": [0, 1],
            "timestamp": [0.0, 60.0],
            "latitude": [12.9, 12.91],
            "longitude": [77.5, 77.51],
            "arc_id": [10, 20],
        })

        dt_df = convert_timestamps_to_datetime(
            reconstructed, start_datetime="2026-11-18 10:00:00", timezone="UTC"
        )

        assert "datetime" in dt_df.columns
        assert dt_df["datetime"].iloc[0] == pd.Timestamp("2026-11-18 10:00:00", tz="UTC")
        assert dt_df["datetime"].iloc[1] == pd.Timestamp("2026-11-18 10:01:00", tz="UTC")
