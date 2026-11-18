"""Unit tests for route coordinate projection and snapping in PathSnap Device 3.

Task 2: Step 4 - route_to_gps_coordinates validation across geometric projections,
clamping, parallel arcs, zero-length arcs, and schema preservation.
"""

import math
import networkx as nx
import numpy as np
import pandas as pd
import pytest

try:
    from modules.device3.decoder.route_projection import (
        RECONSTRUCTED_OUTPUT_COLUMNS,
        route_to_gps_coordinates,
    )
except (ImportError, ModuleNotFoundError):
    from decoder.route_projection import (
        RECONSTRUCTED_OUTPUT_COLUMNS,
        route_to_gps_coordinates,
    )


@pytest.fixture
def simple_road_graph() -> nx.MultiDiGraph:
    """Fixture providing a simple multi-segment road graph."""
    g = nx.MultiDiGraph()
    # Node 0 at (0.0, 0.0), Node 1 at (0.01, 0.0), Node 2 at (0.01, 0.01), Node 3 at (0.01, 0.01) [zero-length]
    g.add_node(0, longitude=0.0, latitude=0.0)
    g.add_node(1, longitude=0.01, latitude=0.0)
    g.add_node(2, longitude=0.01, latitude=0.01)
    g.add_node(3, longitude=0.01, latitude=0.01)

    # Arc 101: 0 -> 1 (Horizontal east)
    g.add_edge(0, 1, key=0, arc_id=101, length=1111.0)
    # Arc 102: 1 -> 2 (Vertical north)
    g.add_edge(1, 2, key=0, arc_id=102, length=1111.0)
    # Arc 103: 0 -> 2 (Diagonal north-east)
    g.add_edge(0, 2, key=0, arc_id=103, length=1571.0)
    # Arc 104: 2 -> 3 (Zero length)
    g.add_edge(2, 3, key=0, arc_id=104, length=0.0)
    # Arc 105: 0 -> 1 (Parallel alternative arc)
    g.add_edge(0, 1, key=1, arc_id=105, length=1111.0)

    return g


class TestRouteCoordinateProjection:
    """Tests for route_to_gps_coordinates."""

    def test_horizontal_road_projection(self, simple_road_graph: nx.MultiDiGraph):
        """GPS point offset perpendicularly from horizontal arc snaps correctly."""
        track_df = pd.DataFrame({
            "obs_idx": [0],
            "longitude": [0.005],
            "latitude": [0.002],  # offset north
            "timestamp": [100.0],
        })
        predicted_arcs = [101]

        res = route_to_gps_coordinates(predicted_arcs, simple_road_graph, track_df, trip_id="trip_test")

        assert len(res) == 1
        assert res.iloc[0]["trip_id"] == "trip_test"
        assert res.iloc[0]["obs_idx"] == 0
        assert res.iloc[0]["timestamp"] == 100.0
        assert res.iloc[0]["arc_id"] == 101
        # Projected onto horizontal arc (lat=0.0, lon=0.005)
        assert pytest.approx(res.iloc[0]["latitude"], abs=1e-6) == 0.0
        assert pytest.approx(res.iloc[0]["longitude"], abs=1e-6) == 0.005

    def test_vertical_road_projection(self, simple_road_graph: nx.MultiDiGraph):
        """GPS point offset perpendicularly from vertical arc snaps correctly."""
        track_df = pd.DataFrame({
            "obs_idx": [0],
            "longitude": [0.012],  # offset east
            "latitude": [0.005],
            "timestamp": [105.0],
        })
        predicted_arcs = [102]

        res = route_to_gps_coordinates(predicted_arcs, simple_road_graph, track_df, trip_id="trip_test")

        assert len(res) == 1
        assert pytest.approx(res.iloc[0]["latitude"], abs=1e-6) == 0.005
        assert pytest.approx(res.iloc[0]["longitude"], abs=1e-6) == 0.01

    def test_diagonal_road_projection(self, simple_road_graph: nx.MultiDiGraph):
        """GPS point snaps to diagonal arc perpendicularly."""
        track_df = pd.DataFrame({
            "obs_idx": [0],
            "longitude": [0.0],
            "latitude": [0.01],
            "timestamp": [110.0],
        })
        predicted_arcs = [103]

        res = route_to_gps_coordinates(predicted_arcs, simple_road_graph, track_df, trip_id="trip_test")

        # Midpoint of (0,0) and (0.01, 0.01) is (0.005, 0.005)
        assert pytest.approx(res.iloc[0]["latitude"], abs=1e-5) == 0.005
        assert pytest.approx(res.iloc[0]["longitude"], abs=1e-5) == 0.005

    def test_endpoint_clamping_start(self, simple_road_graph: nx.MultiDiGraph):
        """GPS point behind arc start clamps to q=0.0."""
        track_df = pd.DataFrame({
            "obs_idx": [0],
            "longitude": [-0.005],  # before start (0,0)
            "latitude": [0.0],
            "timestamp": [115.0],
        })
        predicted_arcs = [101]

        res = route_to_gps_coordinates(predicted_arcs, simple_road_graph, track_df, trip_id="trip_test")

        assert pytest.approx(res.iloc[0]["latitude"], abs=1e-6) == 0.0
        assert pytest.approx(res.iloc[0]["longitude"], abs=1e-6) == 0.0

    def test_endpoint_clamping_end(self, simple_road_graph: nx.MultiDiGraph):
        """GPS point beyond arc end clamps to q=1.0."""
        track_df = pd.DataFrame({
            "obs_idx": [0],
            "longitude": [0.015],  # past end (0.01, 0.0)
            "latitude": [0.0],
            "timestamp": [120.0],
        })
        predicted_arcs = [101]

        res = route_to_gps_coordinates(predicted_arcs, simple_road_graph, track_df, trip_id="trip_test")

        assert pytest.approx(res.iloc[0]["latitude"], abs=1e-6) == 0.0
        assert pytest.approx(res.iloc[0]["longitude"], abs=1e-6) == 0.01

    def test_zero_length_arc(self, simple_road_graph: nx.MultiDiGraph):
        """Zero-length arc snaps to identical node coordinates."""
        track_df = pd.DataFrame({
            "obs_idx": [0],
            "longitude": [0.015],
            "latitude": [0.015],
            "timestamp": [125.0],
        })
        predicted_arcs = [104]  # 2 -> 3 (zero length)

        res = route_to_gps_coordinates(predicted_arcs, simple_road_graph, track_df, trip_id="trip_test")

        assert pytest.approx(res.iloc[0]["latitude"], abs=1e-6) == 0.01
        assert pytest.approx(res.iloc[0]["longitude"], abs=1e-6) == 0.01

    def test_parallel_arcs_differentiation(self, simple_road_graph: nx.MultiDiGraph):
        """Parallel arcs between identical endpoints preserve specified arc_id."""
        track_df = pd.DataFrame({
            "obs_idx": [0, 1],
            "longitude": [0.005, 0.005],
            "latitude": [0.0, 0.0],
            "timestamp": [130.0, 131.0],
        })
        predicted_arcs = [101, 105]

        res = route_to_gps_coordinates(predicted_arcs, simple_road_graph, track_df, trip_id="trip_parallel")

        assert list(res["arc_id"]) == [101, 105]

    def test_timestamp_and_obs_idx_preservation(self, simple_road_graph: nx.MultiDiGraph):
        """Observation timestamps in seconds and obs_idx are strictly preserved."""
        track_df = pd.DataFrame({
            "obs_idx": [10, 20],
            "longitude": [0.002, 0.008],
            "latitude": [0.001, -0.001],
            "timestamp": [1700000000.5, 1700000005.75],
        })
        predicted_arcs = [101, 101]

        res = route_to_gps_coordinates(predicted_arcs, simple_road_graph, track_df, trip_id="trip_ts")

        assert list(res["obs_idx"]) == [10, 20]
        assert list(res["timestamp"]) == [1700000000.5, 1700000005.75]
        assert np.issubdtype(res["timestamp"].dtype, np.floating)
        assert np.issubdtype(res["obs_idx"].dtype, np.integer)

    def test_output_columns_and_schema(self, simple_road_graph: nx.MultiDiGraph):
        """Output schema strictly conforms to the 6 expected columns."""
        track_df = pd.DataFrame({
            "obs_idx": [0],
            "longitude": [0.005],
            "latitude": [0.0],
            "timestamp": [1.0],
        })
        res = route_to_gps_coordinates([101], simple_road_graph, track_df, trip_id="trip_1")

        assert list(res.columns) == RECONSTRUCTED_OUTPUT_COLUMNS
        assert list(res.columns) == [
            "trip_id",
            "obs_idx",
            "timestamp",
            "latitude",
            "longitude",
            "arc_id",
        ]

    def test_length_mismatch_raises_error(self, simple_road_graph: nx.MultiDiGraph):
        """Length mismatch between predicted_arcs and track_df raises ValueError."""
        track_df = pd.DataFrame({
            "obs_idx": [0, 1],
            "longitude": [0.0, 0.0],
            "latitude": [0.0, 0.0],
            "timestamp": [1.0, 2.0],
        })
        with pytest.raises(ValueError, match="Length mismatch"):
            route_to_gps_coordinates([101], simple_road_graph, track_df, trip_id="trip_1")

    def test_unknown_arc_id_raises_key_error(self, simple_road_graph: nx.MultiDiGraph):
        """Predicted arc ID not present in road graph raises KeyError."""
        track_df = pd.DataFrame({
            "obs_idx": [0],
            "longitude": [0.0],
            "latitude": [0.0],
            "timestamp": [1.0],
        })
        with pytest.raises(KeyError, match="Predicted arc ID 9999"):
            route_to_gps_coordinates([9999], simple_road_graph, track_df, trip_id="trip_1")

    def test_empty_track_raises_error(self, simple_road_graph: nx.MultiDiGraph):
        """Empty track_df raises ValueError."""
        track_df = pd.DataFrame(columns=["obs_idx", "longitude", "latitude", "timestamp"])
        with pytest.raises(ValueError, match="track_df cannot be empty"):
            route_to_gps_coordinates([], simple_road_graph, track_df, trip_id="trip_1")
