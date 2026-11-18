"""Unit tests for candidate_io: loading, validation, and emission normalization.

Task 2: Step 1 of PathSnap Device 3.
"""

from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from decoder.candidate_io import (
    RECONSTRUCTED_OUTPUT_COLUMNS,
    load_candidate_predictions,
    load_track,
    normalize_emission_probabilities,
    reconstruct_trip,
    validate_candidates_against_graph,
)
from graph.graph_builder import build_road_graph


@pytest.fixture
def sample_parquet_file(tmp_path: Path) -> Path:
    """Fixture providing a valid 2-observation Parquet file with differing candidate counts."""
    data = {
        "trip_id": ["trip_test"] * 5,
        "obs_idx": [0, 0, 1, 1, 1],  # Obs 0 has 2 candidates, Obs 1 has 3 candidates
        "arc_id": [10, 20, 10, 30, 40],
        "match_probability": [0.85, 0.15, 0.60, 0.30, 0.10],
    }
    df = pd.DataFrame(data)
    parquet_path = tmp_path / "valid_candidates.parquet"
    df.to_parquet(parquet_path, engine="pyarrow")
    return parquet_path


@pytest.fixture
def sample_track_file(tmp_path: Path) -> Path:
    """Fixture providing a 3-point GPS track file."""
    content = (
        "8.581679 49.857880 1.0\n"
        "8.582679 49.858880 2.0\n"
        "8.583679 49.859880 3.0\n"
    )
    track_path = tmp_path / "sample.track"
    track_path.write_text(content, encoding="utf-8")
    return track_path


# 1. Parquet loading
def test_load_candidate_predictions_success(sample_parquet_file: Path):
    df = load_candidate_predictions(sample_parquet_file)
    assert len(df) == 5
    assert list(df.columns) == ["trip_id", "obs_idx", "arc_id", "match_probability"]
    assert df["trip_id"].iloc[0] == "trip_test"


# 2. Correct schema and data types
def test_candidate_predictions_dtypes(sample_parquet_file: Path):
    df = load_candidate_predictions(sample_parquet_file)
    assert df["trip_id"].dtype == object or pd.api.types.is_string_dtype(df["trip_id"])
    assert df["obs_idx"].dtype == np.int64
    assert df["arc_id"].dtype == np.int64
    assert df["match_probability"].dtype == np.float64


# 3. Invalid probabilities
def test_invalid_probabilities_rejection(tmp_path: Path):
    # Probability > 1.0
    bad_prob_df = pd.DataFrame({
        "trip_id": ["t1"],
        "obs_idx": [0],
        "arc_id": [1],
        "match_probability": [1.5],
    })
    p_path = tmp_path / "bad_prob.parquet"
    bad_prob_df.to_parquet(p_path, engine="pyarrow")
    with pytest.raises(ValueError, match="out of bounds"):
        load_candidate_predictions(p_path)

    # Probability < 0.0
    neg_prob_df = pd.DataFrame({
        "trip_id": ["t1"],
        "obs_idx": [0],
        "arc_id": [1],
        "match_probability": [-0.1],
    })
    p_path2 = tmp_path / "neg_prob.parquet"
    neg_prob_df.to_parquet(p_path2, engine="pyarrow")
    with pytest.raises(ValueError, match="out of bounds"):
        load_candidate_predictions(p_path2)

    # NaN probability
    nan_prob_df = pd.DataFrame({
        "trip_id": ["t1"],
        "obs_idx": [0],
        "arc_id": [1],
        "match_probability": [np.nan],
    })
    p_path3 = tmp_path / "nan_prob.parquet"
    nan_prob_df.to_parquet(p_path3, engine="pyarrow")
    with pytest.raises(ValueError, match="null values"):
        load_candidate_predictions(p_path3)


# 4. Missing fields
def test_missing_fields_rejection(tmp_path: Path):
    bad_schema_df = pd.DataFrame({
        "trip_id": ["t1"],
        "obs_idx": [0],
        "arc_id": [1],
        # 'match_probability' is missing
    })
    p_path = tmp_path / "missing_col.parquet"
    bad_schema_df.to_parquet(p_path, engine="pyarrow")
    with pytest.raises(ValueError, match="Missing required column"):
        load_candidate_predictions(p_path)


# 5. Duplicate candidates
def test_duplicate_candidates_rejection(tmp_path: Path):
    dup_df = pd.DataFrame({
        "trip_id": ["t1", "t1"],
        "obs_idx": [0, 0],
        "arc_id": [10, 10],  # Duplicate candidate for same observation
        "match_probability": [0.6, 0.4],
    })
    p_path = tmp_path / "dup.parquet"
    dup_df.to_parquet(p_path, engine="pyarrow")
    with pytest.raises(ValueError, match="duplicate.*candidate row"):
        load_candidate_predictions(p_path)


# 6. Multiple candidates per observation with differing candidate counts
def test_multiple_candidates_differing_counts(sample_parquet_file: Path):
    df = load_candidate_predictions(sample_parquet_file)
    counts = df.groupby("obs_idx").size().to_dict()
    assert counts[0] == 2
    assert counts[1] == 3


# 7. Normalization summing to 1.0
def test_normalization_sums_to_one(sample_parquet_file: Path):
    df = load_candidate_predictions(sample_parquet_file)
    norm_df = normalize_emission_probabilities(df, temperature=1.0)

    assert "emission_probability" in norm_df.columns
    assert "match_probability" in norm_df.columns

    # Verify every observation sums to 1.0
    for obs, group in norm_df.groupby("obs_idx"):
        assert group["emission_probability"].sum() == pytest.approx(1.0, abs=1e-6)


# 8. Temperature validation
def test_temperature_validation(sample_parquet_file: Path):
    df = load_candidate_predictions(sample_parquet_file)

    # Negative or zero temperature must raise ValueError
    with pytest.raises(ValueError, match="Temperature must be a positive"):
        normalize_emission_probabilities(df, temperature=0.0)
    with pytest.raises(ValueError, match="Temperature must be a positive"):
        normalize_emission_probabilities(df, temperature=-1.5)

    # Temperature = 0.5 (sharpening top candidate)
    sharp_df = normalize_emission_probabilities(df, temperature=0.5)
    # Temperature = 2.0 (softening distribution towards uniform)
    soft_df = normalize_emission_probabilities(df, temperature=2.0)

    norm_t1 = normalize_emission_probabilities(df, temperature=1.0)

    p_t1 = norm_t1.loc[norm_t1["obs_idx"] == 0, "emission_probability"].iloc[0]
    p_sharp = sharp_df.loc[sharp_df["obs_idx"] == 0, "emission_probability"].iloc[0]
    p_soft = soft_df.loc[soft_df["obs_idx"] == 0, "emission_probability"].iloc[0]

    assert p_sharp > p_t1
    assert p_soft < p_t1


# 9. Arc ID validation against road graph
def test_validate_candidates_against_graph(sample_parquet_file: Path):
    candidates_df = load_candidate_predictions(sample_parquet_file)

    G = nx.MultiDiGraph()
    G.add_edge(0, 1, key=10, arc_id=10, length_m=50.0)
    G.add_edge(1, 2, key=20, arc_id=20, length_m=60.0)
    arc_lookup = {10: (0, 1), 20: (1, 2)}

    report = validate_candidates_against_graph(candidates_df, G, arc_lookup=arc_lookup, strict=False)
    assert not report["is_valid"]
    assert 30 in report["invalid_arc_ids"]
    assert 40 in report["invalid_arc_ids"]

    with pytest.raises(ValueError, match="Candidate validation against road graph failed"):
        validate_candidates_against_graph(candidates_df, G, arc_lookup=arc_lookup, strict=True)


# 10. GPS observation index validation against track file
def test_validate_obs_idx_against_track(sample_parquet_file: Path, sample_track_file: Path):
    candidates_df = load_candidate_predictions(sample_parquet_file)
    track_df = load_track(sample_track_file)
    assert len(track_df) == 3

    G = nx.MultiDiGraph()
    for arc in [10, 20, 30, 40]:
        G.add_edge(0, 1, key=arc, arc_id=arc, length_m=50.0)
    arc_lookup = {arc: (0, 1) for arc in [10, 20, 30, 40]}

    report = validate_candidates_against_graph(
        candidates_df, G, arc_lookup=arc_lookup, track_df=track_df, strict=True
    )
    assert report["is_valid"]
    assert len(report["out_of_bounds_obs_indices"]) == 0

    bad_candidates = candidates_df.copy()
    bad_candidates.loc[0, "obs_idx"] = 99
    with pytest.raises(ValueError, match="outside track range"):
        validate_candidates_against_graph(
            bad_candidates, G, arc_lookup=arc_lookup, track_df=track_df, strict=True
        )


# 11. Test actual workspace sample files: candidate_predictions_sample.parquet
def test_actual_workspace_sample_parquet():
    possible_paths = [
        Path("candidate_predictions_sample.parquet"),
        Path(__file__).resolve().parent.parent / "candidate_predictions_sample.parquet",
        Path(__file__).resolve().parent.parent.parent.parent / "candidate_predictions_sample.parquet",
    ]
    sample_file = next((p for p in possible_paths if p.is_file()), None)
    assert sample_file is not None and sample_file.is_file(), "candidate_predictions_sample.parquet must exist."

    df = load_candidate_predictions(sample_file)
    assert len(df) == 20
    assert df["trip_id"].unique().tolist() == ["trip_000"]
    assert list(df["obs_idx"].unique()) == [0, 1, 2, 3]

    norm_df = normalize_emission_probabilities(df, temperature=1.0)
    for _, group in norm_df.groupby(["trip_id", "obs_idx"]):
        assert group["emission_probability"].sum() == pytest.approx(1.0, abs=1e-6)


# 12. Support prediction_probability column from Device 2
def test_load_candidate_predictions_with_prediction_probability(tmp_path: Path):
    data = {
        "trip_id": ["trip_dev2"] * 3,
        "obs_idx": [0, 0, 1],
        "arc_id": [101, 102, 103],
        "prediction_probability": [0.75, 0.25, 0.90],
    }
    parquet_path = tmp_path / "dev2_candidates.parquet"
    pd.DataFrame(data).to_parquet(parquet_path, engine="pyarrow")

    loaded_df = load_candidate_predictions(parquet_path)
    assert len(loaded_df) == 3
    # Standard internal format
    assert "match_probability" in loaded_df.columns
    # Device 2 alias column preserved
    assert "prediction_probability" in loaded_df.columns
    assert loaded_df["match_probability"].tolist() == [0.75, 0.25, 0.90]
    assert loaded_df["prediction_probability"].tolist() == [0.75, 0.25, 0.90]
    assert loaded_df["obs_idx"].dtype == np.int64
    assert loaded_df["arc_id"].dtype == np.int64


# 13. Support additional candidate features and metadata
def test_load_candidate_predictions_preserves_additional_metadata(tmp_path: Path):
    data = {
        "trip_id": ["trip_features"] * 2,
        "obs_idx": [0, 0],
        "arc_id": [5, 6],
        "prediction_probability": [0.6, 0.4],
        "dist_to_arc": [12.5, 34.2],
        "heading_delta": [5.1, 45.0],
    }
    parquet_path = tmp_path / "features_candidates.parquet"
    pd.DataFrame(data).to_parquet(parquet_path, engine="pyarrow")

    loaded_df = load_candidate_predictions(parquet_path)
    assert "dist_to_arc" in loaded_df.columns
    assert "heading_delta" in loaded_df.columns
    assert loaded_df["dist_to_arc"].tolist() == [12.5, 34.2]
    assert loaded_df["heading_delta"].tolist() == [5.1, 45.0]


# 14. Normalization works on prediction_probability directly
def test_normalize_emission_with_prediction_probability(tmp_path: Path):
    data = {
        "trip_id": ["trip_dev2"] * 3,
        "obs_idx": [0, 0, 0],
        "arc_id": [1, 2, 3],
        "prediction_probability": [0.2, 0.3, 0.5],
    }
    df = pd.DataFrame(data)
    norm_df = normalize_emission_probabilities(df, temperature=1.0)
    assert "emission_probability" in norm_df.columns
    assert norm_df["emission_probability"].sum() == pytest.approx(1.0, abs=1e-6)
    assert norm_df["emission_probability"].tolist() == [0.2, 0.3, 0.5]


# 15. Mixed consistent probability columns
def test_mixed_consistent_probability_columns(tmp_path: Path):
    data = {
        "trip_id": ["trip_mix"] * 2,
        "obs_idx": [0, 0],
        "arc_id": [1, 2],
        "match_probability": [0.7, 0.3],
        "prediction_probability": [0.7, 0.3],
    }
    p_path = tmp_path / "consistent_mix.parquet"
    pd.DataFrame(data).to_parquet(p_path, engine="pyarrow")

    df = load_candidate_predictions(p_path)
    assert len(df) == 2
    assert df["match_probability"].tolist() == [0.7, 0.3]
    assert df["prediction_probability"].tolist() == [0.7, 0.3]


# 16. Conflicting probability columns without explicit selection
def test_conflicting_probability_columns_rejection(tmp_path: Path):
    data = {
        "trip_id": ["trip_conflict"] * 2,
        "obs_idx": [0, 0],
        "arc_id": [1, 2],
        "match_probability": [0.8, 0.2],
        "prediction_probability": [0.4, 0.6],  # Conflicting values
    }
    p_path = tmp_path / "conflict.parquet"
    pd.DataFrame(data).to_parquet(p_path, engine="pyarrow")

    with pytest.raises(ValueError, match="Conflicting probability columns"):
        load_candidate_predictions(p_path)


# 17. Conflicting probability columns with explicit selection
def test_conflicting_probability_columns_explicit_selection(tmp_path: Path):
    data = {
        "trip_id": ["trip_conflict"] * 2,
        "obs_idx": [0, 0],
        "arc_id": [1, 2],
        "match_probability": [0.8, 0.2],
        "prediction_probability": [0.4, 0.6],
    }
    p_path = tmp_path / "conflict_explicit.parquet"
    pd.DataFrame(data).to_parquet(p_path, engine="pyarrow")

    # Select match_probability explicitly
    df_match = load_candidate_predictions(p_path, probability_col="match_probability")
    assert df_match["match_probability"].tolist() == [0.8, 0.2]

    # Select prediction_probability explicitly
    df_pred = load_candidate_predictions(p_path, probability_col="prediction_probability")
    assert df_pred["match_probability"].tolist() == [0.4, 0.6]
    assert df_pred["prediction_probability"].tolist() == [0.4, 0.6]


# 18. Rejection of binary prediction column as probability
def test_reject_binary_prediction_column_selection(tmp_path: Path):
    data = {
        "trip_id": ["trip_bin"] * 2,
        "obs_idx": [0, 0],
        "arc_id": [1, 2],
        "match_probability": [0.9, 0.1],
        "prediction": [1, 0],  # Binary classification label
    }
    p_path = tmp_path / "binary_col.parquet"
    pd.DataFrame(data).to_parquet(p_path, engine="pyarrow")

    with pytest.raises(ValueError, match="Cannot use binary prediction column"):
        load_candidate_predictions(p_path, probability_col="prediction")


# 19. Task 4 Interface stub: reconstruct_trip and schema verification
def test_reconstruct_trip_interface_stub():
    assert RECONSTRUCTED_OUTPUT_COLUMNS == [
        "trip_id",
        "obs_idx",
        "timestamp",
        "latitude",
        "longitude",
        "arc_id",
    ]

    dummy_df = pd.DataFrame({
        "trip_id": ["t1"],
        "obs_idx": [0],
        "arc_id": [1],
        "match_probability": [1.0],
    })
    dummy_track = pd.DataFrame({
        "trip_id": ["t1"],
        "obs_idx": [0],
        "longitude": [8.5],
        "latitude": [49.8],
        "timestamp": [1.0],
    })
    G = nx.MultiDiGraph()
    G.add_node(0, longitude=8.5, latitude=49.8)
    G.add_node(1, longitude=8.51, latitude=49.8)
    G.add_edge(0, 1, key=1, arc_id=1, length_m=10.0)

    res = reconstruct_trip(dummy_df, G, dummy_track, beta=5.0)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 1
    assert list(res.columns) == RECONSTRUCTED_OUTPUT_COLUMNS
    assert res.iloc[0]["arc_id"] == 1

