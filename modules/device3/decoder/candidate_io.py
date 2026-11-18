"""Candidate predictions I/O and emission probability processing for PathSnap Device 3.

Task 2: Step 1 - Device 2 ML Output Integration.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import networkx as nx
import numpy as np
import pandas as pd


REQUIRED_BASE_COLUMNS = ["trip_id", "obs_idx", "arc_id"]
PROBABILITY_COLUMN_CANDIDATES = ["match_probability", "prediction_probability"]
REQUIRED_CANDIDATE_COLUMNS = ["trip_id", "obs_idx", "arc_id", "match_probability"]
RECONSTRUCTED_OUTPUT_COLUMNS = [
    "trip_id",
    "obs_idx",
    "timestamp",
    "latitude",
    "longitude",
    "arc_id",
]


def load_candidate_predictions(
    parquet_path: Union[str, Path],
    probability_col: Optional[str] = None,
) -> pd.DataFrame:
    """Load and validate candidate predictions Parquet file from Device 2 or Device 4.

    Supports both 'match_probability' and 'prediction_probability' as candidate scores,
    normalizing internally to 'match_probability' while preserving original score columns
    and any additional candidate features or metadata.

    Parameters
    ----------
    parquet_path : Union[str, Path]
        Path to the candidate predictions .parquet file.
    probability_col : Optional[str], default=None
        Explicit probability column to select. If None, auto-detects
        'match_probability' or 'prediction_probability'. If both exist, requires
        them to be consistent or requires explicit selection.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame with columns:
        - trip_id: str
        - obs_idx: int64
        - arc_id: int64
        - match_probability: float64
        - prediction_probability: float64 (if present in input)
        - any additional candidate features/metadata columns preserved

    Raises
    ------
    FileNotFoundError
        If parquet_path does not exist.
    ValueError
        If required columns are missing, values are null/infinite/out-of-bounds,
        conflicting probability columns exist without explicit selection,
        a binary prediction column is used as probability, or duplicate candidate
        entries are present.
    """
    path = Path(parquet_path)
    if not path.is_file():
        raise FileNotFoundError(f"Candidate predictions file not found: {path}")

    try:
        df = pd.read_parquet(path, engine="pyarrow")
    except Exception as e:
        raise ValueError(f"Failed to read Parquet file at {path}: {e}") from e

    if df.empty:
        raise ValueError(f"Candidate predictions Parquet file is empty: {path}")

    # Check required base columns
    missing_base = [c for c in REQUIRED_BASE_COLUMNS if c not in df.columns]
    if missing_base:
        raise ValueError(
            f"Missing required base column(s) in {path}: {missing_base}. "
            f"Expected base columns: {REQUIRED_BASE_COLUMNS}"
        )

    # Resolve score column
    if probability_col is not None:
        if probability_col not in df.columns:
            raise ValueError(
                f"Specified probability column '{probability_col}' not found in {path}. "
                f"Available columns: {list(df.columns)}"
            )
        # Ensure binary label/prediction column is not used as continuous emission probability
        if probability_col in ["prediction", "predicted_label", "binary_prediction", "label"]:
            raise ValueError(
                f"Cannot use binary prediction column '{probability_col}' as emission probability. "
                f"A continuous probability score in [0.0, 1.0] representing P(match=1) is required."
            )
        chosen_prob_col = probability_col
    else:
        present_prob_cols = [c for c in PROBABILITY_COLUMN_CANDIDATES if c in df.columns]
        if len(present_prob_cols) == 0:
            raise ValueError(
                f"Missing required column(s) in {path}: no supported probability column found. "
                f"Expected at least one of {PROBABILITY_COLUMN_CANDIDATES}. Found: {list(df.columns)}"
            )
        elif len(present_prob_cols) == 1:
            chosen_prob_col = present_prob_cols[0]
        else:
            # Both 'match_probability' and 'prediction_probability' exist in the file
            vals_match = pd.to_numeric(df["match_probability"], errors="coerce")
            vals_pred = pd.to_numeric(df["prediction_probability"], errors="coerce")
            # If both columns are numerically consistent within tolerance, accept either
            if np.allclose(vals_match.dropna(), vals_pred.dropna(), atol=1e-5, equal_nan=False):
                chosen_prob_col = "match_probability"
            else:
                raise ValueError(
                    f"Conflicting probability columns in {path}: both 'match_probability' and "
                    f"'prediction_probability' exist with differing values. Please specify "
                    f"probability_col explicitly to select which score to use."
                )

    # Check for null / missing values in required columns
    checked_cols = REQUIRED_BASE_COLUMNS + [chosen_prob_col]
    null_counts = df[checked_cols].isna().sum()
    if null_counts.any():
        bad_cols = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Candidate predictions contains null values in columns: {bad_cols}")

    # Preserve trip_id as string
    trip_id_series = df["trip_id"].astype(str)

    # Validate obs_idx and arc_id
    obs = pd.to_numeric(df["obs_idx"], errors="coerce")
    arc = pd.to_numeric(df["arc_id"], errors="coerce")

    if obs.isna().any() or arc.isna().any():
        raise ValueError("obs_idx and arc_id must contain valid numeric entries.")

    if ((obs % 1 != 0) | (arc % 1 != 0)).any():
        raise ValueError("obs_idx and arc_id must be integer values.")

    if (obs < 0).any():
        first_bad = obs[obs < 0].iloc[0]
        raise ValueError(f"Found negative obs_idx: {first_bad}. Observation index must be non-negative.")

    if (arc < 0).any():
        first_bad = arc[arc < 0].iloc[0]
        raise ValueError(f"Found negative arc_id: {first_bad}. Arc ID must be non-negative.")

    # Validate score / probability
    prob = pd.to_numeric(df[chosen_prob_col], errors="coerce")
    if prob.isna().any():
        raise ValueError(f"{chosen_prob_col} must contain valid numeric entries.")

    if not np.all(np.isfinite(prob.values)):
        raise ValueError(f"{chosen_prob_col} values must be finite (not NaN or Inf).")

    if ((prob < 0.0) | (prob > 1.0)).any():
        bad_prob = prob[(prob < 0.0) | (prob > 1.0)].iloc[0]
        raise ValueError(
            f"{chosen_prob_col} value {bad_prob} is out of bounds. Must be between 0.0 and 1.0."
        )

    # Check duplicate (trip_id, obs_idx, arc_id)
    dup_mask = df.duplicated(subset=["trip_id", "obs_idx", "arc_id"])
    if dup_mask.any():
        num_dups = int(dup_mask.sum())
        sample_dups = df.loc[dup_mask, ["trip_id", "obs_idx", "arc_id"]].head(3).to_dict("records")
        raise ValueError(
            f"Found {num_dups} duplicate (trip_id, obs_idx, arc_id) candidate row(s). Samples: {sample_dups}"
        )

    clean_dict: Dict[str, Any] = {
        "trip_id": trip_id_series,
        "obs_idx": obs.astype(np.int64),
        "arc_id": arc.astype(np.int64),
        "match_probability": prob.astype(np.float64),
    }

    # If prediction_probability was in input, preserve it
    if "prediction_probability" in df.columns:
        if chosen_prob_col == "prediction_probability":
            clean_dict["prediction_probability"] = prob.astype(np.float64)
        else:
            clean_dict["prediction_probability"] = pd.to_numeric(
                df["prediction_probability"], errors="coerce"
            ).astype(np.float64)

    # Preserve any additional candidate features and metadata
    for col in df.columns:
        if col not in clean_dict:
            clean_dict[col] = df[col]

    clean_df = pd.DataFrame(clean_dict)
    return clean_df


def load_track(track_path: Union[str, Path]) -> pd.DataFrame:
    """Load and validate the vehicle trajectory GPS .track file.

    Parameters
    ----------
    track_path : Union[str, Path]
        Path to the .track file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - obs_idx: int64 (0-based row index)
        - longitude: float64
        - latitude: float64
        - timestamp: float64

    Raises
    ------
    FileNotFoundError
        If track_path does not exist.
    ValueError
        If the file has invalid column counts, missing values, or out-of-bounds coordinates.
    """
    path = Path(track_path)
    if not path.is_file():
        raise FileNotFoundError(f"Track file not found: {path}")

    try:
        df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    except pd.errors.ParserError as e:
        raise ValueError(f"Malformed track file at {path}: {e}") from e

    if df.empty:
        raise ValueError(f"Track file is empty: {path}")

    if df.shape[1] != 3:
        raise ValueError(
            f"Malformed track file at {path}: expected exactly 3 columns (lon, lat, timestamp), "
            f"found {df.shape[1]} columns."
        )

    if df.isna().any().any():
        raise ValueError(f"Track file at {path} contains missing or unparseable values.")

    lon = pd.to_numeric(df[0], errors="coerce")
    lat = pd.to_numeric(df[1], errors="coerce")
    ts = pd.to_numeric(df[2], errors="coerce")

    if lon.isna().any() or lat.isna().any() or ts.isna().any():
        raise ValueError(f"Track file at {path} contains non-numeric entries.")

    if not np.all(np.isfinite(lon.values)) or not np.all(np.isfinite(lat.values)) or not np.all(np.isfinite(ts.values)):
        raise ValueError(f"Track file at {path} contains non-finite values.")

    if ((lon < -180.0) | (lon > 180.0)).any():
        raise ValueError(f"Track file at {path} contains invalid longitude values outside [-180, 180].")

    if ((lat < -90.0) | (lat > 90.0)).any():
        raise ValueError(f"Track file at {path} contains invalid latitude values outside [-90, 90].")

    if (ts < 0.0).any():
        raise ValueError(f"Track file at {path} contains negative timestamp values.")

    obs_indices = np.arange(len(df), dtype=np.int64)
    track_df = pd.DataFrame({
        "obs_idx": obs_indices,
        "longitude": lon.astype(np.float64),
        "latitude": lat.astype(np.float64),
        "timestamp": ts.astype(np.float64),
    })

    return track_df


def validate_candidates_against_graph(
    candidates_df: pd.DataFrame,
    road_graph: nx.MultiDiGraph,
    arc_lookup: Optional[Dict[int, Tuple[int, int]]] = None,
    track_df: Optional[pd.DataFrame] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """Validate candidate predictions against a trip's specific road graph and GPS trajectory.

    Parameters
    ----------
    candidates_df : pd.DataFrame
        DataFrame of candidate predictions with columns ['trip_id', 'obs_idx', 'arc_id', 'match_probability'].
    road_graph : nx.MultiDiGraph
        Road network MultiDiGraph for this specific trip.
    arc_lookup : Optional[Dict[int, Tuple[int, int]]], default=None
        Lookup dictionary mapping arc_id -> (u, v) for the road graph.
    track_df : Optional[pd.DataFrame], default=None
        Loaded track DataFrame for bounds checking obs_idx.
    strict : bool, default=False
        If True, raises ValueError upon detecting any invalid arc references or out-of-bounds indices.

    Returns
    -------
    Dict[str, Any]
        Validation report dictionary with integrity checks and candidate distribution stats.

    Raises
    ------
    ValueError
        If strict=True and invalid candidate arcs or out-of-bounds obs_idx are identified.
    """
    errors: List[str] = []

    # Valid arc IDs in the graph
    if arc_lookup is not None:
        valid_arc_ids: Set[int] = set(arc_lookup.keys())
    else:
        valid_arc_ids = {
            data["arc_id"]
            for _, _, _, data in road_graph.edges(keys=True, data=True)
            if "arc_id" in data
        }

    candidate_arc_ids = set(candidates_df["arc_id"].unique())
    invalid_arcs = sorted(list(candidate_arc_ids - valid_arc_ids))
    if invalid_arcs:
        errors.append(
            f"Found {len(invalid_arcs)} candidate arc_id(s) that do not exist in the road graph: {invalid_arcs[:5]}..."
        )

    # Observation index bounds check
    out_of_bounds_obs: List[int] = []
    if track_df is not None:
        max_obs = len(track_df) - 1
        bad_obs_mask = (candidates_df["obs_idx"] < 0) | (candidates_df["obs_idx"] > max_obs)
        if bad_obs_mask.any():
            out_of_bounds_obs = sorted(list(candidates_df.loc[bad_obs_mask, "obs_idx"].unique()))
            errors.append(
                f"Found obs_idx values outside track range [0, {max_obs}]: {out_of_bounds_obs[:5]}..."
            )

    # Group candidate counts
    group_sizes = candidates_df.groupby(["trip_id", "obs_idx"]).size()
    min_candidates = int(group_sizes.min()) if len(group_sizes) > 0 else 0
    max_candidates = int(group_sizes.max()) if len(group_sizes) > 0 else 0
    avg_candidates = float(group_sizes.mean()) if len(group_sizes) > 0 else 0.0

    is_valid = len(errors) == 0

    report: Dict[str, Any] = {
        "is_valid": is_valid,
        "errors": errors,
        "trip_ids": candidates_df["trip_id"].unique().tolist(),
        "total_candidate_rows": len(candidates_df),
        "unique_observations": len(group_sizes),
        "min_candidates_per_obs": min_candidates,
        "max_candidates_per_obs": max_candidates,
        "avg_candidates_per_obs": avg_candidates,
        "invalid_arc_ids": invalid_arcs,
        "invalid_arc_count": len(invalid_arcs),
        "out_of_bounds_obs_indices": out_of_bounds_obs,
    }

    if strict and not is_valid:
        raise ValueError(
            f"Candidate validation against road graph failed with {len(errors)} error(s):\n"
            + "\n".join(f"- {e}" for e in errors)
        )

    return report


def normalize_emission_probabilities(
    candidates_df: pd.DataFrame,
    temperature: float = 1.0,
    epsilon: float = 1e-12,
    probability_col: Optional[str] = None,
) -> pd.DataFrame:
    """Normalize candidate match probabilities into a categorical emission probability distribution.

    For every (trip_id, obs_idx) observation group:
    1. Reads score values p_i from 'match_probability' or 'prediction_probability'.
    2. Applies a probability floor: p_i' = max(p_i, epsilon).
    3. If temperature == 1.0:
       normalized_p_i = p_i' / sum(p_j')
       If temperature != 1.0:
       normalized_p_i = softmax(log(p_i') / temperature)

    Parameters
    ----------
    candidates_df : pd.DataFrame
        DataFrame with columns ['trip_id', 'obs_idx', 'arc_id'] and either
        'match_probability' or 'prediction_probability'.
    temperature : float, default=1.0
        Temperature parameter for probability scaling. Must be strictly positive.
    epsilon : float, default=1e-12
        Probability floor to prevent log(0) or division by zero.
    probability_col : Optional[str], default=None
        Explicit column name to normalize. If None, prefers 'match_probability'
        falling back to 'prediction_probability'.

    Returns
    -------
    pd.DataFrame
        Copy of candidates_df with preserved score columns and an added
        'emission_probability' column, normalized to sum to 1.0 per observation.

    Raises
    ------
    ValueError
        If temperature is non-positive or non-finite, or if no valid probability column is found.
    """
    if temperature <= 0.0 or not math.isfinite(temperature):
        raise ValueError(f"Temperature must be a positive finite float, got: {temperature}")

    if epsilon <= 0.0 or not math.isfinite(epsilon):
        raise ValueError(f"Epsilon must be a positive finite float, got: {epsilon}")

    if probability_col is not None:
        target_col = probability_col
        if target_col not in candidates_df.columns:
            raise ValueError(f"Specified probability column '{target_col}' not found in candidates_df.")
    elif "match_probability" in candidates_df.columns:
        target_col = "match_probability"
    elif "prediction_probability" in candidates_df.columns:
        target_col = "prediction_probability"
    else:
        raise ValueError(
            "candidates_df must contain either 'match_probability' or 'prediction_probability'."
        )

    out_df = candidates_df.copy()

    def _normalize_group(series: pd.Series) -> np.ndarray:
        p_raw = series.values
        # Apply probability floor
        p_floor = np.maximum(p_raw, epsilon)

        if temperature == 1.0:
            s = np.sum(p_floor)
            if s > 0:
                return p_floor / s
            return np.full_like(p_floor, 1.0 / len(p_floor))
        else:
            # Softmax(log(p) / T) with numerical max subtraction
            z = np.log(p_floor) / temperature
            z_max = np.max(z)
            exp_z = np.exp(z - z_max)
            s = np.sum(exp_z)
            if s > 0:
                return exp_z / s
            return np.full_like(exp_z, 1.0 / len(exp_z))

    out_df["emission_probability"] = (
        out_df.groupby(["trip_id", "obs_idx"])[target_col]
        .transform(_normalize_group)
        .astype(np.float64)
    )

    return out_df


def reconstruct_trip(
    candidates_df: pd.DataFrame,
    road_graph: nx.MultiDiGraph,
    track_df: pd.DataFrame,
    beta: float = 5.0,
    *,
    return_diagnostics: bool = False,
    arc_lookup: Optional[Dict[int, Tuple[int, int]]] = None,
    shortest_path_cache: Optional[Dict[Tuple[int, int], float]] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """Reconstruct vehicle trajectory by map-matching GPS track candidates onto the road network.

    Delegates to `decoder.reconstruction.reconstruct_trip`.

    Parameters
    ----------
    candidates_df : pd.DataFrame
        Candidate road predictions containing ['trip_id', 'obs_idx', 'arc_id']
        and 'match_probability' or 'prediction_probability'.
    road_graph : nx.MultiDiGraph
        Directed road network MultiDiGraph for the specific trip.
    track_df : pd.DataFrame
        GPS trajectory DataFrame with columns ['obs_idx', 'longitude', 'latitude', 'timestamp'].
    beta : float, default=5.0
        Transition probability scale parameter for network distance difference penalty.
    return_diagnostics : bool, default=False
        Whether to return diagnostic telemetry alongside the reconstructed trajectory.
    arc_lookup : Optional[Dict[int, Tuple[int, int]]], default=None
        Lookup dictionary mapping arc_id -> (u, v) for road_graph edges.
    shortest_path_cache : Optional[Dict[Tuple[int, int], float]], default=None
        Cache for shortest path network distances.

    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]
        Reconstructed trajectory with columns:
        - trip_id: str
        - obs_idx: int64
        - timestamp: float64 (preserves original track timestamp in seconds)
        - latitude: float64 (reconstructed coordinate on matched road arc)
        - longitude: float64 (reconstructed coordinate on matched road arc)
        - arc_id: int64 (Viterbi-matched road segment ID)
    """
    try:
        from modules.device3.decoder.reconstruction import reconstruct_trip as _rec
    except (ImportError, ModuleNotFoundError):
        try:
            from decoder.reconstruction import reconstruct_trip as _rec
        except (ImportError, ModuleNotFoundError):
            from .reconstruction import reconstruct_trip as _rec

    return _rec(
        candidates_df=candidates_df,
        road_graph=road_graph,
        track_df=track_df,
        beta=beta,
        return_diagnostics=return_diagnostics,
        arc_lookup=arc_lookup,
        shortest_path_cache=shortest_path_cache,
    )

