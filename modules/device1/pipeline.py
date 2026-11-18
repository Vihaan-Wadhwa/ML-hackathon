"""Backend-facing helpers around spatial.features.extract_features_for_trip."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from spatial.features import extract_features_for_trip

# The 12 model inputs (same names/order as backend/predictor.py FEATURE_COLUMNS).
FEATURE_COLUMNS = [
    "cand_rank", "perp_dist_m", "bearing_diff_deg", "speed_mps", "dt_s", "gap_flag",
    "dist_to_start_m", "dist_to_end_m", "frac_along", "seg_len_m", "n_candidates", "has_heading",
]


def locate_trip(trip_id: str, data_dir) -> Path:
    """Folder holding <trip_id>.track anywhere under data_dir (any partition layout)."""
    name = f"{trip_id}.track"
    hits = sorted(Path(d) / name for d, _, files in os.walk(data_dir, followlinks=True) if name in files)
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly one {trip_id}.track under {data_dir}, found {len(hits)}")
    return hits[0].parent


def extract_trip_features(trip_id: str, data_dir, with_labels: bool = False) -> pd.DataFrame:
    """Candidate features for one trip, frozen schema v1.0.

    with_labels=False (default, and mandatory in the API) never opens the .route file,
    so is_match / ref_arc_id are null and evaluator labels stay out of prediction.
    """
    trip_dir = locate_trip(str(trip_id), data_dir)
    return extract_features_for_trip(str(trip_id), trip_dir.parent, with_labels=with_labels)


def model_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows to score: drops the arc_id = -1 placeholder rows (observations with no road
    within R). Keep the original frame to know which observations had no candidate."""
    return df[df["arc_id"] >= 0].reset_index(drop=True)
