
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from modules.device1 import (
    FEATURE_COLUMNS,
    extract_trip_features,
    model_rows,
)


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "final_model.joblib"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


def _score_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Generate Device 2 predictions for valid Device 1 candidate rows."""

    missing_columns = [
        col for col in FEATURE_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing model features: {missing_columns}"
        )

    if "arc_id" not in df.columns:
        raise ValueError("Missing required column: arc_id")

    # Device 3 cannot process placeholder candidates.
    candidates = df[df["arc_id"] >= 0].copy()

    if candidates.empty:
        raise ValueError("No valid road candidates available for prediction.")

    X = (
        candidates[FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .to_numpy()
    )

    model = load_model()

    candidates["prediction"] = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    # Keep the original name and provide Device 3's expected name.
    candidates["prediction_probability"] = probabilities
    candidates["match_probability"] = probabilities

    return candidates


def predict_candidates(input_path: str, output_path: str):
    """Batch prediction using an existing candidate Parquet file."""

    df = pd.read_parquet(input_path)

    candidates = _score_candidates(df)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    candidates.to_parquet(output, index=False)

    return {
        "status": "success",
        "input_rows": len(df),
        "predicted_rows": len(candidates),
        "dropped_placeholder_rows": len(df) - len(candidates),
        "output_file": str(output),
        "positive_predictions": int(candidates["prediction"].sum()),
    }


def predict_trip(trip_id: str, data_dir: str = "data") -> pd.DataFrame:
    """
    Live prediction pipeline:

    Device 1 feature extraction -> Device 2 model predictions.

    Uses with_labels=False so the API never reads ground-truth routes.
    Returns candidate rows with match_probability for Device 3.
    """

    full_df = extract_trip_features(
        str(trip_id),
        data_dir,
        with_labels=False,
    )

    candidates = model_rows(full_df)

    if candidates.empty:
        raise ValueError(
            f"No valid road candidates found for trip {trip_id}"
        )

    return _score_candidates(candidates)
