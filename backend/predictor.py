
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "final_model.joblib"

FEATURE_COLUMNS = [
    "cand_rank",
    "perp_dist_m",
    "bearing_diff_deg",
    "speed_mps",
    "dt_s",
    "gap_flag",
    "dist_to_start_m",
    "dist_to_end_m",
    "frac_along",
    "seg_len_m",
    "n_candidates",
    "has_heading",
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


def predict_candidates(input_path: str, output_path: str):
    df = pd.read_parquet(input_path)

    missing_columns = [
        col for col in FEATURE_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing model features: {missing_columns}"
        )

    X = (
        df[FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    model = load_model()

    df["prediction"] = model.predict(X)
    df["prediction_probability"] = model.predict_proba(X)[:, 1]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output, index=False)

    return {
        "status": "success",
        "input_rows": len(df),
        "predicted_rows": len(df),
        "output_file": str(output),
        "positive_predictions": int(df["prediction"].sum()),
    }
