"""Device 1 (spatial feature extraction) entry point for the backend.

    from modules.device1 import extract_trip_features, model_rows, FEATURE_COLUMNS

The implementation lives in modules/device1/src/spatial (package `spatial`).
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from .pipeline import FEATURE_COLUMNS, extract_trip_features, locate_trip, model_rows  # noqa: E402,F401
