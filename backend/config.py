
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Backend settings
HOST = "0.0.0.0"
PORT = 8000

# Evaluation settings
EXPECTED_TRIP_COUNT = 10
