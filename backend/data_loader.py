
from backend.config import DATA_DIR


def get_available_trips():
    """Find all available .track files in the data directory."""

    if not DATA_DIR.exists():
        return []

    track_files = sorted(DATA_DIR.rglob("*.track"))

    return [
        {
            "trip_id": file.stem,
            "file_path": str(file)
        }
        for file in track_files
    ]
