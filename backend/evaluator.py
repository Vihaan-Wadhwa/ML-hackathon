
from backend.data_loader import get_available_trips


def evaluate_trip(trip_id: str):
    """
    Evaluation pipeline placeholder.

    Actual evaluation will be connected after receiving:
    Device 1: Feature extraction
    Device 2: Trained ML model
    Device 3: Route reconstruction
    """

    trips = get_available_trips()

    available_trip_ids = [
        trip["trip_id"] for trip in trips
    ]

    if trip_id not in available_trip_ids:
        return {
            "trip_id": trip_id,
            "status": "not_found",
            "message": "Trip data is not available"
        }

    return {
        "trip_id": trip_id,
        "status": "pending",
        "message": "Waiting for Device 1, Device 2 and Device 3 integration"
    }
