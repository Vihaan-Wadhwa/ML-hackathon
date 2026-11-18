
from fastapi import FastAPI
from datetime import datetime

from backend.data_loader import get_available_trips
from backend.evaluator import evaluate_trip

app = FastAPI(
    title="GPS Route Reconstruction API",
    description="Backend API for distributed GPS route reconstruction",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "GPS Route Reconstruction API is running",
        "status": "online"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/trips")
def get_trips():
    trips = get_available_trips()

    return {
        "trips": [
            {"trip_id": trip["trip_id"]}
            for trip in trips
        ],
        "count": len(trips),
        "message": "Trip data loaded successfully"
        if trips else "No .track files found"
    }


@app.post("/evaluate/{trip_id}")
def evaluate(trip_id: str):
    return evaluate_trip(trip_id)
