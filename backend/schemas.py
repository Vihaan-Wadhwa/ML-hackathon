
from pydantic import BaseModel
from typing import List, Optional


class TripListResponse(BaseModel):
    trips: List[dict]
    count: int
    message: str


class EvaluationResponse(BaseModel):
    trip_id: str
    status: str
    predicted_arcs: List[int] = []
    accuracy: Optional[float] = None
    continuity: Optional[float] = None
    latency_ms: Optional[float] = None
    message: str
