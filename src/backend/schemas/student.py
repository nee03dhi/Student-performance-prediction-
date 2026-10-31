"""Request and response shapes. Pydantic rejects out-of-range input automatically."""
from pydantic import BaseModel, Field


class StudentInput(BaseModel):
    study_hours: int = Field(..., ge=1, le=11, description="Study hours (1-11)")
    attendance: float = Field(..., ge=40, le=100, description="Attendance % (40-100)")
    sleep_hours: int = Field(..., ge=4, le=9, description="Sleep hours (4-9)")
    internet_usage: int = Field(..., ge=1, le=11, description="Internet usage (1-11)")
    assignments_completed: int = Field(..., ge=0, le=20, description="Assignments completed (0-20)")
    previous_score: float = Field(..., ge=35, le=95, description="Previous score (35-95)")


class PredictionResponse(BaseModel):
    predicted_score: float
    range_low: float
    range_high: float
    outcome: str
    borderline: bool