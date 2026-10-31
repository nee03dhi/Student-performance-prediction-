"""API routes."""
from fastapi import APIRouter, Request

from src.backend.schemas.student import PredictionResponse, StudentInput

router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
def predict(student: StudentInput, request: Request):
    return request.app.state.service.predict(student)


@router.get("/model-info")
def model_info(request: Request):
    service = request.app.state.service
    return {"metrics": service.metrics, "typical_error": service.typical_error}