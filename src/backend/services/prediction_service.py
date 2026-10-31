"""Loads the saved model once and turns a StudentInput into a prediction."""
import json
import logging

import joblib
import numpy as np

from src.backend.config.settings import DEFAULT_TYPICAL_ERROR
from src.backend.schemas.student import PredictionResponse, StudentInput
from src.common import settings
from src.common.preprocessing_util import to_frame

log = logging.getLogger(__name__)


class PredictionService:
    def __init__(self):
        if not settings.MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {settings.MODEL_PATH}. "
                "Run: python -m src.training.student_performance")
        self.model = joblib.load(settings.MODEL_PATH)
        self.metrics = self._load_metrics()
        self.typical_error = (self.metrics.get("gradient_boosting_tuned", {})
                              .get("mae", DEFAULT_TYPICAL_ERROR))
        log.info("Model loaded, typical error %.2f", self.typical_error)

    @staticmethod
    def _load_metrics() -> dict:
        if settings.METRICS_PATH.exists():
            return json.loads(settings.METRICS_PATH.read_text())
        return {}

    def predict(self, student: StudentInput) -> PredictionResponse:
        row = to_frame(student.model_dump())
        score = float(np.clip(self.model.predict(row)[0], 0, settings.MAX_SCORE))
        low = max(score - self.typical_error, 0.0)
        high = min(score + self.typical_error, settings.MAX_SCORE)
        return PredictionResponse(
            predicted_score=round(score, 1),
            range_low=round(low, 1),
            range_high=round(high, 1),
            outcome="Likely PLACED" if score >= settings.PASS_MARK else "AT RISK",
            borderline=low < settings.PASS_MARK < high,
        )