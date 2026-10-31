"""Frontend settings. Self-contained so Streamlit pages need no backend imports."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api/predict")
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api")
PASS_MARK = float(os.getenv("PASS_MARK", 70))
PAGE_TITLE = "Student Performance Predictor"

# label, min, max, default (match the training data ranges)
SLIDERS = {
    "study_hours": ("Study hours", 1, 11, 6),
    "attendance": ("Attendance (%)", 40, 100, 70),
    "sleep_hours": ("Sleep hours", 4, 9, 7),
    "internet_usage": ("Internet usage", 1, 11, 6),
    "assignments_completed": ("Assignments completed", 0, 20, 10),
    "previous_score": ("Previous score", 35, 95, 65),
}