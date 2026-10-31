"""Reads .env once and exposes project-wide settings."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]          # project root
load_dotenv(ROOT / ".env")


def _path(name: str, default: str) -> Path:
    return ROOT / os.getenv(name, default)


DATASET_PATH = _path("DATASET_PATH", "data/raw/student_dataset.csv")
MODEL_PATH = _path("MODEL_PATH", "models/student_model.joblib")
METRICS_PATH = _path("METRICS_PATH", "reports/model_metrics.json")
HYPERPARAMS_PATH = _path("HYPERPARAMS_PATH", "src/training/config/hyperparameters.yaml")
LOG_PATH = _path("LOG_PATH", "logs/app.log")

TARGET = os.getenv("TARGET_COLUMN", "exam_score")
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))
PASS_MARK = float(os.getenv("PASS_MARK", 70))
MAX_SCORE = float(os.getenv("MAX_SCORE", 100))