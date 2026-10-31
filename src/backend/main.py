"""FastAPI entry point.

Run from the project root:
    uvicorn src.backend.main:app --reload
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.backend.api.predict import router
from src.backend.config.settings import APP_TITLE, APP_VERSION
from src.backend.services.prediction_service import PredictionService
from src.common import settings

settings.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.FileHandler(settings.LOG_PATH), logging.StreamHandler()],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = PredictionService()      # load the model once at startup
    yield


app = FastAPI(title=APP_TITLE, version=APP_VERSION, lifespan=lifespan)
app.include_router(router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok"}