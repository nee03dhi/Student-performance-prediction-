"""Feature definitions and preprocessing shared by training and the API."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# name: (min, max) seen in the training data
FEATURE_LIMITS = {
    "study_hours": (1, 11),
    "attendance": (40, 100),
    "sleep_hours": (4, 9),
    "internet_usage": (1, 11),
    "assignments_completed": (0, 20),
    "previous_score": (35, 95),
}
FEATURES = list(FEATURE_LIMITS)


def build_preprocessor(scale: bool = False) -> ColumnTransformer:
    """All six features are numeric, so the only optional step is scaling."""
    scaler = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([("num", scaler, FEATURES)])


def to_frame(values: dict) -> pd.DataFrame:
    """One-row DataFrame with columns in the exact order the model expects."""
    return pd.DataFrame([values])[FEATURES]