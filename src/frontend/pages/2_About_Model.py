"""About page: metrics from the API plus limitations."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # project root

import pandas as pd
import requests
import streamlit as st

from src.frontend.config.settings import API_BASE_URL, PAGE_TITLE

st.set_page_config(page_title=f"About | {PAGE_TITLE}", page_icon="🎓", layout="wide")
st.title("About the model")

try:
    info = requests.get(f"{API_BASE_URL}/model-info", timeout=10).json()
    metrics = info["metrics"]
    table = pd.DataFrame({
        "Linear Regression (baseline)": metrics["baseline_linear_regression"],
        "Gradient Boosting (tuned)": metrics["gradient_boosting_tuned"],
    }).T
    st.subheader("Test-set results")
    st.dataframe(table.rename(columns=str.upper))
    st.caption(f"Typical prediction error: about {info['typical_error']} marks.")
except requests.exceptions.RequestException:
    st.warning("Could not load metrics from the API. Is the backend running?")
except KeyError:
    st.warning("Metrics file not found. Run the training script first.")

st.subheader("Limitations")
st.markdown(
    """
- About 35% of scores in the data are capped at 100, so top students are hard to separate.
- The model under-predicts capped scores and over-predicts very low scores.
- `placement_status` is excluded from the features because it is derived from the exam score.
- The dataset looks synthetic, so results may not hold for real students.
- Predictions near the cutoff are uncertain; treat them as a screening signal.
"""
)