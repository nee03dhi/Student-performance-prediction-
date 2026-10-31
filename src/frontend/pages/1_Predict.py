"""Prediction page: sliders -> API -> result."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # project root

import requests
import streamlit as st

from src.frontend.config.settings import API_URL, PAGE_TITLE, PASS_MARK, SLIDERS

st.set_page_config(page_title=f"Predict | {PAGE_TITLE}", page_icon="🎓", layout="wide")
st.title("Predict a student's exam score")

st.sidebar.header("Student details")
payload = {
    name: st.sidebar.slider(label, low, high, default)
    for name, (label, low, high, default) in SLIDERS.items()
}

if st.button("Predict", type="primary"):
    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the API. Start it with: uvicorn src.backend.main:app --reload")
        st.stop()
    except requests.exceptions.RequestException as exc:
        st.error(f"API error: {exc}")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted exam score", f"{result['predicted_score']:.1f} / 100")
    c2.metric("Likely range", f"{result['range_low']:.0f} - {result['range_high']:.0f}")
    c3.metric("Outcome", result["outcome"])
    st.progress(int(round(result["predicted_score"])))

    if result["borderline"]:
        st.warning(f"The likely range crosses the placement cutoff ({PASS_MARK:.0f}), so this prediction is borderline.")
    elif result["outcome"] == "Likely PLACED":
        st.success("This profile is predicted to clear the placement cutoff.")
    else:
        st.error("This profile is predicted to fall below the placement cutoff.")
else:
    st.write("Set the values in the sidebar, then click **Predict**.")