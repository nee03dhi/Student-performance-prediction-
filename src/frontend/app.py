"""Streamlit home page.

Run from the project root (backend must be running):
    streamlit run src/frontend/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # project root

import streamlit as st

from src.frontend.config.settings import PAGE_TITLE, PASS_MARK

st.set_page_config(page_title=PAGE_TITLE, page_icon="🎓", layout="wide")

st.title("🎓 Student Performance Prediction System")
st.write(
    "Estimate a student's final exam score from six study-related inputs. "
    f"Students predicted below {PASS_MARK:.0f} are flagged as at risk."
)
st.info("Use the pages in the sidebar: **Predict** to score a student, **About Model** for results and limitations.")