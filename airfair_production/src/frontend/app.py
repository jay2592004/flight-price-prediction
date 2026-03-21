"""
src/frontend/app.py  —  AirFair Vista Streamlit Frontend
All 4 pages in one file using st.navigation / st.Page (Streamlit ≥ 1.36)
with graceful fallback to st.sidebar radio for older versions.

Loads artefacts from models/ at startup — zero coupling to training code.
"""

import streamlit as st
import os, sys

# Make sure src is importable when running from project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.frontend.pages import home, eda, model_comparison, about

st.set_page_config(
    page_title="AirFair Vista ✈️",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared CSS injected once ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]  { font-family:'DM Sans',sans-serif; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0A1931 0%,#0D2550 100%); }
[data-testid="stSidebar"] * { color:#C8DEFF !important; }
[data-testid="stSidebar"] label { color:#90B8F0 !important; font-size:.76rem !important;
  text-transform:uppercase; letter-spacing:.07em; }
.stButton>button { background:linear-gradient(135deg,#1565C0,#1E88E5) !important;
  color:#fff !important; border:none !important; border-radius:10px !important;
  padding:.65rem 2rem !important; font-weight:600 !important;
  width:100% !important; box-shadow:0 4px 14px rgba(21,101,192,.3) !important; }
</style>""", unsafe_allow_html=True)

# ── Sidebar nav ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 1.5rem;'>
      <div style='font-size:2.4rem;'>✈️</div>
      <div style='font-family:"Playfair Display",serif;font-size:1.35rem;
           font-weight:900;color:#fff;'>AirFair Vista</div>
      <div style='font-size:.7rem;color:#7098CC;letter-spacing:.1em;
           text-transform:uppercase;margin-top:.2rem;'>Flight Price Intelligence</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Predict Price", "📊 EDA Dashboard", "🤖 Model Comparison", "ℹ️ About"],
        label_visibility="collapsed",
    )

# ── Route to page ─────────────────────────────────────────────────────────────
if   page == "🏠 Predict Price":   home.render()
elif page == "📊 EDA Dashboard":   eda.render()
elif page == "🤖 Model Comparison": model_comparison.render()
elif page == "ℹ️ About":           about.render()
