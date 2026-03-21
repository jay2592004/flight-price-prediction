"""
app.py  —  AirFair Vista Streamlit entry point
Run: streamlit run app.py
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from utils.style import GLOBAL_CSS

st.set_page_config(
    page_title="AirFair Vista ✈️",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

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
        ["🏠  Predict Price",
         "📊  EDA & Insights",
         "⚙️  Feature Engineering",
         "🤖  Model Comparison",
         "ℹ️  About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:.7rem;color:#506080;line-height:1.8;'>
      <b style='color:#7098CC;'>How to train</b><br>
      <code style='background:rgba(255,255,255,.1);padding:2px 5px;border-radius:4px;'>
      cd ml_pipeline<br>python train.py</code>
    </div>""", unsafe_allow_html=True)

# ── Route ─────────────────────────────────────────────────────────────────────
if   page == "🏠  Predict Price":
    from pages.p1_predict import render; render()
elif page == "📊  EDA & Insights":
    from pages.p2_eda import render; render()
elif page == "⚙️  Feature Engineering":
    from pages.p3_features import render; render()
elif page == "🤖  Model Comparison":
    from pages.p4_models import render; render()
elif page == "ℹ️  About":
    from pages.p5_about import render; render()
