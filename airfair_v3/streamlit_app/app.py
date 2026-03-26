"""
app.py  —  AirFair Vista
Run: streamlit run app.py
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from utils.style  import GLOBAL_CSS
from utils.loader import load_model_artefacts

st.set_page_config(
    page_title="AirFair Vista",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# Load model once — sidebar uses live metrics
_, __, ___, meta, loaded = load_model_artefacts()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown("""
    <div style='padding:1.4rem 1rem .9rem;border-bottom:1px solid #1E3A5F;margin-bottom:1rem;'>
      <div style='display:flex;align-items:center;gap:.65rem;'>
        <div style='font-size:1.5rem;line-height:1;'>✈</div>
        <div>
          <div style='font-family:Georgia,serif;font-size:1.15rem;font-weight:700;
               color:#F5F3EE;'>AirFair Vista</div>
          <div style='font-size:.62rem;color:#C5A028;letter-spacing:.16em;
               text-transform:uppercase;margin-top:.1rem;'>Flight Price Intelligence</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Navigation — clean P1–P5 labels
    st.markdown(
        "<div style='font-size:.65rem;color:#4A5568;font-weight:700;"
        "letter-spacing:.11em;text-transform:uppercase;"
        "margin-bottom:.35rem;padding-left:.1rem;'>Pages</div>",
        unsafe_allow_html=True,
    )
    page = st.radio(
        "pages",
        [
            "P1  —  Predict",
            "P2  —  EDA",
            "P3  —  Features",
            "P4  —  Models",
            "P5  —  About",
        ],
        label_visibility="collapsed",
    )

    # Status
    st.markdown("<hr style='border-color:#1E3A5F;margin:.9rem 0;'>",
                unsafe_allow_html=True)
    sc  = "#276749" if loaded else "#9B2335"
    si  = "●" if loaded else "○"
    stx = "Ready" if loaded else "Not trained"
    live = ""
    if loaded:
        live = (
            f"<div style='margin-top:.5rem;font-size:.74rem;color:#718096;line-height:2;'>"
            f"<span style='color:#4A5568;'>Model</span>&emsp;{meta.get('model_name','—')}<br>"
            f"<span style='color:#4A5568;'>R²</span>&emsp;&emsp;&emsp;{meta.get('r2',0)*100:.1f}%<br>"
            f"<span style='color:#4A5568;'>MAPE</span>&emsp;&ensp;{meta.get('mape',0):.1f}%<br>"
            f"<span style='color:#4A5568;'>Rows</span>&emsp;&ensp;{meta.get('total_rows',0):,}"
            f"</div>"
        )
    st.markdown(
        f"<div style='padding:.1rem .3rem;'>"
        f"<div style='font-size:.63rem;color:#4A5568;font-weight:700;"
        f"letter-spacing:.1em;text-transform:uppercase;margin-bottom:.4rem;'>Status</div>"
        f"<div style='font-size:.8rem;color:{sc};font-weight:600;'>{si} {stx}</div>"
        f"{live}</div>",
        unsafe_allow_html=True,
    )

    # Train hint
    st.markdown("<hr style='border-color:#1E3A5F;margin:.8rem 0;'>",
                unsafe_allow_html=True)
    st.markdown("""
    <div style='padding:.1rem .3rem;'>
      <div style='font-size:.63rem;color:#4A5568;font-weight:700;
           letter-spacing:.1em;text-transform:uppercase;margin-bottom:.35rem;'>Train</div>
      <code style='background:#1E3A5F;color:#C5A028;padding:3px 6px;
           border-radius:4px;font-size:.7rem;line-height:2;display:block;'>
        cd ml_pipeline<br>python train.py</code>
    </div>""", unsafe_allow_html=True)

# ── Route ─────────────────────────────────────────────────────────────────────
if   page == "P1  —  Predict":
    from pages.p1_predict import render;  render()
elif page == "P2  —  EDA":
    from pages.p2_eda import render;      render()
elif page == "P3  —  Features":
    from pages.p3_features import render; render()
elif page == "P4  —  Models":
    from pages.p4_models import render;   render()
elif page == "P5  —  About":
    from pages.p5_about import render;    render()
