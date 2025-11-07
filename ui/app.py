# ============================================================
# app.py
# ------------------------------------------------------------
# Streamlit Footfall Counter UI
# - Equal button sizes
# - Page navigation (sidebar + footer controls)
# - Page routing
# ============================================================

import streamlit as st
from pages import about, analysis, inference

# ------------------------------------------------------------
# Streamlit Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="AI Footfall Counter",
    page_icon="👣",
    layout="wide",
)


# Hide Streamlit's default multipage navigation
st.markdown("""
    <style>
    div[data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Initialize Session State
# ------------------------------------------------------------
if "selected_page" not in st.session_state:
    st.session_state["selected_page"] = "🏠 Overview"

# ------------------------------------------------------------
# Sidebar Button Styling
# ------------------------------------------------------------
st.markdown("""
<style>
[data-testid="stSidebar"] {
    text-align: center;
    background-color: #F8F9FA;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    text-align: center;
}
div.stButton > button {
    width: 180px;
    height: 50px;
    background-color: #2B7A78;
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    margin: 12px auto;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    font-weight: 500;
    letter-spacing: 0.3px;
    transition: all 0.25s ease-in-out;
    cursor: pointer;
    white-space: nowrap;
    text-align: center;
}
div.stButton > button:hover {
    background-color: #3AAFA9;
    transform: translateY(-2px);
}
div.stButton > button:active {
    transform: scale(0.98);
    background-color: #2B7A78 !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------------------
with st.sidebar:
    st.title("👣 Footfall Counter")
    st.caption("AI-based People Detection & Counting System")
    st.markdown("---")

    st.subheader("📂 Navigation")

    col_l, col_c, col_r = st.columns([0.1, 0.8, 0.1])

    with col_c:
        if st.button("🏠 Overview"):
            st.session_state["selected_page"] = "🏠 Overview"

        if st.button("🧪 Test & Analyze"):
            st.session_state["selected_page"] = "🧪 Test & Analyze"

        if st.button("📹 Live Inference"):
            st.session_state["selected_page"] = "📹 Live Inference"

# ------------------------------------------------------------
# Page Routing
# ------------------------------------------------------------
page = st.session_state["selected_page"]

if page == "🏠 Overview":
    about.render_page()

elif page == "🧪 Test & Analyze":
    analysis.render_page()

elif page == "📹 Live Inference":
    inference.render_page()

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; font-size:14px;'>"
    "Developed for AI Assignment — Footfall Counter using Computer Vision | YOLOv8 + DeepSORT"
    "</p>",
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# Bottom Navigation Controls
# ------------------------------------------------------------
col_prev, col_spacer, col_next = st.columns([1, 6, 1])

with col_prev:
    if page == "🧪 Test & Analyze":
        if st.button("⬅️ Overview"):
            st.session_state["selected_page"] = "🏠 Overview"
            st.rerun()
    elif page == "📹 Live Inference":
        if st.button("⬅️ Test & Analyze"):
            st.session_state["selected_page"] = "🧪 Test & Analyze"
            st.rerun()

with col_next:
    if page == "🏠 Overview":
        if st.button("➡️ Test & Analyze"):
            st.session_state["selected_page"] = "🧪 Test & Analyze"
            st.rerun()
    elif page == "🧪 Test & Analyze":
        if st.button("➡️ Live Inference"):
            st.session_state["selected_page"] = "📹 Live Inference"
            st.rerun()