"""
╔══════════════════════════════════════════════════════════════╗
║  TRINETRA — AI-Powered Intelligent Traffic Surveillance      ║
║  Smart City Command Center                                   ║
║                                                              ║
║  Run:  streamlit run app.py                                  ║
║                                                              ║
║  Login:  admin / trinetra@2024                               ║
║          officer / police@123                                ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st

# ── Page config — MUST be first Streamlit call ────────────────
st.set_page_config(
    page_title="Trinetra — AI Traffic Intelligence",
    page_icon="🔱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports after page config ─────────────────────────────────
from app.components.styles import get_global_css
from app.components.sidebar import render_sidebar
from app.pages.login import login_page
from app.pages.dashboard import dashboard
from app.pages.violations import violations_page
from app.pages.analytics import analytics_page
from app.pages.maps import maps_page
from app.pages.settings import settings_page

# ── Inject global CSS ─────────────────────────────────────────
st.markdown(get_global_css(), unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "page" not in st.session_state:
    st.session_state.page = "dashboard"

# ── Auth gate ─────────────────────────────────────────────────
if not st.session_state.authenticated:
    login_page()
    st.stop()

# ── Authenticated: render sidebar + route ─────────────────────
render_sidebar()

page = st.session_state.get("page", "dashboard")

PAGE_MAP = {
    "dashboard":  dashboard,
    "violations": violations_page,
    "analytics":  analytics_page,
    "maps":       maps_page,
    "settings":   settings_page,
}

render_fn = PAGE_MAP.get(page, dashboard)
render_fn()
