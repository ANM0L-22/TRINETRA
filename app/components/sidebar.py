"""
Trinetra — Sidebar Navigation
"""

import streamlit as st
from datetime import datetime


NAV_PAGES = [
    ("🖥️",  "Dashboard",      "dashboard"),
    ("⚠️",  "Violations",     "violations"),
    ("📊",  "Analytics",      "analytics"),
    ("🗺️",  "Maps / Traffic", "maps"),
    ("⚙️",  "System Status",  "settings"),
]


def render_sidebar():
    with st.sidebar:
        # Logo
        st.markdown("""
        <div style="text-align:center;padding:16px 0 8px;">
            <div style="font-size:32px;">🔱</div>
            <div style="font-family:'Orbitron',monospace;font-size:20px;font-weight:900;
                        color:#e2e8f0;letter-spacing:4px;margin-top:4px;">TRINETRA</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                        letter-spacing:3px;color:#00e5ff;margin-top:4px;">
                AI TRAFFIC INTEL
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # System status
        ts = datetime.now().strftime("%H:%M:%S")
        st.markdown(f"""
        <div style="background:rgba(0,230,118,0.06);border:1px solid rgba(0,230,118,0.2);
                    border-radius:6px;padding:8px 12px;margin-bottom:12px;">
            <div style="display:flex;align-items:center;gap:8px;">
                <div class="active-dot"></div>
                <span style="font-family:'JetBrains Mono',monospace;font-size:10px;
                             color:#00e676;letter-spacing:2px;">SYSTEM ONLINE</span>
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                        color:#334155;margin-top:4px;">{ts}</div>
        </div>
        """, unsafe_allow_html=True)

        # Navigation
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                    letter-spacing:3px;color:#334155;text-transform:uppercase;
                    margin-bottom:8px;">Navigation</div>
        """, unsafe_allow_html=True)

        current = st.session_state.get("page", "dashboard")

        for icon, label, page_key in NAV_PAGES:
            is_active = current == page_key
            bg        = "rgba(0,229,255,0.1)"  if is_active else "transparent"
            border    = "rgba(0,229,255,0.3)"  if is_active else "transparent"
            color     = "#00e5ff"               if is_active else "#94a3b8"

            if st.button(
                f"{icon}  {label}",
                key=f"nav_{page_key}",
                use_container_width=True,
            ):
                st.session_state.page = page_key
                st.rerun()

        st.markdown("<hr style='margin:16px 0;'>", unsafe_allow_html=True)

        # User info
        name = st.session_state.get("display_name", "User")
        role = st.session_state.get("role", "—")
        user = st.session_state.get("username", "—")

        st.markdown(f"""
        <div style="background:rgba(11,17,32,0.8);border:1px solid rgba(0,229,255,0.1);
                    border-radius:8px;padding:12px;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                        color:#64748b;letter-spacing:2px;margin-bottom:6px;">LOGGED IN AS</div>
            <div style="font-family:'Rajdhani',sans-serif;font-size:15px;font-weight:600;
                        color:#e2e8f0;">{name}</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                        color:#00e5ff;margin-top:2px;">@{user}</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                        color:#64748b;margin-top:2px;">{role}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)

        if st.button("🔒  LOGOUT", use_container_width=True, key="logout_btn"):
            for key in ["authenticated", "username", "role", "display_name", "page"]:
                st.session_state.pop(key, None)
            st.rerun()

        # Version
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:8px;
                    color:#1e293b;text-align:center;margin-top:16px;letter-spacing:2px;">
            TRINETRA v1.0.0 · JAIPUR NODE
        </div>
        """, unsafe_allow_html=True)
