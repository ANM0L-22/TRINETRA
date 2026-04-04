"""
Trinetra — Login Page
Session-based authentication. Credentials are hardcoded for demo/prototype.
In production: replace with DB lookup + bcrypt.
"""

import streamlit as st
from app.components.styles import get_global_css

CREDENTIALS = {
    "admin":    {"password": "trinetra@2024", "role": "Administrator",  "name": "Admin"},
    "officer":  {"password": "police@123",    "role": "Field Officer",  "name": "Officer"},
    "analyst":  {"password": "analyst@123",   "role": "Data Analyst",   "name": "Analyst"},
}


def login_page():
    st.markdown(get_global_css(), unsafe_allow_html=True)
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(ellipse at 20% 50%, rgba(0,229,255,0.04) 0%, transparent 60%),
                    radial-gradient(ellipse at 80% 20%, rgba(124,58,237,0.06) 0%, transparent 60%),
                    #04060d !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Center the login card
    c1, c2, c3 = st.columns([1, 1.2, 1])
    with c2:
        st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)

        # Logo & title
        st.markdown("""
        <div style="text-align:center;margin-bottom:40px;">
            <div style="font-size:48px;margin-bottom:8px;">🔱</div>
            <div style="font-family:'Orbitron',monospace;font-size:32px;font-weight:900;
                        color:#e2e8f0;letter-spacing:4px;">TRINETRA</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:11px;
                        letter-spacing:4px;color:#00e5ff;margin-top:6px;text-transform:uppercase;">
                AI Traffic Intelligence System
            </div>
            <div style="width:60px;height:2px;background:linear-gradient(90deg,#00e5ff,#7c3aed);
                        margin:12px auto 0;"></div>
        </div>
        """, unsafe_allow_html=True)

        # Login card
        st.markdown("""
        <div style="background:rgba(11,17,32,0.9);border:1px solid rgba(0,229,255,0.15);
                    border-radius:12px;padding:32px;backdrop-filter:blur(16px);
                    box-shadow:0 0 60px rgba(0,229,255,0.06);">
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                    letter-spacing:3px;color:#64748b;margin-bottom:20px;
                    text-transform:uppercase;">Secure Access Portal</div>
        """, unsafe_allow_html=True)

        username = st.text_input("USERNAME", placeholder="Enter username",
                                  key="login_user",
                                  label_visibility="visible")
        password = st.text_input("PASSWORD", type="password",
                                  placeholder="Enter password",
                                  key="login_pass")

        col_btn, col_hint = st.columns([1, 1])
        with col_btn:
            login_clicked = st.button("AUTHENTICATE", use_container_width=True)
        with col_hint:
            st.markdown("""
            <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                        color:#334155;padding-top:10px;line-height:1.6;">
                admin / trinetra@2024<br>
                officer / police@123
            </div>
            """, unsafe_allow_html=True)

        if login_clicked:
            if username in CREDENTIALS and CREDENTIALS[username]["password"] == password:
                st.session_state.authenticated = True
                st.session_state.username      = username
                st.session_state.role          = CREDENTIALS[username]["role"]
                st.session_state.display_name  = CREDENTIALS[username]["name"]
                st.rerun()
            else:
                st.markdown("""
                <div style="background:rgba(255,60,60,0.1);border:1px solid rgba(255,60,60,0.3);
                            border-left:3px solid #ff3c3c;border-radius:6px;padding:10px 14px;
                            margin-top:12px;font-family:'JetBrains Mono',monospace;font-size:11px;
                            color:#ff3c3c;">
                    ⚠ ACCESS DENIED — Invalid credentials
                </div>
                """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Status footer
        st.markdown("""
        <div style="text-align:center;margin-top:24px;">
            <span style="font-family:'JetBrains Mono',monospace;font-size:9px;
                         color:#1e293b;letter-spacing:2px;">
                SYSTEM STATUS: <span style="color:#00e676;">ONLINE</span> &nbsp;·&nbsp;
                VERSION 1.0.0 &nbsp;·&nbsp; SECURE CHANNEL
            </span>
        </div>
        """, unsafe_allow_html=True)
