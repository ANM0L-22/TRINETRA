"""
Trinetra — System Status & Settings
"""

import random
import time
import streamlit as st
from datetime import datetime, timedelta

from app.components.styles import section_header, metric_chip, card


MODULES = [
    ("Vehicle Detector",   "YOLOv8m",          True,  "12ms avg"),
    ("Object Tracker",     "DeepSORT",          True,  "8ms avg"),
    ("ANPR / OCR",         "PaddleOCR",         True,  "35ms avg"),
    ("Violation Detector", "Custom CNN",         True,  "18ms avg"),
    ("Density Analyzer",   "Custom Algorithm",  True,  "4ms avg"),
    ("Report Engine",      "Python/Pandas",     True,  "2ms avg"),
    ("Dashboard Server",   "Streamlit 1.28",    True,  "Active"),
    ("Database",           "PostgreSQL",        False, "Offline"),
]

CAMERAS = [
    ("CAM-01", "NH-48 · Sector 7",      True,  "1080p · 30fps"),
    ("CAM-02", "MG Road · Junction 3",  True,  "1080p · 25fps"),
    ("CAM-03", "Ring Road · Gate 12",   True,  "720p · 30fps"),
    ("CAM-04", "Outer Ring · Node 6",   True,  "1080p · 25fps"),
    ("CAM-05", "City Center · Hub 1",   False, "Reconnecting…"),
]


def settings_page():
    st.markdown(section_header("SYSTEM STATUS", "Infrastructure & Configuration"), unsafe_allow_html=True)

    # ── System health KPIs ─────────────────────────────────────
    uptime_h = 14
    kc1, kc2, kc3, kc4 = st.columns(4)
    with kc1:
        st.metric("System Uptime", f"{uptime_h}h 23m")
    with kc2:
        st.metric("CPU Usage", f"{random.randint(28,48)}%")
    with kc3:
        st.metric("RAM Usage", f"{random.randint(3,6)} GB / 16 GB")
    with kc4:
        st.metric("GPU VRAM", "3.2 GB / 8 GB")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Module status ─────────────────────────────────────────
    col_mod, col_cam = st.columns(2)

    with col_mod:
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:3px;
                    color:#64748b;text-transform:uppercase;margin-bottom:10px;">
            AI Pipeline Modules
        </div>
        """, unsafe_allow_html=True)

        for name, tech, online, latency in MODULES:
            color   = "#00e676" if online else "#ff3c3c"
            status  = "ONLINE" if online else "OFFLINE"
            dot_cls = "active-dot" if online else "blink-dot"
            st.markdown(f"""
            <div style="background:rgba(11,17,32,0.8);border:1px solid rgba(0,229,255,0.08);
                        border-radius:6px;padding:10px 14px;margin-bottom:5px;
                        display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <div style="font-family:'Rajdhani',sans-serif;font-size:14px;
                                font-weight:600;color:#e2e8f0;">{name}</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                                color:#64748b;margin-top:1px;">{tech}</div>
                </div>
                <div style="text-align:right;">
                    <div style="display:flex;align-items:center;gap:6px;justify-content:flex-end;">
                        <span class="{dot_cls}"></span>
                        <span style="font-family:'JetBrains Mono',monospace;font-size:10px;
                                     color:{color};letter-spacing:2px;">{status}</span>
                    </div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                                color:#64748b;margin-top:2px;">{latency}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_cam:
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:3px;
                    color:#64748b;text-transform:uppercase;margin-bottom:10px;">
            Camera Network
        </div>
        """, unsafe_allow_html=True)

        for cam_id, location, online, info in CAMERAS:
            color  = "#00e676" if online else "#ff3c3c"
            status = "LIVE" if online else "OFFLINE"
            st.markdown(f"""
            <div style="background:rgba(11,17,32,0.8);border:1px solid rgba(0,229,255,0.08);
                        border-radius:6px;padding:10px 14px;margin-bottom:5px;
                        display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <div style="display:flex;gap:8px;align-items:center;">
                        <span style="font-family:'Orbitron',monospace;font-size:11px;
                                     font-weight:700;color:{color};">{cam_id}</span>
                        <span style="font-family:'Rajdhani',sans-serif;font-size:13px;
                                     color:#94a3b8;">{location}</span>
                    </div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                                color:#64748b;margin-top:2px;">{info}</div>
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                             color:{color};letter-spacing:2px;">{status}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Settings ──────────────────────────────────────────────
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:3px;
                color:#64748b;text-transform:uppercase;margin-bottom:14px;">
        Configuration
    </div>
    """, unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)

    with s1:
        st.markdown("**Detection Settings**")
        st.slider("Confidence Threshold", 0.30, 0.95, 0.45, 0.05, key="conf_thresh")
        st.slider("IOU Threshold",        0.30, 0.80, 0.50, 0.05, key="iou_thresh")
        st.selectbox("Model", ["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l"], index=2,
                      key="model_size")

    with s2:
        st.markdown("**Violation Settings**")
        st.slider("Helmet Conf. Threshold", 0.40, 0.95, 0.55, 0.05, key="helm_thresh")
        st.slider("Wrong-Side Angle (°)",   60, 150, 120, 10, key="ws_angle")
        st.toggle("Save Evidence Crops", True, key="save_crops")

    with s3:
        st.markdown("**Notification Settings**")
        st.toggle("E-Challan Auto-Issue", False, key="echallan")
        st.toggle("WhatsApp Alerts",      False, key="whatsapp")
        st.toggle("Email Reports",        True,  key="email_rep")
        st.selectbox("Report Interval", ["Hourly","Daily","Weekly"], index=1,
                      key="report_interval")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    save_col, _ = st.columns([1, 4])
    with save_col:
        if st.button("💾  Save Configuration", use_container_width=True):
            st.success("✅ Configuration saved successfully")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Activity log ──────────────────────────────────────────
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:3px;
                color:#64748b;text-transform:uppercase;margin-bottom:10px;">
        System Activity Log
    </div>
    """, unsafe_allow_html=True)

    now = datetime.now()
    log_entries = [
        (now - timedelta(seconds=5),  "INFO",    "Frame 001247 processed — 8 vehicles, 1 violation"),
        (now - timedelta(seconds=12), "WARNING", "Violation: NO_HELMET — Plate RJ14XX9821"),
        (now - timedelta(seconds=28), "INFO",    "ANPR: 6 plates read successfully this batch"),
        (now - timedelta(seconds=45), "INFO",    "Congestion: MODERATE (score=0.48) on NH-48"),
        (now - timedelta(seconds=62), "INFO",    "Model YOLOv8m loaded — device: CPU"),
        (now - timedelta(seconds=90), "INFO",    "Dashboard server started on port 8501"),
        (now - timedelta(seconds=120),"INFO",    "System initialised — all modules online"),
    ]

    log_colors = {"INFO":"#00e5ff","WARNING":"#ff8800","ERROR":"#ff3c3c","DEBUG":"#64748b"}
    log_html = ""
    for ts, level, msg in log_entries:
        c = log_colors.get(level, "#64748b")
        log_html += f'<div style="display:flex;gap:12px;padding:5px 0;border-bottom:1px solid rgba(0,229,255,0.04);font-family:\'JetBrains Mono\',monospace;"><span style="color:#334155;font-size:9px;min-width:72px;">{ts.strftime("%H:%M:%S")}</span><span style="color:{c};font-size:9px;min-width:56px;letter-spacing:1px;">[{level}]</span><span style="color:#94a3b8;font-size:10px;">{msg}</span></div>'

    st.markdown(f'<div style="background:rgba(4,6,13,0.9);border:1px solid rgba(0,229,255,0.1);border-radius:8px;padding:12px;font-family:\'JetBrains Mono\',monospace;max-height:280px;overflow-y:auto;">{log_html}</div>', unsafe_allow_html=True)
