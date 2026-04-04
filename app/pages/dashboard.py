"""
Trinetra — Dashboard (3-Layer Video-Driven)

Layer 1 (Image 1): Video Analysis Panel + Frame Intelligence + Intersection/Alerts
Layer 2 (Image 2): Live Video Feed with AI Detection + Analytics left + Violations right
Layer 3:           Full scrollable analytics — charts, class dist, density, decisions
"""

import os
import time
import tempfile
import threading
import random
import math
from datetime import datetime
from collections import Counter
from pathlib import Path

try:
    import cv2
    _CV2_AVAILABLE = True
except Exception:
    cv2 = None
    _CV2_AVAILABLE = False
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from app.data.video_engine import (
    analyze_frame, get_frame_at, get_video_info, bulk_analyze,
    reset_engine_state,
)
from app.components.styles import get_global_css, section_header, alert_card, pipeline_strip

UPLOAD_DIR = Path("static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── colour helpers ────────────────────────────────────────────
CONG_COLOR = {"LOW": "#00e676", "MEDIUM": "#ffb400",
               "HIGH": "#ff8800", "CRITICAL": "#ff3c3c"}
CLS_COLORS  = {"car": "#00e5ff", "bus": "#ffb400", "bike": "#7c3aed",
               "truck": "#ff8800", "auto": "#00e676"}
VIOL_ICONS  = {"Helmet Violation": "🪖", "Wrong-side Driving": "↩️",
               "Mobile Usage": "📱", "Tampered Plate": "🚫", "No Seatbelt": "🔒"}


def _dark_fig():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", color="#94a3b8", size=10),
        margin=dict(l=4, r=4, t=28, b=4),
        xaxis=dict(showgrid=True, gridcolor="rgba(0,229,255,0.06)", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,229,255,0.06)", zeroline=False),
        showlegend=False,
    )


def _gauge_fig(value, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number=dict(font=dict(family="Orbitron, monospace", size=26, color=color)),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(size=8)),
            bar=dict(color=color, thickness=0.22),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0,  25], color="rgba(0,230,118,0.1)"),
                dict(range=[25, 50], color="rgba(255,180,0,0.08)"),
                dict(range=[50, 75], color="rgba(255,136,0,0.08)"),
                dict(range=[75,100], color="rgba(255,60,60,0.1)"),
            ],
        ),
        title=dict(text=f"<span style='font-size:10px;color:#64748b'>{title}</span>"),
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="JetBrains Mono,monospace", color="#94a3b8"),
                      height=180, margin=dict(l=8, r=8, t=16, b=4))
    return fig


def _append_violations(violations):
    if "viol_log" not in st.session_state:
        st.session_state.viol_log = []
    if "viol_keys" not in st.session_state:
        st.session_state.viol_keys = set()
    for v in violations:
        v["frame_id"] = v.get("frame_id", st.session_state.get("current_frame", 0))
        key = f"{v.get('type')}|track:{v.get('track_id','')}|plate:{v.get('plate','UNKNOWN')}|bbox:{v.get('detected_bbox',[])}"
        if key not in st.session_state.viol_keys:
            st.session_state.viol_keys.add(key)
            st.session_state.viol_log.append(v)


# ── session init ──────────────────────────────────────────────
def _init():
    defaults = {
        "video_path":     None,
        "video_info":     None,
        "bulk_data":      [],
        "frame_data":     None,
        "current_frame":  0,
        "playing":        False,
        "bulk_ready":     False,
        "viol_log":       [],
        "viol_keys":      set(),
        "frame2_data":    None,
        "frame_slider":   0,
        "is_image":       False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════
def dashboard():
    _init()
    st.markdown(get_global_css(), unsafe_allow_html=True)

    if not _CV2_AVAILABLE:
        st.warning(
            "OpenCV is not available in this deployment environment, so video and image analysis are disabled until cv2 can be imported."
        )

    st.markdown("""
    <style>
    /* Extra dashboard CSS */
    .layer-divider {
        border: none;
        border-top: 2px solid rgba(0,229,255,0.12);
        margin: 40px 0 32px 0;
    }
    .layer-label {
        font-family:'Orbitron',monospace;font-size:11px;font-weight:700;
        letter-spacing:4px;color:rgba(0,229,255,0.5);
        text-transform:uppercase;margin-bottom:14px;
        display:flex;align-items:center;gap:12px;
    }
    .layer-label::after{content:'';flex:1;height:1px;background:rgba(0,229,255,0.1);}
    .vcard {
        background:rgba(11,17,32,0.85);border:1px solid rgba(0,229,255,0.12);
        border-radius:8px;padding:12px 14px;margin-bottom:6px;
    }
    .vcard-crit { border-left:3px solid #ff3c3c;background:rgba(255,60,60,0.06); }
    .vcard-high { border-left:3px solid #ff8800;background:rgba(255,136,0,0.05); }
    .vcard-med  { border-left:3px solid #ffb400;background:rgba(255,180,0,0.04); }
    .panel-head {
        font-family:'Orbitron',monospace;font-size:13px;font-weight:700;
        color:#e2e8f0;letter-spacing:1px;margin-bottom:10px;
        display:flex;align-items:center;gap:8px;
    }
    .stat-row {
        display:flex;justify-content:space-between;align-items:center;
        padding:5px 0;border-bottom:1px solid rgba(0,229,255,0.05);
    }
    .stat-key { font-family:'JetBrains Mono',monospace;font-size:10px;color:#64748b; }
    .stat-val { font-family:'Orbitron',monospace;font-size:13px;font-weight:700;color:#00e5ff; }
    </style>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────
    # TOP NAV BAR
    # ─────────────────────────────────────────────────────────
    now = datetime.now()
    fd  = st.session_state.get("frame_data", {}) or {}
    is_image = st.session_state.get("is_image", False)
    fps_val  = "N/A" if is_image else fd.get("fps", "—")
    cong     = fd.get("congestion", "—")
    n_veh    = fd.get("n_vehicles", 0)
    cong_col = CONG_COLOR.get(cong, "#00e5ff")
    mode_str = "IMAGE ANALYSIS" if is_image else "VIDEO ANALYSIS"

    # Build header HTML with status boxes
    header_html = '<div style="background:rgba(6,11,22,0.95);border:1px solid rgba(0,229,255,0.12);border-radius:10px;padding:10px 20px;display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;gap:8px;flex-wrap:wrap;">'
    header_html += '<div style="display:flex;align-items:center;gap:10px;">'
    header_html += '<span style="font-size:22px;">🔱</span>'
    header_html += '<div>'
    header_html += '<div style="font-family:\'Orbitron\',monospace;font-size:15px;font-weight:900;color:#e2e8f0;letter-spacing:3px;">TRINETRA</div>'
    header_html += '<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;color:#00e5ff;letter-spacing:3px;">AI TRAFFIC INTELLIGENCE</div>'
    header_html += '</div></div>'
    header_html += '<div style="display:flex;gap:24px;align-items:center;flex-wrap:wrap;">'
    
    # Add status boxes
    status_data = [
        ("Status", "ACTIVE", "#00e676"),
        ("FPS", str(fps_val), "#00e5ff"),
        ("Vehicles", str(n_veh), "#ffb400"),
        ("Congestion", cong, cong_col),
        ("Mode", mode_str, "#7c3aed"),
        ("Time", now.strftime("%H:%M:%S"), "#e2e8f0"),
    ]
    
    for lbl, val, col in status_data:
        header_html += f'<div style="text-align:center;"><div style="font-family:\'JetBrains Mono\',monospace;font-size:8px;letter-spacing:2px;color:#334155;text-transform:uppercase;">{lbl}</div><div style="font-family:\'Orbitron\',monospace;font-size:13px;font-weight:700;color:{col};">{val}</div></div>'
    
    header_html += '</div></div>'
    
    st.markdown(header_html, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────
    # LAYER 1 — Image 1 layout
    # ─────────────────────────────────────────────────────────
    # (layer label removed for cleaner packing in same space)
    main_left, main_center, main_right = st.columns([1.1, 2.6, 1.3])

    # ── LAYER 1 LEFT: Sidebar nav + upload (logo strip removed) ───
    with main_left:
        st.markdown("""
        <div style="background:rgba(6,11,22,0.9);border:1px solid rgba(0,229,255,0.12);
                    border-radius:10px;padding:16px;min-height:420px;">
        """, unsafe_allow_html=True)

        nav_items = [
            ("📹","Video Input","video"),("🔍","Detection","det"),
            ("🎯","Tracking","track"),("🔤","OCR","ocr"),
            ("⚠️","Violations","viol"),("📊","Analytics","anal"),
            ("🖥️","Dashboard","dash"),
        ]
        for icon, label, key in nav_items:
            active = key == "video"
            bg = "rgba(0,229,255,0.12)" if active else "transparent"
            border = "1px solid rgba(0,229,255,0.3)" if active else "1px solid transparent"
            color  = "#00e5ff" if active else "#94a3b8"
            st.markdown(f"""
            <div style="background:{bg};border:{border};border-radius:6px;
                        padding:7px 12px;margin-bottom:4px;display:flex;
                        align-items:center;gap:10px;cursor:pointer;">
                <span style="font-size:14px;">{icon}</span>
                <span style="font-family:'Rajdhani',sans-serif;font-size:14px;
                             font-weight:500;color:{color};">{label}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Upload widget
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        uploaded = st.file_uploader("📂 Upload Traffic Video/Image",
                                     type=["mp4","avi","mov","mkv","jpg","jpeg","png"],
                                     key="vid_upload_l1",
                                     help="Upload CCTV/dashcam video or static image")
        if uploaded and uploaded.name != st.session_state.get("_last_upload_name", ""):
            st.session_state["_last_upload_name"] = uploaded.name
            save_path = UPLOAD_DIR / f"upload_{uploaded.name.replace(' ','_')}"
            save_path.write_bytes(uploaded.read())
            
            # Check if image or video
            ext = save_path.suffix.lower()
            is_image = ext in [".jpg", ".jpeg", ".png"]
            st.session_state.is_image = is_image
            
            if is_image:
                # For images, treat as single frame
                st.session_state.video_path = str(save_path)
                st.session_state.video_info = None  # No video info for images
                st.session_state.bulk_ready = True
                # Analyze the image as frame 0
                if cv2 is not None:
                    frame_bgr = cv2.imread(str(save_path))
                    if frame_bgr is not None:
                        fd = analyze_frame(frame_bgr, 0, 1)
                        if fd:
                            fd["frame_id"] = 0
                            st.session_state.frame_data = fd
                            st.session_state.bulk_data = [fd]  # Single frame data
                            st.session_state.viol_log = []
                            st.session_state.viol_keys = set()
                            _append_violations(fd.get("violations", []))
                else:
                    st.error("OpenCV is not available, so image analysis cannot run in this deployment.")
                st.session_state.current_frame = 0
                reset_engine_state()
            else:
                # Video processing
                info = get_video_info(str(save_path))
                st.session_state.video_path = str(save_path)
                st.session_state.video_info = info
                st.session_state.bulk_ready = False
                st.session_state.bulk_data = []
                st.session_state.current_frame = 0
                st.session_state.viol_log = []
                st.session_state.viol_keys = set()
                reset_engine_state()
                # Bulk analyze in thread
                def _bg():
                    data = bulk_analyze(str(save_path), max_samples=180)
                    st.session_state["bulk_data"] = data
                    st.session_state["bulk_ready"] = True
                threading.Thread(target=_bg, daemon=True).start()
            st.rerun()

    # ── LAYER 1 CENTER: Video/Image panel ───────────────────────────
    with main_center:
        st.markdown("""
        <div class="panel-head">📹 Video/Image Analysis Panel
            <span style="font-family:'JetBrains Mono',monospace;font-size:9px;
                         color:#64748b;margin-left:auto;letter-spacing:2px;">
                AI-Powered Intelligent Traffic Monitoring
            </span>
        </div>
        """, unsafe_allow_html=True)

        is_image = st.session_state.get("is_image", False)

        if is_image:
            # For images, show analyzed image directly
            fd = st.session_state.get("frame_data", {}) or {}
            b64 = fd.get("frame_b64", "")
            if b64:
                st.markdown(f"""
                <div style="border:1px solid rgba(0,229,255,0.15);border-radius:8px;
                            overflow:hidden;position:relative;">
                    <img src="data:image/jpeg;base64,{b64}"
                         style="width:100%;display:block;">
                </div>
                """, unsafe_allow_html=True)

                hud_den = fd.get("density", 0)
                hud_veh = fd.get("n_vehicles", 0)
                hud_cng = fd.get("congestion", "—")
                st.markdown(f"""
                <div style="text-align:center;font-family:'JetBrains Mono',monospace;
                            font-size:11px;color:#64748b;padding:4px 0;">
                    Static Image Analysis &nbsp;|&nbsp; Vehicles: {hud_veh}
                    &nbsp;|&nbsp; Congestion: {int(hud_den*100)}%
                </div>
                """, unsafe_allow_html=True)
        else:
            # Video processing
            # Frame selector
            info = st.session_state.video_info
            total_frames = info.get("total_frames", 100) if info else 100

            col_mode1, col_mode2 = st.columns([1, 1])
            with col_mode1:
                view_mode = st.radio("View", ["Normal Video", "Frame Analysis"],
                                      horizontal=True, key="view_mode_l1",
                                      label_visibility="collapsed")
            with col_mode2:
                frame_num = st.slider("Select Frame", 0,
                                       max(total_frames - 1, 1),
                                       st.session_state.current_frame,
                                       key="frame_slider_l1",
                                       label_visibility="collapsed")
                st.session_state.current_frame = frame_num

            # Video / frame display
            if st.session_state.video_path:
                if view_mode == "Normal Video":
                    st.video(st.session_state.video_path)
                else:
                    # Analyse selected frame
                    if (not st.session_state.get("frame_data") or
                            (st.session_state.get("frame_data") or {}).get("frame_id") != frame_num):
                        fd_new = get_frame_at(st.session_state.video_path, frame_num)
                        if fd_new:
                            st.session_state.frame_data = fd_new
                            # Append new violations to log deduplicated by stable keys
                            for v in fd_new.get("violations", []):
                                v["frame_id"] = frame_num
                                key = f"{v.get('type')}|track:{v.get('track_id','')}|plate:{v.get('plate','UNKNOWN')}|bbox:{v.get('detected_bbox',[])}"
                                if key not in st.session_state.viol_keys:
                                    st.session_state.viol_keys.add(key)
                                    st.session_state.viol_log.append(v)

                    fd = st.session_state.get("frame_data", {}) or {}
                    b64 = fd.get("frame_b64", "")
                    if b64:
                        st.markdown(f"""
                        <div style="border:1px solid rgba(0,229,255,0.15);border-radius:8px;
                                    overflow:hidden;position:relative;">
                            <img src="data:image/jpeg;base64,{b64}"
                                 style="width:100%;display:block;">
                        </div>
                        """, unsafe_allow_html=True)

                    hud_den = fd.get("density", 0)
                    hud_veh = fd.get("n_vehicles", 0)
                    hud_cng = fd.get("congestion", "—")
                    st.markdown(f"""
                    <div style="text-align:center;font-family:'JetBrains Mono',monospace;
                                font-size:11px;color:#64748b;padding:4px 0;">
                        Frame {frame_num} &nbsp;|&nbsp; Vehicles: {hud_veh}
                        &nbsp;|&nbsp; Congestion: {int(hud_den*100)}%
                    </div>
                    """, unsafe_allow_html=True)
            else:
                no_video_html = """
                <div style="background:rgba(11,17,32,0.8);border:2px dashed rgba(0,229,255,0.2);
                            border-radius:10px;padding:60px;text-align:center;">
                    <div style="font-size:40px;margin-bottom:12px;">📹</div>
                    <div style="font-family:'Orbitron',monospace;font-size:13px;color:#64748b;
                                letter-spacing:2px;">Upload a traffic video or image to begin</div>
                </div>
                """
                st.markdown(no_video_html, unsafe_allow_html=True)

        # Controls
        if not is_image:
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("▶ Analyse Frame", use_container_width=True, key="btn_anal"):
                    if st.session_state.video_path:
                        fd_new = get_frame_at(st.session_state.video_path, frame_num)
                        if fd_new:
                            st.session_state.frame_data = fd_new
                            _append_violations(fd_new.get("violations", []))
            with c2:
                if st.button("⏭ Next Frame", use_container_width=True, key="btn_next"):
                    nxt = min(frame_num + 1, total_frames - 1)
                    st.session_state.current_frame = nxt
                    st.session_state.frame_data = None
                    if st.session_state.video_path:
                        fd_next = get_frame_at(st.session_state.video_path, nxt)
                        if fd_next:
                            st.session_state.frame_data = fd_next
                            _append_violations(fd_next.get("violations", []))
            with c3:
                if st.button("🔄 Refresh All", use_container_width=True, key="btn_ref"):
                    st.session_state.frame_data = None
                    st.session_state.viol_log = []
                    st.session_state.viol_keys = set()
        else:
            # For images, single re-analyse button
            if st.button("🔄 Re-analyse Image", use_container_width=True, key="btn_reanal_img"):
                if st.session_state.video_path:
                    frame_bgr = cv2.imread(st.session_state.video_path)
                    if frame_bgr is not None:
                        fd_new = analyze_frame(frame_bgr, 0, 1)
                        if fd_new:
                            fd_new["frame_id"] = 0
                            st.session_state.frame_data = fd_new
                            st.session_state.bulk_data = [fd_new]
                            st.session_state.viol_log = []
                            st.session_state.viol_keys = set()
                            _append_violations(fd_new.get("violations", []))

        # Pipeline strip (collapsed by default)
        with st.expander("AI Pipeline (Video→Detect→Track→OCR→Violations→Analyze→Dashboard)", expanded=False):
            st.markdown(pipeline_strip(), unsafe_allow_html=True)

        # ── Frame Intelligence Panel (below video) ────────────
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'Orbitron',monospace;font-size:13px;font-weight:700;
                    color:#e2e8f0;letter-spacing:1px;margin-bottom:10px;">
            🔱 Frame Intelligence Panel
        </div>
        """, unsafe_allow_html=True)

        fi_l, fi_r = st.columns(2)

        with fi_l:
            st.markdown("""
            <div class="panel-head" style="font-size:12px;">📊 Performance Analytics</div>
            """, unsafe_allow_html=True)
            fd = st.session_state.get("frame_data", {}) or {}
            fps_v  = fd.get("fps", 31)
            ms_v   = fd.get("proc_ms", 28)
            den_v  = int(fd.get("density", 0.46) * 100)
            cong_v = fd.get("congestion", "MEDIUM")

            st.plotly_chart(_gauge_fig(fps_v, "FPS", "#00e5ff"),
                             use_container_width=True,
                             config={"displayModeBar": False}, key="g_fps")
            st.markdown(f"""
            <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:4px;">
                <div class="vcard" style="flex:1;text-align:center;">
                    <div style="font-family:'JetBrains Mono',monospace;font-size:8px;
                                color:#64748b;letter-spacing:2px;">AVG PROC TIME</div>
                    <div style="font-family:'Orbitron',monospace;font-size:20px;
                                font-weight:700;color:#00e5ff;">{fps_v} <span style="font-size:11px;">FPS</span></div>
                </div>
                <div class="vcard" style="flex:1;text-align:center;">
                    <div style="font-family:'JetBrains Mono',monospace;font-size:8px;
                                color:#64748b;letter-spacing:2px;">DETECT ACCURACY</div>
                    <div style="font-family:'Orbitron',monospace;font-size:20px;
                                font-weight:700;color:#7c3aed;">{ms_v} <span style="font-size:11px;">ms</span></div>
                </div>
            </div>
            <div style="text-align:center;margin-top:8px;background:rgba(255,180,0,0.08);
                        border:1px solid rgba(255,180,0,0.2);border-radius:6px;padding:8px;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                            color:#64748b;letter-spacing:2px;">{cong_v} CONGESTION</div>
                <div style="font-family:'Orbitron',monospace;font-size:22px;font-weight:700;
                            color:{CONG_COLOR.get(cong_v,'#ffb400')};">{den_v}%</div>
            </div>
            """, unsafe_allow_html=True)

        with fi_r:
            st.markdown("""
            <div class="panel-head" style="font-size:12px;">⚠️ Violations</div>
            """, unsafe_allow_html=True)
            fd     = st.session_state.get("frame_data", {}) or {}
            viols  = fd.get("violations", [])
            vcount = Counter(v["type"] for v in st.session_state.viol_log)

            if vcount:
                types  = list(vcount.keys())
                values = list(vcount.values())
                colors = ["#ff3c3c","#ff8800","#ffb400","#7c3aed","#00e5ff"][:len(types)]
                fig    = go.Figure(go.Bar(
                    x=[t.replace(" ", "<br>") for t in types],
                    y=values,
                    marker=dict(color=colors, opacity=0.85,
                                line=dict(color=colors, width=0.5)),
                    text=values, textposition="outside",
                    textfont=dict(size=11, color="#e2e8f0"),
                ))
                layout_cfg = {**_dark_fig(), 'height': 200, 'bargap': 0.25, 'xaxis': dict(tickfont=dict(size=8), showgrid=False)}
                fig.update_layout(layout_cfg)  # type: ignore
                st.plotly_chart(fig, use_container_width=True,
                                 config={"displayModeBar": False}, key="viol_bar_l1")
            else:
                st.info("No violations yet — analyse frames to detect")

            # Violation table
            all_v = st.session_state.viol_log[-6:][::-1]
            for v in all_v:
                icon  = VIOL_ICONS.get(v["type"], "⚠️")
                vehicle_class = v.get("related_vehicle", {}).get("class", "UNKNOWN")
                plate = v.get("plate", "N/A")
                st.markdown(f"""
                <div class="vcard vcard-crit">
                    <span style="font-family:'JetBrains Mono',monospace;font-size:10px;
                                 color:#ff3c3c;">{icon} {v['type']}</span>
                    &nbsp;·&nbsp;
                    <span style="font-family:'JetBrains Mono',monospace;font-size:10px;
                                 color:#94a3b8;">{vehicle_class}</span>
                    &nbsp;·&nbsp;
                    <span style="font-family:'Orbitron',monospace;font-size:10px;
                                 color:#ffb400;">{plate}</span>
                </div>
                """, unsafe_allow_html=True)

    # ── LAYER 1 RIGHT: Intersection + Alerts ─────────────────
    with main_right:
        st.markdown("""
        <div class="panel-head">📍 Intersection View</div>
        """, unsafe_allow_html=True)

        fd = st.session_state.get("frame_data", {}) or {}
        density_pct = int(fd.get("density", 0.43) * 100)
        cong_lv     = fd.get("congestion", "MEDIUM")
        cong_c      = CONG_COLOR.get(cong_lv, "#ffb400")

        # Map placeholder (styled card)
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0a1428 0%,#0d1f3c 50%,#0a1428 100%);
                    border:1px solid rgba(0,229,255,0.15);border-radius:8px;
                    height:200px;position:relative;overflow:hidden;margin-bottom:10px;">
            <div style="position:absolute;inset:0;
                background:repeating-linear-gradient(rgba(0,229,255,0.03) 1px,transparent 1px,transparent 30px),
                           repeating-linear-gradient(90deg,rgba(0,229,255,0.03) 1px,transparent 1px,transparent 30px);">
            </div>
            <!-- road lines -->
            <div style="position:absolute;top:50%;left:0;right:0;height:2px;
                        background:rgba(255,180,0,0.4);transform:translateY(-50%)"></div>
            <div style="position:absolute;top:0;bottom:0;left:50%;width:2px;
                        background:rgba(255,180,0,0.3);transform:translateX(-50%)"></div>
            <!-- location pin -->
            <div style="position:absolute;top:38%;left:52%;transform:translate(-50%,-50%);
                        font-size:28px;">📍</div>
            <!-- density ring -->
            <div style="position:absolute;top:28%;left:43%;
                        width:60px;height:60px;border-radius:50%;
                        border:3px solid {cong_c};opacity:0.5;
                        animation:ring-pulse 2s infinite;"></div>
            <style>@keyframes ring-pulse{{0%,100%{{transform:scale(1);opacity:.5}}50%{{transform:scale(1.15);opacity:.2}}}}</style>
            <!-- bottom label -->
            <div style="position:absolute;bottom:8px;left:0;right:0;text-align:center;
                        font-family:'JetBrains Mono',monospace;font-size:9px;color:#64748b;
                        letter-spacing:2px;">NH-48 · SECTOR 7 · JAIPUR</div>
        </div>
        <div style="background:rgba(11,17,32,0.8);border:1px solid rgba(0,229,255,0.12);
                    border-radius:8px;padding:10px 14px;margin-bottom:10px;
                    display:flex;align-items:center;gap:12px;">
            <div style="width:12px;height:12px;border-radius:50%;background:{cong_c};
                        box-shadow:0 0 8px {cong_c};flex-shrink:0;"></div>
            <span style="font-family:'Orbitron',monospace;font-size:14px;font-weight:700;
                         color:{cong_c};">{cong_lv}</span>
            <span style="margin-left:auto;font-family:'Orbitron',monospace;font-size:20px;
                         font-weight:900;color:{cong_c};">{density_pct}%</span>
        </div>
        """, unsafe_allow_html=True)

        if st.button("➕ Add New Intersection", use_container_width=True, key="btn_add_int"):
            st.toast("Feature available in production deployment", icon="ℹ️")

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="panel-head">⚠️ Alerts</div>
        """, unsafe_allow_html=True)

        # Location alerts
        locations = [
            ("Golf Course Rd", density_pct, cong_c),
            ("MG Road",        max(0, density_pct - 12), CONG_COLOR.get("MEDIUM","#ffb400")),
            ("Ring Road",      min(99, density_pct + 18), CONG_COLOR.get("HIGH","#ff8800")),
        ]
        for loc, pct, lc in locations:
            st.markdown(f"""
            <div style="background:{lc}11;border:1px solid {lc}33;border-radius:6px;
                        padding:8px 12px;margin-bottom:5px;
                        display:flex;justify-content:space-between;align-items:center;">
                <span style="font-family:'Rajdhani',sans-serif;font-size:14px;
                             font-weight:600;color:#e2e8f0;">{loc}</span>
                <span style="font-family:'Orbitron',monospace;font-size:15px;
                             font-weight:700;color:{lc};">{pct}%</span>
            </div>
            """, unsafe_allow_html=True)

        # Live violations in alert table
        st.markdown("""
        <div style="margin-top:8px;font-family:'JetBrains Mono',monospace;font-size:9px;
                    color:#334155;letter-spacing:2px;text-transform:uppercase;">
            Time &nbsp;·&nbsp; Violation &nbsp;·&nbsp; Type &nbsp;·&nbsp; Vehicle &nbsp;·&nbsp; Plate
        </div>
        """, unsafe_allow_html=True)
        for v in st.session_state.viol_log[-6:][::-1]:
            ts   = datetime.now().strftime("%H:%M")
            icon = VIOL_ICONS.get(v["type"], "⚠️")
            vehicle_class = v.get("related_vehicle", {}).get("class", "UNKNOWN")[:4]
            plate = v.get("plate", "N/A")
            viol_type = v.get("type", "UNKNOWN").split()[0]
            st.markdown(f"""
            <div style="display:flex;gap:6px;padding:5px 0;
                        border-bottom:1px solid rgba(0,229,255,0.05);
                        font-family:'JetBrains Mono',monospace;font-size:10px;align-items:center;">
                <span style="color:#64748b;min-width:38px;">{ts}</span>
                <span style="color:#ff3c3c;">{icon}</span>
                <span style="color:#94a3b8;flex:1;overflow:hidden;white-space:nowrap;
                             text-overflow:ellipsis;">{viol_type}</span>
                <span style="color:#64748b;">MG Rd</span>
                <span style="color:#94a3b8;">{vehicle_class}</span>
                <span style="color:#ffb400;min-width:72px;">{plate}</span>
            </div>
            """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # LAYER 2 — Image 2 layout
    # ═══════════════════════════════════════════════════════════
    st.markdown('<hr class="layer-divider">', unsafe_allow_html=True)
    st.markdown('<div class="layer-label">▎ LAYER 2 · LIVE VIDEO FEED WITH AI DETECTION</div>',
                unsafe_allow_html=True)

    l2_left, l2_center, l2_right = st.columns([1.2, 2.4, 1.4])

    # ── L2 LEFT: Analytics summary ────────────────────────────
    with l2_left:
        st.markdown("""
        <div class="panel-head">📊 Analytics</div>
        <div class="panel-head" style="font-size:11px;margin-top:4px;">🏙️ Traffic Analytics</div>
        """, unsafe_allow_html=True)

        bulk = st.session_state.bulk_data
        if bulk:
            vc_y = [d["n_vehicles"] for d in bulk]
            vc_x = list(range(len(vc_y)))
            fig_vc = go.Figure(go.Scatter(
                x=vc_x, y=vc_y, mode="lines",
                line=dict(color="#00e5ff", width=2),
                fill="tozeroy", fillcolor="rgba(0,229,255,0.07)",
            ))
            fig_vc.update_layout(**_dark_fig(), height=140,  # type: ignore
                                  title=dict(text="Vehicle Count Over Time",
                                             font=dict(size=10, color="#94a3b8")))
            st.plotly_chart(fig_vc, use_container_width=True,
                             config={"displayModeBar": False}, key="vc_l2")

            den_y = [d["density"] for d in bulk]
            fig_den = go.Figure(go.Scatter(
                x=vc_x, y=[d*100 for d in den_y], mode="lines",
                line=dict(color="#7c3aed", width=2),
                fill="tozeroy", fillcolor="rgba(124,58,237,0.07)",
            ))
            fig_den.update_layout({
                                   **_dark_fig(),
                                   "height": 140,
                                   "title": dict(text="Congestion Score Over Time",
                                                  font=dict(size=10, color="#94a3b8")),
                                   "yaxis": dict(
                                       showgrid=True,
                                       gridcolor="rgba(0,229,255,0.06)",
                                       zeroline=False,
                                       range=[0, 100],
                                   )
                                   })
            st.plotly_chart(fig_den, use_container_width=True,
                             config={"displayModeBar": False}, key="den_l2")
        else:
            st.info("Analysing video… charts appear after upload")

        # Violations detected list
        vcount = Counter(v["type"] for v in st.session_state.viol_log)
        if vcount:
            st.markdown("<div style='margin-top:8px;font-family:Rajdhani,sans-serif;font-size:14px;font-weight:600;color:#e2e8f0;'>Violations Detected:</div>", unsafe_allow_html=True)
            for vtype, cnt in vcount.most_common():
                icon = VIOL_ICONS.get(vtype, "⚠️")
                st.markdown(f"""
                <div style="padding:4px 0;font-family:'Rajdhani',sans-serif;font-size:13px;
                            color:#94a3b8;">• {icon} {vtype}: <b style='color:#ff3c3c;'>{cnt}</b></div>
                """, unsafe_allow_html=True)

        # Class distribution pie
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'Rajdhani',sans-serif;font-size:14px;font-weight:600;
                    color:#e2e8f0;margin-bottom:6px;">Vehicle Class Distribution</div>
        """, unsafe_allow_html=True)

        cls_totals = {}
        for d in st.session_state.bulk_data:
            for k, v in d.get("counts", {}).items():
                cls_totals[k] = cls_totals.get(k, 0) + v
        if not cls_totals:
            fd_ = st.session_state.get("frame_data", {}) or {}
            cls_totals = fd_.get("counts", {"car": 4, "bike": 3, "bus": 1, "truck": 2, "auto": 2})

        labels = list(cls_totals.keys())
        values = list(cls_totals.values())
        colors = [CLS_COLORS.get(l, "#64748b") for l in labels]
        fig_pie = go.Figure(go.Pie(
            labels=[l.title() for l in labels], values=values,
            hole=0.45, marker=dict(colors=colors, line=dict(color="#04060d", width=2)),
            textfont=dict(size=9),
        ))
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                               showlegend=True, height=200,
                               margin=dict(l=4, r=4, t=4, b=4),
                               legend=dict(bgcolor="rgba(0,0,0,0)",
                                           font=dict(size=9, color="#94a3b8")),
                               font=dict(color="#94a3b8"))
        st.plotly_chart(fig_pie, use_container_width=True,
                         config={"displayModeBar": False}, key="pie_l2")

    # ── L2 CENTER: Live feed with frame slider ─────────────────
    with l2_center:
        st.markdown("""
        <div class="panel-head">📷 Live Video Feed with AI Detection</div>
        """, unsafe_allow_html=True)

        video_info = st.session_state.get("video_info") or {}
        total_frames = video_info.get("total_frames", 1)
        frame2_num = st.slider("Frame #", 0,
                                max(total_frames - 1, 1),
                                st.session_state.get("current_frame", 0),
                                key="frame_slider_l2")

        if st.session_state.get("video_path"):
            if st.session_state.get("is_image", False):
                fd2 = st.session_state.get("frame_data", {}) or {}
            else:
                video_path_val: str = st.session_state.get("video_path", "")
                fd2 = get_frame_at(video_path_val, frame2_num)

            if fd2:
                st.session_state.frame_data = fd2
                b64_2 = fd2.get("frame_b64", "")
                if b64_2:
                    st.markdown(f"""
                    <div style="border:1px solid rgba(0,229,255,0.15);border-radius:8px;
                                overflow:hidden;position:relative;">
                        <img src="data:image/jpeg;base64,{b64_2}" style="width:100%;display:block;">
                    </div>
                    """, unsafe_allow_html=True)
                den2 = int(fd2.get("density", 0) * 100)
                nv2  = fd2.get("n_vehicles", 0)
                st.markdown(f"""
                <div style="text-align:center;font-family:'JetBrains Mono',monospace;
                            font-size:11px;color:#64748b;padding:4px 0;">
                    Frame {frame2_num} &nbsp;|&nbsp; Vehicles: {nv2}
                    &nbsp;|&nbsp; Congestion: {den2}%
                </div>
                """, unsafe_allow_html=True)
                # Detected vehicles breakdown
                for v in fd2.get("vehicles", [])[:4]:
                    col_v = CLS_COLORS.get(v["class"], "#00e5ff")
                    viol_mark = " 🚨" if v.get("violation") else ""
                    track_id = v.get("track_id", "?")
                    confidence = v.get("conf", 0)
                    plate = v.get("plate", "")
                    st.markdown(f"""
                    <div style="display:flex;gap:8px;padding:3px 0;
                                border-bottom:1px solid rgba(0,229,255,0.05);
                                font-family:'JetBrains Mono',monospace;font-size:10px;align-items:center;">
                        <span style="color:{col_v};min-width:50px;">{v['class']}</span>
                        <span style="color:#94a3b8;">#{track_id}</span>
                        <span style="color:#64748b;">{confidence:.2f}</span>
                        <span style="color:#ffb400;margin-left:auto;">{plate}</span>
                        <span>{viol_mark}</span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            no_feed_html = """
            <div style="background:rgba(11,17,32,0.8);border:2px dashed rgba(0,229,255,0.15);
                        border-radius:10px;padding:40px;text-align:center;height:280px;
                        display:flex;flex-direction:column;justify-content:center;align-items:center;">
                <div style="font-size:36px;margin-bottom:8px;">📷</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#334155;">
                    Upload video in Layer 1 to see live feed
                </div>
            </div>
            """
            st.markdown(no_feed_html, unsafe_allow_html=True)

    # ── L2 RIGHT: Real-time violations alert ──────────────────
    with l2_right:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
            <span style="font-size:18px;">🚨</span>
            <div class="panel-head" style="margin-bottom:0;">Violations Alert</div>
        </div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
            <span style="font-size:16px;">🚨</span>
            <div style="font-family:'Orbitron',monospace;font-size:12px;font-weight:700;
                        color:#e2e8f0;letter-spacing:1px;">Real-Time Violations</div>
        </div>
        """, unsafe_allow_html=True)

        live_viols = st.session_state.get("viol_log", [])
        if live_viols:
            live_viols = live_viols[::-1]
            page_size = 6
            total_pages = math.ceil(len(live_viols) / page_size)
            page = st.slider("Violation page", 1, total_pages, key="viol_page", label_visibility="collapsed") if total_pages > 1 else 1
            page_items = live_viols[(page - 1) * page_size: page * page_size]
            st.markdown('<div style="max-height:560px;overflow-y:auto;padding-right:6px;">', unsafe_allow_html=True)
            for v in page_items:
                icon = VIOL_ICONS.get(v["type"], "⚠️")
                type_label = v.get("type", "Violation").replace("_", " ").title()
                sev  = "crit" if "wrong" in v["type"] or "tampered" in v["type"] else "high"
                cls_s = "vcard-crit" if sev == "crit" else "vcard-high"
                ts    = v.get("event_time", datetime.now().strftime("%H:%M:%S"))
                vehicle_class = v.get("related_vehicle", {}).get("class", "UNKNOWN").title()
                plate = v.get("plate", "N/A")
                confidence = v.get("confidence", 0)
                event_line = f"Frame {v.get('frame_id', '—')}"
                if v.get("event_time"):
                    event_line += f" · {ts}"
                st.markdown(f"""
                <div class="vcard {cls_s}">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span style="font-family:'Orbitron',monospace;font-size:10px;
                                     color:{'#ff3c3c' if sev=='crit' else '#ff8800'};
                                     font-weight:700;">{icon} {type_label}</span>
                        <span style="font-family:'JetBrains Mono',monospace;font-size:8px;
                                     color:#64748b;">{ts}</span>
                    </div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                                color:#94a3b8;margin-top:3px;">
                        {vehicle_class} &nbsp;·&nbsp;
                        <span style="color:#ffb400;">{plate}</span>
                    </div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                                color:#64748b;margin-top:2px;">{event_line} · Conf: {confidence:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:rgba(0,230,118,0.08);border:1px solid rgba(0,230,118,0.3);
                        border-radius:6px;padding:12px;text-align:center;
                        font-family:'JetBrains Mono',monospace;font-size:11px;color:#00e676;">
                ✓ No violations detected
            </div>
            """, unsafe_allow_html=True)

        # Violation type summary
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        vcount2 = Counter(v["type"] for v in st.session_state.viol_log)
        for vtype, cnt in vcount2.most_common(5):
            icon = VIOL_ICONS.get(vtype, "⚠️")
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:6px 0;border-bottom:1px solid rgba(0,229,255,0.05);">
                <span style="font-family:'Rajdhani',sans-serif;font-size:13px;color:#e2e8f0;">
                    {icon} {vtype}</span>
                <span style="font-family:'Orbitron',monospace;font-size:14px;
                             font-weight:700;color:#ff3c3c;">{cnt}</span>
            </div>
            """, unsafe_allow_html=True)

        # AI decisions
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="panel-head" style="font-size:11px;">🧠 AI Decisions</div>
        """, unsafe_allow_html=True)
        fd_dec = st.session_state.frame_data or {}
        for dec in fd_dec.get("decisions", [])[:3]:
            c2 = "#ff3c3c" if "🔴" in dec else "#ffb400" if "🟡" in dec else "#00e676"
            st.markdown(f"""
            <div style="background:{c2}0d;border-left:3px solid {c2};border-radius:4px;
                        padding:6px 10px;margin-bottom:5px;font-family:'Rajdhani',sans-serif;
                        font-size:12px;color:{c2};">{dec}</div>
            """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # LAYER 3 — Full analytics
    # ═══════════════════════════════════════════════════════════
    st.markdown('<hr class="layer-divider">', unsafe_allow_html=True)
    st.markdown('<div class="layer-label">▎ LAYER 3 · DEEP ANALYTICS & SYSTEM INTELLIGENCE</div>',
                unsafe_allow_html=True)

    bulk = st.session_state.get("bulk_data", []) or []
    fd3  = st.session_state.get("frame_data", {}) or {}
    bulk_ready = st.session_state.get("bulk_ready", False)

    # Bulk analysis status indicator
    if st.session_state.get("video_path"):
        if bulk_ready and bulk:
            st.success(f"Bulk analysis complete: {len(bulk)} frames processed.")
        else:
            st.info("Bulk analysis in progress... please wait (no charts yet until done)")

    # KPI strip
    all_viols_count = len(st.session_state.viol_log)
    avg_density_pct = int(sum(d["density"] for d in bulk) / len(bulk) * 100) if bulk else None
    peak_veh        = max((d["n_vehicles"] for d in bulk), default=None)
    total_plates    = sum(len(d.get("plates",[])) for d in bulk)
    video_info      = st.session_state.get("video_info", {}) or {}
    total_frames    = video_info.get("total_frames") if video_info else (len(bulk) or None)

    k1, k2, k3, k4, k5 = st.columns(5)
    for col, lbl, val, col_v in [
        (k1, "Total Frames",    total_frames or "—",        "#00e5ff"),
        (k2, "Peak Vehicles",   peak_veh or "—",          "#ffb400"),
        (k3, "Avg Density",     f"{avg_density_pct}%" if avg_density_pct is not None else "—",    "#7c3aed"),
        (k4, "Total Violations",all_viols_count,           "#ff3c3c"),
        (k5, "Plates Read",     total_plates or "—",      "#00e676"),
    ]:
        with col:
            st.markdown(f"""
            <div style="background:rgba(11,17,32,0.85);border:1px solid rgba(0,229,255,0.1);
                        border-radius:8px;padding:12px;text-align:center;margin-bottom:8px;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:8px;
                            letter-spacing:2px;color:#64748b;text-transform:uppercase;">{lbl}</div>
                <div style="font-family:'Orbitron',monospace;font-size:22px;font-weight:700;
                            color:{col_v};line-height:1.2;">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Row A: Vehicle count timeline + Density trend
    ra1, ra2 = st.columns(2)
    if bulk:
        with ra1:
            vc_y = [d["n_vehicles"] for d in bulk]
            vc_x = list(range(len(vc_y)))
            fig_a1 = go.Figure()
            fig_a1.add_trace(go.Scatter(
                x=vc_x, y=vc_y, name="Total",
                mode="lines", line=dict(color="#00e5ff", width=2.5),
                fill="tozeroy", fillcolor="rgba(0,229,255,0.06)",
            ))
            fig_a1.update_layout(**_dark_fig(), height=250,  # type: ignore
                                  title=dict(text="Vehicle Count Over Time",
                                             font=dict(size=12, color="#e2e8f0")))
            st.plotly_chart(fig_a1, use_container_width=True,
                             config={"displayModeBar": False}, key="vc_l3")

        with ra2:
            den_y = [d["density"]*100 for d in bulk]
            colors_d = [CONG_COLOR.get(d["congestion"], "#00e5ff") for d in bulk]
            fig_a2 = go.Figure(go.Scatter(
                x=vc_x, y=den_y, mode="lines",
                line=dict(color="#7c3aed", width=2.5),
                fill="tozeroy", fillcolor="rgba(124,58,237,0.07)",
            ))
            # Threshold lines
            for y_v, clr, lbl in [(25,"rgba(0,230,118,.5)","LOW"),
                                    (50,"rgba(255,180,0,.5)","MEDIUM"),
                                    (75,"rgba(255,136,0,.5)","HIGH")]:
                fig_a2.add_hline(y=y_v, line=dict(color=clr, dash="dot", width=1),
                                  annotation_text=lbl,
                                  annotation_font=dict(size=9, color="#64748b"))
            fig_a2.update_layout({
                                  **_dark_fig(),
                                  "height": 250,
                                  "title": dict(text="Traffic Density Trend (%)",
                                                 font=dict(size=12, color="#e2e8f0")),
                                  "yaxis": dict(
                                      showgrid=True,
                                      gridcolor="rgba(0,229,255,0.06)",
                                      zeroline=False,
                                      range=[0, 105],
                                  )
                                  })
            st.plotly_chart(fig_a2, use_container_width=True,
                             config={"displayModeBar": False}, key="den_l3")
    else:
        st.info("Upload a video to see full analytics")

    # Row B: Violation bar + Class pie + Congestion dist
    rb1, rb2, rb3 = st.columns(3)

    with rb1:
        vcount3 = Counter(v["type"] for v in st.session_state.viol_log)
        if vcount3:
            types  = list(vcount3.keys())
            values = list(vcount3.values())
            colors = ["#ff3c3c","#ff8800","#ffb400","#7c3aed","#00e5ff"]
            fig_b1 = go.Figure(go.Bar(
                x=values, y=[t.replace(" ", "<br>") for t in types],
                orientation="h",
                marker=dict(color=colors[:len(types)], opacity=0.85),
                text=values, textposition="outside",
                textfont=dict(size=10, color="#e2e8f0"),
            ))
            fig_b1.update_layout(**_dark_fig(), height=260,  # type: ignore
                                  title=dict(text="Violation Frequency",
                                             font=dict(size=12, color="#e2e8f0")))
            fig_b1.update_layout(xaxis=dict(showgrid=False),
                                  yaxis=dict(showgrid=False, tickfont=dict(size=9)))
            st.plotly_chart(fig_b1, use_container_width=True,
                             config={"displayModeBar": False}, key="vf_l3")
        else:
            no_viols_html = """
            <div style="border:1px solid rgba(0,229,255,0.1);border-radius:8px;
                        padding:24px;text-align:center;height:260px;display:flex;
                        flex-direction:column;justify-content:center;align-items:center;">
                <div style="font-size:28px;">📊</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                            color:#334155;margin-top:8px;">No violations recorded</div>
            </div>
            """
            st.markdown(no_viols_html, unsafe_allow_html=True)

    with rb2:
        labels = list(cls_totals.keys()) if cls_totals else ["car","bike","truck","bus","auto"]
        values = [cls_totals.get(l, random.randint(2, 10)) for l in labels]
        colors = [CLS_COLORS.get(l, "#64748b") for l in labels]
        fig_b2 = go.Figure(go.Pie(
            labels=[l.title() for l in labels], values=values,
            hole=0.5, marker=dict(colors=colors, line=dict(color="#04060d", width=2)),
            textfont=dict(size=9),
        ))
        fig_b2.update_layout(paper_bgcolor="rgba(0,0,0,0)", showlegend=True,
                              height=260, margin=dict(l=4, r=4, t=28, b=4),
                              title=dict(text="Class Distribution",
                                         font=dict(size=12, color="#e2e8f0")),
                              legend=dict(bgcolor="rgba(0,0,0,0)",
                                          font=dict(size=9, color="#94a3b8")),
                              font=dict(color="#94a3b8"))
        st.plotly_chart(fig_b2, use_container_width=True,
                         config={"displayModeBar": False}, key="cls_l3")

    with rb3:
        cong_dist = Counter(d["congestion"] for d in bulk) if bulk else {"LOW":5,"MEDIUM":10,"HIGH":6,"CRITICAL":2}
        cnames = ["LOW","MEDIUM","HIGH","CRITICAL"]
        cvals  = [cong_dist.get(c, 0) for c in cnames]
        cclrs  = [CONG_COLOR[c] for c in cnames]
        fig_b3 = go.Figure(go.Bar(
            x=cnames, y=cvals,
            marker=dict(color=cclrs, opacity=0.85,
                        line=dict(color=cclrs, width=0.5)),
            text=cvals, textposition="outside",
            textfont=dict(size=11, color="#e2e8f0"),
        ))
        fig_b3.update_layout(**_dark_fig(), height=260, bargap=0.3,  # type: ignore
                              title=dict(text="Congestion Distribution",
                                         font=dict(size=12, color="#e2e8f0")))
        fig_b3.update_layout(xaxis=dict(showgrid=False))
        st.plotly_chart(fig_b3, use_container_width=True,
                         config={"displayModeBar": False}, key="cng_l3")

    # Row C: Speed distribution + AI decision log + plates table
    rc1, rc2 = st.columns([1.4, 1.6])

    with rc1:
        speeds = []
        for d in bulk:
            for v in d.get("vehicles", []):
                if "speed_kmh" in v:
                    speeds.append(v["speed_kmh"])
        if not speeds:
            fd_current = st.session_state.get("frame_data", {}) or {}
            for v in fd_current.get("vehicles", []):
                if "speed_kmh" in v:
                    speeds.append(v["speed_kmh"])
        if speeds:
            fig_spd = go.Figure(go.Histogram(
                x=speeds, nbinsx=20,
                marker=dict(color="#00e5ff", opacity=0.75,
                            line=dict(color="#00e5ff", width=0.5)),
            ))
            fig_spd.update_layout(**_dark_fig(), height=240,  # type: ignore
                                   title=dict(text="Speed Distribution (km/h)",
                                              font=dict(size=12, color="#e2e8f0")),
                                   bargap=0.05)
            st.plotly_chart(fig_spd, use_container_width=True,
                             config={"displayModeBar": False}, key="spd_l3")
        else:
            st.markdown("""
            <div style="background:rgba(11,17,32,0.8);border:2px dashed rgba(0,229,255,0.2);
                        border-radius:10px;padding:40px;text-align:center;height:250px;
                        display:flex;flex-direction:column;justify-content:center;align-items:center;">
                <div style="font-size:36px;margin-bottom:8px;">📐</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#334155;">
                    Speed data appears after video analysis
                </div>
            </div>
            """, unsafe_allow_html=True)

    with rc2:
        st.markdown("""
        <div class="panel-head" style="font-size:12px;">🧠 AI Decision Log</div>
        """, unsafe_allow_html=True)
        fd3_dec = st.session_state.get("frame_data", {}) or {}
        decisions = fd3_dec.get("decisions", [
            "🟢 Traffic nominal → Standard cycle maintained",
            "📡 All cameras operational",
        ])
        for dec in decisions:
            c3 = "#ff3c3c" if "🔴" in dec else "#ffb400" if "🟡" in dec else "#00e676" if "🟢" in dec else "#00e5ff"
            st.markdown(f"""
            <div style="background:{c3}0d;border:1px solid {c3}22;border-left:3px solid {c3};
                        border-radius:6px;padding:10px 12px;margin-bottom:6px;
                        font-family:'Rajdhani',sans-serif;font-size:13px;color:{c3};">
                {dec}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="panel-head" style="font-size:12px;">🔤 Recent Plates (OCR)</div>
        """, unsafe_allow_html=True)

        all_plates = []
        for d in (bulk[-20:] if bulk else []):
            for p in d.get("plates", []):
                all_plates.append(p)
        if not all_plates:
            fd_current = st.session_state.get("frame_data", {}) or {}
            for p in fd_current.get("plates", []):
                all_plates.append(p)
        recent_plates = all_plates[-8:][::-1]
        if recent_plates:
            for p in recent_plates:
                plate_str = p.get("plate", "N/A")
                plate_conf = p.get("confidence", p.get("conf", 0))
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;
                            padding:4px 0;border-bottom:1px solid rgba(0,229,255,0.05);">
                    <span style="font-family:'Orbitron',monospace;font-size:12px;
                                 color:#ffb400;letter-spacing:1px;">{plate_str}</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:10px;
                                 color:#64748b;">conf {plate_conf:.2f}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                        color:#334155;padding:8px 0;">Plates appear after frame analysis</div>
            """, unsafe_allow_html=True)

    # Bulk-ready indicator
    if st.session_state.bulk_ready:
        st.success(f"✅ Full video analysis complete — {len(st.session_state.bulk_data)} frames sampled")
    elif st.session_state.video_path and not st.session_state.bulk_data:
        st.info("⏳ Background analysis running — charts will populate automatically")

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
