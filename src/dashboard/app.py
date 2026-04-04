"""
Trinetra — Live Dashboard
Streamlit-based real-time monitoring dashboard.

Run:
    streamlit run src/dashboard/app.py
"""

import time
import threading
import queue
from pathlib import Path
from collections import deque, defaultdict
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Trinetra — Traffic Surveillance",
    page_icon="🔱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background: #0e1117; }
    .metric-card {
        background: #1a1d27;
        border: 1px solid #2d3145;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .violation-badge {
        background: #3d1a1a;
        border-left: 4px solid #ff4444;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 4px;
        font-size: 13px;
    }
    .congestion-free     { color: #00dd77; font-weight: bold; }
    .congestion-moderate { color: #00ddcc; font-weight: bold; }
    .congestion-heavy    { color: #ffaa00; font-weight: bold; }
    .congestion-gridlock { color: #ff4444; font-weight: bold; }
    div[data-testid="metric-container"] {
        background: #1a1d27;
        border: 1px solid #2d3145;
        border-radius: 8px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────
def init_state():
    defaults = {
        'running': False,
        'frame_queue': queue.Queue(maxsize=2),
        'stats_history': deque(maxlen=300),
        'violation_log': [],
        'current_frame': None,
        'current_stats': {
            'total_vehicles': 0,
            'congestion_score': 0.0,
            'congestion_level': 'FREE',
            'avg_speed_kmh': None,
            'count_by_class': {},
            'violations_total': 0,
            'fps': 0.0,
            'frame_id': 0,
        },
        'pipeline': None,
        'video_source': None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🔱 Trinetra")
    st.markdown("**Intelligent Traffic Surveillance**")
    st.divider()

    st.subheader("Video Source")
    source_type = st.radio("Source", ["Video File", "Webcam", "RTSP Stream"])

    video_source = None
    if source_type == "Video File":
        uploaded = st.file_uploader("Upload video", type=['mp4', 'avi', 'mov', 'mkv'])
        if uploaded:
            tmp = Path("data/raw") / uploaded.name
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(uploaded.read())
            video_source = str(tmp)
            st.success(f"Loaded: {uploaded.name}")
    elif source_type == "Webcam":
        cam_idx = st.number_input("Camera index", 0, 4, 0)
        video_source = int(cam_idx)
    else:
        rtsp_url = st.text_input("RTSP URL", "rtsp://")
        video_source = rtsp_url if rtsp_url != "rtsp://" else None

    st.divider()

    st.subheader("Settings")
    show_trajectories = st.toggle("Show trajectories", True)
    show_heatmap      = st.toggle("Show density heatmap", False)
    conf_threshold    = st.slider("Detection confidence", 0.3, 0.9, 0.45, 0.05)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("▶ Start", use_container_width=True,
                               type="primary", disabled=video_source is None)
    with col2:
        stop_btn = st.button("⏹ Stop", use_container_width=True,
                              disabled=not st.session_state.running)

    if stop_btn:
        st.session_state.running = False

    if start_btn and video_source is not None:
        st.session_state.running = True
        st.session_state.video_source = video_source
        st.session_state.violation_log = []
        st.session_state.stats_history = deque(maxlen=300)


# ── Main layout ───────────────────────────────────────────────
st.title("🔱 Trinetra — Traffic Surveillance Dashboard")

tab_live, tab_analytics, tab_violations, tab_report = st.tabs([
    "📹 Live Feed", "📊 Analytics", "⚠️ Violations", "📋 Report"
])


# ── TAB 1: Live Feed ──────────────────────────────────────────
with tab_live:
    col_feed, col_stats = st.columns([3, 1])

    with col_feed:
        frame_placeholder = st.empty()
        status_placeholder = st.empty()

    with col_stats:
        st.subheader("Live Stats")
        m1 = st.empty()
        m2 = st.empty()
        m3 = st.empty()
        m4 = st.empty()
        st.divider()
        st.subheader("Class Breakdown")
        class_chart = st.empty()
        st.divider()
        st.subheader("Recent Violations")
        violation_feed = st.empty()


# ── TAB 2: Analytics ─────────────────────────────────────────
with tab_analytics:
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        congestion_chart = st.empty()
    with col_a2:
        vehicle_count_chart = st.empty()

    col_a3, col_a4 = st.columns(2)
    with col_a3:
        speed_chart = st.empty()
    with col_a4:
        class_pie = st.empty()


# ── TAB 3: Violations ────────────────────────────────────────
with tab_violations:
    vio_summary = st.empty()
    vio_table   = st.empty()


# ── TAB 4: Report ────────────────────────────────────────────
with tab_report:
    report_placeholder = st.empty()
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        dl_violations = st.empty()
    with dl_col2:
        dl_density = st.empty()


# ── Background pipeline thread ────────────────────────────────

def run_pipeline_thread(source, frame_q: queue.Queue, cfg_overrides: dict):
    """Runs Trinetra pipeline in background thread, pushing frames to queue."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        from src.pipeline import TrinetraPipeline
        from src.utils.video_io import VideoReader

        pipeline = TrinetraPipeline()
        pipeline.cfg['inference']['conf_threshold'] = cfg_overrides.get('conf', 0.45)
        st.session_state.pipeline = pipeline

        with VideoReader(source) as reader:
            fps_counter = deque(maxlen=30)
            t_prev = time.time()

            for frame_id, frame in reader:
                if not st.session_state.running:
                    break

                t_now = time.time()
                fps_counter.append(1.0 / max(t_now - t_prev, 0.001))
                t_prev = t_now

                annotated = pipeline.process_frame(frame, frame_id)
                snap = pipeline._last_density_snap

                # Build stats dict
                stats = {
                    'total_vehicles': snap.total_vehicles,
                    'congestion_score': snap.congestion_score,
                    'congestion_level': snap.congestion_level,
                    'avg_speed_kmh': snap.avg_speed_kmh,
                    'count_by_class': snap.count_by_class,
                    'violations_total': len(pipeline.violation.events),
                    'fps': round(float(np.mean(fps_counter)), 1),
                    'frame_id': frame_id,
                    'timestamp': t_now,
                }

                # Add new violations to log
                recent_viols = [
                    e for e in pipeline.violation.events
                    if t_now - e.timestamp < 1.5
                ]
                for ev in recent_viols:
                    if ev not in st.session_state.violation_log:
                        st.session_state.violation_log.append(ev)

                st.session_state.stats_history.append(stats)
                st.session_state.current_stats = stats

                # Encode frame for display
                _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not frame_q.full():
                    frame_q.put_nowait(buf.tobytes())

    except Exception as e:
        st.session_state.running = False
        st.error(f"Pipeline error: {e}")


# ── Start thread ──────────────────────────────────────────────
if st.session_state.running and st.session_state.video_source is not None:
    if 'pipeline_thread' not in st.session_state or \
       not st.session_state.get('pipeline_thread', None) or \
       not st.session_state.pipeline_thread.is_alive():
        t = threading.Thread(
            target=run_pipeline_thread,
            args=(
                st.session_state.video_source,
                st.session_state.frame_queue,
                {'conf': conf_threshold},
            ),
            daemon=True,
        )
        t.start()
        st.session_state.pipeline_thread = t


# ── UI update loop ────────────────────────────────────────────
def update_ui():
    stats = st.session_state.current_stats
    history = list(st.session_state.stats_history)

    # ── Live feed ──
    try:
        frame_bytes = st.session_state.frame_queue.get_nowait()
        frame_placeholder.image(frame_bytes, channels='BGR', use_column_width=True)
    except queue.Empty:
        if not st.session_state.running:
            frame_placeholder.info("No video source running. Select a source and press Start.")

    # ── Metrics ──
    level = stats['congestion_level']
    color_map = {'FREE': 'normal', 'MODERATE': 'normal', 'HEAVY': 'off', 'GRIDLOCK': 'inverse'}
    m1.metric("Vehicles", stats['total_vehicles'])
    m2.metric("Congestion", f"{level} ({stats['congestion_score']:.2f})")
    m3.metric("Violations", stats['violations_total'])
    speed_val = f"{stats['avg_speed_kmh']:.1f}" if stats['avg_speed_kmh'] else "N/A"
    m4.metric("Avg Speed", f"{speed_val} km/h")

    # ── Class bar chart ──
    cls = stats['count_by_class']
    if cls:
        fig = go.Figure(go.Bar(
            x=list(cls.values()), y=list(cls.keys()),
            orientation='h',
            marker_color=['#FFc800', '#00c8FF', '#00FF64', '#FF5000', '#B400FF', '#dcdcdc'],
        ))
        fig.update_layout(
            height=200, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='white', xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
        )
        class_chart.plotly_chart(fig, use_container_width=True, key='class_bar')

    # ── Violation feed ──
    recent_viols = st.session_state.violation_log[-5:][::-1]
    if recent_viols:
        html = ""
        for ev in recent_viols:
            t_str = datetime.fromtimestamp(ev.timestamp).strftime('%H:%M:%S')
            html += f'<div class="violation-badge">⚠ {ev.violation_type.replace("_"," ").upper()}<br><small>{ev.plate_number} — {t_str}</small></div>'
        violation_feed.markdown(html, unsafe_allow_html=True)
    else:
        violation_feed.info("No violations detected")

    # ── Analytics charts ──
    if history:
        df = pd.DataFrame(history)
        df['time'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%H:%M:%S')

        # Congestion timeline
        fig_cong = go.Figure()
        fig_cong.add_trace(go.Scatter(
            x=df['time'], y=df['congestion_score'],
            fill='tozeroy', line=dict(color='#ff6644', width=2),
            name='Congestion Score',
        ))
        fig_cong.update_layout(
            title='Congestion Score', height=220,
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,39,1)',
            font_color='white', yaxis=dict(range=[0, 1]),
            showlegend=False,
        )
        congestion_chart.plotly_chart(fig_cong, use_container_width=True, key='cong')

        # Vehicle count
        fig_count = go.Figure()
        fig_count.add_trace(go.Scatter(
            x=df['time'], y=df['total_vehicles'],
            fill='tozeroy', line=dict(color='#44aaff', width=2),
            name='Vehicle Count',
        ))
        fig_count.update_layout(
            title='Vehicle Count', height=220,
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,39,1)',
            font_color='white', showlegend=False,
        )
        vehicle_count_chart.plotly_chart(fig_count, use_container_width=True, key='vcount')

        # Speed
        speed_data = df['avg_speed_kmh'].dropna()
        if not speed_data.empty:
            fig_spd = go.Figure()
            fig_spd.add_trace(go.Scatter(
                x=df['time'].iloc[speed_data.index], y=speed_data,
                line=dict(color='#44ffaa', width=2), name='Speed km/h',
            ))
            fig_spd.update_layout(
                title='Avg Speed (km/h)', height=220,
                margin=dict(l=0, r=0, t=30, b=0),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,29,39,1)',
                font_color='white', showlegend=False,
            )
            speed_chart.plotly_chart(fig_spd, use_container_width=True, key='spd')

        # Class pie
        last_cls = history[-1].get('count_by_class', {})
        if last_cls:
            fig_pie = go.Figure(go.Pie(
                labels=list(last_cls.keys()),
                values=list(last_cls.values()),
                hole=0.4,
                marker_colors=['#FFc800', '#00c8FF', '#00FF64', '#FF5000', '#B400FF', '#dcdcdc'],
            ))
            fig_pie.update_layout(
                title='Vehicle Mix', height=220,
                margin=dict(l=0, r=0, t=30, b=0),
                paper_bgcolor='rgba(0,0,0,0)', font_color='white',
            )
            class_pie.plotly_chart(fig_pie, use_container_width=True, key='pie')

    # ── Violations table ──
    vlog = st.session_state.violation_log
    if vlog:
        rows = [ev.to_dict() for ev in vlog]
        df_v = pd.DataFrame(rows)
        df_v['timestamp'] = pd.to_datetime(df_v['timestamp'], unit='s').dt.strftime('%H:%M:%S')

        # Summary
        from collections import Counter
        counts = Counter(ev.violation_type for ev in vlog)
        cols = vio_summary.columns(len(counts))
        for i, (vtype, cnt) in enumerate(counts.items()):
            with cols[i]:
                st.metric(vtype.replace('_', ' ').title(), cnt)

        vio_table.dataframe(
            df_v[['event_id', 'violation_type', 'plate_number', 'timestamp', 'confidence']],
            use_container_width=True, height=400,
        )

        # Download
        dl_violations.download_button(
            "⬇ Download Violations CSV",
            df_v.to_csv(index=False).encode(),
            "violations.csv", "text/csv",
            use_container_width=True,
        )
    else:
        vio_table.info("No violations recorded yet.")

    # ── Report tab ──
    if history:
        total_frames = history[-1]['frame_id'] if history else 0
        avg_score    = float(np.mean([h['congestion_score'] for h in history]))
        avg_count    = float(np.mean([h['total_vehicles'] for h in history]))

        report_placeholder.markdown(f"""
### Session Report
| Metric | Value |
|--------|-------|
| Frames processed | {total_frames} |
| Total violations | {len(vlog)} |
| Avg vehicle count | {avg_count:.1f} |
| Avg congestion | {avg_score:.3f} |
| Session start | {datetime.fromtimestamp(history[0]['timestamp']).strftime('%Y-%m-%d %H:%M:%S') if history else '—'} |
        """)

        if history:
            df_density = pd.DataFrame([{
                'frame_id': h['frame_id'],
                'timestamp': datetime.fromtimestamp(h['timestamp']).strftime('%H:%M:%S'),
                'total_vehicles': h['total_vehicles'],
                'congestion_score': h['congestion_score'],
                'congestion_level': h['congestion_level'],
                'avg_speed_kmh': h['avg_speed_kmh'],
            } for h in history])
            dl_density.download_button(
                "⬇ Download Density CSV",
                df_density.to_csv(index=False).encode(),
                "density.csv", "text/csv",
                use_container_width=True,
            )


# ── Auto-refresh ──────────────────────────────────────────────
if st.session_state.running:
    update_ui()
    time.sleep(0.05)
    st.rerun()
else:
    update_ui()
    status_placeholder.info("Pipeline stopped. Select a source and press ▶ Start to begin.")
