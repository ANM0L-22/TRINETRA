"""
Trinetra — Analytics Page (video-driven)
All charts update based on video analysis data from session state.
"""

import random
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from collections import Counter

from app.data.simulator import get_historical_data, get_violation_history
from app.components.styles import section_header
from app.components.charts import (violation_frequency_bar, hourly_violations,
                                    density_trend_area, speed_histogram,
                                    vehicle_count_line, class_distribution_pie)

CONG_COLOR = {"LOW":"#00e676","MEDIUM":"#ffb400","HIGH":"#ff8800","CRITICAL":"#ff3c3c"}
CLS_COLORS = {"car":"#00e5ff","bus":"#ffb400","bike":"#7c3aed",
              "truck":"#ff8800","auto":"#00e676","pedestrian":"#94a3b8"}


def _dark_fig():
    return dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="JetBrains Mono,monospace",color="#94a3b8",size=10),
                margin=dict(l=4,r=4,t=32,b=4),showlegend=False,
                xaxis=dict(showgrid=True,gridcolor="rgba(0,229,255,0.06)",zeroline=False),
                yaxis=dict(showgrid=True,gridcolor="rgba(0,229,255,0.06)",zeroline=False))


def _init():
    if "analytics_history" not in st.session_state:
        st.session_state.analytics_history = get_historical_data(hours=8)
    if "analytics_violations" not in st.session_state:
        st.session_state.analytics_violations = get_violation_history(120)


def analytics_page():
    _init()
    st.markdown(section_header("ANALYTICS","Traffic Intelligence & Insights"),unsafe_allow_html=True)

    # Pull live video data if available
    bulk       = st.session_state.get("bulk_data", [])
    viol_log   = st.session_state.get("viol_log", [])
    frame_data = st.session_state.get("frame_data", {}) or {}
    has_video  = bool(bulk)

    # ── Refresh controls ──────────────────────────────────────
    rc1, rc2, rc3 = st.columns([2.2,1.1,4])
    with rc1:
        time_range = st.selectbox("Time Range",
            ["From Video","Last 1 Hour","Last 2 Hours","Last 6 Hours"],
            index=0, key="anal_range")
    with rc2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("🔄 Refresh", use_container_width=True):
            st.session_state.analytics_history    = get_historical_data(hours=8)
            st.session_state.analytics_violations = get_violation_history(120)
            st.rerun()
    with rc3:
        if has_video:
            st.success(f"📊 Showing live data from {len(bulk)} analysed frames")
        else:
            st.info("Upload a video in Dashboard to see live analytics")

    # Use video data if available, else fall back to simulated history
    if has_video and time_range == "From Video":
        history    = bulk
        violations = viol_log
        # Adapt bulk format to history format
        h = [{"time": str(i), "vehicles": d["n_vehicles"],
               "density": d["density"], "speed_avg": 35.0,
               "congestion": d["congestion"],
               "violations": len(d.get("violations",[]))} for i, d in enumerate(bulk)]
    else:
        limits = {"Last 1 Hour":60,"Last 2 Hours":120,"Last 6 Hours":360,"From Video":360}
        lim    = limits.get(time_range, 360)
        h      = st.session_state.analytics_history[-lim:]
        violations = st.session_state.analytics_violations

    df = pd.DataFrame(h)

    # ── KPI row ───────────────────────────────────────────────
    k1,k2,k3,k4,k5 = st.columns(5)
    with k1: st.metric("Peak Vehicles",  int(df["vehicles"].max()) if not df.empty else "—")
    with k2: st.metric("Avg Vehicles",   round(df["vehicles"].mean(),1) if not df.empty else "—")
    with k3: st.metric("Max Density",    f"{int(df['density'].max()*100)}%" if not df.empty else "—")
    with k4: st.metric("Total Violations", len(violations))
    with k5:
        avs = df["speed_avg"].mean() if "speed_avg" in df.columns and not df.empty else None
        st.metric("Avg Speed", f"{avs:.1f} km/h" if avs else "—")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Row 1: Vehicle count timeline + Class pie ─────────────
    r1,r2 = st.columns([2,1])
    with r1:
        if not df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.get("time", list(range(len(df)))),
                y=df["vehicles"], mode="lines",
                line=dict(color="#00e5ff",width=2.5),
                fill="tozeroy", fillcolor="rgba(0,229,255,0.06)", name="Vehicles",
            ))
            fig.update_layout(**_dark_fig(), height=280,  # type: ignore
                title=dict(text="Vehicle Count Over Time",font=dict(size=12,color="#e2e8f0")))
            st.plotly_chart(fig, use_container_width=True,
                             config={"displayModeBar":False}, key="vc_anal")
        else:
            st.info("No data")

    with r2:
        # Class distribution from video or fallback
        cls_totals = {}
        for d in bulk:
            for k,v in d.get("counts",{}).items():
                cls_totals[k] = cls_totals.get(k,0) + v
        if not cls_totals:
            cls_totals = frame_data.get("counts",{}) or {"car":8,"bike":6,"bus":2,"truck":3,"auto":4}
        labels = list(cls_totals.keys())
        values = list(cls_totals.values())
        colors = [CLS_COLORS.get(l,"#64748b") for l in labels]
        fig_pie = go.Figure(go.Pie(
            labels=[l.title() for l in labels], values=values, hole=0.5,
            marker=dict(colors=colors,line=dict(color="#04060d",width=2)),
            textfont=dict(size=9),
        ))
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)",showlegend=True,height=280,
            margin=dict(l=4,r=4,t=28,b=4),
            title=dict(text="Class Distribution",font=dict(size=12,color="#e2e8f0")),
            legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=9,color="#94a3b8")),
            font=dict(color="#94a3b8"))
        st.plotly_chart(fig_pie, use_container_width=True,
                         config={"displayModeBar":False}, key="pie_anal")

    # ── Row 2: Density trend + Violation frequency ────────────
    r3,r4 = st.columns([2,1])
    with r3:
        if not df.empty:
            den_vals = df["density"] * 100
            fig = go.Figure(go.Scatter(
                x=df.get("time", list(range(len(df)))), y=den_vals, mode="lines",
                line=dict(color="#7c3aed",width=2.5),
                fill="tozeroy",fillcolor="rgba(124,58,237,0.07)",
            ))
            for yv,lbl in [(25,"LOW"),(50,"MEDIUM"),(75,"HIGH")]:
                fig.add_hline(y=yv,line=dict(color="rgba(255,255,255,0.15)",dash="dot",width=1),
                               annotation_text=lbl,annotation_font=dict(size=9,color="#64748b"))
            layout_cfg = {**_dark_fig(), 'height': 250, 'title': dict(text="Traffic Density Trend (%)",font=dict(size=12,color="#e2e8f0")), 'yaxis': dict(range=[0,105])}
            fig.update_layout(layout_cfg)  # type: ignore
            st.plotly_chart(fig, use_container_width=True,
                             config={"displayModeBar":False}, key="den_anal")
        else:
            st.info("No density data")

    with r4:
        vcount = Counter(v["type"] if isinstance(v,dict) and "type" in v
                          else v.get("type","Unknown") for v in violations)
        if vcount:
            types  = list(vcount.keys())
            values = list(vcount.values())
            clrs   = ["#ff3c3c","#ff8800","#ffb400","#7c3aed","#00e5ff"][:len(types)]
            fig = go.Figure(go.Bar(
                x=values, y=[t.replace(" ","<br>") for t in types],
                orientation="h",
                marker=dict(color=clrs,opacity=0.85),
                text=values, textposition="outside",
                textfont=dict(size=11,color="#e2e8f0"),
            ))
            layout_cfg = {**_dark_fig(), 'height': 250, 'title': dict(text="Violations",font=dict(size=12,color="#e2e8f0")), 'xaxis': dict(showgrid=False), 'yaxis': dict(showgrid=False,tickfont=dict(size=9))}
            fig.update_layout(layout_cfg)  # type: ignore
            st.plotly_chart(fig, use_container_width=True,
                             config={"displayModeBar":False}, key="vf_anal")
        else:
            st.info("No violations")

    # ── Row 3: Speed + Congestion dist ───────────────────────
    r5,r6 = st.columns(2)
    with r5:
        speeds = []
        for d in bulk:
            for v in d.get("vehicles",[]):
                if "speed_kmh" in v:
                    speeds.append(v["speed_kmh"])
        if speeds:
            fig = go.Figure(go.Histogram(
                x=speeds, nbinsx=20,
                marker=dict(color="#00e5ff",opacity=0.75,
                            line=dict(color="rgba(0,229,255,0.27)",width=0.5)),
            ))
            fig.update_layout(**_dark_fig(), height=240,  # type: ignore
                title=dict(text="Speed Distribution (km/h)",font=dict(size=12,color="#e2e8f0")),
                bargap=0.05)
            st.plotly_chart(fig, use_container_width=True,
                             config={"displayModeBar":False}, key="spd_anal")
        elif "speed_avg" in df.columns and not df.empty:
            fig = go.Figure(go.Histogram(
                x=df["speed_avg"].tolist(), nbinsx=15,
                marker=dict(color="#00e5ff",opacity=0.75),
            ))
            fig.update_layout(**_dark_fig(), height=240,  # type: ignore
                title=dict(text="Speed Distribution",font=dict(size=12,color="#e2e8f0")))
            st.plotly_chart(fig, use_container_width=True,
                             config={"displayModeBar":False}, key="spd_anal2")
        else:
            st.info("Speed data from video analysis")

    with r6:
        cong_src = [d["congestion"] for d in bulk] if bulk else (df["congestion"].tolist() if "congestion" in df.columns else [])
        cong_dist = Counter(cong_src) if cong_src else {"LOW":10,"MEDIUM":20,"HIGH":10,"CRITICAL":3}
        cnames = ["LOW","MEDIUM","HIGH","CRITICAL"]
        cvals  = [cong_dist.get(c,0) for c in cnames]
        cclrs  = [CONG_COLOR[c] for c in cnames]
        fig = go.Figure(go.Bar(
            x=cnames, y=cvals,
            marker=dict(color=cclrs,opacity=0.85,line=dict(color=cclrs,width=0.5)),
            text=cvals, textposition="outside",
            textfont=dict(size=11,color="#e2e8f0"),
        ))
        layout_cfg = {**_dark_fig(), 'height': 240, 'bargap': 0.3, 'title': dict(text="Congestion Distribution",font=dict(size=12,color="#e2e8f0")), 'xaxis': dict(showgrid=False)}
        fig.update_layout(layout_cfg)  # type: ignore
        st.plotly_chart(fig, use_container_width=True,
                         config={"displayModeBar":False}, key="cng_anal")

    # ── Intersection table ─────────────────────────────────────
    st.markdown("<hr>",unsafe_allow_html=True)
    st.markdown("""<div style="font-family:'JetBrains Mono',monospace;font-size:10px;
    letter-spacing:3px;color:#64748b;text-transform:uppercase;margin-bottom:8px;">
    Intersection Comparison</div>""", unsafe_allow_html=True)

    locations = ["NH-48 · Sector 7","MG Road · Junction 3","Ring Road · Gate 12",
                 "Outer Ring · Node 6","City Center · Hub 1"]
    idata = {
        "Intersection": locations,
        "Vehicles":     [random.randint(8,22) for _ in locations],
        "Violations":   [random.randint(0,8)  for _ in locations],
        "Avg Speed":    [round(random.uniform(18,55),1) for _ in locations],
        "Density":      [f"{random.randint(20,90)}%" for _ in locations],
    }
    st.dataframe(pd.DataFrame(idata), use_container_width=True, hide_index=True)
