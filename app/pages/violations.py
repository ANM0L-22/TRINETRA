"""
Trinetra — Violations Page (video-driven)
Shows violations from actual video analysis + historical log.
"""
import streamlit as st
import pandas as pd
from datetime import datetime
from collections import Counter

from app.data.simulator import get_violation_history
from app.components.styles import section_header

VIOL_ICONS = {"Helmet Violation":"🪖","Wrong-side Driving":"↩️",
              "Mobile Usage":"📱","Tampered Plate":"🚫","No Seatbelt":"🔒"}
SEV_MAP    = {"Helmet Violation":"high","Wrong-side Driving":"critical",
              "Mobile Usage":"medium","Tampered Plate":"critical","No Seatbelt":"high"}


def _init():
    if "viol_history" not in st.session_state:
        st.session_state.viol_history = get_violation_history(80)


def violations_page():
    _init()
    st.markdown(section_header("VIOLATIONS","Enforcement Log & Analytics"), unsafe_allow_html=True)

    # Merge live video violations with historical
    live_viols = st.session_state.get("viol_log", [])
    hist       = list(st.session_state.viol_history)

    # Convert live viol format to table format
    live_rows = []
    for i, v in enumerate(live_viols):
        live_rows.append({
            "id":         f"TRN-LIVE-{i+1:03d}",
            "type":       v.get("type","Unknown"),
            "plate":      v.get("plate","UNKNOWN"),
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "location":   st.session_state.get("selected_cam","NH-48 · Sector 7"),
            "confidence": v.get("conf", 0.8),
            "severity":   SEV_MAP.get(v.get("type",""),"medium"),
            "fine_inr":   1000,
            "status":     "Pending",
        })

    all_records = live_rows + hist

    # ── Summary metrics ───────────────────────────────────────
    vtype_counts = Counter(r["type"] for r in all_records)
    m1,m2,m3,m4,m5 = st.columns(5)
    with m1: st.metric("Total Violations",  len(all_records))
    with m2: st.metric("From Video",        len(live_rows), delta="Live" if live_rows else None)
    with m3: st.metric("No Helmet",         vtype_counts.get("Helmet Violation",0))
    with m4: st.metric("Wrong Side",        vtype_counts.get("Wrong-side Driving",0), delta="⚠️ Critical" if vtype_counts.get("Wrong-side Driving",0)>0 else None)
    with m5: st.metric("Tampered Plate",    vtype_counts.get("Tampered Plate",0))

    if live_rows:
        st.success(f"🎯 {len(live_rows)} violations detected from video analysis")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Filters ───────────────────────────────────────────────
    f1,f2,f3,f4 = st.columns([2,2,2,1])
    with f1:
        vtype_filter = st.multiselect("Violation Type",
            list(VIOL_ICONS.keys()), default=[], key="vtype_f")
    with f2:
        sev_filter = st.multiselect("Severity",
            ["critical","high","medium"], default=[], key="sev_f")
    with f3:
        src_filter = st.multiselect("Source",
            ["Live Video","Historical"], default=[], key="src_f")
    with f4:
        plate_q = st.text_input("Plate Search", placeholder="MH12", key="plate_q")

    df = pd.DataFrame(all_records)
    if df.empty:
        st.info("No violations recorded yet."); return

    df["source"] = ["Live Video"] * len(live_rows) + ["Historical"] * len(hist)
    if vtype_filter: df = df[df["type"].isin(vtype_filter)]
    if sev_filter:   df = df[df["severity"].isin(sev_filter)]
    if src_filter:   df = df[df["source"].isin(src_filter)]
    if plate_q:      df = df[df["plate"].str.contains(plate_q.upper(), na=False)]

    st.markdown(f"""
    <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                letter-spacing:2px;color:#64748b;margin:8px 0;">
        SHOWING {len(df)} RECORDS
    </div>""", unsafe_allow_html=True)

    if df.empty:
        st.info("No violations match filters."); return

    disp = df[["id","type","plate","timestamp","location","confidence","severity","fine_inr","status","source"]].copy()
    disp.columns = ["ID","Type","Plate","Timestamp","Location","Confidence","Severity","Fine (₹)","Status","Source"]
    disp["Confidence"] = disp["Confidence"].map("{:.2f}".format)

    def _style(row):
        s = row["Severity"]
        if s == "critical": return ["background-color:rgba(255,60,60,0.08)"] * len(row)
        elif s == "high":   return ["background-color:rgba(255,136,0,0.06)"] * len(row)
        else:               return ["background-color:rgba(255,234,0,0.04)"] * len(row)

    st.dataframe(disp.style.apply(_style, axis=1), width='stretch', height=440)

    ex1, ex2, _ = st.columns([1,1,3])
    with ex1:
        st.download_button("⬇ Export CSV", df.to_csv(index=False).encode(),
                            "violations.csv", "text/csv", width='stretch')
    with ex2:
        if st.button("🔄 Refresh Historical", width='stretch'):
            st.session_state.viol_history = get_violation_history(80)
            st.rerun()

    # ── Critical spotlight ────────────────────────────────────
    crits = df[df["Severity"]=="critical"] if "Severity" in df.columns else pd.DataFrame()
    if not crits.empty:
        st.markdown("<hr>",unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'Orbitron',monospace;font-size:13px;font-weight:700;
                    color:#ff3c3c;letter-spacing:2px;margin-bottom:10px;">
            🚨 CRITICAL VIOLATIONS
        </div>""", unsafe_allow_html=True)
        cols = st.columns(min(len(crits),4))
        for i, (_, row) in enumerate(crits.head(4).iterrows()):
            with cols[i]:
                icon = VIOL_ICONS.get(row.get("Type",""),"⚠️")
                st.markdown(f"""
                <div style="background:rgba(255,60,60,0.08);border:1px solid rgba(255,60,60,0.3);
                            border-top:3px solid #ff3c3c;border-radius:8px;padding:14px;">
                    <div style="font-family:'Orbitron',monospace;font-size:10px;color:#ff3c3c;
                                letter-spacing:2px;margin-bottom:6px;">{icon} {row.get('Type','')}</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:16px;
                                font-weight:700;color:#e2e8f0;margin-bottom:4px;">{row.get('Plate','')}</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                                color:#64748b;">{row.get('Timestamp','')}</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                                color:#64748b;margin-top:2px;">{row.get('Source','')}</div>
                </div>
                """, unsafe_allow_html=True)
