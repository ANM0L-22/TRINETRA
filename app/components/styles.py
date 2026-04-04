"""
Trinetra — Global Streamlit CSS
Injected via st.markdown(get_global_css(), unsafe_allow_html=True)
"""

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600&family=JetBrains+Mono:wght@300;400&display=swap');

/* ── Root vars ── */
:root {
  --bg:         #04060d;
  --surface:    #0b1120;
  --panel:      #0f1628;
  --border:     rgba(0,229,255,0.12);
  --cyan:       #00e5ff;
  --cyan-dim:   rgba(0,229,255,0.15);
  --purple:     #7c3aed;
  --purple-dim: rgba(124,58,237,0.15);
  --gold:       #ffb400;
  --red:        #ff3c3c;
  --red-dim:    rgba(255,60,60,0.15);
  --green:      #00e676;
  --yellow:     #ffea00;
  --text:       #e2e8f0;
  --muted:      #64748b;
}

/* ── Global reset ── */
html, body, [class*="css"] {
  font-family: 'Rajdhani', sans-serif !important;
  background-color: var(--bg) !important;
  color: var(--text) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0.5rem 1.5rem 2rem !important; max-width: 100% !important; }
[data-testid="stAppViewContainer"] { background: var(--bg) !important; }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #060b16 0%, #0a1225 100%) !important;
  border-right: 1px solid var(--border) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] .block-container { padding: 1rem !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  padding: 12px !important;
}
[data-testid="metric-container"] label {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 10px !important;
  letter-spacing: 2px !important;
  color: var(--muted) !important;
  text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  font-family: 'Orbitron', monospace !important;
  font-size: 24px !important;
  font-weight: 700 !important;
  color: var(--cyan) !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 11px !important;
}

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, rgba(0,229,255,0.1), rgba(124,58,237,0.1)) !important;
  color: var(--cyan) !important;
  border: 1px solid rgba(0,229,255,0.3) !important;
  border-radius: 6px !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 11px !important;
  letter-spacing: 2px !important;
  text-transform: uppercase !important;
  transition: all 0.2s !important;
}
.stButton > button:hover {
  background: rgba(0,229,255,0.15) !important;
  border-color: var(--cyan) !important;
  box-shadow: 0 0 20px rgba(0,229,255,0.3) !important;
}

/* ── Selectbox, inputs ── */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 6px !important;
  font-family: 'Rajdhani', sans-serif !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}
.dvn-scroller { background: transparent !important; }
[data-testid="stDataFrame"] table { background: transparent !important; }
[data-testid="stDataFrame"] tbody tr { background: transparent !important; }
[data-testid="stDataFrame"] thead th { background: rgba(11,17,32,0.6) !important; }

/* ── Progress bar ── */
.stProgress > div > div > div {
  background: linear-gradient(90deg, var(--cyan), var(--purple)) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--surface) !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 11px !important;
  letter-spacing: 2px !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
  background: transparent !important;
  border: none !important;
  padding: 10px 20px !important;
}
.stTabs [aria-selected="true"] {
  color: var(--cyan) !important;
  border-bottom: 2px solid var(--cyan) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: rgba(0,229,255,0.3); border-radius: 2px; }

/* ── Glass card ── */
.glass-card {
  background: rgba(11,17,32,0.8);
  border: 1px solid rgba(0,229,255,0.12);
  border-radius: 10px;
  padding: 16px;
  backdrop-filter: blur(10px);
}

/* ── Violation badge ── */
.vbadge-critical { background:#ff3c3c22; border-left:3px solid #ff3c3c; padding:4px 8px; border-radius:4px; margin:2px 0; font-size:12px; }
.vbadge-high     { background:#ff880022; border-left:3px solid #ff8800; padding:4px 8px; border-radius:4px; margin:2px 0; font-size:12px; }
.vbadge-medium   { background:#ffea0022; border-left:3px solid #ffea00; padding:4px 8px; border-radius:4px; margin:2px 0; font-size:12px; }

/* ── Status badges ── */
.status-online { color:#00e676; font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:2px; }
.status-offline{ color:#ff3c3c; font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:2px; }

/* ── Divider ── */
hr { border: none; border-top: 1px solid var(--border) !important; margin: 10px 0 !important; }

/* ── Animation keyframes ── */
@keyframes blink-red {
  0%,100% { opacity:1; box-shadow: 0 0 8px #ff3c3c; }
  50%     { opacity:0.5; box-shadow: none; }
}
@keyframes pulse-cyan {
  0%,100% { box-shadow: 0 0 0 0 rgba(0,229,255,0.4); }
  50%     { box-shadow: 0 0 0 8px rgba(0,229,255,0); }
}
.blink-dot {
  display:inline-block; width:8px; height:8px; border-radius:50%;
  background:#ff3c3c; animation:blink-red 1s infinite;
}
.active-dot {
  display:inline-block; width:8px; height:8px; border-radius:50%;
  background:#00e676; animation:pulse-cyan 2s infinite;
}
</style>
"""

def get_global_css():
    return GLOBAL_CSS


# ── Component helpers ─────────────────────────────────────────

def card(content_html: str, border_color: str = "rgba(0,229,255,0.15)") -> str:
    return f"""
    <div style="background:rgba(11,17,32,0.85);border:1px solid {border_color};
                border-radius:10px;padding:16px;backdrop-filter:blur(8px);margin-bottom:8px;">
        {content_html}
    </div>"""


def badge(text: str, color: str = "#00e5ff") -> str:
    return f"""<span style="background:{color}22;color:{color};border:1px solid {color}44;
               border-radius:4px;padding:2px 8px;font-family:'JetBrains Mono',monospace;
               font-size:10px;letter-spacing:1px;">{text}</span>"""


def metric_chip(label: str, value: str, color: str = "#00e5ff") -> str:
    return f"""
    <div style="background:rgba(11,17,32,0.9);border:1px solid {color}33;border-radius:8px;
                padding:12px 16px;text-align:center;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:2px;
                    color:#64748b;text-transform:uppercase;margin-bottom:4px;">{label}</div>
        <div style="font-family:'Orbitron',monospace;font-size:22px;font-weight:700;
                    color:{color};line-height:1;">{value}</div>
    </div>"""


def section_header(title: str, subtitle: str = "") -> str:
    return f"""
    <div style="margin-bottom:20px;">
        <div style="font-family:'Orbitron',monospace;font-size:18px;font-weight:700;
                    color:#e2e8f0;letter-spacing:1px;">{title}</div>
        {"<div style='font-family:JetBrains Mono,monospace;font-size:10px;letter-spacing:3px;color:#64748b;text-transform:uppercase;margin-top:4px;'>"+subtitle+"</div>" if subtitle else ""}
        <div style="width:40px;height:2px;background:linear-gradient(90deg,#00e5ff,#7c3aed);margin-top:8px;"></div>
    </div>"""


def alert_card(vtype: str, plate: str, ts: str, severity: str = "critical") -> str:
    colors = {"critical": "#ff3c3c", "high": "#ff8800", "medium": "#ffea00"}
    c = colors.get(severity, "#ff3c3c")
    icons = {
        "no_helmet":      "🪖",
        "no_seatbelt":    "🔒",
        "phone_use":      "📱",
        "wrong_side":     "↩️",
        "tampered_plate": "🚫",
    }
    icon = icons.get(vtype, "⚠️")
    label = vtype.replace("_", " ").upper()
    return f"""
    <div style="background:{c}15;border:1px solid {c}44;border-left:3px solid {c};
                border-radius:6px;padding:10px 12px;margin-bottom:6px;
                animation:blink-red 2s infinite;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="font-family:'Orbitron',monospace;font-size:11px;color:{c};
                         font-weight:700;">{icon} {label}</span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:9px;
                         color:#64748b;">{ts}</span>
        </div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:11px;
                    color:#94a3b8;margin-top:4px;">Plate: <span style="color:#e2e8f0;
                    font-weight:600;">{plate}</span></div>
    </div>"""


def pipeline_strip() -> str:
    steps = [
        ("📹", "VIDEO"),
        ("🔍", "DETECT"),
        ("🎯", "TRACK"),
        ("🔤", "OCR"),
        ("⚠️", "VIOLATE"),
        ("📊", "ANALYZE"),
        ("📡", "DASHBOARD"),
    ]
    items = ""
    for i, (icon, label) in enumerate(steps):
        arrow = "<span style='color:rgba(0,229,255,0.4);font-size:14px;'>→</span>" if i < len(steps)-1 else ""
        items += f'<div style="display:flex;align-items:center;gap:6px;"><div style="background:rgba(0,229,255,0.1);border:1px solid rgba(0,229,255,0.25);border-radius:6px;padding:6px 12px;text-align:center;min-width:80px;"><div style="font-size:14px;">{icon}</div><div style="font-family:\'JetBrains Mono\',monospace;font-size:8px;letter-spacing:1px;color:#00e5ff;margin-top:2px;">{label}</div></div>{arrow}</div>'
    
    html = '<div style="background:rgba(6,11,22,0.9);border:1px solid rgba(0,229,255,0.12);border-radius:8px;padding:10px 16px;margin-top:8px;">'
    html += '<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;letter-spacing:3px;color:#64748b;margin-bottom:8px;">AI PIPELINE</div>'
    html += '<div style="display:flex;align-items:center;gap:4px;overflow-x:auto;padding-bottom:4px;">'
    html += items
    html += '</div></div>'
    return html


def congestion_bar(score: float, level: str) -> str:
    colors = {"FREE": "#00e676", "MODERATE": "#ffea00", "HEAVY": "#ff8800", "GRIDLOCK": "#ff3c3c"}
    c      = colors.get(level, "#00e5ff")
    pct    = int(score * 100)
    return f"""
    <div style="background:rgba(11,17,32,0.9);border:1px solid rgba(0,229,255,0.12);
                border-radius:10px;padding:14px 16px;">
        <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="font-family:'JetBrains Mono',monospace;font-size:9px;
                         letter-spacing:3px;color:#64748b;text-transform:uppercase;">
                Traffic Density</span>
            <span style="font-family:'Orbitron',monospace;font-size:12px;
                         font-weight:700;color:{c};">{level}</span>
        </div>
        <div style="background:rgba(255,255,255,0.06);border-radius:4px;height:8px;
                    overflow:hidden;margin-bottom:8px;">
            <div style="width:{pct}%;height:100%;background:linear-gradient(90deg,{c}88,{c});
                        border-radius:4px;transition:width 0.5s ease;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;">
            <span style="font-family:'Orbitron',monospace;font-size:20px;font-weight:700;
                         color:{c};">{pct}<span style="font-size:11px;">%</span></span>
            <div style="display:flex;gap:6px;align-items:center;">
                {"".join([f'<div style="width:10px;height:10px;border-radius:2px;background:{{"FREE":"#00e67622","MODERATE":"#ffea0022","HEAVY":"#ff880022","GRIDLOCK":"#ff3c3c44"}}.get(l,"")+";border:1px solid {{"FREE":"#00e676","MODERATE":"#ffea00","HEAVY":"#ff8800","GRIDLOCK":"#ff3c3c"}}.get(l,"");"></div>' for l in ["FREE","MODERATE","HEAVY","GRIDLOCK"]])}
            </div>
        </div>
    </div>"""
