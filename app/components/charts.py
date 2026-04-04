"""
Trinetra — Plotly Chart Builders
All charts use the dark futuristic theme.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from collections import Counter

PLOT_BG    = "rgba(6,11,22,0.0)"
PAPER_BG   = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(0,229,255,0.06)"
FONT_COLOR = "#94a3b8"
FONT_MONO  = "JetBrains Mono, monospace"

BASE_LAYOUT = dict(
    paper_bgcolor=PAPER_BG,
    plot_bgcolor=PLOT_BG,
    font=dict(family=FONT_MONO, color=FONT_COLOR, size=11),
    margin=dict(l=8, r=8, t=32, b=8),
    showlegend=True,
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(0,229,255,0.15)",
        font=dict(size=10),
    ),
    xaxis=dict(showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
               tickfont=dict(size=9), linecolor="rgba(0,229,255,0.1)"),
    yaxis=dict(showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
               tickfont=dict(size=9), linecolor="rgba(0,229,255,0.1)"),
)

CLASS_COLORS = {
    "car":          "#00e5ff",
    "bike":         "#7c3aed",
    "bus":          "#ffb400",
    "truck":        "#ff8800",
    "auto_rickshaw":"#00e676",
    "pedestrian":   "#94a3b8",
}

VIOLATION_COLORS = {
    "no_helmet":      "#ff3c3c",
    "no_seatbelt":    "#ff8800",
    "phone_use":      "#ffb400",
    "wrong_side":     "#7c3aed",
    "tampered_plate": "#00e5ff",
}


def vehicle_count_line(history: list) -> go.Figure:
    """Vehicle count over time — multi-line by class."""
    df = pd.DataFrame(history[-120:])  # last 2 hours
    if df.empty:
        return _empty_fig("No data")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["vehicles"],
        name="Total", mode="lines",
        line=dict(color="#00e5ff", width=2),
        fill="tozeroy", fillcolor="rgba(0,229,255,0.05)",
    ))
    for cls in ["cars", "bikes", "buses", "trucks"]:
        if cls in df.columns:
            fig.add_trace(go.Scatter(
                x=df["time"], y=df[cls],
                name=cls.title(), mode="lines",
                line=dict(width=1.2, dash="dot"),
                visible="legendonly",
            ))
    fig.update_layout(**BASE_LAYOUT,  # type: ignore
        title=dict(text="Vehicle Count / Time", font=dict(size=12, color="#e2e8f0")),
        height=280,
    )
    return fig


def class_distribution_pie(counts: dict) -> go.Figure:
    labels = [k for k, v in counts.items() if v > 0]
    values = [v for v in counts.values() if v > 0]
    colors = [CLASS_COLORS.get(l, "#64748b") for l in labels]

    fig = go.Figure(go.Pie(
        labels=[l.replace("_", " ").title() for l in labels],
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="#04060d", width=2)),
        textfont=dict(family=FONT_MONO, size=10),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(**BASE_LAYOUT,  # type: ignore
        title=dict(text="Class Distribution", font=dict(size=12, color="#e2e8f0")),
        showlegend=True, height=280,
        annotations=[dict(text="FLEET", x=0.5, y=0.5, font_size=11,
                           font_color="#64748b", showarrow=False,
                           font_family=FONT_MONO)],
    )
    return fig


def violation_frequency_bar(violations: list) -> go.Figure:
    if not violations:
        return _empty_fig("No violations")

    counts = Counter(v["type"] for v in violations)
    types  = list(counts.keys())
    values = list(counts.values())
    colors = [VIOLATION_COLORS.get(t, "#64748b") for t in types]

    fig = go.Figure(go.Bar(
        x=[t.replace("_", " ").upper() for t in types],
        y=values,
        marker=dict(color=colors, opacity=0.85,
                    line=dict(color=[c + "88" for c in colors], width=1)),
        text=values, textposition="outside",
        textfont=dict(size=10, color="#e2e8f0"),
        hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
    ))
    fig.update_layout(**BASE_LAYOUT,  # type: ignore
        title=dict(text="Violation Frequency", font=dict(size=12, color="#e2e8f0")),
        height=280, bargap=0.3,
        xaxis=dict(tickfont=dict(size=9), showgrid=False),
    )
    return fig


def density_trend_area(history: list) -> go.Figure:
    df = pd.DataFrame(history[-120:])
    if df.empty:
        return _empty_fig("No data")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["density"],
        name="Density", mode="lines",
        line=dict(color="#7c3aed", width=2),
        fill="tozeroy", fillcolor="rgba(124,58,237,0.08)",
    ))
    # Threshold bands
    for y_val, color, label in [
        (0.25, "rgba(0,230,118,0.15)", "FREE"),
        (0.50, "rgba(255,234,0,0.12)", "MODERATE"),
        (0.75, "rgba(255,136,0,0.10)", "HEAVY"),
    ]:
        fig.add_hline(y=y_val, line=dict(color=color.replace("0.15","0.4")
                                              .replace("0.12","0.4")
                                              .replace("0.10","0.4"), dash="dot", width=1),
                      annotation_text=label,
                      annotation_font=dict(size=9, color=FONT_COLOR))

    fig.update_layout(**BASE_LAYOUT,  # type: ignore
        title=dict(text="Traffic Density Trend", font=dict(size=12, color="#e2e8f0")),
        yaxis=dict(range=[0, 1.05], tickformat=".0%", showgrid=True, gridcolor=GRID_COLOR),
        height=280,
    )
    return fig


def speed_histogram(history: list) -> go.Figure:
    df = pd.DataFrame(history)
    if df.empty or "speed_avg" not in df.columns:
        return _empty_fig("No speed data")

    fig = go.Figure(go.Histogram(
        x=df["speed_avg"],
        nbinsx=20,
        marker=dict(color="#00e5ff", opacity=0.7,
                    line=dict(color="#00e5ff", width=0.5)),
        hovertemplate="Speed: %{x:.1f} km/h<br>Count: %{y}<extra></extra>",
    ))
    fig.update_layout(**BASE_LAYOUT,  # type: ignore
        title=dict(text="Speed Distribution (km/h)", font=dict(size=12, color="#e2e8f0")),
        height=240, bargap=0.05,
        xaxis_title="km/h", yaxis_title="Frequency",
    )
    return fig


def hourly_violations(violations: list) -> go.Figure:
    if not violations:
        return _empty_fig("No violations")

    hours = [int(v["timestamp"].split(" ")[1].split(":")[0]) for v in violations
             if " " in v["timestamp"]]
    counts = Counter(hours)
    xs     = list(range(24))
    ys     = [counts.get(h, 0) for h in xs]

    fig = go.Figure(go.Bar(
        x=[f"{h:02d}:00" for h in xs], y=ys,
        marker=dict(color="#ff3c3c", opacity=0.75,
                    line=dict(color="#ff3c3c88", width=0.5)),
    ))
    fig.update_layout(**BASE_LAYOUT,  # type: ignore
        title=dict(text="Violations by Hour", font=dict(size=12, color="#e2e8f0")),
        height=240, bargap=0.2,
        xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
    )
    return fig


def live_gauge(score: float, level: str) -> go.Figure:
    colors_map = {"FREE": "#00e676", "MODERATE": "#ffea00", "HEAVY": "#ff8800", "GRIDLOCK": "#ff3c3c"}
    color = colors_map.get(level, "#00e5ff")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score * 100, 1),
        number=dict(suffix="%", font=dict(family="Orbitron, monospace",
                                           size=28, color=color)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickfont=dict(size=9)),
            bar=dict(color=color, thickness=0.25),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0,  25], color="rgba(0,230,118,0.12)"),
                dict(range=[25, 50], color="rgba(255,234,0,0.10)"),
                dict(range=[50, 75], color="rgba(255,136,0,0.08)"),
                dict(range=[75,100], color="rgba(255,60,60,0.12)"),
            ],
            threshold=dict(line=dict(color=color, width=2), value=score * 100),
        ),
        title=dict(text=f"<span style='font-size:12px;color:#64748b;font-family:JetBrains Mono'>CONGESTION · {level}</span>"),
    ))
    fig.update_layout(
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(family=FONT_MONO, color=FONT_COLOR),
        height=220, margin=dict(l=16, r=16, t=20, b=8),
    )
    return fig


def _empty_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                       font=dict(color=FONT_COLOR, size=13), showarrow=False)
    fig.update_layout(paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
                      height=200, margin=dict(l=8, r=8, t=8, b=8),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig
