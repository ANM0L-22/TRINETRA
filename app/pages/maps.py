"""
Trinetra — Maps / Traffic View
PyDeck-based geospatial traffic density visualization.
"""

import random
import streamlit as st
import pandas as pd
import pydeck as pdk

from app.data.simulator import get_frame_data
from app.components.styles import section_header, metric_chip


INTERSECTION_BASE = [
    {"name": "NH-48 · Sector 7",      "lat": 26.9124, "lon": 75.7873},
    {"name": "MG Road · Junction 3",   "lat": 26.9010, "lon": 75.8000},
    {"name": "Ring Road · Gate 12",    "lat": 26.8890, "lon": 75.7700},
    {"name": "Outer Ring · Node 6",    "lat": 26.9200, "lon": 75.8100},
    {"name": "City Center · Hub 1",    "lat": 26.9300, "lon": 75.7600},
    {"name": "Airport Road · Gate 2",  "lat": 26.8740, "lon": 75.8150},
    {"name": "Vaishali Nagar · Jct",   "lat": 26.9050, "lon": 75.7450},
    {"name": "Tonk Road · Sec 12",     "lat": 26.8650, "lon": 75.8050},
]


def _density_to_color(d: float):
    """Returns RGBA for a density score 0–1."""
    if   d < 0.25: return [0,   230, 118, 180]
    elif d < 0.50: return [255, 234,   0, 180]
    elif d < 0.75: return [255, 136,   0, 200]
    else:          return [255,  60,  60, 220]


def _generate_map_data():
    rows = []
    for loc in INTERSECTION_BASE:
        d = random.uniform(0.1, 0.95)
        vc = int(d * 20 + random.randint(0, 5))
        rows.append({
            "name":          loc["name"],
            "lat":           loc["lat"] + random.uniform(-0.001, 0.001),
            "lon":           loc["lon"] + random.uniform(-0.001, 0.001),
            "density":       round(d, 3),
            "vehicle_count": vc,
            "congestion":    "GRIDLOCK" if d>0.75 else "HEAVY" if d>0.5 else "MODERATE" if d>0.25 else "FREE",
            "color":         _density_to_color(d),
            "radius":        int(d * 300 + 100),
            "violations":    random.randint(0, int(d * 8)),
        })
    return rows


def maps_page():
    st.markdown(section_header("MAPS", "Geospatial Traffic Intelligence"), unsafe_allow_html=True)

    # Controls
    mc1, mc2, mc3 = st.columns([2.2, 1.1, 4])
    with mc1:
        map_layer = st.selectbox("Layer", ["Density Heatmap","Vehicle Count","Congestion Zones","Violations"],
                                  key="map_layer")
    with mc2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Map", use_container_width=True):
            if "map_data" in st.session_state:
                del st.session_state["map_data"]

    if "map_data" not in st.session_state:
        st.session_state.map_data = _generate_map_data()

    data = st.session_state.map_data
    df   = pd.DataFrame(data)

    # ── Summary chips ──────────────────────────────────────────
    congestion_counts = df["congestion"].value_counts()
    chip_cols = st.columns(4)
    chip_data = [
        ("FREE",     congestion_counts.get("FREE",0),     "#00e676"),
        ("MODERATE", congestion_counts.get("MODERATE",0), "#ffea00"),
        ("HEAVY",    congestion_counts.get("HEAVY",0),    "#ff8800"),
        ("GRIDLOCK", congestion_counts.get("GRIDLOCK",0), "#ff3c3c"),
    ]
    for col, (label, val, color) in zip(chip_cols, chip_data):
        with col:
            st.markdown(metric_chip(label, str(val), color), unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Map ───────────────────────────────────────────────────
    # Scatter layer (circles per intersection)
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_radius="radius",
        pickable=True,
        opacity=0.85,
        stroked=True,
        get_line_color=[255, 255, 255, 60],
        line_width_min_pixels=1,
    )

    # Text labels
    text_layer = pdk.Layer(
        "TextLayer",
        data=df,
        get_position=["lon", "lat"],
        get_text="name",
        get_size=11,
        get_color=[200, 220, 255, 220],
        get_alignment_baseline="'bottom'",
        get_anchor="'middle'",
        get_pixel_offset=[0, -20],
        font_family="JetBrains Mono, monospace",
    )

    # Column layer for vehicle count
    column_layer = pdk.Layer(
        "ColumnLayer",
        data=df,
        get_position=["lon", "lat"],
        get_elevation="vehicle_count",
        elevation_scale=80,
        radius=120,
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
        opacity=0.7,
    )

    active_layers = [scatter_layer, text_layer]
    if map_layer == "Vehicle Count":
        active_layers = [column_layer, text_layer]

    view_state = pdk.ViewState(
        latitude=26.9050,
        longitude=75.7800,
        zoom=12,
        pitch=40 if map_layer == "Vehicle Count" else 0,
        bearing=0,
    )

    tooltip = {
        "html": """
        <div style='background:rgba(6,11,22,0.95);border:1px solid rgba(0,229,255,0.3);
                    border-radius:8px;padding:12px;font-family:monospace;'>
            <b style='color:#00e5ff;font-size:12px;'>{name}</b><br>
            <span style='color:#94a3b8;font-size:10px;'>Density: </span>
            <span style='color:#e2e8f0;'>{density}</span><br>
            <span style='color:#94a3b8;font-size:10px;'>Vehicles: </span>
            <span style='color:#e2e8f0;'>{vehicle_count}</span><br>
            <span style='color:#94a3b8;font-size:10px;'>Status: </span>
            <span style='color:#ffb400;'>{congestion}</span><br>
            <span style='color:#94a3b8;font-size:10px;'>Violations: </span>
            <span style='color:#ff3c3c;'>{violations}</span>
        </div>
        """,
        "style": {"backgroundColor": "transparent", "border": "none"},
    }

    deck = pdk.Deck(
        layers=active_layers,
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="mapbox://styles/mapbox/dark-v11",
        map_provider="mapbox",
    )

    st.pydeck_chart(deck, use_container_width=True, height=520)

    # ── Intersection table ─────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:3px;
                color:#64748b;text-transform:uppercase;margin-bottom:8px;">
        Intersection Status
    </div>
    """, unsafe_allow_html=True)

    display = df[["name","congestion","vehicle_count","density","violations"]].copy()
    display.columns = ["Intersection","Congestion","Vehicles","Density","Violations"]
    display["Density"] = display["Density"].map("{:.1%}".format)

    def color_congestion(val):
        colors = {"FREE":"color:#00e676","MODERATE":"color:#ffea00",
                  "HEAVY":"color:#ff8800","GRIDLOCK":"color:#ff3c3c"}
        return colors.get(val, "")

    st.dataframe(
        display.style.applymap(color_congestion, subset=["Congestion"]),
        use_container_width=True, hide_index=True, height=320,
    )
