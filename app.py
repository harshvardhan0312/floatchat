from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FloatChat | Ocean Observation Portal",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional Ocean Research Dashboard Styling (Muted Deep Navy Palette)
CSS_THEME = """
<style>
    /* Dark Theme Core Colors */
    .stApp {
        background-color: #0b132b;
        color: #e0e1dd;
    }

    /* Subdued Gradient Top Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b263b 50%, #1c3144 100%);
        border: 1px solid #2b3a4e;
        border-radius: 12px;
        padding: 24px 30px;
        margin-bottom: 25px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
    }
    .hero-title {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #7f5a83;
        color: #48cae4;
        font-size: 0.95rem;
        margin-top: 6px;
        margin-bottom: 0;
        font-weight: 400;
        opacity: 0.9;
    }

    /* Visual Glassmorphic Cards for Data Summaries */
    .stat-card {
        background: rgba(27, 38, 59, 0.6);
        border: 1px solid #2b3a4e;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        backdrop-filter: blur(5px);
    }
    .stat-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        color: #8d99ae;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #00b4d8;
    }

    /* Styled Chat Container Adjustments */
    .stChatMessage {
        border-radius: 10px;
        border: 1px solid #1c2541;
        margin-bottom: 10px;
    }

    /* Clean Sidebar Divider */
    .sidebar-section {
        background: #1b263b;
        padding: 14px;
        border-radius: 8px;
        border: 1px solid #2b3a4e;
        margin-bottom: 12px;
    }
</style>
"""
st.markdown(CSS_THEME, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DATA CONFIGURATION & HELPERS
# -----------------------------------------------------------------------------
DATA_PATH = Path(__file__).parent / "ocean_data.csv"
REQUIRED_COLUMNS = [
    "float_id",
    "latitude",
    "longitude",
    "date",
    "temperature_celsius",
    "salinity_psu",
    "depth_meters",
]

CITY_COORDINATES = {
    "chennai": (13.0827, 80.2707),
    "mumbai": (19.0760, 72.8777),
    "singapore": (1.3521, 103.8198),
    "sydney": (-33.8688, 151.2093),
    "colombo": (6.9271, 79.8612),
    "dubai": (25.2048, 55.2708),
    "cape town": (-33.9249, 18.4241),
    "san francisco": (37.7749, -122.4194),
}
CITY_RADIUS_KM = 90


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load, validate, and normalize the ocean observations."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find {DATA_PATH.name} in the app folder.")

    data = pd.read_csv(DATA_PATH)
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required CSV columns: {', '.join(missing)}")

    data = data[REQUIRED_COLUMNS].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    numeric_columns = [
        "latitude",
        "longitude",
        "temperature_celsius",
        "salinity_psu",
        "depth_meters",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["date", "latitude", "longitude"])
    return data.sort_values("date").reset_index(drop=True)


def parse_question(question: str) -> tuple[str | None, int | None]:
    normalized = question.casefold()
    city = next(
        (name for name in sorted(CITY_COORDINATES, key=len, reverse=True) if name in normalized),
        None,
    )
    year_match = re.search(r"\b(19|20)\d{2}\b", normalized)
    year = int(year_match.group(0)) if year_match else None
    return city, year


def haversine_km(
    latitudes: pd.Series,
    longitudes: pd.Series,
    target_latitude: float,
    target_longitude: float,
) -> pd.Series:
    """Calculate approximate distance from every float to a city in kilometers."""
    from math import asin, cos, radians, sin, sqrt

    lat1 = latitudes.map(radians)
    lat2 = radians(target_latitude)
    delta_lat = lat2 - lat1
    delta_lon = longitudes.map(radians) - radians(target_longitude)
    haversine = (
        delta_lat.map(sin).pow(2)
        + lat1.map(cos) * cos(lat2) * delta_lon.map(sin).pow(2)
    )
    return haversine.clip(lower=0, upper=1).map(
        lambda value: 2 * 6371 * asin(sqrt(value))
    )


def filter_data(data: pd.DataFrame, city: str | None, year: int | None) -> pd.DataFrame:
    filtered = data.copy()
    if year is not None:
        filtered = filtered[filtered["date"].dt.year == year]
    if city is not None:
        latitude, longitude = CITY_COORDINATES[city]
        distances = haversine_km(filtered["latitude"], filtered["longitude"], latitude, longitude)
        filtered = filtered[distances <= CITY_RADIUS_KM].copy()
        filtered["distance_km"] = distances[distances <= CITY_RADIUS_KM].round(1)
    return filtered.sort_values("date").reset_index(drop=True)


# -----------------------------------------------------------------------------
# VISUAL RENDERING DASHBOARD
# -----------------------------------------------------------------------------
def render_results(results: pd.DataFrame, city: str | None, year: int | None) -> None:
    # 1. Summary Cards (Visual Overview Panel)
    avg_temp = results["temperature_celsius"].mean()
    avg_sal = results["salinity_psu"].mean()
    avg_depth = results["depth_meters"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="stat-card"><div class="stat-label">Matches</div><div class="stat-value">{len(results)}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        val = f"{avg_temp:.1f} °C" if pd.notnull(avg_temp) else "N/A"
        st.markdown(
            f'<div class="stat-card"><div class="stat-label">Avg Temp</div><div class="stat-value">{val}</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        val = f"{avg_sal:.2f} PSU" if pd.notnull(avg_sal) else "N/A"
        st.markdown(
            f'<div class="stat-card"><div class="stat-label">Avg Salinity</div><div class="stat-value">{val}</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        val = f"{avg_depth:.0f} m" if pd.notnull(avg_depth) else "N/A"
        st.markdown(
            f'<div class="stat-card"><div class="stat-label">Avg Depth</div><div class="stat-value">{val}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### 📄 Observations Table")
    
    display_columns = REQUIRED_COLUMNS + (["distance_km"] if "distance_km" in results else [])
    st.dataframe(
        results[display_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "date": st.column_config.DateColumn("Date", format="MMM D, YYYY"),
            "temperature_celsius": st.column_config.NumberColumn("Temperature", format="%.1f °C"),
            "salinity_psu": st.column_config.NumberColumn("Salinity", format="%.2f PSU"),
            "depth_meters": st.column_config.NumberColumn("Depth", format="%.0f m"),
            "distance_km": st.column_config.NumberColumn("Distance", format="%.1f km"),
        },
    )

    left, right = st.columns([1.35, 1])
    with left:
        metric = st.radio(
            "Select Metric to Plot",
            ["Temperature", "Salinity"],
            horizontal=True,
            key=f"metric_{id(results)}",
        )
        value_column = "temperature_celsius" if metric == "Temperature" else "salinity_psu"
        y_label = "Temperature (°C)" if metric == "Temperature" else "Salinity (PSU)"
        chart_data = results.dropna(subset=[value_column])

        if chart_data.empty:
            st.info(f"No {metric.lower()} values are available for these observations.")
        else:
            # Custom Dark Charting: Muted colors for maximum visibility on dark backgrounds
            figure = px.line(
                chart_data,
                x="date",
                y=value_column,
                color="float_id",
                markers=True,
                labels={"date": "Observation Date", value_column: y_label, "float_id": "Float ID"},
                template="plotly_dark",
                # Professional Ocean Palette (Steel Blue, Soft Turquoise, Pale Ice Cyan)
                color_discrete_sequence=["#48cae4", "#00b4d8", "#90e0ef", "#52b788", "#74c69d"],
            )
            figure.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(11, 19, 43, 0.7)",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    title_text="",
                ),
                margin=dict(l=10, r=10, t=25, b=10),
                xaxis=dict(showgrid=True, gridcolor="#1c2541"),
                yaxis=dict(showgrid=True, gridcolor="#1c2541"),
            )
            st.plotly_chart(figure, use_container_width=True)

    with right:
        st.markdown("##### 🌐 Spatial Scatter Map")
        # Visual Geo Map using Plotly Dark Map Box for better visual aesthetics
        if city and city in CITY_COORDINATES:
            lat_c, lon_c = CITY_COORDINATES[city]
            map_fig = px.scatter_mapbox(
                results,
                lat="latitude",
                lon="longitude",
                color="float_id",
                size="depth_meters",
                hover_data=["date", "temperature_celsius", "salinity_psu"],
                zoom=6,
                center={"lat": lat_c, "lon": lon_c},
                mapbox_style="carto-darkmatter",
                color_discrete_sequence=["#48cae4", "#00b4d8", "#90e0ef", "#52b788"],
            )
            map_fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                height=350,
            )
            st.plotly_chart(map_fig, use_container_width=True)
        else:
            st.map(
                results.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]],
                use_container_width=True,
            )


# -----------------------------------------------------------------------------
# MAIN APPLICATION
# -----------------------------------------------------------------------------
def main() -> None:
    # Top Professional Banner Header
    st.markdown(
        """
        <div class="hero-banner">
            <h1 class="hero-title">🌊 FloatChat</h1>
            <p class="hero-subtitle">SIH25040 Solution Prototype | Autonomous Ocean Observation Analysis Platform</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        data = load_data()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.stop()

    with st.sidebar:
        st.markdown("### ⚓ Dataset Overview")
        
        # Sidebar Visual Summary Cards
        st.markdown(
            f"""
            <div class="sidebar-section">
                <small style="color:#8d99ae;">TOTAL OBSERVATIONS</small>
                <h3 style="color:#48cae4; margin:0;">{len(data):,}</h3>
            </div>
            <div class="sidebar-section">
                <small style="color:#8d99ae;">ACTIVE FLOATS</small>
                <h3 style="color:#48cae4; margin:0;">{data['float_id'].nunique()}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown("**Query Syntax Guidance**")
        st.caption(
            "Query observations by combining a recognized coastal region and year:\n\n"
            "• *`temperature near Chennai 2023`*\n"
            "• *`salinity near Singapore 2022`*"
        )
        st.divider()
        st.markdown("**Indexed Coastal Regions**")
        st.caption(" • ".join(name.title() for name in CITY_COORDINATES))

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Welcome to the FloatChat analysis console. Enter geographic and temporal keywords "
                    "to query oceanographic float data (e.g., **temperature near Chennai 2023**)."
                ),
            }
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("results") is not None:
                render_results(message["results"], message.get("city"), message.get("year"))

    if question := st.chat_input("Query ocean float observations..."):
        city, year = parse_question(question)
        results = filter_data(data, city, year)
        st.session_state.messages.append({"role": "user", "content": question})

        if city is None and year is None:
            response = (
                "Unrecognized query scope. Please specify a valid coastal city or observation year. "
                "Try a query like **temperature near Chennai 2023**."
            )
            st.session_state.messages.append({"role": "assistant", "content": response})
        elif results.empty:
            requested_city = city.title() if city else "all locations"
            requested_year = str(year) if year else "all years"
            response = (
                f"I identified keywords for **{requested_city}** ({requested_year}), "
                "but no matching float records exist in the current dataset."
            )
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            scope = []
            if city:
                scope.append(f"within {CITY_RADIUS_KM} km of {city.title()}")
            if year:
                scope.append(f"in {year}")
            response = (
                f"Retrieved **{len(results)} observation record(s)** "
                f"{' '.join(scope)}. Analytical results provided below:"
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response,
                    "results": results,
                    "city": city,
                    "year": year,
                }
            )
        st.rerun()


if __name__ == "__main__":
    main()
