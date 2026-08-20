from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# Page setup
st.set_page_config(
    page_title="FloatChat | Ocean Observation Portal",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional Ocean Styling (Deep Slate Navy + Subdued Ocean Accents)
CSS_THEME = """
<style>
    /* Primary Color Variables */
    :root {
        --bg-color: #0b132b;
        --card-bg: #1c2541;
        --text-primary: #e0e1dd;
        --accent-teal: #48cae4;
        --border-color: #3a506b;
    }

    /* Professional Top Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0b132b 0%, #1c2541 60%, #1c3144 100%);
        padding: 24px 28px;
        border-radius: 10px;
        border: 1px solid #3a506b;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .hero-title {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #90e0ef;
        font-size: 0.95rem;
        margin-top: 6px;
        margin-bottom: 0;
        font-weight: 400;
    }

    /* Subdued Status Badges */
    .metric-badge {
        background-color: #1c2541;
        border: 1px solid #3a506b;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 600;
        color: #48cae4;
    }
</style>
"""
st.markdown(CSS_THEME, unsafe_allow_html=True)

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

# Coordinates lookup
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


def render_results(results: pd.DataFrame, city: str | None, year: int | None) -> None:
    st.markdown("##### 📊 Observation Records")
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
            "Trend Parameter",
            ["Temperature", "Salinity"],
            horizontal=True,
            key=f"metric_{id(results)}"
        )
        value_column = "temperature_celsius" if metric == "Temperature" else "salinity_psu"
        y_label = "Temperature (°C)" if metric == "Temperature" else "Salinity (PSU)"
        chart_data = results.dropna(subset=[value_column])
        
        if chart_data.empty:
            st.info(f"No {metric.lower()} values available for these parameters.")
        else:
            # Professional plot theme: Low contrast dark base with readable muted lines
            figure = px.line(
                chart_data,
                x="date",
                y=value_column,
                color="float_id",
                markers=True,
                labels={"date": "Observation Date", value_column: y_label, "float_id": "Float ID"},
                template="plotly_dark",
                color_discrete_sequence=["#48cae4", "#00b4d8", "#90e0ef", "#0077b6", "#caf0f8"]
            )
            figure.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(11,19,43,0.5)",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    title_text=""
                ),
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis=dict(showgrid=True, gridcolor="#1c2541"),
                yaxis=dict(showgrid=True, gridcolor="#1c2541"),
            )
            st.plotly_chart(figure, use_container_width=True)

    with right:
        st.markdown("##### 📍 Geo-Location Map")
        st.map(
            results.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]],
            use_container_width=True,
            zoom=4 if city else None,
        )


def main() -> None:
    # Title Banner (Clean, Professional, Academic Header)
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
        st.metric("Total Observations", f"{len(data):,}")
        st.metric("Active Float Nodes", data["float_id"].nunique())
        
        st.divider()
        st.markdown("**Query Syntax Guidance**")
        st.caption(
            "Combine a target geographic node with an observation year:\n\n"
            "• *`temperature near Chennai 2023`*\n"
            "• *`salinity near Mumbai 2022`*"
        )
        st.divider()
        st.markdown("**Indexed Coastal Regions**")
        st.caption(" • ".join(name.title() for name in CITY_COORDINATES))

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Welcome to the FloatChat analysis console. Input location and year queries "
                    "to retrieve observation metrics (e.g., **temperature near Chennai 2023**)."
                ),
            }
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("results") is not None:
                render_results(message["results"], message.get("city"), message.get("year"))

    if question := st.chat_input("Query float observations by location or year..."):
        city, year = parse_question(question)
        results = filter_data(data, city, year)
        st.session_state.messages.append({"role": "user", "content": question})

        if city is None and year is None:
            response = (
                "Unrecognized query scope. Please specify a valid city region or observation year. "
                "Example: **temperature near Chennai 2023**."
            )
            st.session_state.messages.append({"role": "assistant", "content": response})
        elif results.empty:
            requested_city = city.title() if city else "all locations"
            requested_year = str(year) if year else "all years"
            response = (
                f"Parameters parsed for **{requested_city}** ({requested_year}), "
                "but no matching float records were retrieved from the current dataset."
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
                f"{' '.join(scope)}. Data visualization provided below:"
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
