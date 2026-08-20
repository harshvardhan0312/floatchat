from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FloatChat | Ocean Data Explorer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Professional ocean-themed styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ================================================================
       FloatChat — restrained SIH / scientific dashboard theme
       ================================================================ */

    .stApp {
        background: #f3f5f6;
        color: #263840;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.25rem;
        padding-bottom: 2.5rem;
    }

    /* Compact, dark header — no bright gradient */
    .floatchat-banner {
        background: #203c49;
        border: 1px solid #304f5c;
        border-radius: 10px;
        padding: 20px 25px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(30, 55, 65, 0.08);
    }

    .floatchat-title {
        color: #f4f7f8;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.35px;
        margin: 0;
    }

    .floatchat-subtitle {
        color: #c8d5d9;
        font-size: 0.9rem;
        margin-top: 5px;
        margin-bottom: 0;
    }

    .banner-badge {
        display: inline-block;
        margin-top: 11px;
        padding: 4px 9px;
        border-radius: 5px;
        background: #294955;
        border: 1px solid #41606b;
        color: #d5e1e4;
        font-size: 0.72rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #e9eef0;
        border-right: 1px solid #d3dde1;
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #294956;
    }

    /* Small neutral metric cards */
    .dataset-card {
        background: #f8fafb;
        border: 1px solid #d6e0e3;
        border-radius: 8px;
        padding: 10px 12px;
        margin: 7px 0;
    }

    .dataset-label {
        color: #667980;
        font-size: 0.70rem;
        text-transform: uppercase;
        letter-spacing: 0.45px;
    }

    .dataset-value {
        color: #294956;
        font-size: 1.25rem;
        font-weight: 650;
        margin-top: 2px;
    }

    /* Section labels */
    .section-title {
        color: #294956;
        font-size: 0.98rem;
        font-weight: 650;
        margin: 16px 0 8px 0;
        padding-left: 8px;
        border-left: 3px solid #607d87;
    }

    /* Chat */
    [data-testid="stChatMessage"] {
        border-radius: 8px;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border: 1px solid #d6e0e3;
        border-radius: 8px;
        overflow: hidden;
        background: #ffffff;
    }

    /* Inputs */
    [data-testid="stChatInput"] {
        border-color: #cbd7db;
    }

    .stRadio label {
        color: #405a63;
        font-weight: 500;
    }

    /* Footer */
    .floatchat-footer {
        margin-top: 30px;
        padding-top: 12px;
        border-top: 1px solid #d3dde1;
        color: #77888e;
        font-size: 0.74rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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

# A small, transparent lookup keeps the app local and avoids an external geocoding API.
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
        (
            name
            for name in sorted(CITY_COORDINATES, key=len, reverse=True)
            if name in normalized
        ),
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


def filter_data(
    data: pd.DataFrame,
    city: str | None,
    year: int | None,
) -> pd.DataFrame:
    filtered = data.copy()

    if year is not None:
        filtered = filtered[filtered["date"].dt.year == year]

    if city is not None:
        latitude, longitude = CITY_COORDINATES[city]
        distances = haversine_km(
            filtered["latitude"],
            filtered["longitude"],
            latitude,
            longitude,
        )
        filtered = filtered[distances <= CITY_RADIUS_KM].copy()
        filtered["distance_km"] = distances[distances <= CITY_RADIUS_KM].round(1)

    return filtered.sort_values("date").reset_index(drop=True)


def render_results(
    results: pd.DataFrame,
    city: str | None,
    year: int | None,
    result_key: str,
) -> None:
    st.markdown('<div class="section-title">Matching observations</div>', unsafe_allow_html=True)

    display_columns = REQUIRED_COLUMNS + (
        ["distance_km"] if "distance_km" in results else []
    )

    st.dataframe(
        results[display_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "date": st.column_config.DateColumn("Date", format="MMM D, YYYY"),
            "temperature_celsius": st.column_config.NumberColumn(
                "Temperature", format="%.1f °C"
            ),
            "salinity_psu": st.column_config.NumberColumn(
                "Salinity", format="%.2f PSU"
            ),
            "depth_meters": st.column_config.NumberColumn("Depth", format="%.0f m"),
            "distance_km": st.column_config.NumberColumn(
                "Distance", format="%.1f km"
            ),
        },
    )

    left, right = st.columns([1.35, 1])

    with left:
        st.markdown(
            '<div class="section-title">Observation trend</div>',
            unsafe_allow_html=True,
        )

        metric = st.radio(
            "Select measurement",
            ["Temperature", "Salinity"],
            horizontal=True,
            key=f"metric_{result_key}",
            label_visibility="collapsed",
        )

        value_column = (
            "temperature_celsius"
            if metric == "Temperature"
            else "salinity_psu"
        )
        y_label = (
            "Temperature (°C)"
            if metric == "Temperature"
            else "Salinity (PSU)"
        )

        chart_data = results.dropna(subset=[value_column])

        if chart_data.empty:
            st.info(f"No {metric.lower()} values are available for these observations.")
        else:
            figure = px.line(
                chart_data,
                x="date",
                y=value_column,
                color="float_id",
                markers=True,
                labels={
                    "date": "Observation date",
                    value_column: y_label,
                    "float_id": "Float",
                },
            )

            figure.update_layout(
                template="plotly_white",
                height=390,
                legend_title_text="Float",
                margin=dict(l=10, r=10, t=15, b=10),
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(color="#263840", size=12),
                colorway=[
                    "#365b69",
                    "#607985",
                    "#7b8f96",
                    "#4b6f7c",
                    "#8b9da2",
                ],
                legend=dict(
                    bgcolor="#ffffff",
                    bordercolor="#d7e0e3",
                    borderwidth=1,
                ),
            )

            figure.update_xaxes(
                showgrid=True,
                gridcolor="#e1e6e8",
                zeroline=False,
            )
            figure.update_yaxes(
                showgrid=True,
                gridcolor="#e1e6e8",
                zeroline=False,
            )

            st.plotly_chart(
                figure,
                use_container_width=True,
                config={"displaylogo": False},
            )

    with right:
        st.markdown(
            '<div class="section-title">Float locations</div>',
            unsafe_allow_html=True,
        )

        st.map(
            results.rename(
                columns={"latitude": "lat", "longitude": "lon"}
            )[["lat", "lon"]],
            use_container_width=True,
            zoom=4 if city else None,
        )


def main() -> None:
    # -----------------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------------
    st.markdown(
        """
        <div class="floatchat-banner">
            <div class="floatchat-title">🌊 FloatChat</div>
            <div class="floatchat-subtitle">
                Ocean Data Exploration &nbsp;•&nbsp; Argo Float Observations
            </div>
            <div class="banner-badge">
                Data-driven ocean observation explorer
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        data = load_data()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.stop()

    # -----------------------------------------------------------------------
    # Sidebar
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.markdown("## 🌐 Dataset")

        st.markdown(
            f"""
            <div class="dataset-card">
                <div class="dataset-label">Observations</div>
                <div class="dataset-value">{len(data):,}</div>
            </div>

            <div class="dataset-card">
                <div class="dataset-label">Active floats</div>
                <div class="dataset-value">{data["float_id"].nunique():,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        st.markdown("### 🔎 Query examples")
        st.caption(
            "Ask for observations using a recognized city and year."
        )

        st.code("temperature near Chennai 2023", language="text")
        st.code("salinity near Dubai 2023", language="text")
        st.code("temperature near Singapore 2024", language="text")

        st.divider()

        st.markdown("### 📍 Recognized locations")
        st.caption(
            ", ".join(name.title() for name in CITY_COORDINATES)
        )

        st.caption(
            f"Search radius: {CITY_RADIUS_KM} km"
        )

    # -----------------------------------------------------------------------
    # Chat state
    # -----------------------------------------------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Welcome to **FloatChat**. Ask for ocean observations by "
                    "city and year, for example **temperature near Chennai 2023**."
                ),
            }
        ]

    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("results") is not None:
                render_results(
                    message["results"],
                    message.get("city"),
                    message.get("year"),
                    result_key=str(index),
                )

    # -----------------------------------------------------------------------
    # Query handling
    # -----------------------------------------------------------------------
    if question := st.chat_input(
        "Ask about a city and year…"
    ):
        city, year = parse_question(question)
        results = filter_data(data, city, year)

        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        if city is None and year is None:
            response = (
                "I couldn't find a recognized city or year in that question. "
                "Try a query like **temperature near Chennai 2023**."
            )

            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )

        elif results.empty:
            requested_city = city.title() if city else "all locations"
            requested_year = str(year) if year else "all years"

            response = (
                f"I found the keywords for **{requested_city}** and "
                f"**{requested_year}**, but there are no matching "
                "observations in the dataset."
            )

            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )

        else:
            scope = []

            if city:
                scope.append(
                    f"within {CITY_RADIUS_KM} km of {city.title()}"
                )

            if year:
                scope.append(f"in {year}")

            response = (
                f"I found **{len(results)} observations** "
                f"{' '.join(scope)}. The table, trend, and map below "
                "show the matching observations."
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

    # -----------------------------------------------------------------------
    # Footer
    # -----------------------------------------------------------------------
    st.markdown(
        """
        <div class="floatchat-footer">
            FloatChat &nbsp;•&nbsp; Ocean observation prototype
            &nbsp;•&nbsp; Argo float data exploration
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
