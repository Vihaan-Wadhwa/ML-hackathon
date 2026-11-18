
import streamlit as st
import requests
import sys
from pathlib import Path

# ---------------- PROJECT PATH ----------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visualization.folium_map import display_route_map

# ---------------- CONFIGURATION ----------------

API_URL = "http://127.0.0.1:8000"

EVAL_TRIPS = [
    "00000015", "00000017", "00000023", "00000030", "00000049",
    "00000059", "00000067", "00000073", "00000087", "00000094"
]

st.set_page_config(
    page_title="GPS Route Reconstruction",
    page_icon="🛰️",
    layout="wide"
)

# ---------------- CUSTOM STYLING ----------------

st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background: #1c2333;
        padding: 18px;
        border-radius: 12px;
    }
    h1, h2, h3 {
        color: #60a5fa;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.title("🛰️ GPS Route Reconstruction")
st.caption("Distributed GPS Processing & Evaluation Dashboard")

st.divider()

# ---------------- API CONNECTION ----------------

def get_api_health():
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def get_trips():
    try:
        response = requests.get(
            f"{API_URL}/trips",
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


# ---------------- SIDEBAR ----------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "Available Trips",
        "Route Evaluation",
        "Route Map Demo"
    ]
)

st.sidebar.divider()
st.sidebar.caption("GPS Route Reconstruction | Device 4")

# ---------------- DASHBOARD ----------------

if page == "Dashboard":

    st.header("System Overview")

    health = get_api_health()
    trips_data = get_trips()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Backend Status",
            "Online" if health else "Offline"
        )

    with col2:
        st.metric(
            "Available Trips",
            trips_data["count"] if trips_data else 0
        )

    with col3:
        st.metric(
            "Evaluation Trips",
            len(EVAL_TRIPS)
        )

    st.divider()

    st.subheader("Distributed Device Status")

    devices = [
        ("Device 1", "Feature Extraction"),
        ("Device 2", "ML Model"),
        ("Device 3", "Route Reconstruction"),
        ("Device 4", "Backend Integration")
    ]

    for name, description in devices:
        if name == "Device 4":
            st.success(f"🟢 {name} — {description}")
        else:
            st.info(
                f"⏳ {name} — {description} (Integration pending)"
            )

    st.divider()

    st.subheader("System Information")

    if health:
        st.json(health)
    else:
        st.error("Cannot connect to FastAPI backend.")

# ---------------- AVAILABLE TRIPS ----------------

elif page == "Available Trips":

    st.header("Available GPS Trips")

    if st.button("Refresh Trips"):
        st.rerun()

    trips_data = get_trips()

    if trips_data is None:
        st.error(
            "Backend connection failed. Start FastAPI first."
        )

    else:
        st.metric("Total Trips", trips_data["count"])

        if trips_data["trips"]:
            st.dataframe(
                trips_data["trips"],
                use_container_width=True
            )
        else:
            st.warning(
                "No GPS .track files found in the data folder."
            )

# ---------------- ROUTE EVALUATION ----------------

elif page == "Route Evaluation":

    st.header("Route Evaluation")

    st.write(
        "Select one of the 10 predefined evaluation trips."
    )

    selected_trip = st.selectbox(
        "Select Evaluation Trip",
        EVAL_TRIPS
    )

    st.info(f"Selected Trip ID: {selected_trip}")

    if st.button("Evaluate Route", type="primary"):

        try:
            response = requests.post(
                f"{API_URL}/evaluate/{selected_trip}",
                timeout=30
            )

            response.raise_for_status()
            result = response.json()

            st.subheader("Evaluation Response")
            st.json(result)

            if result.get("status") == "pending":
                st.warning(
                    "Waiting for Device 1, Device 2 "
                    "and Device 3 integration."
                )

            elif result.get("status") == "not_found":
                st.warning(
                    "Trip data is not available yet. "
                    "The corresponding data files are required."
                )

        except requests.RequestException as error:
            st.error(f"Evaluation failed: {error}")

# ---------------- ROUTE MAP DEMO ----------------

elif page == "Route Map Demo":

    st.header("Interactive GPS Route Map")

    st.info(
        "This is a demonstration route using sample coordinates. "
        "Actual GPS route data will be connected after module integration."
    )

    demo_coordinates = [
        (12.9716, 77.5946),
        (12.9730, 77.5960),
        (12.9750, 77.5980),
        (12.9770, 77.6000),
        (12.9790, 77.6020)
    ]

    st.subheader("Sample Reconstructed Route")

    display_route_map(
        demo_coordinates,
        map_key="demo_route"
    )

    st.caption(
        "Sample coordinates are used for demonstration only."
    )

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "GPS Route Reconstruction | Distributed Computing "
    "| FastAPI + Streamlit + Folium"
)
