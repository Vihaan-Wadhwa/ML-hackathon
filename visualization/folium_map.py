
import folium
import streamlit as st
from streamlit_folium import st_folium


def display_route_map(coordinates, map_key="route_map"):
    """
    Display a GPS route on an interactive Folium map.

    coordinates format:
    [
        (latitude, longitude),
        (latitude, longitude)
    ]
    """

    if not coordinates or len(coordinates) < 2:
        st.info("At least two GPS coordinates are required to display a route.")
        return

    route_map = folium.Map(
        location=coordinates[0],
        zoom_start=14,
        tiles="OpenStreetMap"
    )

    folium.PolyLine(
        locations=coordinates,
        color="#3388ff",
        weight=5,
        opacity=0.85,
        tooltip="Reconstructed GPS Route"
    ).add_to(route_map)

    folium.Marker(
        coordinates[0],
        tooltip="Start",
        icon=folium.Icon(color="green", icon="play")
    ).add_to(route_map)

    folium.Marker(
        coordinates[-1],
        tooltip="End",
        icon=folium.Icon(color="red", icon="stop")
    ).add_to(route_map)

    route_map.fit_bounds(coordinates)

    st_folium(
        route_map,
        width=None,
        height=500,
        key=map_key
    )
