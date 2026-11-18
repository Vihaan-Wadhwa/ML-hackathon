"""Road network visualization module for PathSnap Device 3.

Task 1: Step 5 - Road Network Visualization.
"""

import math
from pathlib import Path
from typing import Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np


def visualize_road_graph(
    road_graph: nx.MultiDiGraph,
    output_path: Union[str, Path],
    title: Optional[str] = None,
    max_arrows: int = 400,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    dpi: int = 200,
) -> Path:
    """Visualize a road network MultiDiGraph using geographic coordinates.

    Uses Matplotlib LineCollection for vectorized high-performance rendering of
    large road networks (over 150,000 edges) without force-directed layout distortion.

    Parameters
    ----------
    road_graph : nx.MultiDiGraph
        Directed multigraph with node attributes ('longitude', 'latitude')
        and edge attributes ('arc_id', 'length_m').
    output_path : Union[str, Path]
        Target file path for the exported PNG image.
    title : Optional[str], default=None
        Custom plot title. If None, an informative default is generated.
    max_arrows : int, default=400
        Maximum number of representative direction arrows to render.
    bbox : Optional[Tuple[float, float, float, float]], default=None
        Optional geographic bounding box filter (min_lon, min_lat, max_lon, max_lat)
        for detailed local subgraph inspection.
    dpi : int, default=200
        Resolution of exported image in dots per inch.

    Returns
    -------
    Path
        Resolved Path to the saved image file.

    Raises
    ------
    ValueError
        If the input graph is empty or missing geographic coordinates.
    """
    if road_graph.number_of_nodes() == 0:
        raise ValueError("Cannot visualize an empty road graph (0 nodes).")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Extract node coordinates
    node_coords = {}
    lons = []
    lats = []
    for n, data in road_graph.nodes(data=True):
        if "longitude" not in data or "latitude" not in data:
            raise ValueError(f"Node {n} is missing 'longitude' or 'latitude' attributes.")
        lon = float(data["longitude"])
        lat = float(data["latitude"])
        node_coords[n] = (lon, lat)
        lons.append(lon)
        lats.append(lat)

    lons_arr = np.array(lons)
    lats_arr = np.array(lats)

    # Apply bounding box filter if specified
    if bbox is not None:
        b_min_lon, b_min_lat, b_max_lon, b_max_lat = bbox
        active_nodes = {
            n
            for n, (lon, lat) in node_coords.items()
            if b_min_lon <= lon <= b_max_lon and b_min_lat <= lat <= b_max_lat
        }
    else:
        active_nodes = set(node_coords.keys())

    # Build line segments for LineCollection
    segments = []
    arrow_starts_x = []
    arrow_starts_y = []
    arrow_dirs_x = []
    arrow_dirs_y = []

    # Collect valid edge coordinates
    candidate_edges = []
    for u, v, k, data in road_graph.edges(keys=True, data=True):
        if u in active_nodes and v in active_nodes:
            p1 = node_coords[u]
            p2 = node_coords[v]
            segments.append([p1, p2])
            candidate_edges.append((p1, p2))

    # Sample representative direction arrows
    total_edges = len(candidate_edges)
    if total_edges > 0 and max_arrows > 0:
        step = max(1, total_edges // max_arrows)
        for i in range(0, total_edges, step):
            p1, p2 = candidate_edges[i]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dist_deg = math.hypot(dx, dy)
            # Avoid placing arrows on zero-length or microscopic segments
            if dist_deg > 1e-6:
                # Place arrow midway along the edge
                mid_x = (p1[0] + p2[0]) / 2.0
                mid_y = (p1[1] + p2[1]) / 2.0
                arrow_starts_x.append(mid_x)
                arrow_starts_y.append(mid_y)
                arrow_dirs_x.append(dx)
                arrow_dirs_y.append(dy)

    # Initialize figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # 1. Render road segments using LineCollection (high-performance batch rendering)
    if segments:
        lc = LineCollection(
            segments,
            colors="#2563EB",  # Vibrant blue
            linewidths=0.6 if total_edges > 10000 else 1.2,
            alpha=0.65,
            zorder=1,
            label=f"Road Arcs ({total_edges:,})",
        )
        ax.add_collection(lc)

    # 2. Render road nodes
    if active_nodes:
        active_lons = [node_coords[n][0] for n in active_nodes]
        active_lats = [node_coords[n][1] for n in active_nodes]
        node_size = 0.5 if len(active_nodes) > 10000 else 4.0
        ax.scatter(
            active_lons,
            active_lats,
            s=node_size,
            c="#0F172A",  # Dark slate
            alpha=0.5,
            zorder=2,
            label=f"Road Nodes ({len(active_nodes):,})",
        )

    # 3. Render sampled direction arrows
    if arrow_starts_x:
        ax.quiver(
            arrow_starts_x,
            arrow_starts_y,
            arrow_dirs_x,
            arrow_dirs_y,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="#DC2626",  # Red arrows
            width=0.002,
            headwidth=3.5,
            headlength=4.5,
            headaxislength=3.5,
            alpha=0.8,
            zorder=3,
            label=f"Direction Indicators (Sampled {len(arrow_starts_x):,})",
        )

    # Configure axes and coordinate bounds
    if bbox is not None:
        plot_min_lon, plot_min_lat, plot_max_lon, plot_max_lat = bbox
    elif active_nodes:
        min_lon = min(active_lons)
        max_lon = max(active_lons)
        min_lat = min(active_lats)
        max_lat = max(active_lats)
        lon_pad = max((max_lon - min_lon) * 0.04, 0.001)
        lat_pad = max((max_lat - min_lat) * 0.04, 0.001)
        plot_min_lon = min_lon - lon_pad
        plot_max_lon = max_lon + lon_pad
        plot_min_lat = min_lat - lat_pad
        plot_max_lat = max_lat + lat_pad
    else:
        plot_min_lon, plot_max_lon = -180.0, 180.0
        plot_min_lat, plot_max_lat = -90.0, 90.0

    ax.set_xlim(plot_min_lon, plot_max_lon)
    ax.set_ylim(plot_min_lat, plot_max_lat)

    # Mercator aspect ratio adjustment: 1 / cos(latitude)
    mid_lat = (plot_min_lat + plot_max_lat) / 2.0
    cos_lat = math.cos(math.radians(mid_lat))
    if cos_lat > 1e-4:
        ax.set_aspect(1.0 / cos_lat)

    # Labels and annotations
    ax.set_xlabel("Longitude (Degrees East)", fontsize=11, fontweight="medium")
    ax.set_ylabel("Latitude (Degrees North)", fontsize=11, fontweight="medium")

    default_title = (
        f"Road Network Topology — Directed MultiDiGraph\n"
        f"({road_graph.number_of_nodes():,} Nodes, {road_graph.number_of_edges():,} Arcs)"
    )
    ax.set_title(title or default_title, fontsize=13, fontweight="bold", pad=12)

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, color="#94A3B8")

    # Custom legend proxies to ensure clean legend display regardless of scatter size
    legend_elements = [
        Line2D([0], [0], color="#2563EB", lw=2, label="Road Segment (Arc)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#0F172A", markersize=6, label="Road Node (Intersection)"),
    ]
    if arrow_starts_x:
        legend_elements.append(
            Line2D([0], [0], marker=">", color="w", markerfacecolor="#DC2626", markersize=8, label="Sampled Flow Direction")
        )

    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)

    return out_file
