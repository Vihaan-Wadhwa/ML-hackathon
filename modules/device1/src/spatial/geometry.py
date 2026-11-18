"""Per-trip metric geometry: projected nodes/arcs, arc table and STRtree (Part 2/3)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .io import Trip
from .projection import project, trip_crs


def point_segment_dist(px, py, ax, ay, bx, by):
    """Broadcasting point->segment distance and clamped projection fraction in [0, 1]."""
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    with np.errstate(invalid="ignore", divide="ignore"):
        u = ((px - ax) * vx + (py - ay) * vy) / L2
    u = np.where(L2 > 0, np.clip(u, 0.0, 1.0), 0.0)
    dx, dy = px - (ax + u * vx), py - (ay + u * vy)
    return np.hypot(dx, dy), u


def compass_bearing(dx, dy):
    """Grid bearing in degrees clockwise from north, [0, 360)."""
    return np.mod(np.degrees(np.arctan2(dx, dy)), 360.0)


@dataclass
class TripGeometry:
    trip_id: str
    crs: str
    node_x: np.ndarray
    node_y: np.ndarray
    from_node: np.ndarray
    to_node: np.ndarray
    ax: np.ndarray
    ay: np.ndarray
    bx: np.ndarray
    by: np.ndarray
    length_m: np.ndarray
    bearing_deg: np.ndarray
    _tree: object = None

    @property
    def tree(self):
        """shapely STRtree over arc LineStrings (built lazily, once per trip)."""
        if self._tree is None:
            import shapely

            coords = np.stack([np.c_[self.ax, self.ay], np.c_[self.bx, self.by]], axis=1)
            self._tree = shapely.STRtree(shapely.linestrings(coords))
        return self._tree


def build_geometry(trip: Trip, crs: str | None = None, crs_kind: str = "tmerc") -> TripGeometry:
    nlon, nlat = trip.nodes["lon"].to_numpy(), trip.nodes["lat"].to_numpy()
    crs = crs or trip_crs(nlon, nlat, crs_kind)
    nx, ny = project(nlon, nlat, crs)          # one vectorised call for all nodes
    f = trip.arcs["from_node"].to_numpy()
    t = trip.arcs["to_node"].to_numpy()
    ax, ay, bx, by = nx[f], ny[f], nx[t], ny[t]
    return TripGeometry(trip.trip_id, crs, nx, ny, f, t, ax, ay, bx, by,
                        np.hypot(bx - ax, by - ay), compass_bearing(bx - ax, by - ay))


def build_geometry_table(trip: Trip, geom: TripGeometry | None = None) -> pd.DataFrame:
    """One row per arc: the Device 3/4 lookup table (network_geometry.parquet)."""
    g = geom or build_geometry(trip)
    nlon, nlat = trip.nodes["lon"].to_numpy(), trip.nodes["lat"].to_numpy()
    f, t = g.from_node, g.to_node
    return pd.DataFrame({
        "trip_id": trip.trip_id,
        "arc_id": np.arange(len(f), dtype=np.int32),
        "from_node": f.astype(np.int32), "to_node": t.astype(np.int32),
        "from_lon": nlon[f], "from_lat": nlat[f], "to_lon": nlon[t], "to_lat": nlat[t],
        "from_x": g.ax.astype(np.float32), "from_y": g.ay.astype(np.float32),
        "to_x": g.bx.astype(np.float32), "to_y": g.by.astype(np.float32),
        "length_m": g.length_m.astype(np.float32), "bearing_deg": g.bearing_deg.astype(np.float32),
    })
