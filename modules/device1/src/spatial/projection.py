"""Metric projection (per trip).

The Part 1 scan showed the 90 trips are worldwide (23 UTM zones, 3 southern-hemisphere
trips, 2 trips straddling a zone edge), so one fixed CRS is wrong. Default: a transverse
Mercator centred on the trip's median node (k=1). Each network spans <= ~1.1 deg, so the
scale error is < ~2e-5 everywhere and there are no zone-edge or hemisphere cases. The
per-trip UTM EPSG is kept as an alternative (Part 2 cross-checks both against Haversine).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

EARTH_R = 6_371_008.8


def local_tmerc(lon0: float, lat0: float) -> str:
    return (f"+proj=tmerc +lat_0={lat0:.6f} +lon_0={lon0:.6f} +k=1 +x_0=0 +y_0=0 "
            "+ellps=WGS84 +units=m +no_defs")


def utm_epsg(lon: float, lat: float) -> int:
    zone = int(np.floor((lon + 180.0) / 6.0)) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


def trip_crs(node_lon, node_lat, kind: str = "tmerc") -> str:
    lon0, lat0 = float(np.median(node_lon)), float(np.median(node_lat))
    if kind == "tmerc":
        return local_tmerc(lon0, lat0)
    if kind == "utm":
        return f"EPSG:{utm_epsg(lon0, lat0)}"
    raise ValueError(kind)


@lru_cache(maxsize=256)
def get_transformer(crs: str):
    from pyproj import Transformer

    return Transformer.from_crs("EPSG:4326", crs, always_xy=True)


def project(lon, lat, crs: str) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised lon/lat (degrees) -> x/y (metres) in `crs`."""
    x, y = get_transformer(crs).transform(np.asarray(lon, float), np.asarray(lat, float))
    return np.asarray(x), np.asarray(y)


def haversine_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))
