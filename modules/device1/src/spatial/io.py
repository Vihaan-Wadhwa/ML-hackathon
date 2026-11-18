"""Raw trip ingestion and per-trip sanity checks.

File formats (verified on the real files, see outputs/part1/):
  <trip>.track  lon lat t     whitespace-separated, headerless, one row per GPS fix
  <trip>.nodes  lon lat       row index = 0-based node_id
  <trip>.arcs   from to       row index = 0-based arc_id (directed)
  <trip>.route  arc_id        chronological reference arcs (optional; absent at inference)

The loader never reorders, drops or repairs rows. Validation only REPORTS problems so
decisions about them are taken (and documented) downstream.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PARTITIONS = ("train", "student_test", "evaluator_test")
_SPECS = {
    "track": (["lon", "lat", "t"], "float64"),
    "nodes": (["lon", "lat"], "float64"),
    "arcs": (["from_node", "to_node"], "int64"),
    "route": (["arc_id"], "int64"),
}


@dataclass
class Trip:
    trip_id: str
    path: Path
    track: pd.DataFrame          # columns lon, lat, t ; index = obs_idx
    nodes: pd.DataFrame          # columns lon, lat    ; index = node_id
    arcs: pd.DataFrame           # columns from_node, to_node ; index = arc_id
    route: pd.Series | None      # arc_id sequence, or None when no .route

    @property
    def has_route(self) -> bool:
        return self.route is not None


def find_trip_dir(trip_id: str, raw_dir) -> Path:
    """Accept either a partition dir (raw/train) or the root holding partitions."""
    raw_dir = Path(raw_dir)
    direct = raw_dir / trip_id
    if direct.is_dir():
        return direct
    hits = [raw_dir / p / trip_id for p in PARTITIONS if (raw_dir / p / trip_id).is_dir()]
    if len(hits) != 1:
        raise FileNotFoundError(f"trip {trip_id!r}: found {len(hits)} dirs under {raw_dir}")
    return hits[0]


def _read(path: Path, kind: str) -> pd.DataFrame:
    names, dtype = _SPECS[kind]
    df = pd.read_csv(path, sep=r"\s+", header=None, dtype="float64", engine="c")
    if df.shape[1] != len(names):
        raise ValueError(f"{path.name}: expected {len(names)} columns, got {df.shape[1]}")
    df.columns = names
    if dtype == "int64":
        if df.isna().any().any():
            raise ValueError(f"{path.name}: NaN in integer file")
        if not np.all(np.mod(df.to_numpy(), 1) == 0):
            raise ValueError(f"{path.name}: non-integer ids")
        df = df.astype("int64")
    df.index.name = {"track": "obs_idx", "nodes": "node_id", "arcs": "arc_id", "route": "route_pos"}[kind]
    return df


def load_trip(trip_id: str, raw_dir, require_route: bool = False, read_route: bool = True) -> Trip:
    """read_route=False never opens .route (inference / evaluator trips: labels stay unread)."""
    trip_id = str(trip_id)
    d = find_trip_dir(trip_id, raw_dir)
    files = {k: d / f"{trip_id}.{k}" for k in _SPECS}
    for k in ("track", "nodes", "arcs"):
        if not files[k].is_file():
            raise FileNotFoundError(files[k])
    route = None
    if not read_route:
        if require_route:
            raise ValueError("require_route and read_route=False are contradictory")
    elif files["route"].is_file() and files["route"].stat().st_size > 0:
        route = _read(files["route"], "route")["arc_id"]
    elif require_route:
        raise FileNotFoundError(files["route"])
    return Trip(trip_id, d, _read(files["track"], "track"), _read(files["nodes"], "nodes"),
                _read(files["arcs"], "arcs"), route)


def list_trips(raw_root) -> pd.DataFrame:
    """Directory listing -> (trip_id, partition). Partition = parent folder name."""
    raw_root = Path(raw_root)
    rows = []
    for part in PARTITIONS:
        pdir = raw_root / part
        if pdir.is_dir():
            rows += [(p.name, part) for p in sorted(pdir.iterdir()) if p.is_dir()]
    return pd.DataFrame(rows, columns=["trip_id", "partition"])


def file_sha1(path: Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- checks
def utm_zone(lon: np.ndarray) -> np.ndarray:
    return (np.floor((np.asarray(lon) + 180.0) / 6.0) + 1).astype(int)


def validate_trip(trip: Trip) -> dict:
    """Return a flat dict of sanity metrics for one trip (no mutation)."""
    tr, nd, ar = trip.track, trip.nodes, trip.arcs
    n_nodes = len(nd)
    out: dict = {"trip_id": trip.trip_id, "n_obs": len(tr), "n_nodes": n_nodes,
                 "n_arcs": len(ar), "has_route": trip.has_route}

    # --- track
    out["track_nan_rows"] = int(tr.isna().any(axis=1).sum())
    lon, lat, t = tr["lon"].to_numpy(), tr["lat"].to_numpy(), tr["t"].to_numpy()
    out["track_lon_out_of_range"] = int(((lon < -180) | (lon > 180)).sum())
    out["track_lat_out_of_range"] = int(((lat < -90) | (lat > 90)).sum())
    out["track_zero_coords"] = int(((lon == 0) & (lat == 0)).sum())
    for k, v in (("lon", lon), ("lat", lat)):
        out[f"track_{k}_min"], out[f"track_{k}_max"] = float(np.nanmin(v)), float(np.nanmax(v))
    out["t_first"], out["t_last"] = float(t[0]), float(t[-1])
    dt = np.diff(t)
    out["dt_negative"] = int((dt < 0).sum())
    out["dt_zero"] = int((dt == 0).sum())
    out["dt_eq_1_share"] = float(np.mean(np.isclose(dt, 1.0))) if len(dt) else np.nan
    out["dt_gt_1"] = int((dt > 1 + 1e-9).sum())
    out["dt_gt_10"] = int((dt > 10).sum())
    out["dt_max"] = float(dt.max()) if len(dt) else np.nan
    out["dt_non_integer"] = int((~np.isclose(dt, np.round(dt))).sum())
    out["t_non_integer"] = int((~np.isclose(t, np.round(t))).sum())
    out["dup_consecutive_fix"] = int(((np.diff(lon) == 0) & (np.diff(lat) == 0)).sum())

    # --- nodes
    nlon, nlat = nd["lon"].to_numpy(), nd["lat"].to_numpy()
    out["nodes_nan_rows"] = int(nd.isna().any(axis=1).sum())
    out["nodes_out_of_range"] = int(((np.abs(nlon) > 180) | (np.abs(nlat) > 90)).sum())
    out["dup_node_coords"] = int(nd.duplicated().sum())
    for k, v in (("lon", nlon), ("lat", nlat)):
        out[f"nodes_{k}_min"], out[f"nodes_{k}_max"] = float(np.nanmin(v)), float(np.nanmax(v))
    # Coordinate-order heuristic: if columns were (lat, lon) the "lat" column would hold
    # the larger magnitude here (77 vs 43). Reported, not assumed.
    out["track_inside_node_bbox_share"] = float(np.mean(
        (lon >= nlon.min()) & (lon <= nlon.max()) & (lat >= nlat.min()) & (lat <= nlat.max())))
    zones = np.unique(np.concatenate([utm_zone(nlon), utm_zone(lon[~np.isnan(lon)])]))
    out["utm_zones"] = ";".join(map(str, zones))

    # --- arcs
    f, to = ar["from_node"].to_numpy(), ar["to_node"].to_numpy()
    bad = (f < 0) | (f >= n_nodes) | (to < 0) | (to >= n_nodes)
    out["arcs_bad_endpoint"] = int(bad.sum())
    out["arcs_self_loop"] = int((f == to).sum())
    out["arcs_duplicate"] = int(ar.duplicated().sum())
    key = f * n_nodes + to
    rkey = to * n_nodes + f
    out["arcs_with_reverse_share"] = float(np.isin(key, rkey).mean()) if len(ar) else np.nan
    used = np.zeros(n_nodes, bool)
    good = ~bad
    used[f[good]] = True
    used[to[good]] = True
    out["isolated_nodes"] = int((~used).sum())
    out.update(_components(n_nodes, f[good], to[good]))

    # --- route
    if trip.has_route:
        r = trip.route.to_numpy()
        out["route_len"] = len(r)
        out["route_bad_id"] = int(((r < 0) | (r >= len(ar))).sum())
        ok = r[(r >= 0) & (r < len(ar))]
        out["route_repeat_consecutive"] = int((np.diff(ok) == 0).sum())
        out["route_unique_arcs"] = int(len(np.unique(ok)))
        if len(ok) > 1:
            conn = to[ok[:-1]] == f[ok[1:]]
            out["route_breaks"] = int((~conn).sum())
            out["route_connected_share"] = float(conn.mean())
        else:
            out["route_breaks"], out["route_connected_share"] = 0, np.nan
        out["obs_per_route_arc"] = len(tr) / max(len(r), 1)
    return out


def _components(n: int, f: np.ndarray, t: np.ndarray) -> dict:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    g = coo_matrix((np.ones(len(f), np.int8), (f, t)), shape=(n, n)).tocsr()
    nw, lw = connected_components(g, directed=True, connection="weak")
    ns, ls = connected_components(g, directed=True, connection="strong")
    return {"weak_components": int(nw), "largest_weak_share": float(np.bincount(lw).max() / n),
            "strong_components": int(ns), "largest_strong_share": float(np.bincount(ls).max() / n)}
