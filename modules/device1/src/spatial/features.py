"""extract_features_for_trip: raw trip -> frozen-schema candidate DataFrame (Part 4).

Same function for training (labels from .route) and the API (no .route -> labels null).
Observation-level features use only the current and PREVIOUS fix (causal), so they are
identical at inference time.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .candidates import generate_candidates
from .config import SpatialConfig, load_config
from .geometry import build_geometry, compass_bearing
from .io import Trip, load_trip
from .labels import align_route, is_match_for
from .projection import project
from .schema import COLUMN_NAMES, NO_CANDIDATE


def wrap_angle_diff(a_deg, b_deg):
    """|a - b| wrapped to [0, 180] (359 vs 1 -> 2)."""
    return np.abs(np.mod(np.asarray(a_deg) - np.asarray(b_deg) + 180.0, 360.0) - 180.0)


def observation_features(px, py, t, min_move_m: float, gap_threshold_s: float) -> dict:
    n = len(px)
    dt = np.full(n, np.nan)
    disp = np.full(n, np.nan)
    heading = np.full(n, np.nan)
    if n > 1:
        dx, dy = np.diff(px), np.diff(py)
        dt[1:] = np.diff(t)
        disp[1:] = np.hypot(dx, dy)
        heading[1:] = compass_bearing(dx, dy)
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.where(dt > 0, disp / dt, np.nan)
    has_heading = (disp >= min_move_m)              # NaN (obs 0) -> False
    heading = np.where(has_heading, heading, np.nan)
    gap = np.nan_to_num(dt, nan=0.0) > gap_threshold_s
    return {"dt_s": dt, "speed_mps": speed, "heading_deg": heading,
            "has_heading": has_heading.astype(np.int8), "gap_flag": gap.astype(np.int8)}


def extract_features_for_trip(trip_id: str, raw_dir=None, cfg: SpatialConfig | None = None,
                              trip: Trip | None = None, with_labels: bool = True,
                              timings: dict | None = None, return_alignment: bool = False):
    """Return the candidate-feature DataFrame (frozen schema v1.0) for one trip.

    trip_id, raw_dir : as load_trip (raw_dir = partition dir or root). Pass `trip` to
                       skip loading. with_labels=False skips alignment even if a .route
                       exists (exactly what the API does for evaluator trips).
    """
    cfg = cfg or load_config()
    tm = {}
    t0 = time.perf_counter()
    trip = trip or load_trip(trip_id, raw_dir, read_route=with_labels)
    tm["load_s"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    geom = build_geometry(trip, crs_kind=cfg.crs_kind)
    px, py = project(trip.track["lon"].to_numpy(), trip.track["lat"].to_numpy(), geom.crs)
    tm["project_s"] = time.perf_counter() - t1

    t1 = time.perf_counter()
    _ = geom.tree
    tm["tree_s"] = time.perf_counter() - t1
    t1 = time.perf_counter()
    c = generate_candidates(geom, px, py, cfg.k, cfg.radius_m, cfg.fallback_radius_m)
    tm["query_s"] = time.perf_counter() - t1

    t1 = time.perf_counter()
    n = len(px)
    ob = observation_features(px, py, trip.track["t"].to_numpy(), cfg.min_move_m, cfg.gap_threshold_s)
    # candidate rows + one placeholder row per empty observation, sorted by obs
    e = c.empty_obs
    obs = np.r_[c.obs_idx, e]
    arc = np.r_[c.arc_id, np.full(len(e), NO_CANDIDATE)]
    rank = np.r_[c.rank, np.full(len(e), NO_CANDIDATE)]
    dist = np.r_[c.dist_m, np.full(len(e), np.nan)]
    frac = np.r_[c.frac, np.full(len(e), np.nan)]
    order = np.lexsort((rank, obs))
    obs, arc, rank, dist, frac = obs[order], arc[order], rank[order], dist[order], frac[order]
    valid = arc >= 0
    a = np.where(valid, arc, 0)

    def cand(v):
        return np.where(valid, v[a], np.nan)

    ax, ay, bx, by = cand(geom.ax), cand(geom.ay), cand(geom.bx), cand(geom.by)
    opx, opy = px[obs], py[obs]
    df = pd.DataFrame({
        "trip_id": trip.trip_id,
        "obs_idx": obs,
        "timestamp": trip.track["t"].to_numpy()[obs],
        "arc_id": arc,
        "cand_rank": rank,
        "perp_dist_m": dist,
        "bearing_diff_deg": wrap_angle_diff(ob["heading_deg"][obs], cand(geom.bearing_deg)),
        "speed_mps": ob["speed_mps"][obs],
        "dt_s": ob["dt_s"][obs],
        "gap_flag": ob["gap_flag"][obs],
        "dist_to_start_m": np.hypot(opx - ax, opy - ay),
        "dist_to_end_m": np.hypot(opx - bx, opy - by),
        "frac_along": frac,
        "seg_len_m": cand(geom.length_m),
        "n_candidates": c.n_per_obs[obs],
        "has_heading": ob["has_heading"][obs],
    })
    tm["features_s"] = time.perf_counter() - t1

    t1 = time.perf_counter()
    al = None
    if with_labels and trip.has_route:
        al = align_route(trip, cfg.align, crs=geom.crs, track_xy=(px, py),
                         node_xy=(geom.node_x, geom.node_y))
        ref, m = is_match_for(arc, obs, al, cfg.align.noise_m)
        df["is_match"] = pd.arrays.IntegerArray(np.nan_to_num(m).astype(np.int8), np.isnan(m))
        df["ref_arc_id"] = pd.arrays.IntegerArray(ref.astype(np.int32), np.zeros(len(df), bool))
    else:
        null = np.ones(len(df), bool)
        df["is_match"] = pd.arrays.IntegerArray(np.zeros(len(df), np.int8), null)
        df["ref_arc_id"] = pd.arrays.IntegerArray(np.zeros(len(df), np.int32), null)
    tm["labels_s"] = time.perf_counter() - t1
    tm["total_s"] = time.perf_counter() - t0
    tm["n_obs"], tm["n_rows"], tm["n_arcs"] = n, len(df), len(geom.length_m)
    if timings is not None:
        timings.update(tm)
    df = df[COLUMN_NAMES]
    return (df, al, geom) if return_alignment else df
