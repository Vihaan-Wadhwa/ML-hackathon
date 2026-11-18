"""Reference-route -> observation alignment (defines `ref_arc_id`, `is_match`, "miss").

Alignment rule (v1, frozen with the schema)
-------------------------------------------
`.route` is a chronological list of M directed arc_ids; `.track` has N >> M fixes, so
there is no row-to-row mapping. Each observation i is assigned a route POSITION p(i) in
0..M-1 by dynamic programming that minimises

    sum_i [ d(z_i, route[p(i)]) + w * (1 - cos(h_i, b_p(i))) / 2 ]  +  lam * sum_i (p(i) - p(i-1))

subject to p(i) >= p(i-1)  (monotonic: the vehicle never moves backwards along the route),

where d is the perpendicular distance in metres (per-trip local TM, clamped to the segment) from the
GPS fix to the arc's straight segment, h_i is the vehicle heading from the centred
difference z(i-1) -> z(i+1) and b the arc's direction. The heading term (w metres when
driving exactly against the arc) is applied only when that displacement >= min_move_m.
It is needed because 99.6% of arcs have a geometrically identical reverse twin: on a
U-turn the last outbound arc and first return arc tie on distance alone. Start and end positions are free and route
positions may be skipped (short arcs can be crossed between two 1 Hz fixes).
ref_arc_id(i) = route[p(i)]. Assigning positions rather than arc_ids keeps loops and
repeated arcs correct. lam (metres per skipped position) is a small tie-breaker that
prefers not jumping ahead; stability under lam is checked in the Part 1 scan.

Consequences
* LABELLED obs (align_dist_m <= noise_m = 50 m): is_match = 1 for the candidate row whose
  arc_id == ref_arc_id(i), 0 for the others.
* An observation is a MISS when it is labelled and ref_arc_id(i) is not in its candidate
  set. noise_m is fixed by this rule and independent of the candidate radius R, so a
  reference arc at 30-50 m with R = 30 m still counts as a miss.
* UNLABELLED obs (align_dist_m > noise_m): the reference does not describe where the fix
  is. Part 1 scan: 97% of these are > 50 m from EVERY route arc (the track continues past
  the route, e.g. train/00000054 after obs ~3000). They keep ref_arc_id (diagnostic) but
  get is_match = null on all rows, are excluded from training labels and from the miss
  rate, and their share is reported separately as label noise.
* Reference routes are not always continuous: 11/90 trips have 1-3 breaks of 10-600 m.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .geometry import point_segment_dist  # noqa: F401  (re-exported)
from .io import Trip
from .projection import project, trip_crs


@dataclass
class AlignConfig:
    lam_m: float = 1.0          # penalty (m) per route position advanced
    heading_w_m: float = 10.0   # cost (m) of travelling exactly against the arc direction
    min_move_m: float = 2.0     # centred displacement needed to trust the heading
    noise_m: float = 50.0       # align_dist_m above this = label noise
    chunk: int = 1024           # observations per distance block (memory bound)


def _unit_heading(px, py, min_move_m):
    """Centred-difference unit heading per fix; (0, 0) where movement < min_move_m."""
    n = len(px)
    if n < 2:
        return np.zeros(n), np.zeros(n)
    i0 = np.clip(np.arange(n) - 1, 0, n - 1)
    i1 = np.clip(np.arange(n) + 1, 0, n - 1)
    dx, dy = px[i1] - px[i0], py[i1] - py[i0]
    L = np.hypot(dx, dy)
    ok = L >= min_move_m
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ok, dx / L, 0.0), np.where(ok, dy / L, 0.0)


def route_segments_xy(trip: Trip, crs: str, node_xy=None):
    nx, ny = node_xy if node_xy is not None else project(
        trip.nodes["lon"].to_numpy(), trip.nodes["lat"].to_numpy(), crs)
    r = trip.route.to_numpy()
    f = trip.arcs["from_node"].to_numpy()[r]
    t = trip.arcs["to_node"].to_numpy()[r]
    return nx[f], ny[f], nx[t], ny[t]


def align_route(trip: Trip, cfg: AlignConfig = AlignConfig(), crs: str | None = None,
                track_xy=None, node_xy=None) -> pd.DataFrame:
    """Return per-observation alignment: obs_idx, route_pos, ref_arc_id, align_dist_m,
    nearest_route_dist_m (unconstrained lower bound), is_noise.
    track_xy / node_xy: already-projected coordinates (in `crs`) to avoid re-projecting."""
    if not trip.has_route:
        raise ValueError(f"trip {trip.trip_id} has no .route")
    crs = crs or trip_crs(trip.nodes["lon"].to_numpy(), trip.nodes["lat"].to_numpy())
    px, py = track_xy if track_xy is not None else project(
        trip.track["lon"].to_numpy(), trip.track["lat"].to_numpy(), crs)
    ax, ay, bx, by = route_segments_xy(trip, crs, node_xy)
    N, M = len(px), len(ax)
    hx, hy = _unit_heading(px, py, cfg.min_move_m)
    sl = np.hypot(bx - ax, by - ay)
    with np.errstate(invalid="ignore", divide="ignore"):
        ux, uy = np.where(sl > 0, (bx - ax) / sl, 0.0), np.where(sl > 0, (by - ay) / sl, 0.0)
    j = np.arange(M, dtype=np.float64)
    bp_dtype = np.int16 if M < np.iinfo(np.int16).max else np.int32
    back = np.empty((N, M), dtype=bp_dtype)   # N*M*2 bytes: 19k x 2.3k ~ 88 MB worst case
    nearest = np.empty(N)
    prev = None
    for s in range(0, N, cfg.chunk):
        e = min(s + cfg.chunk, N)
        C, _ = point_segment_dist(px[s:e, None], py[s:e, None], ax, ay, bx, by)  # (chunk, M)
        nearest[s:e] = C.min(axis=1)
        if cfg.heading_w_m:
            cos = hx[s:e, None] * ux + hy[s:e, None] * uy          # 0 when heading unknown
            has = (hx[s:e] != 0) | (hy[s:e] != 0)
            C = C + np.where(has[:, None], cfg.heading_w_m * (1.0 - cos) / 2.0, 0.0)
        for k in range(e - s):
            i = s + k
            if prev is None:
                cur = C[k].copy()          # free start position
                back[i] = -1
            else:
                E = prev - cfg.lam_m * j
                pm = np.minimum.accumulate(E)
                arg = np.maximum.accumulate(np.where(E == pm, np.arange(M), 0))
                cur = C[k] + cfg.lam_m * j + pm
                back[i] = arg
            prev = cur
    pos = np.empty(N, dtype=np.int64)
    pos[-1] = int(np.argmin(prev))
    for i in range(N - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    d, _ = point_segment_dist(px, py, ax[pos], ay[pos], bx[pos], by[pos])
    route = trip.route.to_numpy()
    return pd.DataFrame({
        "obs_idx": np.arange(N, dtype=np.int32),
        "route_pos": pos.astype(np.int32),
        "ref_arc_id": route[pos].astype(np.int32),
        "align_dist_m": d.astype(np.float32),
        "nearest_route_dist_m": nearest.astype(np.float32),
        "is_noise": (d > cfg.noise_m).astype(np.int8),
    })


def alignment_summary(al: pd.DataFrame, route_len: int, noise_m: float) -> dict:
    d = al["align_dist_m"].to_numpy()
    covered = np.unique(al["route_pos"].to_numpy())
    return {
        "align_dist_median_m": float(np.median(d)),
        "align_dist_p95_m": float(np.percentile(d, 95)),
        "align_dist_max_m": float(d.max()),
        "align_excess_over_nearest_p95_m": float(np.percentile(d - al["nearest_route_dist_m"], 95)),
        "obs_gt_30m": int((d > 30).sum()),
        "obs_gt_50m": int((d > 50).sum()),
        "noise_share": float((d > noise_m).mean()),
        "route_pos_covered_share": float(len(covered) / max(route_len, 1)),
        "first_route_pos": int(al["route_pos"].iloc[0]),
        "last_route_pos": int(al["route_pos"].iloc[-1]),
    }


def is_match_for(arc_id: np.ndarray, obs_idx: np.ndarray, al: pd.DataFrame,
                 noise_m: float = AlignConfig.noise_m) -> tuple[np.ndarray, np.ndarray]:
    """Candidate-row labels from an alignment. Returns (ref_arc_id int32, is_match
    float with NaN = null for unlabelled observations)."""
    ref = al["ref_arc_id"].to_numpy()[obs_idx]
    ok = al["align_dist_m"].to_numpy()[obs_idx] <= noise_m
    m = np.where(ok, (arc_id == ref).astype(float), np.nan)
    return ref.astype(np.int32), m


def miss_rate(cands: pd.DataFrame) -> dict:
    """Per-trip miss rate from a frozen-schema candidate frame (one trip or many).
    Unlabelled observations (is_match null) are excluded and counted separately."""
    per_obs = cands.groupby(["trip_id", "obs_idx"])["is_match"].agg(["max", "count"])
    labelled = per_obs["count"] > 0
    hit = per_obs.loc[labelled, "max"]
    return {"n_obs": int(len(per_obs)), "n_labelled": int(labelled.sum()),
            "unlabelled_share": float(1 - labelled.mean()) if len(per_obs) else float("nan"),
            "miss_rate": float(1.0 - hit.mean()) if len(hit) else float("nan")}
