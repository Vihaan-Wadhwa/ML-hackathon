"""Top-K candidate arcs per GPS fix (Part 3). Vectorised: no Python loop over rows.

1. STRtree "dwithin" query for every fix at once -> (obs, arc) pairs within R.
2. Exact point->segment distance for every pair (numpy).
3. Sort by (obs, distance, arc_id) and keep the first K per obs (cand_rank 0..K-1).
4. Fixes with no pair: optional fallback to the nearest arc within fallback_radius_m;
   otherwise they are returned in `empty_obs` (never dropped).
Ties (an arc and its reverse twin, duplicate arcs) are broken by arc_id, deterministically.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import TripGeometry, point_segment_dist


@dataclass
class Candidates:
    obs_idx: np.ndarray     # int64, sorted
    arc_id: np.ndarray      # int64
    dist_m: np.ndarray      # float64
    frac: np.ndarray        # float64 in [0, 1]
    rank: np.ndarray        # int64
    n_per_obs: np.ndarray   # int64, len = n_obs
    empty_obs: np.ndarray   # obs with no candidate


def generate_candidates(geom: TripGeometry, px: np.ndarray, py: np.ndarray, k: int,
                        radius_m: float, fallback_radius_m: float = 0.0) -> Candidates:
    import shapely

    n = len(px)
    pts = shapely.points(px, py)
    obs, arc = geom.tree.query(pts, predicate="dwithin", distance=radius_m)
    if fallback_radius_m > radius_m:
        has = np.zeros(n, bool)
        has[obs] = True
        miss = np.flatnonzero(~has)
        if len(miss):
            fo, fa = geom.tree.query_nearest(pts[miss], max_distance=fallback_radius_m, all_matches=True)
            obs = np.r_[obs, miss[fo]]
            arc = np.r_[arc, fa]
    obs = obs.astype(np.int64)
    arc = arc.astype(np.int64)
    d, u = point_segment_dist(px[obs], py[obs], geom.ax[arc], geom.ay[arc], geom.bx[arc], geom.by[arc])
    order = np.lexsort((arc, d, obs))
    obs, arc, d, u = obs[order], arc[order], d[order], u[order]
    if len(obs):
        start = np.r_[0, np.flatnonzero(np.diff(obs)) + 1]
        counts = np.diff(np.r_[start, len(obs)])
        rank = np.arange(len(obs)) - np.repeat(start, counts)
    else:
        rank = np.zeros(0, np.int64)
    keep = rank < k
    obs, arc, d, u, rank = obs[keep], arc[keep], d[keep], u[keep], rank[keep]
    n_per = np.bincount(obs, minlength=n)
    return Candidates(obs, arc, d, u, rank, n_per, np.flatnonzero(n_per == 0))
