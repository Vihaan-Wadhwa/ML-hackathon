"""Write outputs/sample_candidates.parquet: 50 synthetic rows in the frozen schema.

The values are DUMMY (trip_id 'DUMMY000'), seeded, and only exist so Device 2 can
build loaders/models against the real column names and dtypes before Part 4 lands.
Layout: obs 0-8 have 5 candidates (45 rows), obs 9 has 4 (49), obs 10 has none (50).
Obs 9's reference arc is not among its candidates (a "miss"); obs 10 has none at all;
obs 8 is UNLABELLED (reference > 50 m away), so its is_match is null on every row.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spatial.schema import NO_CANDIDATE, write_parquet  # noqa: E402


def make_sample(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_cands = [5] * 9 + [4, 0]
    rows = []
    for obs, k in enumerate(n_cands):
        ts = float(obs + 1) + (2.0 if obs >= 6 else 0.0)  # a 3 s gap before obs 6
        dt = np.nan if obs == 0 else (3.0 if obs == 6 else 1.0)
        speed = np.nan if obs == 0 else float(rng.uniform(0.0, 15.0))
        has_heading = int(obs != 0 and speed > 0.5)
        ref = 100 + obs // 3
        base = {
            "trip_id": "DUMMY000", "obs_idx": obs, "timestamp": ts,
            "speed_mps": speed, "dt_s": dt, "gap_flag": int(dt > 1.0) if obs else 0,
            "n_candidates": k, "has_heading": has_heading, "ref_arc_id": ref,
        }
        if k == 0:
            rows.append({**base, "arc_id": NO_CANDIDATE, "cand_rank": NO_CANDIDATE,
                         "perp_dist_m": np.nan, "bearing_diff_deg": np.nan,
                         "dist_to_start_m": np.nan, "dist_to_end_m": np.nan,
                         "frac_along": np.nan, "seg_len_m": np.nan, "is_match": 0})
            continue
        dists = np.sort(rng.uniform(0.5, 45.0, size=k))
        pool = np.setdiff1d(np.arange(90, 130), [ref])
        arcs = rng.choice(pool, size=k, replace=False)
        if obs != 9:  # put the reference arc in the set (rank 0 most of the time)
            arcs[0 if rng.random() < 0.8 else 1] = ref
        for rank in range(k):
            seg = float(rng.uniform(10.0, 200.0))
            frac = float(rng.uniform(0.0, 1.0))
            rows.append({**base, "arc_id": int(arcs[rank]), "cand_rank": rank,
                         "perp_dist_m": dists[rank],
                         "bearing_diff_deg": float(rng.uniform(0, 180)) if has_heading else np.nan,
                         "dist_to_start_m": float(np.hypot(frac * seg, dists[rank])),
                         "dist_to_end_m": float(np.hypot((1 - frac) * seg, dists[rank])),
                         "frac_along": frac, "seg_len_m": seg,
                         "is_match": int(arcs[rank] == ref)})
    df = pd.DataFrame(rows)
    df["is_match"] = df["is_match"].astype("Int8")
    df.loc[df["obs_idx"] == 8, "is_match"] = pd.NA
    assert len(df) == 50
    return df


if __name__ == "__main__":
    out = ROOT / "outputs" / "sample_candidates.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(make_sample(), out)
    print(f"wrote {out}")
