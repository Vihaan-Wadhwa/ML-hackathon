"""Part 5: miss-rate K/R sweep + timing benchmark (HEAVY: run on a remote CPU, not the demo laptop).

Uses train + student_test only (no evaluator data exists here or is ever read).
1. Wide run per trip (K_MAX, R_MAX): for every labelled fix, record the rank and
   distance of its reference arc among candidates. Because candidates are ranked by
   distance, the fix is a hit at (K, R) iff ref_rank < K and ref_dist <= R, so the whole
   grid follows exactly from one run.
2. Timing of extract_features_for_trip with the configured K/R, per trip: inference
   path (no labels, as in the API) and full path (with labels).
Outputs (OUT_DIR): obs_ref_ranks.parquet, kr_grid.csv, miss_by_trip.csv, timing.csv,
validation_summary.json
Usage: python scripts/validate_spatial.py RAW_ROOT OUT_DIR
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spatial.config import load_config  # noqa: E402
from spatial.features import extract_features_for_trip  # noqa: E402
from spatial.io import list_trips, load_trip  # noqa: E402

K_MAX, R_MAX = 16, 100.0
K_GRID = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]
R_GRID = [20.0, 30.0, 40.0, 50.0, 60.0, 75.0, 100.0]


def ref_ranks(df: pd.DataFrame, al: pd.DataFrame, n_obs: int) -> pd.DataFrame:
    """Per observation: ref rank in the wide candidate set (-1 = not within K_MAX/R_MAX),
    exact distance to the ref arc (from the alignment) and counts within each R."""
    lab = df.groupby("obs_idx")["is_match"].count() > 0            # labelled obs
    hit = df[df["is_match"] == 1]
    out = pd.DataFrame({"obs_idx": np.arange(n_obs)})
    out["labelled"] = lab.reindex(out.obs_idx, fill_value=False).to_numpy()
    out["ref_rank"] = hit.set_index("obs_idx")["cand_rank"].reindex(out.obs_idx).fillna(-1).astype(int).to_numpy()
    out["ref_dist_m"] = al["align_dist_m"].to_numpy()
    valid = df[df["arc_id"] >= 0]
    for r in R_GRID:
        cnt = valid[valid["perp_dist_m"] <= r].groupby("obs_idx").size()
        out[f"n_within_{int(r)}"] = cnt.reindex(out.obs_idx, fill_value=0).to_numpy()
    return out


def main(raw_root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    wide = cfg.with_(k=K_MAX, radius_m=R_MAX, fallback_radius_m=0.0)
    splits = list_trips(raw_root)
    obs_tables, timing = [], []
    t0 = time.time()
    # warm-up so the first timed trip does not pay import / JIT-cache costs
    tid0, part0 = splits.iloc[0]
    extract_features_for_trip(tid0, raw_root / part0, cfg=cfg, with_labels=False)
    for n, (tid, part) in enumerate(splits.itertuples(index=False), 1):
        trip = load_trip(tid, raw_root / part)
        df, al, _ = extract_features_for_trip(tid, cfg=wide, trip=trip, return_alignment=True)
        rr = ref_ranks(df, al, len(trip.track))
        rr.insert(0, "trip_id", tid)
        rr.insert(0, "partition", part)
        obs_tables.append(rr)
        for mode, lab in (("inference", False), ("full", True)):
            tm = {}
            extract_features_for_trip(tid, raw_root / part, cfg=cfg, with_labels=lab, timings=tm)
            timing.append({"partition": part, "trip_id": tid, "mode": mode, **tm})
        print(f"[{n}/{len(splits)}] {part}/{tid} {time.time() - t0:.1f}s", flush=True)

    obs = pd.concat(obs_tables, ignore_index=True)
    obs.to_parquet(out_dir / "obs_ref_ranks.parquet", index=False)
    lab = obs[obs.labelled]

    rows = []
    for part, g_all in [("train", lab[lab.partition == "train"]),
                        ("student_test", lab[lab.partition == "student_test"]),
                        ("train+student_test", lab)]:
        base = obs[obs.partition.isin(["train", "student_test"] if "+" in part else [part])]
        for r in R_GRID:
            for k in K_GRID:
                hit = (g_all.ref_rank >= 0) & (g_all.ref_rank < k) & (g_all.ref_dist_m <= r)
                per_trip = hit.groupby(g_all.trip_id).mean()
                n_rows = np.maximum(np.minimum(base[f"n_within_{int(r)}"], k), 1).sum()
                rows.append({"partition": part, "k": k, "radius_m": r,
                             "miss_rate_pooled": float(1 - hit.mean()),
                             "miss_rate_trip_mean": float(1 - per_trip.mean()),
                             "miss_rate_trip_max": float(1 - per_trip.min()),
                             "trips_over_5pct": int((1 - per_trip > 0.05).sum()),
                             "miss_from_radius": float((g_all.ref_dist_m > r).mean()),
                             "rows": int(n_rows), "rows_per_obs": float(n_rows / len(base)),
                             "positive_rate": float(hit.sum() / n_rows)})
    grid = pd.DataFrame(rows)
    grid.to_csv(out_dir / "kr_grid.csv", index=False)

    # per-trip miss rate at the configured K/R, with the cause of each miss
    k, r = cfg.k, cfg.radius_m
    hit = (lab.ref_rank >= 0) & (lab.ref_rank < k) & (lab.ref_dist_m <= r)
    ref_far = lab.ref_dist_m > r       # ref beyond R: no K can recover it
    by_trip = pd.DataFrame({"partition": lab.partition, "trip_id": lab.trip_id, "miss": ~hit,
                            "miss_radius": ~hit & ref_far, "miss_rank": ~hit & ~ref_far}) \
        .groupby(["partition", "trip_id"]).mean()
    cnt = obs.groupby(["partition", "trip_id"]).agg(n_obs=("obs_idx", "size"),
                                                    unlabelled_share=("labelled", lambda s: 1 - s.mean()))
    by_trip = cnt.join(by_trip).rename(columns={"miss": "miss_rate"}).reset_index()
    by_trip.sort_values("miss_rate", ascending=False).to_csv(out_dir / "miss_by_trip.csv", index=False)

    tdf = pd.DataFrame(timing)
    tdf.to_csv(out_dir / "timing.csv", index=False)

    def tstats(mode):
        s = tdf[tdf["mode"] == mode]["total_s"]
        return {"median_s": float(s.median()), "p95_s": float(s.quantile(0.95)),
                "max_s": float(s.max()), "trips_over_1_5s": int((s > 1.5).sum()),
                "slowest_trip": tdf[tdf["mode"] == mode].sort_values("total_s").iloc[-1]["trip_id"]}

    def at(part):
        g = grid[(grid.partition == part) & (grid.k == k) & (grid.radius_m == r)].iloc[0]
        return g[["miss_rate_pooled", "miss_rate_trip_mean", "miss_rate_trip_max",
                  "trips_over_5pct", "rows_per_obs", "positive_rate"]].to_dict()

    summary = {
        "config": cfg.to_dict(), "machine": platform.platform(), "cpu": platform.processor(),
        "labelled_obs": {p: int((lab.partition == p).sum()) for p in ("train", "student_test")},
        "at_config": {p: at(p) for p in ("train", "student_test", "train+student_test")},
        "miss_cause_at_config_pooled": {"ref_beyond_radius": float((~hit & ref_far).mean()),
                                        "ref_beyond_rank_k": float((~hit & ~ref_far).mean())},
        "timing_inference": tstats("inference"), "timing_full": tstats("full"),
        "seconds": time.time() - t0,
    }
    (out_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
