"""Part 1 full-dataset scan (HEAVY: run on a remote CPU, not the demo laptop).

For every trip in train/ and student_test/:
  inventory (files, bytes, rows, sha1), sanity checks (io.validate_trip),
  network sharing (hash of .nodes+.arcs), route connectivity, and the route->obs
  alignment with a lam stability check.
Outputs (OUT_DIR, default outputs/part1/), every row tagged with its partition:
  inventory.csv, trip_checks.csv, alignment_summary.csv, alignment_obs.parquet,
  network_groups.csv, dataset_summary.json, errors.csv
Usage: python scripts/eda_part1.py RAW_ROOT [OUT_DIR]
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spatial.io import file_sha1, find_trip_dir, list_trips, load_trip, validate_trip  # noqa: E402
from spatial.labels import AlignConfig, align_route, alignment_summary  # noqa: E402
from spatial.projection import trip_crs  # noqa: E402

# stability check around the default rule (first entry = default AlignConfig())
VARIANTS = [dict(lam_m=1.0, heading_w_m=10.0), dict(lam_m=0.0, heading_w_m=10.0),
            dict(lam_m=5.0, heading_w_m=10.0), dict(lam_m=1.0, heading_w_m=0.0),
            dict(lam_m=1.0, heading_w_m=20.0)]
NOISE_M = 50.0


def main(raw_root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = list_trips(raw_root)
    assert not splits["trip_id"].duplicated().any(), "trip in two partitions"
    splits.to_csv(out_dir / "splits.csv", index=False)
    inv, checks, aligns, obs_tables, errors = [], [], [], [], []
    t0 = time.time()
    for n, (tid, part) in enumerate(splits.itertuples(index=False), 1):
        try:
            d = find_trip_dir(tid, raw_root / part)
            for p in sorted(d.iterdir()):
                ext = p.suffix.lstrip(".")
                rows = sum(1 for _ in open(p, "rb")) if p.is_file() else None
                inv.append({"partition": part, "trip_id": tid, "file": p.name, "ext": ext,
                            "bytes": p.stat().st_size, "rows": rows, "sha1": file_sha1(p)})
            trip = load_trip(tid, raw_root / part)
            c = validate_trip(trip)
            c["partition"] = part
            c["crs"] = trip_crs(trip.nodes["lon"].to_numpy(), trip.nodes["lat"].to_numpy())
            c["utm_crs_alt"] = trip_crs(trip.nodes["lon"].to_numpy(), trip.nodes["lat"].to_numpy(), "utm")
            c["network_sha1"] = file_sha1(d / f"{tid}.nodes") + ":" + file_sha1(d / f"{tid}.arcs")
            checks.append(c)
            if trip.has_route and c.get("route_bad_id", 0) == 0:
                base = None
                for vi, kw in enumerate(VARIANTS):
                    ts = time.time()
                    al = align_route(trip, AlignConfig(noise_m=NOISE_M, **kw), crs=c["crs"])
                    s = {"partition": part, "trip_id": tid, "variant": vi, "is_default": vi == 0,
                         **kw, "align_seconds": time.time() - ts,
                         **alignment_summary(al, len(trip.route), NOISE_M)}
                    if vi == 0:
                        base = al
                        al.insert(0, "trip_id", tid)
                        al.insert(0, "partition", part)
                        obs_tables.append(al)
                    aligns.append((s, al))
                # stability: share of obs whose ref arc differs from the default rule
                for s, al in aligns[-len(VARIANTS):]:
                    s["changed_vs_default_share"] = float(
                        (al["ref_arc_id"].to_numpy() != base["ref_arc_id"].to_numpy()).mean())
        except Exception as e:  # keep scanning; report every failure
            errors.append({"partition": part, "trip_id": tid, "error": repr(e),
                           "trace": traceback.format_exc()})
        print(f"[{n}/{len(splits)}] {part}/{tid}  {time.time() - t0:.1f}s", flush=True)

    pd.DataFrame(inv).to_csv(out_dir / "inventory.csv", index=False)
    ck = pd.DataFrame(checks)
    lead = ["partition", "trip_id"]
    ck = ck[lead + [c for c in ck.columns if c not in lead]]
    ck.to_csv(out_dir / "trip_checks.csv", index=False)
    asum = pd.DataFrame([s for s, _ in aligns])
    asum.to_csv(out_dir / "alignment_summary.csv", index=False)
    if obs_tables:
        pd.concat(obs_tables, ignore_index=True).to_parquet(out_dir / "alignment_obs.parquet", index=False)
    pd.DataFrame(errors, columns=["partition", "trip_id", "error", "trace"]).to_csv(
        out_dir / "errors.csv", index=False)

    groups = ck.groupby("network_sha1").agg(n_trips=("trip_id", "size"),
                                            trips=("trip_id", lambda s: ";".join(s)),
                                            partitions=("partition", lambda s: ";".join(sorted(set(s)))))
    groups.sort_values("n_trips", ascending=False).to_csv(out_dir / "network_groups.csv")

    default = asum[asum["is_default"]] if len(asum) else asum
    summary = {
        "trips_per_partition": splits["partition"].value_counts().to_dict(),
        "evaluator_trips_present": int((splits["partition"] == "evaluator_test").sum()),
        "trips_scanned_ok": len(ck), "trips_failed": len(errors),
        "files_per_ext": pd.DataFrame(inv).groupby("ext").size().to_dict() if inv else {},
        "distinct_networks": int(ck["network_sha1"].nunique()),
        "utm_zones_seen": sorted(set(";".join(ck["utm_zones"]).split(";"))),
        "total_obs": int(ck["n_obs"].sum()),
        "obs_per_trip": ck["n_obs"].describe().round(1).to_dict(),
        "dt_eq_1_share_overall": float(np.average(ck["dt_eq_1_share"], weights=ck["n_obs"] - 1)),
        "trips_with_negative_dt": int((ck["dt_negative"] > 0).sum()),
        "trips_with_zero_dt": int((ck["dt_zero"] > 0).sum()),
        "trips_with_gap_gt_10s": int((ck["dt_gt_10"] > 0).sum()),
        "max_dt_s": float(ck["dt_max"].max()),
        "trips_with_nan_track_rows": int((ck["track_nan_rows"] > 0).sum()),
        "trips_with_bad_arc_endpoint": int((ck["arcs_bad_endpoint"] > 0).sum()),
        "trips_with_route_bad_id": int((ck.get("route_bad_id", pd.Series(0)) > 0).sum()),
        "trips_with_route_breaks": int((ck.get("route_breaks", pd.Series(0)) > 0).sum()),
        "route_connected_share_min": float(ck.get("route_connected_share", pd.Series(np.nan)).min()),
        "track_inside_node_bbox_share_min": float(ck["track_inside_node_bbox_share"].min()),
        "alignment_default": VARIANTS[0], "noise_threshold_m": NOISE_M,
        "obs_gt_30m_total": int(default["obs_gt_30m"].sum()) if len(default) else None,
        "obs_gt_50m_total": int(default["obs_gt_50m"].sum()) if len(default) else None,
        "align_p95_m_worst_trip": float(default["align_dist_p95_m"].max()) if len(default) else None,
        "changed_share_by_variant_max": asum.groupby("variant")["changed_vs_default_share"].max().round(4).to_dict() if len(asum) else None,
        "align_seconds_max": float(asum["align_seconds"].max()) if len(asum) else None,
        "changed_share_by_variant_median": asum.groupby("variant")["changed_vs_default_share"].median().round(4).to_dict() if len(asum) else None,
        **_label_stats(obs_tables, ck),
        "scan_seconds": time.time() - t0,
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


def _label_stats(obs_tables, ck) -> dict:
    if not obs_tables:
        return {}
    o = pd.concat(obs_tables, ignore_index=True)
    unl = o["align_dist_m"] > NOISE_M
    per = o.assign(unl=unl).groupby(["partition", "trip_id"])["unl"].mean()
    return {
        "labelled_obs": int((~unl).sum()), "unlabelled_obs": int(unl.sum()),
        "unlabelled_share_overall": float(unl.mean()),
        "unlabelled_by_partition": o.assign(unl=unl).groupby("partition")["unl"].mean().round(4).to_dict(),
        "unlabelled_with_no_route_arc_within_noise": int((unl & (o["nearest_route_dist_m"] > NOISE_M)).sum()),
        "unlabelled_share_per_trip": per.describe().round(4).to_dict(),
        "trips_unlabelled_gt_5pct": per[per > 0.05].round(3).reset_index().to_dict("records"),
    }


if __name__ == "__main__":
    raw = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "outputs" / "part1"
    main(raw, out)
