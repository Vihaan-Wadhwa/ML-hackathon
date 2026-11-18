"""Part 6 batch build (HEAVY: run on a remote CPU, not the demo laptop).

Writes to OUT_DIR:
  train_candidates.parquet          80 trips, frozen schema v1.0
  student_test_candidates.parquet   10 trips
  network_geometry.parquet          one row per (trip_id, arc_id), Device 3/4 lookup
  network_crs.json                  trip_id -> CRS (proj string) used for x/y
  build_report.csv                  per trip: rows, miss rate, unlabelled share, timings
  build_manifest.json               config, schema version, row counts, file sizes
Usage: python scripts/build_all_features.py RAW_ROOT OUT_DIR
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spatial.config import load_config  # noqa: E402
from spatial.features import extract_features_for_trip  # noqa: E402
from spatial.geometry import build_geometry_table  # noqa: E402
from spatial.io import list_trips, load_trip  # noqa: E402
from spatial.labels import miss_rate  # noqa: E402
from spatial.schema import ARROW_SCHEMA, SCHEMA_VERSION, to_arrow, validate_table  # noqa: E402


def main(raw_root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    splits = list_trips(raw_root)
    assert not splits.trip_id.duplicated().any()
    writers, report, geoms, crs = {}, [], [], {}
    t0 = time.time()
    for n, (tid, part) in enumerate(splits.itertuples(index=False), 1):
        trip = load_trip(tid, raw_root / part)
        tm = {}
        df, _, geom = extract_features_for_trip(tid, cfg=cfg, trip=trip, timings=tm, return_alignment=True)
        tbl = to_arrow(df)
        validate_table(tbl)
        if part not in writers:
            writers[part] = pq.ParquetWriter(out_dir / f"{part}_candidates.parquet", ARROW_SCHEMA,
                                             compression="zstd")
        writers[part].write_table(tbl)
        geoms.append(build_geometry_table(trip, geom))
        crs[tid] = geom.crs
        mr = miss_rate(df)
        report.append({"partition": part, "trip_id": tid, **mr, "rows": len(df),
                       "positive_rate": float(df.is_match.mean()) if mr["n_labelled"] else None,
                       "zero_candidate_obs": int((df.n_candidates == 0).sum()), **tm})
        print(f"[{n}/{len(splits)}] {part}/{tid} rows={len(df)} miss={mr['miss_rate']:.4f} "
              f"{time.time() - t0:.1f}s", flush=True)
    for w in writers.values():
        w.close()
    geo = pd.concat(geoms, ignore_index=True)
    pq.write_table(pa.Table.from_pandas(geo, preserve_index=False), out_dir / "network_geometry.parquet",
                   compression="zstd")
    (out_dir / "network_crs.json").write_text(json.dumps(
        {"note": "x/y in network_geometry.parquet are metres in this per-trip CRS "
                 "(local transverse Mercator at the median node, WGS84 ellipsoid)",
         "crs": crs}, indent=2))
    rep = pd.DataFrame(report)
    rep.to_csv(out_dir / "build_report.csv", index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION, "config": cfg.to_dict(),
        "partitions": {}, "geometry_rows": len(geo), "seconds": time.time() - t0,
    }
    for part, g in rep.groupby("partition"):
        f = out_dir / f"{part}_candidates.parquet"
        lab = g.n_labelled.sum()
        manifest["partitions"][part] = {
            "trips": len(g), "rows": int(g.rows.sum()), "obs": int(g.n_obs.sum()),
            "labelled_obs": int(lab),
            "miss_rate_pooled": float((g.miss_rate * g.n_labelled).sum() / lab) if lab else None,
            "miss_rate_trip_max": float(g.miss_rate.max()),
            "unlabelled_share": float(1 - lab / g.n_obs.sum()),
            "zero_candidate_obs": int(g.zero_candidate_obs.sum()),
            "file_mb": round(f.stat().st_size / 1e6, 2)}
    (out_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
