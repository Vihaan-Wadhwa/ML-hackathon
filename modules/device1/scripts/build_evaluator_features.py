"""Evaluator-trip features, INFERENCE MODE ONLY (exactly what the API computes).

* .route files are never opened (load_trip(read_route=False)); is_match / ref_arc_id are null.
* Same frozen config (configs/spatial.yaml) chosen on train + student_test only.
* For final evaluation only: never train, tune, select or threshold on this file.
Usage: python scripts/build_evaluator_features.py [RAW_ROOT] [OUT_PARQUET]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spatial.config import load_config  # noqa: E402
from spatial.features import extract_features_for_trip  # noqa: E402
from spatial.io import list_trips  # noqa: E402
from spatial.schema import ARROW_SCHEMA, SCHEMA_VERSION, to_arrow, validate_table  # noqa: E402


def main(raw_root: Path, out: Path) -> None:
    cfg = load_config()
    trips = list_trips(raw_root)
    ev = trips[trips.partition == "evaluator_test"]
    others = set(trips[trips.partition != "evaluator_test"].trip_id)
    assert len(ev) and not (set(ev.trip_id) & others), "evaluator trips missing or overlapping"
    out.parent.mkdir(parents=True, exist_ok=True)
    tables, report = [], []
    for tid in ev.trip_id:
        tm = {}
        df = extract_features_for_trip(tid, raw_root / "evaluator_test", cfg=cfg, with_labels=False, timings=tm)
        assert df.is_match.isna().all() and df.ref_arc_id.isna().all()
        tbl = to_arrow(df)
        validate_table(tbl)
        tables.append(tbl)
        report.append({"trip_id": tid, "n_obs": tm["n_obs"], "rows": tm["n_rows"],
                       "zero_candidate_obs": int((df.n_candidates == 0).sum()),
                       "total_s": round(tm["total_s"], 4)})
    pq.write_table(pa.concat_tables(tables).cast(ARROW_SCHEMA), out, compression="zstd")
    meta = {"partition": "evaluator_test", "mode": "inference (no .route read, labels null)",
            "schema_version": SCHEMA_VERSION, "config": cfg.to_dict(), "trips": report,
            "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    raw = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "outputs" / "evaluator" / "evaluator_candidates.parquet"
    main(raw, out)
