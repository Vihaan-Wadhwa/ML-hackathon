"""Time extract_features_for_trip on THIS machine for a few trips (light; demo-laptop check).

Default trips = the slowest ones on Kaggle (from part5/timing.csv) plus the largest network.
Usage: python scripts/time_local.py RAW_ROOT OUT_CSV [TRIP_ID ...]
"""
from __future__ import annotations

import platform
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spatial.features import extract_features_for_trip  # noqa: E402
from spatial.io import find_trip_dir  # noqa: E402

DEFAULT = ["00000092", "00000084", "00000042", "00000013", "00000054"]


def main(raw: Path, out: Path, trips: list[str]) -> pd.DataFrame:
    extract_features_for_trip(trips[0], raw, with_labels=False)  # warm-up
    rows = []
    for tid in trips:
        part = find_trip_dir(tid, raw).parent.name
        for mode, lab in (("inference", False), ("full", True)):
            tm = {}
            extract_features_for_trip(tid, raw, with_labels=lab, timings=tm)
            rows.append({"machine": f"{platform.system()} {platform.machine()}", "partition": part,
                         "trip_id": tid, "mode": mode, **tm})
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df[["trip_id", "mode", "n_obs", "n_arcs", "total_s"]].to_string(index=False))
    return df


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:] or DEFAULT)
