"""data/splits.csv from the directory listing (partition = parent folder name).

The evaluator partition is held by the organisers and is not in the folder, so it has
no rows here; the pipeline never needs it. Fails if a trip id appears in two partitions.
Usage: python scripts/make_splits.py [RAW_ROOT] [OUT_CSV]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spatial.io import list_trips  # noqa: E402


def make_splits(raw_root, out_csv):
    df = list_trips(raw_root)
    dup = df[df.duplicated("trip_id", keep=False)]
    if len(dup):
        raise SystemExit(f"trip ids in more than one partition:\n{dup}")
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


if __name__ == "__main__":
    raw = sys.argv[1] if len(sys.argv) > 1 else ROOT
    out = sys.argv[2] if len(sys.argv) > 2 else ROOT / "data" / "splits.csv"
    df = make_splits(raw, out)
    print(df["partition"].value_counts().to_string())
    print(f"wrote {out}")
