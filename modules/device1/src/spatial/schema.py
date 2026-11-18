"""Frozen candidate-feature schema (Device 1 -> Device 2/3/4 contract).

Schema version 1.0, frozen in Part 1. Any later change must be ADDITIVE only
(new columns appended at the end); existing names, order and dtypes never change.

One row = one (observation, candidate arc) pair.

Conventions
-----------
* An observation with zero candidates is NOT dropped. It gets exactly one row with
  ``arc_id = -1``, ``cand_rank = -1``, ``n_candidates = 0``, all candidate-level
  float features NaN and ``is_match = 0`` (null if the obs is unlabelled).
* ``is_match`` / ``ref_arc_id`` are null when the trip has no ``.route`` (inference).
  ``is_match`` is also null for UNLABELLED observations (reference arc > 50 m away; see
  labels.py). Train only on rows with non-null ``is_match``. In pandas a column with
  nulls reads back as float64 (NaN); without nulls as int8.
* ``ref_arc_id`` and ``is_match`` are labels. Using either as a model input is leakage.
* ``has_heading = 0`` -> ``bearing_diff_deg`` is NaN (first point / stationary vehicle).
"""
from __future__ import annotations

import pandas as pd
import pyarrow as pa

SCHEMA_VERSION = "1.0"

# (name, arrow type, nullable, description)
COLUMNS: list[tuple[str, pa.DataType, bool, str]] = [
    ("trip_id", pa.string(), False, "Trip folder name, zero-padded (e.g. '00000007')"),
    ("obs_idx", pa.int32(), False, "0-based row index into the trip's .track"),
    ("timestamp", pa.float64(), False, "Raw .track timestamp (seconds)"),
    ("arc_id", pa.int32(), False, "0-based row index into .arcs; -1 = no candidate"),
    ("cand_rank", pa.int32(), False, "Rank by perp_dist_m, 0 = closest; -1 = no candidate"),
    ("perp_dist_m", pa.float32(), True, "Distance GPS point -> candidate segment (m)"),
    ("bearing_diff_deg", pa.float32(), True, "|vehicle heading - arc bearing| wrapped to [0,180]"),
    ("speed_mps", pa.float32(), True, "Displacement from previous obs / dt (m/s); NaN at obs 0"),
    ("dt_s", pa.float32(), True, "Time since previous obs (s); NaN at obs 0"),
    ("gap_flag", pa.int8(), False, "1 if dt_s > 1 s else 0"),
    ("dist_to_start_m", pa.float32(), True, "Distance GPS point -> arc from_node (m)"),
    ("dist_to_end_m", pa.float32(), True, "Distance GPS point -> arc to_node (m)"),
    ("frac_along", pa.float32(), True, "Projection position along the arc, clamped to [0,1]"),
    ("seg_len_m", pa.float32(), True, "Arc length (m)"),
    ("n_candidates", pa.int32(), False, "Candidates found for this observation"),
    ("has_heading", pa.int8(), False, "1 if vehicle heading is reliable for this obs"),
    ("is_match", pa.int8(), True, "1 if arc_id == ref_arc_id else 0; null without .route or when the obs is unlabelled (align_dist > 50 m)"),
    ("ref_arc_id", pa.int32(), True, "LABEL column (never a feature): reference arc from route alignment; null without .route"),
]

COLUMN_NAMES = [c[0] for c in COLUMNS]
ARROW_SCHEMA = pa.schema(
    [pa.field(n, t, nullable=nl) for n, t, nl, _ in COLUMNS],
    metadata={"schema_version": SCHEMA_VERSION},
)
NO_CANDIDATE = -1


def to_arrow(df: pd.DataFrame) -> pa.Table:
    """Cast a feature DataFrame to the frozen schema (fixed order + dtypes)."""
    missing = [c for c in COLUMN_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"missing schema columns: {missing}")
    extra = [c for c in df.columns if c not in COLUMN_NAMES]
    if extra:
        raise ValueError(f"columns not in frozen schema v{SCHEMA_VERSION}: {extra}")
    return pa.Table.from_pandas(df[COLUMN_NAMES], schema=ARROW_SCHEMA, preserve_index=False)


def write_parquet(df: pd.DataFrame, path) -> None:
    import pyarrow.parquet as pq

    pq.write_table(to_arrow(df), path)


def validate_table(table: pa.Table) -> None:
    """Raise if a table does not match the frozen schema exactly."""
    got = [(f.name, f.type) for f in table.schema]
    want = [(f.name, f.type) for f in ARROW_SCHEMA]
    if got != want:
        raise ValueError(f"schema mismatch:\n got  {got}\n want {want}")


def schema_markdown() -> str:
    lines = ["| # | Column | Arrow type | Nullable | Meaning |", "|---|---|---|---|---|"]
    for i, (n, t, nl, d) in enumerate(COLUMNS):
        lines.append(f"| {i} | `{n}` | {t} | {'yes' if nl else 'no'} | {d} |")
    return "\n".join(lines)
