# Device 1: spatial engine and feature extraction

For each GPS fix in a trip, Device 1 finds up to 8 nearby road arcs (candidates) and computes the features Device 2's classifier scores. Device 3's HMM then picks one candidate per fix. The same function serves training (with labels) and the API (without labels).

## Use it from the backend

```python
from modules.device1 import extract_trip_features, model_rows, FEATURE_COLUMNS

df = extract_trip_features("00000015", "data")   # finds data/**/00000015.track; never opens .route
rows = model_rows(df)                             # drop arc_id = -1 placeholder rows before scoring
proba = model.predict_proba(rows[FEATURE_COLUMNS].to_numpy())[:, 1]
```

- Run it from the repo root (so `modules` is importable).
- `extract_trip_features` takes at most 0.15 s per trip on the demo laptop.
- `FEATURE_COLUMNS` matches `backend/predictor.py` exactly.

## Input format

The function expects one folder per trip, anywhere under `data_dir`:

```
<trip_id>/<trip_id>.track   lon lat t        GPS fixes (whitespace-separated, no header, WGS84 degrees, seconds)
<trip_id>/<trip_id>.nodes   lon lat          row index = node_id (0-based)
<trip_id>/<trip_id>.arcs    from to          row index = arc_id (0-based, directed)
<trip_id>/<trip_id>.route   arc_id           reference labels: training only, never read by the API path
```

Raw data is never committed. The root `.gitignore` blocks `*.track`, `*.nodes`, `*.arcs` and `*.route`.

## Output schema v1.0 (frozen; later changes only append columns)

One row is one (GPS fix, candidate arc) pair, ranked by distance.

| # | Column | Arrow type | Nullable | Meaning |
|---|---|---|---|---|
| 0 | `trip_id` | string | no | Trip folder name, zero-padded (e.g. '00000007') |
| 1 | `obs_idx` | int32 | no | 0-based row index into the trip's .track |
| 2 | `timestamp` | double | no | Raw .track timestamp (seconds) |
| 3 | `arc_id` | int32 | no | 0-based row index into .arcs; -1 = no candidate |
| 4 | `cand_rank` | int32 | no | Rank by perp_dist_m, 0 = closest; -1 = no candidate |
| 5 | `perp_dist_m` | float | yes | Distance GPS point -> candidate segment (m) |
| 6 | `bearing_diff_deg` | float | yes | |vehicle heading - arc bearing| wrapped to [0,180] |
| 7 | `speed_mps` | float | yes | Displacement from previous obs / dt (m/s); NaN at obs 0 |
| 8 | `dt_s` | float | yes | Time since previous obs (s); NaN at obs 0 |
| 9 | `gap_flag` | int8 | no | 1 if dt_s > 1 s else 0 |
| 10 | `dist_to_start_m` | float | yes | Distance GPS point -> arc from_node (m) |
| 11 | `dist_to_end_m` | float | yes | Distance GPS point -> arc to_node (m) |
| 12 | `frac_along` | float | yes | Projection position along the arc, clamped to [0,1] |
| 13 | `seg_len_m` | float | yes | Arc length (m) |
| 14 | `n_candidates` | int32 | no | Candidates found for this observation |
| 15 | `has_heading` | int8 | no | 1 if vehicle heading is reliable for this obs |
| 16 | `is_match` | int8 | yes | 1 if arc_id == ref_arc_id else 0; null without .route or when the obs is unlabelled (align_dist > 50 m) |
| 17 | `ref_arc_id` | int32 | yes | LABEL column (never a feature): reference arc from route alignment; null without .route |

- **Model inputs:** the 12 `FEATURE_COLUMNS`.
- **Keys:** `trip_id`, `obs_idx`, `timestamp`, `arc_id`.
- **Labels (never model inputs):** `is_match` and `ref_arc_id`. Both are null in the API and evaluator output.
- **No road within 50 m:** one row with `arc_id = -1`, `cand_rank = -1`, `n_candidates = 0` and NaN features. `model_rows()` drops it; the HMM should treat that fix as a gap. Device 3's loader rejects negative `arc_id`, so always drop these rows first.
- **Expected NaNs:** `bearing_diff_deg` when `has_heading = 0`, and `dt_s`/`speed_mps` at `obs_idx = 0`.

## Batch file for the 10 evaluator trips

```bash
cd modules/device1
python scripts/build_evaluator_features.py <RAW_ROOT> outputs/evaluator/evaluator_candidates.parquet
```

`RAW_ROOT` must contain `evaluator_test/<trip_id>/`. The script never reads `.route`, so labels stay null.

**This file is not training input.** Device 2 trains and tunes only on train (80 trips) and student_test (10 trips). Evaluator trips go through the API live (Device 1 → Device 2 model → Device 3 HMM). The precomputed file is only for the final evaluation run or for checking the API output. Accuracy against evaluator routes is computed afterwards by a separate scoring step.

## Results (train + student_test only; details in `docs/` and `results/`)

| | train | student_test |
|---|---|---|
| Candidate miss rate (labelled fixes; target < 5%) | 0.87% | 0.29% |
| Miss rate counting unlabelled fixes as misses | 11.9% | 0.43% |
| Rows per fix / share of rows with `is_match = 1` | 5.29 / 16.6% | 5.93 / 16.8% |

The slowest trip extracts in 0.86 s with labels on a Kaggle CPU, against a target of < 1.5 s. Without labels it takes at most 0.32 s on Kaggle and 0.15 s on the demo laptop.

## Layout

```
modules/device1/
├── __init__.py, pipeline.py   backend entry point (extract_trip_features, model_rows)
├── src/spatial/               io, projection, geometry, candidates, features, labels, schema, config
├── configs/spatial.yaml       K=8, R=50 m, alignment rule (frozen)
├── scripts/                   EDA scan, K/R sweep + timing, batch build, EDA figures, evaluator build
├── tests/                     unit tests on synthetic data (python -m pytest -q modules/device1/tests)
├── docs/                      part1_data_contract.md, device1_interface.md, data_quality_log.md
├── results/                   part1 (data checks), part5 (miss rate, K/R grid, timing), eda (figures, tables)
└── data/splits.csv            trip_id -> partition
```

The parquet files (train, student_test, evaluator candidates and the network geometry) are derived data. They are kept out of this public repo, because the hackathon forbids sharing derived datasets with other teams. Ask Device 1 for them, or rebuild them with the scripts.
