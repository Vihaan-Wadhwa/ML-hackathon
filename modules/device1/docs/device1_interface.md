# Device 1 interface: spatial engine handoff

For Device 2 (models), Device 3 (HMM/Viterbi, API) and Device 4 (UI, maps). The field-level schema contract and the label rule are in [part1_data_contract.md](part1_data_contract.md). This page covers what you receive and how to call the code.

## Files delivered (built from all 90 train + student_test trips)

| File | Who uses it | Contents |
|---|---|---|
| `train_candidates.parquet` | Device 2 | 80 trips, 208,417 observations, 1,103,060 rows, schema v1.0 (29 MB) |
| `student_test_candidates.parquet` | Device 2 | 10 trips, 19,160 observations, 113,639 rows (3 MB) |
| `network_geometry.parquet` | Devices 3, 4 | one row per (`trip_id`, `arc_id`): 4,061,945 rows (182 MB) |
| `network_crs.json` | Devices 3, 4 | `trip_id` → the CRS (proj string) that `from_x/y` and `to_x/y` are in |
| `build_report.csv` | everyone | per trip: rows, miss rate, unlabelled share, positive rate, timings |

Location: the parquets are derived data and are **not in this public repo**. Ask Device 1 for the files (team-only: the hackathon forbids sharing derived datasets with other teams), or rebuild them with the commands below.

### `network_geometry.parquet` columns

`trip_id` (str), `arc_id`, `from_node`, `to_node` (int32, 0-based row ids of `.arcs` and `.nodes`), `from_lon`, `from_lat`, `to_lon`, `to_lat` (float64, degrees, for maps), `from_x`, `from_y`, `to_x`, `to_y` (float32, metres in that trip's CRS, for HMM distances), `length_m` (float32), `bearing_deg` (float32, grid bearing clockwise from north, from start to end; arcs are directed).

The x/y values are only comparable **within one trip**, because every trip has its own CRS. For graph transitions, use `from_node`/`to_node`: arc *a* can be followed by arc *b* when `to_node[a] == from_node[b]`.

## Function signature (the same call in training and in the API)

```python
import sys; sys.path.insert(0, "src")
from spatial.features import extract_features_for_trip
from spatial.config import load_config

df = extract_features_for_trip(trip_id, raw_dir)                     # labels if a .route exists
df = extract_features_for_trip(trip_id, raw_dir, with_labels=False)  # API: never reads .route
df, alignment, geom = extract_features_for_trip(trip_id, raw_dir, return_alignment=True)
```

- `raw_dir` is either a partition folder (`train/`) or the root that contains partition folders.
- The result is a pandas DataFrame in frozen schema v1.0, in fixed column order. Write it with `spatial.schema.write_parquet` to enforce dtypes.
- **Evaluator trips arrive without labels.** The API must call the same function with `with_labels=False`, which leaves `is_match` and `ref_arc_id` null. The 12 feature columns are identical in both modes (a unit test checks this).
- `geom` (`spatial.geometry.TripGeometry`) gives the projected arcs and the cached STRtree, which Device 3 can reuse for transitions.

## Parameters used (`configs/spatial.yaml`)

| Parameter | Value | Why (evidence: `results/part5/`, `results/eda/kr_grid_miss_rate.png`) |
|---|---|---|
| `k` | 8 | Pooled miss rate 0.81% (train + student_test). K=6 gives 1.35%; K=16 gives 0.42% but 40% more rows. K counts *directed* arcs, and almost every road has a reverse twin at the same distance, so K=8 is about 4 roads. K=1 misses 45% for that reason. |
| `radius_m` | 50 | 40 m → 1.04%, 30 m → 1.85%. Above 50 m nothing changes, because labelled fixes have their reference within 50 m by definition. |
| `fallback_radius_m` | 0 (off) | All 18,230 zero-candidate fixes are train fixes where the vehicle is outside the supplied network (16,519 in 00000054). None has a label, and the nearest arc would be a wrong match. |
| `crs_kind` | tmerc | Trips are worldwide (23 UTM zones). The projection is a local transverse Mercator at each trip's median node. |
| `min_move_m` | 1.0 | `has_heading = 0` below 1 m of displacement from the previous fix. |
| `gap_threshold_s` | 1.0 | `gap_flag = 1` when Δt > 1 s. |

## Results (all from code; train and student_test only, no evaluator data)

| | train | student_test |
|---|---|---|
| Miss rate, pooled over labelled fixes (target < 5%) | **0.87%** | **0.29%** |
| Strict miss rate (unlabelled fixes counted as misses) | 11.9% | 0.43% |
| Worst trip | 19.9% (00000033) | 2.4% (00000009) |
| Trips over 5% | 4 (00000033, 00000072, 00000054, 00000061) | 0 |
| Rows per observation, positive rate | 5.29, 16.6% | 5.93, 16.8% |

Timing of `extract_features_for_trip` (target < 1.5 s per trip):

| Machine | Path | Median | p95 | Max |
|---|---|---|---|---|
| Kaggle CPU, all 90 trips | inference (no labels) | 0.045 s | 0.17 s | 0.32 s |
| Kaggle CPU, all 90 trips | with labels | 0.074 s | 0.38 s | 0.86 s |
| Demo laptop (arm64), 5 slowest/largest trips | inference | 0.081 s | 0.14 s | 0.15 s |
| Demo laptop (arm64), 5 slowest/largest trips | with labels | 0.32 s | 0.44 s | 0.46 s |

The label alignment (a DP with one Python loop over observations) is the slowest step. The API never runs it.

## Notes for each device

**Device 2 (models)**
- Use only the 12 feature columns. **Train on rows where `is_match` is not null.**
- Group the CV by `trip_id`.
- Fit imputers and scalers on training trips only (`bearing_diff_deg` is NaN when `has_heading = 0`, and `dt_s`/`speed_mps` are NaN at obs 0).
- Rows with `arc_id = -1` are placeholders for observations with no candidate. Drop them for training, since they never have a label.
- Expect ties: an arc and its reverse twin share every geometric feature except `bearing_diff_deg`, `dist_to_start_m`, `dist_to_end_m` and `frac_along`. With no heading, they are indistinguishable.

**Device 3 (HMM)**
- Hidden states are the candidate rows for each observation. Emissions come from Device 2's scores or from `perp_dist_m`. Transitions come from `network_geometry.parquet` (node connectivity plus `length_m`).
- Treat an observation with `n_candidates = 0` as a gap: skip it and carry the previous column forward.
- Use `dt_s` and `gap_flag` for the transition time budget. Gaps reach 42 h (train/00000090).
- Reference routes are not always continuous: 11 trips have 1–3 breaks of 10–600 m. Do not score a break as a model error.
- Misses happen at junctions full of 1–5 m micro-segments. An HMM with a K=8 ceiling cannot recover them, which caps accuracy at about 99% of labelled fixes.

**Device 4 (UI and maps)**
- Use `from_lon/lat` and `to_lon/lat` to draw arcs.
- Example Folium maps are in `results/eda/map_*.html`. Hits are blue, misses red and unlabelled points grey, with layers for the reference route and the rank-0 arc.

## Reproduce

Run from `modules/device1/`. RAW is the folder that holds `train/`, `student_test/` and `evaluator_test/` (raw data is never committed).

```bash
pip install -r requirements.txt
python -m pytest -q tests                                  # unit tests (synthetic data)
python scripts/make_sample_candidates.py                   # dummy parquet (outputs/)
python scripts/make_splits.py RAW data/splits.csv          # split manifest
python scripts/run_pipeline.py RAW outputs part1 part5 build   # full scan, K/R sweep + timing, batch build (heavy)
python scripts/eda_spatial.py RAW outputs outputs/eda      # tables, figures, Folium maps
python scripts/build_evaluator_features.py RAW outputs/evaluator/evaluator_candidates.parquet
```

The heavy steps (`part1`, `part5`, `build`) ran on a Kaggle CPU; the figures and tables in `results/` come from those runs.
