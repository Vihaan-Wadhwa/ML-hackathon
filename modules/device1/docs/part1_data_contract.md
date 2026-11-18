# Part 1: data contract and dataset findings (Device 1)

For Devices 2, 3 and 4. Schema version **1.0 is frozen**. Any later change only appends columns.

## 1. Candidate feature schema v1.0

One row is one (observation, candidate arc) pair. The source of truth is `src/spatial/schema.py`, and `schema.write_parquet` enforces column order and dtypes.

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

Rules you can rely on:

- **No observation is dropped.** An observation with no candidate within R gets one row with `arc_id = -1`, `cand_rank = -1`, `n_candidates = 0` and NaN candidate features.
- **`is_match` is null** when (a) the trip has no `.route` (evaluator trips in the API) or (b) the observation is *unlabelled* (see section 2). **Train only on rows where `is_match` is not null.** In pandas the column reads as `int8` if it has no nulls and as float64/NaN otherwise.
- **`is_match` and `ref_arc_id` are labels, not features.** Feeding either into a model is leakage.
- `has_heading = 0` means `bearing_diff_deg` is NaN.
- The dummy file `outputs/sample_candidates.parquet` (50 rows, `trip_id = DUMMY000`) uses exactly this schema and includes every edge case: a miss (obs 9), zero candidates (obs 10), an unlabelled observation (obs 8), a missing heading (obs 0) and a gap (obs 6). Its values are synthetic.

## 2. Label alignment rule v1 (defines `ref_arc_id`, `is_match` and "miss")

`.route` is a chronological list of directed arc_ids. It is **not** one entry per GPS fix: there are 1.3 to 57 fixes per route arc (median 5.3). Each fix *i* is assigned a route **position** p(i) by a monotonic DP (`src/spatial/labels.py: align_route`). The DP minimises

  sum of [ perpendicular distance (m) to arc route[p(i)] + 10 m x (1 - cos(heading, arc direction)) / 2 ] + 1 m x (positions advanced)

with p(i) >= p(i-1). Start and end are free, and positions may be skipped. The heading term only applies when the centred displacement is at least 2 m. It is needed because the median trip has 95% of its arcs with a geometrically identical reverse twin, so on a U-turn distance alone ties. `ref_arc_id(i) = route[p(i)]`.

- **Labelled:** distance to the reference arc is at most **50 m**. `is_match = 1` on the candidate row whose `arc_id == ref_arc_id` and 0 on the others.
- **Unlabelled:** distance is above 50 m. `is_match` is null on all rows.
- **Miss:** a labelled observation whose `ref_arc_id` is not among its candidates. The 50 m threshold is fixed by this rule and does not depend on the candidate radius R, so the miss rate cannot be lowered by redefining labels.
- **Stability** (all 90 trips, `results/part1/alignment_summary.csv`): with lambda in {0, 5} instead of 1, at most 0.45% of assignments change. With the heading weight at 20 m instead of 10, the median change is 0.67% and the maximum 3.2%. Removing the heading term changes a median of 2.1% and up to 13.7%, which is the U-turn ambiguity it exists to fix.

Assumption: `.route` is reference information, not verified truth. It is not always continuous: 11 of 90 trips have 1 to 3 breaks of 10 to 600 m where arcs are skipped (see section 3).

## 3. Dataset findings

Source: `results/part1/`, produced by `scripts/eda_part1.py` (run on a Kaggle CPU). Each row there is tagged with its partition.

| Finding | Value (train + student_test, 90 trips) |
|---|---|
| Trips | train 80, student_test 10, evaluator 0 present (held by the organisers) |
| Files | every trip has `.track`, `.nodes`, `.arcs`, `.route`; scan errors: 0 |
| Partition overlap | none (`data/splits.csv`) |
| **Networks** | **each trip has its own network: 90 distinct (nodes, arcs) hashes**; 649 to 307,392 arcs (median 18,762) |
| **Location** | **worldwide, not only Almaty:** 23 UTM zones, 3 southern-hemisphere trips, 2 trips that straddle a zone edge |
| Coordinate order | lon, lat confirmed: every GPS fix lies inside its network's node bounding box |
| GPS observations | 227,577 total; 262 to 19,313 per trip (median 1,497) |
| Sampling | 96.7% of steps are exactly 1 s; no negative or zero Δt; 3 trips have non-integer timestamps; 42 trips have a gap over 10 s; the largest gap is 153,064 s (42.5 h, train/00000090) |
| Invalid rows | no NaN or out-of-range coordinates; 6 repeated consecutive fixes |
| Arcs | no endpoints that point to missing nodes; 24 self-loops; **1,903 duplicate (from, to) arcs** (the same road with different arc_ids); 0 isolated nodes |
| Connectivity | median 19 weakly connected components per trip; the largest component holds 69% to 100% of nodes |
| Route | 59 to 2,266 arcs; no invalid ids; 11 trips have breaks (connected share at least 98.1%) |
| Unlabelled fixes | 23,295 (10.2%). 22,603 of these are more than 50 m from *every* route arc: the track continues beyond the route. They are concentrated in 5 train trips: 00000054 (89%), 00000033 (59%), 00000002 (12%), 00000066 (9%) and 00000043 (7%). student_test is only 0.14% unlabelled |

Consequences for later parts:

- **CRS (changes plan Part 2).** One EPSG:32643 projection is wrong for most trips. `projection.trip_crs` uses a transverse Mercator centred on each trip's median node. Scale error is below 2e-5 and there are no zone-edge or hemisphere cases. `tests/test_part1.py` checks it against Haversine at four locations. Per-trip UTM is available as an alternative.
- **Duplicate arcs:** when two arc_ids share geometry, only the one in the route is labelled 1. Device 2 should expect ties that features cannot separate. Device 3's HMM may want to merge them.
- **Large gaps** (up to 42 h) need the HMM's gap handling. `gap_flag` and `dt_s` are in the schema.
- **Radius floor for R:** 2,922 labelled fixes (1.4% of 204,282) are 30 to 50 m from their reference arc. With R = 30 m every one of them is a guaranteed miss, which already uses up more than a quarter of the 5% budget. Part 5 tunes R with this in mind.

## 4. Interfaces

```python
from spatial.io import load_trip, validate_trip, list_trips
trip = load_trip("00000007", raw_dir)     # raw_dir = partition dir or the root holding train/ and student_test/
trip.track   # DataFrame lon, lat, t      (index obs_idx, 0-based)
trip.nodes   # DataFrame lon, lat         (index node_id, 0-based)
trip.arcs    # DataFrame from_node, to_node (index arc_id, 0-based)
trip.route   # Series of arc_id, or None when there is no .route (API inference)

from spatial.labels import align_route, is_match_for, miss_rate
al = align_route(trip)                    # per obs: route_pos, ref_arc_id, align_dist_m, nearest_route_dist_m, is_noise
```

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
