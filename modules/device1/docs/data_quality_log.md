# Data-quality log (Device 1, for report section 2.2)

Partitions: train (80 trips) and student_test (10 trips). No evaluator data was present or used. Each number comes from the file named in its row.

## Issues found

| # | Issue | Partition(s) | Size | Evidence |
|---|---|---|---|---|
| 1 | Trips are worldwide, so one CRS does not fit | both | 23 UTM zones, 3 southern-hemisphere trips, 2 trips straddling a zone edge | `results/part1/trip_checks.csv` (`utm_zones`) |
| 2 | Each trip has its own road network | both | 90 distinct (nodes, arcs) hashes | `results/part1/network_groups.csv` |
| 3 | Track continues beyond the reference route | train (5 trips), student_test (minor) | 23,295 fixes (10.2%) more than 50 m from the aligned reference arc; 22,603 of them more than 50 m from *every* route arc | `results/part1/dataset_summary.json`, `alignment_obs.parquet` (not committed: per-observation data) |
| 4 | Vehicle leaves the supplied network | train | 18,230 fixes with no arc within 50 m (16,519 in 00000054) | `results/part5/build_report.csv` (`zero_candidate_obs`) |
| 5 | Large GPS gaps | both | 42 trips have a gap over 10 s; the largest is 153,064 s (train/00000090) | `results/eda/obs_and_dt_summary.csv` |
| 6 | Breaks in the reference route | train 10, student_test 1 | 1–3 breaks of 10–600 m per trip | `results/part1/trip_checks.csv` (`route_breaks`) |
| 7 | Duplicate arcs | both | 1,903 duplicate (from, to) pairs; 24 self-loops | `results/eda/missing_invalid_raw.csv` |
| 8 | Dense junction micro-segments (arcs 1–5 m long) | both | cause of most of the 1,657 candidate misses at K=8 | `obs_ref_ranks.parquet` (not committed: per-observation data) |
| 9 | Non-integer timestamps | train | 3 trips | `results/part1/trip_checks.csv` (`t_non_integer`) |
| 10 | Repeated consecutive fixes | both | 6 | `results/part1/trip_checks.csv` (`dup_consecutive_fix`) |

Not found: NaN or out-of-range coordinates, negative or zero Δt, arc endpoints pointing to missing nodes, invalid route arc ids, isolated nodes. Coordinate order (lon, lat) is confirmed because every fix lies inside its network's node bounding box. 96.7% of steps are exactly 1 s.

## Decisions and assumptions

1. **CRS:** each trip uses its own transverse Mercator centred on the median node, with distances in metres. Distances agree with Haversine to within 0.4% (sphere vs ellipsoid) at four test locations (`tests/test_part1.py`).
2. **Labels:** `.route` is treated as reference information, not verified truth. Fixes are aligned to route positions by a monotonic DP using distance plus a direction term (rule v1, `src/spatial/labels.py`). The result is stable when its parameters are varied (`results/part1/alignment_summary.csv`).
3. **Unlabelled fixes:** fixes more than 50 m from their reference arc get `is_match = null`. They are excluded from training labels and from the miss rate, and they are reported separately as a strict miss rate (11.9% train, 0.43% student_test). The 50 m threshold was fixed in Part 1, before K and R were tuned.
4. **Nothing is dropped or repaired** in the raw data. Gaps, duplicates and route breaks are kept and flagged (`gap_flag`, `dt_s`, `n_candidates = 0` rows).
5. **The true arc is never inserted** into the candidate set. Misses are reported as they are.
6. **No learned preprocessing** happens in Device 1. Imputation and scaling belong to Device 2, fitted on training data.
7. **K = 8, R = 50 m** were chosen on train + student_test only (`results/part5/miss_rate_results.csv`, `results/eda/kr_grid_miss_rate.png`).
