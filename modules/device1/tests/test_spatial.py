"""Parts 2-5 tests: geometry, candidates, features, schema and edge cases (synthetic)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spatial.config import SpatialConfig, load_config  # noqa: E402
from spatial.features import extract_features_for_trip, wrap_angle_diff  # noqa: E402
from spatial.geometry import build_geometry, build_geometry_table  # noqa: E402
from spatial.io import load_trip  # noqa: E402
from spatial.labels import miss_rate  # noqa: E402
from spatial.projection import haversine_m  # noqa: E402
from spatial.schema import COLUMN_NAMES, to_arrow, validate_table  # noqa: E402

LON0, LAT0 = 77.12, 43.39
DLON = 0.001  # ~81 m at this latitude


def write_trip(root: Path, tid: str, track, route=True, part="train"):
    """Grid: an east-west street (nodes 0-3) and a parallel street 150 m north (4-7)."""
    d = root / part / tid
    d.mkdir(parents=True)
    nodes = [[LON0 + k * DLON, LAT0] for k in range(4)] + \
            [[LON0 + k * DLON, LAT0 + 0.00135] for k in range(4)]
    arcs = [[0, 1], [1, 2], [2, 3], [1, 0], [2, 1], [3, 2],      # 0..5 south street
            [4, 5], [5, 6], [6, 7], [5, 4], [6, 5], [7, 6]]      # 6..11 north street
    np.savetxt(d / f"{tid}.nodes", nodes, fmt="%.6f", delimiter="\t")
    np.savetxt(d / f"{tid}.arcs", arcs, fmt="%d", delimiter="\t")
    np.savetxt(d / f"{tid}.track", np.atleast_2d(track), fmt="%.6f", delimiter="\t")
    if route:
        np.savetxt(d / f"{tid}.route", [0, 1, 2], fmt="%d")
    return d


def eastbound(n=20, t=None, offset_lat=0.00003):
    xs = np.linspace(0.05, 2.95, n)
    t = np.arange(1, n + 1, dtype=float) if t is None else t
    return np.c_[LON0 + xs * DLON, np.full(n, LAT0 + offset_lat), t]


CFG = SpatialConfig(k=4, radius_m=50.0)


def run(tmp_path, track, route=True, tid="T", cfg=CFG, **kw):
    write_trip(tmp_path, tid, track, route)
    return extract_features_for_trip(tid, tmp_path, cfg=cfg, **kw)


# ----------------------------------------------------------------- Part 2: geometry
def test_geometry_table_lengths_match_haversine(tmp_path):
    write_trip(tmp_path, "G", eastbound())
    trip = load_trip("G", tmp_path)
    tbl = build_geometry_table(trip)
    assert len(tbl) == len(trip.arcs) and (tbl["arc_id"] == np.arange(len(tbl))).all()
    hv = haversine_m(tbl.from_lon, tbl.from_lat, tbl.to_lon, tbl.to_lat)
    assert np.allclose(tbl.length_m, hv, rtol=4e-3)
    # arcs 0 (east) and 3 (west) are reverse twins
    assert abs(tbl.bearing_deg[0] - 90) < 0.5 and abs(tbl.bearing_deg[3] - 270) < 0.5


def test_bearing_wrap():
    assert wrap_angle_diff(359, 1) == pytest.approx(2)
    assert wrap_angle_diff(1, 359) == pytest.approx(2)
    assert wrap_angle_diff(90, 270) == pytest.approx(180)
    assert wrap_angle_diff(10, 10) == 0


# ----------------------------------------------------------- Part 3/4: candidates
def test_features_schema_and_labels(tmp_path):
    df = run(tmp_path, eastbound())
    validate_table(to_arrow(df))
    assert list(df.columns) == COLUMN_NAMES
    assert (df.groupby("obs_idx")["cand_rank"].apply(lambda r: list(r) == list(range(len(r))))).all()
    assert (df.groupby("obs_idx")["perp_dist_m"].apply(lambda d: d.is_monotonic_increasing)).all()
    assert df["perp_dist_m"].max() <= 50.0 + 1e-6
    # every fix is on the south street, eastbound: the reference is a 0..2 arc, always found
    assert miss_rate(df)["miss_rate"] == 0.0
    hit = df[df.is_match == 1]
    assert set(hit.arc_id) <= {0, 1, 2} and hit.obs_idx.is_unique
    # eastbound vehicle: the reverse twin has bearing diff ~180, the true arc ~0
    moving = df[(df.has_heading == 1) & (df.cand_rank <= 1) & (df.perp_dist_m < 10)]
    assert set(np.round(moving.bearing_diff_deg / 180)) == {0.0, 1.0}
    assert (df[df.is_match == 1].query("has_heading == 1").bearing_diff_deg < 5).all()


def test_k_truncation_and_radius(tmp_path):
    df = run(tmp_path, eastbound(), cfg=SpatialConfig(k=2, radius_m=200.0))
    assert df.groupby("obs_idx").size().max() == 2 and (df.n_candidates <= 2).all()
    df = run(tmp_path, eastbound(), tid="T2", cfg=SpatialConfig(k=20, radius_m=200.0))
    assert df.groupby("obs_idx").size().max() > 4  # north street (150 m) now included


# --------------------------------------------------------------- Part 5: edge cases
def test_missing_route_labels_null(tmp_path):
    df = run(tmp_path, eastbound(), route=False)
    assert df.is_match.isna().all() and df.ref_arc_id.isna().all()
    validate_table(to_arrow(df))


def test_with_labels_false_matches_inference(tmp_path):
    a = run(tmp_path, eastbound(), tid="A")
    b = extract_features_for_trip("A", tmp_path, cfg=CFG, with_labels=False)
    feats = [c for c in COLUMN_NAMES if c not in ("is_match", "ref_arc_id")]
    pd.testing.assert_frame_equal(a[feats], b[feats])
    assert b.is_match.isna().all()


def test_single_point_trip(tmp_path):
    df = run(tmp_path, eastbound(1))
    assert df.obs_idx.nunique() == 1 and df.has_heading.eq(0).all() and df.dt_s.isna().all()
    validate_table(to_arrow(df))


def test_duplicate_timestamps(tmp_path):
    t = np.arange(1, 21, dtype=float)
    t[5] = t[4]
    df = run(tmp_path, eastbound(t=t))
    row = df[df.obs_idx == 5].iloc[0]
    assert row.dt_s == 0 and np.isnan(row.speed_mps)


def test_large_gap(tmp_path):
    t = np.arange(1, 21, dtype=float)
    t[10:] += 600
    df = run(tmp_path, eastbound(t=t))
    r = df[df.obs_idx == 10].iloc[0]
    assert r.gap_flag == 1 and r.dt_s == 601
    assert df[df.obs_idx == 11].gap_flag.iloc[0] == 0


def test_stationary_vehicle(tmp_path):
    tr = np.repeat(eastbound(1), 10, axis=0)
    tr[:, 2] = np.arange(1, 11)
    df = run(tmp_path, tr)
    assert df.has_heading.eq(0).all() and df.bearing_diff_deg.isna().all()
    assert np.allclose(df.speed_mps.dropna(), 0)


def test_far_point_gets_placeholder_row(tmp_path):
    tr = eastbound(5)
    tr[2, 1] = LAT0 - 0.01  # ~1.1 km south of everything
    df = run(tmp_path, tr)
    r = df[df.obs_idx == 2]
    assert len(r) == 1 and r.arc_id.iloc[0] == -1 and r.n_candidates.iloc[0] == 0
    assert r.cand_rank.iloc[0] == -1 and np.isnan(r.perp_dist_m.iloc[0])
    assert df.obs_idx.nunique() == 5
    # fallback radius brings back the nearest arc instead
    df2 = extract_features_for_trip("T", tmp_path, cfg=SpatialConfig(k=4, radius_m=50, fallback_radius_m=2000))
    assert (df2[df2.obs_idx == 2].arc_id >= 0).all()


def test_config_file_loads():
    cfg = load_config()
    assert cfg.k >= 1 and cfg.radius_m > 0 and cfg.align.noise_m == 50.0


def test_inference_never_opens_route(tmp_path, monkeypatch):
    write_trip(tmp_path, "R", eastbound())
    import spatial.io as sio
    real = sio._read
    def guard(path, kind):
        assert kind != "route", "route file read in inference mode"
        return real(path, kind)
    monkeypatch.setattr(sio, "_read", guard)
    df = extract_features_for_trip("R", tmp_path, cfg=CFG, with_labels=False)
    assert df.is_match.isna().all()


def test_backend_entry_point(tmp_path):
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("device1_pkg", root / "__init__.py",
                                                  submodule_search_locations=[str(root)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["device1_pkg"] = pkg
    spec.loader.exec_module(pkg)
    write_trip(tmp_path / "data", "E1", eastbound(), part="evaluator_test")
    tr = eastbound(5)
    tr[2, 1] = LAT0 - 0.01
    write_trip(tmp_path / "data", "E2", tr, part="evaluator_test")
    df = pkg.extract_trip_features("E2", tmp_path / "data")
    assert df.is_match.isna().all() and (df.arc_id == -1).any()
    rows = pkg.model_rows(df)
    assert (rows.arc_id >= 0).all() and rows[pkg.FEATURE_COLUMNS].shape[1] == 12
    with pytest.raises(FileNotFoundError):
        pkg.locate_trip("NOPE", tmp_path / "data")
