"""Light Part 1 tests (synthetic data + the dummy parquet). Run: pytest tests/"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from spatial.io import load_trip, validate_trip  # noqa: E402
from spatial.labels import AlignConfig, align_route  # noqa: E402
from spatial.schema import ARROW_SCHEMA, COLUMN_NAMES, to_arrow, validate_table  # noqa: E402


def _write_trip(root: Path, tid: str, track, nodes, arcs, route=None, part="train"):
    d = root / part / tid
    d.mkdir(parents=True)
    np.savetxt(d / f"{tid}.track", track, fmt="%.6f", delimiter="\t")
    np.savetxt(d / f"{tid}.nodes", nodes, fmt="%.6f", delimiter="\t")
    np.savetxt(d / f"{tid}.arcs", arcs, fmt="%d", delimiter="\t")
    if route is not None:
        np.savetxt(d / f"{tid}.route", route, fmt="%d")


@pytest.fixture
def out_and_back(tmp_path):
    # straight east-west street of 4 nodes ~ 80 m apart, arcs both directions
    lon0, lat0 = 77.12, 43.39
    nodes = np.array([[lon0 + k * 0.001, lat0] for k in range(4)])
    arcs = np.array([[0, 1], [1, 2], [2, 3], [3, 2], [2, 1], [1, 0]])  # ids 0..5
    route = np.array([0, 1, 2, 3, 4, 5])                                # out and back
    xs = np.r_[np.linspace(0, 3, 16), np.linspace(3, 0, 16)[1:]]
    track = np.c_[lon0 + xs * 0.001, np.full(len(xs), lat0 + 0.00003), np.arange(1, len(xs) + 1)]
    _write_trip(tmp_path, "T1", track, nodes, arcs, route)
    _write_trip(tmp_path, "T2", track, nodes, arcs, None, part="student_test")
    return tmp_path


def test_load_trip_ids_and_route(out_and_back):
    t = load_trip("T1", out_and_back)
    assert list(t.track.columns) == ["lon", "lat", "t"]
    assert t.arcs.index[0] == 0 and t.nodes.index[0] == 0
    assert t.has_route and len(t.route) == 6
    t2 = load_trip("T2", out_and_back)  # no .route: must not fail
    assert not t2.has_route


def test_validate_flags(out_and_back):
    v = validate_trip(load_trip("T1", out_and_back))
    assert v["arcs_bad_endpoint"] == 0 and v["route_breaks"] == 0
    assert v["arcs_with_reverse_share"] == 1.0 and v["utm_zones"] == "43"
    assert v["dt_negative"] == 0 and v["weak_components"] == 1


def test_alignment_monotonic_and_direction(out_and_back):
    t = load_trip("T1", out_and_back)
    al = align_route(t, AlignConfig(lam_m=1.0))
    assert (np.diff(al["route_pos"]) >= 0).all()
    # first half travels east (arcs 0-2), second half west (arcs 3-5)
    assert set(al["ref_arc_id"].iloc[1:14]) <= {0, 1, 2}
    assert set(al["ref_arc_id"].iloc[-13:]) <= {3, 4, 5}
    assert al["align_dist_m"].max() < 5


def test_alignment_requires_route(out_and_back):
    with pytest.raises(ValueError):
        align_route(load_trip("T2", out_and_back))


def test_sample_parquet_matches_schema():
    from make_sample_candidates import make_sample

    tbl = to_arrow(make_sample())
    validate_table(tbl)
    p = ROOT / "outputs" / "sample_candidates.parquet"
    if p.exists():
        validate_table(pq.read_table(p))
    df = tbl.to_pandas()
    assert len(df) == 50 and list(df.columns) == COLUMN_NAMES
    assert (df.groupby("obs_idx")["is_match"].sum() <= 1).all()
    empty = df[df["n_candidates"] == 0]
    assert (empty["arc_id"] == -1).all() and empty["perp_dist_m"].isna().all()


def test_schema_rejects_extra_columns():
    df = pd.DataFrame({c: [] for c in COLUMN_NAMES + ["oops"]})
    with pytest.raises(ValueError):
        to_arrow(df)
    assert ARROW_SCHEMA.metadata[b"schema_version"] == b"1.0"


@pytest.mark.parametrize("lon0,lat0", [(77.19, 43.44), (-58.45, -34.62), (12.0, 56.0), (137.2, 36.7)])
def test_local_tmerc_matches_haversine(lon0, lat0):
    from spatial.projection import haversine_m, local_tmerc, project

    rng = np.random.default_rng(1)
    lon = lon0 + rng.uniform(-0.5, 0.5, (200, 2))
    lat = lat0 + rng.uniform(-0.3, 0.3, (200, 2))
    x, y = project(lon.ravel(), lat.ravel(), local_tmerc(lon0, lat0))
    x, y = x.reshape(200, 2), y.reshape(200, 2)
    d_proj = np.hypot(x[:, 0] - x[:, 1], y[:, 0] - y[:, 1])
    d_hav = haversine_m(lon[:, 0], lat[:, 0], lon[:, 1], lat[:, 1])
    assert np.max(np.abs(d_proj / d_hav - 1)) < 4e-3  # sphere vs ellipsoid dominates


def test_is_match_for_and_miss_rate():
    from spatial.labels import is_match_for, miss_rate

    al = pd.DataFrame({"ref_arc_id": [7, 7, 9], "align_dist_m": [3.0, 80.0, 10.0]})
    obs = np.array([0, 0, 1, 1, 2, 2])
    arc = np.array([7, 8, 7, 8, 1, 2])
    ref, m = is_match_for(arc, obs, al)
    assert ref.tolist() == [7, 7, 7, 7, 9, 9]
    assert m[:2].tolist() == [1.0, 0.0] and np.isnan(m[2:4]).all() and m[4:].tolist() == [0, 0]
    r = miss_rate(pd.DataFrame({"trip_id": "X", "obs_idx": obs, "is_match": m}))
    assert r["n_labelled"] == 2 and r["miss_rate"] == 0.5 and abs(r["unlabelled_share"] - 1 / 3) < 1e-9
