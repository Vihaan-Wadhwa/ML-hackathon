"""Part 6.4 EDA artefacts (tables, PNG figures, Folium maps). Every artefact is labelled
with the partition(s) it uses. Reads the outputs of eda_part1 / validate_spatial /
build_all_features; re-extracts only the few trips that get a Folium map.

Usage: python scripts/eda_spatial.py RAW_ROOT OUT_ROOT [EDA_DIR]
         [--part1 DIR] [--part5 DIR] [--build DIR]   (defaults: OUT_ROOT/part1 etc.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spatial.config import load_config  # noqa: E402
from spatial.features import extract_features_for_trip  # noqa: E402
from spatial.geometry import build_geometry  # noqa: E402
from spatial.io import load_trip  # noqa: E402

# reference palette (dataviz skill, light mode): categorical slots 1-2, text, surface
C = {"train": "#2a78d6", "student_test": "#eb6834", "ink": "#0b0b0b", "ink2": "#52514e",
     "muted": "#8a8984", "grid": "#e4e3df", "surface": "#fcfcfb", "net": "#c9c8c2"}
PARTS = ["train", "student_test"]
LABEL = {"train": "train (80 trips)", "student_test": "student_test (10 trips)"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
        "axes.edgecolor": C["grid"], "axes.labelcolor": C["ink2"], "text.color": C["ink"],
        "xtick.color": C["ink2"], "ytick.color": C["ink2"], "axes.grid": True,
        "grid.color": C["grid"], "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "legend.frameon": False, "savefig.dpi": 150,
        "savefig.bbox": "tight"})
    return plt


def _save(fig, path):
    fig.savefig(path)
    import matplotlib.pyplot as plt
    plt.close(fig)


# ---------------------------------------------------------------------- tables
def tables(p1: Path, p5: Path, build: Path, eda: Path) -> dict:
    ck = pd.read_csv(p1 / "trip_checks.csv", dtype={"trip_id": str})
    inv = pd.read_csv(p1 / "inventory.csv", dtype={"trip_id": str})
    size = inv.pivot_table(index="trip_id", columns="ext", values="bytes", aggfunc="sum").add_suffix("_bytes")
    inventory = ck[["partition", "trip_id", "n_obs", "n_nodes", "n_arcs", "route_len", "has_route",
                    "t_first", "t_last"]].merge(size, left_on="trip_id", right_index=True)
    inventory["duration_s"] = inventory.t_last - inventory.t_first
    inventory.to_csv(eda / "dataset_inventory.csv", index=False)

    rows = []
    for part, g in ck.groupby("partition"):
        steps = (g.n_obs - 1).sum()
        rows.append({"partition": part, "trips": len(g), "obs_total": int(g.n_obs.sum()),
                     "obs_min": int(g.n_obs.min()), "obs_median": float(g.n_obs.median()),
                     "obs_max": int(g.n_obs.max()),
                     "dt_eq_1s_share": float((g.dt_eq_1_share * (g.n_obs - 1)).sum() / steps),
                     "steps_dt_gt_1s": int(g.dt_gt_1.sum()), "steps_dt_gt_10s": int(g.dt_gt_10.sum()),
                     "trips_with_gap_gt_10s": int((g.dt_gt_10 > 0).sum()), "dt_max_s": float(g.dt_max.max()),
                     "negative_dt": int(g.dt_negative.sum()), "zero_dt": int(g.dt_zero.sum())})
    pd.DataFrame(rows).to_csv(eda / "obs_and_dt_summary.csv", index=False)

    mv = ck.groupby("partition")[["track_nan_rows", "track_lon_out_of_range", "track_lat_out_of_range",
                                  "track_zero_coords", "dup_consecutive_fix", "nodes_nan_rows",
                                  "nodes_out_of_range", "dup_node_coords", "arcs_bad_endpoint",
                                  "arcs_self_loop", "arcs_duplicate", "route_bad_id", "route_breaks",
                                  "t_non_integer"]].sum().T
    mv.index.name = "raw_check"
    mv.to_csv(eda / "missing_invalid_raw.csv")
    nulls = {}
    for part in PARTS:
        f = build / f"{part}_candidates.parquet"
        if f.exists():
            df = pd.read_parquet(f)
            nulls[part] = df.isna().mean().round(4)
    if nulls:
        nt = pd.DataFrame(nulls)
        nt.index.name = "column (null share of candidate rows)"
        nt.to_csv(eda / "missing_values_candidates.csv")

    net = ck[["partition", "trip_id", "n_nodes", "n_arcs", "isolated_nodes", "weak_components",
              "largest_weak_share", "strong_components", "largest_strong_share",
              "arcs_with_reverse_share", "arcs_duplicate", "arcs_self_loop"]]
    net.to_csv(eda / "network_size_connectivity.csv", index=False)
    return {"checks": ck}


# ---------------------------------------------------------------------- figures
def fig_obs_dt(ck, eda):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    bins = np.logspace(np.log10(200), np.log10(25000), 25)
    for part in PARTS:
        ax.hist(ck[ck.partition == part].n_obs, bins=bins, color=C[part], alpha=0.85,
                label=LABEL[part], edgecolor=C["surface"], linewidth=1)
    ax.set_xscale("log")
    ax.set(xlabel="GPS observations per trip (log scale)", ylabel="trips",
           title="Observations per trip (train + student_test)")
    ax.legend()
    _save(fig, eda / "obs_per_trip.png")

    fig, ax = plt.subplots(figsize=(7, 3.2))
    width = 0.38
    for i, part in enumerate(PARTS):
        g = ck[ck.partition == part]
        steps = (g.n_obs - 1).sum()
        eq1 = (g.dt_eq_1_share * (g.n_obs - 1)).sum()
        gt1, gt10 = g.dt_gt_1.sum(), g.dt_gt_10.sum()
        vals = np.array([eq1, gt1 - gt10, gt10], float) / steps * 100
        ax.bar(np.arange(3) + (i - 0.5) * width, vals, width=width - 0.04, color=C[part], label=LABEL[part])
    ax.set_xticks(range(3), ["Δt = 1 s", "1 s < Δt ≤ 10 s", "Δt > 10 s"])
    ax.set_yscale("log")
    ax.set(ylabel="% of steps (log scale)", title="Sampling interval Δt (train + student_test)")
    ax.legend()
    _save(fig, eda / "dt_distribution.png")


def fig_network(ck, eda):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for part in PARTS:
        g = ck[ck.partition == part]
        ax.scatter(g.n_arcs, g.weak_components, s=28, color=C[part], label=LABEL[part],
                   edgecolor=C["surface"], linewidth=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlabel="arcs in trip network (log)", ylabel="weakly connected components (log)",
           title="Network size vs fragmentation (train + student_test)")
    ax.legend()
    _save(fig, eda / "network_size_connectivity.png")


def fig_trajectories(raw: Path, ck, eda):
    plt = _plt()
    picks = []
    for part in PARTS:   # shortest, median and longest trip of each partition
        g = ck[ck.partition == part].sort_values("n_obs")
        picks.append(g.iloc[sorted({0, len(g) // 2, len(g) - 1})])
    pick = pd.concat(picks, ignore_index=True)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, (_, r) in zip(axes.ravel(), pick.iterrows()):
        trip = load_trip(r.trip_id, raw / r.partition)
        geom = build_geometry(trip)
        from spatial.projection import project
        px, py = project(trip.track.lon, trip.track.lat, geom.crs)
        pad = 300
        box = (px.min() - pad, px.max() + pad, py.min() - pad, py.max() + pad)
        m = ((geom.ax > box[0]) & (geom.ax < box[1]) & (geom.ay > box[2]) & (geom.ay < box[3]))
        seg = np.stack([np.c_[geom.ax[m], geom.ay[m]], np.c_[geom.bx[m], geom.by[m]]], 1)
        from matplotlib.collections import LineCollection
        ax.add_collection(LineCollection(seg, colors=C["net"], linewidths=0.5))
        if trip.has_route:
            rr = trip.route.to_numpy()
            rs = np.stack([np.c_[geom.ax[rr], geom.ay[rr]], np.c_[geom.bx[rr], geom.by[rr]]], 1)
            ax.add_collection(LineCollection(rs, colors=C["ink"], linewidths=1.2, alpha=0.8, zorder=2))
        ax.scatter(px, py, s=5, color=C[r.partition], linewidths=0, zorder=3)
        ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3])
        ax.set_aspect("equal"); ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{r.partition}/{r.trip_id}  ({r.n_obs} fixes)", fontsize=9)
    for ax in axes.ravel()[len(pick):]:
        ax.set_visible(False)
    fig.suptitle("GPS fixes (colour), reference route (black), road network (grey)", fontsize=11)
    _save(fig, eda / "trajectories.png")


def fig_miss_and_grid(p5: Path, eda, cfg):
    plt = _plt()
    mt = pd.read_csv(p5 / "miss_by_trip.csv", dtype={"trip_id": str})
    fig, ax = plt.subplots(figsize=(7, 3.2))
    bins = np.linspace(0, max(0.1, mt.miss_rate.max() + 0.005), 30)
    ax.hist([mt[mt.partition == p].miss_rate * 100 for p in PARTS], bins=bins * 100, stacked=True,
            color=[C[p] for p in PARTS], label=[LABEL[p] for p in PARTS], edgecolor=C["surface"],
            linewidth=1)
    ax.axvline(5, color=C["ink2"], linewidth=1, linestyle="--")
    ax.text(5, ax.get_ylim()[1] * 0.92, " 5% target", color=C["ink2"], fontsize=9)
    ax.set(xlabel="candidate miss rate per trip (%)", ylabel="trips",
           title=f"Miss rate per trip at K={cfg.k}, R={cfg.radius_m:g} m (labelled fixes)")
    ax.legend()
    _save(fig, eda / "miss_rate_hist.png")

    grid = pd.read_csv(p5 / "kr_grid.csv")
    g = grid[grid.partition == "train+student_test"]
    piv = g.pivot(index="k", columns="radius_m", values="miss_rate_pooled") * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(piv.values, cmap="Blues", aspect="auto", origin="lower", vmin=0, vmax=10)
    ax.set_xticks(range(piv.shape[1]), [f"{c:g}" for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 6 else C["ink"])
    ax.grid(False)
    ax.set(xlabel="radius R (m)", ylabel="K (max candidates)",
           title="Pooled miss rate % (train + student_test, labelled fixes)")
    fig.colorbar(im, ax=ax, label="miss rate % (colour capped at 10)")
    _save(fig, eda / "kr_grid_miss_rate.png")

    tm = pd.read_csv(p5 / "timing.csv", dtype={"trip_id": str})
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for mode, col in (("inference", C["train"]), ("full", C["student_test"])):
        s = tm[tm["mode"] == mode]
        ax.scatter(s.n_arcs, s.total_s, s=22, color=col, edgecolor=C["surface"], linewidth=1,
                   label={"inference": "inference (no labels)", "full": "with labels"}[mode])
    ax.axhline(1.5, color=C["ink2"], linewidth=1, linestyle="--")
    ax.set_xscale("log")
    ax.set(xlabel="arcs in trip network (log)", ylabel="seconds per trip",
           title="extract_features_for_trip time (train + student_test, Kaggle CPU)")
    ax.legend()
    _save(fig, eda / "timing.png")
    return mt


# ---------------------------------------------------------------------- Folium
def folium_map(raw: Path, part: str, tid: str, out: Path, cfg, max_pts: int = 4000) -> bool:
    try:
        import folium
    except ImportError:
        print("folium not installed: skipping", out.name)
        return False
    trip = load_trip(tid, raw / part)
    df, al, geom = extract_features_for_trip(tid, cfg=cfg, trip=trip, return_alignment=True)
    lon, lat = trip.track.lon.to_numpy(), trip.track.lat.to_numpy()
    nlon, nlat = trip.nodes.lon.to_numpy(), trip.nodes.lat.to_numpy()
    m = folium.Map(location=[float(np.median(lat)), float(np.median(lon))], zoom_start=14,
                   tiles="OpenStreetMap")
    if trip.has_route:
        fg = folium.FeatureGroup("reference route (.route)")
        for a in trip.route.to_numpy():
            f, t = geom.from_node[a], geom.to_node[a]
            folium.PolyLine([(nlat[f], nlon[f]), (nlat[t], nlon[t])], color="#0b0b0b", weight=4,
                            opacity=0.7).add_to(fg)
        fg.add_to(m)
    top = df[df.cand_rank == 0]
    fg = folium.FeatureGroup("rank-0 candidate arc", show=False)
    for a in np.unique(top.arc_id[top.arc_id >= 0]):
        f, t = geom.from_node[a], geom.to_node[a]
        folium.PolyLine([(nlat[f], nlon[f]), (nlat[t], nlon[t])], color="#1baf7a", weight=3).add_to(fg)
    fg.add_to(m)
    per = df.groupby("obs_idx").agg(hit=("is_match", "max"), lab=("is_match", "count"),
                                    n=("n_candidates", "first"))
    step = max(1, len(lon) // max_pts)
    groups = {"hit": folium.FeatureGroup("GPS: reference arc in candidates"),
              "miss": folium.FeatureGroup("GPS: MISS"),
              "unl": folium.FeatureGroup("GPS: unlabelled (ref > 50 m)")}
    colour = {"hit": "#2a78d6", "miss": "#e34948", "unl": "#8a8984"}
    for i in range(0, len(lon), step):
        r = per.loc[i]
        kind = "unl" if r.lab == 0 else ("hit" if r.hit == 1 else "miss")
        if kind == "hit" and i % (step * 2):
            continue
        folium.CircleMarker((lat[i], lon[i]), radius=3 if kind == "hit" else 4, color=colour[kind],
                            fill=True, fill_opacity=0.9, weight=0,
                            tooltip=f"obs {i}  {kind}  n_cand={int(r.n)}  "
                                    f"ref_dist={al.align_dist_m.iloc[i]:.1f} m" if al is not None else f"obs {i}"
                            ).add_to(groups[kind])
    for g in groups.values():
        g.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    title = (f"<div style='position:fixed;top:10px;left:50px;z-index:9999;background:#fcfcfb;"
             f"padding:6px 10px;font:13px sans-serif;border:1px solid #e4e3df'>"
             f"<b>{part}/{tid}</b> &middot; K={cfg.k}, R={cfg.radius_m:g} m &middot; "
             f"blue = hit, red = miss, grey = unlabelled</div>")
    m.get_root().html.add_child(folium.Element(title))
    m.save(str(out))
    return True


def main(raw: Path, out_root: Path, eda: Path, part1=None, part5=None, build=None) -> None:
    raw, out_root, eda = Path(raw), Path(out_root), Path(eda)
    p1 = Path(part1) if part1 else out_root / "part1"
    p5 = Path(part5) if part5 else out_root / "part5"
    b = Path(build) if build else out_root / "build"
    eda.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    ck = tables(p1, p5, b, eda)["checks"]
    fig_obs_dt(ck, eda)
    fig_network(ck, eda)
    fig_trajectories(raw, ck, eda)
    mt = fig_miss_and_grid(p5, eda, cfg)
    maps = []
    worst = mt.sort_values("miss_rate", ascending=False)
    choices = [tuple(worst.iloc[0][["partition", "trip_id"]]),
               tuple(worst[worst.partition == "student_test"].iloc[0][["partition", "trip_id"]]),
               ("train", "00000054")]
    present = set(zip(ck.partition, ck.trip_id))
    for part, tid in [c for c in dict.fromkeys(choices) if c in present]:
        out = eda / f"map_{part}_{tid}.html"
        if folium_map(raw, part, tid, out, cfg):
            maps.append(out.name)
    (eda / "eda_index.json").write_text(json.dumps(
        {"partitions_used": PARTS, "evaluator_data_used": False,
         "files": sorted(p.name for p in eda.iterdir())}, indent=2))
    print("EDA artefacts:", sorted(p.name for p in eda.iterdir()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out_root")
    ap.add_argument("eda", nargs="?")
    ap.add_argument("--part1")
    ap.add_argument("--part5")
    ap.add_argument("--build")
    a = ap.parse_args()
    main(a.raw, a.out_root, a.eda or Path(a.out_root) / "eda", a.part1, a.part5, a.build)
