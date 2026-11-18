"""Run several Device 1 jobs in one process (we ran it on a Kaggle CPU).

Usage: python scripts/run_pipeline.py RAW_ROOT OUT_ROOT STEP [STEP ...]
Steps: part1  -> OUT_ROOT/part1   (eda_part1: inventory, checks, alignment scan)
       part5  -> OUT_ROOT/part5   (validate_spatial: K/R sweep, timing)
       build  -> OUT_ROOT/build   (build_all_features: parquets + geometry)
       eda    -> OUT_ROOT/eda     (eda_spatial: plots, tables, Folium maps; needs the others)
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = {"part1": "eda_part1.py", "part5": "validate_spatial.py",
         "build": "build_all_features.py", "eda": "eda_spatial.py"}


def run_step(name: str, raw: Path, out_root: Path) -> None:
    spec = importlib.util.spec_from_file_location(name, HERE / STEPS[name])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    t = time.time()
    print(f"===== {name} =====", flush=True)
    if name == "eda":
        mod.main(raw, out_root, out_root / "eda")
    else:
        mod.main(raw, out_root / name)
    print(f"===== {name} done in {time.time() - t:.1f}s =====", flush=True)


if __name__ == "__main__":
    raw, out_root = Path(sys.argv[1]), Path(sys.argv[2])
    for step in sys.argv[3:]:
        run_step(step, raw, out_root)
