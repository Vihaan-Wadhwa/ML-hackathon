"""Spatial-engine configuration (configs/spatial.yaml)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .labels import AlignConfig

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "spatial.yaml"


@dataclass(frozen=True)
class SpatialConfig:
    crs_kind: str = "tmerc"
    k: int = 8
    radius_m: float = 50.0
    fallback_radius_m: float = 0.0
    min_move_m: float = 1.0
    gap_threshold_s: float = 1.0
    align: AlignConfig = field(default_factory=AlignConfig)

    def with_(self, **kw) -> "SpatialConfig":
        return replace(self, **kw)

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path=None) -> SpatialConfig:
    path = Path(path) if path else DEFAULT_PATH
    if not path.is_file():
        return SpatialConfig()
    import yaml

    raw = yaml.safe_load(path.read_text()) or {}
    align = AlignConfig(**raw.pop("align", {}))
    return SpatialConfig(**raw, align=align)
