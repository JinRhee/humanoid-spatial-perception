from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required to load configs/pipeline.yaml") from exc


@dataclass(frozen=True)
class PipelineConfig:
    raw: dict[str, Any]

    @property
    def paths(self) -> dict[str, Any]:
        return self.raw["paths"]

    @property
    def keyframe(self) -> dict[str, Any]:
        return self.raw["keyframe"]

    @property
    def geometry(self) -> dict[str, Any]:
        return self.raw["geometry"]

    @property
    def backbone(self) -> dict[str, Any]:
        return self.raw["backbone"]

    @property
    def slam(self) -> dict[str, Any]:
        return self.raw["slam"]

    @property
    def semantics(self) -> dict[str, Any]:
        return self.raw["semantics"]

    @property
    def sam3(self) -> dict[str, Any]:
        return self.raw["sam3"]

    @property
    def segmast3r(self) -> dict[str, Any]:
        return self.raw["segmast3r"]

    @property
    def fusion(self) -> dict[str, Any]:
        return self.raw["fusion"]

    @property
    def memory(self) -> dict[str, Any]:
        return self.raw["memory"]


def _deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_config(config_path: Path, preset: str | None) -> PipelineConfig:
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    required_top = {
        "paths",
        "keyframe",
        "geometry",
        "backbone",
        "slam",
        "semantics",
        "sam3",
        "segmast3r",
        "fusion",
        "memory",
        "presets",
    }
    missing = required_top.difference(raw)
    if missing:
        raise ValueError(f"Config missing top-level sections: {sorted(missing)}")

    resolved = _deep_update(raw, raw["presets"].get("balanced", {}))
    if preset:
        if preset not in raw["presets"]:
            raise ValueError(f"Unknown preset '{preset}'. Available: {sorted(raw['presets'])}")
        resolved = _deep_update(resolved, raw["presets"][preset])

    return PipelineConfig(raw=resolved)


def read_yaml_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Expected YAML mapping at {path}")
        return data


def dump_yaml_file(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
