from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins on conflict."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def load_pipeline_config(
    config_path: str | Path,
    preset: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict:
    """
    Load the pipeline config yaml, apply an optional preset, then
    apply any programmatic overrides from CLI args.

    Returns a dict named pipeline_cfg in calling scope to avoid
    shadowing MASt3R-SLAM's module-level `config` global.
    """
    config_path = Path(config_path)
    with config_path.open("r") as f:
        pipeline_cfg = yaml.safe_load(f)

    if preset is not None:
        presets = pipeline_cfg.get("presets", {})
        if preset not in presets:
            raise ValueError(
                f"Unknown preset '{preset}'. "
                f"Available: {list(presets.keys())}"
            )
        pipeline_cfg = _deep_merge(pipeline_cfg, presets[preset])

    pipeline_cfg.pop("presets", None)

    if overrides:
        pipeline_cfg = _deep_merge(pipeline_cfg, overrides)

    return pipeline_cfg


def overrides_from_args(args) -> dict:
    """
    Build an overrides dict from parsed CLI args.
    Only includes keys that differ from their argparse defaults,
    so yaml values are not silently clobbered by unfired defaults.
    """
    overrides: dict = {}

    if args.dataset:
        overrides.setdefault("dataset", {})["path"] = str(args.dataset)
    if args.calib:
        overrides.setdefault("dataset", {})["calib"] = str(args.calib)
    if getattr(args, "save_as", None) and args.save_as != "default":
        overrides.setdefault("dataset", {})["save_as"] = args.save_as
    if getattr(args, "no_viz", False):
        overrides.setdefault("slam", {})["no_viz"] = True
    if getattr(args, "seg_conf", None) is not None:
        overrides.setdefault("segmentation", {})["conf"] = args.seg_conf
    if getattr(args, "seg_iou", None) is not None:
        overrides.setdefault("segmentation", {})["iou"] = args.seg_iou
    if getattr(args, "seg_imgsz", None) is not None:
        overrides.setdefault("segmentation", {})["imgsz"] = args.seg_imgsz

    return overrides