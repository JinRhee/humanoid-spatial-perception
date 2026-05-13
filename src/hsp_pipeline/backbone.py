from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from .types import (
    BackboneFrameOutput,
    BackboneOutput,
    FrameRecord,
    Mast3rFeaturesV1,
    Mast3rFeaturesV2,
    PointMap,
)


@dataclass(frozen=True)
class BackboneConfig:
    mode: str
    factory: str | None
    max_image_resolution: int


def _load_callable(path: str) -> Callable[..., Any]:
    if ":" not in path:
        raise ValueError(f"Invalid factory path '{path}'. Expected module:function")
    module_name, func_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def _load_rgb(path: Path, max_image_resolution: int) -> np.ndarray:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        if max_image_resolution and max_image_resolution > 0:
            w0, h0 = rgb.size
            long_edge = max(w0, h0)
            if long_edge > max_image_resolution:
                scale = max_image_resolution / long_edge
                rgb = rgb.resize((int(w0 * scale), int(h0 * scale)))
        return np.asarray(rgb, dtype=np.float32)


def _dense_pointmap(rgb: np.ndarray) -> PointMap:
    h, w, _ = rgb.shape
    yy, xx = np.mgrid[0:h, 0:w]
    z = (rgb.mean(axis=2) / 255.0) * 2.0 + 0.5
    x = (xx - (w / 2.0)) / max(w, 1)
    y = (yy - (h / 2.0)) / max(h, 1)
    pts = np.stack([x, y, z], axis=-1).astype(np.float32)
    conf = np.clip(1.0 - np.abs(z - z.mean()) / (z.std() + 1e-6), 0.0, 1.0).astype(np.float32)
    return PointMap(points=pts, confidence=conf)


def _simple_features(rgb: np.ndarray) -> tuple[Mast3rFeaturesV1, Mast3rFeaturesV2]:
    norm = rgb / 255.0
    mean = norm.mean(axis=2, keepdims=True)
    feats = np.concatenate([norm, mean, norm * norm], axis=2).astype(np.float32)
    conf = np.clip(mean[..., 0], 0.0, 1.0).astype(np.float32)
    return (
        Mast3rFeaturesV1(grid=feats, confidence=conf),
        Mast3rFeaturesV2(grid=feats, confidence=conf),
    )


def _validate_frame_output(frame: FrameRecord, out: BackboneFrameOutput) -> None:
    if out.pointmap.points.shape[:2] != out.pointmap.confidence.shape:
        raise ValueError(f"Pointmap/conf shape mismatch for frame {frame.path}")
    if out.features_v1.grid.shape[:2] != out.features_v1.confidence.shape:
        raise ValueError(f"Features v1/conf shape mismatch for frame {frame.path}")
    if out.features_v2.grid.shape[:2] != out.features_v2.confidence.shape:
        raise ValueError(f"Features v2/conf shape mismatch for frame {frame.path}")


def _run_stub_backbone(frames: list[FrameRecord], cfg: BackboneConfig) -> BackboneOutput:
    outputs: dict[tuple[int, int], BackboneFrameOutput] = {}
    pairs: list[tuple[FrameRecord, FrameRecord]] = []
    for i, fr in enumerate(frames):
        rgb = _load_rgb(fr.path, cfg.max_image_resolution)
        pointmap = _dense_pointmap(rgb)
        feat_v1, feat_v2 = _simple_features(rgb)
        out = BackboneFrameOutput(frame=fr, rgb=rgb, features_v1=feat_v1, features_v2=feat_v2, pointmap=pointmap)
        _validate_frame_output(fr, out)
        outputs[(fr.sec, fr.nsec)] = out
        if i > 0:
            pairs.append((frames[i - 1], fr))
    return BackboneOutput(frames=outputs, pairs=pairs)


def _coerce_frame_output(frame: FrameRecord, rgb: np.ndarray, raw: Any) -> BackboneFrameOutput:
    if isinstance(raw, BackboneFrameOutput):
        return raw
    if not isinstance(raw, dict):
        raise TypeError("Backbone adapter must return dict or BackboneFrameOutput")
    pointmap = PointMap(points=np.asarray(raw["pointmap"], dtype=np.float32), confidence=np.asarray(raw["pointmap_confidence"], dtype=np.float32))
    v1 = Mast3rFeaturesV1(
        grid=np.asarray(raw["features_v1"], dtype=np.float32),
        confidence=np.asarray(raw["features_v1_confidence"], dtype=np.float32),
    )
    v2 = Mast3rFeaturesV2(
        grid=np.asarray(raw["features_v2"], dtype=np.float32),
        confidence=np.asarray(raw["features_v2_confidence"], dtype=np.float32),
    )
    return BackboneFrameOutput(frame=frame, rgb=rgb, features_v1=v1, features_v2=v2, pointmap=pointmap)


def _run_mast3r_backbone(frames: list[FrameRecord], cfg: BackboneConfig, checkpoints: dict[str, Any], device: str, fp16: bool) -> BackboneOutput:
    if not cfg.factory:
        raise RuntimeError(
            "MASt3R backbone factory not configured. Set backbone.factory to a callable module:function."
        )
    factory = _load_callable(cfg.factory)
    adapter = factory(checkpoints=checkpoints, device=device, fp16=fp16, config={"max_image_resolution": cfg.max_image_resolution})
    outputs: dict[tuple[int, int], BackboneFrameOutput] = {}
    pairs: list[tuple[FrameRecord, FrameRecord]] = []
    for i in range(len(frames) - 1):
        fa, fb = frames[i], frames[i + 1]
        rgb_a = _load_rgb(fa.path, cfg.max_image_resolution)
        rgb_b = _load_rgb(fb.path, cfg.max_image_resolution)
        raw_a, raw_b = adapter.run_pair(rgb_a, rgb_b)
        out_a = _coerce_frame_output(fa, rgb_a, raw_a)
        out_b = _coerce_frame_output(fb, rgb_b, raw_b)
        _validate_frame_output(fa, out_a)
        _validate_frame_output(fb, out_b)
        outputs[(fa.sec, fa.nsec)] = out_a
        outputs[(fb.sec, fb.nsec)] = out_b
        pairs.append((fa, fb))
    if len(frames) == 1:
        fr = frames[0]
        rgb = _load_rgb(fr.path, cfg.max_image_resolution)
        raw = adapter.run_single(rgb)
        out = _coerce_frame_output(fr, rgb, raw)
        _validate_frame_output(fr, out)
        outputs[(fr.sec, fr.nsec)] = out
    return BackboneOutput(frames=outputs, pairs=pairs)


def run_backbone(frames: list[FrameRecord], cfg: dict[str, Any], memory_cfg: dict[str, Any], checkpoints: dict[str, Any]) -> BackboneOutput:
    mode = str(cfg.get("mode", "mast3r"))
    max_res = int(cfg.get("max_image_resolution", 0) or memory_cfg.get("max_image_resolution", 0) or 0)
    factory = cfg.get("factory")
    device = str(cfg.get("device", memory_cfg.get("device", "cuda")))
    fp16 = bool(cfg.get("fp16", memory_cfg.get("fp16", False)))
    backbone_cfg = BackboneConfig(mode=mode, factory=factory, max_image_resolution=max_res)
    if mode == "stub":
        return _run_stub_backbone(frames, backbone_cfg)
    if mode != "mast3r":
        raise ValueError(f"Unknown backbone mode '{mode}'")
    return _run_mast3r_backbone(frames, backbone_cfg, checkpoints, device=device, fp16=fp16)
