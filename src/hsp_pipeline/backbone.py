from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
from PIL import Image

from .types import (
    BackboneFrameOutput,
    BackboneOutput,
    FrameRecord,
    Mast3rFeaturesV1,
    Mast3rFeaturesV2,
    PairwiseBackboneOutput,
    PointMap,
)
from .unified_inference import mast3r_unified_inference


@dataclass(frozen=True)
class BackboneConfig:
    mode: str
    factory: str | None
    model_factory: str | None
    max_image_resolution: int


def _load_callable(path: str) -> Callable[..., Any]:
    if ":" not in path:
        raise ValueError(f"Invalid factory path '{path}'. Expected module:function")
    module_name, func_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _make_frame(rgb: np.ndarray, device: str):
    import torch

    arr = np.asarray(rgb, dtype=np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device)
    return SimpleNamespace(
        img=tensor,
        img_true_shape=tensor.shape[-2:],
        feat=None,
        pos=None,
    )


class UnifiedMast3RBackboneAdapter:
    def __init__(self, model: Any, device: str) -> None:
        self.model = model
        self.device = device

    def _frame_result(self, ref: dict[str, Any], alt: dict[str, Any]) -> dict[str, Any]:
        return {
            "pointmap": _to_numpy(ref["pts3d"][0]).astype(np.float32),
            "pointmap_confidence": _to_numpy(ref["conf"][0]).astype(np.float32),
            "features_v1": _to_numpy(ref["desc"][0]).astype(np.float32),
            "features_v1_confidence": _to_numpy(ref["desc_conf"][0]).astype(np.float32),
            "features_v2": _to_numpy(alt["desc"][0]).astype(np.float32),
            "features_v2_confidence": _to_numpy(alt["desc_conf"][0]).astype(np.float32),
        }

    def run_pair(self, rgb_i: np.ndarray, rgb_j: np.ndarray) -> dict[str, Any]:
        frame_i = _make_frame(rgb_i, self.device)
        frame_j = _make_frame(rgb_j, self.device)
        out = mast3r_unified_inference(self.model, frame_i, frame_j)
        slam = {
            "X": _to_numpy(out.X).astype(np.float32),
            "C": _to_numpy(out.C).astype(np.float32),
            "D": _to_numpy(out.D).astype(np.float32),
            "Q": _to_numpy(out.Q).astype(np.float32),
        }
        seg = {
            "desc_i": _to_numpy(out.desc_i).astype(np.float32),
            "desc_j": _to_numpy(out.desc_j).astype(np.float32),
        }
        return {
            "frame_i": self._frame_result(out.res11, out.res12),
            "frame_j": self._frame_result(out.res21, out.res22),
            "slam": slam,
            "seg": seg,
        }

    def run_single(self, rgb: np.ndarray) -> dict[str, Any]:
        return self.run_pair(rgb, rgb)["frame_i"]


def create_mast3r_backbone(*_args: Any, **_kwargs: Any):
    config = _kwargs.get("config") or {}
    checkpoints = _kwargs.get("checkpoints", {})
    device = str(_kwargs.get("device", "cuda"))
    fp16 = bool(_kwargs.get("fp16", False))
    model_factory = config.get("model_factory")
    if not model_factory:
        raise RuntimeError(
            "MASt3R backbone model_factory not configured. Set backbone.model_factory to a model loader callable."
        )
    loader = _load_callable(str(model_factory))
    model = loader(checkpoints=checkpoints, device=device, fp16=fp16, config=config)
    return UnifiedMast3RBackboneAdapter(model=model, device=device)


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


def _extract_pair_outputs(raw_pair: Any) -> tuple[Any, Any, dict[str, Any] | None]:
    if isinstance(raw_pair, tuple) and len(raw_pair) == 2:
        return raw_pair[0], raw_pair[1], None
    if isinstance(raw_pair, dict):
        if "frame_i" in raw_pair and "frame_j" in raw_pair:
            pair = {}
            if "slam" in raw_pair:
                pair["slam"] = raw_pair["slam"]
            if "seg" in raw_pair:
                pair["seg"] = raw_pair["seg"]
            return raw_pair["frame_i"], raw_pair["frame_j"], (pair or None)
    raise TypeError("Backbone adapter run_pair output must be (frame_i, frame_j) or dict with frame_i/frame_j")


def _run_mast3r_backbone(frames: list[FrameRecord], cfg: BackboneConfig, checkpoints: dict[str, Any], device: str, fp16: bool) -> BackboneOutput:
    if not cfg.factory:
        raise RuntimeError(
            "MASt3R backbone factory not configured. Set backbone.factory to a callable module:function."
        )
    factory = _load_callable(cfg.factory)
    adapter_config: dict[str, Any] = {"max_image_resolution": cfg.max_image_resolution}
    if cfg.model_factory:
        adapter_config["model_factory"] = cfg.model_factory
    adapter = factory(checkpoints=checkpoints, device=device, fp16=fp16, config=adapter_config)
    outputs: dict[tuple[int, int], BackboneFrameOutput] = {}
    pairs: list[tuple[FrameRecord, FrameRecord]] = []
    pairwise_outputs: list[PairwiseBackboneOutput] = []
    for i in range(len(frames) - 1):
        fa, fb = frames[i], frames[i + 1]
        rgb_a = _load_rgb(fa.path, cfg.max_image_resolution)
        rgb_b = _load_rgb(fb.path, cfg.max_image_resolution)
        raw_a, raw_b, pair_data = _extract_pair_outputs(adapter.run_pair(rgb_a, rgb_b))
        out_a = _coerce_frame_output(fa, rgb_a, raw_a)
        out_b = _coerce_frame_output(fb, rgb_b, raw_b)
        _validate_frame_output(fa, out_a)
        _validate_frame_output(fb, out_b)
        outputs[(fa.sec, fa.nsec)] = out_a
        outputs[(fb.sec, fb.nsec)] = out_b
        pairs.append((fa, fb))
        if pair_data is not None:
            pairwise_outputs.append(
                PairwiseBackboneOutput(
                    frame_i=(fa.sec, fa.nsec),
                    frame_j=(fb.sec, fb.nsec),
                    slam=dict(pair_data.get("slam", {})),
                    seg=dict(pair_data.get("seg", {})),
                )
            )
    if len(frames) == 1:
        fr = frames[0]
        rgb = _load_rgb(fr.path, cfg.max_image_resolution)
        raw = adapter.run_single(rgb)
        out = _coerce_frame_output(fr, rgb, raw)
        _validate_frame_output(fr, out)
        outputs[(fr.sec, fr.nsec)] = out
    return BackboneOutput(frames=outputs, pairs=pairs, pairwise_outputs=pairwise_outputs)


def run_backbone(frames: list[FrameRecord], cfg: dict[str, Any], memory_cfg: dict[str, Any], checkpoints: dict[str, Any]) -> BackboneOutput:
    mode = str(cfg.get("mode", "mast3r"))
    cfg_max_res = cfg.get("max_image_resolution")
    if cfg_max_res is None:
        cfg_max_res = memory_cfg.get("max_image_resolution")
    max_res = int(cfg_max_res) if cfg_max_res is not None else 0
    factory = cfg.get("factory")
    model_factory = cfg.get("model_factory")
    device = str(cfg.get("device", memory_cfg.get("device", "cuda")))
    fp16 = bool(cfg.get("fp16", memory_cfg.get("fp16", False)))
    backbone_cfg = BackboneConfig(mode=mode, factory=factory, model_factory=model_factory, max_image_resolution=max_res)
    if mode == "stub":
        return _run_stub_backbone(frames, backbone_cfg)
    if mode != "mast3r":
        raise ValueError(f"Unknown backbone mode '{mode}'")
    return _run_mast3r_backbone(frames, backbone_cfg, checkpoints, device=device, fp16=fp16)
