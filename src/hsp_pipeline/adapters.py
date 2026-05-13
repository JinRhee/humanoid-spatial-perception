from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .types import Mast3rFeaturesV1, Mast3rFeaturesV2


@dataclass(frozen=True)
class Sam3MaskResult:
    label: str
    mask: np.ndarray
    score: float


def _load_callable(path: str) -> Callable[..., Any]:
    if ":" not in path:
        raise ValueError(f"Invalid factory path '{path}'. Expected module:function")
    module_name, func_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def _simple_prompt_masks(h: int, w: int, prompt_count: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    stripe_w = max(1, w // max(prompt_count, 1))
    for i in range(prompt_count):
        x0 = i * stripe_w
        x1 = w if i == prompt_count - 1 else min(w, (i + 1) * stripe_w)
        mask = np.zeros((h, w), dtype=bool)
        mask[:, x0:x1] = True
        masks.append(mask)
    return masks


def _resample_descriptor(vec: np.ndarray, target_dim: int) -> np.ndarray:
    src = vec.astype(np.float32).reshape(-1)
    if src.size == 0:
        out = np.ones((target_dim,), dtype=np.float32) * 1e-6
        return out
    if src.size == target_dim:
        out = src
    else:
        xp = np.linspace(0.0, 1.0, num=src.size)
        xq = np.linspace(0.0, 1.0, num=target_dim)
        out = np.interp(xq, xp, src).astype(np.float32)
    norm = np.linalg.norm(out) + 1e-9
    return out / norm


class Sam3Adapter:
    def __init__(self, cfg: dict[str, Any], checkpoints: dict[str, Any]) -> None:
        self.mode = str(cfg.get("mode", "sam3"))
        self.factory = cfg.get("factory")
        self.checkpoint_key = str(cfg.get("checkpoint_key", "sam3"))
        self._model = None
        if self.mode == "sam3":
            if not self.factory:
                raise RuntimeError("SAM3 factory not configured. Set sam3.factory to a callable module:function.")
            factory = _load_callable(self.factory)
            self._model = factory(checkpoint=checkpoints.get(self.checkpoint_key, ""))
        elif self.mode != "stub":
            raise ValueError(f"Unknown SAM3 mode '{self.mode}'")

    def predict(self, image: np.ndarray, prompts: list[str], max_masks: int) -> list[Sam3MaskResult]:
        if self.mode == "stub":
            h, w, _ = image.shape
            limited = prompts[:max_masks] if max_masks > 0 else prompts
            masks = _simple_prompt_masks(h, w, len(limited))
            return [
                Sam3MaskResult(label=label, mask=mask, score=float(mask.mean()))
                for label, mask in zip(limited, masks)
            ]
        raw = self._model.predict(image=image, prompts=prompts, max_masks=max_masks)
        results: list[Sam3MaskResult] = []
        for idx, item in enumerate(raw):
            if isinstance(item, Sam3MaskResult):
                results.append(item)
                continue
            if not isinstance(item, dict):
                raise TypeError("SAM3 adapter must return dicts with mask/score or Sam3MaskResult")
            label = str(item.get("label", prompts[idx] if idx < len(prompts) else ""))
            mask = np.asarray(item["mask"], dtype=bool)
            score = float(item.get("score", 0.0))
            results.append(Sam3MaskResult(label=label, mask=mask, score=score))
        return results


class SegMASt3RAdapter:
    def __init__(self, cfg: dict[str, Any], checkpoints: dict[str, Any]) -> None:
        self.mode = str(cfg.get("mode", "segmast3r"))
        self.factory = cfg.get("factory")
        self.checkpoint_key = str(cfg.get("checkpoint_key", "segmast3r_head"))
        self.descriptor_dim = int(cfg.get("descriptor_dim", 24))
        if self.mode == "segmast3r":
            if not self.factory:
                raise RuntimeError(
                    "SegMASt3R factory not configured. Set segmast3r.factory to a callable module:function."
                )
            factory = _load_callable(self.factory)
            self._model = factory(checkpoint=checkpoints.get(self.checkpoint_key, ""))
        elif self.mode == "stub":
            self._model = None
        else:
            raise ValueError(f"Unknown SegMASt3R mode '{self.mode}'")

    def encode_mask(self, features: Mast3rFeaturesV1 | Mast3rFeaturesV2, mask: np.ndarray) -> np.ndarray:
        if self.mode == "segmast3r":
            raw = self._model.encode_mask(features=features.grid, mask=mask)
            return _resample_descriptor(np.asarray(raw, dtype=np.float32), self.descriptor_dim)
        feats = features.grid
        if feats.shape[:2] != mask.shape:
            raise ValueError("SegMASt3R stub expects mask and features grid to match")
        if feats.ndim != 3:
            raise ValueError("SegMASt3R stub expects features grid shape (H, W, C)")
        vec = feats[mask].mean(axis=0) if np.any(mask) else np.zeros((feats.shape[2],), dtype=np.float32)
        return _resample_descriptor(vec, self.descriptor_dim)


SegMast3rAdapter = SegMASt3RAdapter
