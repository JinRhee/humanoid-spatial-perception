from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import numpy as np

from .unified_inference import mast3r_unified_inference


def _missing_adapter(name: str, hint: str) -> RuntimeError:
    return RuntimeError(
        f"{name} adapter not available. {hint} You can set the corresponding *.factory config entry "
        "to a custom module:function that returns a compatible adapter."
    )


def _load_callable(path: str):
    if ":" not in path:
        raise RuntimeError(f"Invalid factory path '{path}'. Use module:function format.")
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
    config = _kwargs.get("config", {})
    checkpoints = _kwargs.get("checkpoints", {})
    device = str(_kwargs.get("device", "cuda"))
    fp16 = bool(_kwargs.get("fp16", False))
    model_factory = config.get("model_factory")
    if not model_factory:
        raise _missing_adapter(
            "MASt3R backbone",
            "Set backbone.model_factory to a model loader callable that returns a MASt3R model.",
        )
    loader = _load_callable(str(model_factory))
    model = loader(checkpoints=checkpoints, device=device, fp16=fp16, config=config)
    return UnifiedMast3RBackboneAdapter(model=model, device=device)


def create_mast3r_slam(*_args: Any, **_kwargs: Any):
    raise _missing_adapter(
        "MASt3R-SLAM",
        "Install external/MASt3R-SLAM and provide an adapter with a run(frames, backbone) method.",
    )


def create_sam3(*_args: Any, **_kwargs: Any):
    raise _missing_adapter(
        "SAM3",
        "Install external/sam3 and provide an adapter with a predict(image, prompts, max_masks) method.",
    )


def create_segmast3r(*_args: Any, **_kwargs: Any):
    raise _missing_adapter(
        "SegMASt3R",
        "Install external/segmast3r and provide an adapter with encode_mask(features, mask) method.",
    )
