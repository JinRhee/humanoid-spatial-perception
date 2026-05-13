from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class UnifiedInferenceResult:
    X: Any
    C: Any
    D: Any
    Q: Any
    desc_i: Any
    desc_j: Any
    res11: dict[str, Any]
    res21: dict[str, Any]
    res22: dict[str, Any]
    res12: dict[str, Any]


def _ensure_encoded(model: Any, frame: Any) -> None:
    if getattr(frame, "feat", None) is not None:
        return
    feat, pos, _ = model._encode_image(frame.img, frame.img_true_shape)
    frame.feat = feat
    frame.pos = pos


def _check_model(model: Any) -> None:
    for method in ("_encode_image", "_decoder", "_downstream_head"):
        if not hasattr(model, method):
            raise RuntimeError(f"MASt3R model is missing required method: {method}")


def _check_result(name: str, value: dict[str, Any]) -> None:
    for key in ("pts3d", "conf", "desc", "desc_conf"):
        if key not in value:
            raise RuntimeError(f"{name} is missing required field: {key}")


def mast3r_unified_inference(model: Any, frame_i: Any, frame_j: Any) -> UnifiedInferenceResult:
    import torch

    with torch.inference_mode():
        return _mast3r_unified_inference_impl(model, frame_i, frame_j)


def _mast3r_unified_inference_impl(model: Any, frame_i: Any, frame_j: Any) -> UnifiedInferenceResult:
    import torch

    _check_model(model)
    _ensure_encoded(model, frame_i)
    _ensure_encoded(model, frame_j)

    feat1, pos1, shape1 = frame_i.feat, frame_i.pos, frame_i.img_true_shape
    feat2, pos2, shape2 = frame_j.feat, frame_j.pos, frame_j.img_true_shape
    device_type = getattr(getattr(feat1, "device", None), "type", "cpu")
    if device_type not in {"cpu", "cuda"}:
        device_type = "cpu"

    dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
    with torch.amp.autocast(enabled=False, device_type=device_type):
        res11 = model._downstream_head(1, [tok.float() for tok in dec1], shape1)
        res21 = model._downstream_head(2, [tok.float() for tok in dec2], shape2)

    dec2r, dec1r = model._decoder(feat2, pos2, feat1, pos1)
    with torch.amp.autocast(enabled=False, device_type=device_type):
        res22 = model._downstream_head(1, [tok.float() for tok in dec2r], shape2)
        res12 = model._downstream_head(2, [tok.float() for tok in dec1r], shape1)

    _check_result("res11", res11)
    _check_result("res21", res21)
    _check_result("res22", res22)
    _check_result("res12", res12)

    rows = [res11, res21, res22, res12]
    X = torch.stack([r["pts3d"][0] for r in rows])
    C = torch.stack([r["conf"][0] for r in rows])
    D = torch.stack([r["desc"][0] for r in rows])
    Q = torch.stack([r["desc_conf"][0] for r in rows])

    desc_i = res11["desc"]
    desc_j = res21["desc"]

    return UnifiedInferenceResult(
        X=X,
        C=C,
        D=D,
        Q=Q,
        desc_i=desc_i,
        desc_j=desc_j,
        res11=res11,
        res21=res21,
        res22=res22,
        res12=res12,
    )
