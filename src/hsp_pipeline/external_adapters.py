from __future__ import annotations

from typing import Any


def _missing_adapter(name: str, hint: str) -> RuntimeError:
    return RuntimeError(
        f"{name} adapter not available. {hint} You can set the corresponding *.factory config entry "
        "to a custom module:function that returns a compatible adapter."
    )


def create_mast3r_backbone(*_args: Any, **_kwargs: Any):
    raise _missing_adapter(
        "MASt3R backbone",
        "Install external/MASt3R-SLAM and provide an adapter with run_pair/run_single methods.",
    )


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
