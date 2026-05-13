from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .types import FrameRecord


@dataclass
class GeometryOutputs:
    trajectory: list[tuple[float, np.ndarray, np.ndarray]]
    pointclouds: dict[tuple[int, int], np.ndarray]
    global_map: np.ndarray


def _image_to_points(path, max_points: int = 3000) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w, _ = arr.shape
    yy, xx = np.mgrid[0:h, 0:w]
    z = (arr.mean(axis=2) / 255.0) * 2.0 + 0.5
    x = (xx - (w / 2.0)) / max(w, 1)
    y = (yy - (h / 2.0)) / max(h, 1)
    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    if pts.shape[0] > max_points:
        idx = np.linspace(0, pts.shape[0] - 1, max_points).astype(int)
        pts = pts[idx]
    return pts


def run_geometry(frames: list[FrameRecord], _cfg: dict) -> GeometryOutputs:
    if not frames:
        return GeometryOutputs([], {}, np.zeros((0, 3), dtype=np.float32))

    trajectory: list[tuple[float, np.ndarray, np.ndarray]] = []
    pointclouds: dict[tuple[int, int], np.ndarray] = {}
    map_parts: list[np.ndarray] = []

    for i, fr in enumerate(frames):
        trans = np.array([0.03 * i, 0.0, 0.0], dtype=np.float32)
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        trajectory.append((fr.timestamp, trans, quat))

        local_pts = _image_to_points(fr.path)
        global_pts = local_pts + trans[None, :]
        pointclouds[(fr.sec, fr.nsec)] = global_pts
        map_parts.append(global_pts)

    global_map = np.concatenate(map_parts, axis=0) if map_parts else np.zeros((0, 3), dtype=np.float32)
    return GeometryOutputs(trajectory, pointclouds, global_map)


def sanity_check_geometry(trajectory: list[tuple[float, np.ndarray, np.ndarray]], pointclouds: dict[tuple[int, int], np.ndarray]) -> None:
    if not trajectory:
        raise RuntimeError("Sanity check failed: trajectory has no poses")

    ts = [x[0] for x in trajectory]
    if any(ts[i + 1] <= ts[i] for i in range(len(ts) - 1)):
        raise RuntimeError("Sanity check failed: trajectory timestamps are not monotonically increasing")

    if not pointclouds:
        raise RuntimeError("Sanity check failed: no pointclouds generated")

    if all(pc.size == 0 for pc in pointclouds.values()):
        raise RuntimeError("Sanity check failed: all pointclouds are empty")
