from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .spatial import pose_matrix, transform_points
from .types import BackboneOutput, FrameRecord, PointMap, PoseEstimate


@dataclass
class SlamResult:
    trajectory: list[tuple[float, np.ndarray, np.ndarray]]
    pose_matrices: dict[float, np.ndarray]
    global_map: np.ndarray


def _load_callable(path: str) -> Callable[..., Any]:
    if ":" not in path:
        raise ValueError(f"Invalid factory path '{path}'. Expected module:function")
    module_name, func_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def _build_pose_matrices(trajectory: list[tuple[float, np.ndarray, np.ndarray]]) -> dict[float, np.ndarray]:
    return {ts: pose_matrix(t, q) for ts, t, q in trajectory}


def _stub_trajectory(frames: list[FrameRecord]) -> list[tuple[float, np.ndarray, np.ndarray]]:
    trajectory: list[tuple[float, np.ndarray, np.ndarray]] = []
    for i, fr in enumerate(frames):
        trans = np.array([0.03 * i, 0.0, 0.0], dtype=np.float32)
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        trajectory.append((fr.timestamp, trans, quat))
    return trajectory


def _build_global_map(frames: list[FrameRecord], backbone: BackboneOutput, pose_mats: dict[float, np.ndarray]) -> np.ndarray:
    parts: list[np.ndarray] = []
    for fr in frames:
        frame_out = backbone.frames.get((fr.sec, fr.nsec))
        if not frame_out:
            continue
        pts = frame_out.pointmap.points.reshape(-1, 3)
        pose = pose_mats.get(fr.timestamp)
        if pose is None:
            continue
        parts.append(transform_points(pts, pose))
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, 3), dtype=np.float32)


def _coerce_slam_result(raw: Any) -> SlamResult:
    if isinstance(raw, SlamResult):
        return raw
    if not isinstance(raw, dict):
        raise TypeError("SLAM adapter must return dict or SlamResult")
    trajectory = raw["trajectory"]
    pose_mats = raw.get("pose_matrices") or _build_pose_matrices(trajectory)
    if "global_map" in raw:
        global_map = np.asarray(raw["global_map"], dtype=np.float32)
    else:
        global_map = np.zeros((0, 3), dtype=np.float32)
    return SlamResult(trajectory=trajectory, pose_matrices=pose_mats, global_map=global_map)


def run_slam(frames: list[FrameRecord], backbone: BackboneOutput, cfg: dict[str, Any]) -> SlamResult:
    mode = str(cfg.get("mode", "mast3r_slam"))
    if mode == "stub":
        trajectory = _stub_trajectory(frames)
        pose_mats = _build_pose_matrices(trajectory)
        global_map = _build_global_map(frames, backbone, pose_mats)
        return SlamResult(trajectory=trajectory, pose_matrices=pose_mats, global_map=global_map)
    if mode != "mast3r_slam":
        raise ValueError(f"Unknown SLAM mode '{mode}'")
    factory = cfg.get("factory")
    if not factory:
        raise RuntimeError("MASt3R-SLAM factory not configured. Set slam.factory to a callable module:function.")
    adapter = _load_callable(factory)(config=cfg)
    raw = adapter.run(frames=frames, backbone=backbone)
    result = _coerce_slam_result(raw)
    if not result.pose_matrices:
        result.pose_matrices = _build_pose_matrices(result.trajectory)
    if result.global_map.size == 0:
        result.global_map = _build_global_map(frames, backbone, result.pose_matrices)
    return result
