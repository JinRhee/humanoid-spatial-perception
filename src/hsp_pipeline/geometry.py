from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .slam import run_slam
from .spatial import transform_points
from .types import BackboneOutput, FrameRecord, PointMap


@dataclass
class GeometryOutputs:
    trajectory: list[tuple[float, np.ndarray, np.ndarray]]
    pointclouds: dict[tuple[int, int], np.ndarray]
    global_map: np.ndarray

    pointmaps: dict[tuple[int, int], PointMap] = field(default_factory=dict)
    pose_matrices: dict[float, np.ndarray] = field(default_factory=dict)


def _pointmap_points(pointmap: PointMap) -> np.ndarray:
    pts = pointmap.points.reshape(-1, 3)
    return pts


def run_geometry(frames: list[FrameRecord], backbone: BackboneOutput, geom_cfg: dict, slam_cfg: dict) -> GeometryOutputs:
    if not frames:
        return GeometryOutputs([], {}, np.zeros((0, 3), dtype=np.float32))

    slam = run_slam(frames, backbone, slam_cfg)
    pointclouds: dict[tuple[int, int], np.ndarray] = {}
    pointmaps: dict[tuple[int, int], PointMap] = {}
    map_parts: list[np.ndarray] = []

    for fr in frames:
        frame_out = backbone.frames.get((fr.sec, fr.nsec))
        if not frame_out:
            raise RuntimeError(f"Missing backbone output for frame {fr.path}")
        pointmaps[(fr.sec, fr.nsec)] = frame_out.pointmap
        pose = slam.pose_matrices.get(fr.timestamp)
        if pose is None:
            raise RuntimeError(f"Missing pose for frame {fr.path}")
        pts_local = _pointmap_points(frame_out.pointmap)
        pts_global = transform_points(pts_local, pose)
        pointclouds[(fr.sec, fr.nsec)] = pts_global
        map_parts.append(pts_global)

    global_map = slam.global_map
    if global_map.size == 0 and map_parts:
        global_map = np.concatenate(map_parts, axis=0)

    outputs = GeometryOutputs(
        trajectory=slam.trajectory,
        pointclouds=pointclouds,
        global_map=global_map,
    )
    outputs.pointmaps = pointmaps
    outputs.pose_matrices = slam.pose_matrices
    return outputs


def sanity_check_geometry(
    trajectory: list[tuple[float, np.ndarray, np.ndarray]],
    pointclouds: dict[tuple[int, int], np.ndarray],
    pointmaps: dict[tuple[int, int], PointMap],
) -> None:
    if not trajectory:
        raise RuntimeError("Sanity check failed: trajectory has no poses")

    ts = [x[0] for x in trajectory]
    if any(ts[i + 1] <= ts[i] for i in range(len(ts) - 1)):
        raise RuntimeError("Sanity check failed: trajectory timestamps are not monotonically increasing")

    if not pointmaps:
        raise RuntimeError("Sanity check failed: no pointmaps generated")
    for key, pm in pointmaps.items():
        if pm.points.size == 0:
            raise RuntimeError(f"Sanity check failed: empty pointmap for {key}")
        if pm.confidence.size == 0:
            raise RuntimeError(f"Sanity check failed: empty pointmap confidence for {key}")
        if np.any(pm.confidence < 0.0) or np.any(pm.confidence > 1.0):
            raise RuntimeError(f"Sanity check failed: pointmap confidence out of range for {key}")

    if not pointclouds:
        raise RuntimeError("Sanity check failed: no pointclouds generated")

    if all(pc.size == 0 for pc in pointclouds.values()):
        raise RuntimeError("Sanity check failed: all pointclouds are empty")
