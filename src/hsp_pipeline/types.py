from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FrameRecord:
    path: Path
    sec: int
    nsec: int
    timestamp: float


@dataclass(frozen=True)
class SegmentRecord:
    frame_ts: float
    sec: int
    nsec: int
    label: str
    canonical_label: str
    sam3_score: float
    descriptor: np.ndarray
    points_3d: np.ndarray
    mast3r_conf: np.ndarray
    centroid_xyz: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    is_structural: bool = False


@dataclass(frozen=True)
class Mast3rFeaturesV1:
    grid: np.ndarray
    confidence: np.ndarray


@dataclass(frozen=True)
class Mast3rFeaturesV2:
    grid: np.ndarray
    confidence: np.ndarray


@dataclass(frozen=True)
class PointMap:
    points: np.ndarray
    confidence: np.ndarray


@dataclass(frozen=True)
class PoseEstimate:
    timestamp: float
    translation: np.ndarray
    quaternion_xyzw: np.ndarray


@dataclass(frozen=True)
class BackboneFrameOutput:
    frame: FrameRecord
    rgb: np.ndarray
    features_v1: Mast3rFeaturesV1
    features_v2: Mast3rFeaturesV2
    pointmap: PointMap


@dataclass(frozen=True)
class PairwiseBackboneOutput:
    frame_i: tuple[int, int]
    frame_j: tuple[int, int]
    slam: dict[str, np.ndarray] = field(default_factory=dict)
    seg: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class BackboneOutput:
    frames: dict[tuple[int, int], BackboneFrameOutput]
    pairs: list[tuple[FrameRecord, FrameRecord]]
    pairwise_outputs: list[PairwiseBackboneOutput] = field(default_factory=list)


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta_before = value - self.mean
        self.mean += delta_before / self.count
        delta_after = value - self.mean
        self.m2 += delta_before * delta_after

    def merge(self, other: "RunningStats") -> None:
        if other.count == 0:
            return
        if self.count == 0:
            self.count = other.count
            self.mean = other.mean
            self.m2 = other.m2
            return
        total = self.count + other.count
        delta = other.mean - self.mean
        self.m2 = self.m2 + other.m2 + delta * delta * self.count * other.count / total
        self.mean = (self.mean * self.count + other.mean * other.count) / total
        self.count = total

    @property
    def std(self) -> float:
        if self.count < 2:
            return 0.0
        return float((self.m2 / (self.count - 1)) ** 0.5)


@dataclass
class GlobalInstance:
    instance_id: int
    canonical_label: str
    points_3d: np.ndarray
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    descriptor: np.ndarray
    sam3_stats: RunningStats = field(default_factory=RunningStats)
    mast3r_conf_mean: float = 0.0
    mast3r_conf_count: int = 0
    support_count: int = 0
    first_ts: float = 0.0
    last_ts: float = 0.0
    source_frame_list: list[float] = field(default_factory=list)

    def update(self, segment: SegmentRecord, ema_decay: float) -> None:
        if segment.points_3d.size:
            self.points_3d = np.concatenate([self.points_3d, segment.points_3d], axis=0)
            self.centroid = self.points_3d.mean(axis=0)
            self.bbox_min = self.points_3d.min(axis=0)
            self.bbox_max = self.points_3d.max(axis=0)

        self.descriptor = ema_decay * self.descriptor + (1.0 - ema_decay) * segment.descriptor
        self.sam3_stats.update(float(segment.sam3_score))

        if segment.mast3r_conf.size:
            total = self.mast3r_conf_mean * self.mast3r_conf_count + float(segment.mast3r_conf.sum())
            self.mast3r_conf_count += int(segment.mast3r_conf.size)
            self.mast3r_conf_mean = total / self.mast3r_conf_count

        self.support_count += 1
        self.last_ts = segment.frame_ts
        if self.support_count == 1:
            self.first_ts = segment.frame_ts
        self.source_frame_list.append(segment.frame_ts)

    def as_json(self) -> dict[str, Any]:
        combined = 0.5 * self.sam3_stats.mean + 0.5 * self.mast3r_conf_mean
        return {
            "instance_id": self.instance_id,
            "canonical_label": self.canonical_label,
            "centroid_xyz": self.centroid.tolist(),
            "bbox_min": self.bbox_min.tolist(),
            "bbox_max": self.bbox_max.tolist(),
            "num_points": int(self.points_3d.shape[0]),
            "support_count": self.support_count,
            "first_ts": self.first_ts,
            "last_ts": self.last_ts,
            "sam3_confidence": self.sam3_stats.mean,
            "sam3_confidence_std": self.sam3_stats.std,
            "mast3r_confidence": self.mast3r_conf_mean,
            "combined_confidence": combined,
            "source_frames": self.source_frame_list,
        }
