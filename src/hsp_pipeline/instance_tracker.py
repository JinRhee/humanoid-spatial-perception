from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .utils_spatial import transform_points


EPS = 1e-8


def _empty_points_3d() -> np.ndarray:
    return np.zeros((0, 3), dtype=np.float32)


@dataclass
class SegmentRecord:
    mask_id: int
    frame_index: int
    descriptor: np.ndarray
    points_world: np.ndarray
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    label: str | None
    score: float
    instance_id: int | None = None
    frame_ts: float | None = None

    @property
    def size(self) -> np.ndarray:
        return self.bbox_max - self.bbox_min


@dataclass
class SegmentationStore:
    segments_by_frame: dict[int, list[SegmentRecord]] = field(default_factory=dict)
    matches_by_pair: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)

    def add_segments(self, frame_index: int, segments: list[SegmentRecord]) -> None:
        self.segments_by_frame[frame_index] = segments

    def add_match_result(self, frame_a: int, frame_b: int, match_result: np.ndarray) -> None:
        self.matches_by_pair[(frame_a, frame_b)] = np.asarray(match_result)

    def get_segments(self, frame_index: int) -> list[SegmentRecord]:
        return self.segments_by_frame.get(frame_index, [])

    def get_match_result(self, frame_a: int, frame_b: int) -> np.ndarray | None:
        return self.matches_by_pair.get((frame_a, frame_b))


@dataclass(frozen=True)
class MergeWeights:
    descriptor: float
    centroid: float
    iou: float
    size: float


@dataclass
class InstanceState:
    instance_id: int
    descriptor: np.ndarray
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    support_count: int
    last_seen: int
    label: str | None
    score: float
    points_3d: np.ndarray = field(default_factory=_empty_points_3d)

    @property
    def size(self) -> np.ndarray:
        return self.bbox_max - self.bbox_min


def _to_numpy(array) -> np.ndarray:
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + EPS
    return float(np.dot(a, b) / denom)


def _centroid_similarity(a: np.ndarray, b: np.ndarray, max_dist: float) -> float:
    if max_dist <= 0:
        return 0.0
    dist = float(np.linalg.norm(a - b))
    return max(0.0, 1.0 - dist / max_dist)


def _size_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(a) + np.linalg.norm(b) + EPS
    diff = np.linalg.norm(a - b)
    return max(0.0, 1.0 - diff / denom)


def _bbox_iou(min_a: np.ndarray, max_a: np.ndarray, min_b: np.ndarray, max_b: np.ndarray) -> float:
    min_a = np.asarray(min_a, dtype=np.float32)
    max_a = np.asarray(max_a, dtype=np.float32)
    min_b = np.asarray(min_b, dtype=np.float32)
    max_b = np.asarray(max_b, dtype=np.float32)
    inter_min = np.maximum(min_a, min_b)
    inter_max = np.minimum(max_a, max_b)
    inter_size = np.maximum(0.0, inter_max - inter_min)
    inter_vol = float(np.prod(inter_size))
    vol_a = float(np.prod(np.maximum(0.0, max_a - min_a)))
    vol_b = float(np.prod(np.maximum(0.0, max_b - min_b)))
    denom = vol_a + vol_b - inter_vol + EPS
    return inter_vol / denom


BOX_EDGES = np.array(
    [
        [0, 1], [0, 2], [1, 3], [2, 3],
        [4, 5], [4, 6], [5, 7], [6, 7],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ],
    dtype=np.int32,
)


def bbox_iou(min_a: np.ndarray, max_a: np.ndarray, min_b: np.ndarray, max_b: np.ndarray) -> float:
    return _bbox_iou(min_a, max_a, min_b, max_b)


def bbox_corners(bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
    bbox_min = np.asarray(bbox_min, dtype=np.float32)
    bbox_max = np.asarray(bbox_max, dtype=np.float32)
    return np.array(
        [
            [bbox_min[0], bbox_min[1], bbox_min[2]],
            [bbox_max[0], bbox_min[1], bbox_min[2]],
            [bbox_min[0], bbox_max[1], bbox_min[2]],
            [bbox_max[0], bbox_max[1], bbox_min[2]],
            [bbox_min[0], bbox_min[1], bbox_max[2]],
            [bbox_max[0], bbox_min[1], bbox_max[2]],
            [bbox_min[0], bbox_max[1], bbox_max[2]],
            [bbox_max[0], bbox_max[1], bbox_max[2]],
        ],
        dtype=np.float32,
    )


def instance_rgba(instance_id: int, alpha: float = 1.0) -> np.ndarray:
    seed = int(np.uint32(instance_id) * np.uint32(2654435761))
    rng = np.random.default_rng(seed)
    rgb = rng.uniform(0.35, 0.95, size=3).astype(np.float32)
    return np.concatenate([rgb, np.array([alpha], dtype=np.float32)])


def build_segment_records(
    frame_index: int,
    masks,
    points_3d,
    conf,
    descriptors,
    pose_world: np.ndarray | None = None,
    frame_ts: float | None = None,
    label: str | None = None,
    score: float = 1.0,
) -> list[SegmentRecord]:
    masks_np = _to_numpy(masks)
    points_np = _to_numpy(points_3d)
    conf_np = _to_numpy(conf)
    desc_np = _to_numpy(descriptors)

    if desc_np.ndim == 3:
        desc_np = desc_np[0]

    if masks_np.ndim == 4:
        masks_np = masks_np[0]

    if masks_np.size == 0:
        return []

    segments: list[SegmentRecord] = []
    for mask_id in range(masks_np.shape[0]):
        mask = masks_np[mask_id] > 0
        if mask.sum() == 0:
            continue
        pts = points_np[mask]
        if pts.size == 0:
            continue
        if pose_world is not None:
            pts = transform_points(pts, pose_world)
        centroid = pts.mean(axis=0)
        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)
        descriptor = desc_np[:, mask_id]
        segments.append(
            SegmentRecord(
                mask_id=mask_id,
                frame_index=frame_index,
                descriptor=descriptor.astype(np.float32, copy=False),
                points_world=pts.astype(np.float32, copy=False),
                centroid=centroid.astype(np.float32, copy=False),
                bbox_min=bbox_min.astype(np.float32, copy=False),
                bbox_max=bbox_max.astype(np.float32, copy=False),
                label=label,
                score=score,
                frame_ts=frame_ts,
            )
        )
    return segments


DEFAULT_INSTANCE_TRACKER_CONFIG = {
    "merge_weights": {
        "descriptor": 0.5,
        "centroid":   0.3,
        "iou":        0.1,
        "size":       0.1,
    },
    "merge_threshold":       0.5,
    "max_centroid_distance": 2.0,   # metres
    "label_mismatch_penalty": 0.5,
    "ema_decay":             0.7,
    "min_support_count":     3,
    "resweep_interval":      50,
    "skip_resweep":          False,
}


class InstanceTracker:
    def __init__(
        self,
        merge_weights: MergeWeights | None = None,
        merge_threshold: float | None = None,
        max_centroid_distance: float | None = None,
        label_mismatch_penalty: float | None = None,
        ema_decay: float | None = None,
        min_support_count: int | None = None,
        resweep_interval: int | None = None,
        skip_resweep: bool | None = None,
    ) -> None:
        cfg = DEFAULT_INSTANCE_TRACKER_CONFIG

        if merge_weights is None:
            w = cfg["merge_weights"]
            merge_weights = MergeWeights(
                descriptor=w["descriptor"],
                centroid=w["centroid"],
                iou=w["iou"],
                size=w["size"],
            )

        self.merge_weights          = merge_weights
        self.merge_threshold        = merge_threshold        if merge_threshold        is not None else cfg["merge_threshold"]
        self.max_centroid_distance  = max_centroid_distance  if max_centroid_distance  is not None else cfg["max_centroid_distance"]
        self.label_mismatch_penalty = label_mismatch_penalty if label_mismatch_penalty is not None else cfg["label_mismatch_penalty"]
        self.ema_decay              = ema_decay              if ema_decay              is not None else cfg["ema_decay"]
        self.min_support_count      = min_support_count      if min_support_count      is not None else cfg["min_support_count"]
        self.resweep_interval       = resweep_interval       if resweep_interval       is not None else cfg["resweep_interval"]
        self.skip_resweep           = skip_resweep           if skip_resweep           is not None else cfg["skip_resweep"]

        self.instances: dict[int, InstanceState] = {}
        self._next_instance_id = 0
        self._prev_segments: list[SegmentRecord] = []
        
    def update_frame(
        self,
        segments: list[SegmentRecord],
        frame_index: int,
        match_result: np.ndarray | None = None,
    ) -> None:
        assignments: dict[int, int] = {}
        if match_result is not None and self._prev_segments:
            match = np.asarray(match_result)
            if match.ndim >= 2:
                # Accept (B, M) or higher-rank outputs by collapsing leading dims and using row 0.
                match = match.reshape(-1, match.shape[-1])[0]
            elif match.ndim == 0:
                match = np.zeros((0,), dtype=np.int64)
            for prev_idx, curr_idx in enumerate(match):
                curr_idx = int(curr_idx)
                if prev_idx >= len(self._prev_segments):
                    continue
                if curr_idx < 0 or curr_idx >= len(segments):
                    continue
                prev_seg = self._prev_segments[prev_idx]
                if prev_seg.instance_id is None:
                    continue
                if curr_idx not in assignments:
                    assignments[curr_idx] = prev_seg.instance_id

        for idx, seg in enumerate(segments):
            instance_id = assignments.get(idx)
            if instance_id is None:
                instance_id, best_score = self._assign_instance(seg)
                if instance_id is None:
                    instance_id = self._create_instance(seg, frame_index)
                else:
                    # Defensive: if the chosen instance_id was removed (e.g. during resweep),
                    # recreate a new instance to avoid KeyError and continue tracking.
                    if instance_id not in self.instances:
                        instance_id = self._create_instance(seg, frame_index)
                        self._update_instance(self.instances[instance_id], seg, frame_index, best_score)
                    else:
                        self._update_instance(self.instances[instance_id], seg, frame_index, best_score)
            else:
                if instance_id not in self.instances:
                    # Previously assigned instance id no longer exists; create replacement
                    instance_id = self._create_instance(seg, frame_index)
                else:
                    self._update_instance(self.instances[instance_id], seg, frame_index, None)
            seg.instance_id = instance_id

        self._prev_segments = segments
        if (
            not self.skip_resweep
            and self.resweep_interval > 0
            and frame_index > 0
            and frame_index % self.resweep_interval == 0
        ):
            self.resweep()

    def resweep(self) -> None:
        if len(self.instances) < 2:
            return
        ids = list(self.instances.keys())
        parent = {iid: iid for iid in ids}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i, id_a in enumerate(ids):
            inst_a = self.instances[id_a]
            for id_b in ids[i + 1 :]:
                inst_b = self.instances[id_b]
                score = self._score_instance_pair(inst_a, inst_b)
                if score >= self.merge_threshold:
                    union(id_a, id_b)

        clusters: dict[int, list[int]] = {}
        for iid in ids:
            root = find(iid)
            clusters.setdefault(root, []).append(iid)

        merged: dict[int, InstanceState] = {}
        for root, cluster_ids in clusters.items():
            if len(cluster_ids) == 1:
                merged[cluster_ids[0]] = self.instances[cluster_ids[0]]
                continue
            merged_inst = self._merge_instances([self.instances[iid] for iid in cluster_ids])
            merged[merged_inst.instance_id] = merged_inst

        self.instances = merged

    def final_split(self) -> tuple[list[InstanceState], list[InstanceState]]:
        if not self.skip_resweep:
            self.resweep()
        keep = [inst for inst in self.instances.values() if inst.support_count >= self.min_support_count]
        low = [inst for inst in self.instances.values() if inst.support_count < self.min_support_count]
        return keep, low

    def summaries(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for inst in self.instances.values():
            rows.append(
                {
                    "instance_id": inst.instance_id,
                    "support_count": inst.support_count,
                    "last_seen": inst.last_seen,
                    "centroid": inst.centroid.tolist(),
                    "bbox_min": inst.bbox_min.tolist(),
                    "bbox_max": inst.bbox_max.tolist(),
                    "label": inst.label,
                    "score": inst.score,
                }
            )
        return rows

    def _assign_instance(self, segment: SegmentRecord) -> tuple[int | None, float]:
        best_id: int | None = None
        best_score = -1.0
        for iid, inst in self.instances.items():
            score = self._score_segment(segment, inst)
            if score > best_score:
                best_score = score
                best_id = iid
        if best_id is None or best_score < self.merge_threshold:
            return None, best_score
        return best_id, best_score

    def _score_segment(self, seg: SegmentRecord, inst: InstanceState) -> float:
        desc_sim = _cosine_similarity(seg.descriptor, inst.descriptor)
        centroid_sim = _centroid_similarity(seg.centroid, inst.centroid, self.max_centroid_distance)
        iou_sim = _bbox_iou(seg.bbox_min, seg.bbox_max, inst.bbox_min, inst.bbox_max)
        size_sim = _size_similarity(seg.size, inst.size)
        score = (
            self.merge_weights.descriptor * desc_sim
            + self.merge_weights.centroid * centroid_sim
            + self.merge_weights.iou * iou_sim
            + self.merge_weights.size * size_sim
        )
        if seg.label and inst.label and seg.label != inst.label:
            score *= self.label_mismatch_penalty
        return score

    def _score_instance_pair(self, inst_a: InstanceState, inst_b: InstanceState) -> float:
        desc_sim = _cosine_similarity(inst_a.descriptor, inst_b.descriptor)
        centroid_sim = _centroid_similarity(inst_a.centroid, inst_b.centroid, self.max_centroid_distance)
        iou_sim = _bbox_iou(inst_a.bbox_min, inst_a.bbox_max, inst_b.bbox_min, inst_b.bbox_max)
        size_sim = _size_similarity(inst_a.size, inst_b.size)
        score = (
            self.merge_weights.descriptor * desc_sim
            + self.merge_weights.centroid * centroid_sim
            + self.merge_weights.iou * iou_sim
            + self.merge_weights.size * size_sim
        )
        if inst_a.label and inst_b.label and inst_a.label != inst_b.label:
            score *= self.label_mismatch_penalty
        return score

    def _create_instance(self, seg: SegmentRecord, frame_index: int) -> int:
        instance_id = self._next_instance_id
        self._next_instance_id += 1
        self.instances[instance_id] = InstanceState(
            instance_id=instance_id,
            descriptor=seg.descriptor.astype(np.float32, copy=True),
            centroid=seg.centroid.astype(np.float32, copy=True),
            bbox_min=seg.bbox_min.astype(np.float32, copy=True),
            bbox_max=seg.bbox_max.astype(np.float32, copy=True),
            support_count=1,
            last_seen=frame_index,
            label=seg.label,
            score=seg.score,
            points_3d=seg.points_world.astype(np.float32, copy=True),
        )
        return instance_id

    def _update_instance(
        self,
        inst: InstanceState,
        seg: SegmentRecord,
        frame_index: int,
        score: float | None,
    ) -> None:
        alpha = self.ema_decay
        inst.descriptor = alpha * inst.descriptor + (1.0 - alpha) * seg.descriptor
        inst.centroid = alpha * inst.centroid + (1.0 - alpha) * seg.centroid
        inst.bbox_min = np.minimum(inst.bbox_min, seg.bbox_min)
        inst.bbox_max = np.maximum(inst.bbox_max, seg.bbox_max)
        inst.support_count += 1
        inst.last_seen = frame_index
        inst.points_3d = np.concatenate([inst.points_3d, seg.points_world], axis=0)
        if seg.label and (inst.label is None or seg.score >= inst.score):
            inst.label = seg.label
        if score is not None:
            inst.score = max(inst.score, score)

    def _merge_instances(self, instances: Iterable[InstanceState]) -> InstanceState:
        inst_list = list(instances)
        total_support = max(sum(inst.support_count for inst in inst_list), 1)
        weights = np.array([inst.support_count for inst in inst_list], dtype=np.float32)
        weights_sum = weights.sum()
        if weights_sum <= 0:
            weights = np.full(len(inst_list), 1.0 / len(inst_list), dtype=np.float32)
        else:
            weights = weights / weights_sum

        def weighted_avg(values: Iterable[np.ndarray]) -> np.ndarray:
            stacked = np.stack([v for v in values], axis=0)
            return (weights[:, None] * stacked).sum(axis=0)

        descriptor = weighted_avg([inst.descriptor for inst in inst_list])
        centroid = weighted_avg([inst.centroid for inst in inst_list])
        bbox_min = np.min([inst.bbox_min for inst in inst_list], axis=0)
        bbox_max = np.max([inst.bbox_max for inst in inst_list], axis=0)
        last_seen = max(inst.last_seen for inst in inst_list)
        points_3d = np.concatenate([inst.points_3d for inst in inst_list], axis=0)
        label, score = self._merge_labels(inst_list)

        keep_id = min(inst.instance_id for inst in inst_list)
        return InstanceState(
            instance_id=keep_id,
            descriptor=descriptor.astype(np.float32, copy=False),
            centroid=centroid.astype(np.float32, copy=False),
            bbox_min=bbox_min.astype(np.float32, copy=False),
            bbox_max=bbox_max.astype(np.float32, copy=False),
            support_count=total_support,
            last_seen=last_seen,
            label=label,
            score=score,
            points_3d=points_3d.astype(np.float32, copy=False),
        )

    def _merge_labels(self, instances: Iterable[InstanceState]) -> tuple[str | None, float]:
        best_label = None
        best_score = -1.0
        for inst in instances:
            if inst.label is None:
                continue
            score = inst.score * inst.support_count
            if score > best_score:
                best_score = score
                best_label = inst.label
        return best_label, max(best_score, 0.0)